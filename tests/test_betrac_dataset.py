"""Synthetic integrity tests; no downloads or model loads."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from voice_to_soap import betrac_dataset as b
from voice_to_soap.aci_soap import load_aci_soap_examples

class Tokenizer:
    chat_template = None
    eos_token_id = 0
    def __call__(self,text,add_special_tokens=False):
        return {'input_ids':list(text)}

class BeTraCDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.config=json.loads(Path('configs/betrac_2000.json').read_text())
        self.config.update(target_examples=10,train_examples=8,validation_examples=1,test_examples=1)

    def archive(self, duplicate=False, count=12):
        path=self.root/'train-00000.tar'
        with tarfile.open(path,'w') as t:
            for i in range(count):
                text=f'DOCTOR: Hello {i}.\nPATIENT: I feel well {i}.'
                if duplicate and i==1:
                    text='DOCTOR: Hello 0.\nPATIENT: I feel well 0.'
                for suffix,data in {'.transcript.txt':text,'.soap.txt':'SUBJECTIVE\nFeels well.',
                                    '.json':json.dumps({'id':str(i)}),'.opus':'not decoded'}.items():
                    raw=data.encode(); m=tarfile.TarInfo(f'dialog_{i:04d}'+suffix);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        return path

    def test_split_integrity_raw_pairing_loader_and_reproducibility(self):
        archive=self.archive()
        outputs=[]
        for name in ('a','b'):
            out=self.root/name
            b.build([('train-00000.tar',archive,{})],Tokenizer(),self.config,out)
            outputs.append(out)
            all_rows=[]
            for split,count in [('train',8),('validation',1),('test',1)]:
                rows=load_aci_soap_examples(out/(split+'.jsonl'))
                self.assertEqual(len(rows),count)
                self.assertTrue(all(r['source_split']=='train' and r['assigned_split']==split for r in rows))
                self.assertTrue(all(r['transcript'].startswith('DOCTOR:') and r['reference_soap_raw']=='SUBJECTIVE\nFeels well.' for r in rows))
                self.assertTrue(all(r['hybrid_sequence_tokens']<=4096 and r['direct_sequence_tokens']<=4096 for r in rows))
                all_rows+=rows
            self.assertEqual(len({r['encounter_id'] for r in all_rows}),10)
            self.assertEqual(len({r['transcript'] for r in all_rows}),10)
            self.assertFalse(list(out.glob('*.opus')))
        self.assertEqual((outputs[0]/'train.jsonl').read_bytes(),(outputs[1]/'train.jsonl').read_bytes())
        self.assertEqual((outputs[0]/'split_manifest.json').read_bytes(),(outputs[1]/'split_manifest.json').read_bytes())

    def test_duplicate_transcript_rejected_and_reported(self):
        out=self.root/'out'
        report=b.build([('train-00000.tar',self.archive(True),{})],Tokenizer(),self.config,out)
        self.assertEqual(report['rejection_reasons']['duplicate_id_transcript_or_metadata_id'],1)

    def test_insufficient_does_not_publish_training_splits(self):
        config={**self.config,'max_transcript_tokens':1}
        out=self.root/'out'
        with self.assertRaisesRegex(ValueError,'Only 0 qualify'):
            b.build([('train-00000.tar',self.archive(),{})],Tokenizer(),config,out)
        self.assertFalse((out/'train.jsonl').exists())
        self.assertEqual(json.loads((out/'selection_report.json').read_text())['status'],'insufficient_eligible_data')

    def test_each_length_limit_is_a_rejection_not_truncation(self):
        text='DOCTOR: Hello.\nPATIENT: I have cough.'
        for limit,reason in [('max_transcript_tokens','transcript_tokens'),('max_soap_tokens','reference_soap_tokens'),
                             ('max_hybrid_facts_tokens','hybrid_facts_tokens'),('max_total_sequence_tokens','direct_sequence_tokens')]:
            config={**self.config,limit:1}
            row,actual=b.qualify('id',text,'SUBJECTIVE\nCough.',{},'train-00000.tar',Tokenizer(),config)
            self.assertIsNone(row)
            self.assertEqual(actual,reason)

    def test_dev_archive_and_unsafe_members_rejected(self):
        with self.assertRaisesRegex(ValueError,'official training'):
            b.build([('dev-00000.tar',self.archive(),{})],Tokenizer(),self.config,self.root/'out')
        path=self.root/'unsafe.tar'
        with tarfile.open(path,'w') as t:
            m=tarfile.TarInfo('../bad.transcript.txt');m.size=1;t.addfile(m,io.BytesIO(b'x'))
        with self.assertRaisesRegex(ValueError,'unsafe'):
            list(b.candidates(path))

    def test_hybrid_sequence_overflow_rejected_even_when_direct_fits(self):
        from voice_to_soap.aci_soap import _encode_example, _betrac_transcript_to_dialogue, split_note_into_soap
        text='DOCTOR: Hello.\nPATIENT: I have cough.'
        reference='SUBJECTIVE\nCough.'
        example={'encounter_id':'id','dialogue':_betrac_transcript_to_dialogue(text,'id'),
                 'soap':split_note_into_soap(reference)[0]}
        direct_length=len(_encode_example(Tokenizer(),example,10000,'direct')['input_ids'])
        config={**self.config,'max_hybrid_facts_tokens':10000,'max_total_sequence_tokens':direct_length}
        row,reason=b.qualify('id',text,reference,{},'train-00000.tar',Tokenizer(),config)
        self.assertIsNone(row)
        self.assertEqual(reason,'hybrid_sequence_tokens')

    def test_no_standalone_facts_cap_retains_facts_and_checks_full_sequence(self):
        from unittest.mock import patch
        facts=[{'concept':'synthetic mention','utterance_text':'x'*800}]
        config={**self.config,'max_hybrid_facts_tokens':None}
        with patch.object(b,'extract_clinical_facts',return_value=facts), patch('voice_to_soap.clinical_facts.extract_clinical_facts',return_value=facts):
            row,reason=b.qualify('id','DOCTOR: Hello.','SUBJECTIVE\nFeels well.',{},'train-00000.tar',Tokenizer(),config)
        self.assertIsNone(reason)
        self.assertGreater(row['facts_tokens'],700)
        self.assertGreater(row['hybrid_sequence_tokens'],row['direct_sequence_tokens'])
        self.assertLessEqual(row['hybrid_sequence_tokens'],4096)

    def test_source_reference_never_reaches_extractor(self):
        from unittest.mock import patch
        with patch.object(b,'extract_clinical_facts',return_value=[]) as extractor:
            row,reason=b.qualify('id','DOCTOR: Hello.','SUBJECTIVE\nREFERENCE_SENTINEL',{},'train-00000.tar',Tokenizer(),self.config)
        self.assertIsNone(reason)
        self.assertNotIn('REFERENCE_SENTINEL',extractor.call_args.args[0])

    def test_exact_production_split_sizes_with_sufficient_synthetic_candidates(self):
        config=json.loads(Path('configs/betrac_2000.json').read_text())
        out=self.root/'production-counts'
        report=b.build([('train-00000.tar',self.archive(count=2000),{})],Tokenizer(),config,out)
        self.assertEqual(report['accepted'],2000)
        for split,count in [('train',1600),('validation',200),('test',200)]:
            self.assertEqual(len(load_aci_soap_examples(out/(split+'.jsonl'))),count)
