"""Dataset-only migration. Uses existing SOAP mapping, prompts, facts and encoder."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
import re
import shutil
import tarfile

from .aci_soap import _betrac_transcript_to_dialogue, _encode_example, split_note_into_soap
from .clinical_facts import extract_clinical_facts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')


def candidates(path):
    """Match observed flat WebDataset members by original key; never extract audio."""
    with tarfile.open(path) as archive:
        groups = {}
        for member in archive.getmembers():
            if not member.isfile() or not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:opus|transcript\.txt|soap\.txt|json)', member.name):
                raise ValueError(f'Unexpected/unsafe archive member: {member.name}')
            suffix = next(s for s in ('.transcript.txt','.soap.txt','.json','.opus') if member.name.endswith(s))
            key = member.name[:-len(suffix)]
            group = groups.setdefault(key,{})
            if suffix in group:
                raise ValueError('Duplicate archive member')
            group[suffix] = member
        for key, group in sorted(groups.items()):
            if not {'.transcript.txt','.soap.txt','.json'}.issubset(group):
                yield key, None, None, None
                continue
            texts = []
            for suffix in ('.transcript.txt','.soap.txt','.json'):
                with archive.extractfile(group[suffix]) as stream:
                    texts.append(stream.read().decode('utf-8'))
            yield key, texts[0], texts[1], json.loads(texts[2])


def qualify(key, transcript, reference, metadata, shard, tokenizer, config):
    if not transcript or not transcript.strip() or not reference or not reference.strip():
        return None, 'empty_or_incomplete_pair'
    if not isinstance(metadata, dict):
        return None, 'invalid_metadata'
    count = lambda text: len(tokenizer(text, add_special_tokens=False)['input_ids'])
    nt, ns = count(transcript), count(reference)
    if nt > config['max_transcript_tokens']:
        return None, 'transcript_tokens'
    if ns > config['max_soap_tokens']:
        return None, 'reference_soap_tokens'
    try:
        dialogue = _betrac_transcript_to_dialogue(transcript, key)
        soap, warnings = split_note_into_soap(reference)
    except ValueError:
        return None, 'existing_loader_or_soap_mapping_incompatible'
    if count(json.dumps(soap, ensure_ascii=False)) > config['max_soap_tokens']:
        return None, 'serialized_target_tokens'
    facts = extract_clinical_facts(dialogue)  # Never receives the reference.
    nf = count(json.dumps(facts, ensure_ascii=False))
    facts_limit = config.get('max_hybrid_facts_tokens')
    if facts_limit is not None and nf > facts_limit:
        return None, 'hybrid_facts_tokens'
    lengths = {}
    for approach in ('direct','hybrid'):
        try:
            encoded = _encode_example(tokenizer, {'encounter_id':key,'dialogue':dialogue,'soap':soap},
                                      config['max_total_sequence_tokens'], approach)
        except ValueError as exc:
            if 'No automatic truncation' not in str(exc):
                raise
            return None, approach+'_sequence_tokens'
        lengths[approach] = len(encoded['input_ids'])
    return {'encounter_id':key,'source_dataset':config['repository'],'source_split':'train',
            'source_shard':shard,'source_metadata':metadata,'target_type':'synthetic_soap',
            'transcript':transcript,'reference_soap_raw':reference,'dialogue':dialogue,'soap':soap,
            'mapping_warnings':warnings,'transcript_tokens':nt,'soap_tokens':ns,'facts_tokens':nf,
            'direct_sequence_tokens':lengths['direct'],'hybrid_sequence_tokens':lengths['hybrid']}, None


def statistics(rows):
    result = {}
    for field in ('transcript_tokens','soap_tokens','facts_tokens','direct_sequence_tokens','hybrid_sequence_tokens'):
        values = sorted(r[field] for r in rows)
        n = len(values)
        result[field] = {'min':values[0], 'median':(values[(n-1)//2]+values[n//2])/2,
                         'p95':values[math.ceil(n*.95)-1], 'max':values[-1]} if n else None
    return result


def build(archives, tokenizer, config, output):
    if config['source_split'] != 'train' or config['target_examples'] != sum(config[k] for k in ('train_examples','validation_examples','test_examples')):
        raise ValueError('Invalid official source or split counts')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    accepted, rejected, seen_ids, seen_text, seen_related, sources = [], [], set(), set(), set(), []
    for shard, path, provenance in archives:
        if not re.fullmatch(r'train-\d{5}\.tar', shard):
            raise ValueError('Only official training shards are accepted')
        sources.append({'shard':shard, **provenance})
        for key, text, note, metadata in candidates(path):
            fingerprint = hashlib.sha256(' '.join((text or '').casefold().split()).encode()).hexdigest()
            related = metadata.get('id') if isinstance(metadata,dict) else None
            related = str(related) if related is not None else None
            if key in seen_ids or fingerprint in seen_text or (related is not None and related in seen_related):
                row, reason = None, 'duplicate_id_transcript_or_metadata_id'
            else:
                seen_ids.add(key)
                seen_text.add(fingerprint)
                if related is not None:
                    seen_related.add(related)
                row, reason = qualify(key,text,note,metadata,shard,tokenizer,config)
            if row is None:
                rejected.append({'encounter_id':key,'source_shard':shard,'reason':reason})
            else:
                accepted.append(row)
            if (len(accepted)+len(rejected)) % 100 == 0:
                print(f'Inspected {len(accepted)+len(rejected)}; eligible {len(accepted)}',flush=True)
            if len(accepted) == config['target_examples']:
                break
        if len(accepted) == config['target_examples']:
            break
    report = {'status':'complete' if len(accepted)==config['target_examples'] else 'insufficient_eligible_data',
              'config':config,'sources':sources,'inspected':len(accepted)+len(rejected),'accepted':len(accepted),
              'rejected':len(rejected),'rejection_reasons':dict(Counter(r['reason'] for r in rejected)),
              'token_statistics':statistics(accepted),'selection':'first eligible records in shard/key order, then seeded split; prefix selection bias',
              'hybrid_validation':'existing unmodified extractor and _encode_example, including chat template, target and EOS',
              'related_encounters':'duplicate IDs, normalized transcripts and source metadata IDs checked; semantic/persona relatedness unverified',
              'limitations':['Synthetic references, not clinical ground truth.','Official dev and ACI are excluded.',
                             'Archive downloads contain audio bytes, but no audio is read into training or extracted.']}
    write_rows(output/'rejections.jsonl',rejected)
    write_json(output/'selection_report.json',report)
    if report['status'] != 'complete':
        write_rows(output/'eligible_incomplete.jsonl',accepted)
        raise ValueError(f"Only {len(accepted)} qualify; required {config['target_examples']}. See selection_report.json; limits unchanged.")
    random.Random(config['random_seed']).shuffle(accepted)
    a, b = config['train_examples'], config['validation_examples']
    splits = {'train':accepted[:a],'validation':accepted[a:a+b],'test':accepted[a+b:]}
    for name, rows in splits.items():
        for row in rows:
            row['assigned_split'] = name
        write_rows(output/(name+'.jsonl'),rows)
    write_rows(output/'all_2000.jsonl',accepted)
    manifest = {'source_split':'train','seed':config['random_seed'],
                'encounter_ids':{s:[r['encounter_id'] for r in rows] for s,rows in splits.items()},
                'sha256':{s:sha(output/(s+'.jsonl')) for s in splits}}
    write_json(output/'split_manifest.json',manifest)
    return report


def official_archives(config, cache):
    from huggingface_hub import HfApi, hf_hub_download
    info = HfApi().dataset_info(config['repository'],revision=config['revision'],files_metadata=True,token=False)
    if info.sha != config['revision'] or info.gated:
        raise ValueError('Unexpected revision or gated access; user must review terms')
    for entry in sorted(info.siblings,key=lambda s:s.rfilename):
        if not re.fullmatch(r'data/train-0000[0-8]\.tar',entry.rfilename):
            continue
        Path(cache).mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(cache).free < entry.size+12*1024**3:
            raise ValueError('Insufficient disk space with 12 GiB reserve')
        print('Acquiring '+entry.rfilename,flush=True)
        path = hf_hub_download(config['repository'],entry.rfilename,repo_type='dataset',
                               revision=config['revision'],cache_dir=cache,token=False)
        expected = entry.lfs.sha256 if not isinstance(entry.lfs,dict) else entry.lfs['sha256']
        actual = sha(path)
        if actual != expected or Path(path).stat().st_size != entry.size:
            raise ValueError('Training archive hash or size mismatch')
        yield Path(entry.rfilename).name,path,{'sha256':actual,'bytes':entry.size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/betrac_2000.json')
    parser.add_argument('--output',default='data/processed/betrac_2000')
    parser.add_argument('--cache',default='data/models/huggingface/hub')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'],revision=config['tokenizer_revision'],
                                              cache_dir=args.cache,local_files_only=True,trust_remote_code=False)
    result = build(official_archives(config,args.cache),tokenizer,config,args.output)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
