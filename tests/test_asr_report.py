import json
from pathlib import Path
import tempfile
import unittest
from xml.etree import ElementTree
from zipfile import ZipFile

from scripts.build_asr_report import assemble, read_archive, word_document
from scripts.prepare_kaggle_asr import archive


class AsrReportTests(unittest.TestCase):
    def records(self):
        prefix = "data/outputs/asr-lora-experiment-run2/training/"
        config = {"training": {"epochs": 5}}
        records = {prefix + "run.json": {"status": "complete", "optimizer_updates": 515,
                   "model_id": "openai/whisper-large-v3", "config": config}}
        for epoch in range(1, 6):
            metrics = {"by_language": {language: {"wer": .6 - epoch * .05, "cer": .1}
                                       for language in ("en", "hi", "mr")}}
            records[prefix + f"epoch-{epoch}/metrics.json"] = metrics
        records[prefix + "best-adapter/asr_adapter.json"] = {
            "status": "complete", "config": config, "epoch": 5,
            "validation": records[prefix + "epoch-5/metrics.json"]}
        return records

    def test_final_report_refuses_unfinished_paired_tests(self):
        with self.assertRaisesRegex(ValueError, "Paired CUDA evaluation is incomplete"):
            assemble(self.records(), require_complete=True)

    def test_selected_epoch_must_match_validation_minimum(self):
        records = self.records()
        records["data/outputs/asr-lora-experiment-run2/training/best-adapter/asr_adapter.json"]["epoch"] = 4
        with self.assertRaisesRegex(ValueError, "validation-only"):
            assemble(records)

    def test_refuses_incomplete_training(self):
        with self.assertRaisesRegex(ValueError, "completed five-epoch"):
            assemble({"data/outputs/asr-lora-experiment-run2/training/run.json":
                      {"status": "paused", "optimizer_updates": 90}})

    def test_report_input_checksum_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "fixture.json"
            source.write_text(json.dumps({"fixture": True}))
            archive({source}, root / "resume-state.tar", root)
            with (root / "resume-state.tar").open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                read_archive(root / "resume-state.tar")

    def test_word_document_contains_valid_xml_and_measured_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            word_document("# Test & report\n\n## Measured results\n\n| Language | WER |\n|---|---:|\n| Marathi | 59.59% |\n", root)
            with ZipFile(root / "ASR_REPORT_SECTION.docx") as document:
                for name in document.namelist():
                    ElementTree.fromstring(document.read(name))
                text = document.read("word/document.xml").decode()
                self.assertIn("59.59%", text)
                self.assertIn("Test &amp; report", text)


if __name__ == "__main__":
    unittest.main()
