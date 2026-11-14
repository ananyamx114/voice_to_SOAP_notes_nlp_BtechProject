import csv
import io
import json
import sys
import tarfile
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from voice_to_soap.aci_soap import (
    InvalidGeneratedSoapError,
    _encode_example,
    _paired_fit_examples,
    _prompt_text,
    evaluate_aci_soap_lora,
    load_aci_soap_examples,
    prepare_combined_soap_data,
    prepare_aci_soap_data,
    split_note_into_soap,
)


class CharacterTokenizer:
    chat_template = None
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


class AciSoapTests(unittest.TestCase):
    @staticmethod
    def fake_rouge_module():
        class FakeRougeScorer:
            def __init__(self, *_args, **_kwargs):
                pass

            def score(self, *_args, **_kwargs):
                return {"rougeL": SimpleNamespace(fmeasure=0.0)}

        module = types.ModuleType("rouge_score")
        module.rouge_scorer = SimpleNamespace(RougeScorer=FakeRougeScorer)
        return module

    def test_maps_standard_headings_and_splits_combined_assessment_plan(self):
        note = """CHIEF COMPLAINT
Patient reports cough.

PHYSICAL EXAM
Lungs clear.

ASSESSMENT AND PLAN
Acute cough without fever. Start cough medicine today. Follow up in one week.
"""
        soap, warnings = split_note_into_soap(note)

        self.assertEqual(soap["subjective"], "Patient reports cough.")
        self.assertEqual(soap["objective"], "Lungs clear.")
        self.assertEqual(soap["assessment"], "Acute cough without fever.")
        self.assertIn("Start cough medicine today.", soap["plan"])
        self.assertIn("Follow up in one week.", soap["plan"])
        self.assertTrue(any("heuristically divided" in item for item in warnings))

    def test_maps_markdown_numbered_betrac_soap_sections(self):
        note = """**1. Subjective**
- **Chief Complaint:** Wants pregnancy.

**2. Objective**
- Omitted (no physical exam findings or test results)

**3. Assessment and Plan**
- **Problem #1: Fertility counseling**
  - **Assessment:** Patient desires pregnancy.
  - **Plan:**
    - Track menstrual period.
    - Schedule follow-up next month.
"""
        soap, warnings = split_note_into_soap(note)

        self.assertIn("Wants pregnancy.", soap["subjective"])
        self.assertIn("Omitted", soap["objective"])
        self.assertIn("Patient desires pregnancy.", soap["assessment"])
        self.assertIn("Track menstrual period.", soap["plan"])
        self.assertIn("Schedule follow-up next month.", soap["plan"])
        self.assertTrue(any("heuristically divided" in warning for warning in warnings))

    def test_prepares_combined_dataset_without_using_held_out_sources_for_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            aci_dir = root / "aci"
            aci_dir.mkdir()
            for split, encounter_id in (
                ("train", "train-1"),
                ("validation", "valid-1"),
                ("test", "test-1"),
            ):
                example = {
                    "encounter_id": encounter_id,
                    "source_split": split,
                    "dialogue": "[patient] I have a cough.",
                    "soap": {
                        "subjective": "Patient reports cough.",
                        "objective": "",
                        "assessment": "",
                        "plan": "",
                    },
                }
                (aci_dir / f"{split}.jsonl").write_text(
                    json.dumps(example) + "\n", encoding="utf-8"
                )
            archive_path = root / "betrac.tar"
            members = {
                "dialog_1.transcript.txt": b"DOCTOR: What hurts?\nPATIENT: My cough.",
                "dialog_1.soap.txt": (
                    b"**1. Subjective**\n- Cough.\n\n"
                    b"**2. Objective**\n- Omitted.\n\n"
                    b"**3. Assessment and Plan**\n- **Assessment:** Cough.\n"
                    b"- **Plan:** Follow up next week."
                ),
                "dialog_1.json": json.dumps({"id": "dialog_1"}).encode(),
                "dialog_1.opus": b"fake audio member for preparation test",
            }
            with tarfile.open(archive_path, "w") as archive:
                for name, payload in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))

            output_dir = root / "combined"
            summary = prepare_combined_soap_data(aci_dir, archive_path, output_dir)
            train = load_aci_soap_examples(output_dir / "train.jsonl")
            validation = load_aci_soap_examples(output_dir / "validation.jsonl")
            test = load_aci_soap_examples(output_dir / "test.jsonl")

            self.assertEqual(len(train), 2)
            betrac = next(item for item in train if item["encounter_id"].startswith("betrac-"))
            self.assertEqual(betrac["source_split"], "dev")
            self.assertEqual(betrac["assigned_split"], "train")
            self.assertIn("[doctor]", betrac["dialogue"])
            self.assertIn("Follow up next week.", betrac["soap"]["plan"])
            self.assertEqual([item["encounter_id"] for item in validation], ["aci-valid-1"])
            self.assertEqual([item["encounter_id"] for item in test], ["aci-test-1"])
            self.assertEqual(summary["be_trac_archive_samples"], 1)
            self.assertEqual(
                summary["split_counts"]["train"]["examples"], 2
            )

    def test_refuses_notes_without_recognized_soap_content(self):
        with self.assertRaisesRegex(ValueError, "no content under recognized SOAP"):
            split_note_into_soap("An unstructured note with no supported headings.")

    def test_prepares_disjoint_train_validation_and_held_out_test_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            dataset.mkdir()
            for filename, encounter_id in (
                ("train.csv", "train-1"),
                ("valid.csv", "valid-1"),
                ("clinicalnlp_taskB_test1.csv", "test-1"),
                ("clinicalnlp_taskC_test2.csv", "test-2"),
                ("clef_taskC_test3.csv", "test-3"),
            ):
                with (dataset / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["encounter_id", "dialogue", "note"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "encounter_id": encounter_id,
                        "dialogue": "Doctor: How are you?\nPatient: Better.",
                        "note": "CHIEF COMPLAINT\nPatient is better.",
                    })

            output = root / "prepared"
            summary = prepare_aci_soap_data(dataset, output)

            self.assertEqual(summary["splits"]["train"]["examples"], 1)
            self.assertEqual(summary["splits"]["validation"]["examples"], 1)
            self.assertEqual(summary["splits"]["test"]["examples"], 3)
            train = load_aci_soap_examples(output / "train.jsonl")
            validation = load_aci_soap_examples(output / "validation.jsonl")
            test = load_aci_soap_examples(output / "test.jsonl")
            self.assertEqual(train[0]["soap"]["subjective"], "Patient is better.")
            self.assertEqual(train[0]["dialogue"], "Doctor: How are you?\nPatient: Better.")
            self.assertNotEqual(train[0]["encounter_id"], validation[0]["encounter_id"])
            self.assertEqual(len(test), 3)
            self.assertTrue((output / "preparation_report.json").is_file())

    def test_rejects_encounter_leakage_between_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in ("train.csv", "valid.csv", "clinicalnlp_taskB_test1.csv",
                             "clinicalnlp_taskC_test2.csv", "clef_taskC_test3.csv"):
                with (root / filename).open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=["encounter_id", "dialogue", "note"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "encounter_id": "duplicate",
                        "dialogue": "Doctor: Hello.",
                        "note": "CHIEF COMPLAINT\nHello.",
                    })

            with self.assertRaisesRegex(ValueError, "more than one split"):
                prepare_aci_soap_data(root, root / "prepared")

    def test_masks_dialogue_tokens_from_training_loss(self):
        example = {
            "encounter_id": "pilot-1",
            "dialogue": "Doctor: How are you?",
            "soap": {
                "subjective": "Patient feels well.",
                "objective": "",
                "assessment": "",
                "plan": "",
            },
        }
        encoded = _encode_example(CharacterTokenizer(), example, max_seq_length=4096)
        first_target = next(index for index, label in enumerate(encoded["labels"]) if label != -100)
        self.assertGreater(first_target, 0)
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_target]))
        self.assertEqual(encoded["labels"][-1], CharacterTokenizer.eos_token_id)

    def test_does_not_silently_truncate_long_dialogue(self):
        example = {
            "encounter_id": "long-1",
            "dialogue": "Doctor: " + ("long dialogue " * 10),
            "soap": {section: "" for section in ("subjective", "objective", "assessment", "plan")},
        }
        with self.assertRaisesRegex(ValueError, "No automatic truncation"):
            _encode_example(CharacterTokenizer(), example, max_seq_length=8)

    def test_hybrid_prompt_includes_facts_while_direct_prompt_does_not(self):
        dialogue = "[patient] I have a fever."
        facts = [{"concept": "fever", "assertion": "present", "evidence": "U1"}]

        direct = _prompt_text(CharacterTokenizer(), dialogue)
        hybrid = _prompt_text(CharacterTokenizer(), dialogue, facts)

        self.assertNotIn("RULE-BASED CLINICAL FACTS", direct)
        self.assertIn("RULE-BASED CLINICAL FACTS", hybrid)
        self.assertIn('"evidence": "U1"', hybrid)

    def test_pair_preflight_uses_the_same_fit_examples_for_both_approaches(self):
        short = {
            "encounter_id": "short",
            "dialogue": "[patient] I have a fever.",
            "soap": {section: "" for section in ("subjective", "objective", "assessment", "plan")},
        }
        long = {
            "encounter_id": "long",
            "dialogue": "[patient] " + ("conversation " * 100),
            "soap": {section: "" for section in ("subjective", "objective", "assessment", "plan")},
        }
        hybrid_size = len(_encode_example(
            CharacterTokenizer(), short, max_seq_length=10000, approach="hybrid"
        )["input_ids"])

        eligible, skipped = _paired_fit_examples(
            [short, long], CharacterTokenizer(), hybrid_size + 1
        )

        self.assertEqual([item["encounter_id"] for item in eligible], ["short"])
        self.assertEqual(skipped, ["long"])

    def test_loader_rejects_malformed_json_lines(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "examples.jsonl"
            path.write_text(json.dumps({"dialogue": "Doctor: Hi", "soap": {}}) + "\n")
            with self.assertRaisesRegex(ValueError, "string fields"):
                load_aci_soap_examples(path)

    @patch("voice_to_soap.aci_soap.AciSoapLoRAGenerator")
    def test_evaluation_records_invalid_json_without_aborting(self, mock_generator):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_path = root / "test.jsonl"
            example = {
                "encounter_id": "test-1",
                "dialogue": "Doctor: What hurts?\nPatient: My arm.",
                "soap": {
                    "subjective": "Patient reports arm pain.",
                    "objective": "",
                    "assessment": "",
                    "plan": "",
                },
            }
            test_path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            generator = mock_generator.return_value
            generator.device = "mps"
            generator.generate.side_effect = InvalidGeneratedSoapError(
                "Model returned invalid JSON", "not json"
            )

            with patch.dict(sys.modules, {"rouge_score": self.fake_rouge_module()}):
                result = evaluate_aci_soap_lora(
                    test_path,
                    "test-model",
                    None,
                    root / "results",
                    device="mps",
                )

            self.assertEqual(result["valid_predictions"], 0)
            self.assertEqual(result["invalid_predictions"], 1)
            self.assertIsNone(result["overall_mean_section_rougeL_fmeasure"])
            prediction = json.loads(
                (root / "results" / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertIsNone(prediction["prediction_soap"])
            self.assertEqual(prediction["raw_model_output"], "not json")

    @patch("voice_to_soap.aci_soap.AciSoapLoRAGenerator")
    def test_hybrid_evaluation_records_facts_and_passes_them_to_generator(
        self, mock_generator
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test_path = root / "test.jsonl"
            example = {
                "encounter_id": "test-hybrid",
                "dialogue": "[patient] I have no fever.",
                "soap": {
                    "subjective": "Patient denies fever.",
                    "objective": "",
                    "assessment": "",
                    "plan": "",
                },
            }
            test_path.write_text(json.dumps(example) + "\n", encoding="utf-8")
            generator = mock_generator.return_value
            generator.device = "cpu"
            generator.generate.return_value = {
                section: "" for section in ("subjective", "objective", "assessment", "plan")
            }

            with patch.dict(sys.modules, {"rouge_score": self.fake_rouge_module()}):
                result = evaluate_aci_soap_lora(
                    test_path,
                    "test-model",
                    None,
                    root / "results",
                    approach="hybrid",
                )

            passed_facts = generator.generate.call_args.kwargs["facts"]
            self.assertEqual(passed_facts[0]["assertion"], "absent")
            self.assertEqual(passed_facts[0]["evidence"], "U1")
            self.assertEqual(result["encounter_ids"], ["test-hybrid"])
            prediction = json.loads(
                (root / "results" / "predictions.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(prediction["input_facts"][0]["assertion"], "absent")


if __name__ == "__main__":
    unittest.main()
