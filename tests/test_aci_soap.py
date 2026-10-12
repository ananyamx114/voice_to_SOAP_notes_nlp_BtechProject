import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from voice_to_soap.aci_soap import (
    InvalidGeneratedSoapError,
    _encode_example,
    evaluate_aci_soap_lora,
    load_aci_soap_examples,
    prepare_aci_soap_data,
    split_note_into_soap,
)


class CharacterTokenizer:
    chat_template = None
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


class AciSoapTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
