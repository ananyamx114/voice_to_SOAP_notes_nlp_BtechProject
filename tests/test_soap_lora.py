import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_to_soap.soap_lora import (
    _tokenize_training_example,
    load_soap_examples,
)
from voice_to_soap.steps import generate_claims


EXAMPLE = {
    "utterances": [
        {
            "id": "U1",
            "role": "doctor",
            "speaker_id": "speaker_0",
            "text_normalized": "Do you have a fever?",
        },
        {
            "id": "U2",
            "role": "patient",
            "speaker_id": "speaker_1",
            "text_normalized": "No, I do not.",
        },
    ],
    "facts": [
        {
            "category": "symptom",
            "text": "fever denied",
            "assertion": "absent",
            "entity_type": "symptom",
            "entity_text": "fever",
            "evidence": ["U2"],
        }
    ],
    "soap": {
        "subjective": [{"text": "Patient denies fever.", "evidence": ["U2"]}],
        "objective": [],
        "assessment": [],
        "plan": [],
    },
}


class CharacterTokenizer:
    chat_template = None
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


class SoapLoRATests(unittest.TestCase):
    def test_loads_example_and_checks_every_citation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text(json.dumps(EXAMPLE) + "\n", encoding="utf-8")

            examples = load_soap_examples(path)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["soap"]["subjective"][0]["evidence"], ["U2"])
            self.assertEqual(examples[0]["soap"]["assessment"], [])

    def test_rejects_training_claim_that_cites_missing_utterance(self):
        bad_example = json.loads(json.dumps(EXAMPLE))
        bad_example["soap"]["subjective"][0]["evidence"] = ["U99"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "examples.jsonl"
            path.write_text(json.dumps(bad_example) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown IDs"):
                load_soap_examples(path)

    def test_masks_prompt_tokens_so_loss_trains_on_soap_target(self):
        encoded = _tokenize_training_example(
            CharacterTokenizer(), EXAMPLE, max_seq_length=2048
        )
        first_target = next(index for index, label in enumerate(encoded["labels"]) if label != -100)

        self.assertGreater(first_target, 0)
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_target]))
        self.assertEqual(
            encoded["input_ids"][first_target:],
            encoded["labels"][first_target:],
        )
        self.assertEqual(encoded["labels"][-1], CharacterTokenizer.eos_token_id)

    def test_refuses_to_truncate_source_evidence(self):
        with self.assertRaisesRegex(ValueError, "Automatic truncation is disabled"):
            _tokenize_training_example(CharacterTokenizer(), EXAMPLE, max_seq_length=5)

    def test_generation_backend_output_still_validates_citations(self):
        generator = Mock()
        generator.generate_json.return_value = {
            "subjective": [{"text": "Patient denies fever.", "evidence": ["U2"]}],
            "objective": [],
            "assessment": [],
            "plan": [],
        }
        utterances = [
            {
                "id": "U1", "role": "doctor", "speaker_id": "speaker_0",
                "text_normalized": "Do you have a fever?",
            },
            {
                "id": "U2", "role": "patient", "speaker_id": "speaker_1",
                "text_normalized": "No, I do not.",
            },
        ]

        claims = generate_claims(Mock(), utterances, [], soap_generator=generator)

        self.assertEqual(claims[0]["evidence"], ["U2"])
        generator.generate_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
