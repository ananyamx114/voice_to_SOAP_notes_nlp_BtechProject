import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from voice_to_soap.normalize import Normalizer
from voice_to_soap.steps import find_omissions, generate_verified_soap


UTTERANCES = [{
    "id": "U1",
    "role": "patient",
    "speaker_id": "speaker_1",
    "text_normalized": "I do not have a fever.",
}]
FACTS = [{
    "category": "symptom",
    "text": "fever denied",
    "assertion": "absent",
    "entity_type": "symptom",
    "entity_text": "fever",
    "evidence": ["U1"],
}]
OMITTED_DRAFT = {
    "subjective": [],
    "objective": [],
    "assessment": [],
    "plan": [],
}
COMPLETE_DRAFT = {
    "subjective": [{"text": "Patient denies fever.", "evidence": ["U1"]}],
    "objective": [],
    "assessment": [],
    "plan": [],
}


class VerificationTests(unittest.TestCase):
    def test_assertion_aware_omission_check_handles_paraphrases(self):
        self.assertEqual(find_omissions(
            FACTS,
            [{"text": "Patient denies fever.", "evidence": ["U1"]}],
        ), [])
        omissions = find_omissions(
            FACTS,
            [{"text": "Patient has fever.", "evidence": ["U1"]}],
        )
        self.assertEqual(omissions[0]["assertion"], "absent")

    def test_retry_feedback_adds_omitted_fact(self):
        generator = Mock()
        generator.generate_json.side_effect = [OMITTED_DRAFT, COMPLETE_DRAFT]
        llm = Mock()
        llm.complete_json.return_value = {"verdict": "supported", "reason": "Cited source agrees"}
        normalizer = Normalizer("/path/with/no/tables")

        outcome = generate_verified_soap(
            llm, UTTERANCES, FACTS, normalizer, generator, max_attempts=2
        )

        self.assertEqual(len(outcome["attempts"]), 2)
        self.assertEqual(outcome["attempts"][0]["status"], "needs_revision")
        self.assertIn("Include the source-supported fact", outcome["attempts"][0]["feedback"][0])
        self.assertEqual(outcome["attempts"][1]["status"], "passed")
        self.assertFalse(outcome["review_required"])
        second_prompt = generator.generate_json.call_args_list[1].args[1]
        self.assertIn("REQUIRED CORRECTIONS", second_prompt)
        self.assertIn("fever denied", second_prompt)

    def test_empty_dosage_table_does_not_invent_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dose_ranges.csv"
            path.write_text(
                "drug,min_value,max_value,unit,basis,route\n",
                encoding="utf-8",
            )
            normalizer = Normalizer(tmp)

            self.assertEqual(normalizer.check_dose_details("Take 500 mg paracetamol."), [])

    def test_dose_range_converts_units_and_flags_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dose_ranges.csv"
            path.write_text(
                "drug,min_value,max_value,unit,basis,route\n"
                "paracetamol,500,1000,mg,per_dose,oral\n",
                encoding="utf-8",
            )
            normalizer = Normalizer(tmp)

            in_range = normalizer.check_dose_details("Take 0.5 g paracetamol.")
            out_of_range = normalizer.check_dose_details("Take 2 g paracetamol.")

            self.assertEqual(in_range, [])
            self.assertEqual(len(out_of_range), 1)
            self.assertIn("configured range", out_of_range[0]["message"])

    def test_dose_warning_requests_review_and_is_preserved_after_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "dose_ranges.csv").write_text(
                "drug,min_value,max_value,unit,basis,route\n"
                "paracetamol,500,1000,mg,per_dose,oral\n",
                encoding="utf-8",
            )
            normalizer = Normalizer(tmp)
            utterances = [{
                "id": "U1",
                "role": "doctor",
                "speaker_id": "speaker_0",
                "text_normalized": "Take 2 g paracetamol.",
            }]
            generator = Mock()
            generator.generate_json.side_effect = [
                {"subjective": [], "objective": [], "assessment": [], "plan": [
                    {"text": "Take 2 g paracetamol.", "evidence": ["U1"]}
                ]},
                {"subjective": [], "objective": [], "assessment": [], "plan": []},
            ]
            llm = Mock()
            llm.complete_json.return_value = {"verdict": "supported", "reason": "Matches source"}

            outcome = generate_verified_soap(
                llm, utterances, [], normalizer, generator, max_attempts=2
            )

            self.assertEqual(outcome["warnings"][0]["detected_on_attempt"], 1)
            self.assertTrue(outcome["review_required"])
            self.assertIn("Do not silently alter", outcome["attempts"][0]["feedback"][0])

    def test_invalid_evidence_triggers_retry_before_verification(self):
        generator = Mock()
        generator.generate_json.side_effect = [
            {
                "subjective": [{"text": "Fever denied.", "evidence": ["U99"]}],
                "objective": [],
                "assessment": [],
                "plan": [],
            },
            COMPLETE_DRAFT,
        ]
        llm = Mock()
        llm.complete_json.return_value = {"verdict": "supported", "reason": "Source agrees"}
        normalizer = Normalizer("/path/with/no/tables")

        outcome = generate_verified_soap(
            llm, UTTERANCES, FACTS, normalizer, generator, max_attempts=2
        )

        self.assertEqual(outcome["attempts"][0]["status"], "invalid_draft")
        self.assertEqual(outcome["attempts"][1]["status"], "passed")
        self.assertEqual(llm.complete_json.call_count, 1)


if __name__ == "__main__":
    unittest.main()
