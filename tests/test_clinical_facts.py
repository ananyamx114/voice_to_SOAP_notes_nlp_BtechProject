import unittest

from voice_to_soap.clinical_facts import (
    extract_clinical_facts,
    parse_dialogue_utterances,
)


class ClinicalFactsTests(unittest.TestCase):
    def test_splits_speaker_labeled_dialogue_into_stable_utterance_ids(self):
        utterances = parse_dialogue_utterances(
            "[doctor] How are you?\n[patient] I have a cough."
        )

        self.assertEqual([item["id"] for item in utterances], ["U1", "U2"])
        self.assertEqual([item["role"] for item in utterances], ["doctor", "patient"])
        self.assertEqual(utterances[1]["text"], "I have a cough.")

    def test_detects_present_absent_uncertain_and_historical_assertions(self):
        examples = (
            ("[patient] I have a fever.", "present", "subjective"),
            ("[patient] I have no fever.", "absent", "subjective"),
            ("[patient] I might have a fever.", "uncertain", "subjective"),
            ("[patient] I had a fever last week.", "historical", "subjective"),
            ("[doctor] Any fever?", "uncertain", "unknown"),
            ("[doctor] Past medical history includes fever.", "historical", "unknown"),
        )
        for dialogue, assertion, section in examples:
            with self.subTest(assertion=assertion):
                facts = extract_clinical_facts(dialogue)
                fever = next(item for item in facts if item["concept"].casefold() == "fever")
                self.assertEqual(fever["assertion"], assertion)
                self.assertEqual(fever["evidence"], "U1")
                self.assertEqual(fever["section"], section)
                self.assertEqual(fever["utterance_text"], dialogue.split("] ", 1)[1])

    def test_assigns_exam_and_plan_facts_with_dose_and_source_span(self):
        dialogue = (
            "[doctor] On examination, there is pitting edema.\n"
            "[doctor] Start paracetamol 500 mg daily."
        )
        facts = extract_clinical_facts(dialogue)
        edema = next(item for item in facts if item["concept"].casefold() == "pitting edema")
        medicine = next(item for item in facts if item["category"] == "medication")

        self.assertEqual(edema["section"], "objective")
        self.assertEqual(edema["evidence"], "U1")
        self.assertEqual(medicine["section"], "plan")
        self.assertEqual(medicine["concept"], "paracetamol")
        self.assertEqual(medicine["normalized_candidate"], "paracetamol")
        self.assertEqual(medicine["dose_text"], "500 mg daily")
        self.assertEqual(
            dialogue.splitlines()[1].split("] ", 1)[1][
                medicine["span_start"]:medicine["span_end"]
            ],
            medicine["concept"],
        )

    def test_unknown_speaker_keeps_uncertain_section_assignment(self):
        facts = extract_clinical_facts("The patient reports cough.")

        self.assertEqual(facts[0]["section"], "unknown")
        self.assertEqual(facts[0]["speaker"], "unknown")

    def test_negation_does_not_cross_a_contrast_boundary(self):
        facts = extract_clinical_facts("[patient] No fever, but I have a cough.")
        by_concept = {fact["concept"].casefold(): fact["assertion"] for fact in facts}

        self.assertEqual(by_concept["fever"], "absent")
        self.assertEqual(by_concept["cough"], "present")

    def test_empty_transcript_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no non-empty utterances"):
            extract_clinical_facts(" \n ")


if __name__ == "__main__":
    unittest.main()
