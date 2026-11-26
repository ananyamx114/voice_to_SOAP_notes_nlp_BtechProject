import copy
import unicodedata
import unittest

from voice_to_soap.asr_scoring import CURRENT, LEGACY, normalize_asr_text, numeric_inventory, score_records


class ASRScoringTests(unittest.TestCase):
    def test_decimal_and_sign_errors_cannot_disappear(self):
        for reference, prediction in (("2.5 mg", "25 mg"), ("-5", "5"), ("-.5", ".5"),
                                      ("1/2 tablet", "12 tablet"), ("5%", "5")):
            with self.subTest(reference=reference):
                rows = [{"reference_transcript": reference, "prediction": prediction}]
                result = score_records(rows)
                self.assertGreater(result["wer"], 0)
                self.assertEqual(result["numeric_inventory_mismatch_examples"], 1)

    def test_legacy_is_explicitly_available_for_reproduction(self):
        rows = [{"reference_transcript": "2.5 mg", "prediction": "25 mg"}]
        self.assertEqual(score_records(rows, LEGACY)["wer"], 0)
        self.assertEqual(score_records(rows)["scoring_protocol"], CURRENT)

    def test_indic_marks_and_negation_remain_distinguishable(self):
        self.assertNotEqual(normalize_asr_text("कला"), normalize_asr_text("किला"))
        self.assertNotEqual(normalize_asr_text("नाही"), normalize_asr_text("नाही का"))
        self.assertEqual(normalize_asr_text("खांसी।नहीं"), "खांसी नहीं")
        self.assertEqual(normalize_asr_text("क़"), normalize_asr_text(unicodedata.normalize("NFD", "क़")))
        result = score_records([{"reference_transcript": "ताप नाही", "prediction": "ताप"}])
        self.assertEqual(result["negation_marker_mismatch_examples"], 1)

    def test_digit_script_and_unicode_minus_preserve_value(self):
        self.assertEqual(normalize_asr_text("−२.५ mg"), "-2.5 mg")
        self.assertEqual(numeric_inventory("−२.५ mg"), numeric_inventory("-2.5 mg"))
        self.assertEqual(normalize_asr_text("−.५"), "-.5")
        self.assertEqual(normalize_asr_text("२٫५"), "2.5")
        self.assertEqual(normalize_asr_text("２．５"), "2.5")
        self.assertNotEqual(numeric_inventory("−.५"), numeric_inventory("−५"))

    def test_punctuation_separates_words_but_keeps_content(self):
        self.assertEqual(normalize_asr_text("patient-reported (no fever)."), "patient reported no fever")
        self.assertEqual(normalize_asr_text("Can't, stop."), "can't stop")
        self.assertNotEqual(normalize_asr_text("can't"), normalize_asr_text("cant"))

    def test_numeric_forms_are_conservative_not_guessed(self):
        self.assertNotEqual(normalize_asr_text("1,000"), normalize_asr_text("1.000"))
        self.assertEqual(normalize_asr_text("range 5-10, at 06:30."), "range 5-10 at 06:30")

    def test_scores_do_not_mutate_raw_input(self):
        rows = [{"reference_transcript": "२.५ mg।", "prediction": "25 mg"}]
        original = copy.deepcopy(rows)
        score_records(rows)
        self.assertEqual(rows, original)

    def test_protocol_mismatch_and_empty_input_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_asr_text("5 mg", "unknown")
        with self.assertRaises(ValueError):
            score_records([])


if __name__ == "__main__":
    unittest.main()
