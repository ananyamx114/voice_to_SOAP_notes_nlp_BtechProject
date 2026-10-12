import tempfile
import unittest
from pathlib import Path

from voice_to_soap.normalize import Normalizer


class NormalizationTests(unittest.TestCase):
    def test_removes_configured_fillers_and_audits_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "disfluencies.csv").write_text("heard\num\nuh\n", encoding="utf-8")
            normalizer = Normalizer(root)
            utterance = {
                "id": "U1",
                "text_raw": "Um, I, uh, have pain.",
                "text_normalized": "Um, I, uh, have pain.",
            }

            result, changes = normalizer.normalize([utterance])

            self.assertEqual(result[0]["text_normalized"], "I, have pain.")
            self.assertEqual(result[0]["text_raw"], utterance["text_raw"])
            self.assertEqual(utterance["text_normalized"], utterance["text_raw"])
            self.assertEqual(changes[0]["before"], "Um, I, uh, have pain.")
            self.assertEqual(changes[0]["after"], "I, have pain.")
            self.assertEqual(
                [operation["type"] for operation in changes[0]["operations"]],
                ["disfluency_removal", "disfluency_removal"],
            )

    def test_does_not_remove_filler_substrings_from_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "disfluencies.csv").write_text("heard\nuh\n", encoding="utf-8")
            normalizer = Normalizer(root)

            result, changes = normalizer.normalize([{
                "id": "U1",
                "text_raw": "I thought about it.",
                "text_normalized": "I thought about it.",
            }])

            self.assertEqual(result[0]["text_normalized"], "I thought about it.")
            self.assertEqual(changes, [])

    def test_maps_longest_exact_brand_phrase_without_touching_substrings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "drug_names.csv").write_text(
                "heard,generic\nbrand forte,medicine\nbrand,other\n",
                encoding="utf-8",
            )
            normalizer = Normalizer(root)

            result, changes = normalizer.normalize([{
                "id": "U1",
                "text_raw": "Take BRAND FORTE; not brandish.",
                "text_normalized": "Take BRAND FORTE; not brandish.",
            }])

            self.assertEqual(
                result[0]["text_normalized"],
                "Take medicine; not brandish.",
            )
            self.assertEqual(
                [operation["heard"] for operation in changes[0]["operations"]],
                ["brand forte"],
            )

    def test_expands_abbreviations_and_keeps_original_utterances(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "abbreviations.csv").write_text(
                "heard,canonical\nbid,twice daily\n",
                encoding="utf-8",
            )
            normalizer = Normalizer(root)
            utterances = [{
                "id": "U1",
                "text_raw": "Take it BID.",
                "text_normalized": "Take it BID.",
            }]

            result, changes = normalizer.normalize(utterances)

            self.assertEqual(result[0]["text_normalized"], "Take it twice daily.")
            self.assertEqual(utterances[0]["text_normalized"], "Take it BID.")
            self.assertEqual(changes[0]["operations"][0]["type"], "abbreviation_expansion")


if __name__ == "__main__":
    unittest.main()
