import tempfile
import unittest
from pathlib import Path

from voice_to_soap.datasets import parse_transcript
from voice_to_soap.normalize import Normalizer
from voice_to_soap.schemas import validate_facts, validate_soap
from voice_to_soap.steps import find_omissions, render_soap


class CoreTests(unittest.TestCase):
    def test_parse_transcript_assigns_ids_and_roles(self):
        rows = parse_transcript("[doctor] Any fever?\n[patient] No fever.")
        self.assertEqual([row["id"] for row in rows], ["U1", "U2"])
        self.assertEqual(rows[0]["role"], "doctor")
        self.assertEqual(rows[1]["role"], "patient")

    def test_citations_are_validated(self):
        with self.assertRaisesRegex(ValueError, "unknown utterance"):
            validate_facts(
                {"facts": [{
                    "category": "symptom", "text": "fever denied", "assertion": "absent",
                    "speaker": "patient", "entity_type": "symptom", "entity_text": "fever",
                    "evidence": ["U9"],
                }]},
                {"U1"},
            )
        with self.assertRaisesRegex(ValueError, "unknown IDs"):
            validate_soap({"plan": [{"text": "Rest", "evidence": ["U2"]}]}, {"U1"})

    def test_lookup_normalization_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "drug_names.csv").write_text("heard,generic\nparacetemol,paracetamol\n")
            normalizer = Normalizer(root)
            rows, changes = normalizer.normalize([{
                "id": "U1", "text_raw": "Take paracetemol", "text_normalized": "Take paracetemol"
            }])
            self.assertEqual(rows[0]["text_normalized"], "Take paracetamol")
            self.assertEqual(changes[0]["utterance_id"], "U1")

    def test_note_rendering_and_omission_output(self):
        claims = [
            {"section": "plan", "text": "Take rest", "evidence": ["U1"], "verdict": "supported"},
            {"section": "assessment", "text": "Unsupported diagnosis", "evidence": ["U1"], "verdict": "unsupported"},
        ]
        note = render_soap(claims, supported_only=True)
        self.assertEqual(len(note["plan"]), 1)
        self.assertEqual(note["assessment"], [])
        self.assertEqual(
            find_omissions([{"text": "fever", "assertion": "absent"}], claims),
            [{
                "fact": "fever",
                "assertion": "absent",
                "entity_text": "",
                "evidence": [],
            }],
        )


if __name__ == "__main__":
    unittest.main()
