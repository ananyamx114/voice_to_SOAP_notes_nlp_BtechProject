import tempfile
import unittest
from pathlib import Path

from voice_to_soap.entity_linking import TerminologyLinker
from voice_to_soap.schemas import validate_facts


class EntityLinkingTests(unittest.TestCase):
    def test_links_drugs_to_rxnorm_and_clinical_entities_to_umls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concepts.csv"
            path.write_text(
                "term,vocabulary,concept_id,canonical_name,entity_type\n"
                "paracetamol,RxNorm,123,acetaminophen 500 MG Oral Tablet,drug\n"
                "fever,UMLS,C0015967,Fever,symptom\n",
                encoding="utf-8",
            )
            linker = TerminologyLinker(path)
            facts = [
                {"entity_type": "drug", "entity_text": "Paracetamol"},
                {"entity_type": "symptom", "entity_text": "FEVER"},
            ]

            linked = linker.link_facts(facts)

            self.assertEqual(linked[0]["entity_link"]["status"], "linked")
            self.assertEqual(linked[0]["entity_link"]["concepts"][0]["vocabulary"], "RxNorm")
            self.assertEqual(linked[1]["entity_link"]["concepts"][0]["concept_id"], "C0015967")

    def test_reports_ambiguous_match_instead_of_selecting_arbitrarily(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concepts.csv"
            path.write_text(
                "term,vocabulary,concept_id,canonical_name,entity_type\n"
                "pain,UMLS,C1,Pain concept one,symptom\n"
                "pain,UMLS,C2,Pain concept two,symptom\n",
                encoding="utf-8",
            )
            linker = TerminologyLinker(path)

            linked = linker.link_facts([{
                "entity_type": "symptom",
                "entity_text": "pain",
            }])

            self.assertEqual(linked[0]["entity_link"]["status"], "ambiguous")
            self.assertEqual(len(linked[0]["entity_link"]["concepts"]), 2)

    def test_explicitly_marks_unmatched_entities(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "concepts.csv"
            path.write_text(
                "term,vocabulary,concept_id,canonical_name,entity_type\n",
                encoding="utf-8",
            )
            linker = TerminologyLinker(path)

            linked = linker.link_facts([{
                "entity_type": "condition",
                "entity_text": "unlisted condition",
            }])

            self.assertEqual(linked[0]["entity_link"]["status"], "not_found")
            self.assertEqual(linked[0]["entity_link"]["concepts"], [])

    def test_fact_validation_requires_cited_evidence_from_named_speaker(self):
        utterances = [{
            "id": "U1",
            "speaker_id": "speaker_1",
            "role": "patient",
        }]
        payload = {
            "facts": [{
                "category": "symptom",
                "text": "fever denied",
                "assertion": "absent",
                "speaker": "doctor",
                "entity_type": "symptom",
                "entity_text": "fever",
                "evidence": ["U1"],
            }]
        }

        with self.assertRaisesRegex(ValueError, "does not match its evidence"):
            validate_facts(payload, {"U1"}, utterances)

    def test_historical_assertion_is_canonicalized(self):
        payload = {
            "facts": [{
                "category": "condition",
                "text": "prior asthma",
                "assertion": "historical",
                "speaker": "patient",
                "entity_type": "condition",
                "entity_text": "asthma",
                "evidence": ["U1"],
            }]
        }

        facts = validate_facts(payload, {"U1"})

        self.assertEqual(facts[0]["assertion"], "historical")


if __name__ == "__main__":
    unittest.main()
