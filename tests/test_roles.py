import unittest
from unittest.mock import Mock

from voice_to_soap.roles import assign_speaker_roles


class RoleAssignmentTests(unittest.TestCase):
    def test_assigns_roles_and_preserves_evidence(self):
        utterances = [
            {
                "id": "U1", "speaker_id": "speaker_0", "role": "unknown",
                "text_raw": "Where does it hurt?", "text_normalized": "Where does it hurt?",
            },
            {
                "id": "U2", "speaker_id": "speaker_1", "role": "unknown",
                "text_raw": "My stomach hurts.", "text_normalized": "My stomach hurts.",
            },
        ]
        llm = Mock()
        llm.complete_json.return_value = {
            "roles": [
                {"speaker_id": "speaker_0", "role": "doctor", "evidence": ["U1"], "reason": "Asks clinical question"},
                {"speaker_id": "speaker_1", "role": "patient", "evidence": ["U2"], "reason": "Reports symptom"},
            ]
        }

        assigned, decisions = assign_speaker_roles(llm, utterances)

        self.assertEqual([item["role"] for item in assigned], ["doctor", "patient"])
        self.assertEqual([item["role"] for item in decisions], ["doctor", "patient"])
        self.assertIn("U1: Where does it hurt?", llm.complete_json.call_args.args[1])

    def test_keeps_uncertain_role_unknown(self):
        utterances = [
            {"id": "U1", "speaker_id": "speaker_0", "role": "unknown", "text_raw": "Hello"},
        ]
        llm = Mock()
        llm.complete_json.return_value = {
            "roles": [{"speaker_id": "speaker_0", "role": "unknown", "evidence": [], "reason": "Insufficient context"}]
        }

        assigned, _ = assign_speaker_roles(llm, utterances)

        self.assertEqual(assigned[0]["role"], "unknown")

    def test_preserves_roles_from_labeled_transcript_without_llm_call(self):
        utterances = [
            {"id": "U1", "speaker_id": "doctor", "role": "doctor", "text_raw": "Hello"},
            {"id": "U2", "speaker_id": "patient", "role": "patient", "text_raw": "Hi"},
        ]
        llm = Mock()

        assigned, decisions = assign_speaker_roles(llm, utterances)

        self.assertEqual([item["role"] for item in assigned], ["doctor", "patient"])
        self.assertEqual(llm.complete_json.call_count, 0)
        self.assertEqual(decisions[0]["source"], "provided_transcript")

    def test_rejects_evidence_from_another_speaker(self):
        utterances = [
            {"id": "U1", "speaker_id": "speaker_0", "role": "unknown", "text_raw": "Any pain?"},
            {"id": "U2", "speaker_id": "speaker_1", "role": "unknown", "text_raw": "Yes."},
        ]
        llm = Mock()
        llm.complete_json.return_value = {
            "roles": [
                {"speaker_id": "speaker_0", "role": "doctor", "evidence": ["U2"]},
                {"speaker_id": "speaker_1", "role": "patient", "evidence": ["U2"]},
            ]
        }

        with self.assertRaisesRegex(ValueError, "invalid utterance IDs"):
            assign_speaker_roles(llm, utterances)

    def test_rejects_missing_speaker_predictions(self):
        utterances = [
            {"id": "U1", "speaker_id": "speaker_0", "role": "unknown", "text_raw": "Any pain?"},
            {"id": "U2", "speaker_id": "speaker_1", "role": "unknown", "text_raw": "Yes."},
        ]
        llm = Mock()
        llm.complete_json.return_value = {
            "roles": [{"speaker_id": "speaker_0", "role": "doctor", "evidence": ["U1"]}]
        }

        with self.assertRaisesRegex(ValueError, "omitted speakers"):
            assign_speaker_roles(llm, utterances)


if __name__ == "__main__":
    unittest.main()
