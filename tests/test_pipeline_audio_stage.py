import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_to_soap.pipeline import _audio_stage
from unittest.mock import Mock


class PipelineAudioStageTests(unittest.TestCase):
    def test_vad_segments_flow_to_asr_and_timestamps_remain_on_full_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "visit.wav"
            sf.write(audio_path, np.ones(32000, dtype=np.float32) * 0.1, 16000)
            row = {"encounter_id": "visit-1", "audio_path": str(audio_path)}
            config = {
                "audio": {
                    "target_sample_rate": 16000,
                    "vad": {"enabled": True},
                    "asr_model_id": "openai/whisper-large-v3",
                    "asr_device": "cuda:0",
                    "diarization_model_id": "pyannote/speaker-diarization-community-1",
                    "diarization": {"enabled": True, "num_speakers": 2},
                    "role_assignment": {"enabled": True},
                }
            }
            speech_segments = [{"start_seconds": 0.5, "end_seconds": 1.5}]
            words = [{"text": "hello", "start": 0.7, "end": 0.9, "confidence": None}]
            speaker_segments = [{"start": 0.6, "end": 1.0, "speaker_id": "speaker_0"}]
            role_utterances = [{
                "id": "U1",
                "start": 0.7,
                "end": 0.9,
                "speaker_id": "speaker_0",
                "role": "patient",
                "text_raw": "hello",
                "text_normalized": "hello",
                "confidence": None,
            }]

            fake_llm = Mock()
            with patch("voice_to_soap.pipeline.detect_speech", return_value=speech_segments) as vad:
                with patch(
                    "voice_to_soap.pipeline.transcribe_speech_segments", return_value=words
                ) as asr:
                    with patch("voice_to_soap.pipeline.diarize", return_value=speaker_segments) as diarizer:
                        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
                            with patch(
                                "voice_to_soap.pipeline.assign_speaker_roles",
                                return_value=(
                                    role_utterances,
                                    [{"speaker_id": "speaker_0", "role": "patient", "evidence": ["U1"]}],
                                ),
                            ) as role_assigner:
                                utterances, metadata = _audio_stage(row, config, fake_llm)

            vad.assert_called_once()
            asr.assert_called_once()
            diarizer.assert_called_once()
            role_assigner.assert_called_once()
            self.assertEqual(asr.call_args.args[2], speech_segments)
            self.assertEqual(utterances[0]["text_raw"], "hello")
            self.assertAlmostEqual(utterances[0]["start"], 0.7)
            self.assertAlmostEqual(utterances[0]["end"], 0.9)
            self.assertEqual(utterances[0]["speaker_id"], "speaker_0")
            self.assertEqual(utterances[0]["role"], "patient")
            self.assertEqual(metadata["speech_segments"], speech_segments)
            self.assertEqual(metadata["diarization_segments"], speaker_segments)
            self.assertEqual(metadata["role_assignments"][0]["role"], "patient")
            self.assertEqual(metadata["duration_seconds"], 2.0)


if __name__ == "__main__":
    unittest.main()
