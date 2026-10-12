import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from voice_to_soap.audio import transcribe, transcribe_speech_segments
from voice_to_soap.cli import main


class AsrTests(unittest.TestCase):
    def test_rejects_unconfigured_model(self):
        with self.assertRaisesRegex(RuntimeError, "model"):
            transcribe(np.ones(16000, dtype=np.float32), 16000, None)

    @patch("voice_to_soap.audio._load_asr_pipeline")
    @patch("voice_to_soap.audio._resolve_asr_device", return_value="cuda:0")
    def test_calls_configured_model_with_word_timestamps(self, resolve_device, load_pipeline):
        fake_asr = Mock(return_value={
            "text": "I have a cough",
            "chunks": [
                {"text": " I", "timestamp": (0.0, 0.2)},
                {"text": " have", "timestamp": (0.2, 0.5)},
            ],
        })
        load_pipeline.return_value = fake_asr
        audio = np.ones(16000, dtype=np.float32)

        words = transcribe(
            audio,
            16000,
            "openai/whisper-large-v3",
            language="marathi",
            chunk_length_s=30,
            stride_length_s=5,
        )

        resolve_device.assert_called_once_with("auto")
        load_pipeline.assert_called_once_with("openai/whisper-large-v3", "cuda:0")
        self.assertEqual(words[0]["text"], "I")
        self.assertEqual(words[1]["start"], 0.2)
        fake_asr.assert_called_once()
        call_kwargs = fake_asr.call_args.kwargs
        self.assertEqual(call_kwargs["return_timestamps"], "word")
        self.assertEqual(call_kwargs["stride_length_s"], (5, 5))
        self.assertEqual(call_kwargs["generate_kwargs"], {
            "task": "transcribe",
            "language": "marathi",
        })

    @patch("voice_to_soap.audio._load_asr_pipeline")
    @patch("voice_to_soap.audio._resolve_asr_device", return_value="cpu")
    def test_returns_full_transcript_when_model_has_no_chunks(self, _resolve_device, load_pipeline):
        fake_asr = Mock(return_value={"text": "नमस्कार"})
        load_pipeline.return_value = fake_asr

        result = transcribe(np.ones(4000, dtype=np.float32), 16000, "openai/whisper-large-v3")

        self.assertEqual(result, [{
            "text": "नमस्कार",
            "start": None,
            "end": None,
            "confidence": None,
        }])

    def test_rejects_invalid_chunk_options(self):
        with self.assertRaisesRegex(ValueError, "chunk/stride"):
            transcribe(
                np.ones(16000, dtype=np.float32),
                16000,
                "openai/whisper-large-v3",
                chunk_length_s=10,
                stride_length_s=5,
            )

    @patch("voice_to_soap.audio.transcribe")
    def test_transcribes_each_vad_segment_and_restores_audio_timestamps(self, mock_transcribe):
        mock_transcribe.side_effect = [
            [{"text": " hello", "start": 0.1, "end": 0.3, "confidence": None}],
            [{"text": " world", "start": 0.0, "end": 0.2, "confidence": None}],
        ]
        audio = np.ones(10 * 16000, dtype=np.float32)
        segments = [
            {"start_seconds": 1.0, "end_seconds": 2.0},
            {"start_seconds": 5.0, "end_seconds": 6.0},
        ]

        result = transcribe_speech_segments(
            audio,
            16000,
            segments,
            "openai/whisper-large-v3",
            language="marathi",
        )

        self.assertEqual([word["text"] for word in result], [" hello", " world"])
        self.assertEqual(result[0]["start"], 1.1)
        self.assertEqual(result[0]["end"], 1.3)
        self.assertEqual(result[1]["start"], 5.0)
        self.assertEqual(result[1]["end"], 5.2)
        self.assertEqual(mock_transcribe.call_count, 2)
        self.assertEqual(len(mock_transcribe.call_args_list[0].args[0]), 16000)
        self.assertEqual(len(mock_transcribe.call_args_list[1].args[0]), 16000)

    def test_fails_if_vad_found_no_speech(self):
        with self.assertRaisesRegex(RuntimeError, "no speech"):
            transcribe_speech_segments(
                np.ones(16000, dtype=np.float32),
                16000,
                [],
                "openai/whisper-large-v3",
            )

    @patch("voice_to_soap.cli.transcribe")
    def test_transcribe_audio_command_saves_result(self, mock_transcribe):
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "sample.wav"
            output_path = Path(tmp) / "transcript.json"
            sf.write(audio_path, np.ones(16000, dtype=np.float32) * 0.1, 16000)
            mock_transcribe.return_value = [{
                "text": " Hello",
                "start": 0.0,
                "end": 0.5,
                "confidence": None,
            }]

            with patch("sys.argv", [
                "voice-to-soap", "transcribe-audio",
                "--audio", str(audio_path),
                "--out", str(output_path),
                "--config", "configs/base.yaml",
            ]):
                main()

            import json
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["model_id"], "openai/whisper-small")
            self.assertEqual(saved["text"], "Hello")
            self.assertEqual(saved["chunks"][0]["start"], 0.0)
            mock_transcribe.assert_called_once()


if __name__ == "__main__":
    unittest.main()
