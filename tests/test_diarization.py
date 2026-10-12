import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from voice_to_soap.audio import assign_speakers, diarize
from voice_to_soap.cli import main


class DiarizationTests(unittest.TestCase):
    def test_diarize_returns_normalized_speaker_segments(self):
        annotation = Mock()
        annotation.itertracks.return_value = iter([
            (SimpleNamespace(start=0.1, end=1.0), "track-a", "SPEAKER_01"),
            (SimpleNamespace(start=1.2, end=2.0), "track-b", "SPEAKER_00"),
            (SimpleNamespace(start=2.2, end=2.8), "track-c", "SPEAKER_01"),
        ])
        result = SimpleNamespace(
            exclusive_speaker_diarization=annotation,
            speaker_diarization=None,
        )
        model = Mock()
        model.return_value = result

        with patch("voice_to_soap.audio._load_diarization_pipeline", return_value=model):
            segments = diarize(
                "consultation.wav",
                "pyannote/speaker-diarization-community-1",
                "test-token",
                device="cuda:0",
                num_speakers=2,
            )

        self.assertEqual(
            segments,
            [
                {"start": 0.1, "end": 1.0, "speaker_id": "speaker_0"},
                {"start": 1.2, "end": 2.0, "speaker_id": "speaker_1"},
                {"start": 2.2, "end": 2.8, "speaker_id": "speaker_0"},
            ],
        )
        model.assert_called_once_with("consultation.wav", num_speakers=2)

    def test_uses_regular_annotation_if_exclusive_is_unavailable(self):
        annotation = Mock()
        annotation.itertracks.return_value = iter([
            (SimpleNamespace(start=0.0, end=0.5), "track-a", "speaker-a"),
        ])
        result = SimpleNamespace(exclusive_speaker_diarization=None, speaker_diarization=annotation)
        model = Mock(return_value=result)

        with patch("voice_to_soap.audio._load_diarization_pipeline", return_value=model):
            segments = diarize("audio.wav", "model-id", "token")

        self.assertEqual(segments[0]["speaker_id"], "speaker_0")

    def test_requires_model_token_and_valid_speaker_constraints(self):
        with self.assertRaisesRegex(RuntimeError, "diarization_model_id"):
            diarize("audio.wav", None, "token")
        with self.assertRaisesRegex(RuntimeError, "token is missing"):
            diarize("audio.wav", "model-id", None)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            diarize("audio.wav", "model-id", "token", min_speakers=3, max_speakers=2)
        with self.assertRaisesRegex(ValueError, "not both"):
            diarize("audio.wav", "model-id", "token", num_speakers=2, min_speakers=2)

    def test_assigns_speaker_with_largest_timestamp_overlap(self):
        utterances = [
            {"id": "U1", "start": 0.2, "end": 0.9, "speaker_id": "unknown"},
            {"id": "U2", "start": None, "end": None, "speaker_id": "unknown"},
        ]
        segments = [
            {"start": 0.0, "end": 0.6, "speaker_id": "speaker_0"},
            {"start": 0.6, "end": 1.0, "speaker_id": "speaker_1"},
        ]

        result = assign_speakers(utterances, segments)

        self.assertEqual(result[0]["speaker_id"], "speaker_0")
        self.assertEqual(result[1]["speaker_id"], "unknown")
        self.assertEqual(utterances[0]["speaker_id"], "unknown")

    @patch("voice_to_soap.cli.diarize")
    def test_cli_saves_diarization_output(self, mock_diarize):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "consultation.wav"
            audio.touch()
            output = Path(tmp) / "diarization.json"
            mock_diarize.return_value = [
                {"start": 0.0, "end": 1.0, "speaker_id": "speaker_0"}
            ]
            with patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
                with patch("sys.argv", [
                    "voice-to-soap", "diarize-audio",
                    "--audio", str(audio),
                    "--out", str(output),
                    "--config", "configs/base.yaml",
                    "--num-speakers", "2",
                ]):
                    main()
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["model_id"], "pyannote/speaker-diarization-community-1")
            self.assertEqual(saved["segments"][0]["speaker_id"], "speaker_0")
            self.assertEqual(mock_diarize.call_args.kwargs["num_speakers"], 2)

    @patch("voice_to_soap.cli.load_manifest")
    @patch("voice_to_soap.cli.diarize")
    def test_manifest_cli_limits_samples_per_language_and_writes_jsonl(
        self, mock_diarize, mock_load_manifest
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_paths = []
            rows = []
            for language, count in (("en", 3), ("hi", 3)):
                for index in range(count):
                    audio_path = root / f"{language}-{index}.mp3"
                    audio_path.touch()
                    audio_paths.append(audio_path)
                    rows.append({
                        "encounter_id": f"{language}-{index}",
                        "language": language,
                        "audio_path": str(audio_path),
                    })
            mock_load_manifest.return_value = rows
            mock_diarize.return_value = [
                {"start": 0.0, "end": 1.0, "speaker_id": "speaker_0"}
            ]
            output = root / "diarization.jsonl"

            with patch.dict("os.environ", {"HF_TOKEN": "test-token"}):
                with patch("sys.argv", [
                    "voice-to-soap", "diarize-manifest",
                    "--manifest", str(root / "manifest.csv"),
                    "--out", str(output),
                    "--config", "configs/base.yaml",
                    "--limit-per-language", "2",
                ]):
                    main()

            saved = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(saved), 4)
            self.assertEqual(
                [item["encounter_id"] for item in saved],
                ["en-0", "en-1", "hi-0", "hi-1"],
            )
            self.assertEqual(mock_diarize.call_count, 4)
            self.assertEqual(saved[0]["segments"][0]["speaker_id"], "speaker_0")


if __name__ == "__main__":
    unittest.main()
