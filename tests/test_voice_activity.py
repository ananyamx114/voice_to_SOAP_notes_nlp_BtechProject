import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_to_soap.audio import normalize_speech_segments, save_speech_segments
from voice_to_soap.cli import main


class VoiceActivityTests(unittest.TestCase):
    def test_clamps_sorts_and_merges_segments(self):
        result = normalize_speech_segments(
            [
                {"start": 1.0, "end": 2.0},
                {"start": -0.2, "end": 0.5},
                {"start": 1.8, "end": 3.0},
                {"start": 4.5, "end": 8.0},
                {"start": 3.0, "end": 3.0},
            ],
            duration_seconds=5.0,
        )

        self.assertEqual(
            result,
            [
                {"start_seconds": 0.0, "end_seconds": 0.5},
                {"start_seconds": 1.0, "end_seconds": 3.0},
                {"start_seconds": 4.5, "end_seconds": 5.0},
            ],
        )

    def test_rejects_nonfinite_segment_times(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            normalize_speech_segments(
                [{"start": float("nan"), "end": 1.0}],
                duration_seconds=2.0,
            )

    def test_saves_speech_segments_and_preserves_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample_rate = 1000
            audio = np.arange(5000, dtype=np.float32) / 5000
            segments = save_speech_segments(
                audio,
                sample_rate,
                [
                    {"start_seconds": 1.0, "end_seconds": 2.0},
                    {"start_seconds": 3.0, "end_seconds": 3.5},
                ],
                tmp,
            )

            self.assertEqual(len(segments), 2)
            self.assertAlmostEqual(segments[0]["start_seconds"], 1.0)
            self.assertAlmostEqual(segments[0]["end_seconds"], 2.0)
            saved_audio, saved_rate = sf.read(segments[1]["audio_path"])
            self.assertEqual(saved_rate, sample_rate)
            self.assertEqual(len(saved_audio), 500)

    def test_detect_speech_cli_writes_segment_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original.wav"
            sf.write(original, np.ones(4000, dtype=np.float32) * 0.2, 16000)
            prepared = root / "prepared" / "visit-1" / "audio_mono.wav"
            prepared.parent.mkdir(parents=True)
            sf.write(prepared, np.ones(4000, dtype=np.float32) * 0.2, 16000)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "encounter_id,audio_path,transcript,reference_note,language,split\n"
                f"visit-1,{original},,,mr,train\n",
                encoding="utf-8",
            )
            output = root / "speech"
            fake_segments = [{"start_seconds": 0.05, "end_seconds": 0.15}]

            with patch("voice_to_soap.cli.detect_speech", return_value=fake_segments):
                with patch("sys.argv", [
                    "voice-to-soap", "detect-speech",
                    "--manifest", str(manifest),
                    "--audio-dir", str(root / "prepared"),
                    "--outdir", str(output),
                ]):
                    main()

            segment_file = output / "visit-1" / "segments" / "speech_0001.wav"
            metadata_file = output / "visit-1" / "speech_segments.json"
            self.assertTrue(segment_file.is_file())
            self.assertTrue(metadata_file.is_file())


if __name__ == "__main__":
    unittest.main()
