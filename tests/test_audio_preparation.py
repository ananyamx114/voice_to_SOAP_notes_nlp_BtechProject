import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from voice_to_soap.audio import load_audio, prepare_audio_file
from voice_to_soap.cli import main


class AudioPreparationTests(unittest.TestCase):
    def test_converts_stereo_audio_to_mono_16khz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "original.wav"
            output = root / "prepared" / "mono.wav"
            sample_rate = 8000
            times = np.arange(sample_rate, dtype=np.float32) / sample_rate
            signal = np.column_stack((
                0.4 * np.sin(2 * np.pi * 440 * times),
                0.4 * np.sin(2 * np.pi * 440 * times),
            ))
            sf.write(source, signal, sample_rate)

            metadata = prepare_audio_file(source, output)
            prepared, rate = sf.read(output, always_2d=True)

            self.assertEqual(rate, 16000)
            self.assertEqual(prepared.shape[1], 1)
            self.assertAlmostEqual(metadata["source_duration_seconds"], 1.0, places=2)
            self.assertAlmostEqual(metadata["prepared_duration_seconds"], 1.0, places=2)
            self.assertEqual(metadata["source_channels"], 2)
            self.assertEqual(metadata["prepared_channels"], 1)
            self.assertEqual(metadata["sample_count"], 16000)

    def test_load_audio_returns_finite_mono_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "audio.wav"
            samples = np.ones((800, 2), dtype=np.float32) * 0.2
            sf.write(source, samples, 8000)

            audio, rate = load_audio(source, 16000)

            self.assertEqual(rate, 16000)
            self.assertEqual(audio.ndim, 1)
            self.assertTrue(np.isfinite(audio).all())

    def test_refuses_to_overwrite_original_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "audio.wav"
            sf.write(source, np.ones(800, dtype=np.float32) * 0.1, 8000)

            with self.assertRaisesRegex(ValueError, "differ from"):
                prepare_audio_file(source, source)

    def test_rejects_nonpositive_target_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "audio.wav"
            sf.write(source, np.ones(800, dtype=np.float32) * 0.1, 8000)

            with self.assertRaisesRegex(ValueError, "positive"):
                prepare_audio_file(source, Path(tmp) / "out.wav", target_rate=0)

    def test_reports_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "not found"):
                prepare_audio_file(Path(tmp) / "missing.wav", Path(tmp) / "out.wav")

    def test_prepare_audio_command_reads_manifest_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "visit.wav"
            sf.write(source, np.ones(8000, dtype=np.float32) * 0.2, 8000)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "encounter_id,audio_path,transcript,reference_note,language,split\n"
                "visit-1,visit.wav,,,mr,train\n",
                encoding="utf-8",
            )
            output = root / "prepared"

            with patch("sys.argv", [
                "voice-to-soap", "prepare-audio", "--manifest", str(manifest),
                "--outdir", str(output),
            ]):
                main()

            self.assertTrue((output / "visit-1" / "audio_mono.wav").is_file())
            self.assertTrue((output / "visit-1" / "audio_metadata.json").is_file())


if __name__ == "__main__":
    unittest.main()
