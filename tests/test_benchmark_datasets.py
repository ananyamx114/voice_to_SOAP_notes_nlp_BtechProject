import io
from pathlib import Path
import tarfile
import tempfile
import unittest

import numpy as np
import soundfile as sf

from voice_to_soap.benchmark_datasets import prepare_betrac
from voice_to_soap.datasets import load_manifest


class BenchmarkDatasetTests(unittest.TestCase):
    def archive(self, root, unsafe=False, incomplete=False):
        audio = io.BytesIO()
        sf.write(audio, np.zeros(1600), 16000, format="WAV")
        archive = root / "sample.tar"
        with tarfile.open(archive, "w") as stream:
            files = {".opus": audio.getvalue(),
                     ".transcript.txt": b"DOCTOR: Hello.\nPATIENT: I have a cough.",
                     ".soap.txt": b"Subjective: Cough.", ".json": b"{}"}
            if incomplete:
                files.pop(".json")
            for suffix, payload in files.items():
                member = tarfile.TarInfo(("../escape" if unsafe else "dialog_1") + suffix)
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))
        return archive

    def test_manifests_keep_gold_text_out_of_asr_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = prepare_betrac(self.archive(root), root / "out")
            audio = load_manifest(report["audio_manifest"])[0]
            gold = load_manifest(report["gold_manifest"])[0]
            self.assertEqual(audio["transcript"], "")
            self.assertEqual(audio["split"], "validation")
            self.assertEqual(audio["source_split"], "dev")
            self.assertIn("[patient] I have a cough.", gold["transcript"])
            self.assertIn("I have a cough.", audio["reference_transcript"])
            self.assertEqual(report["exported_samples"], 1)
            self.assertTrue(Path(audio["audio_path"]).is_file())
            with self.assertRaises(FileExistsError):
                prepare_betrac(root / "sample.tar", root / "out")

    def test_rejects_traversal_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                prepare_betrac(self.archive(root, unsafe=True), root / "out")
            self.assertFalse((root / "out").exists())

    def test_rejects_missing_reference_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                prepare_betrac(self.archive(root, incomplete=True), root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
