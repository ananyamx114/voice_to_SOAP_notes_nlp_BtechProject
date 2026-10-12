import tempfile
import unittest
from pathlib import Path

from voice_to_soap.datasets import load_manifest


HEADER = "encounter_id,audio_path,transcript,reference_note,language,split\n"


class ManifestTests(unittest.TestCase):
    def test_loads_transcript_only_encounter(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(
                HEADER + 'visit-1,,"[patient] I have a cough.",,hi-en,train\n',
                encoding="utf-8",
            )

            rows = load_manifest(manifest)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["encounter_id"], "visit-1")
            self.assertEqual(rows[0]["transcript"], "[patient] I have a cough.")
            self.assertEqual(rows[0]["split"], "train")

    def test_resolves_relative_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = root / "audio" / "visit.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"test placeholder")
            manifest = root / "manifest.csv"
            manifest.write_text(
                HEADER + "visit-1,audio/visit.wav,,,mr,validation\n",
                encoding="utf-8",
            )

            row = load_manifest(manifest)[0]

            self.assertEqual(row["audio_path"], str(audio.resolve()))

    def test_rejects_duplicate_encounter_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(
                HEADER
                + 'visit-1,,"hello",note,hi,train\n'
                + 'visit-1,,"hello",note,hi,test\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate encounter_id"):
                load_manifest(manifest)

    def test_rejects_missing_data_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(HEADER + "visit-1,,,,hi,train\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "needs audio_path or transcript"):
                load_manifest(manifest)

    def test_rejects_missing_audio_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(
                HEADER + "visit-1,missing.wav,,,mr,train\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Audio file"):
                load_manifest(manifest)

    def test_rejects_missing_language_or_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(
                HEADER + 'visit-1,,"[doctor] Hello.",,mr,\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "no split"):
                load_manifest(manifest)

    def test_rejects_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "manifest.csv"
            manifest.write_text(HEADER, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "no encounters"):
                load_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
