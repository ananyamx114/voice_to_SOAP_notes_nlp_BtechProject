import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from voice_to_soap.asr_datasets import (
    evaluate_asr_manifest,
    import_eka_asr_dataset,
)
from voice_to_soap.datasets import load_manifest


def _wav_bytes() -> bytes:
    output = io.BytesIO()
    sf.write(output, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return output.getvalue()


def _write_shard(path: Path, audio: bytes, text: str, language: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([{
        "audio": audio,
        "text": text,
        "file_name": f"sample-{language}.wav",
        "audio_language": language,
        "text_language": language,
        "duration": 0.1,
        "session_id": "session-1",
        "speaker": "speaker-1",
        "type_concept": "symptom",
        "recording_context": "conversation",
        "medical_entities": "[]",
    }])
    pq.write_table(table, path)


class EkaASRDatasetTests(unittest.TestCase):
    def test_exports_audio_references_metadata_and_project_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_root = root / "processed"
            manifest_path = root / "manifests" / "eka-asr.csv"
            audio = _wav_bytes()
            _write_shard(dataset_root / "en" / "test-00000.parquet", audio, "A cough.", "en")
            _write_shard(dataset_root / "hi" / "test-00000.parquet", audio, "खांसी।", "hi")

            result = import_eka_asr_dataset(
                dataset_root, output_root, manifest_path=manifest_path
            )

            self.assertEqual(result["examples_by_language"], {"en": 1, "hi": 1})
            self.assertEqual(result["total_examples"], 2)
            rows = load_manifest(manifest_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["split"], "test")
            self.assertEqual(rows[0]["task"], "asr_evaluation")
            self.assertEqual(rows[0]["transcript"], "")
            self.assertEqual(rows[0]["reference_transcript"], "A cough.")
            self.assertTrue(Path(rows[0]["audio_path"]).is_file())
            self.assertEqual(sf.info(rows[0]["audio_path"]).samplerate, 16000)

    def test_refuses_to_overwrite_existing_export_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_root = root / "processed"
            manifest_path = root / "manifests" / "eka-asr.csv"
            _write_shard(dataset_root / "en" / "test-00000.parquet", _wav_bytes(), "A cough.", "en")
            import_eka_asr_dataset(
                dataset_root,
                output_root,
                manifest_path=manifest_path,
                languages=["en"],
            )

            with self.assertRaisesRegex(FileExistsError, "Manifest already exists"):
                import_eka_asr_dataset(
                    dataset_root,
                    output_root,
                    manifest_path=manifest_path,
                    languages=["en"],
                )

    def test_batch_evaluation_uses_reference_only_for_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_root = root / "processed"
            manifest_path = root / "manifests" / "eka-asr.csv"
            _write_shard(dataset_root / "en" / "test-00000.parquet", _wav_bytes(), "A cough.", "en")
            imported = import_eka_asr_dataset(
                dataset_root,
                output_root,
                manifest_path=manifest_path,
                languages=["en"],
            )

            with patch(
                "voice_to_soap.audio.transcribe",
                return_value=[{"text": "a cough", "start": None, "end": None, "confidence": None}],
            ) as mock_transcribe:
                result = evaluate_asr_manifest(
                    imported["manifest_path"],
                    model_id="test-model",
                    output_dir=root / "results",
                    languages=["en"],
                )

            self.assertEqual(result["examples"], 1)
            self.assertEqual(result["metrics_by_language"]["en"]["wer"], 0.0)
            mock_transcribe.assert_called_once()
            self.assertEqual(mock_transcribe.call_args.kwargs["language"], "english")
            prediction = json.loads((root / "results" / "predictions.jsonl").read_text())
            self.assertEqual(prediction["prediction"], "a cough")
            metrics = json.loads((root / "results" / "metrics.json").read_text())
            self.assertEqual(metrics["overall"]["wer"], 0.0)

    def test_rejects_non_test_rows_for_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "dataset"
            output_root = root / "processed"
            manifest_path = root / "manifests" / "eka-asr.csv"
            _write_shard(dataset_root / "en" / "test-00000.parquet", _wav_bytes(), "A cough.", "en")
            imported = import_eka_asr_dataset(
                dataset_root,
                output_root,
                manifest_path=manifest_path,
                languages=["en"],
            )
            manifest_path = Path(imported["manifest_path"])
            with manifest_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
                fields = list(rows[0])
            rows[0]["split"] = "train"
            with manifest_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "not marked as an ASR test"):
                evaluate_asr_manifest(
                    manifest_path,
                    model_id="test-model",
                    output_dir=root / "results",
                    languages=["en"],
                )


if __name__ == "__main__":
    unittest.main()
