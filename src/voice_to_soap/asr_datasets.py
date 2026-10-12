from __future__ import annotations

import csv
import io
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import soundfile as sf


EKA_MANIFEST_FIELDS = (
    "encounter_id",
    "dataset",
    "task",
    "split",
    "language",
    "audio_path",
    "transcript",
    "reference_transcript",
    "reference_note",
    "source_file",
    "source_row",
    "audio_sample_rate",
    "duration_seconds",
    "session_id",
    "speaker",
    "recording_context",
    "type_concept",
    "medical_entities",
)
LANGUAGE_NAMES = {"en": "english", "hi": "hindi"}
AUDIO_EXTENSIONS = {"MP3": ".mp3", "WAV": ".wav"}


def _audio_bytes(value: Any, encounter_id: str) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if not isinstance(value, (bytes, bytearray)) or not value:
        raise ValueError(f"Eka sample {encounter_id} has no embedded audio bytes")
    return bytes(value)


def _serialize_metadata(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_asr_text(text: str) -> str:
    without_punctuation = "".join(
        character
        for character in text.casefold()
        if not unicodedata.category(character).startswith("P")
    )
    return " ".join(without_punctuation.split())


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from jiwer import cer, wer

    references = [
        _normalize_asr_text(item["reference_transcript"]) for item in rows
    ]
    hypotheses = [_normalize_asr_text(item["prediction"]) for item in rows]
    return {
        "examples": len(rows),
        "wer": float(wer(references, hypotheses)),
        "cer": float(cer(references, hypotheses)),
    }


def _iter_parquet_rows(path: Path) -> Iterable[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Reading Eka Parquet shards requires pyarrow; install the project's "
            "Hugging Face datasets dependencies."
        ) from exc

    columns = [
        "audio",
        "text",
        "file_name",
        "audio_language",
        "text_language",
        "duration",
        "session_id",
        "speaker",
        "type_concept",
        "recording_context",
        "medical_entities",
    ]
    parquet_file = pq.ParquetFile(path)
    missing = set(columns) - set(parquet_file.schema_arrow.names)
    if missing:
        raise ValueError(f"Eka shard {path} is missing columns: {sorted(missing)}")
    for batch in parquet_file.iter_batches(columns=columns, batch_size=32):
        yield from batch.to_pylist()


def import_eka_asr_dataset(
    dataset_dir: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path = "data/manifests/eka-medical-asr-evaluation.csv",
    languages: Iterable[str] = ("en", "hi"),
    limit_per_language: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_root = Path(dataset_dir).resolve()
    output_root = Path(output_dir).resolve()
    destination_manifest = Path(manifest_path).resolve()
    selected_languages = tuple(dict.fromkeys(language.strip().lower() for language in languages))
    if not selected_languages:
        raise ValueError("Select at least one Eka language subset")
    unknown_languages = set(selected_languages) - LANGUAGE_NAMES.keys()
    if unknown_languages:
        raise ValueError(f"Unsupported Eka language subset(s): {sorted(unknown_languages)}")
    if limit_per_language is not None and limit_per_language < 1:
        raise ValueError("limit_per_language must be positive when provided")

    if destination_manifest.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {destination_manifest}; pass overwrite=True "
            "to refresh it"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    for language in selected_languages:
        shard_paths = sorted((source_root / language).glob("test-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No test Parquet shards found for Eka language {language!r} under "
                f"{source_root / language}"
            )
        count = 0
        audio_dir = output_root / language / "test" / "audio"
        for shard_path in shard_paths:
            for row in _iter_parquet_rows(shard_path):
                if limit_per_language is not None and count >= limit_per_language:
                    break
                encounter_id = f"eka-{language}-test-{count + 1:05d}"
                reference = row.get("text")
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError(f"Eka sample {encounter_id} has an empty reference text")
                audio_data = _audio_bytes(row.get("audio"), encounter_id)
                try:
                    audio_info = sf.info(io.BytesIO(audio_data))
                except (RuntimeError, OSError) as exc:
                    raise ValueError(
                        f"Could not decode embedded audio for Eka sample {encounter_id}: {exc}"
                    ) from exc
                extension = AUDIO_EXTENSIONS.get(audio_info.format.upper())
                if extension is None:
                    raise ValueError(
                        f"Unsupported audio format {audio_info.format!r} in "
                        f"Eka sample {encounter_id}"
                    )

                audio_path = audio_dir / f"{encounter_id}{extension}"
                if audio_path.exists() and not overwrite:
                    raise FileExistsError(
                        f"Audio export already exists: {audio_path}; pass overwrite=True "
                        "to refresh it"
                    )
                audio_dir.mkdir(parents=True, exist_ok=True)
                audio_path.write_bytes(audio_data)
                relative_audio_path = Path(
                    os.path.relpath(audio_path, destination_manifest.parent)
                ).as_posix()
                manifest_rows.append({
                    "encounter_id": encounter_id,
                    "dataset": "eka-medical-asr-evaluation",
                    "task": "asr_evaluation",
                    "split": "test",
                    "language": language,
                    "audio_path": relative_audio_path,
                    "transcript": "",
                    "reference_transcript": reference.strip(),
                    "reference_note": "",
                    "source_file": str(shard_path.relative_to(source_root)),
                    "source_row": str(count),
                    "audio_sample_rate": str(audio_info.samplerate),
                    "duration_seconds": str(row.get("duration", audio_info.duration)),
                    "session_id": _serialize_metadata(row.get("session_id")),
                    "speaker": _serialize_metadata(row.get("speaker")),
                    "recording_context": _serialize_metadata(row.get("recording_context")),
                    "type_concept": _serialize_metadata(row.get("type_concept")),
                    "medical_entities": _serialize_metadata(row.get("medical_entities")),
                })
                count += 1
            if limit_per_language is not None and count >= limit_per_language:
                break
        if count == 0:
            raise ValueError(f"No Eka examples were imported for language {language!r}")
        counts[language] = count

    temporary_manifest = destination_manifest.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EKA_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(destination_manifest)
    return {
        "manifest_path": str(destination_manifest),
        "examples_by_language": counts,
        "total_examples": sum(counts.values()),
        "audio_files": sum(counts.values()),
        "split": "test",
    }


def evaluate_asr_manifest(
    manifest_path: str | Path,
    model_id: str,
    output_dir: str | Path,
    device: str = "auto",
    sample_rate: int = 16000,
    languages: Iterable[str] = ("en", "hi"),
    limit_per_language: int | None = None,
    chunk_length_s: int = 30,
    stride_length_s: int = 5,
) -> dict[str, Any]:
    from .audio import load_audio, transcribe
    from .datasets import load_manifest

    selected_languages = set(language.strip().lower() for language in languages)
    if not selected_languages or not selected_languages <= LANGUAGE_NAMES.keys():
        raise ValueError("languages must contain one or both of: en, hi")
    if limit_per_language is not None and limit_per_language < 1:
        raise ValueError("limit_per_language must be positive when provided")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    rows = load_manifest(manifest_path)
    predictions: list[dict[str, Any]] = []
    seen_by_language: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        language = row.get("language", "").lower()
        if language not in selected_languages:
            continue
        if row.get("task") != "asr_evaluation" or row.get("split") != "test":
            raise ValueError(
                f"Encounter {row['encounter_id']!r} is not marked as an ASR test example"
            )
        reference = row.get("reference_transcript", "").strip()
        if not reference:
            raise ValueError(
                f"ASR test encounter {row['encounter_id']!r} has no reference_transcript"
            )
        if (
            limit_per_language is not None
            and seen_by_language[language] >= limit_per_language
        ):
            continue

        waveform, rate = load_audio(row["audio_path"], sample_rate)
        chunks = transcribe(
            waveform,
            rate,
            model_id,
            device=device,
            language=LANGUAGE_NAMES[language],
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
        )
        hypothesis = " ".join(chunk["text"] for chunk in chunks).strip()
        if not hypothesis:
            raise RuntimeError(
                f"ASR model {model_id!r} returned an empty transcript for "
                f"{row['encounter_id']!r}"
            )
        predictions.append({
            "encounter_id": row["encounter_id"],
            "language": language,
            "recording_context": row.get("recording_context", ""),
            "audio_path": row["audio_path"],
            "reference_transcript": reference,
            "prediction": hypothesis,
        })
        seen_by_language[language] += 1

    if not predictions:
        raise ValueError("No matching ASR test examples were found in the manifest")

    metrics_by_language: dict[str, dict[str, Any]] = {}
    for language in sorted({item["language"] for item in predictions}):
        language_rows = [item for item in predictions if item["language"] == language]
        metrics_by_language[language] = _metric_summary(language_rows)
    metrics_by_context: dict[str, dict[str, Any]] = {}
    for context in sorted({item["recording_context"] for item in predictions}):
        context_rows = [
            item for item in predictions if item["recording_context"] == context
        ]
        metrics_by_context[context or "unknown"] = _metric_summary(context_rows)
    results = {
        "model_id": model_id,
        "split": "test",
        "examples": len(predictions),
        "metrics_by_language": metrics_by_language,
        "metrics_by_recording_context": metrics_by_context,
        "overall": _metric_summary(predictions),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for prediction in predictions:
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    (destination / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
