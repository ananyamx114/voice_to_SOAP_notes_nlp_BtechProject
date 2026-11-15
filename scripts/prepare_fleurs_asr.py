"""Prepare pinned FLEURS whole-clip supervision; keep official splits and raw audio."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import math

import pyarrow.parquet as pq
import soundfile as sf

from voice_to_soap.asr_component import file_sha256
from voice_to_soap.audio import load_audio
from voice_to_soap.asr_training_data import encode_labels
from voice_to_soap.io_utils import write_json

REVISION = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"


def select_candidates(rows, seed, count=None, seconds=None, accepts=None):
    """Choose a seeded subset from metadata, never predictions or test scores."""
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    selected, rejected, duration = set(), [], 0.0
    for index in indices:
        row = rows[index]
        length = row["num_samples"] / 16000
        reason = None
        if not row["raw_transcription"].strip() or not 0 < length <= 30:
            reason = "empty_target_or_duration_above_30s"
        elif accepts is not None and not accepts(row):
            reason = "target_exceeds_decoder_context"
        if reason:
            rejected.append({"source_index": index, "reason": reason})
            continue
        selected.add(index)
        duration += length
        if (seconds is not None and duration >= seconds) or (count is not None and len(selected) >= count):
            return selected, rejected
    raise ValueError("Eligible source exhausted before the requested count/duration; no partial training selection")


def prepare(source, output, train_count, validation_count, seed, test_count=0, train_seconds=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if min(train_count, validation_count) < 1 or test_count < 0:
        raise ValueError("Select positive train and validation counts")
    if train_seconds is not None and (not math.isfinite(train_seconds) or train_seconds <= 0 or test_count):
        raise ValueError("Positive training duration is incompatible with test-only preparation")
    manifest = json.loads((source / "download_manifest.json").read_text(encoding="utf-8"))
    if manifest["repository"] != "google/fleurs" or manifest["revision"] != REVISION:
        raise ValueError("Expected the pinned FLEURS training selection")
    if manifest.get("license_from_card") != "cc-by-4.0" or file_sha256(source / "README.md") != manifest.get("source_card_sha256"):
        raise ValueError("Retain the verified source card and attribution before preparing FLEURS")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "repository": "google/fleurs", "revision": REVISION,
        "selection_seed": seed, "requested_per_language": {"test": test_count} if test_count else {"train": train_count, "validation": validation_count},
        "minimum_training_audio_seconds_per_language": train_seconds,
        "files": [], "limitations": ["General read speech, not clinical-domain supervision.",
            "Dataset clip/transcript pairing is not a new human review.",
            "Test-only preparations never feed the training or validation manifests."]}
    evaluation_rows = []
    tokenizer = None
    if train_seconds is not None:
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3",
            revision="06f233fe06e710322aca913c1bc4249a0d71fce1")
    try:
        selections = (("test", test_count),) if test_count else (("train", train_count), ("validation", validation_count))
        for split, requested in selections:
            with (output / (split + ".jsonl")).open("w", encoding="utf-8") as stream:
                for language in ("hi", "mr"):
                    relative = f"{language}_in/{split}/0000.parquet"
                    archive = source / relative
                    expected = next(r for r in manifest["files"] if r["path"] == relative)
                    checksum = file_sha256(archive)
                    if checksum != expected["sha256"] or checksum != expected["upstream_sha256"]:
                        raise ValueError("FLEURS source differs from pinned verified bytes")
                    parquet = pq.ParquetFile(archive)
                    if train_seconds is None and requested > parquet.metadata.num_rows:
                        raise ValueError("Requested more examples than the source split contains")
                    indices = list(range(parquet.metadata.num_rows))
                    random.Random(f"{seed}-{split}-{language}").shuffle(indices)
                    # Select a deterministic candidate pool before looking at ASR output.
                    selected = set(indices[:requested])
                    preselection_rejected = []
                    if train_seconds is not None:
                        metadata = parquet.read(columns=["id", "num_samples", "raw_transcription"]).to_pylist()
                        def fits(row):
                            try:
                                encode_labels(tokenizer, row["raw_transcription"].strip(), language, 448)
                                return True
                            except ValueError:
                                return False
                        selected, preselection_rejected = select_candidates(metadata,
                            f"{seed}-{split}-{language}",
                            seconds=train_seconds + 30 if split == "train" else None,
                            count=requested if split != "train" else None, accepts=fits)
                    accepted, rejected, offset = [], [], 0
                    seen_audio = set()
                    for batch in parquet.iter_batches(batch_size=8):
                        if all(offset + i not in selected for i in range(batch.num_rows)):
                            offset += batch.num_rows
                            continue
                        for index, row in enumerate(batch.to_pylist(), offset):
                            if index not in selected:
                                continue
                            uid = f"fleurs-{language}-{split}-{index:05d}"
                            payload = row["audio"]["bytes"]
                            if not payload:
                                raise ValueError("FLEURS row has no embedded audio")
                            original = output / (uid + ".original")
                            original.write_bytes(payload)
                            waveform, rate = load_audio(original, 16000)
                            prepared = output / (uid + ".flac")
                            sf.write(prepared, waveform, rate, subtype="PCM_24")
                            duration = sf.info(prepared).duration
                            text = row["raw_transcription"].strip()
                            write_json(output / (uid + ".source.json"),
                                {k: v for k, v in row.items() if k != "audio"})
                            if not text or not 0 < duration <= 30:
                                rejected.append({"id": uid, "duration": duration,
                                                 "reason": "empty_target_or_duration_above_30s"})
                                continue
                            source_hash = file_sha256(prepared)
                            if train_seconds is not None and source_hash in seen_audio:
                                rejected.append({"id": uid, "reason": "duplicate_audio"})
                                continue
                            seen_audio.add(source_hash)
                            item = {"id": uid, "encounter_id": uid,
                                "group_id": f"fleurs-sentence-{row['id']}",
                                "dataset": "google/fleurs", "dataset_revision": REVISION,
                                "split": split, "language": language, "text": text,
                                "audio_path": prepared.name, "source_sha256": source_hash,
                                "original_audio_path": original.name,
                                "original_audio_sha256": hashlib.sha256(payload).hexdigest(),
                                "start": 0, "end": duration,
                                "alignment": {"method": "dataset_clip_pair", "status": "accepted",
                                    "human_reviewed": False, "source_text_field": "raw_transcription"}}
                            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                            if split == "test":
                                evaluation_rows.append({"encounter_id": uid, "audio_path": prepared.name,
                                    "transcript": "", "reference_note": "", "language": language,
                                    "split": "test", "reference_transcript": text})
                            accepted.append({"id": uid, "source_sentence_id": row["id"], "duration": duration})
                        offset += batch.num_rows
                    if not accepted:
                        raise ValueError("No selected FLEURS examples passed the supervision contract")
                    accepted_seconds = sum(item["duration"] for item in accepted)
                    if train_seconds is not None and split == "train" and accepted_seconds < train_seconds:
                        raise ValueError("Decoded accepted audio does not meet the training duration; no partial training cohort")
                    report["files"].append({"path": relative, "sha256": checksum, "accepted": accepted,
                        "rejected": rejected, "preselection_rejected": preselection_rejected,
                        "accepted_audio_seconds": accepted_seconds, "source_recordings": parquet.metadata.num_rows})
                    print(f"FLEURS {language}/{split}: {len(accepted)} accepted, {accepted_seconds/60:.2f} minutes, {len(rejected)} rejected", flush=True)
                    write_json(output / "preparation.json", report)
        if evaluation_rows:
            with (output / "test_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(evaluation_rows[0]))
                writer.writeheader()
                writer.writerows(evaluation_rows)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(output / "preparation.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/raw/fleurs-training")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--train-per-language", type=int, default=64)
    parser.add_argument("--validation-per-language", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-per-language", type=int, default=0, help="Prepare test only; never combine into training")
    parser.add_argument("--train-seconds-per-language", type=float, help="Minimum accepted training seconds per language; overrides train count")
    args = parser.parse_args()
    prepare(args.dataset_dir, args.outdir, args.train_per_language, args.validation_per_language, args.seed,
            args.test_per_language, args.train_seconds_per_language)


if __name__ == "__main__":
    main()
