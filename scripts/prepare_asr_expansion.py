"""Build expanded Hindi/Marathi whole-clip supervision without touching frozen validation.

Uses existing private prepared ASR archive and already-downloaded pinned FLEURS
training Parquet files. Downloads nothing; keeps clinical English and validation
identities/text/boundaries/audio bytes. Never reads FLEURS test for selection.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import posixpath
import sys
import tarfile
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-archive", type=Path, required=True)
    parser.add_argument("--fleurs-source", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="Locally cached Whisper tokenizer directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve existing experiment data"
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from transformers import WhisperTokenizer
    from voice_to_soap.asr_training_data import check_disjoint, encode_labels, read_segments
    inv = json.loads(args.prepared_archive.with_suffix(".inventory.json").read_text())
    assert digest(args.prepared_archive) == inv["sha256"]
    source_manifest = json.loads((args.fleurs_source / "download_manifest.json").read_text())
    revision = "168de341b3db6859a9bac1c50a2ef5e3b47647e0"
    assert source_manifest["repository"] == "google/fleurs" and source_manifest["revision"] == revision
    assert source_manifest["license_from_card"] == "cc-by-4.0"
    assert digest(args.fleurs_source / "README.md") == source_manifest["source_card_sha256"]
    tokenizer = WhisperTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    output = args.output.resolve()
    if shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free < 8 * 1024**3:
        raise OSError("Preserve at least8 GiB free before expansion; no data was written")
    output.mkdir(parents=True)
    shutil.copy2(args.fleurs_source / "README.md", output / "FLEURS_SOURCE_CARD.md")
    shutil.copy2(ROOT / "datasets/betrac_2000/SOURCE_DATASET_CARD.md", output / "BETRAC_SOURCE_CARD.md")
    source_records = {r["path"]: r for r in inv["files"]}
    prefix = "data/outputs/asr-lora-experiment-run3/data/"
    rejected = Counter()
    train = []
    validation = []

    def save_audio(row, payload):
        sha = hashlib.sha256(payload).hexdigest()
        assert sha == row["source_sha256"]
        target = output / "audio" / row["language"] / (sha + ".flac")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(payload)
        return {**row, "audio_path": str(target.relative_to(output))}

    with tarfile.open(args.prepared_archive) as tar:
        def manifest(split):
            name = prefix + split + ".jsonl"
            raw = tar.extractfile(name).read()
            assert hashlib.sha256(raw).hexdigest() == source_records[name]["sha256"]
            return [json.loads(line) for line in raw.splitlines() if line.strip()]

        old_train, old_validation = manifest("train"), manifest("validation")
        validation_groups = {r["group_id"] for r in old_validation}
        for row in [r for r in old_train if r["language"] == "en"] + old_validation:
            name = posixpath.normpath(prefix + row["audio_path"])
            if name in source_records:
                raw = tar.extractfile(name).read()
            else:
                raw = (ROOT / name).read_bytes()
            target = save_audio(row, raw)
            (train if row["split"] == "train" else validation).append(target)
    assert len(validation) == 239
    seen_hashes = {r["source_sha256"] for r in train + validation}
    for language in ("hi", "mr"):
        relative = f"{language}_in/train/0000.parquet"
        path = args.fleurs_source / relative
        expected = next(r for r in source_manifest["files"] if r["path"] == relative)
        assert digest(path) == expected["sha256"] == expected["upstream_sha256"]
        index = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8,
                columns=["id", "num_samples", "raw_transcription", "audio"]):
            for source in batch.to_pylist():
                uid = f"fleurs-{language}-train-{index:05d}"
                index += 1
                group = f"fleurs-sentence-{source['id']}"
                text = source["raw_transcription"].strip()
                if group in validation_groups:
                    rejected["validation_group"] += 1
                    continue
                if not text:
                    rejected["empty_text"] += 1
                    continue
                try:
                    encode_labels(tokenizer, text, language, 448)
                except ValueError:
                    rejected["target_exceeds_context"] += 1
                    continue
                payload = source["audio"]["bytes"]
                waveform, rate = sf.read(io.BytesIO(payload), dtype="float32")
                duration = len(waveform) / rate
                if rate != 16000 or waveform.ndim != 1 or not np.isfinite(waveform).all() or not 0 < duration <= 30:
                    rejected["audio_contract"] += 1
                    continue
                if abs(duration - source["num_samples"] / 16000) > 1 / 16000:
                    raise ValueError("Source metadata/audio duration mismatch")
                target = output / "audio" / language / (uid + ".flac")
                if shutil.disk_usage(output).free < 4 * 1024**3:
                    raise OSError("Storage reserve reached; preserve partial data and inspect before continuing")
                target.parent.mkdir(parents=True, exist_ok=True)
                sf.write(target, waveform, rate, subtype="PCM_24")
                sha = digest(target)
                if sha in seen_hashes:
                    target.unlink()
                    rejected["duplicate_audio"] += 1
                    continue
                seen_hashes.add(sha)
                train.append({"id": uid, "encounter_id": uid, "group_id": group,
                    "dataset": "google/fleurs", "dataset_revision": revision, "split": "train",
                    "language": language, "text": text, "audio_path": str(target.relative_to(output)),
                    "source_sha256": sha, "original_audio_sha256": hashlib.sha256(payload).hexdigest(),
                    "start": 0, "end": duration,
                    "alignment": {"status": "accepted", "method": "dataset_clip_pair",
                                  "human_reviewed": False, "source_text_field": "raw_transcription"}})
        print(language, "decoded and accepted", sum(r["language"] == language for r in train), flush=True)
    for split, rows in (("train", train), ("validation", validation)):
        with (output / (split + ".jsonl")).open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    validated_train = read_segments(output / "train.jsonl")
    validated_validation = read_segments(output / "validation.jsonl")
    audit = check_disjoint(validated_train, validated_validation)
    identity = ("id", "encounter_id", "group_id", "language", "text", "start", "end", "source_sha256")
    assert [[r[k] for k in identity] for r in validation] == [[r[k] for k in identity] for r in old_validation]
    report = {"status": "complete", "validation_identities_text_boundaries_audio_unchanged": True,
        "prepared_source_archive_sha256": inv["sha256"], "train_sha256": digest(output / "train.jsonl"),
        "validation_sha256": digest(output / "validation.jsonl"), "audit": audit,
        "languages": {l: {"segments": sum(r["language"] == l for r in train),
            "hours": sum(r["end"] - r["start"] for r in train if r["language"] == l) / 3600,
            "distinct_texts": len({r["text"] for r in train if r["language"] == l})} for l in ("en", "hi", "mr")},
        "rejected": dict(rejected), "training_started": False,
        "limitations": ["FLEURS is general read speech, not clinical-domain validation", "Native human review pending"]}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
