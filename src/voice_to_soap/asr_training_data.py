"""Strict ASR supervision contract: aligned audio, original split, no truncation."""
import hashlib
import json
import math
from pathlib import Path
import re

import soundfile as sf

from .asr_component import file_sha256


def read_segments(path):
    source = Path(path).resolve()
    rows = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("id", "encounter_id", "group_id", "dataset", "dataset_revision",
                    "split", "audio_path", "source_sha256", "start", "end", "text",
                    "language", "alignment")
        if any(key not in row for key in required):
            raise ValueError(f"Missing supervision/provenance fields on line {line_number}")
        if any(not isinstance(row[k], str) or not row[k].strip() for k in
               ("id", "encounter_id", "group_id", "dataset", "text", "language")):
            raise ValueError("Empty or non-string training identity/text/language")
        if row["language"] not in {"en", "hi", "mr"}:
            raise ValueError("Supported training languages: en, hi, mr")
        if not re.fullmatch(r"[a-f0-9]{40}", row["dataset_revision"]):
            raise ValueError("Dataset revision must be pinned to an immutable 40-character ID")
        if not re.fullmatch(r"[a-f0-9]{64}", row["source_sha256"]):
            raise ValueError("Invalid alignment source hash")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Preserve the original train/validation/test split")
        if row["dataset"] == "ekacare/eka-medical-asr-evaluation-dataset" and row["split"] != "test":
            raise ValueError("Eka is evaluation-only")
        alignment = row["alignment"]
        if not isinstance(alignment, dict) or alignment.get("status") != "accepted":
            raise ValueError("Only accepted aligned segments can enter training")
        if alignment.get("method") not in {"ctc_forced_alignment", "human_reviewed", "dataset_clip_pair"}:
            raise ValueError("Unknown alignment method; do not infer times from text events")
        start, end = float(row["start"]), float(row["end"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end
                and end - start <= 30):
            raise ValueError("Segments must be finite, aligned and <=30s; never truncate")
        audio = (source.parent / row["audio_path"]).resolve()
        info = sf.info(audio)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError("Training audio must already be 16kHz mono")
        if end > info.duration + 1 / info.samplerate:
            raise ValueError("Aligned segment extends beyond audio")
        if alignment["method"] == "dataset_clip_pair" and (start != 0 or abs(end - info.duration) > 1 / info.samplerate):
            raise ValueError("Dataset clip supervision must cover the complete source clip")
        row = {**row, "audio_path": str(audio), "start": start, "end": end}
        rows.append(row)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Supervision must be nonempty with unique segment IDs")
    # Hash each source once. A changed audio file invalidates all of its labels.
    hashes = {}
    for row in rows:
        audio = row["audio_path"]
        if audio not in hashes:
            hashes[audio] = file_sha256(Path(audio))
        if hashes[audio] != row["source_sha256"]:
            raise ValueError("Audio content differs from alignment source hash")
    return rows


def check_disjoint(train, validation):
    if not train or not validation:
        raise ValueError("Training and validation must both be nonempty")
    if any(r["split"] != "train" for r in train):
        raise ValueError("Training input contains non-training data")
    if any(r["split"] != "validation" for r in validation):
        raise ValueError("Validation input contains non-validation data")
    for key in ("id", "encounter_id", "group_id", "source_sha256"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"Train/validation leakage in {key}")
    if ({r["encounter_text_sha256"] for r in train if r.get("encounter_text_sha256")} &
            {r["encounter_text_sha256"] for r in validation if r.get("encounter_text_sha256")}):
        raise ValueError("Train/validation leakage in complete encounter transcript")
    text_hash = lambda r: hashlib.sha256(" ".join(r["text"].casefold().split()).encode()).hexdigest()
    # Entire encounter copies are excluded separately via group/source hashes.
    # Common short utterances can recur naturally; report rather than erase them.
    return {"train_segments": len(train), "validation_segments": len(validation),
            "train_encounters": len({r["encounter_id"] for r in train}),
            "validation_encounters": len({r["encounter_id"] for r in validation}),
            "shared_segment_texts": len({text_hash(r) for r in train} &
                                        {text_hash(r) for r in validation}),
            "train_audio_seconds": sum(r["end"] - r["start"] for r in train),
            "validation_audio_seconds": sum(r["end"] - r["start"] for r in validation)}


def encode_labels(tokenizer, text, language, maximum):
    tokenizer.set_prefix_tokens(language={"en": "english", "hi": "hindi", "mr": "marathi"}[language],
                                task="transcribe", predict_timestamps=False)
    labels = tokenizer(text, truncation=False).input_ids
    # Whisper shifts labels internally; do not duplicate its decoder start token.
    if labels and labels[0] == tokenizer.convert_tokens_to_ids("<|startoftranscript|>"):
        labels = labels[1:]
    if not labels or len(labels) > maximum:
        raise ValueError("Target exceeds decoder context; re-align smaller segments, never truncate")
    return labels
