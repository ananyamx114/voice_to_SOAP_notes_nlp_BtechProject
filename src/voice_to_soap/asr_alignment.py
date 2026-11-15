"""Auditable English CTC alignment; never use ASR predictions as gold labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import random
import tarfile
import unicodedata

import numpy as np
import soundfile as sf

from .asr_component import file_sha256
from .audio import load_audio
from .io_utils import write_json

ALIGNER_ID = "facebook/wav2vec2-base-960h"
ALIGNER_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
BETRAC_REVISION = "96f7b94e662dac0d1394cfdf55e8b8ae81422663"


def transcript_units(raw, words_per_unit=20):
    """Retain spoken wording/punctuation; log the explicit stage-direction cleanup.

    Unit lengths here are text lengths only. Audio boundaries come from CTC.
    """
    cleaned = re.sub(r"(?im)^(?:DOCTOR|PATIENT):\s*", "", raw)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = cleaned.replace("\\n", " ").replace("*", "")
    words = cleaned.split()
    units = []
    for i in range(0, len(words), words_per_unit):
        unit = " ".join(words[i:i + words_per_unit])
        if units and not any(c.isalnum() for c in unit):
            units[-1] += " " + unit
        else:
            units.append(unit)
    return units


def alignment_text(text):
    from num2words import num2words

    text = text.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Only the aligner's view expands digits; original Whisper targets stay intact.
    text = re.sub(r"\d+(?:\.\d+)?", lambda m: num2words(m[0]), text)
    if any(c.isalpha() and not c.isascii() for c in text):
        raise ValueError("English aligner cannot align non-Latin text")
    text = re.sub(r"[^A-Za-z' ]", " ", text).upper()
    text = " ".join(text.split())
    if not text:
        raise ValueError("Empty aligner target")
    return text


def ctc_path(log_probs, labels, blank=0, max_cells=2_048_000_000):
    """Exact CTC Viterbi, with repeats requiring blanks and bounded memory.

    Return target-token frame spans and mean token log probability. The path
    consumes the complete reference and complete audio; unmatched speech is not
    silently removed from the reference. Blank-only leading/trailing audio is OK.
    """
    scores = np.asarray(log_probs, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if (scores.ndim != 2 or not np.isfinite(scores).all() or not len(labels)
            or labels.ndim != 1 or (labels == blank).any()
            or labels.min() < 0 or labels.max() >= scores.shape[1]):
        raise ValueError("Invalid CTC emissions or target IDs")
    steps = scores.shape[0]
    states = np.full(2 * len(labels) + 1, blank, dtype=np.int64)
    states[1::2] = labels
    if steps * len(states) > max_cells:
        raise ValueError("Alignment exceeds memory budget; split source with reviewed boundaries")
    if steps < len(labels) + int(np.sum(labels[1:] == labels[:-1])):
        raise ValueError("Audio has too few CTC frames for the complete reference")
    # A predecessor has only three values. Pack four into each byte so long
    # consultations retain exact alignment without a four-times-larger table.
    packed_columns = (len(states) + 3) // 4
    trace = np.zeros((steps, packed_columns), dtype=np.uint8)
    padded = np.zeros(packed_columns * 4, dtype=np.uint8)
    previous = np.full(len(states), -np.inf, dtype=np.float32)
    previous[0], previous[1] = scores[0, blank], scores[0, labels[0]]
    can_skip = np.zeros(len(states), dtype=bool)
    can_skip[2:] = (states[2:] != blank) & (states[2:] != states[:-2])
    for frame in range(1, steps):
        advance = np.concatenate(([-np.inf], previous[:-1]))
        skip = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        skip[~can_skip] = -np.inf
        moves = np.stack((previous, advance, skip))
        choice = np.argmax(moves, axis=0)
        padded[:len(states)] = choice
        trace[frame] = (padded[0::4] | (padded[1::4] << 2) |
                        (padded[2::4] << 4) | (padded[3::4] << 6))
        previous = moves[choice, np.arange(len(states))] + scores[frame, states]
    state = len(states) - 1 if previous[-1] >= previous[-2] else len(states) - 2
    if not np.isfinite(previous[state]):
        raise ValueError("No complete CTC alignment")
    path = np.zeros(steps, dtype=np.int64)
    for frame in range(steps - 1, -1, -1):
        path[frame] = state
        if frame:
            state -= int((trace[frame, state // 4] >> (2 * (state % 4))) & 3)
    spans = []
    for index, label in enumerate(labels):
        positions = np.flatnonzero(path == 2 * index + 1)
        if not len(positions):
            raise ValueError("Alignment omitted a reference token")
        spans.append((int(positions[0]), int(positions[-1]),
                      float(scores[positions, label].mean())))
    return spans


def emissions(audio, model, processor):
    """CPU CTC in 20s cores with 1s context; keep the exact convolution time grid."""
    import torch

    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    core, context = 20 * 16000, 16000
    if core % stride or context % stride:
        raise ValueError("CTC core/context must align with the convolution stride")
    probabilities, times = [], []
    for start in range(0, len(audio), core):
        left, right = max(0, start - context), min(len(audio), start + core + context)
        if right - left < receptive:
            raise ValueError("Audio is too short for the CTC encoder")
        values = processor(audio[left:right], sampling_rate=16000, return_tensors="pt").input_values
        with torch.inference_mode():
            logits = model(values).logits[0].float().log_softmax(-1).cpu().numpy()
        centers = left + (receptive - 1) / 2 + np.arange(len(logits)) * stride
        keep = (centers >= start) & (centers < min(start + core, len(audio)))
        probabilities.append(logits[keep])
        times.append(centers[keep] / 16000)
    grid = np.concatenate(times)
    if len(grid) > 1 and not np.allclose(np.diff(grid), stride / 16000, atol=1e-6):
        raise ValueError("Non-contiguous CTC frame grid")
    return np.concatenate(probabilities), grid, stride / 16000


def align_encounter(audio, units, model, processor, min_log_probability=-1.5):
    normalized = [alignment_text(t) for t in units]
    tokens = [processor.tokenizer(t, add_special_tokens=False).input_ids for t in normalized]
    if any(processor.tokenizer.unk_token_id in t or not t for t in tokens):
        raise ValueError("Reference contains unknown CTC tokens; refusing silent deletion")
    # Explicit word separators between units. Keep token spans for each original target.
    delimiter = processor.tokenizer.word_delimiter_token_id
    labels, offsets = [], []
    for unit in tokens:
        if labels:
            labels.append(delimiter)
        offsets.append((len(labels), len(labels) + len(unit)))
        labels.extend(unit)
    log_probs, centers, frame_seconds = emissions(audio, model, processor)
    spans = ctc_path(log_probs, labels, blank=model.config.pad_token_id)
    # Share each inter-unit gap at its midpoint, rather than cutting at a CTC
    # token spike and clipping an initial/final phoneme. Edge padding stays in
    # the source waveform. These are inferred boundaries, never gold timings.
    speech_starts = [max(0.0, centers[spans[first][0]] - frame_seconds / 2)
                     for first, last in offsets]
    speech_ends = [min(len(audio) / 16000, centers[spans[last - 1][1]] + frame_seconds / 2)
                   for first, last in offsets]
    boundaries = [max(0.0, speech_starts[0] - 0.1)]
    boundaries.extend((speech_ends[i - 1] + speech_starts[i]) / 2 for i in range(1, len(units)))
    boundaries.append(min(len(audio) / 16000, speech_ends[-1] + 0.1))
    candidates = []
    previous_end = 0.0
    for index, (text, (first, last)) in enumerate(zip(units, offsets)):
        selected = spans[first:last]
        start, end = boundaries[index:index + 2]
        confidence = float(np.mean([t[2] for t in selected]))
        worst = float(min(t[2] for t in selected))
        # All token scores matter: a good mean must not hide one forced wrong word.
        reasons = []
        if not (0 <= start < end <= len(audio) / 16000) or start < previous_end - 1e-6:
            reasons.append("invalid_or_overlapping_boundaries")
        if not 0.5 <= end - start <= 29.5:
            reasons.append("duration_outside_0.5_to_29.5_seconds")
        if confidence < min_log_probability or worst < -5:
            reasons.append("low_alignment_confidence")
        previous_end = end
        candidates.append({"text": text, "start": float(start), "end": float(end),
            "alignment": {"method": "ctc_forced_alignment", "model_id": ALIGNER_ID,
                "revision": ALIGNER_REVISION, "status": "rejected" if reasons else "accepted",
                "mean_token_log_probability": confidence, "worst_token_log_probability": worst,
                "minimum_mean_log_probability": min_log_probability,
                "reasons": reasons, "human_reviewed": False}})
    return candidates


def prepare(archive, output, split, limit=None, selection_seed=None, exclude=(), resume=False,
            min_accepted_seconds=None):
    from transformers import AutoModelForCTC, AutoProcessor

    source, root = Path(archive), Path(output).resolve()
    expected = "train-" if split == "train" else "dev-"
    if not source.name.startswith(expected):
        raise ValueError("Archive name does not match its original split")
    if limit is not None and limit <= 0:
        raise ValueError("Encounter limit must be positive")
    if min_accepted_seconds is not None and (not np.isfinite(min_accepted_seconds) or min_accepted_seconds <= 0):
        raise ValueError("Accepted-audio target must be finite and positive")
    manifest_root = source.resolve().parent.parent
    download = json.loads((manifest_root / "download_manifest.json").read_text(encoding="utf-8"))
    relative = source.resolve().relative_to(manifest_root).as_posix()
    archive_hash = file_sha256(source)
    if (download["repository"] != "BeTraC/betrac-2026" or download["revision"] != BETRAC_REVISION
            or not any(f["path"] == relative and f["sha256"] == archive_hash
                       and f.get("upstream_sha256") == archive_hash for f in download["files"])):
        raise ValueError("Archive does not match the pinned, hash-verified download manifest")
    root.mkdir(parents=True, exist_ok=resume)
    report = {"status": "running", "dataset": "BeTraC/betrac-2026", "revision": BETRAC_REVISION,
              "archive_sha256": archive_hash, "split": split, "encounter_limit": limit,
              "selection_seed": selection_seed, "excluded_encounters": list(exclude),
              "minimum_accepted_audio_seconds": min_accepted_seconds,
              "alignment_model": ALIGNER_ID, "alignment_revision": ALIGNER_REVISION,
              "encounters": [], "accepted": 0, "rejected": 0, "accepted_audio_seconds": 0.0,
              "limitations": ["Synthetic English speech; no Hindi/Marathi accuracy claim.",
                "CTC boundaries and confidence are automatic, not human-reviewed gold.",
                "Text-generation event timestamps are never audio alignment.",
                "Review aligned audio before substantive training; inspect dose/name targets."]}
    cached = {}
    if resume:
        previous = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
        for field in ("dataset", "revision", "archive_sha256", "split", "encounter_limit",
                      "selection_seed", "excluded_encounters", "alignment_model", "alignment_revision",
                      "minimum_accepted_audio_seconds"):
            if previous.get(field) != report[field]:
                raise ValueError(f"Preparation resume mismatch in {field}")
        write_json(root / "preparation.previous.json", previous)
        cached = {r["encounter_id"]: r for r in previous["encounters"]}
    write_json(root / "preparation.json", report)
    try:
        # Loaded independently from Whisper; no simultaneous model memory use.
        model = AutoModelForCTC.from_pretrained(ALIGNER_ID, revision=ALIGNER_REVISION,
                    use_safetensors=True).to("cpu").eval()
        processor = AutoProcessor.from_pretrained(ALIGNER_ID, revision=ALIGNER_REVISION)
        with tarfile.open(source) as tar, (root / "segments.jsonl.partial").open("w", encoding="utf-8") as accepted:
            members = {m.name: m for m in tar.getmembers() if m.isfile()}
            keys = sorted(n[:-5] for n in members if n.endswith(".opus"))
            keys = [key for key in keys if key not in set(exclude)]
            if not keys:
                raise ValueError("Archive contains no audio encounters")
            if limit and limit > len(keys):
                raise ValueError("Requested more encounters than the eligible source contains")
            if selection_seed is not None and limit:
                keys = sorted(random.Random(selection_seed).sample(keys, limit))
            else:
                keys = keys[:limit] if limit else keys
                if min_accepted_seconds is not None and selection_seed is not None:
                    random.Random(selection_seed).shuffle(keys)
            report["selected_encounters"] = keys
            write_json(root / "preparation.json", report)
            for number, key in enumerate(keys, 1):
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", key):
                    raise ValueError("Unsafe encounter key")
                raw = tar.extractfile(members[key + ".transcript.txt"]).read().decode("utf-8")
                metadata = json.load(tar.extractfile(members[key + ".json"]))
                payload = tar.extractfile(members[key + ".opus"]).read()
                original_file = root / (key + ".opus")
                original_file.write_bytes(payload)
                # libsndfile may decode Opus at 48kHz irrespective of its input
                # rate header. Resample explicitly; never relabel samples as 16kHz.
                original_info = sf.info(original_file)
                audio, rate = load_audio(original_file, 16000)
                audio_file = root / (key + ".flac")
                sf.write(audio_file, audio, rate, subtype="PCM_24")
                # Align exactly the samples that training will read after encoding.
                audio, rate = sf.read(audio_file, dtype="float32")
                (root / (key + ".transcript.txt")).write_text(raw, encoding="utf-8")
                write_json(root / (key + ".metadata.json"), metadata)
                source_hash = file_sha256(audio_file)
                units = transcript_units(raw)
                print(f"Aligning {number}: {key}, {len(audio)/rate:.1f}s, {len(units)} units", flush=True)
                try:
                    if key in cached:
                        candidates = json.loads((root / (key + ".alignment.json")).read_text(encoding="utf-8"))
                        if cached[key]["source_sha256"] != source_hash or [c["text"] for c in candidates] != units:
                            raise ValueError("Preparation resume source or target changed")
                        print("  reused hash-checked completed alignment", flush=True)
                    else:
                        candidates = align_encounter(audio, units, model, processor)
                except ValueError as exc:
                    known_rejections = ("Empty aligner target", "English aligner cannot",
                        "Reference contains unknown CTC", "Alignment exceeds memory budget",
                        "Audio has too few CTC frames", "No complete CTC alignment",
                        "Alignment omitted a reference token")
                    if not str(exc).startswith(known_rejections):
                        raise
                    print(f"  REJECTED encounter alignment: {exc}", flush=True)
                    candidates = [{"text": text, "start": None, "end": None,
                        "alignment": {"method": "ctc_forced_alignment", "status": "rejected",
                            "human_reviewed": False, "reasons": [str(exc)]}} for text in units]
                group = str(metadata.get("parentId") or metadata.get("id") or key)
                for i, segment in enumerate(candidates):
                    row = {"id": f"{key}-{i:04d}", "encounter_id": key,
                        "group_id": group, "dataset": "BeTraC/betrac-2026",
                        "dataset_revision": BETRAC_REVISION, "split": split,
                        "audio_path": audio_file.name, "source_sha256": source_hash,
                        "original_audio_path": original_file.name,
                        "original_audio_sha256": file_sha256(original_file),
                        "original_decoded_sample_rate": original_info.samplerate,
                        "encounter_text_sha256": hashlib.sha256(" ".join(raw.casefold().split()).encode()).hexdigest(),
                        "language": "en", **segment}
                    if segment["alignment"]["status"] == "accepted":
                        accepted.write(json.dumps(row, ensure_ascii=False) + "\n")
                        accepted.flush()
                        report["accepted"] += 1
                        report["accepted_audio_seconds"] += row["end"] - row["start"]
                    else:
                        report["rejected"] += 1
                write_json(root / (key + ".alignment.json"), candidates)
                report["encounters"].append({"encounter_id": key, "group_id": group,
                    "source_sha256": source_hash, "source_seconds": len(audio) / rate,
                    "accepted": sum(c["alignment"]["status"] == "accepted" for c in candidates),
                    "rejected": sum(c["alignment"]["status"] != "accepted" for c in candidates)})
                write_json(root / "preparation.json", report)
                if min_accepted_seconds is not None:
                    print(f"  accepted audio: {report['accepted_audio_seconds']/60:.2f} / {min_accepted_seconds/60:.2f} minutes", flush=True)
                    if report["accepted_audio_seconds"] >= min_accepted_seconds:
                        break
        if not report["accepted"]:
            raise ValueError("No alignment passed quality checks; review the source and alignment")
        if min_accepted_seconds is not None and report["accepted_audio_seconds"] < min_accepted_seconds:
            raise ValueError("Eligible source exhausted before the quality-screened duration target; do not train a partial cohort")
        report["selected_encounters"] = [item["encounter_id"] for item in report["encounters"]]
        (root / "segments.jsonl.partial").replace(root / "segments.jsonl")
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(root / "preparation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--limit", type=int, help="Explicit integration subset; omit for full shard")
    parser.add_argument("--selection-seed", type=int, help="Reproducible random encounter selection")
    parser.add_argument("--exclude-encounter", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-accepted-seconds", type=float, help="Stop after complete encounters meet this quality-screened duration")
    args = parser.parse_args()
    prepare(args.archive, args.outdir, args.split, args.limit, args.selection_seed, args.exclude_encounter, args.resume,
            args.min_accepted_seconds)


if __name__ == "__main__":
    main()
