"""Run real sequential Whisper on complete encounters with progress and provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

from .audio import load_audio
from .asr_datasets import _metric_summary
from .config import load_config
from .datasets import load_manifest
from .io_utils import append_jsonl, write_json
from .whisper_asr import sequential_transcribe, speech_context_regions


class ResumeMismatchError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if sys.platform == "darwin" else value * 1024
    except ImportError:
        return None


def run_component(config: dict, manifests: list[str], output: str, device: str | None = None,
                  vad_dir: str | None = None, resume: bool = False) -> dict:
    audio_config = config["audio"]
    options = audio_config["asr_options"]
    if options.get("mode") != "sequential":
        raise ValueError("ASR component requires explicit sequential mode")
    root = Path(output).resolve()
    rows = [row for manifest in manifests for row in load_manifest(manifest)]
    ids = [row["encounter_id"] for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", uid) for uid in ids):
        raise ValueError("Encounter IDs must be unique safe path components")
    if any(not row["audio_path"] for row in rows):
        raise ValueError("ASR requires actual audio; supplied transcript is not used as input")
    root.mkdir(parents=True, exist_ok=True)
    run_config = {"config": config, "device_override": device, "use_vad": bool(vad_dir)}
    config_path = root / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ResumeMismatchError("Existing run configuration differs; choose a new output directory")
    write_json(config_path, run_config)
    records = []
    for position, row in enumerate(rows, 1):
        uid = row["encounter_id"]
        target = root / uid / "asr.json"
        if target.exists() and not resume:
            raise FileExistsError(f"Encounter output exists; use a new directory or explicit --resume: {uid}")
        print(f"[{position}/{len(rows)}] ASR {uid} ({row['language']})", flush=True)
        started = perf_counter()
        signature = None
        record = {"encounter_id": uid, "language": row["language"], "status": "failed"}
        try:
            waveform, rate = load_audio(row["audio_path"], int(audio_config.get("target_sample_rate", 16000)))
            duration = len(waveform) / rate
            vad = None
            regions = [{"start_seconds": 0.0, "end_seconds": duration}]
            if vad_dir:
                vad = json.loads((Path(vad_dir) / uid / "speech_segments.json").read_text())
                if abs(vad["source_duration_seconds"] - duration) > 2 / rate:
                    raise ValueError("VAD source duration differs from ASR source")
                regions = speech_context_regions(vad["segments"], duration,
                                                 float(audio_config.get("vad_join_gap_seconds", 2.0)))
            signature_data = {"audio_sha256": file_sha256(Path(row["audio_path"])),
                              "config": config, "device": device, "regions": regions,
                              "language": row["language"], "reference": row.get("reference_transcript", "")}
            signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
            if target.exists() and resume:
                previous = json.loads(target.read_text())
                if previous.get("input_signature") != signature:
                    raise ResumeMismatchError("Resume source/config/reference changed; choose a new output directory")
                if previous.get("status") == "complete":
                    records.append(previous)
                    print("  resumed existing complete result", flush=True)
                    continue
            import torch
            seed = (int(config.get("seed", 42)) + int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16)) % (2**32)
            torch.manual_seed(seed)
            language = {"en": "english", "hi": "hindi", "mr": "marathi"}.get(row["language"])
            chunks, texts, warnings = [], [], []
            inference_started = perf_counter()
            for region_index, region in enumerate(regions):
                start = int(round(region["start_seconds"] * rate))
                end = min(len(waveform), int(round(region["end_seconds"] * rate)))
                result = sequential_transcribe(waveform[start:end], rate, audio_config["asr_model_id"],
                    device or audio_config.get("asr_device", "auto"), language, options)
                texts.append(result["text"])
                warnings.extend({"region": region_index, **warning} for warning in result["warnings"])
                for chunk in result["chunks"]:
                    chunks.append({**chunk, "region": region_index,
                        "start": chunk["start"] + start / rate if chunk["start"] is not None else None,
                        "end": chunk["end"] + start / rate if chunk["end"] is not None else None})
            inference_seconds = perf_counter() - inference_started
            hypothesis = " ".join(texts).strip()
            record.update(status="complete", input_signature=signature, model=result["metadata"],
                          seed=seed, audio_sha256=signature_data["audio_sha256"],
                          source_duration_seconds=duration, decoded_regions=regions,
                          vad_applied=bool(vad_dir), chunks=chunks, prediction=hypothesis,
                          warnings=warnings, inference_seconds=inference_seconds,
                          real_time_factor=inference_seconds / duration,
                          elapsed_seconds=perf_counter() - started,
                          process_peak_rss_bytes=peak_rss_bytes(),
                          reference_transcript=row.get("reference_transcript", ""),
                          recording_context=row.get("recording_context", ""))
            if row.get("reference_transcript", "").strip():
                record["metrics"] = _metric_summary([record])
            utterances = [{"id": f"U{i}", "start": item["start"], "end": item["end"],
                "text_raw": item["text"], "text_normalized": item["text"],
                "speaker_id": "unknown", "role": "unknown", "confidence": item["confidence"]}
                for i, item in enumerate(chunks, 1)]
            write_json(target.parent / "transcript_raw.json", utterances)
            print(f"  complete: {duration:.1f}s audio, {inference_seconds:.1f}s processing, {len(chunks)} timestamp chunks", flush=True)
        except Exception as exc:
            if isinstance(exc, ResumeMismatchError):
                raise
            record.update(error_type=type(exc).__name__, error=str(exc),
                          elapsed_seconds=perf_counter() - started, process_peak_rss_bytes=peak_rss_bytes())
            print(f"  FAILED: {type(exc).__name__}", flush=True)
            if signature is not None:
                record["input_signature"] = signature
        write_json(target, record)
        append_jsonl(root / "progress.jsonl", {k: record[k] for k in
                     ("encounter_id", "status", "elapsed_seconds") if k in record})
        records.append(record)
        if record["status"] == "failed":
            # Do not repeat a resource/model failure across long encounters.
            break
    complete = [r for r in records if r["status"] == "complete"]
    scored = [r for r in complete if r.get("reference_transcript", "").strip()]
    summary = {"requested": len(rows), "completed": len(complete),
               "failed": len(records) - len(complete), "not_processed": len(rows) - len(records),
               "metrics_by_language": {lang: _metric_summary([r for r in scored if r["language"] == lang])
                                       for lang in sorted({r["language"] for r in scored})},
               "overall": _metric_summary(scored) if scored else None,
               "encounters": [{k: r[k] for k in ("encounter_id", "status", "metrics",
                              "inference_seconds", "real_time_factor", "process_peak_rss_bytes",
                              "model", "warnings", "error_type") if k in r} for r in records],
               "limitations": ["WER/CER are transcript overlap, not clinical correctness.",
                    "Process peak RSS includes model loading; it is not peak total accelerator memory.",
                    "No claims of Marathi/code-mixed accuracy without scored recordings."]}
    write_json(root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr-large-v3.yaml")
    parser.add_argument("--manifest", required=True, action="append")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", help="Explicit cpu, mps or cuda:0; no fallback on errors")
    parser.add_argument("--vad-dir", help="Require prior VAD artifacts for every input encounter")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    summary = run_component(load_config(args.config), args.manifest, args.outdir,
                            args.device, args.vad_dir, args.resume)
    print(json.dumps({k: v for k, v in summary.items() if k != "encounters"}, indent=2))
    return 0 if summary["completed"] == summary["requested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
