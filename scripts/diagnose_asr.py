"""Resumable full-Whisper decoding/generation-gap diagnostics; never trains or selects on test."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def subset(rows, count, seed):
    chosen = []
    for language in sorted({r["language"] for r in rows}):
        candidates = [r for r in rows if r["language"] == language]
        random.Random(f"{seed}-{language}").shuffle(candidates)
        seen = set()
        for row in candidates:
            if row["group_id"] not in seen:
                chosen.append(row)
                seen.add(row["group_id"])
            if len(seen) == count:
                break
        if len(seen) != count:
            raise ValueError("Not enough distinct training groups for diagnostic subset")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr-lora-balanced.yaml")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-per-language", type=int, default=32)
    parser.add_argument("--beams", type=int, nargs="+", default=[1, 5])
    parser.add_argument("--seconds", type=float, default=4500)
    args = parser.parse_args()
    if args.train_per_language < 1 or any(b < 1 for b in args.beams):
        raise ValueError("Positive subset/beam counts required")
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, WhisperForConditionalGeneration
    from voice_to_soap.asr_scoring import CURRENT, LEGACY, score_records
    from voice_to_soap.asr_training_data import check_disjoint, read_segments
    from voice_to_soap.config import load_config
    from voice_to_soap.io_utils import write_json
    from voice_to_soap.whisper_lora import amp_context, batch_for, validate_adapter
    started = time.monotonic()
    cfg = load_config(args.config)
    train = read_segments(args.train)
    validation = read_segments(args.validation)
    check_disjoint(train, validation)
    adapter_metadata = validate_adapter(args.adapter, cfg["model_id"], cfg["revision"])
    train_subset = subset(train, args.train_per_language, cfg["seed"])
    args.outdir.mkdir(parents=True, exist_ok=True)
    protocol = {"model_id": cfg["model_id"], "revision": cfg["revision"],
                "adapter_sha256": adapter_metadata["adapter_sha256"],
                "adapter_source_epoch": adapter_metadata.get("epoch"), "device": args.device,
                "train_manifest_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
                "validation_manifest_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
                "seed": cfg["seed"], "train_per_language": args.train_per_language,
                "beams": args.beams, "max_new_tokens": 440, "training_performed": False}
    path = args.outdir / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Diagnostic inputs/protocol changed; use a new directory")
    write_json(path, protocol)
    torch.manual_seed(cfg["seed"])
    processor = AutoProcessor.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    model = WhisperForConditionalGeneration.from_pretrained(cfg["model_id"], revision=cfg["revision"],
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        low_cpu_mem_usage=True, attn_implementation="sdpa", use_safetensors=True)
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(model, args.adapter).to(args.device).eval()
    tasks = [("train-subset-beam1", train_subset, 1)] + [
        (f"validation-beam{beam}", validation, beam) for beam in args.beams]
    summaries = {}
    for name, rows, beam in tasks:
        folder = args.outdir / name
        folder.mkdir(exist_ok=True)
        records = []
        for index, row in enumerate(rows, 1):
            assert Path(row["id"]).name == row["id"]
            target = folder / (row["id"] + ".json")
            if target.exists():
                record = json.loads(target.read_text())
                assert record["id"] == row["id"] and record["reference_transcript"] == row["text"]
            else:
                if time.monotonic() - started > args.seconds:
                    write_json(args.outdir / "status.json", {"status": "paused_budget", "stage": name,
                        "completed_stage_examples": len(records), "training_performed": False})
                    print("DIAGNOSTIC_PAUSED_SAFELY", name, len(records), flush=True)
                    return
                batch = batch_for(row, processor, model.config.max_target_positions, args.device)
                del batch["labels"]
                tick = time.monotonic()
                language = {"en": "english", "hi": "hindi", "mr": "marathi"}[row["language"]]
                with torch.inference_mode(), amp_context(args.device):
                    tokens = model.generate(**batch, language=language, task="transcribe", do_sample=False,
                        num_beams=beam, max_new_tokens=440, return_timestamps=False, use_cache=True)
                prediction = processor.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
                record = {"id": row["id"], "encounter_id": row["encounter_id"], "language": row["language"],
                          "reference_transcript": row["text"], "prediction": prediction,
                          "seconds": time.monotonic() - tick, "num_beams": beam,
                          "generated_sequence_tokens": tokens.shape[-1],
                          "ends_with_eos": tokens[0, -1].item() == processor.tokenizer.eos_token_id}
                write_json(target, record)
                del batch, tokens
            records.append(record)
            print("DIAGNOSTIC_PROGRESS", name, index, len(rows), flush=True)
        report = {"examples": len(records), "num_beams": beam, "protocols": {}}
        for scoring in (LEGACY, CURRENT):
            groups = {l: score_records([r for r in records if r["language"] == l], scoring)
                      for l in sorted({r["language"] for r in records})}
            report["protocols"][scoring] = {"by_language": groups,
                "macro_language_wer": sum(r["wer"] for r in groups.values()) / len(groups)}
        report["seconds"] = sum(r["seconds"] for r in records)
        write_json(folder / "summary.json", report)
        summaries[name] = report
        write_json(args.outdir / "summary.json", {"protocol": protocol, "stages": summaries,
            "status": "running", "scope": "Diagnostic training-subset/validation; not independent test"})
    write_json(args.outdir / "summary.json", {"protocol": protocol, "stages": summaries,
        "status": "complete", "scope": "Diagnostic training-subset/validation; not independent test"})
    write_json(args.outdir / "status.json", {"status": "complete", "training_performed": False})
    print("DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
