"""Read active DGX SOAP experiment artifacts and export aggregate report metrics only.

Does not load a model, modify checkpoints, or stop jobs. Private artifacts remain
in place; no dialogues, predictions, encounter IDs, credentials or machine paths
are copied to the shareable JSON.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def selected(source: dict, keys: tuple[str, ...]) -> dict:
    return {key: source[key] for key in keys if key in source}


def export(root: Path) -> dict:
    pin = read(root / ".local/dgx-migration/qwen7b-model-pin.json")
    assert pin.get("model_id") == "Qwen/Qwen2.5-7B-Instruct"
    result = {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "status": "interim_report_snapshot_not_final_pipeline_evaluation",
        "model": selected(pin, ("model_id", "revision")),
        "dataset": {"name": "BeTraC-2000", "train": 1600, "validation": 200,
                    "internal_test": 200, "source": "official BeTraC train",
                    "references": "synthetic, heuristically mapped four-string SOAP",
                    "known_internal_test_encounters_used_for_asr_training": 2},
        "training": {}, "evaluation": {},
        "limitations": [
            "Text-only SOAP results do not measure end-to-end audio-to-SOAP performance.",
            "Validation loss and ROUGE-L are not clinical accuracy.",
            "Partial evaluations cover different subsets; do not rank them as a paired full-test comparison.",
            "The four-string SOAP experiment is separate from the evidence-cited claim schema.",
            "All generated notes require clinician approval.",
        ],
    }
    for stage in ("direct", "hybrid"):
        folder = root / "data/outputs/dgx-qwen7b-soap" / stage
        config = read(folder / "run_config.json")
        complete = read(folder / "completed.json")
        checkpoints = sorted(folder.glob("checkpoint-*"),
                             key=lambda p: int(p.name.split("-")[-1]), reverse=True)
        state = {}
        for checkpoint in checkpoints:
            state = read(checkpoint / "trainer_state.json")
            if state:
                break
        history = [selected(row, ("step", "epoch", "eval_loss", "eval_runtime"))
                   for row in state.get("log_history", []) if "eval_loss" in row]
        best = complete.get("best_validation_loss", state.get("best_metric"))
        asynchronous = read(folder / "async-validation-summary.json")
        if asynchronous:
            best = complete.get("best_validation_loss", asynchronous["best_validation_loss"])
            history = [selected(row, ("step", "epoch", "eval_loss", "eval_runtime"))
                       for row in asynchronous["validation_history"]]
        result["training"][stage] = {
            "status": "complete" if complete else "incomplete_checkpoint_snapshot" if state else "not_started",
            "saved_updates": complete.get("global_step", state.get("global_step")),
            "planned_updates": 4800,
            "saved_epoch": 3.0 if complete else state.get("epoch"),
            "best_validation_loss": best,
            "selected_checkpoint_step": (int(str(complete.get("best_checkpoint") or state.get("best_model_checkpoint")).split("checkpoint-")[-1])
                if complete.get("best_checkpoint") or state.get("best_model_checkpoint") else None),
            "validation_history": history,
            "settings": selected(config, ("epochs", "batch_size", "gradient_accumulation_steps",
                "learning_rate", "weight_decay", "seed", "max_seq_length", "lora_rank", "lora_alpha",
                "lora_dropout", "target_modules", "selection_metric", "split_sha256")),
        }
        if asynchronous:
            result["training"][stage].update(
                selected_checkpoint_step=asynchronous["selected_checkpoint_step"],
                last_validated_update=asynchronous["last_evaluated_update"],
                validation_execution="asynchronous_on_separate_gpu")
    keys = ("examples", "valid_predictions", "invalid_predictions", "by_section_rougeL_all_examples",
            "overall_mean_section_rougeL_all_examples", "overall_mean_section_rougeL_valid_only",
            "generation_seconds", "invalid_outputs_scored_as_zero", "clinical_factuality_measured",
            "adapter_sha256", "test_sha256", "max_new_tokens", "do_sample", "seed", "precision")
    for stage in ("base", "direct", "hybrid"):
        folder = root / "data/outputs/dgx-qwen7b-soap-evaluation" / stage
        metrics = read(folder / "metrics.json")
        if metrics:
            assert metrics["examples"] == 200
            result["evaluation"][stage] = {"status": "complete", **selected(metrics, keys)}
            continue
        records = [read(p) for p in sorted((folder / "encounters").glob("*.json"))]
        records = [r for r in records if r and r.get("stage") == stage]
        n = len(records)
        valid = sum(r.get("prediction_soap") is not None for r in records)
        result["evaluation"][stage] = {
            "status": "partial" if n else "not_started", "examples": n,
            "planned_examples": 200, "valid_predictions": valid, "invalid_predictions": n - valid,
            "overall_mean_section_rougeL_all_examples": (
                sum(r["mean_section_rougeL"] for r in records) / n if n else None),
            "invalid_outputs_scored_as_zero": True,
        }
    result["software"] = {}
    for package in ("torch", "transformers", "peft", "accelerate", "rouge-score"):
        try:
            result["software"][package] = version(package)
        except PackageNotFoundError:
            result["software"][package] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"snapshot_utc": result["snapshot_utc"],
                      "training": {s: {k: v for k, v in r.items() if k in
                          ("status", "saved_updates", "best_validation_loss")} for s, r in result["training"].items()},
                      "evaluation": result["evaluation"]}, indent=2))


if __name__ == "__main__":
    main()
