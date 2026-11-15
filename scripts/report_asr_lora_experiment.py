"""Compile measured local ASR results; refuse incomplete experiments."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re

from voice_to_soap.asr_datasets import _normalize_asr_text
from voice_to_soap.io_utils import write_json

NEGATIONS = {"no", "not", "never", "without", "denies", "denied", "negative",
             "नहीं", "नही", "ना", "नाही", "नाहीत", "नसून"}


def inventory(text):
    return {"numeric_literals": Counter(re.findall(r"\d+(?:[.,]\d+)?", text)),
            "negation_words": Counter(w for w in _normalize_asr_text(text).split() if w in NEGATIONS)}


def compile_report(directory):
    root = Path(directory)
    experiment = json.loads((root / "experiment.json").read_text(encoding="utf-8"))
    if experiment["status"] != "complete":
        raise ValueError("Experiment is incomplete; report actual failure/progress before compiling final scores")
    training = experiment["training"]
    adapter = json.loads((root / "training/best-adapter/asr_adapter.json").read_text(encoding="utf-8"))
    baseline_validation = adapter["baseline_validation"]
    selected_validation = adapter["validation"]
    rows, reviews, decoder_hashes = [], [], set()
    for dataset, comparison in experiment["evaluations"].items():
        for language, values in comparison["by_language"].items():
            before, after = values["base"], values["adapter"]
            rows.append({"dataset": dataset, "language": language, "examples": before["examples"],
                "base_wer": before["wer"], "adapter_wer": after["wer"],
                "wer_change_percentage_points": 100 * values["wer_change"],
                "relative_wer_reduction_percent": 100 * (1 - after["wer"] / before["wer"]) if before["wer"] else None,
                "base_cer": before["cer"], "adapter_cer": after["cer"]})
        for uid in comparison["paired_ids"]:
            entry = {"dataset": dataset, "encounter_id": uid}
            for mode in ("base", "adapter"):
                path = root / (dataset + "-" + mode) / uid / "asr.json"
                if not path.is_file():
                    raise ValueError("Detailed report requires the rerun base artifacts too")
                record = json.loads(path.read_text(encoding="utf-8"))
                decoder_hashes.add(record["model"].get("implementation_sha256"))
                if mode == "adapter" and record["model"]["adapter"]["sha256"] != adapter["adapter_sha256"]:
                    raise ValueError("Evaluation recordings used different adapter weights; do not combine them")
                reference, prediction = inventory(record["reference_transcript"]), inventory(record["prediction"])
                entry[mode] = {"numeric_inventory_differs": reference["numeric_literals"] != prediction["numeric_literals"],
                    "negation_inventory_differs": reference["negation_words"] != prediction["negation_words"],
                    "timestamp_warning_count": len(record.get("warnings", [])),
                    "source_seconds": record["source_duration_seconds"],
                    "inference_seconds_including_model_load": record["inference_seconds"]}
            reviews.append(entry)
    improved = sum(r["adapter_wer"] < r["base_wer"] for r in rows)
    regressed = sum(r["adapter_wer"] > r["base_wer"] for r in rows)
    events = [json.loads(line) for line in (root / "training/training.jsonl").read_text(encoding="utf-8").splitlines()]
    past_sessions = [json.loads(path.read_text(encoding="utf-8"))
                     for path in (root / "training/session-history").glob("*.json")]
    summary = {"status": "complete", "all_dataset_language_groups_improved": improved == len(rows),
        "improved_groups": improved, "regressed_groups": regressed,
        "unchanged_groups": len(rows) - improved - regressed,
        "evaluated_recordings": sum(r["examples"] for r in rows),
        "training_data": experiment["audit"], "training_by_language": experiment["training_by_language"],
        "model_id": training["model_id"], "revision": training["revision"],
        "adapter_sha256": adapter["adapter_sha256"], "selected_epoch": adapter["epoch"],
        "decoder_implementation_sha256": sorted(value for value in decoder_hashes if value),
        "evaluation_recovery_notes": [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "evaluation-recovery").glob("*/reason.json"))],
        "parameters": training["parameters"], "optimizer_updates": training["optimizer_updates"],
        "config": training["config"], "device": training["device"], "base_precision": training["base_precision"],
        "quantization": training["quantization"], "versions": training["versions"],
        "elapsed_training_and_validation_seconds_current_session": training["elapsed_seconds"],
        "elapsed_training_and_validation_seconds_all_attempts": training["elapsed_seconds"] + sum(
            session.get("elapsed_seconds", 0) for session in past_sessions),
        "previous_attempts": [{key: session.get(key) for key in
            ("status", "error_type", "error", "implementation_sha256", "elapsed_seconds")} for session in past_sessions],
        "maximum_observed_mps_driver_bytes_at_update_boundaries": max(
            (e.get("mps_driver_allocated_bytes", 0) for e in events), default=0),
        "validation_base": baseline_validation, "validation_selected_adapter": selected_validation,
        "paired_evaluation": rows, "review_flags": reviews,
        "limitations": [*experiment["limitations"],
            "Checkpoint selection used greedy short-segment validation; production evaluation uses sequential decoding with temperature fallback and word timestamps.",
            "Numeric/negation inventory differences are lexical review flags, not clinical error diagnoses.",
            "Matching inventories do not prove correct dosage, assertion, speaker, or clinical meaning.",
            "Timestamp consistency warnings are not gold word-boundary accuracy.",
            "MPS driver samples at update boundaries are not measured peak total-system memory.",
            "Every downstream generated note still requires clinician approval."]}
    write_json(root / "compiled-results.json", summary)
    with (root / "compiled-results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Measured local Whisper LoRA results", "",
        f"WER decreased in {improved}/{len(rows)} dataset/language groups, increased in {regressed}, "
        f"and was unchanged in {len(rows)-improved-regressed}.", "",
        f"Evaluated {summary['evaluated_recordings']} selected recordings with both full base and LoRA models.", "",
        "| Dataset | Language | Clips | Base WER | LoRA WER | Change (pp) | Base CER | LoRA CER |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['language']} | {row['examples']} | "
            f"{100*row['base_wer']:.2f}% | {100*row['adapter_wer']:.2f}% | "
            f"{row['wer_change_percentage_points']:+.2f} | {100*row['base_cer']:.2f}% | {100*row['adapter_cer']:.2f}% |")
    lines += ["", "## Training and selection", "",
        f"Full {training['model_id']} at `{training['revision']}`, {training['base_precision']}, no quantization.",
        f"{training['parameters']['trainable_parameters']:,} trainable LoRA parameters; "
        f"{training['parameters']['total_parameters']:,} total parameters.",
        f"{experiment['audit']['train_segments']} training segments / "
        f"{experiment['audit']['train_audio_seconds']/60:.2f} minutes of speech; "
        f"{experiment['audit']['validation_segments']} separate validation segments.",
        f"Selected epoch {adapter['epoch']} using {adapter.get('selection_metric', 'overall_wer')}; "
        f"{training['optimizer_updates']} optimizer updates executed.", "",
        "| Validation language | Base WER | Selected LoRA WER |", "|---|---:|---:|"]
    for language in baseline_validation["by_language"]:
        lines.append(f"| {language} | {100*baseline_validation['by_language'][language]['wer']:.2f}% | "
                     f"{100*selected_validation['by_language'][language]['wer']:.2f}% |")
    lines += ["", "## Review and limits", "", *["- " + text for text in summary["limitations"]], "",
        "See `compiled-results.json` for all numeric/negation/timestamp review flags and exact provenance.",
        "WER/CER use the same scorer for both models. This selected sample is not a full-corpus or clinical validation claim."]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    result = compile_report(parser.parse_args().run_dir)
    print(json.dumps({key: result[key] for key in ("improved_groups", "regressed_groups", "unchanged_groups",
                                                   "evaluated_recordings")}, indent=2))
