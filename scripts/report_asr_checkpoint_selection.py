"""Report language-validation checkpoint selection from complete paired runs.

This compiles existing predictions; it does not implement automatic routing or
choose checkpoints using test scores. Keep the shared-checkpoint report too.
"""
import argparse
import csv
import json
from pathlib import Path

from voice_to_soap.asr_comparison import compare, load_run
from voice_to_soap.asr_component import file_sha256
from voice_to_soap.io_utils import write_json


def report(shared_run, english_run, english_adapter, output):
    shared, english, adapter, target = map(Path, (shared_run, english_run, english_adapter, output))
    primary = json.loads((shared / "experiment.json").read_text())
    candidate = json.loads((english / "experiment.json").read_text())
    if primary["status"] != "complete" or candidate["status"] != "complete":
        raise ValueError("Both complete experiments are required")
    shared_adapter = shared / "training/best-adapter"
    original = json.loads((shared_adapter / "asr_adapter.json").read_text())
    chosen = json.loads((adapter / "asr_adapter.json").read_text())
    for key in ("task", "model_id", "revision", "training_manifest_sha256", "validation_manifest_sha256"):
        if original[key] != chosen[key]:
            raise ValueError(f"Checkpoints do not share the training experiment: {key}")
    if chosen.get("selection_language") != "en" or chosen.get("selection_metric") != "language_validation_wer":
        raise ValueError("English checkpoint must be selected using English validation WER")
    for directory, metadata in ((shared_adapter, original), (adapter, chosen)):
        if metadata["status"] != "complete" or file_sha256(directory / "adapter_model.safetensors") != metadata["adapter_sha256"]:
            raise ValueError("Incomplete or modified adapter")
    epochs = [(int(path.parent.name.split("-")[-1]), json.loads(path.read_text()))
              for path in (shared / "training").glob("epoch-*/metrics.json")]
    languages = {row["language"] for name in primary["evaluations"] for row in load_run(shared / (name + "-base")).values()}
    selected_epochs = {}
    for language in sorted(languages):
        best = min(epochs, key=lambda item: (item[1]["by_language"][language]["wer"], item[0]))[0]
        selected = chosen if language == "en" else original
        if selected["epoch"] != best:
            raise ValueError(f"Available checkpoint is not validation-best for {language}; export/evaluate it first")
        selected_epochs[language] = best
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in primary["evaluations"]:
        shared_comparison = compare(shared / (name + "-base"), shared / (name + "-adapter"), target / (name + "-shared.json"))
        if any(record["model"]["adapter"]["sha256"] != original["adapter_sha256"]
               for record in load_run(shared / (name + "-adapter")).values()):
            raise ValueError("Shared predictions used different adapter weights")
        for language, shared_values in shared_comparison["by_language"].items():
            if language == "en":
                if name not in candidate["comparisons"]:
                    raise ValueError(f"Missing English candidate coverage: {name}")
                baseline = english / "eka-base" if name == "eka" else shared / (name + "-base")
                expected_base = {uid: record for uid, record in load_run(shared / (name + "-base")).items()
                                 if record["language"] == language}
                candidate_base = {uid: record for uid, record in load_run(baseline).items()
                                  if record["language"] == language}
                if expected_base.keys() != candidate_base.keys():
                    raise ValueError("Candidate and shared English encounter IDs differ")
                for uid in expected_base:
                    for key in ("audio_sha256", "reference_transcript", "language", "decoded_regions",
                                "seed", "model", "prediction", "chunks"):
                        if expected_base[uid][key] != candidate_base[uid][key]:
                            raise ValueError(f"English control recordings differ in {key}: {uid}")
                values = compare(baseline, english / (name + "-adapter"), target / (name + "-english.json"))["by_language"][language]
                directory, metadata = english / (name + "-adapter"), chosen
            else:
                values, directory, metadata = shared_values, shared / (name + "-adapter"), original
            if values["base"] != shared_values["base"]:
                raise ValueError("Candidate and shared baseline metrics differ")
            for record in load_run(directory).values():
                if record["language"] == language and record["model"]["adapter"]["sha256"] != metadata["adapter_sha256"]:
                    raise ValueError("Predictions used different adapter weights")
            rows.append({"dataset": name, "language": language, "examples": values["base"]["examples"],
                "selected_epoch": selected_epochs[language], "base_wer": values["base"]["wer"],
                "shared_adapter_wer": shared_values["adapter"]["wer"], "selected_adapter_wer": values["adapter"]["wer"],
                "base_cer": values["base"]["cer"], "selected_adapter_cer": values["adapter"]["cer"],
                "wer_change_percentage_points": 100 * values["wer_change"]})
    result = {"status": "complete", "exploratory": True,
        "selection": "Minimum language validation WER across completed epochs; ties choose earlier epoch",
        "selected_epochs": selected_epochs, "evaluated_recordings": sum(row["examples"] for row in rows),
        "improved_wer_groups": sum(row["selected_adapter_wer"] < row["base_wer"] for row in rows),
        "regressed_wer_groups": sum(row["selected_adapter_wer"] > row["base_wer"] for row in rows),
        "all_wer_groups_improved": all(row["selected_adapter_wer"] < row["base_wer"] for row in rows),
        "all_cer_groups_non_regressing": all(row["selected_adapter_cer"] <= row["base_cer"] for row in rows),
        "adapter_sha256_by_language": {language: (chosen if language == "en" else original)["adapter_sha256"] for language in selected_epochs},
        "comparisons": rows, "limitations": [*primary["limitations"], *candidate["limitations"],
            "Greedy short-segment validation differs from production decoding with temperature fallback and word timestamps.",
            "These are separate verified runs using known language hints; automatic adapter routing is not implemented.",
            "Retain the original shared-checkpoint results, including its English regressions.",
            "WER gains do not imply CER gains or clinical correctness."]}
    write_json(target / "compiled-results.json", result)
    with (target / "compiled-results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    lines = ["# Exploratory language-validation checkpoint selection", "", result["selection"], "",
        "| Dataset | Language | Clips | Epoch | Base WER | Shared WER | Selected WER | Base CER | Selected CER |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['dataset']} | {row['language']} | {row['examples']} | {row['selected_epoch']} | "
            f"{100*row['base_wer']:.2f}% | {100*row['shared_adapter_wer']:.2f}% | {100*row['selected_adapter_wer']:.2f}% | "
            f"{100*row['base_cer']:.2f}% | {100*row['selected_adapter_cer']:.2f}% |")
    lines += ["", *["- " + value for value in result["limitations"]]]
    (target / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-run", required=True)
    parser.add_argument("--english-run", required=True)
    parser.add_argument("--english-adapter", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    result = report(args.shared_run, args.english_run, args.english_adapter, args.outdir)
    print(json.dumps({key: result[key] for key in ("status", "selected_epochs", "improved_wer_groups", "regressed_wer_groups")}, indent=2))
