"""Build dated report tables from aggregate-only ASR and SOAP snapshots."""
import argparse
import csv
import json
from pathlib import Path


def table(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(asr, soap, out):
    assert asr["status"] == "interim_training_validation_only"
    assert soap["status"] == "interim_report_snapshot_not_final_pipeline_evaluation"
    assert soap["dataset"]["internal_test"] == 200
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("asr-snapshot.json", asr), ("soap-snapshot.json", soap)):
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    validation = []
    for row in asr["completed_validation_epochs"]:
        for language, metric in row["by_language"].items():
            validation.append({"epoch": row["epoch"], "language": language,
                "segments": metric["examples"], "wer": metric["wer"], "cer": metric["cer"],
                "macro_language_wer": row["macro_language_wer"],
                "selected": row["epoch"] == asr["selected_validation_epoch"]})
    table(out / "asr-validation.csv", validation)
    train_rows = [{"approach": stage, **{k: row[k] for k in ("status", "saved_updates",
        "planned_updates", "saved_epoch", "best_validation_loss", "selected_checkpoint_step")}}
        for stage, row in soap["training"].items()]
    table(out / "soap-training.csv", train_rows)
    curves = [{"approach": stage, **r} for stage, row in soap["training"].items()
              for r in row["validation_history"]]
    table(out / "soap-validation.csv", curves)
    evaluation = []
    for stage, row in soap["evaluation"].items():
        n = row["examples"]
        assert n == row["valid_predictions"] + row["invalid_predictions"]
        evaluation.append({"model": stage, "status": row["status"], "examples": n,
            "valid_outputs": row["valid_predictions"], "invalid_outputs": row["invalid_predictions"],
            "valid_output_rate": row["valid_predictions"] / n if n else None,
            "mean_section_rougeL_all_examples": row["overall_mean_section_rougeL_all_examples"],
            "invalid_outputs_scored_as_zero": row["invalid_outputs_scored_as_zero"]})
    table(out / "soap-evaluation.csv", evaluation)
    selected = asr["selected_validation_metrics"]
    snapshot_date = soap["snapshot_utc"][:10]
    lines = [f"# Interim report metrics — {snapshot_date}", "",
        "This is a frozen interim snapshot, not final pipeline results. Jobs continued running during export.",
        f"SOAP snapshot: `{soap['snapshot_utc']}`. ASR progress observed: `{asr['progress_observed_utc']}`.", "",
        "## ASR: larger-cohort Whisper Large-v3 LoRA continuation", "",
        f"Training uses {asr['training_segments']:,} segments / {asr['training_hours']:.2f} hours, roughly four hours per language.",
        f"Validation uses {asr['validation_segments']} segments from {asr['validation_encounters']} encounters.",
        f"Live progress was update {asr['latest_live_update']}; the latest fully verified local continuation archive was update {asr['latest_locally_verified_update']}.",
        f"The selected checkpoint is epoch **{asr['selected_validation_epoch']}**, chosen using macro-language validation WER, not the latest update.", "",
        "| Language | Selected validation WER | Selected validation CER |", "|---|---:|---:|"]
    for language, metric in selected["by_language"].items():
        lines.append(f"| {language} | {100*metric['wer']:.2f}% | {100*metric['cer']:.2f}% |")
    lines += ["", f"Selected macro-language WER: **{100*sum(r['wer'] for r in selected['by_language'].values())/3:.2f}%**.",
        "These are diagnostic validation scores; no new independent test evaluation has completed for this continuation.",
        "[All available validation epochs, including regressions](asr-validation.csv) and [configuration/provenance](asr-snapshot.json).", "",
        "The completed [run2 paired test](../asr-lora-run2/README.md) remains a different experiment. Do not replace its test table with these validation scores.", "",
        "## SOAP: Qwen2.5-7B-Instruct with rank-8 LoRA", "",
        "BeTraC-2000 contains 1,600 training, 200 validation and 200 internal-test encounters, selected from official training data.",
        "Conversations/references are synthetic; four-string SOAP targets use heuristic mapping. This is not the evidence-cited claim schema.",
        "DIRECT uses the transcript. HYBRID adds the existing rule-based clinical facts; it is not a trained entity-extraction model.", "",
        "| Approach | Status | Saved updates | Epoch | Best validation loss |", "|---|---|---:|---:|---:|"]
    for row in train_rows:
        lines.append(f"| {row['approach']} | {row['status']} | {row['saved_updates']}/{row['planned_updates']} | {row['saved_epoch']} | {row['best_validation_loss']:.6f} |")
    lines += ["", "Validation loss is token prediction loss, not WER, ROUGE-L or clinical accuracy. Different inputs and training positions preclude claiming a winner from these losses.", "",
        "| Model | Evaluation status | Encounters | Valid / invalid | Mean section ROUGE-L |", "|---|---|---:|---:|---:|"]
    for row in evaluation:
        score = row["mean_section_rougeL_all_examples"]
        value = f"{score:.6f}" if score is not None else "not measured"
        lines.append(f"| {row['model']} | {row['status']} | {row['examples']}/200 | {row['valid_outputs']} / {row['invalid_outputs']} | {value} |")
    partial_note = ("**DIRECT evaluation is partial in this snapshot. Do not subtract its subset score from the full baseline score or describe the difference as a completed paired improvement.**"
                    if soap["evaluation"]["direct"]["status"] != "complete" else
                    "DIRECT and baseline evaluations are complete; verify identical test hashes and generation settings before claiming a paired comparison.")
    baseline = soap["evaluation"]["base"]
    direct = soap["training"]["direct"]
    language_wer = "/".join(f"{100*selected['by_language'][language]['wer']:.2f}%" for language in ("en", "hi", "mr"))
    lines += ["", "Invalid generations count as zero in the all-example ROUGE-L score. The valid-only baseline score is retained separately in JSON.",
        partial_note,
        "Output validity and factual correctness are different outcomes; regressions in validity must also be reported.",
        f"HYBRID generation evaluation status: {soap['evaluation']['hybrid']['status']}. Test cases must not be used for checkpoint selection or hyperparameter tuning.", "",
        "[Training summary](soap-training.csv), [validation curves](soap-validation.csv), [evaluation table](soap-evaluation.csv), [exact exported SOAP metrics](soap-snapshot.json).", "",
        "## Report wording and limits", "",
        f"> Full-precision Qwen2.5-7B-Instruct was adapted with rank-8 LoRA on 1,600 synthetic BeTraC encounters. DIRECT training status was {direct['status']} ({direct['saved_updates']} updates), with validation-selected loss {direct['best_validation_loss']:.6f}. The pretrained baseline evaluated {baseline['examples']} internal-test encounters ({baseline['valid_predictions']} valid, {baseline['invalid_predictions']} invalid; failure-aware mean section ROUGE-L {baseline['overall_mean_section_rougeL_all_examples']:.6f}). DIRECT generation evaluation status was {soap['evaluation']['direct']['status']}; HYBRID training status was {soap['training']['hybrid']['status']}. Incomplete outcomes must remain explicitly labelled in the report.", "",
        f"> Whisper Large-v3 rank-16 LoRA training continued on a roughly 12-hour multilingual cohort. The validation-selected epoch-{asr['selected_validation_epoch']} checkpoint achieved English/Hindi/Marathi WER of {language_wer}. These are validation results, not a new held-out ASR benchmark. The below-5% target is an unmet research objective for languages whose reported WER exceeds it.", "",
        "Two SOAP internal-test encounters also occur in ASR training. This does not contaminate the text-only SOAP training split, but those encounters cannot support independent audio-to-SOAP test claims.",
        "No end-to-end audio-to-SOAP evaluation, clinical factuality validation, multilingual SOAP evaluation or clinician approval is established by these tables.",
        "Every generated note still requires clinician approval. Model weights, optimizer state, transcripts, predictions, raw datasets and credentials are excluded from Git.", "",
        "## Refresh without stopping jobs", "",
        "On the GPU host, run the aggregate-only exporter:", "", "```bash",
        "python scripts/export_soap_interim.py --output .local/interim-soap-results.json", "```", "",
        "Transfer only that JSON through your own authenticated connection. Keep the ASR aggregate snapshot separate from private controller records.",
        "Rebuild a new dated snapshot using aggregate inputs; preserve older snapshots and completed experiments:", "", "```bash",
        "python scripts/build_interim_report.py --asr ASR_AGGREGATE.json --soap SOAP_AGGREGATE.json --outdir results/interim-2026-11-26", "```", "",
        "This builder exports tables only; it does not start, pause or otherwise control model jobs."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", type=Path, required=True)
    parser.add_argument("--soap", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    build(json.loads(args.asr.read_text()), json.loads(args.soap.read_text()), args.outdir)


if __name__ == "__main__":
    main()
