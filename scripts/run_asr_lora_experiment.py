"""Run local Whisper training and paired ASR evaluation sequentially, preserving artifacts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from voice_to_soap.asr_comparison import compare
from voice_to_soap.asr_training_data import check_disjoint, read_segments
from voice_to_soap.io_utils import write_json

ROOT = Path(__file__).resolve().parents[1]


def combine(inputs, target, resume=False):
    rows = []
    for filename in inputs:
        filename = Path(filename).resolve()
        preparation = json.loads((filename.parent / "preparation.json").read_text(encoding="utf-8"))
        if preparation["status"] != "complete":
            raise ValueError("Do not train on partial/failed preparations")
        for line in filename.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("audio_path", "original_audio_path"):
                if row.get(key):
                    absolute = (filename.parent / row[key]).resolve()
                    row[key] = Path(os.path.relpath(absolute, target.parent)).as_posix()
            rows.append(row)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if target.exists() and (not resume or target.read_text(encoding="utf-8") != payload):
        raise ValueError("Combined manifest already exists or its sources changed")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def run(args):
    root = Path(args.outdir).resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    started = perf_counter()
    report = {"status": "running", "stage": "data_audit", "evaluations": {},
        "evaluation_scope": ("previous seven diagnostic recordings" if not args.full_eka else
            "full Eka plus the two previous BeTraC and one Sully recordings") +
            (" plus the separately prepared held-out FLEURS selection" if args.fleurs_test else ""),
        "publication": "local only, explicitly requested by user", "rerun_base": args.rerun_base,
        "limitations": ["Validation selects the adapter; final comparisons are separate.",
            "Previously inspected Eka examples are diagnostic, not untouched final test data.",
            "Automatic alignment is not human-reviewed gold; general FLEURS is not clinical speech.",
            "A measured improvement on these recordings does not establish clinical readiness."]}
    def stage(name, command):
        report["stage"] = name
        write_json(root / "experiment.json", report)
        print(f"\nStarting stage: {name}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/project.py"), *command], cwd=ROOT, check=True)
    try:
        train_file, validation_file = root / "data/train.jsonl", root / "data/validation.jsonl"
        report["train_manifest_sha256"] = combine([
            Path(args.clinical_train) / "segments.jsonl", Path(args.fleurs) / "train.jsonl"], train_file, args.resume)
        report["validation_manifest_sha256"] = combine([
            Path(args.clinical_validation) / "segments.jsonl", Path(args.fleurs) / "validation.jsonl"], validation_file, args.resume)
        training, validation = read_segments(train_file), read_segments(validation_file)
        report["audit"] = check_disjoint(training, validation)
        report["training_by_language"] = {lang: {"segments": sum(r["language"] == lang for r in training),
            "audio_seconds": sum(r["end"] - r["start"] for r in training if r["language"] == lang)}
            for lang in sorted({r["language"] for r in training})}
        if args.fleurs_test:
            heldout = read_segments(Path(args.fleurs_test) / "test.jsonl")
            for key in ("id", "encounter_id", "group_id", "source_sha256"):
                if {r[key] for r in training + validation} & {r[key] for r in heldout}:
                    raise ValueError(f"Held-out FLEURS leakage in {key}")
            report["additional_fleurs_test_examples"] = len(heldout)
        print(json.dumps({"audit": report["audit"], "training_by_language": report["training_by_language"]}, indent=2), flush=True)
        if args.prepare_only:
            report.update(status="data_ready", stage="data_ready")
            return
        training_dir = root / "training"
        trainer_state = training_dir / "run.json"
        trained = args.resume and trainer_state.is_file() and json.loads(trainer_state.read_text(encoding="utf-8"))["status"] == "complete"
        if not trained:
            command = ["asr-train", "--config", args.config, "--train", str(train_file),
                "--validation", str(validation_file), "--outdir", str(training_dir), "--device", args.device]
            if args.resume and training_dir.exists():
                command.append("--resume")
                if args.resume_source_patch:
                    command += ["--resume-source-patch", args.resume_source_patch]
            stage("training_and_validation", command)
        report["training"] = json.loads(trainer_state.read_text(encoding="utf-8"))
        adapter = training_dir / "best-adapter"
        cases = [
            ("eka", "data/manifests/eka-medical-asr-evaluation.csv" if args.full_eka else "data/manifests/eka-smoke.csv",
             "data/outputs/asr-large-v3-eka-run2", None),
            ("betrac", "data/processed/betrac-smoke/audio_manifest.csv",
             "data/outputs/asr-large-v3-betrac", "data/outputs/vad-component-01"),
            ("sully", "data/manifests/sully-asr-selection.csv", "data/outputs/asr-large-v3-sully", None),
        ]
        if args.fleurs_test:
            cases.append(("fleurs", str(Path(args.fleurs_test) / "test_manifest.csv"), None, None))
        for name, manifest, historic_base, vad in cases:
            shared = ["--manifest", manifest, "--device", args.device]
            if vad:
                shared += ["--vad-dir", vad]
            if args.resume:
                shared += ["--resume"]
            rerun_base = args.rerun_base or historic_base is None or (name == "eka" and args.full_eka)
            base = root / (name + "-base") if rerun_base else ROOT / historic_base
            if rerun_base:
                stage(name + "_base", ["asr-run", *shared, "--outdir", str(base)])
            adapted = root / (name + "-adapter")
            stage(name + "_adapter", ["asr-run", *shared, "--adapter", str(adapter), "--outdir", str(adapted)])
            report["evaluations"][name] = compare(base, adapted, root / (name + "-comparison.json"))
            write_json(root / "experiment.json", report)
        report.update(status="complete", stage="complete")
        table = ["# Local Whisper LoRA experiment", "", "WER is a proportion; lower is better.", "",
                 "| Dataset | Language | Base WER | LoRA WER | Change |", "|---|---|---:|---:|---:|"]
        for name, result in report["evaluations"].items():
            for lang, metrics in result["by_language"].items():
                table.append(f"| {name} | {lang} | {metrics['base']['wer']:.5f} | {metrics['adapter']['wer']:.5f} | {metrics['wer_change']:+.5f} |")
        table += ["", report["evaluation_scope"], "", *report["limitations"]]
        (root / "RESULTS.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds_this_session"] = perf_counter() - started
        write_json(root / "experiment.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical-train", required=True)
    parser.add_argument("--clinical-validation", required=True)
    parser.add_argument("--fleurs", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--config", default="configs/asr-lora.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerun-base", action="store_true")
    parser.add_argument("--full-eka", action="store_true", help="Score all 3939 Eka clips for both base and adapter")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Build and audit manifests without loading models")
    parser.add_argument("--fleurs-test", help="Separate prepared held-out Hindi/Marathi test clips")
    parser.add_argument("--resume-source-patch", help="Document a compatible trainer correction when resuming")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
