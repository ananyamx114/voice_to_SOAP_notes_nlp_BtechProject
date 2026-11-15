"""Diagnostic English comparison of a checkpoint chosen only by English validation."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from voice_to_soap.asr_comparison import compare
from voice_to_soap.io_utils import write_json


def evaluate(run_dir, adapter, output, device, resume=False):
    root = Path(__file__).resolve().parents[1]
    original, target = Path(run_dir).resolve(), Path(output).resolve()
    if json.loads((original / "experiment.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise ValueError("Run model stages sequentially; finish the shared-adapter experiment first")
    target.mkdir(parents=True, exist_ok=resume)
    source = root / "data/manifests/eka-smoke.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8")))
    english = [row for row in rows if row["language"] == "en"]
    for row in english:
        row["audio_path"] = Path(os.path.relpath((source.parent / row["audio_path"]).resolve(), target)).as_posix()
    manifest = target / "eka-english.csv"
    with manifest.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(english[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(english)
    report = {"status": "running", "selection": "minimum English validation WER; no test-score selection",
              "comparisons": {}, "limitations": ["Exploratory diagnostic comparison on previously inspected clinical samples.",
                  "Do not choose a winner per test encounter; retain every candidate result."]}
    def run(arguments):
        subprocess.run([sys.executable, str(root / "scripts/project.py"), "asr-run", *arguments], cwd=root, check=True)
    try:
        shared = ["--device", device] + (["--resume"] if resume else [])
        run([*shared, "--manifest", str(manifest), "--outdir", str(target / "eka-base")])
        cases = [("eka", str(manifest), target / "eka-base", None),
                 ("betrac", "data/processed/betrac-smoke/audio_manifest.csv", original / "betrac-base", "data/outputs/vad-component-01"),
                 ("sully", "data/manifests/sully-asr-selection.csv", original / "sully-base", None)]
        for name, audio_manifest, baseline, vad in cases:
            report["stage"] = name
            write_json(target / "experiment.json", report)
            arguments = [*shared, "--manifest", audio_manifest, "--adapter", str(Path(adapter).resolve()),
                         "--outdir", str(target / (name + "-adapter"))]
            if vad:
                arguments += ["--vad-dir", vad]
            run(arguments)
            report["comparisons"][name] = compare(baseline, target / (name + "-adapter"), target / (name + "-comparison.json"))
            write_json(target / "experiment.json", report)
        report.update(status="complete", stage="complete")
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_json(target / "experiment.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    evaluate(args.run_dir, args.adapter, args.outdir, args.device, args.resume)
