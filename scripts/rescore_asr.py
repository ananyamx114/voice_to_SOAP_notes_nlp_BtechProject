"""Rescore saved ASR predictions under both protocols; no inference or raw-text export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from voice_to_soap.asr_scoring import CURRENT, LEGACY, score_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    text = args.predictions.read_text(encoding="utf-8")
    rows = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get(key), str) for row in rows
                       for key in ("reference_transcript", "prediction", "language")):
        raise ValueError("Every prediction needs its raw reference, hypothesis and language")
    report = {"source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              "examples": len(rows), "protocols": {}, "scope": "Rescoring saved text; no new ASR run or clinical validation"}
    for protocol in (LEGACY, CURRENT):
        groups = {lang: score_records([r for r in rows if r["language"] == lang], protocol)
                  for lang in sorted({r["language"] for r in rows})}
        report["protocols"][protocol] = {"overall": score_records(rows, protocol), "by_language": groups,
            "macro_language_wer": sum(m["wer"] for m in groups.values()) / len(groups)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == args.predictions.resolve():
        raise ValueError("Never overwrite prediction evidence")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
