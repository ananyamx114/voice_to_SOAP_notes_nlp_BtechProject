"""Compare matched full-encounter ASR runs without hiding missing/failed audio."""
import argparse
import json
from pathlib import Path
import re

from .asr_datasets import _metric_summary
from .io_utils import write_json
from .asr_scoring import CURRENT, PROTOCOLS


def load_run(directory):
    root = Path(directory)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if summary["requested"] != summary["completed"] or summary["failed"] or summary["not_processed"]:
        raise ValueError("Compare only complete runs; do not discard failed encounters")
    rows = {}
    for encounter in summary["encounters"]:
        uid = encounter["encounter_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", uid) or uid in rows:
            raise ValueError("Invalid or duplicate encounter ID")
        row = json.loads((root / uid / "asr.json").read_text(encoding="utf-8"))
        if row["status"] != "complete" or not row.get("reference_transcript", "").strip():
            raise ValueError("Every encounter needs completed ASR and a reference")
        rows[uid] = row
    if len(rows) != summary["requested"] or not rows:
        raise ValueError("Run does not contain every requested encounter")
    return rows


def compare(base_directory, adapted_directory, output, scoring_protocol=CURRENT):
    base, adapted = load_run(base_directory), load_run(adapted_directory)
    if base.keys() != adapted.keys():
        raise ValueError("Base and adapted runs must cover exactly the same encounters")
    for uid in base:
        left, right = base[uid], adapted[uid]
        for key in ("audio_sha256", "reference_transcript", "language", "decoded_regions", "seed"):
            if left[key] != right[key]:
                raise ValueError(f"Paired evaluation mismatch in {key}: {uid}")
        for key in ("model_id", "revision", "precision", "device", "generation", "algorithm", "timestamps"):
            if left["model"][key] != right["model"][key]:
                raise ValueError(f"Paired model/decoding mismatch in {key}: {uid}")
        if left["model"].get("implementation_sha256") != right["model"].get("implementation_sha256"):
            raise ValueError("Paired decoder implementations differ")
        if left["model"].get("adapter") or not right["model"].get("adapter"):
            raise ValueError("Expected an unmodified base run and an explicit ASR adapter run")
    report = {"encounters": len(base), "paired_ids": sorted(base), "by_language": {},
              "scoring_protocol": scoring_protocol,
              "limitations": ["WER/CER are lexical overlap, not clinical accuracy.",
                "Previously inspected Eka samples are diagnostic, not an untouched final test.",
                "A gain on synthetic English does not establish Hindi/Marathi improvement.",
                "Review medication/dose, negation, omissions and timestamps separately."]}
    for language in sorted({r["language"] for r in base.values()}):
        ids = [uid for uid in base if base[uid]["language"] == language]
        before = _metric_summary([base[uid] for uid in ids], scoring_protocol)
        after = _metric_summary([adapted[uid] for uid in ids], scoring_protocol)
        report["by_language"][language] = {"base": before, "adapter": after,
            "wer_change": after["wer"] - before["wer"], "cer_change": after["cer"] - before["cer"],
            "improved_wer": after["wer"] < before["wer"]}
    write_json(Path(output), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scoring-protocol", choices=PROTOCOLS, default=CURRENT)
    args = parser.parse_args()
    print(json.dumps(compare(args.base, args.adapted, args.output, args.scoring_protocol), indent=2))


if __name__ == "__main__":
    main()
