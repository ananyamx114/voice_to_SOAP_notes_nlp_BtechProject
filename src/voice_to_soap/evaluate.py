import argparse
import json
from pathlib import Path
from rouge_score import rouge_scorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated note JSONL with reference note JSONL.")
    parser.add_argument("--predictions", required=True, help="JSONL rows: encounter_id, note")
    parser.add_argument("--references", required=True, help="JSONL rows: encounter_id, reference_note")
    args = parser.parse_args()

    def load(path: str, key: str) -> dict[str, str]:
        data = {}
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row.get("encounter_id") and row.get(key):
                        data[row["encounter_id"]] = row[key]
        return data

    predictions = load(args.predictions, "note")
    references = load(args.references, "reference_note")
    ids = sorted(predictions.keys() & references.keys())
    if not ids:
        raise ValueError("No shared encounter IDs with non-empty notes were found")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(references[encounter_id], predictions[encounter_id])["rougeL"].fmeasure
        for encounter_id in ids
    ]
    print(json.dumps({"n": len(scores), "rougeL_mean": sum(scores) / len(scores)}, indent=2))


if __name__ == "__main__":
    main()
