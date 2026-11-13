"""Prepare BeTraC audio and gold-transcript manifests without executing dataset code."""
import argparse
import csv
import json
from pathlib import Path
import re
import shutil
import tarfile

import soundfile as sf

from .io_utils import write_json


SUFFIXES = (".opus", ".transcript.txt", ".soap.txt", ".json")
FIELDS = ("encounter_id", "audio_path", "transcript", "reference_note", "language",
          "split", "dataset", "source_id", "source_split", "reference_transcript",
          "task", "reference_kind", "metadata_path")


def prepare_betrac(archive: str | Path, output_dir: str | Path,
                   limit: int = 2) -> dict:
    """Export dev samples, retaining synthetic note provenance and speaker labels.

    Gold references are never sent to ASR: the audio manifest's transcript field
    is empty. A separate manifest explicitly supplies the labeled transcript.
    This exporter does not create utterance-level evidence labels.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    rows = []
    with tarfile.open(archive, "r:*") as stream:
        groups = {}
        for member in stream.getmembers():
            path = Path(member.name)
            if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
                raise ValueError("Unsafe or nested archive member name")
            if not member.isfile():
                raise ValueError("Archive contains a non-regular member")
            suffix = next((s for s in SUFFIXES if member.name.endswith(s)), None)
            if suffix is None:
                continue
            key = member.name[:-len(suffix)]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
                raise ValueError("Unsafe sample identifier")
            record = groups.setdefault(key, {})
            if suffix in record:
                raise ValueError(f"Duplicate member for {key}")
            record[suffix] = member
        selected = list(groups.items())[:limit]
        if not selected:
            raise ValueError("No BeTraC samples found")
        for key, members in selected:
            if set(members) != set(SUFFIXES):
                raise ValueError(f"Incomplete BeTraC sample: {key}")
        output.mkdir(parents=True)
        for key, members in selected:
            paths = {}
            for suffix, member in members.items():
                target = output / (key + suffix)
                with stream.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                paths[suffix] = target
            transcript = paths[".transcript.txt"].read_text(encoding="utf-8").strip()
            note = paths[".soap.txt"].read_text(encoding="utf-8").strip()
            metadata = json.loads(paths[".json"].read_text(encoding="utf-8"))
            if not transcript or not note or not isinstance(metadata, dict):
                raise ValueError(f"Empty transcript/note or invalid metadata: {key}")
            info = sf.info(paths[".opus"])
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError(f"Invalid audio: {key}")
            labeled = re.sub(r"(?m)^(DOCTOR|PATIENT):\s*",
                             lambda m: f"[{m[1].lower()}] ", transcript)
            reference = re.sub(r"(?m)^(DOCTOR|PATIENT):\s*", "", transcript)
            rows.append({"encounter_id": f"betrac-dev-{key}",
                         "audio_path": paths[".opus"].name, "transcript": "",
                         "reference_note": note, "language": "en", "split": "validation",
                         "dataset": "BeTraC/betrac-2026", "source_id": key,
                         "source_split": "dev", "reference_transcript": reference,
                         "task": "audio_to_soap_evaluation",
                         "reference_kind": "synthetic_soap_without_evidence_labels",
                         "metadata_path": paths[".json"].name,
                         "_gold_transcript": labeled})
        for filename, gold in (("audio_manifest.csv", False), ("gold_manifest.csv", True)):
            with (output / filename).open("w", newline="", encoding="utf-8") as destination:
                writer = csv.DictWriter(destination, fieldnames=FIELDS)
                writer.writeheader()
                for row in rows:
                    record = {key: value for key, value in row.items() if key in FIELDS}
                    if gold:
                        record["transcript"] = row["_gold_transcript"]
                    writer.writerow(record)
    report = {"dataset": "BeTraC/betrac-2026", "source_split": "dev",
              "manifest_split": "validation", "archive": str(Path(archive).resolve()),
              "available_samples": len(groups), "exported_samples": len(rows),
              "audio_manifest": str(output / "audio_manifest.csv"),
              "gold_manifest": str(output / "gold_manifest.csv"),
              "limitations": ["Synthetic conversations and SOAP references.",
                              "No gold utterance-level claim citations.",
                              "Development subset; not an untouched final test set."]}
    write_json(output / "preparation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(prepare_betrac(args.archive, args.output_dir, args.limit), indent=2))


if __name__ == "__main__":
    main()
