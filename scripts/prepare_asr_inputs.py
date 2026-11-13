"""Prepare bounded real-audio ASR inputs from the explicitly downloaded selections."""
import csv
import json
import os
from pathlib import Path
import re

from voice_to_soap.benchmark_datasets import prepare_betrac
from voice_to_soap.asr_datasets import import_eka_asr_dataset


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    betrac = Path("data/processed/betrac-smoke")
    if not (betrac / "audio_manifest.csv").exists():
        prepare_betrac("data/raw/betrac/data/dev-00000.tar", betrac, 2)
    eka = Path("data/manifests/eka-smoke.csv")
    if not eka.exists():
        import_eka_asr_dataset("data/raw/eka", "data/processed/eka-smoke",
                              manifest_path=eka, limit_per_language=2)
    sully = Path("data/manifests/sully-asr-selection.csv")
    if not sully.exists():
        uid = "pm57-d1c01"
        rows = [json.loads(line) for line in Path("data/raw/sully-smoke/encounters.jsonl").read_text().splitlines() if line.strip()]
        selected = next(row for row in rows if row["encounter_id"] == uid)
        reference = re.sub(r"(?im)^(Clinician|Doctor|Patient|Family member):\s*", "", selected["transcript"])
        audio = Path("data/raw/sully-smoke/audio") / (uid + ".mp3")
        if not audio.is_file():
            raise FileNotFoundError("Selected Sully audio must be downloaded first")
        row = {"encounter_id": uid, "audio_path": os.path.relpath(audio, sully.parent),
               "transcript": "", "reference_note": "", "language": "en", "split": "validation",
               "reference_transcript": reference, "reference_transcript_raw": selected["transcript"],
               "dataset": "SullyHQ/open-clinical-conversations", "task": "asr_evaluation",
               "source": selected["source"], "source_split": "collection; no held-out claim",
               "license": selected["license"], "recording_context": "role_played_consultation"}
        sully.parent.mkdir(parents=True, exist_ok=True)
        with sully.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
    print("Prepared/reused four Eka clips, two complete BeTraC consultations, and one complete Sully consultation.")


if __name__ == "__main__":
    main()
