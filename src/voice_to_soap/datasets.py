import csv
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {"encounter_id", "audio_path", "transcript", "reference_note", "language", "split"}


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest}")

    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        normalized_headers = [header.strip() for header in headers]
        if len(normalized_headers) != len(set(normalized_headers)):
            raise ValueError("Manifest contains duplicate column names")
        columns = set(normalized_headers)
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for line, source_row in enumerate(reader, start=2):
            if None in source_row:
                raise ValueError(f"Manifest line {line} has more values than there are columns")
            row = {
                (key or "").strip(): (value or "").strip()
                for key, value in source_row.items()
            }
            encounter_id = row.get("encounter_id", "")
            if not encounter_id:
                raise ValueError(f"Missing encounter_id on manifest line {line}")
            if encounter_id in seen:
                raise ValueError(f"Duplicate encounter_id {encounter_id!r}")
            seen.add(encounter_id)

            for key in REQUIRED_COLUMNS:
                row.setdefault(key, "")
            if not row["language"]:
                raise ValueError(f"Encounter {encounter_id!r} has no language on line {line}")
            if not row["split"]:
                raise ValueError(f"Encounter {encounter_id!r} has no split on line {line}")
            if not row["transcript"] and not row["audio_path"]:
                raise ValueError(f"Encounter {encounter_id!r} needs audio_path or transcript")

            if row["audio_path"]:
                audio_path = Path(row["audio_path"])
                if not audio_path.is_absolute():
                    audio_path = manifest.parent / audio_path
                audio_path = audio_path.resolve()
                if not audio_path.is_file():
                    raise FileNotFoundError(
                        f"Audio file for encounter {encounter_id!r} not found: {audio_path}"
                    )
                row["audio_path"] = str(audio_path)
            rows.append(row)

    if not rows:
        raise ValueError(f"Dataset manifest contains no encounters: {manifest}")
    return rows


def parse_transcript(text: str) -> list[dict[str, Any]]:
    utterances = []
    for index, raw_line in enumerate(text.replace("\r", "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        speaker = "unknown"
        content = line
        if line.startswith("[") and "]" in line:
            label, content = line[1:].split("]", 1)
            speaker = label.strip().lower()
            content = content.strip()
        role = speaker if speaker in {"doctor", "patient"} else "unknown"
        utterances.append({
            "id": f"U{len(utterances) + 1}",
            "start": None,
            "end": None,
            "speaker_id": speaker,
            "role": role,
            "text_raw": content,
            "text_normalized": content,
            "confidence": None,
        })
    if not utterances and text.strip():
        utterances.append({
            "id": "U1", "start": None, "end": None, "speaker_id": "unknown",
            "role": "unknown", "text_raw": text.strip(), "text_normalized": text.strip(),
            "confidence": None,
        })
    return utterances
