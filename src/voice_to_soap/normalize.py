from pathlib import Path
import csv
import re
from typing import Any


_MASS_TO_MG = {
    "mg": 1.0,
    "g": 1000.0,
    "mcg": 0.001,
    "ug": 0.001,
    "μg": 0.001,
}


def _read_lookup(path: Path, key: str, value: str) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row[key].strip().lower(): row[value].strip()
            for row in csv.DictReader(stream)
            if row.get(key, "").strip() and row.get(value, "").strip()
        }


def _read_terms(path: Path, column: str) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        return {
            row[column].strip().lower()
            for row in csv.DictReader(stream)
            if row.get(column, "").strip()
        }


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?<!\w){re.escape(phrase.strip())}(?!\w)",
        flags=re.IGNORECASE,
    )


class Normalizer:
    def __init__(self, lookup_dir: str | Path):
        root = Path(lookup_dir)
        self.drugs = _read_lookup(root / "drug_names.csv", "heard", "generic")
        self.abbreviations = _read_lookup(root / "abbreviations.csv", "heard", "canonical")
        self.disfluencies = _read_terms(root / "disfluencies.csv", "heard")
        self._disfluency_patterns = [
            (phrase, _phrase_pattern(phrase))
            for phrase in sorted(self.disfluencies, key=len, reverse=True)
        ]
        self._drug_patterns = [
            (phrase, replacement, _phrase_pattern(phrase))
            for phrase, replacement in sorted(
                self.drugs.items(), key=lambda item: len(item[0]), reverse=True
            )
        ]
        self._abbreviation_patterns = [
            (phrase, replacement, _phrase_pattern(phrase))
            for phrase, replacement in sorted(
                self.abbreviations.items(), key=lambda item: len(item[0]), reverse=True
            )
        ]
        self.dose_ranges: list[dict[str, Any]] = []
        dose_path = root / "dose_ranges.csv"
        if dose_path.exists():
            with dose_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames:
                    legacy_columns = {"drug", "min_mg", "max_mg"}
                    current_columns = {
                        "drug", "min_value", "max_value", "unit", "basis", "route"
                    }
                    if not legacy_columns.issubset(reader.fieldnames) and not current_columns.issubset(
                        reader.fieldnames
                    ):
                        raise ValueError(
                            f"{dose_path} must use either the legacy drug,min_mg,max_mg "
                            "columns or drug,min_value,max_value,unit,basis,route"
                        )
                    for line_number, row in enumerate(reader, start=2):
                        drug = row.get("drug", "").strip().lower()
                        if not drug:
                            continue
                        if legacy_columns.issubset(reader.fieldnames):
                            minimum = row.get("min_mg", "").strip()
                            maximum = row.get("max_mg", "").strip()
                            unit, basis, route = "mg", "unspecified", ""
                        else:
                            minimum = row.get("min_value", "").strip()
                            maximum = row.get("max_value", "").strip()
                            unit = row.get("unit", "").strip().lower()
                            basis = row.get("basis", "").strip().lower()
                            route = row.get("route", "").strip().lower()
                        if not minimum and not maximum:
                            continue
                        try:
                            low, high = float(minimum), float(maximum)
                        except ValueError as exc:
                            raise ValueError(
                                f"{dose_path}:{line_number} has invalid dose bounds"
                            ) from exc
                        if low < 0 or high < low:
                            raise ValueError(
                                f"{dose_path}:{line_number} has invalid dose range {low}..{high}"
                            )
                        if unit not in {*_MASS_TO_MG, "ml", "tablet", "tablets", "capsule", "capsules"}:
                            raise ValueError(
                                f"{dose_path}:{line_number} has unsupported unit {unit!r}"
                            )
                        self.dose_ranges.append({
                            "drug": drug,
                            "min_value": low,
                            "max_value": high,
                            "unit": unit,
                            "basis": basis or "unspecified",
                            "route": route,
                        })

    @staticmethod
    def _clean_spacing(text: str) -> str:
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,;:])\s*[,;:]+", r"\1", text)
        text = re.sub(r"^[,;:]\s*", "", text)
        text = re.sub(r"\s*[,;:]\s*$", "", text)
        return re.sub(r"\s+", " ", text).strip()

    def normalize(
        self, utterances: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        changes: list[dict[str, Any]] = []
        normalized_utterances = []
        for utterance in utterances:
            original = utterance["text_raw"]
            text = original
            operations: list[dict[str, str]] = []
            for phrase, pattern in self._disfluency_patterns:
                text, count = pattern.subn("", text)
                if count:
                    operations.append({
                        "type": "disfluency_removal",
                        "heard": phrase,
                        "replacement": "",
                    })
            text = self._clean_spacing(text)

            for phrase, generic, pattern in self._drug_patterns:
                text, count = pattern.subn(lambda _: generic, text)
                if count:
                    operations.append({
                        "type": "drug_name_mapping",
                        "heard": phrase,
                        "replacement": generic,
                    })
            for phrase, canonical, pattern in self._abbreviation_patterns:
                text, count = pattern.subn(lambda _: canonical, text)
                if count:
                    operations.append({
                        "type": "abbreviation_expansion",
                        "heard": phrase,
                        "replacement": canonical,
                    })
            text = self._clean_spacing(text)

            normalized = dict(utterance)
            normalized["text_normalized"] = text
            normalized_utterances.append(normalized)
            if original != text:
                changes.append({
                    "utterance_id": utterance["id"],
                    "before": original,
                    "after": text,
                    "operations": operations,
                })
        return normalized_utterances, changes

    def check_dose_details(self, text: str) -> list[dict[str, str]]:
        warnings: list[dict[str, str]] = []
        if not self.dose_ranges:
            return warnings
        dose_pattern = re.compile(
            r"(?<!\w)(\d+(?:\.\d+)?)\s*"
            r"(mg|g|mcg|ug|μg|ml|tablets?|capsules?)(?!\w)",
            flags=re.IGNORECASE,
        )
        for dose_range in self.dose_ranges:
            drug_pattern = _phrase_pattern(dose_range["drug"])
            for sentence in re.split(r"(?<=[.!?;])\s+", text):
                if not drug_pattern.search(sentence):
                    continue
                doses = list(dose_pattern.finditer(sentence))
                for dose_match in doses:
                    amount = float(dose_match.group(1))
                    observed_unit = dose_match.group(2).lower()
                    expected_unit = dose_range["unit"]
                    message: str | None = None
                    if expected_unit in _MASS_TO_MG:
                        if observed_unit not in _MASS_TO_MG:
                            message = (
                                f"Cannot compare {dose_range['drug']} dose {amount:g} "
                                f"{observed_unit} with the table unit {expected_unit}"
                            )
                        else:
                            observed_mg = amount * _MASS_TO_MG[observed_unit]
                            low_mg = dose_range["min_value"] * _MASS_TO_MG[expected_unit]
                            high_mg = dose_range["max_value"] * _MASS_TO_MG[expected_unit]
                            if not low_mg <= observed_mg <= high_mg:
                                message = (
                                    f"Check {dose_range['drug']} dose {amount:g} "
                                    f"{observed_unit} against the configured range "
                                    f"{dose_range['min_value']:g}-{dose_range['max_value']:g} "
                                    f"{expected_unit} ({dose_range['basis']})"
                                )
                    elif observed_unit.rstrip("s") != expected_unit.rstrip("s"):
                        message = (
                            f"Cannot compare {dose_range['drug']} dose {amount:g} "
                            f"{observed_unit} with the table unit {expected_unit}"
                        )
                    elif not dose_range["min_value"] <= amount <= dose_range["max_value"]:
                        message = (
                            f"Check {dose_range['drug']} dose {amount:g} {observed_unit} "
                            f"against the configured range {dose_range['min_value']:g}-"
                            f"{dose_range['max_value']:g} {expected_unit} "
                            f"({dose_range['basis']})"
                        )
                    if message:
                        warnings.append({
                            "drug": dose_range["drug"],
                            "dose": f"{amount:g} {observed_unit}",
                            "basis": dose_range["basis"],
                            "message": message,
                        })
        return warnings

    def check_doses(self, text: str) -> list[str]:
        return [warning["message"] for warning in self.check_dose_details(text)]
