from __future__ import annotations

import csv
import re
from pathlib import Path


VOCABULARIES = {"UMLS", "RxNorm"}
ENTITY_TYPES = {"symptom", "condition", "drug"}


def _normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip()).casefold()


class TerminologyLinker:
    """Resolve extracted mentions using a user-provided local terminology table."""

    def __init__(self, terminology_path: str | Path):
        self.terminology_path = Path(terminology_path)
        if not self.terminology_path.is_file():
            raise FileNotFoundError(
                f"Terminology table not found: {self.terminology_path}"
            )
        self._index: dict[tuple[str, str], dict[tuple[str, str], dict[str, str]]] = {}
        with self.terminology_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            expected_columns = {
                "term", "vocabulary", "concept_id", "canonical_name", "entity_type"
            }
            if not reader.fieldnames or not expected_columns.issubset(reader.fieldnames):
                raise ValueError(
                    "Terminology table must include columns: "
                    + ", ".join(sorted(expected_columns))
                )
            for line_number, row in enumerate(reader, start=2):
                term = row["term"].strip()
                vocabulary = row["vocabulary"].strip()
                concept_id = row["concept_id"].strip()
                canonical_name = row["canonical_name"].strip()
                entity_type = row["entity_type"].strip().lower()
                if not all((term, vocabulary, concept_id, canonical_name, entity_type)):
                    raise ValueError(
                        f"Terminology table row {line_number} has an empty required value"
                    )
                if vocabulary not in VOCABULARIES:
                    raise ValueError(
                        f"Terminology table row {line_number} has unsupported vocabulary "
                        f"{vocabulary!r}"
                    )
                if entity_type not in ENTITY_TYPES:
                    raise ValueError(
                        f"Terminology table row {line_number} has unsupported entity type "
                        f"{entity_type!r}"
                    )
                concept = {
                    "vocabulary": vocabulary,
                    "concept_id": concept_id,
                    "canonical_name": canonical_name,
                    "entity_type": entity_type,
                }
                key = (_normalize_term(term), entity_type)
                concept_key = (vocabulary, concept_id)
                self._index.setdefault(key, {})[concept_key] = concept

    def link_facts(self, facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        linked_facts = []
        for fact in facts:
            linked = dict(fact)
            entity_type = fact["entity_type"]
            preferred_vocabulary = "RxNorm" if entity_type == "drug" else "UMLS"
            candidates = list(
                self._index.get(
                    (_normalize_term(fact["entity_text"]), entity_type), {}
                ).values()
            )
            preferred = [
                candidate for candidate in candidates
                if candidate["vocabulary"] == preferred_vocabulary
            ]
            selected = preferred or candidates
            if len(selected) == 1:
                status = "linked"
            elif selected:
                status = "ambiguous"
            else:
                status = "not_found"
            linked["entity_link"] = {
                "status": status,
                "preferred_vocabulary": preferred_vocabulary,
                "concepts": selected,
            }
            linked_facts.append(linked)
        return linked_facts
