from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any


SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")
_ROLE_PREFIX = re.compile(r"^\s*\[(doctor|patient|unknown)\]\s*(.*)$", re.IGNORECASE)
_COLON_ROLE_PREFIX = re.compile(r"^\s*(doctor|patient)\s*:\s*(.*)$", re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(r"[.!?;]|\b(?:but|however|although)\b", re.IGNORECASE)
_DOSE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|mL|units?)\b"
    r"(?:\s*(?:by mouth|orally|daily|once|twice|three times|every \w+))?",
    re.IGNORECASE,
)

_TERMS: dict[str, tuple[str, ...]] = {
    "symptom": (
        "shortness of breath", "chest pain", "nasal congestion", "abdominal pain",
        "sore throat", "swelling", "nausea", "vomiting", "cough", "fever",
        "headache", "pain", "fatigue", "dizziness",
    ),
    "condition": (
        "congestive heart failure", "heart failure", "hypertension",
        "high blood pressure", "diabetes", "depression", "asthma",
    ),
    "finding": (
        "blood pressure", "pitting edema", "edema", "murmur", "lungs",
        "heart exam", "cholesterol",
    ),
    "procedure": (
        "echocardiogram", "lipid panel", "mammogram", "lab test", "x-ray",
    ),
    "allergy": ("allergy", "allergies"),
}
_PLAN_CUES = re.compile(
    r"\b(?:plan|start|continue|increase|decrease|stop|order|repeat|schedule|"
    r"recommend|advise|follow[- ]?up|record|monitor|take)\b",
    re.IGNORECASE,
)
_OBJECTIVE_CUES = re.compile(
    r"\b(?:on (?:physical )?exam|examination|vitals?|today in the office|"
    r"showed|shows|result(?:s)?|measured|auscultat\w*)\b",
    re.IGNORECASE,
)
_ASSESSMENT_CUES = re.compile(
    r"\b(?:assessment|diagnos\w*|impression|due to|consistent with|"
    r"from a .* standpoint)\b",
    re.IGNORECASE,
)
_ABSENT_CUES = re.compile(
    r"\b(?:no|not|never|denies?|without|negative for|free of|absent)\b"
    r"|n't\b",
    re.IGNORECASE,
)
_UNCERTAIN_CUES = re.compile(
    r"\b(?:possible|possibly|may|might|could|suspected|uncertain|"
    r"question of|rule out|unclear)\b",
    re.IGNORECASE,
)
_HISTORICAL_CUES = re.compile(
    r"\b(?:history of|past medical history|medical history|past history|"
    r"historically|previously|prior|last (?:week|month|year)|resolved|used to|had)\b",
    re.IGNORECASE,
)


def parse_dialogue_utterances(dialogue: str) -> list[dict[str, Any]]:
    """Split line-oriented transcripts while preserving the original utterance text."""
    utterances = []
    for line in dialogue.splitlines():
        text = line.strip()
        if not text:
            continue
        match = _ROLE_PREFIX.match(text) or _COLON_ROLE_PREFIX.match(text)
        if match:
            role, content = match.group(1).casefold(), match.group(2).strip()
        else:
            role, content = "unknown", text
        if content:
            utterances.append({
                "id": f"U{len(utterances) + 1}",
                "role": role,
                "text": content,
            })
    if not utterances:
        raise ValueError("Transcript contains no non-empty utterances")
    return utterances


def _medications(lookup_path: str | Path | None) -> list[tuple[str, str]]:
    path = Path(lookup_path) if lookup_path else Path("data/lookup_tables/drug_names.csv")
    if not path.is_file():
        return []
    results = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            mention = (row.get("heard") or "").strip()
            normalized = (row.get("generic") or "").strip()
            if mention:
                results.append((mention, normalized or mention))
    return results


def _clause_bounds(text: str, start: int, end: int) -> tuple[str, str]:
    left_matches = list(_CLAUSE_BOUNDARY.finditer(text, 0, start))
    right_match = _CLAUSE_BOUNDARY.search(text, end)
    left = left_matches[-1].end() if left_matches else 0
    right = right_match.start() if right_match else len(text)
    suffix = text[end:right]
    if right_match and right_match.group(0) == "?":
        suffix += "?"
    return text[left:start], suffix


def _section_for(
    role: str,
    clause: str,
    category: str,
    assertion: str,
) -> str:
    if role == "patient":
        return "subjective"
    if _PLAN_CUES.search(clause):
        return "plan"
    if _OBJECTIVE_CUES.search(clause) or (
        category == "finding" and re.search(r"\b(?:exam|vitals?|today)\b", clause, re.I)
    ):
        return "objective"
    if category == "condition" and assertion == "historical":
        return "subjective"
    if _ASSESSMENT_CUES.search(clause):
        return "assessment"
    if role == "doctor" and category == "condition":
        return "assessment"
    return "unknown"


def _assertion_for(text: str, start: int, end: int) -> str:
    prefix, suffix = _clause_bounds(text, start, end)
    prefix = prefix[-100:]
    if _ABSENT_CUES.search(prefix) or re.match(r"\s*(?:is|are)\s+absent\b", suffix, re.I):
        return "absent"
    if (
        _UNCERTAIN_CUES.search(prefix)
        or re.match(r"\s*(?:may|might|could)\b", suffix, re.I)
        or "?" in suffix
    ):
        return "uncertain"
    if _HISTORICAL_CUES.search(prefix):
        return "historical"
    return "present"


def extract_clinical_facts(
    dialogue: str,
    medication_lookup: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Extract lexicon-matched clinical mentions with simple assertion and section rules."""
    utterances = parse_dialogue_utterances(dialogue)
    terms = [
        (term, category)
        for category, values in _TERMS.items()
        for term in values
    ]
    medication_candidates = {
        mention.casefold(): normalized
        for mention, normalized in _medications(medication_lookup)
    }
    terms.extend((mention, "medication") for mention in medication_candidates)
    terms.sort(key=lambda item: (-len(item[0]), item[0].casefold()))
    facts: list[dict[str, Any]] = []
    for utterance in utterances:
        text = utterance["text"]
        matches: list[tuple[int, int, str, str]] = []
        for term, category in terms:
            pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
            for match in pattern.finditer(text):
                matches.append((match.start(), match.end(), category, term))
        matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        accepted: list[tuple[int, int, str, str]] = []
        for match in matches:
            start, end = match[:2]
            if any(start < prev_end and end > prev_start for prev_start, prev_end, *_ in accepted):
                continue
            accepted.append(match)
        for start, end, category, _ in accepted:
            mention = text[start:end]
            clause_prefix, clause_suffix = _clause_bounds(text, start, end)
            clause = f"{clause_prefix}{mention}{clause_suffix}"
            assertion = _assertion_for(text, start, end)
            fact: dict[str, Any] = {
                "concept": mention,
                "category": category,
                "assertion": assertion,
                "speaker": utterance["role"],
                "section": _section_for(
                    utterance["role"], clause, category, assertion
                ),
                "evidence": utterance["id"],
                "utterance_text": text,
                "span_start": start,
                "span_end": end,
            }
            if category == "medication":
                fact["normalized_candidate"] = medication_candidates.get(mention.casefold())
                dose = _DOSE.search(text[max(0, start - 45):min(len(text), end + 80)])
                if dose:
                    fact["dose_text"] = dose.group(0)
            facts.append(fact)
    return facts
