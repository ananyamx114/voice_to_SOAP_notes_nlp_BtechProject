from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class Utterance:
    id: str
    start: float | None
    end: float | None
    speaker_id: str
    role: str
    text_raw: str
    text_normalized: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_facts(
    payload: Any,
    valid_ids: set[str],
    utterances: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), list):
        raise ValueError("Fact extraction response must contain a 'facts' list")
    facts = payload["facts"]
    utterance_by_id = {item["id"]: item for item in utterances or []}
    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            raise ValueError(f"Fact {index} is not an object")
        required = (
            "category", "text", "assertion", "speaker", "entity_type",
            "entity_text", "evidence",
        )
        if not all(fact.get(key) for key in required):
            raise ValueError(f"Fact {index} is missing required fields")
        if fact["assertion"] not in {"present", "absent", "uncertain", "historical", "past"}:
            raise ValueError(f"Fact {index} has invalid assertion {fact['assertion']!r}")
        if fact["assertion"] == "past":
            fact["assertion"] = "historical"
        if fact["entity_type"] not in {"symptom", "condition", "drug", "other"}:
            raise ValueError(f"Fact {index} has invalid entity type {fact['entity_type']!r}")
        if not all(isinstance(fact[key], str) for key in required[:-1]):
            raise ValueError(f"Fact {index} text and classification fields must be strings")
        if not isinstance(fact["evidence"], list) or not fact["evidence"]:
            raise ValueError(f"Fact {index} must cite at least one utterance")
        if any(not isinstance(value, str) for value in fact["evidence"]):
            raise ValueError(f"Fact {index} evidence IDs must be strings")
        unknown = set(fact["evidence"]) - valid_ids
        if unknown:
            raise ValueError(f"Fact {index} cites unknown utterance IDs: {sorted(unknown)}")
        if utterance_by_id:
            evidence_roles = {
                utterance_by_id[uid].get("role", "unknown")
                for uid in fact["evidence"]
            }
            evidence_speakers = {
                utterance_by_id[uid].get("speaker_id", "unknown")
                for uid in fact["evidence"]
            }
            if fact["speaker"] not in evidence_roles | evidence_speakers:
                raise ValueError(
                    f"Fact {index} speaker {fact['speaker']!r} does not match its evidence"
                )
    return facts


def validate_soap(payload: Any, valid_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("SOAP response must be a JSON object")
    claims: list[dict[str, Any]] = []
    for section in ("subjective", "objective", "assessment", "plan"):
        values = payload.get(section, [])
        if not isinstance(values, list):
            raise ValueError(f"SOAP section {section!r} must be a list")
        for index, claim in enumerate(values):
            if (
                not isinstance(claim, dict)
                or not isinstance(claim.get("text"), str)
                or not claim["text"].strip()
            ):
                raise ValueError(f"Claim {section}[{index}] must have text")
            evidence = claim.get("evidence")
            if (
                not isinstance(evidence, list)
                or not evidence
                or any(not isinstance(value, str) or not value for value in evidence)
            ):
                raise ValueError(f"Claim {section}[{index}] must cite utterance IDs")
            unknown = set(evidence) - valid_ids
            if unknown:
                raise ValueError(f"Claim {section}[{index}] cites unknown IDs: {sorted(unknown)}")
            claims.append({"section": section, "text": claim["text"], "evidence": evidence})
    return claims
