from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .llm import LocalLLM


ALLOWED_ROLES = {"doctor", "patient", "unknown"}


def assign_speaker_roles(
    llm: LocalLLM,
    utterances: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    speaker_ids = sorted({
        item["speaker_id"]
        for item in utterances
        if item.get("speaker_id") and item["speaker_id"] != "unknown"
    })
    if not speaker_ids:
        return utterances, []

    known_roles = {
        item["speaker_id"]: item["role"]
        for item in utterances
        if item.get("speaker_id") in speaker_ids
        and item.get("role") in {"doctor", "patient"}
    }
    unresolved = [speaker_id for speaker_id in speaker_ids if speaker_id not in known_roles]
    if not unresolved:
        return utterances, [
            {
                "speaker_id": speaker_id,
                "role": known_roles[speaker_id],
                "evidence": [],
                "source": "provided_transcript",
            }
            for speaker_id in speaker_ids
        ]

    transcript_by_speaker = {
        speaker_id: [
            item for item in utterances
            if item.get("speaker_id") == speaker_id and item.get("text_raw", "").strip()
        ]
        for speaker_id in unresolved
    }
    evidence_lines = []
    for speaker_id in unresolved:
        lines = [
            f"{item['id']}: {item['text_raw'].strip()}"
            for item in transcript_by_speaker[speaker_id]
        ]
        evidence_lines.append(f"{speaker_id}:\n" + "\n".join(lines))

    response = llm.complete_json(
        "You assign conversation roles from a clinical consultation transcript. "
        "Return only valid JSON. Be conservative and use unknown when the text does not "
        "clearly distinguish the role. Speaker IDs are arbitrary and do not imply a role.",
        """For each listed speaker choose exactly one role: doctor, patient, or unknown.
Use the words as evidence: clinician questions, examination, medical instructions, or
explicit care-plan language may support doctor; first-person symptoms, history, and
answers about the patient's condition may support patient. Do not infer from speaking
time, voice identity, or speaker numbering. If evidence is mixed or weak, use unknown.
For doctor or patient, cite one or more utterance IDs belonging to that speaker.
Return {"roles":[{"speaker_id":"speaker_0","role":"unknown","evidence":[],"reason":"brief"}]}.

SPEAKER TRANSCRIPTS:
"""
        + "\n\n".join(evidence_lines),
    )
    predictions = response.get("roles")
    if not isinstance(predictions, list):
        raise ValueError("Role assignment response must contain a roles list")

    expected = set(unresolved)
    assignments: dict[str, dict[str, Any]] = {}
    valid_ids_by_speaker = {
        speaker_id: {item["id"] for item in transcript_by_speaker[speaker_id]}
        for speaker_id in unresolved
    }
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Role assignment entries must be objects")
        speaker_id, role = item.get("speaker_id"), item.get("role")
        if speaker_id not in expected:
            raise ValueError(f"Role assignment returned unexpected speaker ID {speaker_id!r}")
        if speaker_id in assignments:
            raise ValueError(f"Role assignment returned duplicate speaker ID {speaker_id!r}")
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Role assignment returned invalid role {role!r}")
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list) or any(not isinstance(value, str) for value in evidence):
            raise ValueError(f"Role assignment evidence for {speaker_id!r} must be a list of IDs")
        invalid_ids = set(evidence) - valid_ids_by_speaker[speaker_id]
        if invalid_ids:
            raise ValueError(
                f"Role assignment for {speaker_id!r} cites invalid utterance IDs: "
                f"{sorted(invalid_ids)}"
            )
        if role != "unknown" and not evidence:
            raise ValueError(f"Role {role!r} for {speaker_id!r} must cite evidence")
        assignments[speaker_id] = {
            "speaker_id": speaker_id,
            "role": role,
            "evidence": evidence,
            "reason": str(item.get("reason", "")),
            "source": "local_llm",
        }

    missing = expected - assignments.keys()
    if missing:
        raise ValueError(f"Role assignment omitted speakers: {sorted(missing)}")

    all_assignments = {
        **{
            speaker_id: {
                "speaker_id": speaker_id,
                "role": known_roles[speaker_id],
                "evidence": [],
                "source": "provided_transcript",
            }
            for speaker_id in known_roles
        },
        **assignments,
    }
    result = []
    for utterance in utterances:
        updated = dict(utterance)
        assignment = all_assignments.get(updated.get("speaker_id"))
        if assignment is not None and updated.get("role") not in {"doctor", "patient"}:
            updated["role"] = assignment["role"]
        result.append(updated)
    return result, [all_assignments[speaker_id] for speaker_id in speaker_ids]
