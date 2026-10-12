import re
from typing import Any
from .llm import LocalLLM
from .normalize import Normalizer
from .schemas import validate_facts, validate_soap


def extract_facts(llm: LocalLLM, utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    transcript = "\n".join(
        f"{u['id']} [{u['role']}|{u['speaker_id']}]: {u['text_normalized']}" for u in utterances
    )
    system = (
        "You extract atomic, explicitly stated clinical facts from a consultation. "
        "Return only valid JSON. Do not diagnose or fill gaps using medical knowledge."
    )
    user = f"""Extract separate facts for clinically meaningful symptoms, conditions,
medicines, allergies, procedures, and findings. Preserve the meaning and the
patient/clinician source. For each fact return:
- category: a short label such as symptom, condition, medication, allergy, procedure, finding
- text: the concise fact including its meaning (for example "fever denied")
- assertion: present, absent, uncertain, or historical. "No fever" is absent;
"might have fever" is uncertain; a symptom that occurred before but is not current is
historical. Do not convert absent, uncertain, or historical facts into present facts.
- speaker: the role or speaker_id of the person whose words support the fact
- entity_type: symptom, condition, drug, or other
- entity_text: the exact clinical concept or drug mention to link (for example "fever")
- evidence: one or more exact utterance IDs from this transcript that support the fact

Every fact must cite evidence. Do not cite IDs that are not shown. Do not add a fact
merely because it is medically plausible. Return
{{"facts":[{{"category":"symptom","text":"fever denied","assertion":"absent",
"speaker":"patient","entity_type":"symptom","entity_text":"fever","evidence":["U1"]}}]}}.

TRANSCRIPT:
{transcript}"""
    payload = llm.complete_json(system, user)
    return validate_facts(payload, {u["id"] for u in utterances}, utterances)


def generate_claims(
    llm: LocalLLM,
    utterances: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    soap_generator: Any | None = None,
    feedback: list[str] | None = None,
) -> list[dict[str, Any]]:
    transcript = "\n".join(
        f"{u['id']} [{u['role']}|{u['speaker_id']}]: {u['text_normalized']}" for u in utterances
    )
    system = "You are a careful clinical scribe. Return only valid JSON."
    user = f"""Write a concise SOAP note using only the transcript and extracted facts.
Every sentence must cite supporting utterance IDs. Do not add diagnoses, findings,
medicines, or advice not stated by the clinician. Keep absent symptoms negative.
Return JSON with arrays subjective, objective, assessment, plan. Each item has text
and evidence, an array of source utterance IDs.

FACTS:
{facts}

TRANSCRIPT:
{transcript}"""
    if feedback:
        user += "\n\nREQUIRED CORRECTIONS TO THE PREVIOUS DRAFT:\n- " + "\n- ".join(feedback)
    if soap_generator is None:
        payload = llm.complete_json(system, user)
    else:
        payload = soap_generator.generate_json(system, user)
    return validate_soap(payload, {u["id"] for u in utterances})


def verify_claims(
    llm: LocalLLM,
    claims: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    normalizer: Normalizer,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    by_id = {u["id"]: u for u in utterances}
    checked, warnings = [], []
    for claim in claims:
        evidence = "\n".join(
            f"{uid}: {by_id[uid]['text_normalized']}" for uid in claim["evidence"]
        )
        result = llm.complete_json(
            "You check whether a clinical note sentence is fully supported by cited transcript evidence. Return JSON only.",
            f"""Claim: {claim['text']}
Cited source:
{evidence}
Return {{"verdict":"supported|unsupported|contradicted","reason":"brief explanation"}}.""",
        )
        verdict = result.get("verdict")
        if verdict not in {"supported", "unsupported", "contradicted"}:
            raise ValueError(f"Verifier returned invalid verdict {verdict!r}")
        item = {**claim, "verdict": verdict, "reason": str(result.get("reason", ""))}
        checked.append(item)
        warnings.extend({
            **warning,
            "claim": claim["text"],
            "evidence": claim["evidence"],
        } for warning in normalizer.check_dose_details(claim["text"]))
    return checked, warnings


def render_soap(claims: list[dict[str, Any]], supported_only: bool = False) -> dict[str, list[dict[str, Any]]]:
    note = {section: [] for section in ("subjective", "objective", "assessment", "plan")}
    for claim in claims:
        if supported_only and claim.get("verdict") != "supported":
            continue
        note[claim["section"]].append({
            "text": claim["text"],
            "evidence": claim["evidence"],
            **({"verdict": claim["verdict"]} if "verdict" in claim else {}),
        })
    return note


def _fact_is_represented(fact: dict[str, Any], note_text: str) -> bool:
    concept = fact.get("entity_text") or fact["text"]
    concept_pattern = re.compile(
        rf"(?<!\w){re.escape(concept.strip())}(?!\w)", flags=re.IGNORECASE
    )
    match = concept_pattern.search(note_text)
    if not match:
        return False
    context = note_text[max(0, match.start() - 60):match.end() + 60].casefold()
    assertion = fact["assertion"]
    cues = {
        "absent": r"\b(no|not|never|denies|denied|without|negative|absent|none)\b",
        "uncertain": r"\b(possible|possibly|might|may|uncertain|unclear|questionable)\b",
        "historical": r"\b(history of|historical|previously|prior|previous|last|ago|resolved)\b",
    }
    cue = cues.get(assertion)
    if cue:
        return re.search(cue, context) is not None
    if assertion == "present":
        negative_or_uncertain = (
            r"\b(no|not|never|denies|denied|without|negative|absent|none|"
            r"possible|possibly|might|may|uncertain|unclear|history of|previously|prior)\b"
        )
        return re.search(negative_or_uncertain, context) is None
    return True


def find_omissions(
    facts: list[dict[str, Any]], claims: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    note_text = " ".join(claim["text"] for claim in claims)
    omitted = []
    for fact in facts:
        if fact.get("assertion") not in {"present", "absent", "uncertain", "historical", "past"}:
            continue
        normalized_fact = {
            **fact,
            "assertion": "historical" if fact["assertion"] == "past" else fact["assertion"],
        }
        if not _fact_is_represented(normalized_fact, note_text):
            omitted.append({
                "fact": fact["text"],
                "assertion": normalized_fact["assertion"],
                "entity_text": fact.get("entity_text", ""),
                "evidence": fact.get("evidence", []),
            })
    return omitted


def generate_verified_soap(
    llm: LocalLLM,
    utterances: list[dict[str, Any]],
    facts: list[dict[str, Any]],
    normalizer: Normalizer,
    soap_generator: Any | None = None,
    max_attempts: int = 2,
) -> dict[str, Any]:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    feedback: list[str] = []
    attempt_history: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    omissions: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    all_dose_warnings: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            claims = generate_claims(
                llm, utterances, facts, soap_generator, feedback or None
            )
        except ValueError as exc:
            feedback = [f"Return valid SOAP JSON and cite only real utterance IDs. Error: {exc}"]
            attempt_history.append({
                "attempt": attempt,
                "status": "invalid_draft",
                "feedback": feedback,
            })
            if attempt == max_attempts:
                raise ValueError(
                    f"SOAP draft remained invalid after {max_attempts} attempt(s): {exc}"
                ) from exc
            continue

        checked, warnings = verify_claims(llm, claims, utterances, normalizer)
        for warning in warnings:
            recorded = {**warning, "detected_on_attempt": attempt}
            if recorded not in all_dose_warnings:
                all_dose_warnings.append(recorded)
        supported = [claim for claim in checked if claim["verdict"] == "supported"]
        omissions = find_omissions(facts, supported)
        attempt_issues: list[str] = []
        unsupported = [claim for claim in checked if claim["verdict"] != "supported"]
        if unsupported:
            attempt_issues.extend(
                f"Remove or correct the {claim['section']} claim {claim['text']!r}; "
                f"its cited evidence {claim['evidence']} was {claim['verdict']}: {claim['reason']}"
                for claim in unsupported
            )
        if omissions:
            attempt_issues.extend(
                f"Include the source-supported fact {item['fact']!r} with assertion "
                f"{item['assertion']!r}, citing {item['evidence']}"
                for item in omissions
            )
        if warnings:
            attempt_issues.extend(
                f"Do not silently alter this source dosage. Omit or clearly flag the "
                f"claim for clinician review: {warning['message']}"
                for warning in warnings
            )
        attempt_history.append({
            "attempt": attempt,
            "status": "needs_revision" if attempt_issues else "passed",
            "unsupported_claims": unsupported,
            "omissions": omissions,
            "dosage_warnings": warnings,
            "feedback": attempt_issues,
        })
        if not attempt_issues:
            break
        feedback = attempt_issues

    supported = [claim for claim in checked if claim["verdict"] == "supported"]
    final_omissions = find_omissions(facts, supported)
    final_warnings = all_dose_warnings
    return {
        "claims": checked,
        "supported_claims": supported,
        "omissions": final_omissions,
        "warnings": final_warnings,
        "attempts": attempt_history,
        "review_required": bool(final_omissions or final_warnings or len(supported) != len(checked)),
    }
