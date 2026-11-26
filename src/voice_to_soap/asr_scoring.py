"""Versioned lexical ASR scoring; raw references and hypotheses are never edited.

numeric_indic_v2 is deliberately conservative: punctuation inside numeric forms
is retained rather than guessing decimal, thousands, date or dose semantics.
These lexical diagnostics do not establish clinical correctness.
"""
from collections import Counter
import re
import unicodedata

LEGACY = "punctuation_delete_v1"
CURRENT = "numeric_indic_v2"
PROTOCOLS = (LEGACY, CURRENT)
_APOSTROPHES = {"'", "’", "‘", "ʼ"}
_MINUSES = {"-", "−", "﹣", "－", "–", "—"}
_NUMERIC_PUNCTUATION = {"٫": ".", "٬": ",", "．": ".", "，": ",", "：": ":", "／": "/", "＋": "+", "％": "%"}
_NUMERIC = re.compile(r"(?<![\w.])[+-]?\s*(?:\d+(?:[.,:/-]\d+)*|\.\d+)(?:\s*%)?")
_NEGATIONS = frozenset(("no", "not", "without", "denies", "denied", "never", "negative",
                        "cannot", "can't", "isn't", "doesn't", "नहीं", "नही", "नाही", "नको"))


def normalize_asr_text(text: str, protocol: str = CURRENT) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown ASR scoring protocol: {protocol}")
    if protocol == LEGACY:
        return " ".join("".join(c for c in text.casefold()
                                if not unicodedata.category(c).startswith("P")).split())
    text = unicodedata.normalize("NFC", text.casefold())
    text = text.translate({**{ord(c): "-" for c in _MINUSES},
                           **{ord(c): replacement for c, replacement in _NUMERIC_PUNCTUATION.items()}})
    text = "".join(str(unicodedata.decimal(c)) if c.isdecimal() else c for c in text)
    output = []
    for index, c in enumerate(text):
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        if c in _MINUSES:
            # Keep signs and numeric ranges; separate ordinary hyphenated words.
            following = text[index + 1:].lstrip()[:2]
            numeric_follows = bool(following) and (following[0].isdigit() or (
                len(following) == 2 and following[0] == "." and following[1].isdigit()))
            output.append("-" if numeric_follows else " ")
        elif c in _APOSTROPHES:
            output.append("'" if before.isalpha() and after.isalpha() else " ")
        elif c in ".,/:" and before.isdigit() and after.isdigit():
            output.append(c)
        elif c == "." and after.isdigit() and (not before or before.isspace() or before in "+-"):
            output.append(c)
        elif c == "%" and text[:index].rstrip()[-1:].isdigit():
            output.append(c)
        elif unicodedata.category(c).startswith("P"):
            output.append(" ")
        elif c == "\ufeff":
            output.append(" ")
        else:
            # In particular, retain Devanagari vowel signs, virama and nukta.
            output.append(c)
    return " ".join("".join(output).split())


def numeric_inventory(text: str) -> Counter:
    normalized = normalize_asr_text(text, CURRENT)
    return Counter(re.sub(r"\s+", "", match.group()) for match in _NUMERIC.finditer(normalized))


def negation_inventory(text: str) -> Counter:
    return Counter(word for word in normalize_asr_text(text, CURRENT).split() if word in _NEGATIONS)


def score_records(rows, protocol: str = CURRENT) -> dict:
    from jiwer import process_characters, process_words

    if not rows:
        raise ValueError("ASR scoring requires at least one reference/hypothesis pair")
    references = [normalize_asr_text(r["reference_transcript"], protocol) for r in rows]
    hypotheses = [normalize_asr_text(r["prediction"], protocol) for r in rows]
    words = process_words(references, hypotheses)
    characters = process_characters(references, hypotheses)
    return {"examples": len(rows), "wer": float(words.wer), "cer": float(characters.cer),
            "scoring_protocol": protocol,
            "word_errors": {k: getattr(words, k) for k in ("substitutions", "deletions", "insertions")},
            "numeric_inventory_mismatch_examples": sum(numeric_inventory(r["reference_transcript"]) !=
                numeric_inventory(r["prediction"]) for r in rows),
            "negation_marker_mismatch_examples": sum(negation_inventory(r["reference_transcript"]) !=
                negation_inventory(r["prediction"]) for r in rows),
            "safety_flags_scope": "Lexical inventories only; no assertion, dose or clinical correctness validation"}
