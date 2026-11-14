from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import tarfile
from pathlib import Path
from typing import Any

SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM_PROMPT = (
    "You are a clinical note writer. Convert the supplied doctor-patient dialogue "
    "into a concise English SOAP note. Return exactly one JSON object with only these "
    "four keys and string values: "
    '{"subjective":"...","objective":"...","assessment":"...","plan":"..."}. '
    "Each value must be plain text, not a nested object or list. Use only information "
    "supported by the dialogue; do not invent findings, diagnoses, or treatment. "
    "Keep patient-reported information separate from examination and test results. "
    "Use an empty string when a section is not supported."
)
PLAN_CUE = re.compile(
    r"\b("
    r"advise[ds]?|recommend(?:ed|s|ing)?|instruct(?:ed|s|ions)?|"
    r"start(?:ed|s|ing)?|initiat(?:e|ed|es|ing)|continue(?:d|s|ing)?|"
    r"increas(?:e|ed|es|ing)|decreas(?:e|ed|es|ing)|discontinu(?:e|ed|es|ing)|"
    r"stop(?:ped|s|ping)?|prescrib(?:e|ed|es|ing)|order(?:ed|s|ing)?|"
    r"follow[- ]?up|return in|monitor|schedule|will begin|will take|"
    r"should take|take .* daily|patient education|additional testing"
    r")\b",
    re.IGNORECASE,
)


class InvalidGeneratedSoapError(ValueError):
    def __init__(self, message: str, raw_output: str):
        super().__init__(message)
        self.raw_output = raw_output


NOTE_HEADINGS = {
    "CHIEF COMPLAINT": "subjective",
    "HISTORY OF PRESENT ILLNESS": "subjective",
    "REVIEW OF SYSTEMS": "subjective",
    "REVIEW OF SYMPTOMS": "subjective",
    "MEDICAL HISTORY": "subjective",
    "PAST MEDICAL HISTORY": "subjective",
    "PAST HISTORY": "subjective",
    "SOCIAL HISTORY": "subjective",
    "FAMILY HISTORY": "subjective",
    "SURGICAL HISTORY": "subjective",
    "PAST SURGICAL HISTORY": "subjective",
    "CURRENT MEDICATIONS": "subjective",
    "MEDICATIONS": "subjective",
    "ALLERGIES": "subjective",
    "BIRTH HISTORY": "subjective",
    "SUBJECTIVE": "subjective",
    "PHYSICAL EXAM": "objective",
    "PHYSICAL EXAMINATION": "objective",
    "EXAM": "objective",
    "VITALS": "objective",
    "VITALS REVIEWED": "objective",
    "RESULTS": "objective",
    "PROCEDURE": "objective",
    "PROCEDURES": "objective",
    "LABS": "objective",
    "IMAGING": "objective",
    "EKG": "objective",
    "ECG": "objective",
    "OBJECTIVE": "objective",
    "ASSESSMENT": "assessment",
    "IMPRESSION": "assessment",
    "DIAGNOSIS": "assessment",
    "ASSESSMENT AND PLAN": "combined",
    "ASSESSMENT & PLAN": "combined",
    "PLAN": "plan",
    "INSTRUCTIONS": "plan",
    "ORDERS": "plan",
    "DISPOSITION": "plan",
}


def _normalized_heading(line: str) -> str:
    heading = re.sub(r"^\s*#{1,6}\s*", "", line.strip())
    heading = re.sub(r"^\s*[-*•]\s*", "", heading)
    heading = heading.strip().strip("*_` ")
    heading = re.sub(r"^\d+[.)]\s*", "", heading)
    return heading.strip().strip("*_` ").rstrip(":").strip().upper()


def split_note_into_soap(note: str) -> tuple[dict[str, str], list[str]]:
    if not note.strip():
        raise ValueError("ACI note is empty")

    sections: dict[str, list[str]] = {section: [] for section in SOAP_SECTIONS}
    warnings: list[str] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    preamble: list[str] = []

    def save_block() -> None:
        nonlocal current_lines, current_heading
        text = "\n".join(line.rstrip() for line in current_lines).strip()
        if not text or current_heading is None:
            current_lines = []
            return
        destination = NOTE_HEADINGS[current_heading]
        if destination == "combined":
            assessment_lines: list[str] = []
            plan_lines: list[str] = []
            combined_section = "assessment"
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                label = re.match(
                    r"^(?:[-*•]\s*)?\*{0,2}(Assessment|Plan)\s*:\*{0,2}\s*(.*)$",
                    line,
                    re.IGNORECASE,
                )
                if label:
                    combined_section = label.group(1).casefold()
                    line = label.group(2).strip()
                    if not line:
                        continue
                elif re.match(
                    r"^(?:[-*•]\s*)?\*{0,2}Problem\s*#?\d+\b",
                    line,
                    re.IGNORECASE,
                ):
                    combined_section = "assessment"
                elif re.match(r"^(?:[-*•]\s*)?\*{0,2}Plan\s*:?\*{0,2}\s*$", line, re.I):
                    combined_section = "plan"
                    continue
                elif re.match(
                    r"^(?:[-*•]\s*)?\*{0,2}Assessment\s*:?\*{0,2}\s*$",
                    line,
                    re.I,
                ):
                    combined_section = "assessment"
                    continue
                content = re.sub(r"^(?:[-*•]\s*)+", "", line).strip()
                sentences = re.split(
                    r"(?<=[.!?])\s+(?=[A-Z0-9])", content
                )
                for sentence in sentences:
                    if not sentence:
                        continue
                    if combined_section == "plan" or PLAN_CUE.search(sentence):
                        plan_lines.append(sentence)
                    else:
                        assessment_lines.append(sentence)
            if assessment_lines:
                sections["assessment"].append("\n".join(assessment_lines))
            if plan_lines:
                sections["plan"].append("\n".join(plan_lines))
            warnings.append(
                "ASSESSMENT AND PLAN was heuristically divided by action-language cues"
            )
        else:
            sections[destination].append(text)
        current_lines = []

    for line in note.splitlines():
        heading = _normalized_heading(line)
        if heading in NOTE_HEADINGS:
            if current_heading is not None:
                save_block()
            elif preamble:
                preamble_text = "\n".join(preamble).strip()
                if preamble_text:
                    sections["subjective"].append(preamble_text)
                    warnings.append(
                        "Text before the first recognized note heading was assigned to subjective"
                    )
            current_heading = heading
            continue
        if current_heading is None:
            preamble.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        save_block()
    elif not any(line.strip() for line in preamble):
        raise ValueError("ACI note contains no usable content")

    mapped = {
        section: "\n\n".join(value for value in sections[section] if value.strip()).strip()
        for section in SOAP_SECTIONS
    }
    if not any(mapped.values()):
        raise ValueError("ACI note has no content under recognized SOAP headings")
    for section in SOAP_SECTIONS:
        if not mapped[section]:
            warnings.append(f"No source note content mapped to {section}")
    return mapped, warnings


def _read_split(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"encounter_id", "dialogue", "note"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            encounter_id = (row.get("encounter_id") or "").strip()
            dialogue = (row.get("dialogue") or "").strip()
            note = (row.get("note") or "").strip()
            if not encounter_id or not dialogue or not note:
                raise ValueError(
                    f"{path}:{line_number} needs encounter_id, dialogue, and note"
                )
            rows.append({
                "encounter_id": encounter_id,
                "dialogue": dialogue,
                "note": note,
            })
    if not rows:
        raise ValueError(f"No ACI examples found in {path}")
    return rows


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_aci_soap_data(
    dataset_dir: str | Path = "data/raw/aci-bench/aci-bench-corpus/challenge_data",
    output_dir: str | Path = "data/processed/aci-bench-soap",
) -> dict[str, Any]:
    source = Path(dataset_dir)
    destination = Path(output_dir)
    split_paths = {
        "train": source / "train.csv",
        "validation": source / "valid.csv",
        "test": [
            source / "clinicalnlp_taskB_test1.csv",
            source / "clinicalnlp_taskC_test2.csv",
            source / "clef_taskC_test3.csv",
        ],
    }
    rows_by_split: dict[str, list[tuple[dict[str, str], str]]] = {}
    seen_ids: set[str] = set()
    for split, paths in split_paths.items():
        selected_paths = paths if isinstance(paths, list) else [paths]
        split_rows = []
        for path in selected_paths:
            for row in _read_split(path):
                encounter_id = row["encounter_id"]
                if encounter_id in seen_ids:
                    raise ValueError(
                        f"Encounter {encounter_id!r} appears in more than one split"
                    )
                seen_ids.add(encounter_id)
                split_rows.append((row, path.name))
        rows_by_split[split] = split_rows

    summary: dict[str, Any] = {"splits": {}, "mapping_method": "heading_rules_v1"}
    for split, rows in rows_by_split.items():
        records = []
        warning_count = 0
        for row, source_file in rows:
            soap, warnings = split_note_into_soap(row["note"])
            warning_count += bool(warnings)
            records.append({
                "encounter_id": row["encounter_id"],
                "source_split": split,
                "source_file": source_file,
                "dialogue": row["dialogue"],
                "soap": soap,
                "mapping_warnings": warnings,
            })
        _write_jsonl(destination / f"{split}.jsonl", records)
        summary["splits"][split] = {
            "examples": len(records),
            "examples_with_mapping_warnings": warning_count,
        }
    summary["important_limitations"] = [
        "SOAP targets are heuristic mappings from reference-note headings, not clinician-reviewed labels.",
        "Combined assessment-and-plan text is divided heuristically using action-language cues.",
        "ACI-Bench dialogues are English; these files do not train or evaluate Hindi-input-to-English-SOAP.",
    ]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "preparation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def load_aci_soap_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            if not isinstance(item.get("dialogue"), str) or not item["dialogue"].strip():
                raise ValueError(f"{path}:{line_number} needs a non-empty dialogue")
            soap = item.get("soap")
            if not isinstance(soap, dict) or any(
                not isinstance(soap.get(section), str) for section in SOAP_SECTIONS
            ):
                raise ValueError(
                    f"{path}:{line_number} needs string fields for all SOAP sections"
                )
            examples.append(item)
    if not examples:
        raise ValueError(f"No SOAP examples found in {path}")
    return examples


def _betrac_transcript_to_dialogue(transcript: str, member_name: str) -> str:
    if not transcript.strip():
        raise ValueError(f"BeTraC transcript is empty: {member_name}")
    dialogue = re.sub(
        r"(?m)^\s*(DOCTOR|PATIENT)\s*:\s*",
        lambda match: f"[{match.group(1).casefold()}] ",
        transcript.strip(),
        flags=re.IGNORECASE,
    )
    if not re.search(r"(?m)^\[(?:doctor|patient)\]\s+", dialogue):
        raise ValueError(f"BeTraC transcript has no recognized speaker turns: {member_name}")
    return dialogue


def prepare_combined_soap_data(
    aci_dir: str | Path,
    betrac_archive: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Combine ACI train and BeTraC dev pairs while preserving ACI evaluation splits."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    aci_root = Path(aci_dir)
    records: dict[str, list[dict[str, Any]]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        source_path = aci_root / f"{split}.jsonl"
        examples = load_aci_soap_examples(source_path)
        converted = []
        for item in examples:
            converted.append({
                "encounter_id": f"aci-{item['encounter_id']}",
                "source_dataset": "ACI-Bench",
                "source_split": item.get("source_split", split),
                "assigned_split": split,
                "target_provenance": "reference_note_heuristically_mapped_to_soap",
                "dialogue": item["dialogue"],
                "soap": item["soap"],
                "mapping_warnings": item.get("mapping_warnings", []),
            })
        records[split] = converted
        source_counts[f"ACI-Bench:{split}"] = {"examples": len(converted)}

    archive_path = Path(betrac_archive)
    groups: dict[str, dict[str, tarfile.TarInfo]] = {}
    suffixes = (".transcript.txt", ".soap.txt", ".json", ".opus")
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or len(path.parts) != 1 or ".." in path.parts:
                raise ValueError(f"Unsafe BeTraC archive member name: {member.name}")
            if not member.isfile():
                raise ValueError(f"BeTraC archive contains a non-file member: {member.name}")
            suffix = next((value for value in suffixes if member.name.endswith(value)), None)
            if suffix is None:
                continue
            sample_id = member.name[:-len(suffix)]
            if not re.fullmatch(r"[A-Za-z0-9_-]+", sample_id):
                raise ValueError(f"Unsafe BeTraC sample identifier: {sample_id!r}")
            sample = groups.setdefault(sample_id, {})
            if suffix in sample:
                raise ValueError(f"Duplicate BeTraC member for {sample_id}{suffix}")
            sample[suffix] = member

        if not groups:
            raise ValueError(f"No BeTraC samples found in {archive_path}")
        betrac_records = []
        mapping_warnings = 0
        for sample_id, members in sorted(groups.items()):
            if set(members) != set(suffixes):
                raise ValueError(f"Incomplete BeTraC sample {sample_id}: {sorted(members)}")
            transcript_stream = archive.extractfile(members[".transcript.txt"])
            note_stream = archive.extractfile(members[".soap.txt"])
            metadata_stream = archive.extractfile(members[".json"])
            if transcript_stream is None or note_stream is None or metadata_stream is None:
                raise ValueError(f"Could not read paired BeTraC files for {sample_id}")
            transcript = transcript_stream.read().decode("utf-8")
            note = note_stream.read().decode("utf-8")
            metadata = json.loads(metadata_stream.read().decode("utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid BeTraC metadata for {sample_id}")
            dialogue = _betrac_transcript_to_dialogue(transcript, sample_id)
            soap, warnings = split_note_into_soap(note)
            mapping_warnings += bool(warnings)
            betrac_records.append({
                "encounter_id": f"betrac-{sample_id}",
                "source_dataset": "BeTraC/betrac-2026",
                "source_split": "dev",
                "assigned_split": "train",
                "target_provenance": "synthetic_dataset_soap_note_heuristically_mapped",
                "dialogue": dialogue,
                "soap": soap,
                "mapping_warnings": warnings,
                "source_metadata_id": str(metadata.get("id", sample_id)),
            })

    records["train"].extend(betrac_records)
    source_counts["BeTraC:dev_assigned_to_train"] = {
        "examples": len(betrac_records),
        "examples_with_mapping_warnings": mapping_warnings,
    }

    all_ids: dict[str, set[str]] = {
        split: {item["encounter_id"] for item in examples}
        for split, examples in records.items()
    }
    if any(len(all_ids[split]) != len(records[split]) for split in records):
        raise ValueError("Duplicate encounter IDs within a combined split")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if all_ids[left] & all_ids[right]:
            raise ValueError(f"Encounter leakage between {left} and {right}")

    destination.mkdir(parents=True)
    for split, examples in records.items():
        _write_jsonl(destination / f"{split}.jsonl", examples)
    summary = {
        "dataset_name": "combined_aci_bench_betrac_soap_experiment",
        "split_counts": {
            split: {
                "examples": len(examples),
                "source_counts": {
                    dataset: counts
                    for dataset, counts in source_counts.items()
                    if (split == "train" and dataset.startswith(("ACI-Bench:train", "BeTraC:")))
                    or (split != "train" and dataset == f"ACI-Bench:{split}")
                },
            }
            for split, examples in records.items()
        },
        "encounter_leakage_check": "passed",
        "be_trac_archive_samples": len(groups),
        "aci_test_ids_preserved": sorted(all_ids["test"]),
        "limitations": [
            "BeTraC source is synthetic and its SOAP notes are heuristically mapped into four fields.",
            "BeTraC dev examples are assigned to training; they cannot also serve as an independent test set.",
            "ACI-Bench SOAP fields are heuristic mappings from reference notes, not clinician-reviewed SOAP labels.",
            "Eka has ASR references but no SOAP targets and is excluded from SOAP training.",
            "Sully local audio overlaps an ACI-Bench encounter and is excluded to avoid leakage; it has no independent paired example here.",
            "No additional synthetic examples were fabricated for this dataset preparation.",
        ],
    }
    (destination / "preparation_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _prompt_text(
    tokenizer: Any,
    dialogue: str,
    facts: list[dict[str, Any]] | None = None,
) -> str:
    system_prompt = SYSTEM_PROMPT
    user_content = (
        "Write the SOAP note from this dialogue. Return all four fields as "
        "JSON string values."
    )
    if facts is not None:
        system_prompt += (
            " You will also receive candidate facts extracted by a transparent "
            "rule-based NLP stage. Use them as a checklist, but rely on the "
            "dialogue for meaning and do not invent unsupported details."
        )
        user_content += (
            "\n\nRULE-BASED CLINICAL FACTS (may be incomplete):\n"
            + json.dumps(facts, ensure_ascii=False)
        )
    user_content += "\n\nDIALOGUE:\n" + dialogue
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    ) + "\nASSISTANT:\n"


def _encode_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_seq_length: int,
    approach: str = "direct",
) -> dict[str, list[int]]:
    if approach not in {"direct", "hybrid"}:
        raise ValueError(f"Unknown SOAP generation approach {approach!r}")
    facts = None
    if approach == "hybrid":
        from .clinical_facts import extract_clinical_facts

        facts = extract_clinical_facts(example["dialogue"])
    prompt = _prompt_text(tokenizer, example["dialogue"], facts)
    target = json.dumps(example["soap"], ensure_ascii=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(target_ids) > max_seq_length:
        raise ValueError(
            f"Encounter {example.get('encounter_id', '<unknown>')} requires "
            f"{len(prompt_ids) + len(target_ids)} tokens, exceeding max_seq_length="
            f"{max_seq_length}. No automatic truncation is performed."
        )
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


def train_aci_soap_lora(
    train_path: str | Path,
    validation_path: str | Path,
    base_model: str,
    output_dir: str | Path,
    epochs: float = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_seq_length: int = 4096,
    lora_rank: int = 8,
    skip_long_examples: bool = False,
    approach: str = "direct",
    seed: int = 42,
    model_revision: str | None = None,
    _examples_override: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not base_model.strip():
        raise ValueError("base_model cannot be empty")
    if epochs <= 0 or batch_size < 1 or max_seq_length < 2 or lora_rank < 1:
        raise ValueError("Training settings must be positive")
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ACI SOAP LoRA training requires torch, transformers, accelerate, and peft"
        ) from exc

    if approach not in {"direct", "hybrid"}:
        raise ValueError(f"Unknown SOAP generation approach {approach!r}")
    train_examples, validation_examples = (
        _examples_override
        if _examples_override is not None
        else (
            load_aci_soap_examples(train_path),
            load_aci_soap_examples(validation_path),
        )
    )
    model_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **model_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    class EncodedDataset(torch.utils.data.Dataset):
        def __init__(self, rows: list[dict[str, list[int]]]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.rows[index]

    def encode(
        examples: list[dict[str, Any]],
        *,
        skip_long: bool = False,
    ) -> tuple[EncodedDataset, list[str]]:
        encoded = []
        skipped = []
        for example in examples:
            try:
                encoded.append(
                    _encode_example(tokenizer, example, max_seq_length, approach)
                )
            except ValueError as exc:
                if not skip_long or "No automatic truncation" not in str(exc):
                    raise
                skipped.append(str(example.get("encounter_id", "<unknown>")))
        if not encoded:
            raise ValueError("No examples fit within max_seq_length")
        return EncodedDataset(encoded), skipped

    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    set_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16 if cuda_available else torch.float32,
        low_cpu_mem_usage=True,
        **model_kwargs,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            lora_dropout=0.05,
            target_modules=target_modules,
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        num_train_epochs=epochs,
        seed=seed,
        data_seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=cuda_available,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,
        report_to="none",
        remove_unused_columns=False,
    )
    encoded_train, skipped_train = encode(train_examples, skip_long=skip_long_examples)
    encoded_validation, skipped_validation = encode(
        validation_examples, skip_long=skip_long_examples
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_train,
        eval_dataset=encoded_validation,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": base_model,
        "adapter_type": "LoRA",
        "task": f"aci_bench_{approach}_dialogue_to_heuristic_soap",
        "approach": approach,
        "train_examples": len(train_examples),
        "train_encounter_ids": [str(item.get("encounter_id", "")) for item in train_examples],
        "trained_examples": len(encoded_train),
        "skipped_long_train_examples": skipped_train,
        "validation_examples": len(validation_examples),
        "validation_encounter_ids": [
            str(item.get("encounter_id", "")) for item in validation_examples
        ],
        "evaluated_validation_examples": len(encoded_validation),
        "skipped_long_validation_examples": skipped_validation,
        "epochs": epochs,
        "seed": seed,
        "model_revision": model_revision,
        "max_seq_length": max_seq_length,
        "lora_rank": lora_rank,
        "training_device": (
            "cuda" if cuda_available else "mps" if mps_available else "cpu"
        ),
        "lora_alpha": lora_rank * 2,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "reference_target_note": (
            "SOAP targets were produced by heading_rules_v1 and are not clinician-reviewed."
        ),
        "input_method": (
            "Transcript only."
            if approach == "direct"
            else "Transcript plus lexicon/pattern clinical facts with rule-based assertion, section, and evidence labels."
        ),
    }
    from importlib.metadata import PackageNotFoundError, version
    import platform

    try:
        datasets_version = version("datasets")
    except PackageNotFoundError:
        datasets_version = None
    metadata["software_versions"] = {
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "datasets": datasets_version,
    }
    Path(output_dir, "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {"metadata": metadata, "train_metrics": trainer.state.log_history}


def _paired_fit_examples(
    examples: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    eligible = []
    skipped = []
    for example in examples:
        fits_both = True
        for approach in ("direct", "hybrid"):
            try:
                _encode_example(tokenizer, example, max_seq_length, approach)
            except ValueError as exc:
                if "No automatic truncation" not in str(exc):
                    raise
                fits_both = False
        if fits_both:
            eligible.append(example)
        else:
            skipped.append(str(example.get("encounter_id", "<unknown>")))
    return eligible, skipped


def run_aci_soap_comparison(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    base_model: str,
    output_dir: str | Path,
    *,
    model_revision: str | None = None,
    epochs: float = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_seq_length: int = 4096,
    lora_rank: int = 8,
    seed: int = 42,
    device: str = "auto",
    max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Train and evaluate direct and fact-preprocessed LoRA runs on matched examples."""
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Matched SOAP comparison requires torch and transformers"
        ) from exc

    split_examples = {
        "train": load_aci_soap_examples(train_path),
        "validation": load_aci_soap_examples(validation_path),
        "test": load_aci_soap_examples(test_path),
    }
    split_ids = {
        split: [str(item.get("encounter_id", "")) for item in examples]
        for split, examples in split_examples.items()
    }
    for split, encounter_ids in split_ids.items():
        if len(encounter_ids) != len(set(encounter_ids)):
            raise ValueError(f"Duplicate encounter IDs within {split} split")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = set(split_ids[left]) & set(split_ids[right])
        if overlap:
            raise ValueError(f"Encounter leakage between {left} and {right}: {sorted(overlap)}")

    tokenizer_kwargs = {"revision": model_revision} if model_revision else {}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    fit: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, list[str]] = {}
    for split, examples in split_examples.items():
        fit[split], skipped[split] = _paired_fit_examples(
            examples, tokenizer, max_seq_length
        )
        if not fit[split]:
            raise ValueError(f"No {split} examples fit both approaches at this sequence length")
    del tokenizer

    train_ids = [str(item.get("encounter_id", "")) for item in fit["train"]]
    validation_ids = [str(item.get("encounter_id", "")) for item in fit["validation"]]
    test_ids = [str(item.get("encounter_id", "")) for item in fit["test"]]
    results_root = Path(output_dir)
    if results_root.exists() and any(results_root.iterdir()):
        raise FileExistsError(
            f"Comparison output directory is not empty: {results_root}; choose a new path"
        )
    adapters_root = results_root / "adapters"
    adapters_root.mkdir(parents=True, exist_ok=True)
    run_config = {
        "study": "matched_direct_vs_rule_based_hybrid_aci_soap",
        "status": "training",
        "base_model": base_model,
        "model_revision": model_revision,
        "seed": seed,
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "selected_training_device": (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            ),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank * 2,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
        "matched_encounter_ids": {
            "train": train_ids,
            "validation": validation_ids,
            "test": test_ids,
        },
        "skipped_for_both": skipped,
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    training: dict[str, Any] = {}
    for approach in ("direct", "hybrid"):
        training[approach] = train_aci_soap_lora(
            train_path,
            validation_path,
            base_model,
            adapters_root / approach,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_seq_length=max_seq_length,
            lora_rank=lora_rank,
            approach=approach,
            seed=seed,
            model_revision=model_revision,
            _examples_override=(fit["train"], fit["validation"]),
        )
        if approach == "direct":
            import gc

            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    evaluations = {}
    for approach in ("direct", "hybrid"):
        evaluations[approach] = evaluate_aci_soap_lora(
            test_path,
            base_model,
            adapters_root / approach,
            results_root / approach,
            device=device,
            max_new_tokens=max_new_tokens,
            approach=approach,
            model_revision=model_revision,
            _examples_override=fit["test"],
        )
    if evaluations["direct"]["encounter_ids"] != evaluations["hybrid"]["encounter_ids"]:
        raise RuntimeError("Comparison evaluations did not use identical test encounter IDs")

    comparison = {
        "study": "matched_direct_vs_rule_based_hybrid_aci_soap",
        "base_model": base_model,
        "model_revision": model_revision,
        "seed": seed,
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_seq_length": max_seq_length,
            "lora_rank": lora_rank,
        },
        "training_runs": training,
        "split_counts": {
            split: {
                "available": len(split_examples[split]),
                "used_by_both": len(fit[split]),
                "skipped_for_both": skipped[split],
            }
            for split in split_examples
        },
        "matched_encounter_ids": {
            "train": train_ids,
            "validation": validation_ids,
            "test": test_ids,
        },
        "approaches": evaluations,
        "interpretation_limits": [
            "ACI-Bench SOAP targets are heuristic mappings, not clinician-reviewed labels.",
            "ROUGE-L measures overlap with those heuristic targets, not clinical correctness.",
            "The hybrid preprocessing is a lexicon/pattern baseline and may miss or misclassify facts.",
            "Invalid generations are excluded from ROUGE means and are reported separately.",
        ],
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    run_config["status"] = "completed"
    (results_root / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison


class AciSoapLoRAGenerator:
    def __init__(
        self,
        base_model: str,
        adapter_dir: str | Path | None = None,
        device: str = "auto",
        model_revision: str | None = None,
    ):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("SOAP inference requires torch, transformers, and peft") from exc
        if adapter_dir is not None:
            metadata_path = Path(adapter_dir) / "adapter_metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"Adapter metadata not found: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("base_model") != base_model:
                raise ValueError(
                    f"Adapter requires base model {metadata.get('base_model')!r}, "
                    f"not {base_model!r}"
                )
            if metadata.get("model_revision") != model_revision:
                raise ValueError(
                    "Configured model revision does not match the adapter's recorded "
                    f"revision {metadata.get('model_revision')!r}"
                )
        if device == "auto":
            device = (
                "mps" if torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(device)
        model_kwargs = {"revision": model_revision} if model_revision else {}
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, **model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
        self.model = (
            PeftModel.from_pretrained(model, str(adapter_dir))
            if adapter_dir is not None
            else model
        ).to(self.device)
        self.model.eval()

    def generate(
        self,
        dialogue: str,
        max_new_tokens: int = 1024,
        facts: list[dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        import torch

        encoded = self.tokenizer(
            _prompt_text(self.tokenizer, dialogue, facts), return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.inference_mode():
            result = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(
            result[0, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        decoded = re.sub(r"^```(?:json)?\s*|\s*```$", "", decoded, flags=re.IGNORECASE)
        try:
            soap = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise InvalidGeneratedSoapError(
                "Model returned invalid JSON", decoded
            ) from exc
        if not isinstance(soap, dict) or any(
            not isinstance(soap.get(section), str) for section in SOAP_SECTIONS
        ):
            raise InvalidGeneratedSoapError(
                "Generated note must contain string values for all SOAP fields",
                decoded,
            )
        return {section: soap[section].strip() for section in SOAP_SECTIONS}


def evaluate_aci_soap_lora(
    test_path: str | Path,
    base_model: str,
    adapter_dir: str | Path | None,
    output_dir: str | Path,
    device: str = "auto",
    limit: int | None = None,
    max_new_tokens: int = 1024,
    approach: str = "direct",
    model_revision: str | None = None,
    _examples_override: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Install rouge-score to evaluate SOAP note text") from exc
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")

    if approach not in {"direct", "hybrid"}:
        raise ValueError(f"Unknown SOAP generation approach {approach!r}")
    examples = (
        _examples_override
        if _examples_override is not None
        else load_aci_soap_examples(test_path)
    )
    if limit is not None:
        examples = examples[:limit]
    generator = AciSoapLoRAGenerator(
        base_model, adapter_dir, device=device, model_revision=model_revision
    )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores: dict[str, list[float]] = {section: [] for section in SOAP_SECTIONS}
    predictions = []
    for example in examples:
        record = {
            "encounter_id": example["encounter_id"],
            "source_split": example.get("source_split", "test"),
            "dialogue": example["dialogue"],
            "reference_soap": example["soap"],
            "mapping_warnings": example.get("mapping_warnings", []),
            "approach": approach,
        }
        try:
            if approach == "hybrid":
                from .clinical_facts import extract_clinical_facts

                facts = extract_clinical_facts(example["dialogue"])
                record["input_facts"] = facts
                generated = generator.generate(
                    example["dialogue"],
                    max_new_tokens=max_new_tokens,
                    facts=facts,
                )
            else:
                generated = generator.generate(
                    example["dialogue"], max_new_tokens=max_new_tokens
                )
        except InvalidGeneratedSoapError as exc:
            record["prediction_soap"] = None
            record["generation_error"] = str(exc)
            record["raw_model_output"] = exc.raw_output
        else:
            record["prediction_soap"] = generated
            for section in SOAP_SECTIONS:
                scores[section].append(
                    scorer.score(
                        example["soap"][section], generated[section]
                    )["rougeL"].fmeasure
                )
        predictions.append(record)

    section_metrics = {
        section: {
            "examples": len(values),
            "rougeL_fmeasure": sum(values) / len(values) if values else None,
        }
        for section, values in scores.items()
    }
    all_scores = [value for values in scores.values() for value in values]
    valid_predictions = sum(item["prediction_soap"] is not None for item in predictions)
    metrics = {
        "base_model": base_model,
        "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
        "model_stage": "lora_adapter" if adapter_dir is not None else "base_model",
        "split": "test",
        "examples": len(examples),
        "encounter_ids": [str(item.get("encounter_id", "")) for item in examples],
        "approach": approach,
        "model_revision": model_revision,
        "valid_predictions": valid_predictions,
        "invalid_predictions": len(examples) - valid_predictions,
        "device": str(generator.device),
        "metric": "mean section-level ROUGE-L F-measure",
        "by_section": section_metrics,
        "overall_mean_section_rougeL_fmeasure": (
            sum(all_scores) / len(all_scores) if all_scores else None
        ),
        "limitations": [
            "Scores measure overlap with automatically mapped reference notes, not clinical correctness.",
            "ACI-Bench test dialogues and notes are English.",
        ],
    }
    if valid_predictions == 0:
        metrics["limitations"].append(
            "No valid SOAP JSON was generated; ROUGE-L scores are null."
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / "predictions.jsonl", predictions)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="ACI-Bench SOAP preparation and LoRA pilot")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="map ACI note headings into SOAP")
    prepare_parser.add_argument(
        "--dataset-dir",
        default="data/raw/aci-bench/aci-bench-corpus/challenge_data",
    )
    prepare_parser.add_argument("--output-dir", default="data/processed/aci-bench-soap")
    combined_parser = commands.add_parser(
        "prepare-combined",
        help="combine ACI train and BeTraC dev pairs while preserving ACI validation/test",
    )
    combined_parser.add_argument(
        "--aci-dir", default="data/processed/aci-bench-soap"
    )
    combined_parser.add_argument(
        "--betrac-archive", default="data/raw/betrac/data/dev-00000.tar"
    )
    combined_parser.add_argument(
        "--output-dir", default="data/processed/aci-betrac-soap-v1"
    )

    train_parser = commands.add_parser("train", help="LoRA fine-tune a small SOAP model")
    train_parser.add_argument("--train", default="data/processed/aci-bench-soap/train.jsonl")
    train_parser.add_argument(
        "--validation", default="data/processed/aci-bench-soap/validation.jsonl"
    )
    train_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    train_parser.add_argument("--model-revision", default=None)
    train_parser.add_argument("--output-dir", default="data/models/aci-qwen25-05b-soap-lora")
    train_parser.add_argument("--epochs", type=float, default=1)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--max-seq-length", type=int, default=4096)
    train_parser.add_argument("--lora-rank", type=int, default=8)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument(
        "--skip-long-examples",
        action="store_true",
        help="skip overlength train/validation encounters without truncating them",
    )

    eval_parser = commands.add_parser("evaluate", help="evaluate LoRA adapter on held-out ACI tests")
    eval_parser.add_argument("--test", default="data/processed/aci-bench-soap/test.jsonl")
    eval_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    eval_parser.add_argument("--adapter-dir", default=None)
    eval_parser.add_argument("--output-dir", default="data/outputs/aci-qwen25-05b-soap")
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--limit", type=int, default=None)
    eval_parser.add_argument("--max-new-tokens", type=int, default=1024)
    compare_parser = commands.add_parser(
        "compare",
        help="train and evaluate matched direct and rule-based hybrid Qwen LoRA approaches",
    )
    compare_parser.add_argument("--train", default="data/processed/aci-bench-soap/train.jsonl")
    compare_parser.add_argument(
        "--validation", default="data/processed/aci-bench-soap/validation.jsonl"
    )
    compare_parser.add_argument("--test", default="data/processed/aci-bench-soap/test.jsonl")
    compare_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    compare_parser.add_argument("--model-revision", default=None)
    compare_parser.add_argument("--output-dir", default="data/outputs/aci-soap-comparison")
    compare_parser.add_argument("--epochs", type=float, default=1)
    compare_parser.add_argument("--batch-size", type=int, default=1)
    compare_parser.add_argument("--learning-rate", type=float, default=1e-4)
    compare_parser.add_argument("--max-seq-length", type=int, default=4096)
    compare_parser.add_argument("--lora-rank", type=int, default=8)
    compare_parser.add_argument("--seed", type=int, default=42)
    compare_parser.add_argument("--device", default="auto")
    compare_parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "prepare":
        print(json.dumps(
            prepare_aci_soap_data(args.dataset_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.command == "prepare-combined":
        print(json.dumps(
            prepare_combined_soap_data(
                args.aci_dir,
                args.betrac_archive,
                args.output_dir,
            ),
            ensure_ascii=False,
            indent=2,
        ))
    elif args.command == "train":
        result = train_aci_soap_lora(
            args.train,
            args.validation,
            args.base_model,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_seq_length=args.max_seq_length,
            lora_rank=args.lora_rank,
            skip_long_examples=args.skip_long_examples,
            seed=args.seed,
            model_revision=args.model_revision,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "evaluate":
        print(json.dumps(
            evaluate_aci_soap_lora(
                args.test,
                args.base_model,
                args.adapter_dir,
                args.output_dir,
                device=args.device,
                limit=args.limit,
                max_new_tokens=args.max_new_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        ))
    else:
        print(json.dumps(
            run_aci_soap_comparison(
                args.train,
                args.validation,
                args.test,
                args.base_model,
                args.output_dir,
                model_revision=args.model_revision,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_seq_length,
                lora_rank=args.lora_rank,
                seed=args.seed,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            ),
            ensure_ascii=False,
            indent=2,
        ))


if __name__ == "__main__":
    main()
