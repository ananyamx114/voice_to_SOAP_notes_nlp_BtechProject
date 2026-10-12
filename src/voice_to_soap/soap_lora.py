from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schemas import validate_soap

SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")
SOAP_SYSTEM_PROMPT = (
    "You are a careful clinical scribe. Use only the supplied transcript and facts. "
    "Return one valid JSON object and no surrounding commentary. Every SOAP claim "
    "must include one or more supporting utterance IDs from the transcript. Do not "
    "invent diagnoses, findings, medicines, or advice. Preserve negation, uncertainty, "
    "and historical status."
)


def load_soap_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(example, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            utterances = example.get("utterances")
            facts = example.get("facts", [])
            soap = example.get("soap")
            if not isinstance(utterances, list) or not utterances:
                raise ValueError(f"{path}:{line_number} needs a non-empty utterances list")
            if not isinstance(facts, list):
                raise ValueError(f"{path}:{line_number} facts must be a list")
            if not isinstance(soap, dict):
                raise ValueError(f"{path}:{line_number} needs a soap object")
            ids = set()
            for utterance in utterances:
                if not isinstance(utterance, dict):
                    raise ValueError(f"{path}:{line_number} utterances must be objects")
                utterance_id = utterance.get("id")
                text = utterance.get("text_normalized", utterance.get("text"))
                if not isinstance(utterance_id, str) or not utterance_id:
                    raise ValueError(f"{path}:{line_number} has an utterance without an ID")
                if utterance_id in ids:
                    raise ValueError(f"{path}:{line_number} has duplicate ID {utterance_id!r}")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(
                        f"{path}:{line_number} utterance {utterance_id!r} has no text"
                    )
                ids.add(utterance_id)
            validate_soap(soap, ids)
            examples.append({
                "utterances": utterances,
                "facts": facts,
                "soap": {
                    section: [
                        {"text": claim["text"], "evidence": claim["evidence"]}
                        for claim in soap.get(section, [])
                    ]
                    for section in SOAP_SECTIONS
                },
            })
    if not examples:
        raise ValueError(f"No SOAP training examples found in {path}")
    return examples


def build_soap_prompt(utterances: list[dict[str, Any]], facts: list[dict[str, Any]]) -> str:
    transcript_lines = []
    for utterance in utterances:
        text = utterance.get("text_normalized", utterance.get("text", ""))
        role = utterance.get("role", "unknown")
        speaker_id = utterance.get("speaker_id", "unknown")
        transcript_lines.append(f"{utterance['id']} [{role}|{speaker_id}]: {text}")
    return (
        "Create a SOAP note. Return all four arrays: subjective, objective, "
        "assessment, and plan. Each item must contain text and evidence, where "
        "evidence is a list of source utterance IDs.\n\n"
        f"FACTS:\n{json.dumps(facts, ensure_ascii=False)}\n\n"
        f"TRANSCRIPT:\n" + "\n".join(transcript_lines)
    )


def _chat_text(tokenizer: Any, messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = "\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
    return rendered + ("\nASSISTANT:\n" if add_generation_prompt else "")


def _tokenize_training_example(
    tokenizer: Any,
    example: dict[str, Any],
    max_seq_length: int,
) -> dict[str, list[int]]:
    user_prompt = build_soap_prompt(example["utterances"], example["facts"])
    prompt_text = _chat_text(
        tokenizer,
        [
            {"role": "system", "content": SOAP_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        add_generation_prompt=True,
    )
    target_text = json.dumps(example["soap"], ensure_ascii=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        target_ids = [*target_ids, eos_token_id]
    if not target_ids:
        raise ValueError("SOAP target tokenization produced an empty sequence")
    if len(prompt_ids) + len(target_ids) > max_seq_length:
        raise ValueError(
            f"Training example is {len(prompt_ids) + len(target_ids)} tokens, exceeding "
            f"max_seq_length={max_seq_length}; shorten the example or raise the limit. "
            "Automatic truncation is disabled so cited transcript evidence is not lost."
        )
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": ([-100] * len(prompt_ids)) + target_ids,
    }


def train_soap_lora(
    train_path: str | Path,
    validation_path: str | Path,
    base_model: str,
    output_dir: str | Path,
    epochs: float = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> None:
    if not base_model.strip():
        raise ValueError("Provide an open-weight base model checkpoint with --base-model")
    if max_seq_length < 2 or batch_size < 1 or epochs <= 0:
        raise ValueError("Training sequence length, batch size, and epochs must be positive")

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "LoRA training requires torch, datasets, transformers, and peft; "
            "install the project's training dependencies first"
        ) from exc

    train_examples = load_soap_examples(train_path)
    validation_examples = load_soap_examples(validation_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token to use as the padding token")
        tokenizer.pad_token = tokenizer.eos_token

    def encode(examples: list[dict[str, Any]]) -> Dataset:
        return Dataset.from_list([
            _tokenize_training_example(tokenizer, example, max_seq_length)
            for example in examples
        ])

    cuda_available = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if cuda_available else torch.float32,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    adapter_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=cuda_available,
        gradient_checkpointing=True,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encode(train_examples),
        eval_dataset=encode(validation_examples),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": base_model,
        "adapter_type": "LoRA",
        "task": "evidence_cited_soap_generation",
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    Path(output_dir, "adapter_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


class LoRASOAPGenerator:
    def __init__(self, base_model: str, adapter_path: str, max_new_tokens: int = 512):
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LoRA SOAP inference requires torch, transformers, and peft"
            ) from exc
        if not base_model or not adapter_path:
            raise ValueError("Both the LoRA base_model and adapter_path must be configured")
        metadata_path = Path(adapter_path) / "adapter_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"LoRA adapter metadata not found: {metadata_path}; use an adapter "
                "created by the project's training script"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("base_model") != base_model:
            raise ValueError(
                "Configured SOAP base_model does not match the adapter's recorded "
                f"base model {metadata.get('base_model')!r}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype)
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate_json(self, system: str, user: str) -> dict[str, Any]:
        import torch

        prompt = _chat_text(
            self.tokenizer,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("LoRA SOAP model did not return valid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("LoRA SOAP model must return a JSON object")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal open-weight LLM with LoRA for cited SOAP generation."
    )
    parser.add_argument("--train", required=True, help="Training JSONL file")
    parser.add_argument("--validation", required=True, help="Validation JSONL file")
    parser.add_argument(
        "--base-model", required=True,
        help="Exact Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for the LoRA adapter")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()
    train_soap_lora(
        train_path=args.train,
        validation_path=args.validation,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )


if __name__ == "__main__":
    main()
