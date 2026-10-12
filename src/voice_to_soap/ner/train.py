import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


def load_annotated(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text")
            entities = item.get("entities")
            if not isinstance(text, str) or not isinstance(entities, list):
                raise ValueError(f"{path}:{line_number} needs text and entities")
            for entity in entities:
                start, end, label = entity.get("start"), entity.get("end"), entity.get("label")
                if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
                    raise ValueError(f"{path}:{line_number} has a malformed entity")
                if start < 0 or end <= start or end > len(text):
                    raise ValueError(f"{path}:{line_number} has an entity span outside text")
            rows.append({"text": text, "entities": entities})
    if not rows:
        raise ValueError(f"No annotated rows found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a clinical token classifier from character spans.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args()

    train_rows, val_rows = load_annotated(args.train), load_annotated(args.validation)
    labels = sorted({e["label"] for row in train_rows + val_rows for e in row["entities"]})
    if not labels:
        raise ValueError("No entity labels found; provide labelled spans for NER training")
    tag_names = ["O"] + [tag for label in labels for tag in (f"B-{label}", f"I-{label}")]
    tag_to_id = {tag: index for index, tag in enumerate(tag_names)}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(f"Model {args.model!r} needs a fast tokenizer for span alignment")

    def encode(rows: list[dict[str, Any]]) -> Dataset:
        def tokenize(batch: dict[str, list]) -> dict[str, list]:
            encoded = tokenizer(batch["text"], truncation=True, return_offsets_mapping=True)
            aligned = []
            for row_index, offsets in enumerate(encoded["offset_mapping"]):
                entities = rows[row_index]["entities"]
                tags = ["O"] * len(offsets)
                for entity in entities:
                    touched = [
                        token_index for token_index, (start, end) in enumerate(offsets)
                        if end > start and start < entity["end"] and end > entity["start"]
                    ]
                    if not touched:
                        raise ValueError(f"Tokenizer could not align entity {entity} in row {row_index}")
                    for place, token_index in enumerate(touched):
                        tags[token_index] = ("B-" if place == 0 else "I-") + entity["label"]
                aligned.append([tag_to_id[tag] if offset != (0, 0) else -100 for tag, offset in zip(tags, offsets)])
            encoded["labels"] = aligned
            encoded.pop("offset_mapping")
            return encoded

        ds = Dataset.from_list(rows)
        return ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    train_data, validation_data = encode(train_rows), encode(val_rows)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(tag_names), id2label=dict(enumerate(tag_names)),
        label2id=tag_to_id,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "labels.json").write_text(json.dumps(tag_names, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
