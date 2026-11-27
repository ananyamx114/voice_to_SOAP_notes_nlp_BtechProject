"""Validate published HYBRID checkpoints on GPU2 while GPU3 keeps training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".local/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicitly select one permitted GPU"
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, set_seed)
    from voice_to_soap.aci_soap import _encode_example, load_aci_soap_examples
    assert torch.cuda.device_count() == 1
    set_seed(42)
    run = ROOT / "data/outputs/dgx-qwen7b-soap/hybrid"
    lock = (run / "async-validator.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pin = read(ROOT / ".local/dgx-migration/qwen7b-model-pin.json")
    config = read(run / "run_config.json")
    while not (mode := read(run / "async-validation-mode.json")):
        print("Waiting for checkpointed training restart.", flush=True)
        time.sleep(10)
    results = run / "external-validation"
    results.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(pin["snapshot_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    path = ROOT / "data/processed/betrac_2000/validation.jsonl"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == config["split_sha256"]["validation"]
    rows = load_aci_soap_examples(path)
    assert len(rows) == 200
    examples = [_encode_example(tokenizer, r, 4096, "hybrid") for r in rows]
    base = AutoModelForCausalLM.from_pretrained(pin["snapshot_path"], local_files_only=True,
        dtype=torch.float16, attn_implementation="sdpa")
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=config["target_modules"], bias="none"))
    trainer = Trainer(model=model,
        args=TrainingArguments(output_dir=str(results / "runtime"), per_device_eval_batch_size=1,
            fp16=True, bf16=False, prediction_loss_only=True, report_to="none",
            remove_unused_columns=False, dataloader_num_workers=0, dataloader_pin_memory=False),
        eval_dataset=examples, processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))

    def load_checkpoint(folder):
        incoming = load_file(str(folder / "adapter_model.safetensors"), device="cpu")
        incompatible = set_peft_model_state_dict(model, incoming)
        assert not incompatible.unexpected_keys, str(incompatible.unexpected_keys)
        assert not [key for key in incompatible.missing_keys if "lora_" in key]
        model.eval()

    def evaluate(folder):
        load_checkpoint(folder)
        metrics = trainer.evaluate()
        value = float(metrics["eval_loss"])
        assert __import__("math").isfinite(value)
        return value, metrics["eval_runtime"]

    parity_file = results / "parity-check.json"
    if not parity_file.exists():
        loss, seconds = evaluate(Path(mode["prior_best_checkpoint"]))
        expected = mode["prior_best_loss"]
        assert abs(loss - expected) < 1e-4, f"Validation parity failed: {loss} versus {expected}"
        write(parity_file, {"status": "passed", "expected_loss": expected, "observed_loss": loss,
                            "absolute_tolerance": 1e-4, "seconds": seconds})
        print("VALIDATION_PARITY_PASSED", loss, flush=True)
    best_loss = mode["prior_best_loss"]
    best_checkpoint = Path(mode["prior_best_checkpoint"])
    history = list(mode["prior_validation_history"])
    for record in sorted(results.glob("checkpoint-*.json"), key=lambda p: int(p.stem.split("-")[-1])):
        row = read(record)
        history.append(row)
        if row["eval_loss"] < best_loss:
            best_loss, best_checkpoint = row["eval_loss"], run / f"checkpoint-{row['step']}"
    while True:
        checkpoints = sorted(run.glob("checkpoint-*/validation-ready.json"),
                             key=lambda p: int(p.parent.name.split("-")[-1]))
        for ready in checkpoints:
            step = int(ready.parent.name.split("-")[-1])
            target = results / f"checkpoint-{step}.json"
            if target.exists():
                continue
            assert read(ready)["step"] == step
            loss, seconds = evaluate(ready.parent)
            row = {"step": step, "epoch": step / 1600, "eval_loss": loss,
                   "eval_runtime": seconds, "validation_physical_gpu": os.environ.get("QWEN_VALIDATION_GPU", "unset")}
            write(target, row)
            history.append(row)
            if loss < best_loss:
                best_loss, best_checkpoint = loss, ready.parent
            write(run / "async-validation-summary.json", {
                "last_evaluated_update": step, "best_validation_loss": best_loss,
                "selected_checkpoint_step": int(best_checkpoint.name.split("-")[-1]),
                "best_checkpoint": str(best_checkpoint), "validation_history": history,
                "status": "running_or_waiting_for_checkpoint"})
            print("VALIDATED_CHECKPOINT", step, "loss", loss, "best", best_loss, flush=True)
        finished = read(run / "async-training-finished.json")
        if finished and (results / "checkpoint-4800.json").exists():
            assert finished["global_step"] == 4800
            load_checkpoint(best_checkpoint)
            selected = run / "selected-adapter"
            assert not selected.exists(), "Selected adapter already exists; preserve it"
            model.save_pretrained(selected)
            tokenizer.save_pretrained(selected)
            metadata = {**config, "global_step": 4800, "best_validation_loss": best_loss,
                        "best_checkpoint": str(best_checkpoint), "adapter_path": str(selected),
                        "execution_mode": "gpu3_training_gpu2_asynchronous_validation",
                        "test_evaluated": False}
            write(selected / "adapter_metadata.json", metadata)
            write(run / "completed.json", metadata)
            summary = read(run / "async-validation-summary.json")
            summary["status"] = "complete"
            write(run / "async-validation-summary.json", summary)
            print("TRAINING AND VALIDATION COMPLETE: selected adapter saved.", flush=True)
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
