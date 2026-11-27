"""Single-V100, resumable DIRECT/HYBRID BeTraC SOAP training; test data is never loaded."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".local/huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Trainer, TrainerCallback, TrainingArguments, set_seed)
from voice_to_soap.aci_soap import _encode_example, load_aci_soap_examples


class SaveAndStop(TrainerCallback):
    requested = False

    def request(self, signum, frame):
        self.requested = True
        print("Pause requested: saving at the next completed optimizer step.", flush=True)

    def on_step_end(self, args, state, control, **kwargs):
        if self.requested:
            control.should_save = True
            control.should_training_stop = True
        return control


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=("direct", "hybrid"), default="direct")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicitly select one permitted GPU"
    assert torch.cuda.device_count() == 1
    set_seed(42)
    local = ROOT / ".local/dgx-migration"
    pin = json.loads((local / "qwen7b-model-pin.json").read_text())
    assert json.loads((local / "qwen7b-training-probe.json").read_text())["status"] == "passed"
    assert json.loads((local / "qwen7b-length-audit.json").read_text())["all_fit"]
    output = ROOT / "data/outputs/dgx-qwen7b-soap" / args.approach
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completed.json").exists():
        print("This run is already complete; existing results preserved.")
        return
    manifest = json.loads((ROOT / "data/processed/betrac_2000/split_manifest.json").read_text())
    config = {"base_model": pin["model_id"], "model_revision": pin["revision"],
              "approach": args.approach, "epochs": 3, "batch_size": 1,
              "gradient_accumulation_steps": 1, "learning_rate": 1e-4,
              "weight_decay": 0.0, "seed": 42, "max_seq_length": 4096,
              "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05,
              "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "save_steps": 100, "eval_steps": 100, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "split_sha256": {s: manifest["sha256"][s] for s in ("train", "validation")},
              "task": "betrac_synthetic_four_string_heuristic_soap",
              "selection_metric": "validation_loss", "clinician_approval_required": True}
    config_path = output / "run_config.json"
    if config_path.exists():
        assert json.loads(config_path.read_text()) == config, "Existing run configuration differs"
    else:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    checkpoints = list(output.glob("checkpoint-*"))
    for checkpoint in checkpoints:
        assert (checkpoint / "trainer_state.json").is_file(), "Incomplete checkpoint: " + str(checkpoint)
        for required in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
            assert (checkpoint / required).is_file(), "Checkpoint missing " + required
    latest = max(checkpoints, key=lambda p: int(p.name.split("-")[-1])) if checkpoints else None
    tokenizer = AutoTokenizer.from_pretrained(pin["snapshot_path"], local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    datasets = {}
    for split, count in (("train", 1600), ("validation", 200)):
        path = ROOT / "data/processed/betrac_2000" / (split + ".jsonl")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config["split_sha256"][split]
        examples = load_aci_soap_examples(path)
        assert len(examples) == count
        datasets[split] = [_encode_example(tokenizer, row, 4096, args.approach) for row in examples]
    model = AutoModelForCausalLM.from_pretrained(
        pin["snapshot_path"], local_files_only=True, dtype=torch.float16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=config["target_modules"]))
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=3, learning_rate=1e-4,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=1, weight_decay=0.0, seed=42, data_seed=42,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=100, save_strategy="steps", save_steps=100,
        save_total_limit=3, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=10, prediction_loss_only=True, dataloader_num_workers=0,
        dataloader_pin_memory=False, report_to="none", remove_unused_columns=False,
        optim="adamw_torch", max_grad_norm=1.0)
    stopper = SaveAndStop()
    signal.signal(signal.SIGINT, stopper.request)
    signal.signal(signal.SIGTERM, stopper.request)
    trainer = Trainer(model=model, args=training_args, train_dataset=datasets["train"],
        eval_dataset=datasets["validation"], processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[stopper])
    (output / "pid.txt").write_text(str(os.getpid()) + "\n")
    print("Resume checkpoint:", str(latest) if latest else "new SOAP run", flush=True)
    trainer.train(resume_from_checkpoint=str(latest) if latest else None)
    if stopper.requested:
        print("PAUSED: checkpoint saved; rerun the same command to resume.", flush=True)
        return
    assert trainer.state.global_step == 4800, "Unexpected completed training position"
    adapter = output / "selected-adapter"
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    metadata = {**config, "global_step": trainer.state.global_step,
                "best_validation_loss": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "adapter_path": str(adapter), "test_evaluated": False}
    (adapter / "adapter_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "completed.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("TRAINING COMPLETE: validation-selected adapter saved; test remains unevaluated.", flush=True)


if __name__ == "__main__":
    main()
