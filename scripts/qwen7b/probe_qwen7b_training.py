"""Disposable full-length training-memory probe; never modifies saved adapters."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from voice_to_soap.aci_soap import _encode_example, load_aci_soap_examples

local = ROOT / ".local/dgx-migration"
assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select one permitted GPU"
assert torch.cuda.device_count() == 1
audit = json.loads((local / "qwen7b-length-audit.json").read_text())
assert audit["all_fit"]
pin = json.loads((local / "qwen7b-model-pin.json").read_text())
assert audit["revision"] == pin["revision"]
set_seed(42)
device = torch.device("cuda:0")
torch.cuda.set_device(device)
tokenizer = AutoTokenizer.from_pretrained(pin["snapshot_path"], local_files_only=True)
rows = load_aci_soap_examples(ROOT / "data/processed/betrac_2000/train.jsonl")
encoded = max((_encode_example(tokenizer, row, 4096, "hybrid") for row in rows),
              key=lambda row: len(row["input_ids"]))
assert len(encoded["input_ids"]) == 4096
print("Loading full FP16 Qwen7B; probing 4096-token training on the selected GPU.", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    pin["snapshot_path"], local_files_only=True, dtype=torch.float16,
    device_map={"": 0}, attn_implementation="sdpa")
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none"))
model.train()
parameters = [p for p in model.parameters() if p.requires_grad]
assert all(p.dtype == torch.float32 for p in parameters)
optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda")
batch = {k: torch.tensor([v], device=device) for k, v in encoded.items()}
batch["attention_mask"] = torch.ones_like(batch["input_ids"])
torch.cuda.reset_peak_memory_stats(device)
torch.cuda.synchronize(device)
started = time.monotonic()
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(**batch).loss
assert torch.isfinite(loss).item(), "Nonfinite loss"
scaler.scale(loss).backward()
scaler.unscale_(optimizer)
grads = [p.grad for p in parameters if p.grad is not None]
assert grads and all(torch.isfinite(g).all().item() for g in grads), "Nonfinite gradients"
norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
assert torch.isfinite(norm).item() and norm.item() > 0
scaler.step(optimizer)
scaler.update()
torch.cuda.synchronize(device)
report = {"status": "passed", "model_id": pin["model_id"], "revision": pin["revision"],
          "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "sequence_tokens": 4096, "batch_size": 1, "lora_rank": 8,
          "trainable_parameters": sum(p.numel() for p in parameters),
          "loss": loss.item(), "gradient_norm_before_clip": norm.item(),
          "step_seconds": round(time.monotonic() - started, 2),
          "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
          "peak_reserved_gib": round(torch.cuda.max_memory_reserved(device) / 1024**3, 2),
          "adapter_saved": False, "production_training_started": False}
(local / "qwen7b-training-probe.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
