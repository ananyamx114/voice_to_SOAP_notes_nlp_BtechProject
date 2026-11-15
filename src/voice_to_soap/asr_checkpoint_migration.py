"""Explicit, audited MPS-to-CUDA relocation of our own optimizer checkpoint.

Run this only in the verified CUDA destination, against a copied training tree.
It does not load Whisper or modify the original local checkpoint.
"""
import copy
from importlib.metadata import version
import json
from pathlib import Path
import shutil

from .asr_component import file_sha256
from .asr_training_data import check_disjoint, read_segments
from .io_utils import write_json


def convert_checkpoint(checkpoint, cuda_rng):
    """Change only backend RNG; retain CPU RNG, optimizer, scaler and offset."""
    required = {"adapter", "optimizer", "scaler", "state", "torch_rng", "mps_rng"}
    if not required <= checkpoint.keys() or "cuda_rng" in checkpoint:
        raise ValueError("Expected an original MPS training checkpoint")
    state = checkpoint["state"]
    if state["updates"] <= 0 or state["next_epoch"] < 0 or state["next_offset"] < 0:
        raise ValueError("Invalid saved training position")
    if not checkpoint["adapter"] or not checkpoint["optimizer"]["state"]:
        raise ValueError("Missing learned adapter/optimizer state")
    if not cuda_rng:
        raise ValueError("Destination CUDA RNG state is required")
    return {**checkpoint, "cuda_rng": cuda_rng}


def migrate(training, train_manifest, validation_manifest, config):
    import torch
    from . import whisper_lora

    training = Path(training)
    receipt = training / "backend-migration.json"
    if receipt.exists():
        # Subsequent Kaggle sessions use the normal exact-backend resume guards.
        return json.loads(receipt.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Migration must be verified on a real CUDA destination")
    previous = json.loads((training / "run.json").read_text())
    if previous["task"] != whisper_lora.TASK or previous["config"] != config or previous["probe_only"]:
        raise ValueError("Task, full training config or checkpoint purpose mismatch")
    if previous["device"] != "mps":
        raise ValueError("Only explicit MPS-to-CUDA migration is supported")
    if previous["implementation_sha256"] != file_sha256(Path(whisper_lora.__file__)):
        raise ValueError("Trainer changed since the saved checkpoint; refusing migration")
    for key, path in (("train_manifest_sha256", train_manifest),
                      ("validation_manifest_sha256", validation_manifest)):
        if previous[key] != file_sha256(Path(path)):
            raise ValueError(f"Manifest bytes changed: {key}")
    rows, validation = read_segments(train_manifest), read_segments(validation_manifest)
    check_disjoint(rows, validation)
    if previous["train_ids"] != [r["id"] for r in rows] or previous["validation_ids"] != [r["id"] for r in validation]:
        raise ValueError("Training order/validation cohort changed")
    versions = {name: version(name) for name in ("torch", "transformers", "peft", "numpy")}
    for name in ("transformers", "peft"):
        if versions[name] != previous["versions"][name]:
            raise ValueError(f"Pin the original {name} version before migrating")
    # Torch/NumPy platform changes are recorded explicitly, not silently waived.
    path = training / "checkpoint.pt"
    old_sha = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    updates_per_epoch = (len(rows) + config["training"]["gradient_accumulation"] - 1) // config["training"]["gradient_accumulation"]
    state = checkpoint["state"]
    offset = state["next_offset"]
    expected = state["next_epoch"] * updates_per_epoch + (offset + config["training"]["gradient_accumulation"] - 1) // config["training"]["gradient_accumulation"]
    if offset > len(rows) or state["updates"] != expected:
        raise ValueError("Optimizer update count does not match saved epoch/sample offset")
    seed = int(config["seed"])
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.cuda.manual_seed_all(seed)
        cuda_rng = torch.cuda.get_rng_state_all()
    converted = convert_checkpoint(checkpoint, cuda_rng)
    source = training / "migration-original"
    source.mkdir(exist_ok=False)
    # Retain original bytes before atomically replacing the destination copy.
    shutil.copyfile(path, source / "checkpoint.pt")
    if file_sha256(source / "checkpoint.pt") != old_sha:
        raise ValueError("Recovery copy checksum mismatch")
    write_json(source / "run.json", previous)
    temporary = training / "checkpoint.pt.partial"
    torch.save(converted, temporary)
    temporary.replace(path)
    updated = copy.deepcopy(previous)
    updated.update(device="cuda", versions=versions, status="paused")
    report = {"source_device": "mps", "destination_device": "cuda",
              "source_checkpoint_sha256": old_sha, "destination_checkpoint_sha256": file_sha256(path),
              "source_versions": previous["versions"], "destination_versions": versions,
              "state": state, "adapter_optimizer_scaler_cpu_rng_preserved": True,
              "cuda_rng_policy": f"New CUDA RNG seeded with {seed}; MPS RNG retained for provenance",
              "bitwise_equivalent_continuation": False}
    updated["backend_migration"] = report
    write_json(training / "run.json", updated)
    write_json(receipt, report)
    return report
