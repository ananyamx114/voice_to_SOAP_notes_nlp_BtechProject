"""Private Kaggle script entry: verified assets, exact resume, persistent outputs.

Attach assets.tar + assets.inventory.json and checkpoint.tar + its inventory.
For later sessions also attach the previous kernel's resume-state.tar/inventory.
No credentials are needed: all base checkpoints used here are publicly accessible.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tarfile
from time import monotonic

# Each T4 has its own memory. The current trainer deliberately uses one GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/kaggle/temp/asr-huggingface"
os.environ["OMP_NUM_THREADS"] = "4"
WORK = Path("/kaggle/working/voice-to-soap")
INPUT = Path("/kaggle/input")
STARTED = monotonic()


def input_files(name):
    found, seen = [], set()
    for folder, directories, files in os.walk(INPUT, followlinks=True):
        resolved = Path(folder).resolve()
        if resolved in seen:
            directories[:] = []
            continue
        seen.add(resolved)
        if name in files:
            found.append(Path(folder) / name)
    return found


def run_settings():
    files = input_files("kaggle-asr-run.json")
    if len(files) > 1:
        raise ValueError("Attach only one Kaggle ASR run settings file")
    settings = json.loads(files[0].read_text()) if files else {
        "mode": "evaluate_completed", "expected_updates": 515}
    if settings.get("mode") not in {"evaluate_completed", "resume_training"}:
        raise ValueError("Unknown Kaggle ASR run mode")
    updates = settings.get("expected_updates")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0:
        raise ValueError("Explicit positive expected_updates is required")
    return settings


def validate_source(state, settings):
    if settings["mode"] == "evaluate_completed":
        require_completed_training(state)
        if settings["expected_updates"] != 515:
            raise ValueError("Completed experiment evaluation requires update 515")
    elif state.get("status") != "paused":
        raise ValueError("Training continuation requires a safely paused source")
    if state.get("optimizer_updates") is not None and state["optimizer_updates"] != settings["expected_updates"]:
        raise ValueError("Source update count differs from the explicitly requested checkpoint")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def restore(name, required=True):
    # Kaggle may unpack tar inputs and may expose versioned inputs via symlinks.
    found = input_files(name + ".inventory.json")
    if not found and not required:
        return False
    if len(found) != 1:
        raise ValueError(f"Expected exactly one {name} archive; found {len(found)}")
    inventory_path = found[0]
    inventory = json.loads(inventory_path.read_text())
    source = inventory_path.parent / (name + ".tar")
    expected = {r["path"]: r for r in inventory["files"]}
    if not source.is_file():
        first = next(iter(expected))
        roots = [inventory_path.parent / name, inventory_path.parent,
                 inventory_path.parent / (name + ".tar")]
        roots = [root for root in roots if (root / first).is_file()]
        if len(roots) != 1:
            raise ValueError(f"Cannot identify unpacked {name} root near {inventory_path}")
        for relative, record in expected.items():
            original = roots[0] / relative
            original.resolve().relative_to(roots[0].resolve())
            target = (WORK / relative).resolve()
            target.relative_to(WORK.resolve())
            if digest(original) != record["sha256"]:
                raise ValueError(f"Unpacked input checksum mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if digest(target) != record["sha256"]:
                raise ValueError(f"Copied input checksum mismatch: {relative}")
        print("RESTORED_VERIFIED_UNPACKED", name, len(expected), flush=True)
        return True
    if digest(source) != inventory["sha256"]:
        raise ValueError(f"{name} archive checksum mismatch")
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Archive inventory/member mismatch")
        for member in members:
            target = (WORK / member.name).resolve()
            target.relative_to(WORK.resolve())
            if not member.isfile():
                raise ValueError("Only regular, inventoried files may be restored")
            stream = archive.extractfile(member)
            value = hashlib.sha256()
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as destination:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
                    destination.write(block)
            if value.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Restored file checksum mismatch: {member.name}")
    print("RESTORED_VERIFIED", name, len(expected), flush=True)
    return True


def command(args, timeout=None):
    print("RUN", args, flush=True)
    process = subprocess.Popen(args, cwd=WORK, env=os.environ.copy(), start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        return False
    if code:
        raise subprocess.CalledProcessError(code, args)
    return True


def save_resume():
    sys.path.insert(0, str(WORK))
    from scripts.prepare_kaggle_asr import archive
    paths = set()
    for folder in ("data/outputs/asr-lora-experiment-run2/training",
                   "data/outputs/asr-lora-kaggle-evaluation"):
        paths.update(p for p in (WORK / folder).rglob("*")
                     if p.is_file() and not p.name.endswith(".partial") and not p.is_symlink())
    result = archive(paths, Path("/kaggle/working/resume-state.tar"), root=WORK)
    print("PERSISTENT_RESUME_ARCHIVE", json.dumps(result), flush=True)


def main():
    settings = run_settings()
    WORK.mkdir(parents=True, exist_ok=True)
    restore("assets")
    restore("checkpoint")
    # Evaluation recovery is fail-closed by default. Training continuation
    # requires an explicit private settings file and exact checkpoint position.
    restore("resume-state", required=settings["mode"] == "evaluate_completed")
    source_training = WORK / "data/outputs/asr-lora-experiment-run2/training/run.json"
    source_state = json.loads(source_training.read_text())
    validate_source(source_state, settings)
    os.environ["PYTHONPATH"] = str(WORK / "src")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Kaggle's Python lacks ensurepip. Inherit its CUDA wheel and pip, then
    # install ASR pins into this fresh environment without touching global pins.
    command([sys.executable, "-m", "venv", "--without-pip", "--system-site-packages", ".venv"])
    runtime_python = str(WORK / ".venv/bin/python")
    # Keep Kaggle's CUDA-enabled torch wheel; never replace it with a CPU wheel.
    command([runtime_python, "-m", "pip", "install", "--no-cache-dir",
             "transformers==4.57.6", "peft==0.17.1", "accelerate>=0.34,<2",
             "huggingface-hub>=0.34,<1", "soundfile>=0.12", "scipy>=1.13",
             "numpy>=1.26,<3", "PyYAML>=6", "jiwer==4.0.0", "num2words==0.5.14",
             "psutil>=5.9,<8"])
    command([runtime_python, "-c", "import torch, transformers, peft, numpy, soundfile, jiwer; "
             "assert torch.cuda.is_available(); "
             "assert transformers.__version__ == '4.57.6'; "
             "assert peft.__version__ == '0.17.1'; "
             "print('CUDA_ASR_ENVIRONMENT_VERIFIED', torch.__version__, transformers.__version__, peft.__version__)"])
    import torch
    from packaging.version import Version
    assert torch.cuda.is_available(), "Real CUDA GPU required"
    assert Version(torch.__version__.split("+")[0]) >= Version("2.8"), "Trainer requires PyTorch >=2.8"
    saved = torch.load(WORK / "data/outputs/asr-lora-experiment-run2/training/checkpoint.pt",
                       map_location="cpu", weights_only=True)
    if saved["state"]["updates"] != settings["expected_updates"]:
        raise ValueError("Checkpoint tensor state differs from explicitly requested update")
    del saved
    hardware = {"torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "visible_gpus": torch.cuda.device_count(), "training_device": "cuda"}
    (WORK / "kaggle-hardware.json").write_text(json.dumps(hardware, indent=2))
    print("KAGGLE_HARDWARE", json.dumps(hardware), flush=True)
    try:
        remaining = 8 * 3600 - (monotonic() - STARTED)
        command([runtime_python, "scripts/run_asr_cloud_session.py", "--seconds", str(max(1, remaining))])
        training = WORK / "data/outputs/asr-lora-experiment-run2/training"
        status = json.loads((training / "run.json").read_text())["status"]
        if status == "complete" and monotonic() - STARTED < 8 * 3600:
            # Separate CUDA controls; MPS baseline results cannot be a matched CUDA comparison.
            evaluation = WORK / "data/outputs/asr-lora-kaggle-evaluation"
            evaluation.mkdir(exist_ok=True)
            link = evaluation / "training"
            if not link.exists():
                link.symlink_to(training, target_is_directory=True)
            command([runtime_python, "scripts/run_asr_lora_experiment.py", "--resume", "--rerun-base",
                     "--clinical-train", "data/outputs/asr-lora-experiment-run2/prepared/clinical-train",
                     "--clinical-validation", "data/processed/asr-lora-clinical-validation-run1",
                     "--fleurs", "data/outputs/asr-lora-experiment-run2/prepared/fleurs",
                     "--fleurs-test", "data/processed/asr-lora-fleurs-test-run1",
                     "--outdir", str(evaluation), "--config", "configs/asr-lora-expanded.yaml",
                     "--device", "cuda"], timeout=max(1, 10 * 3600 - (monotonic() - STARTED)))
    finally:
        save_resume()


def require_completed_training(state):
    if state.get("status") != "complete" or state.get("optimizer_updates") != 515:
        raise ValueError("Evaluation resume requires the completed update-515 checkpoint; refusing to retrain an older snapshot")


if __name__ == "__main__":
    main()
