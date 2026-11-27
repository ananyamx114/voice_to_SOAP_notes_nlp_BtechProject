"""Pause only the owned HYBRID trainer, back up recovery state, launch split validation."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
assert os.environ["QWEN_TRAIN_GPU"] != os.environ["QWEN_VALIDATION_GPU"], "Choose distinct permitted GPUs"
run = ROOT / "data/outputs/dgx-qwen7b-soap/hybrid"
local = ROOT / ".local/dgx-migration"
assert not (run / "completed.json").exists(), "Training already complete; do not restart"
for session in ("vsoap-hybrid-async", "vsoap-validation-async"):
    assert subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0, "Async session already exists"
assert subprocess.run(["pgrep", "-af", "[e]valuate_qwen7b_resumable.py"], capture_output=True).returncode == 1, "Generation evaluation is still active"
pid = int((run / "pid.txt").read_text())
process = Path(f"/proc/{pid}")
if process.exists():
    command = (process / "cmdline").read_bytes()
    assert b"train_qwen7b_soap.py" in command and b"hybrid" in command
    assert (process / "cwd").resolve() == ROOT.resolve()
    os.kill(pid, signal.SIGINT)
    print("Requested checkpointed pause of HYBRID PID", pid, flush=True)
    for _ in range(120):
        if not process.exists():
            break
        time.sleep(5)
    else:
        raise RuntimeError("Trainer has not exited; no new jobs launched. Inspect old log.")
checkpoints = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
assert checkpoints
latest = checkpoints[-1]
state = json.loads((latest / "trainer_state.json").read_text())
assert state["global_step"] == int(latest.name.split("-")[-1]) < 4800
assert state.get("best_model_checkpoint") and state.get("best_metric") is not None
required = ("trainer_state.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth", "scaler.pt")
assert all((latest / name).is_file() for name in required), "Recovery checkpoint incomplete"
backup = local / "pre-async-backup" / latest.name
assert not backup.exists(), "Existing migration backup preserved; inspect before repeating"
backup.mkdir(parents=True)
records = []
for source in {latest, Path(state["best_model_checkpoint"])}:
    target = backup / source.name
    shutil.copytree(source, target)
    for p in source.rglob("*"):
        if p.is_file():
            relative = p.relative_to(source)
            with p.open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            with (target / relative).open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == sha
            records.append({"checkpoint": source.name, "file": str(relative), "sha256": sha})
(backup / "recovery-inventory.json").write_text(json.dumps({"update": state["global_step"], "files": records}, indent=2) + "\n")
print("Preserved and verified latest and selected checkpoints at", state["global_step"], flush=True)
for session, gpu, script, arguments, log in (
    ("vsoap-validation-async", os.environ["QWEN_VALIDATION_GPU"], "validate_qwen7b_async.py", [], "qwen7b-validation.log"),
    ("vsoap-hybrid-async", os.environ["QWEN_TRAIN_GPU"], "train_qwen7b_async.py", ["--approach", "hybrid"], "qwen7b-hybrid-async.log"),
):
    command = shlex.join(["env", "CUDA_DEVICE_ORDER=PCI_BUS_ID", f"CUDA_VISIBLE_DEVICES={gpu}",
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "scripts/qwen7b" / script), *arguments])
    command += " > " + shlex.quote(str(local / log)) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(ROOT), command], check=True)
    print("Launched", session, "on physical GPU", gpu, flush=True)
print("Inspect both logs: training must advance and validation must report VALIDATION_PARITY_PASSED.", flush=True)
