"""Local checkpoint-boundary pause/resume controller; no model or API polling.

Job manifests, PIDs and resource settings belong under ignored .local/.
The existing trainer's atomic optimizer/scaler/RNG checkpoint remains authoritative.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / ".local/asr-expanded-run2.job.json"


def detached_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def owner(job):
    try:
        process = psutil.Process(job["pid"])
        if abs(process.create_time() - job["create_time"]) > 0.01:
            raise ValueError("PID identity changed; refusing to control another process")
        return process if process.is_running() and process.status() != psutil.STATUS_ZOMBIE else None
    except psutil.NoSuchProcess:
        return None


def active_owner(job):
    try:
        return owner(job)
    except ValueError:
        # PID reuse after shutdown is not a running instance of this job.
        return None


def checkpoint(job):
    return Path(job["run_dir"]) / "training/checkpoint.pt"


def safe_point(job):
    root = Path(job["run_dir"])
    stage = read(root / "experiment.json").get("stage", "")
    if stage.endswith(("_base", "_adapter")):
        dataset, mode = stage.rsplit("_", 1)
        path = root / (dataset + "-" + mode) / "progress.jsonl"
    else:
        path = checkpoint(job)
    return {"path": str(path), "token": token(path)}


def token(path):
    try:
        stat = path.stat()
        return [stat.st_ino, stat.st_mtime_ns, stat.st_size]
    except FileNotFoundError:
        return None


def stop_owned(job):
    process = owner(job)
    if process is None:
        return
    tree = process.children(recursive=True) + [process]
    for item in reversed(tree):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(tree, timeout=15)
    for item in remaining:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(remaining, timeout=5)


def request_pause(job_file):
    job_file = Path(job_file)
    job = read(job_file)
    if active_owner(job) is None:
        raise ValueError("Job is not running; use status or resume")
    request = {"point_before_request": safe_point(job), "requested_at": time.time()}
    write(job_file.with_suffix(".pause.json"), request)
    return {"status": "pause_requested", "detail": "Will stop after the next saved optimizer checkpoint or completed ASR recording; wait for paused before shutting down"}


def pause_if_ready(job_file):
    job_file = Path(job_file)
    job = read(job_file)
    request_file = job_file.with_suffix(".pause.json")
    request = read(request_file)
    current = safe_point(job)
    if not request or current["token"] is None or current == request["point_before_request"]:
        return False
    # Validate process identity again immediately before touching its process tree.
    stop_owned(job)
    root = Path(job["run_dir"])
    events = root / "training/training.jsonl"
    latest = {}
    if events.exists():
        for line in reversed(events.read_text().splitlines()):
            try:
                latest = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    for relative in ("expanded-run.json", "experiment.json", "training/run.json"):
        path = root / relative
        metadata = read(path)
        if metadata:
            if relative == "training/run.json" and metadata.get("status") == "complete":
                continue
            metadata.update(status="paused", pause_reason="User requested pause at a saved batch boundary")
            if relative == "training/run.json" and latest:
                metadata["optimizer_updates"] = latest["update"]
                metadata["elapsed_seconds"] = latest["elapsed_seconds"] + max(
                    0, checkpoint(job).stat().st_mtime - events.stat().st_mtime)
            write(path, metadata)
    write(job_file.with_suffix(".state.json"), {"status": "paused", "checkpoint": current,
        "latest_logged_update": latest.get("update"), "paused_at": time.time()})
    return True


def watch(job_file):
    while True:
        job = read(job_file)
        run_state = read(Path(job["run_dir"]) / "expanded-run.json").get("status")
        if run_state in {"complete", "failed"} or active_owner(job) is None:
            return
        if pause_if_ready(job_file):
            return
        time.sleep(2)


def start_watcher(job_file):
    job_file = Path(job_file).resolve()
    existing = read(job_file.with_suffix(".watcher.json"))
    if existing and active_owner(existing) is not None:
        return
    with job_file.with_suffix(".controller.log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watch", "--job", str(job_file)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            **detached_options())
    write(job_file.with_suffix(".watcher.json"), {"pid": process.pid,
                                               "create_time": psutil.Process(process.pid).create_time()})


def resume(job_file, watching=True):
    job_file = Path(job_file)
    job = read(job_file)
    if active_owner(job) is not None:
        if watching:
            start_watcher(job_file)
        return {"status": "already_running", "pid": job["pid"]}
    if read(Path(job["run_dir"]) / "expanded-run.json").get("status") == "complete":
        return {"status": "complete"}
    if not checkpoint(job).is_file():
        raise ValueError("No saved model checkpoint to resume yet")
    # The previous watcher may still be exiting after completing the pause.
    previous_watcher = read(job_file.with_suffix(".watcher.json"))
    if previous_watcher:
        worker = active_owner(previous_watcher)
        if worker is not None:
            try:
                worker.terminate()
                worker.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
    job_file.with_suffix(".pause.json").unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.update(job.get("environment", {}))
    command = list(job["command"])
    if "--resume" not in command:
        command.append("--resume")
    with Path(job["log"]).open("a") as log:
        process = subprocess.Popen(command, cwd=job["cwd"], env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            **detached_options())
    job.update(pid=process.pid, create_time=psutil.Process(process.pid).create_time())
    write(job_file, job)
    write(job_file.with_suffix(".state.json"), {"status": "running", "resumed_at": time.time()})
    if watching:
        start_watcher(job_file)
    return {"status": "resumed", "pid": process.pid}


def status(job_file):
    job_file = Path(job_file)
    job = read(job_file)
    running = active_owner(job) is not None
    state = read(job_file.with_suffix(".state.json"))
    run_status = read(Path(job["run_dir"]) / "expanded-run.json").get("status")
    stopped_status = "paused" if state.get("status") == "paused" else (
        run_status if run_status in {"complete", "failed"} else "not_running")
    result = {"status": "running" if running else stopped_status,
              "pid": job["pid"], "checkpoint_saved": checkpoint(job).exists(),
              "pause_requested": job_file.with_suffix(".pause.json").exists()}
    if not running:
        result["run_status"] = read(Path(job["run_dir"]) / "expanded-run.json").get("status")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pause", "resume", "status", "watch", "attach"))
    parser.add_argument("--job", type=Path, default=DEFAULT_JOB)
    args = parser.parse_args()
    if args.action == "watch":
        watch(args.job)
    elif args.action == "attach":
        start_watcher(args.job)
        print(json.dumps(status(args.job)))
    elif args.action == "pause":
        result = request_pause(args.job)
        start_watcher(args.job)
        print(json.dumps(result))
    elif args.action == "resume":
        print(json.dumps(resume(args.job)))
    else:
        print(json.dumps(status(args.job)))
