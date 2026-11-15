"""Periodically monitor a private Kaggle ASR job without loading any model.

Keeps raw live logs, quality flags, and a readable report under ignored .local/.
Downloads and verifies the saved resume archive when the session terminates.
Never changes training parameters, starts another job, or publishes anything.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time


def read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def progress(text, baseline, initial_checkpoint=None):
    updates = re.findall(r"epoch (\d+) update (\d+): loss=([^\s]+)", text)
    latest_update = max(updates, key=lambda item: int(item[1])) if updates else None
    validations = re.findall(r"validation (\d+)/(\d+):", text)
    checkpoints = [initial_checkpoint] if initial_checkpoint else []
    for line in text.splitlines():
        if line.startswith("CLOUD_CHECKPOINT_SAVED "):
            try:
                checkpoints.append(json.loads(line.split(" ", 1)[1]))
            except json.JSONDecodeError:
                pass  # A live stream can end in the middle of a line.
    latest = max(checkpoints, key=lambda item: (item.get("updates", 0), item.get("next_epoch", 0))) if checkpoints else {}
    best = latest.get("best_metrics")
    comparisons = {}
    flags = []
    if best:
        for language, metric in best["by_language"].items():
            control = baseline.get("by_language", {}).get(language)
            if not control:
                continue
            comparisons[language] = {name: {"historical_base": control[name],
                "best_validation": metric[name], "change": metric[name] - control[name]}
                for name in ("wer", "cer")}
            for name in ("wer", "cer"):
                if metric[name] > control[name]:
                    flags.append(f"{language} best-validation {name.upper()} exceeds historical MPS base")
    if re.search(r"Traceback \(most recent call last\)|CUDA out of memory|Nonfinite loss", text):
        flags.append("Runtime error detected; inspect live.log and remote status")
    stages = re.findall(r"^Starting stage: (.+)$", text, re.MULTILINE)
    cases = re.findall(r"^\[(\d+)/(\d+)\] ASR ([^\s]+)", text, re.MULTILINE)
    return {"latest_update": int(latest_update[1]) if latest_update else latest.get("updates"),
            "latest_training_epoch": int(latest_update[0]) if latest_update else None,
            "latest_training_loss": float(latest_update[2]) if latest_update else None,
            "latest_validation_progress": list(map(int, validations[-1])) if validations else None,
            "last_saved_checkpoint": latest,
            "latest_evaluation_stage": stages[-1] if stages else None,
            "latest_asr_record": list(cases[-1]) if cases else None,
            "completed_validation_epochs": sorted({s["next_epoch"] for s in checkpoints
                if s.get("next_offset") == 0 and s.get("next_epoch", 0) > 0} |
                set((initial_checkpoint or {}).get("saved_validation_epochs", []))),
            "best_validation": best, "diagnostic_comparison": comparisons, "flags": flags,
            "comparison_limit": "Historical base validation ran on MPS; current validation runs on CUDA. "
                "These are diagnostic comparisons. Completed paired CUDA test comparisons are separate. "
                "The logged metrics describe the best checkpoint so far, not every epoch's current score."}


def verify_download(folder):
    folder = Path(folder)
    source, inventory_path = folder / "resume-state.tar", folder / "resume-state.inventory.json"
    inventory = read(inventory_path)
    if not source.is_file() or not inventory:
        raise ValueError("Saved resume archive/inventory not yet available")
    if sha256(source) != inventory["sha256"]:
        raise ValueError("Downloaded resume archive SHA-256 mismatch")
    expected = {r["path"]: r for r in inventory["files"]}
    epoch_metrics, paired = {}, {}
    with tarfile.open(source) as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError("Downloaded resume archive members differ from inventory")
        for member in members:
            if not member.isfile():
                raise ValueError("Unexpected archive member type")
            value = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected[member.name]["sha256"]:
                raise ValueError(f"Downloaded member checksum mismatch: {member.name}")
            if re.search(r"/training/epoch-\d+/metrics\.json$", member.name):
                epoch_metrics[member.name] = json.load(archive.extractfile(member))
            if re.search(r"/asr-lora-kaggle-evaluation/[^/]+-comparison\.json$", member.name):
                paired[member.name] = json.load(archive.extractfile(member))
    return {"status": "verified", "archive_sha256": inventory["sha256"],
            "files_verified": len(members), "epoch_validation_metrics": epoch_metrics,
            "paired_cuda_comparisons": paired, "quality_flags": paired_flags(paired)}


def paired_flags(comparisons):
    flags = []
    for path, result in comparisons.items():
        dataset = Path(path).name.removesuffix("-comparison.json")
        for language, scores in result.get("by_language", {}).items():
            for name in ("wer", "cer"):
                if scores["adapter"][name] > scores["base"][name]:
                    flags.append(f"Paired CUDA test regression: {dataset}/{language} {name.upper()}")
    return flags


def report(folder, state):
    lines = ["# Kaggle ASR monitor", "", f"Checked (UTC): {state['checked_at_utc']}",
             f"Remote status: {state['remote_status']}",
             f"Latest optimizer update: {state['latest_update']}",
             f"Latest logged validation progress: {state['latest_validation_progress']}",
             f"Completed validation epochs: {state['completed_validation_epochs']}", "",
             f"Latest evaluation stage: {state.get('latest_evaluation_stage')}",
             f"Latest ASR recording: {state.get('latest_asr_record')}", "",
             "Training loss is not an ASR quality result.", ""]
    if state["diagnostic_comparison"]:
        lines += ["Best validation so far versus historical base (percent):", "",
                  "| Language | Base WER | Best WER | WER change (pp) | Base CER | Best CER |",
                  "|---|---:|---:|---:|---:|---:|"]
        for language, scores in state["diagnostic_comparison"].items():
            wer, cer = scores["wer"], scores["cer"]
            lines.append(f"| {language} | {wer['historical_base']*100:.2f} | {wer['best_validation']*100:.2f} | "
                         f"{wer['change']*100:+.2f} | {cer['historical_base']*100:.2f} | {cer['best_validation']*100:.2f} |")
    else:
        lines += ["No completed validation quality metrics have been observed yet."]
    lines += ["", state["comparison_limit"], "", "Flags:", ""]
    lines += [f"- {flag}" for flag in state["flags"]] or ["- None currently detected."]
    if state.get("download"):
        lines += ["", f"Checkpoint download: {state['download'].get('status')}"]
    (folder / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(folder / "status.json", state)
    with (folder / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(state) + "\n")


def main():
    import psutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--outdir", default=".local/kaggle-asr/monitor")
    parser.add_argument("--cli", default=".local/kaggle-cli-venv/" + ("Scripts/kaggle.exe" if os.name == "nt" else "bin/kaggle"))
    parser.add_argument("--baseline", default="data/outputs/asr-lora-experiment-run2/training/base-validation/metrics.json")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--initial-checkpoint", help="Verified saved checkpoint summary when monitoring evaluation-only recovery")
    parser.add_argument("--report-training-archive", help="Build the final report after complete, verified evaluation")
    parser.add_argument("--report-python", default=sys.executable)
    parser.add_argument("--report-outdir", default="results/asr-lora-run2")
    args = parser.parse_args()
    if args.interval < 60:
        parser.error("Use a monitoring interval of at least 60 seconds")
    folder = Path(args.outdir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    previous = read(folder / "monitor.json")
    if previous.get("pid"):
        try:
            old = psutil.Process(previous["pid"])
            if abs(old.create_time() - previous["created_at"]) < .01 and old.is_running():
                raise RuntimeError("This monitor is already running")
        except psutil.NoSuchProcess:
            pass
    write(folder / "monitor.json", {"pid": os.getpid(), "created_at": psutil.Process().create_time(),
          "kernel": args.kernel, "interval_seconds": args.interval})
    baseline = read(args.baseline)
    initial_checkpoint = read(args.initial_checkpoint) if args.initial_checkpoint else None
    cli = str(Path(args.cli).resolve())
    live = None
    with (folder / "live.log").open("a", encoding="utf-8") as log:
        try:
            while True:
                stale = time.time() - (folder / "live.log").stat().st_mtime > 600
                if stale and live is not None and live.poll() is None:
                    live.terminate()
                    try:
                        live.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        live.kill()
                        live.wait(timeout=15)
                if live is None or live.poll() is not None:
                    live = subprocess.Popen([cli, "kernels", "logs", "--follow", args.kernel],
                                            stdout=log, stderr=subprocess.STDOUT)
                try:
                    result = subprocess.run([cli, "kernels", "status", args.kernel],
                                            capture_output=True, text=True, timeout=90)
                    match = re.search(r"KernelWorkerStatus\.([A-Z_]+)", result.stdout)
                    remote = match.group(1) if result.returncode == 0 and match else "UNKNOWN"
                except (OSError, subprocess.SubprocessError):
                    remote = "UNKNOWN"
                terminal = remote in {"COMPLETE", "ERROR", "CANCELLED", "CANCELED", "CANCEL_ACK", "CANCEL_ACKNOWLEDGED"}
                text = (folder / "live.log").read_text(encoding="utf-8", errors="replace")
                if terminal:
                    try:
                        persisted = subprocess.run([cli, "kernels", "logs", args.kernel],
                            capture_output=True, text=True, timeout=90)
                        if persisted.returncode == 0 and persisted.stdout.strip():
                            payload = json.loads(persisted.stdout)
                            text = "".join(item.get("data", "") for item in payload)
                            (folder / "completed.log").write_text(text, encoding="utf-8")
                    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, AttributeError):
                        pass
                state = {**progress(text, baseline, initial_checkpoint), "remote_status": remote,
                         "checked_at_utc": datetime.now(timezone.utc).isoformat()}
                if remote == "UNKNOWN":
                    state["flags"].append("Remote status unavailable; retrying next check")
                if stale and not terminal:
                    state["flags"].append("Live logs are stale; reconnecting. Current training progress is unconfirmed")
                if terminal:
                    if remote != "COMPLETE":
                        state["flags"].append("Remote session did not complete successfully")
                    try:
                        destination = folder / "downloaded"
                        if shutil.disk_usage(folder).free < 4 * 1024**3:
                            raise ValueError("Less than 4 GiB free; download deferred")
                        subprocess.run([cli, "kernels", "output", args.kernel, "--path", str(destination),
                            "--file-pattern", r"(^|/)resume-state\.(tar|inventory\.json)$", "--page-size", "200", "--quiet"],
                            check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
                        state["download"] = verify_download(destination)
                        state["flags"].extend(state["download"]["quality_flags"])
                        if remote == "COMPLETE" and args.report_training_archive:
                            try:
                                subprocess.run([args.report_python, str(Path(__file__).with_name("build_asr_report.py")),
                                    "--training-archive", args.report_training_archive,
                                    "--evaluation-archive", str(destination / "resume-state.tar"),
                                    "--outdir", args.report_outdir, "--require-complete", "--plots", "--docx"],
                                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                                state["report"] = {"status": "complete", "outdir": args.report_outdir}
                            except (OSError, subprocess.SubprocessError) as exc:
                                state["report"] = {"status": "failed", "error": str(exc)}
                                state["flags"].append("Final report export needs attention; verified evaluation archive is preserved")
                    except (ValueError, OSError, subprocess.SubprocessError) as exc:
                        state["download"] = {"status": "pending_retry", "error": str(exc)}
                        state["flags"].append("Resume archive download/verification needs attention")
                report(folder, state)
                print(json.dumps({key: state[key] for key in ("checked_at_utc", "remote_status", "latest_update", "flags")}), flush=True)
                if terminal and state.get("download", {}).get("status") == "verified":
                    break
                time.sleep(args.interval)
        finally:
            if live is not None and live.poll() is None:
                live.terminate()
                live.wait(timeout=30)


if __name__ == "__main__":
    main()
