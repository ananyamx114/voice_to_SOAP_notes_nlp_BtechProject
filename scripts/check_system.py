"""Inspect this computer; save local hardware facts without model downloads."""
import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def physical_memory(system: str | None = None) -> int | None:
    system = system or platform.system()
    try:
        if system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        "physical", "available", "pagefile", "available_pagefile",
                        "virtual", "available_virtual", "extended"
                    )]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.physical
        elif system == "Darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], check=True,
                                    capture_output=True, text=True, timeout=5)
            return int(result.stdout.strip())
        else:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def collect_system(root: Path) -> dict:
    usage = shutil.disk_usage(root)
    report = {"os": platform.system(), "os_release": platform.release(),
              "architecture": platform.machine(), "python": platform.python_version(),
              "ram_bytes": physical_memory(), "disk_free_bytes": usage.free,
              "disk_total_bytes": usage.total, "ffmpeg_available": bool(shutil.which("ffmpeg")),
              "ollama_available": bool(shutil.which("ollama")), "accelerators": [],
              "model_fit": "Not established: measure each selected model and context."}
    try:
        if importlib.util.find_spec("torch") is None:
            report["torch_status"] = "not installed; accelerator availability not checked"
            return report
        import torch
        report["torch_status"] = torch.__version__
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                report["accelerators"].append({"type": "cuda", "index": index,
                    "name": properties.name, "memory_bytes": properties.total_memory})
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            report["accelerators"].append({"type": "mps", "memory": "shared system RAM"})
    except Exception as exc:
        report["torch_status"] = f"Unable to inspect accelerators ({type(exc).__name__})"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".local/system.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = collect_system(root)
    path = Path(args.output)
    if not path.is_absolute():
        path = root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Hardware report saved locally; no model downloaded or executed.")


if __name__ == "__main__":
    main()
