"""Portable launcher: use python3 on macOS/Linux or py on Windows."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def environment_python(root: Path, windows: bool | None = None) -> Path:
    windows = os.name == "nt" if windows is None else windows
    return root / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def command_for(action: str, root: Path, extra: list[str], windows: bool | None = None) -> list[str]:
    python = str(environment_python(root, windows))
    if action == "hf-login":
        executable = Path(python).parent / ("hf.exe" if python.endswith(".exe") else "hf")
        return [str(executable), "auth", "login", *extra]
    if action == "test":
        return [python, "-m", "unittest", "discover", "-s", "tests", "-v", *extra]
    if action == "cli":
        return [python, "-m", "voice_to_soap.cli", *extra]
    if action == "asr-run":
        return [python, "-m", "voice_to_soap.asr_component", *extra]
    script = {"system": "check_system.py", "check": "check_setup.py",
              "hf-check": "check_huggingface.py", "download-data": "download_benchmarks.py",
              "download-asr": "download_asr.py", "prepare-asr-data": "prepare_asr_inputs.py"}[action]
    return [python, str(root / "scripts" / script), *extra]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "system", "check", "test", "hf-login",
                                           "hf-check", "download-data", "download-asr", "prepare-asr-data", "asr-run", "cli"))
    # Parse only the action here; flags such as cli --help belong to the child.
    args = parser.parse_args(sys.argv[1:2])
    extra = sys.argv[2:]
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(root / "data/models/huggingface"))
    env["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    python = environment_python(root)
    if args.action == "setup":
        setup_parser = argparse.ArgumentParser(prog=f"{parser.prog} setup")
        setup_parser.add_argument("--profile", choices=("dev", "asr", "vad"), default="dev")
        setup = setup_parser.parse_args(extra)
        if sys.version_info < (3, 10):
            parser.error("Python >=3.10 required; Python 3.12 is the tested starting point")
        if not python.exists():
            if (root / ".venv").exists():
                parser.error("Existing .venv has no interpreter for this OS; preserve it and recreate locally")
            subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")],
                           cwd=root, env=env, check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-e", ".[dev,hub]"],
                       cwd=root, env=env, check=True)
        if setup.profile in {"asr", "vad"}:
            requirements = {"asr": "requirements-local-asr.txt", "vad": "requirements-vad.txt"}
            subprocess.run([str(python), "-m", "pip", "install", "-r", requirements[setup.profile]],
                           cwd=root, env=env, check=True)
        subprocess.run([str(python), str(root / "scripts/check_system.py")],
                       cwd=root, env=env, check=True)
        return 0
    if args.action == "system" and not python.exists():
        return subprocess.run([sys.executable, str(root / "scripts/check_system.py"), *extra],
                              cwd=root, env=env).returncode
    if not python.exists():
        parser.error("Create this computer's environment first: python scripts/project.py setup")
    return subprocess.run(command_for(args.action, root, extra), cwd=root, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
