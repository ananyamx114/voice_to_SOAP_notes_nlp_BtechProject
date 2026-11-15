"""Build private, hash-verified ASR transfer archives without Git or credentials.

Assets can be prepared while training continues. Checkpoint export requires the
registered job to have stopped at a checkpoint boundary. Original files stay put.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("data/outputs/asr-lora-experiment-run2")
PREPARED = [RUN / "prepared/clinical-train",
            RUN / "prepared/fleurs",
            Path("data/processed/asr-lora-clinical-validation-run1"),
            Path("data/processed/asr-lora-fleurs-test-run1")]
CSV_MANIFESTS = [Path("data/manifests/eka-smoke.csv"),
                 Path("data/processed/betrac-smoke/audio_manifest.csv"),
                 Path("data/manifests/sully-asr-selection.csv"),
                 PREPARED[-1] / "test_manifest.csv"]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def internal(path, root=ROOT):
    path = Path(path).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def manifest_assets(path, root=ROOT):
    """Preserve manifest bytes and relative paths, including original audio."""
    path = internal(path, root)
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    files = {path}
    hashes = {}
    for row in rows:
        for key in ("audio_path", "original_audio_path", "metadata_path"):
            if row.get(key):
                if Path(row[key]).is_absolute():
                    raise ValueError(f"Nonportable absolute {key} in {path}")
                source = internal(path.parent / row[key], root)
                files.add(source)
                expected = row.get("source_sha256" if key == "audio_path" else
                                   "original_audio_sha256" if key == "original_audio_path" else "")
                if expected:
                    if source not in hashes:
                        hashes[source] = digest(source)
                    if hashes[source] != expected:
                        raise ValueError(f"Audio provenance hash changed: {source}")
    return files


def archive(files, target, root=ROOT):
    """Atomic tar, exact inventory and re-read verification; no symlinks."""
    root = Path(root).resolve()
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to replace an existing snapshot: {target}")
    records = []
    partial = target.with_suffix(target.suffix + ".partial")
    with tarfile.open(partial, "w") as output:
        for path in sorted(set(files)):
            path = internal(path, root)
            relative = path.relative_to(root).as_posix()
            before = digest(path)
            output.add(path, arcname=relative, recursive=False)
            if digest(path) != before:
                raise ValueError(f"File changed during snapshot: {relative}")
            records.append({"path": relative, "sha256": before, "bytes": path.stat().st_size})
    with tarfile.open(partial) as saved:
        for record, member in zip(records, saved.getmembers(), strict=True):
            if not member.isfile() or member.name != record["path"]:
                raise ValueError("Archive member mismatch")
            value = hashlib.sha256()
            with saved.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != record["sha256"]:
                raise ValueError("Archive checksum mismatch")
    partial.replace(target)
    inventory = {"archive": target.name, "sha256": digest(target), "files": records,
                 "total_source_bytes": sum(row["bytes"] for row in records)}
    target.with_suffix(".inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return {"archive": str(target), "files": len(records),
            "bytes": target.stat().st_size, "sha256": inventory["sha256"]}


def assets():
    files = set()
    for folder in ("src", "scripts", "configs", "docs", "tests"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
                if path.suffix in {".py", ".yaml", ".md", ".json"} and ".local." not in path.name:
                    files.add(path)
    files.update(ROOT.glob("requirements*.txt"))
    files = {p for p in files if ".local." not in p.name}
    files.update(ROOT.glob("LICENSE*"))
    files.update(ROOT / name for name in ("pyproject.toml", "README.md", "AGENTS.md"))
    for path in (RUN / "data/train.jsonl", RUN / "data/validation.jsonl"):
        files.update(manifest_assets(ROOT / path))
    for folder in PREPARED:
        for path in (ROOT / folder).glob("*.json*"):
            if path.suffix == ".jsonl":
                files.update(manifest_assets(path))
            else:
                files.add(path)
    for path in CSV_MANIFESTS:
        files.update(manifest_assets(ROOT / path))
    files.update((ROOT / "data/outputs/vad-component-01").glob("*/speech_segments.json"))
    for pattern in ("*.json",):
        files.update((ROOT / RUN).glob(pattern))
    return {p for p in files if ".local." not in p.name}


def checkpoint_assets():
    # Import process inspection only for the explicitly stopped snapshot action.
    from asr_job import active_owner, read
    job = read(ROOT / ".local/asr-expanded-run2.job.json")
    if not job or active_owner(job) is not None:
        raise ValueError("Pause the registered local job and wait until it is stopped first")
    training = ROOT / RUN / "training"
    metadata = json.loads((training / "run.json").read_text())
    if metadata.get("status") not in {"paused", "complete"}:
        raise ValueError("Checkpoint snapshot requires paused or completed training")
    if not (training / "checkpoint.pt").is_file():
        raise ValueError("No saved optimizer checkpoint")
    return {p for p in training.rglob("*") if p.is_file() and not p.name.endswith(".partial")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("assets", "checkpoint"))
    parser.add_argument("--outdir", default=".local/kaggle-asr/transfer")
    args = parser.parse_args()
    files = assets() if args.kind == "assets" else checkpoint_assets()
    print(json.dumps(archive(files, Path(args.outdir) / f"{args.kind}.tar"), indent=2))


if __name__ == "__main__":
    main()
