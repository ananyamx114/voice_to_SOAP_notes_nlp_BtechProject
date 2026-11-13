"""Download the initial public audio benchmarks, pinned and SHA-256 checked.

Uses httpx already installed with the project's local LLM client. No login or
remote dataset code execution is needed for the selected public repositories.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import shutil
import time

import httpx


SOURCES = {
    "betrac": ("BeTraC/betrac-2026", ["README.md", "data/dev-00000.tar"]),
    "eka": ("ekacare/eka-medical-asr-evaluation-dataset",
            ["README.md", "dataset_info.json", "en/test-*.parquet", "hi/test-*.parquet"]),
    "sully-smoke": ("SullyHQ/open-clinical-conversations",
                    ["README.md", "ATTRIBUTION.md", "CITATIONS.bib", "encounters.jsonl",
                     "audio/metadata.jsonl", "audio/pm57-d1c01.mp3"]),
}

REVISIONS = {
    "betrac": "96f7b94e662dac0d1394cfdf55e8b8ae81422663",
    "eka": "433e66037c8aebb8f396b938624a5ab6afc4ca36",
    "sully-smoke": "06053cd9be2ea5fbfaefebb1db5be907b53a8c44",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def download(name: str, root: Path, client: httpx.Client) -> None:
    repo, patterns = SOURCES[name]
    response = client.get(f"https://huggingface.co/api/datasets/{repo}/revision/{REVISIONS[name]}",
                          params={"blobs": "true"})
    response.raise_for_status()
    metadata = response.json()
    if metadata.get("gated"):
        raise RuntimeError("Dataset now requires access approval; review its current terms.")
    revision = metadata["sha"]
    if revision != REVISIONS[name]:
        raise RuntimeError("Upstream revision does not match the pinned benchmark revision")
    files = [item for item in metadata["siblings"]
             if any(fnmatch.fnmatchcase(item["rfilename"], p) for p in patterns)]
    if any(not any(fnmatch.fnmatchcase(f["rfilename"], p) for f in files) for p in patterns):
        raise RuntimeError("Expected dataset files are absent; audit the upstream layout.")
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    total = sum(item["size"] for item in files)
    if total > 2 * 1024**3:
        raise RuntimeError("Selected files exceed the initial 2 GiB per-dataset budget.")
    if shutil.disk_usage(destination).free < total + 12 * 1024**3:
        raise RuntimeError("Insufficient free space to preserve 12 GiB for models/system use.")
    print(f"{name}: {len(files)} files, {total / 1024**2:.1f} MiB, revision {revision}", flush=True)
    records = []
    for item in files:
        relative = Path(item["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("Unsafe upstream filename")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get("lfs", {}).get("sha256")
        valid = (target.is_file() and target.stat().st_size == item["size"]
                 and expected is not None and digest(target) == expected)
        if not valid:
            partial = target.with_name(target.name + ".part")
            for attempt in range(3):
                try:
                    with client.stream("GET", f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative.as_posix()}") as stream:
                        stream.raise_for_status()
                        with partial.open("wb") as output:
                            for block in stream.iter_bytes(1024 * 1024):
                                output.write(block)
                    if partial.stat().st_size != item["size"]:
                        raise RuntimeError("Downloaded size does not match upstream metadata")
                    if expected and digest(partial) != expected:
                        raise RuntimeError("Downloaded SHA-256 does not match upstream metadata")
                    partial.replace(target)
                    break
                except (httpx.HTTPError, RuntimeError):
                    if attempt == 2:
                        raise
                    time.sleep(1)
        records.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                        "sha256": digest(target), "upstream_sha256": expected})
        print(f"  checked {relative.as_posix()}", flush=True)
    manifest = {"repository": repo, "revision": revision,
                "license_from_card": metadata.get("cardData", {}).get("license"),
                "files": records, "total_bytes": total}
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*SOURCES, "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "data/raw"
    with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90, connect=20)) as client:
        for name in SOURCES if args.dataset == "all" else [args.dataset]:
            download(name, root, client)


if __name__ == "__main__":
    main()
