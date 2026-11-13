"""Download only the pinned full Whisper checkpoint and processor files."""
import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/asr-large-v3.yaml")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / "data/models/huggingface"))
    from huggingface_hub import HfApi, snapshot_download
    from voice_to_soap.config import load_config
    audio = load_config(args.config)["audio"]
    repo, revision = audio["asr_model_id"], audio["asr_options"]["revision"]
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    weight = next(f for f in info.siblings if f.rfilename == "model.safetensors")
    cache = Path(os.environ["HF_HOME"])
    cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(cache).free < weight.size + 8 * 1024**3:
        raise RuntimeError("Insufficient space to download checkpoint while preserving 8 GiB free")
    print(f"Full checkpoint {repo}@{revision}: {weight.size} weight bytes", flush=True)
    folder = snapshot_download(repo, revision=revision, allow_patterns=["model.safetensors", "*.json", "merges.txt", "README.md", "LICENSE*"], max_workers=2)
    print(json.dumps({"repository": repo, "revision": revision,
                      "weight_bytes": (Path(folder) / "model.safetensors").stat().st_size,
                      "quantization": "none", "download_complete": True}, indent=2))


if __name__ == "__main__":
    main()
