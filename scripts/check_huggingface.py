"""Verify saved HF authentication and selected model file access without inference."""
import argparse
import json
import os
from pathlib import Path


def check_models() -> dict:
    root = Path(__file__).resolve().parents[1]
    # Set before importing huggingface_hub: it resolves cache locations at import.
    os.environ.setdefault("HF_HOME", str(root / "data/models/huggingface"))
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
    from voice_to_soap.config import huggingface_token

    token = huggingface_token()
    report = {"authenticated": False, "models": [],
              "scope": "Small config downloads and weight metadata; no model inference"}
    if not token:
        report["error"] = "No token found. Set HF_HOME and run .venv/bin/hf auth login."
        return report
    api = HfApi(token=token)
    try:
        api.whoami()
        report["authenticated"] = True
    except Exception as exc:
        report["error"] = type(exc).__name__
        return report
    for repo in ("openai/whisper-small", "pyannote/speaker-diarization-community-1",
                 "Qwen/Qwen2.5-3B-Instruct"):
        item = {"repository": repo, "accessible": False}
        try:
            info = api.model_info(repo)
            files = [entry.rfilename for entry in info.siblings]
            config = "config.yaml" if "config.yaml" in files else "config.json"
            hf_hub_download(repo, config, revision=info.sha, token=token)
            weights = [name for name in files
                       if name.endswith((".safetensors", "pytorch_model.bin"))]
            if not weights:
                raise ValueError("No recognized weight file")
            for name in weights:
                get_hf_file_metadata(hf_hub_url(repo, name, revision=info.sha), token=token)
            item.update(accessible=True, revision=info.sha, weight_files_checked=len(weights))
        except Exception as exc:
            # Do not serialize exception messages: these can include URLs/headers.
            item["error"] = type(exc).__name__
            item["http_status"] = getattr(getattr(exc, "response", None), "status_code", None)
        report["models"].append(item)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optionally save the credential-free JSON result")
    args = parser.parse_args()
    report = check_models()
    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    return 0 if report["authenticated"] and all(
        item["accessible"] for item in report["models"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
