"""Check local prerequisites without model downloads or credential disclosure."""
from importlib.util import find_spec
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-llm", action="store_true",
                        help="Query the configured local server's model list (no generation)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / "data/models/huggingface"))
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"Virtual environment: {sys.prefix != sys.base_prefix}")
    missing = []
    for name in ("voice_to_soap", "openai", "yaml", "numpy", "scipy", "soundfile",
                 "pyarrow", "jiwer", "rouge_score", "pytest"):
        available = find_spec(name) is not None
        print(f"{'OK' if available else 'MISSING'}: {name}")
        if not available:
            missing.append(name)
    for executable in ("ffmpeg", "ollama"):
        print(f"{executable}: {'available' if shutil.which(executable) else 'not on PATH'}")
    for name in ("torch", "transformers", "silero_vad", "pyannote", "peft"):
        print(f"Optional {name}: {'installed' if find_spec(name) else 'not installed'}")
    from voice_to_soap.config import huggingface_token
    print(f"HF authentication: {'available (environment or saved login)' if huggingface_token() else 'not configured'}")
    if args.check_llm:
        import yaml
        config = yaml.safe_load((root / "configs/base.yaml").read_text())["llm"]
        base = os.environ.get(config["base_url_env"], config["default_base_url"])
        parsed = urlparse(base)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {
            "localhost", "127.0.0.1", "::1"
        } or parsed.username or parsed.password:
            print("LLM check skipped: this probe accepts only local HTTP(S) endpoints.")
            return 1
        model = os.environ.get(config["model_env"], config["default_model"])
        key = os.environ.get(config["api_key_env"], "ollama")
        request = Request(base.rstrip("/") + "/models",
                          headers={"Authorization": f"Bearer {key}"})
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.load(response)
            ids = {item["id"] for item in payload["data"]}
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError):
            print("LLM: unavailable or invalid model-list response; no inference tested.")
            return 1
        if model not in ids:
            print("LLM: reachable, but the configured model is not listed.")
            return 1
        print("LLM: configured model listed; inference and output quality remain untested.")
    else:
        print("LLM: not probed (use --check-llm).")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
