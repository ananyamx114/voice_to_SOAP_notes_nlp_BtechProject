from pathlib import Path
from typing import Any
import os
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")
    return config


def env_value(config: dict[str, Any], key: str, fallback: str | None = None) -> str | None:
    name = config.get(key)
    return os.environ.get(name, fallback) if name else fallback


def huggingface_token(token_env: str = "HF_TOKEN") -> str | None:
    """Use an explicit token first, then the Hugging Face CLI's saved login."""
    explicit = os.environ.get(token_env)
    if explicit:
        return explicit
    try:
        from huggingface_hub import get_token
    except ImportError:
        return None
    return get_token()
