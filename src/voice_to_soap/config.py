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
