from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_overrides(cfg: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merge_overrides(cfg[k], v)
        else:
            cfg[k] = v
    return cfg
