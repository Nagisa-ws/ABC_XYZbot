"""shared/config_utils.py -- loader YAML + helper path proyek."""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def resolve_path(path) -> Path:
    """Path relatif diartikan relatif terhadap root proyek."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def deep_update(base: dict, override: dict) -> dict:
    """Merge rekursif (tidak memodifikasi argumen)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
