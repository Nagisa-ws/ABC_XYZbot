"""
shared/exporter/versioning.py
-----------------------------
Penomoran versi berkas ONNX (models/<prefix>_v<N>.onnx). Sengaja BEBAS torch/stable-baselines3 supaya
evaluasi (yang hanya membutuhkan onnxruntime) tidak ikut memuat keduanya.
"""
from __future__ import annotations

import re
from pathlib import Path


def next_versioned_onnx_path(models_dir: str, prefix: str) -> Path:
    """models/<prefix>_v<N>.onnx dengan N = versi terbesar yang ada + 1 (mulai dari 1)."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.onnx$")
    vers = [int(m.group(1)) for f in d.iterdir() if (m := pat.match(f.name))]
    return d / f"{prefix}_v{(max(vers) + 1) if vers else 1}.onnx"


def latest_versioned_onnx_path(models_dir: str, prefix: str) -> Path | None:
    d = Path(models_dir)
    pat = re.compile(rf"^{re.escape(prefix)}_v(\d+)\.onnx$")
    found = [(int(m.group(1)), f) for f in d.iterdir() if (m := pat.match(f.name))] if d.exists() else []
    return max(found)[1] if found else None
