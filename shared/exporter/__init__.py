# shared/exporter/__init__.py
# Fungsi versi ONNX diimpor langsung (tanpa torch). Sisanya dimuat LAZY (PEP 562): `import shared.exporter` tidak lagi
# memuat torch, jadi evaluasi tidak menyentuh torch/stable-baselines3. `from shared.exporter import X` tetap berfungsi.
from shared.exporter.versioning import latest_versioned_onnx_path, next_versioned_onnx_path

_LAZY = {
    "export_policy_to_onnx": "shared.exporter.to_onnx",
    "build_normalized_policy": "shared.exporter.to_onnx",
    "save_vecnormalize_stats": "shared.exporter.to_onnx",
    "load_vecnormalize_stats": "shared.exporter.to_onnx",
    "vecnormalize_stats_from_pkl": "shared.exporter.to_onnx",
    "write_onnx_metadata": "shared.exporter.to_onnx",
    "NormalizedOnnxablePolicy": "shared.exporter.to_onnx",
    "verify_onnx": "shared.exporter.verify_onnx",
}

__all__ = ["next_versioned_onnx_path", "latest_versioned_onnx_path", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        import importlib
        value = getattr(importlib.import_module(_LAZY[name]), name)
        globals()[name] = value          # mengatasi bentrok nama submodul `verify_onnx` vs fungsi `verify_onnx`
        return value
    raise AttributeError(f"module 'shared.exporter' has no attribute {name!r}")
