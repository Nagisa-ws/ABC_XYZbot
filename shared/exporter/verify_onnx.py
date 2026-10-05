"""
shared/exporter/verify_onnx.py
------------------------------
Bandingkan output checkpoint SB3 vs ONNX pada observasi MENTAH acak.
Dipakai OTOMATIS oleh train_missionN.py setelah ekspor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import onnxruntime as ort

from shared.envs.obs_layout import OBS_DIM
from shared.exporter.to_onnx import build_normalized_policy, vecnormalize_stats_from_pkl


def verify_onnx(model, obs_mean, obs_var, clip_obs, epsilon, onnx_path: str,
                 obs_dim: int = OBS_DIM, n_samples: int = 20, tol: float = 1e-4, seed: int = 0,
                 obs_batch: np.ndarray | None = None):
    """Bandingkan output PyTorch (normalisasi + policy) vs ONNX pada observasi MENTAH.

    `obs_batch` sebaiknya observasi NYATA dari rollout env (agar dimensi aktif dan
    inaktif punya distribusi realistis); bila None dipakai derau acak."""
    onnxable = build_normalized_policy(model, obs_mean, obs_var, clip_obs, epsilon)
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    if obs_batch is None:
        rng = np.random.default_rng(seed)
        obs_batch = rng.normal(size=(n_samples, obs_dim)).astype(np.float32)
    obs_batch = np.asarray(obs_batch, dtype=np.float32)

    with torch.no_grad():
        torch_out = onnxable(torch.from_numpy(obs_batch)).numpy()
    onnx_out = ort_session.run(["action"], {"observation": obs_batch})[0]

    max_abs_diff = float(np.max(np.abs(torch_out - onnx_out)))
    mean_abs_diff = float(np.mean(np.abs(torch_out - onnx_out)))
    passed = max_abs_diff <= tol
    return max_abs_diff, mean_abs_diff, passed


def main():
    parser = argparse.ArgumentParser(description="Verifikasi ONNX vs checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, required=True, help="file VecNormalize .pkl")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--obs_dim", type=int, default=OBS_DIM)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, device="cpu")
    obs_mean, obs_var, clip_obs, epsilon = vecnormalize_stats_from_pkl(args.vecnormalize)

    max_diff, mean_diff, passed = verify_onnx(
        model, obs_mean, obs_var, clip_obs, epsilon, args.onnx, args.obs_dim,
        args.n_samples, args.tol,
    )
    print(f"Max abs diff : {max_diff:.8f}")
    print(f"Mean abs diff: {mean_diff:.8f}")
    if passed:
        print(f"VERIFIKASI SUKSES (toleransi {args.tol}).")
    else:
        print(f"PERINGATAN: perbedaan melebihi toleransi ({args.tol}).")
        sys.exit(1)


if __name__ == "__main__":
    main()