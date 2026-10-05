"""
shared/exporter/to_onnx.py
---------------------------
Ekspor policy PPO ke ONNX, DENGAN NORMALISASI OBSERVASI dibakar ke graph.

Dipakai OTOMATIS oleh train_missionN.py di akhir training.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pickle
import re

import numpy as np
import torch
import torch.nn as nn

from shared.envs.obs_layout import OBS_DIM


class NormalizedOnnxablePolicy(nn.Module):
    """MlpExtractor + action_net SB3, dengan normalisasi observasi fixed
    di paling depan. Return mean action (deterministic)."""

    def __init__(self, policy, obs_mean, obs_var, clip_obs, epsilon):
        super().__init__()
        self.register_buffer("obs_mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_std",
                             torch.sqrt(torch.as_tensor(obs_var, dtype=torch.float32) + epsilon))
        self.clip_obs = float(clip_obs)
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        normed = (raw_observation - self.obs_mean) / self.obs_std
        normed = torch.clamp(normed, -self.clip_obs, self.clip_obs)
        features = self.mlp_extractor.forward_actor(normed)
        mean_actions = self.action_net(features)
        return torch.clamp(mean_actions, -1.0, 1.0)


def save_vecnormalize_stats(vecnormalize, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs_mean=vecnormalize.obs_rms.mean,
        obs_var=vecnormalize.obs_rms.var,
        clip_obs=np.array(vecnormalize.clip_obs),
        epsilon=np.array(vecnormalize.epsilon),
    )


def load_vecnormalize_stats(path: str):
    data = np.load(path)
    return data["obs_mean"], data["obs_var"], float(data["clip_obs"]), float(data["epsilon"])


def vecnormalize_stats_from_pkl(path: str):
    """Baca statistik dari file VecNormalize (.pkl) TANPA perlu membuat env."""
    with open(path, "rb") as f:
        vn = pickle.load(f)
    return (np.array(vn.obs_rms.mean), np.array(vn.obs_rms.var),
            float(vn.clip_obs), float(vn.epsilon))


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


def write_onnx_metadata(onnx_path: str, meta: dict):
    Path(onnx_path).with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))


def build_normalized_policy(model, obs_mean, obs_var, clip_obs, epsilon) -> NormalizedOnnxablePolicy:
    return NormalizedOnnxablePolicy(model.policy, obs_mean, obs_var, clip_obs, epsilon).eval()


def export_policy_to_onnx(model, obs_mean, obs_var, clip_obs, epsilon,
                           output_path: str, obs_dim: int = OBS_DIM, opset: int = 17):
    onnxable = build_normalized_policy(model, obs_mean, obs_var, clip_obs, epsilon)
    dummy_input = torch.zeros(1, obs_dim)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        onnxable, dummy_input, str(output_path),
        input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return onnxable


def main():
    parser = argparse.ArgumentParser(description="Re-export checkpoint SB3 ke ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vecnormalize", type=str, required=True, help="file VecNormalize .pkl")
    parser.add_argument("--output", type=str, required=True,
                         help="mis. models/onnx_mission1_v1.onnx")
    parser.add_argument("--obs_dim", type=int, default=OBS_DIM)
    args = parser.parse_args()

    from stable_baselines3 import PPO
    model = PPO.load(args.checkpoint, device="cpu")
    obs_mean, obs_var, clip_obs, epsilon = vecnormalize_stats_from_pkl(args.vecnormalize)
    export_policy_to_onnx(model, obs_mean, obs_var, clip_obs, epsilon,
                           args.output, args.obs_dim)
    print(f"Model diekspor ke: {args.output}")


if __name__ == "__main__":
    main()