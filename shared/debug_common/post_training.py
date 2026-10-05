"""
shared/debug_common/post_training.py
------------------------------------
Debug pasca-training yang sama untuk semua misi. Dipanggil dari train_missionN.py
lewat hook `post_debug`:

  1. kurva training dari TensorBoard            -> training_curves.png
  2. ONNX vs checkpoint pada rollout nyata      -> onnx_vs_checkpoint.png
  3. replay episode dgn policy ONNX             -> replay_*.png
  4. robustness terhadap domain randomization   -> robustness.png
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from stable_baselines3 import PPO

from shared.debug_common.common import (make_predict_fn_onnx, make_predict_fn_sb3,
                                        plot_onnx_vs_checkpoint_diff, plot_randomization_robustness,
                                        plot_replay_trajectory, plot_training_curves,
                                        run_policy_episode)
from shared.exporter import vecnormalize_stats_from_pkl


def onnx_session(path: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def evaluate_onnx(env, onnx_path: str, n_episodes: int, seed0: int = 500,
                  wall_clock_timeout_sec: float = 60.0):
    """Jalankan n episode dgn policy ONNX. Return (list[EpisodeResult], list[record dict])."""
    fn = make_predict_fn_onnx(onnx_session(onnx_path))
    results, records = [], []
    for i in range(n_episodes):
        obs, info0 = env.reset(seed=seed0 + i)
        res = run_policy_episode(env, fn, obs, env.max_steps, wall_clock_timeout_sec)
        results.append(res)
        records.append({"seed": seed0 + i, "success": res.termination_reason == "success",
                        "reason": res.termination_reason, "steps": res.n_steps,
                        "final_distance": float(res.info.get("final_distance", np.nan)),
                        "payload_mass_extra": float(info0.get("payload_mass_extra", 0.0)),
                        "n_obstacles": int(info0.get("n_obstacles", 0)),
                        "cross_track_mean": float(res.info.get("cross_track_mean", np.nan)),
                        "cross_track_max": float(res.info.get("cross_track_max", np.nan)),
                        "min_clearance": float(res.clearance.min()) if res.clearance.size else np.nan})
    return results, records


def onnx_vs_checkpoint(env, onnx_path: str, out_path: str, n_steps: int = 300, log=print):
    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text())   # sumber ONNX
    model = PPO.load(meta["source_model"], device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(meta["vecnormalize"])
    f_sb3 = make_predict_fn_sb3(model, mean, var, clip, eps)
    f_onnx = make_predict_fn_onnx(onnx_session(onnx_path))
    obs, _ = env.reset(seed=4242)
    diffs = []
    for _ in range(n_steps):
        a1, a2 = np.clip(f_sb3(obs), -1, 1), f_onnx(obs)
        diffs.append(float(np.max(np.abs(a1 - a2))))
        obs, _, term, trunc, _ = env.step(a2)
        if term or trunc:
            obs, _ = env.reset()
    diffs = np.array(diffs)
    plot_onnx_vs_checkpoint_diff(diffs, out_path)
    log(f"[debug] ONNX vs checkpoint (rollout {n_steps} step): max={diffs.max():.2e}")
    return diffs


def robustness_study(env_factory, onnx_path: str, out_path: str, n_episodes: int = 20,
                     seed0: int = 9000, log=print):
    """Domain randomization AKTIF (payload, gravitasi, noise awal) tanpa noise sensor."""
    env = env_factory(training=False, randomize=True, seed=seed0)
    _, records = evaluate_onnx(env, onnx_path, n_episodes, seed0)
    plot_randomization_robustness(records, out_path)
    sr = np.mean([r["success"] for r in records])
    log(f"[debug] robustness (DR aktif): success {sr * 100:.0f}% dari {n_episodes} episode "
        f"{dict(Counter(r['reason'] for r in records))}")
    return records


def generic_post_training(ctx, result: dict, out_dir: str | Path, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        plot_training_curves(result["tb_logdir"], str(out / "training_curves.png"),
                             title=f"Misi {ctx.mission_id} - training")
        log(f"[debug] kurva training -> {out / 'training_curves.png'}")
    except Exception as e:
        log(f"[debug] kurva training dilewati: {type(e).__name__}: {e}")
    onnx_vs_checkpoint(ctx.make_raw_env(training=True, randomize=True, seed=4242),
                       result["onnx_path"], str(out / "onnx_vs_checkpoint.png"), log=log)
    dbg = ctx.cfg.get("debug", {})
    robustness_study(ctx.make_raw_env, result["onnx_path"], str(out / "robustness.png"),
                     int(dbg.get("robustness_episodes", 20)), log=log)
