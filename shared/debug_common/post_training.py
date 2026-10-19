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

from shared.debug_common.common import (make_predict_fn_onnx, make_predict_fn_sb3, make_predict_fn_scripted,
                                        plot_onnx_vs_checkpoint_diff, plot_randomization_robustness,
                                        plot_replay_trajectory, plot_training_curves,
                                        run_policy_episode)


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
                        "gravity_scale": float(info0.get("gravity_scale", np.nan)),
                        "n_obstacles": int(info0.get("n_obstacles", 0)),
                        "cross_track_mean": float(res.info.get("cross_track_mean", np.nan)),
                        "cross_track_max": float(res.info.get("cross_track_max", np.nan)),
                        "min_clearance": float(res.clearance.min()) if res.clearance.size else np.nan})
    return results, records


def onnx_vs_checkpoint(env, onnx_path: str, out_path: str, n_steps: int = 300, log=print):
    from stable_baselines3 import PPO                        # lazy: evaluasi tidak membutuhkan SB3/torch
    from shared.exporter import vecnormalize_stats_from_pkl
    meta = json.loads(Path(onnx_path).with_suffix(".json").read_text(encoding="utf-8"))   # sumber ONNX
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


def _phase_a_deviation(res) -> float:
    """Penyimpangan maksimum lintasan fase A dari garis lurus titik-awal -> T1 (m)."""
    tr = res.trajectory
    if res.T1 is None or len(tr) < 2:
        return float("nan")
    k = len(tr)
    if res.phase.size and (res.phase == 1).any():
        k = int(np.argmax(res.phase == 1)) + 1
    a, b = tr[0], res.T1
    ab = b - a
    t = np.clip((tr[:k] - a) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.max(np.linalg.norm(tr[:k] - (a + t[:, None] * ab), axis=1)))


def path_quality_study(env_factory, onnx_path: str, out_path: str, n_episodes: int = 30,
                       seed0: int = 7000, log=print) -> dict:
    """Kualitas lintasan policy vs controller referensi garis lurus pada skenario yang SAMA.

    Metrik per episode: rasio panjang jalur fase A (1 = lurus), penyimpangan maksimum dari garis lurus
    awal->T1, total putaran gripper, dan rata-rata perubahan aksi per langkah (kehalusan).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = env_factory(training=False, randomize=False, seed=seed0)
    rl_fn = make_predict_fn_onnx(onnx_session(onnx_path))
    rows = {"policy": [], "referensi lurus": []}
    for i in range(n_episodes):
        for name in rows:
            obs, _ = env.reset(seed=seed0 + i)
            fn = rl_fn if name == "policy" else make_predict_fn_scripted(env)
            res = run_policy_episode(env, fn, obs, env.max_steps)
            rows[name].append({"success": res.termination_reason == "success",
                               "path_ratio_A": float(res.info.get("path_ratio_A", np.nan)),
                               "dev_mm": 1000.0 * _phase_a_deviation(res),
                               "rot_travel_deg": float(res.info.get("rot_travel_deg", np.nan)),
                               "action_jerk": float(res.info.get("action_jerk", np.nan))})
    metrics = [("path_ratio_A", "Rasio panjang jalur fase A (1 = lurus)"),
               ("dev_mm", "Penyimpangan maks. dari garis lurus awal-T1 (mm)"),
               ("rot_travel_deg", "Total putaran gripper / episode (deg)"),
               ("action_jerk", "Perubahan aksi / langkah  mean||a_t - a_(t-1)||^2")]
    summary = {}
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))
    for ax, (k, lab) in zip(axes, metrics):
        data, names = [], []
        for name, rr in rows.items():
            v = np.array([r[k] for r in rr], dtype=float)
            v = v[np.isfinite(v)]
            summary.setdefault(name, {})[k] = float(np.median(v)) if v.size else float("nan")
            data.append(v if v.size else np.array([np.nan]))
            names.append(f"{name}\nmedian {np.median(v):.3g}" if v.size else f"{name}\n(tidak ada data)")
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(lab, fontsize=10)
        ax.grid(alpha=0.3)
    for name, rr in rows.items():
        summary[name]["success"] = float(np.mean([r["success"] for r in rr]))
    fig.suptitle(f"Kualitas lintasan ({n_episodes} skenario sama): sukses policy "
                 f"{summary['policy']['success'] * 100:.0f}%, referensi {summary['referensi lurus']['success'] * 100:.0f}%",
                 fontsize=11)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    p = summary["policy"]
    ratio = (f"{p['path_ratio_A']:.2f}" if np.isfinite(p["path_ratio_A"])
             else "- (belum ada episode yang mencapai T1)")
    log(f"[debug] kualitas lintasan: sukses {p['success'] * 100:.0f}% | rasio jalur {ratio} | penyimpangan "
        f"{p['dev_mm']:.0f} mm | putaran gripper {p['rot_travel_deg']:.0f} deg | perubahan aksi "
        f"{p['action_jerk']:.3f} -> {out_path}")
    return summary


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
    try:
        path_quality_study(ctx.make_raw_env, result["onnx_path"], str(out / "path_quality.png"),
                           int(dbg.get("path_quality_episodes", 30)), log=log)
    except Exception as e:                                          # debug tidak boleh menggagalkan hasil
        log(f"[debug] kualitas lintasan dilewati: {type(e).__name__}: {e}")
