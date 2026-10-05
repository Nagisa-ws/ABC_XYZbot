"""
Mission2/debug/plot_cross_track.py
----------------------------------
Debug Misi 2 (dipanggil otomatis oleh train_mission2.py):
  * cross_track.png : cross-track error fase B per step, histogram mean/max per episode
  * replay_*.png    : replay 3D (T1, T2, garis, lintasan EE)

Standalone:  python Mission2/debug/plot_cross_track.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.debug_common import evaluate_onnx, plot_replay_trajectory


def run(env_factory, onnx_path: str, out_dir: str, n_episodes: int = 15, n_replay: int = 3, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = env_factory(training=False, randomize=False, seed=321)
    results, rec = evaluate_onnx(env, onnx_path, n_episodes, seed0=700)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    means, maxs = [], []
    for r, rc in zip(results, rec):
        if r.phase.size == 0 or not (r.phase == 1).any():
            continue
        b = r.phase == 1
        ct = 1000 * r.cross_track[b]
        ax[0].plot(ct, color="tab:green" if rc["success"] else "tab:red", alpha=0.6)
        means.append(ct.mean()); maxs.append(ct.max())
    ax[0].set_title("Cross-track error fase B (mm) per step"); ax[0].set_xlabel("step fase B")
    ax[0].grid(alpha=0.3)
    if means:
        ax[1].hist(means, bins=10, color="tab:blue"); ax[1].set_title("Cross-track rata-rata / episode (mm)")
        ax[2].hist(maxs, bins=10, color="tab:orange"); ax[2].set_title("Cross-track maksimum / episode (mm)")
    plt.tight_layout()
    fig.savefig(out / "cross_track.png", dpi=110)
    plt.close(fig)
    for i, r in enumerate(results[:n_replay]):
        plot_replay_trajectory(r, str(out / f"replay_{i}.png"), title=f"Misi 2 replay {i}")
    sr = np.mean([x["success"] for x in rec])
    log(f"[debug-M2] success {sr * 100:.0f}% | cross-track rata-rata "
        f"{np.mean(means) if means else float('nan'):.1f} mm | plot -> {out}")
    return {"success_rate": float(sr), "cross_track_mean_mm": float(np.mean(means)) if means else None}


def main():
    from shared.config_utils import load_yaml
    from shared.evaluation.viewer_eval import resolve_onnx
    from Mission2.eval_envs import Stage2PathEvalEnv

    pcfg = load_yaml(_ROOT / "Mission2" / "configs" / "eval_config.yaml")["paths"]
    onnx = resolve_onnx(_ROOT, pcfg)
    factory = lambda training=False, randomize=False, seed=0: Stage2PathEvalEnv(  # noqa: E731
        model_path=str(_ROOT / pcfg["robot_model_path"]), seed=seed, randomize=randomize)
    run(factory, str(onnx), str(_ROOT / pcfg["debug_dir"]))


if __name__ == "__main__":
    main()
