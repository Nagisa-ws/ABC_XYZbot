"""
Mission3/debug/plot_obstacle_clearance.py
-----------------------------------------
Debug Misi 3 (dipanggil otomatis oleh train_mission3.py):
  * obstacle_clearance.png : clearance badan-robot vs rintangan per step, histogram
                             clearance minimum per episode, jumlah tabrakan
  * replay_*.png           : replay 3D dgn rintangan, T1, T2, garis, lintasan EE

Standalone:  python Mission3/debug/plot_obstacle_clearance.py
"""
from __future__ import annotations

import sys
from collections import Counter
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
    d_safe = float(getattr(env, "_d_safe", 0.12))
    results, rec = evaluate_onnx(env, onnx_path, n_episodes, seed0=700)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    mins = []
    for r, rc in zip(results, rec):
        if r.clearance.size:
            ax[0].plot(1000 * np.clip(r.clearance, -0.05, 0.4),
                       color="tab:green" if rc["success"] else "tab:red", alpha=0.6)
            mins.append(1000 * float(r.clearance.min()))
    ax[0].axhline(1000 * d_safe, color="orange", ls="--", label="d_safe")
    ax[0].axhline(0, color="k", lw=1, label="tabrakan")
    ax[0].set_title("Clearance minimum badan-rintangan (mm)"); ax[0].set_xlabel("step")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    if mins:
        ax[1].hist(mins, bins=10, color="tab:blue")
        ax[1].axvline(1000 * d_safe, color="orange", ls="--")
    ax[1].set_title("Clearance minimum per episode (mm)")
    reasons = Counter(x["reason"] for x in rec)
    ax[2].bar(list(reasons.keys()), list(reasons.values()),
              color=["tab:green" if k == "success" else "tab:red" for k in reasons])
    ax[2].set_title("Alasan terminasi")
    plt.tight_layout()
    fig.savefig(out / "obstacle_clearance.png", dpi=110)
    plt.close(fig)
    for i, r in enumerate(results[:n_replay]):
        plot_replay_trajectory(r, str(out / f"replay_{i}.png"), title=f"Misi 3 replay {i}")
    sr = np.mean([x["success"] for x in rec])
    log(f"[debug-M3] success {sr * 100:.0f}% | alasan {dict(reasons)} | clearance min "
        f"{min(mins) if mins else float('nan'):.1f} mm | plot -> {out}")
    return {"success_rate": float(sr), "reasons": dict(reasons)}


def main():
    from shared.config_utils import load_yaml
    from shared.evaluation.viewer_eval import resolve_onnx
    from Mission3.eval_envs import Stage3ObstacleEvalEnv

    pcfg = load_yaml(_ROOT / "Mission3" / "configs" / "eval_config.yaml")["paths"]
    onnx = resolve_onnx(_ROOT, pcfg)
    factory = lambda training=False, randomize=False, seed=0: Stage3ObstacleEvalEnv(  # noqa: E731
        model_path=str(_ROOT / pcfg["robot_model_path"]), seed=seed, randomize=randomize)
    run(factory, str(onnx), str(_ROOT / pcfg["debug_dir"]))


if __name__ == "__main__":
    main()
