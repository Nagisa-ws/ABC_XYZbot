"""
Mission1/debug/plot_replay_episode.py
-------------------------------------
Debug Misi 1 (dipanggil otomatis oleh train_mission1.py setelah training):
  * threshold_sweep.png : success rate & jarak akhir pada tiap threshold curriculum
  * replay_*.png        : replay 3D episode dengan policy ONNX (target, lintasan EE, jarak)

Bisa juga dijalankan sendiri (memakai ONNX terbaru dari eval_config.yaml):
    python Mission1/debug/plot_replay_episode.py
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


def run(env_factory, onnx_path: str, out_dir: str, thresholds=(0.10, 0.05, 0.02, 0.005),
        n_per_threshold: int = 10, n_replay: int = 3, log=print):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = env_factory(training=False, randomize=False, seed=321)

    rates, dists = [], []
    for thr in thresholds:
        env.set_curriculum(thr, 1.0)
        _, rec = evaluate_onnx(env, onnx_path, n_per_threshold, seed0=700)
        rates.append(float(np.mean([r["success"] for r in rec])))
        dists.append(1000 * float(np.nanmean([r["final_distance"] for r in rec])))
        log(f"[debug-M1] threshold {thr * 100:5.1f} cm -> success {rates[-1] * 100:3.0f}% "
            f"| jarak akhir rata-rata {dists[-1]:.1f} mm")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    lab = [f"{t * 100:g} cm" for t in thresholds]
    ax[0].bar(lab, rates, color="tab:green")
    ax[0].set_ylim(0, 1.05); ax[0].set_title("Success rate per threshold"); ax[0].grid(alpha=0.3)
    ax[1].bar(lab, dists, color="tab:blue")
    ax[1].plot(lab, [1000 * t for t in thresholds], "r--", label="threshold")
    ax[1].set_title("Jarak akhir rata-rata (mm)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=110)
    plt.close(fig)

    env.set_curriculum(thresholds[-1], 1.0)
    results, _ = evaluate_onnx(env, onnx_path, n_replay, seed0=800)
    for i, r in enumerate(results):
        plot_replay_trajectory(r, str(out / f"replay_{i}.png"), title=f"Misi 1 replay {i}")
    log(f"[debug-M1] plot -> {out}")
    return {"rates": rates, "dists_mm": dists}


def main():
    from shared.config_utils import load_yaml
    from shared.evaluation.viewer_eval import resolve_onnx
    from Mission1.eval_envs import Stage1ReachEvalEnv

    cfg = load_yaml(_ROOT / "Mission1" / "configs" / "eval_config.yaml")
    pcfg = cfg["paths"]
    onnx = resolve_onnx(_ROOT, pcfg)
    factory = lambda training=False, randomize=False, seed=0: Stage1ReachEvalEnv(  # noqa: E731
        model_path=str(_ROOT / pcfg["robot_model_path"]), seed=seed, randomize=randomize)
    run(factory, str(onnx), str(_ROOT / pcfg["debug_dir"]))


if __name__ == "__main__":
    main()
