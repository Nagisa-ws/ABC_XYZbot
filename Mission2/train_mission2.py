"""
Mission2/train_mission2.py
--------------------------
Misi 2: Target Reaching + Path Following (garis lurus T1->T2).

Bobot awal dimuat dari Misi 1 (blok 'transfer' di configs/ppo_config.yaml), jadi
jalankan training Misi 1 lebih dulu.

Jalankan (tanpa argumen):
    python train_mission2.py            # dari folder Mission2/
    python Mission2/train_mission2.py   # dari root proyek
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission2.debug import plot_cross_track
from Mission2.envs.stage2_path_env import Stage2PathEnv, DEFAULT_CONFIG, DEFAULT_MODEL
from shared.debug_common import generic_post_training, run_all_sanity
from shared.training import MissionHooks, train_mission

MISSION = 2


def pre_debug(ctx):
    dbg = ctx.cfg["debug"]
    run_all_sanity(lambda: ctx.make_raw_env(training=False, randomize=False, seed=1),
                   dbg.get("sanity_n_ik_targets", 20), dbg.get("sanity_n_resets", 10),
                   dbg.get("sanity_n_scripted_episodes", 10))


def post_debug(ctx, result):
    out = ctx.project_root / ctx.cfg["debug"]["output_dir"]
    generic_post_training(ctx, result, out)
    plot_cross_track.run(ctx.make_raw_env, result["onnx_path"], str(out),
             n_replay=ctx.cfg["debug"].get("replay_episodes", 3))


def main():
    return train_mission(
        mission_id=MISSION, project_root=ROOT,
        ppo_cfg_path=str(ROOT / "Mission2" / "configs" / "ppo_config.yaml"),
        env_cls=Stage2PathEnv, env_cfg_path=DEFAULT_CONFIG, model_path=DEFAULT_MODEL,
        hooks=MissionHooks(pre_debug=pre_debug, post_debug=post_debug))


if __name__ == "__main__":
    main()
