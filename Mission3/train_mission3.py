"""
Mission3/train_mission3.py
--------------------------
Misi 3: Path Following + Obstacle Avoidance (konstelasi stargaze).

Bobot awal dimuat dari Misi 2 (blok 'transfer' di configs/ppo_config.yaml), jadi
jalankan training Misi 2 lebih dulu.

Jalankan (tanpa argumen):
    python train_mission3.py            # dari folder Mission3/
    python Mission3/train_mission3.py   # dari root proyek
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission3.debug import plot_obstacle_clearance
from Mission3.envs.stage3_obstacle_env import Stage3ObstacleEnv, DEFAULT_CONFIG, DEFAULT_MODEL
from shared.debug_common import generic_post_training, run_all_sanity
from shared.training import DifficultyCurriculumCallback, MissionHooks, train_mission

MISSION = 3


def pre_debug(ctx):
    dbg = ctx.cfg["debug"]
    run_all_sanity(lambda: ctx.make_raw_env(training=False, randomize=False, seed=1),
                   dbg.get("sanity_n_ik_targets", 20), dbg.get("sanity_n_resets", 10),
                   dbg.get("sanity_n_scripted_episodes", 10))


def build_callbacks(ctx):
    """Kurikulum kesulitan rintangan (blok `curriculum` di ppo_config.yaml)."""
    c = ctx.cfg.get("curriculum", {}) or {}
    if not c.get("enabled", False):
        return []
    cb = DifficultyCurriculumCallback(
        levels=c["levels"], advance_rate=c.get("advance_success_rate", 0.80),
        check_every=c.get("check_every_n_episodes", 50), warmup=c.get("warmup_episodes", 100),
        window=c.get("window_episodes", 100), eval_env=ctx.eval_env,
        state_path=str(ctx.ckpt_dir / "curriculum_state.json"))
    ctx.level_fn = lambda: cb.level
    return [cb]


def post_debug(ctx, result):
    out = ctx.project_root / ctx.cfg["debug"]["output_dir"]
    generic_post_training(ctx, result, out)
    plot_obstacle_clearance.run(ctx.make_raw_env, result["onnx_path"], str(out),
             n_replay=ctx.cfg["debug"].get("replay_episodes", 3))


def main():
    return train_mission(
        mission_id=MISSION, project_root=ROOT,
        ppo_cfg_path=str(ROOT / "Mission3" / "configs" / "ppo_config.yaml"),
        env_cls=Stage3ObstacleEnv, env_cfg_path=DEFAULT_CONFIG, model_path=DEFAULT_MODEL,
        hooks=MissionHooks(pre_debug=pre_debug, build_callbacks=build_callbacks, post_debug=post_debug))


if __name__ == "__main__":
    main()
