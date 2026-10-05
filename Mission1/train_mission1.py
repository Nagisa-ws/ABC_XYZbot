"""
Mission1/train_mission1.py
--------------------------
Misi 1: Target Reaching dengan ADAPTIVE THRESHOLD (0.10 -> 0.05 -> 0.02 -> 0.005 m).

Jalankan (tanpa argumen):
    python train_mission1.py            # dari folder Mission1/
    python Mission1/train_mission1.py   # dari root proyek

Yang dilakukan: sanity check (debug tools) -> training PPO + adaptive threshold
(TensorBoard) -> simpan best/final -> ekspor ONNX berversi ke models/ ->
verify_onnx -> debug pasca-training (plot).
Semua parameter ada di Mission1/configs/ppo_config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission1.debug import plot_replay_episode
from Mission1.envs import Stage1ReachEnv
from Mission1.envs.stage1_reach_env import DEFAULT_CONFIG, DEFAULT_MODEL
from shared.debug_common import generic_post_training, run_all_sanity
from shared.training import AdaptiveThresholdCallback, MissionHooks, train_mission

MISSION = 1


def pre_debug(ctx):
    dbg = ctx.cfg["debug"]
    run_all_sanity(lambda: ctx.make_raw_env(training=False, randomize=False, seed=1),
                   dbg.get("sanity_n_ik_targets", 20), dbg.get("sanity_n_resets", 10),
                   dbg.get("sanity_n_scripted_episodes", 10))


def build_callbacks(ctx):
    a = ctx.cfg["adaptive"]
    if not a.get("enabled", True):                       # tanpa curriculum: langsung threshold final
        ctx.train_env.env_method("set_curriculum", float(a["thresholds"][-1]), 1.0)
        ctx.eval_env.env_method("set_curriculum", float(a["thresholds"][-1]), 1.0)
        return []
    cb = AdaptiveThresholdCallback(
        thresholds=a["thresholds"], advance_rate=a["success_rate_to_advance"],
        check_every=a["check_every_n_episodes"], warmup=a["warmup_episodes"],
        window=a.get("window_episodes", 100), multiplier_per_level=a["reward_success_multiplier"],
        eval_env=ctx.eval_env, state_path=str(ctx.ckpt_dir / "curriculum_state.json"))
    ctx.level_fn = lambda: cb.level
    return [cb]


def post_debug(ctx, result):
    out = ctx.project_root / ctx.cfg["debug"]["output_dir"]
    generic_post_training(ctx, result, out)
    plot_replay_episode.run(ctx.make_raw_env, result["onnx_path"], str(out),
                            thresholds=ctx.cfg["adaptive"]["thresholds"],
                            n_replay=ctx.cfg["debug"].get("replay_episodes", 3))


def main():
    return train_mission(
        mission_id=MISSION, project_root=ROOT,
        ppo_cfg_path=str(ROOT / "Mission1" / "configs" / "ppo_config.yaml"),
        env_cls=Stage1ReachEnv, env_cfg_path=DEFAULT_CONFIG, model_path=DEFAULT_MODEL,
        hooks=MissionHooks(pre_debug=pre_debug, build_callbacks=build_callbacks, post_debug=post_debug))


if __name__ == "__main__":
    main()
