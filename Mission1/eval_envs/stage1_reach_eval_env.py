"""
Mission1/eval_envs/stage1_reach_eval_env.py
-------------------------------------------
Env EVALUASI Misi 1: reaching target tunggal (lingkungan kosong).

Subclass dari env training -> fungsi observasi, IK, validasi skenario, dan reward
IDENTIK dengan training (tidak ada lagi mismatch dimensi / urutan observasi).
Perbedaan hanya:
  * model = robot_model/scene_eval.xml (mesh visual; geom collision identik dgn training)
  * training=False  -> tanpa derau sensor; domain randomization hanya bila diminta
"""
from __future__ import annotations

from pathlib import Path

from Mission1.envs.stage1_reach_env import Stage1ReachEnv

_ROOT = Path(__file__).resolve().parents[2]
EVAL_MODEL = str(_ROOT / "robot_model" / "scene_eval.xml")
EVAL_CONFIG = str(_ROOT / "Mission1" / "configs" / "domain_randomization.yaml")


class Stage1ReachEvalEnv(Stage1ReachEnv):
    def __init__(self, model_path: str = EVAL_MODEL, config_path: str = EVAL_CONFIG,
                 seed: int | None = None, randomize: bool = False, obs_noise_std: float = 0.0,
                 success_pos_threshold: float | None = None):
        super().__init__(model_path=model_path, config_path=config_path,
                         obs_noise_std=obs_noise_std, training=False, seed=seed,
                         randomize=randomize)
        if success_pos_threshold is not None:
            self.set_curriculum(float(success_pos_threshold), 1.0)   # threshold final, tanpa skala reward
