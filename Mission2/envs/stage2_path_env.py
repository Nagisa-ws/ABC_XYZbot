"""
Mission2/envs/stage2_path_env.py
---------------------------------
Misi 2: Target Reaching & Straight-Line Path Following (tanpa rintangan).

  Fase A: home -> T1 (reaching)
  Fase B: T1 -> T2 menyusuri garis lurus, penalti cross-track R_path = -w_p * d_perp^2
Komponen reward: distance, progress, smoothness, cross-track (fase B), success (FIXED).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.envs.base_arm_env import BaseArmEnv
from shared.envs.path_task import PathTaskMixin

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_PROJECT_ROOT / "robot_model" / "scene_train.xml")
DEFAULT_CONFIG = str(_PROJECT_ROOT / "Mission2" / "configs" / "domain_randomization.yaml")


class Stage2PathEnv(PathTaskMixin, BaseArmEnv):
    MISSION_ID = 2

    def __init__(self, model_path: str = DEFAULT_MODEL, config_path: str = DEFAULT_CONFIG,
                 obs_noise_std: float | None = None, training: bool = True,
                 seed: int | None = None, randomize: bool | None = None):
        super().__init__(model_path=model_path, config_path=config_path,
                         obs_noise_std=obs_noise_std, training=training,
                         seed=seed, randomize=randomize)

    def _sample_scenario(self):
        """T1 acak di workspace; T2 = T1 + arah acak x panjang segmen.
        Reachability & bebas self-collision diperiksa di _validate_scenario()."""
        self.T1 = self._sample_workspace_point()
        lo, hi = self.cfg["path_task"]["segment_length_range"]
        direction = self._rng.normal(size=3)
        direction /= np.linalg.norm(direction) + 1e-9
        self.T2 = self.T1 + direction * self._rng.uniform(lo, hi)

    def _compute_reward_and_termination(self):
        reward, terminated, reason = self._path_reward_and_termination()
        pen, term, creason = self._contact_termination()
        if term:
            return reward + pen, True, creason
        return reward, terminated, reason
