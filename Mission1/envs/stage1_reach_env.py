"""
Mission1/envs/stage1_reach_env.py
----------------------------------
Misi 1: Adaptive Threshold Target Reaching (lingkungan KOSONG).

Hanya robot + 1 titik target acak. Komponen reward:
  * Distance penalty   R_dist   = -w_d * ||P_target - P_ee||
  * Progress shaping   (simetris, tidak mengubah solusi optimal)
  * Smoothness penalty R_smooth = -w_s * ||a_t - a_{t-1}||^2
  * Success reward     R_base x multiplier(r_thresh)   <- SATU-SATUNYA yang adaptif
  * Safety             kontak self/lantai -> terminasi
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from shared.envs.base_arm_env import BaseArmEnv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_PROJECT_ROOT / "robot_model" / "scene_train.xml")
DEFAULT_CONFIG = str(_PROJECT_ROOT / "Mission1" / "configs" / "domain_randomization.yaml")


class Stage1ReachEnv(BaseArmEnv):
    MISSION_ID = 1

    def __init__(self, model_path: str = DEFAULT_MODEL, config_path: str = DEFAULT_CONFIG,
                 obs_noise_std: float | None = None, training: bool = True,
                 seed: int | None = None, randomize: bool | None = None):
        super().__init__(model_path=model_path, config_path=config_path,
                         obs_noise_std=obs_noise_std, training=training,
                         seed=seed, randomize=randomize)
        self.success_reward_multiplier = 1.0

    # ---- API curriculum ----
    def set_success_reward_multiplier(self, value: float):
        self.success_reward_multiplier = float(value)

    def set_curriculum(self, threshold: float, multiplier: float):
        """Dipanggil callback adaptive lewat VecEnv.env_method (satu panggilan)."""
        self.set_success_threshold(threshold)
        self.set_success_reward_multiplier(multiplier)

    # ---- skenario ----
    def _sample_scenario(self):
        self.T1 = self._sample_workspace_point()
        self.T2 = None

    # ---- reward ----
    def _compute_reward_and_termination(self):
        dist = float(np.linalg.norm(self.T1 - self.tcp_pos))
        reward = self._common_reward_terms(dist)

        pen, term, reason = self._contact_termination()
        if term:
            return reward + pen, True, reason

        if dist < self.success_pos_threshold:
            r_succ = self._w("success_reward", 10.0) * self.success_reward_multiplier
            self._rc["success"] = r_succ
            return reward + r_succ, True, "success"
        return reward, False, None
