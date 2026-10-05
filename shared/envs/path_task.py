"""
shared/envs/path_task.py
------------------------
Logika bersama Misi 2 & 3: fase A (reaching T1) -> fase B (path following T1->T2).

Dipisah ke shared supaya Mission3 tidak perlu meng-import Mission2
(aturan proyek: kode yang dipakai bersama ada di luar folder misi).

Reward fase B:
  R_path = -w_p * d_perp^2          (cross-track error ke segmen T1T2, sesuai spesifikasi)
"""
from __future__ import annotations

import numpy as np

from ik.pose_utils import perpendicular_distance_to_segment


class PathTaskMixin:
    """Gunakan sebagai: class StageXEnv(PathTaskMixin, BaseArmEnv)."""

    # ---------------- konfigurasi ----------------
    @property
    def t1_reach_threshold(self) -> float:
        return float(self.cfg.get("path_task", {}).get("t1_reach_threshold", 0.02))

    # ---------------- hook reset ----------------
    def _on_reset(self):
        self._ct_sum = 0.0
        self._ct_max = 0.0
        self._ct_n = 0

    # ---------------- validasi skenario ----------------
    def _validate_path_scenario(self, obstacles=None, margin: float = 0.0,
                                approach_margin: float | None = None) -> bool:
        """Fase A (home->T1) cukup reachable & bebas kontak (margin rintangan longgar,
        policy bebas memilih jalurnya); fase B (T1->T2) harus bebas rintangan dgn `margin`."""
        if approach_margin is None:
            approach_margin = margin
        if self.T1 is None or self.T2 is None:
            return False
        if np.linalg.norm(self.T2 - self.T1) < 0.05:
            return False
        if min(self.T1[2], self.T2[2]) < 0.12:          # terlalu dekat lantai
            return False
        chk = self.checker
        ok, q1, segs_a, _ = chk.approach(self.home_qpos.copy(), self.home_tcp, self.T1,
                                         self._nominal_quat, self.base_position[:2])
        if not ok:
            return False
        if obstacles is not None and obstacles[0].shape[0] > 0 and \
                np.min(chk.min_clearance_along(segs_a, *obstacles)) <= approach_margin:
            return False
        ok, _ = chk.follow(q1, chk.waypoints(self.T1, self.T2), self._nominal_quat,
                           obstacles, margin)
        return ok

    def _validate_scenario(self) -> bool:
        return self._validate_path_scenario()

    # ---------------- reward fase A/B ----------------
    def _path_reward_and_termination(self):
        tcp = self.tcp_pos
        if self.phase == 0:
            dist = float(np.linalg.norm(self.T1 - tcp))
            reward = self._common_reward_terms(dist)
            if dist < self.t1_reach_threshold:
                self.phase = 1
                self._prev_dist = float(np.linalg.norm(self.T2 - tcp))
                bonus = self._w("phase_transition_bonus", 5.0)
                self._rc["phase_bonus"] = bonus
                reward += bonus
            return reward, False, None

        dist = float(np.linalg.norm(self.T2 - tcp))
        reward = self._common_reward_terms(dist)
        d_perp = perpendicular_distance_to_segment(tcp, self.T1, self.T2)
        # R_path = -w * d_perp^2 (sesuai spesifikasi), dengan d_perp DIBATASI `cross_track_cap`.
        # Tanpa batas, d = 30 cm memberi -13.5/langkah (w=150) sehingga return melonjak dan
        # value loss meledak (terukur sampai 35000), terutama saat policy sedang menyimpang.
        cap = float(self.cfg.get("path_task", {}).get("cross_track_cap", 0.08))
        r_path = -abs(self._w("cross_track_penalty", 150.0)) * min(d_perp, cap) ** 2
        self._rc["cross_track"] = r_path
        reward += r_path
        self._ct_sum += d_perp
        self._ct_max = max(self._ct_max, d_perp)
        self._ct_n += 1
        if dist < self.success_pos_threshold:
            r_succ = self._w("success_reward", 20.0)
            self._rc["success"] = r_succ
            return reward + r_succ, True, "success"
        return reward, False, None

    # ---------------- info ----------------
    def _get_info(self) -> dict:
        info = super()._get_info()
        if self.T1 is not None and self.T2 is not None:
            info["cross_track"] = perpendicular_distance_to_segment(self.tcp_pos, self.T1, self.T2)
        # statistik cross-track hanya ada bila robot SUDAH masuk fase B (bukan 0 palsu)
        if getattr(self, "_ct_n", 0) > 0:
            info["cross_track_mean"] = self._ct_sum / self._ct_n
            info["cross_track_max"] = float(self._ct_max)
        return info
