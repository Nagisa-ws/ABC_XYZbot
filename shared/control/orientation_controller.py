"""
shared/control/orientation_controller.py
----------------------------------------
Controller orientasi: menyejajarkan MONCONG gripper (sumbu z TCP) dengan arah jalur T1->T2.

Membaca observasi 57-dim saja: tcp_rot6 (dua kolom pertama matriks rotasi TCP) dan path_dir.

Plant: aksi rotasi env = vektor rotasi kerangka DUNIA * max_rot_delta, diintegrasikan ke target_quat
(PERINTAH); IK baru mengikutinya dengan jeda. Karena itu galat dihitung terhadap perintah yang dilacak
sendiri (R_cmd), BUKAN orientasi aktual. Menghitung dari orientasi aktual membuat perintah "windup"
(berlari hingga ~180 derajat di depan orientasi aktual) saat IK tertinggal, dan lengan menabrak lantai
(terukur).

Anti-windup: bila perintah mendahului orientasi aktual lebih dari lead_max_deg, rotasi dihentikan sampai
orientasi aktual menyusul.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.envs.obs_layout import OBS_SLICES


def _rodrigues(rotvec: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(rotvec))
    if th < 1e-12:
        return np.eye(3)
    k = rotvec / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * K @ K


@dataclass
class OrientationConfig:
    k_rot: float = 0.3            # fraksi galat sudut yang dikoreksi per langkah
    max_rot_delta: float = 0.08   # harus sama dengan env (action.max_rot_delta)
    deadband_deg: float = 0.5
    lead_max_deg: float = 15.0    # batas perintah mendahului orientasi aktual (anti-windup)


class OrientationController:
    def __init__(self, cfg: OrientationConfig | None = None):
        self.cfg = cfg or OrientationConfig()
        self.reset()

    def reset(self):
        self.R_cmd = None

    @staticmethod
    def actual_rotation(obs: np.ndarray) -> np.ndarray:
        r6 = np.asarray(obs[OBS_SLICES["tcp_rot6"]], dtype=float)
        c0 = r6[:3] / max(np.linalg.norm(r6[:3]), 1e-12)
        c1 = r6[3:] - (r6[3:] @ c0) * c0
        c1 /= max(np.linalg.norm(c1), 1e-12)
        return np.column_stack([c0, c1, np.cross(c0, c1)])

    def nose_error_deg(self, obs: np.ndarray) -> float:
        d = np.asarray(obs[OBS_SLICES["path_dir"]], dtype=float)
        d /= max(np.linalg.norm(d), 1e-12)
        return float(np.degrees(np.arccos(np.clip(self.actual_rotation(obs)[:, 2] @ d, -1.0, 1.0))))

    def act(self, obs: np.ndarray) -> np.ndarray:
        c = self.cfg
        R_act = self.actual_rotation(obs)
        if self.R_cmd is None:
            self.R_cmd = R_act.copy()                  # inisialisasi dari orientasi aktual saat pertama dipanggil
        d = np.asarray(obs[OBS_SLICES["path_dir"]], dtype=float)
        d /= max(np.linalg.norm(d), 1e-12)

        z_cmd, z_act = self.R_cmd[:, 2], R_act[:, 2]
        lead = float(np.degrees(np.arccos(np.clip(z_cmd @ z_act, -1.0, 1.0))))
        axis = np.cross(z_cmd, d)
        s = float(np.linalg.norm(axis))
        theta = float(np.arctan2(s, float(z_cmd @ d)))
        if np.degrees(theta) < c.deadband_deg or lead >= c.lead_max_deg:
            return np.zeros(3)                         # sejajar, atau perintah terlalu jauh di depan -> tunggu
        if s < 1e-9:
            axis, s = self.R_cmd[:, 0].copy(), 1.0
        a = (axis / s) * theta * c.k_rot / c.max_rot_delta
        m = float(np.max(np.abs(a)))
        a = a / m if m > 1.0 else a
        self.R_cmd = _rodrigues(a * c.max_rot_delta) @ self.R_cmd      # melacak perintah yang diterapkan
        return a
