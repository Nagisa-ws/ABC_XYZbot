"""
shared/envs/validation.py
-------------------------
Validasi skenario: apakah target / lintasan T1->T2 benar-benar reachable,
bebas self-collision/lantai, dan (Misi 3) bebas tabrakan dengan rintangan.

Dijalankan saat reset() menggunakan IK yang SAMA dengan yang dipakai env,
sehingga target yang lolos validasi dijamin bisa dicapai secara kinematik.
"""
from __future__ import annotations

import numpy as np
import mujoco

from ik.ik_solver import IKSolver
from shared.envs.robot_geometry import (RobotCapsules, obstacle_clearance,
                                        points_to_segments, world_segments)


class ReachabilityChecker:
    def __init__(self, model: mujoco.MjModel, ik_solver: IKSolver, caps: RobotCapsules,
                 pos_tol: float = 2e-3, waypoint_step: float = 0.04):
        self.model = model
        self.ik = ik_solver
        self.caps = caps
        self.pos_tol = pos_tol
        self.waypoint_step = waypoint_step
        self._data = mujoco.MjData(model)
        self.last_wps: np.ndarray | None = None      # waypoint jalur pendekatan yang lolos terakhir

    # ------------------------------------------------------------------ #
    def has_contact(self, q_full: np.ndarray) -> bool:
        d = self._data
        d.qpos[:] = q_full
        mujoco.mj_forward(self.model, d)
        return d.ncon > 0

    def polar_waypoints(self, p_from: np.ndarray, p_to: np.ndarray, base_xy: np.ndarray) -> np.ndarray:
        """Jalur melingkar mengitari sumbu base (azimuth linear, radius & tinggi linear).
        Dipakai untuk fase pendekatan home->T1 bila garis lurus melewati badan robot."""
        a, b = np.asarray(p_from) - np.r_[base_xy, 0.0], np.asarray(p_to) - np.r_[base_xy, 0.0]
        r0, r1 = np.hypot(a[0], a[1]), np.hypot(b[0], b[1])
        th0, th1 = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
        dth = (th1 - th0 + np.pi) % (2 * np.pi) - np.pi                  # rute terpendek
        arc = 0.5 * (r0 + r1) * abs(dth)
        length = float(np.hypot(arc, np.hypot(r1 - r0, b[2] - a[2])))
        n = max(2, int(np.ceil(length / self.waypoint_step)) + 1)
        t = np.linspace(0.0, 1.0, n)[1:]
        r, th, z = r0 + (r1 - r0) * t, th0 + dth * t, a[2] + (b[2] - a[2]) * t
        return np.stack([base_xy[0] + r * np.cos(th), base_xy[1] + r * np.sin(th), z], axis=1)

    def approach(self, q_start: np.ndarray, p_from: np.ndarray, p_to: np.ndarray, quat: np.ndarray,
                 base_xy: np.ndarray):
        """Coba jalur lurus lalu jalur polar. Return (ok, q_akhir, segs, nama_jalur)."""
        for name, wps in (("lurus", self.waypoints(p_from, p_to)),
                          ("polar", self.polar_waypoints(p_from, p_to, base_xy))):
            ok, q, segs = self.rollout(q_start, wps, quat)
            if ok:
                self.last_wps = np.array(wps)
                return True, q, segs, name
        return False, q_start, [], "gagal"

    def waypoints(self, p_from: np.ndarray, p_to: np.ndarray) -> np.ndarray:
        n = max(2, int(np.ceil(np.linalg.norm(p_to - p_from) / self.waypoint_step)) + 1)
        return np.linspace(p_from, p_to, n)[1:]          # tanpa titik awal

    def follow(self, q_full_start: np.ndarray, waypoints: np.ndarray, quat: np.ndarray,
               obstacles: tuple[np.ndarray, np.ndarray] | None = None,
               obstacle_margin: float = 0.0):
        """Ikuti `waypoints` berurutan dengan IK warm-start.
        Return (ok, q_full_akhir). Gagal bila IK tidak konvergen, ada kontak, atau
        badan robot terlalu dekat rintangan."""
        q_full = np.array(q_full_start, dtype=float, copy=True)
        for wp in waypoints:
            q_arm, info = self.ik.solve(q_full, wp, quat)
            if info["position_error"] > self.pos_tol:
                return False, q_full
            q_full[self.ik.qpos_adr] = q_arm
            if self.has_contact(q_full):
                return False, q_full
            if obstacles is not None and obstacles[0].shape[0] > 0:
                clear, _ = obstacle_clearance(self.caps, self._data, obstacles[0], obstacles[1])
                if float(np.min(clear)) <= obstacle_margin:
                    return False, q_full
        return True, q_full

    # ------------------------------------------------------------------ #
    def rollout(self, q_full_start: np.ndarray, waypoints: np.ndarray, quat: np.ndarray,
                include_start: bool = False):
        """Seperti follow() tetapi TANPA cek rintangan; merekam segmen kapsul robot
        (dunia) di tiap waypoint supaya clearance terhadap rintangan bisa dihitung
        sekaligus (dan rintangan pelanggar bisa dipangkas).

        Return (ok, q_full_akhir, segs) dengan segs = list[(a (C,3), b (C,3))].
        """
        q_full = np.array(q_full_start, dtype=float, copy=True)
        segs = []
        if include_start:
            self.has_contact(q_full)
            segs.append(world_segments(self.caps, self._data))
        for wp in waypoints:
            q_arm, info = self.ik.solve(q_full, wp, quat)
            if info["position_error"] > self.pos_tol:
                return False, q_full, segs
            q_full[self.ik.qpos_adr] = q_arm
            if self.has_contact(q_full):
                return False, q_full, segs
            segs.append(world_segments(self.caps, self._data))
        return True, q_full, segs

    def min_clearance_along(self, segs, points: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Clearance permukaan minimum tiap rintangan (M,) sepanjang seluruh `segs`."""
        out = np.full(points.shape[0], np.inf)
        for a, b in segs:
            dist, _ = points_to_segments(points, a, b)
            surf = dist - self.caps.radii[None, :] - radii[:, None]
            out = np.minimum(out, surf.min(axis=1))
        return out
