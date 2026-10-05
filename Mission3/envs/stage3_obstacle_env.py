"""
Mission3/envs/stage3_obstacle_env.py
-------------------------------------
Misi 3: Reaching + Path Following + Obstacle Avoidance.

Skenario (konsep stargaze, lihat obstacles/stargaze.py):
  T2 (titik akhir) = nucleus constellation (titik acak di workspace)
  T1 (titik awal)  = titik pada face terluar kubus, dipilih dgn clearance garis
                     T1->T2 terhadap rintangan >= min_line_clearance
  garis T1->T2     = lintasan lurus yang harus disusuri
  rintangan        = titik-titik constellation (bola kecil, bukan geom fisika)

Reward tambahan Misi 3:
  Obstacle potential  R_obs = -w_o * (1/(d+eps) - 1/(d_safe+eps))  untuk d < d_safe
                      (inversely-proportional ke jarak, dibuat kontinu di d = d_safe)
  Collision           R = collision_penalty (-100), episode berhenti
  d = clearance PERMUKAAN antara seluruh badan robot (kapsul) dan rintangan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from obstacles.stargaze import ConstellationGenerator
from shared.envs.base_arm_env import BaseArmEnv
from shared.envs.obs_layout import OBS_FAR
from shared.envs.path_task import PathTaskMixin
from shared.envs.robot_geometry import obstacle_clearance

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = str(_PROJECT_ROOT / "robot_model" / "scene_train.xml")
DEFAULT_CONFIG = str(_PROJECT_ROOT / "Mission3" / "configs" / "domain_randomization.yaml")


def _dist_to_polyline(p: np.ndarray, pts: np.ndarray) -> float:
    """Jarak titik p ke polyline pts (N,3)."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p[None, :] - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(p[None, :] - (a + t[:, None] * ab), axis=1)))


class Stage3ObstacleEnv(PathTaskMixin, BaseArmEnv):
    MISSION_ID = 3

    def __init__(self, model_path: str = DEFAULT_MODEL, config_path: str = DEFAULT_CONFIG,
                 obs_noise_std: float | None = None, training: bool = True,
                 seed: int | None = None, randomize: bool | None = None):
        super().__init__(model_path=model_path, config_path=config_path,
                         obs_noise_std=obs_noise_std, training=training,
                         seed=seed, randomize=randomize)
        m3 = self.cfg["mission3"]
        self._gen = ConstellationGenerator(
            L_values=tuple(m3["constellation_L_values"]),
            grid_size=int(m3["constellation_grid_size"]),
            n_points_per_face_range=tuple(m3["constellation_n_points_per_face_range"]),
            jitter=float(m3["constellation_jitter"]),
            min_radius_from_center=float(m3["constellation_min_radius"]),
        )
        self._obstacle_radius = float(m3["obstacle_radius"])
        self._d_safe = float(m3["d_safe"])
        self._collision_margin = float(m3.get("collision_margin", 0.0))
        self._min_line_clearance = float(m3["min_line_clearance"])
        self._start_mode = str(m3.get("start_selection", "random_safe"))
        self._forbidden_faces = list(m3.get("forbidden_faces", ["-z"]))
        self._initial_face = str(m3.get("initial_face", "auto"))
        self._reach = m3.get("start_reach", {"r_min": 0.25, "r_max": 0.95, "z_min": 0.12, "z_max": 0.95})
        self._reach_full = dict(self._reach)            # batas penuh (kurikulum menyempitkan sementara)
        val = self.cfg.get("validation", {})
        self._obstacle_margin = float(val.get("obstacle_margin", 0.01))
        self._approach_margin = float(val.get("approach_obstacle_margin", 0.0))
        self._home_keepout = float(m3.get("home_keepout", 0.06))
        self._max_prune_fraction = float(m3.get("max_prune_fraction", 0.6))
        self._min_obstacles = int(m3.get("min_obstacles", 6))
        self._prune_vs_chord = bool(m3.get("prune_vs_chord", True))
        self.constellation_face_id = -1
        self._ref_poly: np.ndarray | None = None   # polyline jalur referensi home->T1 (tervalidasi bebas rintangan)
        self._difficulty_id = -1          # -1 = tanpa kurikulum kesulitan (margin dari config)
        self._prev_phi = 0.0
        # MjData pose-home terpisah (untuk keep-out; self.data berubah selama episode)
        import mujoco
        self._home_data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self._home_data, self.home_key_id)
        mujoco.mj_forward(self.model, self._home_data)

    # ---------------- kurikulum kesulitan ----------------
    def set_difficulty(self, level: int, obstacle_margin: float, approach_margin: float,
                       reach_r_max: float | None = None, reach_z_max: float | None = None):
        """Atur kesulitan skenario (dipanggil callback kurikulum; berlaku mulai reset berikutnya).

        obstacle_margin / approach_margin : lebar koridor aman di sekitar lintasan referensi
            (T1->T2 / home->T1). Rintangan yang lebih dekat dari margin ke SAPUAN badan robot
            dipangkas saat validasi. Besar = rintangan jauh dari lintasan (mudah).
        reach_r_max / reach_z_max : batas radius & tinggi titik awal T1. Policy Misi 1-2 hanya pernah
            menyelesaikan reaching di ruang kerja sempit (r 0.35-0.48 m, z 0.43-0.63 m); T1 yang jauh
            di luar itu membuat policy berhenti 2-5 cm dari T1 (timeout). Dilonggarkan bertahap.
        """
        self._difficulty_id = int(level)
        self._obstacle_margin = float(obstacle_margin)
        self._approach_margin = float(approach_margin)
        self._reach = dict(self._reach_full)
        if reach_r_max is not None:
            self._reach["r_max"] = min(float(reach_r_max), self._reach_full["r_max"])
        if reach_z_max is not None:
            self._reach["z_max"] = min(float(reach_z_max), self._reach_full["z_max"])

    # ---------------- skenario ----------------
    def _start_reach_filter(self, pts: np.ndarray) -> np.ndarray:
        rel = pts - self.base_position[None, :]
        r = np.linalg.norm(rel[:, :2], axis=1)
        z = pts[:, 2]
        R = self._reach
        return (r >= R["r_min"]) & (r <= R["r_max"]) & (z >= R["z_min"]) & (z <= R["z_max"])

    def _valid_obstacle_mask(self, pts: np.ndarray) -> np.ndarray:
        """Buang titik yang berada di dalam badan base robot / di bawah lantai."""
        rel = pts - self.base_position[None, :]
        r = np.linalg.norm(rel[:, :2], axis=1)
        return (pts[:, 2] > 0.03) & ~((r < 0.22) & (pts[:, 2] < 0.45))

    def _home_keepout_mask(self, pts: np.ndarray) -> np.ndarray:
        """Buang rintangan yang terlalu dekat dengan badan robot di pose HOME
        (robot selalu mulai dari home; rintangan di situ = tabrakan tak terhindarkan)."""
        if pts.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        clear, _ = obstacle_clearance(self.caps, self._home_data, pts,
                                      np.full(pts.shape[0], self._obstacle_radius))
        return clear > self._home_keepout

    def _sample_scenario(self):
        nucleus = self._sample_workspace_point()
        res = self._gen.sample(nucleus, self._rng)
        keep = self._valid_obstacle_mask(res.points) if res.points.shape[0] else np.zeros(0, bool)
        if keep.any():
            keep[keep] = self._home_keepout_mask(res.points[keep])
        pts = res.points[keep] if res.points.shape[0] else res.points
        res.points = pts
        faces = self._gen.allowed_start_faces(nucleus, self.base_position,
                                              self._forbidden_faces, self._initial_face)
        sel = self._gen.select_start_point(res, self._rng, faces, self._start_reach_filter,
                                           self._start_mode, self._min_line_clearance)
        self.T2 = nucleus
        if sel is None:                 # tidak ada kandidat reachable -> akan gagal validasi
            self.T1 = nucleus + np.array([0.3, 0.0, 0.0])
            self.constellation_face_id = -1
        else:
            self.T1, _, self.constellation_face_id = sel
        self.obstacle_points = pts
        self.obstacle_radii = np.full(pts.shape[0], self._obstacle_radius)

    def _validate_scenario(self) -> bool:
        """Reject-or-repair.

        Lintasan referensi home->T1->T2 dijalankan SEKALI lewat IK (tidak bergantung
        pada rintangan). Rintangan yang ditembus badan robot sepanjang lintasan itu
        DIPANGKAS; skenario ditolak hanya bila terlalu banyak rintangan terpangkas
        atau sisa rintangan terlalu sedikit. Karena lintasan tidak berubah setelah
        pemangkasan, skenario hasil validasi dijamin punya solusi bebas-tabrakan.
        (Menolak seluruh konstelasi, seperti versi sebelumnya, hanya lolos ~8%.)
        """
        if self.T1 is None or self.T2 is None:
            return False
        if np.linalg.norm(self.T2 - self.T1) < 0.05 or min(self.T1[2], self.T2[2]) < 0.12:
            return False
        chk = self.checker
        ok, q1, segs_a, _ = chk.approach(self.home_qpos.copy(), self.home_tcp, self.T1,
                                         self._nominal_quat, self.base_position[:2])
        if not ok:
            return False
        ok, _, segs_b = chk.rollout(q1, chk.waypoints(self.T1, self.T2), self._nominal_quat,
                                    include_start=True)
        if not ok:
            return False

        pts, rad = self.obstacle_points, self.obstacle_radii
        m = pts.shape[0]
        if m == 0:
            return True
        bad = (chk.min_clearance_along(segs_a, pts, rad) <= self._approach_margin) | \
              (chk.min_clearance_along(segs_b, pts, rad) <= self._obstacle_margin)
        if self._prune_vs_chord:
            # Jalur referensi bisa berupa jalur POLAR (mengitari base) bila garis lurus home->T1 tidak
            # layak, tetapi jalur alami policy cenderung lebih dekat ke garis lurus (terukur: 8% sukses,
            # 17 dari 25 skenario polar menabrak di fase A). Pangkas juga rintangan di dekat garis lurus itu.
            chord = np.vstack([self.home_tcp[None, :], chk.waypoints(self.home_tcp, self.T1)])
            bad |= np.array([_dist_to_polyline(p, chord) - r for p, r in zip(pts, rad)]) < self._approach_margin
        keep = ~bad
        if keep.sum() < self._min_obstacles or bad.sum() > self._max_prune_fraction * m:
            return False
        self.obstacle_points, self.obstacle_radii = pts[keep], rad[keep]
        return True

    # ---------------- reward ----------------
    def _phi(self, clear: float) -> float:
        """Potensial rintangan: Phi(c) = -k * (1/(c+eps) - 1/(d_safe+eps)) untuk c < d_safe, selain itu 0."""
        if not np.isfinite(clear):
            return 0.0
        c = max(float(clear), 0.0)
        if c >= self._d_safe:
            return 0.0
        eps = self._w("obstacle_eps", 0.01)
        return -abs(self._w("obstacle_potential_k", 0.3)) * (1.0 / (c + eps) - 1.0 / (self._d_safe + eps))

    def _on_reset(self):
        super()._on_reset()
        self._prev_phi = self._phi(self._min_clear)
        wps = self.ref_approach_wps
        self._ref_poly = (np.vstack([self.home_tcp[None, :], wps])
                          if (wps is not None and len(wps) > 0) else None)

    def _compute_reward_and_termination(self):
        phase_before = self.phase
        reward, terminated, reason = self._path_reward_and_termination()
        clear = self._min_clear

        # Cross-track fase PENDEKATAN: R = -w * min(d, cap)^2, d = jarak TCP ke jalur referensi home->T1.
        # Jalur referensi sudah dijamin bebas rintangan (margin pendekatan) saat validasi skenario, tetapi
        # jalur alami policy (hasil Misi 1-2) menyimpang 5-15 cm darinya -> 70% tabrakan terjadi di fase A.
        # Tanpa sinyal ini policy tak punya alasan menempel ke jalur yang aman.
        w_a = abs(self._w("approach_track_penalty", 0.0))
        if w_a > 0.0 and phase_before == 0 and self._ref_poly is not None:
            # batas TERPISAH dari cross-track fase B: penyimpangan fase A bisa 8-20 cm (median ~8 cm);
            # dengan batas 8 cm penalti jenuh dan gradiennya nol.
            cap = float(self.cfg.get("path_task", {}).get("approach_track_cap", 0.25))
            d_ref = min(_dist_to_polyline(self.tcp_pos, self._ref_poly), cap)
            r_ref = -w_a * d_ref ** 2
            self._rc["approach_track"] = r_ref
            reward += r_ref
        potential_mode = str(self.W.get("obstacle_mode", "state")) == "potential"

        if np.isfinite(clear):
            if potential_mode:
                # Shaping berbasis potensial: F = gamma*Phi(s') - Phi(s). Memberi gradien menjauhi
                # rintangan tanpa membuat 'diam di dekat rintangan' lebih murah daripada bergerak
                # (penalti per-langkah lama menciptakan minimum lokal itu: timeout 18% -> 69%).
                phi = self._phi(clear)
                r_obs = self._w("obstacle_gamma", 0.995) * phi - self._prev_phi
                self._prev_phi = phi
                self._rc["obstacle"] = r_obs
                reward += r_obs
            if clear <= self._collision_margin:
                pen = self._w("collision_penalty", -100.0)
                self._rc["collision"] = pen
                return reward + pen, True, "collision"
            if not potential_mode and clear < self._d_safe:
                eps = self._w("obstacle_eps", 0.01)
                r_obs = -abs(self._w("obstacle_penalty", 0.02)) * (
                    1.0 / (clear + eps) - 1.0 / (self._d_safe + eps))
                self._rc["obstacle"] = r_obs
                reward += r_obs

        pen, term, creason = self._contact_termination()
        if term:
            return reward + pen, True, creason
        return reward, terminated, reason

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["min_clearance"] = float(self._min_clear) if np.isfinite(self._min_clear) else float(OBS_FAR)
        if self._ref_poly is not None:
            info["approach_dev"] = _dist_to_polyline(self.tcp_pos, self._ref_poly)
        info["difficulty_id"] = int(self._difficulty_id)
        info["obstacle_margin"] = float(self._obstacle_margin)
        return info
