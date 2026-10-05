"""
ik/ik_solver.py
---------------
Numerical IK solver berbasis Jacobian Damped Least-Squares (DLS).

Dipakai setiap step oleh environment training/evaluasi untuk menerjemahkan
target pose Cartesian end-effector (aksi DRL) menjadi sudut sendi.

Perbaikan dibanding versi awal:
  * Toleransi default jauh lebih ketat (0.1 mm). Versi lama 5 mm == threshold
    akhir Misi 1, sehingga gerakan < 5 mm dianggap "sudah konvergen" dan robot
    tidak bergerak sama sekali.
  * Tiap iterasi hanya memakai mj_kinematics + mj_comPos (bukan mj_forward
    penuh) -> jauh lebih cepat, hasil Jacobian identik.
  * Bobot orientasi (rot_weight) supaya error rotasi tidak mendominasi posisi.
  * Sudut akhir selalu di-clamp ke joint range SEBENARNYA (versi lama hanya
    di-clip ke +-2*pi sehingga elbow (+-pi) bisa keluar range).
  * Buffer Jacobian dialokasikan sekali (tidak per iterasi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco


@dataclass
class IKConfig:
    max_iters: int = 60
    pos_tol: float = 1e-4            # 0.1 mm
    rot_tol: float = 2e-3            # rad
    rot_weight: float = 0.5          # bobot error rotasi relatif terhadap posisi
    damping_min: float = 1e-4
    damping_max: float = 5e-2
    manipulability_singular_thresh: float = 0.02
    step_scale: float = 1.0
    max_joint_step: float = 0.3      # batas dq per iterasi (rad)


def _quat_to_axis_angle_error(quat_current: np.ndarray, quat_target: np.ndarray) -> np.ndarray:
    """Error orientasi (axis * angle, frame dunia) dari current -> target."""
    qc = quat_current / (np.linalg.norm(quat_current) + 1e-12)
    qt = quat_target / (np.linalg.norm(quat_target) + 1e-12)
    quat_conj = qc.copy()
    quat_conj[1:] *= -1.0
    quat_err = np.zeros(4)
    mujoco.mju_mulQuat(quat_err, qt, quat_conj)
    if quat_err[0] < 0:
        quat_err = -quat_err
    axis_angle = np.zeros(3)
    mujoco.mju_quat2Vel(axis_angle, quat_err, 1.0)
    return axis_angle


class IKSolver:
    """IK solver terikat ke satu mujoco.MjModel (memakai MjData scratch sendiri)."""

    def __init__(self, model: mujoco.MjModel, site_name: str,
                 joint_names: list[str], cfg: IKConfig | None = None):
        self.model = model
        self.cfg = cfg if cfg is not None else IKConfig()
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' tidak ditemukan pada model.")
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in self.joint_ids):
            raise ValueError("Salah satu joint_names tidak ditemukan pada model.")
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in self.joint_ids], dtype=int)
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in self.joint_ids], dtype=int)
        self.joint_range = np.array([model.jnt_range[j] for j in self.joint_ids], dtype=float)
        self.n_joints = len(self.joint_ids)

        self._data = mujoco.MjData(model)        # scratch (tidak menyentuh state env)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._w = np.array([1.0, 1.0, 1.0] + [self.cfg.rot_weight] * 3)

    # ------------------------------------------------------------------ #
    def joint_limits_ok(self, q: np.ndarray, margin: float = 0.0) -> bool:
        lo = self.joint_range[:, 0] + margin
        hi = self.joint_range[:, 1] - margin
        return bool(np.all(q >= lo) and np.all(q <= hi))

    def clamp_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.joint_range[:, 0], self.joint_range[:, 1])

    def unwrap_to_current(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """Pilih representasi sudut (q + k*2pi) yang paling dekat dengan q_current
        dan masih di dalam joint range. Mencegah 'lompatan' 2*pi pada sendi
        berputar penuh (range +-2pi)."""
        two_pi = 2.0 * np.pi
        out = q_target.copy()
        for i in range(self.n_joints):
            lo, hi = self.joint_range[i]
            best, best_cost = None, np.inf
            for k in (-1, 0, 1):
                cand = q_target[i] + k * two_pi
                if lo - 1e-9 <= cand <= hi + 1e-9:
                    cost = abs(cand - q_current[i])
                    if cost < best_cost:
                        best, best_cost = cand, cost
            out[i] = best if best is not None else np.clip(q_target[i], lo, hi)
        return out

    # alias kompatibel dengan nama lama
    def unwrap_clip_unwind(self, q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        return self.unwrap_to_current(q_target, q_current)

    # ------------------------------------------------------------------ #
    def _update_kinematics(self, q_full: np.ndarray):
        d = self._data
        d.qpos[:] = q_full
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

    def site_pose(self, q_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: (pos, rotmat 3x3) site TCP untuk konfigurasi q_full."""
        self._update_kinematics(q_full)
        return (self._data.site_xpos[self.site_id].copy(),
                self._data.site_xmat[self.site_id].reshape(3, 3).copy())

    def manipulability_from_data(self, data: mujoco.MjData) -> float:
        """Yoshikawa manipulability (Jacobian posisi) dari MjData yang SUDAH
        melalui mj_kinematics + mj_comPos."""
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, self.site_id)
        Jp = self._jacp[:, self.dof_adr]
        return float(np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0)))

    def manipulability(self, q_full: np.ndarray) -> float:
        self._update_kinematics(q_full)
        return self.manipulability_from_data(self._data)

    # ------------------------------------------------------------------ #
    def solve(self, q_full_init: np.ndarray, target_pos: np.ndarray,
              target_quat: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Selesaikan IK dengan warm-start dari q_full_init.

        Return (q_arm (n_joints,), info). q_arm selalu berada dalam joint range.
        """
        cfg = self.cfg
        d = self._data
        q_full = np.array(q_full_init, dtype=float, copy=True)
        q_arm = q_full[self.qpos_adr].copy()
        q_arm0 = q_arm.copy()
        use_rot = target_quat is not None
        w = self._w if use_rot else self._w[:3]

        manip = 0.0
        pos_err_norm = np.inf
        rot_err_norm = 0.0
        converged = False
        n_iters = 0
        site_quat = np.zeros(4)

        for it in range(cfg.max_iters):
            n_iters = it + 1
            d.qpos[:] = q_full
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)

            pos_err = target_pos - d.site_xpos[self.site_id]
            pos_err_norm = float(np.linalg.norm(pos_err))

            mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site_id)
            Jp = self._jacp[:, self.dof_adr]

            if use_rot:
                mujoco.mju_mat2Quat(site_quat, d.site_xmat[self.site_id])
                rot_err = _quat_to_axis_angle_error(site_quat, target_quat)
                rot_err_norm = float(np.linalg.norm(rot_err))
                J = np.vstack([Jp, self._jacr[:, self.dof_adr]])
                err = np.concatenate([pos_err, rot_err])
            else:
                J, err = Jp, pos_err

            if pos_err_norm < cfg.pos_tol and rot_err_norm < cfg.rot_tol:
                converged = True
                break

            manip = float(np.sqrt(max(np.linalg.det(Jp @ Jp.T), 0.0)))
            if manip < cfg.manipulability_singular_thresh:
                ratio = np.clip(1.0 - manip / cfg.manipulability_singular_thresh, 0.0, 1.0)
                damping = cfg.damping_min + (cfg.damping_max - cfg.damping_min) * ratio
            else:
                damping = cfg.damping_min

            Jw = J * w[:, None]
            ew = err * w
            JJt = Jw @ Jw.T + (damping ** 2) * np.eye(Jw.shape[0])
            dq = Jw.T @ np.linalg.solve(JJt, ew)
            dq = np.clip(dq * cfg.step_scale, -cfg.max_joint_step, cfg.max_joint_step)

            q_arm = self.clamp_to_joint_limits(q_arm + dq)
            q_full[self.qpos_adr] = q_arm

        q_arm = self.unwrap_to_current(q_arm, q_arm0)
        info = {
            "n_iters": n_iters,
            "position_error": pos_err_norm,
            "rotation_error": rot_err_norm,
            "converged": converged,
            "manipulability": manip,
            "near_singularity": manip < cfg.manipulability_singular_thresh,
        }
        return q_arm, info
