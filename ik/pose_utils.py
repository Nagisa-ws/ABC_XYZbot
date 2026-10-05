"""
ik/pose_utils.py
----------------
Helper geometri pose (quaternion, wxyz) dan segmen garis untuk environment.
"""
from __future__ import annotations

import numpy as np
import mujoco


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dua vektor unit yang tegak lurus terhadap `axis`."""
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u) + 1e-9
    v = np.cross(axis, u)
    return u, v


def quat_from_approach_axis(approach_dir: np.ndarray,
                            up_hint: np.ndarray | None = None) -> np.ndarray:
    """Quaternion (wxyz) sehingga +Z lokal end-effector sejajar `approach_dir`."""
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0])
    z = approach_dir / (np.linalg.norm(approach_dir) + 1e-9)
    if abs(np.dot(z, up_hint)) > 0.98:
        up_hint = np.array([1.0, 0.0, 0.0])
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x) + 1e-9
    y = np.cross(z, x)
    rot_mat = np.column_stack([x, y, z]).flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_mat)
    return quat


def integrate_quat(quat: np.ndarray, delta_rotvec: np.ndarray) -> np.ndarray:
    """Integrasikan delta rotasi kecil (axis*angle, frame DUNIA) ke quaternion."""
    norm = float(np.linalg.norm(delta_rotvec))
    if norm < 1e-12:
        return quat.copy()
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, delta_rotvec / norm, norm)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, dq, quat)
    out /= np.linalg.norm(out) + 1e-12
    return out


def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Sudut (rad, 0..pi) antara dua orientasi."""
    dot = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2) + 1e-12))
    return 2.0 * float(np.arccos(np.clip(dot, -1.0, 1.0)))


def limit_quat_deviation(quat: np.ndarray, nominal: np.ndarray, max_angle: float) -> np.ndarray:
    """Jaga `quat` tetap dalam kerucut `max_angle` (rad) dari `nominal` (slerp ke batas).
    Mencegah random-walk orientasi target yang membuat IK tidak terselesaikan."""
    ang = quat_angle(quat, nominal)
    if ang <= max_angle or ang < 1e-9:
        return quat
    q = quat if np.dot(quat, nominal) >= 0 else -quat
    t = max_angle / ang
    out = (1.0 - t) * nominal + t * q          # nlerp (cukup akurat utk sudut kecil-menengah)
    return out / (np.linalg.norm(out) + 1e-12)


def project_point_on_segment(point: np.ndarray, seg_a: np.ndarray,
                             seg_b: np.ndarray) -> tuple[float, np.ndarray]:
    """(t, titik_proyeksi), t in [0,1] = posisi sepanjang segmen A->B."""
    ab = seg_b - seg_a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-12:
        return 0.0, seg_a.copy()
    t = float(np.dot(point - seg_a, ab) / ab_sq)
    t = max(0.0, min(1.0, t))
    return t, seg_a + t * ab


def perpendicular_distance_to_segment(point: np.ndarray, seg_a: np.ndarray,
                                      seg_b: np.ndarray) -> float:
    """Jarak tegak lurus terdekat dari `point` ke segmen A->B (cross-track error)."""
    _, proj = project_point_on_segment(point, seg_a, seg_b)
    return float(np.linalg.norm(point - proj))
