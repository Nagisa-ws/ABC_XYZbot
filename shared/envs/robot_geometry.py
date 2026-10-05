"""
shared/envs/robot_geometry.py
-----------------------------
Model geometri robot untuk perhitungan clearance terhadap rintangan titik.

Badan robot didekati oleh kumpulan kapsul (segmen + radius) yang diambil dari
primitif collision UR10e di XML (capsule/cylinder pada link lengan) ditambah
satu kapsul gripper dari origin Servo-v1 sampai tcp_site.

Karena hanya memakai geom primitif yang IDENTIK di ur10e_train.xml dan
ur10e_eval.xml, hasil clearance sama persis di training & validasi.

(Versi lama hanya memeriksa ORIGIN empat link -> lengan atas/bawah bisa
menembus rintangan tanpa terdeteksi.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

_EXCLUDE_BODIES = {"world", "base"}
_GRIPPER_BODY = "Servo-v1"


@dataclass
class RobotCapsules:
    body_ids: np.ndarray        # (C,)
    p0_local: np.ndarray        # (C,3) ujung 1 pada frame body
    p1_local: np.ndarray        # (C,3) ujung 2 pada frame body
    radii: np.ndarray           # (C,)
    names: list[str]

    @property
    def n(self) -> int:
        return len(self.radii)


def build_robot_capsules(model: mujoco.MjModel, tcp_site_name: str = "tcp_site",
                         gripper_radius: float = 0.04) -> RobotCapsules:
    body_ids, p0s, p1s, radii, names = [], [], [], [], []
    mat = np.zeros(9)
    for g in range(model.ngeom):
        # enum pybind11 tidak == int -> bandingkan lewat nilai int
        if int(model.geom_type[g]) not in (int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                                           int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            continue
        bid = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname in _EXCLUDE_BODIES:
            continue
        mujoco.mju_quat2Mat(mat, model.geom_quat[g])
        axis = mat.reshape(3, 3)[:, 2]
        half = float(model.geom_size[g][1])
        center = model.geom_pos[g]
        body_ids.append(bid)
        p0s.append(center - axis * half)
        p1s.append(center + axis * half)
        radii.append(float(model.geom_size[g][0]))
        names.append(f"{bname}:g{g}")

    gb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _GRIPPER_BODY)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site_name)
    if gb >= 0 and site >= 0:
        body_ids.append(gb)
        p0s.append(np.zeros(3))
        p1s.append(model.site_pos[site].copy())
        radii.append(float(gripper_radius))
        names.append(f"{_GRIPPER_BODY}:gripper")

    return RobotCapsules(
        body_ids=np.array(body_ids, dtype=int),
        p0_local=np.array(p0s, dtype=float),
        p1_local=np.array(p1s, dtype=float),
        radii=np.array(radii, dtype=float),
        names=names,
    )


def world_segments(caps: RobotCapsules, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Ujung kapsul dalam koordinat dunia (butuh data.xpos/xmat terbaru)."""
    pos = data.xpos[caps.body_ids]                          # (C,3)
    rot = data.xmat[caps.body_ids].reshape(-1, 3, 3)        # (C,3,3)
    a = pos + np.einsum("cij,cj->ci", rot, caps.p0_local)
    b = pos + np.einsum("cij,cj->ci", rot, caps.p1_local)
    return a, b


def points_to_segments(points: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Jarak titik (M,3) ke segmen (C) -> (dist (M,C), closest (M,C,3))."""
    ab = b - a                                               # (C,3)
    ab2 = np.maximum(np.einsum("ck,ck->c", ab, ab), 1e-12)   # (C,)
    ap = points[:, None, :] - a[None, :, :]                  # (M,C,3)
    t = np.clip(np.einsum("mck,ck->mc", ap, ab) / ab2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[..., None] * ab[None, :, :]
    dist = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return dist, closest


def obstacle_clearance(caps: RobotCapsules, data: mujoco.MjData,
                       points: np.ndarray, obs_radii: np.ndarray):
    """Clearance PERMUKAAN robot-rintangan.

    Return:
      clear     (M,)   clearance minimum tiap rintangan terhadap seluruh badan robot
      witness   (M,3)  titik pada robot yang paling dekat dengan tiap rintangan
    """
    a, b = world_segments(caps, data)
    dist, closest = points_to_segments(points, a, b)
    surf = dist - caps.radii[None, :] - obs_radii[:, None]   # (M,C)
    idx = np.argmin(surf, axis=1)
    rows = np.arange(points.shape[0])
    return surf[rows, idx], closest[rows, idx]
