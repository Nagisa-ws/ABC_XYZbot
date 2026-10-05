"""
shared/control/hybrid.py
------------------------
Policy hibrida: RL (ONNX) + controller LQR posisi.

    obs mentah (57) --> RL  -> aksi 6D  (rotasi & pendekatan jauh)
                   \\--> LQR -> aksi posisi 3D (presisi menuju T1, menyusuri T1->T2, menggapai T2)

Serah-terima (latch, satu arah per episode): controller mengambil alih POSISI begitu fase 1 dimulai
atau jarak ke target aktif < handover_distance. Controller dijalankan di SETIAP langkah (menjaga
riwayat posisi untuk estimasi kecepatan), tetapi hanya dipakai setelah serah-terima.

Penggunaan (deploy):
    policy = make_hybrid_from_onnx("models/onnx_mission3_v1.onnx")
    policy.reset()                      # awal tiap episode
    action = policy(obs_mentah)         # obs mentah; normalisasi sudah ada di dalam ONNX
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from shared.config_utils import load_yaml
from shared.control.orientation_controller import OrientationConfig, OrientationController
from shared.control.path_controller import ControllerConfig, PathController
from shared.envs.obs_layout import OBS_SLICES

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "controller.yaml"


def load_control_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or DEFAULT_CONFIG_PATH)


class HybridPolicy:
    def __init__(self, rl_fn: Callable[[np.ndarray], np.ndarray], config: dict | None = None,
                 orientation: bool | None = None):
        cfg = config or load_control_config()
        self.rl_fn = rl_fn
        self.hcfg = dict(cfg.get("hybrid", {}))
        self.enabled = bool(self.hcfg.get("enabled", True))
        self.handover = float(self.hcfg.get("handover_distance", 0.08))
        self.blend = int(self.hcfg.get("blend_steps", 0))
        self.controller = PathController(ControllerConfig(**cfg.get("controller", {})))
        oc = dict(cfg.get("orientation", {}))
        self.orient_enabled = bool(oc.get("enabled", False)) if orientation is None else bool(orientation)
        self.gate_deg = float(oc.get("gate_angle_deg", 60.0))
        self.o_start = float(oc.get("start_distance", 0.40))
        self.hold_deg = float(oc.get("hold_until_deg", 10.0))
        self.max_hold = int(oc.get("max_hold_steps", 80))
        self.orient = OrientationController(OrientationConfig(
            k_rot=float(oc.get("k_rot", 0.3)), max_rot_delta=float(oc.get("max_rot_delta", 0.08)),
            lead_max_deg=float(oc.get("lead_max_deg", 15.0))))
        self.reset()

    def reset(self):
        self.controller.reset()
        self.orient.reset()
        self.latched = False
        self.cone_request = False      # True -> pemanggil harus melonggarkan batas kerucut orientasi (lihat bind_to_env)
        self._t = 0
        self._hold_steps = 0

    def path_angle_from_down_deg(self, obs: np.ndarray) -> float:
        d = np.asarray(obs[OBS_SLICES["path_dir"]], dtype=float)
        d = d / max(np.linalg.norm(d), 1e-12)
        return float(np.degrees(np.arccos(np.clip(-d[2], -1.0, 1.0))))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        a_rl = np.asarray(self.rl_fn(obs), dtype=float)
        phase = int(round(float(obs[OBS_SLICES["phase"]][0])))
        dist = float(np.linalg.norm(obs[OBS_SLICES["rel_target"]]))
        use_orient = self.orient_enabled and self.path_angle_from_down_deg(obs) < self.gate_deg
        hold_now = bool(use_orient and phase == 1 and self._hold_steps < self.max_hold
                        and self.orient.nose_error_deg(obs) > self.hold_deg)
        if hold_now:
            self._hold_steps += 1
        a_c = self.controller.act(obs, hold_along=hold_now)   # selalu dijalankan (riwayat kecepatan)
        if not self.enabled:
            return a_rl
        handover = self.o_start if use_orient else self.handover
        if not self.latched and (phase == 1 or dist < handover):
            self.latched, self._t = True, 0
            self.cone_request = bool(use_orient)
        if not self.latched:
            return a_rl
        w = 1.0 if self.blend <= 0 else min(1.0, (self._t + 1) / self.blend)
        self._t += 1
        out = a_rl.copy()
        out[:3] = (1.0 - w) * a_rl[:3] + w * a_c[:3]
        if use_orient:
            out[3:] = self.orient.act(obs)
        return out


def bind_to_env(policy: HybridPolicy, env):
    """Hubungkan policy ke env: melonggarkan env.max_orient_dev bila policy memintanya, dan
    mengembalikannya di awal tiap episode. Dipakai oleh benchmark & evaluasi viewer.
    Pada robot nyata, antarmuka perintah harus menerima rentang orientasi lebih lebar saat cone_request."""
    default = float(env.max_orient_dev)

    class _Bound:
        def reset(self):
            policy.reset()
            env.max_orient_dev = default

        def __call__(self, obs):
            a = policy(obs)
            env.max_orient_dev = float(np.pi) if policy.cone_request else default
            return a

    return _Bound()


def make_hybrid_from_onnx(onnx_path: str | Path, config_path: str | Path | None = None,
                          orientation: bool | None = None) -> HybridPolicy:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    def rl_fn(obs):
        return sess.run(["action"], {"observation": np.asarray(obs, dtype=np.float32)[None, :]})[0][0]

    return HybridPolicy(rl_fn, load_control_config(config_path), orientation=orientation)
