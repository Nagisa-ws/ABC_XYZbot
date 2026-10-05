"""
shared/envs/base_arm_env.py
---------------------------
Environment dasar untuk SEMUA misi curriculum (UR10e 6-DoF + gripper, MuJoCo).

Tanggung jawab kelas ini:
  * Memuat model MuJoCo, IK solver, kapsul geometri robot.
  * Action space 6-D ternormalisasi [-1,1]  ->  (dx,dy,dz, drx,dry,drz) EE yang
    diselesaikan IK menjadi target sudut sendi (position actuator).
  * Observasi terpadu (lihat obs_layout.py) -> transfer antar misi bisa jalan.
  * Domain randomization (payload, gravitasi residual, noise awal, noise sensor).
  * Validasi skenario (reachable, bebas kontak) saat reset.
  * Komponen reward yang dipakai ulang (distance, progress, smoothness, dst.).

Subclass misi meng-override:
  _sample_scenario()                   -> isi self.T1 (+ self.T2 / obstacle)
  _validate_scenario()                 -> (opsional) validasi tambahan
  _compute_reward_and_termination()    -> reward & terminasi misi
"""
from __future__ import annotations

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from ik.ik_solver import IKSolver, IKConfig
from ik.pose_utils import (integrate_quat, limit_quat_deviation,
                           project_point_on_segment)
from shared.config_utils import load_yaml
from shared.envs.obs_layout import (OBS_DIM, OBS_SLICES, NOISE_MASK, ACTIVE_DIMS,
                                    N_NEAREST_OBSTACLES, OBS_FAR)
from shared.envs.robot_geometry import build_robot_capsules, obstacle_clearance
from shared.envs.validation import ReachabilityChecker

ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]
ARM_ACTUATOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
TCP_SITE = "tcp_site"
GRIPPER_ACTUATOR = "gripper_actuator"
PAYLOAD_BODY = "Servo-v1"
NOMINAL_PAYLOAD_BODY_MASS = 3.424


class BaseArmEnv(gym.Env):
    """Base env abstrak (dipakai lintas misi)."""

    metadata = {"render_modes": []}
    MISSION_ID = 1

    def __init__(self,
                 model_path: str,
                 config_path: str,
                 obs_noise_std: float | None = None,
                 training: bool = True,
                 seed: int | None = None,
                 randomize: bool | None = None):
        super().__init__()
        self.config_path = str(config_path)
        self.cfg = load_yaml(config_path)
        self.training = bool(training)
        self.randomize = self.training if randomize is None else bool(randomize)

        # ---------------- model MuJoCo ----------------
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mj = mujoco
        name2id = mj.mj_name2id
        self.arm_joint_ids = [name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINT_NAMES]
        self.arm_qpos_adr = np.array([self.model.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dof_adr = np.array([self.model.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_actuator_ids = [name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATOR_NAMES]
        self.gripper_actuator_id = name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        self.tcp_site_id = name2id(self.model, mj.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.home_key_id = name2id(self.model, mj.mjtObj.mjOBJ_KEY, "home")
        self.base_body_id = name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.payload_body_id = name2id(self.model, mj.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
        for nm, i in [("tcp_site", self.tcp_site_id), ("home", self.home_key_id),
                      ("base", self.base_body_id), (PAYLOAD_BODY, self.payload_body_id)]:
            if i < 0:
                raise ValueError(f"'{nm}' tidak ditemukan pada model {model_path}")
        self.arm_joint_range = np.array([self.model.jnt_range[j] for j in self.arm_joint_ids])

        # ---------------- konfigurasi ----------------
        cfg = self.cfg
        ep = cfg["episode"]
        self.max_steps = int(ep["max_steps"])
        self.success_pos_threshold = float(ep["success_pos_threshold"])

        act = cfg.get("action", {})
        self.max_pos_delta = float(act.get("max_pos_delta", 0.05))
        self.max_rot_delta = float(act.get("max_rot_delta", 0.08))
        self.max_joint_delta = float(act.get("max_joint_delta", 0.25))
        self.max_orient_dev = np.deg2rad(float(act.get("max_orientation_dev_deg", 30.0)))

        phys = cfg.get("physics", {})
        self.n_substeps = int(phys.get("n_substeps", 10))
        self.gravity_compensation = bool(phys.get("gravity_compensation", True))

        safety = cfg.get("safety", {})
        self.terminate_on_contact = bool(safety.get("terminate_on_contact", True))
        self.gripper_radius = float(safety.get("gripper_collision_radius", 0.04))

        rnd = cfg.get("randomization", {})
        self.init_joint_noise = float(rnd.get("init_joint_noise", 0.05))
        # derau sensor: default dari config saat training, 0 saat evaluasi (kecuali diminta eksplisit)
        if obs_noise_std is None:
            obs_noise_std = float(rnd.get("obs_noise_std", 0.005)) if self.training else 0.0
        self.obs_noise_std = float(obs_noise_std)
        # Derau HANYA pada dimensi yang aktif di misi ini. Dimensi belum-aktif harus tetap
        # konstan: bila diberi derau, VecNormalize (var ~ derau^2) menormalkannya menjadi N(0,1)
        # acak yang masuk ke policy dan merusak transfer antar misi.
        self._noise_mask = (NOISE_MASK & ACTIVE_DIMS[self.MISSION_ID]).astype(np.float64)

        self.W = dict(cfg.get("reward", {}))

        ik_cfg = IKConfig(**cfg.get("ik", {}))
        self.ik_solver = IKSolver(self.model, TCP_SITE, ARM_JOINT_NAMES, ik_cfg)
        self.caps = build_robot_capsules(self.model, TCP_SITE, self.gripper_radius)
        val = cfg.get("validation", {})
        self.max_scenario_attempts = int(val.get("max_scenario_attempts", 40))
        self.checker = ReachabilityChecker(self.model, self.ik_solver, self.caps,
                                           pos_tol=float(val.get("ik_pos_tol", 2e-3)),
                                           waypoint_step=float(val.get("waypoint_step", 0.04)))

        # pose home
        mj.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mj.mj_forward(self.model, self.data)
        self.base_position = self.data.xpos[self.base_body_id].copy()
        self.home_qpos = self.data.qpos.copy()
        self.home_ctrl = self.data.ctrl.copy()
        self._nominal_quat = np.zeros(4)
        mj.mju_mat2Quat(self._nominal_quat, self.data.site_xmat[self.tcp_site_id])
        self.home_tcp = self.data.site_xpos[self.tcp_site_id].copy()

        # ---------------- spaces ----------------
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)

        # ---------------- state ----------------
        self._rng = np.random.default_rng(seed)
        self.T1: np.ndarray | None = None
        self.T2: np.ndarray | None = None
        self.phase = 0
        self.obstacle_points = np.zeros((0, 3))
        self.obstacle_radii = np.zeros(0)
        self.target_quat = self._nominal_quat.copy()
        self.step_count = 0
        self._prev_action = np.zeros(6)
        self._cur_action = np.zeros(6)
        self._prev_dist = 0.0
        self._payload_mass_extra = 0.0
        self._gravity_scale = 1.0
        self._min_clear = np.inf
        self._obs_clear = np.zeros(0)
        self._obs_witness = np.zeros((0, 3))
        self._manip = 0.0
        self._contact_reason: str | None = None
        self._ik_info: dict = {}
        self._rc: dict = {}                  # komponen reward step terakhir (debug)
        self._scenario_ok = True
        self._scenario_fail_count = 0
        self.ref_approach_wps: np.ndarray | None = None   # jalur pendekatan tervalidasi (untuk controller referensi)
        self._apply_domain_randomization()   # set gravcomp nominal

    # ------------------------------------------------------------------ #
    # Properti kompatibilitas
    # ------------------------------------------------------------------ #
    @property
    def finish_point(self) -> np.ndarray | None:
        return self.T1

    @property
    def tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tcp_site_id].copy()

    def active_target(self) -> np.ndarray:
        if self.T2 is not None and self.phase == 1:
            return self.T2
        return self.T1

    # API curriculum (dipanggil callback via VecEnv.env_method)
    def set_success_threshold(self, value: float):
        self.success_pos_threshold = float(value)

    def get_success_threshold(self) -> float:
        return self.success_pos_threshold

    def set_reward_overrides(self, **kwargs):
        self.W.update({k: float(v) for k, v in kwargs.items()})

    # ------------------------------------------------------------------ #
    # Workspace & domain randomization
    # ------------------------------------------------------------------ #
    def _sample_workspace_point(self) -> np.ndarray:
        ws = self.cfg["workspace"]
        r = self._rng.uniform(*ws["finish_point_radius_range"])
        h = self._rng.uniform(*ws["finish_point_height_range"])
        az = self._rng.uniform(*ws["finish_point_azimuth_range"])
        z_off = float(ws.get("base_height_offset", 0.0))
        return self.base_position + np.array([r * np.cos(az), r * np.sin(az), h + z_off])

    def _apply_domain_randomization(self):
        dyn = self.cfg.get("robot_dynamics", {})
        if self.randomize:
            extra = float(self._rng.uniform(*dyn.get("payload_mass_range", [0.0, 0.0])))
            scale = float(self._rng.uniform(*dyn.get("gravity_scale_range", [1.0, 1.0])))
        else:
            extra, scale = 0.0, 1.0
        self._payload_mass_extra = extra
        self._gravity_scale = scale
        m = self.model
        m.body_mass[self.payload_body_id] = NOMINAL_PAYLOAD_BODY_MASS + extra
        m.opt.gravity[:] = np.array([0.0, 0.0, -9.81 * scale])
        if self.gravity_compensation:
            # Kompensasi gravitasi NOMINAL: gaya = -m*g*scale*gc. Dengan gc = 1/scale
            # gravitasi nominal terkompensasi persis; yang tersisa hanyalah
            # residual (scale-1)*g dan massa payload ekstra -> perturbasi nyata.
            gc = np.full(m.nbody, 1.0 / scale)
            gc[0] = 0.0
            gc[self.payload_body_id] = (NOMINAL_PAYLOAD_BODY_MASS / scale) / (NOMINAL_PAYLOAD_BODY_MASS + extra)
            m.body_gravcomp[:] = gc
        else:
            m.body_gravcomp[:] = 0.0

    # ------------------------------------------------------------------ #
    # Hook subclass
    # ------------------------------------------------------------------ #
    def _sample_scenario(self):
        """Isi self.T1 (dan self.T2 / obstacle bila perlu)."""
        raise NotImplementedError

    def _validate_scenario(self) -> bool:
        """Default: T1 harus reachable dari home lewat jalur lurus ATAU polar, bebas kontak."""
        ok, _, _, _ = self.checker.approach(self.home_qpos.copy(), self.home_tcp, self.T1,
                                            self._nominal_quat, self.base_position[:2])
        return ok

    def _compute_reward_and_termination(self) -> tuple[float, bool, str | None]:
        raise NotImplementedError

    def _on_reset(self):
        """Hook setelah skenario siap (inisialisasi variabel reward misi)."""

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        if self.randomize and self.init_joint_noise > 0:
            noise = self._rng.uniform(-self.init_joint_noise, self.init_joint_noise, 6)
            q = np.clip(self.data.qpos[self.arm_qpos_adr] + noise,
                        self.arm_joint_range[:, 0], self.arm_joint_range[:, 1])
            self.data.qpos[self.arm_qpos_adr] = q
            self.data.ctrl[self.arm_actuator_ids] = q
        self._apply_domain_randomization()
        mujoco.mj_forward(self.model, self.data)

        self.phase = 0
        self.T1 = self.T2 = None
        self.obstacle_points = np.zeros((0, 3))
        self.obstacle_radii = np.zeros(0)
        self._scenario_ok = False
        for _ in range(self.max_scenario_attempts):
            self._sample_scenario()
            self.checker.last_wps = None
            if self._validate_scenario():
                self._scenario_ok = True
                self.ref_approach_wps = self.checker.last_wps
                break
        if not self._scenario_ok:
            self._scenario_fail_count += 1     # skenario terakhir tetap dipakai (dilaporkan di info)

        mujoco.mju_mat2Quat(self.target_quat, self.data.site_xmat[self.tcp_site_id])
        self.step_count = 0
        self._prev_action = np.zeros(6)
        self._cur_action = np.zeros(6)
        self._update_derived_state()
        self._prev_dist = float(np.linalg.norm(self.active_target() - self.tcp_pos))
        self._rc = {}
        self._on_reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(6), -1.0, 1.0)
        self._cur_action = action
        pos_delta = action[:3] * self.max_pos_delta
        rot_delta = action[3:] * self.max_rot_delta

        q_now = self.data.qpos[self.arm_qpos_adr].copy()
        tcp_pos = self.tcp_pos
        target_pos = tcp_pos + pos_delta
        quat = integrate_quat(self.target_quat, rot_delta)
        self.target_quat = limit_quat_deviation(quat, self._nominal_quat, self.max_orient_dev)

        q_target, ik_info = self.ik_solver.solve(self.data.qpos, target_pos, self.target_quat)
        dq = np.clip(q_target - q_now, -self.max_joint_delta, self.max_joint_delta)
        self.data.ctrl[self.arm_actuator_ids] = q_now + dq
        self._ik_info = ik_info

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._update_derived_state()

        reward, terminated, reason = self._compute_reward_and_termination()
        truncated = (not terminated) and self.step_count >= self.max_steps
        if truncated:
            reason = "timeout"

        self._prev_action = action
        obs = self._get_obs()
        info = self._get_info()
        info["ik"] = ik_info
        info["termination_reason"] = reason
        info["is_success"] = bool(reason == "success")
        info["reward_components"] = dict(self._rc)
        if terminated or truncated:
            info["final_distance"] = float(np.linalg.norm(self.active_target() - self.tcp_pos))
            info["final_phase"] = int(self.phase)
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ------------------------------------------------------------------ #
    # State turunan (kinematika, kontak, clearance)
    # ------------------------------------------------------------------ #
    def _update_derived_state(self):
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)          # xpos/site_xpos sesuai qpos TERBARU
        mujoco.mj_comPos(m, d)
        self._manip = self.ik_solver.manipulability_from_data(d)

        self._contact_reason = None
        if d.ncon > 0:
            floor = False
            for i in range(d.ncon):
                c = d.contact[i]
                if m.geom_bodyid[c.geom1] == 0 or m.geom_bodyid[c.geom2] == 0:
                    floor = True
                    break
            self._contact_reason = "floor_contact" if floor else "self_collision"

        if self.obstacle_points.shape[0] > 0:
            self._obs_clear, self._obs_witness = obstacle_clearance(
                self.caps, d, self.obstacle_points, self.obstacle_radii)
            self._min_clear = float(np.min(self._obs_clear))
        else:
            self._obs_clear = np.zeros(0)
            self._obs_witness = np.zeros((0, 3))
            self._min_clear = np.inf

    # ------------------------------------------------------------------ #
    # Observasi
    # ------------------------------------------------------------------ #
    def _path_features(self, tcp: np.ndarray):
        if self.T2 is None or self.T1 is None:
            return np.zeros(3), np.zeros(3), 0.0
        seg = self.T2 - self.T1
        direction = seg / (np.linalg.norm(seg) + 1e-9)
        t, proj = project_point_on_segment(tcp, self.T1, self.T2)
        return direction, proj - tcp, t

    def _get_obs(self) -> np.ndarray:
        d = self.data
        tcp = d.site_xpos[self.tcp_site_id]
        rot = d.site_xmat[self.tcp_site_id].reshape(3, 3)
        S = OBS_SLICES
        obs = np.zeros(OBS_DIM, dtype=np.float64)

        target = self.active_target()
        obs[S["rel_target"]] = target - tcp
        obs[S["q"]] = d.qpos[self.arm_qpos_adr]
        obs[S["qd"]] = d.qvel[self.arm_dof_adr]
        obs[S["tcp_pos"]] = tcp - self.base_position
        obs[S["tcp_rot6"]] = np.concatenate([rot[:, 0], rot[:, 1]])
        obs[S["manip"]] = self._manip
        obs[S["prev_action"]] = self._prev_action
        obs[S["time_left"]] = 1.0 - self.step_count / self.max_steps

        if self.T2 is not None:
            direction, ct_vec, prog = self._path_features(tcp)
            obs[S["phase"]] = float(self.phase)
            obs[S["path_dir"]] = direction
            obs[S["cross_track_vec"]] = ct_vec
            obs[S["path_progress"]] = prog

        K = N_NEAREST_OBSTACLES
        rel = np.zeros((K, 3))
        dist = np.full(K, OBS_FAR)
        if self._obs_clear.size > 0:
            order = np.argsort(self._obs_clear)[:K]
            k = len(order)
            rel[:k] = self.obstacle_points[order] - self._obs_witness[order]
            dist[:k] = np.clip(self._obs_clear[order], 0.0, OBS_FAR)
            obs[S["min_clearance"]] = float(np.clip(self._min_clear, 0.0, OBS_FAR))
        else:
            obs[S["min_clearance"]] = OBS_FAR
        obs[S["obs_rel"]] = rel.reshape(-1)
        obs[S["obs_dist"]] = dist

        if self.obs_noise_std > 0:
            noise = self._rng.normal(0.0, self.obs_noise_std, OBS_DIM)
            obs = obs + noise * self._noise_mask
        return obs.astype(np.float32)

    def _get_info(self) -> dict:
        return {
            "T1": None if self.T1 is None else self.T1.copy(),
            "T2": None if self.T2 is None else self.T2.copy(),
            "phase": int(self.phase),
            "n_obstacles": int(self.obstacle_points.shape[0]),
            "payload_mass_extra": float(self._payload_mass_extra),
            "success_threshold": float(self.success_pos_threshold),
            "scenario_ok": bool(self._scenario_ok),
        }

    # ------------------------------------------------------------------ #
    # Helper reward (dipakai subclass)
    # ------------------------------------------------------------------ #
    def _w(self, key: str, default: float = 0.0) -> float:
        return float(self.W.get(key, default))

    def _common_reward_terms(self, dist: float) -> float:
        """time + distance + progress (simetris) + smoothness. Dicatat di self._rc."""
        r_time = self._w("time_penalty")
        r_dist = -abs(self._w("distance_penalty", 0.5)) * dist          # -w_d * ||P_t - P_ee||
        r_prog = self._w("progress_weight") * (self._prev_dist - dist)  # simetris (tidak bias)
        r_smooth = self._smoothness_term()
        self._prev_dist = dist
        self._rc = {"time": r_time, "distance": r_dist, "progress": r_prog, "smooth": r_smooth}
        return r_time + r_dist + r_prog + r_smooth

    def _smoothness_term(self) -> float:
        """-w_s * ||a_t - a_{t-1}||^2 pada aksi ternormalisasi (bukan target sendi).
        (Versi lama membandingkan target sendi (rad) dengan aksi [-1,1] -> tidak bermakna.)"""
        delta = self._cur_action - self._prev_action
        return -abs(self._w("smoothness_penalty", 0.01)) * float(np.sum(delta ** 2))

    def _contact_termination(self):
        """Return (penalty, terminated, reason) untuk kontak self/lantai."""
        if self.terminate_on_contact and self._contact_reason is not None:
            self._rc["contact"] = self._w("contact_penalty", -10.0)
            return self._w("contact_penalty", -10.0), True, self._contact_reason
        return 0.0, False, None
