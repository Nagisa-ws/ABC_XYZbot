"""
shared/evaluation/viewer_eval.py
--------------------------------
Evaluasi policy ONNX dengan MuJoCo viewer (dipakai eval_mission1/2/3.py).

  * Env evaluasi (eval_envs/) = env training + scene_eval.xml; observasi dibangun
    oleh kelas env yang SAMA dengan training -> tidak ada mismatch observasi.
  * Marker di viewer: T1 (hijau), T2 (merah), garis T1-T2, rintangan (bola) +
    zona aman, jejak end-effector.
  * Headless otomatis bila viewer dimatikan di YAML / tidak ada layar.
  * Hasil: ringkasan di terminal, CSV per skenario, plot replay terbaik/terburuk.
"""
from __future__ import annotations

import contextlib
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np
import mujoco

from shared.config_utils import load_yaml
from shared.debug_common.common import plot_replay_trajectory, run_policy_episode
from shared.debug_common.post_training import onnx_session, onnx_vs_checkpoint
from shared.debug_common.common import make_predict_fn_onnx
from shared.exporter import latest_versioned_onnx_path

_I3 = np.eye(3).flatten()
_GREEN = np.array([0.1, 0.9, 0.2, 1.0], np.float32)
_RED = np.array([0.95, 0.15, 0.15, 1.0], np.float32)
_ORANGE = np.array([1.0, 0.55, 0.0, 0.95], np.float32)
_ZONE = np.array([1.0, 0.55, 0.0, 0.10], np.float32)
_LINE = np.array([0.1, 0.1, 0.1, 0.8], np.float32)
_TRAIL = np.array([0.1, 0.4, 1.0, 0.9], np.float32)


# ---------------------------------------------------------------------- #
class MarkerScene:
    """Menulis geom dekorasi ke `scene.geoms` (viewer.user_scn atau MjvScene)."""
    MAX_TRAIL = 600

    def __init__(self, scn, show_trail=True, draw_obstacles=True, draw_zone=True):
        self.scn = scn
        self.show_trail, self.draw_obstacles, self.draw_zone = show_trail, draw_obstacles, draw_zone
        self.n_static = 0
        self.n_trail = 0

    def _sphere(self, pos, r, rgba):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        g = self.scn.geoms[self.scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                            np.asarray(pos, float), _I3, rgba)
        self.scn.ngeom += 1

    def _line(self, p0, p1, width, rgba):
        if self.scn.ngeom >= self.scn.maxgeom:
            return
        g = self.scn.geoms[self.scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.zeros(9), rgba)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(p0, float), np.asarray(p1, float))
        self.scn.ngeom += 1

    def set_episode(self, env):
        self.scn.ngeom = 0
        if env.T1 is not None:
            self._sphere(env.T1, 0.018, _GREEN)
        if env.T2 is not None:
            self._sphere(env.T2, 0.018, _RED)
            self._line(env.T1, env.T2, 0.004, _LINE)
        if self.draw_obstacles and env.obstacle_points.shape[0]:
            d_safe = float(getattr(env, "_d_safe", 0.0))
            for p, r in zip(env.obstacle_points, env.obstacle_radii):
                self._sphere(p, float(r), _ORANGE)
                if self.draw_zone and d_safe > 0:
                    self._sphere(p, float(r) + d_safe, _ZONE)
        self.n_static, self.n_trail = self.scn.ngeom, 0

    def add_trail(self, p):
        if self.show_trail and self.n_trail < self.MAX_TRAIL:
            self.scn.ngeom = self.n_static + self.n_trail
            self._sphere(p, 0.005, _TRAIL)
            self.n_trail += 1


class _HeadlessViewer:
    """Pengganti viewer (tanpa layar) - memakai MjvScene nyata agar kode marker tetap teruji."""

    def __init__(self, model):
        self.user_scn = mujoco.MjvScene(model, maxgeom=2000)

    def is_running(self): return True
    def sync(self): pass
    def close(self): pass

    @contextlib.contextmanager
    def lock(self):
        yield


class _LockedEnv:
    """Proxy: step()/reset() dijalankan di bawah viewer.lock() (thread-safe)."""

    def __init__(self, env, viewer):
        self._env, self._viewer = env, viewer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, a):
        with self._viewer.lock():
            return self._env.step(a)

    def reset(self, **kw):
        with self._viewer.lock():
            return self._env.reset(**kw)


def _display_available() -> bool:
    """GLFW yang gagal init memanggil exit() di level C (tidak bisa ditangkap try/except),
    jadi ketersediaan layar diperiksa dulu. macOS/Windows selalu dianggap punya layar."""
    import os
    import sys
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def open_viewer(model, data, vcfg: dict, log=print):
    if not vcfg.get("enabled", True):
        return _HeadlessViewer(model), False
    if not _display_available():
        log("[eval] tidak ada layar (DISPLAY kosong) -> mode headless")
        return _HeadlessViewer(model), False
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data), True
    except Exception as e:                                        # tidak ada display / GLFW
        log(f"[eval] viewer tidak bisa dibuka ({type(e).__name__}: {e}) -> mode headless")
        return _HeadlessViewer(model), False


# ---------------------------------------------------------------------- #
def resolve_onnx(root: Path, pcfg: dict) -> Path:
    p = pcfg.get("onnx_model", "latest")
    if str(p).lower() == "latest":
        found = latest_versioned_onnx_path(str(root / pcfg["onnx_dir"]), pcfg["onnx_prefix"])
        if found is None:
            raise FileNotFoundError(
                f"Belum ada {pcfg['onnx_prefix']}_v*.onnx di {root / pcfg['onnx_dir']}. "
                f"Jalankan training misi ini dulu (python train_missionN.py).")
        return found
    path = root / p if not Path(p).is_absolute() else Path(p)
    if not path.exists():
        raise FileNotFoundError(f"ONNX tidak ditemukan: {path}")
    return path


def run_evaluation(*, mission_id: int, project_root: Path, eval_cfg_path: str, env_cls: type,
                   log=print) -> dict:
    root = Path(project_root)
    cfg = load_yaml(eval_cfg_path)
    pc, rc, vc, dc = cfg["paths"], cfg["rollout"], cfg.get("viewer", {}), cfg.get("debug", {})
    onnx_path = resolve_onnx(root, pc)
    log(f"[eval] Misi {mission_id} | ONNX: {onnx_path.name} | model: {pc['robot_model_path']}")

    env = env_cls(model_path=str(root / pc["robot_model_path"]), config_path=str(root / pc["env_config"]),
                  seed=int(rc.get("seed_offset", 1000)), randomize=bool(rc.get("randomize", False)),
                  obs_noise_std=float(rc.get("obs_noise_std", 0.0)),
                  success_pos_threshold=rc.get("success_pos_threshold"))
    if rc.get("t1_reach_threshold") is not None:           # ambang T1 (Misi 2/3); null -> ikut env_config
        env.cfg.setdefault("path_task", {})["t1_reach_threshold"] = float(rc["t1_reach_threshold"])
    hy = cfg.get("hybrid", {}) or {}
    if hy.get("enabled", False):                            # RL + controller LQR (shared/control)
        from shared.control import bind_to_env, make_hybrid_from_onnx
        predict = bind_to_env(make_hybrid_from_onnx(onnx_path, hy.get("config"),
                                                    orientation=hy.get("orientation")), env)
        log("[eval] mode HIBRIDA: RL + controller LQR"
            + (" + penyejajaran moncong (jalur menurun curam)" if hy.get("orientation") else "")
            + " (parameter di shared/control/controller.yaml)")
    else:
        predict = make_predict_fn_onnx(onnx_session(str(onnx_path)))

    viewer, real = open_viewer(env.model, env.data, vc, log)
    ov = cfg.get("obstacles_view", {})
    markers = MarkerScene(viewer.user_scn, bool(vc.get("show_trail", True)),
                          bool(ov.get("draw_obstacles", True)), bool(ov.get("draw_safety_zone", True)))
    lenv = _LockedEnv(env, viewer)
    dt_ctrl = env.n_substeps * env.model.opt.timestep
    rtf = float(vc.get("realtime_factor", 1.0))

    def _on_step(e, info):
        markers.add_trail(e.tcp_pos)
        viewer.sync()
        if rtf > 0:
            time.sleep(dt_ctrl / rtf)
        return viewer.is_running()

    rows, results = [], []
    n = int(rc["n_scenarios"])
    t_start = time.time()
    for i in range(n):
        if not viewer.is_running():
            log("[eval] viewer ditutup - evaluasi dihentikan.")
            break
        seed = int(rc.get("seed_offset", 1000)) + i
        if hasattr(predict, "reset"):                       # policy hibrida punya state per episode
            predict.reset()
        obs, info0 = lenv.reset(seed=seed)
        with viewer.lock():
            markers.set_episode(env)
        viewer.sync()
        res = run_policy_episode(lenv, predict, obs, env.max_steps,
                                 float(rc.get("max_seconds_per_scenario", 60)), _on_step)
        results.append(res)
        inf = res.info
        path_len = float(np.sum(np.linalg.norm(np.diff(res.trajectory, axis=0), axis=1)))
        row = {"scenario": i, "seed": seed, "success": int(res.termination_reason == "success"),
               "reason": res.termination_reason, "steps": res.n_steps,
               "sim_time_s": round(res.n_steps * dt_ctrl, 3),
               "final_distance_mm": round(1000 * float(inf.get("final_distance", np.nan)), 2),
               "path_length_m": round(path_len, 4),
               "final_phase": int(inf.get("final_phase", 0)),
               "cross_track_mean_mm": round(1000 * float(inf["cross_track_mean"]), 2)
               if "cross_track_mean" in inf else "",
               "cross_track_max_mm": round(1000 * float(inf["cross_track_max"]), 2)
               if "cross_track_max" in inf else "",
               "min_clearance_mm": round(1000 * float(res.clearance.min()), 2) if res.clearance.size else "",
               "n_obstacles": int(info0.get("n_obstacles", 0)),
               "payload_extra_kg": round(float(info0.get("payload_mass_extra", 0.0)), 3),
               "scenario_ok": int(info0.get("scenario_ok", True))}
        rows.append(row)
        log(f"[eval] skenario {i + 1}/{n} seed={seed}: {res.termination_reason:<14} "
            f"steps={res.n_steps:<4} final={row['final_distance_mm']:.1f} mm"
            + (f" | cross-track mean={row['cross_track_mean_mm']} mm" if row["cross_track_mean_mm"] != "" else "")
            + (f" | min clearance={row['min_clearance_mm']} mm" if row["min_clearance_mm"] != "" else ""))
        time.sleep(float(vc.get("pause_between_scenarios_sec", 0.0)) if real else 0.0)

    summary = summarize(rows, log)
    summary["wall_sec"] = time.time() - t_start

    # ---- CSV ----
    if rows:
        csv_path = root / pc["log_csv"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"[eval] log CSV -> {csv_path}")

    # ---- debug ----
    if dc.get("enabled", True) and results:
        out = root / pc["debug_dir"]
        if dc.get("replay_best_episode", True):
            ok = [k for k, r in enumerate(rows) if r["success"]]
            best = min(ok, key=lambda k: rows[k]["steps"]) if ok else int(np.argmin(
                [r["final_distance_mm"] for r in rows]))
            worst = max(range(len(rows)), key=lambda k: (1 - rows[k]["success"], rows[k]["final_distance_mm"]))
            for tag, k in (("best", best), ("worst", worst)):
                plot_replay_trajectory(results[k], str(out / f"replay_{tag}.png"),
                                       title=f"Misi {mission_id} skenario {k} ({tag})")
            log(f"[eval] plot replay -> {out}")
        if dc.get("compare_onnx_checkpoint", False):
            try:
                onnx_vs_checkpoint(env_cls(model_path=str(root / pc["robot_model_path"]),
                                           config_path=str(root / pc["env_config"]), seed=1),
                                   str(onnx_path), str(out / "onnx_vs_checkpoint.png"), log=log)
            except Exception as e:
                log(f"[eval] perbandingan ONNX-checkpoint dilewati: {type(e).__name__}: {e}")

    if real and vc.get("keep_open_when_done", True):
        log("[eval] selesai. Tutup jendela viewer untuk keluar.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)
    viewer.close()
    summary["rows"] = rows
    return summary


def summarize(rows: list[dict], log=print) -> dict:
    if not rows:
        return {}
    n = len(rows)
    succ = sum(r["success"] for r in rows)
    reasons = Counter(r["reason"] for r in rows)
    s = {"n": n, "success_rate": succ / n, "reasons": dict(reasons),
         "mean_steps": float(np.mean([r["steps"] for r in rows])),
         "mean_final_distance_mm": float(np.mean([r["final_distance_mm"] for r in rows]))}
    ct = [r["cross_track_mean_mm"] for r in rows if r["cross_track_mean_mm"] != ""]
    if ct:
        s["cross_track_mean_mm"] = float(np.nanmean(ct))
    cl = [r["min_clearance_mm"] for r in rows if r["min_clearance_mm"] != ""]
    if cl:
        s["min_clearance_mm"] = float(np.min(cl))
    log("\n" + "=" * 60)
    log(f"RINGKASAN EVALUASI: {succ}/{n} sukses ({100 * s['success_rate']:.0f}%)  alasan={dict(reasons)}")
    log(f"  langkah rata-rata {s['mean_steps']:.0f} | jarak akhir rata-rata {s['mean_final_distance_mm']:.1f} mm"
        + (f" | cross-track rata-rata {s['cross_track_mean_mm']:.1f} mm" if ct else "")
        + (f" | clearance minimum {s['min_clearance_mm']:.1f} mm" if cl else ""))
    log("=" * 60)
    return s
