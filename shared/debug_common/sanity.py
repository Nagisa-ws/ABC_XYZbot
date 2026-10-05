"""
shared/debug_common/sanity.py
-----------------------------
Pemeriksaan sanity SEBELUM training (dipanggil train_missionN.py):

  sanity_check_ik          : IK dapat mencapai titik workspace acak (error posisi kecil)
  sanity_check_env         : reset/step normal, dimensi obs benar, tidak ada NaN,
                             skenario valid, tidak ada kontak di pose home
  scripted_controller_check: controller P lurus-ke-target harus berhasil pada env
                             -> bukti bahwa tugas SOLVABLE & reward/terminasi konsisten
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from shared.debug_common.common import make_predict_fn_scripted, run_policy_episode
from shared.envs.obs_layout import OBS_DIM


def sanity_check_ik(env, n_targets: int = 20, tol: float = 2e-3, log=print) -> dict:
    """IK dijalankan PER WAYPOINT (warm-start) dari home ke titik workspace acak, lewat
    jalur lurus atau polar -- sama seperti validasi skenario env. (Satu lompatan IK dari
    home ke titik jauh memang tidak selalu konvergen; bukan cara IK dipakai di env.)"""
    import mujoco
    ok, errs, t_solve, paths = 0, [], [], []
    for _ in range(n_targets):
        p = env._sample_workspace_point()
        t0 = time.perf_counter()
        good, q, _, name = env.checker.approach(env.home_qpos.copy(), env.home_tcp, p,
                                                env._nominal_quat, env.base_position[:2])
        t_solve.append((time.perf_counter() - t0) / max(1, len(env.checker.last_wps) if good else 1))
        ok += bool(good)
        paths.append(name)
        if good:
            d = env.checker._data
            d.qpos[:] = q
            mujoco.mj_kinematics(env.model, d)
            errs.append(float(np.linalg.norm(d.site_xpos[env.tcp_site_id] - p)))
    res = {"ik_reached": ok, "n": n_targets, "solve_ms": 1000 * float(np.mean(t_solve)),
           "max_err_mm": 1000 * max(errs) if errs else float("nan"),
           "mean_err_mm": 1000 * float(np.mean(errs)) if errs else float("nan"),
           "polar_paths": paths.count("polar")}
    log(f"[sanity-IK] {ok}/{n_targets} target tercapai dari home ({paths.count('lurus')} jalur lurus, "
        f"{paths.count('polar')} polar) | error akhir rata-rata {res['mean_err_mm']:.2f} mm, "
        f"maks {res['max_err_mm']:.2f} mm | ~{res['solve_ms']:.2f} ms/waypoint")
    return res


def sanity_check_env(env, n_resets: int = 10, log=print) -> dict:
    problems = []
    t0 = time.perf_counter()
    scen_ok = 0
    for i in range(n_resets):
        obs, info = env.reset(seed=10_000 + i)
        scen_ok += bool(info["scenario_ok"])
        if obs.shape != (OBS_DIM,):
            problems.append(f"dimensi obs {obs.shape} != ({OBS_DIM},)")
        if not np.all(np.isfinite(obs)):
            problems.append("obs mengandung NaN/Inf")
        if env._contact_reason is not None:
            problems.append(f"kontak di pose awal: {env._contact_reason}")
        obs, r, term, trunc, info = env.step(np.zeros(6))
        if not np.isfinite(r):
            problems.append("reward NaN")
    dt = (time.perf_counter() - t0) / n_resets
    res = {"scenario_ok_rate": scen_ok / n_resets, "reset_step_ms": 1000 * dt,
           "problems": sorted(set(problems))}
    log(f"[sanity-env] skenario valid {scen_ok}/{n_resets} | reset+step {1000 * dt:.1f} ms | "
        f"masalah: {res['problems'] or 'tidak ada'}")
    return res


def scripted_controller_check(env, n_episodes: int = 10, log=print) -> dict:
    fn = make_predict_fn_scripted(env)
    reasons = Counter()
    for i in range(n_episodes):
        obs, _ = env.reset(seed=20_000 + i)
        res = run_policy_episode(env, fn, obs, env.max_steps, 30.0)
        reasons[res.termination_reason] += 1
    rate = reasons.get("success", 0) / n_episodes
    log(f"[sanity-scripted] success {rate * 100:.0f}% ({dict(reasons)}) - controller referensi (jalur tervalidasi)")
    return {"success_rate": rate, "reasons": dict(reasons)}


def run_all_sanity(env_factory, n_ik=20, n_resets=10, n_scripted=10, log=print) -> dict:
    """env_factory() -> env BARU (training=False, randomize=False)."""
    env = env_factory()
    out = {"ik": sanity_check_ik(env, n_ik, log=log),
           "env": sanity_check_env(env, n_resets, log=log),
           "scripted": scripted_controller_check(env, n_scripted, log=log)}
    return out
