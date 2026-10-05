"""
shared/control/benchmark.py
---------------------------
Uji penerimaan: RL saja vs hibrida (RL + LQR) pada skenario TETAP, pada ambang sukses standar (2 cm)
dan ketat (5 mm). Dipakai untuk memastikan orang lain mendapat hasil dalam rentang yang sama dengan
hasil referensi, dan untuk melihat manfaat controller secara adil (skenario berpasangan).

Dijalankan lewat Mission2/benchmark_hybrid_mission2.py dan Mission3/benchmark_hybrid_mission3.py
(tanpa argumen CLI). Parameter di shared/control/controller.yaml (bagian `benchmark`).
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from shared.config_utils import load_yaml
from shared.control.hybrid import HybridPolicy, bind_to_env, load_control_config
from shared.debug_common.common import make_predict_fn_onnx, run_policy_episode
from shared.debug_common.post_training import onnx_session
from shared.evaluation.viewer_eval import resolve_onnx


def _wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def run_benchmark(*, mission_id: int, project_root: Path, env, onnx_path: Path,
                  control_cfg_path=None, csv_path: Path | None = None, log=print) -> list[dict]:
    cfg = load_control_config(control_cfg_path)
    b = cfg.get("benchmark", {})
    n, seed0 = int(b.get("n_scenarios", 100)), int(b.get("seed_start", 20000))
    thresholds = [float(t) for t in b.get("thresholds", [0.02, 0.005])]
    if abs(float(env.max_pos_delta) - float(cfg["controller"]["max_pos_delta"])) > 1e-9:
        raise ValueError(f"max_pos_delta env ({env.max_pos_delta}) != controller.yaml "
                         f"({cfg['controller']['max_pos_delta']}); samakan keduanya.")
    ocfg = cfg.get("orientation", {})
    if abs(float(env.max_rot_delta) - float(ocfg.get("max_rot_delta", env.max_rot_delta))) > 1e-9:
        raise ValueError("max_rot_delta env != controller.yaml (orientation.max_rot_delta); samakan keduanya.")
    gate = float(ocfg.get("gate_angle_deg", 60.0))
    default_cone = float(env.max_orient_dev)
    sess = onnx_session(str(onnx_path))

    def rl_fn(o):
        return np.asarray(sess.run(["action"], {"observation": np.asarray(o, dtype=np.float32)[None, :]})[0][0])

    policies = {
        "RL saja": rl_fn,
        "RL + LQR (posisi)": bind_to_env(HybridPolicy(rl_fn, cfg, orientation=False), env),
        "RL + LQR + orientasi": bind_to_env(HybridPolicy(rl_fn, cfg, orientation=True), env),
    }
    log(f"[benchmark] Misi {mission_id} | ONNX: {Path(onnx_path).name} | {n} skenario tetap (seed {seed0}+) "
        f"| serah-terima posisi {cfg['hybrid']['handover_distance'] * 100:.0f} cm; orientasi hanya jalur menurun "
        f"curam (< {gate:.0f} deg dari 'lurus ke bawah'), mulai {ocfg.get('start_distance', 0.4) * 100:.0f} cm")
    rows, outcome = [], {}
    for thr in thresholds:
        env.set_success_threshold(thr)
        env.cfg.setdefault("path_task", {})["t1_reach_threshold"] = thr
        for name, pol in policies.items():
            succ = []
            for i in range(n):
                env.max_orient_dev = default_cone
                if hasattr(pol, "reset"):
                    pol.reset()
                obs, _ = env.reset(seed=seed0 + i)
                d = (env.T2 - env.T1) / np.linalg.norm(env.T2 - env.T1)
                path_angle = float(np.degrees(np.arccos(np.clip(-d[2], -1.0, 1.0))))
                nose = []

                def _cb(e, info, nose=nose, d=d):
                    if e.phase == 1:
                        z = e.data.site_xmat[e.tcp_site_id].reshape(3, 3)[:, 2]
                        nose.append(float(np.degrees(np.arccos(np.clip(z @ d, -1.0, 1.0)))))
                    return True

                res = run_policy_episode(env, pol, obs, env.max_steps, 60.0, step_callback=_cb)
                reason = res.termination_reason or "timeout"
                ct = res.info.get("cross_track_mean", float("nan"))
                nz = np.array(nose)
                rows.append({"mission": mission_id, "threshold_mm": thr * 1000, "policy": name, "seed": seed0 + i,
                             "reason": reason, "steps": res.n_steps, "path_angle_from_down_deg": round(path_angle, 1),
                             "steep_descent": int(path_angle < gate),
                             "cross_track_mean_mm": round(1000 * float(ct), 2) if np.isfinite(ct) else "",
                             "nose_err_mean_deg": round(float(nz.mean()), 1) if nz.size else "",
                             "nose_lt10_frac": round(float(np.mean(nz < 10)), 3) if nz.size else ""})
                succ.append(reason == "success")
            outcome[(thr, name)] = np.array(succ)
        env.max_orient_dev = default_cone
    # ------------- ringkasan -------------
    log("\n" + "=" * 112)
    for thr in thresholds:
        log(f"--- ambang T1/T2 = {thr * 1000:.0f} mm ---")
        log(f"{'policy':22s} | {'sukses':>7} {'(CI 95%)':>12} | tab timeout self floor | langkah | cross-track | "
            f"jalur menurun curam: n, sukses, moncong rata2, <10 deg")
        for name in policies:
            sub = [r for r in rows if r["threshold_mm"] == thr * 1000 and r["policy"] == name]
            c = Counter(r["reason"] for r in sub)
            k = c["success"]
            lo, hi = _wilson(k, len(sub))
            cts = [r["cross_track_mean_mm"] for r in sub if r["cross_track_mean_mm"] != ""]
            st = [r for r in sub if r["steep_descent"]]
            nz = [r["nose_err_mean_deg"] for r in st if r["nose_err_mean_deg"] != ""]
            l10 = [r["nose_lt10_frac"] for r in st if r["nose_lt10_frac"] != ""]
            log(f"{name:22s} | {k / len(sub):6.1%} ({lo:4.0%}-{hi:4.0%}) | {c['collision']:3d} {c['timeout']:7d} "
                f"{c['self_collision']:4d} {c['floor_contact']:5d} | {np.mean([r['steps'] for r in sub]):7.1f} | "
                f"{np.mean(cts) if cts else float('nan'):6.1f} mm | n={len(st):2d} "
                f"{np.mean([r['reason'] == 'success' for r in st]) if st else float('nan'):5.0%} "
                f"{np.mean(nz) if nz else float('nan'):6.1f} deg {np.mean(l10) if l10 else float('nan'):5.0%}")
        base = outcome[(thr, "RL + LQR (posisi)")]
        ori = outcome[(thr, "RL + LQR + orientasi")]
        log(f"{'':22s}   berpasangan (orientasi vs posisi saja): orientasi menang {int((~base & ori).sum())}, "
            f"posisi saja menang {int((base & ~ori).sum())}")
    log("=" * 112)
    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log(f"[benchmark] CSV -> {csv_path}")
    return rows


def run_benchmark_for_mission(*, mission_id: int, project_root: Path, env_cls: type, eval_cfg_path: str,
                              log=print):
    root = Path(project_root)
    ecfg = load_yaml(eval_cfg_path)
    onnx_path = resolve_onnx(root, ecfg["paths"])
    env = env_cls(seed=1, randomize=False, obs_noise_std=0.0)
    return run_benchmark(mission_id=mission_id, project_root=root, env=env, onnx_path=onnx_path,
                         csv_path=root / "checkpoints" / f"mission{mission_id}" / "hybrid_benchmark.csv", log=log)
