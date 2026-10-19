"""
shared/debug_common/common.py
-----------------------------
Alat debug bersama lintas misi:
  * Rollout episode yang DIJAMIN berhenti (loop terbatas + watchdog) dan merekam
    lintasan, fase, cross-track, clearance rintangan.
  * Progress tracker (ETA, SPS).
  * Pembaca TensorBoard + plot kurva training, replay 3D, ONNX-vs-checkpoint,
    robustness terhadap domain randomization.
Semua plot headless (backend Agg).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------- #
def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class ProgressTracker:
    def __init__(self, total_units: int, label: str = "progress"):
        self.total_units = max(1, total_units)
        self.label = label
        self.start_time = time.time()

    def report(self, done_units: int) -> str:
        elapsed = time.time() - self.start_time
        frac = min(1.0, done_units / self.total_units)
        rate = done_units / elapsed if elapsed > 0 else 0.0
        eta = (self.total_units - done_units) / rate if rate > 0 else float("inf")
        eta_s = format_duration(eta) if np.isfinite(eta) else "?"
        return (f"[{self.label}] {done_units}/{self.total_units} ({frac * 100:.1f}%) "
                f"| elapsed={format_duration(elapsed)} | ETA={eta_s}")


# ---------------------------------------------------------------------- #
# Prediksi
# ---------------------------------------------------------------------- #
def make_predict_fn_onnx(session):
    """Observasi MENTAH -> aksi (normalisasi sudah ada di dalam graph ONNX)."""
    return lambda obs: session.run(["action"], {"observation": obs[None, :].astype(np.float32)})[0][0]


def make_predict_fn_sb3(model, obs_mean=None, obs_var=None, clip_obs=10.0, epsilon=1e-8):
    """Observasi MENTAH -> aksi; normalisasi VecNormalize diterapkan manual."""
    if obs_mean is None:
        return lambda obs: model.predict(obs, deterministic=True)[0]
    std = np.sqrt(obs_var + epsilon)
    return lambda obs: model.predict(np.clip((obs - obs_mean) / std, -clip_obs, clip_obs),
                                     deterministic=True)[0]


def make_predict_fn_random(env, scale: float = 0.3):
    return lambda obs: env.action_space.sample() * scale


def make_predict_fn_scripted(env, gain: float = 0.8, lookahead: int = 2):
    """Controller referensi (bukti 'solvable').

    Fase reaching: ikuti jalur pendekatan YANG DIVALIDASI env (lurus atau polar mengitari
    base) dengan waypoint terdekat + lookahead. Fase path-following: lurus ke T2.
    Bukan policy RL - hanya memastikan reward/terminasi/skenario konsisten dan tugas layak."""
    def _fn(obs):
        tcp = env.tcp_pos
        tgt = env.active_target()
        wps = getattr(env, "ref_approach_wps", None)
        if env.phase == 0 and wps is not None and len(wps) > 0:
            i = int(np.argmin(np.linalg.norm(wps - tcp, axis=1)))
            tgt = wps[min(i + lookahead, len(wps) - 1)]
        v = (tgt - tcp) / env.max_pos_delta * gain
        n = np.linalg.norm(v)
        a = np.zeros(6)
        a[:3] = v if n <= 1.0 else v / n
        return a
    return _fn


# ---------------------------------------------------------------------- #
# Rollout
# ---------------------------------------------------------------------- #
@dataclass
class EpisodeResult:
    trajectory: np.ndarray                 # (T+1,3) posisi TCP
    dist_to_target: np.ndarray             # (T+1,)  jarak ke target aktif
    termination_reason: str | None
    n_steps: int
    phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cross_track: np.ndarray = field(default_factory=lambda: np.zeros(0))
    clearance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    T1: np.ndarray | None = None
    T2: np.ndarray | None = None
    obstacles: tuple | None = None         # (points (M,3), radii (M,))
    info: dict = field(default_factory=dict)
    actions: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))   # aksi (terpotong [-1,1]) per langkah
    tilt_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))       # kemiringan gripper thd pose nominal


def run_policy_episode(env, predict_fn: Callable[[np.ndarray], np.ndarray], obs: np.ndarray,
                       max_steps: int, wall_clock_timeout_sec: float = 60.0,
                       step_callback: Callable | None = None) -> EpisodeResult:
    """Jalankan SATU episode dari state env saat ini (setelah reset).

    step_callback(env, info) dipanggil setelah tiap step (mis. sinkron viewer);
    bila mengembalikan False episode dihentikan (alasan 'viewer_closed')."""
    traj = [env.tcp_pos]
    dist = [float(np.linalg.norm(env.active_target() - env.tcp_pos))]
    phase, ct, clr, acts, tilt = [], [], [], [], []
    reason, info = None, {}
    t0 = time.time()
    n = 0
    for n in range(1, max_steps + 1):
        if time.time() - t0 > wall_clock_timeout_sec:
            reason = "watchdog_timeout"
            break
        act = np.clip(np.asarray(predict_fn(obs), dtype=np.float64).reshape(-1), -1.0, 1.0)
        obs, _, term, trunc, info = env.step(act)
        acts.append(act)
        tilt.append(float(np.degrees(getattr(env, "_tilt", np.nan))))
        traj.append(env.tcp_pos)
        dist.append(float(np.linalg.norm(env.active_target() - env.tcp_pos)))
        phase.append(int(env.phase))
        if "cross_track" in info:          # hanya bermakna di fase B; fase A -> NaN (panjang tetap sejajar)
            ct.append(float(info["cross_track"]) if int(env.phase) == 1 else float("nan"))
        if "min_clearance" in info:
            clr.append(float(info["min_clearance"]))
        if step_callback is not None and (term or trunc):
            step_callback(env, info)
        if term or trunc:
            reason = info.get("termination_reason") or "timeout"
            break
        if step_callback is not None and step_callback(env, info) is False:
            reason = "viewer_closed"
            break
    reason = reason or "timeout"
    return EpisodeResult(
        trajectory=np.array(traj), dist_to_target=np.array(dist), termination_reason=reason,
        n_steps=n, phase=np.array(phase), cross_track=np.array(ct), clearance=np.array(clr),
        T1=None if env.T1 is None else env.T1.copy(),
        T2=None if env.T2 is None else env.T2.copy(),
        obstacles=(env.obstacle_points.copy(), env.obstacle_radii.copy())
        if env.obstacle_points.shape[0] else None,
        info=info, actions=np.array(acts).reshape(-1, 6), tilt_deg=np.array(tilt))


# ---------------------------------------------------------------------- #
# TensorBoard
# ---------------------------------------------------------------------- #
def _read_one_run(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
    acc.Reload()
    out = {}
    for tag in acc.Tags().get("scalars", []):
        ev = acc.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def read_tb_scalars(logdir: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Skalar run TensorBoard TERBARU di `logdir` -> {tag: (steps, values)}.

    Run sebelumnya ikut digabung HANYA bila langkahnya bersambung (training dilanjutkan dengan `resume`,
    langkah run baru mulai setelah langkah terakhir run lama). Training ulang dari nol (langkah tumpang
    tindih) tidak digabung."""
    p = Path(logdir)
    runs = sorted([d for d in p.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime) if p.exists() else []
    if not runs:
        return _read_one_run(p)
    chain = [_read_one_run(runs[-1])]
    first_step = lambda sc: min((float(v[0].min()) for v in sc.values() if len(v[0])), default=0.0)
    last_step = lambda sc: max((float(v[0].max()) for v in sc.values() if len(v[0])), default=-1.0)
    for d in reversed(runs[:-1]):
        prev = _read_one_run(d)
        if prev and first_step(chain[0]) > 0 and last_step(prev) < first_step(chain[0]):
            chain.insert(0, prev)
        else:
            break
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sc in chain:
        for tag, (x, y) in sc.items():
            if tag in out:
                out[tag] = (np.concatenate([out[tag][0], x]), np.concatenate([out[tag][1], y]))
            else:
                out[tag] = (x, y)
    return out


def _no_data(ax, msg: str):
    """Tulis keterangan di panel yang tidak punya data (bukan kotak kosong tanpa penjelasan)."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="0.4",
            wrap=True)


def _plot_tag(ax, sc, tag, title, color, ylim=None, label=None, empty_msg=None):
    has = tag in sc
    if has:
        ax.plot(*sc[tag], color=color, label=label)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("timesteps")
    ax.grid(alpha=0.3)
    if ylim:
        ax.set_ylim(*ylim)
    if not has and empty_msg:
        _no_data(ax, empty_msg)
    return has


def plot_training_curves(logdir: str, out_path: str, title: str = ""):
    """Kurva training 3x3 dari log TensorBoard.

    Baris 1: reward, panjang episode, success (training + eval)
    Baris 2: entropy loss, value loss, alasan terminasi (+ threshold/level kurikulum)
    Baris 3: kualitas lintasan - rasio panjang jalur fase A, putaran/kemiringan gripper, kehalusan aksi
    """
    sc = read_tb_scalars(logdir)
    fig, axes = plt.subplots(3, 3, figsize=(16, 11.5))
    old_log = "tidak ada di log ini\n(log dari versi kode lama)"
    _plot_tag(axes[0, 0], sc, "rollout/ep_rew_mean", "Reward rata-rata / episode", "tab:blue")
    _plot_tag(axes[0, 1], sc, "rollout/ep_len_mean", "Panjang episode (langkah)", "tab:orange")
    _plot_tag(axes[0, 2], sc, "rollout/success_rate", "Success rate", "tab:green", (0, 1.02), label="training")
    if "eval/success_rate" in sc:
        axes[0, 2].plot(*sc["eval/success_rate"], color="tab:olive", ls="--", label="eval")
    if "rollout/success_rate" in sc or "eval/success_rate" in sc:
        axes[0, 2].legend(fontsize=8)
    _plot_tag(axes[1, 0], sc, "train/entropy_loss", "Entropy loss", "tab:purple")
    _plot_tag(axes[1, 1], sc, "train/value_loss", "Value loss", "tab:red")

    ax = axes[1, 2]
    drawn = False
    for r, c in [("success", "green"), ("timeout", "gray"), ("collision", "red"),
                 ("self_collision", "brown"), ("floor_contact", "black")]:
        # callbacks.py mencatat "reasons/<alasan>_rate"; versi lama memakai "failure_breakdown/". Baca keduanya.
        t = next((f"{pre}/{r}_rate" for pre in ("reasons", "failure_breakdown") if f"{pre}/{r}_rate" in sc), None)
        if t is not None:
            ax.plot(*sc[t], color=c, label=r)
            drawn = True
    for tag, lab in (("curriculum/threshold_m", "threshold (m)"), ("curriculum/level", "level kurikulum")):
        if tag in sc:
            ax2 = ax.twinx()
            ax2.step(*sc[tag], color="tab:blue", ls=":", where="post")
            ax2.set_ylabel(lab, color="tab:blue")
            break
    ax.set_title("Alasan terminasi episode", fontsize=10)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("timesteps")
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=8)
    else:
        _no_data(ax, "tidak ada data alasan terminasi")

    # ---- baris 3: kualitas lintasan ----
    ax = axes[2, 0]
    h1 = _plot_tag(ax, sc, "rollout/path_ratio_A", "Rasio panjang jalur fase A (1 = lurus)", "tab:cyan",
                   label="training")
    if "eval/path_ratio_A" in sc:
        ax.plot(*sc["eval/path_ratio_A"], color="tab:blue", ls="--", label="eval")
    if h1 or "eval/path_ratio_A" in sc:
        ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
        ax.legend(fontsize=8)
    else:
        _no_data(ax, "belum ada data: rasio hanya dihitung untuk episode\nyang mencapai T1 (atau log dari versi kode lama)")
    ax = axes[2, 1]
    h2 = _plot_tag(ax, sc, "rollout/rot_travel_deg", "Putaran gripper / episode (deg)", "tab:brown",
                   label="total putaran")
    h3 = "rollout/tilt_mean_deg" in sc
    if h3:
        ax.plot(*sc["rollout/tilt_mean_deg"], color="tab:pink", label="kemiringan rata-rata")
    if h2 or h3:
        ax.legend(fontsize=8)
    else:
        _no_data(ax, old_log)
    _plot_tag(axes[2, 2], sc, "rollout/action_jerk", "Perubahan aksi / langkah  mean||a_t - a_(t-1)||^2",
              "tab:gray", empty_msg=old_log)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return sc


# ---------------------------------------------------------------------- #
# Plot replay / diagnosis
# ---------------------------------------------------------------------- #
def _draw_sphere(ax, c, r, color="orange", alpha=0.35):
    u, v = np.mgrid[0:2 * np.pi:12j, 0:np.pi:8j]
    ax.plot_surface(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v),
                    c[2] + r * np.cos(v), color=color, alpha=alpha, linewidth=0)


def plot_replay_trajectory(res: EpisodeResult, out_path: str, title: str = ""):
    """Replay satu episode: lintasan 3D, jarak ke target, cross-track/clearance, aksi & kemiringan gripper."""
    fig = plt.figure(figsize=(21, 5.6))
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    tr = res.trajectory
    if res.phase.size:
        ph = np.concatenate([[0], res.phase])
        for p, col in [(0, "tab:blue"), (1, "tab:cyan")]:
            m = ph == p
            if m.any():
                ax.plot(tr[m, 0], tr[m, 1], tr[m, 2], ".", c=col, ms=3, label=f"EE fase {'AB'[p]}")
    else:
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], c="tab:blue", marker=".", ms=3, label="EE")
    ax.scatter(*tr[0], c="k", s=30, marker="o", label="awal")
    if res.T1 is not None:
        ax.plot(*zip(tr[0], res.T1), c="0.6", ls=":", lw=1, label="garis lurus awal-T1")
        ax.scatter(*res.T1, c="limegreen", s=120, marker="*", label="T1")
    if res.T2 is not None:
        ax.scatter(*res.T2, c="red", s=120, marker="*", label="T2")
        ax.plot(*zip(res.T1, res.T2), c="k", ls="--", lw=1, label="garis T1-T2")
    if res.obstacles is not None:
        for c, r in zip(*res.obstacles):
            _draw_sphere(ax, c, r)
    pts = [tr] + [np.array([p]) for p in (res.T1, res.T2) if p is not None]
    allp = np.vstack(pts)
    mid, span = allp.mean(0), max(np.ptp(allp, axis=0).max(), 0.3) / 2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)
    pq = []
    if "path_ratio_A" in res.info:
        pq.append(f"rasio jalur {res.info['path_ratio_A']:.2f}")
    if "rot_travel_deg" in res.info:
        pq.append(f"putaran gripper {res.info['rot_travel_deg']:.0f} deg")
    ax.set_title(f"{title} (akhir={res.termination_reason})" + ("\n" + ", ".join(pq) if pq else ""), fontsize=10)
    ax.legend(fontsize=7)

    a2 = fig.add_subplot(1, 4, 2)
    a2.plot(res.dist_to_target)
    a2.set_xlabel("step"); a2.set_ylabel("jarak ke target aktif (m)")
    a2.set_title("Jarak EE -> target aktif"); a2.grid(alpha=0.3)

    a3 = fig.add_subplot(1, 4, 3)
    has_ct = res.cross_track.size and np.isfinite(res.cross_track).any()
    if has_ct:
        a3.plot(res.cross_track * 1000, label="cross-track fase B (mm)", c="tab:red")
    if res.clearance.size:
        a3.plot(np.clip(res.clearance, 0, 0.5) * 1000, label="min clearance rintangan (mm)", c="tab:green")
    a3.set_xlabel("step"); a3.grid(alpha=0.3)
    if a3.lines:
        a3.legend(fontsize=8)
    if not has_ct and not res.clearance.size:
        _no_data(a3, "Misi 1: tidak ada jalur T1-T2" if res.T2 is None else
                 "T1 tidak tercapai:\nfase B (penyusuran T1-T2) tidak pernah dimulai")
    elif not has_ct and res.T2 is not None:
        a3.text(0.02, 0.97, "fase B tidak dimulai (T1 tidak tercapai)", transform=a3.transAxes, va="top",
                fontsize=8, color="0.4")
    a3.set_title("Cross-track & clearance")

    a4 = fig.add_subplot(1, 4, 4)
    if res.actions.size:
        a4.plot(np.linalg.norm(res.actions[:, :3], axis=1), c="tab:blue", lw=1, label="|aksi posisi|")
        a4.plot(np.linalg.norm(res.actions[:, 3:], axis=1), c="tab:orange", lw=1, label="|aksi rotasi|")
        a4.set_ylabel("besar aksi (maks sqrt(3))")
        a4.set_ylim(0, 1.8)
        if res.tilt_deg.size and np.isfinite(res.tilt_deg).any():
            a5 = a4.twinx()
            a5.plot(res.tilt_deg, c="tab:purple", lw=1, ls="--", label="kemiringan gripper (deg)")
            a5.set_ylabel("kemiringan gripper (deg)", color="tab:purple")
            a5.set_ylim(0, 35)
            h1, l1 = a4.get_legend_handles_labels()
            h2, l2 = a5.get_legend_handles_labels()
            a4.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
        else:
            a4.legend(fontsize=8)
    else:
        _no_data(a4, "tidak ada data aksi")
    a4.set_xlabel("step"); a4.grid(alpha=0.3)
    a4.set_title("Aksi & kemiringan gripper per langkah")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_randomization_robustness(records: list[dict], out_path: str):
    """records: [{success, payload_mass_extra, gravity_scale?, n_obstacles?, final_distance}, ...]

    Panel tengah = jumlah rintangan bila ada rintangan (Misi 3); selain itu skala gravitasi yang diacak
    (sebelumnya Misi 1-2 menampilkan panel 'jumlah rintangan' yang selalu 0)."""
    n_obs = np.array([r.get("n_obstacles", 0) or 0 for r in records], dtype=float)
    mid = ("n_obstacles", "Jumlah rintangan") if np.nanmax(n_obs, initial=0) > 0 else \
        ("gravity_scale", "Skala gravitasi (domain randomization)")
    keys = [("payload_mass_extra", "Massa payload tambahan (kg)"), mid,
            ("final_distance", "Jarak akhir ke target (m)")]
    succ = np.array([bool(r["success"]) for r in records])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"Robustness (domain randomization aktif): sukses {succ.mean() * 100:.0f}% dari {len(succ)} episode",
                 fontsize=11)
    for ax, (k, lab) in zip(axes, keys):
        x = np.array([r.get(k, np.nan) for r in records], dtype=float)
        jit = np.random.default_rng(0).uniform(-0.05, 0.05, len(x))
        ax.scatter(x[succ], 1 + jit[succ], c="green", alpha=0.5, label="sukses")
        ax.scatter(x[~succ], jit[~succ], c="red", alpha=0.5, label="gagal")
        ax.set_xlabel(lab); ax.set_yticks([0, 1]); ax.set_yticklabels(["gagal", "sukses"])
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        if not np.isfinite(x).any():
            _no_data(ax, "tidak ada data")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_onnx_vs_checkpoint_diff(diffs: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(diffs)
    ax.set_xlabel("step (rollout nyata)")
    ax.set_ylabel("max |diff| aksi (checkpoint vs ONNX)")
    ax.set_title(f"Selisih ONNX vs checkpoint (max={diffs.max():.2e}, mean={diffs.mean():.2e})")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
