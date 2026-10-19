"""
shared/evaluation/video_render.py
---------------------------------
Rendering OFFSCREEN episode evaluasi menjadi video (untuk Google Colab / server tanpa layar).

MuJoCo viewer interaktif tidak bisa dibuka di Colab; modul ini menggantinya dengan `mujoco.Renderer` (EGL pada
runtime GPU, OSMesa pada runtime CPU) dan menulis MP4 lewat `ffmpeg`. Marker sama seperti di viewer:
T1 (hijau), T2 (merah), garis T1-T2, rintangan (oranye) + zona aman, dan jejak end-effector.

PENTING: variabel lingkungan MUJOCO_GL ("egl" atau "osmesa") harus diatur SEBELUM `import mujoco`.

Contoh:
    policy = make_policy(onnx_path, env, mode="hybrid+orientasi")
    frames, hasil = render_episodes(env, policy, seeds=[1000, 1001], every=2)
    write_mp4(frames, "evaluasi.mp4", fps=25)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import mujoco

_I3 = np.eye(3).flatten()
_GREEN = np.array([0.1, 0.9, 0.2, 1.0], np.float32)
_RED = np.array([0.95, 0.15, 0.15, 1.0], np.float32)
_ORANGE = np.array([1.0, 0.55, 0.0, 1.0], np.float32)
_ZONE = np.array([1.0, 0.55, 0.0, 0.10], np.float32)
_LINE = np.array([0.05, 0.05, 0.05, 0.9], np.float32)
_TRAIL = np.array([0.1, 0.4, 1.0, 1.0], np.float32)

MODES = ("rl", "hybrid", "hybrid+orientasi")


# ---------------------------------------------------------------------- #
def _sphere(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), np.asarray(pos, float), _I3, rgba)
    scn.ngeom += 1


def _line(scn, p0, p1, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def append_markers(scn, env, trail, draw_zone: bool = True):
    """Tambahkan marker ke scene yang SUDAH berisi geom model (tidak mereset ngeom)."""
    if env.T1 is not None:
        _sphere(scn, env.T1, 0.018, _GREEN)
    if env.T2 is not None:
        _sphere(scn, env.T2, 0.018, _RED)
        if env.T1 is not None:
            _line(scn, env.T1, env.T2, 0.004, _LINE)
    if env.obstacle_points.shape[0]:
        d_safe = float(getattr(env, "_d_safe", 0.0))
        for p, r in zip(env.obstacle_points, env.obstacle_radii):
            _sphere(scn, p, float(r), _ORANGE)
            if draw_zone and d_safe > 0:
                _sphere(scn, p, float(r) + d_safe, _ZONE)
    for p in trail:
        _sphere(scn, p, 0.005, _TRAIL)


def make_camera(env, azimuth_offset: float = 90.0, distance: float = 2.3, elevation: float = -22.0):
    """Kamera bebas yang melihat robot dan jalur dari samping."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    anchor = env.T2 if env.T2 is not None else env.T1
    mid = 0.5 * (np.asarray(env.T1) + np.asarray(anchor)) if env.T1 is not None else np.array([0.4, 0.0, 0.5])
    cam.lookat[:] = [0.5 * mid[0], 0.5 * mid[1], 0.45]
    cam.distance = distance
    cam.azimuth = float(np.degrees(np.arctan2(mid[1], mid[0]))) + azimuth_offset
    cam.elevation = elevation
    return cam


def make_policy(onnx_path, env, mode: str = "rl", control_cfg_path=None):
    """Policy dari ONNX. mode: 'rl' | 'hybrid' (RL+LQR) | 'hybrid+orientasi' (juga sejajarkan moncong)."""
    if mode not in MODES:
        raise ValueError(f"mode harus salah satu dari {MODES}")
    if mode == "rl":
        from shared.debug_common.common import make_predict_fn_onnx
        from shared.debug_common.post_training import onnx_session
        return make_predict_fn_onnx(onnx_session(str(onnx_path)))
    from shared.control import bind_to_env, make_hybrid_from_onnx
    return bind_to_env(make_hybrid_from_onnx(onnx_path, control_cfg_path, orientation=(mode == "hybrid+orientasi")), env)


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return img
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, im.width, 18], fill=(0, 0, 0))
    dr.text((6, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def render_episodes(env, policy, seeds, *, width: int = 640, height: int = 480, every: int = 2,
                    max_steps: int | None = None, hold_frames: int = 12, label: str = "", fast: bool = True,
                    trail_stride: int = 3, trail_max: int = 100, log=print):
    """Jalankan `policy` pada tiap seed, render tiap `every` langkah kontrol.

    fast=True mematikan bayangan dan pantulan (jauh lebih cepat pada rendering software/OSMesa; tampilan sedikit
    lebih datar). Jejak end-effector digambar tiap `trail_stride` langkah, maksimal `trail_max` titik.

    Return (frames: list[np.ndarray HxWx3 uint8], hasil: list[dict]).
    """
    model = env.model
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
    renderer = mujoco.Renderer(model, int(height), int(width))
    if fast:
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    frames, results = [], []
    max_steps = int(max_steps or env.max_steps)
    try:
        for i, seed in enumerate(seeds):
            if hasattr(policy, "reset"):
                policy.reset()
            obs, _ = env.reset(seed=int(seed))
            cam = make_camera(env)
            trail: list[np.ndarray] = []
            info, reason, n = {}, None, 0

            def grab(txt):
                renderer.update_scene(env.data, cam)
                append_markers(renderer.scene, env, trail)
                return _caption(renderer.render().copy(), txt)

            head = f"{label} skenario {i + 1}/{len(seeds)} seed={seed}"
            frames.append(grab(f"{head} | langkah 0"))
            for n in range(1, max_steps + 1):
                obs, _, term, trunc, info = env.step(policy(obs))
                if n % trail_stride == 0:
                    trail.append(env.tcp_pos.copy())
                    if len(trail) > trail_max:
                        trail.pop(0)
                if n % every == 0:
                    frames.append(grab(f"{head} | langkah {n} | fase {'B' if env.phase else 'A'}"))
                if term or trunc:
                    reason = info.get("termination_reason") or "timeout"
                    break
            reason = reason or "timeout"
            last = grab(f"{head} | AKHIR: {reason}")
            frames.extend([last] * hold_frames)
            results.append({"seed": int(seed), "reason": reason, "steps": n,
                            "final_distance_mm": round(1000.0 * float(info.get("final_distance", float("nan"))), 1)})
            log(f"[video] {head}: {reason} dalam {n} langkah")
    finally:
        renderer.close()
    return frames, results


def write_mp4(frames, path, fps: int = 25) -> Path:
    """Tulis MP4 (H.264) lewat ffmpeg. Lebar/tinggi dipaksa genap (syarat yuv420p)."""
    frames = list(frames)
    if not frames:
        raise ValueError("tidak ada frame")
    h, w = frames[0].shape[:2]
    w2, h2 = w - (w % 2), h - (h % 2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}",
           "-r", str(int(fps)), "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
        p.stdin.close()
        err = p.stderr.read().decode(errors="ignore")
        if p.wait() != 0:
            raise RuntimeError(f"ffmpeg gagal: {err}")
    finally:
        if p.poll() is None:
            p.kill()
    return path
