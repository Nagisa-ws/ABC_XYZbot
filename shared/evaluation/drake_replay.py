"""
shared/evaluation/drake_replay.py
---------------------------------
Viewer 3D INTERAKTIF (Meshcat milik Drake) untuk hasil evaluasi, sebagai cadangan bila MuJoCo viewer tidak bisa
dipakai (mis. Google Colab, atau perangkat yang tidak kuat menjalankan viewer).

Cara kerja: episode dijalankan di MuJoCo seperti biasa (fisika tidak berubah), posisi sendi direkam, lalu diputar
ulang di Drake sebagai animasi Meshcat dan diekspor menjadi HTML statis. HTML itu mandiri: dapat dibuka di
browser (putar kamera, geser timeline, play/pause) tanpa server.

Drake TIDAK menjalankan MuJoCo viewer dan bukan simulator pengganti; ia hanya menggambar ulang pose yang direkam.
Drake dimuat secara lazy (pip install drake; butuh Python >= 3.12), jadi modul ini aman diimpor tanpa Drake.

Tingkat detail model (parameter `detail`):
    "mesh_ringan"  mesh UR10e disederhanakan (vertex clustering 4 mm) + gripper - HTML ~3-4 MB (default)
    "primitif"     model training (kapsul/silinder) - HTML ~2.5 MB, paling ringan
    "mesh_penuh"   mesh asli - HTML ~47 MB; berat, tidak disarankan di Colab
Parser MJCF Drake hanya mendukung mesh .obj, jadi gripper (.stl) dikonversi otomatis ke .obj
(skala 0.001 diterapkan ke vertex).
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np

DETAILS = ("primitif", "mesh_ringan", "mesh_penuh")


# ---------------------------------------------------------------------- #
# Ketersediaan Drake
# ---------------------------------------------------------------------- #
def drake_status() -> tuple[bool, str]:
    """(tersedia?, pesan). Tidak melempar error."""
    try:
        import importlib.metadata as md
        import pydrake.all  # noqa: F401
        return True, f"Drake {md.version('drake')}"
    except Exception as e:  # ImportError atau kegagalan pustaka native
        return False, f"Drake tidak dapat dimuat ({type(e).__name__}: {str(e)[:120]}). Pasang dengan: pip install drake"


def _require_drake():
    ok, msg = drake_status()
    if not ok:
        raise RuntimeError(msg)
    logging.getLogger("drake").setLevel(logging.ERROR)     # diamkan peringatan parser MJCF (gravcomp, site, dst.)


# ---------------------------------------------------------------------- #
# Konversi / penyederhanaan mesh
# ---------------------------------------------------------------------- #
def _read_stl(path) -> np.ndarray:
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else -1
    if len(data) == 84 + 50 * n:                                             # STL biner
        rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n)
        return rec["v"].reshape(-1, 3, 3).astype(np.float64)
    txt = data.decode("ascii", errors="ignore")                              # STL ASCII
    vs = np.array([[float(x) for x in m.groups()] for m in re.finditer(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", txt)])
    return vs.reshape(-1, 3, 3)


def _read_obj(path) -> tuple[np.ndarray, np.ndarray]:
    v, f = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(v) + i for i in idx]
                for k in range(1, len(idx) - 1):                             # fan-triangulasi
                    f.append([idx[0], idx[k], idx[k + 1]])
    return np.array(v, dtype=np.float64), np.array(f, dtype=np.int64)


def _write_obj(verts: np.ndarray, faces: np.ndarray, path) -> None:
    """Tulis OBJ dengan NORMAL vertex (area-weighted). Tanpa normal, mesin render VTK menolak berkas dan mesh dapat
    tampil gelap/datar di Meshcat."""
    n = np.zeros_like(verts)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    with open(path, "w") as fh:
        fh.write("\n".join("v %.6f %.6f %.6f" % tuple(p) for p in verts) + "\n")
        fh.write("\n".join("vn %.5f %.5f %.5f" % tuple(p) for p in n) + "\n")
        fh.write("\n".join("f %d//%d %d//%d %d//%d" % (a + 1, a + 1, b + 1, b + 1, c + 1, c + 1) for a, b, c in faces) + "\n")


def _decimate(verts: np.ndarray, faces: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: simpul dalam satu sel kubus `cell` (m) digabung ke rata-ratanya; segitiga yang runtuh dibuang."""
    key = np.floor(verts / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    new_v = np.stack([np.bincount(inv, weights=verts[:, k], minlength=n) / cnt for k in range(3)], axis=1)
    f_all = inv[faces]
    keep = (f_all[:, 0] != f_all[:, 1]) & (f_all[:, 1] != f_all[:, 2]) & (f_all[:, 0] != f_all[:, 2])
    f_keep = f_all[keep]                                          # buang segitiga yang runtuh
    _, first = np.unique(np.sort(f_keep, axis=1), axis=0, return_index=True)   # buang duplikat, orientasi asli dijaga
    return new_v, f_keep[np.sort(first)]


def prepare_model(robot_model_dir, work_dir, detail: str = "mesh_ringan", cell: float = 0.004) -> Path:
    """Kembalikan path XML yang dapat dimuat parser MJCF Drake.

    primitif    -> robot_model/ur10e_train.xml apa adanya.
    mesh_*      -> salinan ur10e_eval.xml di `work_dir` dengan gripper .stl dikonversi ke .obj
                   (dan mesh lengan disederhanakan bila mesh_ringan).
    """
    if detail not in DETAILS:
        raise ValueError(f"detail harus salah satu dari {DETAILS}")
    robot_model_dir = Path(robot_model_dir)
    if detail == "primitif":
        return robot_model_dir / "ur10e_train.xml"
    work_dir = Path(work_dir)
    out_assets = work_dir / "assets"
    shutil.rmtree(work_dir, ignore_errors=True)
    out_assets.mkdir(parents=True)
    src_assets = robot_model_dir / "assets"
    xml = (robot_model_dir / "ur10e_eval.xml").read_text(encoding="utf-8")

    for m in re.finditer(r'<mesh name="([^"]+)" file="([^"]+\.stl)" scale="([^"]+)"/>', xml):   # STL -> OBJ
        _, fn, sc = m.groups()
        tri = _read_stl(src_assets / fn).reshape(-1, 3)
        uniq, inv = np.unique(np.round(tri, 7), axis=0, return_inverse=True)
        _write_obj(uniq * float(sc.split()[0]), inv.reshape(-1, 3), out_assets / (Path(fn).stem + ".obj"))
    xml = re.sub(r'file="([^"]+)\.stl" scale="[^"]+"', r'file="\1.obj"', xml)

    for f in src_assets.glob("*.obj"):                                                          # mesh lengan
        dst = out_assets / f.name
        if detail == "mesh_ringan":
            v, fa = _read_obj(f)
            nv, nf = _decimate(v, fa, cell)
            _write_obj(nv, nf, dst)
        else:
            try:
                os.symlink(f, dst)
            except OSError:
                shutil.copy(f, dst)
    out = work_dir / "ur10e_drake.xml"
    out.write_text(xml, encoding="utf-8")
    return out


# ---------------------------------------------------------------------- #
# Perekaman episode (MuJoCo) -> data replay
# ---------------------------------------------------------------------- #
def record_episode(env, policy, seed: int, max_steps: int | None = None) -> dict:
    """Jalankan satu episode di MuJoCo dan rekam qpos tiap langkah + data skenario."""
    import mujoco
    if hasattr(policy, "reset"):
        policy.reset()
    obs, _ = env.reset(seed=int(seed))
    m = env.model
    joints = []
    for j in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        if name and int(m.jnt_type[j]) in (mujoco.mjtJoint.mjJNT_HINGE.value, mujoco.mjtJoint.mjJNT_SLIDE.value):
            joints.append((name, int(m.jnt_qposadr[j])))
    qpos, tcp = [env.data.qpos.copy()], [np.asarray(env.tcp_pos, float).copy()]
    reason, info, n = None, {}, 0
    for n in range(1, int(max_steps or env.max_steps) + 1):
        obs, _, term, trunc, info = env.step(policy(obs))
        qpos.append(env.data.qpos.copy())
        tcp.append(np.asarray(env.tcp_pos, float).copy())
        if term or trunc:
            reason = info.get("termination_reason") or "timeout"
            break
    return {
        "seed": int(seed), "reason": reason or "timeout", "steps": n, "joints": joints,
        "qpos": np.array(qpos), "tcp": np.array(tcp), "dt": float(env.n_substeps * env.model.opt.timestep),
        "T1": None if env.T1 is None else np.asarray(env.T1, float).copy(),
        "T2": None if env.T2 is None else np.asarray(env.T2, float).copy(),
        "obstacles": np.asarray(env.obstacle_points, float).copy(),
        "obstacle_r": np.asarray(env.obstacle_radii, float).copy(),
        "d_safe": float(getattr(env, "_d_safe", 0.0)),
    }


# ---------------------------------------------------------------------- #
# Putar ulang di Drake -> HTML
# ---------------------------------------------------------------------- #
def _setup_drake(model_xml, joints):
    """Muat model ke Drake + Meshcat. Pemetaan sendi MuJoCo -> Drake dilakukan lewat NAMA sendi."""
    _require_drake()
    from pydrake.geometry import Meshcat, MeshcatVisualizer
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    meshcat = Meshcat()
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    Parser(plant).AddModels(str(model_xml))
    plant.Finalize()
    vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    mapping = []
    for name, adr in joints:
        try:
            mapping.append((adr, plant.GetJointByName(name).position_start()))
        except Exception:
            pass
    if not mapping:
        raise RuntimeError("Tidak ada sendi MuJoCo yang cocok dengan model Drake (periksa nama sendi di XML).")
    return {"meshcat": meshcat, "plant": plant, "vis": vis, "diagram": diagram, "ctx": ctx,
            "pctx": plant.GetMyContextFromRoot(ctx), "mapping": mapping}


def _publish_pose(d: dict, qpos: np.ndarray, t: float) -> None:
    q = d["plant"].GetPositions(d["pctx"]).copy()
    for adr, start in d["mapping"]:
        q[start] = qpos[adr]
    d["plant"].SetPositions(d["pctx"], q)
    d["ctx"].SetTime(float(t))
    d["diagram"].ForcedPublish(d["ctx"])


def _add_scenario_objects(meshcat, rec: dict, root: str, zones: bool, trail: bool) -> None:
    """Penanda T1/T2, garis, rintangan (+zona aman) dan jejak end-effector di bawah path `root`."""
    from pydrake.geometry import Rgba, Sphere
    from pydrake.math import RigidTransform
    if rec["T1"] is not None:
        meshcat.SetObject(f"{root}/T1", Sphere(0.018), Rgba(0.1, 0.9, 0.2, 1.0))
        meshcat.SetTransform(f"{root}/T1", RigidTransform(rec["T1"]))
    if rec["T2"] is not None:
        meshcat.SetObject(f"{root}/T2", Sphere(0.018), Rgba(0.95, 0.15, 0.15, 1.0))
        meshcat.SetTransform(f"{root}/T2", RigidTransform(rec["T2"]))
        if rec["T1"] is not None:
            meshcat.SetLine(f"{root}/garis_T1_T2", np.stack([rec["T1"], rec["T2"]], axis=1), 3.0, Rgba(0.05, 0.05, 0.05, 1.0))
    for i, (pt, r) in enumerate(zip(rec["obstacles"], rec["obstacle_r"])):
        meshcat.SetObject(f"{root}/rintangan/{i}", Sphere(float(r)), Rgba(1.0, 0.55, 0.0, 1.0))
        meshcat.SetTransform(f"{root}/rintangan/{i}", RigidTransform(pt))
        if zones and rec["d_safe"] > 0:
            meshcat.SetObject(f"{root}/zona/{i}", Sphere(float(r) + rec["d_safe"]), Rgba(1.0, 0.55, 0.0, 0.10))
            meshcat.SetTransform(f"{root}/zona/{i}", RigidTransform(pt))
    if trail and len(rec["tcp"]) > 1:
        meshcat.SetLine(f"{root}/jejak_EE", rec["tcp"].T, 2.0, Rgba(0.1, 0.4, 1.0, 1.0))


def _set_camera(meshcat, recs: list, distance: float, elevation: float, azimuth_offset: float) -> None:
    """Kamera dari samping, diarahkan ke rata-rata titik tengah jalur semua skenario."""
    mids = []
    for rec in recs:
        anchor = rec["T2"] if rec["T2"] is not None else rec["T1"]
        mids.append(0.5 * (rec["T1"] + anchor) if rec["T1"] is not None else np.array([0.4, 0.0, 0.5]))
    mid = np.mean(mids, axis=0)
    look = np.array([0.5 * mid[0], 0.5 * mid[1], 0.45])
    az, el = np.arctan2(mid[1], mid[0]) + np.deg2rad(azimuth_offset), np.deg2rad(elevation)
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    meshcat.SetCameraPose(look - distance * fwd, look)


def _add_floor(meshcat) -> None:
    from pydrake.geometry import Box, Rgba
    from pydrake.math import RigidTransform
    meshcat.SetObject("scene/lantai", Box(3.0, 3.0, 0.004), Rgba(0.22, 0.30, 0.42, 1.0))
    meshcat.SetTransform("scene/lantai", RigidTransform([0.0, 0.0, -0.002]))


def _frames_per_second(anim) -> float:
    f = anim.frames_per_second
    return float(f() if callable(f) else f)


def build_replay_html(rec: dict, model_xml, *, zones: bool = True, trail: bool = True,
                      distance: float = 2.3, elevation: float = -22.0, azimuth_offset: float = 90.0,
                      return_info: bool = False):
    """Putar ulang SATU episode `rec` (hasil record_episode) di Meshcat Drake dan kembalikan HTML statis.
    return_info=True -> (html, {"frames": jumlah frame animasi, "sendi_terpetakan": n})."""
    from pydrake.geometry import MeshcatAnimation
    d = _setup_drake(model_xml, rec["joints"])
    meshcat, vis = d["meshcat"], d["vis"]
    _add_floor(meshcat)
    _add_scenario_objects(meshcat, rec, "scene", zones, trail)
    _set_camera(meshcat, [rec], distance, elevation, azimuth_offset)

    # animasi: waktu konteks = waktu simulasi nyata (k * dt) agar pemutaran real-time
    # (Drake 1.57: StartRecording tidak punya parameter fps; frame dipetakan dari waktu pada fps bawaan Drake.)
    vis.StartRecording(set_transforms_while_recording=False)
    for k, qpos in enumerate(rec["qpos"]):
        _publish_pose(d, qpos, k * rec["dt"])
    vis.StopRecording()
    anim = vis.get_mutable_recording()
    anim.set_autoplay(True)
    anim.set_loop_mode(MeshcatAnimation.LoopMode.kLoopRepeat)
    vis.PublishRecording()
    html = meshcat.StaticHtml()
    if return_info:
        return html, {"frames": int(anim.frame((len(rec["qpos"]) - 1) * rec["dt"])) + 1,
                      "sendi_terpetakan": len(d["mapping"])}
    return html


def build_multi_replay_html(recs: list, model_xml, *, gap_s: float = 1.0, zones: bool = True, trail: bool = True,
                            distance: float = 2.3, elevation: float = -22.0, azimuth_offset: float = 90.0):
    """Putar BEBERAPA episode berurutan dalam satu timeline Meshcat.

    Penanda, rintangan, dan jejak tiap skenario berada di grup `scene/s<i>`; visibilitasnya digeser lewat keyframe
    animasi ("visible") sehingga hanya skenario yang sedang diputar yang terlihat. Antar skenario ada jeda `gap_s`
    detik (robot menahan pose akhir). Return (html, info) dengan info["starts_s"] = detik mulai tiap skenario.
    """
    if not recs:
        raise ValueError("recs kosong")
    from pydrake.geometry import MeshcatAnimation
    d = _setup_drake(model_xml, recs[0]["joints"])
    meshcat, vis = d["meshcat"], d["vis"]
    _add_floor(meshcat)
    for i, rec in enumerate(recs):
        _add_scenario_objects(meshcat, rec, f"scene/s{i}", zones, trail)
        meshcat.SetProperty(f"scene/s{i}", "visible", i == 0)           # keadaan awal (sebelum animasi diputar)
    _set_camera(meshcat, recs, distance, elevation, azimuth_offset)

    anim = vis.StartRecording(set_transforms_while_recording=False)
    one_frame = 1.0 / _frames_per_second(anim)
    t, starts = 0.0, []
    for rec in recs:
        starts.append(t)
        for k, qpos in enumerate(rec["qpos"]):
            _publish_pose(d, qpos, t + k * rec["dt"])
        end = t + (len(rec["qpos"]) - 1) * rec["dt"]
        _publish_pose(d, rec["qpos"][-1], end + gap_s)                  # tahan pose akhir selama gap_s
        t = end + gap_s + one_frame                                      # skenario berikut mulai satu frame kemudian
    vis.StopRecording()
    anim = vis.get_mutable_recording()
    for i, t0 in enumerate(starts):                                      # keyframe visibilitas di awal tiap skenario
        fr = int(anim.frame(t0))
        for j in range(len(recs)):
            anim.SetProperty(fr, f"scene/s{j}", "visible", j == i)
    anim.set_autoplay(True)
    anim.set_loop_mode(MeshcatAnimation.LoopMode.kLoopRepeat)
    vis.PublishRecording()
    html = meshcat.StaticHtml()
    return html, {"starts_s": [round(x, 2) for x in starts], "total_s": round(t, 2),
                  "frames": int(anim.frame(t)) + 1, "sendi_terpetakan": len(d["mapping"])}


def episode_to_html(env, policy, seed: int, robot_model_dir, work_dir, detail: str = "mesh_ringan", **kw):
    """Rekam satu episode lalu bangun HTML replay. Return (html, ringkasan)."""
    rec = record_episode(env, policy, seed)
    model_xml = prepare_model(robot_model_dir, work_dir, detail)
    html, info = build_replay_html(rec, model_xml, return_info=True, **kw)
    return html, {"seed": rec["seed"], "reason": rec["reason"], "steps": rec["steps"],
                  "html_mb": round(len(html) / 1e6, 2), "detail": detail, **info}


def episodes_to_html(env, policy, seeds, robot_model_dir, work_dir, detail: str = "mesh_ringan",
                     gap_s: float = 1.0, **kw):
    """Rekam beberapa episode lalu putar berurutan dalam SATU viewer. Return (html, ringkasan)."""
    recs = [record_episode(env, policy, sd) for sd in seeds]
    model_xml = prepare_model(robot_model_dir, work_dir, detail)
    html, info = build_multi_replay_html(recs, model_xml, gap_s=gap_s, **kw)
    episodes = [{"skenario": i + 1, "seed": r["seed"], "reason": r["reason"], "steps": r["steps"],
                 "mulai_detik": info["starts_s"][i]} for i, r in enumerate(recs)]
    return html, {"episodes": episodes, "html_mb": round(len(html) / 1e6, 2), "detail": detail,
                  "total_detik": info["total_s"], "frames": info["frames"], "sendi_terpetakan": info["sendi_terpetakan"]}


def notebook_iframe(html: str, height: int = 650) -> str:
    """Bungkus HTML penuh Meshcat dalam <iframe srcdoc> agar tampil terisolasi di Jupyter/Colab:
    display(HTML(notebook_iframe(html)))."""
    import html as _html
    return (f'<iframe srcdoc="{_html.escape(html, quote=True)}" width="100%" height="{int(height)}" '
            f'style="border:0" allowfullscreen></iframe>')
