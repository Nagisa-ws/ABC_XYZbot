"""
shared/control/path_controller.py
---------------------------------
Controller umpan-balik untuk posisi TCP (reaching T1, menyusuri garis T1->T2, menggapai T2).

Hanya membaca vektor observasi 57-dim (shared/envs/obs_layout.py), sehingga dapat dijalankan di sisi
pengguna bersama ONNX tanpa sensor tambahan:
    rel_target      : target aktif - posisi EE   (fase 0: T1, fase 1: T2)
    tcp_pos         : posisi EE relatif base
    phase           : 0 = menuju T1, 1 = menyusuri T1->T2
    path_dir        : arah unit T1->T2
    cross_track_vec : vektor EE -> titik terdekat pada garis T1T2

Plant yang dikendalikan (hasil identifikasi, per sumbu, satuan meter & langkah kontrol):
    v_k = a1 v_{k-1} + a2 v_{k-2} + b0 u_k + b1 u_{k-1},    p_{k+1} = p_k + v_k
dengan u_k = aksi * max_pos_delta. Kecepatan tunak = 0.17 * u (linier pada seluruh rentang aksi).

Dua hukum kendali dengan logika jalur yang SAMA (agar perbandingan adil):
    "lqr" : u = u_ff - K s,  K dari DARE, s = [e, v_{k-1}, v_{k-2}, u_{k-1}]
    "p"   : u = u_ff + kp * (p_ref - p)

Aksi rotasi (indeks 3..5) dikembalikan nol (orientasi nominal). Pada mode hibrida, bagian rotasi
tetap dari policy RL.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shared.envs.obs_layout import OBS_SLICES

# Hasil identifikasi (lihat catatan di README kontrol): model skalar yang sama untuk x, y, z.
DEFAULT_ARX = dict(a1=0.9909, a2=-0.2513, b0=0.0980, b1=-0.0535)


def dc_gain(m: dict) -> float:
    return (m["b0"] + m["b1"]) / (1.0 - m["a1"] - m["a2"])


def _solve_dare(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                max_iter: int = 200_000, tol: float = 1e-12) -> np.ndarray:
    """Persamaan Riccati diskret lewat iterasi nilai (tanpa scipy; sistem 4x4 konvergen cepat)."""
    P = Q.copy()
    for _ in range(max_iter):
        BtP = B.T @ P
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + BtP @ B, BtP @ A)
        if np.max(np.abs(P_next - P)) <= tol * max(1.0, np.max(np.abs(P_next))):
            return P_next
        P = P_next
    raise RuntimeError("DARE tidak konvergen; periksa bobot LQR / model plant")


def design_lqr(m: dict, q_e: float, q_v: float, r: float, q_u: float = 0.0) -> np.ndarray:
    """K (1x4) untuk u = -K s, s = [e, v_{k-1}, v_{k-2}, u_{k-1}]."""
    a1, a2, b0, b1 = m["a1"], m["a2"], m["b0"], m["b1"]
    A = np.array([[1.0, a1, a2, b1],
                  [0.0, a1, a2, b1],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0]])
    B = np.array([[b0], [b0], [0.0], [1.0]])
    Q = np.diag([q_e, q_v, 0.0, q_u])
    R = np.array([[r]])
    P = _solve_dare(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


@dataclass
class ControllerConfig:
    law: str = "lqr"                 # "lqr" | "p"
    max_pos_delta: float = 0.05      # harus sama dengan env (action.max_pos_delta)
    # --- LQR: bobot hasil penalaan acak (70 kombinasi, 40 skenario) lalu divalidasi pada 60 skenario lain ---
    q_e_cross: float = 1115.0
    q_v_cross: float = 6336.6
    q_e_along: float = 27.1
    q_v_along: float = 701.4
    r: float = 4.523
    # --- hukum P ---
    kp: float = 0.4
    # --- logika jalur ---
    lead_approach: float = 0.25     # carrot di depan EE saat menuju T1 (m)
    lead_line: float = 0.03         # carrot di depan proyeksi pada garis (m)
    v_cruise: float = 0.004          # kecepatan jelajah sepanjang garis (m/langkah)
    taper_gain: float = 0.25         # perlambatan menjelang T2: v = min(v_cruise, taper_gain * sisa)
    arx: dict = field(default_factory=lambda: dict(DEFAULT_ARX))


class PathController:
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()
        c = self.cfg
        self.G = dc_gain(c.arx)
        self.K_cross = design_lqr(c.arx, c.q_e_cross, c.q_v_cross, c.r)[0]
        self.K_along = design_lqr(c.arx, c.q_e_along, c.q_v_along, c.r)[0]
        self.reset()

    def reset(self):
        self._p1 = None            # p_{k-1}
        self._p2 = None            # p_{k-2}
        self._u1 = np.zeros(3)     # u_{k-1} (perintah yang DITERAPKAN)

    # ------------------------------------------------------------------ #
    def _reference(self, obs: np.ndarray, hold_along: bool = False):
        """Return (p, p_ref, w, d) : posisi, titik acuan (carrot), kecepatan acuan, arah garis (atau None)."""
        c = self.cfg
        p = np.asarray(obs[OBS_SLICES["tcp_pos"]], dtype=float)
        rel = np.asarray(obs[OBS_SLICES["rel_target"]], dtype=float)
        phase = int(round(float(obs[OBS_SLICES["phase"]][0])))
        if phase == 0:
            dist = float(np.linalg.norm(rel))
            if dist < 1e-9:
                return p, p.copy(), np.zeros(3), None
            p_ref = p + rel / dist * min(dist, c.lead_approach)
            return p, p_ref, np.zeros(3), None
        d = np.asarray(obs[OBS_SLICES["path_dir"]], dtype=float)
        d = d / max(np.linalg.norm(d), 1e-9)
        cross = np.asarray(obs[OBS_SLICES["cross_track_vec"]], dtype=float)
        q = p + cross                                  # proyeksi EE pada garis
        s_rem = max(0.0, float(d @ (p + rel - q)))     # sisa jarak sepanjang garis sampai T2
        if hold_along:                                 # tahan kemajuan sepanjang garis (mis. menunggu orientasi sejajar)
            return p, q.copy(), np.zeros(3), d
        p_ref = q + d * min(c.lead_line, s_rem)
        w = d * min(c.v_cruise, c.taper_gain * s_rem)
        return p, p_ref, w, d

    def act(self, obs: np.ndarray, hold_along: bool = False) -> np.ndarray:
        c = self.cfg
        p, p_ref, w, d = self._reference(obs, hold_along)
        p1 = p if self._p1 is None else self._p1
        p2 = p1 if self._p2 is None else self._p2
        v1, v2 = p - p1, p1 - p2                       # v_{k-1}, v_{k-2} (estimasi dari posisi)
        e = p - p_ref
        u_ff = w * (1.0 - c.arx["a1"] - c.arx["a2"]) / (c.arx["b0"] + c.arx["b1"])

        if c.law == "p":
            u = u_ff - c.kp * e
        else:
            s_e, s_v1, s_v2, s_u = e, v1 - w, v2 - w, self._u1 - u_ff

            if d is None:                              # fase 0: satu set gain (sepanjang = tegak lurus)
                u = u_ff - (self.K_along[0] * s_e + self.K_along[1] * s_v1
                            + self.K_along[2] * s_v2 + self.K_along[3] * s_u)
            else:                                      # fase 1: pisahkan komponen sepanjang / tegak lurus garis
                P_a = np.outer(d, d)
                P_c = np.eye(3) - P_a
                u = u_ff - (P_a @ (self.K_along[0] * s_e + self.K_along[1] * s_v1
                                    + self.K_along[2] * s_v2 + self.K_along[3] * s_u)
                            + P_c @ (self.K_cross[0] * s_e + self.K_cross[1] * s_v1
                                     + self.K_cross[2] * s_v2 + self.K_cross[3] * s_u))

        a = np.clip(u / c.max_pos_delta, -1.0, 1.0)
        self._u1 = a * c.max_pos_delta                 # yang benar-benar diterapkan (setelah saturasi)
        self._p2, self._p1 = p1, p
        out = np.zeros(6)
        out[:3] = a
        return out
