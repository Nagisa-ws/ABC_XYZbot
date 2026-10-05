"""
shared/envs/obs_layout.py
-------------------------
Layout observasi TERPADU untuk ketiga misi.

Kenapa terpadu?
  Bobot policy Misi N dipakai sebagai inisialisasi Misi N+1 (transfer). Itu
  hanya mungkin bila dimensi observasi IDENTIK di semua misi. Versi lama
  menambah dimensi tiap misi (26 -> 31 -> 36) sehingga PPO.load() gagal dan
  `observation_space` tidak cocok dengan observasi sebenarnya.

  Sekarang dimensi tetap. Bagian yang belum relevan untuk sebuah misi diisi
  nilai konstan (nol / "jauh") dan ditandai tidak-aktif pada ACTIVE_DIMS,
  sehingga transfer bisa menangani statistik normalisasi & bobot input-nya.
"""
from __future__ import annotations

import numpy as np

N_NEAREST_OBSTACLES = 4          # jumlah rintangan terdekat yang diamati
OBS_FAR = 1.0                    # jarak saturasi (m) bila tidak ada rintangan

# (nama, panjang) berurutan
_FIELDS = [
    ("rel_target", 3),        # P_target_aktif - P_ee
    ("q", 6),                 # sudut sendi
    ("qd", 6),                # kecepatan sendi
    ("tcp_pos", 3),           # posisi EE relatif base
    ("tcp_rot6", 6),          # 2 kolom pertama matriks rotasi EE
    ("manip", 1),             # manipulability (deteksi singularitas)
    ("prev_action", 6),       # aksi step sebelumnya
    ("time_left", 1),         # 1 - step/max_steps
    # ---- aktif mulai Misi 2 ----
    ("phase", 1),             # 0 = reaching T1, 1 = path following T1->T2
    ("path_dir", 3),          # arah unit T1->T2
    ("cross_track_vec", 3),   # vektor EE -> titik terdekat pada garis T1T2
    ("path_progress", 1),     # posisi proyeksi EE sepanjang segmen (0..1)
    # ---- aktif mulai Misi 3 ----
    ("obs_rel", 3 * N_NEAREST_OBSTACLES),   # vektor titik-robot-terdekat -> pusat rintangan
    ("obs_dist", N_NEAREST_OBSTACLES),      # jarak permukaan robot-rintangan (clip 0..OBS_FAR)
    ("min_clearance", 1),                   # clearance minimum seluruh badan robot
]

OBS_SLICES: dict[str, slice] = {}
_i = 0
for _name, _n in _FIELDS:
    OBS_SLICES[_name] = slice(_i, _i + _n)
    _i += _n
OBS_DIM = _i

_BASE_FIELDS = ["rel_target", "q", "qd", "tcp_pos", "tcp_rot6", "manip", "prev_action", "time_left"]
_PATH_FIELDS = ["phase", "path_dir", "cross_track_vec", "path_progress"]
_OBST_FIELDS = ["obs_rel", "obs_dist", "min_clearance"]


def _mask(fields: list[str]) -> np.ndarray:
    m = np.zeros(OBS_DIM, dtype=bool)
    for f in fields:
        m[OBS_SLICES[f]] = True
    return m


# Dimensi observasi yang BERVARIASI (bermakna) pada tiap misi.
ACTIVE_DIMS = {
    1: _mask(_BASE_FIELDS),
    2: _mask(_BASE_FIELDS + _PATH_FIELDS),
    3: _mask(_BASE_FIELDS + _PATH_FIELDS + _OBST_FIELDS),
}

# Dimensi yang diberi derau sensor saat training (bukan flag/indikator).
NOISY_FIELDS = ["rel_target", "q", "qd", "tcp_pos", "tcp_rot6", "manip",
                "cross_track_vec", "obs_rel", "obs_dist", "min_clearance"]
NOISE_MASK = _mask(NOISY_FIELDS)
