"""
obstacles/stargaze.py
---------------------
Konsep obstacle "constellation" yang diadaptasi dari stargaze_visualizer.py,
versi headless (tanpa UI).

Konsep asli (stargaze):
  * 1 titik nucleus di pusat (0,0,0)  -> TARGET AKHIR
  * 3 lapis kubus konsentris (L = half-side tiap lapis)
  * 6 face (+-x, +-y, +-z); tiap face: grid 3x3 per lapis, tiap sel max 1 titik,
    3-5 titik per face tersebar di 3 lapis (tidak ada lapis kosong)
  * Titik awal S dipilih pada face terluar; garis S -> nucleus dipilih yang
    memberi CLEARANCE maksimum terhadap seluruh titik rintangan
    ("Initial Search Face" + opposite face diabaikan + "Forbidden Face").

Adaptasi untuk Misi 3:
  T1 (titik awal)  = S pada face terluar
  T2 (titik akhir) = nucleus
  garis T1->T2     = garis lurus penghubung yang harus disusuri robot
  rintangan        = titik-titik constellation (bola kecil)

Perubahan dibanding versi sebelumnya:
  * Hanya SATU constellation per episode; T1, T2 dan rintangan konsisten
    (sebelumnya T2 diambil dari constellation berbeda dari rintangan).
  * Titik awal dipilih lewat pencarian clearance garis (vectorized) seperti
    stargaze, dengan filter reachability dari environment.
  * Jarak ke segmen (bukan garis tak hingga) -> benar untuk lintasan berhingga.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FACES = [
    {"name": "+x", "axis": 0, "sign": +1},
    {"name": "-x", "axis": 0, "sign": -1},
    {"name": "+y", "axis": 1, "sign": +1},
    {"name": "-y", "axis": 1, "sign": -1},
    {"name": "+z", "axis": 2, "sign": +1},
    {"name": "-z", "axis": 2, "sign": -1},
]
_FACE_INDEX = {f["name"]: i for i, f in enumerate(FACES)}


@dataclass
class ConstellationResult:
    nucleus: np.ndarray                  # (3,)   -> T2
    points: np.ndarray                   # (N,3)  posisi dunia rintangan
    layer_ids: np.ndarray                # (N,)
    face_ids: np.ndarray                 # (N,)
    L_values: tuple = (0.15, 0.30, 0.45)
    face_names: list = field(default_factory=lambda: [f["name"] for f in FACES])


class ConstellationGenerator:
    def __init__(self,
                 L_values=(0.15, 0.30, 0.45),
                 grid_size: int = 3,
                 n_points_per_face_range=(3, 5),
                 jitter: float = 0.35,
                 min_radius_from_center: float = 0.05):
        self.L_values = tuple(float(x) for x in L_values)
        self.grid_size = int(grid_size)
        self.n_points_per_face_range = tuple(n_points_per_face_range)
        self.jitter = float(jitter)
        self.min_radius_from_center = float(min_radius_from_center)

    # ------------------------------------------------------------------ #
    def _distribute_points_over_layers(self, total: int, rng) -> list[int]:
        n_layers = len(self.L_values)
        if total <= n_layers:
            return [1] * n_layers
        counts = [1] * n_layers
        for _ in range(total - n_layers):
            counts[int(rng.integers(0, n_layers))] += 1
        return counts

    def _sample_face(self, face, rng):
        pts = []
        n_total = int(rng.integers(self.n_points_per_face_range[0],
                                   self.n_points_per_face_range[1] + 1))
        layer_counts = self._distribute_points_over_layers(n_total, rng)
        axis, sign = face["axis"], face["sign"]
        cells = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)]
        for layer_idx, n_in_layer in enumerate(layer_counts):
            L = self.L_values[layer_idx]
            cell = (2 * L) / self.grid_size
            chosen = rng.choice(len(cells), size=min(n_in_layer, len(cells)), replace=False)
            for c in chosen:
                i, j = cells[int(c)]
                margin = cell * (1.0 - self.jitter) * 0.5 * 0.5   # jitter tetap di dalam sel
                u = rng.uniform(-L + i * cell + margin, -L + (i + 1) * cell - margin)
                v = rng.uniform(-L + j * cell + margin, -L + (j + 1) * cell - margin)
                pos = np.zeros(3)
                pos[axis] = sign * L
                pos[(axis + 1) % 3] = u
                pos[(axis + 2) % 3] = v
                pts.append((pos, layer_idx))
        return pts

    def sample(self, center: np.ndarray, rng: np.random.Generator) -> ConstellationResult:
        """Constellation baru di sekitar `center` (= nucleus = T2)."""
        center = np.asarray(center, dtype=float)
        points, layer_ids, face_ids = [], [], []
        for fi, face in enumerate(FACES):
            for pos_local, layer_idx in self._sample_face(face, rng):
                if np.linalg.norm(pos_local) < self.min_radius_from_center:
                    continue
                points.append(center + pos_local)
                layer_ids.append(layer_idx)
                face_ids.append(fi)
        return ConstellationResult(
            nucleus=center.copy(),
            points=np.array(points) if points else np.zeros((0, 3)),
            layer_ids=np.array(layer_ids, dtype=np.int32),
            face_ids=np.array(face_ids, dtype=np.int32),
            L_values=self.L_values,
        )

    # ------------------------------------------------------------------ #
    # Pemilihan titik awal (konsep "find_path" stargaze)
    # ------------------------------------------------------------------ #
    def face_candidates(self, face_names: list[str], resolution: int = 30):
        """Grid kandidat titik awal (koordinat LOKAL thd nucleus) pada face terluar."""
        L = self.L_values[-1]
        g = np.linspace(-L, L, resolution)
        G1, G2 = np.meshgrid(g, g)
        G1, G2 = G1.ravel(), G2.ravel()
        pts, fids = [], []
        for name in face_names:
            fi = _FACE_INDEX[name]
            axis, sign = FACES[fi]["axis"], FACES[fi]["sign"]
            P = np.zeros((G1.size, 3))
            P[:, axis] = sign * L
            P[:, (axis + 1) % 3] = G1
            P[:, (axis + 2) % 3] = G2
            pts.append(P)
            fids.append(np.full(G1.size, fi))
        return np.vstack(pts), np.concatenate(fids)

    @staticmethod
    def segment_clearance(points_local: np.ndarray, starts_local: np.ndarray) -> np.ndarray:
        """Jarak tiap titik rintangan (M,3) ke segmen start_n -> origin (N,3) => (M,N)."""
        S = starts_local
        S2 = np.maximum(np.einsum("nk,nk->n", S, S), 1e-12)
        dots = points_local @ S.T                                   # (M,N)
        t = np.clip(dots / S2[None, :], 0.0, 1.0)
        p2 = np.einsum("mk,mk->m", points_local, points_local)[:, None]
        d2 = p2 - 2.0 * t * dots + (t ** 2) * S2[None, :]
        return np.sqrt(np.maximum(d2, 0.0))

    def select_start_point(self, result: ConstellationResult, rng: np.random.Generator,
                           allowed_faces: list[str], reach_filter=None,
                           mode: str = "random_safe", min_clearance: float = 0.08,
                           resolution: int = 30):
        """Pilih titik awal S (dunia) pada face terluar.

        mode = "max"         : clearance garis maksimum (persis stargaze)
        mode = "random_safe" : acak di antara kandidat dengan clearance >= min_clearance
                               (lebih beragam & lebih menyulitkan); fallback ke "max".
        reach_filter(points_world (N,3)) -> bool (N,) memfilter kandidat yang reachable.
        Return (S_world, clearance, face_id) atau None bila tidak ada kandidat.
        """
        cand_local, fids = self.face_candidates(allowed_faces, resolution)
        cand_world = result.nucleus[None, :] + cand_local
        keep = np.ones(len(cand_local), dtype=bool)
        if reach_filter is not None:
            keep &= np.asarray(reach_filter(cand_world), dtype=bool)
        if not np.any(keep):
            return None
        cand_local, cand_world, fids = cand_local[keep], cand_world[keep], fids[keep]

        if result.points.shape[0] == 0:
            clear = np.full(len(cand_local), np.inf)
        else:
            pts_local = result.points - result.nucleus[None, :]
            clear = self.segment_clearance(pts_local, cand_local).min(axis=0)   # (N,)

        if mode == "random_safe":
            ok = np.where(clear >= min_clearance)[0]
            if ok.size > 0:
                i = int(rng.choice(ok))
                return cand_world[i], float(clear[i]), int(fids[i])
        i = int(np.argmax(clear))
        return cand_world[i], float(clear[i]), int(fids[i])

    def allowed_start_faces(self, nucleus: np.ndarray, base_position: np.ndarray,
                            forbidden: list[str] | None = None,
                            initial_face: str = "auto") -> list[str]:
        """Face pencarian = face awal + 4 face tetangga (face berseberangan dibuang) - forbidden."""
        names = [f["name"] for f in FACES]
        if initial_face == "auto":          # face yang menghadap base robot
            to_base = np.asarray(base_position, dtype=float) - np.asarray(nucleus, dtype=float)
            to_base[2] = 0.0                # hanya arah horizontal (lantai ditangani 'forbidden')
            scores = [FACES[i]["sign"] * to_base[FACES[i]["axis"]] for i in range(6)]
            initial_face = names[int(np.argmax(scores))]
        ini = FACES[_FACE_INDEX[initial_face]]
        opposite = names[_FACE_INDEX[("-" if ini["sign"] > 0 else "+") + "xyz"[ini["axis"]]]]
        out = [n for n in names if n != opposite]
        for f in (forbidden or []):
            if f in out:
                out.remove(f)
        return out
