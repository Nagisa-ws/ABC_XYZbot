"""
shared/training/transfer.py
---------------------------
Transfer bobot Misi N -> Misi N+1 (curriculum).

Karena observasi terpadu berdimensi tetap (obs_layout.OBS_DIM), arsitektur sama
dan bobot bisa dimuat langsung. Yang perlu ditangani adalah dimensi yang BARU
AKTIF di misi baru (mis. fitur path di Misi 2, fitur rintangan di Misi 3):

  * Di misi lama dimensi itu konstan -> varians statistik normalisasi ~ 0 dan
    bobot input-nya tidak pernah mendapat gradien (masih acak/ortogonal).
  * Bila dibiarkan, observasi baru ternormalisasi -> nilai ekstrem (kena clip)
    dan bobot acak mengacaukan policy hasil Misi N.

Penanganan:
  1. Bobot layer pertama (policy & value) untuk dimensi baru di-NOL-kan, sehingga
     di awal Misi N+1 perilaku = policy Misi N, lalu fitur baru dipelajari bertahap.
  2. obs_rms: statistik dimensi LAMA dipertahankan (input actor tidak bergeser ->
     perilaku policy lama terjaga). Statistik dimensi BARU diisi dari data nyata env
     misi baru (rollout dengan policy hasil transfer, `warm_stats_steps` langkah),
     bukan tebakan mean 0 / var 1. `count` dibuat cukup besar (`obs_rms_count`) agar
     statistik tidak melompat pada iterasi awal.
  3. log_std: dipertahankan dari misi sebelumnya. Opsional: `log_std_max` (batas atas)
     atau `log_std_init` (ganti seragam). Mereset ke nilai besar menambah derau aksi
     dan merusak ketelitian yang sudah dipelajari.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from shared.envs.obs_layout import ACTIVE_DIMS, OBS_DIM

_MIN_VAR = 1.0e-4        # lantai varians untuk dimensi baru (hindari pembagian ~0)


def _load_vecnorm_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _collect_raw_obs(model: PPO, vecnorm, n_steps: int) -> np.ndarray:
    """Observasi MENTAH dari env misi baru, digerakkan oleh policy hasil transfer (stokastik)."""
    inner = vecnorm.venv
    raw = inner.reset()
    out = [raw.copy()]
    for _ in range(max(1, n_steps // max(1, inner.num_envs))):
        act, _ = model.predict(vecnorm.normalize_obs(raw), deterministic=False)
        raw, _, _, _ = inner.step(act)
        out.append(raw.copy())
    return np.concatenate(out, axis=0)


def transfer_from_previous(model: PPO, vecnorm, tcfg: dict, project_root: Path,
                           target_mission: int, log=print) -> dict:
    """Muat bobot + statistik normalisasi Misi sebelumnya ke `model` & `vecnorm` (in-place).

    tcfg: {source_model, source_vecnormalize, source_mission, zero_new_input_weights,
           obs_rms_count, warm_stats_steps, log_std_init, log_std_max}
    """
    src_model = Path(project_root) / tcfg["source_model"]
    src_norm = Path(project_root) / tcfg["source_vecnormalize"]
    if not src_model.exists():
        raise FileNotFoundError(
            f"Checkpoint Misi {tcfg.get('source_mission')} tidak ditemukan: {src_model}\n"
            f"Jalankan training misi sebelumnya dulu.")
    if not src_norm.exists():
        raise FileNotFoundError(f"Statistik VecNormalize tidak ditemukan: {src_norm}")

    prev = PPO.load(str(src_model), device="cpu")
    if prev.observation_space.shape != model.observation_space.shape:
        raise ValueError(f"Dimensi observasi berbeda: {prev.observation_space.shape} vs "
                         f"{model.observation_space.shape}")
    model.policy.load_state_dict(prev.policy.state_dict(), strict=True)

    src_mission = int(tcfg.get("source_mission", target_mission - 1))
    new_dims = ACTIVE_DIMS[target_mission] & ~ACTIVE_DIMS[src_mission]
    idx = torch.as_tensor(np.where(new_dims)[0], dtype=torch.long)

    if tcfg.get("zero_new_input_weights", True) and len(idx) > 0:
        with torch.no_grad():
            for first in (model.policy.mlp_extractor.policy_net[0],
                          model.policy.mlp_extractor.value_net[0]):
                first.weight[:, idx] = 0.0

    # ---- statistik normalisasi ----
    old = _load_vecnorm_pickle(str(src_norm))
    mean = np.array(old.obs_rms.mean, dtype=np.float64)
    var = np.array(old.obs_rms.var, dtype=np.float64)
    mean[new_dims], var[new_dims] = 0.0, 1.0                      # sementara
    vecnorm.obs_rms.mean, vecnorm.obs_rms.var = mean, var
    vecnorm.obs_rms.count = float(tcfg.get("obs_rms_count", 50_000.0))

    warm = int(tcfg.get("warm_stats_steps", 0))
    if warm > 0 and new_dims.any():
        raw = _collect_raw_obs(model, vecnorm, warm)
        mean[new_dims] = raw[:, new_dims].mean(axis=0)
        var[new_dims] = np.maximum(raw[:, new_dims].var(axis=0), _MIN_VAR)
        vecnorm.obs_rms.mean, vecnorm.obs_rms.var = mean, var
        log(f"[transfer] statistik {int(new_dims.sum())} dimensi baru diisi dari {len(raw)} observasi nyata")

    # ---- eksplorasi (log_std) ----
    with torch.no_grad():
        if tcfg.get("log_std_init") is not None:
            model.policy.log_std.fill_(float(tcfg["log_std_init"]))
        if tcfg.get("log_std_max") is not None:
            model.policy.log_std.clamp_(max=float(tcfg["log_std_max"]))
    std = np.exp(model.policy.log_std.detach().numpy())

    info = {"source_model": str(src_model), "n_new_dims": int(new_dims.sum()),
            "new_dims": np.where(new_dims)[0].tolist(), "obs_dim": OBS_DIM}
    log(f"[transfer] bobot Misi {src_mission} dimuat dari {src_model.name}; "
        f"{info['n_new_dims']} dimensi baru aktif (bobot di-nol-kan: "
        f"{bool(tcfg.get('zero_new_input_weights', True))}) | std aksi = {np.round(std, 2).tolist()}")
    return info
