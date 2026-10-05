"""
shared/training/trainer.py
--------------------------
Pipeline training generik untuk semua misi:

  1. sanity check (IK / env / controller skrip)            [debug tools]
  2. VecEnv (Subproc/Dummy) + VecNormalize (+ env eval terpisah)
  3. PPO (hyperparameter dari ppo_config.yaml); opsional TRANSFER dari misi sebelumnya
  4. callback: reason-rate, (adaptive), progress/ETA, best-model eval, checkpoint
  5. TensorBoard (PPO bawaan + callback di atas)
  6. simpan final + best; ekspor ONNX berversi -> models/onnx_missionN_vK.onnx
  7. verify_onnx pada observasi mentah NYATA                [exporter]
  8. debug pasca-training (kurva, replay, ONNX vs checkpoint, robustness)

train_missionN.py hanya menyiapkan env class + hook khusus misi lalu memanggil
`train_mission()`.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from shared.config_utils import load_yaml
from shared.envs.obs_layout import OBS_DIM
from shared.exporter import (export_policy_to_onnx, next_versioned_onnx_path, verify_onnx,
                             vecnormalize_stats_from_pkl, write_onnx_metadata)
from shared.training.callbacks import (BestModelEvalCallback, CriticWarmupCallback,
                                       ProgressCallback, ReasonRateCallback)
from shared.training.transfer import transfer_from_previous

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU}


@dataclass
class TrainContext:
    mission_id: int
    project_root: Path
    cfg: dict
    env_cls: type
    env_cfg_path: str
    model_path: str
    ckpt_dir: Path
    logdir: Path
    models_dir: Path
    train_env: object = None
    eval_env: object = None
    model: PPO | None = None
    level_fn: Callable[[], int] = field(default=lambda: 0)
    extras: dict = field(default_factory=dict)

    def make_raw_env(self, training: bool = False, randomize: bool | None = None, seed: int = 0):
        """Satu env mentah (bukan VecEnv) untuk debug/verifikasi."""
        return self.env_cls(model_path=self.model_path, config_path=self.env_cfg_path,
                            training=training, randomize=randomize, seed=seed)


@dataclass
class MissionHooks:
    pre_debug: Callable[[TrainContext], None] | None = None
    build_callbacks: Callable[[TrainContext], list] | None = None
    post_debug: Callable[[TrainContext, dict], None] | None = None


# ---------------------------------------------------------------------- #
def _make_env_thunk(ctx_env_cls, model_path, env_cfg_path, seed, training, randomize):
    def _init():
        env = ctx_env_cls(model_path=model_path, config_path=env_cfg_path, seed=seed,
                          training=training, randomize=randomize)
        return Monitor(env)
    return _init


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lr(cfg_ppo: dict):
    lr = float(cfg_ppo.get("learning_rate", 3e-4))
    if str(cfg_ppo.get("lr_schedule", "constant")) == "linear":
        return lambda progress: lr * max(progress, 0.05)
    return lr


def build_ppo(cfg: dict, train_env, logdir: Path, seed: int, device: str) -> PPO:
    p = cfg["ppo"]
    arch = list(p.get("net_arch", [128, 128]))
    policy_kwargs = dict(net_arch=dict(pi=arch, vf=arch),
                         activation_fn=_ACT[str(p.get("activation_fn", "tanh")).lower()])
    return PPO(
        "MlpPolicy", train_env, learning_rate=_lr(p), n_steps=int(p["n_steps"]),
        batch_size=int(p["batch_size"]), n_epochs=int(p["n_epochs"]), gamma=float(p["gamma"]),
        gae_lambda=float(p["gae_lambda"]), clip_range=float(p["clip_range"]),
        ent_coef=float(p["ent_coef"]), vf_coef=float(p["vf_coef"]),
        max_grad_norm=float(p["max_grad_norm"]),
        target_kl=None if p.get("target_kl") is None else float(p["target_kl"]),
        policy_kwargs=policy_kwargs, tensorboard_log=str(logdir), seed=seed, device=device,
        verbose=1)


# ---------------------------------------------------------------------- #
def collect_raw_observations(ctx: TrainContext, model_zip: str, vecnorm_pkl: str,
                             n_steps: int = 300, log=print) -> np.ndarray:
    """Observasi MENTAH nyata (env mode training, dgn noise) yang dihasilkan policy."""
    from shared.debug_common.common import make_predict_fn_sb3
    model = PPO.load(model_zip, device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(vecnorm_pkl)
    fn = make_predict_fn_sb3(model, mean, var, clip, eps)
    env = ctx.make_raw_env(training=True, randomize=True, seed=777)
    obs, _ = env.reset(seed=777)
    out = [obs]
    for _ in range(n_steps):
        obs, _, term, trunc, _ = env.step(fn(obs))
        out.append(obs)
        if term or trunc:
            obs, _ = env.reset()
    return np.array(out, dtype=np.float32)


def export_and_verify(ctx: TrainContext, model_zip: str, vecnorm_pkl: str, meta: dict,
                      log=print) -> dict:
    ecfg = ctx.cfg["export"]
    model = PPO.load(model_zip, device="cpu")
    mean, var, clip, eps = vecnormalize_stats_from_pkl(vecnorm_pkl)
    onnx_path = next_versioned_onnx_path(str(ctx.models_dir), ecfg["name_prefix"])
    export_policy_to_onnx(model, mean, var, clip, eps, str(onnx_path), OBS_DIM)
    write_onnx_metadata(str(onnx_path), {"mission": ctx.mission_id, "obs_dim": OBS_DIM,
                                         "source_model": model_zip, "vecnormalize": vecnorm_pkl,
                                         **meta})
    log(f"[onnx] diekspor -> {onnx_path}")
    obs_batch = collect_raw_observations(ctx, model_zip, vecnorm_pkl, log=log)
    max_d, mean_d, ok = verify_onnx(model, mean, var, clip, eps, str(onnx_path), OBS_DIM,
                                    tol=float(ecfg.get("verify_tolerance", 1e-4)),
                                    obs_batch=obs_batch)
    log(f"[onnx] verifikasi ({len(obs_batch)} observasi nyata): max|diff|={max_d:.2e} "
        f"mean|diff|={mean_d:.2e} -> {'SUKSES' if ok else 'GAGAL'}")
    return {"onnx_path": str(onnx_path), "max_diff": max_d, "mean_diff": mean_d, "verified": ok}


# ---------------------------------------------------------------------- #
def train_mission(*, mission_id: int, project_root: Path, ppo_cfg_path: str, env_cls: type,
                  env_cfg_path: str, model_path: str, hooks: MissionHooks | None = None,
                  log=print) -> dict:
    hooks = hooks or MissionHooks()
    root = Path(project_root)
    cfg = load_yaml(ppo_cfg_path)
    run, rt = cfg["run"], cfg.get("runtime", {})
    torch.set_num_threads(int(rt.get("torch_num_threads", 1)))
    device = str(rt.get("device", "cpu"))
    seed = int(run.get("seed", 0))
    _set_seeds(seed)

    # gagal-cepat: transfer butuh checkpoint misi sebelumnya (sebelum sanity check/pembuatan env)
    tc = cfg.get("transfer", {}) or {}
    if tc.get("enabled", False):
        for key in ("source_model", "source_vecnormalize"):
            if not (root / tc[key]).exists():
                raise FileNotFoundError(
                    f"Misi {mission_id} memuat bobot dari Misi {tc.get('source_mission', mission_id - 1)}, "
                    f"tetapi file belum ada: {root / tc[key]}\n"
                    f"Jalankan training Misi {tc.get('source_mission', mission_id - 1)} lebih dulu "
                    f"(python train_mission{tc.get('source_mission', mission_id - 1)}.py).")

    ctx = TrainContext(
        mission_id=mission_id, project_root=root, cfg=cfg, env_cls=env_cls,
        env_cfg_path=str(env_cfg_path), model_path=str(model_path),
        ckpt_dir=root / run["ckpt_dir"], logdir=root / run["logdir"],
        models_dir=root / cfg["export"]["onnx_dir"])
    for d in (ctx.ckpt_dir, ctx.logdir, ctx.models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. debug pra-training ----
    if hooks.pre_debug and cfg.get("debug", {}).get("run_sanity_check_before_training", True):
        log(f"\n=== Misi {mission_id}: sanity check pra-training ===")
        hooks.pre_debug(ctx)

    # ---- 2. environment ----
    n_envs = int(run["n_envs"])
    thunks = [_make_env_thunk(env_cls, ctx.model_path, ctx.env_cfg_path, seed + 1000 * i,
                              True, None) for i in range(n_envs)]
    kind = str(run.get("vec_env", "auto"))
    vec_cls = SubprocVecEnv if (kind == "subproc" or (kind == "auto" and n_envs > 1)) else DummyVecEnv
    norm = cfg["normalization"]
    train_env = VecNormalize(vec_cls(thunks), norm_obs=True, norm_reward=bool(norm.get("norm_reward", False)),
                             clip_obs=float(norm["clip_obs"]), epsilon=float(norm["epsilon"]))
    eval_env = VecNormalize(
        DummyVecEnv([_make_env_thunk(env_cls, ctx.model_path, ctx.env_cfg_path, seed + 99_999,
                                     False, False)]),
        norm_obs=True, norm_reward=False, training=False,
        clip_obs=float(norm["clip_obs"]), epsilon=float(norm["epsilon"]))
    ctx.train_env, ctx.eval_env = train_env, eval_env

    # ---- 3. PPO (+ transfer) ----
    model = build_ppo(cfg, train_env, ctx.logdir, seed, device)
    ctx.model = model
    tcfg = cfg.get("transfer", {}) or {}
    if tcfg.get("enabled", False):
        transfer_from_previous(model, train_env, tcfg, root, mission_id, log=log)
        # eval env memakai stats train (disinkron oleh callback), tetapi inisialisasi juga:
        eval_env.obs_rms = train_env.obs_rms
        # Bekukan statistik normalisasi. Bila terus diperbarui, mean/var dimensi lama bergeser ke
        # distribusi misi baru -> input actor berubah -> policy hasil transfer rusak MESKIPUN bobot
        # actor tidak diubah (terukur: 75% -> 0% dalam 75k langkah dengan actor dibekukan).
        if tcfg.get("freeze_obs_rms", True):
            train_env.training = False
            log("[transfer] statistik normalisasi observasi DIBEKUKAN (tidak diperbarui selama training)")

    # ---- 4. callbacks ----
    reason_cb = ReasonRateCallback(window=100)
    extra = hooks.build_callbacks(ctx) if hooks.build_callbacks else []
    total = int(run["timesteps"])
    warmup = []
    if tcfg.get("enabled", False) and int(tcfg.get("critic_warmup_steps", 0)) > 0:
        warmup = [CriticWarmupCallback(int(tcfg["critic_warmup_steps"]))]
    callbacks = [reason_cb, *warmup, *extra,
                 ProgressCallback(total, float(rt.get("progress_print_interval_sec", 15)), reason_cb),
                 BestModelEvalCallback(eval_env, int(run.get("eval_episodes", 20)),
                                       int(run.get("eval_freq", 25_000)), str(ctx.ckpt_dir),
                                       level_fn=lambda: ctx.level_fn(),
                                       eval_seed=int(run.get("eval_seed", 424_242)),
                                       eval_at_start=bool(tcfg.get("enabled", False))),
                 CheckpointCallback(max(1, int(run["ckpt_freq"]) // n_envs), str(ctx.ckpt_dir / "periodic"),
                                    name_prefix=f"mission{mission_id}", save_vecnormalize=True)]

    # ---- 5. training ----
    log(f"\n=== Misi {mission_id}: training {total:,} steps | {n_envs} env ({vec_cls.__name__}) "
        f"| TensorBoard: tensorboard --logdir {ctx.logdir} ===")
    t0 = time.time()
    interrupted = False
    try:
        model.learn(total_timesteps=total, callback=CallbackList(callbacks),
                    tb_log_name=f"mission{mission_id}", progress_bar=False)
    except KeyboardInterrupt:
        interrupted = True
        log("\n[train] dihentikan pengguna (Ctrl+C) - menyimpan & mengekspor model terbaik...")
    wall = time.time() - t0

    # ---- 6. simpan ----
    final_zip = ctx.ckpt_dir / "final_model.zip"
    final_pkl = ctx.ckpt_dir / "final_vecnormalize.pkl"
    model.save(str(final_zip))
    train_env.save(str(final_pkl))
    best_zip, best_pkl = ctx.ckpt_dir / "best_model.zip", ctx.ckpt_dir / "best_vecnormalize.pkl"
    use_best = str(cfg["export"].get("source", "best")) == "best" and best_zip.exists() and best_pkl.exists()
    src_zip, src_pkl = (best_zip, best_pkl) if use_best else (final_zip, final_pkl)
    log(f"[train] selesai dalam {wall / 60:.1f} menit. Sumber ONNX: {'best' if use_best else 'final'}")

    # ---- 7. ONNX + verifikasi ----
    export = export_and_verify(ctx, str(src_zip), str(src_pkl),
                               {"source": "best" if use_best else "final",
                                "timesteps": int(model.num_timesteps), "interrupted": interrupted})
    result = {"final_model": str(final_zip), "best_model": str(best_zip) if best_zip.exists() else None,
              "interrupted": interrupted, "wall_sec": wall, **export, "tb_logdir": str(ctx.logdir)}

    # ---- 8. debug pasca-training ----
    if hooks.post_debug and cfg.get("debug", {}).get("enabled", True):
        log(f"\n=== Misi {mission_id}: debug pasca-training ===")
        try:
            hooks.post_debug(ctx, result)
        except Exception as e:                                   # debug tidak boleh menggagalkan hasil
            log(f"[debug] peringatan: {type(e).__name__}: {e}")

    train_env.close()
    eval_env.close()
    log(f"\nSELESAI. ONNX: {export['onnx_path']}\nTensorBoard: tensorboard --logdir {ctx.logdir}")
    return result
