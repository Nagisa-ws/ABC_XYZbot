"""
shared/training/callbacks.py
----------------------------
Callback SB3 yang dipakai bersama oleh semua misi.

  ProgressCallback          : cetak progres / SPS / ETA berkala
  ReasonRateCallback        : log alasan terminasi episode (success, collision, ...) ke TensorBoard
  AdaptiveThresholdCallback : curriculum threshold adaptif Misi 1
  BestModelEvalCallback     : evaluasi deterministik berkala + simpan best_model
                              (kriteria sadar-curriculum, bukan sekadar mean reward)

Semua callback membaca `info["termination_reason"]` yang diisi BaseArmEnv.step().
"""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization

from shared.debug_common.common import format_duration

REASONS = ("success", "timeout", "collision", "self_collision", "floor_contact")


# ---------------------------------------------------------------------- #
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, interval_sec: float = 15.0, reason_cb=None):
        super().__init__()
        self.total = max(1, int(total_timesteps))
        self.interval = float(interval_sec)
        self.reason_cb = reason_cb
        self._t0 = self._last = 0.0

    def _on_training_start(self):
        self._t0 = self._last = time.time()
        self._n0 = self.num_timesteps

    def _on_step(self) -> bool:
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            done = self.num_timesteps - self._n0
            sps = done / max(now - self._t0, 1e-9)
            eta = (self.total - done) / sps if sps > 0 else float("inf")
            extra = ""
            if self.reason_cb is not None and self.reason_cb.n_episodes > 0:
                extra = f" | success={self.reason_cb.rate('success') * 100:.0f}%"
            print(f"[train] {self.num_timesteps}/{self.total} ({100 * done / self.total:.1f}%)"
                  f" | {sps:.0f} SPS | elapsed={format_duration(now - self._t0)}"
                  f" | ETA={format_duration(eta) if np.isfinite(eta) else '?'}{extra}", flush=True)
        return True


# ---------------------------------------------------------------------- #
class ReasonRateCallback(BaseCallback):
    """Statistik episode training bergulir (window) -> TensorBoard."""

    def __init__(self, window: int = 100):
        super().__init__()
        self.window = int(window)
        self._reasons: deque = deque(maxlen=self.window)
        self._extras: dict[str, deque] = {k: deque(maxlen=self.window) for k in
                                          ("final_distance", "cross_track_mean", "cross_track_max",
                                           "min_clearance")}
        self.n_episodes = 0

    # ---- akses statistik (dipakai callback lain) ----
    def rate(self, reason: str) -> float:
        return self._reasons.count(reason) / len(self._reasons) if self._reasons else 0.0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.n_episodes += 1
            self._reasons.append(info.get("termination_reason") or "timeout")
            for k, buf in self._extras.items():
                v = info.get("final_distance") if k == "final_distance" else info.get(k)
                if v is not None and np.isfinite(v):
                    buf.append(float(v))
        return True

    def _on_rollout_end(self):
        if not self._reasons:
            return
        c = Counter(self._reasons)
        n = len(self._reasons)
        for r in REASONS:
            self.logger.record(f"reasons/{r}_rate", c.get(r, 0) / n)
        self.logger.record("rollout/success_rate", c.get("success", 0) / n)
        for k, buf in self._extras.items():
            if buf:
                self.logger.record(f"rollout/{k}", float(np.mean(buf)))


# ---------------------------------------------------------------------- #
class AdaptiveThresholdCallback(BaseCallback):
    """Curriculum Misi 1: r_thresh 0.10 -> 0.05 -> 0.02 -> 0.005 m.

    Aturan:
      * Hanya episode yang SELESAI pada threshold aktif yang dihitung
        (info["success_threshold"]) -> episode sisa dari level lama tidak
        mencemari statistik level baru.
      * Evaluasi tiap `check_every` episode setelah `warmup_episodes`; hanya
        SATU level naik per evaluasi, lalu window dikosongkan (tidak ada cascade).
      * Bila success rate (window terakhir) >= `advance_rate` -> level++.
      * Hanya reward_success yang diskalakan: R_base * multiplier^level; semua
        suku reward lain tetap.
      * State disimpan ke JSON (resume aman), bukan lewat file temp/config.
    """

    def __init__(self, thresholds, advance_rate=0.85, check_every=50, warmup=100,
                 window=100, multiplier_per_level=1.5, eval_env=None,
                 state_path: str | None = None, reason_cb: ReasonRateCallback | None = None):
        super().__init__()
        self.thresholds = [float(t) for t in thresholds]
        self.advance_rate = float(advance_rate)
        self.check_every = int(check_every)
        self.warmup = int(warmup)
        self.window = int(window)
        self.mult = float(multiplier_per_level)
        self.eval_env = eval_env
        self.state_path = Path(state_path) if state_path else None
        self.level = 0
        self._buf: deque = deque(maxlen=self.window)
        self._eps_at_level = 0
        self._since_check = 0
        self.history: list[dict] = []

    # ---- API ----
    @property
    def threshold(self) -> float:
        return self.thresholds[self.level]

    @property
    def multiplier(self) -> float:
        return self.mult ** self.level

    @property
    def is_final_level(self) -> bool:
        return self.level >= len(self.thresholds) - 1

    def _apply(self):
        for env in (self.training_env, self.eval_env):
            if env is not None:
                env.env_method("set_curriculum", self.threshold, self.multiplier)

    def _save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(
                {"level": self.level, "threshold": self.threshold, "multiplier": self.multiplier,
                 "timesteps": int(self.num_timesteps), "history": self.history}, indent=2),
                encoding="utf-8")

    def _on_training_start(self):
        self._apply()
        self._save()
        print(f"[adaptive] level 0: threshold={self.threshold * 100:.1f} cm, "
              f"success x{self.multiplier:.2f}", flush=True)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            if abs(float(info.get("success_threshold", -1)) - self.threshold) > 1e-9:
                continue                       # episode dari level sebelumnya
            self._buf.append(1.0 if info.get("termination_reason") == "success" else 0.0)
            self._eps_at_level += 1
            self._since_check += 1
            if (self._eps_at_level >= self.warmup and self._since_check >= self.check_every
                    and len(self._buf) >= min(self.window, self.check_every)):
                self._since_check = 0
                rate = float(np.mean(self._buf))
                self.logger.record("curriculum/window_success_rate", rate)
                if rate >= self.advance_rate and not self.is_final_level:
                    self.history.append({"level": self.level, "threshold": self.threshold,
                                         "success_rate": rate, "timesteps": int(self.num_timesteps)})
                    self.level += 1
                    self._buf.clear()
                    self._eps_at_level = 0
                    self._apply()
                    self._save()
                    print(f"[adaptive] NAIK ke level {self.level}: threshold="
                          f"{self.threshold * 100:.1f} cm, success x{self.multiplier:.2f} "
                          f"(success rate {rate * 100:.0f}% @ {self.num_timesteps} steps)", flush=True)
        return True

    def _on_rollout_end(self):
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/threshold_m", self.threshold)
        self.logger.record("curriculum/success_reward_multiplier", self.multiplier)


# ---------------------------------------------------------------------- #
class DifficultyCurriculumCallback(BaseCallback):
    """Kurikulum kesulitan (Misi 3): koridor aman di sekitar lintasan referensi menyempit bertahap.

    `levels` = daftar {obstacle_margin, approach_margin} dari mudah (margin besar) ke sulit
    (margin kecil). Naik SATU level bila success rate pada level aktif >= `advance_rate`
    (window episode terakhir). Aturan sama dengan AdaptiveThresholdCallback Misi 1:
      * hanya episode yang SELESAI pada level aktif yang dihitung (info["difficulty_id"]),
      * minimum `warmup` episode per level, evaluasi tiap `check_every` episode,
      * window dikosongkan setiap naik level (tidak ada cascade),
      * state disimpan ke JSON.
    Env harus punya `set_difficulty(level, obstacle_margin, approach_margin)`.
    """

    def __init__(self, levels, advance_rate=0.80, check_every=50, warmup=100, window=100,
                 eval_env=None, state_path: str | None = None):
        super().__init__()
        self.levels = [dict(l) for l in levels]
        self.advance_rate = float(advance_rate)
        self.check_every = int(check_every)
        self.warmup = int(warmup)
        self.window = int(window)
        self.eval_env = eval_env
        self.state_path = Path(state_path) if state_path else None
        self.level = 0
        self._buf: deque = deque(maxlen=self.window)
        self._eps_at_level = 0
        self._since_check = 0
        self.history: list[dict] = []

    @property
    def is_final_level(self) -> bool:
        return self.level >= len(self.levels) - 1

    def _apply(self):
        lv = self.levels[self.level]
        for env in (self.training_env, self.eval_env):
            if env is not None:
                env.env_method("set_difficulty", self.level, **{k: float(v) for k, v in lv.items()})

    def _save(self):
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(
                {"level": self.level, "levels": self.levels, "timesteps": int(self.num_timesteps),
                 "history": self.history}, indent=2), encoding="utf-8")

    def _describe(self) -> str:
        lv = self.levels[self.level]
        reach = ""
        if "reach_r_max" in lv or "reach_z_max" in lv:
            reach = f", T1 r<={lv.get('reach_r_max', '-')} z<={lv.get('reach_z_max', '-')} m"
        return (f"level {self.level}/{len(self.levels) - 1}: margin lintasan "
                f"{lv['obstacle_margin'] * 100:.0f} cm, margin pendekatan {lv['approach_margin'] * 100:.0f} cm{reach}")

    def _on_training_start(self):
        self._apply()
        self._save()
        print(f"[kurikulum] {self._describe()}", flush=True)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done or int(info.get("difficulty_id", -1)) != self.level:
                continue
            self._buf.append(1.0 if info.get("termination_reason") == "success" else 0.0)
            self._eps_at_level += 1
            self._since_check += 1
            if (self._eps_at_level >= self.warmup and self._since_check >= self.check_every
                    and len(self._buf) >= min(self.window, self.check_every)):
                self._since_check = 0
                rate = float(np.mean(self._buf))
                self.logger.record("curriculum/window_success_rate", rate)
                if rate >= self.advance_rate and not self.is_final_level:
                    self.history.append({"level": self.level, "success_rate": rate,
                                         "timesteps": int(self.num_timesteps)})
                    self.level += 1
                    self._buf.clear()
                    self._eps_at_level = 0
                    self._apply()
                    self._save()
                    print(f"[kurikulum] NAIK ke {self._describe()} (success {rate * 100:.0f}% "
                          f"@ {self.num_timesteps} langkah)", flush=True)
        return True

    def _on_rollout_end(self):
        lv = self.levels[self.level]
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/obstacle_margin_m", lv["obstacle_margin"])


# ---------------------------------------------------------------------- #
class CriticWarmupCallback(BaseCallback):
    """Fine-tuning aman setelah transfer: bekukan ACTOR, latih hanya CRITIC selama `steps`.

    Masalah yang diatasi: critic (value network) Misi baru masih acak. Pada update PPO
    pertama, advantage dihitung dari critic yang salah -> gradien actor menyesatkan dan
    policy hasil transfer yang sudah bagus rusak total (terukur: 75% sukses -> 0% dalam
    satu iterasi evaluasi). Selama warm-up, policy tetap = policy Misi sebelumnya sehingga
    data rollout bagus, dan critic belajar menilai policy itu dulu.
    """

    def __init__(self, steps: int):
        super().__init__()
        self.steps = int(steps)
        self._frozen = False

    def _actor_params(self):
        pol = self.model.policy
        return [*pol.mlp_extractor.policy_net.parameters(), *pol.action_net.parameters(), pol.log_std]

    def _set_frozen(self, frozen: bool):
        for prm in self._actor_params():
            prm.requires_grad_(not frozen)
        self._frozen = frozen

    def _on_training_start(self):
        if self.steps > 0:
            self._set_frozen(True)
            print(f"[warmup] actor dibekukan; hanya critic yang dilatih sampai {self.steps:,} langkah", flush=True)

    def _on_step(self) -> bool:
        if self._frozen and self.num_timesteps >= self.steps:
            self._set_frozen(False)
            print(f"[warmup] selesai @ {self.num_timesteps:,} langkah -> actor dilatih bersama critic", flush=True)
        return True

    def _on_rollout_end(self):
        self.logger.record("train/actor_frozen", float(self._frozen))


# ---------------------------------------------------------------------- #
class BestModelEvalCallback(BaseCallback):
    """Evaluasi deterministik berkala pada env eval (tanpa noise/randomisasi sensor).

    Kriteria 'terbaik' (leksikografis): (level curriculum, success_rate, -mean_final_dist).
    Mean reward TIDAK dipakai karena skalanya berubah saat curriculum naik.
    Menyimpan best_model.zip + best_vecnormalize.pkl di `save_dir`.
    """

    def __init__(self, eval_env, n_episodes: int = 20, eval_freq: int = 25_000,
                 save_dir: str = "checkpoints", level_fn=None, max_steps: int = 2000,
                 eval_seed: int = 424_242, eval_at_start: bool = False):
        super().__init__()
        self.eval_env = eval_env
        self.n_episodes = int(n_episodes)
        self.eval_freq = int(eval_freq)
        self.save_dir = Path(save_dir)
        self.level_fn = level_fn or (lambda: 0)
        self.max_steps = int(max_steps)
        self.eval_seed = int(eval_seed)          # skenario evaluasi SAMA tiap putaran -> perbandingan adil
        self.eval_at_start = bool(eval_at_start) # evaluasi policy awal (mis. hasil transfer) sebagai baseline
        self._best_key = None
        self._last_eval = 0
        self.best_info: dict = {}

    def _evaluate(self) -> dict:
        sync_envs_normalization(self.training_env, self.eval_env)
        rets, succ, dists, steps, reasons = [], [], [], [], Counter()
        ct_mean, min_cl = [], []
        for ep in range(self.n_episodes):
            self.eval_env.seed(self.eval_seed + ep)
            obs = self.eval_env.reset()
            done, ret, n = False, 0.0, 0
            while not done and n < self.max_steps:
                act, _ = self.model.predict(obs, deterministic=True)
                obs, r, dones, infos = self.eval_env.step(act)
                ret += float(r[0]); n += 1
                done = bool(dones[0])
            info = infos[0]
            reason = info.get("termination_reason") or "timeout"
            reasons[reason] += 1
            succ.append(reason == "success")
            rets.append(ret); steps.append(n)
            dists.append(float(info.get("final_distance", np.nan)))
            if "cross_track_mean" in info:
                ct_mean.append(float(info["cross_track_mean"]))
            if "min_clearance" in info:
                min_cl.append(float(info["min_clearance"]))
        res = {"success_rate": float(np.mean(succ)), "mean_return": float(np.mean(rets)),
               "mean_steps": float(np.mean(steps)), "mean_final_dist": float(np.nanmean(dists)),
               "reasons": dict(reasons)}
        if ct_mean:
            res["cross_track_mean"] = float(np.mean(ct_mean))
        if min_cl:
            res["min_clearance"] = float(np.min(min_cl))
        return res

    def _run_and_maybe_save(self, tag: str = ""):
        res = self._evaluate()
        lvl = int(self.level_fn())
        for k, v in res.items():
            if isinstance(v, (int, float)):
                self.logger.record(f"eval/{k}", v)
        # kriteria: (level curriculum, success rate, -ketelitian). Ketelitian = cross-track bila ada
        # (tugas path), selain itu jarak akhir. Mean reward tidak dipakai (skalanya berubah).
        precision = res.get("cross_track_mean", res["mean_final_dist"])
        key = (lvl, round(res["success_rate"], 4), -precision)
        improved = self._best_key is None or key > self._best_key
        if improved:
            self._best_key = key
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(self.save_dir / "best_model")
            self.model.get_vec_normalize_env().save(str(self.save_dir / "best_vecnormalize.pkl"))
            self.best_info = {**res, "level": lvl, "timesteps": int(self.num_timesteps)}
            (self.save_dir / "best_model_info.json").write_text(json.dumps(self.best_info, indent=2),
                                                                encoding="utf-8")
        ct = f" cross-track={res['cross_track_mean'] * 1000:.0f}mm" if "cross_track_mean" in res else ""
        print(f"[eval]{tag} steps={self.num_timesteps} level={lvl} success={res['success_rate'] * 100:.0f}%"
              f" final_dist={res['mean_final_dist'] * 100:.1f}cm{ct} reasons={res['reasons']}"
              f"{'  <- best' if improved else ''}", flush=True)

    def _on_training_start(self):
        if self.eval_at_start:
            self._run_and_maybe_save(" [awal/baseline]")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps
        self._run_and_maybe_save()
        return True
