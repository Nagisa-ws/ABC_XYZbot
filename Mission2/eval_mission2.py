"""
Mission2/eval_mission2.py
-------------------------
Evaluasi Misi 2 (Reaching T1 + Path Following T1->T2) dengan MuJoCo viewer, memakai policy ONNX terbaru.

Jalankan (tanpa argumen):
    python eval_mission2.py            # dari folder Mission2/
    python Mission2/eval_mission2.py   # dari root proyek

Semua pengaturan (jumlah skenario, viewer, ONNX, CSV) ada di configs/eval_config.yaml.
Di macOS, jalankan dengan 'mjpython' (syarat viewer MuJoCo).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission2.eval_envs import Stage2PathEvalEnv
from shared.evaluation import run_evaluation


def main():
    return run_evaluation(mission_id=2, project_root=ROOT,
                          eval_cfg_path=str(ROOT / "Mission2" / "configs" / "eval_config.yaml"),
                          env_cls=Stage2PathEvalEnv)


if __name__ == "__main__":
    main()
