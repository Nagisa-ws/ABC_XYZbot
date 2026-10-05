"""
Mission1/eval_mission1.py
-------------------------
Evaluasi Misi 1 (Target Reaching (threshold final 5 mm)) dengan MuJoCo viewer, memakai policy ONNX terbaru.

Jalankan (tanpa argumen):
    python eval_mission1.py            # dari folder Mission1/
    python Mission1/eval_mission1.py   # dari root proyek

Semua pengaturan (jumlah skenario, viewer, ONNX, CSV) ada di configs/eval_config.yaml.
Di macOS, jalankan dengan 'mjpython' (syarat viewer MuJoCo).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission1.eval_envs import Stage1ReachEvalEnv
from shared.evaluation import run_evaluation


def main():
    return run_evaluation(mission_id=1, project_root=ROOT,
                          eval_cfg_path=str(ROOT / "Mission1" / "configs" / "eval_config.yaml"),
                          env_cls=Stage1ReachEvalEnv)


if __name__ == "__main__":
    main()
