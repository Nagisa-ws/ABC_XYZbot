"""
Mission3/benchmark_hybrid_mission3.py
-----------------------------------
Uji penerimaan Misi 3: RL saja vs hibrida (RL + LQR) pada skenario tetap, ambang 2 cm dan 5 mm.
Memakai ONNX terbaru di models/ dan parameter di shared/control/controller.yaml.

Jalankan (tanpa argumen):
    python Mission3/benchmark_hybrid_mission3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mission3.eval_envs import Stage3ObstacleEvalEnv
from shared.control.benchmark import run_benchmark_for_mission


def main():
    return run_benchmark_for_mission(mission_id=3, project_root=ROOT, env_cls=Stage3ObstacleEvalEnv,
                                     eval_cfg_path=str(ROOT / "Mission3" / "configs" / "eval_config.yaml"))


if __name__ == "__main__":
    main()
