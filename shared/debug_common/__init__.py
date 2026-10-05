# shared/debug_common/__init__.py
from shared.debug_common.common import (
    EpisodeResult, ProgressTracker, format_duration, run_policy_episode,
    make_predict_fn_onnx, make_predict_fn_random, make_predict_fn_sb3, make_predict_fn_scripted,
    read_tb_scalars, plot_training_curves, plot_replay_trajectory,
    plot_randomization_robustness, plot_onnx_vs_checkpoint_diff,
)
from shared.debug_common.sanity import (
    sanity_check_ik, sanity_check_env, scripted_controller_check, run_all_sanity,
)

__all__ = [
    "EpisodeResult", "ProgressTracker", "format_duration", "run_policy_episode",
    "make_predict_fn_onnx", "make_predict_fn_random", "make_predict_fn_sb3",
    "make_predict_fn_scripted", "read_tb_scalars", "plot_training_curves",
    "plot_replay_trajectory", "plot_randomization_robustness", "plot_onnx_vs_checkpoint_diff",
    "sanity_check_ik", "sanity_check_env", "scripted_controller_check", "run_all_sanity",
]

from shared.debug_common.post_training import (
    evaluate_onnx, generic_post_training, onnx_session, onnx_vs_checkpoint, robustness_study,
)
__all__ += ["evaluate_onnx", "generic_post_training", "onnx_session", "onnx_vs_checkpoint",
            "robustness_study"]
