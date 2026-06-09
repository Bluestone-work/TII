"""
Method3 兼容导出层。

历史上，观测摘要、maneuver 执行接口和高层 reward 都混在这个文件里。
现在已经按职责拆分到:

- interaction_observation_utils.py
- interaction_execution_utils.py
- interaction_reward_utils.py

保留这个文件只为兼容旧 import，避免其他脚本/checkpoint 工具直接引用时报错。
"""

from gnn_marl_training.interaction_execution_utils import (
    build_interaction_subgoal_offset,
    compute_tracking_controller_cmd,
)
from gnn_marl_training.interaction_observation_utils import (
    SocialRiskSummary,
    build_high_level_policy_features,
    build_interaction_neighbor_token,
    compute_progress_delta_signal,
    compute_social_risk_summary,
    compute_stuck_score,
)
from gnn_marl_training.interaction_reward_utils import (
    Method3RewardTerms,
    compute_method3_reward_terms,
)

__all__ = [
    "SocialRiskSummary",
    "Method3RewardTerms",
    "build_high_level_policy_features",
    "build_interaction_neighbor_token",
    "build_interaction_subgoal_offset",
    "compute_method3_reward_terms",
    "compute_progress_delta_signal",
    "compute_social_risk_summary",
    "compute_stuck_score",
    "compute_tracking_controller_cmd",
]
