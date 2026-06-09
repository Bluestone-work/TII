#!/usr/bin/env python3
"""
MAPPO 训练脚本（逐步构建版本）

--model_type mlp  : 基础 MAPPO-MLP-LSTM（无 GNN，带 LSTM）— 用于先跑通 MAPPO
--model_type gat  : GNN-MAPPO（GAT + LSTM）+ 课程学习
"""
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import argparse
import os
import sys
import subprocess
import logging
import warnings
import math
import numbers
import re
import numpy as np
import json
import csv
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO']  = '0'   # 消除 Ray GPU FutureWarning
os.environ['RAY_DISABLE_METRICS_COLLECTION']       = '1'   # 彻底关闭 metrics 采集（解决 rpc_code:14）
os.environ['RAY_DISABLE_IMPORT_METRICS_REPORTER']  = '1'   # 禁用 Prometheus metrics exporter
os.environ['RAY_memory_monitor_refresh_ms']        = '0'   # 关闭 Ray 内存监视器，避免误判杀进程
os.environ['RAY_event_stats_print_interval_ms']    = '0'   # 减少 raylet 事件统计后台负担
os.environ['RAY_metrics_export_port']              = '0'   # 不绑定 metrics 端口
warnings.filterwarnings('ignore', category=FutureWarning, module='ray')
# Ray 2.54 旧 API 栈内部会触发 `_get_slice_indices` 的 DeprecationWarning 日志。
# 该告警来自 ray._common.deprecation 的 logger.warning（非 Python warnings）。
# 仅降噪日志，不改变训练行为。
logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)

class MARLMetricsCallback(DefaultCallbacks):
    """
    自定义回调函数：用于把环境 info 字典里的详细子奖励和状态
    提取出来，发送给 TensorBoard 进行可视化监控。
    """
    def on_episode_step(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # 遍历环境中的每一个智能体 (agent_0, agent_1...)
        for agent_id in episode.get_agents():
            # 获取环境返回的 info 字典
            info = episode.last_info_for(agent_id)
            if info:
                # 初始化 user_data 用于在 episode 内累计数据
                if "custom_metrics_history" not in episode.user_data:
                    episode.user_data["custom_metrics_history"] = {}
                
                # 我们挑选 info 字典里想要监控的关键字段
                keys_to_track = [
                    'path_progress_reward', 'goal_progress_reward',
                    'progress_reward', 'heading_reward', 'lateral_penalty', 
                    'obstacle_penalty', 'social_keep_right_reward', 
                    'wrong_dir_penalty', 'min_dist', 'front_min', 'side_min',
                    'predictive_social_risk', 'predictive_front_risk',
                    'predictive_social_penalty', 'predictive_front_penalty',
                    'social_proximity_penalty', 'predictive_penalty',
                    'close_obstacle_penalty', 'front_close_penalty', 'side_close_penalty',
                    'subgoal_progress_reward', 'yield_compliance_reward',
                    'interaction_mode_reward', 'interaction_mode_penalty',
                    'risk_aware_forward_penalty', 'safe_turn_reward',
                    'head_on_avoidance_reward', 'corner_escape_reward', 'corner_escape_active',
                    'stuck_score', 'clear_reward', 'social_risk', 'social_risk_delta',
                    'window_progress_reward', 'window_path_progress_reward', 'window_goal_progress_reward',
                    'replan_cost', 'replan_freq_penalty', 'replan_time_penalty',
                    'replan_wall_time_sec', 'replan_attempted', 'replan_success',
                    'recent_replan_count',
                    'reward_risk_signal', 'reward_risk_gate',
                    'reward_navigation_scale', 'reward_avoidance_scale',
                    'effective_time_penalty', 'front_close_ratio', 'side_close_ratio',
                    'interaction_mode_id', 'interaction_in_conflict', 'interaction_has_token',
                    'interaction_wait_age_norm', 'interaction_severity', 'interaction_turn_sign',
                    'interaction_partner_dist', 'policy_interaction_mode_id',
                    'effective_interaction_mode_id', 'executed_behavior_mode_id',
                    'best_gap_width', 'best_gap_clearance',
                    'high_level_nav_reward', 'high_level_interaction_reward',
                    'high_level_safety_reward', 'high_level_efficiency_penalty',
                    'high_level_policy_penalty',
                    # Phase 6/7: option outcome reward
                    'option_progress_reward', 'option_clearance_reward', 'option_safety_reward',
                    'option_completion_bonus', 'option_failure_penalty', 'option_timeout_penalty',
                    'safe_turn_reward_outcome', 'wrong_turn_penalty', 'random_turn_penalty',
                    'spin_without_progress_penalty', 'idle_without_progress_penalty',
                    'conservative_mode_penalty',
                    'option_switch_penalty', 'infeasible_action_penalty',
                    'pair_cooperative_reward', 'pair_competitive_penalty',
                    'obstacle_proximity_penalty', 'backoff_release_reward',
                    'detour_loop_penalty', 'detour_active_penalty',
                    # Option state
                    'active_option_mode', 'option_hold_remaining_frac', 'option_elapsed_frac',
                    'detour_phase', 'detour_lateral_displacement', 'detour_active',
                    'rolling_subgoal_suppressed',
                    # Feasibility
                    'feasible_go', 'feasible_wait', 'feasible_backoff',
                    'feasible_detour_left', 'feasible_detour_right', 'feasible_slow_follow',
                    'action_mask_any_zero',
                    # Safe turn
                    'risk_gate', 'ttc_risk', 'left_safety_score', 'right_safety_score',
                    'correct_turn', 'wrong_turn',
                    # Collision
                    'collision_self_fault',
                    # Path projection progress
                    'path_s', 'prev_path_s', 'path_projection_progress_delta', 'path_projection_progress_window',
                    'closest_dist_to_path', 'cross_track_error', 'cross_track_penalty',
                    'goal_progress_delta', 'local_goal_progress_delta', 'guide_target_progress_delta',
                    'positive_path_projection_progress', 'negative_path_projection_progress',
                    'progress_positive', 'progress_source_id', 'R_progress',
                    'obstacle_risk_drop', 'ttc_improvement', 'front_blocked_ratio_delta',
                    'risk_reduced', 'path_projection_valid',
                    'front_left_min', 'front_center_min', 'front_right_min', 'rear_min',
                    'clearance_asymmetry',
                    # Potential-drop reward
                    'phi_goal_prev', 'phi_goal_curr', 'phi_goal_drop',
                    'phi_obs_prev', 'phi_obs_curr', 'phi_obs_drop',
                    'phi_agent_prev', 'phi_agent_curr', 'phi_agent_drop',
                    'phi_path_prev', 'phi_path_curr', 'phi_path_drop',
                    'front_obstacle_potential', 'side_obstacle_potential', 'corner_obstacle_potential',
                    'r_potential', 'r_event', 'r_pair', 'r_terminal', 'final_reward',
                    'time_penalty_step', 'spin_without_progress', 'stuck_long',
                    'stuck_long_penalty', 'detour_success_bonus', 'corner_clear_bonus',
                    'no_progress', 'progress_positive_simple', 'local_head_on_pass_event',
                    # Pair credit
                    'pair_event_reward', 'pair_collision_penalty', 'pair_near_miss_penalty',
                    'mutual_yield_penalty', 'yield_pass_credit', 'team_mean_reward',
                ]
                
                for k in keys_to_track:
                    if k in info:
                        value = info[k]
                        if not isinstance(value, (int, float, np.floating, np.integer)):
                            continue
                        if not math.isfinite(float(value)):
                            continue
                        # 按 agent_id 和 字段名 存入列表
                        metric_key = f"{agent_id}/{k}"
                        episode.user_data["custom_metrics_history"].setdefault(metric_key, []).append(value)

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        # 在回合结束时，把累计的数据求平均，并写入 custom_metrics 
        # (RLlib 会自动把 custom_metrics 里的内容画到 TensorBoard 上)
        if "custom_metrics_history" in episode.user_data:
            for metric_key, values in episode.user_data["custom_metrics_history"].items():
                if len(values) > 0:
                    # 计算该回合的平均值
                    episode.custom_metrics[metric_key] = sum(values) / len(values)

def _inject_workspace_paths():
    repo_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        repo_root,
        repo_root / "build" / "gnn_marl_training",
        repo_root / "build" / "gnn_marl_training" / "build" / "lib",
    ]
    ordered_existing_paths = [str(path) for path in candidate_paths if path.exists()]

    # 强制优先级: src > build > build/lib（即使这些路径已存在于 sys.path 中）
    for p in ordered_existing_paths:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(ordered_existing_paths):
        sys.path.insert(0, p)

    # 显式写入 PYTHONPATH，确保 Ray Worker 子进程与主进程一致地优先加载 src
    py_path_entries = ordered_existing_paths + [
        p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    deduped = []
    for entry in py_path_entries:
        if entry not in deduped:
            deduped.append(entry)
    os.environ["PYTHONPATH"] = os.pathsep.join(deduped)

    # ── 修复：将本地工作空间 install/ 下所有包前缀加入 AMENT_PREFIX_PATH ──
    # Ray RolloutWorker 是独立子进程，不会继承 `source install/setup.bash`
    # 必须显式注入，否则 get_package_share_directory() 只能搜索 /opt/ros/humble
    install_dir = repo_root / "install"
    if install_dir.exists():
        ws_prefixes = [
            str(p) for p in install_dir.iterdir()
            if p.is_dir() and not p.name.startswith('_') and p.name != 'COLCON_IGNORE'
        ]
        existing_ament = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
        ament_entries = ws_prefixes + [p for p in existing_ament if p not in ws_prefixes]
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(ament_entries)

    return os.environ["PYTHONPATH"], os.environ.get("AMENT_PREFIX_PATH", "")


WORKSPACE_PYTHONPATH, WORKSPACE_AMENT_PATH = _inject_workspace_paths()

from gnn_marl_training.gnn_marl_env    import env_creator
from gnn_marl_training.gat_rllib_model   import GATRLlibModel,  MODEL_NAME       as MODEL_NAME_GAT
from gnn_marl_training.mappo_mlp_model   import MAPPOMLPModel,  MODEL_NAME_MLP
from gnn_marl_training.counterfactual_ppo_policy import (
    CounterfactualPPOTorchPolicy,
    register_counterfactual_policy,
)
register_counterfactual_policy()


ENV_CURRICULUM = {
    1: {
        "name": "Stage 1 · 静态入门",
        "map_number": 5,          # warehouse_aisles，与 run_curriculum.sh STAGE_MAP_NUM 保持同步
        "max_episode_steps": 2000,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "description": "warehouse_aisles + 无动态障碍，先学会朝 waypoint 平滑前进。",
    },
    2: {
        "name": "Stage 2 · 静态变长",
        "map_number": 3,          # corridor_swap，与 run_curriculum.sh STAGE_MAP_NUM 保持同步
        "max_episode_steps": 2000,
        "num_obstacles": 2,
        "obs_speed_scale": 0.35,
        "description": "corridor_swap 轻量动态障碍预热，开始引入避碰压力。",
    },
    3: {
        "name": "Stage 3 · 慢速动态障碍",
        "map_number": 3,
        "max_episode_steps": 2500,
        "num_obstacles": 6,
        "obs_speed_scale": 0.65,
        "description": "走廊交换地图 + 中高密度动态障碍，会车/汇入冲突更频繁。",
    },
    4: {
        "name": "Stage 4 · 完整任务",
        "map_number": 3,
        "max_episode_steps": 3000,
        "num_obstacles": 8,
        "obs_speed_scale": 1.3,
        "description": "高密度高速动态障碍 + 路径冲突场景，强化最终避碰鲁棒性。",
    },
    5: {
        "name": "Stage 5 (interaction) · Option Selection Basics",
        "map_number": 3,
        "max_episode_steps": 2500,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "description": "Learn feasible option selection from action mask, no dynamic obstacles.",
    },
    6: {
        "name": "Stage 6 (interaction) · Pair Coordination",
        "map_number": 3,
        "max_episode_steps": 3000,
        "num_obstacles": 4,
        "obs_speed_scale": 0.45,
        "description": "Pair event reward + collision attribution with light dynamic obstacles.",
    },
    7: {
        "name": "Stage 7 (interaction) · Full Deployment",
        "map_number": 3,
        "max_episode_steps": 3500,
        "num_obstacles": 8,
        "obs_speed_scale": 1.0,
        "description": "Full difficulty with option outcome rewards, collision attribution.",
    },
}


def _deep_get(mapping, path):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _to_finite_float(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _collect_numeric_by_key_fragments(obj, fragments):
    values = []

    def _walk(node):
        if isinstance(node, dict):
            for key, val in node.items():
                key_s = str(key).lower()
                if isinstance(val, numbers.Real) and not isinstance(val, bool):
                    fv = _to_finite_float(val)
                    if fv is not None and all(frag in key_s for frag in fragments):
                        values.append(fv)
                else:
                    _walk(val)
        elif isinstance(node, (list, tuple)):
            for item in node:
                _walk(item)

    _walk(obj)
    return values


def _extract_reward(result):
    reward_paths = [
        ("episode_reward_mean",),
        ("episode_return_mean",),
        ("env_runners", "episode_reward_mean"),
        ("env_runners", "episode_return_mean"),
        ("sampler_results", "episode_reward_mean"),
        ("sampler_results", "episode_return_mean"),
    ]
    for path in reward_paths:
        reward = _to_finite_float(_deep_get(result, path))
        if reward is not None:
            return reward

    policy_reward_mean = _deep_get(result, ("policy_reward_mean",))
    if isinstance(policy_reward_mean, dict) and policy_reward_mean:
        vals = [_to_finite_float(v) for v in policy_reward_mean.values()]
        vals = [v for v in vals if v is not None]
        if vals:
            return float(sum(vals) / len(vals))
    return None


def _extract_episodes_total(result):
    for path in [
        ("episodes_total",),
        ("env_runners", "episodes_total"),
        ("sampler_results", "episodes_total"),
        ("info", "episodes_total"),
    ]:
        v = _to_finite_float(_deep_get(result, path))
        if v is not None:
            return int(max(0, v))
    return None


def _extract_episodes_this_iter(result):
    for path in [
        ("episodes_this_iter",),
        ("env_runners", "episodes_this_iter"),
        ("sampler_results", "episodes_this_iter"),
        ("info", "episodes_this_iter"),
    ]:
        v = _to_finite_float(_deep_get(result, path))
        if v is not None:
            return int(max(0, v))
    return None


def _extract_timesteps(result, prev_steps=0):
    total_paths = [
        ("timesteps_total",),
        ("num_env_steps_sampled_lifetime",),
        ("num_env_steps_sampled",),
        ("num_env_steps_trained_lifetime",),
        ("num_env_steps_trained",),
        ("agent_timesteps_total",),
        ("env_runners", "num_env_steps_sampled_lifetime"),
        ("env_runners", "num_env_steps_sampled"),
        ("sampler_results", "num_env_steps_sampled"),
        ("info", "num_env_steps_sampled"),
        ("counters", "num_env_steps_sampled"),
        ("counters", "num_env_steps_trained"),
    ]
    total_vals = []
    for path in total_paths:
        value = _to_finite_float(_deep_get(result, path))
        if value is not None:
            total_vals.append(max(0.0, value))
    if total_vals:
        return int(max(prev_steps, max(total_vals)))

    iter_paths = [
        ("timesteps_this_iter",),
        ("num_env_steps_sampled_this_iter",),
        ("env_runners", "num_env_steps_sampled_this_iter"),
        ("sampler_results", "timesteps_this_iter"),
        ("sampler_results", "num_env_steps_sampled_this_iter"),
        ("info", "num_env_steps_sampled_this_iter"),
    ]
    iter_vals = []
    for path in iter_paths:
        value = _to_finite_float(_deep_get(result, path))
        if value is not None:
            iter_vals.append(max(0, int(value)))
    if iter_vals:
        return int(max(prev_steps, prev_steps + max(iter_vals)))

    # 兜底：不同 Ray 版本可能改名，按 key 片段搜索可用步数字段
    for fragments in [
        ("step", "sample"),
        ("env", "step"),
        ("timestep",),
    ]:
        vals = _collect_numeric_by_key_fragments(result, fragments)
        if vals:
            return int(max(prev_steps, max(vals)))

    return int(max(0, prev_steps))


def _extract_timesteps_this_iter(result):
    """提取当前迭代新增步数（而非全生命周期累计步数）。"""
    iter_paths = [
        ("timesteps_this_iter",),
        ("num_env_steps_sampled_this_iter",),
        ("env_runners", "num_env_steps_sampled_this_iter"),
        ("sampler_results", "timesteps_this_iter"),
        ("sampler_results", "num_env_steps_sampled_this_iter"),
        ("info", "num_env_steps_sampled_this_iter"),
    ]
    iter_vals = []
    for path in iter_paths:
        value = _to_finite_float(_deep_get(result, path))
        if value is not None:
            iter_vals.append(max(0, int(value)))
    if iter_vals:
        return int(max(iter_vals))
    return None


def _extract_entropy(result):
    entropy_paths = [
        ("info", "learner", "shared_policy", "entropy"),
        ("info", "learner", "shared_policy", "learner_stats", "entropy"),
        ("info", "learner", "default_policy", "entropy"),
        ("info", "learner", "default_policy", "learner_stats", "entropy"),
        ("learner", "shared_policy", "entropy"),
        ("learner", "shared_policy", "learner_stats", "entropy"),
        ("learner", "default_policy", "entropy"),
        ("learner", "default_policy", "learner_stats", "entropy"),
    ]
    for path in entropy_paths:
        value = _to_finite_float(_deep_get(result, path))
        if value is not None and value >= 0.0:
            return value

    vals = _collect_numeric_by_key_fragments(result, ("entropy",))
    vals = [v for v in vals if v is not None and v >= 0.0]
    return vals[0] if vals else None


def _extract_episode_len_mean(result):
    for path in [
        ("episode_len_mean",),
        ("env_runners", "episode_len_mean"),
        ("sampler_results", "episode_len_mean"),
    ]:
        value = _to_finite_float(_deep_get(result, path))
        if value is not None:
            return value
    return None


def _extract_custom_metrics_dict(result):
    candidates = [
        _deep_get(result, ("custom_metrics",)),
        _deep_get(result, ("env_runners", "custom_metrics")),
        _deep_get(result, ("sampler_results", "custom_metrics")),
        _deep_get(result, ("info", "learner", "shared_policy", "custom_metrics")),
    ]
    for node in candidates:
        if isinstance(node, dict) and node:
            out = {}
            for key, value in node.items():
                fv = _to_finite_float(value)
                if fv is not None:
                    out[str(key)] = fv
            if out:
                return out
    return {}


def _extract_learner_scalars(result):
    paths = [
        ("info", "learner", "shared_policy"),
        ("info", "learner", "default_policy"),
        ("learner", "shared_policy"),
        ("learner", "default_policy"),
    ]
    learner = None
    for path in paths:
        node = _deep_get(result, path)
        if isinstance(node, dict) and node:
            learner = node
            break
    if learner is None:
        return {}

    search_nodes = [learner]
    for subkey in ["learner_stats", "stats"]:
        sub = learner.get(subkey)
        if isinstance(sub, dict) and sub:
            search_nodes.append(sub)

    key_candidates = {
        "policy_loss": ["policy_loss", "pi_loss"],
        "vf_loss": ["vf_loss", "value_loss", "vf_loss_unclipped"],
        "total_loss": ["total_loss", "loss"],
        "kl": ["kl", "kl_loss", "mean_kl_loss", "curr_kl_coeff"],
        "entropy": ["entropy"],
    }

    out = {}
    for out_key, cands in key_candidates.items():
        found = None
        for node in search_nodes:
            for cand in cands:
                value = _to_finite_float(node.get(cand))
                if value is not None:
                    found = value
                    break
            if found is not None:
                break
        if found is None:
            fragments_map = {
                "policy_loss": [("policy", "loss"), ("pi", "loss")],
                "vf_loss": [("vf", "loss"), ("value", "loss")],
                "total_loss": [("total", "loss"), ("loss",)],
                "kl": [("mean", "kl"), ("kl",)],
                "entropy": [("entropy",)],
            }
            for node in search_nodes:
                for fragments in fragments_map.get(out_key, []):
                    vals = _collect_numeric_by_key_fragments(node, fragments)
                    vals = [v for v in vals if v is not None and math.isfinite(v)]
                    if out_key == "entropy":
                        vals = [v for v in vals if v >= 0.0]
                    if vals:
                        found = vals[0]
                        break
                if found is not None:
                    break
        out[out_key] = found
    return out


def _sanitize_tb_tag(key: str) -> str:
    return str(key).replace(" ", "_").replace(":", "_")


def _save_training_plot(history, plot_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if not history:
        return False

    steps = [item.get("timesteps", 0) for item in history]
    rewards = [item.get("episode_reward_mean") for item in history]
    ep_lens = [item.get("episode_len_mean") for item in history]
    ent = [item.get("entropy") for item in history]
    pol = [item.get("policy_loss") for item in history]
    vfl = [item.get("vf_loss") for item in history]
    tll = [item.get("total_loss") for item in history]
    klv = [item.get("kl") for item in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    ax1.plot(steps, rewards, label="episode_reward_mean", linewidth=1.8)
    ax1.set_title("Reward")
    ax1.set_xlabel("Timesteps")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(steps, ep_lens, label="episode_len_mean", linewidth=1.5)
    ax2.plot(steps, ent, label="entropy", linewidth=1.2)
    ax2.set_title("Episode / Entropy")
    ax2.set_xlabel("Timesteps")
    ax2.grid(alpha=0.25)
    ax2.legend()

    ax3.plot(steps, pol, label="policy_loss", linewidth=1.2)
    ax3.plot(steps, vfl, label="vf_loss", linewidth=1.2)
    ax3.plot(steps, tll, label="total_loss", linewidth=1.2)
    ax3.plot(steps, klv, label="kl", linewidth=1.2)
    ax3.set_title("Learner Stats")
    ax3.set_xlabel("Timesteps")
    ax3.grid(alpha=0.25)
    ax3.legend()

    custom_keys = sorted({
        key
        for item in history
        for key in item.get("custom_metrics_mean", {}).keys()
        if key.endswith("_mean")
    })
    preferred_fragments = [
        "progress_reward",
        "heading_reward",
        "min_dist",
        "obstacle_penalty",
        "predictive_social_penalty",
        "predictive_front_penalty",
    ]
    picked_keys = []
    for frag in preferred_fragments:
        for key in custom_keys:
            if frag in key and key not in picked_keys:
                picked_keys.append(key)
                break
    for key in custom_keys:
        if len(picked_keys) >= 4:
            break
        if key not in picked_keys:
            picked_keys.append(key)
    for key in picked_keys[:4]:
        vals = [item.get("custom_metrics_mean", {}).get(key, float("nan")) for item in history]
        ax4.plot(steps, vals, label=key, linewidth=1.2)
    ax4.set_title("Custom Metrics")
    ax4.set_xlabel("Timesteps")
    ax4.grid(alpha=0.25)
    if picked_keys:
        ax4.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return True


def _write_monitor_csv_header(csv_path, fieldnames):
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_monitor_csv(csv_path, fieldnames, row):
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({k: row.get(k) for k in fieldnames})


def _load_monitor_history(jsonl_path):
    history = []
    if not os.path.exists(jsonl_path):
        return history

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                if not isinstance(item.get("custom_metrics_mean"), dict):
                    item["custom_metrics_mean"] = {}
                history.append(item)
    return history


def _load_monitor_offsets(jsonl_path, csv_path):
    history = _load_monitor_history(jsonl_path)
    if history:
        last = history[-1]
        return (
            history,
            int(last.get("iteration", 0) or 0),
            int(last.get("timesteps", 0) or 0),
        )

    if not os.path.exists(csv_path):
        return [], 0, 0

    try:
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return [], 0, 0

    if not rows:
        return [], 0, 0

    last = rows[-1]
    return [], int(float(last.get("iteration", 0) or 0)), int(float(last.get("timesteps", 0) or 0))


def _build_launch_command(stage_cfg, num_agents):
    return (
        "ros2 launch start_rl_environment_tb3 main.launch.py "
        f"map_number:={stage_cfg['map_number']} robot_number:={num_agents} "
        f"num_obstacles:={stage_cfg['num_obstacles']} "
        f"obs_speed_scale:={stage_cfg['obs_speed_scale']:.1f}"
    )


def _run_training(
    config,
    run_name,
    train_steps,
    checkpoint_freq,
    storage_path,
    init_checkpoint=None,
    monitor_print_every=5,
    monitor_plot_every=5,
    early_stop_patience_iters=25,
    early_stop_min_steps=400000,
    early_stop_min_delta=5.0,
    early_stop_abs_drop=60.0,
    early_stop_drop_ratio=0.15,
    early_stop_plateau_window_iters=12,
    early_stop_plateau_min_gain=5.0,
):
    run_dir = os.path.join(storage_path, run_name)
    os.makedirs(run_dir, exist_ok=True)
    monitor_jsonl = os.path.join(run_dir, "training_monitor.jsonl")
    monitor_csv = os.path.join(run_dir, "training_monitor.csv")
    monitor_plot = os.path.join(run_dir, "training_monitor.png")
    tb_dir = os.path.join(run_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)

    csv_fields = [
        "iteration",
        "timesteps",
        "episode_reward_mean",
        "episode_len_mean",
        "entropy",
        "policy_loss",
        "vf_loss",
        "total_loss",
        "kl",
        "num_custom_metrics_mean",
    ]
    resume_monitoring = bool(init_checkpoint)
    monitor_history, iteration_offset, timesteps_offset = ([], 0, 0)
    if resume_monitoring:
        monitor_history, iteration_offset, timesteps_offset = _load_monitor_offsets(
            monitor_jsonl, monitor_csv
        )
        if monitor_history or iteration_offset > 0 or timesteps_offset > 0:
            print(
                "检测到已有训练监控，续写日志: "
                f"iteration_offset={iteration_offset} timesteps_offset={timesteps_offset}"
            )

    if not resume_monitoring:
        open(monitor_jsonl, "w", encoding="utf-8").close()
        _write_monitor_csv_header(monitor_csv, csv_fields)
    else:
        if not os.path.exists(monitor_jsonl):
            open(monitor_jsonl, "a", encoding="utf-8").close()
        if not os.path.exists(monitor_csv) or os.path.getsize(monitor_csv) == 0:
            _write_monitor_csv_header(monitor_csv, csv_fields)

    tb_writer = SummaryWriter(tb_dir) if SummaryWriter is not None else None
    if tb_writer is None:
        print("⚠️ TensorBoard 未启用：torch.utils.tensorboard 不可用")

    algo = config.build()
    if init_checkpoint:
        print(f"尝试从 Checkpoint 恢复: {init_checkpoint}")
        try:
            # 【核心修改】尝试完整恢复（包含优化器状态、步数、KL系数），这是断点续训的正确姿势！
            algo.restore(init_checkpoint)
            print("完整恢复成功（包含优化器历史动量）")
        except Exception as e:
            # 如果环境形状变了（比如跨课程阶段），restore 会失败，此时退化为仅迁移权重
            print(f"完整恢复失败，退化为仅迁移网络权重 (Donor Mode): {e}")
            donor = config.build()
            donor.restore(init_checkpoint)
            algo.set_weights(donor.get_weights())
            donor.stop()

    best_reward = float("-inf")
    best_sig_reward = float("-inf")
    best_ckpt = None
    best_iter = 0
    best_sig_iter = 0
    no_improve_iters = 0
    last_ckpt = None
    final_ckpt = None
    start_time = __import__('time').time()
    iteration = 0
    done_steps = 0
    total_steps_lifetime = 0
    last_valid_reward = None
    episodes_total_prev = 0

    try:
        env_max_steps = int(config.to_dict().get("env_config", {}).get("max_episode_steps", 0))
    except Exception:
        env_max_steps = 0

    while True:
        iteration += 1
        result = algo.train()
        prev_done_steps = done_steps
        prev_total_steps_lifetime = total_steps_lifetime
        total_steps_lifetime = _extract_timesteps(result, prev_steps=total_steps_lifetime)

        # 关键修复：训练进度使用“本次运行新增步数”，避免 restore 后被历史累计步数秒停。
        steps_this_iter = _extract_timesteps_this_iter(result)
        if steps_this_iter is not None:
            done_steps += max(0, int(steps_this_iter))
        else:
            done_steps += max(0, int(total_steps_lifetime - prev_total_steps_lifetime))
        reward = _extract_reward(result)

        episodes_total = _extract_episodes_total(result)
        episodes_this_iter = _extract_episodes_this_iter(result)
        if episodes_this_iter is None and episodes_total is not None:
            episodes_this_iter = max(0, episodes_total - episodes_total_prev)

        if done_steps == prev_done_steps and env_max_steps > 0 and episodes_this_iter and episodes_this_iter > 0:
            done_steps = int(done_steps + episodes_this_iter * env_max_steps)

        if episodes_total is not None:
            episodes_total_prev = max(episodes_total_prev, episodes_total)
        if reward is not None:
            last_valid_reward = reward
        reward_for_print = reward if reward is not None else (
            last_valid_reward if last_valid_reward is not None else 0.0
        )
        entropy = _extract_entropy(result)
        episode_len_mean = _extract_episode_len_mean(result)
        learner_stats = _extract_learner_scalars(result)
        custom_metrics = {
            k: v for k, v in _extract_custom_metrics_dict(result).items()
            if str(k).endswith("_mean")
        }
        log_iteration = iteration_offset + iteration
        log_timesteps = timesteps_offset + done_steps

        monitor_record = {
            "iteration": int(log_iteration),
            "timesteps": int(log_timesteps),
            "episode_reward_mean": float(reward_for_print),
            "episode_len_mean": (float(episode_len_mean) if episode_len_mean is not None else None),
            "entropy": (float(entropy) if entropy is not None else None),
            "policy_loss": learner_stats.get("policy_loss"),
            "vf_loss": learner_stats.get("vf_loss"),
            "total_loss": learner_stats.get("total_loss"),
            "kl": learner_stats.get("kl"),
            "custom_metrics_mean": custom_metrics,
        }
        monitor_history.append(monitor_record)
        with open(monitor_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(monitor_record, ensure_ascii=False) + "\n")
        csv_row = {
            **monitor_record,
            "num_custom_metrics_mean": len(custom_metrics),
        }
        _append_monitor_csv(monitor_csv, csv_fields, csv_row)

        if tb_writer is not None:
            tb_writer.add_scalar("train/episode_reward_mean", float(reward_for_print), int(log_timesteps))
            if episode_len_mean is not None:
                tb_writer.add_scalar("train/episode_len_mean", float(episode_len_mean), int(log_timesteps))
            if entropy is not None:
                tb_writer.add_scalar("train/entropy", float(entropy), int(log_timesteps))
            for key in ["policy_loss", "vf_loss", "total_loss", "kl"]:
                value = learner_stats.get(key)
                if value is not None:
                    tb_writer.add_scalar(f"learner/{key}", float(value), int(log_timesteps))
            for key, value in sorted(custom_metrics.items()):
                tb_writer.add_scalar(f"custom/{_sanitize_tb_tag(key)}", float(value), int(log_timesteps))

        pct = min(100.0, done_steps / max(train_steps, 1) * 100)
        elapsed = __import__('time').time() - start_time
        if done_steps > 0:
            eta_s = elapsed / done_steps * max(train_steps - done_steps, 0)
            eta_str = f"  ETA {int(eta_s//3600):02d}:{int(eta_s%3600//60):02d}:{int(eta_s%60):02d}"
        else:
            eta_str = ""
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(
            f"\r  [{bar}] {pct:5.1f}%  {done_steps:>8,} / {train_steps:,} 步"
            f"  回报 {reward_for_print:7.2f}{eta_str}",
            end="",
            flush=True,
        )

        if done_steps == prev_done_steps and iteration <= 3:
            raw_total = {
                'timesteps_total': result.get('timesteps_total'),
                'num_env_steps_sampled_lifetime': result.get('num_env_steps_sampled_lifetime'),
                'num_env_steps_sampled': result.get('num_env_steps_sampled'),
                'timesteps_this_iter': result.get('timesteps_this_iter'),
                'num_env_steps_sampled_this_iter': result.get('num_env_steps_sampled_this_iter'),
                'lifetime_steps': total_steps_lifetime,
                'episodes_total': episodes_total,
                'episodes_this_iter': episodes_this_iter,
                'env_max_steps': env_max_steps,
            }
            print(f"\n  [debug] timesteps未增长，关键字段: {raw_total}")

        if iteration % max(1, int(monitor_print_every)) == 0:
            print("")
            entropy_str = f"{entropy:.4f}" if entropy is not None else "NA"
            ep_len_str = f"{episode_len_mean:.1f}" if episode_len_mean is not None else "NA"
            pol_loss = learner_stats.get("policy_loss")
            vf_loss = learner_stats.get("vf_loss")
            total_loss = learner_stats.get("total_loss")
            kl = learner_stats.get("kl")
            pol_loss_str = f"{pol_loss:.4f}" if pol_loss is not None else "NA"
            vf_loss_str = f"{vf_loss:.4f}" if vf_loss is not None else "NA"
            total_loss_str = f"{total_loss:.4f}" if total_loss is not None else "NA"
            kl_str = f"{kl:.5f}" if kl is not None else "NA"
            print(
                f"  [monitor] iter={log_iteration} steps={log_timesteps} "
                f"reward={reward_for_print:.3f} ep_len={ep_len_str} entropy={entropy_str}"
            )
            print(
                f"  [monitor] policy_loss={pol_loss_str} vf_loss={vf_loss_str} "
                f"total_loss={total_loss_str} kl={kl_str}"
            )
            if custom_metrics:
                top_items = sorted(custom_metrics.items())[:6]
                custom_line = " | ".join([f"{k}={v:.4f}" for k, v in top_items])
                print(f"  [monitor] custom_metrics -> {custom_line}")

        if iteration % max(1, int(monitor_plot_every)) == 0:
            _save_training_plot(monitor_history, monitor_plot)
            if tb_writer is not None:
                tb_writer.flush()

        if reward is not None and reward > best_reward:
            best_reward = reward
            best_ckpt = algo.save(os.path.join(run_dir, "best"))
            best_iter = iteration
        if reward is not None:
            min_delta = float(max(0.0, early_stop_min_delta))
            if (not math.isfinite(best_sig_reward)) or reward > (best_sig_reward + min_delta):
                best_sig_reward = reward
                best_sig_iter = iteration
                no_improve_iters = 0
            else:
                no_improve_iters += 1

        if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
            last_ckpt = algo.save(run_dir)

        if (
            reward is not None
            and math.isfinite(best_reward)
            and int(max(0, early_stop_patience_iters)) > 0
            and done_steps >= int(max(0, early_stop_min_steps))
        ):
            reward_drop = float(best_reward - reward)
            drop_threshold = max(
                float(max(0.0, early_stop_abs_drop)),
                abs(float(best_reward)) * float(max(0.0, early_stop_drop_ratio)),
            )
            if no_improve_iters >= int(early_stop_patience_iters) and reward_drop >= drop_threshold:
                print("")
                print(
                    "  [early-stop] 触发提前停止: "
                    f"iter={iteration} steps={done_steps} "
                    f"best_iter={best_iter} best_reward={best_reward:.3f} "
                    f"current_reward={reward:.3f} drop={reward_drop:.3f} "
                    f"patience={no_improve_iters}"
                )
                break

        if (
            int(max(0, early_stop_patience_iters)) > 0
            and done_steps >= int(max(0, early_stop_min_steps))
            and len(monitor_history) >= 2 * int(max(1, early_stop_plateau_window_iters))
            and no_improve_iters >= int(early_stop_patience_iters)
        ):
            window = int(max(1, early_stop_plateau_window_iters))
            recent_rewards = [
                float(r["episode_reward_mean"])
                for r in monitor_history[-window:]
                if r.get("episode_reward_mean") is not None
            ]
            prev_rewards = [
                float(r["episode_reward_mean"])
                for r in monitor_history[-2 * window:-window]
                if r.get("episode_reward_mean") is not None
            ]
            if len(recent_rewards) == window and len(prev_rewards) == window:
                recent_gain = float(sum(recent_rewards) / window - sum(prev_rewards) / window)
                if recent_gain <= float(early_stop_plateau_min_gain):
                    print("")
                    print(
                        "  [early-stop] 触发平台期提前停止: "
                        f"iter={iteration} steps={done_steps} "
                        f"best_iter={best_iter} best_reward={best_reward:.3f} "
                        f"best_sig_iter={best_sig_iter} recent_gain={recent_gain:.3f} "
                        f"window={window} patience={no_improve_iters}"
                    )
                    break

        if done_steps >= train_steps:
            break

    print()
    _save_training_plot(monitor_history, monitor_plot)
    final_ckpt = algo.save(os.path.join(run_dir, "final"))
    algo.stop()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()

    ckpt = best_ckpt or last_ckpt or final_ckpt
    summary_reward = (
        best_reward if math.isfinite(best_reward)
        else (last_valid_reward if last_valid_reward is not None else 0.0)
    )
    print(f"  平均回报: {summary_reward:.2f}")
    print(f"  训练步数: {train_steps:,}")
    print(f"  最佳 Checkpoint: {ckpt}")
    print(f"  监控日志(JSONL): {monitor_jsonl}")
    print(f"  监控日志(CSV):   {monitor_csv}")
    print(f"  监控图像(PNG):   {monitor_plot}")
    print(f"  TensorBoard:     {tb_dir}")
    return ckpt


def _resolve_ppo_training_kwargs(args):
    """返回 interaction_mode 专用 PPO 超参数。"""
    profile = str(getattr(args, "ppo_profile", "auto")).strip().lower()

    if profile not in ("auto", "interaction"):
        raise ValueError(f"Unsupported --ppo_profile: {profile}")

    cfg = {
        "clip_param": 0.15,
        "entropy_coeff": 0.008,
        "vf_clip_param": 20.0,
        "grad_clip": 0.7,
        "minibatch_size": 1024,
        "num_epochs": 8,
    }
    resolved_profile = "interaction_mode"

    # 显式 CLI 覆盖优先
    overrides = {
        "clip_param": getattr(args, "clip_param", None),
        "entropy_coeff": getattr(args, "entropy_coeff", None),
        "vf_clip_param": getattr(args, "vf_clip_param", None),
        "grad_clip": getattr(args, "grad_clip", None),
        "minibatch_size": getattr(args, "minibatch_size", None),
        "num_epochs": getattr(args, "num_epochs", None),
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    cfg["minibatch_size"] = int(max(64, int(cfg["minibatch_size"])))
    cfg["num_epochs"] = int(max(1, int(cfg["num_epochs"])))
    if cfg["minibatch_size"] > int(args.train_batch_size):
        print(
            f"⚠️ minibatch_size={cfg['minibatch_size']} > train_batch_size={args.train_batch_size}，"
            f"自动裁剪为 {args.train_batch_size}"
        )
        cfg["minibatch_size"] = int(args.train_batch_size)

    return cfg, resolved_profile, True


def _load_optional_json_dict(raw_value, *, arg_name):
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{arg_name} must decode to a JSON object")
    return value


def _load_optional_json_dict_file(path_value, *, arg_name):
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser().resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{arg_name} could not read file {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{arg_name} file must contain valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{arg_name} file must contain a JSON object")
    return value


def _resolve_reward_override_dict(args, inline_attr, file_attr, *, arg_name):
    inline_value = _load_optional_json_dict(getattr(args, inline_attr, None), arg_name=arg_name)
    file_value = _load_optional_json_dict_file(getattr(args, file_attr, None), arg_name=f"{arg_name}_file")
    merged = {}
    if file_value:
        merged.update(file_value)
    if inline_value:
        merged.update(inline_value)
    return merged or None


def _interaction_reward_profile_defaults(profile_name: str):
    profile = str(profile_name or "potential").strip().lower()
    reward_overrides = {}
    potential_overrides = {}

    if profile == "potential":
        return reward_overrides, potential_overrides
    if profile == "progress_safe":
        potential_overrides.update({
            "goal_drop_weight": 0.45,
            "obs_drop_weight": 0.45,
            "agent_drop_weight": 0.55,
            "path_drop_weight": 0.65,
            "event_reward_scale": 0.80,
            "spin_penalty_scale": 1.40,
            "reverse_penalty_scale": 1.60,
            "stuck_penalty_scale": 1.35,
            "detour_bonus_scale": 0.60,
            "corner_bonus_scale": 0.70,
        })
        return reward_overrides, potential_overrides
    if profile == "anti_reverse":
        potential_overrides.update({
            "goal_drop_weight": 0.35,
            "obs_drop_weight": 0.40,
            "agent_drop_weight": 0.60,
            "path_drop_weight": 0.70,
            "event_reward_scale": 0.75,
            "spin_penalty_scale": 1.60,
            "reverse_penalty_scale": 2.50,
            "stuck_penalty_scale": 1.50,
            "detour_bonus_scale": 0.45,
            "corner_bonus_scale": 0.60,
        })
        return reward_overrides, potential_overrides
    if profile == "no_detour_loop":
        potential_overrides.update({
            "goal_drop_weight": 0.45,
            "obs_drop_weight": 0.45,
            "agent_drop_weight": 0.55,
            "path_drop_weight": 0.75,
            "event_reward_scale": 0.70,
            "spin_penalty_scale": 1.90,
            "reverse_penalty_scale": 2.20,
            "stuck_penalty_scale": 1.70,
            "detour_bonus_scale": 0.0,
            "detour_active_penalty_scale": 1.80,
            "corner_bonus_scale": 0.45,
        })
        return reward_overrides, potential_overrides
    if profile == "event_light":
        potential_overrides.update({
            "goal_drop_weight": 0.55,
            "obs_drop_weight": 0.35,
            "agent_drop_weight": 0.40,
            "path_drop_weight": 0.55,
            "event_reward_scale": 0.45,
            "spin_penalty_scale": 1.25,
            "reverse_penalty_scale": 1.40,
            "stuck_penalty_scale": 1.20,
            "detour_bonus_scale": 0.25,
            "corner_bonus_scale": 0.35,
        })
        return reward_overrides, potential_overrides
    raise ValueError(
        "--interaction_reward_profile must be one of: "
        "potential, progress_safe, anti_reverse, no_detour_loop, event_light"
    )


def _sanitize_run_name_suffix(raw_suffix: str | None) -> str:
    if raw_suffix is None:
        return ""
    suffix = str(raw_suffix).strip()
    if not suffix:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", suffix)


def _resolve_storage_path_and_run_name(base_name: str, args):
    storage_root = os.path.abspath(str(getattr(args, "output_dir", "./ray_results")))
    suffix = _sanitize_run_name_suffix(getattr(args, "run_name_suffix", ""))
    run_name = f"{base_name}_{suffix}" if suffix else base_name
    return storage_root, run_name


def _build_ppo_config(args, env_config, model_name, model_cfg):
    """根据参数构建 PPOConfig。"""
    policy_name = "shared_policy"
    ppo_kwargs, _, _ = _resolve_ppo_training_kwargs(args)

    config = (
        PPOConfig()
        .environment(
            env="gnn_marl",
            env_config=env_config,
            disable_env_checking=True,
        )
        .callbacks(MARLMetricsCallback)
        .framework("torch")
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            sample_timeout_s=max(60, int(args.sample_timeout_s)),
            rollout_fragment_length=max(20, int(args.rollout_fragment_length)),
            batch_mode=args.batch_mode,
        )
        .training(
            lr=args.lr,
            gamma=0.99,
            lambda_=0.95,
            train_batch_size=args.train_batch_size,
            clip_param=float(ppo_kwargs["clip_param"]),
            entropy_coeff=float(ppo_kwargs["entropy_coeff"]),
            vf_clip_param=float(ppo_kwargs["vf_clip_param"]),
            grad_clip=float(ppo_kwargs["grad_clip"]),
            grad_clip_by="global_norm",
            minibatch_size=int(ppo_kwargs["minibatch_size"]),
            num_epochs=int(ppo_kwargs["num_epochs"]),
            model=model_cfg,
        )
        .multi_agent(
            policies={
                policy_name: (
                    CounterfactualPPOTorchPolicy,
                    None,
                    None,
                    {
                        "counterfactual_advantage_coef": float(args.counterfactual_advantage_coef),
                        "counterfactual_credit_clip": float(args.counterfactual_credit_clip),
                    },
                )
            },
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            policies_to_train=[policy_name],
        )
        .resources(
            num_gpus=int(__import__('torch').cuda.is_available())
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )
    return config, policy_name


def _run_direct_env_sanity(env_config, sanity_steps: int = 20) -> dict:
    env = env_creator(env_config)
    completed_steps = 0
    episodes_started = 0
    try:
        observations, _ = env.reset()
        episodes_started += 1
        for _ in range(max(1, int(sanity_steps))):
            action_dict = {}
            for agent_id, obs in observations.items():
                action_space = env.action_space
                if isinstance(action_space, dict):
                    space = action_space[agent_id]
                else:
                    space = action_space
                mask = None
                if isinstance(obs, dict):
                    mask = obs.get("action_mask")
                if mask is not None:
                    mask_arr = np.asarray(mask)
                    valid = np.flatnonzero(mask_arr > 0)
                    if valid.size > 0:
                        action_dict[agent_id] = int(valid[0])
                        continue
                sampled = space.sample()
                if isinstance(sampled, np.ndarray) and sampled.ndim == 0:
                    sampled = sampled.item()
                action_dict[agent_id] = sampled
            observations, rewards, terminateds, truncateds, infos = env.step(action_dict)
            completed_steps += 1
            episode_done = bool(terminateds.get("__all__", False) or truncateds.get("__all__", False))
            if episode_done and completed_steps < int(sanity_steps):
                observations, _ = env.reset()
                episodes_started += 1
        return {
            "completed_steps": completed_steps,
            "episodes_started": episodes_started,
            "active_agents": len(observations),
        }
    finally:
        env.close()


def _run_tune(
    config,
    run_name,
    train_steps,
    checkpoint_freq,
    storage_path,
    init_checkpoint=None,
    monitor_print_every=5,
    monitor_plot_every=5,
    early_stop_patience_iters=25,
    early_stop_min_steps=300000,
    early_stop_min_delta=5.0,
    early_stop_abs_drop=60.0,
    early_stop_drop_ratio=0.15,
    early_stop_plateau_window_iters=12,
    early_stop_plateau_min_gain=5.0,
):
    """兼容旧调用名；内部改为支持权重迁移的手动训练循环。"""
    return _run_training(
        config,
        run_name,
        train_steps,
        checkpoint_freq,
        storage_path,
        init_checkpoint,
        monitor_print_every,
        monitor_plot_every,
        early_stop_patience_iters,
        early_stop_min_steps,
        early_stop_min_delta,
        early_stop_abs_drop,
        early_stop_drop_ratio,
        early_stop_plateau_window_iters,
        early_stop_plateau_min_gain,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MAPPO 训练脚本（逐步构建版）\n"
            "  --model_type mlp : 基础 MAPPO-MLP-LSTM，先跑通再加 GNN\n"
            "  --model_type gat : GNN-MAPPO + 课程学习（需先用 mlp 验证）"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # ── 通用参数 ──────────────────────────────────────────────────────────
    parser.add_argument("--model_type",       type=str,   default="mlp",
                        choices=["mlp", "gat"],
                        help="mlp = 纯MLP+LSTM基础MAPPO（推荐先用这个）\n"
                             "gat = GAT+LSTM的GNN-MAPPO（验证完mlp再用）")
    parser.add_argument("--num_agents",       type=int,   default=3)
    parser.add_argument("--communication_range", type=float, default=3.5,
                        help="通信范围(米)，仅 gat 模式下有意义")
    parser.add_argument("--interaction_neighbor_perception_range", type=float, default=3.5,
                        help="Method3 中层 actor 的最近邻感知半径(米)")
    parser.add_argument("--num_workers",      type=int,   default=2)
    parser.add_argument("--sample_timeout_s", type=int,   default=1800,
                        help="采样超时秒数（环境较慢时需要调大，避免 No samples returned）")
    parser.add_argument("--rollout_fragment_length", type=int, default=200,
                        help="每次采样回传步数（truncate_episodes 模式下建议 100~400）")
    parser.add_argument("--batch_mode",       type=str,   default="truncate_episodes",
                        choices=["truncate_episodes", "complete_episodes"],
                        help="采样拼接模式：truncate_episodes 更稳，complete_episodes 更严格")
    parser.add_argument("--train_steps",      type=int,   default=300000)
    parser.add_argument("--direct_env_sanity_steps", type=int, default=0,
                        help=">0 时跳过 Ray/RLlib 训练，仅直接驱动环境 step 若干步做最小可运行性检查")
    parser.add_argument("--checkpoint_freq",  type=int,   default=20)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--train_batch_size", type=int,   default=2000)
    parser.add_argument("--ppo_profile",      type=str,   default="auto",
                        choices=["auto", "interaction"],
                        help="PPO超参模板；当前仅保留 interaction_mode")
    parser.add_argument("--clip_param",       type=float, default=None,
                        help="可选覆盖 PPO clip_param")
    parser.add_argument("--entropy_coeff",    type=float, default=None,
                        help="可选覆盖 PPO entropy_coeff")
    parser.add_argument("--vf_clip_param",    type=float, default=None,
                        help="可选覆盖 PPO vf_clip_param")
    parser.add_argument("--grad_clip",        type=float, default=None,
                        help="可选覆盖 PPO grad_clip")
    parser.add_argument("--minibatch_size",   type=int,   default=None,
                        help="可选覆盖 PPO minibatch_size")
    parser.add_argument("--num_epochs",       type=int,   default=None,
                        help="可选覆盖 PPO num_epochs")
    parser.add_argument("--monitor_print_every", type=int, default=5,
                        help="每隔多少次迭代打印一次训练摘要")
    parser.add_argument("--monitor_plot_every", type=int, default=5,
                        help="每隔多少次迭代刷新一次训练曲线图")
    parser.add_argument("--early_stop_patience_iters", type=int, default=25,
                        help="连续多少个迭代未明显刷新最佳回报时，允许提前停止")
    parser.add_argument("--early_stop_min_steps", type=int, default=300000,
                        help="至少训练多少步后才启用早停")
    parser.add_argument("--early_stop_min_delta", type=float, default=5.0,
                        help="只有回报超过 best+delta 才算有效刷新")
    parser.add_argument("--early_stop_abs_drop", type=float, default=60.0,
                        help="当前回报相对最佳回报的最小绝对跌幅，满足后才触发早停")
    parser.add_argument("--early_stop_drop_ratio", type=float, default=0.15,
                        help="当前回报相对最佳回报的最小相对跌幅，满足后才触发早停")
    parser.add_argument("--early_stop_plateau_window_iters", type=int, default=12,
                        help="平台期早停的滑动窗口长度（按迭代数）")
    parser.add_argument("--early_stop_plateau_min_gain", type=float, default=5.0,
                        help="最近一个窗口相对前一窗口的最小均值提升，低于该值视为平台期")
    parser.add_argument("--hidden_dim",       type=int,   default=256,
                        help="MLP/LSTM 隐藏层维度（mlp 模式）")
    parser.add_argument("--rolling_lookahead_dist", type=float, default=0.0,
                        help="rolling subgoal 前瞻距离(米)，<=0 表示关闭")
    parser.add_argument("--obstacle_filter_range", type=float, default=2.0,
                        help="局部避障观测半径(米): 仅保留该距离内障碍点")
    parser.add_argument("--obstacle_filter_fov_deg", type=float, default=360.0,
                        help="局部避障观测扇区角度(度): 360 表示全向")
    parser.add_argument("--obstacle_top_k", type=int, default=9,
                        help="每帧保留最近障碍点数量 Top-K（定长编码）")
    parser.add_argument("--angular_bins", type=int, default=64,
                        help="LiDAR 全向角分辨率 bins 数（用于 1D CNN 扫描编码器；≤0 时回退到 obstacle_top_k）")
    parser.add_argument("--disable_obstacle_motion_features", action="store_true",
                        help="关闭基于 LiDAR 扇区历史构建的动态障碍 motion token")
    parser.add_argument("--progress_reward_scale", type=float, default=1.2,
                        help="基础导航进度奖励系数")
    parser.add_argument("--path_progress_reward_scale", type=float, default=0.0,
                        help="沿 A* 路径累计进度的额外奖励系数")
    parser.add_argument("--goal_progress_reward_scale", type=float, default=4.0,
                        help="朝最终目标点径向靠近的额外奖励系数")
    parser.add_argument("--goal_reward", type=float, default=24.0,
                        help="到达目标点时的一次性奖励")
    parser.add_argument("--collision_penalty", type=float, default=20.0,
                        help="碰撞事件的一次性惩罚")
    parser.add_argument("--time_penalty", type=float, default=0.0,
                        help="每步时间惩罚；建议仅小幅启用")
    parser.add_argument("--close_obstacle_penalty_scale", type=float, default=0.30,
                        help="近距离贴障的软惩罚系数")
    parser.add_argument("--close_obstacle_dist", type=float, default=0.55,
                        help="近距离贴障软惩罚的触发距离(米)")
    parser.add_argument("--predictive_horizon_sec", type=float, default=1.2,
                        help="预测式避碰的短时前视窗口(秒)")
    parser.add_argument("--predictive_social_ttc_safe", type=float, default=2.2,
                        help="多机器人预测避碰 TTC 安全阈值(秒)")
    parser.add_argument("--predictive_front_ttc_safe", type=float, default=1.2,
                        help="前向障碍预测 TTC 安全阈值(秒)")
    parser.add_argument("--predictive_min_sep", type=float, default=0.55,
                        help="预测式避碰最小安全间距(米)")
    parser.add_argument("--predictive_social_range", type=float, default=2.5,
                        help="预测式社交避碰感知半径(米)")
    parser.add_argument("--predictive_social_penalty_scale", type=float, default=0.17,
                        help="社交预测风险惩罚系数")
    parser.add_argument("--predictive_front_penalty_scale", type=float, default=0.16,
                        help="前向预测风险惩罚系数")
    parser.add_argument("--social_proximity_risk_scale", type=float, default=0.34,
                        help="近距离会车风险门控系数；越大越早惩罚近距接近")
    parser.add_argument("--gap_feature_enable", type=int, default=1,
                        help="是否在观测中加入最佳通行缝隙特征(0/1)")
    parser.add_argument("--yielding_enable", type=int, default=1,
                        help="是否启用基于优先权的让行/commit 局部引导(0/1)")
    parser.add_argument("--yielding_soft_dist", type=float, default=0.90,
                        help="会车让行软触发距离(米)")
    parser.add_argument("--yielding_stop_dist", type=float, default=0.50,
                        help="会车让行强制减速距离(米)")
    parser.add_argument("--yielding_hard_stop_dist", type=float, default=0.30,
                        help="会车让行近停距离(米)")
    parser.add_argument("--yielding_ttc", type=float, default=2.4,
                        help="会车让行 TTC 触发阈值(秒)")
    parser.add_argument("--yielding_commit_steps", type=int, default=5,
                        help="让行决策保持步数，降低互相反复试探")
    parser.add_argument("--interaction_reward_profile", type=str, default="potential",
                        choices=["potential", "progress_safe", "anti_reverse", "no_detour_loop", "event_light"],
                        help="interaction_mode 奖励预设；手动 JSON override 会覆盖预设")
    parser.add_argument("--replan_fixed_cost", type=float, default=0.03,
                        help="Method3 中选择 replan 的固定惩罚")
    parser.add_argument("--replan_freq_cost", type=float, default=0.012,
                        help="Method3 中短窗口重复 replan 的频率惩罚系数")
    parser.add_argument("--replan_time_cost", type=float, default=0.015,
                        help="Method3 中按规划耗时追加的惩罚系数")
    parser.add_argument("--replan_time_budget_sec", type=float, default=0.08,
                        help="Method3 中规划耗时惩罚的归一化预算(秒)")
    parser.add_argument("--replan_window_steps", type=int, default=80,
                        help="Method3 统计近期 replan 频率的时间窗口(步)")
    parser.add_argument("--method3_reward_window_steps", type=int, default=8,
                        help="Method3 高层 reward 的短窗口 credit 步数")
    parser.add_argument("--subgoal_block_front_dist", type=float, default=0.55,
                        help="触发局部绕行 subgoal 的前向受阻阈值(米)")
    parser.add_argument("--subgoal_deadlock_front_dist", type=float, default=0.60,
                        help="死锁判定使用的前向受阻阈值(米)")
    parser.add_argument("--subgoal_deadlock_steps", type=int, default=6,
                        help="连续死锁多少步后触发强制绕行或重规划")
    parser.add_argument("--subgoal_detour_lateral_gain", type=float, default=1.15,
                        help="局部绕行横向增益，越大越敢绕")
    parser.add_argument("--subgoal_detour_hold_steps", type=int, default=12,
                        help="局部绕行 subgoal 保持步数")
    parser.add_argument("--dynamic_replan_neighbor_dist", type=float, default=1.8,
                        help="deadlock 动态重规划时考虑邻居为临时障碍的范围(米)")
    parser.add_argument("--dynamic_replan_ttc", type=float, default=2.6,
                        help="deadlock 动态重规划时考虑邻居为临时障碍的 TTC 阈值(秒)")
    parser.add_argument("--dynamic_replan_block_radius", type=float, default=0.55,
                        help="deadlock 动态重规划时临时障碍膨胀半径(米)")
    parser.add_argument("--subgoal_progress_reward_scale", type=float, default=1.2,
                        help="绕行/让行阶段朝局部 subgoal 靠近的奖励系数")
    parser.add_argument("--detour_progress_relax", type=float, default=0.30,
                        help="绕行/让行模式下对全局路径进度奖励的保留比例")
    parser.add_argument("--risk_aware_forward_penalty_scale", type=float, default=0.28,
                        help="高风险时继续前冲的惩罚系数")
    parser.add_argument("--safe_turn_reward_scale", type=float, default=0.15,
                        help="高风险时朝最佳 gap 转向的奖励系数")
    parser.add_argument("--head_on_avoidance_reward_scale", type=float, default=0.90,
                        help="迎面冲突时，对减速+横向让开的规则奖励系数")
    parser.add_argument("--risk_gate_soft", type=float, default=0.08,
                        help="risk-gated reward 的软启动阈值")
    parser.add_argument("--risk_gate_hard", type=float, default=0.50,
                        help="risk-gated reward 的满激活阈值")
    parser.add_argument("--avoidance_low_risk_scale", type=float, default=0.45,
                        help="低风险时避碰 shaping 的保底权重")
    parser.add_argument("--navigation_high_risk_scale", type=float, default=0.80,
                        help="高风险时导航 shaping 仍保留的权重")
    parser.add_argument("--time_penalty_risk_relax", type=float, default=0.65,
                        help="高风险时对时间惩罚的放松比例")
    parser.add_argument("--team_reward_lambda", type=float, default=1.0,
                        help="团队奖励混合系数")
    parser.add_argument("--min_active_agents_to_continue", type=int, default=2,
                        help="partial-done 语义下，少于多少活跃机器人就提前结束 episode；设为 0 关闭")
    parser.add_argument("--max_failed_agents_before_cutoff", type=int, default=2,
                        help="partial-done 语义下，失败机器人达到多少个就提前结束 episode；设为 0 关闭")
    parser.add_argument("--output_dir", type=str, default="./ray_results",
                        help="Ray/Tune 输出根目录")
    parser.add_argument("--run_name_suffix", type=str, default="",
                        help="附加到实验目录名末尾的后缀，便于区分不同批次")
    parser.add_argument("--reward_aggregation_overrides_json", type=str, default="",
                        help="JSON object for reward aggregation override weights/clips")
    parser.add_argument("--reward_aggregation_overrides_file", type=str, default="",
                        help="Path to JSON file for reward aggregation overrides")
    parser.add_argument("--interaction_potential_overrides_json", type=str, default="",
                        help="JSON object for interaction potential reward overrides")
    parser.add_argument("--interaction_potential_overrides_file", type=str, default="",
                        help="Path to JSON file for interaction potential reward overrides")
    parser.add_argument("--env_stage",        type=int,   default=1,
                        choices=sorted(ENV_CURRICULUM.keys()),
                        help="环境课程学习阶段；切换阶段后需按提示重新 ros2 launch 环境")
    parser.add_argument("--map_number",      type=int,   default=None,
                        help="可选：覆盖 env_stage 的默认地图编号（用于与外部 launch 强制对齐）")
    parser.add_argument("--num_obstacles",   type=int,   default=None,
                        help="可选：覆盖 env_stage 的默认动态障碍数量")
    parser.add_argument("--obs_speed_scale", type=float, default=None,
                        help="可选：覆盖 env_stage 的默认动态障碍速度缩放")


    # ── GAT 专用：课程学习 ────────────────────────────────────────────────
    parser.add_argument("--curriculum_stage", type=int,   default=1,
                        choices=[1, 2],
                        help="[仅 gat 模式]\n"
                             "  1 = 独立导航阶段 (comm_dropout=1.0)\n"
                             "  2 = 协作通信阶段 (从阶段1 checkpoint 加载)")
    parser.add_argument("--resume_checkpoint", type=str,  default=None,
                        help="[仅 gat 模式阶段2] 阶段1 checkpoint 路径")
    parser.add_argument("--comm_mode",          type=str,   default="decentralized",
                        choices=["decentralized", "centralized_oracle", "ros2_bridge"])
    parser.add_argument("--comm_dropout_prob",  type=float, default=0.05)
    parser.add_argument("--comm_latency_steps", type=int,   default=1)
    parser.add_argument("--comm_jitter_steps",  type=int,   default=1)
    parser.add_argument("--comm_noise_std",     type=float, default=0.05)
    parser.add_argument("--gat_actor_graph", type=str, default="local_risk",
                        choices=["social_risk", "local_risk", "neighbor"],
                        help="GAT actor 的图节点来源：social_risk/local_risk=本机局部风险token，neighbor=最近邻感知token")
    parser.add_argument("--gat_critic_mode", type=str, default="mlp",
                        choices=["mlp", "gat"],
                        help="GAT 分支 critic 结构：mlp 更稳，gat 更复杂")
    parser.add_argument("--gat_risk_bias_scale", type=float, default=2.5,
                        help="GAT 注意力先验偏置强度（距离/风险越高，bias 越大）")
    parser.add_argument("--high_conflict_mode", type=str, default="mixed",
                        choices=["off", "mixed", "aggressive"],
                        help="高冲突路线采样模式：off 关闭，mixed 按概率混合，aggressive 持续使用冲突路线")
    parser.add_argument("--high_conflict_prob", type=float, default=0.85,
                        help="当 high_conflict_mode=mixed 时，启用高冲突路线采样的概率 [0,1]")
    parser.add_argument("--failure_replay_enable", type=int, default=1,
                        help="是否启用失败场景 replay buffer (0/1)")
    parser.add_argument("--failure_replay_buffer_size", type=int, default=96,
                        help="失败场景 replay buffer 容量")
    parser.add_argument("--failure_replay_base_prob", type=float, default=0.22,
                        help="失败场景 replay 的基础触发概率")
    parser.add_argument("--failure_replay_max_prob", type=float, default=0.90,
                        help="失败场景 replay 的最大触发概率")
    parser.add_argument("--failure_replay_success_threshold", type=float, default=0.60,
                        help="场景成功率阈值；超过该阈值后 replay 概率会自动下调")
    parser.add_argument("--corner_curriculum_enable", type=int, default=1,
                        help="是否启用 corner curriculum 训练分布 (0/1)")
    parser.add_argument("--corner_curriculum_prob", type=float, default=0.35,
                        help="每个 episode 采样 corner curriculum 的概率 [0,1]")
    parser.add_argument("--corner_curriculum_set", type=str, default="corner_curriculum_v1",
                        help="corner curriculum 使用的固定路线集合名")
    parser.add_argument("--corner_curriculum_mix_conflict", type=int, default=1,
                        help="corner curriculum 未占满 agent 时是否混入冲突路线 (0/1)")
    parser.add_argument("--counterfactual_advantage_coef", type=float, default=0.15,
                        help="counterfactual advantage 混入系数，>0 时启用 leave-one-out credit shaping")
    parser.add_argument("--counterfactual_credit_clip", type=float, default=2.5,
                        help="counterfactual advantage 标准化后裁剪阈值")
    args = parser.parse_args()
    try:
        profile_reward_defaults, profile_potential_defaults = _interaction_reward_profile_defaults(
            args.interaction_reward_profile
        )
        reward_aggregation_overrides = _resolve_reward_override_dict(
            args,
            "reward_aggregation_overrides_json",
            "reward_aggregation_overrides_file",
            arg_name="--reward_aggregation_overrides_json",
        )
        interaction_potential_overrides = _resolve_reward_override_dict(
            args,
            "interaction_potential_overrides_json",
            "interaction_potential_overrides_file",
            arg_name="--interaction_potential_overrides_json",
        )
        reward_aggregation_overrides = {
            **profile_reward_defaults,
            **(reward_aggregation_overrides or {}),
        } or None
        interaction_potential_overrides = {
            **profile_potential_defaults,
            **(interaction_potential_overrides or {}),
        } or None
    except ValueError as exc:
        parser.error(str(exc))

    if args.gat_actor_graph == "social_risk":
        print("⚠️ 检测到旧参数 --gat_actor_graph social_risk，自动映射为 local_risk。")
        args.gat_actor_graph = "local_risk"

    stage_cfg = dict(ENV_CURRICULUM[args.env_stage])
    if args.map_number is not None:
        stage_cfg["map_number"] = int(args.map_number)
    if args.num_obstacles is not None:
        stage_cfg["num_obstacles"] = int(args.num_obstacles)
    if args.obs_speed_scale is not None:
        stage_cfg["obs_speed_scale"] = float(args.obs_speed_scale)
    args.action_mode = "interaction_mode"

    if int(stage_cfg.get("map_number", 0)) == 6:
        if args.num_obstacles is None and int(stage_cfg.get("num_obstacles", 0)) < 2:
            stage_cfg["num_obstacles"] = 2
        if args.obs_speed_scale is None and float(stage_cfg.get("obs_speed_scale", 0.0)) <= 0.0:
            stage_cfg["obs_speed_scale"] = 0.35
        print(
            "ℹ️ Method3 + map6(interaction_hub) 自动启用交互专项默认值: "
            f"num_obstacles={stage_cfg['num_obstacles']}, obs_speed_scale={stage_cfg['obs_speed_scale']}"
        )
    launch_cmd = _build_launch_command(stage_cfg, args.num_agents)
    resume_path = None
    if args.resume_checkpoint:
        resume_path = os.path.abspath(os.path.expanduser(args.resume_checkpoint))
        if not os.path.exists(resume_path):
            parser.error(f"--resume_checkpoint 路径不存在: {resume_path}")

    obs_top_k = max(1, min(int(args.obstacle_top_k), 64))
    predictive_feature_dim = 6
    gap_feature_dim = 3 if bool(int(args.gap_feature_enable)) else 0
    neighbor_prediction_top_k = 2
    neighbor_prediction_feature_dim = 6
    obstacle_motion_top_k = 3
    obstacle_motion_feature_dim = 6
    obstacle_motion_dim = 0 if args.disable_obstacle_motion_features else (
        obstacle_motion_top_k * obstacle_motion_feature_dim
    )
    local_obs_dim = (
        obs_top_k * 4
        + 2
        + 2
        + 7
        + predictive_feature_dim
        + gap_feature_dim
        + neighbor_prediction_top_k * neighbor_prediction_feature_dim
        + obstacle_motion_dim
    )
    if resume_path and local_obs_dim != 47:
        print(
            f"⚠️ 当前 obstacle_top_k={obs_top_k} -> local_obs_dim={local_obs_dim}，"
            "与旧版默认 47 维不同；resume checkpoint 可能维度不匹配。"
        )

    ppo_kwargs, ppo_profile_name, _ = _resolve_ppo_training_kwargs(args)
    mode_tag = "Interact"

    # ── 初始化 Ray ────────────────────────────────────────────────────────
    try:
        subprocess.run(
            ["ray", "stop", "--force"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass

    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        num_cpus=max(1, int(args.num_workers) + 1),
        local_mode=bool(int(args.num_workers) == 0),
        object_store_memory=256 * 1024 * 1024,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": WORKSPACE_PYTHONPATH,
                "AMENT_PREFIX_PATH": WORKSPACE_AMENT_PATH,
            }
        },
        _system_config={
            "metrics_report_interval_ms": 999999999,  # 禁用定时 metrics 上报
        },
    )
    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)

    # ════════════════════════════════════════════════════════════════════
    #  MLP 模式：基础 MAPPO-LSTM，单阶段训练，无通信
    # ════════════════════════════════════════════════════════════════════
    if args.model_type == "mlp":
        print(f"{'='*80}")
        print(f"🧱 MAPPO-MLP-LSTM 基础训练（无 GNN，带 LSTM）")
        print(f"{'='*80}")
        print(f"  机器人数量:  {args.num_agents}")
        print(f"  环境阶段:    {args.env_stage} · {stage_cfg['name']}")
        print(f"  并行Workers: {args.num_workers}")
        print(f"  采样设置:    timeout={args.sample_timeout_s}s fragment={args.rollout_fragment_length} mode={args.batch_mode}")
        print(f"  训练步数:    {args.train_steps:,}")
        print(f"  学习率:      {args.lr}")
        print(f"  动作模式:    {args.action_mode}")
        print(f"  PPO配置:     {ppo_profile_name}"
              f"  (clip={ppo_kwargs['clip_param']}, ent={ppo_kwargs['entropy_coeff']},"
              f" mb={ppo_kwargs['minibatch_size']}, epochs={ppo_kwargs['num_epochs']})")
        print(f"  隐藏维度:    {args.hidden_dim}")
        print(
            f"  Actor 输入:  {local_obs_dim}  "
            f"(TopK扇区({obs_top_k})×4帧 + goal(2)+vel(2)+safety(7)+predictive(6)"
            f"+neighbor_pred({neighbor_prediction_top_k}x{neighbor_prediction_feature_dim})"
            f"+obstacle_motion({obstacle_motion_dim}))"
        )
        print(f"  Critic 输入: {args.num_agents * local_obs_dim}  ({args.num_agents} 个机器人全局状态)")
        print(f"  阶段描述:    {stage_cfg['description']}")
        print(f"  冲突路线:    mode={args.high_conflict_mode} prob={args.high_conflict_prob:.2f}")
        print(
            f"  预测避碰:    horizon={args.predictive_horizon_sec:.2f}s"
            f" social_ttc={args.predictive_social_ttc_safe:.2f}s"
            f" front_ttc={args.predictive_front_ttc_safe:.2f}s"
            f" min_sep={args.predictive_min_sep:.2f}m"
        )
        print(
            f"  控制方式:    纯端到端RL"
            f"（goal-directed，path-tracking reward=OFF）"
        )
        print(f"  需要环境:    {launch_cmd}")
        print(f"{'='*80}\n")

        env_config = {
            "num_agents":          args.num_agents,
            "map_number":          stage_cfg['map_number'],
            "max_episode_steps":   stage_cfg['max_episode_steps'],
            "communication_range": args.communication_range,
            "interaction_neighbor_perception_range": args.interaction_neighbor_perception_range,
            # MLP 模式：关闭邻居观测，obs = [local(B)] + [reset_flag(1)] + [global(N*B)]
            "enable_neighbor_obs": True,
            "enable_local_map":    False,
            "comm_mode":           "decentralized",
            "comm_dropout_prob":   1.0,   # 无意义，但保持接口一致
            "comm_latency_steps":  1,
            "comm_jitter_steps":   0,
            "comm_noise_std":      0.0,
            # 纯端到端 RL：不再追 rolling subgoal
            "reset_on_collision_event": True,
            "collision_hard_dist": 0.22,
            "collision_persist_dist": 0.24,
            "collision_persist_steps": 2,
            "rolling_lookahead_dist": args.rolling_lookahead_dist,
            "action_mode": "interaction_mode",
            "obstacle_filter_range": args.obstacle_filter_range,
            "obstacle_filter_fov_deg": args.obstacle_filter_fov_deg,
            "obstacle_top_k": obs_top_k,
            "angular_bins": int(args.angular_bins),
            "predictive_feature_enable": True,
            "predictive_horizon_sec": args.predictive_horizon_sec,
            "predictive_social_ttc_safe": args.predictive_social_ttc_safe,
            "predictive_front_ttc_safe": args.predictive_front_ttc_safe,
            "predictive_min_sep": args.predictive_min_sep,
            "predictive_social_range": args.predictive_social_range,
            "predictive_social_penalty_scale": args.predictive_social_penalty_scale,
            "predictive_front_penalty_scale": args.predictive_front_penalty_scale,
            "social_proximity_risk_scale": args.social_proximity_risk_scale,
            "gap_feature_enable": bool(int(args.gap_feature_enable)),
            "yielding_enable": bool(int(args.yielding_enable)),
            "yielding_soft_dist": args.yielding_soft_dist,
            "yielding_stop_dist": args.yielding_stop_dist,
            "yielding_hard_stop_dist": args.yielding_hard_stop_dist,
            "yielding_ttc": args.yielding_ttc,
            "yielding_commit_steps": args.yielding_commit_steps,
            "replan_fixed_cost": args.replan_fixed_cost,
            "replan_freq_cost": args.replan_freq_cost,
            "replan_time_cost": args.replan_time_cost,
            "replan_time_budget_sec": args.replan_time_budget_sec,
            "replan_window_steps": args.replan_window_steps,
            "method3_reward_window_steps": args.method3_reward_window_steps,
            "neighbor_prediction_top_k": neighbor_prediction_top_k,
            "obstacle_motion_feature_enable": not args.disable_obstacle_motion_features,
            "obstacle_motion_top_k": obstacle_motion_top_k,
            "obs_target_dist_clip": 6.0,
            "obs_target_filter_alpha": 0.35,
            "obs_target_max_step": 0.45,
            "progress_reward_scale": args.progress_reward_scale,
            "path_progress_reward_scale": args.path_progress_reward_scale,
            "goal_progress_reward_scale": args.goal_progress_reward_scale,
            "goal_reward": args.goal_reward,
            "collision_penalty": args.collision_penalty,
            "time_penalty": args.time_penalty,
            "close_obstacle_penalty_scale": args.close_obstacle_penalty_scale,
            "close_obstacle_dist": args.close_obstacle_dist,
            "subgoal_block_front_dist": args.subgoal_block_front_dist,
            "subgoal_deadlock_front_dist": args.subgoal_deadlock_front_dist,
            "subgoal_deadlock_steps": args.subgoal_deadlock_steps,
            "subgoal_detour_lateral_gain": args.subgoal_detour_lateral_gain,
            "subgoal_detour_hold_steps": args.subgoal_detour_hold_steps,
            "dynamic_replan_neighbor_dist": args.dynamic_replan_neighbor_dist,
            "dynamic_replan_ttc": args.dynamic_replan_ttc,
            "dynamic_replan_block_radius": args.dynamic_replan_block_radius,
            "subgoal_progress_reward_scale": args.subgoal_progress_reward_scale,
            "detour_progress_relax": args.detour_progress_relax,
            "risk_aware_forward_penalty_scale": args.risk_aware_forward_penalty_scale,
            "safe_turn_reward_scale": args.safe_turn_reward_scale,
            "head_on_avoidance_reward_scale": args.head_on_avoidance_reward_scale,
            "risk_gate_soft": args.risk_gate_soft,
            "risk_gate_hard": args.risk_gate_hard,
            "avoidance_low_risk_scale": args.avoidance_low_risk_scale,
            "navigation_high_risk_scale": args.navigation_high_risk_scale,
            "time_penalty_risk_relax": args.time_penalty_risk_relax,
            "team_reward_lambda": args.team_reward_lambda,
            "reward_aggregation_overrides": reward_aggregation_overrides,
            "interaction_potential_overrides": interaction_potential_overrides,
            # 碰撞后单机器人重生，其余继续训练；只有超时结束 episode
            "auto_reset_agents":   True,
            "min_active_agents_to_continue": 0,
            "max_failed_agents_before_cutoff": 0,
            "high_conflict_mode": args.high_conflict_mode,
            "high_conflict_prob": float(np.clip(args.high_conflict_prob, 0.0, 1.0)),
            "failure_replay_enable": bool(int(args.failure_replay_enable)),
            "failure_replay_buffer_size": int(args.failure_replay_buffer_size),
            "failure_replay_base_prob": float(args.failure_replay_base_prob),
            "failure_replay_max_prob": float(args.failure_replay_max_prob),
            "failure_replay_success_threshold": float(args.failure_replay_success_threshold),
            "corner_curriculum_enable": bool(int(args.corner_curriculum_enable)),
            "corner_curriculum_prob": float(np.clip(args.corner_curriculum_prob, 0.0, 1.0)),
            "corner_curriculum_set": str(args.corner_curriculum_set),
            "corner_curriculum_mix_conflict": bool(int(args.corner_curriculum_mix_conflict)),
            "stall_global_replan_enable": True,
            "stall_global_replan_sec": 3.0,
            "subgoal_deadlock_front_dist": 0.52,
            "subgoal_deadlock_steps": 5,
            # 动态障碍物参数（当前由 Gazebo Actor 驱动，保持接口一致）
            "num_dynamic_obstacles": stage_cfg['num_obstacles'],
            "obs_speed":           0.3 * stage_cfg['obs_speed_scale'],
        }
        model_cfg = {
            "custom_model": MODEL_NAME_MLP,
            "custom_model_config": {
                "num_agents": args.num_agents,
                "max_neighbors": min(args.num_agents - 1, 5),
                "neighbor_feature_dim": 5,
                "use_neighbor_obs": True,
                "action_mode": "interaction_mode",
                "hidden_dim": args.hidden_dim,
                "scan_history_len": 4,
                "obstacle_top_k": obs_top_k,
                "angular_bins": int(args.angular_bins),
                "scan_emb_dim": 128,
                "base_safety_feature_dim": 14,
                "predictive_feature_dim": 6,
                "gap_feature_dim": gap_feature_dim,
                "neighbor_prediction_dim": neighbor_prediction_top_k * neighbor_prediction_feature_dim,
                "obstacle_motion_dim": obstacle_motion_dim,
                "interaction_base_ego_dim": 8,
                "interaction_ego_state_dim": 27,
                "option_state_dim": 11,
            },
            # 策略级 LSTM：每条轨迹被切成长度 20 的序列送入训练
            # 20 步 × ~0.1s/步 ≈ 2 秒，足以覆盖动态障碍物一次穿越过程
            "max_seq_len": 32,
        }
        config, _ = _build_ppo_config(args, env_config, MODEL_NAME_MLP, model_cfg)
        storage_path, run_name = _resolve_storage_path_and_run_name(
            f"MAPPO_MLP_LSTM_Stage{args.env_stage}_{mode_tag}",
            args,
        )
        if args.env_stage > 1 and not resume_path:
            print("⚠️ 当前为高阶段课程学习，但未提供 --resume_checkpoint，将从随机初始化开始。\n")
        ckpt = _run_tune(
            config,
            run_name,
            args.train_steps,
            args.checkpoint_freq,
            storage_path,
            init_checkpoint=resume_path,
            monitor_print_every=args.monitor_print_every,
            monitor_plot_every=args.monitor_plot_every,
            early_stop_patience_iters=args.early_stop_patience_iters,
            early_stop_min_steps=args.early_stop_min_steps,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_abs_drop=args.early_stop_abs_drop,
            early_stop_drop_ratio=args.early_stop_drop_ratio,
            early_stop_plateau_window_iters=args.early_stop_plateau_window_iters,
            early_stop_plateau_min_gain=args.early_stop_plateau_min_gain,
        )
        print(f"\n{'='*80}")
        print(f"✅ MLP-LSTM 训练完成！最佳 Checkpoint: {ckpt}")
        if args.env_stage < max(ENV_CURRICULUM):
            next_stage = args.env_stage + 1
            next_cfg = ENV_CURRICULUM[next_stage]
            next_launch = _build_launch_command(next_cfg, args.num_agents)
            ckpt_path = ckpt.path if hasattr(ckpt, "path") else str(ckpt)
            print("   下一阶段请先重启环境：")
            print(f"   {next_launch}")
            print("   然后继续训练：")
            print(f"   python build/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py --model_type mlp \\")
            print(f"     --env_stage {next_stage} --resume_checkpoint {ckpt_path} \\")
            print(f"     --num_agents {args.num_agents} --num_workers {args.num_workers} \\")
            print(f"     --ppo_profile {args.ppo_profile}")
        else:
            print(f"   验证成功后可切换到 GNN 模式：--model_type gat")
        print(f"{'='*80}\n")

    # ════════════════════════════════════════════════════════════════════
    #  GAT 模式：GNN-MAPPO + 课程学习
    # ════════════════════════════════════════════════════════════════════
    else:
        actor_uses_neighbor_graph = (args.gat_actor_graph == "neighbor")
        # 参数校验
        if args.curriculum_stage == 2 and actor_uses_neighbor_graph:
            if not resume_path:
                parser.error("--curriculum_stage 2 需要指定 --resume_checkpoint")

        # 训练期先固定为完美通信：零丢包、零时延、零抖动、零噪声。
        effective_dropout = 0.0
        effective_latency_steps = 0
        effective_jitter_steps = 0
        effective_noise_std = 0.0
        if actor_uses_neighbor_graph:
            stage_name = "最近邻感知图"
        else:
            stage_name = "局部风险注意力 (感知版 token)"
        print(f"{'='*80}")
        print(f"🚀 GNN-MAPPO 课程学习训练  [{stage_name}]")
        print(f"{'='*80}")
        print(f"  环境阶段:     {args.env_stage} · {stage_cfg['name']}")
        print(f"  机器人数量:   {args.num_agents}")
        print(f"  通信范围:     {args.communication_range} m")
        print(f"  感知邻域:     {args.interaction_neighbor_perception_range} m")
        print(f"  并行Workers:  {args.num_workers}")
        print(f"  采样设置:     timeout={args.sample_timeout_s}s fragment={args.rollout_fragment_length} mode={args.batch_mode}")
        print(f"  训练步数:     {args.train_steps:,}")
        print(f"  动作模式:     {args.action_mode}")
        print(f"  PPO配置:      {ppo_profile_name}"
              f"  (clip={ppo_kwargs['clip_param']}, ent={ppo_kwargs['entropy_coeff']},"
              f" mb={ppo_kwargs['minibatch_size']}, epochs={ppo_kwargs['num_epochs']})")
        print(f"  GAT Actor:    {args.gat_actor_graph}  bias={args.gat_risk_bias_scale:.2f}")
        print(f"  GAT Critic:   {args.gat_critic_mode}")
        print(f"  需要环境:     {launch_cmd}")
        print(
            f"  通信设置:     perfect_comm"
            f"  (dropout={effective_dropout:.0%}, latency={effective_latency_steps},"
            f" jitter={effective_jitter_steps}, noise={effective_noise_std:.2f})"
        )
        print(f"  冲突路线:     mode={args.high_conflict_mode} prob={args.high_conflict_prob:.2f}")
        if resume_path:
            print(f"  恢复自:       {resume_path}")
        print(f"{'='*80}\n")

        env_config = {
            "num_agents":          args.num_agents,
            "map_number":          stage_cfg['map_number'],
            "max_episode_steps":   stage_cfg['max_episode_steps'],
            "communication_range": args.communication_range,
            "interaction_neighbor_perception_range": args.interaction_neighbor_perception_range,
            "enable_neighbor_obs": True,   # 保留邻居槽位，保证两阶段 obs 维度一致
            "enable_local_map":    False,
            "comm_mode":           args.comm_mode,
            "comm_dropout_prob":   effective_dropout,
            "comm_latency_steps":  effective_latency_steps,
            "comm_jitter_steps":   effective_jitter_steps,
            "comm_noise_std":      effective_noise_std,
            # 纯端到端 RL：不再追 rolling subgoal
            "reset_on_collision_event": True,
            "collision_hard_dist": 0.22,
            "collision_persist_dist": 0.24,
            "collision_persist_steps": 2,
            "rolling_lookahead_dist": args.rolling_lookahead_dist,
            "action_mode": "interaction_mode",
            "obstacle_filter_range": args.obstacle_filter_range,
            "obstacle_filter_fov_deg": args.obstacle_filter_fov_deg,
            "obstacle_top_k": obs_top_k,
            "angular_bins": int(args.angular_bins),
            "predictive_feature_enable": True,
            "predictive_horizon_sec": args.predictive_horizon_sec,
            "predictive_social_ttc_safe": args.predictive_social_ttc_safe,
            "predictive_front_ttc_safe": args.predictive_front_ttc_safe,
            "predictive_min_sep": args.predictive_min_sep,
            "predictive_social_range": args.predictive_social_range,
            "predictive_social_penalty_scale": args.predictive_social_penalty_scale,
            "predictive_front_penalty_scale": args.predictive_front_penalty_scale,
            "social_proximity_risk_scale": args.social_proximity_risk_scale,
            "gap_feature_enable": bool(int(args.gap_feature_enable)),
            "yielding_enable": bool(int(args.yielding_enable)),
            "yielding_soft_dist": args.yielding_soft_dist,
            "yielding_stop_dist": args.yielding_stop_dist,
            "yielding_hard_stop_dist": args.yielding_hard_stop_dist,
            "yielding_ttc": args.yielding_ttc,
            "yielding_commit_steps": args.yielding_commit_steps,
            "replan_fixed_cost": args.replan_fixed_cost,
            "replan_freq_cost": args.replan_freq_cost,
            "replan_time_cost": args.replan_time_cost,
            "replan_time_budget_sec": args.replan_time_budget_sec,
            "replan_window_steps": args.replan_window_steps,
            "method3_reward_window_steps": args.method3_reward_window_steps,
            "neighbor_prediction_top_k": neighbor_prediction_top_k,
            "obstacle_motion_feature_enable": not args.disable_obstacle_motion_features,
            "obstacle_motion_top_k": obstacle_motion_top_k,
            "obs_target_dist_clip": 6.0,
            "obs_target_filter_alpha": 0.35,
            "obs_target_max_step": 0.45,
            "progress_reward_scale": args.progress_reward_scale,
            "path_progress_reward_scale": args.path_progress_reward_scale,
            "goal_progress_reward_scale": args.goal_progress_reward_scale,
            "goal_reward": args.goal_reward,
            "collision_penalty": args.collision_penalty,
            "time_penalty": args.time_penalty,
            "close_obstacle_penalty_scale": args.close_obstacle_penalty_scale,
            "close_obstacle_dist": args.close_obstacle_dist,
            "subgoal_block_front_dist": args.subgoal_block_front_dist,
            "subgoal_deadlock_front_dist": args.subgoal_deadlock_front_dist,
            "subgoal_deadlock_steps": args.subgoal_deadlock_steps,
            "subgoal_detour_lateral_gain": args.subgoal_detour_lateral_gain,
            "subgoal_detour_hold_steps": args.subgoal_detour_hold_steps,
            "dynamic_replan_neighbor_dist": args.dynamic_replan_neighbor_dist,
            "dynamic_replan_ttc": args.dynamic_replan_ttc,
            "dynamic_replan_block_radius": args.dynamic_replan_block_radius,
            "subgoal_progress_reward_scale": args.subgoal_progress_reward_scale,
            "detour_progress_relax": args.detour_progress_relax,
            "risk_aware_forward_penalty_scale": args.risk_aware_forward_penalty_scale,
            "safe_turn_reward_scale": args.safe_turn_reward_scale,
            "head_on_avoidance_reward_scale": args.head_on_avoidance_reward_scale,
            "risk_gate_soft": args.risk_gate_soft,
            "risk_gate_hard": args.risk_gate_hard,
            "avoidance_low_risk_scale": args.avoidance_low_risk_scale,
            "navigation_high_risk_scale": args.navigation_high_risk_scale,
            "time_penalty_risk_relax": args.time_penalty_risk_relax,
            "team_reward_lambda": args.team_reward_lambda,
            "reward_aggregation_overrides": reward_aggregation_overrides,
            "interaction_potential_overrides": interaction_potential_overrides,
            # 碰撞后单机器人重生，其余继续训练；只有超时结束 episode
            "auto_reset_agents":   True,
            "min_active_agents_to_continue": 0,
            "max_failed_agents_before_cutoff": 0,
            "high_conflict_mode": args.high_conflict_mode,
            "high_conflict_prob": float(np.clip(args.high_conflict_prob, 0.0, 1.0)),
            "failure_replay_enable": bool(int(args.failure_replay_enable)),
            "failure_replay_buffer_size": int(args.failure_replay_buffer_size),
            "failure_replay_base_prob": float(args.failure_replay_base_prob),
            "failure_replay_max_prob": float(args.failure_replay_max_prob),
            "failure_replay_success_threshold": float(args.failure_replay_success_threshold),
            "corner_curriculum_enable": bool(int(args.corner_curriculum_enable)),
            "corner_curriculum_prob": float(np.clip(args.corner_curriculum_prob, 0.0, 1.0)),
            "corner_curriculum_set": str(args.corner_curriculum_set),
            "corner_curriculum_mix_conflict": bool(int(args.corner_curriculum_mix_conflict)),
            "stall_global_replan_enable": True,
            "stall_global_replan_sec": 3.0,
            "subgoal_deadlock_front_dist": 0.52,
            "subgoal_deadlock_steps": 5,
            # 动态障碍物参数（当前由 Gazebo Actor 驱动，保持接口一致）
            "num_dynamic_obstacles": stage_cfg['num_obstacles'],
            "obs_speed":           0.3 * stage_cfg['obs_speed_scale'],
        }
        if int(args.direct_env_sanity_steps) > 0:
            summary = _run_direct_env_sanity(env_config, sanity_steps=args.direct_env_sanity_steps)
            print("\n" + "=" * 80)
            print("✅ Direct env sanity completed")
            print(f"  completed_steps: {summary['completed_steps']}")
            print(f"  episodes_started: {summary['episodes_started']}")
            print(f"  active_agents: {summary['active_agents']}")
            print("=" * 80 + "\n")
            return
        model_cfg = {
            "custom_model": MODEL_NAME_GAT,
            "custom_model_config": {
                "num_agents":    args.num_agents,
                "max_neighbors": min(args.num_agents - 1, 5),
                "neighbor_feature_dim": 5,
                "action_mode": "interaction_mode",
                "hidden_dim":    128,
                "gat_hidden_dim": 128,
                "lstm_hidden_dim": 256,
                "n_gat_heads":   4,
                "actor_graph_mode": args.gat_actor_graph,
                "critic_mode": args.gat_critic_mode,
                "risk_bias_scale": args.gat_risk_bias_scale,
                "scan_history_len": 4,
                "base_safety_feature_dim": 14,
                "predictive_feature_dim": 6,
                "gap_feature_dim": gap_feature_dim,
                "neighbor_prediction_dim": neighbor_prediction_top_k * neighbor_prediction_feature_dim,
                "neighbor_prediction_feature_dim": neighbor_prediction_feature_dim,
                "obstacle_motion_dim": obstacle_motion_dim,
                "obstacle_motion_feature_dim": obstacle_motion_feature_dim,
                "obstacle_top_k": obs_top_k,
                "angular_bins": int(args.angular_bins),
                "scan_emb_dim": 128,
                "interaction_base_ego_dim": 8,
                "interaction_ego_state_dim": 27,
                "option_state_dim": 11,
            },
            "max_seq_len": 32,
        }
        config, _ = _build_ppo_config(args, env_config, MODEL_NAME_GAT, model_cfg)
        storage_path, run_name = _resolve_storage_path_and_run_name(
            f"GNN_MAPPO_Stage{args.curriculum_stage}_{mode_tag}",
            args,
        )

        if args.curriculum_stage == 1 and actor_uses_neighbor_graph:
            print("📚 阶段1：保持完美通信，直接学习通信版 social-risk 图...\n")
            ckpt = _run_tune(
                config,
                f"{run_name}_EnvStage{args.env_stage}",
                args.train_steps,
                args.checkpoint_freq,
                storage_path,
                # Stage 1 neighbor-graph fine-tuning must also honor resume_checkpoint.
                init_checkpoint=resume_path,
                monitor_print_every=args.monitor_print_every,
                monitor_plot_every=args.monitor_plot_every,
                early_stop_patience_iters=args.early_stop_patience_iters,
                early_stop_min_steps=args.early_stop_min_steps,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_abs_drop=args.early_stop_abs_drop,
                early_stop_drop_ratio=args.early_stop_drop_ratio,
                early_stop_plateau_window_iters=args.early_stop_plateau_window_iters,
                early_stop_plateau_min_gain=args.early_stop_plateau_min_gain,
            )
            ckpt_path = ckpt.path if hasattr(ckpt, "path") else str(ckpt)
            print(f"\n{'='*80}")
            print(f"✅ 阶段1 完成！下一步：")
            if args.env_stage < max(ENV_CURRICULUM):
                next_stage = args.env_stage + 1
                next_cfg = ENV_CURRICULUM[next_stage]
                next_launch = _build_launch_command(next_cfg, args.num_agents)
                print(f"   先重启环境：{next_launch}")
            print(f"   python train_gnn_mappo_full.py --model_type gat \\")
            print(f"     --env_stage {min(args.env_stage + 1, max(ENV_CURRICULUM))} --curriculum_stage 2 \\")
            print(f"     --resume_checkpoint {ckpt_path} \\")
            print(f"     --num_agents {args.num_agents} --num_workers {args.num_workers} \\")
            print(f"     --ppo_profile {args.ppo_profile}")
            print(f"{'='*80}\n")
        else:
            if actor_uses_neighbor_graph:
                print(f"🤝 阶段2：从阶段1 checkpoint 恢复，开放通信...\n")
            else:
                print(f"🎯 局部风险图注意力：不依赖通信，直接训练 actor 的风险关注模式...\n")
            ckpt = _run_tune(
                config,
                f"{run_name}_EnvStage{args.env_stage}",
                args.train_steps,
                args.checkpoint_freq,
                storage_path,
                init_checkpoint=resume_path,
                monitor_print_every=args.monitor_print_every,
                monitor_plot_every=args.monitor_plot_every,
                early_stop_patience_iters=args.early_stop_patience_iters,
                early_stop_min_steps=args.early_stop_min_steps,
                early_stop_min_delta=args.early_stop_min_delta,
                early_stop_abs_drop=args.early_stop_abs_drop,
                early_stop_drop_ratio=args.early_stop_drop_ratio,
                early_stop_plateau_window_iters=args.early_stop_plateau_window_iters,
                early_stop_plateau_min_gain=args.early_stop_plateau_min_gain,
            )
            print(f"\n{'='*80}")
            if actor_uses_neighbor_graph:
                print(f"✅ 阶段2 完成！最佳 Checkpoint: {ckpt}")
            else:
                print(f"✅ 局部风险图注意力训练完成！最佳 Checkpoint: {ckpt}")
            print(f"{'='*80}\n")

    ray.shutdown()


if __name__ == "__main__":
    main()
