#!/usr/bin/env python3
"""
Evaluation entry point for MAPPO/GAT checkpoints.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=UserWarning, module="ray")

# Allow loading checkpoints pickled with an older `packaging` (e.g. seed2/seed3
# p0_core_v1 models trained on another machine). Must run before any pickle.load.
from gnn_marl_training import _packaging_pickle_compat  # noqa: F401

import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
import argparse
import csv
import gzip
import os
import json
import numpy as np
import random
import time
import torch

from gnn_marl_training.gnn_marl_env import env_creator, GNNMARLEnv
from gnn_marl_training.gat_rllib_model import GATRLlibModel, MODEL_NAME as MODEL_NAME_GAT
from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel, MODEL_NAME_MLP
from gnn_marl_training.counterfactual_ppo_policy import (
    CounterfactualPPOTorchPolicy,  # noqa: F401
    register_counterfactual_policy,
)
from gnn_marl_training.paper_metrics import (
    EpisodeMetricTracker,
    MetricThresholds,
    initial_state_description,
    polyline_length,
)
from gnn_marl_training.formal_scenario_bank import get_scenario
register_counterfactual_policy()


def _decode_action_for_diag(agent_env, action):
    try:
        return agent_env._decode_action_to_cmd_vel(action)
    except Exception:
        if getattr(agent_env, "action_mode", "") == "continuous":
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            a_lin = float(np.clip(arr[0], -1.0, 1.0))
            a_ang = float(np.clip(arr[1], -1.0, 1.0))
            if a_lin >= 0.0:
                linear_vel = a_lin * float(agent_env.max_forward_vel)
            else:
                linear_vel = a_lin * float(agent_env.max_reverse_vel)
            angular_vel = a_ang * float(agent_env.max_angular_vel)
            return float(linear_vel), float(angular_vel)
        action_id = int(np.asarray(action).reshape(-1)[0]) if isinstance(action, np.ndarray) else int(action)
        primitive = agent_env.discrete_action_primitives[int(np.clip(action_id, 0, len(agent_env.discrete_action_primitives) - 1))]
        return float(primitive[0]), float(primitive[1])


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _open_jsonl_append(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if str(path).endswith(".gz"):
        return gzip.open(path, "at", encoding="utf-8", compresslevel=3)
    return open(path, "a", encoding="utf-8")


def _append_csv_record(path, record):
    """Append one scalar episode record while preserving a stable CSV schema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    row = _json_safe(record)
    for key, value in list(row.items()):
        if isinstance(value, (dict, list)):
            row[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    has_header = os.path.exists(path) and os.path.getsize(path) > 0
    fieldnames = list(row)
    if has_header:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            existing_fields = next(csv.reader(handle))
        unexpected = [k for k in row if k not in existing_fields]
        if unexpected:
            # Different maps/episode outcomes can introduce extra scalar columns
            # mid-run (e.g. reset/protocol metadata). Rather than abort the whole
            # eval (losing a full map), widen the schema: rewrite the file with the
            # union of columns so old rows keep their values and new columns are
            # simply blank for earlier episodes.
            with open(path, "r", newline="", encoding="utf-8") as handle:
                old_rows = list(csv.DictReader(handle))
            fieldnames = existing_fields + unexpected
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(old_rows)
        else:
            fieldnames = existing_fields
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def _compute_counterfactual_step(model, observations):
    """Return pre-action V(s), V(s\\i), and credit for each active agent."""
    if not observations or not hasattr(model, "compute_counterfactual_values"):
        return {}
    agent_ids = list(observations)
    obs_batch = np.stack([observations[aid] for aid in agent_ids], axis=0)
    agent_indices = [int(aid.rsplit("_", 1)[-1]) for aid in agent_ids]
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    with torch.no_grad():
        obs_t = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        agent_idx_t = torch.as_tensor(agent_indices, dtype=torch.long, device=device)
        if not hasattr(model, "compute_full_values"):
            raise AttributeError("model does not implement compute_full_values")
        v_full = model.compute_full_values(obs_t).detach().reshape(-1)
        interventions = {}
        for strategy in ('zero_agent', 'mean_agent', 'zero_social', 'zero_obstacle'):
            values = model.compute_intervention_values(
                obs_t, agent_idx_t, strategy).detach().reshape(-1)
            interventions[strategy] = values
        v_loo = interventions['zero_agent']
        credit = v_full - v_loo
    return {
        aid: {
            "v_full": float(v_full[idx].cpu().item()),
            "v_loo": float(v_loo[idx].cpu().item()),
            "counterfactual_credit": float(credit[idx].cpu().item()),
            **{
                f"v_{strategy}": float(values[idx].cpu().item())
                for strategy, values in interventions.items()
            },
            **{
                f"delta_v_{strategy}": float((v_full[idx] - values[idx]).cpu().item())
                for strategy, values in interventions.items()
            },
        }
        for idx, aid in enumerate(agent_ids)
    }


def main():
    parser = argparse.ArgumentParser(description="GNN-MAPPO Auto-Test")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="模型Checkpoint路径")
    parser.add_argument("--num_episodes", type=int, default=5, help="测试回合数")
    parser.add_argument("--seed", type=int, default=0, help="测试场景采样随机种子")
    parser.add_argument("--training_seed", type=int, default=None,
                        help="checkpoint 对应的训练种子，仅用于结果追踪")
    parser.add_argument("--num_agents", type=int, default=None,
                        help="零样本覆盖物理机器人数量；actor 观测保持 checkpoint 兼容")
    parser.add_argument("--explore", action="store_true", default=False, help="开启探索噪声(打破狭窄走廊对称死锁)")
    parser.add_argument("--diag_steps", type=int, default=15, help="打印前N步诊断信息")
    parser.add_argument("--map_number", type=int, default=None, help="覆盖 checkpoint 中的 map_number，使全局规划与当前测试地图一致")
    parser.add_argument("--num_dynamic_obstacles", type=int, default=None, help="覆盖动态障碍物数量")
    parser.add_argument("--num_static_obstacles", type=int, default=None, help="覆盖随机静态障碍物数量")
    parser.add_argument("--obs_speed_scale", type=float, default=None,
                        help="覆盖动态障碍物速度系数，实际 obs_speed=0.22*scale，与 obstacle_mover 一致")
    parser.add_argument("--obs_speed_mps", type=float, default=None,
                        help="直接覆盖动态障碍物绝对速度(m/s)；优先于 --obs_speed_scale")
    parser.add_argument("--max_episode_steps", type=int, default=None,
                        help="覆盖 checkpoint 中的 max_episode_steps；大地图需调大(如 map3/4/5=2000)，map9 可维持 1000")
    parser.add_argument("--max_failed_agents_before_cutoff", type=int, default=None,
                        help="覆盖碰撞 cutoff；评估时设 0 表示不因碰撞提前结束，让未碰撞的车继续跑到目标或超时")
    parser.add_argument("--capture_attention", type=int, default=0,
                        help="1=评估时逐步记录 GNN ego->token 注意力，配合 --attention_out 落盘用于热力图")
    parser.add_argument("--attention_out", type=str, default=None,
                        help="注意力/风险场逐步日志的 .npz 输出路径（未提供时不落盘）")
    parser.add_argument("--viz_stride", type=int, default=1,
                        help="注意力、轨迹和风险记录的采样间隔（步）；大规模评估建议 10~25")
    parser.add_argument("--shield_enable", type=int, default=None, help="可选：覆盖 checkpoint 中的 shield 开关 (0/1)")
    parser.add_argument("--shield_front_slow_dist", type=float, default=None)
    parser.add_argument("--shield_front_stop_dist", type=float, default=None)
    parser.add_argument("--shield_neighbor_slow_dist", type=float, default=None)
    parser.add_argument("--shield_linear_slow", type=float, default=None)
    parser.add_argument("--shield_linear_stop", type=float, default=None)
    parser.add_argument("--shield_turn_bias", type=float, default=None)
    parser.add_argument("--turn_in_place_front_dist", type=float, default=None)
    parser.add_argument("--turn_in_place_angle_thresh", type=float, default=None)
    parser.add_argument("--turn_in_place_w", type=float, default=None)
    parser.add_argument("--auto_reset_agents", type=int, default=None,
                        help="可选：覆盖 checkpoint 中的 auto_reset_agents (0/1)")
    parser.add_argument("--reset_on_collision_event", type=int, choices=(0, 1), default=None,
                        help="是否在碰撞后终止对应机器人；固定场景机器人级失败协议设为1")
    parser.add_argument("--collision_ends_episode", type=int, choices=(0, 1), default=None,
                        help="单机器人碰撞是否终止整局；固定场景机器人级失败协议设为0")
    parser.add_argument("--min_active_agents_to_continue", type=int, default=0,
                        help="测试时剩余 active agent 少于该值才提前结束；默认0表示允许其他机器人继续完成")
    # 2026-06-29: 论文评估用，把每个 episode 的指标写入 JSONL 便于后续聚合
    parser.add_argument("--results_jsonl", type=str, default=None,
                        help="若提供，每个 episode 的统计会以 JSONL 写入此路径")
    parser.add_argument("--results_csv", type=str, default=None,
                        help="若提供，每个 episode 的原始标量统计同步写入 CSV")
    parser.add_argument("--trajectory_jsonl", type=str, default=None,
                        help="可选逐步机器人轨迹记录，支持 .jsonl.gz")
    parser.add_argument("--trajectory_stride", type=int, default=5,
                        help="轨迹采样间隔（控制步）")
    parser.add_argument("--method_label", type=str, default=None,
                        help="若提供，写入 JSONL 的每条记录里附加 method 字段（如 'ours' / 'social_only'）")
    parser.add_argument("--scenario_label", type=str, default=None,
                        help="若提供，写入 JSONL 的每条记录里附加 scenario 字段（如 'circle_swap_4'）")
    parser.add_argument("--experiment_condition", type=str, default=None,
                        help="可选实验条件标识，用于参数/鲁棒性聚合")
    parser.add_argument("--cf_trace_jsonl", type=str, default=None,
                        help="可选逐时刻 CF credit/事件/干预轨迹，支持 .jsonl.gz")
    parser.add_argument("--near_collision_clearance_m", type=float, default=0.45)
    parser.add_argument("--deadlock_speed_mps", type=float, default=0.03)
    parser.add_argument("--deadlock_progress_m", type=float, default=0.005)
    parser.add_argument("--deadlock_window_steps", type=int, default=50)
    parser.add_argument("--oscillation_angular_speed_rps", type=float, default=0.20)
    parser.add_argument("--sensor_noise_std", type=float, default=0.0)
    parser.add_argument("--comm_dropout_prob", type=float, default=None)
    parser.add_argument("--comm_latency_steps", type=int, default=None)
    parser.add_argument("--comm_jitter_steps", type=int, default=None)
    parser.add_argument("--comm_noise_std", type=float, default=None)
    parser.add_argument("--scenario_bank", type=str, default=None)
    parser.add_argument("--scenario_case", type=str, default=None)
    parser.add_argument("--fixed_scenario_file", type=str, default=None,
                        help="复现包中的 mapX_scenarios.json；逐局严格复用路线和朝向")
    parser.add_argument("--fixed_dynamic_state_file", type=str, default=None,
                        help="复现包中的 mapX_dynamic_initial_states.json")
    parser.add_argument("--fixed_scenario_start_index", type=int, default=0,
                        help="从固定场景文件的第几局开始")
    parser.add_argument("--fixed_start_tolerance_m", type=float, default=0.05,
                        help="固定场景重置后允许的 Gazebo 起点落地误差（米）")
    parser.add_argument("--paired_dynamic_obstacles", type=int, choices=(0, 1), default=0)
    parser.add_argument("--controller", choices=["policy", "orca_dwa"], default="policy")
    parser.add_argument("--strict_step_time_control", type=int, choices=(0, 1), default=1,
                        help="1=评估时在策略推理间隙暂停Gazebo，动作只执行一个control_dt")
    parser.add_argument("--override_counterfactual_mask", type=str,
                        choices=["zero_agent", "mean_agent", "zero_social", "zero_obstacle"],
                        default=None,
                        help="Override counterfactual_mask_strategy from checkpoint for credit tracing ablation")
    args = parser.parse_args()
    args.viz_stride = max(1, int(args.viz_stride))
    args.fixed_start_tolerance_m = max(0.0, float(args.fixed_start_tolerance_m))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint_path))
    if not os.path.exists(checkpoint_path):
        print(f" 找不到 checkpoint: {checkpoint_path}")
        return
    
    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)

    # 注册环境与模型
    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)

    print(f"{'='*80}")
    print("正在读取 Checkpoint 配置...")
    print(f"{'='*80}")

    # 直接从 algorithm_state.pkl 读 config dict（避免触发 Algorithm.from_checkpoint
    # 创建 env_runner actor 与 ROS2 daemon 冲突），再用 Policy.from_checkpoint
    # 单独加载 policy 权重。
    import pickle as _pickle
    algo_state_pkl = os.path.join(checkpoint_path, 'algorithm_state.pkl')
    with open(algo_state_pkl, 'rb') as _f:
        _algo_state = _pickle.load(_f)
    trained_config = _algo_state['config']
    env_config = dict(trained_config.get('env_config', {}))
    model_cfg = trained_config.get('model', {})
    custom_model_cfg = dict(model_cfg.get('custom_model_config', {}))

    # Override counterfactual_mask_strategy if requested
    if args.override_counterfactual_mask is not None:
        custom_model_cfg['counterfactual_mask_strategy'] = args.override_counterfactual_mask
        print(f"  Overriding counterfactual_mask_strategy: {args.override_counterfactual_mask}")

    policy_overrides = {}
    try:
        shared_policy_spec = trained_config.get("policies", {}).get("shared_policy")
        if isinstance(shared_policy_spec, (tuple, list)) and len(shared_policy_spec) >= 4:
            policy_overrides = dict(shared_policy_spec[3] or {})
    except Exception:
        policy_overrides = {}

    policy_path = os.path.join(checkpoint_path, 'policies', 'shared_policy')
    # 评估只需策略权重，不需要 optimizer。训练后模型壳体（如 CF reference critic）
    # 的参数分组可能与保存时不同，直接 Policy.from_checkpoint 会在恢复 optimizer
    # 时抛 "parameter group that doesn't match the size of optimizer's group"。
    # 这里手动读取 policy_state，剥离 _optimizer_variables 后用 from_state 重建，
    # RLlib 的 set_state 在该键缺失时会跳过 optimizer 恢复，只加载权重。
    _pol_state_file = os.path.join(policy_path, 'policy_state.pkl')
    with open(_pol_state_file, 'rb') as _pf:
        _pol_state = _pickle.load(_pf)
    _pol_state.pop('_optimizer_variables', None)

    # Apply custom_model_cfg override to policy state
    if args.override_counterfactual_mask is not None:
        if 'config' in _pol_state and 'model' in _pol_state['config']:
            if 'custom_model_config' not in _pol_state['config']['model']:
                _pol_state['config']['model']['custom_model_config'] = {}
            _pol_state['config']['model']['custom_model_config']['counterfactual_mask_strategy'] = args.override_counterfactual_mask

    policy = Policy.from_state(_pol_state)
    latest_gate_stats = {}
    gate_hook_handle = None
    graph_gate = getattr(policy.model, "graph_gate", None)
    if graph_gate is not None and hasattr(graph_gate, "register_forward_hook"):
        def _capture_graph_gate(_module, _inputs, output):
            try:
                values = output.detach().float().cpu().numpy()
                latest_gate_stats["mean"] = float(np.mean(values))
                latest_gate_stats["std"] = float(np.std(values))
            except Exception:
                latest_gate_stats.clear()

        gate_hook_handle = graph_gate.register_forward_hook(_capture_graph_gate)

    trained_num_agents = int(env_config.get("num_agents", 2))
    action_mode = env_config.get("action_mode", "discrete_primitive")
    model_type_used = model_cfg.get("custom_model", "<unknown>")

    print(f" 成功克隆训练架构:")
    print(f"   - 模型结构: {model_type_used}")
    print(f"   - 训练机器人数量: {trained_num_agents}")
    print(f"   - 动作模式: {action_mode}")
    graph_ablation = custom_model_cfg.get("graph_ablation", "<unset>")
    obstacle_token_dim = custom_model_cfg.get("obstacle_motion_dim", "<unset>")
    social_token_dim = custom_model_cfg.get("neighbor_prediction_dim", "<unset>")
    print(f"   - 图消融: {graph_ablation}")
    print(f"   - social token dim: {social_token_dim}")
    print(f"   - obstacle token dim: {obstacle_token_dim}")
    print(f"{'='*80}\n")
    
    # 默认保留 checkpoint 中的大部分环境语义。测试时默认关闭 too_few_active
    # 截断，避免 2 车评估中一个 agent 到达后另一个 agent 被提前停掉。
    env_config["env_log_level"] = "WARNING"
    if args.auto_reset_agents is not None:
        env_config["auto_reset_agents"] = bool(int(args.auto_reset_agents))
    if args.reset_on_collision_event is not None:
        env_config["reset_on_collision_event"] = bool(int(args.reset_on_collision_event))
    if args.collision_ends_episode is not None:
        env_config["collision_ends_episode"] = bool(int(args.collision_ends_episode))
    if args.min_active_agents_to_continue is not None:
        env_config["min_active_agents_to_continue"] = int(args.min_active_agents_to_continue)
    if args.shield_enable is not None:
        env_config["shield_enable"] = bool(int(args.shield_enable))
    if args.shield_front_slow_dist is not None:
        env_config["shield_front_slow_dist"] = float(args.shield_front_slow_dist)
    if args.shield_front_stop_dist is not None:
        env_config["shield_front_stop_dist"] = float(args.shield_front_stop_dist)
    if args.shield_neighbor_slow_dist is not None:
        env_config["shield_neighbor_slow_dist"] = float(args.shield_neighbor_slow_dist)
    if args.shield_linear_slow is not None:
        env_config["shield_linear_slow"] = float(args.shield_linear_slow)
    if args.shield_linear_stop is not None:
        env_config["shield_linear_stop"] = float(args.shield_linear_stop)
    if args.shield_turn_bias is not None:
        env_config["shield_turn_bias"] = float(args.shield_turn_bias)
    if args.turn_in_place_front_dist is not None:
        env_config["turn_in_place_front_dist"] = float(args.turn_in_place_front_dist)
    if args.turn_in_place_angle_thresh is not None:
        env_config["turn_in_place_angle_thresh"] = float(args.turn_in_place_angle_thresh)
    if args.turn_in_place_w is not None:
        env_config["turn_in_place_w"] = float(args.turn_in_place_w)
    if args.num_agents is not None:
        if not 1 <= int(args.num_agents) <= 24:
            parser.error("--num_agents must be in [1, 24]")
        env_config["num_agents"] = int(args.num_agents)

    if args.map_number is not None:
        env_config["map_number"] = int(args.map_number)
    if args.num_dynamic_obstacles is not None:
        env_config["num_dynamic_obstacles"] = int(args.num_dynamic_obstacles)
    if args.num_static_obstacles is not None:
        env_config["num_static_obstacles"] = int(args.num_static_obstacles)
        env_config["random_obstacles"] = True
    if args.obs_speed_mps is not None:
        env_config["obs_speed"] = float(args.obs_speed_mps)
    elif args.obs_speed_scale is not None:
        scale = float(args.obs_speed_scale)
        env_config["obs_speed"] = 0.22 * scale
    if args.max_failed_agents_before_cutoff is not None:
        env_config["max_failed_agents_before_cutoff"] = int(args.max_failed_agents_before_cutoff)
    env_config["strict_step_time_control"] = bool(int(args.strict_step_time_control))
    env_config["sensor_noise_std"] = max(0.0, float(args.sensor_noise_std))
    env_config["paired_dynamic_obstacles"] = bool(args.paired_dynamic_obstacles)
    if args.comm_dropout_prob is not None:
        env_config["comm_dropout_prob"] = min(1.0, max(0.0, float(args.comm_dropout_prob)))
    if args.comm_latency_steps is not None:
        env_config["comm_latency_steps"] = max(0, int(args.comm_latency_steps))
    if args.comm_jitter_steps is not None:
        env_config["comm_jitter_steps"] = max(0, int(args.comm_jitter_steps))
    if args.comm_noise_std is not None:
        env_config["comm_noise_std"] = max(0.0, float(args.comm_noise_std))

    fixed_scenario = None
    fixed_scenario_records = []
    fixed_dynamic_records = []
    fixed_scenario_protocol = None
    if args.fixed_scenario_file:
        if args.scenario_case:
            parser.error("--fixed_scenario_file and --scenario_case are mutually exclusive")
        scenario_path = os.path.abspath(os.path.expanduser(args.fixed_scenario_file))
        with open(scenario_path, 'r', encoding='utf-8') as handle:
            fixed_payload = json.load(handle)
        fixed_scenario_protocol = str(fixed_payload.get('protocol', ''))
        fixed_map = int(fixed_payload.get('map_number', -1))
        fixed_agents = int(fixed_payload.get('num_agents', -1))
        if args.map_number is not None and int(args.map_number) != fixed_map:
            parser.error("--map_number does not match --fixed_scenario_file")
        if args.num_agents is not None and int(args.num_agents) != fixed_agents:
            parser.error("--num_agents does not match --fixed_scenario_file")
        start_index = max(0, int(args.fixed_scenario_start_index))
        all_scenarios = list(fixed_payload.get('scenarios', []))
        stop_index = start_index + int(args.num_episodes)
        if stop_index > len(all_scenarios):
            parser.error(
                f"fixed scenario slice [{start_index}:{stop_index}] exceeds "
                f"{len(all_scenarios)} records"
            )
        fixed_scenario_records = all_scenarios[start_index:stop_index]
        for offset, record in enumerate(fixed_scenario_records):
            expected_index = start_index + offset
            if int(record.get('episode', -1)) != expected_index:
                parser.error(f"fixed scenario episode index mismatch at {expected_index}")
            routes = dict(record.get('route_plan', {}))
            expected_agents = {f'agent_{idx}' for idx in range(fixed_agents)}
            if set(routes) != expected_agents:
                parser.error(f"fixed scenario agents mismatch at episode {expected_index}")
            for aid, route in routes.items():
                if not isinstance(route, dict) or set(('start', 'goal', 'yaw')) - set(route):
                    parser.error(f"incomplete fixed route for {aid} at episode {expected_index}")

        map_config = dict(fixed_payload.get('map_config', {}))
        protocol_obstacles = int(map_config['num_dynamic_obstacles'])
        protocol_steps = int(map_config['max_episode_steps'])
        protocol_speed = float(map_config['obs_speed_mps'])
        if args.num_dynamic_obstacles is not None and int(args.num_dynamic_obstacles) != protocol_obstacles:
            parser.error("--num_dynamic_obstacles does not match the fixed protocol")
        if args.max_episode_steps is not None and int(args.max_episode_steps) != protocol_steps:
            parser.error("--max_episode_steps does not match the fixed protocol")
        requested_speed = args.obs_speed_mps
        if requested_speed is None and args.obs_speed_scale is not None:
            requested_speed = 0.22 * float(args.obs_speed_scale)
        if requested_speed is not None and abs(float(requested_speed) - protocol_speed) > 1e-9:
            parser.error("obstacle speed does not match the fixed protocol")
        if args.num_static_obstacles is not None and int(args.num_static_obstacles) != 0:
            parser.error("fixed protocol requires --num_static_obstacles 0")
        if protocol_obstacles > 0 and not bool(args.paired_dynamic_obstacles):
            parser.error("fixed dynamic scenes require --paired_dynamic_obstacles 1")
        if protocol_obstacles > 0 and not args.fixed_dynamic_state_file:
            parser.error("fixed dynamic scenes require --fixed_dynamic_state_file")

        if args.fixed_dynamic_state_file:
            dynamic_path = os.path.abspath(os.path.expanduser(args.fixed_dynamic_state_file))
            with open(dynamic_path, 'r', encoding='utf-8') as handle:
                dynamic_payload = json.load(handle)
            if int(dynamic_payload.get('map_number', -1)) != fixed_map:
                parser.error("--fixed_dynamic_state_file map does not match the scenario file")
            by_index = {
                int(record['episode_index']): record
                for record in dynamic_payload.get('episodes', [])
            }
            for offset, scenario in enumerate(fixed_scenario_records):
                episode_index = start_index + offset
                dynamic = by_index.get(episode_index)
                if dynamic is None:
                    parser.error(f"missing fixed dynamic state for episode {episode_index}")
                if int(dynamic['episode_seed']) != int(scenario['seed']):
                    parser.error(f"fixed dynamic seed mismatch at episode {episode_index}")
                fixed_dynamic_records.append(dynamic)

        env_config["num_agents"] = fixed_agents
        env_config["map_number"] = fixed_map
        env_config["max_episode_steps"] = protocol_steps
        env_config["num_dynamic_obstacles"] = protocol_obstacles
        env_config["num_static_obstacles"] = 0
        env_config["random_obstacles"] = True
        env_config["obs_speed"] = protocol_speed
        env_config["sensor_noise_std"] = 0.0
        env_config["high_conflict_mode"] = "off"
    if args.scenario_case:
        bank_name = args.scenario_bank or "paper_micro_v1"
        fixed_scenario = get_scenario(args.scenario_case, bank=bank_name)
        if args.num_agents is not None and int(args.num_agents) != fixed_scenario.num_agents:
            parser.error("--num_agents does not match fixed scenario")
        if args.map_number is not None and int(args.map_number) != fixed_scenario.map_number:
            parser.error("--map_number does not match fixed scenario")
        env_config["num_agents"] = fixed_scenario.num_agents
        env_config["map_number"] = fixed_scenario.map_number
        env_config["max_episode_steps"] = fixed_scenario.max_episode_steps
        env_config["num_dynamic_obstacles"] = fixed_scenario.num_dynamic_obstacles
        env_config["obs_speed"] = fixed_scenario.obstacle_speed_mps
        env_config["sensor_noise_std"] = fixed_scenario.sensor_noise_std
        # seeded_random 场景（如 held-out 的 Map 5）不提供固定 route_plan，
        # reset 时 options 为 None。若沿用 checkpoint 里训练用的 high_conflict_mode
        # (mixed/aggressive)，env 会从 _MAP_COLLISION_ROUTE_LIBRARY 采样强制冲突路线，
        # 而该库中部分 Map 5 路线与仓储走廊几何不兼容，触发
        # "fixed route is invalid for map 5" 直接抛错。评估这类场景应关闭强制冲突
        # 路线注入，改用环境自带的随机安全采样，与 scenario_bank 的 seeded_random 语义一致。
        if fixed_scenario.route_mode != "fixed":
            env_config["high_conflict_mode"] = "off"
    elif not fixed_scenario_records:
        # Cross-map eval without a fixed scenario: disable conflict-route injection
        # from the training config. The training env_config may have
        # high_conflict_mode="mixed"/"aggressive" which causes the env to sample
        # from _MAP_COLLISION_ROUTE_LIBRARY on every reset. Some of those library
        # routes are incompatible with the current map geometry and raise
        # "fixed route is invalid for map N". Forcing "off" uses the env's own
        # random safe-spawn logic, which is correct for generalisation evaluation.
        env_config["high_conflict_mode"] = "off"

    if args.max_episode_steps is not None and not fixed_scenario_records:
        env_config["max_episode_steps"] = int(args.max_episode_steps)

    print("测试环境关键参数:")
    print(f"   - map_number: {env_config.get('map_number')}")
    print(f"   - num_dynamic_obstacles: {env_config.get('num_dynamic_obstacles')}")
    print(f"   - num_static_obstacles: {env_config.get('num_static_obstacles')}")
    print(f"   - obs_speed: {env_config.get('obs_speed')}")
    print(f"   - auto_reset_agents: {env_config.get('auto_reset_agents')}")
    print(f"   - reset_on_collision_event: {env_config.get('reset_on_collision_event')}")
    print(f"   - collision_ends_episode: {env_config.get('collision_ends_episode')}")
    print(f"   - min_active_agents_to_continue: {env_config.get('min_active_agents_to_continue')}")
    print(f"   - max_episode_steps: {env_config.get('max_episode_steps')}")
    print(f"   - max_failed_agents_before_cutoff: {env_config.get('max_failed_agents_before_cutoff')}")
    print(f"   - shield_enable: {env_config.get('shield_enable')}")
    print(f"   - strict_step_time_control: {env_config.get('strict_step_time_control')}")
    print(
        "   - communication: "
        f"dropout={env_config.get('comm_dropout_prob')} "
        f"latency={env_config.get('comm_latency_steps')} "
        f"jitter={env_config.get('comm_jitter_steps')} "
        f"noise={env_config.get('comm_noise_std')}"
    )
    
    print("创建测试环境...")
    env = GNNMARLEnv(env_config)
    num_agents = int(env_config.get("num_agents", trained_num_agents))
    policy_obs_dim = int(policy.observation_space.shape[0])
    actor_prefix_dim = int(
        env.base_obs_dim + env.local_map_dim + env.neighbor_dim + env.reset_flag_dim
    )

    def _policy_compatible_obs(observation):
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if array.size == policy_obs_dim:
            return array
        if actor_prefix_dim > array.size or actor_prefix_dim > policy_obs_dim:
            raise ValueError(
                f"cannot adapt observation: source={array.size}, target={policy_obs_dim}, "
                f"actor_prefix={actor_prefix_dim}"
            )
        adapted = np.zeros(policy_obs_dim, dtype=np.float32)
        adapted[:actor_prefix_dim] = array[:actor_prefix_dim]
        source_global = array[actor_prefix_dim:]
        target_global_size = policy_obs_dim - actor_prefix_dim
        copied = min(source_global.size, target_global_size)
        adapted[actor_prefix_dim:actor_prefix_dim + copied] = source_global[:copied]
        return adapted

    if num_agents != trained_num_agents:
        print(
            f"[zero-shot team-size] trained_N={trained_num_agents} eval_N={num_agents} "
            f"obs={env.observation_space.shape[0]} -> policy_obs={policy_obs_dim}"
        )

    external_controller = None
    if args.controller == "orca_dwa":
        from gnn_bc_tools.expert_orca_dwa import ORCADWATeacher
        external_controller = ORCADWATeacher(
            communication_range=float(env_config.get("communication_range", 5.0)),
            intent_seed=int(args.seed),
        )
        print("[controller] ORCA+DWA system enabled; checkpoint supplies environment protocol only")

    # ── 可视化数据采集：GNN 注意力(F) + 风险场(D) ──────────────────────
    capture_attention = bool(int(args.capture_attention)) and args.attention_out is not None
    attn_records = []   # 每条: dict(episode, step, agent, branch, token_kind, layer2[list], heads[list[list]])
    risk_records = []   # 每条: dict(episode, step, agent, x, y, min_dist, front_min, dyn_ttc_risk, dyn_min_dist, r_ttc)
    if capture_attention:
        try:
            policy.model.enable_attention_capture(True)
            print(f"[viz] 注意力采集已开启 -> {args.attention_out}")
        except AttributeError:
            capture_attention = False
            print("[viz] 当前模型不支持注意力采集(非 GAT 模型)，跳过")

    def _grab_attention(episode_idx, step_idx, aid):
        try:
            cap = policy.model.get_last_attention()
        except Exception:
            return
        for branch, d in cap.get("branches", {}).items():
            l2 = d.get("layer2")
            heads = d.get("layer1_heads") or []
            attn_records.append({
                "episode": int(episode_idx), "step": int(step_idx), "agent": aid,
                "branch": branch, "token_kind": d.get("token_kind", branch),
                "layer2": None if l2 is None else np.asarray(l2, dtype=np.float32).tolist(),
                "heads": [np.asarray(h, dtype=np.float32).tolist() for h in heads if h is not None],
            })

    discrete_primitives = [
        (0.00, 0.00), (0.08, 0.00), (0.14, 0.00), (0.20, 0.00),
        (0.08, 0.55), (0.08, -0.55), (0.00, 1.00), (0.00, -1.00), (-0.06, 0.00)
    ]

    episode_stats = []
    cf_trace_handle = _open_jsonl_append(args.cf_trace_jsonl) if args.cf_trace_jsonl else None
    trajectory_handle = (
        _open_jsonl_append(args.trajectory_jsonl)
        if args.trajectory_jsonl else None
    )
    cf_trace_error_reported = False
    
    try:
        for ep in range(args.num_episodes):
            packaged_scenario = fixed_scenario_records[ep] if fixed_scenario_records else None
            packaged_dynamic = fixed_dynamic_records[ep] if fixed_dynamic_records else None
            episode_output_index = (
                int(packaged_scenario['episode'])
                if packaged_scenario is not None else int(ep)
            )
            episode_seed = (
                int(packaged_scenario['seed'])
                if packaged_scenario is not None
                else int(args.seed) + int(ep)
            )
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            print(f"{'='*80}")
            print(f"Episode {ep + 1}/{args.num_episodes}")
            
            if packaged_scenario is not None:
                reset_options = {
                    "route_plan": packaged_scenario['route_plan'],
                    "strict_fixed_route": True,
                    "_paired_episode_seed": episode_seed,
                    "fixed_dynamic_initial_positions": packaged_dynamic['positions'],
                }
            else:
                reset_options = (
                    {"route_plan": fixed_scenario.route_plan()}
                    if fixed_scenario is not None and fixed_scenario.route_mode == "fixed" else None
                )
            # Gazebo multi-robot spawn is physically non-deterministic: the same
            # target spawn pose occasionally settles outside fixed_start_tolerance_m.
            # Retry the reset a few times so a one-off settle jitter re-settles back
            # within tolerance instead of aborting the whole eval. The strict
            # validation below (lines ~944+) still enforces tolerance on the final
            # attempt, so persistent drift remains fatal. Default 1 == old behavior.
            _spawn_max_retries = max(1, int(os.environ.get("FIXED_SPAWN_MAX_RETRIES", "1") or "1"))
            _spawn_attempt = 0
            while True:
                obs_dict, _ = env.reset(seed=episode_seed, options=reset_options)
                _spawn_attempt += 1
                _spawn_drift = 0.0
                if packaged_scenario is not None:
                    for _aid, _route in packaged_scenario['route_plan'].items():
                        _act = np.asarray(env.robot_positions[_aid], dtype=np.float64)[:2]
                        _exp = np.asarray(_route['start'], dtype=np.float64)
                        _spawn_drift = max(_spawn_drift, float(np.linalg.norm(_act - _exp)))
                if _spawn_drift <= float(args.fixed_start_tolerance_m) or _spawn_attempt >= _spawn_max_retries:
                    break
                print(f"[spawn-retry] episode={ep} max_drift={_spawn_drift:.4f}m > tol={args.fixed_start_tolerance_m}m, "
                      f"re-reset attempt {_spawn_attempt}/{_spawn_max_retries}")
            if external_controller is not None:
                external_controller.reset()
            dones = {"__all__": False}
            agent_ids = [f"agent_{i}" for i in range(num_agents)]
            shortest_path_lengths = {}
            initial_goals = {}
            for aid in agent_ids:
                agent_env = env.agents.get(aid)
                initial_goals[aid] = getattr(agent_env, "goal_pos", None) if agent_env is not None else None
                waypoints = getattr(agent_env, "global_waypoints", []) if agent_env is not None else []
                shortest = polyline_length(waypoints, start=env.robot_positions.get(aid))
                if shortest <= 0.0 and agent_env is not None:
                    start = np.asarray(env.robot_positions.get(aid), dtype=np.float64).reshape(-1)[:2]
                    goal = np.asarray(getattr(agent_env, "goal_pos", start), dtype=np.float64).reshape(-1)[:2]
                    shortest = float(np.linalg.norm(goal - start))
                shortest_path_lengths[aid] = shortest
            packaged_routes = (
                {
                    aid: (route['start'], route['goal'])
                    for aid, route in packaged_scenario['route_plan'].items()
                }
                if packaged_scenario is not None else None
            )
            reference_routes = packaged_routes or (
                fixed_scenario.route_plan()
                if fixed_scenario is not None and fixed_scenario.route_mode == "fixed"
                else None
            )
            initial_start_goal, initial_state_fingerprint = initial_state_description(
                agent_ids,
                env.robot_positions,
                initial_goals,
                reference_routes=reference_routes,
                reference_tolerance_m=max(0.0, float(args.fixed_start_tolerance_m)),
            )
            initial_yaws = {
                aid: float(env.agents[aid].current_pose['yaw'])
                for aid in agent_ids
            }
            if packaged_scenario is not None:
                for aid, route in packaged_scenario['route_plan'].items():
                    actual_start = np.asarray(env.robot_positions[aid], dtype=np.float64)[:2]
                    actual_goal = np.asarray(initial_goals[aid], dtype=np.float64)[:2]
                    expected_start = np.asarray(route['start'], dtype=np.float64)
                    expected_goal = np.asarray(route['goal'], dtype=np.float64)
                    yaw_error = abs(
                        (initial_yaws[aid] - float(route['yaw']) + np.pi) % (2.0 * np.pi) - np.pi
                    )
                    if float(np.linalg.norm(actual_start - expected_start)) > float(
                        args.fixed_start_tolerance_m
                    ):
                        raise RuntimeError(
                            f"fixed scenario start drift: episode={ep} {aid} "
                            f"expected={expected_start.tolist()} actual={actual_start.tolist()}"
                        )
                    if float(np.linalg.norm(actual_goal - expected_goal)) > 1e-6:
                        raise RuntimeError(
                            f"fixed scenario goal mismatch: episode={ep} {aid} "
                            f"expected={expected_goal.tolist()} actual={actual_goal.tolist()}"
                        )
                    if yaw_error > 0.03:
                        raise RuntimeError(
                            f"fixed scenario yaw drift: episode={ep} {aid} "
                            f"expected={route['yaw']} actual={initial_yaws[aid]}"
                        )
            trajectory_last_positions = {}
            trajectory_last_step = -1

            def record_trajectory_sample(sample_step, step_infos=None, force=False):
                nonlocal trajectory_last_step
                if trajectory_handle is None:
                    return
                for aid in agent_ids:
                    position = getattr(env, "robot_positions", {}).get(aid)
                    if position is None:
                        continue
                    pos = np.asarray(position, dtype=np.float64).reshape(-1)
                    if (
                        pos.size >= 2
                        and np.all(np.isfinite(pos[:2]))
                        and np.all(np.abs(pos[:2]) < 20.0)
                    ):
                        trajectory_last_positions[aid] = [float(pos[0]), float(pos[1])]
                stride = max(1, int(args.trajectory_stride))
                if not force and int(sample_step) % stride != 0:
                    return
                if int(sample_step) == trajectory_last_step:
                    return
                info_map = step_infos if isinstance(step_infos, dict) else {}
                samples = []
                clearances = {}
                for aid in agent_ids:
                    xy = trajectory_last_positions.get(aid)
                    if xy is None:
                        continue
                    goal = initial_goals.get(aid)
                    goal_arr = (
                        np.asarray(goal, dtype=np.float64).reshape(-1)
                        if goal is not None else np.asarray([], dtype=np.float64)
                    )
                    agent_env = env.agents.get(aid)
                    pose = getattr(agent_env, "current_pose", {}) if agent_env is not None else {}
                    agent_info = info_map.get(aid, {})
                    if not isinstance(agent_info, dict):
                        agent_info = {}
                    try:
                        clearance = float(agent_info.get("min_dist"))
                    except (TypeError, ValueError):
                        clearance = float("nan")
                    if np.isfinite(clearance):
                        clearances[aid] = clearance
                    samples.append({
                        "agent": aid,
                        "x": xy[0],
                        "y": xy[1],
                        "yaw": float(pose.get("yaw", 0.0)) if isinstance(pose, dict) else 0.0,
                        "goal_x": float(goal_arr[0]) if goal_arr.size >= 2 else None,
                        "goal_y": float(goal_arr[1]) if goal_arr.size >= 2 else None,
                        "done": bool(aid in getattr(env, "dones", set()) or dones.get(aid, False)),
                        "event": str(agent_info.get("event", "")),
                    })
                trajectory_handle.write(json.dumps(_json_safe({
                    "method": args.method_label,
                    "controller": args.controller,
                    "scenario": args.scenario_label,
                    "map_number": int(env_config.get("map_number", -1)),
                    "episode": episode_output_index,
                    "fixed_scenario_source_index": (
                        episode_output_index if packaged_scenario is not None else None
                    ),
                    "episode_seed": int(episode_seed),
                    "scenario_id": (
                        packaged_scenario.get('scenario_id')
                        if packaged_scenario is not None else None
                    ),
                    "step": int(sample_step),
                    "control_dt": float(
                        getattr(env.agents.get("agent_0"), "control_dt", 0.1)
                    ),
                    "initial_state_fingerprint": initial_state_fingerprint,
                    "dynamic_obstacle_trajectory_fingerprint": str(
                        getattr(env, "dynamic_obstacle_trajectory_fingerprint", "") or ""
                    ),
                    "agent_min_lidar_clearance": clearances,
                    "min_lidar_clearance": min(clearances.values()) if clearances else None,
                    "agents": samples,
                }), ensure_ascii=False) + "\n")
                trajectory_last_step = int(sample_step)

            record_trajectory_sample(0, force=True)
            thresholds = MetricThresholds(
                control_dt=float(getattr(env.agents.get("agent_0"), "control_dt", 0.1)),
                near_collision_clearance_m=max(0.0, float(args.near_collision_clearance_m)),
                deadlock_speed_mps=max(0.0, float(args.deadlock_speed_mps)),
                deadlock_progress_m=max(0.0, float(args.deadlock_progress_m)),
                deadlock_window_steps=max(1, int(args.deadlock_window_steps)),
                oscillation_angular_speed_rps=max(0.0, float(args.oscillation_angular_speed_rps)),
            )
            metric_tracker = EpisodeMetricTracker(
                agent_ids,
                env.robot_positions,
                shortest_path_lengths,
                thresholds,
            )
            
            total_rewards = {aid: 0.0 for aid in agent_ids}
            successes     = {aid: 0 for aid in agent_ids}
            collisions    = {aid: 0 for aid in agent_ids}
            step_count = 0
            min_dists = []
            social_risks = []
            front_risks = []
            front_mins = []
            r_ttcs = []
            dynamic_obs_counts = []
            dynamic_obs_min_dists = []
            dynamic_obs_min_ttcs = []
            dynamic_obs_ttc_risks = []
            shield_active_values = []
            motion_filter_values = []
            motion_safety_margins = []
            inference_ms = []
            controller_ms = []
            safety_filter_ms = []
            inference_calls = 0
            agent_pairwise_min_dists = []
            reached_agents = set()
            collided_agents = set()
            truncated_episode = False
            episode_cf_credits = []
            
            # 安全初始化 LSTM 状态
            states = {aid: policy.get_initial_state() for aid in [f"agent_{i}" for i in range(num_agents)]}
            episode_wall_start = time.perf_counter()
            
            while not dones["__all__"] and step_count < env_config.get("max_episode_steps", 1000):
                action_dict = {}
                step_policy_obs = {}
                step_gate_stats = {}
                controller_actions = {}
                if external_controller is not None:
                    controller_start_ns = time.perf_counter_ns()
                    controller_actions = external_controller.compute_actions(env, obs_dict)
                    controller_elapsed_ms = (
                        time.perf_counter_ns() - controller_start_ns
                    ) / 1.0e6
                else:
                    controller_elapsed_ms = None

                for aid in agent_ids:
                    if aid in obs_dict and aid not in env.dones:
                        # 执行推理
                        if external_controller is None:
                            policy_obs = _policy_compatible_obs(obs_dict[aid])
                            inference_start_ns = time.perf_counter_ns()
                            action, state_out, _ = policy.compute_single_action(
                                obs=policy_obs,
                                state=states[aid],
                                explore=args.explore,
                            )
                            elapsed_ms = (time.perf_counter_ns() - inference_start_ns) / 1.0e6
                            step_policy_obs[aid] = policy_obs
                        else:
                            if aid not in controller_actions:
                                continue
                            action = controller_actions[aid]
                            state_out = states[aid]
                            elapsed_ms = controller_elapsed_ms / max(1, len(controller_actions))
                        if inference_calls >= 10:
                            if external_controller is None:
                                inference_ms.append(float(elapsed_ms))
                            else:
                                controller_ms.append(float(elapsed_ms))
                        inference_calls += 1
                        action_dict[aid] = action
                        if latest_gate_stats:
                            step_gate_stats[aid] = dict(latest_gate_stats)
                        # F: 抓取本 agent 本步的 ego->token 注意力（forward 刚在上面执行过）
                        if capture_attention and step_count % args.viz_stride == 0:
                            _grab_attention(ep, step_count, aid)
                        states[aid] = state_out

                        # 诊断输出
                        if args.diag_steps > 0 and step_count < args.diag_steps:
                            agent_env = env.agents[aid]
                            sector_metrics = agent_env._scan_sector_metrics()
                            front_min = float(sector_metrics.get("front_min", agent_env.scan_max_range))
                            left_min = float(sector_metrics.get("left_min", agent_env.scan_max_range))
                            right_min = float(sector_metrics.get("right_min", agent_env.scan_max_range))
                            pred = getattr(agent_env, "_last_predictive_metrics", {}) or {}
                            lin_vel, ang_vel = _decode_action_for_diag(agent_env, action)
                            if action_mode == 'continuous':
                                act_str = f"[{float(action[0]):.3f}, {float(action[1]):.3f}]"
                            else:
                                action_id = int(action)
                                act_str = f"{action_id}"

                            print(
                                f"  [diag] {aid} front={front_min:.2f} left={left_min:.2f} right={right_min:.2f}"
                                f" social_risk={float(pred.get('social_risk', 0.0)):.3f}"
                                f" front_risk={float(pred.get('front_risk', 0.0)):.3f}"
                                f" | raw={act_str} -> v={lin_vel:.3f}, w={ang_vel:.3f}"
                            )
                
                step_cf = {}
                if cf_trace_handle is not None and external_controller is None:
                    try:
                        step_cf = _compute_counterfactual_step(policy.model, step_policy_obs)
                    except Exception as exc:
                        if not cf_trace_error_reported:
                            print(f"[cf-trace] counterfactual credit unavailable: {exc}")
                            cf_trace_error_reported = True

                obs_dict, rewards, dones, truncateds, infos = env.step(action_dict)
                step_count += 1
                trajectory_event = any(
                    (
                        str(info.get("event", "")) == "goal"
                        and aid not in reached_agents
                    )
                    or (
                        str(info.get("event", "")) == "collision"
                        and aid not in collided_agents
                    )
                    for aid, info in (infos.items() if isinstance(infos, dict) else [])
                    if isinstance(info, dict)
                )
                record_trajectory_sample(
                    step_count,
                    infos,
                    force=bool(dones.get("__all__", False) or trajectory_event),
                )

                step_commands = {}
                for aid in action_dict:
                    agent_env = env.agents.get(aid)
                    shield = getattr(agent_env, "_last_shield_info", {}) if agent_env is not None else {}
                    gate = step_gate_stats.get(aid, {})
                    shield_latency = shield.get("safety_filter_latency_ms")
                    if shield_latency is not None and np.isfinite(shield_latency):
                        safety_filter_ms.append(float(shield_latency))
                    step_commands[aid] = {
                        "raw_linear_vel": shield.get("raw_linear_vel"),
                        "raw_angular_vel": shield.get("raw_angular_vel"),
                        "executed_linear_vel": shield.get("shielded_linear_vel"),
                        "executed_angular_vel": shield.get("shielded_angular_vel"),
                        "gate_mean": gate.get("mean"),
                        "gate_std": gate.get("std"),
                    }
                metric_tracker.record_step(
                    positions=getattr(env, "robot_positions", {}),
                    velocities=getattr(env, "robot_velocities", {}),
                    infos=infos if isinstance(infos, dict) else {},
                    commands=step_commands,
                    active_ids=action_dict.keys(),
                )

                if cf_trace_handle is not None:
                    for aid, action in action_dict.items():
                        info = infos.get(aid, {}) if isinstance(infos, dict) else {}
                        agent_env = env.agents.get(aid)
                        shield = getattr(agent_env, "_last_shield_info", {}) if agent_env is not None else {}
                        pos = getattr(env, "robot_positions", {}).get(aid)
                        cf_values = step_cf.get(aid, {})
                        credit = cf_values.get("counterfactual_credit")
                        if credit is not None and np.isfinite(credit):
                            episode_cf_credits.append(float(credit))
                        trace_record = {
                            "episode": int(ep),
                            "episode_seed": int(episode_seed),
                            "step": int(step_count),
                            "agent": aid,
                            "agent_index": int(aid.rsplit("_", 1)[-1]),
                            "map_number": int(env_config.get("map_number", -1)),
                            "method": args.method_label,
                            "experiment_condition": args.experiment_condition,
                            "training_seed": args.training_seed,
                            "eval_seed": int(args.seed),
                            **cf_values,
                            "action": np.asarray(action).reshape(-1).tolist(),
                            "reward": rewards.get(aid),
                            "event": str(info.get("event", "")),
                            "terminated": bool(dones.get(aid, False)),
                            "truncated": bool(truncateds.get(aid, False)),
                            "x": (float(pos[0]) if pos is not None else None),
                            "y": (float(pos[1]) if pos is not None else None),
                            "dist_to_goal": info.get("dist_to_goal"),
                            "min_dist": info.get("min_dist"),
                            "front_min": info.get("front_min"),
                            "dynamic_obs_min_dist": info.get("dynamic_obs_min_dist"),
                            "dynamic_obs_min_ttc": info.get("dynamic_obs_min_ttc"),
                            "dynamic_obs_ttc_risk": info.get("dynamic_obs_ttc_risk"),
                            "r_ttc": info.get("r_ttc"),
                            "subgoal_mode": str(info.get("subgoal_mode", "")),
                            "shield_active": info.get("shield_active", 0.0),
                            "swept_filter_active": info.get("swept_filter_active", 0.0),
                            "motion_filter_active": info.get("motion_filter_active", 0.0),
                            "motion_safety_margin": info.get("motion_safety_margin"),
                            "motion_safety_source": str(info.get("motion_safety_source", "none")),
                            "raw_linear_vel": shield.get("raw_linear_vel"),
                            "raw_angular_vel": shield.get("raw_angular_vel"),
                            "executed_linear_vel": shield.get("shielded_linear_vel"),
                            "executed_angular_vel": shield.get("shielded_angular_vel"),
                            "gate_mean": step_gate_stats.get(aid, {}).get("mean"),
                            "gate_std": step_gate_stats.get(aid, {}).get("std"),
                            "turn_in_place": shield.get("turn_in_place", False),
                            "yield_active": str(info.get("subgoal_mode", "")) == "yield",
                            "yield_partner": getattr(agent_env, "_yield_partner", "") if agent_env is not None else "",
                            "yield_hold_steps": getattr(agent_env, "_yield_hold_steps", 0) if agent_env is not None else 0,
                        }
                        cf_trace_handle.write(json.dumps(_json_safe(trace_record), ensure_ascii=False) + "\n")

                active_positions = []
                done_set = getattr(env, "dones", set())
                for active_aid, active_pos in getattr(env, "robot_positions", {}).items():
                    if active_aid in done_set or active_pos is None:
                        continue
                    pos_arr = np.asarray(active_pos, dtype=np.float32).reshape(-1)
                    if pos_arr.size >= 2 and np.all(np.isfinite(pos_arr[:2])) and np.all(np.abs(pos_arr[:2]) < 20.0):
                        active_positions.append(pos_arr[:2])
                if len(active_positions) >= 2:
                    pairwise = [
                        float(np.linalg.norm(active_positions[i] - active_positions[j]))
                        for i in range(len(active_positions))
                        for j in range(i + 1, len(active_positions))
                    ]
                    if pairwise:
                        agent_pairwise_min_dists.append(min(pairwise))

                # D: 风险场逐步日志——每个 agent 的位置 + 风险指标（后续按网格聚合成热力场）
                # 只记录仍在活动的车：已到达/碰撞出局的车会被环境 park 到远处哨兵位(如 100,100)，
                # 记录它们会污染风险场网格范围，故按 env.dones 跳过。
                if capture_attention and step_count % args.viz_stride == 0:
                    done_set = getattr(env, "dones", set())
                    for aid in [f"agent_{i}" for i in range(num_agents)]:
                        if aid in done_set:
                            continue
                        pos = env.robot_positions.get(aid)
                        if pos is None:
                            continue
                        # 兜底：跳过明显越界的哨兵坐标（真实地图远小于 ±20m）
                        if abs(float(pos[0])) > 20.0 or abs(float(pos[1])) > 20.0:
                            continue
                        inf = infos.get(aid, {}) if isinstance(infos, dict) else {}
                        def _f(k):
                            try:
                                return float(inf[k])
                            except Exception:
                                return float("nan")
                        risk_records.append({
                            "episode": int(ep), "step": int(step_count), "agent": aid,
                            "x": float(pos[0]), "y": float(pos[1]),
                            "min_dist": _f("min_dist"), "front_min": _f("front_min"),
                            "dyn_ttc_risk": _f("dynamic_obs_ttc_risk"),
                            "dyn_min_dist": _f("dynamic_obs_min_dist"), "r_ttc": _f("r_ttc"),
                        })

                for aid, r in rewards.items():
                    total_rewards[aid] += r
                    if aid in infos:
                        event = infos[aid].get('event', '')
                        if 'min_dist' in infos[aid]:
                            try:
                                min_dists.append(float(infos[aid]['min_dist']))
                            except Exception:
                                pass
                        if 'front_min' in infos[aid]:
                            try:
                                front_mins.append(float(infos[aid]['front_min']))
                            except Exception:
                                pass
                        if 'r_ttc' in infos[aid]:
                            try:
                                r_ttcs.append(float(infos[aid]['r_ttc']))
                            except Exception:
                                pass
                        if 'dynamic_obs_count' in infos[aid]:
                            try:
                                dynamic_obs_counts.append(float(infos[aid]['dynamic_obs_count']))
                            except Exception:
                                pass
                        if 'dynamic_obs_min_dist' in infos[aid]:
                            try:
                                dynamic_obs_min_dists.append(float(infos[aid]['dynamic_obs_min_dist']))
                            except Exception:
                                pass
                        if 'dynamic_obs_min_ttc' in infos[aid]:
                            try:
                                dynamic_obs_min_ttcs.append(float(infos[aid]['dynamic_obs_min_ttc']))
                            except Exception:
                                pass
                        if 'dynamic_obs_ttc_risk' in infos[aid]:
                            try:
                                dynamic_obs_ttc_risks.append(float(infos[aid]['dynamic_obs_ttc_risk']))
                            except Exception:
                                pass
                        if 'shield_active' in infos[aid]:
                            try:
                                shield_active_values.append(float(infos[aid]['shield_active']))
                            except Exception:
                                pass
                        if 'motion_filter_active' in infos[aid]:
                            try:
                                motion_filter_values.append(float(infos[aid]['motion_filter_active']))
                            except Exception:
                                pass
                        if 'motion_safety_margin' in infos[aid]:
                            try:
                                margin = float(infos[aid]['motion_safety_margin'])
                                if np.isfinite(margin):
                                    motion_safety_margins.append(margin)
                            except Exception:
                                pass
                        if 'predictive_social_risk' in infos[aid]:
                            try:
                                social_risks.append(float(infos[aid]['predictive_social_risk']))
                            except Exception:
                                pass
                        if 'predictive_front_risk' in infos[aid]:
                            try:
                                front_risks.append(float(infos[aid]['predictive_front_risk']))
                            except Exception:
                                pass
                        if event == 'goal':
                            successes[aid] += 1
                            reached_agents.add(aid)
                            metric_tracker.mark_event(aid, event, step_count)
                            print(f" {aid} 到达目标 (step={step_count})")
                        elif event == 'collision':
                            collisions[aid] += 1
                            collided_agents.add(aid)
                            metric_tracker.mark_event(aid, event, step_count)
                            print(f"碰撞: {aid} (step={step_count})")
                if any(bool(v) for v in truncateds.values()):
                    truncated_episode = True
                
                if step_count % 50 == 0:
                    print(f"Step {step_count}...", end="\r")

            record_trajectory_sample(step_count, infos if step_count else None, force=True)
            if trajectory_handle is not None:
                trajectory_handle.flush()
            
            total_success   = sum(successes.values())
            total_collision = sum(collisions.values())
            episode_wall_time_s = max(time.perf_counter() - episode_wall_start, 1.0e-9)
            simulated_time_s = float(step_count) * float(thresholds.control_dt)
            paper_metrics = metric_tracker.summary(
                reached_agents=reached_agents,
                steps=step_count,
            )
            ep_summary = {
                'steps': step_count,
                'episode_wall_time_s': episode_wall_time_s,
                'real_time_factor': simulated_time_s / episode_wall_time_s,
                'total_success': total_success,
                'total_collision': total_collision, 'avg_reward': sum(total_rewards.values()) / num_agents,
                'min_dist': min(min_dists) if min_dists else float("nan"),
                'avg_front_min': float(np.mean(front_mins)) if front_mins else float("nan"),
                'avg_r_ttc': float(np.mean(r_ttcs)) if r_ttcs else 0.0,
                'avg_dynamic_obs_count': float(np.mean(dynamic_obs_counts)) if dynamic_obs_counts else 0.0,
                'min_dynamic_obs_dist': min(dynamic_obs_min_dists) if dynamic_obs_min_dists else float("nan"),
                'min_dynamic_obs_ttc': min(dynamic_obs_min_ttcs) if dynamic_obs_min_ttcs else float("nan"),
                'avg_dynamic_obs_ttc_risk': float(np.mean(dynamic_obs_ttc_risks)) if dynamic_obs_ttc_risks else 0.0,
                'max_dynamic_obs_ttc_risk': max(dynamic_obs_ttc_risks) if dynamic_obs_ttc_risks else 0.0,
                'avg_shield_active': float(np.mean(shield_active_values)) if shield_active_values else 0.0,
                'avg_motion_filter_active': float(np.mean(motion_filter_values)) if motion_filter_values else 0.0,
                'min_motion_safety_margin': min(motion_safety_margins) if motion_safety_margins else float("nan"),
                'avg_social_risk': float(np.mean(social_risks)) if social_risks else 0.0,
                'avg_front_risk': float(np.mean(front_risks)) if front_risks else 0.0,
                'inference_ms_mean': float(np.mean(inference_ms)) if inference_ms else float("nan"),
                'inference_ms_p50': float(np.percentile(inference_ms, 50)) if inference_ms else float("nan"),
                'inference_ms_p95': float(np.percentile(inference_ms, 95)) if inference_ms else float("nan"),
                'inference_samples': len(inference_ms),
                'policy_only_inference_latency_ms_mean': float(np.mean(inference_ms)) if inference_ms else float("nan"),
                'policy_only_inference_latency_ms_p50': float(np.percentile(inference_ms, 50)) if inference_ms else float("nan"),
                'policy_only_inference_latency_ms_p95': float(np.percentile(inference_ms, 95)) if inference_ms else float("nan"),
                'controller_latency_ms_mean': float(np.mean(controller_ms)) if controller_ms else float("nan"),
                'controller_latency_ms_p95': float(np.percentile(controller_ms, 95)) if controller_ms else float("nan"),
                'safety_filter_latency_ms_mean': float(np.mean(safety_filter_ms)) if safety_filter_ms else float("nan"),
                'safety_filter_latency_ms_p50': float(np.percentile(safety_filter_ms, 50)) if safety_filter_ms else float("nan"),
                'safety_filter_latency_ms_p95': float(np.percentile(safety_filter_ms, 95)) if safety_filter_ms else float("nan"),
                'safety_filter_latency_ms_p99': float(np.percentile(safety_filter_ms, 99)) if safety_filter_ms else float("nan"),
                'safety_filter_latency_ms_max': max(safety_filter_ms) if safety_filter_ms else float("nan"),
                'safety_filter_latency_samples': len(safety_filter_ms),
                'min_agent_pairwise_dist': min(agent_pairwise_min_dists) if agent_pairwise_min_dists else float("nan"),
                'reached_agents': len(reached_agents),
                'collided_agents': len(collided_agents),
                'truncated': truncated_episode,
                'agent_success_rate': len(reached_agents) / max(1, num_agents),
                'team_success': bool(len(reached_agents) == num_agents),
                'collision_rate': len(collided_agents) / max(1, num_agents),
                'timeout_rate': float(truncated_episode),
                'SR': len(reached_agents) / max(1, num_agents),
                'TSR': float(len(reached_agents) == num_agents),
                'CR': len(collided_agents) / max(1, num_agents),
                'robot_failure_continue_protocol': bool(
                    env_config.get('reset_on_collision_event', False)
                    and not env_config.get('collision_ends_episode', False)
                    and not env_config.get('auto_reset_agents', False)
                    and int(env_config.get('min_active_agents_to_continue', 0)) == 0
                    and int(env_config.get('max_failed_agents_before_cutoff', 0)) == 0
                ),
                'counterfactual_credit_mean': (
                    float(np.mean(episode_cf_credits)) if episode_cf_credits else float("nan")
                ),
                'counterfactual_credit_std': (
                    float(np.std(episode_cf_credits)) if episode_cf_credits else float("nan")
                ),
                'counterfactual_credit_abs_mean': (
                    float(np.mean(np.abs(episode_cf_credits))) if episode_cf_credits else float("nan")
                ),
                'counterfactual_credit_samples': len(episode_cf_credits),
                **paper_metrics,
            }
            episode_stats.append(ep_summary)
            if cf_trace_handle is not None:
                cf_trace_handle.flush()

            # 2026-06-29: 论文评估用，写入 JSONL
            if args.results_jsonl:
                import json as _json
                _record = dict(ep_summary)
                _record['episode_index'] = len(episode_stats) - 1
                _record['episode_seed'] = episode_seed
                _record['initial_start_goal'] = initial_start_goal
                _record['initial_state_fingerprint'] = initial_state_fingerprint
                _record['fixed_start_tolerance_m'] = float(args.fixed_start_tolerance_m)
                dynamic_obstacles = int(env_config.get('num_dynamic_obstacles') or 0)
                trajectory_fingerprint = str(
                    getattr(env, 'dynamic_obstacle_trajectory_fingerprint', '') or '')
                trajectory_spec = dict(
                    getattr(env, 'dynamic_obstacle_trajectory_spec', {}) or {})
                dynamic_initial_state = dict(
                    trajectory_spec.get('dynamic_initial_state', {}) or {})
                if packaged_dynamic is not None:
                    expected_dynamic_state = {
                        'episode_index': int(packaged_dynamic['episode_index']),
                        'seed': int(packaged_dynamic['episode_seed']),
                        'count': int(packaged_dynamic['count']),
                        'positions': packaged_dynamic['positions'],
                        'state_hash': str(packaged_dynamic['state_hash']),
                    }
                    if dynamic_initial_state != expected_dynamic_state:
                        raise RuntimeError(
                            f"fixed dynamic initial state mismatch at episode {ep}: "
                            f"expected={expected_dynamic_state} actual={dynamic_initial_state}"
                        )
                fixed_scene_active = fixed_scenario is not None or packaged_scenario is not None
                _record['paired_scenario_eligible'] = bool(
                    initial_state_fingerprint
                    and fixed_scene_active
                    and (dynamic_obstacles == 0 or trajectory_fingerprint)
                )
                _record['pairing_basis'] = (
                    'fixed_start_goal_and_dynamic_trajectory'
                    if fixed_scene_active and dynamic_obstacles > 0 and trajectory_fingerprint else (
                        'fixed_scenario_bank'
                        if fixed_scene_active else (
                        'start_goal_static_environment'
                        if dynamic_obstacles == 0 else 'unverified_dynamic_obstacle_trajectory'
                        )
                    )
                )
                _record['dynamic_obstacle_trajectory_fingerprint'] = trajectory_fingerprint
                _record['dynamic_obstacle_trajectory_spec'] = trajectory_spec
                _record['dynamic_obstacle_initial_state'] = dynamic_initial_state
                _record['num_agents'] = num_agents
                _record['trained_num_agents'] = trained_num_agents
                _record['zero_shot_team_size'] = bool(num_agents != trained_num_agents)
                _record['agent_id_encoding_capacity'] = 8
                _record['agent_id_aliasing'] = bool(num_agents > 8)
                _record['map_number'] = env_config.get('map_number')
                _record['num_dynamic_obstacles'] = env_config.get('num_dynamic_obstacles')
                _record['num_static_obstacles'] = env_config.get('num_static_obstacles')
                _record['obs_speed'] = env_config.get('obs_speed')
                _record['obs_speed_scale'] = args.obs_speed_scale
                _record['sensor_noise_std'] = env_config.get('sensor_noise_std', 0.0)
                _record['comm_dropout_prob'] = env_config.get('comm_dropout_prob', 0.0)
                _record['comm_latency_steps'] = env_config.get('comm_latency_steps', 0)
                _record['comm_jitter_steps'] = env_config.get('comm_jitter_steps', 0)
                _record['comm_noise_std'] = env_config.get('comm_noise_std', 0.0)
                _record['max_episode_steps'] = env_config.get('max_episode_steps')
                _record['predictive_horizon_sec'] = env_config.get('predictive_horizon_sec')
                _record['shield_enable'] = bool(env_config.get('shield_enable', False))
                _record['reset_on_collision_event'] = bool(
                    env_config.get('reset_on_collision_event', False))
                _record['collision_ends_episode'] = bool(
                    env_config.get('collision_ends_episode', False))
                _record['auto_reset_agents'] = bool(
                    env_config.get('auto_reset_agents', False))
                _record['min_active_agents_to_continue'] = int(
                    env_config.get('min_active_agents_to_continue', 0))
                _record['max_failed_agents_before_cutoff'] = int(
                    env_config.get('max_failed_agents_before_cutoff', 0))
                _record['evaluation_system'] = (
                    'orca_dwa_system'
                    if external_controller is not None else (
                        'policy_plus_safety_filter'
                        if bool(env_config.get('shield_enable', False)) else 'policy_only'
                    )
                )
                _record['controller'] = args.controller
                _record['graph_ablation'] = custom_model_cfg.get('graph_ablation')
                _record['graph_architecture'] = custom_model_cfg.get('graph_architecture', 'dual')
                _record['graph_gate_mode'] = custom_model_cfg.get('graph_gate_mode', 'vector')
                _record['graph_gate_fixed_value'] = custom_model_cfg.get('graph_gate_fixed_value')
                _record['counterfactual_mask_strategy'] = custom_model_cfg.get(
                    'counterfactual_mask_strategy', 'zero_agent'
                )
                _record['counterfactual_critic_source'] = custom_model_cfg.get(
                    'counterfactual_critic_source', 'online'
                )
                _record['counterfactual_advantage_coef'] = policy_overrides.get(
                    'counterfactual_advantage_coef', 0.0
                )
                _record['risk_bias_scale'] = custom_model_cfg.get('risk_bias_scale')
                if fixed_scenario is not None:
                    _record['scenario_bank'] = args.scenario_bank or 'paper_micro_v1'
                    _record['scenario_case'] = fixed_scenario.case_id
                    _record['scenario_category'] = fixed_scenario.category
                if packaged_scenario is not None:
                    _record['fixed_scenario_protocol'] = fixed_scenario_protocol
                    _record['scenario_id'] = str(packaged_scenario['scenario_id'])
                    _record['scenario_category'] = str(packaged_scenario['scenario_type'])
                    _record['route_plan'] = packaged_scenario['route_plan']
                    _record['initial_yaws'] = initial_yaws
                    _record['fixed_scenario_source_index'] = int(packaged_scenario['episode'])
                _record['seed'] = int(args.seed)
                _record['training_seed'] = args.training_seed
                _record['checkpoint_path'] = checkpoint_path
                # 用 NaN→None 转换以保证 JSON 合法
                for _k, _v in list(_record.items()):
                    if isinstance(_v, float) and np.isnan(_v):
                        _record[_k] = None
                if args.method_label:
                    _record['method'] = args.method_label
                if args.scenario_label:
                    _record['scenario'] = args.scenario_label
                if args.experiment_condition:
                    _record['experiment_condition'] = args.experiment_condition
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(args.results_jsonl)), exist_ok=True)
                except Exception:
                    pass
                with open(args.results_jsonl, 'a', encoding='utf-8') as _f:
                    _f.write(_json.dumps(_record, ensure_ascii=False) + '\n')
                if args.results_csv:
                    _append_csv_record(args.results_csv, _record)
            print(
                f"  [episode-summary] steps={ep_summary['steps']} "
                f"avg_reward={ep_summary['avg_reward']:.2f} "
                f"success={ep_summary['total_success']} collision={ep_summary['total_collision']} "
                f"reached_agents={ep_summary['reached_agents']} collided_agents={ep_summary['collided_agents']} "
                f"min_dist={ep_summary['min_dist']:.3f} "
                f"avg_r_ttc={ep_summary['avg_r_ttc']:.3f} "
                f"min_dyn_ttc={ep_summary['min_dynamic_obs_ttc']:.3f} "
                f"dyn_risk={ep_summary['avg_dynamic_obs_ttc_risk']:.3f} "
                f"avg_social_risk={ep_summary['avg_social_risk']:.3f} "
                f"avg_front_risk={ep_summary['avg_front_risk']:.3f} "
                f"truncated={ep_summary['truncated']}"
            )
            
        print(f"\n{'='*80}")
        print(f"总体测试结果 ({args.num_episodes} Episodes)")
        print(f"{'='*80}")
        avg_steps    = sum(s['steps'] for s in episode_stats) / len(episode_stats)
        all_success  = sum(s['total_success'] for s in episode_stats)
        all_collision= sum(s['total_collision'] for s in episode_stats)
        avg_reward = sum(s['avg_reward'] for s in episode_stats) / len(episode_stats)
        valid_min_dists = [s['min_dist'] for s in episode_stats if not np.isnan(s['min_dist'])]
        avg_min_dist = float(np.mean(valid_min_dists)) if valid_min_dists else float("nan")
        avg_social_risk = sum(s['avg_social_risk'] for s in episode_stats) / len(episode_stats)
        avg_front_risk = sum(s['avg_front_risk'] for s in episode_stats) / len(episode_stats)
        avg_r_ttc = sum(s['avg_r_ttc'] for s in episode_stats) / len(episode_stats)
        valid_dyn_ttc = [s['min_dynamic_obs_ttc'] for s in episode_stats if not np.isnan(s['min_dynamic_obs_ttc'])]
        min_dyn_ttc = float(np.min(valid_dyn_ttc)) if valid_dyn_ttc else float("nan")
        avg_dyn_risk = sum(s['avg_dynamic_obs_ttc_risk'] for s in episode_stats) / len(episode_stats)
        avg_shield_active = sum(s['avg_shield_active'] for s in episode_stats) / len(episode_stats)
        total_reached_agents = sum(s['reached_agents'] for s in episode_stats)
        total_collided_agents = sum(s['collided_agents'] for s in episode_stats)
        truncated_eps = sum(1 for s in episode_stats if s['truncated'])
        print(f"   总计到达: {all_success} 次  |  总计碰撞: {all_collision} 次")
        print(f"   平均回报: {avg_reward:.2f}  |  平均步数: {avg_steps:.1f}")
        print(f"   平均最小间距: {avg_min_dist:.3f} m")
        print(f"   平均社交风险: {avg_social_risk:.3f}  |  平均前向风险: {avg_front_risk:.3f}")
        print(f"   平均 r_ttc: {avg_r_ttc:.3f}  |  最小动态障碍 TTC: {min_dyn_ttc:.3f} s")
        print(f"   平均动态障碍 TTC 风险: {avg_dyn_risk:.3f}  |  平均 shield 激活率: {avg_shield_active:.3f}")
        print(f"   到达过目标的 agent 数: {total_reached_agents}  |  发生过碰撞的 agent 数: {total_collided_agents}")
        print(f"   被时间截断的 Episode: {truncated_eps}/{len(episode_stats)}")
        print(f"{'='*80}\n")
        
    except KeyboardInterrupt:
        print("\n测试中断")
    finally:
        if gate_hook_handle is not None:
            gate_hook_handle.remove()
        if cf_trace_handle is not None:
            cf_trace_handle.close()
        if trajectory_handle is not None:
            trajectory_handle.close()
        # 落盘注意力(F) + 风险场(D) 逐步日志，供绘图脚本使用
        if capture_attention and args.attention_out and (attn_records or risk_records):
            try:
                os.makedirs(os.path.dirname(os.path.abspath(args.attention_out)), exist_ok=True)
                np.savez_compressed(
                    args.attention_out,
                    attention_records=json.dumps(attn_records),
                    risk_records=json.dumps(risk_records),
                    map_number=int(env_config.get("map_number", -1)),
                    num_agents=int(num_agents),
                    scenario=str(args.scenario_label or ""),
                )
                print(f"[viz] 已保存 {len(attn_records)} 条注意力 + {len(risk_records)} 条风险场记录 -> {args.attention_out}")
            except Exception as _e:
                print(f"[viz] 保存注意力/风险场日志失败: {_e}")
        env.close()
        ray.shutdown()

if __name__ == "__main__":
    main()
