"""
GNN-MAPPO 环境包装器
基于动态图神经网络的多智能体强化学习环境
"""
from __future__ import annotations

import os
import time
import random
import math
import logging
import inspect
import yaml
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
from PIL import Image
import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetEntityState
from std_msgs.msg import Float32MultiArray
from gnn_marl_training.global_planner import AStarPlanner, WaypointExtractor, PathTrackingUtils
from gnn_marl_training.fixed_benchmark_scenarios import get_fixed_benchmark_cases
from gnn_marl_training.interaction_execution_utils import (
    build_interaction_subgoal_offset,
    compute_tracking_controller_cmd,
)
from gnn_marl_training.interaction_observation_utils import (
    build_high_level_policy_features,
    build_interaction_neighbor_token,
    compute_progress_delta_signal,
    compute_social_risk_summary,
    compute_stuck_score,
    project_point_to_polyline_arclength,
)
from gnn_marl_training.interaction_reward_utils import (
    compute_method3_reward_terms,
    compute_option_outcome_reward,
    OptionOutcomeRewardTerms,
)
from gnn_marl_training.interaction_option_definitions import (
    TRAINING_TO_FEASIBILITY_OPTION,
    CANONICAL_MODE_BY_TRAINING_OPTION,
    NUM_TRAINING_OPTIONS,
    DetourPhase,
    DETOUR_ENTER_MIN_STEPS,
    DETOUR_PASS_MIN_STEPS,
    DETOUR_LATERAL_DISPLACEMENT_THRESH,
    DETOUR_FRONT_CLEAR_THRESH,
    DETOUR_FRONT_RISK_THRESH,
    OptionOutcome,
    CollisionAttribution,
)
from gnn_marl_training.env_space_reward_utils import (
    ClassicNavigationRewardTerms,
    PotentialRewardConfig,
    RewardAggregationOverrides,
    build_action_mask_features,
    build_independent_env_observation,
    build_observation_schema_spec,
    build_option_state_features,
    build_tracking_target_features,
    compute_classic_navigation_reward,
    compute_interaction_potential_reward,
    compute_pairwise_local_rewards,
    configure_independent_env_action_observation_spaces,
    configure_multi_agent_observation_space,
)
from gnn_marl_training.option_feasibility import (
    evaluate_option_feasibility,
    build_interaction_action_mask,
    OptionFeasibilityResult,
)
from gnn_marl_training.waypoint_visualizer import WaypointVisualizer


def _setup_env_logger(log_path: str, worker_id: int = 0) -> logging.Logger:
    """
    创建每个 Worker 独立的日志记录器。
    格式: [时间] [LEVEL] [worker_id] 消息内容
    """
    logger = logging.getLogger(f'gnn_marl_env.worker{worker_id}')
    if logger.handlers:          # 已初始化则直接返回（多次实例化保护）
        return logger
    logger.setLevel(logging.DEBUG)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d [%(levelname)s] [W%(worker_id)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # 同时保留到控制台（WARNING 及以上）
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


class _LogAdapter(logging.LoggerAdapter):
    """自动插入 worker_id 到日志记录的适配器。"""
    def process(self, msg, kwargs):
        kwargs.setdefault('extra', {})['worker_id'] = self.extra.get('worker_id', '?')
        return msg, kwargs


class GNNMARLEnv(MultiAgentEnv):
    """
    GNN-MAPPO 环境包装器
    
    核心特性：
    1. 动态图构建：只连接近距离机器人
    2. 消息传递：机器人间信息交换
    3. 增强观测：局部地图 + 邻居状态
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self._num_agents = int(config.get('num_agents', 3))
        self.communication_range = config.get('communication_range', 3.5)  # 米, 与LDS-01激光雷达量程一致(URDF <max>3.5</max>)
        self.interaction_neighbor_perception_range = float(
            max(0.5, config.get('interaction_neighbor_perception_range', self.communication_range))
        )
        self.enable_local_map = config.get('enable_local_map', False)
        self.enable_neighbor_obs = config.get('enable_neighbor_obs', True)

        # ===== 通信建模（面向真实机器人部署 sim2real） =====
        # centralized_oracle : 理想化全局同步（仿真上界，用于对比）
        # decentralized       : 模拟真实通信（延迟/抖动/丢包/噪声）
        self.comm_mode         = config.get('comm_mode', 'decentralized')
        self.comm_dropout_prob = float(config.get('comm_dropout_prob', 0.05))   # 单次消息丢包率
        self.comm_latency_steps= int(config.get('comm_latency_steps', 1))       # 固定延迟(步)
        self.comm_jitter_steps = int(config.get('comm_jitter_steps', 1))        # 随机抖动上限(步)
        self.comm_noise_std    = float(config.get('comm_noise_std', 0.02))      # 位置/速度测量噪声σ(m)
        _history_len = max(32, self.comm_latency_steps + self.comm_jitter_steps + 4)
        self._state_history: deque = deque(maxlen=_history_len)
        self.rng = np.random.default_rng(config.get('comm_seed', None))

        # ── 步进语义开关 ─────────────────────────────────────────────────
        # auto_reset_agents=True  : 连续任务流（agent done 后立刻在原 episode 重置）
        # auto_reset_agents=False : 正统 partial-done 语义（agent 终止后退出当前 episode）
        self.auto_reset_agents = bool(config.get('auto_reset_agents', False))
        # 是否将“collision 事件”作为局部重置触发条件（默认开启：判定为碰撞才重置）
        self.reset_on_collision_event = bool(config.get('reset_on_collision_event', True))
        self.min_active_agents_to_continue = int(max(0, config.get('min_active_agents_to_continue', 2)))
        self.max_failed_agents_before_cutoff = int(max(0, config.get('max_failed_agents_before_cutoff', 2)))
        # 高冲突训练模式：通过更激进的起终点路由采样制造会车/交叉冲突。
        self.high_conflict_mode = str(config.get('high_conflict_mode', 'off')).strip().lower()
        if self.high_conflict_mode not in ('off', 'mixed', 'aggressive'):
            self.high_conflict_mode = 'off'
        self.high_conflict_prob = float(np.clip(float(config.get('high_conflict_prob', 0.75)), 0.0, 1.0))
        self.failure_replay_enable = bool(config.get('failure_replay_enable', False))
        self.failure_replay_buffer_size = int(max(1, config.get('failure_replay_buffer_size', 64)))
        self.failure_replay_base_prob = float(
            np.clip(float(config.get('failure_replay_base_prob', 0.10)), 0.0, 1.0)
        )
        self.failure_replay_max_prob = float(
            np.clip(float(config.get('failure_replay_max_prob', 0.70)), 0.0, 1.0)
        )
        if self.failure_replay_max_prob < self.failure_replay_base_prob:
            self.failure_replay_max_prob = self.failure_replay_base_prob
        self.failure_replay_success_threshold = float(
            np.clip(float(config.get('failure_replay_success_threshold', 0.75)), 0.0, 1.0)
        )
        self.failure_replay_posterior_samples = int(
            max(8, config.get('failure_replay_posterior_samples', 48))
        )
        self.reward_aggregation_overrides = RewardAggregationOverrides(
            **dict(config.get('reward_aggregation_overrides', {}) or {})
        )
        self.interaction_potential_overrides = PotentialRewardConfig(
            **dict(config.get('interaction_potential_overrides', {}) or {})
        )
        self.failure_replay_evidence_smoothing = float(
            max(0.1, config.get('failure_replay_evidence_smoothing', 2.0))
        )
        self._failure_replay_rng = random.Random(config.get('failure_replay_seed', None))
        self._failure_replay_records: Dict[
            Tuple[Tuple[str, float, float, float, float], ...],
            Dict[str, Any],
        ] = {}
        self._episode_counter = 0
        self._episode_route_plan_source = 'random'
        self._episode_route_plan_assigned: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self._episode_route_plan_actual: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self.fixed_route_plan_sequence = [
            self._materialize_route_plan(self._normalize_route_plan(plan))
            for plan in (config.get('fixed_route_plan_sequence') or [])
            if isinstance(plan, dict) and plan
        ]
        self.fixed_route_plan_cycle = bool(config.get('fixed_route_plan_cycle', False))
        self.corner_curriculum_enable = bool(config.get('corner_curriculum_enable', False))
        self.corner_curriculum_prob = float(np.clip(float(config.get('corner_curriculum_prob', 0.0)), 0.0, 1.0))
        self.corner_curriculum_set = str(config.get('corner_curriculum_set', 'corner_curriculum_v1')).strip() or 'corner_curriculum_v1'
        self.corner_curriculum_cycle = bool(config.get('corner_curriculum_cycle', True))
        self.corner_curriculum_mix_conflict = bool(config.get('corner_curriculum_mix_conflict', True))
        self._corner_curriculum_plans = self._load_corner_curriculum_plans(self.corner_curriculum_set)
        self._corner_curriculum_index = 0
        self._fixed_route_plan_index = 0

        # 创建独立环境实例
        self.agents = {}
        env_signature = inspect.signature(IndependentRobotEnv.__init__)
        for i in range(self._num_agents):
            candidate_kwargs = {
                'robot_id': i,
                'map_number': config.get('map_number', 3),
                'max_episode_steps': config.get('max_episode_steps', 1000),
                'communication_range': float(config.get('communication_range', 3.5)),
                'interaction_neighbor_perception_range': float(config.get('interaction_neighbor_perception_range', 0.0)),
                'collision_ends_episode': bool(config.get('collision_ends_episode', False)),
                'collision_hard_dist': float(config.get('collision_hard_dist', 0.20)),
                'collision_persist_dist': float(config.get('collision_persist_dist', 0.26)),
                'collision_persist_steps': int(config.get('collision_persist_steps', 2)),
                'waypoint_reach_radius': float(config.get('waypoint_reach_radius', 0.8)),
                'waypoint_distance_threshold': float(config.get('waypoint_distance_threshold', 1.2)),
                'waypoint_min_clearance_m': float(config.get('waypoint_min_clearance_m', 0.40)),
                'use_voronoi_planner': bool(config.get('use_voronoi_planner', False)),
                'voronoi_min_clearance_m': float(config.get('voronoi_min_clearance_m', 0.35)),
                'num_dynamic_obstacles': config.get('num_dynamic_obstacles', 8),
                'obs_speed': config.get('obs_speed', 0.3),
                'rolling_lookahead_dist': float(config.get('rolling_lookahead_dist', 0.8)),
                'subgoal_block_front_dist': float(config.get('subgoal_block_front_dist', 0.42)),
                'subgoal_min_side_clearance': float(config.get('subgoal_min_side_clearance', 0.20)),
                'subgoal_detour_forward_gain': float(config.get('subgoal_detour_forward_gain', 0.55)),
                'subgoal_detour_lateral_gain': float(config.get('subgoal_detour_lateral_gain', 0.75)),
                'subgoal_detour_hold_steps': int(config.get('subgoal_detour_hold_steps', 8)),
                'subgoal_deadlock_front_dist': float(config.get('subgoal_deadlock_front_dist', 0.48)),
                'subgoal_deadlock_speed_thresh': float(config.get('subgoal_deadlock_speed_thresh', 0.03)),
                'subgoal_deadlock_steps': int(config.get('subgoal_deadlock_steps', 10)),
                'replan_on_deadlock': bool(config.get('replan_on_deadlock', True)),
                'replan_cooldown_steps': int(config.get('replan_cooldown_steps', 25)),
                'stall_global_replan_enable': bool(config.get('stall_global_replan_enable', False)),
                'stall_global_replan_sec': float(config.get('stall_global_replan_sec', 5.0)),
                'stall_replan_position_epsilon': float(config.get('stall_replan_position_epsilon', 0.18)),
                'stall_replan_progress_epsilon': float(config.get('stall_replan_progress_epsilon', 0.12)),
                'dynamic_replan_neighbor_dist': float(config.get('dynamic_replan_neighbor_dist', 1.8)),
                'dynamic_replan_ttc': float(config.get('dynamic_replan_ttc', 2.6)),
                'dynamic_replan_block_radius': float(config.get('dynamic_replan_block_radius', 0.55)),
                'obs_target_dist_clip': float(config.get('obs_target_dist_clip', 6.0)),
                'obs_target_filter_alpha': float(config.get('obs_target_filter_alpha', 0.35)),
                'obs_target_max_step': float(config.get('obs_target_max_step', 0.45)),
                'progress_reward_scale': float(config.get('progress_reward_scale', 0.0)),
                'path_progress_reward_scale': float(config.get('path_progress_reward_scale', 0.0)),
                'goal_progress_reward_scale': float(config.get('goal_progress_reward_scale', 4.0)),
                'goal_reward': float(config.get('goal_reward', 20.0)),
                'collision_penalty': float(config.get('collision_penalty', 20.0)),
                'time_penalty': float(config.get('time_penalty', 0.01)),
                'close_obstacle_penalty_scale': float(config.get('close_obstacle_penalty_scale', 0.30)),
                'close_obstacle_dist': float(config.get('close_obstacle_dist', 0.55)),
                'team_reward_lambda': float(config.get('team_reward_lambda', 1.0)),
                'use_gazebo_collision': bool(config.get('use_gazebo_collision', True)),
                'lidar_collision_fallback': bool(config.get('lidar_collision_fallback', True)),
                'obstacle_filter_range': float(config.get('obstacle_filter_range', 2.0)),
                'obstacle_filter_fov_deg': float(config.get('obstacle_filter_fov_deg', 360.0)),
                'obstacle_top_k': int(config.get('obstacle_top_k', 9)),
                'angular_bins': int(config.get('angular_bins', 64)),
                'predictive_feature_enable': bool(config.get('predictive_feature_enable', True)),
                'predictive_horizon_sec': float(config.get('predictive_horizon_sec', 1.2)),
                'predictive_social_ttc_safe': float(config.get('predictive_social_ttc_safe', 2.2)),
                'predictive_front_ttc_safe': float(config.get('predictive_front_ttc_safe', 1.2)),
                'predictive_min_sep': float(config.get('predictive_min_sep', 0.55)),
                'predictive_social_range': float(config.get('predictive_social_range', 2.5)),
                'predictive_social_penalty_scale': float(config.get('predictive_social_penalty_scale', 0.17)),
                'predictive_front_penalty_scale': float(config.get('predictive_front_penalty_scale', 0.16)),
                'social_proximity_risk_scale': float(config.get('social_proximity_risk_scale', 0.34)),
                'neighbor_prediction_top_k': int(config.get('neighbor_prediction_top_k', 2)),
                'gap_feature_enable': bool(config.get('gap_feature_enable', True)),
                'yielding_enable': bool(config.get('yielding_enable', True)),
                'yielding_soft_dist': float(config.get('yielding_soft_dist', 0.90)),
                'yielding_stop_dist': float(config.get('yielding_stop_dist', 0.50)),
                'yielding_hard_stop_dist': float(config.get('yielding_hard_stop_dist', 0.30)),
                'yielding_ttc': float(config.get('yielding_ttc', 2.4)),
                'yielding_commit_steps': int(config.get('yielding_commit_steps', 5)),
                'replan_fixed_cost': float(config.get('replan_fixed_cost', 0.03)),
                'replan_freq_cost': float(config.get('replan_freq_cost', 0.012)),
                'replan_time_cost': float(config.get('replan_time_cost', 0.015)),
                'replan_time_budget_sec': float(config.get('replan_time_budget_sec', 0.08)),
                'replan_window_steps': int(config.get('replan_window_steps', 80)),
                'method3_reward_window_steps': int(config.get('method3_reward_window_steps', 8)),
                'obstacle_motion_feature_enable': bool(config.get('obstacle_motion_feature_enable', True)),
                'obstacle_motion_top_k': int(config.get('obstacle_motion_top_k', 3)),
                'subgoal_progress_reward_scale': float(config.get('subgoal_progress_reward_scale', 1.2)),
                'detour_progress_relax': float(config.get('detour_progress_relax', 0.30)),
                'risk_aware_forward_penalty_scale': float(config.get('risk_aware_forward_penalty_scale', 0.28)),
                'safe_turn_reward_scale': float(config.get('safe_turn_reward_scale', 0.15)),
                'head_on_avoidance_reward_scale': float(config.get('head_on_avoidance_reward_scale', 0.90)),
                'risk_gate_soft': float(config.get('risk_gate_soft', 0.08)),
                'risk_gate_hard': float(config.get('risk_gate_hard', 0.50)),
                'avoidance_low_risk_scale': float(config.get('avoidance_low_risk_scale', 0.45)),
                'navigation_high_risk_scale': float(config.get('navigation_high_risk_scale', 0.80)),
                'time_penalty_risk_relax': float(config.get('time_penalty_risk_relax', 0.65)),
                'reward_aggregation_overrides': config.get('reward_aggregation_overrides', None),
                'interaction_potential_overrides': config.get('interaction_potential_overrides', None),
                'action_mode': 'interaction_mode',
            }
            supported_kwargs = {
                key: value for key, value in candidate_kwargs.items()
                if key in env_signature.parameters
            }
            self.agents[f"agent_{i}"] = IndependentRobotEnv(**supported_kwargs)
            self.agents[f"agent_{i}"].parent_env = self
        
        self.agent_ids = list(self.agents.keys())
        self.current_step_count = 0
        self.max_steps = config.get('max_episode_steps', 1000)
        self.dones = set()  # 已完成的智能体
        self.failed_agents = set()  # 因碰撞等失败退出的智能体
        self._agent_ids = set(self.agent_ids)
        self.possible_agents = list(self.agent_ids)
        
        # 机器人位置缓存（用于构建图）
        self.robot_positions = {aid: np.zeros(2) for aid in self.agent_ids}
        # 机器人速度缓存（用于协作奖励和碰撞预测）
        self.robot_velocities = {aid: np.zeros(2) for aid in self.agent_ids}

        # episode 级累计统计（auto-reset 模式下每个 agent 在一个 episode 内的成功/碰撞次数）
        self.episode_successes  = {aid: 0 for aid in self.agent_ids}
        self.episode_collisions = {aid: 0 for aid in self.agent_ids}
        self.episode_stats = {
            'total_collisions': 0,
            'near_misses': 0,
            'successful_navigations': 0,
        }
        self._pair_event_memory = {}
        self.team_reward_lambda = float(config.get('team_reward_lambda', 1.0))
        self._pair_event_memory: Dict[Tuple[str, str], Dict[str, float]] = {}
        self._interaction_contexts: Dict[str, Dict[str, Any]] = {
            aid: {
                'agent_id': aid,
                'mode': 'idle',
                'mode_id': 0.0,
                'in_conflict': 0.0,
                'has_token': 0.0,
                'should_yield': 0.0,
                'partner': '',
                'partner_dist': float('inf'),
                'closing_speed': 0.0,
                'ttc': float('inf'),
                'severity': 0.0,
                'turn_sign': 0.0,
                'component_size': 1.0,
                'wait_steps': 0.0,
                'wait_age_norm': 0.0,
            }
            for aid in self.agent_ids
        }
        self._interaction_wait_steps = {aid: 0 for aid in self.agent_ids}

        # 定义增强后的观测空间
        self._define_observation_space()
        
        # 动作空间与 IndependentRobotEnv 保持一致（支持连续 v,w 或离散动作原语）
        self.action_space = self.agents['agent_0'].action_space

        # ── ROS2 通信桥接（仅 comm_mode='ros2_bridge' 时启用） ──────────────
        # 训练时：Bridge 节点代替每台机器人发布状态到 /gnn_swarm/robot_X/state，
        # 邻居信息从 DDS 消息缓冲区按时间戳年龄过滤，而非按步数计数。
        # 话题名称/消息格式与 robot_policy_node.py 部署代码完全一致。
        self._bridge_node: Any = None
        self._ros2_state_pubs: Dict[str, Any] = {}
        # 按发送方 aid 存储收到的 DDS 消息: deque of (recv_wall_sec, [id,x,y,vx,vy,send_ts])
        self._ros2_neighbor_bufs: Dict[str, deque] = {
            aid: deque(maxlen=64) for aid in self.agent_ids
        }

        # ── 日志记录器（每个 Worker 写入独立文件） ─────────────────────────────
        _worker_id = config.get('worker_index', os.getpid())
        _log_dir   = config.get('log_dir', os.path.expanduser('~/ray_results/gnn_marl_logs'))
        _log_file  = os.path.join(_log_dir, f'env_worker{_worker_id}.log')
        self.logger = _LogAdapter(
            _setup_env_logger(_log_file, _worker_id),
            {'worker_id': _worker_id}
        )
        self.log_every_n_steps = int(config.get('log_every_n_steps', 50))
        self.logger.info(
            '✅ GNNMARLEnv init: agents=%d comm_mode=%s range=%.1fm latency=%d noise=%.3f',
            self._num_agents, self.comm_mode, self.communication_range,
            self.comm_latency_steps, self.comm_noise_std
        )
        self.logger.info(
            'obs dims: base=%d neighbor=%d reset=%d global=%d total=%d',
            self.base_obs_dim,
            self.neighbor_dim,
            self.reset_flag_dim,
            self.global_state_dim,
            int(self.observation_space.shape[0]),
        )

        if self.comm_mode == 'ros2_bridge':
            self._setup_ros2_comm_bridge()

    def _sample_conflict_routes(self, n_routes: int):
        master = self.agents.get('agent_0')
        if master is None:
            return []

        route_lib_map = getattr(master, '_MAP_COLLISION_ROUTE_LIBRARY', {})
        route_lib = route_lib_map.get(master.map_number, [])
        if not route_lib:
            route_lib = master._MAP_FALLBACK_POSES.get(master.map_number, [])
        if not route_lib:
            return []

        picks = random.sample(route_lib, min(max(1, n_routes), len(route_lib)))
        out = []
        for (sx, sy), (gx, gy) in picks:
            if self.high_conflict_mode == 'aggressive' and random.random() < 0.35:
                out.append(((gx, gy), (sx, sy)))
            else:
                out.append(((sx, sy), (gx, gy)))
        return out

    def _load_corner_curriculum_plans(self, set_name: str) -> List[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]]:
        try:
            cases = get_fixed_benchmark_cases(set_name)
        except Exception:
            self.logger.warning('[corner_curriculum] failed to load set=%s', set_name)
            return []

        plans: List[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]] = []
        for case in cases:
            if int(case.num_agents) > self._num_agents:
                continue
            plan = self._materialize_route_plan(self._normalize_route_plan(case.route_plan))
            if plan:
                plans.append(plan)
        return plans

    def _consume_corner_curriculum_plan(self) -> Tuple[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] | None, Dict[str, Any] | None]:
        if not self.corner_curriculum_enable or not self._corner_curriculum_plans:
            return None, None
        if random.random() > self.corner_curriculum_prob:
            return None, {
                'corner_curriculum_set': self.corner_curriculum_set,
                'selected': False,
                'prob': self.corner_curriculum_prob,
            }
        if self._corner_curriculum_index >= len(self._corner_curriculum_plans):
            if not self.corner_curriculum_cycle:
                return None, {
                    'corner_curriculum_set': self.corner_curriculum_set,
                    'corner_curriculum_exhausted': True,
                    'total': len(self._corner_curriculum_plans),
                }
            self._corner_curriculum_index = 0
        plan = self._corner_curriculum_plans[self._corner_curriculum_index]
        meta = {
            'corner_curriculum_set': self.corner_curriculum_set,
            'corner_curriculum_index': self._corner_curriculum_index,
            'corner_curriculum_total': len(self._corner_curriculum_plans),
            'selected': True,
            'prob': self.corner_curriculum_prob,
        }
        self._corner_curriculum_index += 1
        return dict(plan), meta

    def _consume_fixed_route_plan(self) -> Tuple[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] | None, Dict[str, Any] | None]:
        if not self.fixed_route_plan_sequence:
            return None, None
        if self._fixed_route_plan_index >= len(self.fixed_route_plan_sequence):
            if not self.fixed_route_plan_cycle:
                return None, {
                    'fixed_route_plan_exhausted': True,
                    'total': len(self.fixed_route_plan_sequence),
                }
            self._fixed_route_plan_index = 0
        plan = self.fixed_route_plan_sequence[self._fixed_route_plan_index]
        meta = {
            'fixed_route_plan_index': self._fixed_route_plan_index,
            'fixed_route_plan_total': len(self.fixed_route_plan_sequence),
        }
        self._fixed_route_plan_index += 1
        return dict(plan), meta

    def _build_episode_route_plan(self):
        fixed_plan, fixed_meta = self._consume_fixed_route_plan()
        if fixed_plan:
            return fixed_plan, 'fixed_benchmark', fixed_meta
        replay_plan, replay_meta = self._select_failure_replay_plan()
        if replay_plan:
            return replay_plan, 'failure_replay', replay_meta
        corner_plan, corner_meta = self._consume_corner_curriculum_plan()
        if corner_plan:
            if self.corner_curriculum_mix_conflict and self._num_agents > len(corner_plan):
                extra_routes = self._sample_conflict_routes(self._num_agents - len(corner_plan))
                extra_iter = iter(extra_routes)
                for aid in self.agent_ids:
                    if aid in corner_plan:
                        continue
                    route = next(extra_iter, None)
                    if route is None:
                        break
                    corner_plan[aid] = route
            return corner_plan, 'corner_curriculum', corner_meta
        if self.high_conflict_mode == 'off':
            return {}, 'random', replay_meta or corner_meta
        if self.high_conflict_mode == 'mixed' and random.random() > self.high_conflict_prob:
            return {}, 'random', replay_meta or corner_meta

        routes = self._sample_conflict_routes(self._num_agents)
        if not routes:
            return {}, 'random', replay_meta or corner_meta

        random.shuffle(routes)
        plan = {}
        for idx, aid in enumerate(self.agent_ids):
            plan[aid] = routes[idx % len(routes)]
        return plan, 'high_conflict', replay_meta or corner_meta

    def _sample_conflict_route_for_respawn(self):
        if self.high_conflict_mode == 'off':
            return None
        if self.high_conflict_mode == 'mixed' and random.random() > self.high_conflict_prob:
            return None
        routes = self._sample_conflict_routes(1)
        if not routes:
            return None
        return routes[0]

    @staticmethod
    def _normalize_route_point(xy: Tuple[float, float]) -> Tuple[float, float]:
        return (round(float(xy[0]), 2), round(float(xy[1]), 2))

    def _normalize_route_plan(
        self,
        plan: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> Tuple[Tuple[str, float, float, float, float], ...]:
        normalized = []
        for aid in sorted(plan.keys()):
            (start_xy, goal_xy) = plan[aid]
            sx, sy = self._normalize_route_point(start_xy)
            gx, gy = self._normalize_route_point(goal_xy)
            normalized.append((aid, sx, sy, gx, gy))
        return tuple(normalized)

    @staticmethod
    def _materialize_route_plan(
        signature: Tuple[Tuple[str, float, float, float, float], ...],
    ) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]:
        plan = {}
        for aid, sx, sy, gx, gy in signature:
            plan[aid] = ((float(sx), float(sy)), (float(gx), float(gy)))
        return plan

    def _estimate_failure_replay_priority(self, record: Dict[str, Any]) -> Tuple[float, float, float]:
        attempts = int(record.get('attempts', 0))
        successes = int(record.get('successes', 0))
        failures = int(record.get('failures', 0))
        if attempts <= 0:
            return 0.0, 0.0, 0.0

        # Jeffreys 先验 Beta(0.5, 0.5) 适合 Bernoulli 路线成功/失败统计，
        # posterior tail probability 直接表示“真实成功率仍低于阈值”的置信度。
        alpha = 0.5 + successes
        beta = 0.5 + failures
        hard_count = 0
        for _ in range(self.failure_replay_posterior_samples):
            if self._failure_replay_rng.betavariate(alpha, beta) < self.failure_replay_success_threshold:
                hard_count += 1
        hard_prob = hard_count / float(self.failure_replay_posterior_samples)
        evidence = attempts / (attempts + self.failure_replay_evidence_smoothing)
        priority = hard_prob * evidence
        return hard_prob, evidence, priority

    def _trim_failure_replay_buffer(self) -> None:
        if len(self._failure_replay_records) <= self.failure_replay_buffer_size:
            return

        ranked = []
        for sig, record in self._failure_replay_records.items():
            hard_prob, evidence, priority = self._estimate_failure_replay_priority(record)
            ranked.append((
                priority,
                int(record.get('last_episode', -1)),
                sig,
                hard_prob,
                evidence,
            ))
        ranked.sort(key=lambda item: (item[0], item[1]))
        to_drop = len(ranked) - self.failure_replay_buffer_size
        for idx in range(max(0, to_drop)):
            _, _, sig, _, _ = ranked[idx]
            self._failure_replay_records.pop(sig, None)

    def _select_failure_replay_plan(self) -> Tuple[Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]], Dict[str, Any]] | Tuple[None, None]:
        if not self.failure_replay_enable or not self._failure_replay_records:
            return None, None

        candidates: List[Tuple[Tuple[Tuple[str, float, float, float, float], ...], Dict[str, Any], float, float, float]] = []
        for sig, record in self._failure_replay_records.items():
            hard_prob, evidence, priority = self._estimate_failure_replay_priority(record)
            if priority <= 1e-5:
                continue
            candidates.append((sig, record, hard_prob, evidence, priority))

        if not candidates:
            return None, None

        top_priority = max(item[4] for item in candidates)
        replay_prob = self.failure_replay_base_prob + (
            self.failure_replay_max_prob - self.failure_replay_base_prob
        ) * top_priority
        replay_prob = float(np.clip(replay_prob, self.failure_replay_base_prob, self.failure_replay_max_prob))
        if self._failure_replay_rng.random() > replay_prob:
            return None, {
                'replay_prob': replay_prob,
                'top_priority': top_priority,
                'selected': False,
            }

        weights = [item[4] for item in candidates]
        total = sum(weights)
        pick = self._failure_replay_rng.random() * total
        acc = 0.0
        chosen = candidates[-1]
        for item, weight in zip(candidates, weights):
            acc += weight
            if pick <= acc:
                chosen = item
                break

        sig, record, hard_prob, evidence, priority = chosen
        return self._materialize_route_plan(sig), {
            'replay_prob': replay_prob,
            'top_priority': top_priority,
            'selected': True,
            'hard_prob': hard_prob,
            'evidence': evidence,
            'priority': priority,
            'attempts': int(record.get('attempts', 0)),
            'successes': int(record.get('successes', 0)),
            'failures': int(record.get('failures', 0)),
        }

    def _record_episode_route_outcome(self, reason: str) -> None:
        if not self.failure_replay_enable or self.auto_reset_agents:
            return
        if not self._episode_route_plan_actual:
            return

        success_count = int(sum(self.episode_successes.values()))
        success_ratio = success_count / max(1, self._num_agents)
        is_success = success_ratio >= self.failure_replay_success_threshold
        signature = self._normalize_route_plan(self._episode_route_plan_actual)
        already_tracked = signature in self._failure_replay_records
        should_track = already_tracked or (not is_success) or (self._episode_route_plan_source == 'failure_replay')
        if not should_track:
            return

        record = self._failure_replay_records.get(signature)
        if record is None:
            record = {
                'plan': self._materialize_route_plan(signature),
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'created_episode': self._episode_counter,
            }
            self._failure_replay_records[signature] = record

        record['attempts'] = int(record.get('attempts', 0)) + 1
        if is_success:
            record['successes'] = int(record.get('successes', 0)) + 1
        else:
            record['failures'] = int(record.get('failures', 0)) + 1
        record['last_episode'] = self._episode_counter
        record['last_reason'] = str(reason)
        record['last_source'] = str(self._episode_route_plan_source)
        record['last_success_ratio'] = float(success_ratio)
        record['last_failed_agents'] = tuple(sorted(self.failed_agents))

        hard_prob, evidence, priority = self._estimate_failure_replay_priority(record)
        record['last_hard_prob'] = hard_prob
        record['last_evidence'] = evidence
        record['last_priority'] = priority
        self._trim_failure_replay_buffer()
        self.logger.info(
            '[failure_replay] update source=%s reason=%s success_ratio=%.2f attempts=%d succ=%d fail=%d hard_prob=%.2f priority=%.2f buffer=%d',
            self._episode_route_plan_source,
            reason,
            success_ratio,
            record['attempts'],
            record['successes'],
            record['failures'],
            hard_prob,
            priority,
            len(self._failure_replay_records),
        )

    def _define_observation_space(self):
        """定义观测空间"""
        configure_multi_agent_observation_space(self)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _empty_interaction_context(self, agent_id: str = '') -> Dict[str, Any]:
        return {
            'agent_id': str(agent_id),
            'mode': 'idle',
            'mode_id': 0.0,
            'in_conflict': 0.0,
            'has_token': 0.0,
            'should_yield': 0.0,
            'partner': '',
            'partner_dist': float('inf'),
            'closing_speed': 0.0,
            'ttc': float('inf'),
            'severity': 0.0,
            'turn_sign': 0.0,
            'front_min': float('inf'),
            'front_blocked_ratio': 0.0,
            'component_size': 1.0,
            'wait_steps': 0.0,
            'wait_age_norm': 0.0,
        }

    def get_agent_interaction_context(self, agent_id: str) -> Dict[str, Any]:
        ctx = self._interaction_contexts.get(agent_id)
        if ctx is None:
            return self._empty_interaction_context(agent_id)
        return dict(ctx)

    def _collect_interaction_snapshots(self, active_aids: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        aids = active_aids if active_aids is not None else [
            aid for aid in self.agent_ids if aid not in self.dones
        ]
        snapshots: Dict[str, Dict[str, float]] = {}
        for aid in aids:
            agent = self.agents[aid]
            pos = np.asarray(self.robot_positions.get(aid, np.zeros(2, dtype=np.float32)), dtype=np.float32)
            vel = np.asarray(self.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)), dtype=np.float32)
            yaw = float(getattr(agent, 'current_pose', {}).get('yaw', 0.0))
            forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
            sectors = agent._scan_sector_metrics()
            if sectors is None:
                front_min = float(getattr(agent, 'scan_max_range', 3.5))
                left_min = front_min
                right_min = front_min
            else:
                front_min = float(sectors.get('front_min', getattr(agent, 'scan_max_range', 3.5)))
                left_min = float(sectors.get('left_min', front_min))
                right_min = float(sectors.get('right_min', front_min))
            side_min = float(min(left_min, right_min))
            goal_pos = np.asarray(getattr(agent, 'goal_pos', pos), dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(goal_pos - pos))
            current_target = getattr(agent, 'current_subgoal', None)
            if current_target is None:
                current_target = tuple(goal_pos.tolist())
            target_vec = np.asarray(current_target, dtype=np.float32) - pos
            target_dist = float(np.linalg.norm(target_vec))
            target_dir = target_vec / max(target_dist, 1e-6)
            front_blocked_ratio = float(np.clip(
                (float(getattr(agent, 'subgoal_block_front_dist', 0.42)) - front_min)
                / max(float(getattr(agent, 'subgoal_block_front_dist', 0.42)), 1e-6),
                0.0,
                1.0,
            ))
            narrow_span_ref = max(
                0.45,
                2.4 * float(getattr(agent, 'subgoal_min_side_clearance', 0.20)),
            )
            corridor_narrow_ratio = float(np.clip(
                (narrow_span_ref - (left_min + right_min)) / max(narrow_span_ref, 1e-6),
                0.0,
                1.0,
            ))
            snapshots[aid] = {
                'x': float(pos[0]),
                'y': float(pos[1]),
                'vx': float(vel[0]),
                'vy': float(vel[1]),
                'yaw': yaw,
                'forward_x': float(forward[0]),
                'forward_y': float(forward[1]),
                'speed': float(max(getattr(agent, 'current_vel_x', 0.0), 0.0)),
                'front_min': front_min,
                'left_min': left_min,
                'right_min': right_min,
                'side_min': side_min,
                'front_blocked_ratio': front_blocked_ratio,
                'corridor_narrow_ratio': corridor_narrow_ratio,
                'dist_to_goal': dist_to_goal,
                'target_dir_x': float(target_dir[0]),
                'target_dir_y': float(target_dir[1]),
                'goal_closeness': float(np.clip(1.0 - dist_to_goal / 6.0, 0.0, 1.0)),
                'path_progress': float(getattr(agent, 'path_progress', 0.0)),
            }
        return snapshots

    def _update_interaction_contexts(self, active_aids: Optional[List[str]] = None) -> None:
        aids = active_aids if active_aids is not None else [
            aid for aid in self.agent_ids if aid not in self.dones
        ]
        aids = [aid for aid in aids if aid in self.agents]
        snapshots = self._collect_interaction_snapshots(aids)
        contexts = {aid: self._empty_interaction_context(aid) for aid in self.agent_ids}
        if not snapshots:
            self._interaction_contexts = contexts
            return

        adjacency: Dict[str, set[str]] = {aid: set() for aid in snapshots}
        edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for i, ai in enumerate(aids):
            if ai not in snapshots:
                continue
            snap_i = snapshots[ai]
            pos_i = np.array([snap_i['x'], snap_i['y']], dtype=np.float32)
            vel_i = np.array([snap_i['vx'], snap_i['vy']], dtype=np.float32)
            fwd_i = np.array([snap_i['forward_x'], snap_i['forward_y']], dtype=np.float32)
            for aj in aids[i + 1:]:
                if aj not in snapshots:
                    continue
                snap_j = snapshots[aj]
                pos_j = np.array([snap_j['x'], snap_j['y']], dtype=np.float32)
                vel_j = np.array([snap_j['vx'], snap_j['vy']], dtype=np.float32)
                fwd_j = np.array([snap_j['forward_x'], snap_j['forward_y']], dtype=np.float32)
                rel = pos_j - pos_i
                dist = float(np.linalg.norm(rel))
                if dist < 1e-6:
                    continue
                neighbor_dist = max(
                    float(getattr(self.agents[ai], 'dynamic_replan_neighbor_dist', 1.8)),
                    float(getattr(self.agents[aj], 'dynamic_replan_neighbor_dist', 1.8)),
            )
                if dist > neighbor_dist:
                    continue

                rel_unit = rel / max(dist, 1e-6)
                closing_speed = float(max(0.0, -np.dot(vel_j - vel_i, rel_unit)))
                ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')
                ttc_safe = max(
                    float(getattr(self.agents[ai], 'yielding_ttc', 2.4)),
                    float(getattr(self.agents[aj], 'yielding_ttc', 2.4)),
            )
                soft_dist = max(
                    float(getattr(self.agents[ai], 'yielding_soft_dist', 0.90)),
                    float(getattr(self.agents[aj], 'yielding_soft_dist', 0.90)),
            )
                my_toward = float(np.dot(fwd_i, rel_unit))
                other_toward = float(np.dot(fwd_j, -rel_unit))
                heading_opposition = float(np.dot(fwd_i, -fwd_j))
                ttc_risk = float(np.clip((ttc_safe - ttc) / max(ttc_safe, 1e-6), 0.0, 1.0)) if math.isfinite(ttc) else 0.0
                proximity_risk = float(np.clip((soft_dist - dist) / max(soft_dist, 1e-6), 0.0, 1.0))
                facing_score = float(np.clip(min(my_toward, other_toward), 0.0, 1.0))
                opposition_score = float(np.clip((heading_opposition + 0.20) / 1.20, 0.0, 1.0))
                narrow_score = max(
                    snap_i['corridor_narrow_ratio'],
                    snap_j['corridor_narrow_ratio'],
                    snap_i['front_blocked_ratio'],
                    snap_j['front_blocked_ratio'],
            )
                severity = max(
                    ttc_risk,
                    proximity_risk,
                    0.60 * min(facing_score, opposition_score) * narrow_score,
            )
                engage_dist = min(
                    neighbor_dist,
                    soft_dist + 0.10 + 0.22 * narrow_score,
            )
                projected_conflict = (
                    math.isfinite(ttc)
                    and closing_speed > 0.05
                    and ttc <= 0.85 * ttc_safe
                    and dist <= min(neighbor_dist, soft_dist + 0.55)
            )
                engage_now = (dist <= engage_dist) or projected_conflict
                head_on_like = (
                    my_toward > 0.15
                    and other_toward > 0.15
                    and heading_opposition > 0.05
            )
                if dist > soft_dist and not projected_conflict:
                    continue
                if not head_on_like and severity < 0.36:
                    continue
                if not engage_now:
                    continue

                bearing_i = float(math.atan2(rel[1], rel[0]))
                bearing_j = float(math.atan2(-rel[1], -rel[0]))
                yaw_err_i = self._wrap_angle(bearing_i - snap_i['yaw'])
                yaw_err_j = self._wrap_angle(bearing_j - snap_j['yaw'])
                edge = {
                    'agents': (ai, aj),
                    'dist': dist,
                    'closing_speed': closing_speed,
                    'ttc': ttc,
                    'severity': float(np.clip(severity, 0.0, 1.0)),
                    'engage_dist': float(engage_dist),
                    'turn_signs': {
                        ai: -1.0 if yaw_err_i > 0.0 else 1.0,
                        aj: -1.0 if yaw_err_j > 0.0 else 1.0,
                    },
                }
                edges[(ai, aj)] = edge
                adjacency[ai].add(aj)
                adjacency[aj].add(ai)

        visited = set()
        for aid in aids:
            if aid in visited or not adjacency.get(aid):
                continue
            stack = [aid]
            component: List[str] = []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                component.append(cur)
                stack.extend(adjacency[cur] - visited)

            comp_set = set(component)
            comp_edges = [
                edge for pair, edge in edges.items()
                if pair[0] in comp_set and pair[1] in comp_set
            ]
            if not comp_edges:
                continue

            scores: Dict[str, float] = {}
            for member in component:
                snap = snapshots[member]
                prev_ctx = self._interaction_contexts.get(member, self._empty_interaction_context(member))
                wait_norm = float(np.clip(self._interaction_wait_steps.get(member, 0) / 12.0, 0.0, 1.0))
                prev_token_bonus = 1.0 if (
                    float(prev_ctx.get('has_token', 0.0)) > 0.5
                    and float(prev_ctx.get('in_conflict', 0.0)) > 0.5
                ) else 0.0
                scores[member] = (
                    1.55 * snap['goal_closeness']
                    + 1.20 * max(snap['front_blocked_ratio'], snap['corridor_narrow_ratio'])
                    + 0.90 * prev_token_bonus
                    + 0.75 * wait_norm
                    + 0.20 * float(np.clip(snap['speed'] / max(self.agents[member].max_forward_vel, 1e-6), 0.0, 1.0))
                )
            for member in component:
                member_edges = [
                    edge for edge in comp_edges
                    if member in edge['agents']
                ]
                primary = max(member_edges, key=lambda edge: (edge['severity'], -edge['dist']))
                partner = primary['agents'][1] if primary['agents'][0] == member else primary['agents'][0]
                pair_members = [member, partner]
                owner = max(pair_members, key=lambda aid_: (scores[aid_], -snapshots[aid_]['dist_to_goal']))
                snap = snapshots[member]
                agent = self.agents[member]
                turn_sign = float(primary['turn_signs'][member])
                side_clear = max(snap['left_min'], snap['right_min'])
                hard_stop_dist = float(getattr(agent, 'yielding_hard_stop_dist', 0.30))
                yielding_ttc = float(getattr(agent, 'yielding_ttc', 2.4))
                side_clear_thresh = float(getattr(agent, 'subgoal_min_side_clearance', 0.20)) + 0.06
                front_blocked = snap['front_blocked_ratio'] > 0.35
                front_clear = (
                    snap['front_blocked_ratio'] < 0.18
                    and snap['front_min'] > float(getattr(agent, 'subgoal_block_front_dist', 0.42)) + 0.08
                )
                urgent_ttc = math.isfinite(primary['ttc']) and primary['ttc'] < (0.55 * yielding_ttc)
                if member == owner:
                    mode = 'go'
                elif (
                    primary['dist'] < hard_stop_dist
                    or (urgent_ttc and (front_blocked or side_clear < side_clear_thresh))
                ):
                    mode = 'backoff'
                elif front_clear or side_clear > side_clear_thresh:
                    mode = 'yield'
                else:
                    mode = 'wait'

                contexts[member] = {
                    'agent_id': member,
                    'mode': mode,
                    'mode_id': {
                        'idle': 0.0,
                        'go': 1.0,
                        'yield': 2.0,
                        'wait': 3.0,
                        'backoff': 4.0,
                    }[mode],
                    'in_conflict': 1.0,
                    'has_token': 1.0 if member == owner else 0.0,
                    'should_yield': 0.0 if member == owner else 1.0,
                    'partner': str(partner),
                    'partner_dist': float(primary['dist']),
                    'closing_speed': float(primary['closing_speed']),
                    'ttc': float(primary['ttc']),
                    'severity': float(primary['severity']),
                    'turn_sign': turn_sign,
                    'front_min': float(snap['front_min']),
                    'front_blocked_ratio': float(snap['front_blocked_ratio']),
                    'component_size': 2.0,
                    'wait_steps': 0.0,
                    'wait_age_norm': 0.0,
                }

        for aid in self.agent_ids:
            ctx = contexts.get(aid, self._empty_interaction_context(aid))
            if float(ctx.get('in_conflict', 0.0)) > 0.5 and str(ctx.get('mode', 'idle')) != 'go':
                self._interaction_wait_steps[aid] = min(self._interaction_wait_steps.get(aid, 0) + 1, 1000)
            else:
                self._interaction_wait_steps[aid] = 0
            ctx['wait_steps'] = float(self._interaction_wait_steps[aid])
            ctx['wait_age_norm'] = float(np.clip(self._interaction_wait_steps[aid] / 12.0, 0.0, 1.0))
            contexts[aid] = ctx

        self._interaction_contexts = contexts
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """重置环境"""
        self._episode_counter += 1
        self.current_step_count = 0
        self.dones = set()
        self.failed_agents = set()
        self.episode_successes  = {aid: 0 for aid in self.agent_ids}
        self.episode_collisions = {aid: 0 for aid in self.agent_ids}
        self.episode_stats = {
            'total_collisions': 0,
            'near_misses': 0,
            'successful_navigations': 0,
        }
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.logger.info('━━━ EPISODE RESET ━━━  seed=%s', seed)

        route_plan, route_plan_source, route_plan_meta = self._build_episode_route_plan()
        self._episode_route_plan_source = route_plan_source
        self._episode_route_plan_assigned = dict(route_plan)
        self._episode_route_plan_actual = {}
        if route_plan_source == 'failure_replay' and route_plan_meta:
            self.logger.info(
                '[reset] 启用 failure replay: replay_prob=%.2f hard_prob=%.2f priority=%.2f attempts=%d succ=%d fail=%d',
                float(route_plan_meta.get('replay_prob', 0.0)),
                float(route_plan_meta.get('hard_prob', 0.0)),
                float(route_plan_meta.get('priority', 0.0)),
                int(route_plan_meta.get('attempts', 0)),
                int(route_plan_meta.get('successes', 0)),
                int(route_plan_meta.get('failures', 0)),
            )
        elif route_plan_source == 'fixed_benchmark' and route_plan_meta:
            self.logger.info(
                '[reset] 启用固定基准路线: index=%d/%d',
                int(route_plan_meta.get('fixed_route_plan_index', 0)) + 1,
                int(route_plan_meta.get('fixed_route_plan_total', 0)),
            )
        elif route_plan:
            self.logger.info('[reset] 启用高冲突路线采样: mode=%s prob=%.2f',
                             self.high_conflict_mode, self.high_conflict_prob)

        obs_dict = {}
        info_dict = {}
        
        base_obs_dict = {}
        agent_starts = []  # list of (x, y)
        for aid, agent in self.agents.items():
            forced = route_plan.get(aid)
            obs, info = agent.reset(other_agent_starts=agent_starts, forced_start_goal=forced)
            base_obs_dict[aid] = obs
            spawn_pos = info.get('start_xy', None)
            if spawn_pos:
                agent_starts.append(spawn_pos)
                self.robot_positions[aid] = np.array(spawn_pos, dtype=np.float32)
            else:
                self.robot_positions[aid] = self._get_robot_position(agent)
            self.robot_velocities[aid] = self._get_robot_velocity(agent)
            info_dict[aid] = info
            self._episode_route_plan_actual[aid] = (
                tuple(getattr(agent, 'last_spawn_pos', info.get('start_xy', (0.0, 0.0)))),
                tuple(getattr(agent, 'goal_pos', (0.0, 0.0))),
            )

            goal = getattr(agent, 'goal_pos', ('?', '?'))
            pos  = self.robot_positions[aid]
            self.logger.info(
                '[reset] %-10s  spawn=(%.3f, %.3f)  goal=(%.3f, %.3f)  '
                'start_xy_from_info=%s',
                aid, pos[0], pos[1], goal[0], goal[1], spawn_pos,
            )

        # 打印所有 agent 间距（验证无重叠）
        n = self._num_agents
        dist_lines = []
        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = f'agent_{i}', f'agent_{j}'
                d = float(np.linalg.norm(self.robot_positions[ai] - self.robot_positions[aj]))
                in_range = d < self.communication_range
                dist_lines.append(f'  {ai}<->{aj}: {d:.3f}m  {"[COMM]" if in_range else "[OUT_OF_RANGE]"}')
        self.logger.info('[reset] 机器人间距:\n' + '\n'.join(dist_lines))

        # 随机化动态障碍物（每 episode 不同，增强避障训练）
        master = self.agents.get('agent_0')
        if master is not None and hasattr(master, 'randomize_obstacles'):
            all_positions = [self.robot_positions[aid].tolist() for aid in self.agent_ids]
            master.randomize_obstacles(all_positions)

        # 预填充通信历史
        if self.comm_mode == 'ros2_bridge':
            for buf in self._ros2_neighbor_bufs.values():
                buf.clear()
            self._broadcast_ros2_states()
        else:
            self._state_history.clear()
            for _ in range(self._state_history.maxlen):
                self._push_state_snapshot()

        self._interaction_wait_steps = {aid: 0 for aid in self.agent_ids}
        self._update_interaction_contexts(self.agent_ids)
        for aid, agent in self.agents.items():
            base_obs_dict[aid] = agent._get_obs()

        adjacency_matrix = self._build_communication_graph()
        self._last_adj_matrix = adjacency_matrix   # 供 RLlib 包装层直接读取，避免二次调用
        
        for aid in self.agent_ids:
            obs_dict[aid] = self._build_enhanced_observation(
                aid, base_obs_dict[aid], adjacency_matrix,
                all_base_obs=base_obs_dict,
                reset_flag=1.0,
            )

        for i, aid in enumerate(self.agent_ids):
            info_dict[aid]['adjacency_row'] = adjacency_matrix[i]
            info_dict[aid]['robot_positions'] = self.robot_positions.copy()
        
        return obs_dict, info_dict
    
    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """环境步进"""
        self.current_step_count += 1
        
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        truncated_dict = {}
        info_dict = {}
        
        # ── 阶段1: 并行发送动作（不等待，立即返回）──────────────────────
        # 对齐 marl_training 的并行范式：先群发，再统一推进时间
        for aid in self.agent_ids:
            if aid in action_dict and aid not in self.dones:
                self.agents[aid].apply_action(action_dict[aid])
            else:
                # 已完成或无动作的 agent：发布停止命令，保持原地
                self.agents[aid]._publish_vel(0.0, 0.0)

        # ── 阶段2: 统一推进仿真时间 + 刷新所有节点传感器 ──────────────
        self._wait_and_spin_all(0.1)

        sync_aids = list(self.agent_ids)
        active_for_sync = [aid for aid in self.agent_ids if aid not in self.dones]
        for aid in sync_aids:
            self.robot_positions[aid] = self._get_robot_position(self.agents[aid])
            self.robot_velocities[aid] = self._get_robot_velocity(self.agents[aid])
        self._update_interaction_contexts(active_for_sync)

        # ── 阶段3: 群收结果（所有 agent 统一在这里采集 obs/rew）────────
        raw_step_results: Dict[str, Dict[str, Any]] = {}
        for aid in self.agent_ids:
            if aid in action_dict and aid not in self.dones:
                obs, rew, done, truncated, info = self.agents[aid].get_step_result()

                # 更新位置和速度缓存
                pos = self._get_robot_position(self.agents[aid])
                vel = self._get_robot_velocity(self.agents[aid])
                self.robot_positions[aid] = pos
                self.robot_velocities[aid] = vel
                raw_step_results[aid] = {
                    'obs': obs,
                    'rew': rew,
                    'done': done,
                    'truncated': truncated,
                    'info': info,
                    'pos': pos,
                    'vel': vel,
                }
            elif aid not in self.dones:
                # 活跃 agent 未收到动作时，仍返回一条零奖励 transition，避免键集合错乱
                obs_dict[aid]  = self.agents[aid]._get_obs()
                rew_dict[aid]  = 0.0
                done_dict[aid] = False
                truncated_dict[aid] = False
                info_dict[aid] = {'status': 'no_action_received'}

        # ── 同步碰撞事件：先统一采样所有 agent 的结果，再对成对碰撞做双向补齐 ──
        active_aids = list(raw_step_results.keys())
        for i in range(len(active_aids)):
            for j in range(i + 1, len(active_aids)):
                ai = active_aids[i]
                aj = active_aids[j]
                info_i = raw_step_results[ai]['info']
                info_j = raw_step_results[aj]['info']
                i_collision = (info_i.get('event') == 'collision')
                j_collision = (info_j.get('event') == 'collision')
                if not (i_collision or j_collision):
                    continue

                pos_i = raw_step_results[ai]['pos']
                pos_j = raw_step_results[aj]['pos']
                pair_dist = float(np.linalg.norm(pos_i - pos_j))
                hard_sync_dist = max(
                    float(getattr(self.agents[ai], 'collision_hard_dist', 0.18)),
                    float(getattr(self.agents[aj], 'collision_hard_dist', 0.18)),
                ) + 0.02
                if pair_dist > hard_sync_dist:
                    continue

                if i_collision and not j_collision:
                    info_j['event'] = 'collision'
                    info_j['collision_source'] = 'pair_sync'
                    info_j['synced_collision_with'] = ai
                    raw_step_results[aj]['rew'] = float(raw_step_results[aj]['rew']) - float(
                        getattr(self.agents[aj], 'collision_penalty', 20.0)
                )
                elif j_collision and not i_collision:
                    info_i['event'] = 'collision'
                    info_i['collision_source'] = 'pair_sync'
                    info_i['synced_collision_with'] = aj
                    raw_step_results[ai]['rew'] = float(raw_step_results[ai]['rew']) - float(
                        getattr(self.agents[ai], 'collision_penalty', 20.0)
                )

        terminal_aids: List[str] = []
        for aid in active_aids:
            item = raw_step_results[aid]
            obs = item['obs']
            rew = item['rew']
            done = bool(item['done'])
            truncated = bool(item['truncated'])
            info = item['info']
            event = info.get('event', '')

            if truncated and not event:
                info['event'] = 'timeout'
                event = 'timeout'

            terminal = bool(
                info.get('need_reset', False)
                or done
                or truncated
                or event == 'goal'
                or (event == 'collision' and self.reset_on_collision_event)
            )
            item['terminal'] = terminal
            terminal_aids.append(aid) if terminal else None

            if event == 'goal':
                self.episode_successes[aid] += 1
                self.logger.info('[step %d] %s 到达目标 (本 episode 第 %d 次)',
                                 self.current_step_count, aid, self.episode_successes[aid])
            elif event == 'collision':
                self.episode_collisions[aid] += 1
                self.logger.info('[step %d] %s 碰撞 (本 episode 第 %d 次, src=%s)',
                                 self.current_step_count, aid, self.episode_collisions[aid],
                                 info.get('collision_source', 'unknown'))

            if not np.isfinite(rew):
                self.logger.warning('[step %d] %s reward=%.4f → 替换为 -0.01',
                                    self.current_step_count, aid, rew)
                rew = -0.01
            item['rew'] = rew
            obs_dict[aid] = obs
            rew_dict[aid] = rew
            done_dict[aid] = False
            truncated_dict[aid] = False
            info_dict[aid] = info

        pair_summary = compute_pairwise_local_rewards(self, info_dict) if info_dict else None

        if self.auto_reset_agents:
            fixed_starts = [
                self.robot_positions[aid].tolist()
                for aid in self.agent_ids
                if aid not in terminal_aids
            ]
            reserved_starts = list(fixed_starts)
            for aid in terminal_aids:
                forced = self._sample_conflict_route_for_respawn()
                new_obs, new_info = self.agents[aid].reset(
                    other_agent_starts=reserved_starts,
                    forced_start_goal=forced,
            )
                new_spawn = new_info.get('start_xy')
                if new_spawn:
                    spawn_arr = np.array(new_spawn, dtype=np.float32)
                    self.robot_positions[aid] = spawn_arr
                    reserved_starts.append(spawn_arr.tolist())
                self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
                obs_dict[aid] = new_obs
                info_dict[aid]['auto_reset'] = True
                info_dict[aid]['new_spawn'] = new_spawn
                # 连续任务流语义：同一 agent 在当前 multi-agent episode 内立即重生。
                # 对 RLlib 旧采样栈，若这里返回 done=True，又在同一步给出 reset 后的新 obs，
                # 会把两条轨迹混进同一个 truncated fragment，触发
                # "Batches sent to postprocessing must only contain steps from a single trajectory."
                # 因此 auto_reset 模式下必须保持 done=False，让 __all__ 仅由 timeout 控制。
                done_dict[aid] = False
                truncated_dict[aid] = False
        else:
            for aid in terminal_aids:
                self.dones.add(aid)
                if info_dict.get(aid, {}).get('event') == 'collision':
                    self.failed_agents.add(aid)
                self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
                done_dict[aid] = True
                truncated_dict[aid] = False

        if pair_summary is not None:
            for aid, pair_rew in pair_summary.rewards.items():
                if aid not in rew_dict:
                    continue
                rew_dict[aid] = float(rew_dict[aid]) + float(pair_rew)
                metrics = pair_summary.metrics.get(aid, {})
                info_dict.setdefault(aid, {}).update(metrics)
                info_dict[aid]['pair_event_reward'] = float(pair_rew)
                info_dict[aid]['r_pair'] = float(pair_rew)
                if float(metrics.get('pair_collision_penalty', 0.0)) < 0.0:
                    info_dict[aid]['collision_type'] = 'agent_collision'
                    info_dict[aid]['collision_responsibility'] = 'mutual_collision'
                elif info_dict[aid].get('event') == 'collision':
                    info_dict[aid]['collision_type'] = 'wall_collision'
                    info_dict[aid]['collision_responsibility'] = 'self_navigation_error'

        # ── 可选的小权重团队均值混合：默认关闭（lambda=1.0，仅保留 own+pair） ─────
        if rew_dict:
            team_mean_reward = float(np.mean(list(rew_dict.values())))
            lam = float(np.clip(self.team_reward_lambda, 0.0, 1.0))
            for aid in list(rew_dict.keys()):
                own_rew = float(rew_dict[aid])
                mixed_rew = lam * own_rew + (1.0 - lam) * team_mean_reward
                rew_dict[aid] = mixed_rew
                info_dict.setdefault(aid, {})['own_reward'] = own_rew
                info_dict[aid]['team_mean_reward'] = team_mean_reward
                info_dict[aid]['mixed_reward'] = mixed_rew
                info_dict[aid]['final_reward'] = mixed_rew

        # ── 构建通信图 & 更新状态历史 ────────────────────────────────────
        adjacency_matrix = self._build_communication_graph()
        if self.comm_mode == 'ros2_bridge':
            self._broadcast_ros2_states()
        else:
            self._push_state_snapshot()
        
        # ── 构建增强观测（邻居信息 + reset_flag + 全局状态）──────────────
        reset_flags = {aid: 0.0 for aid in obs_dict.keys()}
        for aid in list(obs_dict.keys()):
            if info_dict.get(aid, {}).get('auto_reset', False):
                reset_flags[aid] = 1.0

        enhanced_obs_dict = {}
        for aid in list(obs_dict.keys()):
            enhanced_obs = self._build_enhanced_observation(
                aid, obs_dict[aid], adjacency_matrix,
                all_base_obs=obs_dict,
                reset_flag=reset_flags[aid],
            )
            enhanced_obs_dict[aid] = enhanced_obs
        
        # ── 添加图信息到 info ─────────────────────────────────────────────
        for aid in list(enhanced_obs_dict.keys()):
            neighbors = np.where(adjacency_matrix[int(aid.split('_')[1])] > 0)[0]
            info_dict[aid]['neighbors']     = [f"agent_{n}" for n in neighbors]
            info_dict[aid]['num_neighbors'] = len(neighbors)

            if info_dict[aid].get('event') == 'collision':
                self.episode_stats['total_collisions'] += 1
            elif info_dict[aid].get('event') == 'goal':
                self.episode_stats['successful_navigations'] += 1
        
        # ── 周期性步骤日志 ────────────────────────────────────────────────
        log_this_step = (self.current_step_count % self.log_every_n_steps == 0)
        if log_this_step:
            pos_lines = []
            for aid in self.agent_ids:
                p = self.robot_positions[aid]
                v = self.robot_velocities[aid]
                r = rew_dict.get(aid, 0.0)
                pos_lines.append(
                    f'  {aid}: pos=({p[0]:6.3f},{p[1]:6.3f})  '
                    f'vel=({v[0]:5.3f},{v[1]:5.3f})  rew={r:7.4f}'
            )
            self.logger.info('[step %4d] 位置/速度/奖励:\n' + '\n'.join(pos_lines),
                             self.current_step_count)

        # ── 全局终止逻辑 ──────────────────────────────────────────────────
        timeout   = (self.current_step_count >= self.max_steps)
        all_done  = (len(self.dones) == self._num_agents)   # 仅 auto_reset=False 时有意义
        active_remaining = self._num_agents - len(self.dones)
        failed_count = len(self.failed_agents)

        if self.auto_reset_agents:
            # 连续任务流：只有超时结束
            episode_over = timeout
            reason = 'timeout' if timeout else ''
        else:
            cutoff_few_active = (
                self.min_active_agents_to_continue > 0
                and active_remaining < self.min_active_agents_to_continue
            )
            cutoff_too_many_failed = (
                self.max_failed_agents_before_cutoff > 0
                and failed_count >= self.max_failed_agents_before_cutoff
            )
            episode_over = all_done or timeout or cutoff_few_active or cutoff_too_many_failed
            if timeout:
                reason = 'timeout'
            elif all_done:
                reason = 'all_done'
            elif cutoff_too_many_failed:
                reason = 'too_many_failed'
            elif cutoff_few_active:
                reason = 'too_few_active'
            else:
                reason = ''

        done_dict["__all__"]      = episode_over
        truncated_dict["__all__"] = timeout

        if episode_over:
            self.logger.info(
                '━━━ EPISODE END (%s, step=%d) dones=%d/%d '
                'successes=%s collisions=%s active_remaining=%d failed=%d ━━━',
                reason, self.current_step_count,
                len(self.dones), self._num_agents,
                {aid: self.episode_successes[aid]  for aid in self.agent_ids},
                {aid: self.episode_collisions[aid] for aid in self.agent_ids},
                active_remaining, failed_count,
            )
            self._record_episode_route_outcome(reason)
            print(
                f"\n{'='*60}\n"
                f"🏁 Episode 结束 ({reason})\n"
                f"   步数: {self.current_step_count}/{self.max_steps}\n"
                f"   完成: {len(self.dones)}/{self._num_agents}\n"
                f"   活跃: {active_remaining}  失败: {failed_count}\n"
                f"{'='*60}\n"
            )
            for aid in list(enhanced_obs_dict.keys()):
                # episode 结束时所有 agent 统一标记终止
                done_dict[aid]      = True
                truncated_dict[aid] = timeout
                info_dict[aid]['episode_successes']  = self.episode_successes[aid]
                info_dict[aid]['episode_collisions'] = self.episode_collisions[aid]
                info_dict[aid]['episode_stats'] = self.episode_stats.copy()
        
        return enhanced_obs_dict, rew_dict, done_dict, truncated_dict, info_dict

    def _wait_and_spin_all(self, seconds: float) -> None:
        """
        统一推进仿真时间，同时刷新所有机器人节点的 ROS2 回调。
        对齐 marl_training 的并行范式：所有机器人发完动作后，统一等待这段时间，
        保证各机器人在同一时间窗内感知刷新（避免串行 step 带来的时间漂移）。
        """
        if not rclpy.ok():
            return
        ref_node = self.agents[self.agent_ids[0]].node
        # 等待时钟就绪
        while rclpy.ok() and ref_node.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(ref_node, timeout_sec=0.01)
        start_ns = ref_node.get_clock().now().nanoseconds
        delta_ns = seconds * 1e9
        while rclpy.ok():
            now_ns = ref_node.get_clock().now().nanoseconds
            if now_ns - start_ns >= delta_ns:
                break
            # 轮询所有机器人节点，保证各自的 scan/odom 回调都能执行
            for agent in self.agents.values():
                rclpy.spin_once(agent.node, timeout_sec=0.001)

    def _build_communication_graph(self) -> np.ndarray:
        """
        构建动态通信图
        返回: adjacency_matrix [n_agents, n_agents]
        """
        n = self._num_agents
        adjacency = np.zeros((n, n))
        
        positions = [self.robot_positions[f"agent_{i}"] for i in range(n)]
        active_mask = [f"agent_{i}" not in self.dones for i in range(n)]
        
        # 计算全量距离矩阵（用于日志）
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                if active_mask[i] and active_mask[j] and d < self.communication_range:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0

        for i in range(n):
            adjacency[i, i] = 1.0 if active_mask[i] else 0.0

        # ── 日志：每步都记录到文件，控制台只在前3步或 debug_comm=True 时打印 ──
        log_lines = [f'[graph] step={self.current_step_count:4d}  comm_range={self.communication_range}m']
        for i in range(n):
            adj_row   = '  '.join(f'{adjacency[i,j]:.0f}' for j in range(n))
            dist_row  = '  '.join(f'{dist_matrix[i,j]:5.2f}' for j in range(n))
            neighbors = [j for j in range(n) if adjacency[i, j] > 0 and j != i]
            pos_i     = positions[i]
            log_lines.append(
                f'  robot_{i} pos=({pos_i[0]:6.3f},{pos_i[1]:6.3f}) status={"active" if active_mask[i] else "done"}')
            log_lines.append(
                f'    adj=[{adj_row}]  dist=[{dist_row}]  neighbors={neighbors}')
        self.logger.debug('\n'.join(log_lines))

        # 控制台：只在前 3 步或 debug_comm=True 时打印（保留原有行为）
        if getattr(self, 'debug_comm', False) or self.current_step_count <= 3:
            print('\n'.join(log_lines))

        return adjacency

    # ------------------------------------------------------------------
    # 通信延迟建模辅助方法
    # ------------------------------------------------------------------

    def _setup_ros2_comm_bridge(self) -> None:
        """
        建立 ROS2 话题通信桥接（仅 comm_mode='ros2_bridge' 时调用）。

        架构（对标 MRS Lab / AWS DeepRacer Bridge 模式）：
          ┌─────────────────────────────────────────────────┐
          │  GNNMARLEnv（训练进程）                          │
          │  Bridge 节点（单个 rclpy 节点）                   │
          │    ├─ Publisher  /gnn_swarm/robot_0/state       │
          │    ├─ Publisher  /gnn_swarm/robot_N/state ──>  DDS
          │    ├─ Subscriber /gnn_swarm/robot_0/state <--  DDS
          │    └─ Subscriber /gnn_swarm/robot_N/state       │
          └─────────────────────────────────────────────────┘
        消息格式 [robot_id, x, y, vx, vy, send_wall_sec] 与
        robot_policy_node.py 完全一致，实现训练/部署代码路径统一。
        """
        if not rclpy.ok():
            rclpy.init()
        node_name = f'gnn_marl_bridge_{random.randint(0, 99999)}'
        self._bridge_node = rclpy.create_node(node_name)

        for aid in self.agent_ids:
            idx   = int(aid.split('_')[1])
            topic = f'/gnn_swarm/robot_{idx}/state'

            # 发布者：Bridge 代替机器人发布状态（训练中无真实机器人节点）
            pub = self._bridge_node.create_publisher(Float32MultiArray, topic, 10)
            self._ros2_state_pubs[aid] = pub

            # 订阅者：DDS 环回后写入缓冲区，供 _encode_neighbor_states 读取
            def _make_cb(sender_id: str):
                def _cb(msg: Float32MultiArray) -> None:
                    # data[5] = send_wall_sec（monotonic，非仿真时间）
                    self._ros2_neighbor_bufs[sender_id].append(
                        (time.monotonic(), list(msg.data))
                )
                return _cb

            self._bridge_node.create_subscription(
                Float32MultiArray, topic, _make_cb(aid), 10
            )

        print(f'[GNNMARLEnv] ROS2 Bridge 节点已启动: {node_name}')
        print(f'[GNNMARLEnv] 话题: /gnn_swarm/robot_{{0..{self._num_agents - 1}}}/state')

    def _broadcast_ros2_states(self) -> None:
        """
        将所有机器人当前状态发布到对应 ROS2 话题，然后 spin_once
        让 DDS 将消息投递到订阅者回调（写入 _ros2_neighbor_bufs）。

        时间戳使用 time.monotonic()（wall clock），不受 Gazebo use_sim_time 影响，
        与 _encode_neighbor_states 中的年龄计算保持一致。
        """
        if self._bridge_node is None:
            return
        now = time.monotonic()
        for aid in self.agent_ids:
            idx = int(aid.split('_')[1])
            pos = self.robot_positions[aid]
            vel = self.robot_velocities[aid]
            msg = Float32MultiArray()
            msg.data = [
                float(idx),
                float(pos[0]), float(pos[1]),
                float(vel[0]), float(vel[1]),
                float(now),
            ]
            self._ros2_state_pubs[aid].publish(msg)
        # 非阻塞 spin：让 DDS 处理刚发出的消息，触发订阅者回调
        rclpy.spin_once(self._bridge_node, timeout_sec=0.0)

    def _push_state_snapshot(self):
        """将当前全局真值状态存入历史队列（仅训练端使用，机器人端不存在此步骤）"""
        snapshot = {
            'positions': {aid: self.robot_positions[aid].copy() for aid in self.agent_ids},
            'velocities': {aid: self.robot_velocities[aid].copy() for aid in self.agent_ids},
        }
        self._state_history.append(snapshot)

    def _get_delayed_snapshot(self) -> Dict:
        """
        按 comm_latency_steps + 随机抖动 取历史快照，
        模拟机器人端通过 ROS2 topic 接收到的、已经产生延迟的消息。
        """
        if not self._state_history:
            return {
                'positions': {aid: self.robot_positions[aid].copy() for aid in self.agent_ids},
                'velocities': {aid: self.robot_velocities[aid].copy() for aid in self.agent_ids},
            }
        if self.comm_mode == 'centralized_oracle':
            # 理想模式：返回最新快照
            return self._state_history[-1]
        jitter = int(self.rng.integers(0, self.comm_jitter_steps + 1))
        delay  = self.comm_latency_steps + jitter
        idx = max(0, len(self._state_history) - 1 - delay)
        return self._state_history[idx]

    def _get_received_neighbor_samples(
        self,
        agent_id: str,
        adjacency_matrix: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, float, np.ndarray, np.ndarray]]:
        """
        统一获取通过通信链路可见的邻居状态。

        返回:
          [(neighbor_idx, dist, n_pos, n_vel), ...]
        """
        agent_idx = int(agent_id.split('_')[1])
        my_pos = self.robot_positions[agent_id]
        my_vel = self.robot_velocities[agent_id]

        if adjacency_matrix is not None:
            candidate_indices = [
                i for i in np.where(adjacency_matrix[agent_idx] > 0)[0]
                if i != agent_idx
            ]
        else:
            candidate_indices = [i for i in range(self._num_agents) if i != agent_idx]

        received: List[Tuple[int, float, np.ndarray, np.ndarray]] = []

        if self.comm_mode == 'ros2_bridge':
            now = time.monotonic()
            step_dt = 0.1
            jitter = int(self.rng.integers(0, self.comm_jitter_steps + 1))
            min_age = (self.comm_latency_steps + jitter) * step_dt

            for n_idx in candidate_indices:
                if self.rng.random() < self.comm_dropout_prob:
                    continue

                n_id = f'agent_{n_idx}'
                buf = self._ros2_neighbor_bufs[n_id]

                selected_data = None
                for _recv_t, data in reversed(buf):
                    if (now - data[5]) >= min_age:
                        selected_data = data
                        break

                if selected_data is None:
                    continue

                n_pos = np.array([selected_data[1], selected_data[2]], dtype=np.float32)
                n_vel = np.array([selected_data[3], selected_data[4]], dtype=np.float32)

                if self.comm_noise_std > 0.0:
                    n_pos = n_pos + self.rng.normal(0.0, self.comm_noise_std, 2).astype(np.float32)
                    n_vel = n_vel + self.rng.normal(0.0, self.comm_noise_std, 2).astype(np.float32)

                dist = float(np.linalg.norm(my_pos - n_pos))
                if dist <= self.communication_range:
                    received.append((n_idx, dist, n_pos, n_vel))
        else:
            snapshot = self._get_delayed_snapshot()

            for n_idx in candidate_indices:
                if self.comm_mode == 'decentralized' and self.rng.random() < self.comm_dropout_prob:
                    continue

                n_id = f'agent_{n_idx}'
                n_pos = snapshot['positions'][n_id].copy()
                n_vel = snapshot['velocities'][n_id].copy()

                if self.comm_mode == 'decentralized' and self.comm_noise_std > 0.0:
                    n_pos = n_pos + self.rng.normal(0.0, self.comm_noise_std, 2)
                    n_vel = n_vel + self.rng.normal(0.0, self.comm_noise_std, 2)

                dist = float(np.linalg.norm(my_pos - n_pos))
                if dist <= self.communication_range:
                    received.append((n_idx, dist, n_pos, n_vel))

        received.sort(key=lambda item: item[1])
        return received

    def _get_perceived_neighbor_samples(
        self,
        agent_id: str,
    ) -> List[Tuple[int, float, np.ndarray, np.ndarray]]:
        """
        获取“最近邻感知图”输入。

        与通信图不同，这里不使用丢包/延迟/通信半径来裁剪邻居，
        只基于当前局部感知半径选择最近邻。这样 Method3 的中层 actor
        学的是“附近谁会影响我的 maneuver 决策”，而不是“DDS 此刻有没有收到消息”。
        """
        my_pos = self.robot_positions[agent_id]
        perceived: List[Tuple[int, float, np.ndarray, np.ndarray]] = []
        max_range = float(max(
            0.5,
            getattr(self, 'interaction_neighbor_perception_range', self.communication_range),
        ))

        for n_idx in range(self._num_agents):
            n_id = f'agent_{n_idx}'
            if n_id == agent_id or n_id in self.dones:
                continue
            n_pos = np.asarray(self.robot_positions[n_id], dtype=np.float32).copy()
            n_vel = np.asarray(self.robot_velocities.get(n_id, np.zeros(2, dtype=np.float32)), dtype=np.float32).copy()
            dist = float(np.linalg.norm(my_pos - n_pos))
            if dist <= max_range:
                perceived.append((n_idx, dist, n_pos, n_vel))

        perceived.sort(key=lambda item: item[1])
        return perceived
    
    def _build_enhanced_observation(
        self,
        agent_id: str,
        base_obs: np.ndarray,
        adjacency_matrix: Optional[np.ndarray] = None,
        all_base_obs: Optional[Dict] = None,
        reset_flag: float = 0.0,
    ) -> np.ndarray:
        """
        构建增强观测

        组成:
        1. 基础观测: lidar + target + velocity (40维)      ← Actor 使用
        2. 邻居状态: K个近邻的相对状态 (K*5维)              ← Actor 使用
          3. reset_flag: 当前观测是否来自局部重置后的首帧       ← Actor 用于清空 LSTM state
          4. 全局状态: 所有智能体基础观测拼接 (N*40维)         ← 集中式 Critic 使用
              固定顺序 agent_0, ..., agent_{N-1}，Critic 输入维度始终一致
        """
        components = [base_obs]

        # 去中心化 Actor 的邻居感知
        if self.enable_neighbor_obs and adjacency_matrix is not None:
            neighbor_obs = self._encode_neighbor_states(agent_id, adjacency_matrix)
            components.append(neighbor_obs)

        components.append(np.array([reset_flag], dtype=np.float32))

        # MAPPO 集中式 Critic 的特权全局状态
        if all_base_obs is not None:
            global_state = np.concatenate([
                all_base_obs.get(f'agent_{i}', np.zeros(self.base_obs_dim, dtype=np.float32))
                for i in range(self._num_agents)
            ]).astype(np.float32)
        else:
            global_state = np.zeros(self.global_state_dim, dtype=np.float32)
        components.append(global_state)

        obs = np.concatenate(components)
        # ── NaN 守卫：Gazebo 初始化/reset 瞬间 odom 可能带 NaN 位置/速度
        # 用 0.0 替换 NaN/Inf，避免 MLP 输出 NaN → loss NaN → 参数崩溃
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"[GNNMARLEnv] enhanced obs shape mismatch for {agent_id}: got={obs.shape}, "
                f"expected={self.observation_space.shape}, base_obs_dim={self.base_obs_dim}, "
                f"neighbor_dim={self.neighbor_dim}, local_map_dim={self.local_map_dim}, "
                f"reset_flag_dim={self.reset_flag_dim}, global_state_dim={self.global_state_dim}"
            )
        return obs

    def _encode_neighbor_states(
        self,
        agent_id: str,
        adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """
        编码邻居状态。

        非 Method3:
          继续使用通信链路观测，支持 centralized_oracle / decentralized / ros2_bridge。

        Method3 (interaction_mode):
          改为“最近邻感知图”输入，不再受通信半径/丢包/延迟支配，
          只保留局部最近邻及其相对状态，避免 actor 学到“有没有收到消息”
          而不是“附近谁会影响我的 maneuver 决策”。

        通信链路模式说明：
          centralized_oracle
            直接读取当前真值，零延迟/零噪声，作为训练上界对比用。

          decentralized  （默认，适合快速训练）
            从 Python deque 取 latency+jitter 步前的快照，叠加丢包/高斯噪声。
            延迟单位为"步数"，与 Gazebo 仿真时间对齐。

          ros2_bridge  （最贴近部署，参考 MRS Lab / AWS DeepRacer Bridge 模式）
            从真实 ROS2 DDS 消息缓冲区读取，按消息中的 send_wall_sec 时间戳
            过滤年龄 >= comm_latency_steps × 0.1s 的消息，叠加丢包/高斯噪声。
            话题名 /gnn_swarm/robot_X/state 与 robot_policy_node.py 完全一致，
            训练代码路径 ≈ 部署代码路径，sim2real gap 最小。
        """
        my_pos        = self.robot_positions[agent_id]
        max_neighbors = min(self._num_agents - 1, 5)
        agent = self.agents.get(agent_id)
        interaction_tokens = bool(getattr(agent, 'action_mode', '') == 'interaction_mode')
        if interaction_tokens:
            received = self._get_perceived_neighbor_samples(agent_id)
        else:
            received = self._get_received_neighbor_samples(agent_id, adjacency_matrix=adjacency_matrix)

        # 编码最近 K 个邻居（不足则填零，保持向量维度固定）
        features_list: List[np.ndarray] = []
        my_yaw = float(getattr(agent, 'current_pose', {}).get('yaw', 0.0)) if agent is not None else 0.0
        if interaction_tokens:
            neighbor_range = float(getattr(self, 'interaction_neighbor_perception_range', self.communication_range))
        elif agent is not None:
            neighbor_range = float(getattr(agent, 'communication_range', self.communication_range))
        else:
            neighbor_range = float(self.communication_range)
        my_vel = self.robot_velocities[agent_id]
        yielding_ttc = float(getattr(agent, 'yielding_ttc', 2.4)) if agent is not None else 2.4
        for k in range(max_neighbors):
            if k < len(received):
                _, dist, n_pos, n_vel = received[k]
                if interaction_tokens:
                    feat = build_interaction_neighbor_token(
                        my_pos=my_pos,
                        my_vel=my_vel,
                        my_yaw=my_yaw,
                        neighbor_pos=n_pos,
                        neighbor_vel=n_vel,
                        perception_range=neighbor_range,
                        yielding_ttc=yielding_ttc,
                )
                else:
                    rel_pos = n_pos - my_pos
                    rel_vel = n_vel - my_vel
                    feat = np.array([
                        rel_pos[0], rel_pos[1],
                        rel_vel[0], rel_vel[1],
                        dist,
                    ], dtype=np.float32)
            else:
                feat = np.zeros(5, dtype=np.float32)
            features_list.append(feat)

        if not features_list:
            return np.zeros(0, dtype=np.float32)
        neighbor_vec = np.concatenate(features_list).astype(np.float32)

        # ── 日志：通信详情（每步写文件，前3步或 debug_comm=True 同时打印） ──
        source_tag = 'perception' if interaction_tokens else 'comm'
        comm_lines = [
            f'[{source_tag}] step={self.current_step_count:4d}  {agent_id}'
            f'  my_pos=({my_pos[0]:.3f},{my_pos[1]:.3f})'
            f'  received={len(received)}'
        ]
        for k, (n_idx, dist, n_pos, n_vel) in enumerate(received):
            comm_lines.append(
                f'    slot[{k}] neighbor=agent_{n_idx} dist={dist:.3f}m  '
                f'n_pos=({n_pos[0]:.3f},{n_pos[1]:.3f})  '
                f'rel_pos=({n_pos[0]-my_pos[0]:.3f},{n_pos[1]-my_pos[1]:.3f})'
            )
        if len(received) == 0:
            if interaction_tokens:
                comm_lines.append('    (无有效近邻: 超出局部感知半径，回退 ego-guidance)')
            else:
                comm_lines.append('    (无有效邻居: 范围外/丢包/延迟填零)')
        self.logger.debug('\n'.join(comm_lines))

        if getattr(self, 'debug_comm', False) or self.current_step_count <= 3:
            print('\n'.join(comm_lines))

        return neighbor_vec
    
    def _get_robot_position(self, agent: IndependentRobotEnv) -> np.ndarray:
        """获取机器人位置（从 odom 回调更新的 current_pose 读取）"""
        pose = getattr(agent, 'current_pose', None)
        if pose is not None:
            return np.array([pose['x'], pose['y']], dtype=np.float32)
        self.logger.warning(
            '[_get_robot_position] robot_%s: current_pose 不存在，返回零向量！'
            ' 检查 odom 话题是否发布。', getattr(agent, 'robot_id', '?'))
        return np.zeros(2, dtype=np.float32)
    
    def _get_robot_velocity(self, agent: IndependentRobotEnv) -> np.ndarray:
        """获取机器人全局坐标系下的速度（使不同机器人速度可比较）"""
        pose  = getattr(agent, 'current_pose', None)
        vel_x = getattr(agent, 'current_vel_x', None)
        if pose is not None and vel_x is not None:
            yaw = pose['yaw']
            vx = vel_x * np.cos(yaw)
            vy = vel_x * np.sin(yaw)
            return np.array([vx, vy], dtype=np.float32)
        self.logger.warning(
            '[_get_robot_velocity] robot_%s: current_pose 或 current_vel_x 不存在，返回零向量！',
            getattr(agent, 'robot_id', '?'))
        return np.zeros(2, dtype=np.float32)
    
    def close(self):
        """关闭环境"""
        for agent in self.agents.values():
            if hasattr(agent, 'close'):
                agent.close()
        if self._bridge_node is not None:
            self._bridge_node.destroy_node()
            self._bridge_node = None


def env_creator(env_config):
    return GNNMARLEnv(env_config)


class IndependentRobotEnv(gym.Env):
    def __init__(self, robot_id=0, map_number=3, max_episode_steps=500, use_random_mode=True,
                 communication_range=3.5,
                 interaction_neighbor_perception_range=0.0,
                 collision_ends_episode=True,
                 collision_hard_dist=0.05,
                 collision_persist_dist=0.15,
                 collision_persist_steps=3,
                 waypoint_reach_radius=0.8,
                 waypoint_distance_threshold=1.2,
                 waypoint_min_clearance_m=0.40,
                 use_voronoi_planner=False,
                 voronoi_min_clearance_m=0.35,
                 num_dynamic_obstacles=8, obs_speed=0.3,
                 rolling_lookahead_dist=0.4,
                 subgoal_block_front_dist=0.42,
                 subgoal_min_side_clearance=0.20,
                 subgoal_detour_forward_gain=0.55,
                 subgoal_detour_lateral_gain=0.75,
                 subgoal_detour_hold_steps=8,
                 subgoal_deadlock_front_dist=0.23,
                 subgoal_deadlock_speed_thresh=0.03,
                 subgoal_deadlock_steps=10,
                 replan_on_deadlock=True,
                 replan_cooldown_steps=25,
                 stall_global_replan_enable=False,
                 stall_global_replan_sec=5.0,
                 stall_replan_position_epsilon=0.18,
                 stall_replan_progress_epsilon=0.12,
                 dynamic_replan_neighbor_dist=1.8,
                 dynamic_replan_ttc=2.6,
                 dynamic_replan_block_radius=0.55,
                 obs_target_dist_clip=6.0,
                 obs_target_filter_alpha=0.35,
                 obs_target_max_step=0.45,
                 progress_reward_scale=0.0,
                 path_progress_reward_scale=0.0,
                 goal_progress_reward_scale=4.0,
                 goal_reward=20.0,
             collision_penalty=50.0,
                 time_penalty=0.01,
                 close_obstacle_penalty_scale=0.30,
                 close_obstacle_dist=0.55,
                 team_reward_lambda=1.0,
                 use_gazebo_collision=True,
                 lidar_collision_fallback=False,
                 obstacle_filter_range=2.0,
                 obstacle_filter_fov_deg=360.0,
                 obstacle_top_k=9,
                 angular_bins=0,
                 predictive_feature_enable=True,
                 predictive_horizon_sec=1.2,
                 predictive_social_ttc_safe=2.2,
                 predictive_front_ttc_safe=1.2,
                 predictive_min_sep=0.55,
                 predictive_social_range=2.5,
                 predictive_social_penalty_scale=0.17,
                 predictive_front_penalty_scale=0.16,
                 social_proximity_risk_scale=0.34,
                 neighbor_prediction_top_k=2,
                 gap_feature_enable=True,
                 yielding_enable=True,
                 yielding_soft_dist=0.90,
                 yielding_stop_dist=0.50,
                 yielding_hard_stop_dist=0.30,
                 yielding_ttc=2.4,
                 yielding_commit_steps=5,
                 replan_fixed_cost=0.03,
                 replan_freq_cost=0.012,
                 replan_time_cost=0.015,
                 replan_time_budget_sec=0.08,
                 replan_window_steps=80,
                 method3_reward_window_steps=8,
                 obstacle_motion_feature_enable=True,
                 obstacle_motion_top_k=3,
                 subgoal_progress_reward_scale=1.2,
                 detour_progress_relax=0.30,
                 risk_aware_forward_penalty_scale=0.28,
             safe_turn_reward_scale=0.15,
             head_on_avoidance_reward_scale=0.90,
                 risk_gate_soft=0.08,
                 risk_gate_hard=0.50,
                 avoidance_low_risk_scale=0.45,
                 navigation_high_risk_scale=0.80,
                 time_penalty_risk_relax=0.65,
                 reward_aggregation_overrides=None,
                 interaction_potential_overrides=None,
                 action_mode='interaction_mode',
                 observation_schema_mode='legacy',
                 include_tracking_target_block=True,
                 include_action_mask_block=True):
        """
        Args:
            collision_ends_episode: 如果为 True，碰撞会结束episode；
                                   如果为 False，碰撞只给惩罚但继续运行
                                   【重要】多机器人训练建议设为True以加快学习
        """
        super(IndependentRobotEnv, self).__init__()

        self.robot_id = robot_id
        self.collision_ends_episode = collision_ends_episode
        self.max_episode_steps = max_episode_steps
        self.use_random_mode = use_random_mode
        self.map_number = int(map_number)
        self.communication_range = float(communication_range)
        self.current_step = 0
        self.collision_hard_dist = float(collision_hard_dist)
        self.collision_persist_dist = float(collision_persist_dist)
        self.collision_persist_steps = int(max(1, collision_persist_steps))
        self.waypoint_reach_radius = float(waypoint_reach_radius)
        self.waypoint_distance_threshold = float(waypoint_distance_threshold)
        self.waypoint_min_clearance_m = float(waypoint_min_clearance_m)
        self.use_voronoi_planner = bool(use_voronoi_planner)
        self.voronoi_min_clearance_m = float(voronoi_min_clearance_m)
        self._close_obstacle_streak = 0

        # 1. 初始化 ROS 节点
        if not rclpy.ok():
            rclpy.init()

        # 强制开启 use_sim_time
        self.node = rclpy.create_node(
            f'gym_env_robot_{robot_id}_{random.randint(0, 100000)}',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        # 2. 命名空间设置
        self.ns = f"/tb3_{robot_id}"
        self.gazebo_model_name = f"tb3_{robot_id}"

        print(f"🤖 环境初始化: Namespace='{self.ns}', ModelName='{self.gazebo_model_name}'")

        # 3. 加载地图 & 初始化规划器
        self.map_image = None
        self.planner = None
        self._valid_spawn_points = []
        self._safe_spawn_mask = None
        self._load_map_data(self.map_number)

        if self.map_image is not None:
            map_data_inverted = 255 - self.map_image
            map_data_for_planner = np.flipud(map_data_inverted)

            self.planner = AStarPlanner(
                map_data_for_planner,
                resolution=self.map_resolution,
                origin=(self.map_origin[0], self.map_origin[1]),
                use_voronoi=self.use_voronoi_planner,
                voronoi_min_clearance_m=self.voronoi_min_clearance_m,
            )
            self.waypoint_extractor = WaypointExtractor(
                distance_threshold=self.waypoint_distance_threshold,
                min_clearance_m=self.waypoint_min_clearance_m,
            )
            print(f"✅ Robot {robot_id}: A*规划器初始化完成")

        self.global_waypoints = []
        self.current_waypoint_index = 0

        self.vis_namespace = f"robot_{robot_id}_waypoints"
        vis_topic = '/waypoint_markers'
        self.vis = WaypointVisualizer(self.node, topic_name=vis_topic)

        # 4. ROS 接口
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.vel_pub = self.node.create_publisher(Twist, f'{self.ns}/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, f'{self.ns}/scan', self._scan_callback, qos)
        self.odom_sub = self.node.create_subscription(Odometry, f'{self.ns}/odom', self._odom_callback, qos)
        self.bumper_sub = self.node.create_subscription(
            ContactsState,
            f'{self.ns}/bumper_states',
            self._bumper_callback,
            qos,
        )

        self.set_state_client = self.node.create_client(SetEntityState, '/set_entity_state')

        self.latest_scan = None
        self.current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_vel_x = 0.0
        self.current_vel_w = 0.0

        # 速度上限（TurtleBot3）
        self.max_forward_vel = 0.22
        self.max_reverse_vel = 0.12
        self.max_angular_vel = 1.2

        self.scan_history_len = 4
        self._scan_history: deque = deque(maxlen=self.scan_history_len)
        self.observation_schema_mode = str(observation_schema_mode).strip().lower() or 'legacy'
        self.include_tracking_target_block = bool(include_tracking_target_block)
        self.include_action_mask_block = bool(include_action_mask_block)
        configure_independent_env_action_observation_spaces(
            self,
            obstacle_top_k=obstacle_top_k,
            obstacle_filter_range=obstacle_filter_range,
            obstacle_filter_fov_deg=obstacle_filter_fov_deg,
            predictive_feature_enable=predictive_feature_enable,
            predictive_horizon_sec=predictive_horizon_sec,
            predictive_social_ttc_safe=predictive_social_ttc_safe,
            predictive_front_ttc_safe=predictive_front_ttc_safe,
            predictive_min_sep=predictive_min_sep,
            predictive_social_range=predictive_social_range,
            interaction_neighbor_perception_range=interaction_neighbor_perception_range,
            communication_range=self.communication_range,
            predictive_social_penalty_scale=predictive_social_penalty_scale,
            predictive_front_penalty_scale=predictive_front_penalty_scale,
            social_proximity_risk_scale=social_proximity_risk_scale,
            gap_feature_enable=gap_feature_enable,
            neighbor_prediction_top_k=neighbor_prediction_top_k,
            obstacle_motion_feature_enable=obstacle_motion_feature_enable,
            obstacle_motion_top_k=obstacle_motion_top_k,
            angular_bins=angular_bins,
        )
        self.yielding_enable = bool(yielding_enable)
        self.yielding_soft_dist = max(0.2, float(yielding_soft_dist))
        self.yielding_stop_dist = max(0.1, min(self.yielding_soft_dist, float(yielding_stop_dist)))
        self.yielding_hard_stop_dist = max(0.05, min(self.yielding_stop_dist, float(yielding_hard_stop_dist)))
        self.yielding_ttc = max(0.5, float(yielding_ttc))
        self.yielding_commit_steps = int(max(1, yielding_commit_steps))
        self.reward_aggregation_overrides = RewardAggregationOverrides(
            **dict(reward_aggregation_overrides or {})
        )
        self.interaction_potential_overrides = PotentialRewardConfig(
            **dict(interaction_potential_overrides or {})
        )
        self.observation_schema_spec = build_observation_schema_spec(self)
        # Closed-loop option tracking; policy output is applied every env step.
        self._active_option_name: str = 'go'
        self._active_option_start_step: int = 0
        self._active_option_duration_steps: int = 1
        self._option_phase: str = DetourPhase.DONE
        self._detour_guide_target_world: Optional[Tuple[float, float]] = None
        self._detour_suppress_rolling: bool = False
        self._detour_lateral_displacement: float = 0.0
        self._detour_active: bool = False
        self._detour_side: str = ''
        self._detour_phase: str = DetourPhase.DONE
        self._detour_targets: List[Tuple[float, float]] = []
        self._detour_target_index: int = 0
        self._detour_hold_remaining: int = 0
        self._detour_min_duration: int = 8
        self._detour_max_duration: int = 20
        self._detour_partner_id: str = ''
        self._detour_done: bool = False
        self._detour_interrupted: bool = False
        # Action mask from last feasibility evaluation
        self._last_action_mask: np.ndarray = np.ones(NUM_TRAINING_OPTIONS, dtype=np.int32)
        # Option outcome tracking (accumulated per lock window)
        self._option_start_snapshot: Dict[str, float] = {}
        self._last_option_outcome_terms = OptionOutcomeRewardTerms()
        # Path projection progress (arc-length along own planned path)
        self.path_projection_ema_alpha = 0.80
        self.path_projection_window_steps = 5
        self.proj_progress_threshold = 0.005
        self.goal_progress_threshold = 0.005
        self.local_progress_threshold = 0.005
        self.guide_progress_threshold = 0.005
        self._path_s: float = 0.0
        self._prev_path_s: float = 0.0
        self._path_s_raw: float = 0.0
        self._prev_path_s_raw: float = 0.0
        self._path_projection_valid: bool = False
        self._path_s_window: "deque[float]" = deque(maxlen=max(3, int(self.path_projection_window_steps)))
        self._closest_dist_to_path: float = 0.0
        self._cross_track_error: float = 0.0
        self._path_projection_progress_delta: float = 0.0
        self._path_projection_progress_window: float = 0.0
        self._guide_target_progress_delta: float = 0.0
        self._prev_dist_to_guide_target: float = float("inf")
        self._last_progress_source: str = "none"
        self._last_progress_source_id: float = 0.0
        self._last_progress_positive: bool = False
        self._last_goal_progress_delta: float = 0.0
        self._last_local_goal_progress_delta: float = 0.0
        self._last_cross_track_penalty: float = 0.0
        self._last_positive_path_projection_progress: float = 0.0
        self._last_negative_path_projection_progress: float = 0.0
        self._last_option_progress_reward: float = 0.0
        self._last_obstacle_risk_drop: float = 0.0
        self._last_risk_reduced: bool = False
        self._last_ttc_improvement: float = 0.0
        self._last_ttc_min: float = float("inf")
        self._prev_phi_goal: Optional[float] = None
        self._prev_phi_obs: Optional[float] = None
        self._prev_phi_agent: Optional[float] = None
        self._prev_phi_path: Optional[float] = None
        self._prev_corner_obstacle_potential: Optional[float] = None
        self._last_local_head_on_pass_event: float = 0.0
        self.control_dt = 0.1
        self._front_min_history: deque = deque(maxlen=self.scan_history_len)
        self._front_sector_dist_history: deque = deque(maxlen=self.scan_history_len)
        self._obstacle_cluster_history: deque = deque(maxlen=max(3, self.scan_history_len))
        self._last_predictive_metrics: Dict[str, float] = {
            'social_ttc': float('inf'),
            'social_min_sep': float('inf'),
            'social_risk': 0.0,
            'front_closing_speed': 0.0,
            'front_ttc': float('inf'),
            'front_risk': 0.0,
        }
        self._last_gap_metrics: Dict[str, float] = {
            'best_gap_angle': 0.0,
            'best_gap_width': 0.0,
            'best_gap_clearance': 0.0,
            'best_gap_score': 0.0,
        }
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0
        self._last_turn_sign = 0.0
        self._last_detour_direction = ''
        self._subgoal_switch_cost = 0.02
        self._subgoal_lateral_deadband = 0.10
        self._turn_flip_hysteresis = 0.18
        self._last_interaction_mode = 'idle'
        self._last_interaction_turn_sign = 0.0
        self._policy_interaction_mode = 'go'
        self._policy_interaction_action = 0
        self._effective_interaction_mode = 'go'
        self._executed_behavior_mode = 'nominal'
        self._cached_step_tracking_target: Optional[Tuple[float, float]] = None
        self._cached_step_tracking_mode = 'nominal'
        self._cached_step_tracking_step = -1
        self._last_nominal_subgoal: Optional[Tuple[float, float]] = None
        self._last_social_risk = 0.0
        self._last_front_blocked_ratio = 0.0
        self._last_stuck_score = 0.0
        self.replan_fixed_cost = max(0.0, float(replan_fixed_cost))
        self.replan_freq_cost = max(0.0, float(replan_freq_cost))
        self.replan_time_cost = max(0.0, float(replan_time_cost))
        self.replan_time_budget_sec = max(1e-3, float(replan_time_budget_sec))
        self.replan_window_steps = int(max(1, replan_window_steps))
        self._recent_replan_steps: deque[int] = deque()
        self.method3_reward_window_steps = int(max(2, method3_reward_window_steps))
        self._method3_credit_history: deque = deque(maxlen=self.method3_reward_window_steps + 1)
        self._last_replan_attempted = False
        self._last_replan_success = False
        self._last_replan_wall_time_sec = 0.0
        self.subgoal_progress_reward_scale = max(0.0, float(subgoal_progress_reward_scale))
        self.detour_progress_relax = float(np.clip(float(detour_progress_relax), 0.0, 1.0))
        self.risk_aware_forward_penalty_scale = max(0.0, float(risk_aware_forward_penalty_scale))
        self.safe_turn_reward_scale = max(0.0, float(safe_turn_reward_scale))
        self.head_on_avoidance_reward_scale = max(0.0, float(head_on_avoidance_reward_scale))
        self.risk_gate_soft = float(np.clip(float(risk_gate_soft), 0.0, 0.95))
        self.risk_gate_hard = max(self.risk_gate_soft + 1e-3, float(risk_gate_hard))
        self.avoidance_low_risk_scale = float(np.clip(float(avoidance_low_risk_scale), 0.0, 1.0))
        self.navigation_high_risk_scale = float(np.clip(float(navigation_high_risk_scale), 0.0, 1.0))
        self.time_penalty_risk_relax = float(np.clip(float(time_penalty_risk_relax), 0.0, 1.0))

        self.goal_pos = (0.0, 0.0)
        self.prev_dist_to_goal = None
        self.prev_dist_to_target = None
        self.prev_target_point = None
        self.prev_abs_target_angle = None

        self.num_dynamic_obstacles = max(0, min(int(num_dynamic_obstacles), 8))
        self.obs_speed = float(obs_speed)
        self.dynamic_obstacle_names: list = [f'dyn_obs_{i}' for i in range(8)]

        self.lookahead_dist = float(rolling_lookahead_dist)
        self.subgoal_block_front_dist = max(0.18, float(subgoal_block_front_dist))
        self.subgoal_min_side_clearance = max(0.10, float(subgoal_min_side_clearance))
        self.subgoal_detour_forward_gain = float(np.clip(float(subgoal_detour_forward_gain), 0.20, 1.20))
        self.subgoal_detour_lateral_gain = float(np.clip(float(subgoal_detour_lateral_gain), 0.20, 1.50))
        self.subgoal_detour_hold_steps = int(max(0, subgoal_detour_hold_steps))
        self.subgoal_deadlock_front_dist = max(0.20, float(subgoal_deadlock_front_dist))
        self.subgoal_deadlock_speed_thresh = max(0.0, float(subgoal_deadlock_speed_thresh))
        self.subgoal_deadlock_steps = int(max(1, subgoal_deadlock_steps))
        self.replan_on_deadlock = bool(replan_on_deadlock)
        self.replan_cooldown_steps = int(max(1, replan_cooldown_steps))
        self.stall_global_replan_enable = bool(stall_global_replan_enable)
        self.stall_global_replan_sec = max(1.0, float(stall_global_replan_sec))
        self.stall_replan_position_epsilon = max(0.02, float(stall_replan_position_epsilon))
        self.stall_replan_progress_epsilon = max(0.01, float(stall_replan_progress_epsilon))
        self.dynamic_replan_neighbor_dist = max(0.5, float(dynamic_replan_neighbor_dist))
        self.dynamic_replan_ttc = max(0.5, float(dynamic_replan_ttc))
        self.dynamic_replan_block_radius = max(0.10, float(dynamic_replan_block_radius))
        self.obs_target_dist_clip = max(0.5, float(obs_target_dist_clip))
        self.obs_target_filter_alpha = float(np.clip(float(obs_target_filter_alpha), 0.0, 1.0))
        self.obs_target_max_step = max(0.05, float(obs_target_max_step))
        self.progress_reward_scale = float(progress_reward_scale)
        self.path_progress_reward_scale = float(path_progress_reward_scale)
        self.goal_progress_reward_scale = float(goal_progress_reward_scale)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        self.close_obstacle_penalty_scale = float(close_obstacle_penalty_scale)
        self.close_obstacle_dist = float(close_obstacle_dist)
        self.team_reward_lambda = float(team_reward_lambda)
        self.robot_radius = 0.10

        self.side_close_dist = max(0.14, min(self.close_obstacle_dist * 0.55, self.subgoal_block_front_dist))
        self.side_close_penalty_scale = 0.35 * self.close_obstacle_penalty_scale
        self.corner_escape_front_dist = max(self.subgoal_block_front_dist, self.subgoal_deadlock_front_dist + 0.05)
        self.corner_escape_angle_thresh = 0.45
        self.corner_escape_speed_thresh = max(0.05, self.subgoal_deadlock_speed_thresh)
        self.corner_escape_commit_steps = max(
            4,
            self.subgoal_detour_hold_steps if self.subgoal_detour_hold_steps > 0 else 6,
        )
        self.corner_escape_forward_gain = 0.22
        self.corner_escape_lateral_gain = max(0.80, min(1.20, self.subgoal_detour_lateral_gain))
        self.use_gazebo_collision = bool(use_gazebo_collision)
        self.lidar_collision_fallback = bool(lidar_collision_fallback)

        # Gazebo 硬碰撞事件（ContactsState）
        self._gazebo_collision_active = False
        self._gazebo_collision_seen = False
        self._gazebo_collision_last_step = -10**9

        self._reset_path_tracking_state()
        self._reset_detour_primitive_state()
        self._subgoal_detour_hold = 0
        self._subgoal_detour_side = 0
        self._subgoal_deadlock_streak = 0
        self._corner_escape_hold_steps = 0
        self._corner_escape_turn_sign = 0.0
        self._next_replan_step = 0
        self._stall_elapsed_sec = 0.0
        self._stall_anchor_pos: Optional[np.ndarray] = None
        self._stall_anchor_path_progress = 0.0
        self._committed_subgoal_target = None
        self._committed_subgoal_mode = 'nominal'
        self._committed_subgoal_partner = ''
        self._committed_subgoal_hold_steps = 0
        self._last_subgoal_mode = 'nominal'
        self._last_interaction_info = {
            'mode': 'idle',
            'mode_id': 0.0,
            'in_conflict': 0.0,
            'has_token': 0.0,
            'should_yield': 0.0,
            'partner': '',
            'partner_dist': float('inf'),
            'closing_speed': 0.0,
            'ttc': float('inf'),
            'severity': 0.0,
            'turn_sign': 0.0,
            'front_min': float('inf'),
            'front_blocked_ratio': 0.0,
            'component_size': 1.0,
            'wait_steps': 0.0,
            'wait_age_norm': 0.0,
        }

    def _reset_path_tracking_state(self, clear_obs_target: bool = True):
        self.current_subgoal = None
        self.current_projection = None
        self.current_path_heading = 0.0
        self.path_progress = 0.0
        self.prev_path_progress = None
        self.current_lateral_error = 0.0
        self.current_waypoint_index = 0
        self._path_s = 0.0
        self._prev_path_s = 0.0
        self._path_s_raw = 0.0
        self._prev_path_s_raw = 0.0
        self._path_projection_valid = False
        self._path_s_window.clear()
        self._closest_dist_to_path = 0.0
        self._cross_track_error = 0.0
        self._path_projection_progress_delta = 0.0
        self._path_projection_progress_window = 0.0
        self._guide_target_progress_delta = 0.0
        self._prev_dist_to_guide_target = float("inf")
        self._last_progress_source = "none"
        self._last_progress_source_id = 0.0
        self._last_progress_positive = False
        self._last_goal_progress_delta = 0.0
        self._last_local_goal_progress_delta = 0.0
        self._last_cross_track_penalty = 0.0
        self._last_positive_path_projection_progress = 0.0
        self._last_negative_path_projection_progress = 0.0
        self._last_option_progress_reward = 0.0
        self._last_obstacle_risk_drop = 0.0
        self._last_risk_reduced = False
        self._last_ttc_improvement = 0.0
        self._last_ttc_min = float("inf")
        self._prev_phi_goal = None
        self._prev_phi_obs = None
        self._prev_phi_agent = None
        self._prev_phi_path = None
        self._prev_corner_obstacle_potential = None
        self._last_local_head_on_pass_event = 0.0
        self._reset_detour_primitive_state()
        if clear_obs_target:
            self._obs_target_state = None

    def _reset_detour_primitive_state(self) -> None:
        self._detour_active = False
        self._detour_side = ''
        self._detour_phase = DetourPhase.DONE
        self._detour_targets = []
        self._detour_target_index = 0
        self._detour_hold_remaining = 0
        self._detour_partner_id = ''
        self._detour_done = False
        self._detour_interrupted = False
        self._detour_suppress_rolling = False
        self._detour_guide_target_world = None
        self._detour_lateral_displacement = 0.0
        self._option_phase = DetourPhase.DONE

    def _get_body_relative_to_agent(self, agent_id: str) -> Optional[np.ndarray]:
        if not agent_id or not hasattr(self, 'parent_env'):
            return None
        other_pos = getattr(self.parent_env, 'robot_positions', {}).get(agent_id)
        if other_pos is None:
            return None
        rel = np.asarray(other_pos, dtype=np.float32) - np.array(
            [float(self.current_pose['x']), float(self.current_pose['y'])],
            dtype=np.float32,
        )
        return self._world_to_body(rel)

    def _start_committed_detour(
        self,
        option_mode: str,
        partner_id: str = '',
    ) -> None:
        side = 'left' if option_mode == 'detour_left' else 'right'
        sign = 1.0 if side == 'left' else -1.0
        body_targets = [
            (0.35, sign * 0.25),
            (0.70, sign * 0.35),
            (1.00, sign * 0.25),
        ]
        self._detour_targets = [
            tuple(self._body_to_world_point(px, py)) for px, py in body_targets
        ]
        self._detour_active = True
        self._detour_side = side
        self._detour_phase = DetourPhase.ENTER
        self._option_phase = self._detour_phase
        self._detour_target_index = 0
        self._detour_hold_remaining = int(self._detour_max_duration)
        self._detour_partner_id = str(partner_id)
        self._detour_done = False
        self._detour_interrupted = False
        self._detour_suppress_rolling = True
        self._detour_guide_target_world = tuple(self._detour_targets[0])
        self._detour_lateral_displacement = abs(float(body_targets[0][1]))

    def _advance_committed_detour(
        self,
        nominal_subgoal: Tuple[float, float],
        interaction_ctx: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._detour_active:
            return

        self._detour_hold_remaining = max(0, int(self._detour_hold_remaining) - 1)
        if self._detour_target_index < len(self._detour_targets):
            target = self._detour_targets[self._detour_target_index]
            dist = math.hypot(
                float(target[0]) - float(self.current_pose['x']),
                float(target[1]) - float(self.current_pose['y']),
            )
            if dist < 0.16 and self._detour_target_index < (len(self._detour_targets) - 1):
                self._detour_target_index += 1

        if self._detour_target_index <= 0:
            self._detour_phase = DetourPhase.ENTER
        elif self._detour_target_index == 1:
            self._detour_phase = DetourPhase.PASS
        else:
            self._detour_phase = DetourPhase.MERGE
        self._option_phase = self._detour_phase

        target_idx = min(self._detour_target_index, max(len(self._detour_targets) - 1, 0))
        if self._detour_targets:
            current_target = self._detour_targets[target_idx]
            if self._detour_phase == DetourPhase.MERGE:
                merge_target = (
                    0.65 * float(current_target[0]) + 0.35 * float(nominal_subgoal[0]),
                    0.65 * float(current_target[1]) + 0.35 * float(nominal_subgoal[1]),
            )
                current_target = merge_target
            self._detour_guide_target_world = tuple(current_target)

        partner_rel = self._get_body_relative_to_agent(self._detour_partner_id)
        partner_passed = bool(partner_rel is not None and float(partner_rel[0]) < -0.05)
        social_risk = float(getattr(self, '_last_social_risk', 0.0))
        ttc_min = float(getattr(self, '_last_ttc_min', float('inf')))
        risk_cleared = social_risk < 0.15 and (
            not math.isfinite(ttc_min) or ttc_min > 2.2
        )
        min_elapsed = int(self.current_step) - int(getattr(self, '_active_option_start_step', 0))
        reached_last_target = (
            self._detour_target_index >= (len(self._detour_targets) - 1)
            and self._detour_targets
            and math.hypot(
                float(self._detour_guide_target_world[0]) - float(self.current_pose['x']),
                float(self._detour_guide_target_world[1]) - float(self.current_pose['y']),
            ) < 0.18
        )
        if min_elapsed >= self._detour_min_duration and (partner_passed or risk_cleared or reached_last_target):
            self._detour_done = True
            self._detour_active = False
            self._detour_suppress_rolling = False
            self._detour_phase = DetourPhase.DONE
            self._option_phase = self._detour_phase

        if self._detour_hold_remaining <= 0:
            self._detour_done = True
            self._detour_active = False
            self._detour_suppress_rolling = False
            self._detour_phase = DetourPhase.DONE
            self._option_phase = self._detour_phase

    def _should_interrupt_detour(self) -> bool:
        if not self._detour_active:
            return False
        sectors = self._scan_sector_metrics()
        front_min = float(sectors.get('front_min', self.scan_max_range))
        front_left_min = float(sectors.get('front_left_min', self.scan_max_range))
        front_right_min = float(sectors.get('front_right_min', self.scan_max_range))
        side_min = float(sectors.get(
            'left_min' if self._detour_side == 'left' else 'right_min',
            self.scan_max_range,
        ))
        imminent_collision = front_min < (self.collision_hard_dist + 0.05)
        corner_blocked = min(front_left_min, front_right_min) < 0.12
        side_blocked = side_min < 0.10
        return bool(imminent_collision or corner_blocked or side_blocked)

    def _clear_committed_subgoal(self):
        self._committed_subgoal_target = None
        self._committed_subgoal_mode = 'nominal'
        self._committed_subgoal_partner = ''
        self._committed_subgoal_hold_steps = 0

    def _commit_subgoal(
        self,
        target: Tuple[float, float],
        mode: str,
        hold_steps: int,
        partner: str = '',
    ) -> Tuple[float, float]:
        target_arr = np.array([float(target[0]), float(target[1])], dtype=np.float32)
        prev_target = getattr(self, '_committed_subgoal_target', None)
        prev_mode = str(getattr(self, '_committed_subgoal_mode', 'nominal'))
        if prev_target is not None and prev_mode == str(mode):
            prev_arr = np.array(prev_target, dtype=np.float32)
            delta = target_arr - prev_arr
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 1e-6:
                alpha = 0.25 if str(mode) in {'wait', 'backoff'} else 0.35
                target_arr = prev_arr + alpha * delta
        self._committed_subgoal_target = (float(target_arr[0]), float(target_arr[1]))
        self._committed_subgoal_mode = str(mode)
        self._committed_subgoal_partner = str(partner)
        self._committed_subgoal_hold_steps = int(max(0, hold_steps))
        return self._committed_subgoal_target

    def _get_committed_subgoal(self, mode: str, partner: str = '') -> Optional[Tuple[float, float]]:
        if self._committed_subgoal_target is None:
            return None
        if str(mode) != self._committed_subgoal_mode:
            return None
        if partner and str(partner) != self._committed_subgoal_partner:
            return None
        if self._committed_subgoal_hold_steps <= 0:
            return None
        if str(mode) in {'wait', 'backoff', 'detour', 'gap_detour'}:
            front_min = float(getattr(self, '_front_min', self.scan_max_range))
            social_risk = float(getattr(self, '_last_social_risk', 0.0))
            if str(mode) in {'wait', 'backoff'} and front_min > max(self.yielding_soft_dist, 0.75) and social_risk < 0.10:
                self._clear_committed_subgoal()
                return None

        dist = math.hypot(
            float(self._committed_subgoal_target[0]) - float(self.current_pose['x']),
            float(self._committed_subgoal_target[1]) - float(self.current_pose['y']),
        )
        if dist < 0.10 or dist > max(1.8, 3.0 * max(0.25, self.lookahead_dist)):
            self._clear_committed_subgoal()
            return None

        self._committed_subgoal_hold_steps = max(0, self._committed_subgoal_hold_steps - 1)
        return tuple(self._committed_subgoal_target)

    def _reset_stall_replan_tracker(
        self,
        anchor_current: bool = False,
        arc_progress: Optional[float] = None,
    ):
        self._stall_elapsed_sec = 0.0
        if anchor_current:
            self._stall_anchor_pos = np.array(
                [float(self.current_pose['x']), float(self.current_pose['y'])],
                dtype=np.float32,
            )
            if arc_progress is None:
                arc_progress = self.path_progress
            self._stall_anchor_path_progress = float(arc_progress)
        else:
            self._stall_anchor_pos = None
            self._stall_anchor_path_progress = 0.0

    def _should_force_global_replan_from_stall(
        self,
        front_min: float,
        arc_progress: Optional[float] = None,
    ) -> bool:
        if not self.stall_global_replan_enable or self.planner is None:
            return False

        blocker_summary = self._collect_replan_blocked_points()

        blocked = bool(
            float(front_min) < max(self.subgoal_block_front_dist, self.subgoal_deadlock_front_dist)
            or self._last_subgoal_mode in {'blocked_nominal', 'deadlock', 'corner_escape', 'yield', 'wait', 'backoff'}
            or bool(blocker_summary.get('path_blocker_ahead', False))
        )
        current_progress = float(self.path_progress if arc_progress is None else arc_progress)
        if not blocked:
            self._reset_stall_replan_tracker(anchor_current=True, arc_progress=current_progress)
            return False

        current_pos = np.array(
            [float(self.current_pose['x']), float(self.current_pose['y'])],
            dtype=np.float32,
        )
        if self._stall_anchor_pos is None:
            self._reset_stall_replan_tracker(anchor_current=True, arc_progress=current_progress)
            return False

        low_motion = abs(float(getattr(self, 'current_vel_x', 0.0))) < max(
            0.05,
            self.subgoal_deadlock_speed_thresh + 0.01,
        )
        moved_dist = float(np.linalg.norm(current_pos - self._stall_anchor_pos))
        progress_gain = max(0.0, current_progress - float(self._stall_anchor_path_progress))

        if low_motion and moved_dist < self.stall_replan_position_epsilon and progress_gain < self.stall_replan_progress_epsilon:
            self._stall_elapsed_sec += self.control_dt
        else:
            self._reset_stall_replan_tracker(anchor_current=True, arc_progress=current_progress)
            return False

        return self._stall_elapsed_sec >= self.stall_global_replan_sec

    def _collect_replan_blocked_points(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'points': [],
            'path_blocker_ahead': False,
            'closest_dist': float('inf'),
        }
        if not hasattr(self, 'parent_env'):
            return summary

        my_aid = f"agent_{self.robot_id}"
        my_pos = np.array([float(self.current_pose['x']), float(self.current_pose['y'])], dtype=np.float32)
        my_yaw = float(self.current_pose['yaw'])
        my_vel = np.array([
            self.current_vel_x * math.cos(my_yaw),
            self.current_vel_x * math.sin(my_yaw),
        ], dtype=np.float32)
        path_points = self.global_waypoints if self.global_waypoints else [self.goal_pos]
        my_proj = None
        if len(path_points) >= 2:
            tracking_kwargs = {}
            if self.current_projection is not None:
                tracking_kwargs = {
                    'anchor_segment_index': int(np.clip(
                        self.current_waypoint_index - 1,
                        0,
                        len(path_points) - 2,
                )),
                    'anchor_arc_progress': float(self.path_progress),
                }
            try:
                my_proj = PathTrackingUtils.get_path_projection(
                    (float(my_pos[0]), float(my_pos[1])),
                    path_points,
                    **tracking_kwargs,
            )
            except Exception:
                my_proj = None

        ahead_arc_horizon = max(self.dynamic_replan_neighbor_dist + 1.2, 3.0)
        lateral_thresh = max(0.55, 2.2 * self.dynamic_replan_block_radius)
        front_band_y = max(0.75, 2.2 * self.dynamic_replan_block_radius)

        for aid, pos in self.parent_env.robot_positions.items():
            if aid == my_aid:
                continue

            other_pos = np.asarray(pos, dtype=np.float32)
            rel = other_pos - my_pos
            dist = float(np.linalg.norm(rel))
            if dist < 1e-6:
                continue

            body_rel = self._world_to_body(rel)
            neighbor_vel = np.asarray(
                self.parent_env.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            other_done = aid in getattr(self.parent_env, 'dones', set())
            static_like = other_done or float(np.linalg.norm(neighbor_vel)) < 0.03

            rel_unit = rel / max(dist, 1e-6)
            closing_speed = float(-np.dot(neighbor_vel - my_vel, rel_unit))
            ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')

            local_front = (
                float(body_rel[0]) > -0.08
                and abs(float(body_rel[1])) <= front_band_y
            )
            local_block = local_front and dist <= max(self.dynamic_replan_neighbor_dist, 2.4)
            projected_conflict = (
                local_front
                and math.isfinite(ttc)
                and closing_speed > 0.05
                and ttc <= self.dynamic_replan_ttc
                and dist <= max(self.dynamic_replan_neighbor_dist, 2.8)
            )

            path_ahead = False
            if my_proj is not None and len(path_points) >= 2:
                try:
                    other_proj = PathTrackingUtils.get_path_projection(
                        (float(other_pos[0]), float(other_pos[1])),
                        path_points,
                        anchor_segment_index=int(np.clip(
                            int(my_proj.get('segment_index', 0)),
                            0,
                            len(path_points) - 2,
                        )),
                        anchor_arc_progress=float(my_proj.get('arc_progress', 0.0)),
                        backtrack_segments=1,
                        forward_search_segments=20,
                        max_backward_progress=0.15,
                        relocalize_lateral_error=max(1.4, lateral_thresh + 0.5),
                    )
                    arc_delta = float(other_proj.get('arc_progress', 0.0)) - float(my_proj.get('arc_progress', 0.0))
                    path_ahead = (
                        0.05 <= arc_delta <= ahead_arc_horizon
                        and float(other_proj.get('lateral_error', float('inf'))) <= lateral_thresh
                    )
                except Exception:
                    path_ahead = False

            should_block = path_ahead or projected_conflict or local_block
            if static_like:
                should_block = path_ahead or local_block
            if not should_block:
                continue

            summary['points'].append((float(other_pos[0]), float(other_pos[1])))
            if not static_like:
                predict_h = min(self.dynamic_replan_ttc, 0.8)
                summary['points'].append((
                    float(other_pos[0] + neighbor_vel[0] * predict_h),
                    float(other_pos[1] + neighbor_vel[1] * predict_h),
                ))
            summary['path_blocker_ahead'] = bool(summary['path_blocker_ahead'] or path_ahead or local_block)
            summary['closest_dist'] = min(float(summary['closest_dist']), dist)

        return summary

    def _load_map_data(self, map_number):
        try:
            try:
                pkg_path = get_package_share_directory('start_rl_environment_tb3')
            except Exception:
                _repo_root = Path(__file__).resolve().parents[3]
                pkg_path = str(_repo_root / 'install' / 'start_rl_environment_tb3'
                               / 'share' / 'start_rl_environment_tb3')
                if not os.path.isdir(pkg_path):
                    raise FileNotFoundError(f'备用路径不存在: {pkg_path}')
            map_mapping = {1: 'map1', 2: 'map2', 3: 'corridor_swap', 4: 'intersection', 5: 'warehouse_aisles', 6: 'interaction_hub'}
            map_name = map_mapping.get(map_number, 'map1')
            yaml_path = os.path.join(pkg_path, 'maps', f'{map_name}.yaml')
            if not os.path.exists(yaml_path):
                return

            with open(yaml_path, 'r') as f:
                map_info = yaml.safe_load(f)
            self.map_resolution = map_info['resolution']
            self.map_origin = map_info['origin']

            image_filename = map_info['image']
            image_path = os.path.join(os.path.dirname(yaml_path), image_filename)
            with Image.open(image_path) as img:
                self.map_image = np.array(img.convert('L'))
                self.map_height, self.map_width = self.map_image.shape

            from scipy.ndimage import binary_erosion as _be
            SPAWN_CLEARANCE_PX = 8
            free_mask = (self.map_image > 200)
            kernel3 = np.ones((3, 3), dtype=bool)
            eroded = _be(free_mask, structure=kernel3, iterations=SPAWN_CLEARANCE_PX)

            world_path = os.path.join(pkg_path, 'worlds', f'{map_name}.world')
            bounds = self._parse_world_bounds(world_path)
            if bounds is not None:
                bx0, bx1, by0, by1 = bounds
                bpx_lo = max(0, int((bx0 - self.map_origin[0]) / self.map_resolution))
                bpx_hi = min(self.map_width - 1, int((bx1 - self.map_origin[0]) / self.map_resolution))
                bpy_lo = max(0, int(self.map_height - 1 - (by1 - self.map_origin[1]) / self.map_resolution))
                bpy_hi = min(self.map_height - 1, int(self.map_height - 1 - (by0 - self.map_origin[1]) / self.map_resolution))
                bmask = np.zeros_like(eroded)
                bmask[bpy_lo:bpy_hi + 1, bpx_lo:bpx_hi + 1] = True
                eroded = eroded & bmask

            self._safe_spawn_mask = eroded.astype(bool)

            vy, vx = np.where(eroded)
            vwx = (self.map_origin[0] + vx * self.map_resolution).tolist()
            vwy = (self.map_origin[1] + (self.map_height - 1 - vy) * self.map_resolution).tolist()
            self._valid_spawn_points = list(zip(vwx, vwy))
            print(f"✅ Map {map_number}: 预计算安全 spawn 点 {len(self._valid_spawn_points)} 个"
                  f" (clearance={SPAWN_CLEARANCE_PX * self.map_resolution:.2f}m)")
        except Exception as e:
            print(f"❌ 加载地图失败: {e}")
            self.map_image = None
            self._valid_spawn_points = []
            self._safe_spawn_mask = None

    def _world_to_map_pixel(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        if self.map_image is None:
            return None
        px = int(round((wx - self.map_origin[0]) / self.map_resolution))
        py = int(round(self.map_height - 1 - (wy - self.map_origin[1]) / self.map_resolution))
        if px < 0 or px >= self.map_width or py < 0 or py >= self.map_height:
            return None
        return px, py

    def _is_safe_spawn_point(
        self,
        wx: float,
        wy: float,
        exclude: Optional[Tuple[float, float]] = None,
        other_agents: Optional[List[Tuple[float, float]]] = None,
        min_agent_sep: float = 1.5,
        min_goal_sep: float = 2.0,
    ) -> bool:
        pix = self._world_to_map_pixel(wx, wy)
        if pix is None:
            return False

        if self._safe_spawn_mask is not None:
            px, py = pix
            if not bool(self._safe_spawn_mask[py, px]):
                return False

        if exclude and math.hypot(wx - exclude[0], wy - exclude[1]) < min_goal_sep:
            return False

        if other_agents and any(
            math.hypot(wx - ax, wy - ay) < min_agent_sep
            for ax, ay in other_agents
        ):
            return False

        dyn_spawns = self._DYN_OBS_SPAWNS.get(self.map_number, [])
        if any(math.hypot(wx - sx, wy - sy) < 1.0 for sx, sy in dyn_spawns):
            return False

        return True

    def _is_valid_start_goal_pair(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        other_agent_starts: Optional[List[Tuple[float, float]]] = None,
        min_agent_sep: float = 1.5,
        min_goal_sep: float = 2.0,
    ) -> bool:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if not self._is_safe_spawn_point(sx, sy, other_agents=other_agent_starts, min_agent_sep=min_agent_sep):
            return False
        if not self._is_safe_spawn_point(gx, gy, exclude=(sx, sy), min_goal_sep=min_goal_sep):
            return False
        if self.planner and not self.planner.plan((sx, sy), (gx, gy)):
            return False
        return True

    def _scan_callback(self, msg):
        self.latest_scan = msg

    def _odom_callback(self, msg):
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_pose['yaw'] = math.atan2(siny_cosp, cosy_cosp)
        self.current_vel_x = msg.twist.twist.linear.x
        self.current_vel_w = msg.twist.twist.angular.z

    def _bumper_callback(self, msg: ContactsState):
        has_contact = len(getattr(msg, 'states', [])) > 0
        if has_contact:
            self._gazebo_collision_active = True
            self._gazebo_collision_seen = True
            self._gazebo_collision_last_step = self.current_step

    def _scan_sector_metrics(self):
        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            return {
                'min_dist': self.scan_max_range,
                'front_min': self.scan_max_range,
                'left_min': self.scan_max_range,
                'right_min': self.scan_max_range,
                'rear_min': self.scan_max_range,
                'front_left_min': self.scan_max_range,
                'front_center_min': self.scan_max_range,
                'front_right_min': self.scan_max_range,
                'clearance_asymmetry': 0.0,
            }

        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.scan_max_range, posinf=self.scan_max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        valid = ranges[(ranges > self.scan_valid_min)]
        min_dist = float(valid.min()) if valid.size else self.scan_max_range

        n = len(ranges)
        if n < 8:
            return {
                'min_dist': min_dist,
                'front_min': min_dist,
                'left_min': min_dist,
                'right_min': min_dist,
                'rear_min': min_dist,
                'front_left_min': min_dist,
                'front_center_min': min_dist,
                'front_right_min': min_dist,
                'clearance_asymmetry': 0.0,
            }

        front_idx = np.r_[0:max(1, n // 18), n - max(1, n // 18):n]
        left_idx = np.arange(n // 6, n // 3)
        right_idx = np.arange(2 * n // 3, 5 * n // 6)
        rear_idx = np.arange(8 * n // 18, 10 * n // 18)
        front_left_idx = np.arange(1 * n // 18, 3 * n // 18)
        front_center_idx = np.r_[0:max(1, n // 36), n - max(1, n // 36):n]
        front_right_idx = np.arange(15 * n // 18, 17 * n // 18)

        def _sector_min(idx):
            vals = ranges[idx]
            vals = vals[(vals > self.scan_valid_min)]
            return float(vals.min()) if vals.size else self.scan_max_range

        left_min = _sector_min(left_idx)
        right_min = _sector_min(right_idx)
        return {
            'min_dist': min_dist,
            'front_min': _sector_min(front_idx),
            'left_min': left_min,
            'right_min': right_min,
            'rear_min': _sector_min(rear_idx),
            'front_left_min': _sector_min(front_left_idx),
            'front_center_min': _sector_min(front_center_idx),
            'front_right_min': _sector_min(front_right_idx),
            'clearance_asymmetry': float(left_min - right_min),
        }

    # def _extract_filtered_scan_features(self, ranges: np.ndarray) -> np.ndarray:
    #     # 将变长障碍点集编码为定长 Top-K 特征，避免 obs 维度变化
    #     feat = np.zeros((self.obstacle_top_k, self.obstacle_point_feature_dim), dtype=np.float32)
    #     n = int(ranges.size)
    #     if n <= 0:
    #         return feat.reshape(-1)

    #     angle_min = -math.pi
    #     angle_inc = (2.0 * math.pi) / max(1, n)
    #     if self.latest_scan is not None and getattr(self.latest_scan, 'ranges', None) is not None:
    #         if len(self.latest_scan.ranges) == n:
    #             a_min = float(getattr(self.latest_scan, 'angle_min', angle_min))
    #             a_inc = float(getattr(self.latest_scan, 'angle_increment', angle_inc))
    #             if math.isfinite(a_min):
    #                 angle_min = a_min
    #             if math.isfinite(a_inc) and abs(a_inc) > 1e-6:
    #                 angle_inc = a_inc

    #     idx = np.arange(n, dtype=np.float32)
    #     angles = angle_min + idx * angle_inc
    #     angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

    #     clipped = np.clip(ranges.astype(np.float32), 0.0, self.scan_max_range)
    #     valid = (clipped > self.scan_valid_min) & (clipped <= self.obstacle_filter_range)

    #     if self.obstacle_filter_fov_deg < 359.9:
    #         half_fov = math.radians(self.obstacle_filter_fov_deg) * 0.5
    #         valid &= (np.abs(angles) <= half_fov)

    #     valid_idx = np.where(valid)[0]
    #     if valid_idx.size == 0:
    #         return feat.reshape(-1)

    #     ordered = valid_idx[np.argsort(clipped[valid_idx])]
    #     picked = ordered[:self.obstacle_top_k]

    #     d = clipped[picked]
    #     theta = angles[picked]
    #     denom = max(self.obstacle_filter_range, 1e-6)
    #     count = int(picked.size)

    #     feat[:count, 0] = (d * np.cos(theta)) / denom
    #     feat[:count, 1] = (d * np.sin(theta)) / denom
    #     feat[:count, 2] = d / denom
    #     feat[:count, 3] = 1.0
    #     return feat.reshape(-1)

    def _compute_angular_scan_dists(self, ranges: np.ndarray) -> np.ndarray:
        """Full 360° fixed angular binning for 1D CNN scan encoder."""
        n = int(ranges.size)
        nbins = max(8, int(getattr(self, 'angular_bins', self.obstacle_top_k)))
        dists = np.full(nbins, self.scan_max_range, dtype=np.float32)
        if n <= 0:
            return dists
        bin_edges = np.linspace(0, n, nbins + 1, dtype=int)
        for i in range(nbins):
            lo = int(bin_edges[i])
            hi = int(np.clip(bin_edges[i + 1], lo + 1, n))
            sector = ranges[lo:hi]
            valid = sector[sector > self.scan_valid_min]
            dists[i] = float(valid.min()) if valid.size > 0 else self.scan_max_range
        return dists

    def _extract_filtered_scan_features(self, ranges: np.ndarray) -> np.ndarray:
        """Normalized angular proximity features for 1D CNN input."""
        dists = self._compute_angular_scan_dists(ranges)
        return np.maximum(
            0.0,
            (self.obstacle_filter_range - dists) / max(self.obstacle_filter_range, 1e-6),
        ).astype(np.float32)

    def _compute_front_sector_min_dists(self, ranges: np.ndarray) -> np.ndarray:
        """Legacy alias — delegates to angular scan dists (full 360°)."""
        return self._compute_angular_scan_dists(ranges)

    def _front_sector_center_angle(self, sector_idx: int) -> float:
        nbins = max(1, int(getattr(self, 'angular_bins', self.obstacle_top_k)))
        if nbins <= 1:
            return 0.0
        return -math.pi + (float(sector_idx) + 0.5) * (2.0 * math.pi / float(nbins))

    def _scan_angles(self, n: int) -> np.ndarray:
        angle_min = -math.pi
        angle_inc = (2.0 * math.pi) / max(1, n)
        if self.latest_scan is not None and getattr(self.latest_scan, 'ranges', None) is not None:
            if len(self.latest_scan.ranges) == n:
                a_min = float(getattr(self.latest_scan, 'angle_min', angle_min))
                a_inc = float(getattr(self.latest_scan, 'angle_increment', angle_inc))
            if math.isfinite(a_min):
                    angle_min = a_min
            if math.isfinite(a_inc) and abs(a_inc) > 1e-6:
                    angle_inc = a_inc
        idx = np.arange(n, dtype=np.float32)
        angles = angle_min + idx * angle_inc
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    def _extract_scan_clusters(self, ranges: np.ndarray) -> List[Dict[str, float]]:
        arr = np.asarray(ranges, dtype=np.float32)
        n = int(arr.size)
        if n <= 0:
            return []

        arr = np.nan_to_num(arr, nan=self.scan_max_range, posinf=self.scan_max_range, neginf=0.0)
        arr = np.clip(arr, 0.0, self.scan_max_range)
        angles = self._scan_angles(n)

        valid = (arr > self.scan_valid_min) & (arr <= self.obstacle_filter_range)
        if self.obstacle_filter_fov_deg < 359.9:
            half_fov = math.radians(self.obstacle_filter_fov_deg) * 0.5
            valid &= (np.abs(angles) <= half_fov)

        if not np.any(valid):
            return []

        clusters: List[List[int]] = []
        current: List[int] = []
        spatial_gap = 0.22
        max_angle_gap = math.radians(8.0)
        valid_idx = np.where(valid)[0]
        for idx in valid_idx:
            if not current:
                current = [int(idx)]
                continue
            prev = current[-1]
            prev_pt = np.array([
                float(arr[prev] * math.cos(float(angles[prev]))),
                float(arr[prev] * math.sin(float(angles[prev]))),
            ], dtype=np.float32)
            cur_pt = np.array([
                float(arr[idx] * math.cos(float(angles[idx]))),
                float(arr[idx] * math.sin(float(angles[idx]))),
            ], dtype=np.float32)
            if (
                int(idx) == prev + 1
                and abs(float(angles[idx] - angles[prev])) <= max_angle_gap
                and float(np.linalg.norm(cur_pt - prev_pt)) <= spatial_gap
            ):
                current.append(int(idx))
            else:
                clusters.append(current)
                current = [int(idx)]
        if current:
            clusters.append(current)

        extracted: List[Dict[str, float]] = []
        for ids in clusters:
            if len(ids) < 2:
                continue
            pts = np.stack([
                np.array([
                    float(arr[i] * math.cos(float(angles[i]))),
                    float(arr[i] * math.sin(float(angles[i]))),
                ], dtype=np.float32)
                for i in ids
            ], axis=0)
            centroid = pts.mean(axis=0)
            dists = np.linalg.norm(pts, axis=1)
            min_dist = float(np.min(dists))
            mean_dist = float(np.mean(dists))
            angle = float(math.atan2(float(centroid[1]), float(centroid[0])))
            span = float(abs(float(angles[ids[-1]]) - float(angles[ids[0]])))
            extracted.append({
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "angle": angle,
                "min_dist": min_dist,
                "mean_dist": mean_dist,
                "span": span,
                "size": float(len(ids)),
            })

        extracted.sort(key=lambda c: c["min_dist"])
        return extracted[: max(self.obstacle_motion_top_k * 3, self.obstacle_motion_top_k)]

    def _match_previous_cluster(
        self,
        cluster: Dict[str, float],
        prev_clusters: List[Dict[str, float]],
    ) -> Optional[Dict[str, float]]:
        best = None
        best_score = float("inf")
        cur = np.array([
            float(cluster.get("xw", cluster["x"])),
            float(cluster.get("yw", cluster["y"])),
        ], dtype=np.float32)
        for prev in prev_clusters:
            prev_xy = np.array([
                float(prev.get("xw", prev["x"])),
                float(prev.get("yw", prev["y"])),
            ], dtype=np.float32)
            dist = float(np.linalg.norm(cur - prev_xy))
            if dist > 0.85:
                continue
            angle_gap = abs(self._wrap_angle(float(cluster["angle"]) - float(prev["angle"])))
            score = dist + 0.25 * angle_gap
            if score < best_score:
                best_score = score
                best = prev
        return best

    def _world_to_body(self, vec_xy: np.ndarray) -> np.ndarray:
        yaw = float(self.current_pose['yaw'])
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([
            c * float(vec_xy[0]) + s * float(vec_xy[1]),
            -s * float(vec_xy[0]) + c * float(vec_xy[1]),
        ], dtype=np.float32)

    def _body_to_world_vec(self, vec_xy: np.ndarray) -> np.ndarray:
        yaw = float(self.current_pose['yaw'])
        c = math.cos(yaw)
        s = math.sin(yaw)
        return np.array([
            c * float(vec_xy[0]) - s * float(vec_xy[1]),
            s * float(vec_xy[0]) + c * float(vec_xy[1]),
        ], dtype=np.float32)

    def _agent_rank(self, aid: str) -> int:
        try:
            return int(str(aid).split('_')[-1])
        except Exception:
            return int(self.robot_id)

    def _get_current_sector_dists(self) -> np.ndarray:
        nbins = int(getattr(self, 'angular_bins', self.obstacle_top_k))
        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            return np.full(nbins, self.scan_max_range, dtype=np.float32)
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        return self._compute_angular_scan_dists(ranges)

    def _compute_gap_metrics(self, sector_dists: np.ndarray, nominal_target_angle: float = 0.0) -> Dict[str, float]:
        sector_arr = np.asarray(sector_dists, dtype=np.float32)
        if sector_arr.size <= 0:
            return {
                'best_gap_angle': 0.0,
                'best_gap_width': 0.0,
                'best_gap_clearance': 0.0,
                'best_gap_score': 0.0,
                'best_sector_idx': -1,
            }

        open_thresh = max(self.subgoal_min_side_clearance, min(self.close_obstacle_dist, 0.45))
        denom = max(self.obstacle_filter_range - open_thresh, 1e-6)
        open_mask = sector_arr >= open_thresh
        best = None

        for idx, dist in enumerate(sector_arr):
            angle = self._front_sector_center_angle(idx)
            clearance_norm = float(np.clip((float(dist) - open_thresh) / denom, 0.0, 1.0))
            if clearance_norm <= 1e-5:
                continue

            left = idx
            right = idx
            while left - 1 >= 0 and open_mask[left - 1]:
                left -= 1
            while right + 1 < sector_arr.size and open_mask[right + 1]:
                right += 1

            width_norm = float(right - left + 1) / float(max(1, sector_arr.size))
            heading_delta = self._wrap_angle(angle - float(nominal_target_angle))
            heading_align = 1.0 - float(np.clip(abs(heading_delta) / (0.5 * math.pi), 0.0, 1.0))
            forwardness = float(np.clip(math.cos(angle), 0.0, 1.0))
            score = (
                0.50 * clearance_norm
                + 0.25 * width_norm
                + 0.15 * heading_align
                + 0.10 * forwardness
            )
            cand = {
                'best_gap_angle': float(angle),
                'best_gap_width': float(width_norm),
                'best_gap_clearance': float(np.clip(float(dist) / max(self.obstacle_filter_range, 1e-6), 0.0, 1.0)),
                'best_gap_score': float(score),
                'best_sector_idx': int(idx),
            }
            if best is None or cand['best_gap_score'] > best['best_gap_score']:
                best = cand

        if best is None:
            return {
                'best_gap_angle': 0.0,
                'best_gap_width': 0.0,
                'best_gap_clearance': 0.0,
                'best_gap_score': 0.0,
                'best_sector_idx': -1,
            }
        return best

    def _get_gap_features(self, sector_dists: np.ndarray, nominal_target_angle: float = 0.0) -> np.ndarray:
        if not self.gap_feature_enable:
            self._last_gap_metrics = {
                'best_gap_angle': 0.0,
                'best_gap_width': 0.0,
                'best_gap_clearance': 0.0,
                'best_gap_score': 0.0,
            }
            return np.zeros(0, dtype=np.float32)

        metrics = self._compute_gap_metrics(sector_dists, nominal_target_angle=nominal_target_angle)
        self._last_gap_metrics = {
            'best_gap_angle': float(metrics['best_gap_angle']),
            'best_gap_width': float(metrics['best_gap_width']),
            'best_gap_clearance': float(metrics['best_gap_clearance']),
            'best_gap_score': float(metrics['best_gap_score']),
        }
        return np.array([
            float(np.clip(metrics['best_gap_angle'] / (0.5 * math.pi), -1.0, 1.0)),
            float(metrics['best_gap_width']),
            float(metrics['best_gap_clearance']),
        ], dtype=np.float32)

    def _get_interaction_context(self) -> Dict[str, Any]:
        if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'get_agent_interaction_context'):
            ctx = self.parent_env.get_agent_interaction_context(f"agent_{self.robot_id}")
            if ctx:
                self._last_interaction_info = dict(ctx)
                return self._last_interaction_info
        self._last_interaction_info = {
            'mode': 'idle',
            'mode_id': 0.0,
            'in_conflict': 0.0,
            'has_token': 0.0,
            'should_yield': 0.0,
            'partner': '',
            'partner_dist': float('inf'),
            'closing_speed': 0.0,
            'ttc': float('inf'),
            'severity': 0.0,
            'turn_sign': 0.0,
            'front_min': float('inf'),
            'front_blocked_ratio': 0.0,
            'component_size': 1.0,
            'wait_steps': 0.0,
            'wait_age_norm': 0.0,
        }
        return self._last_interaction_info

    def _get_interaction_obs_features(self) -> np.ndarray:
        ctx = self._get_interaction_context()
        severity = float(np.clip(float(ctx.get('severity', 0.0)), 0.0, 1.0))
        turn_sign = float(np.clip(float(ctx.get('turn_sign', 0.0)), -1.0, 1.0))
        return np.array([
            float(ctx.get('in_conflict', 0.0)),
            float(ctx.get('has_token', 0.0)),
            float(ctx.get('should_yield', 0.0)),
            severity,
            turn_sign,
            float(np.clip(float(ctx.get('wait_age_norm', 0.0)), 0.0, 1.0)),
            float(np.clip(float(ctx.get('front_blocked_ratio', 0.0)), 0.0, 1.0)),
        ], dtype=np.float32)

    def _resolve_interaction_mode(
        self,
        requested_mode: str,
        interaction_ctx: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply the policy-selected interaction option immediately."""
        _ = interaction_ctx
        requested_mode = str(requested_mode)
        if requested_mode != getattr(self, "_active_option_name", "go"):
            self._active_option_start_step = int(self.current_step)
            self._option_start_snapshot = self._capture_option_start_snapshot()
        self._active_option_name = requested_mode
        self._active_option_duration_steps = max(
            1,
            int(self.current_step) - int(getattr(self, "_active_option_start_step", 0)) + 1,
        )
        return requested_mode

    def _capture_option_start_snapshot(self) -> Dict[str, float]:
        """Snapshot key metrics at the start of a new option lock window."""
        px = float(self.current_pose["x"])
        py = float(self.current_pose["y"])
        snapshot_sectors = self._scan_sector_metrics()
        front_min = float(snapshot_sectors.get("front_min", 3.5))
        front_safe = 0.40
        current_target = getattr(self, "_cached_step_tracking_target", None)
        if current_target is None:
            current_target = getattr(self, "current_subgoal", None)
        if current_target is None:
            current_target = self.goal_pos
        current_target = (float(current_target[0]), float(current_target[1]))
        social_snapshot = self._compute_social_risk_summary()
        interaction_snapshot = dict(getattr(self, "_last_interaction_info", {}) or {})
        front_blocked_ratio = float(
            np.clip(
                (self.subgoal_block_front_dist - front_min) / max(self.subgoal_block_front_dist, 1e-6),
                0.0,
                1.0,
            )
        )
        return {
            "path_progress": float(getattr(self, "path_progress", 0.0)),
            "dist_to_target": math.hypot(current_target[0] - px, current_target[1] - py),
            "dist_to_goal": math.hypot(
                float(self.goal_pos[0]) - px,
                float(self.goal_pos[1]) - py,
            ),
            "front_min": front_min,
            "front_blocked_ratio": front_blocked_ratio,
            "front_obstacle_risk": float(np.clip((front_safe - max(0.0, front_min)) / front_safe, 0.0, 1.0)),
            "interaction_social_risk": float(social_snapshot.get("social_risk", getattr(self, "_last_social_risk", 0.0))),
            "ttc_min": float(interaction_snapshot.get("ttc", getattr(self, "_last_ttc_min", float("inf")))),
            "pos_x": float(self.current_pose["x"]),
            "pos_y": float(self.current_pose["y"]),
            "path_s": float(getattr(self, "_path_s", 0.0)),
        }

    def _update_path_projection(self) -> None:
        """Update arc-length projection of robot onto its own planned path.

        Stores _path_s, _closest_dist_to_path, _cross_track_error,
        _path_projection_progress_delta, and a smoothed window.
        """
        px, py = float(self.current_pose["x"]), float(self.current_pose["y"])
        path = self.global_waypoints if self.global_waypoints and len(self.global_waypoints) >= 2 else None
        self._prev_path_s = float(getattr(self, "_path_s", 0.0))
        self._prev_path_s_raw = float(getattr(self, "_path_s_raw", 0.0))

        if path is None:
            self._path_projection_valid = False
            self._closest_dist_to_path = 0.0
            self._cross_track_error = 0.0
            self._path_projection_progress_delta = 0.0
            self._path_projection_progress_window = 0.0
            self._path_s_window.clear()
        else:
            s_proj, closest_dist, seg_idx, proj_xy = project_point_to_polyline_arclength(
                (px, py), path
            )
            self._path_projection_valid = True
            self._path_s_raw = float(s_proj)
            if not self._path_s_window:
                self._path_s = float(s_proj)
            else:
                alpha = float(np.clip(self.path_projection_ema_alpha, 0.0, 0.98))
                self._path_s = float(alpha * self._prev_path_s + (1.0 - alpha) * float(s_proj))
            self._closest_dist_to_path = float(closest_dist)
            self._cross_track_error = float(closest_dist)
            self._path_s_window.append(float(self._path_s))
            if len(self._path_s_window) >= self._path_s_window.maxlen:
                oldest = float(self._path_s_window[0])
                self._path_projection_progress_window = float(self._path_s - oldest)
            else:
                self._path_projection_progress_window = 0.0
            self._path_projection_progress_delta = float(self._path_s - self._prev_path_s)

        # Guide target progress (distance to current guide target, only for closed-loop options)
        guide_target = getattr(self, "_detour_guide_target_world", None)
        guide_mode = str(getattr(self, "_effective_interaction_mode", "go"))
        if guide_target is not None and guide_mode in {"detour_left", "detour_right", "backoff", "wait"}:
            cur_dist = math.hypot(px - float(guide_target[0]), py - float(guide_target[1]))
            prev_dist = float(getattr(self, "_prev_dist_to_guide_target", float("inf")))
            if math.isfinite(prev_dist):
                self._guide_target_progress_delta = float(prev_dist - cur_dist)
            self._prev_dist_to_guide_target = float(cur_dist)
        else:
            self._guide_target_progress_delta = 0.0
            self._prev_dist_to_guide_target = float("inf")

    # ── Closed-loop detour guide target (Phase 3) ─────────────────────────

    def _build_detour_guide_target(
        self,
        direction: str,
        phase: str,
    ) -> Tuple[float, float]:
        """Dynamic guide target for closed-loop detour execution.

        Generates a body-frame (forward, lateral) offset that is converted
        to a world-frame guide target.  The magnitude depends on the current
        detour phase.
        """
        from gnn_marl_training.interaction_execution_utils import build_interaction_subgoal_offset

        # Phase-dependent magnitudes
        if phase == DetourPhase.ENTER:
            forward_base = 0.30
            lateral_base = 0.28
        elif phase == DetourPhase.PASS:
            forward_base = 0.40
            lateral_base = 0.22
        else:  # MERGE
            forward_base = 0.45
            lateral_base = 0.10

        # Scale by local clearance
        front_min = float(getattr(self, "_front_min", 0.5))
        if direction == "detour_left":
            side_min = float(getattr(self, "_left_min", front_min))
            turn_sign = 1.0
        else:
            side_min = float(getattr(self, "_right_min", front_min))
            turn_sign = -1.0

        free_front_ratio = np.clip(front_min / 0.60, 0.0, 1.0)
        side_clearance = np.clip(side_min / 0.50, 0.0, 1.0)

        forward = float(np.clip(forward_base + 0.25 * free_front_ratio, 0.22, 0.75))
        lateral = float(np.clip(lateral_base + 0.20 * side_clearance, 0.12, 0.45))

        prev_turn_sign = float(getattr(self, '_last_turn_sign', 0.0))
        if abs(prev_turn_sign) > 1e-6 and turn_sign * prev_turn_sign < 0.0 and abs(float(side_min - front_min)) < float(getattr(self, '_subgoal_lateral_deadband', 0.10)):
            turn_sign = prev_turn_sign

        body_offset = (forward, turn_sign * lateral)

        # Convert to world frame
        guide_target = self._body_to_world_point(body_offset[0], body_offset[1])
        self._detour_guide_target_world = tuple(guide_target)
        self._detour_lateral_displacement = float(lateral)
        self._last_turn_sign = float(turn_sign)
        return guide_target

    def _update_detour_phase(self) -> str:
        """Advance detour phase based on clearance and elapsed steps."""
        current_phase = str(getattr(self, "_option_phase", DetourPhase.ENTER))
        if current_phase not in (DetourPhase.ENTER, DetourPhase.PASS, DetourPhase.MERGE):
            return DetourPhase.ENTER

        steps_elapsed = int(self.current_step) - int(getattr(self, "_active_option_start_step", 0))
        front_min = float(getattr(self, "_front_min", 0.3))
        front_risk = float(
            getattr(self, "_last_predictive_metrics", {}).get("front_risk", 0.5)
        )
        lateral_disp = float(getattr(self, "_detour_lateral_displacement", 0.0))

        if current_phase == DetourPhase.ENTER:
            if steps_elapsed >= DETOUR_ENTER_MIN_STEPS and lateral_disp > DETOUR_LATERAL_DISPLACEMENT_THRESH:
                return DetourPhase.PASS

        elif current_phase == DetourPhase.PASS:
            if (steps_elapsed >= DETOUR_ENTER_MIN_STEPS + DETOUR_PASS_MIN_STEPS
                    and front_min > DETOUR_FRONT_CLEAR_THRESH
                    and front_risk < DETOUR_FRONT_RISK_THRESH):
                return DetourPhase.MERGE

        elif current_phase == DetourPhase.MERGE:
            if lateral_disp < 0.08:
                return DetourPhase.DONE

        return current_phase

    def _build_option_tracking_target(
        self,
        option_mode: str,
        nominal_subgoal: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Per-mode tracking target for the closed-loop option pipeline."""
        if option_mode in ("detour_left", "detour_right"):
            ctx = self._get_interaction_context()
            partner_id = str(ctx.get('partner', ''))
            expected_side = 'left' if option_mode == 'detour_left' else 'right'
            if (not self._detour_active) or self._detour_side != expected_side:
                self._detour_active = True
                self._detour_side = expected_side
                self._detour_phase = DetourPhase.ENTER
                self._option_phase = self._detour_phase
                self._detour_partner_id = partner_id
                self._detour_hold_remaining = int(self._detour_max_duration)
                self._detour_done = False
                self._detour_interrupted = False
                self._detour_lateral_displacement = 0.0

            if self._should_interrupt_detour():
                self._detour_interrupted = True
                self._reset_detour_primitive_state()
                return nominal_subgoal

            self._detour_hold_remaining = max(0, int(self._detour_hold_remaining) - 1)
            next_phase = self._update_detour_phase()
            if next_phase == DetourPhase.DONE:
                self._detour_done = True
                self._reset_detour_primitive_state()
                return nominal_subgoal

            self._detour_phase = str(next_phase)
            self._option_phase = self._detour_phase
            guide_target = self._build_detour_guide_target(option_mode, self._detour_phase)

            partner_rel = self._get_body_relative_to_agent(self._detour_partner_id)
            partner_passed = bool(partner_rel is not None and float(partner_rel[0]) < -0.05)
            social_risk = float(getattr(self, '_last_social_risk', 0.0))
            ttc_min = float(getattr(self, '_last_ttc_min', float('inf')))
            in_conflict = float(ctx.get('in_conflict', 0.0)) > 0.5
            risk_cleared = social_risk < 0.15 and (
                not math.isfinite(ttc_min) or ttc_min > 2.2
            )
            min_elapsed = int(self.current_step) - int(getattr(self, '_active_option_start_step', 0))
            hold_exhausted = self._detour_hold_remaining <= 0
            merge_complete = self._detour_phase == DetourPhase.MERGE and (risk_cleared or not in_conflict)

            if min_elapsed >= self._detour_min_duration and (partner_passed or merge_complete or hold_exhausted):
                self._detour_done = True
                self._reset_detour_primitive_state()
                return nominal_subgoal

            self._detour_suppress_rolling = True
            return tuple(guide_target)

        if option_mode in ("wait", "backoff"):
            # Rebuild the guide target every step so the local target keeps
            # sliding with the live scene instead of freezing for a hold window.
            self._reset_detour_primitive_state()
            mode_key = "wait" if option_mode == "wait" else "backoff"
            offset = build_interaction_subgoal_offset(
                mode=mode_key,
                adaptive_lookahead=float(getattr(self, "rolling_lookahead_dist", 0.8)),
                turn_sign=float(getattr(self, "_last_interaction_turn_sign", 0.0) or 1.0),
                fallback_turn_sign=1.0,
                gap_angle=float(
                    getattr(self, "_last_gap_metrics", {}).get("best_gap_angle", 0.0)
            ),
            )
            if offset is not None:
                self._detour_guide_target_world = tuple(
                    self._body_to_world_point(offset[0], offset[1])
            )
            else:
                sign = float(getattr(self, "_last_interaction_turn_sign", 1.0) or 1.0)
                fallback_offset = (
                    (0.03, 0.10 * sign) if option_mode == "wait"
                    else (-0.22, 0.16 * sign)
            )
                self._detour_guide_target_world = tuple(
                    self._body_to_world_point(fallback_offset[0], fallback_offset[1])
            )
            self._detour_suppress_rolling = True
            return self._detour_guide_target_world

        # go / slow_follow: track the nominal subgoal
        if self._detour_active and not self._detour_done:
            if self._should_interrupt_detour():
                self._detour_interrupted = True
            self._reset_detour_primitive_state()
            self._detour_suppress_rolling = False
            return nominal_subgoal
        self._reset_detour_primitive_state()
        return nominal_subgoal

    def _compute_nominal_tracking_info(self) -> Dict[str, Any]:
        pos = (self.current_pose['x'], self.current_pose['y'])
        path_points = self.global_waypoints if self.global_waypoints else [self.goal_pos]
        tracking_kwargs = {}
        if len(path_points) >= 2 and self.current_projection is not None:
            tracking_kwargs = {
                'anchor_segment_index': int(np.clip(
                    self.current_waypoint_index - 1,
                    0,
                    len(path_points) - 2,
            )),
                'anchor_arc_progress': float(self.path_progress),
            }

        if self.lookahead_dist <= 0.0:
            proj = PathTrackingUtils.get_path_projection(pos, path_points, **tracking_kwargs)
            seg_idx = int(proj.get('segment_index', 0))
            path_heading = 0.0
            if len(path_points) >= 2:
                i = int(np.clip(seg_idx, 0, len(path_points) - 2))
                a, b = path_points[i], path_points[i + 1]
                path_heading = float(math.atan2(b[1] - a[1], b[0] - a[0]))
            return {
                'subgoal': tuple(self.goal_pos),
                'projection': tuple(proj['projection']),
                'arc_progress': float(proj.get('arc_progress', 0.0)),
                'lateral_error': float(proj.get('lateral_error', 0.0)),
                'segment_index': seg_idx,
                'path_heading': path_heading,
                'adaptive_lookahead': 0.0,
            }

        base = PathTrackingUtils.get_rolling_subgoal(
            pos,
            path_points,
            self.lookahead_dist,
            **tracking_kwargs,
        )
        heading_error = self._get_target_angle(base['subgoal'])
        sectors = self._scan_sector_metrics()
        front_min = float(sectors['front_min'])
        adaptive = self._get_adaptive_lookahead(front_min, heading_error)
        info = PathTrackingUtils.get_rolling_subgoal(
            pos,
            path_points,
            adaptive,
            **tracking_kwargs,
        )
        info = dict(info)
        info['subgoal'] = tuple(info['subgoal'])
        info['projection'] = tuple(info['projection'])
        info['adaptive_lookahead'] = float(adaptive)
        return info

    def _apply_nominal_tracking_info(self, info: Dict[str, Any]):
        projection = tuple(info.get('projection', (self.current_pose['x'], self.current_pose['y'])))
        path_heading = float(info.get('path_heading', 0.0))
        arc_progress = float(info.get('arc_progress', 0.0))
        lateral_error = float(info.get('lateral_error', 0.0))
        seg_idx = int(info.get('segment_index', 0))
        path_points = self.global_waypoints if self.global_waypoints else [self.goal_pos]

        self.current_projection = projection
        self.current_path_heading = path_heading
        self.path_progress = arc_progress
        self.current_lateral_error = lateral_error
        self.current_waypoint_index = int(np.clip(seg_idx + 1, 0, max(len(path_points) - 1, 0)))

    def _compute_social_risk_summary(self) -> Dict[str, float]:
        if not hasattr(self, 'parent_env'):
            return {
                'social_risk': 0.0,
                'distance_risk': 0.0,
                'ttc_risk': 0.0,
                'rel_dist': 1.0,
                'rel_bearing': 0.0,
                'closing_speed': 0.0,
                'ttc': 1.0,
                'comm_valid': 0.0,
            }
        return compute_social_risk_summary(
            current_pose=self.current_pose,
            current_vel_x=self.current_vel_x,
            self_agent_id=f"agent_{self.robot_id}",
            robot_positions=getattr(self.parent_env, 'robot_positions', {}),
            robot_velocities=getattr(self.parent_env, 'robot_velocities', {}),
            communication_range=float(
                getattr(
                    self,
                    'interaction_neighbor_perception_range',
                    getattr(getattr(self, 'parent_env', None), 'interaction_neighbor_perception_range', self.scan_max_range),
            )
            ),
            scan_max_range=self.scan_max_range,
            predictive_social_range=self.predictive_social_range,
            predictive_social_ttc_safe=self.predictive_social_ttc_safe,
            yielding_ttc=self.yielding_ttc,
        )

    def _compute_progress_delta_signal(self) -> float:
        return compute_progress_delta_signal(
            path_progress=self.path_progress,
            prev_path_progress=self.prev_path_progress,
            lookahead_dist=self.lookahead_dist,
        )

    def _compute_stuck_score(self, front_blocked_ratio: float) -> float:
        return compute_stuck_score(
            current_vel_x=self.current_vel_x,
            progress_delta=self._compute_progress_delta_signal(),
            stall_elapsed_sec=float(getattr(self, '_stall_elapsed_sec', 0.0)),
            stall_global_replan_sec=float(self.stall_global_replan_sec),
            front_blocked_ratio=front_blocked_ratio,
        )

    def _get_method3_credit_window_metrics(
        self,
        *,
        dist_to_target: float,
        dist_to_goal: float,
        path_progress: float,
        social_risk: float,
        blocked_score: float,
        stuck_score: float,
    ) -> Dict[str, float]:
        if self._method3_credit_history:
            ref = self._method3_credit_history[0]
        else:
            ref = {
                'dist_to_target': float(dist_to_target),
                'dist_to_goal': float(dist_to_goal),
                'path_progress': float(path_progress),
                'social_risk': float(social_risk),
                'blocked_score': float(blocked_score),
                'stuck_score': float(stuck_score),
            }

        progress_delta = float(np.clip(float(ref['dist_to_target']) - float(dist_to_target), -1.0, 1.0))
        path_progress_delta = float(np.clip(float(path_progress) - float(ref['path_progress']), -1.0, 1.0))
        goal_progress_delta = float(np.clip(float(ref['dist_to_goal']) - float(dist_to_goal), -1.0, 1.0))
        social_risk_delta = float(np.clip(float(ref['social_risk']) - float(social_risk), -1.0, 1.0))
        blocked_relief = max(0.0, float(ref['blocked_score']) - float(blocked_score))
        stuck_relief = max(0.0, float(ref['stuck_score']) - float(stuck_score))
        clear_reward = 0.65 * blocked_relief + 0.35 * stuck_relief
        return {
            'progress_delta': progress_delta,
            'local_goal_progress_delta': progress_delta,
            'path_progress_delta': path_progress_delta,
            'goal_progress_delta': goal_progress_delta,
            'social_risk_delta': social_risk_delta,
            'clear_reward': float(clear_reward),
        }

    def _append_method3_credit_snapshot(
        self,
        *,
        dist_to_target: float,
        dist_to_goal: float,
        path_progress: float,
        social_risk: float,
        blocked_score: float,
        stuck_score: float,
    ) -> None:
        self._method3_credit_history.append({
            'dist_to_target': float(dist_to_target),
            'dist_to_goal': float(dist_to_goal),
            'path_progress': float(path_progress),
            'social_risk': float(social_risk),
            'blocked_score': float(blocked_score),
            'stuck_score': float(stuck_score),
        })

    def _build_high_level_policy_features(
        self,
        sectors: Dict[str, float],
        target_x_body: float,
        target_y_body: float,
    ) -> np.ndarray:
        ctx = self._get_interaction_context()
        front_min = float(sectors['front_min'])
        front_blocked_ratio = float(np.clip((self.subgoal_block_front_dist - front_min) / max(self.subgoal_block_front_dist, 1e-6), 0.0, 1.0))
        wait_age_norm = float(np.clip(float(ctx.get('wait_age_norm', 0.0)), 0.0, 1.0))
        progress_delta = self._compute_progress_delta_signal()
        social_summary = self._compute_social_risk_summary()
        social_risk = float(social_summary['social_risk'])
        stuck_score = self._compute_stuck_score(front_blocked_ratio)
        return build_high_level_policy_features(
            stuck_score=stuck_score,
            front_blocked_ratio=front_blocked_ratio,
            wait_age_norm=wait_age_norm,
            progress_delta=progress_delta,
            social_risk=social_risk,
            hold_fraction=0.0,
            target_x_body=target_x_body,
            target_y_body=target_y_body,
        )

    def _build_option_state_features(self) -> np.ndarray:
        """Option state for actor observation: one-hot + fractions + detour phase."""
        return build_option_state_features(self)

    def _build_action_mask_features(self) -> np.ndarray:
        """Action mask as observation features (informs actor of feasible actions)."""
        return build_action_mask_features(self)

    def _build_tracking_target_features(self) -> np.ndarray:
        """Actual tracking target in body frame."""
        return build_tracking_target_features(self)

    def _build_interaction_subgoal(
        self,
        mode: str,
        adaptive_lookahead: float,
        turn_sign: float,
    ) -> Optional[Tuple[float, float]]:
        offset = build_interaction_subgoal_offset(
            mode=mode,
            adaptive_lookahead=adaptive_lookahead,
            turn_sign=turn_sign,
            fallback_turn_sign=self._corner_escape_turn_sign or 1.0,
            gap_angle=float(self._last_gap_metrics.get('best_gap_angle', 0.0)),
        )
        if offset is None:
            return None
        return self._body_to_world_point(offset[0], offset[1])

    def _apply_interaction_shield(
        self,
        linear_vel: float,
        angular_vel: float,
        front_min: float,
        left_min: float,
        right_min: float,
    ) -> Tuple[float, float, Optional[str]]:
        ctx = self._get_interaction_context()
        if float(ctx.get('in_conflict', 0.0)) <= 0.5:
            return float(linear_vel), float(angular_vel), None

        mode = str(ctx.get('mode', 'idle'))
        if mode in ('idle', 'go'):
            return float(linear_vel), float(angular_vel), None

        turn_sign = float(ctx.get('turn_sign', 0.0))
        if abs(turn_sign) < 1e-6:
            turn_sign = 1.0 if left_min >= right_min else -1.0
        severity = float(np.clip(float(ctx.get('severity', 0.0)), 0.0, 1.0))
        ttc = float(ctx.get('ttc', float('inf')))
        short_ttc = math.isfinite(ttc) and ttc < (0.50 * self.yielding_ttc)
        front_constrained = front_min < self.corner_escape_front_dist
        v = float(linear_vel)
        w = float(angular_vel)
        if mode == 'yield':
            target_speed = 0.12 if (not front_constrained and not short_ttc and severity < 0.55) else 0.08
            v = min(v, target_speed)
            if front_min < self.yielding_stop_dist or short_ttc:
                v = min(v, 0.04)
            w = float(np.clip(w + (0.25 + 0.20 * severity) * turn_sign, -self.max_angular_vel, self.max_angular_vel))
            return v, w, 'interaction_yield'
        if mode == 'wait':
            if front_constrained or short_ttc:
                v = min(v, 0.02)
            else:
                v = min(v, 0.06)
            if front_constrained or short_ttc or abs(w) < 0.20:
                w = float(np.clip((0.30 + 0.30 * severity) * turn_sign, -self.max_angular_vel, self.max_angular_vel))
            return v, w, 'interaction_wait'
        if mode == 'backoff':
            reverse_floor = 0.07 + 0.04 * severity
            if front_min < self.yielding_hard_stop_dist + 0.05:
                reverse_floor += 0.02
            v = min(v, -min(reverse_floor, self.max_reverse_vel))
            if abs(w) < 0.30:
                w = float(np.clip((0.45 + 0.30 * severity) * turn_sign, -self.max_angular_vel, self.max_angular_vel))
            return v, w, 'interaction_backoff'
        return float(linear_vel), float(angular_vel), None

    def _get_head_on_conflict_state(self) -> Optional[Dict[str, float]]:
        interaction = self._get_interaction_context()
        if float(interaction.get('in_conflict', 0.0)) > 0.5:
            mode = str(getattr(self, '_executed_behavior_mode', getattr(self, '_policy_interaction_mode', 'go')))
            return {
                'partner': str(interaction.get('partner', '')),
                'dist': float(interaction.get('partner_dist', float('inf'))),
                'closing_speed': float(interaction.get('closing_speed', 0.0)),
                'ttc': float(interaction.get('ttc', float('inf'))),
                'turn_sign': float(interaction.get('turn_sign', 0.0)),
                'severity': float(interaction.get('severity', 0.0)),
                'should_yield': 1.0 if float(interaction.get('should_yield', 0.0)) > 0.5 else 0.0,
                'mode': mode,
                'has_token': float(interaction.get('has_token', 0.0)),
            }
        if not self.yielding_enable or not hasattr(self, 'parent_env'):
            return None

        my_aid = f"agent_{self.robot_id}"
        my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        my_yaw = float(self.current_pose['yaw'])
        my_vel = np.array([
            self.current_vel_x * math.cos(my_yaw),
            self.current_vel_x * math.sin(my_yaw),
        ], dtype=np.float32)
        my_forward = np.array([math.cos(my_yaw), math.sin(my_yaw)], dtype=np.float32)
        best = None

        for aid, pos in self.parent_env.robot_positions.items():
            if aid == my_aid:
                continue

            rel = np.asarray(pos, dtype=np.float32) - my_pos
            dist = float(np.linalg.norm(rel))
            if dist > self.dynamic_replan_neighbor_dist or dist < 1e-6:
                continue

            body_rel = self._world_to_body(rel)
            if float(body_rel[0]) < -0.15:
                continue

            rel_unit = rel / max(dist, 1e-6)
            neighbor_vel = np.asarray(
                self.parent_env.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            rel_vel = neighbor_vel - my_vel
            closing_speed = float(-np.dot(rel_vel, rel_unit))
            ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')
            bearing = float(math.atan2(rel[1], rel[0]))
            yaw_err = self._wrap_angle(bearing - my_yaw)
            turn_sign = -1.0 if yaw_err > 0.0 else 1.0

            my_toward_neighbor = float(np.dot(my_forward, rel_unit))
            other_speed = float(np.linalg.norm(neighbor_vel))
            other_toward_me = 0.0
            if other_speed > 0.05:
                other_toward_me = float(np.dot(neighbor_vel / max(other_speed, 1e-6), -rel_unit))
            head_on_like = (my_toward_neighbor > 0.25) and (other_toward_me > 0.20)
            if not head_on_like:
                continue

            ttc_risk = (
                float(np.clip((self.yielding_ttc - ttc) / self.yielding_ttc, 0.0, 1.0))
            if math.isfinite(ttc) else 0.0
            )
            proximity_risk = float(np.clip((self.yielding_soft_dist - dist) / max(self.yielding_soft_dist, 1e-6), 0.0, 1.0))
            severity = max(ttc_risk, proximity_risk)
            cand = {
                'partner': str(aid),
                'dist': float(dist),
                'closing_speed': float(max(closing_speed, 0.0)),
                'ttc': float(ttc),
                'turn_sign': float(turn_sign),
                'severity': float(severity),
                'should_yield': 1.0 if self._agent_rank(my_aid) > self._agent_rank(str(aid)) else 0.0,
            }
            if best is None or cand['severity'] > best['severity']:
                best = cand

        return best

    def reward_head_on_avoidance(
        self,
        conflict: Optional[Dict[str, float]],
        forward_speed: float,
        turn_rate: float,
        front_min: float,
        left_min: float,
        right_min: float,
    ) -> float:
        if conflict is None:
            return 0.0

        severity = float(np.clip(conflict.get('severity', 0.0), 0.0, 1.0))
        if severity <= 0.0:
            return 0.0

        desired_turn = float(conflict.get('turn_sign', 0.0))
        should_yield = float(conflict.get('should_yield', 0.0)) > 0.5
        mode = str(conflict.get('mode', 'go'))
        if should_yield:
            slow_score = float(np.clip((0.12 - max(forward_speed, 0.0)) / 0.12, 0.0, 1.0))
            turn_align = 0.0
            if abs(turn_rate) > 0.05 and abs(desired_turn) > 1e-6:
                turn_align = 1.0 if math.copysign(1.0, turn_rate) == math.copysign(1.0, desired_turn) else 0.0
            turn_score = turn_align * float(np.clip(abs(turn_rate) / max(self.max_angular_vel, 1e-6), 0.0, 1.0))
            target_side_clearance = left_min if desired_turn > 0.0 else right_min
            lateral_score = float(np.clip((target_side_clearance - front_min) / 0.45, 0.0, 1.0))
            if mode == 'backoff':
                reverse_score = float(np.clip(max(-forward_speed, 0.0) / max(self.max_reverse_vel, 1e-6), 0.0, 1.0))
                return self.head_on_avoidance_reward_scale * severity * (
                    0.62 * reverse_score + 0.23 * turn_score + 0.15 * lateral_score
            )
            if mode == 'wait':
                return self.head_on_avoidance_reward_scale * severity * (
                    0.65 * slow_score + 0.35 * turn_score
            )
            return self.head_on_avoidance_reward_scale * severity * (
                0.45 * slow_score + 0.35 * turn_score + 0.20 * lateral_score
            )

        controlled_forward = float(np.clip(forward_speed / max(self.max_forward_vel, 1e-6), 0.0, 1.0))
        side_bias = right_min if desired_turn > 0.0 else left_min
        keep_clear_score = float(np.clip((front_min + side_bias - 0.55) / 0.55, 0.0, 1.0))
        return 0.35 * self.head_on_avoidance_reward_scale * severity * (
            0.70 * controlled_forward + 0.30 * keep_clear_score
        )

    def _get_neighbor_prediction_features(self) -> np.ndarray:
        if self.neighbor_prediction_dim <= 0:
            return np.zeros(self.neighbor_prediction_dim, dtype=np.float32)

        my_aid = f"agent_{self.robot_id}"
        my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        my_vel = np.array([
            self.current_vel_x * math.cos(self.current_pose['yaw']),
            self.current_vel_x * math.sin(self.current_pose['yaw']),
        ], dtype=np.float32)
        candidates = []
        adjacency_matrix = None
        if hasattr(self, 'parent_env'):
            adjacency_matrix = getattr(self.parent_env, '_last_adj_matrix', None)
        received = self.parent_env._get_received_neighbor_samples(my_aid, adjacency_matrix=adjacency_matrix) \
            if hasattr(self, 'parent_env') else []

        for n_idx, dist, n_pos, n_vel in received:
            if dist > self.predictive_social_range:
                continue

            rel_pos = np.asarray(n_pos, dtype=np.float32) - my_pos
            neighbor_vel = np.asarray(n_vel, dtype=np.float32)
            rel_vel = neighbor_vel - my_vel
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))
            if rel_speed_sq > 1e-6:
                t_star = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, self.predictive_horizon_sec))
            else:
                t_star = 0.0
            min_sep = float(np.linalg.norm(rel_pos + rel_vel * t_star))
            approach_speed = max(0.0, float(-np.dot(rel_pos, rel_vel) / max(dist, 1e-6)))
            ttc = float(dist / approach_speed) if approach_speed > 1e-3 else float('inf')
            sep_risk = float(np.clip((self.predictive_min_sep - min_sep) / self.predictive_min_sep, 0.0, 1.0))
            ttc_risk = (
                float(np.clip((self.predictive_social_ttc_safe - ttc) / self.predictive_social_ttc_safe, 0.0, 1.0))
            if math.isfinite(ttc) else 0.0
            )
            proximity_risk = float(np.clip((self.predictive_social_range - dist) / self.predictive_social_range, 0.0, 1.0))
            risk = max(sep_risk, ttc_risk, 0.5 * proximity_risk)
            body_rel = self._world_to_body(rel_pos)
            candidates.append((
                -risk,
                dist,
                np.array([
                    float(np.clip(body_rel[0] / self.predictive_social_range, -1.0, 1.0)),
                    float(np.clip(body_rel[1] / self.predictive_social_range, -1.0, 1.0)),
                    float(np.clip(approach_speed / 0.8, 0.0, 1.0)),
                    float(np.clip(ttc / self.predictive_social_ttc_safe, 0.0, 1.0)) if math.isfinite(ttc) else 1.0,
                    float(np.clip(min_sep / self.predictive_min_sep, 0.0, 1.0)),
                    float(risk),
                ], dtype=np.float32),
            ))

        candidates.sort(key=lambda item: (item[0], item[1]))
        features = np.zeros(self.neighbor_prediction_dim, dtype=np.float32)
        for idx, (_, _, token) in enumerate(candidates[:self.neighbor_prediction_top_k]):
            start = idx * self.neighbor_prediction_feature_dim
            end = start + self.neighbor_prediction_feature_dim
            features[start:end] = token
        return features

    def _get_obstacle_motion_features(self, sector_dists: np.ndarray) -> np.ndarray:
        if self.obstacle_motion_dim <= 0:
            return np.zeros(0, dtype=np.float32)
        features = np.zeros(self.obstacle_motion_dim, dtype=np.float32)

        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            self._obstacle_cluster_history.append([])
            return features

        current_clusters_body = self._extract_scan_clusters(
            np.asarray(self.latest_scan.ranges, dtype=np.float32)
        )
        prev_clusters = self._obstacle_cluster_history[-1] if self._obstacle_cluster_history else []
        robot_world = np.array(
            [float(self.current_pose['x']), float(self.current_pose['y'])],
            dtype=np.float32,
        )
        current_clusters: List[Dict[str, float]] = []
        for cluster in current_clusters_body:
            body_xy = np.array([float(cluster["x"]), float(cluster["y"])], dtype=np.float32)
            world_xy = robot_world + self._body_to_world_vec(body_xy)
            enriched = dict(cluster)
            enriched["xw"] = float(world_xy[0])
            enriched["yw"] = float(world_xy[1])
            current_clusters.append(enriched)
        self._obstacle_cluster_history.append(current_clusters)

        candidates = []
        predict_h = min(max(self.predictive_horizon_sec, 0.3), 0.8)
        denom = max(self.obstacle_filter_range, 1e-6)
        corridor_half_width = max(self.predictive_min_sep, 0.45)

        for cluster in current_clusters:
            matched = self._match_previous_cluster(cluster, prev_clusters)
            vx_world = 0.0
            vy_world = 0.0
            if matched is not None:
                vx_world = float((float(cluster["xw"]) - float(matched["xw"])) / self.control_dt)
                vy_world = float((float(cluster["yw"]) - float(matched["yw"])) / self.control_dt)

            rel_world = np.array([
                float(cluster["xw"]) - float(robot_world[0]),
                float(cluster["yw"]) - float(robot_world[1]),
            ], dtype=np.float32)
            rel_body = self._world_to_body(rel_world)
            future_world = np.array([
                float(cluster["xw"]) + vx_world * predict_h,
                float(cluster["yw"]) + vy_world * predict_h,
            ], dtype=np.float32)
            future_rel_world = future_world - robot_world
            future_rel_body = self._world_to_body(future_rel_world)
            vel_body = self._world_to_body(np.array([vx_world, vy_world], dtype=np.float32))

            x = float(rel_body[0])
            y = float(rel_body[1])
            vx = float(vel_body[0])
            vy = float(vel_body[1])
            future_x = float(future_rel_body[0])
            future_y = float(future_rel_body[1])
            dist = float(math.hypot(x, y))
            future_dist = float(math.hypot(future_x, future_y))

            close_risk = float(np.clip((self.close_obstacle_dist - dist) / self.close_obstacle_dist, 0.0, 1.0))
            future_risk = float(np.clip((self.predictive_min_sep - future_dist) / self.predictive_min_sep, 0.0, 1.0))
            crossing_gate = float(np.clip(1.0 - abs(future_y) / corridor_half_width, 0.0, 1.0))
            forward_gate = float(np.clip((future_x + 0.15) / max(self.obstacle_filter_range, 1e-6), 0.0, 1.0))
            transverse_speed = abs(vy)
            crossing_risk = crossing_gate * forward_gate * float(np.clip(transverse_speed / 0.6, 0.0, 1.0))
            closing_speed = float(max(0.0, -(x * vx + y * vy) / max(dist, 1e-6)))
            ttc = float(dist / closing_speed) if closing_speed > 1e-3 else float("inf")
            ttc_risk = (
                float(np.clip((self.predictive_front_ttc_safe - ttc) / self.predictive_front_ttc_safe, 0.0, 1.0))
            if math.isfinite(ttc) else 0.0
            )
            risk = max(close_risk, future_risk, crossing_risk, ttc_risk)
            if risk <= 1e-4:
                continue

            token = np.array([
                float(np.clip(x / denom, -1.0, 1.0)),
                float(np.clip(y / denom, -1.0, 1.0)),
                float(np.clip(vx / 0.8, -1.0, 1.0)),
                float(np.clip(vy / 0.8, -1.0, 1.0)),
                float(np.clip(future_x / denom, -1.0, 1.0)),
                float(np.clip(future_y / denom, -1.0, 1.0)),
            ], dtype=np.float32)
            candidates.append((-risk, dist, token))

        candidates.sort(key=lambda item: (item[0], item[1]))
        for idx, (_, _, token) in enumerate(candidates[: self.obstacle_motion_top_k]):
            start = idx * self.obstacle_motion_feature_dim
            end = start + self.obstacle_motion_feature_dim
            features[start:end] = token
        return features

    def _wrap_angle(self, angle):
        return (float(angle) + math.pi) % (2 * math.pi) - math.pi

    def _get_target_angle(self, target):
        tgt_angle = math.atan2(target[1] - self.current_pose['y'], target[0] - self.current_pose['x'])
        return self._wrap_angle(tgt_angle - self.current_pose['yaw'])

    def _get_adaptive_lookahead(self, front_min, heading_error):
        if self.lookahead_dist <= 0.0:
            return 0.0

        lookahead = float(self.lookahead_dist)
        if front_min < 0.8:
            lookahead *= 0.75
        if front_min < 0.5:
            lookahead *= 0.60
        if front_min < 0.3:
            lookahead *= 0.45
        if abs(heading_error) > 0.35:
            lookahead *= 0.70
        if abs(heading_error) > 0.70:
            lookahead *= 0.50
        min_lh = min(0.25, self.lookahead_dist)
        return float(np.clip(lookahead, min_lh, self.lookahead_dist))

    def _body_to_world_point(self, x_body: float, y_body: float) -> Tuple[float, float]:
        yaw = float(self.current_pose['yaw'])
        c = math.cos(yaw)
        s = math.sin(yaw)
        xw = float(self.current_pose['x']) + c * x_body - s * y_body
        yw = float(self.current_pose['y']) + s * x_body + c * y_body
        return (xw, yw)

    def _path_to_world_point(self, x_path: float, y_path: float) -> Tuple[float, float]:
        """Convert path-frame (forward, lateral) offset to world coordinates.

        x_path: forward distance along the path heading
        y_path: lateral offset (+ = left of path, - = right of path)
        """
        heading = float(getattr(self, 'current_path_heading', 0.0))
        c = math.cos(heading)
        s = math.sin(heading)
        xw = float(self.current_pose['x']) + c * x_path - s * y_path
        yw = float(self.current_pose['y']) + s * x_path + c * y_path
        return (xw, yw)

    @staticmethod
    def _repulsive_penalty(dist: float, safe_dist: float, scale: float) -> float:
        if dist >= safe_dist:
            return 0.0
        dist = max(float(dist), 0.10)
        return -scale * ((1.0 / dist) - (1.0 / safe_dist)) ** 2

    def _clear_corner_escape(self) -> None:
        self._corner_escape_hold_steps = 0
        self._corner_escape_turn_sign = 0.0

    def _build_corner_escape_subgoal(self, adaptive_lookahead: float, turn_sign: float) -> Tuple[float, float]:
        lookahead = max(0.22, float(adaptive_lookahead))
        x_body = max(0.06, lookahead * self.corner_escape_forward_gain)
        y_body = math.copysign(max(0.18, lookahead * self.corner_escape_lateral_gain), turn_sign)
        return self._body_to_world_point(x_body, y_body)

    def _pick_corner_escape_turn_sign(
        self,
        target_angle: float,
        left_min: float,
        right_min: float,
        gap: Optional[Dict[str, float]] = None,
    ) -> float:
        preferred = 1.0 if left_min >= right_min else -1.0
        if abs(float(target_angle)) > 0.12:
            preferred = math.copysign(1.0, float(target_angle))

        if gap and gap.get('best_sector_idx', -1) >= 0:
            gap_angle = float(gap.get('best_gap_angle', 0.0))
            gap_clearance = float(gap.get('best_gap_clearance', 0.0))
            if abs(gap_angle) > 0.10 and gap_clearance > 0.25:
                preferred = math.copysign(1.0, gap_angle)

        if preferred > 0.0 and left_min < self.subgoal_min_side_clearance and right_min > left_min + 0.03:
            preferred = -1.0
        elif preferred < 0.0 and right_min < self.subgoal_min_side_clearance and left_min > right_min + 0.03:
            preferred = 1.0

        chosen_clearance = left_min if preferred > 0.0 else right_min
        return preferred if chosen_clearance > 0.12 else 0.0

    def _select_corner_escape_subgoal(
        self,
        adaptive_lookahead: float,
        front_min: float,
        left_min: float,
        right_min: float,
        target_angle: float,
        gap: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
        if self._corner_escape_hold_steps > 0:
            turn_sign = self._corner_escape_turn_sign or self._pick_corner_escape_turn_sign(
                target_angle,
            left_min,
            right_min,
                gap,
            )
            if abs(turn_sign) > 1e-6 and front_min < (self.corner_escape_front_dist * 1.25):
                self._corner_escape_turn_sign = turn_sign
                self._corner_escape_hold_steps = max(0, self._corner_escape_hold_steps - 1)
                return self._build_corner_escape_subgoal(adaptive_lookahead, turn_sign), 'corner_escape'
            self._clear_corner_escape()
            return None, None

        stalled = (
            abs(float(getattr(self, 'current_vel_x', 0.0))) < self.corner_escape_speed_thresh
            or self._subgoal_deadlock_streak >= max(2, self.subgoal_deadlock_steps // 2)
        )
        if (
            front_min >= self.corner_escape_front_dist
            or abs(float(target_angle)) < self.corner_escape_angle_thresh
            or not stalled
        ):
            return None, None

        turn_sign = self._pick_corner_escape_turn_sign(target_angle, left_min, right_min, gap)
        if abs(turn_sign) < 1e-6:
            return None, None

        self._corner_escape_turn_sign = turn_sign
        self._corner_escape_hold_steps = self.corner_escape_commit_steps
        return self._build_corner_escape_subgoal(adaptive_lookahead, turn_sign), 'corner_escape'

    def _compute_clearance_context(
        self,
        front_min: float,
        left_min: float,
        right_min: float,
        front_penalty_relax: float,
        side_penalty_relax: float,
        corner_escape_active: bool,
    ) -> Dict[str, float]:
        side_min = float(min(left_min, right_min))
        front_blocked_ratio = float(np.clip((0.34 - front_min) / 0.34, 0.0, 1.0))
        front_close_ratio = float(np.clip(
            (self.close_obstacle_dist - front_min) / max(self.close_obstacle_dist, 1e-6),
            0.0,
            1.0,
        ))
        side_close_ratio = float(np.clip(
            (self.side_close_dist - side_min) / max(self.side_close_dist, 1e-6),
            0.0,
            1.0,
        ))
        front_potential_penalty = self._repulsive_penalty(front_min, 0.38, 0.18) * front_penalty_relax
        side_wall_penalty = (
            self._repulsive_penalty(left_min, 0.18, 0.010)
            + self._repulsive_penalty(right_min, 0.18, 0.010)
        ) * side_penalty_relax
        front_close_penalty = -self.close_obstacle_penalty_scale * (front_close_ratio ** 2)
        side_close_penalty = -self.side_close_penalty_scale * (side_close_ratio ** 2)
        if corner_escape_active:
            side_wall_penalty *= 0.35
            side_close_penalty *= 0.25
        return {
            'side_min': side_min,
            'front_blocked_ratio': front_blocked_ratio,
            'front_close_ratio': front_close_ratio,
            'side_close_ratio': side_close_ratio,
            'front_potential_penalty': float(front_potential_penalty),
            'side_wall_penalty': float(side_wall_penalty),
            'front_close_penalty': float(front_close_penalty),
            'side_close_penalty': float(side_close_penalty),
        }

    def _try_replan_due_to_deadlock(self) -> bool:
        if not self.replan_on_deadlock or self.planner is None:
            return False
        if self.current_step < self._next_replan_step:
            return False

        start_time = time.perf_counter()
        self._last_replan_attempted = True
        self._last_replan_success = False
        self._last_replan_wall_time_sec = 0.0
        try:
            start = (float(self.current_pose['x']), float(self.current_pose['y']))
            goal = (float(self.goal_pos[0]), float(self.goal_pos[1]))
            blocked_summary = self._collect_replan_blocked_points()
            blocked = list(blocked_summary.get('points', []))
            path = self.planner.plan_with_dynamic_obstacles(
                start,
                goal,
                blocked_world_points=blocked,
                block_radius_m=self.dynamic_replan_block_radius,
            )
            if path is None:
                path = self.planner.plan(start, goal)
            self._next_replan_step = self.current_step + self.replan_cooldown_steps
            if not path:
                self._last_replan_wall_time_sec = float(time.perf_counter() - start_time)
                self._recent_replan_steps.append(int(self.current_step))
                self._reset_stall_replan_tracker(anchor_current=True)
                return False
            self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
            self._reset_path_tracking_state()
            self._clear_committed_subgoal()
            self._subgoal_deadlock_streak = 0
            self._clear_corner_escape()
            self._reset_stall_replan_tracker(anchor_current=True)
            self._last_replan_success = True
            self._last_replan_wall_time_sec = float(time.perf_counter() - start_time)
            self._recent_replan_steps.append(int(self.current_step))
            if hasattr(self, 'vis') and self.vis:
                self.vis.clear_waypoints(namespace=self.vis_namespace)
                self.vis.publish_waypoints(
                    self.global_waypoints,
                    robot_id=self.robot_id,
                    namespace=self.vis_namespace,
            )
            return True
        except Exception:
            self._next_replan_step = self.current_step + self.replan_cooldown_steps
            self._reset_stall_replan_tracker(anchor_current=True)
            self._last_replan_wall_time_sec = float(time.perf_counter() - start_time)
            self._recent_replan_steps.append(int(self.current_step))
            return False

    def _select_local_detour_subgoal(
        self,
        nominal_subgoal: Tuple[float, float],
        adaptive_lookahead: float,
        front_min: float,
        left_min: float,
        right_min: float,
        sector_dists: Optional[np.ndarray] = None,
        candidate_arc_progress: Optional[float] = None,
    ) -> Tuple[Tuple[float, float], str]:
        pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        nominal = np.asarray(nominal_subgoal, dtype=np.float32)
        rel_nominal = nominal - pos
        rel_body = self._world_to_body(rel_nominal)
        nominal_target_angle = float(self._get_target_angle(nominal_subgoal))
        sector_arr = (
            np.asarray(sector_dists, dtype=np.float32)
            if sector_dists is not None else self._get_current_sector_dists()
        )
        side_clear = max(left_min, right_min)
        side_clear_thresh = self.subgoal_min_side_clearance + 0.06

        forward_speed = abs(float(getattr(self, 'current_vel_x', 0.0)))
        if front_min < self.subgoal_deadlock_front_dist and forward_speed < self.subgoal_deadlock_speed_thresh:
            self._subgoal_deadlock_streak += 1
        else:
            self._subgoal_deadlock_streak = max(0, self._subgoal_deadlock_streak - 1)

        blocked = (front_min < self.subgoal_block_front_dist) and (float(rel_body[0]) > 0.12)
        force_detour = self._subgoal_deadlock_streak >= self.subgoal_deadlock_steps
        allow_global_replan = (
            float(getattr(self, '_stall_elapsed_sec', 0.0)) >= (2.0 * float(self.stall_global_replan_sec))
        )
        if force_detour and allow_global_replan and self._try_replan_due_to_deadlock():
            self._clear_committed_subgoal()
            return nominal_subgoal, 'replan'
        if allow_global_replan and self._should_force_global_replan_from_stall(front_min, arc_progress=candidate_arc_progress):
            if self._try_replan_due_to_deadlock():
                self._clear_committed_subgoal()
                return nominal_subgoal, 'replan'

        interaction = self._get_interaction_context()
        interaction_turn_sign = float(interaction.get('turn_sign', 0.0))
        policy_mode = str(getattr(self, '_effective_interaction_mode', getattr(self, '_policy_interaction_mode', 'go')))
        interaction_partner = str(interaction.get('partner', ''))
        if policy_mode == 'replan':
            self._clear_committed_subgoal()
            if self.current_step >= self._next_replan_step:
                self._try_replan_due_to_deadlock()
            self._yield_hold_steps = 0
            self._yield_partner = ''
            self._yield_turn_sign = 0.0
            return nominal_subgoal, 'replan'
        if policy_mode in {'wait', 'backoff', 'detour'}:
            committed = self._get_committed_subgoal(policy_mode, partner=interaction_partner)
            if committed is not None:
                return committed, policy_mode
            protocol_subgoal = self._build_interaction_subgoal(
            policy_mode,
                adaptive_lookahead,
                interaction_turn_sign,
            )
            if protocol_subgoal is not None:
                self._clear_corner_escape()
                self._yield_hold_steps = 0
                self._yield_partner = interaction_partner
                self._yield_turn_sign = interaction_turn_sign
                hold_steps = self.yielding_commit_steps if policy_mode == 'wait' else max(5, self.subgoal_detour_hold_steps)
                committed = self._commit_subgoal(
                    protocol_subgoal,
                policy_mode,
                    hold_steps,
                    partner=interaction_partner,
            )
                return committed, policy_mode
        if policy_mode == 'go':
            self._clear_committed_subgoal()
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0

        gap = None
        if front_min < self.corner_escape_front_dist or blocked or force_detour:
            gap = self._compute_gap_metrics(sector_arr, nominal_target_angle=nominal_target_angle)
            escape_subgoal, escape_mode = self._select_corner_escape_subgoal(
                adaptive_lookahead,
            front_min,
            left_min,
            right_min,
                nominal_target_angle,
                gap,
            )
            if escape_subgoal is not None:
                committed = self._get_committed_subgoal(escape_mode)
                if committed is not None:
                    return committed, escape_mode
                committed = self._commit_subgoal(
                    escape_subgoal,
                    escape_mode,
                    self.corner_escape_commit_steps,
            )
                return committed, escape_mode

        if not blocked and not force_detour:
            self._subgoal_detour_hold = max(0, self._subgoal_detour_hold - 1)
            if self._subgoal_detour_hold == 0:
                self._subgoal_detour_side = 0
            self._clear_corner_escape()
            self._clear_committed_subgoal()
            return nominal_subgoal, 'nominal'

        committed_mode = self._committed_subgoal_mode
        if committed_mode in {'gap_detour', 'detour'}:
            committed = self._get_committed_subgoal(committed_mode)
            if committed is not None:
                return committed, committed_mode

        if gap is None:
            gap = self._compute_gap_metrics(sector_arr, nominal_target_angle=nominal_target_angle)
        if gap.get('best_sector_idx', -1) >= 0:
            gap_angle = float(gap['best_gap_angle'])
            gap_clearance = float(gap['best_gap_clearance'])
            gap_width = float(gap['best_gap_width'])
            lookahead = max(0.25, float(adaptive_lookahead))
            radial = lookahead * (0.60 + 0.40 * max(gap_clearance, gap_width))
            if force_detour:
                radial *= 0.90
            x_body = radial * math.cos(gap_angle)
            y_body = radial * math.sin(gap_angle)
            if x_body > 0.10:
                cand = self._body_to_world_point(x_body, y_body)
                if abs(self._get_target_angle(cand)) <= 1.52:
                    self._clear_corner_escape()
                    self._subgoal_detour_side = 1 if y_body >= 0.0 else -1
                    self._subgoal_detour_hold = self.subgoal_detour_hold_steps
                    committed = self._commit_subgoal(
                        cand,
                        'gap_detour',
                        max(3, self.subgoal_detour_hold_steps),
                )
                    return committed, 'gap_detour'

        if self._subgoal_detour_hold > 0 and self._subgoal_detour_side != 0:
            preferred_side = self._subgoal_detour_side
        else:
            preferred_side = 1 if left_min >= right_min else -1

        # 若首选侧净空不足，尝试另一侧。
        if preferred_side > 0 and left_min < self.subgoal_min_side_clearance and right_min > left_min:
            preferred_side = -1
        elif preferred_side < 0 and right_min < self.subgoal_min_side_clearance and left_min > right_min:
            preferred_side = 1

        lookahead = max(0.25, float(adaptive_lookahead))
        forward_step = lookahead * self.subgoal_detour_forward_gain
        lateral_step = lookahead * self.subgoal_detour_lateral_gain
        if force_detour:
            forward_step *= 0.65

        candidates = [preferred_side, -preferred_side] if preferred_side != 0 else [1, -1]
        for side in candidates:
            side_clear = left_min if side > 0 else right_min
            if side_clear < self.subgoal_min_side_clearance:
                continue
            cand = self._body_to_world_point(forward_step, side * lateral_step)
            if abs(self._get_target_angle(cand)) > 1.45:
                continue
            self._clear_corner_escape()
            if self._subgoal_detour_side != 0 and int(side) != int(self._subgoal_detour_side):
                info = getattr(self, '_last_step_debug_metrics', None)
                if isinstance(info, dict):
                    info['subgoal_flip_penalty'] = -float(getattr(self, '_subgoal_switch_cost', 0.02))
            self._subgoal_detour_side = int(side)
            self._last_turn_sign = float(side)
            self._subgoal_detour_hold = max(self.subgoal_detour_hold_steps, 10)
            committed = self._commit_subgoal(
                cand,
                'detour',
                max(3, self.subgoal_detour_hold_steps),
            )
            return committed, 'detour'

        self._clear_corner_escape()
        self._clear_committed_subgoal()
        return nominal_subgoal, ('deadlock' if force_detour else 'blocked_nominal')

    def _get_tracking_target(self):
        try:
            info = self._compute_nominal_tracking_info()
            self._apply_nominal_tracking_info(info)
            nominal_subgoal = tuple(info['subgoal'])
            self._last_nominal_subgoal = nominal_subgoal
            adaptive = float(info.get('adaptive_lookahead', 0.0))
            sectors = self._scan_sector_metrics()
            front_min = float(sectors['front_min'])
            left_min = float(sectors['left_min'])
            right_min = float(sectors['right_min'])
            sector_dists = self._get_current_sector_dists()
            chosen_subgoal, mode = self._select_local_detour_subgoal(
                nominal_subgoal,
                adaptive,
            front_min,
            left_min,
            right_min,
                sector_dists=sector_dists,
                candidate_arc_progress=float(info.get('arc_progress', self.path_progress)),
            )
            if mode == 'replan':
                refreshed = self._compute_nominal_tracking_info()
                self._apply_nominal_tracking_info(refreshed)
                chosen_subgoal = tuple(refreshed['subgoal'])
            prev_subgoal = getattr(self, 'current_subgoal', None)
            chosen_subgoal = tuple(chosen_subgoal)
            if prev_subgoal is not None:
                prev_arr = np.asarray(prev_subgoal, dtype=np.float32)
                next_arr = np.asarray(chosen_subgoal, dtype=np.float32)
                delta = next_arr - prev_arr
                dist = float(np.linalg.norm(delta))
                if dist > 1e-6:
                    alpha = 0.35 if mode in {'detour', 'gap_detour', 'wait', 'backoff'} else 0.20
                    chosen_subgoal = tuple((prev_arr + alpha * delta).tolist())
            self.current_subgoal = chosen_subgoal
            self._last_subgoal_mode = mode
            return self.current_subgoal
        except Exception:
            self.current_projection = None
            self._last_nominal_subgoal = tuple(self.goal_pos)
            self.current_subgoal = tuple(self.goal_pos)
            self.current_path_heading = 0.0
            self.path_progress = 0.0
            self.current_lateral_error = 0.0
            self._last_subgoal_mode = 'tracking_fallback'
            return self.current_subgoal

    def _publish_tracking_visuals(self, target_pos):
        if not (hasattr(self, 'vis') and self.vis):
            return
        try:
            nominal_target = getattr(self, '_last_nominal_subgoal', None)
            mode = str(getattr(self, '_last_subgoal_mode', 'nominal'))
            same_target = False
            if nominal_target is not None and target_pos is not None:
                same_target = bool(
                    np.linalg.norm(
                        np.asarray(target_pos, dtype=np.float32)
                        - np.asarray(nominal_target, dtype=np.float32)
                ) <= 1e-3
            )
            label = (
                f'R{self.robot_id} {mode} | actual==nominal'
                if same_target else
                f'R{self.robot_id} {mode} | actual!=nominal'
            )
            self.vis.publish_tracking_state(
                robot_pos=(self.current_pose['x'], self.current_pose['y']),
                target_pos=target_pos,
                nominal_target_pos=nominal_target,
                projection_pos=self.current_projection,
                robot_id=self.robot_id,
                namespace=self.vis_namespace,
                label=label,
            )
        except Exception as _vis_e:
            self.node.get_logger().warn(f'publish_tracking_state failed: {_vis_e}')

    def _check_collision_event(self, min_dist: float, info: Dict[str, Any]) -> bool:
        """碰撞判定：优先 Gazebo 硬碰撞事件，必要时回退雷达阈值。"""
        if self.use_gazebo_collision and self._gazebo_collision_active:
            info['collision_source'] = 'gazebo'
            # 消费事件，避免一次接触被重复多步计数
            self._gazebo_collision_active = False
            return True

        if self.lidar_collision_fallback:
            if min_dist < self.collision_persist_dist:
                self._close_obstacle_streak += 1
            else:
                self._close_obstacle_streak = 0

            hard_collision = (min_dist < self.collision_hard_dist)
            persistent_collision = (self._close_obstacle_streak >= self.collision_persist_steps)
            if hard_collision or persistent_collision:
                info['collision_source'] = 'lidar_fallback'
                return True

        return False

    def _predict_social_risk_metrics(self) -> Dict[str, float]:
        metrics = {
            'social_ttc': float('inf'),
            'social_min_sep': float('inf'),
            'social_risk': 0.0,
        }
        if not self.predictive_feature_enable or not hasattr(self, 'parent_env'):
            return metrics

        my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        my_vel = np.array([
            self.current_vel_x * math.cos(self.current_pose['yaw']),
            self.current_vel_x * math.sin(self.current_pose['yaw']),
        ], dtype=np.float32)
        best_ttc = float('inf')
        best_min_sep = float('inf')
        max_risk = 0.0
        my_aid = f"agent_{self.robot_id}"

        for aid, pos in self.parent_env.robot_positions.items():
            if aid == my_aid:
                continue

            rel_pos = np.asarray(pos, dtype=np.float32) - my_pos
            dist = float(np.linalg.norm(rel_pos))
            if dist > self.predictive_social_range:
                continue

            neighbor_vel = np.asarray(
                self.parent_env.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
                dtype=np.float32,
            )
            rel_vel = neighbor_vel - my_vel
            rel_speed_sq = float(np.dot(rel_vel, rel_vel))

            if rel_speed_sq > 1e-6:
                t_star = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, self.predictive_horizon_sec))
            else:
                t_star = 0.0
            min_sep = float(np.linalg.norm(rel_pos + rel_vel * t_star))

            approach_speed = 0.0
            if dist > 1e-6:
                approach_speed = max(0.0, float(-np.dot(rel_pos, rel_vel) / dist))
            ttc = float(dist / approach_speed) if approach_speed > 1e-3 else float('inf')

            sep_risk = float(np.clip((self.predictive_min_sep - min_sep) / self.predictive_min_sep, 0.0, 1.0))
            if math.isfinite(ttc):
                ttc_risk = float(np.clip((self.predictive_social_ttc_safe - ttc) / self.predictive_social_ttc_safe, 0.0, 1.0))
            else:
                ttc_risk = 0.0
            proximity_risk = float(np.clip((self.predictive_social_range - dist) / self.predictive_social_range, 0.0, 1.0))
            risk = max(sep_risk, ttc_risk, min(1.0, self.social_proximity_risk_scale * proximity_risk))

            best_ttc = min(best_ttc, ttc)
            best_min_sep = min(best_min_sep, min_sep)
            max_risk = max(max_risk, risk)

        metrics['social_ttc'] = best_ttc
        metrics['social_min_sep'] = best_min_sep
        metrics['social_risk'] = max_risk
        return metrics

    def _predict_front_risk_metrics(self, front_min: float) -> Dict[str, float]:
        metrics = {
            'front_closing_speed': 0.0,
            'front_ttc': float('inf'),
            'front_risk': 0.0,
        }
        if not self.predictive_feature_enable:
            return metrics

        front_trace = list(self._front_min_history)
        closing_speed = 0.0
        if len(front_trace) >= 2:
            deltas = []
            for prev_d, cur_d in zip(front_trace[:-1], front_trace[1:]):
                deltas.append(max(0.0, float(prev_d - cur_d)) / self.control_dt)
            if deltas:
                closing_speed = float(np.mean(deltas))

        ttc = float('inf')
        if closing_speed > 1e-3:
            ttc = float(front_min / max(closing_speed, 1e-3))

        future_clearance = float(front_min - closing_speed * self.predictive_horizon_sec)
        clearance_risk = float(np.clip((self.predictive_min_sep - future_clearance) / self.predictive_min_sep, 0.0, 1.0))
        if math.isfinite(ttc):
            ttc_risk = float(np.clip((self.predictive_front_ttc_safe - ttc) / self.predictive_front_ttc_safe, 0.0, 1.0))
        else:
            ttc_risk = 0.0
        close_risk = float(np.clip((self.close_obstacle_dist - front_min) / self.close_obstacle_dist, 0.0, 1.0))
        metrics['front_closing_speed'] = closing_speed
        metrics['front_ttc'] = ttc
        metrics['front_risk'] = max(clearance_risk, ttc_risk, close_risk)
        return metrics

    def _get_predictive_obs_features(self, front_min: float) -> np.ndarray:
        if not self.predictive_feature_enable:
            self._last_predictive_metrics = {
                'social_ttc': float('inf'),
                'social_min_sep': float('inf'),
                'social_risk': 0.0,
                'front_closing_speed': 0.0,
                'front_ttc': float('inf'),
                'front_risk': 0.0,
            }
            return np.zeros(0, dtype=np.float32)

        self._front_min_history.append(float(front_min))
        social_metrics = self._predict_social_risk_metrics()
        front_metrics = self._predict_front_risk_metrics(float(front_min))
        self._last_predictive_metrics = {
            **social_metrics,
            **front_metrics,
        }

        social_ttc_norm = 1.0
        if math.isfinite(social_metrics['social_ttc']):
            social_ttc_norm = float(np.clip(social_metrics['social_ttc'] / self.predictive_social_ttc_safe, 0.0, 1.0))
        social_min_sep_norm = 1.0
        if math.isfinite(social_metrics['social_min_sep']):
            social_min_sep_norm = float(np.clip(social_metrics['social_min_sep'] / self.predictive_min_sep, 0.0, 1.0))

        front_ttc_norm = 1.0
        if math.isfinite(front_metrics['front_ttc']):
            front_ttc_norm = float(np.clip(front_metrics['front_ttc'] / self.predictive_front_ttc_safe, 0.0, 1.0))
        front_closing_norm = float(np.clip(front_metrics['front_closing_speed'] / 0.6, 0.0, 1.0))

        return np.array([
            social_ttc_norm,
            social_min_sep_norm,
            float(social_metrics['social_risk']),
            front_closing_norm,
            front_ttc_norm,
            float(front_metrics['front_risk']),
        ], dtype=np.float32)

    def reset(self, seed=None, options=None, other_agent_starts=None, forced_start_goal=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._publish_vel(0.0, 0.0)
        self._reset_path_tracking_state()
        self.collision_history = []
        self._close_obstacle_streak = 0
        self._gazebo_collision_active = False
        self._gazebo_collision_seen = False
        self._gazebo_collision_last_step = -10**9
        self._front_min_history.clear()
        self._front_sector_dist_history.clear()
        self._obstacle_cluster_history.clear()
        self._last_predictive_metrics = {
            'social_ttc': float('inf'),
            'social_min_sep': float('inf'),
            'social_risk': 0.0,
            'front_closing_speed': 0.0,
            'front_ttc': float('inf'),
            'front_risk': 0.0,
        }
        self._last_gap_metrics = {
            'best_gap_angle': 0.0,
            'best_gap_width': 0.0,
            'best_gap_clearance': 0.0,
            'best_gap_score': 0.0,
        }
        self._subgoal_detour_hold = 0
        self._subgoal_detour_side = 0
        self._subgoal_deadlock_streak = 0
        self._corner_escape_hold_steps = 0
        self._corner_escape_turn_sign = 0.0
        self._next_replan_step = 0
        self._reset_stall_replan_tracker(anchor_current=False)
        self._clear_committed_subgoal()
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0
        self._last_turn_sign = 0.0
        self._last_detour_direction = ''
        self._last_interaction_mode = 'idle'
        self._last_interaction_turn_sign = 0.0
        self._policy_interaction_mode = 'go'
        self._policy_interaction_action = 0
        self._effective_interaction_mode = 'go'
        self._executed_behavior_mode = 'nominal'
        self._cached_step_tracking_target = None
        self._cached_step_tracking_mode = 'nominal'
        self._cached_step_tracking_step = -1
        self._last_nominal_subgoal = None
        self._last_social_risk = 0.0
        self._last_front_blocked_ratio = 0.0
        self._last_stuck_score = 0.0
        self._method3_credit_history.clear()
        self._recent_replan_steps.clear()
        self._last_replan_attempted = False
        self._last_replan_success = False
        self._last_replan_wall_time_sec = 0.0
        self._last_interaction_info = {
            'mode': 'idle',
            'mode_id': 0.0,
            'in_conflict': 0.0,
            'has_token': 0.0,
            'should_yield': 0.0,
            'partner': '',
            'partner_dist': float('inf'),
            'closing_speed': 0.0,
            'ttc': float('inf'),
            'severity': 0.0,
            'turn_sign': 0.0,
            'front_min': float('inf'),
            'front_blocked_ratio': 0.0,
            'component_size': 1.0,
            'wait_steps': 0.0,
            'wait_age_norm': 0.0,
        }

        if hasattr(self, 'vis') and self.vis:
            self.vis.clear_waypoints(namespace=self.vis_namespace)

        forced_ok = False
        if forced_start_goal is not None:
            try:
                (start_xy, goal_xy) = forced_start_goal
                start_x, start_y = float(start_xy[0]), float(start_xy[1])
                goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
                forced_ok = self._is_valid_start_goal_pair(
                    (start_x, start_y),
                    (goal_x, goal_y),
                    other_agent_starts=other_agent_starts,
                    min_agent_sep=1.0,
            )
                if not forced_ok:
                    print(
                        f"⚠️ Robot {self.robot_id}: 强制路线非法，回退随机安全采样 "
                        f"start=({start_x:.2f},{start_y:.2f}) goal=({goal_x:.2f},{goal_y:.2f})"
                )
            except Exception:
                forced_ok = False

        if not forced_ok and self.use_random_mode and self._valid_spawn_points:
            found_path = False
            for min_sep in [1.5, 1.0, 0.5]:
                for _ in range(60):
                    start_x, start_y = self._get_random_valid_point(
                        other_agents=other_agent_starts,
                        min_agent_sep=min_sep,
                )
                    goal_x, goal_y = self._get_random_valid_point(exclude=(start_x, start_y))
                    if self._is_valid_start_goal_pair(
                        (start_x, start_y),
                        (goal_x, goal_y),
                        other_agent_starts=other_agent_starts,
                        min_agent_sep=min_sep,
                ):
                        found_path = True
                        break
                if found_path:
                    break
            if not found_path:
                fallback_list = self._MAP_FALLBACK_POSES.get(
                    self.map_number, [((0.0, 0.0), (2.0, 2.0))])
                valid_fallbacks = [
                    pair for pair in fallback_list
                    if self._is_valid_start_goal_pair(
                        pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=0.5
                )
                ]
                fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
                (start_x, start_y), (goal_x, goal_y) = fallback
                print(f"⚠️ Reset robot_{self.robot_id}: 随机点生成失败，使用备用位置 {fallback}")
        elif not forced_ok:
            fallback_list = self._MAP_FALLBACK_POSES.get(
                self.map_number, [((0.0, 0.0), (5.0, 5.0))])
            valid_fallbacks = [
                pair for pair in fallback_list
                if self._is_valid_start_goal_pair(
                    pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=0.5
            )
            ]
            fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
            (start_x, start_y), (goal_x, goal_y) = fallback

        if forced_ok and not self._is_valid_start_goal_pair(
            (start_x, start_y),
            (goal_x, goal_y),
            other_agent_starts=other_agent_starts,
            min_agent_sep=1.0,
        ):
                fallback_list = self._MAP_COLLISION_ROUTE_LIBRARY.get(
                    self.map_number,
                    self._MAP_FALLBACK_POSES.get(self.map_number, [((0.0, 0.0), (5.0, 5.0))]),
            )
                valid_fallbacks = [
                    pair for pair in fallback_list
                    if self._is_valid_start_goal_pair(
                        pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=0.5
                )
                ]
                fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
                (start_x, start_y), (goal_x, goal_y) = fallback

        self.last_spawn_pos = (start_x, start_y)
        self.goal_pos = (goal_x, goal_y)

        if self.planner:
            path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
            if path:
                self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
                self.current_waypoint_index = 0
                self.vis.publish_waypoints(
                    self.global_waypoints,
                    robot_id=self.robot_id,
                    namespace=self.vis_namespace
            )
            else:
                print(f"⚠️ Robot {self.robot_id}: A* plan 失败 start={start_x:.2f},{start_y:.2f} "
                      f"goal={goal_x:.2f},{goal_y:.2f}，退化为直线")
                self.global_waypoints = [self.goal_pos]
                self.vis.publish_waypoints(
                    [self.goal_pos],
                    robot_id=self.robot_id,
                    namespace=self.vis_namespace
            )

        yaw = random.uniform(-3.14, 3.14)
        self._set_robot_pose(start_x, start_y, yaw)

        self.latest_scan = None
        self._scan_history.clear()
        self._obstacle_cluster_history.clear()
        self._wait_for_sim_time(0.2)

        self.prev_dist_to_goal = math.hypot(
            self.goal_pos[0] - self.current_pose['x'],
            self.goal_pos[1] - self.current_pose['y']
        )
        self._reset_stall_replan_tracker(anchor_current=True, arc_progress=0.0)

        current_target = self._get_tracking_target()
        self._publish_tracking_visuals(current_target)
        self._obs_target_state = np.array(current_target, dtype=np.float32)
        self.prev_target_point = tuple(current_target)
        self.prev_dist_to_target = math.hypot(
            current_target[0] - self.current_pose['x'],
            current_target[1] - self.current_pose['y']
        )
        self.prev_path_progress = self.path_progress
        self.prev_abs_target_angle = abs(float(self._get_target_angle(current_target)))

        return self._get_obs(), {
            'start_xy': (start_x, start_y),
            'goal_xy': (goal_x, goal_y),
            'route_source': 'forced' if forced_ok else 'random',
        }

    def _decode_action_to_cmd_vel(self, action) -> Tuple[float, float]:
        """将策略动作解码为底盘速度 (v, w)。"""
        action_id = int(np.asarray(action).reshape(-1)[0]) if isinstance(action, np.ndarray) else int(action)
        action_id = int(np.clip(action_id, 0, len(self.learned_interaction_modes) - 1))
        requested_mode = self.learned_interaction_modes[action_id]

        try:
            _, feas_result = build_interaction_action_mask(
                self,
                option_state=str(getattr(self, "_active_option_name", "go")),
                include_replan=False,
            )
            self._last_action_mask = np.ones(NUM_TRAINING_OPTIONS, dtype=np.int32)
            self._last_feasibility = feas_result
        except Exception:
            self._last_action_mask = np.ones(NUM_TRAINING_OPTIONS, dtype=np.int32)

        effective_mode = self._resolve_interaction_mode(requested_mode)
        self._policy_interaction_mode = requested_mode
        self._policy_interaction_action = action_id
        self._effective_interaction_mode = str(effective_mode)

        if effective_mode in ("detour_left", "detour_right"):
            expected_side = 'left' if effective_mode == 'detour_left' else 'right'
            if not (
                bool(getattr(self, '_detour_done', False))
                and str(getattr(self, '_detour_side', '')) == expected_side
            ):
                self._start_committed_detour(
                effective_mode,
                    partner_id=str(getattr(self, '_last_interaction_info', {}).get('partner', '')),
            )
            self._option_phase = str(getattr(self, '_detour_phase', DetourPhase.ENTER))
        else:
            self._option_phase = DetourPhase.DONE

        nominal_info = self._compute_nominal_tracking_info()
        nominal_subgoal = nominal_info.get("subgoal", self.goal_pos)
        tracking_target = self._build_option_tracking_target(effective_mode, nominal_subgoal)

        canonical_mode = CANONICAL_MODE_BY_TRAINING_OPTION.get(effective_mode, "go")
        self._executed_behavior_mode = canonical_mode
        self._last_subgoal_mode = canonical_mode
        self._cached_step_tracking_target = tuple(tracking_target)
        self._cached_step_tracking_mode = canonical_mode
        self._cached_step_tracking_step = int(self.current_step)
        self._last_nominal_subgoal = tuple(nominal_subgoal)

        return self._compute_tracking_controller_cmd(tracking_target, canonical_mode)

    def _compute_tracking_controller_cmd(
        self,
        tracking_target: Tuple[float, float],
        behavior_mode: str,
    ) -> Tuple[float, float]:
        sectors = self._scan_sector_metrics()
        front_min = float(sectors['front_min'])
        left_min = float(sectors['left_min'])
        right_min = float(sectors['right_min'])
        rear_min = float(sectors.get('rear_min', self.scan_max_range))
        interaction_ctx = self._get_interaction_context()
        turn_sign = float(interaction_ctx.get('turn_sign', 0.0))
        severity = float(np.clip(float(interaction_ctx.get('severity', 0.0)), 0.0, 1.0))
        ttc = float(interaction_ctx.get('ttc', float('inf')))
        cmd_v, cmd_w = compute_tracking_controller_cmd(
            tracking_target=tracking_target,
            current_pose=self.current_pose,
            max_forward_vel=self.max_forward_vel,
            max_reverse_vel=self.max_reverse_vel,
            max_angular_vel=self.max_angular_vel,
            corner_escape_front_dist=self.corner_escape_front_dist,
            yielding_stop_dist=self.yielding_stop_dist,
            yielding_ttc=self.yielding_ttc,
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
            turn_sign=turn_sign,
            severity=severity,
            ttc=ttc,
            gap_angle=float(self._last_gap_metrics.get('best_gap_angle', 0.0)),
            behavior_mode=behavior_mode,
        )
        if behavior_mode == "backoff" and rear_min < 0.34:
            cmd_v = 0.0
            turn_dir = 1.0 if left_min >= right_min else -1.0
            cmd_w = float(np.clip(0.35 * turn_dir, -self.max_angular_vel, self.max_angular_vel))
        return float(cmd_v), float(cmd_w)

    def apply_action(self, action, debug=False):
        self.current_step += 1
        self._last_replan_attempted = False
        self._last_replan_success = False
        self._last_replan_wall_time_sec = 0.0
        linear_vel, angular_vel = self._decode_action_to_cmd_vel(action)

        sectors = self._scan_sector_metrics()
        front_min = float(sectors['front_min'])
        left_min = float(sectors['left_min'])
        right_min = float(sectors['right_min'])

        raw_linear_vel = float(linear_vel)
        raw_angular_vel = float(angular_vel)
        interaction_reason = f"policy_{self._effective_interaction_mode}"
        self._executed_behavior_mode = str(self._effective_interaction_mode)

        self._last_control_info = {
            'front_min': front_min,
            'left_min': left_min,
            'right_min': right_min,
            'raw_linear_vel': raw_linear_vel,
            'raw_angular_vel': raw_angular_vel,
            'applied_linear_vel': float(linear_vel),
            'applied_angular_vel': float(angular_vel),
            'interaction_reason': interaction_reason or '',
        }

        if debug and abs(linear_vel) < 0.01 and abs(angular_vel) < 0.01:
            print(f"⚠️  Robot {self.robot_id}: 零动作! option={self._effective_interaction_mode} -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")

        self._publish_vel(linear_vel, angular_vel)

    def get_step_result(self):
            """
            融合增强版奖励函数:
            保留了原版的【A*路径进度】与【防死锁脱困机制】
            引入了新的【严惩南辕北辙】与【靠右会车社会规范】
            """
            reward = 0.0
            done = False
            truncated = False
            info = {}

            if (
                self._cached_step_tracking_target is not None
                and int(self._cached_step_tracking_step) == int(self.current_step)
            ):
                current_target = tuple(self._cached_step_tracking_target)
                self._last_subgoal_mode = str(self._cached_step_tracking_mode)
            else:
                current_target = self._get_tracking_target()
                self._cached_step_tracking_target = tuple(current_target)
                self._cached_step_tracking_mode = str(self._last_subgoal_mode)
                self._cached_step_tracking_step = int(self.current_step)
            self._publish_tracking_visuals(current_target)
            obs = self._get_obs(target_override=current_target)

            dist_to_goal = math.hypot(
                self.goal_pos[0] - self.current_pose['x'],
                self.goal_pos[1] - self.current_pose['y'],
            )
            dist_to_target = math.hypot(
                current_target[0] - self.current_pose['x'],
                current_target[1] - self.current_pose['y'],
            )
            target_angle = self._get_target_angle(current_target)
            abs_target_angle = abs(float(target_angle))
            goal_progress_delta = 0.0
            if self.prev_dist_to_goal is not None:
                goal_progress_delta = float(np.clip(self.prev_dist_to_goal - dist_to_goal, -1.0, 1.0))
            local_goal_progress_delta = 0.0
            if self.prev_dist_to_target is not None:
                local_goal_progress_delta = float(np.clip(self.prev_dist_to_target - dist_to_target, -1.0, 1.0))
            self._last_goal_progress_delta = float(goal_progress_delta)
            self._last_local_goal_progress_delta = float(local_goal_progress_delta)

            # 当前速度状态
            signed_linear_speed = float(getattr(self, 'current_vel_x', 0.0))
            signed_angular_vel = float(getattr(self, 'current_vel_w', 0.0))
            forward_speed = max(signed_linear_speed, 0.0)
            turn_rate = float(getattr(self, 'current_vel_w', 0.0))
            abs_turn_rate = abs(turn_rate)
            interaction_ctx = self._get_interaction_context()
            ttc_min = float(interaction_ctx.get('ttc', float('inf')))
            prev_ttc_min = float(getattr(self, '_last_ttc_min', float('inf')))
            ttc_delta = 0.0
            if math.isfinite(prev_ttc_min):
                current_ttc_for_delta = float(ttc_min)
                if not math.isfinite(current_ttc_for_delta):
                    current_ttc_for_delta = max(float(self.predictive_social_ttc_safe), prev_ttc_min)
                ttc_delta = float(np.clip(current_ttc_for_delta - prev_ttc_min, -5.0, 5.0))
            interaction_in_conflict = float(interaction_ctx.get('in_conflict', 0.0)) > 0.5
            interaction_mode = str(getattr(self, '_executed_behavior_mode', 'go'))
            policy_mode_for_credit = str(getattr(self, '_policy_interaction_mode', interaction_mode))
            self._update_path_projection()

            interaction_mode_reward = 0.0
            interaction_mode_penalty = 0.0

            # ==========================================
            # 1. 核心导航：沿 A* 路径的进度 (恢复原版的稳定性)
            # ==========================================
            attractive_gain = 1.6
            progress_reward = 0.0
            path_progress_reward = 0.0
            goal_progress_reward = 0.0
            
            if self.prev_path_progress is not None:
                progress_reward = attractive_gain * self.progress_reward_scale * (
                    self.path_progress - self.prev_path_progress
            )
            elif self.prev_dist_to_target is not None:
                progress_reward = attractive_gain * self.progress_reward_scale * (
                    self.prev_dist_to_target - dist_to_target
            )
            if self.prev_path_progress is not None:
                path_progress_reward = attractive_gain * self.path_progress_reward_scale * (
                    self.path_progress - self.prev_path_progress
            )
            if self.prev_dist_to_goal is not None:
                goal_progress_reward = self.goal_progress_reward_scale * (
                    self.prev_dist_to_goal - dist_to_goal
            )

            # ==========================================
            # 2. 朝向惩罚与奖励 (融入“防南辕北辙漏洞”)
            # ==========================================
            heading_reward = 0.0
            wrong_direction_penalty = 0.0
            turn_alignment_reward = 0.0
            subgoal_progress_reward = 0.0
            yield_compliance_reward = 0.0
            risk_aware_forward_penalty = 0.0
            safe_turn_reward = 0.0
            head_on_avoidance_reward = 0.0
            detour_active = self._last_subgoal_mode in {
                'detour',
                'gap_detour',
                'deadlock',
                'yield',
                'wait',
                'backoff',
                'replan',
                'corner_escape',
            }

            if abs_target_angle <= math.pi / 2.0:
                # 车头对准时：有前向速度才给 heading_reward
                if forward_speed > 0.02:
                    heading_reward = 0.10 * math.cos(target_angle)
                
                # 恢复原版的对齐奖励，帮助更丝滑地切入弯道
                turnaround_gate = float(np.clip((abs_target_angle - 1.0) / 1.2, 0.0, 1.0))
                if self.prev_abs_target_angle is not None:
                    angle_improvement = float(self.prev_abs_target_angle - abs_target_angle)
                    if angle_improvement > 0.0 and turnaround_gate > 0.0:
                        turn_alignment_reward = 0.18 * angle_improvement * turnaround_gate
            else:
                # 车头背向时：严惩前进，没收进度分
                if forward_speed > 0.02:
                    wrong_direction_penalty = -0.30 * forward_speed * (abs_target_angle / math.pi)
                    progress_reward = min(0.0, progress_reward)
                    path_progress_reward = min(0.0, path_progress_reward)
                    goal_progress_reward = min(0.0, goal_progress_reward)

            if self.prev_dist_to_target is not None:
                subgoal_progress_reward = self.subgoal_progress_reward_scale * (
                    self.prev_dist_to_target - dist_to_target
            )
            if detour_active:
                progress_reward *= self.detour_progress_relax
                path_progress_reward *= self.detour_progress_relax
            if interaction_in_conflict:
                prev_mode = self._last_interaction_mode
                if prev_mode not in ('idle', interaction_mode) and interaction_mode != 'idle':
                    interaction_mode_penalty = -0.025
                self._last_interaction_turn_sign = float(interaction_ctx.get('turn_sign', 0.0))
            else:
                self._last_interaction_turn_sign = 0.0

            # ==========================================
            # 3. 会车规范与循迹 (融入“靠右行驶”)
            # ==========================================
            lateral_penalty = -0.05 * min(abs(float(self.current_lateral_error)), 1.0)
            social_keep_right_reward = 0.0
            if detour_active:
                lateral_penalty *= 0.15
            
            # 寻找最近邻居
            min_n_dist = float('inf')
            nearest_n_dx, nearest_n_dy = 0.0, 0.0
            if hasattr(self, 'parent_env'):
                my_pos = np.array([self.current_pose['x'], self.current_pose['y']])
                for aid, pos in self.parent_env.robot_positions.items():
                    if aid != f"agent_{self.robot_id}":
                        d = float(np.linalg.norm(pos - my_pos))
                        if d < min_n_dist:
                            min_n_dist, nearest_n_dx, nearest_n_dy = d, pos[0] - my_pos[0], pos[1] - my_pos[1]

            # 1.5米内发生会车：鼓励靠右，但不要完全丢掉中心线约束。
            if min_n_dist < 1.5:
                yaw = self.current_pose['yaw']
                n_angle = math.atan2(nearest_n_dy, nearest_n_dx)
                rel_n_angle = (n_angle - yaw + math.pi) % (2 * math.pi) - math.pi
                
                if abs(rel_n_angle) < math.pi / 3.0:
                    dist_factor = max(0.0, (1.5 - min_n_dist) / 1.5)
                    social_keep_right_reward = 0.15 * dist_factor * rel_n_angle
                    if forward_speed < 0.05:
                        social_keep_right_reward *= 0.2

            ma_ttc_penalty = 0.0
            if hasattr(self, 'parent_env'):
                my_pos = np.array([self.current_pose['x'], self.current_pose['y']])
                my_vel = np.array([
                    self.current_vel_x * math.cos(self.current_pose['yaw']),
                    self.current_vel_x * math.sin(self.current_pose['yaw'])
                ])
                
                for aid, pos in self.parent_env.robot_positions.items():
                    if aid != f"agent_{self.robot_id}":
                        neighbor_vel = self.parent_env.robot_velocities[aid]
                        
                        rel_pos = pos - my_pos
                        rel_vel = neighbor_vel - my_vel # 相对速度矢量
                        dist = np.linalg.norm(rel_pos)
                        
                        if dist < 1.5:
                            # 相对速度在相对位置向量上的投影 (如果<0说明正在靠近)
                            approach_speed = -np.dot(rel_pos, rel_vel) / (dist + 1e-6)
                            
                            if approach_speed > 0.05: # 如果正在以明显速度靠近
                                ttc = dist / approach_speed
                                safe_ttc = float(self.predictive_social_ttc_safe)
                                if ttc < safe_ttc:
                                    # 离得越近、靠近越快，惩罚越大（平方级）
                                    ma_ttc_penalty -= 0.15 * ((safe_ttc - ttc) / safe_ttc) ** 2

            # ==========================================
            # 4. 雷达扫描与脱困机制 (恢复原版的防死锁逻辑)
            # ==========================================
            sectors = self._scan_sector_metrics()
            min_dist = float(sectors.get('min_dist', 10.0))
            front_min = float(sectors.get('front_min', min_dist))
            left_min = float(sectors.get('left_min', min_dist))
            right_min = float(sectors.get('right_min', min_dist))
            corridor_span = float(left_min + right_min)
            narrow_span_ref = max(0.45, 2.4 * float(self.subgoal_min_side_clearance))
            corridor_narrow_ratio = float(np.clip(
                (narrow_span_ref - corridor_span) / max(narrow_span_ref, 1e-6),
                0.0,
                1.0,
            ))
            speed_norm = float(np.clip(forward_speed / max(self.max_forward_vel, 1e-6), 0.0, 1.0))
            path_heading_error = float(
                (float(self.current_path_heading) - float(self.current_pose['yaw']) + math.pi) % (2.0 * math.pi) - math.pi
            )
            path_heading_align = float(math.cos(path_heading_error))

            if forward_speed > 0.02 and path_heading_align > 0.0:
                heading_reward += 0.08 * (0.35 + 0.65 * corridor_narrow_ratio) * speed_norm * path_heading_align
            wrong_direction_penalty += -0.10 * corridor_narrow_ratio * speed_norm * float(
            np.clip(abs(path_heading_error) / (0.5 * math.pi), 0.0, 1.0)
            )
            lateral_penalty = -0.05 * (1.0 + 1.8 * corridor_narrow_ratio) * min(
                abs(float(self.current_lateral_error)),
                1.0,
            )
            if detour_active:
                lateral_penalty *= (0.45 + 0.35 * (1.0 - corridor_narrow_ratio))
            if min_n_dist < 1.5:
                lateral_penalty *= (0.25 + 0.75 * corridor_narrow_ratio)

            corner_escape_active = self._last_subgoal_mode == 'corner_escape'
            turn_escape_reward = 0.0
            corner_escape_reward = 0.0
            front_penalty_relax = 1.0
            side_penalty_relax = 1.0
            effective_time_penalty = float(self.time_penalty)

            # 如果面壁思过被卡住了，引导它原地转弯逃脱
            in_place_turn = (forward_speed < 0.04) and (abs_turn_rate > 0.20)
            narrow_front_dist = 0.34
            front_blocked_ratio = float(np.clip((narrow_front_dist - front_min) / narrow_front_dist, 0.0, 1.0))
            heading_need = float(np.clip(abs_target_angle / math.pi, 0.0, 1.0))

            if in_place_turn and front_blocked_ratio > 0.0:
                preferred_turn = self._corner_escape_turn_sign or (1.0 if left_min >= right_min else -1.0)
                turning_dir = float(np.sign(turn_rate))
                dir_match = 1.0 if (turning_dir * preferred_turn) > 0.0 else 0.35
                turn_escape_reward = 0.12 * front_blocked_ratio * heading_need * dir_match
                front_penalty_relax = 0.45
                side_penalty_relax = 0.70
                effective_time_penalty *= 0.4

            if corner_escape_active and front_blocked_ratio > 0.0:
                desired_turn = self._corner_escape_turn_sign or (1.0 if left_min >= right_min else -1.0)
                turn_match = 1.0 if (abs_turn_rate > 0.08 and math.copysign(1.0, turn_rate) == math.copysign(1.0, desired_turn)) else 0.0
                turn_rate_norm = float(np.clip(abs_turn_rate / max(self.max_angular_vel, 1e-6), 0.0, 1.0))
                low_speed_gate = float(np.clip(
                    (self.corner_escape_speed_thresh - forward_speed) / max(self.corner_escape_speed_thresh, 1e-6),
                    0.0,
                    1.0,
            ))
                corner_escape_reward = 0.14 * front_blocked_ratio * heading_need * (
                    0.55 * low_speed_gate + 0.45 * turn_match * turn_rate_norm
            )
                front_penalty_relax = min(front_penalty_relax, 0.70)
                side_penalty_relax = min(side_penalty_relax, 0.45)
                effective_time_penalty *= 0.55

            clearance_ctx = self._compute_clearance_context(
            front_min,
            left_min,
            right_min,
                front_penalty_relax,
                side_penalty_relax,
                corner_escape_active,
            )
            side_min = float(clearance_ctx['side_min'])
            front_blocked_ratio = float(clearance_ctx['front_blocked_ratio'])
            front_close_ratio = float(clearance_ctx['front_close_ratio'])
            side_close_ratio = float(clearance_ctx['side_close_ratio'])
            front_potential_penalty = float(clearance_ctx['front_potential_penalty'])
            side_wall_penalty = float(clearance_ctx['side_wall_penalty'])
            front_close_penalty = float(clearance_ctx['front_close_penalty'])
            side_close_penalty = float(clearance_ctx['side_close_penalty'])
            obstacle_penalty = front_potential_penalty + side_wall_penalty
            close_obstacle_penalty = front_close_penalty + side_close_penalty

            # ==========================================
            # 5. 精细化斥力场 (恢复原版细腻的碰撞检测)
            # ==========================================
            predictive_social_risk = float(self._last_predictive_metrics.get('social_risk', 0.0))
            predictive_front_risk = float(self._last_predictive_metrics.get('front_risk', 0.0))
            social_proximity_ratio = 0.0
            if math.isfinite(min_n_dist):
                social_proximity_ratio = float(np.clip((self.predictive_social_range - min_n_dist) / self.predictive_social_range, 0.0, 1.0))
            social_summary = self._compute_social_risk_summary()
            interaction_social_risk = float(social_summary['social_risk'])
            front_proximity_ratio = max(front_blocked_ratio, front_close_ratio)
            predictive_social_penalty = -self.predictive_social_penalty_scale * (
                0.35 * predictive_social_risk + 0.65 * (predictive_social_risk ** 2)
            ) * (0.40 + 0.60 * social_proximity_ratio)
            predictive_front_penalty = -self.predictive_front_penalty_scale * (
                0.35 * predictive_front_risk + 0.65 * (predictive_front_risk ** 2)
            ) * (0.40 + 0.60 * front_proximity_ratio)
            social_proximity_penalty = -self.social_proximity_risk_scale * social_proximity_ratio * predictive_social_risk
            combined_risk = max(
                predictive_front_risk,
                predictive_social_risk,
                interaction_social_risk,
            front_blocked_ratio,
                front_close_ratio,
            )
            risk_aware_forward_penalty = -self.risk_aware_forward_penalty_scale * forward_speed * (
                0.60 * (predictive_front_risk ** 2) + 0.40 * (predictive_social_risk ** 2)
            )
            gap_angle = float(self._last_gap_metrics.get('best_gap_angle', 0.0))
            if combined_risk > 0.05 and abs_turn_rate > 0.05 and abs(gap_angle) > 0.08:
                turn_dir = math.copysign(1.0, turn_rate)
                desired_dir = math.copysign(1.0, gap_angle)
                if turn_dir == desired_dir:
                    safe_turn_reward = self.safe_turn_reward_scale * combined_risk * min(
                        1.0,
                        abs_turn_rate / max(self.max_angular_vel, 1e-6),
                ) * max(0.4, float(self._last_gap_metrics.get('best_gap_clearance', 0.0)))

            conflict = {
                'partner_id': str(interaction_ctx.get('partner', '')),
                'partner_mode': str(interaction_ctx.get('mode', 'idle')),
                'dist': float(interaction_ctx.get('partner_dist', float('inf'))),
                'closing_speed': float(interaction_ctx.get('closing_speed', 0.0)),
                'ttc': float(interaction_ctx.get('ttc', float('inf'))),
                'turn_sign': float(interaction_ctx.get('turn_sign', 0.0)),
                'severity': float(interaction_ctx.get('severity', 0.0)),
            } if interaction_in_conflict else None
            high_level_nav_reward = 0.0
            high_level_interaction_reward = 0.0
            high_level_safety_reward = 0.0
            high_level_efficiency_penalty = 0.0
            high_level_policy_penalty = 0.0
            social_risk_delta = 0.0
            clear_reward = 0.0
            blocked_score = max(front_blocked_ratio, front_close_ratio)
            stuck_score = self._compute_stuck_score(front_blocked_ratio)
            method3_terms = None
            clear_reward = 0.0
            blocked_score = max(front_blocked_ratio, front_close_ratio)
            stuck_score = self._compute_stuck_score(front_blocked_ratio)
            method3_window_progress_reward = 0.0
            method3_window_path_progress_reward = 0.0
            method3_window_goal_progress_reward = 0.0
            option_progress_reward_outcome = 0.0
            while self._recent_replan_steps and (self.current_step - self._recent_replan_steps[0]) > self.replan_window_steps:
                self._recent_replan_steps.popleft()
            recent_replan_count = sum(
                1 for step in self._recent_replan_steps if step < self.current_step
            )
            replan_cost = 0.0
            replan_freq_penalty = 0.0
            replan_time_penalty = 0.0
            wait_age_norm = float(np.clip(float(interaction_ctx.get('wait_age_norm', 0.0)), 0.0, 1.0))
            method3_window_metrics = self._get_method3_credit_window_metrics(
                dist_to_target=float(dist_to_target),
                dist_to_goal=float(dist_to_goal),
                path_progress=float(self.path_progress),
            social_risk=float(interaction_social_risk),
            blocked_score=float(blocked_score),
            stuck_score=float(stuck_score),
            )
            method3_window_progress_reward = attractive_gain * self.progress_reward_scale * float(method3_window_metrics['progress_delta'])
            method3_window_path_progress_reward = attractive_gain * self.path_progress_reward_scale * float(method3_window_metrics['path_progress_delta'])
            method3_window_goal_progress_reward = self.goal_progress_reward_scale * float(method3_window_metrics['goal_progress_delta'])
            method3_terms = compute_method3_reward_terms(
                policy_mode=policy_mode_for_credit,
                    interaction_social_risk=interaction_social_risk,
                progress_delta_window=float(method3_window_metrics['progress_delta']),
                path_progress_delta_window=float(method3_window_metrics['path_progress_delta']),
                goal_progress_delta_window=float(method3_window_metrics['goal_progress_delta']),
                social_risk_delta_window=float(method3_window_metrics['social_risk_delta']),
                clear_reward_window=float(method3_window_metrics['clear_reward']),
                blocked_score=blocked_score,
                stuck_score=stuck_score,
                    wait_age_norm=wait_age_norm,
                    front_close_ratio=front_close_ratio,
                    side_close_ratio=side_close_ratio,
                    stall_elapsed_sec=float(getattr(self, '_stall_elapsed_sec', 0.0)),
                    stall_global_replan_sec=float(self.stall_global_replan_sec),
                    last_subgoal_mode=str(self._last_subgoal_mode),
                    replan_attempted=bool(self._last_replan_attempted),
                    replan_recent_count=int(recent_replan_count),
                    replan_wall_time_sec=float(self._last_replan_wall_time_sec),
                    replan_time_budget_sec=float(self.replan_time_budget_sec),
                    replan_fixed_cost=float(self.replan_fixed_cost),
                    replan_freq_cost=float(self.replan_freq_cost),
                    replan_time_cost=float(self.replan_time_cost),
            )
            social_risk_delta = float(method3_terms.social_risk_delta)
            clear_reward = float(method3_terms.clear_reward)
            blocked_score = float(method3_terms.blocked_score)
            stuck_score = float(method3_terms.stuck_score)
            replan_cost = float(method3_terms.replan_cost)
            replan_freq_penalty = float(method3_terms.replan_freq_penalty)
            replan_time_penalty = float(method3_terms.replan_time_penalty)
            interaction_mode_reward += float(method3_terms.interaction_mode_reward)
            interaction_mode_penalty += float(method3_terms.interaction_mode_penalty)
            high_level_interaction_reward += float(method3_terms.high_level_interaction_reward)
            high_level_safety_reward += float(method3_terms.high_level_safety_reward)
            high_level_efficiency_penalty += float(method3_terms.high_level_efficiency_penalty)
            high_level_policy_penalty += float(method3_terms.high_level_policy_penalty)

            # ── New: Option Outcome Reward (Phase 6/7) ──
            try:
                _eff_mode = str(getattr(self, '_effective_interaction_mode', policy_mode_for_credit))
                _opt_elapsed = max(0, int(self.current_step) - int(getattr(self, '_active_option_start_step', 0)))
                _opt_total = max(1, int(getattr(self, '_active_option_duration_steps', 1)))
                _opt_just_done = True
                # Path-projection-aware progress (avoids Euclidean-goal-only trap)
                _proj_prog_delta = float(getattr(self, '_path_projection_progress_delta', 0.0))
                _proj_prog_window = float(getattr(self, '_path_projection_progress_window', 0.0))
                _guide_prog_delta = float(getattr(self, '_guide_target_progress_delta', 0.0))
                _proj_thresh = float(self.proj_progress_threshold)
                _goal_thresh = float(self.goal_progress_threshold)
                _local_thresh = float(self.local_progress_threshold)
                _guide_thresh = float(self.guide_progress_threshold)
                _progress_pos = (
                    _proj_prog_delta > _proj_thresh
                    or goal_progress_delta > _goal_thresh
                    or local_goal_progress_delta > _local_thresh
                    or _guide_prog_delta > _guide_thresh
                )
                _progress_source = "none"
                if _proj_prog_delta > _proj_thresh:
                    _progress_source = "projection"
                elif goal_progress_delta > _goal_thresh:
                    _progress_source = "goal"
                elif local_goal_progress_delta > _local_thresh:
                    _progress_source = "local_goal"
                elif _guide_prog_delta > _guide_thresh:
                    _progress_source = "guide_target"
                self._last_progress_source = str(_progress_source)
                self._last_progress_source_id = {
                    "none": 0.0,
                    "projection": 1.0,
                    "goal": 2.0,
                    "local_goal": 3.0,
                    "guide_target": 4.0,
                }.get(_progress_source, 0.0)
                self._last_progress_positive = bool(_progress_pos)

                _option_snapshot = dict(getattr(self, '_option_start_snapshot', {}) or {})
                _start_ttc = float(_option_snapshot.get('ttc_min', float('inf')))
                _ttc_improvement = 0.0
                if math.isfinite(_start_ttc) and math.isfinite(float(ttc_min)):
                    _ttc_improvement = float(np.clip(float(ttc_min) - _start_ttc, -5.0, 5.0))
                _front_blocked_delta = float(front_blocked_ratio) - float(
                    _option_snapshot.get('front_blocked_ratio', front_blocked_ratio)
                )
                _front_safe = 0.40
                _start_front_obstacle_risk = float(_option_snapshot.get('front_obstacle_risk', 0.0))
                _current_front_obstacle_risk = float(
                    np.clip((_front_safe - max(0.0, float(front_min))) / _front_safe, 0.0, 1.0)
                )
                _obstacle_risk_drop = float(_start_front_obstacle_risk - _current_front_obstacle_risk)
                _risk_dropped = bool(
                    float(method3_window_metrics['social_risk_delta']) > 0.02
                    or _ttc_improvement > 0.10
                    or _front_blocked_delta < -0.05
                    or _obstacle_risk_drop > 0.05
                )
                _pair_comp = False
                if conflict and conflict.get('partner_id'):
                    _p_mode = str(conflict.get('partner_mode', ''))
                    _pair_comp = (
                        (_eff_mode == 'go' and _p_mode == 'wait')
                        or (_eff_mode == 'wait' and _p_mode == 'go')
                        or (_eff_mode in ('detour_left', 'detour_right') and _p_mode in ('wait', 'backoff'))
                    )
                opt_outcome = compute_option_outcome_reward(
                    effective_mode=_eff_mode,
                    policy_mode=str(getattr(self, '_policy_interaction_mode', _eff_mode)),
                    progress_delta=float(method3_window_metrics['progress_delta']),
                    path_progress_delta=float(method3_window_metrics['path_progress_delta']),
                    goal_progress_delta=float(goal_progress_delta),
                    local_goal_progress_delta=float(local_goal_progress_delta),
                    path_projection_progress_delta=float(_proj_prog_delta),
                    path_projection_progress_window=float(_proj_prog_window),
                    guide_target_progress_delta=float(_guide_prog_delta),
                    closest_dist_to_path=float(getattr(self, '_closest_dist_to_path', 0.0)),
                    cross_track_error=float(getattr(self, '_cross_track_error', 0.0)),
                    social_risk=float(interaction_social_risk),
                    social_risk_delta=float(method3_window_metrics['social_risk_delta']),
                    front_risk=float(getattr(self, '_last_predictive_metrics', {}).get('front_risk', 0.0)),
                    ttc_min=float(ttc_min),
                    ttc_delta=float(ttc_delta),
                    front_min=float(front_min),
                    left_min=float(left_min),
                    right_min=float(right_min),
                    front_blocked_ratio=float(front_blocked_ratio),
                    blocked_score=float(blocked_score),
                    stuck_score=float(stuck_score),
                    option_elapsed=_opt_elapsed,
                    option_duration_steps=_opt_total,
                    option_just_completed=_opt_just_done,
                    option_success=False,
                    option_failed=False,
                    detour_lateral_displacement=float(getattr(self, '_detour_lateral_displacement', 0.0)),
                    applied_angular_vel=float(signed_angular_vel),
                    applied_linear_vel=float(signed_linear_speed),
                    action_was_feasible=True,
                    policy_vs_effective_mismatch=(str(getattr(self, '_policy_interaction_mode', '')) != _eff_mode),
                    pair_partner_id=str(conflict.get('partner_id', '') if conflict else ''),
                    pair_partner_mode=str(conflict.get('partner_mode', '') if conflict else ''),
                    pair_dist=float(conflict.get('dist', float('inf')) if conflict else float('inf')),
                    pair_closing_speed=float(conflict.get('closing_speed', 0.0) if conflict else 0.0),
                    pair_ttc=float(conflict.get('ttc', float('inf')) if conflict else float('inf')),
                    pair_mode_complementary=_pair_comp,
                    progress_positive=_progress_pos,
                    progress_source=_progress_source,
                    risk_reduced=_risk_dropped,
                    front_blocked_ratio_delta=float(_front_blocked_delta),
                    obstacle_risk_drop=float(_obstacle_risk_drop),
                    safe_turn_reward_scale=float(self.safe_turn_reward_scale),
                    collision_penalty_base=float(self.collision_penalty),
                )
                option_progress_reward_outcome += float(opt_outcome.option_progress_reward)
                interaction_mode_reward += float(opt_outcome.interaction_mode_reward)
                interaction_mode_penalty += float(opt_outcome.interaction_mode_penalty)
                self._last_option_outcome_terms = opt_outcome
                self._last_cross_track_penalty = float(opt_outcome.cross_track_penalty)
                self._last_positive_path_projection_progress = float(opt_outcome.positive_path_projection_progress)
                self._last_negative_path_projection_progress = float(opt_outcome.negative_path_projection_progress)
                self._last_option_progress_reward = float(opt_outcome.option_progress_reward)
                self._last_obstacle_risk_drop = float(opt_outcome.obstacle_risk_drop)
                self._last_risk_reduced = bool(opt_outcome.risk_reduced > 0.5)
                self._last_ttc_improvement = float(opt_outcome.ttc_improvement)
            except Exception:
                self._last_option_outcome_terms = OptionOutcomeRewardTerms()
                self._last_cross_track_penalty = 0.0
                self._last_positive_path_projection_progress = 0.0
                self._last_negative_path_projection_progress = 0.0
                self._last_option_progress_reward = 0.0
                self._last_obstacle_risk_drop = 0.0
                self._last_risk_reduced = False
                self._last_ttc_improvement = 0.0

            head_on_avoidance_reward = self.reward_head_on_avoidance(
                conflict,
                signed_linear_speed,
                turn_rate,
            front_min,
            left_min,
            right_min,
            )
            conflict_severity = float(conflict.get('severity', 0.0)) if conflict is not None else 0.0

            # ==========================================
            # 6. 事件与汇总
            # ==========================================
            collision_penalty = 0.0
            goal_bonus = 0.0
            
            if self._check_collision_event(min_dist, info):
                collision_penalty = -self.collision_penalty
                info['event'] = 'collision'
                if self.collision_ends_episode: done = True

            if dist_to_goal < 0.35:
                goal_bonus = self.goal_reward
                done = True
                info['event'] = 'goal'

            if self.current_step >= self.max_episode_steps:
                truncated = True
            sectors_now = self._scan_sector_metrics()
            front_left_min = float(sectors_now.get('front_left_min', front_min))
            front_right_min = float(sectors_now.get('front_right_min', front_min))
            local_head_on_pass_event = 1.0 if (
                self._detour_done
                and (
                    (conflict is not None and float(conflict.get('dist', float('inf'))) > (self.yielding_soft_dist + 0.25))
                    or float(interaction_social_risk) < 0.15
                )
            ) else 0.0
            self._last_local_head_on_pass_event = float(local_head_on_pass_event)
            potential_terms = compute_interaction_potential_reward(
                self,
                dist_to_target=float(dist_to_target),
                dist_to_goal=float(dist_to_goal),
                front_min=float(front_min),
                left_min=float(left_min),
                right_min=float(right_min),
                front_left_min=float(front_left_min),
                front_right_min=float(front_right_min),
                social_risk_max=float(interaction_social_risk),
                ttc_min=float(ttc_min),
                front_risk=float(predictive_front_risk),
                cross_track_error=float(getattr(self, '_cross_track_error', 0.0)),
                effective_mode=str(getattr(self, '_effective_interaction_mode', interaction_mode)),
                applied_linear_vel=float(signed_linear_speed),
                applied_angular_vel=float(signed_angular_vel),
                local_goal_progress_delta=float(local_goal_progress_delta),
                goal_progress_delta=float(goal_progress_delta),
                path_projection_progress_delta=float(getattr(self, '_path_projection_progress_delta', 0.0)),
                stuck_score=float(stuck_score),
                detour_active=bool(getattr(self, '_detour_active', False) or getattr(self, '_detour_suppress_rolling', False)),
                detour_done=bool(getattr(self, '_detour_done', False)),
                head_on_pass_event=bool(local_head_on_pass_event > 0.5),
                collision=bool(info.get('event') == 'collision'),
                timeout=bool(truncated),
                goal_reached=bool(info.get('event') == 'goal'),
                config=self.interaction_potential_overrides,
            )
            option_terms = ClassicNavigationRewardTerms(
                progress_reward=0.0,
                heading_reward=0.0,
                obstacle_penalty=0.0,
                predictive_penalty=0.0,
                time_penalty=0.0,
                terminal_reward=0.0,
                total_reward=0.0,
            )
            classic_reward_terms = compute_classic_navigation_reward(
                path_projection_progress_delta=float(getattr(self, '_path_projection_progress_delta', 0.0)),
                target_angle=float(target_angle),
                forward_speed=float(forward_speed),
                front_potential_penalty=float(front_potential_penalty),
                side_wall_penalty=float(side_wall_penalty),
                close_obstacle_penalty=float(close_obstacle_penalty),
                predictive_social_penalty=float(predictive_social_penalty),
                predictive_front_penalty=float(predictive_front_penalty),
                time_penalty=float(self.time_penalty),
                goal_reached=bool(info.get('event') == 'goal'),
                collision=bool(info.get('event') == 'collision'),
                timeout=bool(truncated),
                goal_reward=float(self.goal_reward),
                collision_penalty=float(self.collision_penalty),
                timeout_penalty=0.0,
                progress_weight=float(max(self.goal_progress_reward_scale, 1.0)),
                heading_weight=0.08,
            )
            path_tracking_reward = float(
                classic_reward_terms.progress_reward + classic_reward_terms.heading_reward
            )
            avoidance_reward = float(
                classic_reward_terms.obstacle_penalty + classic_reward_terms.predictive_penalty
            )
            reward = float(classic_reward_terms.total_reward)
            high_level_nav_reward = float(path_tracking_reward)
            high_level_interaction_reward = 0.0
            high_level_safety_reward = 0.0
            high_level_efficiency_penalty = 0.0
            high_level_policy_penalty = 0.0
            effective_time_penalty = float(-classic_reward_terms.time_penalty)
            risk_signal = float(max(predictive_front_risk, predictive_social_risk, interaction_social_risk))
            risk_gate = float(risk_signal)
            navigation_scale = 1.0
            avoidance_scale = 1.0

            # 更新历史变量与 Info
            self.prev_dist_to_goal = dist_to_goal
            self.prev_dist_to_target = dist_to_target
            self.prev_path_progress = self.path_progress
            self.prev_target_point = tuple(current_target)
            self.prev_abs_target_angle = abs_target_angle
            self._last_interaction_mode = interaction_mode if interaction_in_conflict else 'idle'
            self._last_social_risk = float(interaction_social_risk)
            self._last_front_blocked_ratio = float(blocked_score)
            self._last_stuck_score = float(stuck_score)
            self._last_ttc_min = float(ttc_min)
            self._append_method3_credit_snapshot(
                dist_to_target=float(dist_to_target),
                dist_to_goal=float(dist_to_goal),
                path_progress=float(self.path_progress),
                social_risk=float(interaction_social_risk),
                blocked_score=float(blocked_score),
                stuck_score=float(stuck_score),
            )
            option_terms = getattr(self, '_last_option_outcome_terms', OptionOutcomeRewardTerms())
            reported_safe_turn_reward = float(option_terms.safe_turn_reward)

            info.update({
                'success_flag': 1.0 if info.get('event') == 'goal' else 0.0,
                'collision_flag': 1.0 if info.get('event') == 'collision' else 0.0,
                'timeout_flag': 1.0 if truncated or info.get('event') == 'timeout' else 0.0,
                'classic_progress_reward': float(classic_reward_terms.progress_reward),
                'classic_heading_reward': float(classic_reward_terms.heading_reward),
                'classic_obstacle_penalty': float(classic_reward_terms.obstacle_penalty),
                'classic_predictive_penalty': float(classic_reward_terms.predictive_penalty),
                'classic_time_penalty': float(classic_reward_terms.time_penalty),
                'classic_terminal_reward': float(classic_reward_terms.terminal_reward),
                'reward_total': float(reward),
                'path_tracking_reward': float(path_tracking_reward),
                'avoidance_reward': float(avoidance_reward),
                'progress_reward': float(progress_reward),
                'path_progress_reward': float(path_progress_reward),
                'goal_progress_reward': float(goal_progress_reward),
                'heading_reward': float(heading_reward),
                'subgoal_progress_reward': float(subgoal_progress_reward),
                'lateral_penalty': float(lateral_penalty),
                'social_keep_right_reward': float(social_keep_right_reward),
                'wrong_dir_penalty': float(wrong_direction_penalty),
                'turn_escape_reward': float(turn_escape_reward),
                'corner_escape_reward': float(corner_escape_reward),
                'corner_escape_active': 1.0 if corner_escape_active else 0.0,
                'stuck_score': float(stuck_score),
                'clear_reward': float(clear_reward),
                'obstacle_penalty': float(obstacle_penalty),
                'close_obstacle_penalty': float(close_obstacle_penalty),
                'front_close_penalty': float(front_close_penalty),
                'side_close_penalty': float(side_close_penalty),
                'predictive_social_risk': float(predictive_social_risk),
                'social_risk': float(interaction_social_risk),
                'social_risk_delta': float(social_risk_delta),
                'window_progress_reward': float(method3_window_progress_reward),
                'window_path_progress_reward': float(method3_window_path_progress_reward),
                'window_goal_progress_reward': float(method3_window_goal_progress_reward),
                'predictive_front_risk': float(predictive_front_risk),
                'predictive_social_penalty': float(predictive_social_penalty),
                'predictive_front_penalty': float(predictive_front_penalty),
                'social_proximity_penalty': float(social_proximity_penalty),
                'predictive_penalty': float(predictive_social_penalty + predictive_front_penalty),
                'yield_compliance_reward': float(yield_compliance_reward),
                'interaction_mode_reward': float(interaction_mode_reward),
                'interaction_mode_penalty': float(interaction_mode_penalty),
                'risk_aware_forward_penalty': float(risk_aware_forward_penalty),
                'safe_turn_reward': float(reported_safe_turn_reward),
                'head_on_avoidance_reward': float(head_on_avoidance_reward),
                'replan_cost': float(replan_cost),
                'replan_freq_penalty': float(replan_freq_penalty),
                'replan_time_penalty': float(replan_time_penalty),
                'replan_wall_time_sec': float(self._last_replan_wall_time_sec),
                'replan_attempted': 1.0 if self._last_replan_attempted else 0.0,
                'replan_success': 1.0 if self._last_replan_success else 0.0,
                'recent_replan_count': float(recent_replan_count),
                'reward_risk_signal': float(risk_signal),
                'reward_risk_gate': float(risk_gate),
                'reward_navigation_scale': float(navigation_scale),
                'reward_avoidance_scale': float(avoidance_scale),
                'effective_time_penalty': float(effective_time_penalty),
                'high_level_nav_reward': float(high_level_nav_reward),
                'high_level_interaction_reward': float(high_level_interaction_reward),
                'high_level_safety_reward': float(high_level_safety_reward),
                'high_level_efficiency_penalty': float(high_level_efficiency_penalty),
                'high_level_policy_penalty': float(high_level_policy_penalty),
                'best_gap_angle': float(self._last_gap_metrics.get('best_gap_angle', 0.0)),
                'best_gap_width': float(self._last_gap_metrics.get('best_gap_width', 0.0)),
                'best_gap_clearance': float(self._last_gap_metrics.get('best_gap_clearance', 0.0)),
                'interaction_mode_id': float(interaction_ctx.get('mode_id', 0.0)),
                'interaction_in_conflict': 1.0 if interaction_in_conflict else 0.0,
                'interaction_has_token': float(interaction_ctx.get('has_token', 0.0)),
                'interaction_wait_age_norm': float(interaction_ctx.get('wait_age_norm', 0.0)),
                'interaction_severity': float(interaction_ctx.get('severity', 0.0)),
                'interaction_turn_sign': float(interaction_ctx.get('turn_sign', 0.0)),
                'interaction_partner_dist': float(interaction_ctx.get('partner_dist', float('inf'))),
                'policy_interaction_mode_id': float(self.learned_interaction_mode_to_id.get(getattr(self, '_policy_interaction_mode', 'go'), 0)),
                'effective_interaction_mode_id': float(self.learned_interaction_mode_to_id.get(getattr(self, '_effective_interaction_mode', interaction_mode), 0)),
                'executed_behavior_mode_id': float(self.learned_interaction_mode_to_id.get(interaction_mode, 0)),
                'interaction_reason': str(self._last_control_info.get('interaction_reason', '')),
                'subgoal_mode': str(self._last_subgoal_mode),
                'subgoal_deadlock_streak': float(self._subgoal_deadlock_streak),
                # Path projection progress
                'path_s': float(getattr(self, '_path_s', 0.0)),
                'prev_path_s': float(getattr(self, '_prev_path_s', 0.0)),
                'path_projection_progress_delta': float(getattr(self, '_path_projection_progress_delta', 0.0)),
                'path_projection_progress_window': float(getattr(self, '_path_projection_progress_window', 0.0)),
                'closest_dist_to_path': float(getattr(self, '_closest_dist_to_path', 0.0)),
                'cross_track_error': float(getattr(self, '_cross_track_error', 0.0)),
                'cross_track_penalty': float(getattr(self, '_last_cross_track_penalty', 0.0)),
                'goal_progress_delta': float(goal_progress_delta),
                'local_goal_progress_delta': float(local_goal_progress_delta),
                'guide_target_progress_delta': float(getattr(self, '_guide_target_progress_delta', 0.0)),
                'positive_path_projection_progress': float(getattr(self, '_last_positive_path_projection_progress', 0.0)),
                'negative_path_projection_progress': float(getattr(self, '_last_negative_path_projection_progress', 0.0)),
                'progress_positive': 1.0 if getattr(self, '_last_progress_positive', False) else 0.0,
                'progress_source': str(getattr(self, '_last_progress_source', 'none')),
                'progress_source_id': float(getattr(self, '_last_progress_source_id', 0.0)),
                'R_progress': float(getattr(self, '_last_option_progress_reward', 0.0)),
                'obstacle_risk_drop': float(getattr(self, '_last_obstacle_risk_drop', 0.0)),
                'ttc_improvement': float(getattr(self, '_last_ttc_improvement', 0.0)),
                'risk_reduced': 1.0 if getattr(self, '_last_risk_reduced', False) else 0.0,
                'path_projection_valid': 1.0 if getattr(self, '_path_projection_valid', False) else 0.0,
                # Option outcome reward breakdown
                'option_progress_reward': float(option_terms.option_progress_reward),
                'option_clearance_reward': float(option_terms.option_clearance_reward),
                'option_safety_reward': float(option_terms.option_safety_reward),
                'option_completion_bonus': float(option_terms.option_completion_bonus),
                'option_failure_penalty': float(option_terms.option_failure_penalty),
                'option_timeout_penalty': float(option_terms.option_timeout_penalty),
                'safe_turn_reward_outcome': float(option_terms.safe_turn_reward),
                'wrong_turn_penalty': float(option_terms.wrong_turn_penalty),
                'random_turn_penalty': float(option_terms.random_turn_penalty),
                'spin_without_progress_penalty': float(option_terms.spin_without_progress_penalty),
                'idle_without_progress_penalty': float(option_terms.idle_without_progress_penalty),
                'conservative_mode_penalty': float(option_terms.conservative_mode_penalty),
                'option_switch_penalty': float(option_terms.option_switch_penalty),
                'infeasible_action_penalty': float(option_terms.infeasible_action_penalty),
                'pair_cooperative_reward': float(option_terms.pair_cooperative_reward),
                'pair_competitive_penalty': float(option_terms.pair_competitive_penalty),
                'obstacle_proximity_penalty': float(option_terms.obstacle_proximity_penalty),
                'backoff_release_reward': float(option_terms.backoff_release_reward),
                'detour_loop_penalty': float(option_terms.detour_loop_penalty),
                'left_safety_score': float(option_terms.left_safety_score),
                'right_safety_score': float(option_terms.right_safety_score),
                'ttc_risk': float(option_terms.ttc_risk),
                'risk_gate': float(option_terms.risk_gate),
                'correct_turn': float(option_terms.correct_turn),
                'wrong_turn': float(option_terms.wrong_turn),
                'front_blocked_ratio_delta': float(option_terms.front_blocked_ratio_delta),
                'stall_replan_elapsed_sec': float(self._stall_elapsed_sec),
                'subgoal_replan_ready': 1.0 if self.current_step >= self._next_replan_step else 0.0,
                'yield_hold_steps': float(self._yield_hold_steps),
                'active_option_mode': float(self.learned_interaction_mode_to_id.get(getattr(self, '_active_option_name', 'go'), 0)),
                'option_hold_remaining_frac': 0.0,
                'option_elapsed_frac': float(np.clip(
                    float(max(0, int(self.current_step) - int(getattr(self, '_active_option_start_step', 0))))
                    / float(max(1, int(getattr(self, '_active_option_duration_steps', 1)))),
                    0.0,
                    1.0,
            )),
                'dist_to_goal': float(dist_to_goal),
                'min_dist': float(min_dist),
                'front_min': float(front_min),
                'left_min': float(left_min),
                'right_min': float(right_min),
                'rear_min': float(sectors_now.get('rear_min', self.scan_max_range)),
                'front_left_min': float(sectors_now.get('front_left_min', self.scan_max_range)),
                'front_center_min': float(sectors_now.get('front_center_min', self.scan_max_range)),
                'front_right_min': float(sectors_now.get('front_right_min', self.scan_max_range)),
                'clearance_asymmetry': float(sectors_now.get('clearance_asymmetry', left_min - right_min)),
                'side_min': float(side_min),
                'front_close_ratio': float(front_close_ratio),
                'side_close_ratio': float(side_close_ratio),
                'corridor_narrow_ratio': float(corridor_narrow_ratio),
                'detour_active': 1.0 if getattr(self, '_detour_active', False) else 0.0,
                'detour_side': 1.0 if getattr(self, '_detour_side', '') == 'left' else (-1.0 if getattr(self, '_detour_side', '') == 'right' else 0.0),
                'detour_phase': str(getattr(self, '_detour_phase', DetourPhase.DONE)),
                'detour_hold_remaining': float(getattr(self, '_detour_hold_remaining', 0)),
                'detour_done': 1.0 if getattr(self, '_detour_done', False) else 0.0,
                'detour_interrupted': 1.0 if getattr(self, '_detour_interrupted', False) else 0.0,
                'rolling_subgoal_suppressed': 1.0 if getattr(self, '_detour_suppress_rolling', False) else 0.0,
                'detour_target_x': float(getattr(self, '_detour_guide_target_world', (self.goal_pos[0], self.goal_pos[1]))[0]) if getattr(self, '_detour_guide_target_world', None) is not None else 0.0,
                'detour_target_y': float(getattr(self, '_detour_guide_target_world', (self.goal_pos[0], self.goal_pos[1]))[1]) if getattr(self, '_detour_guide_target_world', None) is not None else 0.0,
                'pair_event_reward': 0.0,
                'local_head_on_pass_event': float(getattr(self, '_last_local_head_on_pass_event', 0.0)),
            })

            if potential_terms is not None:
                info.update({
                    'phi_goal_prev': float(potential_terms.phi_goal_prev),
                    'phi_goal_curr': float(potential_terms.phi_goal_curr),
                    'phi_goal_drop': float(potential_terms.phi_goal_drop),
                    'phi_obs_prev': float(potential_terms.phi_obs_prev),
                    'phi_obs_curr': float(potential_terms.phi_obs_curr),
                    'phi_obs_drop': float(potential_terms.phi_obs_drop),
                    'phi_agent_prev': float(potential_terms.phi_agent_prev),
                    'phi_agent_curr': float(potential_terms.phi_agent_curr),
                    'phi_agent_drop': float(potential_terms.phi_agent_drop),
                    'phi_path_prev': float(potential_terms.phi_path_prev),
                    'phi_path_curr': float(potential_terms.phi_path_curr),
                    'phi_path_drop': float(potential_terms.phi_path_drop),
                    'front_obstacle_potential': float(potential_terms.front_obstacle_potential),
                    'side_obstacle_potential': float(potential_terms.side_obstacle_potential),
                    'corner_obstacle_potential': float(potential_terms.corner_obstacle_potential),
                    'r_potential': float(potential_terms.r_potential),
                    'r_event': float(potential_terms.r_event),
                    'r_pair': 0.0,
                    'r_terminal': float(potential_terms.r_terminal),
                    'final_reward': float(reward),
                    'time_penalty_step': float(potential_terms.time_penalty),
                    'spin_without_progress': float(1.0 if potential_terms.spin_without_progress_penalty < 0.0 else 0.0),
                    'spin_without_progress_penalty': float(potential_terms.spin_without_progress_penalty),
                    'reverse_without_risk_penalty': float(potential_terms.reverse_without_risk_penalty),
                    'stuck_long': float(potential_terms.stuck_long),
                'stuck_long_penalty': float(potential_terms.stuck_long_penalty),
                'detour_active_penalty': float(potential_terms.detour_active_penalty),
                'detour_success_bonus': float(potential_terms.detour_success_bonus),
                    'corner_clear_bonus': float(potential_terms.corner_clear_bonus),
                    'no_progress': float(potential_terms.no_progress),
                    'progress_positive_simple': float(potential_terms.progress_positive),
                    'ttc_risk': float(potential_terms.ttc_risk),
                })

            return obs, reward, done, truncated, info

    def step(self, action):
        self.apply_action(action)
        self._wait_for_sim_time(0.1)
        return self.get_step_result()

    def _wait_for_sim_time(self, seconds):
        if not rclpy.ok():
            return
        while rclpy.ok() and self.node.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        start_time = self.node.get_clock().now().nanoseconds
        delta_ns = seconds * 1e9

        while rclpy.ok():
            now = self.node.get_clock().now().nanoseconds
            if now - start_time >= delta_ns:
                break
            rclpy.spin_once(self.node, timeout_sec=0.01)

    @staticmethod
    def _parse_world_bounds(world_path: str, margin: float = 0.25):
        try:
            import xml.etree.ElementTree as ET
            import math as _math

            root = ET.parse(world_path).getroot()
            world_el = root.find('world')
            if world_el is None:
                return None

            h_walls: list = []
            v_walls: list = []
            seen: set = set()

            for model in world_el.findall('model'):
                name = model.get('name', '')
                if name in ('ground_plane', 'sun'):
                    continue
                if name.startswith('dyn_obs') or name.startswith('tb3'):
                    continue

                pose_el = model.find('pose')
                if pose_el is None:
                    continue
                pose = pose_el.text.split()
                mx, my = float(pose[0]), float(pose[1])
                myaw = float(pose[5]) if len(pose) > 5 else 0.0

                key = (round(mx, 2), round(my, 2))
                if key in seen:
                    continue
                seen.add(key)

                for box in model.iter('box'):
                    sz = box.find('size')
                    if sz is None:
                        continue
                    dims = sz.text.split()
                    lx, ly = float(dims[0]), float(dims[1])
                    c = abs(_math.cos(myaw))
                    s = abs(_math.sin(myaw))
                    dx = lx * c + ly * s
                    dy = lx * s + ly * c
                    xmin, xmax = mx - dx / 2, mx + dx / 2
                    ymin, ymax = my - dy / 2, my + dy / 2
                    if dx > dy:
                        h_walls.append((xmin, xmax, ymin, ymax, my))
                    else:
                        v_walls.append((xmin, xmax, ymin, ymax, mx))
                    break

            if not h_walls or not v_walls:
                return None

            outer_top = max(h_walls, key=lambda e: e[4])
            outer_bottom = min(h_walls, key=lambda e: e[4])
            outer_right = max(v_walls, key=lambda e: e[4])
            outer_left = min(v_walls, key=lambda e: e[4])

            inner_top = outer_top[2]
            inner_bottom = outer_bottom[3]
            inner_right = outer_right[0]
            inner_left = outer_left[1]

            return (inner_left + margin, inner_right - margin,
                    inner_bottom + margin, inner_top - margin)

        except Exception as _e:
            print(f"⚠️  _parse_world_bounds({world_path}) 解析失败: {_e}")
            return None

    _MAP_SAFE_MARGIN = {
        1: 7,
        2: 7,
        3: 6,
        4: 5,
        5: 5,
        6: 5,
    }

    _MAP_FALLBACK_POSES = {
        1: [
            ((-0.5, -5.0), (1.1, -1.8)),
            ((1.1, -5.0), (-0.5, -1.8)),
            ((-0.5, -4.5), (1.1, -2.5)),
            ((0.8, -2.2), (-0.4, -4.8)),
            ((0.3, -5.5), (0.3, -1.6)),
        ],
        2: [
            ((-0.5, -5.5), (4.5, -1.5)),
            ((4.5, -1.5), (-0.5, -5.5)),
            ((0.3, -5.8), (4.5, -1.0)),
            ((-0.5, -3.0), (4.5, -1.0)),
            ((1.0, -5.0), (4.0, -1.5)),
        ],
        3: [
            ((-4.5, 0.0), (4.5, 0.0)),
            ((0.0, -4.5), (0.0, 4.5)),
            ((-4.0, 2.0), (4.0, -2.0)),
            ((-3.0, -3.0), (3.0, 3.0)),
            ((4.0, 1.0), (-4.0, -1.0)),
        ],
        4: [
            ((-4.0, 0.0), (4.0, 0.0)),
            ((0.0, -4.0), (0.0, 4.0)),
            ((-3.0, 0.5), (3.0, -0.5)),
            ((0.5, -3.5), (-0.5, 3.5)),
            ((-2.0, 0.0), (2.0, 0.0)),
        ],
        5: [
            ((-5.0, 0.0), (5.0, 0.0)),
            ((0.0, -4.0), (0.0, 4.0)),
            ((-4.0, -3.0), (4.0, 3.0)),
            ((-4.0, 2.0), (4.0, -2.0)),
            ((2.0, -4.0), (-2.0, 4.0)),
        ],
        6: [
            ((-4.5, 0.8), (4.5, -0.8)),
            ((-4.5, -0.8), (4.5, 0.8)),
            ((0.0, -4.5), (0.0, 4.5)),
            ((4.2, 0.0), (-4.2, 0.0)),
            ((-3.0, -3.0), (3.0, 3.0)),
        ],
    }

    _MAP_COLLISION_ROUTE_LIBRARY = {
        3: [
            ((-4.8, 0.0), (4.8, 0.0)),
            ((4.8, 0.0), (-4.8, 0.0)),
            ((-4.8, 0.6), (4.8, 0.6)),
            ((4.8, 0.6), (-4.8, 0.6)),
            ((-4.8, -0.6), (4.8, -0.6)),
            ((4.8, -0.6), (-4.8, -0.6)),
            ((4.8, 0.2), (-4.8, 0.2)),
            ((-4.5, 1.8), (4.5, -1.8)),
            ((4.5, -1.8), (-4.5, 1.8)),
            ((-3.8, -2.4), (3.8, 2.4)),
            ((3.8, 2.4), (-3.8, -2.4)),
            ((-4.2, 1.0), (4.2, -1.0)),
            ((4.2, -1.0), (-4.2, 1.0)),
            ((-3.6, 2.6), (3.6, 2.6)),
            ((3.6, 2.6), (-3.6, 2.6)),
            ((-3.6, -2.6), (3.6, -2.6)),
            ((3.6, -2.6), (-3.6, -2.6)),
        ],
        4: [
            ((-4.2, 0.0), (4.2, 0.0)),
            ((4.2, 0.0), (-4.2, 0.0)),
            ((0.0, -4.2), (0.0, 4.2)),
            ((0.0, 4.2), (0.0, -4.2)),
            ((-3.2, 0.6), (0.0, -4.0)),
            ((0.0, -4.0), (3.2, 0.6)),
            ((3.2, -0.6), (0.0, 4.0)),
            ((0.0, 4.0), (-3.2, -0.6)),
        ],
        5: [
            ((-5.0, -2.8), (5.0, -2.8)),
            ((5.0, -1.6), (-5.0, -1.6)),
            ((-4.6, 2.2), (4.6, 2.2)),
            ((4.6, 1.0), (-4.6, 1.0)),
            ((-3.8, -3.5), (3.8, 2.8)),
            ((3.8, 2.8), (-3.8, -3.5)),
            ((-2.0, -3.8), (2.0, 3.8)),
            ((2.0, 3.8), (-2.0, -3.8)),
        ],
        6: [
            ((-4.8, 0.8), (4.8, -0.8)),
            ((4.8, 0.8), (-4.8, -0.8)),
            ((0.0, -4.8), (0.0, 4.8)),
            ((0.0, 4.8), (0.0, -4.8)),
            ((-4.6, 0.0), (4.6, 0.0)),
            ((4.6, 0.0), (-4.6, 0.0)),
            ((-4.2, 1.4), (4.2, 1.4)),
            ((4.2, 1.4), (-4.2, 1.4)),
            ((-4.2, -1.4), (4.2, -1.4)),
            ((4.2, -1.4), (-4.2, -1.4)),
            ((-3.8, 2.4), (3.8, -2.4)),
            ((3.8, 2.4), (-3.8, -2.4)),
            ((-2.6, -3.8), (2.6, 3.8)),
            ((2.6, -3.8), (-2.6, 3.8)),
        ],
    }

    _DYN_OBS_SPAWNS = {
        1: [(0.4, -0.7), (0.4, -1.4), (0.4, -2.2), (0.4, -2.9),
            (0.4, -3.6), (0.4, -4.3), (0.4, -5.0), (0.4, -5.8)],
        2: [(0.3, -0.9), (0.3, -2.4), (0.3, -3.9), (0.3, -5.7),
            (3.3, -0.9), (5.4, -1.2), (3.3, -4.2), (5.4, -5.7)],
        3: [(-4.5, -4.0), (-4.5, 4.0), (-2.0, -4.5), (-2.0, 4.5),
            (2.0, -4.5), (2.0, 4.5), (4.5, -4.0), (4.5, 4.0)],
        4: [(-2.7, 0.5), (-2.7, -0.5), (2.7, 0.5), (2.7, -0.5),
            (0.5, -2.7), (-0.5, -2.7), (0.5, 2.7), (-0.5, 2.7)],
        5: [(-5.1, -3.6), (-5.1, 2.4), (-2.7, -3.6), (-0.3, -3.0),
            (-0.3, 2.4), (2.1, -3.6), (4.5, -3.6), (4.5, 2.4)],
    }

    def _get_random_valid_point(self, exclude=None, other_agents=None, min_agent_sep=1.5):
        if not self._valid_spawn_points:
            return 0.0, 0.0

        shuffled = random.sample(self._valid_spawn_points, len(self._valid_spawn_points))
        for wx, wy in shuffled:
            if self._is_safe_spawn_point(
                wx,
                wy,
                exclude=exclude,
                other_agents=other_agents,
                min_agent_sep=min_agent_sep,
            ):
                return wx, wy

        fallback_candidates = [
            (wx, wy)
            for wx, wy in self._valid_spawn_points
            if self._is_safe_spawn_point(
                wx,
                wy,
                exclude=exclude,
                other_agents=None,
                min_agent_sep=0.0,
            )
        ]
        if fallback_candidates:
            return random.choice(fallback_candidates)
        return random.choice(self._valid_spawn_points)

    def randomize_obstacles(self, robot_positions: list):
        pass

    def _set_obstacle_pose(self, model_name: str, x: float, y: float, z: float):
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            return

        req = SetEntityState.Request()
        req.state.name = model_name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.w = 1.0
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = 0.0

        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

    def _get_obs(self, target_override=None):
        return build_independent_env_observation(self, target_override=target_override)

    def _publish_vel(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.vel_pub.publish(msg)

    def _set_robot_pose(self, x, y, yaw):
        if not self.set_state_client.wait_for_service(timeout_sec=1.0):
            return

        req = SetEntityState.Request()
        req.state.name = self.gazebo_model_name

        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.1
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)

        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

    def close(self):
        self.node.destroy_node()
