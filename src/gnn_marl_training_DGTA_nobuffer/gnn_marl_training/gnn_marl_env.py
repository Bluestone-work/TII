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
import hashlib
import json
import yaml
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
from PIL import Image
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity
from example_interfaces.srv import AddTwoInts
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty, Trigger
from gnn_marl_training.global_planner import AStarPlanner, WaypointExtractor, PathTrackingUtils
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


# =====================================================================
# ---------------------------------------------------------------------
#   r_progress   : 朝目标前进 (path_delta + heading 塑形)   [-0.3, +0.3]
#   r_static     : 障碍避碰 (斥力 + 减速 + 前方预测风险)     [-1.0, 0]
#   r_ttc        : 统一TTC惩罚 (智能体+动态障碍物)          [-2.0, 0]
#   r_collision  : 碰撞终端                                 {-collision_penalty, 0}
#   r_goal       : 到达终端                                 {0, +goal_reward}
#   r_time       : 时间压力                                 -time_penalty
# 每项语义正交、互不重叠，便于做消融分析。
# =====================================================================

RWD_PROGRESS_CLIP        = 0.30   # r_progress 单步截断
RWD_STATIC_CLIP          = 1.00   # r_static 单步截断（仅负值方向）
RWD_TTC_CLIP             = 2.00   # r_ttc 单步截断 (统一TTC惩罚)
RWD_HEADING_COEF         = 0.10   # heading shaping 系数
RWD_HEADING_MIN_FWD_VEL  = 0.02   # 前向速度低于此则不给 heading shaping
RWD_STATIC_D0            = 0.50   # 斥力场起作用距离（0.5m 平衡避障和通过性）
RWD_STATIC_D_MIN         = 0.15   # 距离下界（防爆炸）
RWD_STATIC_SPEED_RISK_D0 = 0.70   # 前方障碍减速门控（0.7m 给足反应时间但不过敏）
RWD_TTC_MAX_DIST         = 2.5    # TTC计算最大距离（米）
RWD_TTC_APPROACH_TH      = 0.05   # 接近速度门（m/s）
RWD_TTC_SAFE             = 2.5    # 安全TTC阈值（秒）
RWD_GOAL_REACH_RADIUS    = 0.35   # 到达目标半径

# =====================================================================
# =====================================================================
ROBOT_RADIUS             = 0.105  # TurtleBot3 Burger 半径 (m)
SAFETY_MARGIN            = 0.15   # 安全裕度 (考虑定位误差和传感器噪声)
INFLATION_RADIUS         = ROBOT_RADIUS + SAFETY_MARGIN  # 总膨胀半径 = 0.255m


def _clearance_potential(distance: float, influence_dist: float, min_dist: float = 0.15) -> float:
    """Negative reciprocal potential inside an influence distance."""
    distance = float(distance)
    influence_dist = max(float(influence_dist), 1e-6)
    if not math.isfinite(distance) or distance >= influence_dist:
        return 0.0
    clamped = max(distance, float(min_dist))
    return -float(((1.0 / clamped) - (1.0 / influence_dist)) ** 2)


def compute_directional_clearance_terms(
    *,
    front_center: float,
    front_left: float,
    front_right: float,
    left: float,
    right: float,
    min_dist: float,
    forward_speed: float,
    angular_vel: float,
    corridor_min_width: float,
    corridor_narrow_width: float,
    corridor_front_open_dist: float,
) -> Dict[str, float]:
    """Build motion-aware static-obstacle terms without punishing parallel walls."""
    width = max(0.0, float(left)) + max(0.0, float(right))
    feasible = width >= max(0.0, float(corridor_min_width))
    front_open = float(front_center) >= float(corridor_front_open_dist)
    narrow_span = max(float(corridor_narrow_width) - float(corridor_min_width), 1e-6)
    narrow_gate = float(np.clip(
        (float(corridor_narrow_width) - width) / narrow_span,
        0.0,
        1.0,
    )) if feasible else 0.0
    corridor_active = bool(feasible and front_open and narrow_gate > 0.0)

    turn_norm = float(np.clip(abs(float(angular_vel)) / 1.2, 0.0, 1.0))
    if float(angular_vel) > 0.0:
        turn_clearance = min(float(front_left), float(left))
    elif float(angular_vel) < 0.0:
        turn_clearance = min(float(front_right), float(right))
    else:
        turn_clearance = min(float(front_left), float(front_right))

    front_potential = _clearance_potential(
        float(front_center),
        RWD_STATIC_D0,
        min_dist=RWD_STATIC_D_MIN,
    )
    turn_potential = 0.65 * turn_norm * _clearance_potential(turn_clearance, 0.40)
    hard_proximity_potential = 0.35 * _clearance_potential(float(min_dist), 0.20, min_dist=0.12)

    speed_risk = 0.0
    speed = max(0.0, float(forward_speed))
    if float(front_center) < RWD_STATIC_SPEED_RISK_D0:
        ratio = (RWD_STATIC_SPEED_RISK_D0 - float(front_center)) / RWD_STATIC_SPEED_RISK_D0
        speed_risk = -speed * float(ratio ** 2)

    center_error = 0.0
    if corridor_active and width > 1e-6:
        center_error = float(np.clip(abs(float(left) - float(right)) / width, 0.0, 1.0))

    return {
        'corridor_width': float(width),
        'corridor_feasible': 1.0 if feasible else 0.0,
        'corridor_active': 1.0 if corridor_active else 0.0,
        'corridor_narrow_gate': float(narrow_gate),
        'corridor_center_error': float(center_error),
        'turn_clearance': float(turn_clearance),
        'front_potential': float(front_potential),
        'turn_potential': float(turn_potential),
        'hard_proximity_potential': float(hard_proximity_potential),
        'speed_risk': float(speed_risk),
    }


def compute_relative_motion_metrics(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    *,
    horizon: float,
    clearance_radius: float = 0.0,
) -> Dict[str, float]:
    """Compute closest approach and TTC from relative position/velocity."""
    rel_pos = np.asarray(relative_position, dtype=np.float32).reshape(2)
    rel_vel = np.asarray(relative_velocity, dtype=np.float32).reshape(2)
    horizon = max(0.0, float(horizon))
    clearance_radius = max(0.0, float(clearance_radius))

    distance = float(np.linalg.norm(rel_pos))
    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq > 1e-8:
        closest_time = float(np.clip(-np.dot(rel_pos, rel_vel) / rel_speed_sq, 0.0, horizon))
    else:
        closest_time = 0.0
    closest_distance = float(np.linalg.norm(rel_pos + rel_vel * closest_time))

    closing_speed = 0.0
    if distance > 1e-8:
        closing_speed = max(0.0, float(-np.dot(rel_pos, rel_vel) / distance))
    ttc = float('inf')
    if closing_speed > 1e-6:
        ttc = max(0.0, distance - clearance_radius) / closing_speed

    return {
        'distance': distance,
        'clearance': distance - clearance_radius,
        'closing_speed': closing_speed,
        'ttc': float(ttc),
        'closest_time': closest_time,
        'closest_distance': closest_distance,
        'closest_clearance': closest_distance - clearance_radius,
    }


def unicycle_body_trajectory(
    linear_vel: float,
    angular_vel: float,
    times: np.ndarray,
) -> np.ndarray:
    """Return unicycle center positions in the initial robot body frame."""
    t = np.asarray(times, dtype=np.float32).reshape(-1)
    v = float(linear_vel)
    w = float(angular_vel)
    if abs(w) < 1e-5:
        return np.stack([v * t, np.zeros_like(t)], axis=1)
    radius = v / w
    return np.stack([
        radius * np.sin(w * t),
        radius * (1.0 - np.cos(w * t)),
    ], axis=1).astype(np.float32)


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
        self.enable_local_map = config.get('enable_local_map', False)
        self.map_number = int(config.get('map_number', 3))
        self.enable_neighbor_obs = config.get('enable_neighbor_obs', True)
        self.paired_dynamic_obstacles = bool(config.get('paired_dynamic_obstacles', False))
        self.dynamic_obstacle_trajectory_fingerprint = ''
        self.dynamic_obstacle_trajectory_spec = {}
        self._paired_trajectory_pending_start = False
        self.strict_step_time_control = bool(config.get('strict_step_time_control', False))
        self._physics_paused = False

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
        self.remove_completed_agents = bool(config.get('remove_completed_agents', True))
        self.completed_agent_parking_origin = tuple(config.get('completed_agent_parking_origin', (100.0, 100.0)))
        self.reset_max_attempts = int(max(1, config.get('reset_max_attempts', 20)))
        self.reset_min_agent_sep = float(config.get('reset_min_agent_sep', 0.85))
        self.reset_min_lidar_clearance = float(config.get('reset_min_lidar_clearance', 0.50))
        self.reset_min_static_obstacle_sep = float(config.get('reset_min_static_obstacle_sep', 0.55))
        # 高冲突训练模式：通过更激进的起终点路由采样制造会车/交叉冲突。
        self.high_conflict_mode = str(config.get('high_conflict_mode', 'off')).strip().lower()
        if self.high_conflict_mode not in ('off', 'mixed', 'aggressive'):
            self.high_conflict_mode = 'off'
        self.high_conflict_prob = float(np.clip(float(config.get('high_conflict_prob', 0.75)), 0.0, 1.0))
        # Circle swap arena (map 8): force aggressive conflict mode for antipodal assignment
        if self.map_number == 8:
            self.high_conflict_mode = 'aggressive'

        # 创建独立环境实例
        self.agents = {}
        env_signature = inspect.signature(IndependentRobotEnv.__init__)
        env_map_number = int(config.get('map_number', 3))
        default_voronoi = env_map_number in (3, 4, 5)
        for i in range(self._num_agents):
            candidate_kwargs = {
                'robot_id': i,
                'map_number': env_map_number,
                'max_episode_steps': config.get('max_episode_steps', 1000),
                'collision_ends_episode': bool(config.get('collision_ends_episode', False)),
                'collision_hard_dist': float(config.get('collision_hard_dist', 0.20)),
                'collision_persist_dist': float(config.get('collision_persist_dist', 0.26)),
                'collision_persist_steps': int(config.get('collision_persist_steps', 2)),
                'near_wall_penalty_dist': float(config.get('near_wall_penalty_dist', 0.30)),
                'waypoint_reach_radius': float(config.get('waypoint_reach_radius', 0.8)),
                'waypoint_distance_threshold': float(config.get('waypoint_distance_threshold', 1.2)),
                'waypoint_min_clearance_m': float(config.get('waypoint_min_clearance_m', 0.40)),
                'planner_inflation_margin_m': float(config.get('planner_inflation_margin_m', 0.28)),
                'use_voronoi_planner': bool(config.get('use_voronoi_planner', default_voronoi)),
                'voronoi_min_clearance_m': float(config.get('voronoi_min_clearance_m', 0.35)),
                'num_dynamic_obstacles': config.get('num_dynamic_obstacles', 8),
                'num_static_obstacles': config.get('num_static_obstacles', 8),
                'random_obstacles': bool(config.get('random_obstacles', True)),
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
                'replan_cooldown_steps': int(config.get('replan_cooldown_steps', 100)),
                'dynamic_replan_neighbor_dist': float(config.get('dynamic_replan_neighbor_dist', 1.8)),
                'dynamic_replan_ttc': float(config.get('dynamic_replan_ttc', 2.6)),
                'dynamic_replan_block_radius': float(config.get('dynamic_replan_block_radius', 0.55)),
                'progress_replan_window_steps': int(config.get('progress_replan_window_steps', 80)),
                'progress_replan_min_delta': float(config.get('progress_replan_min_delta', 0.10)),
                'obs_target_dist_clip': float(config.get('obs_target_dist_clip', 6.0)),
                'obs_target_filter_alpha': float(config.get('obs_target_filter_alpha', 0.35)),
                'obs_target_max_step': float(config.get('obs_target_max_step', 0.45)),
                'guidance_lookahead_m': float(config.get('guidance_lookahead_m', 3.0)),
                'target_obs_dim': int(config.get('target_obs_dim', 6)),
                'progress_reward_scale': float(config.get('progress_reward_scale', 0.0)),
                'path_progress_reward_scale': float(config.get('path_progress_reward_scale', 0.0)),
                'goal_progress_reward_scale': float(config.get('goal_progress_reward_scale', 4.0)),
                'goal_reward': float(config.get('goal_reward', 20.0)),
                'collision_penalty': float(config.get('collision_penalty', 20.0)),
                'time_penalty': float(config.get('time_penalty', 0.01)),
                'lateral_penalty_scale': float(config.get('lateral_penalty_scale', 0.0)),
                'heading_align_reward_scale': float(config.get('heading_align_reward_scale', 0.0)),
                'narrow_forward_penalty_scale': float(config.get('narrow_forward_penalty_scale', 0.35)),
                'close_obstacle_penalty_scale': float(config.get('close_obstacle_penalty_scale', 0.30)),
                'close_obstacle_dist': float(config.get('close_obstacle_dist', 0.55)),
                'team_reward_lambda': float(config.get('team_reward_lambda', 0.7)),
                'shield_enable': bool(config.get('shield_enable', False)),
                'shield_front_slow_dist': float(config.get('shield_front_slow_dist', 0.50)),
                'shield_front_stop_dist': float(config.get('shield_front_stop_dist', 0.20)),
                'shield_neighbor_slow_dist': float(config.get('shield_neighbor_slow_dist', 0.35)),
                'shield_linear_slow': float(config.get('shield_linear_slow', 0.12)),
                'shield_linear_stop': float(config.get('shield_linear_stop', 0.04)),
                'shield_turn_bias': float(config.get('shield_turn_bias', 0.35)),
                'turn_in_place_front_dist': float(config.get('turn_in_place_front_dist', 0.35)),
                'turn_in_place_angle_thresh': float(config.get('turn_in_place_angle_thresh', 0.45)),
                'turn_in_place_w': float(config.get('turn_in_place_w', 0.90)),
                'corridor_min_width': float(config.get('corridor_min_width', 0.50)),
                'corridor_narrow_width': float(config.get('corridor_narrow_width', 1.00)),
                'corridor_front_open_dist': float(config.get('corridor_front_open_dist', 0.55)),
                'corridor_center_penalty_scale': float(config.get('corridor_center_penalty_scale', 0.08)),
                'path_clearance_margin': float(config.get('path_clearance_margin', 0.10)),
                'swept_footprint_horizon': float(config.get('swept_footprint_horizon', 0.70)),
                'swept_footprint_hard_clearance': float(config.get('swept_footprint_hard_clearance', 0.24)),
                'swept_footprint_soft_clearance': float(config.get('swept_footprint_soft_clearance', 0.30)),
                'use_gazebo_collision': bool(config.get('use_gazebo_collision', True)),
                'lidar_collision_fallback': bool(config.get('lidar_collision_fallback', True)),
                'obstacle_filter_range': float(config.get('obstacle_filter_range', 2.0)),
                'sensor_noise_std': float(config.get('sensor_noise_std', 0.0)),
                'obstacle_filter_fov_deg': float(config.get('obstacle_filter_fov_deg', 360.0)),
                'obstacle_top_k': int(config.get('obstacle_top_k', 9)),
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
                'obstacle_motion_feature_enable': bool(config.get('obstacle_motion_feature_enable', True)),
                'obstacle_motion_top_k': int(config.get('obstacle_motion_top_k', 3)),
                'subgoal_progress_reward_scale': float(config.get('subgoal_progress_reward_scale', 1.2)),
                'detour_progress_relax': float(config.get('detour_progress_relax', 0.30)),
                'risk_aware_forward_penalty_scale': float(config.get('risk_aware_forward_penalty_scale', 0.28)),
                'safe_turn_reward_scale': float(config.get('safe_turn_reward_scale', 0.15)),
                'head_on_avoidance_reward_scale': float(config.get('head_on_avoidance_reward_scale', 0.90)),
                'action_mode': config.get('action_mode', 'discrete_primitive'),
                'reward_mode': config.get('reward_mode', 'dgr_vds'),
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
        # 随机spawn的静态障碍物位置（用于重规划）
        self.spawned_static_obstacles = []

        # episode 级累计统计（auto-reset 模式下每个 agent 在一个 episode 内的成功/碰撞次数）
        self.episode_successes  = {aid: 0 for aid in self.agent_ids}
        self.episode_collisions = {aid: 0 for aid in self.agent_ids}
        self.episode_stats = {
            'total_collisions': 0,
            'near_misses': 0,
            'successful_navigations': 0,
        }
        self.team_reward_lambda = float(config.get('team_reward_lambda', 0.5))

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
        _log_dir   = config.get('log_dir', os.path.abspath('./artifacts/ray_results/gnn_marl_logs'))
        _log_file  = os.path.join(_log_dir, f'env_worker{_worker_id}.log')
        self.logger = _LogAdapter(
            _setup_env_logger(_log_file, _worker_id),
            {'worker_id': _worker_id}
        )
        self.log_every_n_steps = int(config.get('log_every_n_steps', 50))
        self.logger.info(
            ' GNNMARLEnv init: agents=%d comm_mode=%s range=%.1fm latency=%d noise=%.3f',
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

        master_node = self.agents['agent_0'].node
        self._dynamic_obstacle_reset_client = master_node.create_client(
            AddTwoInts, '/reset_dynamic_obstacles')
        self._dynamic_obstacle_start_client = master_node.create_client(
            Trigger, '/start_dynamic_obstacles')
        self._dynamic_obstacle_advance_client = master_node.create_client(
            Trigger, '/advance_dynamic_obstacles')
        self._dynamic_obstacle_sync_client = master_node.create_client(
            Trigger, '/sync_dynamic_obstacles')
        self._pause_physics_client = master_node.create_client(Empty, '/pause_physics')
        self._unpause_physics_client = master_node.create_client(Empty, '/unpause_physics')

    def _call_empty_service(self, client, request, name: str, timeout: float = 5.0):
        master = self.agents['agent_0']
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{name} unavailable')
        future = client.call_async(request)
        deadline = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(master.node, timeout_sec=0.01)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'{name} timed out')
        return future.result()

    def _pause_physics(self) -> None:
        if not self.strict_step_time_control:
            return
        self._call_empty_service(self._pause_physics_client, Empty.Request(), '/pause_physics')
        self._physics_paused = True

    def _unpause_physics(self) -> None:
        if not self.strict_step_time_control:
            return
        self._call_empty_service(self._unpause_physics_client, Empty.Request(), '/unpause_physics')
        self._physics_paused = False

    def _call_obstacle_service(self, client, request, name: str, timeout: float = 5.0):
        master = self.agents['agent_0']
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{name} unavailable; strict trajectory pairing cannot be guaranteed')
        future = client.call_async(request)
        deadline = time.time() + timeout
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(master.node, timeout_sec=0.01)
        if not future.done() or future.result() is None:
            raise RuntimeError(f'{name} timed out; strict trajectory pairing cannot be guaranteed')
        return future.result()

    def _reset_paired_dynamic_obstacles(self, seed: int) -> None:
        if not self.paired_dynamic_obstacles:
            return
        num_obstacles = int(getattr(self.agents['agent_0'], 'num_dynamic_obstacles', 0))
        if num_obstacles <= 0:
            self.dynamic_obstacle_trajectory_fingerprint = ''
            self.dynamic_obstacle_trajectory_spec = {}
            return
        request = AddTwoInts.Request()
        request.a = int(seed)
        request.b = int(self.max_steps)
        response = self._call_obstacle_service(
            self._dynamic_obstacle_reset_client,
            request,
            '/reset_dynamic_obstacles',
            timeout=30.0,
        )
        if int(response.sum) != int(seed):
            raise RuntimeError('dynamic-obstacle reset did not acknowledge the requested seed')
        self.dynamic_obstacle_trajectory_spec = {}
        self.dynamic_obstacle_trajectory_fingerprint = ''
        self._paired_trajectory_pending_start = True

    def _start_paired_dynamic_obstacles(self) -> None:
        if not self._paired_trajectory_pending_start:
            return
        # Gazebo may still be servicing the reset request when the first start
        # request arrives.  In that window the Trigger callback can return an
        # empty/non-manifest response even though the trajectory is valid.  Keep
        # retrying briefly so a transient startup race does not invalidate an
        # otherwise paired episode.
        deadline = time.time() + 30.0
        last_error = 'unknown start-service failure'
        while time.time() < deadline:
            try:
                response = self._call_obstacle_service(
                    self._dynamic_obstacle_start_client,
                    Trigger.Request(),
                    '/start_dynamic_obstacles',
                    timeout=min(5.0, max(0.25, deadline - time.time())),
                )
                if not bool(response.success):
                    last_error = 'dynamic-obstacle start service rejected the paired trajectory'
                else:
                    try:
                        payload = json.loads(str(response.message))
                        fingerprint = str(payload['fingerprint'])
                        spec = dict(payload['spec'])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        last_error = f'dynamic-obstacle service omitted the frozen trajectory manifest: {exc}'
                    else:
                        if len(fingerprint) == 64:
                            self.dynamic_obstacle_trajectory_fingerprint = fingerprint
                            self.dynamic_obstacle_trajectory_spec = spec
                            self._paired_trajectory_pending_start = False
                            return
                        last_error = 'dynamic-obstacle trajectory fingerprint is not SHA-256'
            except RuntimeError as exc:
                last_error = str(exc)
            time.sleep(0.25)
        raise RuntimeError(last_error)

    def _advance_paired_dynamic_obstacles(self) -> None:
        if not self.paired_dynamic_obstacles:
            return
        num_obstacles = int(getattr(self.agents['agent_0'], 'num_dynamic_obstacles', 0))
        if num_obstacles <= 0:
            return
        response = self._call_obstacle_service(
            self._dynamic_obstacle_advance_client,
            Trigger.Request(),
            '/advance_dynamic_obstacles',
        )
        if not bool(response.success):
            raise RuntimeError(f'dynamic-obstacle advance rejected: {response.message}')
        deadline = time.time() + 5.0
        while time.time() < deadline:
            sync = self._call_obstacle_service(
                self._dynamic_obstacle_sync_client,
                Trigger.Request(),
                '/sync_dynamic_obstacles',
            )
            if bool(sync.success):
                return
            if str(sync.message).startswith('apply_failed:'):
                raise RuntimeError(f'dynamic-obstacle Gazebo update failed: {sync.message}')
            time.sleep(0.005)
        raise RuntimeError('dynamic-obstacle frame did not reach Gazebo before the control step')

    # ===================================================================
    # 高冲突路由采样 —— 制造会车/交叉冲突的起终点路线
    # ===================================================================
    @staticmethod
    def _procedural_antipodal_routes(n_routes: int):
        """Build up to 24 unique Map8/9 routes on a 1.1m-spaced grid."""
        if n_routes < 1 or n_routes > 24:
            raise ValueError(f"procedural Map8/9 routes support 1..24 agents, got {n_routes}")
        coords = (-2.2, -1.1, 0.0, 1.1, 2.2)
        representatives = [
            (x, y)
            for x in coords
            for y in coords
            if (x > 0.0 or (x == 0.0 and y > 0.0))
        ]
        representatives.sort(
            key=lambda point: (-(point[0] ** 2 + point[1] ** 2), math.atan2(point[1], point[0]))
        )
        routes = []
        for x, y in representatives:
            routes.append(((x, y), (-x, -y)))
            routes.append(((-x, -y), (x, y)))
        return routes[:n_routes]

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

        if master.map_number in (8, 9) and n_routes > len(route_lib):
            picks = self._procedural_antipodal_routes(n_routes)
        # Circle swap (map 8): routes are organized as antipodal PAIRS
        # [0,1] are a head-on pair, [2,3] are a pair, etc.
        # For 2/4/6 agents, sample complete pairs to guarantee head-on conflict.
        elif master.map_number == 8 and len(route_lib) >= 2:
            n_pairs = max(1, n_routes // 2)
            total_pairs = len(route_lib) // 2
            selected_pairs = random.sample(range(total_pairs), min(n_pairs, total_pairs))
            picks = []
            for pi in selected_pairs:
                picks.append(route_lib[pi * 2])
                picks.append(route_lib[pi * 2 + 1])
            picks = picks[:n_routes]
        else:
            picks = random.sample(route_lib, min(max(1, n_routes), len(route_lib)))

        out = []
        for (sx, sy), (gx, gy) in picks:
            if self.high_conflict_mode == 'aggressive' and random.random() < 0.35:
                out.append(((gx, gy), (sx, sy)))
            else:
                out.append(((sx, sy), (gx, gy)))
        return out

    def _build_episode_route_plan(self):
        if self.high_conflict_mode == 'off':
            return {}
        if self.high_conflict_mode == 'mixed' and random.random() > self.high_conflict_prob:
            return {}

        routes = self._sample_conflict_routes(self._num_agents)
        if not routes:
            return {}

        random.shuffle(routes)
        plan = {}
        for idx, aid in enumerate(self.agent_ids):
            plan[aid] = routes[idx % len(routes)]
        return plan

    def _sample_conflict_route_for_respawn(self):
        if self.high_conflict_mode == 'off':
            return None
        if self.high_conflict_mode == 'mixed' and random.random() > self.high_conflict_prob:
            return None
        routes = self._sample_conflict_routes(1)
        if not routes:
            return None
        return routes[0]
    
    # ===================================================================
    # 观测 / 动作空间定义
    # ===================================================================
    def _define_observation_space(self):
        """定义观测空间"""
        # 从实际 agent 动态获取 base_obs_dim，避免与 IndependentRobotEnv.obs_dim 不一致
        # IndependentRobotEnv.obs_dim = scan_dim*scan_history_len + 2 + 2 + safety_feature_dim
        # 其中 scan_dim 由 Top-K 障碍特征编码决定（默认 top_k=9 -> 36），默认总维度仍为 155
        base_obs_dim = self.agents['agent_0'].obs_dim
        
        # 可选：邻居状态（最多 K 个近邻）
        if self.enable_neighbor_obs:
            # 最多邻居数 = min(其他机器人数量, 5)
            # Fixed 5 neighbor slots regardless of num_agents — ensures model is
            # deployable at any N (unused slots are zero-padded).
            max_neighbors = 5
            # 每个邻居: body-frame rel_pos(2) + rel_vel(2) + dist(1) + sin/cos rel_heading(2) = 7
            neighbor_dim = max_neighbors * 7
        else:
            neighbor_dim = 0
        
        # 可选：局部地图（32x32 ego-centric grid * 2 frames = 2048 flat）
        if self.enable_local_map:
            local_map_dim = 32 * 32 * 4
        else:
            local_map_dim = 0
        
        # 单 agent 局部 reset 标记：用于通知模型清空该 agent 的 LSTM state
        self.reset_flag_dim = 1

        # 全局状态（集中式 Critic 输入）：所有智能体基础观测按编号顺序拼接
        # 训练时 Critic 享有"特权信息"；部署时 Critic 不运行，无需传递
        self.global_state_dim = self._num_agents * base_obs_dim

        total_dim = (
            base_obs_dim + neighbor_dim + local_map_dim + self.reset_flag_dim + self.global_state_dim
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        self.base_obs_dim = base_obs_dim
        self.neighbor_dim = neighbor_dim
        self.local_map_dim = local_map_dim
    
    # ===================================================================
    # 主循环入口：reset / step
    # ===================================================================
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """重置环境"""
        if self.strict_step_time_control:
            # Reset uses sim-time waits for spawn settle and sensor refresh.
            # Unpause here, then freeze again before handing observations to policy inference.
            self._unpause_physics()
        options = dict(options or {})
        post_plan_retry = int(max(0, options.get('_post_plan_retry', 0)))
        paired_episode_seed = options.get('_paired_episode_seed', seed)
        fixed_dynamic_positions = options.get('fixed_dynamic_initial_positions')
        if fixed_dynamic_positions is not None:
            expected_names = {
                f'dyn_obs_{index}'
                for index in range(int(self.agents['agent_0'].num_dynamic_obstacles))
            }
            provided = dict(fixed_dynamic_positions)
            if set(provided) != expected_names:
                raise ValueError(
                    'fixed dynamic-obstacle slots do not match the environment: '
                    f'expected={sorted(expected_names)} found={sorted(provided)}'
                )
            self._fixed_dynamic_initial_positions = [
                tuple(float(value) for value in provided[name])
                for name in sorted(provided, key=lambda item: int(item.rsplit('_', 1)[-1]))
            ]
        else:
            self._fixed_dynamic_initial_positions = None
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

        if paired_episode_seed is not None:
            self._reset_paired_dynamic_obstacles(int(paired_episode_seed))

        self._remove_inactive_dynamic_obstacles()
        self._ensure_static_obstacles_before_robot_reset()

        forced_route_plan = options.get("route_plan")
        strict_fixed_route = bool(options.get("strict_fixed_route", False))
        route_yaws = {}
        if forced_route_plan is not None:
            route_plan = {}
            for aid, route in dict(forced_route_plan).items():
                aid = str(aid)
                if isinstance(route, dict):
                    start_xy = route["start"]
                    goal_xy = route["goal"]
                    if route.get("yaw") is not None:
                        route_yaws[aid] = float(route["yaw"])
                else:
                    start_xy, goal_xy = route[0], route[1]
                route_plan[aid] = (
                    tuple(float(value) for value in start_xy),
                    tuple(float(value) for value in goal_xy),
                )
            if set(route_plan) != set(self.agent_ids):
                raise ValueError(
                    "fixed route_plan agents do not match environment: "
                    f"routes={sorted(route_plan)} env={sorted(self.agent_ids)}"
                )
        else:
            route_plan = self._build_episode_route_plan()
        if route_plan:
            self.logger.info('[reset] 启用高冲突路线采样: mode=%s prob=%.2f',
                             self.high_conflict_mode, self.high_conflict_prob)

        obs_dict = {}
        info_dict = {}

        base_obs_dict = {}
        reset_reject_reason = ""
        for reset_attempt in range(1, self.reset_max_attempts + 1):
            base_obs_dict = {}
            info_dict = {}
            agent_starts = []  # list of (x, y)
            for aid, agent in self.agents.items():
                forced = route_plan.get(aid)
                obs, info = agent.reset(
                    other_agent_starts=agent_starts,
                    forced_start_goal=forced,
                    forced_yaw=route_yaws.get(aid),
                    strict_forced_route=strict_fixed_route,
                )
                base_obs_dict[aid] = obs
                spawn_pos = info.get('start_xy', None)
                if spawn_pos:
                    agent_starts.append(spawn_pos)
                    self.robot_positions[aid] = np.array(spawn_pos, dtype=np.float32)
                else:
                    self.robot_positions[aid] = self._get_robot_position(agent)
                self.robot_velocities[aid] = self._get_robot_velocity(agent)
                info_dict[aid] = info

                goal = getattr(agent, 'goal_pos', ('?', '?'))
                pos  = self.robot_positions[aid]
                self.logger.info(
                    '[reset] attempt=%d %-10s  spawn=(%.3f, %.3f)  goal=(%.3f, %.3f)  '
                    'start_xy_from_info=%s',
                    reset_attempt, aid, pos[0], pos[1], goal[0], goal[1], spawn_pos,
                )

            # Gazebo SetEntityState and per-robot odom callbacks are not atomic.
            # Before building the initial multi-agent observation, spin all nodes
            # and refresh poses so the first policy step sees the settled reset.
            self._wait_and_spin_all(0.5)
            for aid, agent in self.agents.items():
                self.robot_positions[aid] = self._get_robot_position(agent)
                self.robot_velocities[aid] = self._get_robot_velocity(agent)
                base_obs_dict[aid] = agent._get_obs()

            reset_ok, reset_reject_reason = self._initial_reset_is_safe(
                min_lidar_clearance=0.20 if strict_fixed_route else None
            )
            if reset_ok:
                if reset_attempt > 1:
                    self.logger.info('[reset] accepted after %d attempts', reset_attempt)
                break

            self.logger.warning(
                '[reset] rejected unsafe initial layout attempt=%d/%d: %s',
                reset_attempt, self.reset_max_attempts, reset_reject_reason,
            )
            for agent in self.agents.values():
                agent._publish_vel(0.0, 0.0)
            self._wait_and_spin_all(0.2)
        else:
            self.logger.warning(
                '[reset] using last layout after %d rejected attempts: %s',
                self.reset_max_attempts, reset_reject_reason,
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

        self._replan_all_agents_with_current_obstacles()

        # Replanning can take long enough for moving obstacles and ROS sensor
        # callbacks to advance. Refresh the first policy observation and verify
        # clearance again at the actual hand-off time, not only before planning.
        self._wait_and_spin_all(0.25)
        for aid, agent in self.agents.items():
            self.robot_positions[aid] = self._get_robot_position(agent)
            self.robot_velocities[aid] = self._get_robot_velocity(agent)
            base_obs_dict[aid] = agent._get_obs()
        post_plan_ok, post_plan_reason = self._initial_reset_is_safe(
            min_lidar_clearance=0.20 if strict_fixed_route else None
        )
        if not post_plan_ok and post_plan_retry < 3:
            self.logger.warning(
                '[reset] post-plan layout became unsafe; resampling retry=%d/3: %s',
                post_plan_retry + 1,
                post_plan_reason,
            )
            for agent in self.agents.values():
                agent._publish_vel(0.0, 0.0)
            next_seed = None if seed is None else int(seed) + 1009 * (post_plan_retry + 1)
            next_options = {'_post_plan_retry': post_plan_retry + 1}
            if paired_episode_seed is not None:
                next_options['_paired_episode_seed'] = int(paired_episode_seed)
            if forced_route_plan is not None:
                next_options['route_plan'] = forced_route_plan
                next_options['strict_fixed_route'] = strict_fixed_route
            if fixed_dynamic_positions is not None:
                next_options['fixed_dynamic_initial_positions'] = fixed_dynamic_positions
            return self.reset(
                seed=next_seed,
                options=next_options,
            )
        if not post_plan_ok:
            self.logger.warning(
                '[reset] accepting post-plan layout after retry limit: %s',
                post_plan_reason,
            )

        # 预填充通信历史
        if self.comm_mode == 'ros2_bridge':
            for buf in self._ros2_neighbor_bufs.values():
                buf.clear()
            self._broadcast_ros2_states()
        else:
            self._state_history.clear()
            for _ in range(self._state_history.maxlen):
                self._push_state_snapshot()

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

        if self.strict_step_time_control:
            self._pause_physics()
        return obs_dict, info_dict
    
    def step(self, action_dict: Dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """环境步进"""
        if self.strict_step_time_control:
            self._pause_physics()
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
        if self.strict_step_time_control:
            self._unpause_physics()
        self._start_paired_dynamic_obstacles()
        self._advance_paired_dynamic_obstacles()
        try:
            self._wait_and_spin_all(0.1)
        finally:
            if self.strict_step_time_control:
                self._pause_physics()

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
                # Stop terminal agents immediately. Otherwise the last non-zero
                # cmd_vel published before goal/collision detection can remain
                # active until a later env.step sends the normal done-agent stop.
                self.agents[aid]._publish_vel(0.0, 0.0)
                self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
                if info_dict.get(aid, {}).get('event') == 'goal':
                    self._remove_completed_agent_from_scene(aid)
                self.dones.add(aid)
                if info_dict.get(aid, {}).get('event') == 'collision':
                    self.failed_agents.add(aid)
                done_dict[aid] = True
                truncated_dict[aid] = False

        # ── 多智能体奖励共享：每个体奖励 = λ * 自身 + (1-λ) * 团队平均 ─────
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
            for aid in self.agent_ids:
                self.agents[aid]._publish_vel(0.0, 0.0)
                self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
            self.logger.info(
                '━━━ EPISODE END (%s, step=%d) dones=%d/%d '
                'successes=%s collisions=%s active_remaining=%d failed=%d ━━━',
                reason, self.current_step_count,
                len(self.dones), self._num_agents,
                {aid: self.episode_successes[aid]  for aid in self.agent_ids},
                {aid: self.episode_collisions[aid] for aid in self.agent_ids},
                active_remaining, failed_count,
            )
            # print(
            #     f"\n{'='*60}\n"
            #     f"🏁 Episode 结束 ({reason})\n"
            #     f"   步数: {self.current_step_count}/{self.max_steps}\n"
            #     f"   完成: {len(self.dones)}/{self._num_agents}\n"
            #     f"   活跃: {active_remaining}  失败: {failed_count}\n"
            #     f"{'='*60}\n"
            # )
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
        spin_timeout = 0.0 if self.strict_step_time_control else 0.001
        while rclpy.ok():
            now_ns = ref_node.get_clock().now().nanoseconds
            if now_ns - start_ns >= delta_ns:
                break
            # 轮询所有机器人节点，保证各自的 scan/odom 回调都能执行
            for agent in self.agents.values():
                rclpy.spin_once(agent.node, timeout_sec=spin_timeout)

    # ===================================================================
    # 通信图构建（按通信半径连边）
    # ===================================================================
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
        # if getattr(self, 'debug_comm', False) or self.current_step_count <= 3:
        #     print('\n'.join(log_lines))

        return adjacency

    # ------------------------------------------------------------------
    # 通信延迟建模辅助方法
    # ------------------------------------------------------------------

    # ===================================================================
    # 通信建模（sim2real）：ROS2 桥接 + 状态历史/延迟/丢包/噪声
    # ===================================================================
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
    
    # ===================================================================
    # 增强观测拼接（local_obs + 邻居编码 + reset_flag + 全局 state）
    # ===================================================================
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

        # Local map: ego-centric occupancy grid (actor only, not in global_state)
        if self.enable_local_map:
            agent_obj = self.agents[agent_id]
            ranges = np.array(
                agent_obj.latest_scan.ranges if agent_obj.latest_scan else [agent_obj.scan_max_range] * 360,
                dtype=np.float32,
            )
            ranges = np.nan_to_num(ranges, nan=agent_obj.scan_max_range, posinf=agent_obj.scan_max_range, neginf=0.0)
            local_map_obs = agent_obj._build_local_map_obs(ranges)
            components.append(local_map_obs)

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
        编码邻居状态。支持三种通信模式：

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
        my_vel        = self.robot_velocities[agent_id]
        max_neighbors = 5  # Fixed slots, zero-padded if fewer neighbors available
        received = self._get_received_neighbor_samples(agent_id, adjacency_matrix=adjacency_matrix)

        agent_obj = self.agents[agent_id]
        my_yaw = float(agent_obj.current_pose['yaw'])
        cos_yaw = math.cos(my_yaw)
        sin_yaw = math.sin(my_yaw)

        # 编码最近 K 个邻居（不足则填零，保持向量维度固定）
        features_list: List[np.ndarray] = []
        for k in range(max_neighbors):
            if k < len(received):
                n_idx, dist, n_pos, n_vel = received[k]
                rel_pos_w = n_pos - my_pos
                rel_vel_w = n_vel - my_vel
                rel_pos_b = np.array([
                    cos_yaw * rel_pos_w[0] + sin_yaw * rel_pos_w[1],
                    -sin_yaw * rel_pos_w[0] + cos_yaw * rel_pos_w[1],
                ], dtype=np.float32)
                rel_vel_b = np.array([
                    cos_yaw * rel_vel_w[0] + sin_yaw * rel_vel_w[1],
                    -sin_yaw * rel_vel_w[0] + cos_yaw * rel_vel_w[1],
                ], dtype=np.float32)
                n_aid = f"agent_{n_idx}"
                n_yaw = float(self.agents[n_aid].current_pose['yaw']) if n_aid in self.agents else my_yaw
                rel_heading = n_yaw - my_yaw
                feat = np.array([
                    rel_pos_b[0], rel_pos_b[1],
                    rel_vel_b[0], rel_vel_b[1],
                    dist,
                    math.sin(rel_heading), math.cos(rel_heading),
                ], dtype=np.float32)
            else:
                feat = np.zeros(7, dtype=np.float32)
            features_list.append(feat)

        neighbor_vec = np.concatenate(features_list).astype(np.float32)

        # ── 日志：通信详情（每步写文件，前3步或 debug_comm=True 同时打印） ──
        comm_lines = [
            f'[comm] step={self.current_step_count:4d}  {agent_id}'
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
            comm_lines.append(f'    (无有效邻居: 范围外/丢包/延迟填零)')
        self.logger.debug('\n'.join(comm_lines))

        # if getattr(self, 'debug_comm', False) or self.current_step_count <= 3:
        #     print('\n'.join(comm_lines))

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

    def _completed_agent_parking_pose(self, aid: str) -> Tuple[float, float, float]:
        try:
            idx = int(str(aid).split('_')[-1])
        except Exception:
            idx = 0
        origin_x, origin_y = self.completed_agent_parking_origin
        return float(origin_x) + 2.0 * idx, float(origin_y), 0.0

    def _remove_completed_agent_from_scene(self, aid: str) -> None:
        if not self.remove_completed_agents:
            return

        agent = self.agents.get(aid)
        if agent is None:
            return

        try:
            if hasattr(agent, 'vis') and agent.vis:
                agent.vis.publish_goal_completed(
                    getattr(agent, 'goal_pos', None),
                    robot_id=agent.robot_id,
                    namespace=agent.vis_namespace,
                )
        except Exception as exc:
            self.logger.warning('[done] %s 绿色终点marker发布失败: %s', aid, exc)

        park_x, park_y, park_yaw = self._completed_agent_parking_pose(aid)
        try:
            ok = bool(agent._set_robot_pose(park_x, park_y, park_yaw))
            agent.current_pose['x'] = park_x
            agent.current_pose['y'] = park_y
            agent.current_pose['yaw'] = park_yaw
            agent.current_vel_x = 0.0
            agent.current_vel_w = 0.0
            self.robot_positions[aid] = np.array([park_x, park_y], dtype=np.float32)
            self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
            self.logger.info(
                '[done] %s 到达目标，已移出场景 park=(%.1f, %.1f), gazebo_ok=%s',
                aid, park_x, park_y, ok,
            )
        except Exception as exc:
            self.logger.warning('[done] %s 移出场景失败: %s', aid, exc)

    def _current_robot_positions_for_obstacle_guard(self) -> List[Tuple[float, float]]:
        positions: List[Tuple[float, float]] = []
        for aid, agent in self.agents.items():
            pos = self._get_robot_position(agent)
            if np.all(np.isfinite(pos)):
                positions.append((float(pos[0]), float(pos[1])))
        return positions

    def _ensure_static_obstacles_before_robot_reset(self) -> None:
        """Create or sync static obstacles before sampling robot start/goal pairs."""
        master = self.agents.get('agent_0')
        if master is None or not hasattr(master, 'randomize_obstacles'):
            return

        num_static = int(getattr(master, 'num_static_obstacles', 0))
        random_obstacles = bool(getattr(master, 'random_obstacles', False))
        if num_static <= 0 or not random_obstacles:
            self.spawned_static_obstacles = []
            for agent in self.agents.values():
                agent.spawned_static_obstacles.clear()
                agent._static_obstacles_spawned = True
            return

        already_spawned = (
            bool(getattr(master, '_static_obstacles_spawned', False))
            and len(getattr(self, 'spawned_static_obstacles', [])) > 0
        )
        if already_spawned:
            self.logger.info(
                '[reset] 先复用静态障碍物布局（共%d个），再采样机器人起点',
                len(self.spawned_static_obstacles),
            )
        else:
            guard_positions = self._current_robot_positions_for_obstacle_guard()
            self.logger.info(
                '[reset] 先生成静态障碍物，再采样机器人起点/终点 guard_positions=%d',
                len(guard_positions),
            )
            if hasattr(master, 'spawn_entity_client'):
                master.spawn_entity_client.wait_for_service(timeout_sec=3.0)

            for attempt in range(1, 3):
                master.randomize_obstacles(guard_positions)
                self._wait_and_spin_all(0.8)
                spawned_count = len(getattr(self, 'spawned_static_obstacles', []))
                if spawned_count >= num_static:
                    break
                self.logger.warning(
                    '[reset] 静态障碍物生成数量不足 attempt=%d/2 spawned=%d expected=%d，等待后重试',
                    attempt, spawned_count, num_static,
                )
                self._wait_and_spin_all(1.0)

            spawned_count = len(getattr(self, 'spawned_static_obstacles', []))
            if spawned_count < num_static:
                self.logger.warning(
                    '[reset] 静态障碍物未完全生成 spawned=%d expected=%d；本次reset仍会由lidar安全门兜底',
                    spawned_count, num_static,
                )
            else:
                self.logger.info('[reset] 静态障碍物已生成并同步：%d个', spawned_count)

        for agent in self.agents.values():
            agent._static_obstacles_spawned = True
            agent.spawned_static_obstacles = list(self.spawned_static_obstacles)

    def _remove_inactive_dynamic_obstacles(self) -> None:
        """Disable unused dynamic-obstacle slots without deleting Gazebo models.

        Runtime DeleteEntity calls race Gazebo's physics thread during GUI and
        multi-robot spawn, which can dereference a released Inertial pointer.
        Keeping the model and parking it outside the map is collision-free and
        preserves the fixed dyn_obs_0..7 model topology expected by the mover.
        """
        master = self.agents.get('agent_0')
        if master is None:
            return

        n_active = int(getattr(master, 'num_dynamic_obstacles', 0))
        n_active = max(0, min(n_active, 8))
        if n_active >= 8:
            return

        client = getattr(master, 'set_state_client', None)
        if client is None or not client.wait_for_service(timeout_sec=1.0):
            self.logger.warning('[reset] /set_entity_state unavailable; obstacle_mover will disable inactive dyn_obs_%d~7', n_active)
            return

        disabled = []
        for i in range(n_active, 8):
            req = SetEntityState.Request()
            req.state.name = f'dyn_obs_{i}'
            req.state.pose.position.x = 20.0 + float(i)
            req.state.pose.position.y = 20.0
            req.state.pose.position.z = 0.4
            req.state.pose.orientation.w = 1.0
            future = client.call_async(req)
            deadline = time.time() + 1.5
            while rclpy.ok() and not future.done() and time.time() < deadline:
                rclpy.spin_once(master.node, timeout_sec=0.02)
            if future.done():
                resp = future.result()
                if resp is None or bool(getattr(resp, 'success', True)):
                    disabled.append(req.state.name)

        if disabled:
            self.logger.info('[reset] parked inactive dynamic obstacles outside map: %s', ', '.join(disabled))
            self._wait_and_spin_all(0.2)

    def _current_blocked_points_for_planning(self, agent: "IndependentRobotEnv") -> List[Tuple[float, float]]:
        blocked_points: List[Tuple[float, float]] = [
            (float(x), float(y))
            for x, y, _ in getattr(self, 'spawned_static_obstacles', [])
        ]

        dyn_obs_positions = agent._active_dyn_obs_spawns()
        for pos in dyn_obs_positions:
            blocked_points.append((float(pos[0]), float(pos[1])))
        return blocked_points

    def _replan_all_agents_with_current_obstacles(self) -> None:
        self.logger.info(
            '[reset] 所有机器人已安全spawn，按当前障碍物重新规划路径 static=%d',
            len(getattr(self, 'spawned_static_obstacles', [])),
        )
        for aid, agent in self.agents.items():
            if not hasattr(agent, 'planner') or not agent.planner:
                continue

            start_pos = tuple(getattr(agent, 'last_spawn_pos', self.robot_positions[aid].tolist()))
            goal_pos = tuple(getattr(agent, 'goal_pos', (0.0, 0.0)))
            blocked_points = self._current_blocked_points_for_planning(agent)
            path = agent._plan_with_retry(
                start_pos,
                goal_pos,
                blocked_points,
                initial_radius=agent._default_planning_block_radius(),
            )

            if path and hasattr(agent, 'waypoint_extractor'):
                agent.global_waypoints = agent.waypoint_extractor.extract(path, planner=agent.planner)
                agent.current_waypoint_index = 0
                agent._global_path_valid = True
                self.logger.info(
                    '  %s: 重新规划成功 (%d waypoints, %d blocked_points)',
                    aid, len(agent.global_waypoints), len(blocked_points),
                )
            else:
                self.logger.warning(
                    '  %s: 重新规划失败！(start=%s, goal=%s, blocked=%d)',
                    aid, start_pos, goal_pos, len(blocked_points),
                )

    def _initial_reset_is_safe(self, min_lidar_clearance=None) -> Tuple[bool, str]:
        """Reject reset layouts that start in near-collision."""
        min_agent_sep = float(self.reset_min_agent_sep)
        min_lidar_clearance = float(
            self.reset_min_lidar_clearance
            if min_lidar_clearance is None else min_lidar_clearance
        )

        for i in range(self._num_agents):
            for j in range(i + 1, self._num_agents):
                ai, aj = f'agent_{i}', f'agent_{j}'
                pi = self.robot_positions.get(ai)
                pj = self.robot_positions.get(aj)
                if pi is None or pj is None:
                    continue
                dist = float(np.linalg.norm(
                    np.asarray(pi, dtype=np.float32) - np.asarray(pj, dtype=np.float32)
                ))
                if dist < min_agent_sep:
                    return False, f'{ai}<->{aj} start distance {dist:.3f}m < {min_agent_sep:.3f}m'

        robot_radius = 0.12
        static_clearance = float(self.reset_min_static_obstacle_sep)
        for aid, pos in self.robot_positions.items():
            p = np.asarray(pos, dtype=np.float32)
            for ox, oy, radius in getattr(self, 'spawned_static_obstacles', []):
                min_required = float(radius) + robot_radius + static_clearance
                dist = float(np.linalg.norm(p - np.array([float(ox), float(oy)], dtype=np.float32)))
                if dist < min_required:
                    return (
                        False,
                        f'{aid} start too close to static obstacle '
                        f'dist={dist:.3f}m < {min_required:.3f}m '
                        f'obstacle=({float(ox):.2f},{float(oy):.2f},r={float(radius):.2f})',
                    )

        for aid, agent in self.agents.items():
            try:
                sectors = agent._scan_sector_metrics()
                min_dist = float(sectors.get('min_dist', agent.scan_max_range))
                front_min = float(sectors.get('front_min', min_dist))
            except Exception as exc:
                return False, f'{aid} scan unavailable during reset safety check: {exc}'
            if min_dist < min_lidar_clearance:
                return False, f'{aid} lidar min_dist {min_dist:.3f}m < {min_lidar_clearance:.3f}m'
            if front_min < min_lidar_clearance:
                return False, f'{aid} front_min {front_min:.3f}m < {min_lidar_clearance:.3f}m'

        return True, ''
    
    def reset_stage_obstacles(self):
        """
        Stage 切换时调用：清理旧的静态障碍物，重置spawn标志
        让下次reset时重新spawn新密度的静态障碍物
        """
        master = self.agents.get('agent_0')
        if master is not None:
            # 清理旧的静态障碍物
            if hasattr(master, '_delete_spawned_obstacles'):
                master._delete_spawned_obstacles()
                self.logger.info('[reset_stage_obstacles] 已清理旧静态障碍物')

            # 重置所有agent的spawn标志
            for agent in self.agents.values():
                agent._static_obstacles_spawned = False
                agent.spawned_static_obstacles.clear()

            # 清理parent_env的记录
            if hasattr(self, 'spawned_static_obstacles'):
                self.spawned_static_obstacles = []

            self.logger.info('[reset_stage_obstacles] 已重置spawn标志，下次reset将spawn新密度障碍物')

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
                 collision_ends_episode=True,
                 collision_hard_dist=0.05,
                 collision_persist_dist=0.15,
                 collision_persist_steps=3,
                 near_wall_penalty_dist=0.20,
                 waypoint_reach_radius=0.8,
                 waypoint_distance_threshold=1.2,
                 waypoint_min_clearance_m=0.40,
                 planner_inflation_margin_m=0.28,
                 use_voronoi_planner=True,
                 voronoi_min_clearance_m=0.35,
                 num_dynamic_obstacles=8, num_static_obstacles=8,
                 random_obstacles=True, obs_speed=0.3,
                 rolling_lookahead_dist=0.8,
                 subgoal_block_front_dist=0.42,
                 subgoal_min_side_clearance=0.20,
                 subgoal_detour_forward_gain=0.55,
                 subgoal_detour_lateral_gain=0.75,
                 subgoal_detour_hold_steps=8,
                 subgoal_deadlock_front_dist=0.48,
                 subgoal_deadlock_speed_thresh=0.03,
                 subgoal_deadlock_steps=10,
                 replan_on_deadlock=True,
                 replan_cooldown_steps=100,
                 dynamic_replan_neighbor_dist=1.8,
                 dynamic_replan_ttc=2.6,
                 dynamic_replan_block_radius=0.55,
                 progress_replan_window_steps=80,
                 progress_replan_min_delta=0.10,
                 obs_target_dist_clip=6.0,
                 obs_target_filter_alpha=0.35,
                 obs_target_max_step=0.45,
                 guidance_lookahead_m=3.0,
                 target_obs_dim=6,
                 progress_reward_scale=0.0,
                 path_progress_reward_scale=0.0,
                 goal_progress_reward_scale=4.0,
                 goal_reward=100.0,
                 collision_penalty=50.0,
                 time_penalty=0.01,
                 lateral_penalty_scale=0.0,
                 heading_align_reward_scale=0.0,
                 narrow_forward_penalty_scale=0.35,
                 close_obstacle_penalty_scale=0.30,
                 close_obstacle_dist=0.55,
                 team_reward_lambda=0.65,
                 shield_enable=False,
                 shield_front_slow_dist=0.50,
                 shield_front_stop_dist=0.20,
                 shield_neighbor_slow_dist=0.35,
                 shield_linear_slow=0.12,
                 shield_linear_stop=0.04,
                 shield_turn_bias=0.35,
                 turn_in_place_front_dist=0.35,
                 turn_in_place_angle_thresh=0.45,
                 turn_in_place_w=0.90,
                 corridor_min_width=0.50,
                 corridor_narrow_width=1.00,
                 corridor_front_open_dist=0.55,
                 corridor_center_penalty_scale=0.08,
                 path_clearance_margin=0.10,
                 swept_footprint_horizon=0.70,
                 swept_footprint_hard_clearance=0.24,
                 swept_footprint_soft_clearance=0.30,
                 use_gazebo_collision=True,
                 lidar_collision_fallback=False,
	                 obstacle_filter_range=2.0,
	                 sensor_noise_std=0.0,
                 obstacle_filter_fov_deg=360.0,
                 obstacle_top_k=9,
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
                 obstacle_motion_feature_enable=True,
                 obstacle_motion_top_k=3,
                 subgoal_progress_reward_scale=1.2,
                 detour_progress_relax=0.30,
                 risk_aware_forward_penalty_scale=0.28,
                 safe_turn_reward_scale=0.15,
                 head_on_avoidance_reward_scale=0.90,
                 progress_scale=3.0,
                 static_scale=0.5,
                 social_scale=0.3,
                 action_mode='discrete_primitive',
                 reward_mode='dgr_vds'):
        """
        Args:
            collision_ends_episode: 如果为 True，碰撞会结束episode；
                                   如果为 False，碰撞只给惩罚但继续运行
                                   多机器人训练建议设为True以加快学习
        """
        super(IndependentRobotEnv, self).__init__()

        self.robot_id = robot_id
        self.collision_ends_episode = collision_ends_episode
        self.max_episode_steps = max_episode_steps
        self.use_random_mode = use_random_mode
        self.map_number = int(map_number)
        self.sensor_noise_std = max(0.0, float(sensor_noise_std))
        self.current_step = 0
        self.collision_hard_dist = float(collision_hard_dist)
        self.collision_persist_dist = float(collision_persist_dist)
        self.collision_persist_steps = int(max(1, collision_persist_steps))
        self.near_wall_penalty_dist = float(near_wall_penalty_dist)
        self.waypoint_reach_radius = float(waypoint_reach_radius)
        self.waypoint_distance_threshold = float(waypoint_distance_threshold)
        self.waypoint_min_clearance_m = float(waypoint_min_clearance_m)
        self.planner_inflation_margin_m = max(ROBOT_RADIUS + 0.05, float(planner_inflation_margin_m))
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
                inflation_margin_m=self.planner_inflation_margin_m,
                clearance_cost_weight=6.0 if self.use_voronoi_planner else 0.0,
                preferred_clearance_m=0.85,
                prune_min_clearance_m=0.55,
            )
            self.waypoint_extractor = WaypointExtractor(
                distance_threshold=self.waypoint_distance_threshold,
                min_clearance_m=self.waypoint_min_clearance_m,
            )
            print(f" Robot {robot_id}: A*规划器初始化完成")

        self.global_waypoints = []
        self.current_waypoint_index = 0
        self._global_path_valid = False

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
        self.delete_entity_client = self.node.create_client(DeleteEntity, '/delete_entity')
        self.spawn_entity_client = self.node.create_client(SpawnEntity, '/spawn_entity')

        self.latest_scan = None
        self.current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_vel_x = 0.0
        self.current_vel_w = 0.0

        # 速度上限（TurtleBot3）
        self.max_forward_vel = 0.22
        self.max_reverse_vel = 0.12
        self.max_angular_vel = 1.2

        mode = str(action_mode).strip().lower()
        if mode in ('continuous', 'cont', 'box'):
            self.action_mode = 'continuous'
        elif mode in ('discrete', 'primitive', 'discrete_primitive'):
            self.action_mode = 'discrete_primitive'
        else:
            raise ValueError(f'Unsupported action_mode: {action_mode}')

        # 离散动作原语: [(v, w), ...]
        self.discrete_action_primitives = [
            (0.00, 0.00),
            (0.08, 0.00),
            (0.14, 0.00),
            (0.20, 0.00),
            (0.08, 0.55),
            (0.08, -0.55),
            (0.00, 1.00),
            (0.00, -1.00),
            (-0.06, 0.00),
        ]
        self._last_action_primitive: Optional[int] = None

        self.scan_max_range = 3.5
        self.scan_valid_min = 0.10
        self.scan_history_len = 8
        self.obstacle_top_k = int(np.clip(int(obstacle_top_k), 1, 64))
        self.obstacle_filter_range = float(np.clip(float(obstacle_filter_range), 0.2, self.scan_max_range))
        self.obstacle_filter_fov_deg = float(np.clip(float(obstacle_filter_fov_deg), 10.0, 360.0))

        # scan_dim 必须与 _extract_filtered_scan_features() 的真实输出维度一致。
        # 当前强化感知实现采用固定扇区池化，每帧只输出 obstacle_top_k 个标量。
        # 例如默认 top_k=9, history=4 时：
        # base_obs = 9*4 + goal(2) + vel(2) + safety(7) + predictive(6) = 53 维。
        self.scan_dim = self.obstacle_top_k
        self._scan_history: deque = deque(maxlen=self.scan_history_len)
        self._local_map_history: deque = deque(maxlen=4)
        self.guidance_lookahead_m = max(0.5, float(guidance_lookahead_m))
        self.target_obs_dim = int(target_obs_dim)
        # Agent ID one-hot embedding (max 8 agents) — breaks symmetry among homogeneous agents
        self.max_agent_id_dim = 8
        self.agent_id_embedding = np.zeros(self.max_agent_id_dim, dtype=np.float32)
        if 0 <= int(self.robot_id) < self.max_agent_id_dim:
            self.agent_id_embedding[int(self.robot_id)] = 1.0
        # SIMPLIFIED OBS: safety reduced 7→2 (front_min, min_dist only)
        # stacked_scan, predictive, gap features REMOVED — covered by local_map CNN
        self.base_safety_feature_dim = 2
        self.predictive_feature_enable = bool(predictive_feature_enable)
        self.predictive_feature_dim = 0  # removed from obs (still used for reward)
        self.predictive_horizon_sec = max(0.2, float(predictive_horizon_sec))
        self.predictive_social_ttc_safe = max(0.2, float(predictive_social_ttc_safe))
        self.predictive_front_ttc_safe = max(0.2, float(predictive_front_ttc_safe))
        self.predictive_min_sep = max(0.15, float(predictive_min_sep))
        self.predictive_social_range = max(self.predictive_min_sep, float(predictive_social_range))
        self.predictive_social_penalty_scale = max(0.0, float(predictive_social_penalty_scale))
        self.predictive_front_penalty_scale = max(0.0, float(predictive_front_penalty_scale))
        self.social_proximity_risk_scale = max(0.0, float(social_proximity_risk_scale))
        self.gap_feature_enable = bool(gap_feature_enable)
        self.gap_feature_dim = 0  # removed from obs
        self.neighbor_prediction_top_k = max(0, int(neighbor_prediction_top_k))
        # NEW: raw neighbor token = [body_rel_x, body_rel_y, body_rel_vx, body_rel_vy, dist_norm, sin_heading, cos_heading]
        self.neighbor_prediction_feature_dim = 7
        self.neighbor_prediction_dim = (
            self.neighbor_prediction_top_k * self.neighbor_prediction_feature_dim
        )
        self.yielding_enable = bool(yielding_enable)
        self.yielding_soft_dist = max(0.2, float(yielding_soft_dist))
        self.yielding_stop_dist = max(0.1, min(self.yielding_soft_dist, float(yielding_stop_dist)))
        self.yielding_hard_stop_dist = max(0.05, min(self.yielding_stop_dist, float(yielding_hard_stop_dist)))
        self.yielding_ttc = max(0.5, float(yielding_ttc))
        self.yielding_commit_steps = int(max(1, yielding_commit_steps))
        self.obstacle_motion_feature_enable = bool(obstacle_motion_feature_enable)
        self.obstacle_motion_top_k = max(0, int(obstacle_motion_top_k))
        self.obstacle_motion_feature_dim = 7
        self.obstacle_motion_dim = (
            self.obstacle_motion_top_k * self.obstacle_motion_feature_dim
            if self.obstacle_motion_feature_enable else 0
        )
        self.control_dt = 0.1
        self._front_min_history: deque = deque(maxlen=self.scan_history_len)
        self._front_sector_dist_history: deque = deque(maxlen=self.scan_history_len)
        self._obstacle_cluster_history: deque = deque(maxlen=max(3, self.scan_history_len))
        self._cluster_velocity_ema: Dict[Tuple[float, float], Tuple[float, float]] = {}
        self._last_motion_features: Optional[np.ndarray] = None  # 保存用于 r_ttc 计算
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
        self.subgoal_progress_reward_scale = max(0.0, float(subgoal_progress_reward_scale))
        self.detour_progress_relax = float(np.clip(float(detour_progress_relax), 0.0, 1.0))
        self.risk_aware_forward_penalty_scale = max(0.0, float(risk_aware_forward_penalty_scale))
        self.safe_turn_reward_scale = max(0.0, float(safe_turn_reward_scale))
        self.head_on_avoidance_reward_scale = max(0.0, float(head_on_avoidance_reward_scale))
        self.progress_scale = max(0.0, float(progress_scale))
        self.static_scale = max(0.0, float(static_scale))
        self.social_scale = max(0.0, float(social_scale))
        self.safety_feature_dim = (
            self.base_safety_feature_dim
            + self.predictive_feature_dim
            + self.gap_feature_dim
        )
        # SIMPLIFIED: removed scan history (covered by local_map CNN)
        self.obs_dim = (
            self.target_obs_dim
            + 2
            + self.safety_feature_dim
            + self.neighbor_prediction_dim
            + self.obstacle_motion_dim
            + self.max_agent_id_dim
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        if self.action_mode == 'continuous':
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(len(self.discrete_action_primitives))

        self.goal_pos = (0.0, 0.0)
        self.prev_dist_to_goal = None
        self.prev_dist_to_target = None
        self.prev_target_point = None
        self.prev_abs_target_angle = None

        self.num_dynamic_obstacles = max(0, min(int(num_dynamic_obstacles), 8))
        self.num_static_obstacles = max(0, min(int(num_static_obstacles), 8))
        self.random_obstacles = bool(random_obstacles)
        self.obs_speed = float(obs_speed)
        self.dynamic_obstacle_names: list = [f'dyn_obs_{i}' for i in range(self.num_dynamic_obstacles)]
        self.static_obstacle_names: list = [f'static_obs_{i}' for i in range(8)]
        self.spawned_static_obstacles: list = []  # 记录spawn的静态障碍物位置 [(x, y, radius), ...]
        self._static_obstacles_spawned = False  # 标记静态障碍物是否已spawn（stage级别，不每episode重置）

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
        self.dynamic_replan_neighbor_dist = max(0.5, float(dynamic_replan_neighbor_dist))
        self.dynamic_replan_ttc = max(0.5, float(dynamic_replan_ttc))
        self.dynamic_replan_block_radius = max(0.10, float(dynamic_replan_block_radius))
        self.progress_replan_window_steps = int(max(1, progress_replan_window_steps))
        self.progress_replan_min_delta = max(0.0, float(progress_replan_min_delta))
        self.obs_target_dist_clip = max(0.5, float(obs_target_dist_clip))
        self.obs_target_filter_alpha = float(np.clip(float(obs_target_filter_alpha), 0.0, 1.0))
        self.obs_target_max_step = max(0.05, float(obs_target_max_step))
        self.progress_reward_scale = float(progress_reward_scale)
        self.path_progress_reward_scale = float(path_progress_reward_scale)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        # 奖励模式: 'dgr_vds'(默认,原精简6项奖励) 或 'rmappo'(基线:Δgeodesic+10·I_goal-0.05·I_collision)
        self.reward_mode = str(reward_mode)
        # goal_progress_reward_scale / heading_align_reward_scale are kept for
        # config compatibility; path and lateral terms are used in get_step_result.
        self.goal_progress_reward_scale = float(goal_progress_reward_scale)   # 未使用
        self.lateral_penalty_scale = max(0.0, float(lateral_penalty_scale))
        self.heading_align_reward_scale = float(heading_align_reward_scale)  # 未使用
        self.narrow_forward_penalty_scale = float(narrow_forward_penalty_scale)
        self.close_obstacle_penalty_scale = float(close_obstacle_penalty_scale)
        self.close_obstacle_dist = float(close_obstacle_dist)
        self.team_reward_lambda = float(team_reward_lambda)

        self.shield_enable = bool(shield_enable)
        self.shield_front_slow_dist = float(shield_front_slow_dist)
        self.shield_front_stop_dist = float(shield_front_stop_dist)
        self.shield_neighbor_slow_dist = float(shield_neighbor_slow_dist)
        self.shield_linear_slow = float(shield_linear_slow)
        self.shield_linear_stop = float(shield_linear_stop)
        self.shield_turn_bias = float(shield_turn_bias)
        self.turn_in_place_front_dist = float(turn_in_place_front_dist)
        self.turn_in_place_angle_thresh = float(turn_in_place_angle_thresh)
        self.turn_in_place_w = float(turn_in_place_w)
        self.corridor_min_width = max(2.0 * ROBOT_RADIUS, float(corridor_min_width))
        self.corridor_narrow_width = max(self.corridor_min_width + 0.05, float(corridor_narrow_width))
        self.corridor_front_open_dist = max(0.25, float(corridor_front_open_dist))
        self.corridor_center_penalty_scale = max(0.0, float(corridor_center_penalty_scale))
        self.path_clearance_margin = max(0.02, float(path_clearance_margin))
        self.swept_footprint_horizon = max(0.2, float(swept_footprint_horizon))
        self.swept_footprint_hard_clearance = max(
            ROBOT_RADIUS + 0.03,
            self.collision_persist_dist,
            float(swept_footprint_hard_clearance),
        )
        self.swept_footprint_soft_clearance = max(
            self.swept_footprint_hard_clearance + 0.03,
            float(swept_footprint_soft_clearance),
        )
        self.use_gazebo_collision = bool(use_gazebo_collision)
        self.lidar_collision_fallback = bool(lidar_collision_fallback)

        # Gazebo 硬碰撞事件（ContactsState）
        self._gazebo_collision_active = False

        self.current_subgoal = None
        self.current_projection = None
        self.current_path_heading = 0.0
        self.path_progress = 0.0
        self.prev_path_progress = None
        self.current_lateral_error = 0.0
        self._obs_target_state = None
        self._subgoal_detour_hold = 0
        self._subgoal_detour_side = 0
        self._subgoal_deadlock_streak = 0
        self._next_replan_step = 0
        self._last_subgoal_mode = 'nominal'
        self._progress_replan_window_start_step = 0
        self._progress_replan_window_start_progress = 0.0
        self._last_guidance_near_weight = 0.0
        self._last_path_curvature = 0.0
        self._last_guidance_lookahead = self.guidance_lookahead_m
        self._last_allowed_lateral_error = 0.35
        self._last_path_clearance = self.scan_max_range
        self._last_swept_clearance = self.scan_max_range
        self._last_swept_filter_active = False
        self._last_motion_safety_margin = self.scan_max_range
        self._last_motion_safety_source = 'none'
        self._last_motion_filter_active = False

    # ===================================================================
    # 地图加载 / spawn 安全检验
    # ===================================================================
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
            map_mapping = {1: 'map1', 2: 'map2', 3: 'corridor_swap', 4: 'intersection', 5: 'warehouse_aisles', 6: 'interaction_hub', 7: 'interaction_hub_mini', 8: 'circle_swap_arena', 9: 'warehouse_dynamic'}
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
            SPAWN_CLEARANCE_PX = 10
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
            print(f" Map {map_number}: 预计算安全 spawn 点 {len(self._valid_spawn_points)} 个"
                  f" (clearance={SPAWN_CLEARANCE_PX * self.map_resolution:.2f}m)")
        except Exception as e:
            print(f" 加载地图失败: {e}")
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

        dyn_spawns = self._active_dyn_obs_spawns()
        fixed_dynamic = getattr(
            getattr(self, 'parent_env', None),
            '_fixed_dynamic_initial_positions',
            None,
        )
        dynamic_spawn_clearance = 0.60 if fixed_dynamic is not None else 1.0
        if any(
            math.hypot(wx - sx, wy - sy) < dynamic_spawn_clearance
            for sx, sy in dyn_spawns
        ):
            return False

        static_obs_list = []
        if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
            static_obs_list = getattr(self.parent_env, 'spawned_static_obstacles', [])
            min_static_sep = float(getattr(self.parent_env, 'reset_min_static_obstacle_sep', 0.55))
        else:
            static_obs_list = getattr(self, 'spawned_static_obstacles', [])
            min_static_sep = 0.55

        robot_radius = 0.12
        for item in static_obs_list:
            try:
                ox, oy, radius = item
                min_required = float(radius) + robot_radius + min_static_sep
                if math.hypot(wx - float(ox), wy - float(oy)) < min_required:
                    return False
            except Exception:
                continue

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

    def _is_valid_fixed_start_goal_pair(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        other_agent_starts: Optional[List[Tuple[float, float]]] = None,
    ) -> bool:
        """Validate a packaged route without the random-spawn erosion margin."""
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if not all(math.isfinite(value) for value in (sx, sy, gx, gy)):
            return False
        if math.hypot(gx - sx, gy - sy) < 2.0:
            return False
        if other_agent_starts and any(
            math.hypot(sx - ax, sy - ay) < 1.0
            for ax, ay in other_agent_starts
        ):
            return False
        dynamic_positions = self._active_dyn_obs_spawns()
        if any(
            math.hypot(px - ox, py - oy) < 0.60
            for px, py in ((sx, sy), (gx, gy))
            for ox, oy in dynamic_positions
        ):
            return False
        if self.planner and not self.planner.plan((sx, sy), (gx, gy)):
            return False
        return True

    # ===================================================================
    # ROS 回调：激光雷达 / 里程计 / 碰撞接触
    # ===================================================================
    def _scan_callback(self, msg):
        if self.sensor_noise_std > 0.0 and getattr(msg, "ranges", None):
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            finite = np.isfinite(ranges)
            noise = np.random.normal(0.0, self.sensor_noise_std, size=ranges.shape)
            ranges[finite] = np.clip(
                ranges[finite] + noise[finite],
                0.0,
                float(self.scan_max_range),
            )
            msg.ranges = ranges.tolist()
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

    # ===================================================================
    # 雷达扇区感知：扇区指标 / 池化特征 / 聚类 / 跟踪
    # ===================================================================
    def _scan_sector_metrics(self):
        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            return {
                'min_dist': self.scan_max_range,
                'front_min': self.scan_max_range,
                'front_left_min': self.scan_max_range,
                'front_center_min': self.scan_max_range,
                'front_right_min': self.scan_max_range,
                'left_min': self.scan_max_range,
                'right_min': self.scan_max_range,
                'rear_min': self.scan_max_range,
                'corridor_width': 2.0 * self.scan_max_range,
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
                'front_left_min': min_dist,
                'front_center_min': min_dist,
                'front_right_min': min_dist,
                'left_min': min_dist,
                'right_min': min_dist,
                'rear_min': min_dist,
                'corridor_width': 2.0 * min_dist,
                'clearance_asymmetry': 0.0,
            }

        angles = np.asarray(self._scan_angles(n), dtype=np.float32)
        angles = (angles + math.pi) % (2.0 * math.pi) - math.pi

        def _sector_min(mask):
            vals = ranges[np.asarray(mask, dtype=bool)]
            vals = vals[(vals > self.scan_valid_min)]
            return float(vals.min()) if vals.size else self.scan_max_range

        deg = math.pi / 180.0
        front_center = _sector_min(np.abs(angles) <= 20.0 * deg)
        front_left = _sector_min((angles > 20.0 * deg) & (angles <= 60.0 * deg))
        front_right = _sector_min((angles < -20.0 * deg) & (angles >= -60.0 * deg))
        left = _sector_min((angles > 60.0 * deg) & (angles <= 120.0 * deg))
        right = _sector_min((angles < -60.0 * deg) & (angles >= -120.0 * deg))
        rear = _sector_min(np.abs(angles) >= 150.0 * deg)

        return {
            'min_dist': min_dist,
            'front_min': front_center,
            'front_left_min': front_left,
            'front_center_min': front_center,
            'front_right_min': front_right,
            'left_min': left,
            'right_min': right,
            'rear_min': rear,
            'corridor_width': float(left + right),
            'clearance_asymmetry': float(left - right),
        }

    def _extract_filtered_scan_features(self, ranges: np.ndarray) -> np.ndarray:
        """强化感知：固定扇区池化 (Fixed Sector Pooling)"""
        sector_dists = self._compute_front_sector_min_dists(ranges)
        feat = np.maximum(
            0.0,
            (self.obstacle_filter_range - sector_dists) / max(self.obstacle_filter_range, 1e-6),
        ).astype(np.float32)
        return feat

    def _compute_front_sector_min_dists(self, ranges: np.ndarray) -> np.ndarray:
        n = int(ranges.size)
        sector_dists = np.full(self.obstacle_top_k, self.scan_max_range, dtype=np.float32)
        if n <= 0:
            return sector_dists

        sector_edges = np.linspace(n // 4, 3 * n // 4, self.obstacle_top_k + 1, dtype=int)
        for i in range(self.obstacle_top_k):
            idx_start = int(np.clip(sector_edges[i], 0, n))
            idx_end = int(np.clip(sector_edges[i + 1], idx_start + 1, n))
            sector_ranges = ranges[idx_start:idx_end]
            valid = sector_ranges[sector_ranges > self.scan_valid_min]
            sector_dists[i] = float(valid.min()) if valid.size > 0 else self.scan_max_range
        return sector_dists

    def _front_sector_center_angle(self, sector_idx: int) -> float:
        if self.obstacle_top_k <= 1:
            return 0.0
        return -0.5 * math.pi + (float(sector_idx) + 0.5) * (math.pi / float(self.obstacle_top_k))

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
        # 取最近的 3*top_k 个候选聚类，后续再按需裁剪到 top_k。
        return extracted[: self.obstacle_motion_top_k * 3]

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

    # ===================================================================
    # 几何工具：体坐标系 ↔ 世界坐标系、角度 wrap、agent 排序
    # ===================================================================
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
        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            return np.full(self.obstacle_top_k, self.scan_max_range, dtype=np.float32)
        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        return self._compute_front_sector_min_dists(ranges)

    # ===================================================================
    # gap / 正面冲突 检测 + head-on 避让奖励塑形
    # ===================================================================
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

    def _get_head_on_conflict_state(self) -> Optional[Dict[str, float]]:
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

    # ===================================================================
    # 图特征：邻居预测特征 / 障碍物运动特征
    # ===================================================================
    def _get_neighbor_prediction_features(self) -> np.ndarray:
        """Raw neighbor token: [body_rel_x, body_rel_y, body_rel_vx, body_rel_vy, dist_norm, sin_heading, cos_heading] x top_k"""
        if self.neighbor_prediction_dim <= 0:
            return np.zeros(self.neighbor_prediction_dim, dtype=np.float32)

        my_aid = f"agent_{self.robot_id}"
        my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        my_yaw = self.current_pose['yaw']
        my_vel = np.array([
            self.current_vel_x * math.cos(my_yaw),
            self.current_vel_x * math.sin(my_yaw),
        ], dtype=np.float32)
        candidates = []
        adjacency_matrix = None
        if hasattr(self, 'parent_env'):
            adjacency_matrix = getattr(self.parent_env, '_last_adj_matrix', None)
        received = self.parent_env._get_received_neighbor_samples(my_aid, adjacency_matrix=adjacency_matrix) \
            if hasattr(self, 'parent_env') else []

        for _idx, dist, n_pos, n_vel in received:
            if dist > self.predictive_social_range:
                continue

            rel_pos = np.asarray(n_pos, dtype=np.float32) - my_pos
            neighbor_vel = np.asarray(n_vel, dtype=np.float32)
            rel_vel = neighbor_vel - my_vel
            body_rel_pos = self._world_to_body(rel_pos)
            body_rel_vel = self._world_to_body(rel_vel)
            dist_norm = float(np.clip(dist / self.predictive_social_range, 0.0, 1.0))

            # Get neighbor heading (from GT in training; from comm yaw in deployment)
            n_yaw = my_yaw
            if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'agents'):
                n_aid = f"agent_{_idx}"
                if n_aid in self.parent_env.agents:
                    n_yaw = float(self.parent_env.agents[n_aid].current_pose['yaw'])
            rel_heading = n_yaw - my_yaw
            token = np.array([
                float(np.clip(body_rel_pos[0] / self.predictive_social_range, -1.0, 1.0)),
                float(np.clip(body_rel_pos[1] / self.predictive_social_range, -1.0, 1.0)),
                float(np.clip(body_rel_vel[0] / 0.8, -1.0, 1.0)),
                float(np.clip(body_rel_vel[1] / 0.8, -1.0, 1.0)),
                dist_norm,
                math.sin(rel_heading),
                math.cos(rel_heading),
            ], dtype=np.float32)
            candidates.append((dist, token))

        candidates.sort(key=lambda item: item[0])
        features = np.zeros(self.neighbor_prediction_dim, dtype=np.float32)
        for idx, (_, token) in enumerate(candidates[:self.neighbor_prediction_top_k]):
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
        denom = max(self.obstacle_filter_range, 1e-6)
        corridor_half_width = max(self.predictive_min_sep, 0.45)
        robot_world_vel = self._body_to_world_vec(np.array(
            [float(self.current_vel_x), 0.0],
            dtype=np.float32,
        ))

        for cluster in current_clusters:
            matched = self._match_previous_cluster(cluster, prev_clusters)
            vx_world = 0.0
            vy_world = 0.0
            if matched is not None:
                raw_vx = float((float(cluster["xw"]) - float(matched["xw"])) / self.control_dt)
                raw_vy = float((float(cluster["yw"]) - float(matched["yw"])) / self.control_dt)

                # Smooth against the matched track. Position-rounded dictionary
                # keys change as an obstacle moves and therefore do not preserve
                # an EMA across frames.
                if "vx_world" in matched and "vy_world" in matched:
                    alpha = 0.3
                    prev_vx = float(matched["vx_world"])
                    prev_vy = float(matched["vy_world"])
                    vx_world = alpha * raw_vx + (1.0 - alpha) * prev_vx
                    vy_world = alpha * raw_vy + (1.0 - alpha) * prev_vy
                else:
                    vx_world, vy_world = raw_vx, raw_vy
            cluster["vx_world"] = float(vx_world)
            cluster["vy_world"] = float(vy_world)

            # Use one explicit horizon for graph tokens, TTC ranking, and the
            # predictive safety terms. This makes horizon sweeps auditable.
            predict_h = float(self.predictive_horizon_sec)

            rel_world = np.array([
                float(cluster["xw"]) - float(robot_world[0]),
                float(cluster["yw"]) - float(robot_world[1]),
            ], dtype=np.float32)
            rel_body = self._world_to_body(rel_world)
            obstacle_world_vel = np.array([vx_world, vy_world], dtype=np.float32)
            relative_world_vel = obstacle_world_vel - robot_world_vel
            future_rel_world = rel_world + relative_world_vel * predict_h
            future_rel_body = self._world_to_body(future_rel_world)
            vel_body = self._world_to_body(relative_world_vel)

            x = float(rel_body[0])
            y = float(rel_body[1])
            vx = float(vel_body[0])
            vy = float(vel_body[1])
            future_x = float(future_rel_body[0])
            future_y = float(future_rel_body[1])
            dist = float(math.hypot(x, y))
            future_dist = float(math.hypot(future_x, future_y))

            effective_dist = max(0.0, dist - INFLATION_RADIUS)
            effective_future_dist = max(0.0, future_dist - INFLATION_RADIUS)

            close_risk = float(np.clip((self.close_obstacle_dist - effective_dist) / self.close_obstacle_dist, 0.0, 1.0))
            future_risk = float(np.clip((self.predictive_min_sep - effective_future_dist) / self.predictive_min_sep, 0.0, 1.0))
            crossing_gate = float(np.clip(1.0 - abs(future_y) / corridor_half_width, 0.0, 1.0))
            forward_gate = float(np.clip((future_x + 0.15) / max(self.obstacle_filter_range, 1e-6), 0.0, 1.0))
            transverse_speed = abs(vy)
            crossing_risk = crossing_gate * forward_gate * float(np.clip(transverse_speed / 0.6, 0.0, 1.0))
            motion_metrics = compute_relative_motion_metrics(
                np.array([x, y], dtype=np.float32),
                np.array([vx, vy], dtype=np.float32),
                horizon=predict_h,
                clearance_radius=INFLATION_RADIUS,
            )
            closing_speed = float(motion_metrics["closing_speed"])
            ttc = float(motion_metrics["ttc"])
            ttc_risk = (
                float(np.clip((self.predictive_front_ttc_safe - ttc) / self.predictive_front_ttc_safe, 0.0, 1.0))
                if math.isfinite(ttc) else 0.0
            )
            risk = max(close_risk, future_risk, crossing_risk, ttc_risk)
            # 原: if risk <= 1e-4: continue
            if dist > self.obstacle_filter_range:
                continue

            obstacle_speed = math.hypot(vx_world, vy_world)
            is_dynamic = 1.0 if obstacle_speed > 0.05 else 0.0
            token = np.array([
                float(np.clip(x / denom, -1.0, 1.0)),
                float(np.clip(y / denom, -1.0, 1.0)),
                float(np.clip(vx / 0.8, -1.0, 1.0)),
                float(np.clip(vy / 0.8, -1.0, 1.0)),
                float(np.clip(future_x / denom, -1.0, 1.0)),
                float(np.clip(future_y / denom, -1.0, 1.0)),
                is_dynamic,
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

    # ===================================================================
    # A* 规划重试辅助方法
    # ===================================================================
    def _plan_with_retry(self, start_pos, goal_pos, blocked_points, initial_radius=0.70):
        """
        统一的A*规划重试方法（带多次重试）

        所有规划点统一应用重试机制

        Args:
            start_pos: (x, y) 起点
            goal_pos: (x, y) 终点
            blocked_points: List[(x, y)] 障碍物列表
            initial_radius: 初始block_radius（米）

        Returns:
            path: List[(x, y)] 或 None
        """
        # 尝试不同的 block_radius。窄通道地图的 initial_radius 通常已经较小，
        # 失败时继续放宽，而不是回到 0.50m 再把通道堵死。
        retry_radii = [initial_radius]
        for radius in [0.50, 0.35, 0.25]:
            if radius < initial_radius - 1e-6 and all(abs(radius - r) > 1e-6 for r in retry_radii):
                retry_radii.append(radius)

        for radius in retry_radii:
            path = self.planner.plan_with_dynamic_obstacles(
                start_pos,
                goal_pos,
                blocked_world_points=blocked_points,
                block_radius_m=radius
            )

            if path:
                if radius != initial_radius:
                    print(f"    Robot {self.robot_id}: A*重试成功 (block_radius={radius}m)")
                return path

        # 所有重试失败，返回None
        return None

    def _default_planning_block_radius(self) -> float:
        """Return the initial dynamic-obstacle inflation radius for A* planning."""
        if int(self.map_number) in (3, 4, 5):
            return 0.35
        return 0.70

    # ===================================================================
    # 子目标 / 重规划：A* 路径跟踪 + 死锁脱困 + 局部绕行
    # ===================================================================
    def _try_replan_due_to_deadlock(self) -> bool:
        if not self.replan_on_deadlock or self.planner is None:
            return False
        if self.current_step < self._next_replan_step:
            return False

        try:
            start = (float(self.current_pose['x']), float(self.current_pose['y']))
            goal = (float(self.goal_pos[0]), float(self.goal_pos[1]))
            blocked = []

            # 添加spawn的静态障碍物到blocked列表（优先从parent_env获取，保证多agent一致性）
            static_obs_list = []
            if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
                static_obs_list = self.parent_env.spawned_static_obstacles
            elif hasattr(self, 'spawned_static_obstacles'):
                static_obs_list = self.spawned_static_obstacles

            for obs_x, obs_y, obs_radius in static_obs_list:
                blocked.append((float(obs_x), float(obs_y)))

            # 添加动态障碍物的典型位置
            # 重规划时也需要考虑这些由obstacle_mover控制的dyn_obs_X
            dyn_obs_positions = self._active_dyn_obs_spawns()
            for pos in dyn_obs_positions:
                blocked.append((float(pos[0]), float(pos[1])))

            # 添加其他机器人位置和预测位置
            if hasattr(self, 'parent_env'):
                my_pos = np.array(start, dtype=np.float32)
                my_yaw = float(self.current_pose['yaw'])
                my_vel = np.array([
                    self.current_vel_x * math.cos(my_yaw),
                    self.current_vel_x * math.sin(my_yaw),
                ], dtype=np.float32)
                for aid, pos in self.parent_env.robot_positions.items():
                    if aid == f"agent_{self.robot_id}":
                        continue
                    rel = np.asarray(pos, dtype=np.float32) - my_pos
                    dist = float(np.linalg.norm(rel))
                    if dist > self.dynamic_replan_neighbor_dist or dist < 1e-6:
                        continue
                    body_rel = self._world_to_body(rel)
                    if float(body_rel[0]) < -0.20:
                        continue
                    neighbor_vel = np.asarray(
                        self.parent_env.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
                        dtype=np.float32,
                    )
                    rel_unit = rel / max(dist, 1e-6)
                    closing_speed = float(-np.dot(neighbor_vel - my_vel, rel_unit))
                    ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')
                    # 注意：上方 2266 行已过滤 dist > neighbor_dist，故此处左支 dist < neighbor_dist
                    # 几乎恒为真——当前等价于"所有近邻一律视为障碍"，TTC 右支实际不生效。
                    # 这是偏保守的安全行为；若想让 dynamic_replan_ttc 真正起筛选作用，
                    # 应把左支改成更小的硬阻塞半径（如 dynamic_replan_block_radius）。保持现状以免改变训练动态。
                    if dist < self.dynamic_replan_neighbor_dist or (math.isfinite(ttc) and ttc < self.dynamic_replan_ttc):
                        blocked.append((float(pos[0]), float(pos[1])))
                        predict_h = min(self.dynamic_replan_ttc, 0.8)
                        blocked.append((
                            float(pos[0] + neighbor_vel[0] * predict_h),
                            float(pos[1] + neighbor_vel[1] * predict_h),
                        ))

            # 使用统一的重试方法
            path = self._plan_with_retry(
                start,
                goal,
                blocked,
                initial_radius=self.dynamic_replan_block_radius
            )

            if path is None:
                print(
                    f"  Robot {self.robot_id}: deadlock/progress replan 失败，"
                    f"保持原 A* 路径，不发布忽略动态/临时障碍物的 fallback 路径"
                )

            self._next_replan_step = self.current_step + self.replan_cooldown_steps
            if not path:
                return False
            self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
            self.current_waypoint_index = 0
            path_clearance = self.planner.path_min_clearance_m(self.global_waypoints)
            if not self.planner.validate_world_path(self.global_waypoints):
                print(
                    f"  Robot {self.robot_id}: replan 路径线段校验失败，"
                    f"min_clearance={path_clearance:.3f}m，拒绝刷新全局路径"
                )
                return False
            self._global_path_valid = True
            self.path_progress = 0.0
            self.prev_path_progress = 0.0
            self._subgoal_deadlock_streak = 0
            self._progress_replan_window_start_step = self.current_step
            self._progress_replan_window_start_progress = 0.0
            if hasattr(self, 'vis') and self.vis:
                try:
                    self.vis.publish_waypoints(
                        self.global_waypoints,
                        robot_id=self.robot_id,
                        namespace=self.vis_namespace,
                    )
                    print(
                        f"🔄 Robot {self.robot_id}: deadlock/progress replan 成功，"
                        f"min_clearance={path_clearance:.3f}m，已刷新 RViz A* 路径"
                    )
                except Exception as exc:
                    self.node.get_logger().warn(f'publish replanned waypoints failed: {exc}')
            return True
        except Exception:
            self._next_replan_step = self.current_step + self.replan_cooldown_steps
            return False

    def _maybe_replan_on_stalled_progress(self, arc_progress: float) -> bool:
        """Trigger A* replan when path progress stalls over a fixed step window."""
        if not self.replan_on_deadlock:
            return False
        if self.current_step < self._next_replan_step:
            return False

        progress = float(arc_progress)
        if self.current_step <= self._progress_replan_window_start_step:
            self._progress_replan_window_start_step = self.current_step
            self._progress_replan_window_start_progress = progress
            return False

        elapsed = self.current_step - self._progress_replan_window_start_step
        delta = progress - self._progress_replan_window_start_progress
        if delta >= self.progress_replan_min_delta:
            self._progress_replan_window_start_step = self.current_step
            self._progress_replan_window_start_progress = progress
            return False
        if elapsed < self.progress_replan_window_steps:
            return False

        if self._try_replan_due_to_deadlock():
            return True

        self._progress_replan_window_start_step = self.current_step
        self._progress_replan_window_start_progress = progress
        return False

    def _select_local_detour_subgoal(
        self,
        nominal_subgoal: Tuple[float, float],
        adaptive_lookahead: float,
        front_min: float,
        left_min: float,
        right_min: float,
        sector_dists: Optional[np.ndarray] = None,
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

        forward_speed = abs(float(getattr(self, 'current_vel_x', 0.0)))
        if front_min < self.subgoal_deadlock_front_dist and forward_speed < self.subgoal_deadlock_speed_thresh:
            self._subgoal_deadlock_streak += 1
        else:
            self._subgoal_deadlock_streak = max(0, self._subgoal_deadlock_streak - 1)

        blocked = (front_min < self.subgoal_block_front_dist) and (float(rel_body[0]) > 0.12)
        force_detour = self._subgoal_deadlock_streak >= self.subgoal_deadlock_steps

        # Event-triggered replan: only when subgoal direction is occluded by lidar
        # (reactive global guidance - closer to Nav2 behavior than periodic replanning)
        # subgoal_blocked: lidar shows obstacle within 0.6m along the subgoal direction
        subgoal_blocked = blocked and float(rel_body[0]) > 0.20 and front_min < 0.60

        if force_detour and self._try_replan_due_to_deadlock():
            return nominal_subgoal, 'replan'
        if subgoal_blocked and self._try_replan_due_to_deadlock():
            return nominal_subgoal, 'replan'

        conflict = self._get_head_on_conflict_state()
        if conflict is not None:
            same_partner = str(conflict['partner']) == str(self._yield_partner)
            should_hold = self._yield_hold_steps > 0 and same_partner and float(conflict['dist']) < (self.yielding_soft_dist + 0.30)
            if should_hold:
                self._yield_hold_steps = max(0, self._yield_hold_steps - 1)
                lookahead = max(0.25, float(adaptive_lookahead))
                x_body = max(0.08, 0.35 * lookahead)
                y_body = self._yield_turn_sign * max(0.18, 0.80 * lookahead)
                return self._body_to_world_point(x_body, y_body), 'yield'

            if (
                float(conflict['should_yield']) > 0.5
                and float(conflict['dist']) < self.yielding_soft_dist
                and (not math.isfinite(float(conflict['ttc'])) or float(conflict['ttc']) < self.yielding_ttc)
            ):
                self._yield_partner = str(conflict['partner'])
                self._yield_turn_sign = float(conflict['turn_sign'])
                self._yield_hold_steps = self.yielding_commit_steps
                lookahead = max(0.25, float(adaptive_lookahead))
                x_body = max(0.06, 0.30 * lookahead)
                y_body = self._yield_turn_sign * max(0.18, 0.90 * lookahead)
                return self._body_to_world_point(x_body, y_body), 'yield'
        else:
            self._yield_hold_steps = 0
            self._yield_partner = ''
            self._yield_turn_sign = 0.0

        if not blocked and not force_detour:
            self._subgoal_detour_hold = max(0, self._subgoal_detour_hold - 1)
            if self._subgoal_detour_hold == 0:
                self._subgoal_detour_side = 0
            return nominal_subgoal, 'nominal'

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
                    self._subgoal_detour_side = 1 if y_body >= 0.0 else -1
                    self._subgoal_detour_hold = self.subgoal_detour_hold_steps
                    return cand, 'gap_detour'

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

        # preferred_side 在上方逻辑中恒为 ±1，按"首选侧、再反向侧"的顺序尝试。
        candidates = [preferred_side, -preferred_side]
        for side in candidates:
            side_clear = left_min if side > 0 else right_min
            if side_clear < self.subgoal_min_side_clearance:
                continue
            cand = self._body_to_world_point(forward_step, side * lateral_step)
            if abs(self._get_target_angle(cand)) > 1.45:
                continue
            self._subgoal_detour_side = int(side)
            self._subgoal_detour_hold = self.subgoal_detour_hold_steps
            return cand, 'detour'

        return nominal_subgoal, ('deadlock' if force_detour else 'blocked_nominal')

    def _get_tracking_target(self):
        pos = (self.current_pose['x'], self.current_pose['y'])
        if self.global_waypoints:
            path_points = self.global_waypoints
        elif self.planner is None:
            path_points = [self.goal_pos]
        else:
            self.current_projection = pos
            self.current_subgoal = pos
            self.current_path_heading = 0.0
            self.path_progress = 0.0
            self.current_lateral_error = 0.0
            self.current_waypoint_index = 0
            self._last_subgoal_mode = 'no_global_path'
            return self.current_subgoal

        # rolling_lookahead_dist <= 0 时，不使用 rolling subgoal，
        # 但仍基于全局路径计算 projection / arc_progress，避免进度奖励退化为欧式距离差。
        if self.lookahead_dist <= 0.0:
            try:
                proj = PathTrackingUtils.get_path_projection(pos, path_points)
                self.current_projection = tuple(proj['projection'])
                self.current_subgoal = tuple(self.goal_pos)
                self.path_progress = float(proj.get('arc_progress', 0.0))
                self.current_lateral_error = float(proj.get('lateral_error', 0.0))

                seg_idx = int(proj.get('segment_index', 0))
                if len(path_points) >= 2:
                    i = int(np.clip(seg_idx, 0, len(path_points) - 2))
                    a, b = path_points[i], path_points[i + 1]
                    self.current_path_heading = float(math.atan2(b[1] - a[1], b[0] - a[0]))
                    self.current_waypoint_index = int(np.clip(i + 1, 0, len(path_points) - 1))
                else:
                    self.current_path_heading = 0.0
                    self.current_waypoint_index = 0
                self._last_subgoal_mode = 'goal_only'
                return self.current_subgoal
            except Exception:
                self.current_projection = None
                self.current_subgoal = tuple(self.goal_pos)
                self.current_path_heading = 0.0
                self.path_progress = 0.0
                self.current_lateral_error = 0.0
                self._last_subgoal_mode = 'goal_only_fallback'
                return self.current_subgoal

        try:
            info = PathTrackingUtils.get_rolling_subgoal(
                pos,
                path_points,
                self.lookahead_dist,
                reference_progress=self.path_progress,
            )
            if self._maybe_replan_on_stalled_progress(float(info.get('arc_progress', 0.0))):
                path_points = self.global_waypoints if self.global_waypoints else path_points
                info = PathTrackingUtils.get_rolling_subgoal(
                    pos,
                    path_points,
                    self.lookahead_dist,
                    reference_progress=self.path_progress,
                )
                self._progress_replan_window_start_step = self.current_step
                self._progress_replan_window_start_progress = float(info.get('arc_progress', 0.0))

            self.current_projection = tuple(info['projection'])
            self.current_subgoal = tuple(info['subgoal'])
            self._last_subgoal_mode = 'path_roll' if self.current_step < self._next_replan_step else 'nominal'
            self.current_path_heading = float(info.get('path_heading', 0.0))
            self.path_progress = float(info.get('arc_progress', 0.0))
            self.current_lateral_error = float(info.get('lateral_error', 0.0))
            seg_idx = int(info.get('segment_index', 0))
            self.current_waypoint_index = int(np.clip(seg_idx + 1, 0, max(len(path_points) - 1, 0)))
            return self.current_subgoal
        except Exception:
            self.current_projection = None
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
            label = f'R{self.robot_id} rolling_subgoal' if self.lookahead_dist > 0.0 else f'R{self.robot_id} goal_target'
            self.vis.publish_tracking_state(
                robot_pos=(self.current_pose['x'], self.current_pose['y']),
                target_pos=target_pos,
                projection_pos=self.current_projection,
                robot_id=self.robot_id,
                namespace=self.vis_namespace,
                label=label,
            )
        except Exception as _vis_e:
            self.node.get_logger().warn(f'publish_tracking_state failed: {_vis_e}')

    # ===================================================================
    # 安全护盾：基于雷达/邻居距离的硬约束（前方阻挡时减速/停止/原地转向）
    # ===================================================================

    def _predict_swept_clearance(self, linear_vel: float, angular_vel: float) -> float:
        """Predict minimum center-to-obstacle clearance along a unicycle arc."""
        if self.latest_scan is None or not getattr(self.latest_scan, 'ranges', None):
            return self.scan_max_range

        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        max_check_range = max(
            1.0,
            abs(float(linear_vel)) * self.swept_footprint_horizon
            + self.swept_footprint_soft_clearance + 0.45,
        )
        valid = (
            (ranges > self.scan_valid_min)
            & (ranges <= min(self.scan_max_range, max_check_range))
        )
        if not np.any(valid):
            return self.scan_max_range

        angles = self._scan_angles(len(ranges))[valid]
        obstacle_x = ranges[valid] * np.cos(angles)
        obstacle_y = ranges[valid] * np.sin(angles)
        horizon = float(self.swept_footprint_horizon)
        times = np.linspace(0.0, horizon, 9, dtype=np.float32)
        v = float(linear_vel)
        w = float(angular_vel)
        if abs(w) < 1e-4:
            center_x = v * times
            center_y = np.zeros_like(times)
        else:
            radius = v / w
            center_x = radius * np.sin(w * times)
            center_y = radius * (1.0 - np.cos(w * times))

        min_clearance = self.scan_max_range
        for cx, cy in zip(center_x, center_y):
            clearance = np.hypot(obstacle_x - float(cx), obstacle_y - float(cy))
            if clearance.size:
                min_clearance = min(min_clearance, float(np.min(clearance)))
        return float(min_clearance)

    def _predict_command_motion_safety(self, linear_vel: float, angular_vel: float) -> Dict[str, Any]:
        """Predict command clearance to moving robots and LiDAR tracks."""
        horizon = max(float(self.control_dt), float(self.predictive_horizon_sec))
        times = np.linspace(
            min(float(self.control_dt), horizon),
            horizon,
            9,
            dtype=np.float32,
        )
        ego_path = unicycle_body_trajectory(linear_vel, angular_vel, times)
        best_margin = float(self.scan_max_range)
        best_clearance = float(self.scan_max_range)
        best_source = 'none'

        def _consider(
            rel_pos_body: np.ndarray,
            object_vel_body: np.ndarray,
            required_clearance: float,
            source: str,
        ) -> None:
            nonlocal best_margin, best_clearance, best_source
            object_path = (
                np.asarray(rel_pos_body, dtype=np.float32).reshape(1, 2)
                + times.reshape(-1, 1) * np.asarray(object_vel_body, dtype=np.float32).reshape(1, 2)
            )
            clearances = np.linalg.norm(object_path - ego_path, axis=1)
            if clearances.size <= 0:
                return
            clearance = float(np.min(clearances))
            margin = clearance - float(required_clearance)
            if margin < best_margin:
                best_margin = margin
                best_clearance = clearance
                best_source = source

        if hasattr(self, 'parent_env'):
            my_aid = f"agent_{self.robot_id}"
            my_pos = np.array(
                [self.current_pose['x'], self.current_pose['y']],
                dtype=np.float32,
            )
            done_agents = set(getattr(self.parent_env, 'dones', set()))
            agent_clearance = max(2.0 * ROBOT_RADIUS + 0.12, 0.42)
            for aid, pos in self.parent_env.robot_positions.items():
                if aid == my_aid or aid in done_agents:
                    continue
                rel_world = np.asarray(pos, dtype=np.float32) - my_pos
                if float(np.linalg.norm(rel_world)) > self.predictive_social_range + 0.5:
                    continue
                neighbor_vel_world = np.asarray(
                    self.parent_env.robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
                    dtype=np.float32,
                )
                _consider(
                    self._world_to_body(rel_world),
                    self._world_to_body(neighbor_vel_world),
                    agent_clearance,
                    f'agent:{aid}',
                )

        if self.obstacle_motion_dim > 0 and self._last_motion_features is not None:
            features = np.asarray(self._last_motion_features, dtype=np.float32)
            pos_scale = max(float(self.obstacle_filter_range), 1e-6)
            current_ego_vel_body = np.array([self.current_vel_x, 0.0], dtype=np.float32)
            obstacle_clearance = max(INFLATION_RADIUS, ROBOT_RADIUS + 0.12)
            for idx in range(self.obstacle_motion_top_k):
                start = idx * self.obstacle_motion_feature_dim
                if start + 6 >= features.size or float(features[start + 6]) < 0.5:
                    continue
                rel_pos_body = np.array([
                    float(features[start]) * pos_scale,
                    float(features[start + 1]) * pos_scale,
                ], dtype=np.float32)
                if float(np.linalg.norm(rel_pos_body)) > self.obstacle_filter_range + 0.2:
                    continue
                rel_vel_body = np.array([
                    float(features[start + 2]) * 0.8,
                    float(features[start + 3]) * 0.8,
                ], dtype=np.float32)
                _consider(
                    rel_pos_body,
                    rel_vel_body + current_ego_vel_body,
                    obstacle_clearance,
                    f'dynamic_obstacle:{idx}',
                )

        return {
            'margin': float(best_margin),
            'clearance': float(best_clearance),
            'source': best_source,
        }

    def _project_motion_safe_command(
        self,
        linear_vel: float,
        angular_vel: float,
        target_angle: float,
        *,
        soft_filter: bool,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Select the closest command whose predicted moving-object margin is safe."""
        initial = self._predict_command_motion_safety(linear_vel, angular_vel)
        target_margin = 0.08 if soft_filter else 0.0
        if float(initial['margin']) >= target_margin:
            return float(linear_vel), float(angular_vel), {**initial, 'active': False}

        original_v = float(linear_vel)
        original_w = float(angular_vel)
        linear_candidates = [
            original_v,
            original_v * 0.75,
            original_v * 0.50,
            original_v * 0.25,
            0.0,
        ]
        turn_sign = float(np.sign(target_angle))
        if abs(turn_sign) < 1e-6:
            turn_sign = 1.0
        angular_candidates = [
            original_w,
            float(np.clip(original_w + 0.45, -self.max_angular_vel, self.max_angular_vel)),
            float(np.clip(original_w - 0.45, -self.max_angular_vel, self.max_angular_vel)),
            float(np.clip(turn_sign * max(0.70, abs(original_w)), -self.max_angular_vel, self.max_angular_vel)),
            float(np.clip(-turn_sign * max(0.70, abs(original_w)), -self.max_angular_vel, self.max_angular_vel)),
        ]

        candidates = []
        seen = set()
        for candidate_v in linear_candidates:
            for candidate_w in angular_candidates:
                key = (round(float(candidate_v), 5), round(float(candidate_w), 5))
                if key in seen:
                    continue
                seen.add(key)
                safety = self._predict_command_motion_safety(candidate_v, candidate_w)
                deviation = (
                    abs(float(candidate_v) - original_v) / max(self.max_forward_vel, 1e-6)
                    + 0.35 * abs(float(candidate_w) - original_w) / max(self.max_angular_vel, 1e-6)
                )
                candidates.append((float(candidate_v), float(candidate_w), safety, float(deviation)))

        safe = [item for item in candidates if float(item[2]['margin']) >= target_margin]
        if safe:
            selected = min(safe, key=lambda item: item[3])
        else:
            selected = max(candidates, key=lambda item: float(item[2]['margin']) - 0.04 * item[3])
        selected_v, selected_w, selected_safety, _ = selected
        return selected_v, selected_w, {**selected_safety, 'active': True}

    def _apply_safety_shield(
        self,
        linear_vel,
        angular_vel,
        front_min,
        left_min,
        right_min,
        target_angle,
        neighbor_min_dist=None,
        front_left_min=None,
        front_right_min=None,
    ):
        front_left_min = float(front_min if front_left_min is None else front_left_min)
        front_right_min = float(front_min if front_right_min is None else front_right_min)
        swept_before = self._predict_swept_clearance(linear_vel, angular_vel)
        motion_before = self._predict_command_motion_safety(linear_vel, angular_vel)
        if not self.shield_enable:
            self._last_swept_clearance = float(swept_before)
            self._last_swept_filter_active = False
            self._last_motion_safety_margin = float(motion_before['margin'])
            self._last_motion_safety_source = str(motion_before['source'])
            self._last_motion_filter_active = False
            return float(linear_vel), float(angular_vel), False

        emergency_turn = (
            front_min < max(self.turn_in_place_front_dist, 0.48)
            and abs(target_angle) > self.turn_in_place_angle_thresh
        )
        emergency_stop = front_min < max(self.shield_front_stop_dist, self.collision_persist_dist + 0.04)
        emergency_swept = swept_before < self.swept_footprint_hard_clearance
        emergency_motion = float(motion_before['margin']) < 0.0
        turn_in_place = False
        if front_min < self.shield_front_slow_dist:
            linear_vel = min(linear_vel, self.shield_linear_slow)
        if front_min < self.shield_front_stop_dist:
            linear_vel = min(linear_vel, self.shield_linear_stop)
        if neighbor_min_dist is not None and neighbor_min_dist < self.shield_neighbor_slow_dist:
            linear_vel = min(linear_vel, self.shield_linear_slow)

        if emergency_turn or (
            front_min < self.turn_in_place_front_dist and abs(target_angle) > self.turn_in_place_angle_thresh
        ):
            turn_in_place = True
            linear_vel = min(linear_vel, 0.02)
            turn_dir = np.sign(target_angle)
            if abs(turn_dir) < 1e-6:
                turn_dir = 1.0 if left_min >= right_min else -1.0
            min_turn_w = max(0.70, self.turn_in_place_w)
            angular_vel = float(np.clip(min_turn_w * turn_dir, -1.2, 1.2))
        elif front_min < self.turn_in_place_front_dist:
            bias = self.shield_turn_bias if left_min >= right_min else -self.shield_turn_bias
            angular_vel = float(np.clip(angular_vel + bias, -1.2, 1.2))

        # A close diagonal wall is safe while moving parallel, but dangerous
        # when the commanded arc sweeps the robot center toward it.
        turning_clearance = (
            min(front_left_min, float(left_min))
            if float(angular_vel) > 0.05 else
            min(front_right_min, float(right_min))
            if float(angular_vel) < -0.05 else
            min(front_left_min, front_right_min)
        )
        directional_turn_risk = (
            abs(float(angular_vel)) > 0.05
            and turning_clearance < self.swept_footprint_soft_clearance
        )
        target_clearance = (
            self.swept_footprint_soft_clearance
            if self.shield_enable else self.swept_footprint_hard_clearance
        )
        swept_filter_active = bool(
            emergency_swept
            or (self.shield_enable and swept_before < self.swept_footprint_soft_clearance)
            or (self.shield_enable and directional_turn_risk)
        )
        if swept_filter_active and abs(float(linear_vel)) > 1e-4:
            original_v = float(linear_vel)
            selected_v = 0.0
            selected_clearance = self._predict_swept_clearance(0.0, angular_vel)
            for scale in (0.75, 0.50, 0.25, 0.0):
                candidate_v = original_v * scale
                candidate_clearance = self._predict_swept_clearance(candidate_v, angular_vel)
                if candidate_clearance >= target_clearance:
                    selected_v = candidate_v
                    selected_clearance = candidate_clearance
                    break
                if scale == 0.0:
                    selected_clearance = candidate_clearance
            linear_vel = float(selected_v)
            self._last_swept_clearance = float(selected_clearance)
        else:
            self._last_swept_clearance = float(
                self._predict_swept_clearance(linear_vel, angular_vel)
            )
        self._last_swept_filter_active = bool(swept_filter_active)
        if swept_filter_active and abs(float(linear_vel)) <= 0.02 and abs(float(angular_vel)) > 0.05:
            turn_in_place = True

        linear_vel, angular_vel, motion_projection = self._project_motion_safe_command(
            linear_vel,
            angular_vel,
            target_angle,
            soft_filter=self.shield_enable,
        )
        self._last_motion_safety_margin = float(motion_projection['margin'])
        self._last_motion_safety_source = str(motion_projection['source'])
        self._last_motion_filter_active = bool(motion_projection['active'])
        if self._last_motion_filter_active and abs(float(linear_vel)) <= 0.02 and abs(float(angular_vel)) > 0.05:
            turn_in_place = True

        return float(linear_vel), float(angular_vel), turn_in_place

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

    # ===================================================================
    # 预测式风险特征（社交 TTC / 前方 TTC）—— 注入观测
    # ===================================================================
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

    # ===================================================================
    # Episode 重置：spawn 起终点、A* 规划路径、清理历史状态
    # ===================================================================
    def reset(
        self,
        seed=None,
        options=None,
        other_agent_starts=None,
        forced_start_goal=None,
        forced_yaw=None,
        strict_forced_route=False,
    ):
        super().reset(seed=seed)
        self.current_step = 0
        self._publish_vel(0.0, 0.0)
        self._close_obstacle_streak = 0
        self._gazebo_collision_active = False
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
        self._next_replan_step = 0
        self._progress_replan_window_start_step = 0
        self._progress_replan_window_start_progress = 0.0
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0

        # 清理全局路径点（RViz可视化 + 内部列表）
        self.global_waypoints = []
        self._global_path_valid = False
        if hasattr(self, 'vis') and self.vis:
            self.vis.clear_waypoints(namespace=self.vis_namespace)

        # 注释掉旧的删除逻辑，改由 GNNMARLEnv.reset() 统一管理
        # 原因：静态障碍物应该在stage级别管理（首次spawn，后续复用），不应该在每次episode reset时删除
        # 删除操作现在只在以下情况发生：
        #   1. Stage切换时：通过 reset_stage_obstacles() 清理旧障碍物
        #   2. 首次spawn时：GNNMARLEnv.reset() 中调用 randomize_obstacles() 会先删后spawn
        # if self.robot_id == 0 and self.map_number in (8, 9) and self.random_obstacles:

        forced_ok = False
        if forced_start_goal is not None:
            try:
                (start_xy, goal_xy) = forced_start_goal
                start_x, start_y = float(start_xy[0]), float(start_xy[1])
                goal_x, goal_y = float(goal_xy[0]), float(goal_xy[1])
                # Random ±0.3m perturbation to break trivial symmetry solutions
                # (e.g., "all advance 0.7m to swap"). Map 8 only — keep other maps deterministic.
                if self.map_number == 8:
                    start_x += random.uniform(-0.3, 0.3)
                    start_y += random.uniform(-0.3, 0.3)
                    goal_x += random.uniform(-0.3, 0.3)
                    goal_y += random.uniform(-0.3, 0.3)
                if strict_forced_route:
                    forced_ok = self._is_valid_fixed_start_goal_pair(
                        (start_x, start_y),
                        (goal_x, goal_y),
                        other_agent_starts=other_agent_starts,
                    )
                else:
                    forced_ok = self._is_valid_start_goal_pair(
                        (start_x, start_y),
                        (goal_x, goal_y),
                        other_agent_starts=other_agent_starts,
                        min_agent_sep=1.0,
                    )
                if not forced_ok:
                    pass
                    # print(
                    #     f" Robot {self.robot_id}: 强制路线非法，回退随机安全采样 "
                    #     f"start=({start_x:.2f},{start_y:.2f}) goal=({goal_x:.2f},{goal_y:.2f})"
                    # )
            except Exception as exc:
                if strict_forced_route:
                    raise ValueError(
                        f'failed to parse fixed route for map {self.map_number}: '
                        f'agent={self.robot_id} route={forced_start_goal}'
                    ) from exc
                forced_ok = False

        if forced_start_goal is not None and strict_forced_route and not forced_ok:
            raise ValueError(
                f'fixed route is invalid for map {self.map_number}: '
                f'agent={self.robot_id} route={forced_start_goal}'
            )

        # 当 forced route 验证失败时（如 map4 库中某些路线在当前 spawn 配置下
        # 不满足 min_agent_sep 或 planner 失败），应该静默回退到下面的随机采样
        # 逻辑（line 4114+），而不是让整个 worker 崩溃。
        # if forced_start_goal is not None and not forced_ok:
        #     raise ValueError(
        #         f'fixed route is invalid for map {self.map_number}: '
        #         f'agent={self.robot_id} route={forced_start_goal}'
        #     )

        if not forced_ok and self.use_random_mode and self._valid_spawn_points:
            found_path = False
            reset_min_sep = float(getattr(getattr(self, 'parent_env', None), 'reset_min_agent_sep', 1.0))
            for min_sep in [1.5, 1.2, max(1.0, reset_min_sep)]:
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
                        pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=1.0
                    )
                ]
                fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
                (start_x, start_y), (goal_x, goal_y) = fallback
                print(f" Reset robot_{self.robot_id}: 随机点生成失败，使用备用位置 {fallback}")
        elif not forced_ok:
            fallback_list = self._MAP_FALLBACK_POSES.get(
                self.map_number, [((0.0, 0.0), (5.0, 5.0))])
            valid_fallbacks = [
                pair for pair in fallback_list
                if self._is_valid_start_goal_pair(
                    pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=1.0
                )
            ]
            fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
            (start_x, start_y), (goal_x, goal_y) = fallback

        fixed_route_still_valid = (
            self._is_valid_fixed_start_goal_pair(
                (start_x, start_y),
                (goal_x, goal_y),
                other_agent_starts=other_agent_starts,
            )
            if strict_forced_route else self._is_valid_start_goal_pair(
                (start_x, start_y),
                (goal_x, goal_y),
                other_agent_starts=other_agent_starts,
                min_agent_sep=1.0,
            )
        )
        if forced_ok and not fixed_route_still_valid:
                if strict_forced_route:
                    raise ValueError(
                        f'fixed route became invalid for map {self.map_number}: '
                        f'agent={self.robot_id} route={forced_start_goal}'
                    )
                fallback_list = self._MAP_COLLISION_ROUTE_LIBRARY.get(
                    self.map_number,
                    self._MAP_FALLBACK_POSES.get(self.map_number, [((0.0, 0.0), (5.0, 5.0))]),
                )
                valid_fallbacks = [
                    pair for pair in fallback_list
                    if self._is_valid_start_goal_pair(
                        pair[0], pair[1], other_agent_starts=other_agent_starts, min_agent_sep=1.0
                    )
                ]
                fallback = random.choice(valid_fallbacks if valid_fallbacks else fallback_list)
                (start_x, start_y), (goal_x, goal_y) = fallback

        self.last_spawn_pos = (start_x, start_y)
        self.goal_pos = (goal_x, goal_y)

        # ═══════════════════════════════════════════════════════════════════
        # 先spawn障碍物，再进行A*规划
        # 原因：A*规划器需要在障碍物存在时规划，否则规划的路径会穿过障碍物
        # 静态障碍物只在stage首次reset时spawn，后续episode复用
        # ═══════════════════════════════════════════════════════════════════

        # 在机器人spawn之前先spawn障碍物（只有robot_id==0 且 首次reset 时执行）
        if self.robot_id == 0 and not self._static_obstacles_spawned and self.num_static_obstacles > 0:
            # 收集所有机器人的spawn位置
            all_robot_positions = [(start_x, start_y)]
            if other_agent_starts:
                all_robot_positions.extend(other_agent_starts)
            print(f"🟫 Robot {self.robot_id}: 首次spawn静态障碍物 (stage级别，不会每episode重置)", flush=True)
            self._spawn_random_obstacles(other_robot_positions=all_robot_positions)
            self._static_obstacles_spawned = True  # 标记已spawn
        elif self.robot_id == 0 and self._static_obstacles_spawned:
            # 已经spawn过，直接使用现有障碍物
            print(f"♻️  Robot {self.robot_id}: 复用现有静态障碍物 (共{len(self.spawned_static_obstacles)}个)", flush=True)
        elif self.robot_id != 0:
            # 其他robot等待robot_0完成spawn（只在首次需要）
            if not self._static_obstacles_spawned and self.num_static_obstacles > 0:
                self._wait_for_sim_time(0.4)

        # 现在进行A*规划（必须使用plan_with_dynamic_obstacles考虑spawn的静态障碍物）
        if self.planner:
            # 获取spawn的静态障碍物位置
            blocked_points = []
            if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
                blocked_points = [(x, y) for x, y, _ in self.parent_env.spawned_static_obstacles]
            elif hasattr(self, 'spawned_static_obstacles'):
                blocked_points = [(x, y) for x, y, _ in self.spawned_static_obstacles]

            # 添加从激光雷达聚类中检测到的动态障碍物
            # 这些是world文件中的dyn_obs_X，由obstacle_mover.py移动
            if hasattr(self, '_obstacle_cluster_history') and self._obstacle_cluster_history:
                # 获取最近一帧的障碍物聚类
                recent_clusters = self._obstacle_cluster_history[-1] if self._obstacle_cluster_history else []
                for cluster in recent_clusters:
                    # cluster格式: {'center': (x, y), 'radius': r, ...}
                    if isinstance(cluster, dict) and 'center' in cluster:
                        cx, cy = cluster['center']
                        blocked_points.append((float(cx), float(cy)))
                    elif isinstance(cluster, (tuple, list)) and len(cluster) >= 2:
                        # 如果是简单的(x,y)元组
                        blocked_points.append((float(cluster[0]), float(cluster[1])))

            # 添加动态障碍物的典型位置（从_DYN_OBS_SPAWNS）
            # 这是保守估计，覆盖obstacle_mover控制的dyn_obs_X的活动区域
            dyn_obs_positions = self._active_dyn_obs_spawns()
            for pos in dyn_obs_positions:
                blocked_points.append((float(pos[0]), float(pos[1])))

            if len(blocked_points) == 0:
                print(f"  Robot {self.robot_id}: blocked_points为空！障碍物可能未spawn完成", flush=True)
                print(f"    hasattr(parent_env)={hasattr(self, 'parent_env')}", flush=True)
                if hasattr(self, 'parent_env'):
                    print(f"    hasattr(parent_env.spawned_static_obstacles)="
                          f"{hasattr(self.parent_env, 'spawned_static_obstacles')}", flush=True)
                    if hasattr(self.parent_env, 'spawned_static_obstacles'):
                        print(f"    len(parent_env.spawned_static_obstacles)="
                              f"{len(self.parent_env.spawned_static_obstacles)}", flush=True)
                print(f"    hasattr(self.spawned_static_obstacles)="
                      f"{hasattr(self, 'spawned_static_obstacles')}", flush=True)
                if hasattr(self, 'spawned_static_obstacles'):
                    print(f"    len(self.spawned_static_obstacles)="
                          f"{len(self.spawned_static_obstacles)}", flush=True)
            else:
                print(f"🗺️  Robot {self.robot_id}: A*规划 start=({start_x:.2f},{start_y:.2f}) "
                      f"goal=({goal_x:.2f},{goal_y:.2f}) blocked_points={len(blocked_points)}")
                if self.robot_id == 0:
                    # 只在robot_0打印详细位置，避免刷屏
                    for i, (bx, by) in enumerate(blocked_points[:3]):
                        print(f"    blocked[{i}]: ({bx:.2f}, {by:.2f})")
                    if len(blocked_points) > 3:
                        print(f"    ... 共{len(blocked_points)}个")

            # 动态/随机障碍物中心点膨胀半径。随机箱子地图保留 0.70m，
            # Map3/4/5 结构化窄通道直接用较小半径，否则会把走廊判死。
            block_radius = self._default_planning_block_radius()

            path = self.planner.plan_with_dynamic_obstacles(
                (start_x, start_y),
                (goal_x, goal_y),
                blocked_world_points=blocked_points,
                block_radius_m=block_radius
            )

            if self.robot_id == 0 and len(blocked_points) > 0:
                print(f"    A*参数: block_radius={block_radius}m, "
                      f"地图分辨率={self.planner.resolution:.3f}m/px")

            # A*失败时的增强诊断和重试机制
            if path:
                self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
                self.current_waypoint_index = 0
                self._global_path_valid = True
                path_clearance = self.planner.path_min_clearance_m(self.global_waypoints)
                print(
                    f" Robot {self.robot_id}: A*规划成功，路径长度={len(path)}点 "
                    f"→ waypoints={len(self.global_waypoints)}点, min_clearance={path_clearance:.3f}m"
                )
                self.vis.publish_waypoints(
                    self.global_waypoints,
                    robot_id=self.robot_id,
                    namespace=self.vis_namespace
                )
            else:
                print(f" Robot {self.robot_id}: A* plan 失败 start=({start_x:.2f},{start_y:.2f}) "
                      f"goal=({goal_x:.2f},{goal_y:.2f})")
                print(f"    blocked_points={len(blocked_points)}, block_radius={block_radius}m")

                # 详细诊断
                print(f"   📍 起点世界坐标: ({start_x:.2f}, {start_y:.2f})")
                print(f"   📍 终点世界坐标: ({goal_x:.2f}, {goal_y:.2f})")
                start_grid = self.planner.world_to_grid(start_x, start_y)
                goal_grid = self.planner.world_to_grid(goal_x, goal_y)
                print(f"   📍 起点网格坐标: {start_grid}")
                print(f"   📍 终点网格坐标: {goal_grid}")
                print(f"   🗺️  地图尺寸: {self.planner.width} × {self.planner.height}")
                print(f"   🗺️  地图原点: {self.planner.origin}")
                print(f"   🚧 障碍物数量: {len(blocked_points)}")

                # 检查边界
                start_in_bounds = (0 <= start_grid[0] < self.planner.width and
                                  0 <= start_grid[1] < self.planner.height)
                goal_in_bounds = (0 <= goal_grid[0] < self.planner.width and
                                 0 <= goal_grid[1] < self.planner.height)

                if not start_in_bounds:
                    print(f"     起点超出地图边界！")
                if not goal_in_bounds:
                    print(f"     终点超出地图边界！")

                # 尝试减小block_radius重试
                if len(blocked_points) > 0 and start_in_bounds and goal_in_bounds:
                    print(f"   🔄 尝试减小block_radius重试...")
                    for retry_radius in [0.50, 0.35, 0.25]:
                        if retry_radius >= block_radius - 1e-6:
                            continue
                        retry_path = self.planner.plan_with_dynamic_obstacles(
                            (start_x, start_y),
                            (goal_x, goal_y),
                            blocked_world_points=blocked_points,
                            block_radius_m=retry_radius
                        )
                        if retry_path:
                            print(f"    用block_radius={retry_radius}m重试成功！")
                            path = retry_path
                            break

                # 最终处理
                if path:
                    self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
                    self.current_waypoint_index = 0
                    self._global_path_valid = True
                    path_clearance = self.planner.path_min_clearance_m(self.global_waypoints)
                    print(f" Robot {self.robot_id}: 重试后A*规划成功, min_clearance={path_clearance:.3f}m")
                    self.vis.publish_waypoints(
                        self.global_waypoints,
                        robot_id=self.robot_id,
                        namespace=self.vis_namespace
                    )
                else:
                    print(f"     所有 A* 重试失败，保持原路径；禁止退化为穿障碍直线路径")

        yaw = (
            float(forced_yaw)
            if forced_ok and forced_yaw is not None
            else random.uniform(-3.14, 3.14)
        )

        self._set_robot_pose(start_x, start_y, yaw)
        pose_synced = self._wait_for_pose_sync(start_x, start_y, yaw)
        if not pose_synced:
            print(
                f" Robot {self.robot_id}: reset pose odom sync timeout "
                f"target=({start_x:.2f},{start_y:.2f}) "
                f"odom=({self.current_pose['x']:.2f},{self.current_pose['y']:.2f})",
                flush=True,
            )

        self.latest_scan = None
        self._scan_history.clear()
        self._obstacle_cluster_history.clear()
        self._wait_for_sim_time(0.2)

        self.prev_dist_to_goal = math.hypot(
            self.goal_pos[0] - self.current_pose['x'],
            self.goal_pos[1] - self.current_pose['y']
        )

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

        return self._get_obs(), {'start_xy': (start_x, start_y), 'start_yaw': yaw}

    # ===================================================================
    # 动作空间：连续 (linear,angular) / 离散原语 → 速度指令
    # ===================================================================
    def _decode_action_to_cmd_vel(self, action) -> Tuple[float, float]:
        """将策略动作解码为底盘速度 (v, w)。"""
        self._last_action_primitive = None

        if self.action_mode == 'continuous':
            arr = np.asarray(action, dtype=np.float32).reshape(-1)
            if arr.size < 2:
                raise ValueError(f'continuous action expects 2 dims, got {action}')

            a_lin = float(np.clip(arr[0], -1.0, 1.0))
            a_ang = float(np.clip(arr[1], -1.0, 1.0))

            if a_lin >= 0.0:
                linear_vel = a_lin * self.max_forward_vel
            else:
                linear_vel = a_lin * self.max_reverse_vel
            angular_vel = a_ang * self.max_angular_vel
            return float(linear_vel), float(angular_vel)

        if isinstance(action, np.ndarray):
            if action.size == 0:
                action_id = 0
            else:
                action_id = int(np.asarray(action).reshape(-1)[0])
        else:
            action_id = int(action)

        action_id = int(np.clip(action_id, 0, len(self.discrete_action_primitives) - 1))
        self._last_action_primitive = action_id
        linear_vel, angular_vel = self.discrete_action_primitives[action_id]
        linear_vel = float(np.clip(linear_vel, -self.max_reverse_vel, self.max_forward_vel))
        angular_vel = float(np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel))
        return linear_vel, angular_vel

    def apply_action(self, action, debug=False):
        self.current_step += 1
        linear_vel, angular_vel = self._decode_action_to_cmd_vel(action)

        sectors = self._scan_sector_metrics()
        front_min = float(sectors['front_min'])
        front_left_min = float(sectors.get('front_left_min', front_min))
        front_right_min = float(sectors.get('front_right_min', front_min))
        left_min = float(sectors['left_min'])
        right_min = float(sectors['right_min'])
        target_ref = self.current_subgoal if self.current_subgoal is not None else self.goal_pos
        target_angle = float(self._get_target_angle(target_ref))

        neighbor_min_dist = None
        if hasattr(self, 'parent_env'):
            my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
            for aid, pos in self.parent_env.robot_positions.items():
                if aid == f"agent_{self.robot_id}":
                    continue
                d = float(np.linalg.norm(np.asarray(pos, dtype=np.float32) - my_pos))
                if neighbor_min_dist is None or d < neighbor_min_dist:
                    neighbor_min_dist = d

        raw_linear_vel = float(linear_vel)
        raw_angular_vel = float(angular_vel)
        shield_start_ns = time.perf_counter_ns()
        linear_vel, angular_vel, turn_in_place = self._apply_safety_shield(
            linear_vel,
            angular_vel,
            front_min,
            left_min,
            right_min,
            target_angle,
            neighbor_min_dist=neighbor_min_dist,
            front_left_min=front_left_min,
            front_right_min=front_right_min,
        )
        shield_latency_ms = (time.perf_counter_ns() - shield_start_ns) / 1.0e6

        self._last_shield_info = {
            'front_min': front_min,
            'front_left_min': front_left_min,
            'front_right_min': front_right_min,
            'left_min': left_min,
            'right_min': right_min,
            'target_angle': target_angle,
            'neighbor_min_dist': float(neighbor_min_dist) if neighbor_min_dist is not None else float('inf'),
            'raw_linear_vel': raw_linear_vel,
            'raw_angular_vel': raw_angular_vel,
            'shielded_linear_vel': float(linear_vel),
            'shielded_angular_vel': float(angular_vel),
            'shield_active': bool(
                abs(raw_linear_vel - float(linear_vel)) > 1e-6
                or abs(raw_angular_vel - float(angular_vel)) > 1e-6
            ),
            'turn_in_place': bool(turn_in_place),
            'swept_clearance': float(self._last_swept_clearance),
            'swept_filter_active': bool(self._last_swept_filter_active),
            'motion_safety_margin': float(self._last_motion_safety_margin),
            'motion_safety_source': str(self._last_motion_safety_source),
            'motion_filter_active': bool(self._last_motion_filter_active),
            'safety_filter_latency_ms': float(shield_latency_ms),
        }

        if debug and abs(linear_vel) < 0.01 and abs(angular_vel) < 0.01:
            if self.action_mode == 'continuous':
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                print(f"  Robot {self.robot_id}: 零动作! action=[{arr[0]:.3f}, {arr[1]:.3f}] -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")
            else:
                print(f"  Robot {self.robot_id}: 零动作! primitive={self._last_action_primitive} -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")

        self._publish_vel(linear_vel, angular_vel)

    # ===================================================================
    # 奖励函数 & step 主流程
    #
    #   总奖励 = path_tracking_reward      (进度/朝向/横向偏差/会车规范)
    #          + avoidance_reward          (TTC/势场/预测/让行/正面避让)
    #          + collision_penalty         (碰撞事件)
    #          + goal_bonus                (到达目标)
    #          - effective_time_penalty    (时间惩罚)
    #
    # 关键调参旋钮（在 train_gnn_mappo_full.py::env_config 设置）：
    #   goal_reward / collision_penalty / time_penalty / progress_reward_scale
    #   subgoal_progress_reward_scale / safe_turn_reward_scale / head_on_avoidance_reward_scale
    #   risk_aware_forward_penalty_scale / predictive_*_penalty_scale
    # ===================================================================
    def get_step_result(self):
            """
            精简奖励函数 — 6 项正交结构:
              r_progress  : 朝目标前进
              r_static    : 静态障碍避碰
              r_social    : 动态避碰 (邻居)
              r_collision : 碰撞终端惩罚
              r_goal      : 到达终端奖励
              r_time      : 时间压力
            """
            reward = 0.0
            done = False
            truncated = False
            info = {}

            current_target = self._get_tracking_target()
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

            forward_speed = max(float(getattr(self, 'current_vel_x', 0.0)), 0.0)

            sectors = self._scan_sector_metrics()
            min_dist = float(sectors.get('min_dist', 10.0))
            front_min = float(sectors.get('front_min', min_dist))
            front_left_min = float(sectors.get('front_left_min', front_min))
            front_right_min = float(sectors.get('front_right_min', front_min))
            left_min = float(sectors.get('left_min', min_dist))
            right_min = float(sectors.get('right_min', min_dist))

            # ==========================================
            # 1. r_progress: 到达目标 + 沿 A* 路径推进 + 轻度贴线约束
            # ==========================================
            goal_dist_delta = 0.0
            if self.prev_dist_to_goal is not None:
                goal_dist_delta = float(self.prev_dist_to_goal - dist_to_goal)

            path_progress_delta = 0.0
            if self.prev_path_progress is not None:
                path_progress_delta = float(self.path_progress - self.prev_path_progress)
            path_progress_delta = float(np.clip(path_progress_delta, -0.25, 0.25))

            # heading shaping uses guidance direction (A* path direction)
            guidance_obs = self._get_guidance_obs()
            guidance_angle = math.atan2(float(guidance_obs[0]), float(guidance_obs[1]))
            heading_shaping = 0.0
            if forward_speed > RWD_HEADING_MIN_FWD_VEL:
                heading_shaping = RWD_HEADING_COEF * math.cos(guidance_angle)

            lateral_penalty = 0.0
            allowed_lateral_error = self._get_clearance_aware_lateral_limit(sectors)
            if self.lateral_penalty_scale > 0.0:
                lateral_excess = max(
                    0.0,
                    float(self.current_lateral_error) - allowed_lateral_error,
                )
                curvature_gate = float(np.clip(
                    (abs(float(self._last_path_curvature)) - 0.18) / 0.70,
                    0.0,
                    1.0,
                ))
                lateral_scale = self.lateral_penalty_scale * (1.0 + 2.0 * curvature_gate)
                lateral_penalty = -lateral_scale * min(lateral_excess, 2.0)

            r_progress = (
                self.progress_scale * (goal_dist_delta + heading_shaping)
                + self.path_progress_reward_scale * path_progress_delta
                + lateral_penalty
            )
            r_progress = float(np.clip(r_progress, -RWD_PROGRESS_CLIP, RWD_PROGRESS_CLIP))

            # ==========================================
            # 2. r_static: 障碍避碰（含动态障碍物）
            #    = 斥力势场 + 高风险减速 + 前方预测风险(含动态obs TTC)
            # ==========================================
            directional_terms = compute_directional_clearance_terms(
                front_center=front_min,
                front_left=front_left_min,
                front_right=front_right_min,
                left=left_min,
                right=right_min,
                min_dist=min_dist,
                forward_speed=forward_speed,
                angular_vel=float(getattr(self, 'current_vel_w', 0.0)),
                corridor_min_width=self.corridor_min_width,
                corridor_narrow_width=self.corridor_narrow_width,
                corridor_front_open_dist=self.corridor_front_open_dist,
            )
            repulsive = float(
                directional_terms['front_potential']
                + directional_terms['turn_potential']
                + directional_terms['hard_proximity_potential']
            )
            speed_risk = float(directional_terms['speed_risk'])
            corridor_center_penalty = -self.corridor_center_penalty_scale * float(
                directional_terms['corridor_narrow_gate']
                * directional_terms['corridor_center_error']
            )

            front_risk = float(self._last_predictive_metrics.get('front_risk', 0.0))
            predictive_front_penalty = -(front_risk ** 2)

            turn_clearance = float(directional_terms['turn_clearance'])
            turn_space = min(front_min, turn_clearance)
            narrow_turn_risk = float(np.clip((0.75 - turn_space) / 0.75, 0.0, 1.0))
            angle_gate = float(np.clip((abs_target_angle - 0.25) / 0.75, 0.0, 1.0))
            speed_norm = float(np.clip(forward_speed / max(self.max_forward_vel, 1e-6), 0.0, 1.0))
            narrow_forward_penalty = -self.narrow_forward_penalty_scale * narrow_turn_risk * angle_gate * speed_norm
            turn_dir = 0.0 if abs_target_angle < 1e-6 else math.copysign(1.0, target_angle)
            turn_alignment = float(np.clip(turn_dir * float(getattr(self, 'current_vel_w', 0.0)) / max(self.max_angular_vel, 1e-6), 0.0, 1.0))
            turn_safety_gate = float(np.clip(
                (turn_clearance - self.swept_footprint_hard_clearance)
                / max(self.swept_footprint_soft_clearance - self.swept_footprint_hard_clearance, 1e-6),
                0.0,
                1.0,
            ))
            safe_turn_reward = self.safe_turn_reward_scale * narrow_turn_risk * angle_gate * turn_alignment * turn_safety_gate

            # Near-miss penalty: only trigger when very close (< 20cm, near TB3 radius)
            # Gentle gradient to allow passage through narrow spaces (e.g. 1.2m corridors)
            near_miss_penalty = 0.0
            if min_dist < 0.18:
                near_miss_ratio = (0.18 - min_dist) / 0.18
                near_miss_penalty = -0.5 * (near_miss_ratio ** 2)  # reduced from -1.0

            r_static = self.static_scale * (
                repulsive
                + speed_risk
                + predictive_front_penalty
                + near_miss_penalty
                + corridor_center_penalty
                + narrow_forward_penalty
                + safe_turn_reward
            )
            r_static = float(max(-RWD_STATIC_CLIP, r_static))

            # ==========================================
            # 3. r_ttc: 统一TTC惩罚 (智能体 + 动态障碍物)
            #    物理本质相同: 避免与所有运动物体碰撞
            # ==========================================
            worst_ttc_penalty = 0.0
            dynamic_obs_count = 0
            dynamic_obs_min_dist = float(RWD_TTC_MAX_DIST)
            dynamic_obs_min_ttc = float(RWD_TTC_SAFE)
            dynamic_obs_ttc_risk = 0.0
            my_pos = np.array([self.current_pose['x'], self.current_pose['y']])
            my_vel = np.array([
                self.current_vel_x * math.cos(self.current_pose['yaw']),
                self.current_vel_x * math.sin(self.current_pose['yaw'])
            ])

            # 3a. 其他智能体的TTC
            if hasattr(self, 'parent_env'):
                for aid, pos in self.parent_env.robot_positions.items():
                    if aid == f"agent_{self.robot_id}":
                        continue
                    rel_pos = pos - my_pos
                    dist = float(np.linalg.norm(rel_pos))
                    if dist >= RWD_TTC_MAX_DIST:
                        continue

                    neighbor_vel = self.parent_env.robot_velocities[aid]
                    rel_vel = neighbor_vel - my_vel
                    motion_metrics = compute_relative_motion_metrics(
                        rel_pos,
                        rel_vel,
                        horizon=RWD_TTC_SAFE,
                        clearance_radius=2.0 * INFLATION_RADIUS,
                    )
                    approach_speed = float(motion_metrics['closing_speed'])

                    if approach_speed > RWD_TTC_APPROACH_TH:
                        ttc = float(motion_metrics['ttc'])
                        if ttc < RWD_TTC_SAFE:
                            penalty = -((RWD_TTC_SAFE - ttc) / RWD_TTC_SAFE) ** 2
                            if penalty < worst_ttc_penalty:
                                worst_ttc_penalty = penalty

            # 3b. 动态障碍物的TTC
            if self.obstacle_motion_dim > 0 and self._last_motion_features is not None:
                motion_features = self._last_motion_features
                for i in range(self.obstacle_motion_top_k):
                    start = i * 7
                    if start + 6 >= len(motion_features):
                        break

                    is_dynamic = motion_features[start + 6]
                    if is_dynamic < 0.5:
                        continue

                    dynamic_obs_count += 1

                    # 反归一化到机器人坐标。token 使用 obstacle_filter_range 归一化，
                    # 不能写死 5m，否则窄通道训练会低估动态障碍 TTC 风险。
                    motion_pos_scale = max(float(self.obstacle_filter_range), 1e-6)
                    obs_x = motion_features[start] * motion_pos_scale
                    obs_y = motion_features[start + 1] * motion_pos_scale
                    obs_vx = motion_features[start + 2] * 0.8
                    obs_vy = motion_features[start + 3] * 0.8

                    # Obstacle tokens store body-frame relative position and
                    # relative velocity (obstacle velocity minus ego velocity).
                    rel_pos = np.array([obs_x, obs_y], dtype=np.float32)
                    rel_vel = np.array([obs_vx, obs_vy], dtype=np.float32)
                    dist = float(np.linalg.norm(rel_pos))
                    dynamic_obs_min_dist = min(dynamic_obs_min_dist, dist)

                    if dist >= RWD_TTC_MAX_DIST:
                        continue

                    motion_metrics = compute_relative_motion_metrics(
                        rel_pos,
                        rel_vel,
                        horizon=RWD_TTC_SAFE,
                        clearance_radius=INFLATION_RADIUS,
                    )
                    approach_speed = float(motion_metrics['closing_speed'])

                    if approach_speed > RWD_TTC_APPROACH_TH:
                        ttc = float(motion_metrics['ttc'])
                        dynamic_obs_min_ttc = min(dynamic_obs_min_ttc, float(ttc))
                        if ttc < RWD_TTC_SAFE:
                            penalty = -((RWD_TTC_SAFE - ttc) / RWD_TTC_SAFE) ** 2
                            dynamic_obs_ttc_risk = max(dynamic_obs_ttc_risk, -penalty)
                            if penalty < worst_ttc_penalty:
                                worst_ttc_penalty = penalty

            r_ttc = self.social_scale * worst_ttc_penalty
            r_ttc = float(max(-RWD_TTC_CLIP, r_ttc))

            # ==========================================
            # 4-6. 终端 / 时间
            # ==========================================
            r_collision = 0.0
            r_goal = 0.0

            if self._check_collision_event(min_dist, info):
                r_collision = -float(self.collision_penalty)
                info['event'] = 'collision'
                if self.collision_ends_episode:
                    done = True

            if dist_to_goal < RWD_GOAL_REACH_RADIUS:
                r_goal = float(self.goal_reward)
                done = True
                info['event'] = 'goal'

            if self.current_step >= self.max_episode_steps:
                truncated = True

            r_time = -float(self.time_penalty)

            if self.reward_mode == 'rmappo':
                # R-MAPPO 基线奖励: Δgeodesic + 10·I_goal − 0.05·I_collision
                #   Δgeodesic  : 沿 A* 路径的弧长推进 (path_progress_delta, 已在上面按 ±0.25 截断)
                #   I_goal     : 到达目标 (r_goal>0 时置 1)
                #   I_collision: 碰撞事件 (r_collision<0 时置 1)
                # 不含 r_static / r_ttc / r_time — 稀疏几何奖励, 用于隔离"奖励设计"单一变量。
                goal_hit = 1.0 if r_goal > 0.0 else 0.0
                coll_hit = 1.0 if r_collision < 0.0 else 0.0
                reward = path_progress_delta + 10.0 * goal_hit - 0.05 * coll_hit
            else:
                reward = r_progress + r_static + r_ttc + r_collision + r_goal + r_time

            # 更新历史变量
            self.prev_dist_to_goal = dist_to_goal
            self.prev_dist_to_target = dist_to_target
            self.prev_path_progress = self.path_progress
            self.prev_target_point = tuple(current_target)
            self.prev_abs_target_angle = abs_target_angle

            info.update({
                'reward_total': float(reward),
                'r_progress': float(r_progress),
                'r_static': float(r_static),
                'r_ttc': float(r_ttc),
                'r_collision': float(r_collision),
                'r_goal': float(r_goal),
                'r_time': float(r_time),
                'dist_to_goal': float(dist_to_goal),
                'path_progress_delta': float(path_progress_delta),
                'lateral_error': float(self.current_lateral_error),
                'allowed_lateral_error': float(allowed_lateral_error),
                'path_clearance': float(self._last_path_clearance),
                'min_dist': float(min_dist),
                'front_min': float(front_min),
                'front_left_min': float(front_left_min),
                'front_right_min': float(front_right_min),
                'left_min': float(left_min),
                'right_min': float(right_min),
                'corridor_width': float(directional_terms['corridor_width']),
                'corridor_feasible': float(directional_terms['corridor_feasible']),
                'corridor_active': float(directional_terms['corridor_active']),
                'corridor_center_error': float(directional_terms['corridor_center_error']),
                'corridor_center_penalty': float(corridor_center_penalty),
                'directional_front_potential': float(directional_terms['front_potential']),
                'directional_turn_potential': float(directional_terms['turn_potential']),
                'directional_hard_proximity_potential': float(directional_terms['hard_proximity_potential']),
                'turn_clearance': float(turn_clearance),
                'guidance_near_weight': float(self._last_guidance_near_weight),
                'path_curvature': float(self._last_path_curvature),
                'guidance_lookahead': float(self._last_guidance_lookahead),
                'dynamic_obs_count': float(dynamic_obs_count),
                'dynamic_obs_min_dist': float(dynamic_obs_min_dist),
                'dynamic_obs_min_ttc': float(dynamic_obs_min_ttc),
                'dynamic_obs_ttc_risk': float(dynamic_obs_ttc_risk),
                'subgoal_mode': str(self._last_subgoal_mode),
                'shield_active': 1.0 if self._last_shield_info.get('shield_active', False) else 0.0,
                'swept_clearance': float(self._last_shield_info.get('swept_clearance', self.scan_max_range)),
                'swept_filter_active': 1.0 if self._last_shield_info.get('swept_filter_active', False) else 0.0,
                'motion_safety_margin': float(self._last_shield_info.get('motion_safety_margin', self.scan_max_range)),
                'motion_safety_source': str(self._last_shield_info.get('motion_safety_source', 'none')),
                'motion_filter_active': 1.0 if self._last_shield_info.get('motion_filter_active', False) else 0.0,
                'best_gap_angle': float(self._last_gap_metrics.get('best_gap_angle', 0.0)),
                'best_gap_width': float(self._last_gap_metrics.get('best_gap_width', 0.0)),
            })

            return obs, reward, done, truncated, info

    # ===================================================================
    # step + 仿真时钟同步
    # ===================================================================
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
    # ===================================================================
    # 世界几何 / 障碍物 / 机器人位姿发布
    # ===================================================================
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
            print(f"  _parse_world_bounds({world_path}) 解析失败: {_e}")
            return None

    _MAP_SAFE_MARGIN = {
        1: 7,
        2: 7,
        3: 6,
        4: 5,
        5: 5,
        6: 5,
        8: 8,  # circle_swap_arena
        9: 8,  # warehouse_dynamic (same as circle_swap_arena)
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
        8: [
            ((2.0, 0.0), (-2.0, 0.0)),
            ((1.4142, 1.4142), (-1.4142, -1.4142)),
            ((0.0, 2.0), (0.0, -2.0)),
            ((-1.4142, 1.4142), (1.4142, -1.4142)),
            ((-2.0, 0.0), (2.0, 0.0)),
            ((-1.4142, -1.4142), (1.4142, 1.4142)),
            ((0.0, -2.0), (0.0, 2.0)),
            ((1.4142, -1.4142), (-1.4142, 1.4142)),
        ],
        9: [  # Map 9使用相同的Circle Swap配置
            ((2.0, 0.0), (-2.0, 0.0)),
            ((1.4142, 1.4142), (-1.4142, -1.4142)),
            ((0.0, 2.0), (0.0, -2.0)),
            ((-1.4142, 1.4142), (1.4142, -1.4142)),
            ((-2.0, 0.0), (2.0, 0.0)),
            ((-1.4142, -1.4142), (1.4142, 1.4142)),
            ((0.0, -2.0), (0.0, 2.0)),
            ((1.4142, -1.4142), (-1.4142, 1.4142)),
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
        8: [
            ((2.0, 0.0), (-2.0, 0.0)),
            ((-2.0, 0.0), (2.0, 0.0)),
            ((1.4142, 1.4142), (-1.4142, -1.4142)),
            ((-1.4142, -1.4142), (1.4142, 1.4142)),
            ((0.0, 2.0), (0.0, -2.0)),
            ((0.0, -2.0), (0.0, 2.0)),
            ((-1.4142, 1.4142), (1.4142, -1.4142)),
            ((1.4142, -1.4142), (-1.4142, 1.4142)),
        ],
        9: [  # Map 9使用相同的Circle Swap配置
            ((2.0, 0.0), (-2.0, 0.0)),
            ((-2.0, 0.0), (2.0, 0.0)),
            ((1.4142, 1.4142), (-1.4142, -1.4142)),
            ((-1.4142, -1.4142), (1.4142, 1.4142)),
            ((0.0, 2.0), (0.0, -2.0)),
            ((0.0, -2.0), (0.0, 2.0)),
            ((-1.4142, 1.4142), (1.4142, -1.4142)),
            ((1.4142, -1.4142), (-1.4142, 1.4142)),
        ],
    }

    _DYN_OBS_SPAWNS = {
        1: [(0.4, -0.7), (0.4, -1.4), (0.4, -2.2), (0.4, -2.9),
            (0.4, -3.6), (0.4, -4.3), (0.4, -5.0), (0.4, -5.8)],
        2: [(0.3, -0.9), (0.3, -2.4), (0.3, -3.9), (0.3, -5.7),
            (3.3, -0.9), (5.4, -1.2), (3.3, -4.2), (5.4, -5.7)],
        3: [(-4.5, -4.0), (-4.5, 4.0), (-2.0, -4.5), (-2.0, 4.5),
            (2.0, -4.5), (2.0, 4.5), (4.5, -4.0), (4.5, 4.0)],
        4: [(-3.0, 0.0), (3.0, 0.0), (0.0, -3.0), (0.0, 3.0),
            (-1.7, 0.0), (1.7, 0.0), (0.0, -1.7), (0.0, 1.7)],
        5: [(-5.1, -3.6), (-5.1, 2.4), (-2.7, -3.6), (-0.3, -3.0),
            (-0.3, 2.4), (2.1, -3.6), (4.5, -3.6), (4.5, 2.4)],
        # Map 8 (circle_swap_arena): 动态障碍物初始spawn在场地边缘
        # 由obstacle_mover控制随机游走，这里标记典型活动区域
        8: [
            # 初始位置（场地边缘）
            (3.5, 3.5), (-3.5, 3.5), (3.5, -3.5), (-3.5, -3.5),
            (3.5, 0.0), (-3.5, 0.0), (0.0, 3.5), (0.0, -3.5),
        ],
        # Map 9 (warehouse_dynamic): must match obstacle_mover.py.
        9: [
            (0.0, 3.0), (0.0, -3.0), (3.0, 0.0), (-3.0, 0.0),
            (2.5, 2.0), (-2.5, -2.0), (2.5, -2.0), (-2.5, 2.0),
        ],
    }

    def _active_dyn_obs_spawns(self):
        """Return only the dyn_obs slots enabled for this environment."""
        fixed_positions = getattr(
            getattr(self, 'parent_env', None),
            '_fixed_dynamic_initial_positions',
            None,
        )
        if fixed_positions is not None:
            return list(fixed_positions)
        all_spawns = self._DYN_OBS_SPAWNS.get(self.map_number, [])
        return all_spawns[:self.num_dynamic_obstacles]

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

    def _delete_spawned_obstacles(self):
        """删除上一轮spawn的静态障碍物（棕色方块）。

        快速异步删除，不等待全部完成：
        - 发起删除请求后立即返回（异步）
        - 在spawn时如果遇到"already exists"再单独处理残留
        - 大幅减少reset耗时（从1.0s降到0.1s）
        """
        # 清空障碍物记录列表
        self.spawned_static_obstacles = []
        if hasattr(self, 'parent_env'):
            self.parent_env.spawned_static_obstacles = []

        if not self.delete_entity_client.wait_for_service(timeout_sec=0.5):
            print(f"  Robot {self.robot_id}: /delete_entity 服务不可用，跳过清理", flush=True)
            return

        # 只删除实际使用的障碍物名称（减少无效RPC调用）
        for i in range(self.num_static_obstacles):
            req = DeleteEntity.Request()
            req.name = f'static_box_{i}'
            self.delete_entity_client.call_async(req)  # 异步，不等待

        # 短暂等待删除请求进入队列（不等待完成）
        self._wait_for_sim_time(0.1)
        print(f"🗑️  Robot {self.robot_id}: 已发起清理 static_box_0~{self.num_static_obstacles-1}（异步）", flush=True)

    def randomize_obstacles(self, robot_positions: list):
        """
        每次 reset 时随机重置静态障碍物（棕色方块）

        Args:
            robot_positions: 所有机器人的spawn位置 [(x, y), ...]
        """
        # 先删除旧的随机障碍物
        self._delete_spawned_obstacles()
        # 再spawn新的随机障碍物
        self._spawn_random_obstacles(other_robot_positions=robot_positions)

    # ===================================================================
    # Guidance observation: A* 粗方向 + goal 并行信号
    # ===================================================================

    def _get_clearance_aware_lateral_limit(self, sectors: Dict[str, float]) -> float:
        projection = self.current_projection
        if projection is None:
            projection = (self.current_pose['x'], self.current_pose['y'])

        path_clearance = self.scan_max_range
        if self.planner is not None:
            try:
                path_clearance = float(self.planner.get_clearance_m(
                    float(projection[0]),
                    float(projection[1]),
                ))
            except Exception:
                path_clearance = self.scan_max_range

        allowed = float(np.clip(
            path_clearance - ROBOT_RADIUS - self.path_clearance_margin,
            0.08,
            0.35,
        ))
        width = float(sectors.get('corridor_width', self.scan_max_range * 2.0))
        if width < self.corridor_narrow_width:
            corridor_limit = max(
                0.08,
                0.5 * width - ROBOT_RADIUS - self.path_clearance_margin,
            )
            allowed = min(allowed, corridor_limit)

        curvature = abs(float(getattr(self, '_last_path_curvature', 0.0)))
        curvature_gate = float(np.clip((curvature - 0.18) / 0.70, 0.0, 1.0))
        allowed = min(allowed, 0.35 - 0.17 * curvature_gate)

        self._last_path_clearance = float(path_clearance)
        self._last_allowed_lateral_error = float(np.clip(allowed, 0.08, 0.35))
        return self._last_allowed_lateral_error

    def _get_guidance_obs(self) -> np.ndarray:
        """6-dim guidance: [guidance_sin, guidance_cos, guidance_dist_norm, goal_sin, goal_cos, goal_dist_norm]"""
        px = self.current_pose['x']
        py = self.current_pose['y']
        yaw = self.current_pose['yaw']

        # Keep the observation shape checkpoint-compatible while blending a
        # near path target into the far guidance around walls and corners.
        path_points = self.global_waypoints if getattr(self, 'global_waypoints', None) else None
        if path_points and len(path_points) >= 2:
            near_lookahead = float(np.clip(
                self.lookahead_dist if self.lookahead_dist > 0.0 else 0.70,
                0.45,
                0.85,
            ))
            near_info = PathTrackingUtils.get_rolling_subgoal(
                (px, py), path_points, near_lookahead
            )
            far_info = PathTrackingUtils.get_rolling_subgoal(
                (px, py), path_points, self.guidance_lookahead_m
            )
            ngx, ngy = near_info['subgoal']
            fgx, fgy = far_info['subgoal']

            near_angle = math.atan2(ngy - py, ngx - px)
            far_angle = math.atan2(fgy - py, fgx - px)
            curvature = abs(self._wrap_angle(far_angle - near_angle))
            curvature_gate = float(np.clip((curvature - 0.12) / 0.75, 0.0, 1.0))

            sectors = self._scan_sector_metrics()
            width = float(sectors.get('corridor_width', self.scan_max_range * 2.0))
            width_span = max(self.corridor_narrow_width - self.corridor_min_width, 1e-6)
            narrow_gate = float(np.clip(
                (self.corridor_narrow_width - width) / width_span,
                0.0,
                1.0,
            ))
            side_min = min(
                float(sectors.get('left_min', self.scan_max_range)),
                float(sectors.get('right_min', self.scan_max_range)),
            )
            wall_gate = float(np.clip((0.45 - side_min) / 0.25, 0.0, 1.0))
            near_weight = float(np.clip(max(
                0.10,
                0.85 * curvature_gate,
                0.75 * narrow_gate,
                0.60 * wall_gate,
            ), 0.10, 0.90))

            near_vec = np.array([ngx - px, ngy - py], dtype=np.float32)
            far_vec = np.array([fgx - px, fgy - py], dtype=np.float32)
            near_dist = float(np.linalg.norm(near_vec))
            far_dist = float(np.linalg.norm(far_vec))
            if near_dist > 1e-6:
                near_vec /= near_dist
            if far_dist > 1e-6:
                far_vec /= far_dist
            blended = near_weight * near_vec + (1.0 - near_weight) * far_vec
            blended_norm = float(np.linalg.norm(blended))
            if blended_norm < 1e-6:
                blended = near_vec if near_dist > 1e-6 else far_vec
                blended_norm = max(float(np.linalg.norm(blended)), 1e-6)
            blended /= blended_norm
            g_dist = near_weight * near_dist + (1.0 - near_weight) * far_dist
            gx = px + float(blended[0]) * g_dist
            gy = py + float(blended[1]) * g_dist

            self._last_guidance_near_weight = near_weight
            self._last_path_curvature = float(curvature)
            self._last_guidance_lookahead = float(
                near_weight * near_lookahead + (1.0 - near_weight) * self.guidance_lookahead_m
            )
        else:
            gx, gy = self.goal_pos
            self._last_guidance_near_weight = 0.0
            self._last_path_curvature = 0.0
            self._last_guidance_lookahead = self.guidance_lookahead_m

        # guidance angle in body frame
        gdx, gdy = gx - px, gy - py
        g_world_angle = math.atan2(gdy, gdx)
        g_rel = (g_world_angle - yaw + math.pi) % (2 * math.pi) - math.pi
        g_dist = math.hypot(gdx, gdy)
        g_dist_norm = float(np.clip(g_dist / self.obs_target_dist_clip, 0.0, 1.0))

        # goal angle in body frame
        goal_dx, goal_dy = self.goal_pos[0] - px, self.goal_pos[1] - py
        goal_world_angle = math.atan2(goal_dy, goal_dx)
        goal_rel = (goal_world_angle - yaw + math.pi) % (2 * math.pi) - math.pi
        goal_dist = math.hypot(goal_dx, goal_dy)
        goal_dist_norm = float(np.clip(goal_dist / self.obs_target_dist_clip, 0.0, 1.0))

        return np.array([
            math.sin(g_rel), math.cos(g_rel), g_dist_norm,
            math.sin(goal_rel), math.cos(goal_rel), goal_dist_norm,
        ], dtype=np.float32)

    # ===================================================================
    # Local map: lidar → ego-centric 32×32 occupancy grid (2 temporal frames)
    # ===================================================================

    def _build_local_map_obs(self, ranges: np.ndarray) -> np.ndarray:
        """Build ego-centric 32x32 binary occupancy grid from lidar, stack 4 frames, return flat 4096-dim."""
        grid_size = 32
        grid_range = 4.0
        cell_size = grid_range / grid_size
        half = grid_size // 2
        max_r = grid_range / 2.0

        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        n_rays = len(ranges)
        angles = self._scan_angles(n_rays)

        valid_mask = (ranges > self.scan_valid_min) & (ranges <= max_r)
        xs = ranges[valid_mask] * np.cos(angles[valid_mask])
        ys = ranges[valid_mask] * np.sin(angles[valid_mask])
        gxs = (xs / cell_size + half).astype(np.int32)
        gys = (ys / cell_size + half).astype(np.int32)
        in_bounds = (gxs >= 0) & (gxs < grid_size) & (gys >= 0) & (gys < grid_size)
        grid[gxs[in_bounds], gys[in_bounds]] = 1.0

        self._local_map_history.append(grid)
        frames = list(self._local_map_history)
        while len(frames) < 4:
            frames.insert(0, frames[0].copy())
        stacked = np.stack(frames[-4:], axis=0)
        return stacked.reshape(-1)

    # ===================================================================
    # 观测空间 _get_obs (SIMPLIFIED): [guidance, vel, safety(2), neighbor_raw_token, obstacle_token]
    # local_map (raw grid) appended by GNNMARLEnv._build_enhanced_observation
    # ===================================================================
    def _get_obs(self, target_override=None):
        ranges = np.array(
            self.latest_scan.ranges if self.latest_scan else [self.scan_max_range] * 360,
            dtype=np.float32,
        )
        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        sector_dists = self._compute_front_sector_min_dists(ranges)
        # Keep scan_history/front_min_history populated for reward/predictive computation
        scan_obs = self._extract_filtered_scan_features(ranges)
        self._scan_history.append(scan_obs.copy())
        self._front_sector_dist_history.append(sector_dists.copy())

        # Guidance-based target (6-dim: A* direction + goal direction)
        target_features = self._get_guidance_obs()

        sectors = self._scan_sector_metrics()
        min_dist = float(sectors['min_dist'])
        front_min = float(sectors['front_min'])
        # Update predictive metrics for reward (not in obs but reward uses _last_predictive_metrics)
        self._get_predictive_obs_features(front_min)

        neighbor_prediction_features = self._get_neighbor_prediction_features()
        obstacle_motion_features = self._get_obstacle_motion_features(sector_dists)
        self._last_motion_features = obstacle_motion_features  # 保存用于奖励计算

        obs = np.concatenate([
            target_features,
            [self.current_vel_x, self.current_vel_w],
            [front_min, min_dist],
            neighbor_prediction_features,
            obstacle_motion_features,
            self.agent_id_embedding,
        ]).astype(np.float32)

        if obs.shape != self.observation_space.shape:
            raise ValueError(
                f"[IndependentRobotEnv] _get_obs shape mismatch: got={obs.shape}, "
                f"expected={self.observation_space.shape}, scan_dim={self.scan_dim}, "
                f"scan_history_len={self.scan_history_len}, obstacle_top_k={self.obstacle_top_k}"
            )
        return obs

    def _publish_vel(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.vel_pub.publish(msg)

    def _set_robot_pose(self, x, y, yaw):
        if not self.set_state_client.wait_for_service(timeout_sec=1.0):
            return False

        req = SetEntityState.Request()
        req.state.name = self.gazebo_model_name
        req.state.reference_frame = 'world'

        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.1
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0

        future = self.set_state_client.call_async(req)
        deadline = time.time() + 2.0
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
        if not future.done():
            self.node.get_logger().warning(
                f'Robot {self.robot_id}: /set_entity_state timed out for '
                f'{self.gazebo_model_name} at ({float(x):.2f}, {float(y):.2f})'
            )
            return False
        try:
            resp = future.result()
        except Exception as exc:
            self.node.get_logger().warning(
                f'Robot {self.robot_id}: /set_entity_state failed for '
                f'{self.gazebo_model_name}: {exc}'
            )
            return False
        return bool(resp is not None and getattr(resp, 'success', True))

    def _wait_for_pose_sync(self, x, y, yaw, timeout_sec=2.0, tolerance=0.08):
        """Wait until odom reflects a just-applied Gazebo pose reset."""
        deadline = time.time() + float(timeout_sec)
        target_x = float(x)
        target_y = float(y)
        synced = False

        while rclpy.ok() and time.time() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.02)
            dx = float(self.current_pose.get('x', 0.0)) - target_x
            dy = float(self.current_pose.get('y', 0.0)) - target_y
            if math.hypot(dx, dy) <= float(tolerance):
                synced = True
                break

        if not synced:
            # Re-apply once; Gazebo can occasionally accept SetEntityState before
            # the model plugin has pushed its next odom sample.
            self._set_robot_pose(target_x, target_y, yaw)
            retry_deadline = time.time() + min(1.0, float(timeout_sec))
            while rclpy.ok() and time.time() < retry_deadline:
                rclpy.spin_once(self.node, timeout_sec=0.02)
                dx = float(self.current_pose.get('x', 0.0)) - target_x
                dy = float(self.current_pose.get('y', 0.0)) - target_y
                if math.hypot(dx, dy) <= float(tolerance):
                    synced = True
                    break

        self.current_vel_x = 0.0
        self.current_vel_w = 0.0
        self._publish_vel(0.0, 0.0)
        return synced

    def _spawn_random_obstacles(self, other_robot_positions=None):
        """
        每次reset时随机spawn静态障碍物（棕色方块）
        动态障碍物（红色圆柱）由obstacle_mover.py管理

        Args:
            other_robot_positions: list of (x, y) tuples for other robots' spawn positions

        注意：此函数只由robot_id==0调用，不要在此清空spawned_static_obstacles列表
        """
        if not self.random_obstacles:
            return

        if not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            return

        # 地图安全spawn区域
        if self.map_number == 9:  # warehouse_dynamic
            spawn_bounds = {'x_min': -3.0, 'x_max': 3.0, 'y_min': -3.0, 'y_max': 3.0}
        elif self.map_number == 8:  # circle_swap_arena
            spawn_bounds = {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5}
        else:
            # 其他地图暂不支持随机障碍物
            return

        # 障碍物类型定义（棕色方块 - 模仿仓库货物）
        box_types = [
            {'size': (0.4, 0.4, 0.5), 'name': 'small_box'},   # 小箱子
            {'size': (0.5, 0.5, 0.6), 'name': 'medium_box'},  # 中箱子
            {'size': (0.6, 0.6, 0.8), 'name': 'large_box'},   # 大箱子
        ]

        # 碰撞检测参数
        MIN_OBSTACLE_SEP = 0.7  # 障碍物之间最小间隔（减小到0.7，在6x6区域能放下8个）
        MIN_ROBOT_SEP = 1.8     # 障碍物与机器人spawn点最小间隔（从1.0增加到1.8）
                                # 原因: A*规划时block_radius=1.0m会膨胀障碍物
                                # 需要确保 MIN_ROBOT_SEP > block_radius + 安全边距
                                # 1.8m = 1.0m(block_radius) + 0.5m(机器人footprint) + 0.3m(安全边距)

        occupied_positions = []  # 存储 (x, y, radius) - radius是该位置的"占用半径"

        # 添加所有已知机器人位置到占用列表
        # 机器人本身很小(~0.2m)，但需要更大的保护区避免障碍物spawn在机器人旁边
        ROBOT_FOOTPRINT = 0.2  # 机器人自身半径
        if other_robot_positions:
            for x, y in other_robot_positions:
                occupied_positions.append((x, y, ROBOT_FOOTPRINT))

        def get_box_footprint(size_tuple):
            """计算方块的碰撞半径（取对角线的一半）"""
            sx, sy, _ = size_tuple
            return math.sqrt(sx**2 + sy**2) / 2

        def is_position_valid(x, y, footprint):
            """
            检查位置是否有效（不与已有障碍物/机器人重叠）

            Args:
                x, y: 待检查位置
                footprint: 待spawn障碍物的碰撞半径

            Returns:
                bool: True=位置有效，False=有冲突
            """
            for ox, oy, o_radius in occupied_positions:
                dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                # 检查是机器人还是障碍物
                if o_radius <= ROBOT_FOOTPRINT:
                    # 与机器人的距离检查：使用较大的MIN_ROBOT_SEP
                    min_required_dist = footprint + o_radius + MIN_ROBOT_SEP
                else:
                    # 与其他障碍物的距离检查：使用MIN_OBSTACLE_SEP
                    min_required_dist = footprint + o_radius + MIN_OBSTACLE_SEP

                if dist < min_required_dist:
                    return False
            return True

        def _spawn_and_confirm(req):
            """同步发起 spawn 并等待响应，返回 (success, message)。"""
            if not self.spawn_entity_client.wait_for_service(timeout_sec=0.5):
                return False, 'spawn service unavailable'
            future = self.spawn_entity_client.call_async(req)
            deadline = time.time() + 2.0
            while rclpy.ok() and not future.done() and time.time() < deadline:
                rclpy.spin_once(self.node, timeout_sec=0.01)
            if not future.done():
                return False, 'spawn timeout'
            resp = future.result()
            if resp is None:
                return False, 'spawn no response'
            return bool(resp.success), getattr(resp, 'status_message', '')

        def spawn_box(name, box_config):
            """尝试spawn一个棕色方块（同步确认成功后才记录，避免旧箱残留时静默失败）"""
            size = box_config['size']
            footprint = get_box_footprint(size)

            for attempt in range(100):
                x = random.uniform(spawn_bounds['x_min'], spawn_bounds['x_max'])
                y = random.uniform(spawn_bounds['y_min'], spawn_bounds['y_max'])

                if is_position_valid(x, y, footprint):
                    # 生成棕色方块SDF模型
                    sdf_xml = self._generate_box_sdf(name=name, size=size)

                    # 随机朝向
                    yaw = random.uniform(0, 2 * math.pi)

                    # SpawnEntity请求
                    req = SpawnEntity.Request()
                    req.name = name
                    req.xml = sdf_xml
                    req.robot_namespace = ''
                    req.initial_pose.position.x = float(x)
                    req.initial_pose.position.y = float(y)
                    req.initial_pose.position.z = size[2] / 2  # 方块高度的一半
                    req.initial_pose.orientation.z = math.sin(yaw / 2)
                    req.initial_pose.orientation.w = math.cos(yaw / 2)

                    # 同步 spawn 并确认成功；失败（如同名实体仍存在）则不记录，换个位置重试
                    ok, msg = _spawn_and_confirm(req)
                    if not ok:
                        # 如果是"already exists"错误，说明删除还没生效
                        # 立即强制删除该实体，短暂等待后重试
                        if "already exists" in msg.lower():
                            del_req = DeleteEntity.Request()
                            del_req.name = name
                            del_future = self.delete_entity_client.call_async(del_req)
                            deadline = time.time() + 1.0
                            while rclpy.ok() and not del_future.done() and time.time() < deadline:
                                rclpy.spin_once(self.node, timeout_sec=0.01)
                            # 等待物理引擎移除（减少到0.2s，配合异步删除）
                            self._wait_for_sim_time(0.2)
                            # 重新尝试spawn
                            ok, msg = _spawn_and_confirm(req)
                            if not ok:
                                print(f"    {name} 强制删除后仍失败: {msg} → 跳过", flush=True)
                                return False
                        else:
                            # 其他错误（如位置冲突）继续重试
                            continue

                    # 确认成功后才记录占用位置 + spawned_static_obstacles（用于A*规划避障）
                    occupied_positions.append((x, y, footprint))
                    obs_info = (float(x), float(y), float(footprint))
                    self.spawned_static_obstacles.append(obs_info)
                    if hasattr(self, 'parent_env'):
                        self.parent_env.spawned_static_obstacles.append(obs_info)

                    print(f"   {name} @ ({x:.2f}, {y:.2f}) size={box_config['name']} footprint={footprint:.2f}m")
                    return True

            # 尝试100次仍失败，警告但继续
            print(f"  Robot {self.robot_id}: 无法为 {name} 找到有效位置（已尝试100次）")
            print(f"    当前已占用位置数: {len(occupied_positions)}")
            print(f"    Spawn区域: x∈[{spawn_bounds['x_min']}, {spawn_bounds['x_max']}], "
                  f"y∈[{spawn_bounds['y_min']}, {spawn_bounds['y_max']}]")
            print(f"    碰撞参数: MIN_OBSTACLE_SEP={MIN_OBSTACLE_SEP}, MIN_ROBOT_SEP={MIN_ROBOT_SEP}")
            return False

        # Spawn静态障碍物（棕色方块）
        success_count = 0
        for i in range(self.num_static_obstacles):
            # 随机选择方块大小
            box_config = random.choice(box_types)
            if spawn_box(f'static_box_{i}', box_config):
                success_count += 1

        print(f"🟫 Robot {self.robot_id}: 成功spawn {success_count}/{self.num_static_obstacles} 个棕色方块")

        # 等待所有spawn请求被Gazebo处理（异步spawn需要更长等待时间）
        self._wait_for_sim_time(0.3)

        # Dynamic obstacles are owned by obstacle_mover.py. Unused slots stay
        # below ground; deleting them at runtime can crash Gazebo's physics
        # thread while GUI/spawn requests are still being processed.

        # 等待物理引擎稳定（避免spawn后的碰撞检测误判）
        self._wait_for_sim_time(0.15)

    def _generate_cylinder_sdf(self, name, radius, height, is_static=True, color=(0.6, 0.6, 0.6, 1.0)):
        """
        生成圆柱体障碍物的SDF模型

        Args:
            name: 模型名称
            radius: 圆柱半径
            height: 圆柱高度
            is_static: 是否为静态障碍物
            color: RGBA颜色元组

        Returns:
            str: SDF XML字符串
        """
        static_tag = 'true' if is_static else 'false'
        mass = 1.0 if is_static else 5.0
        inertial_block = '' if is_static else f'''
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>0.1</ixx><iyy>0.1</iyy><izz>0.1</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>'''

        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>{static_tag}</static>
    <link name="link">{inertial_block}
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        return sdf

    def _generate_box_sdf(self, name, size):
        """
        生成方块障碍物的SDF模型（棕色，模仿仓库货物）

        Args:
            name: 模型名称
            size: (sx, sy, sz) 方块尺寸元组

        Returns:
            str: SDF XML字符串
        """
        sx, sy, sz = size

        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>{sx} {sy} {sz}</size>
          </box>
        </geometry>
        <material>
          <ambient>0.7 0.5 0.3 1</ambient>
          <diffuse>0.7 0.5 0.3 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        return sdf


    def close(self):
        self.node.destroy_node()
