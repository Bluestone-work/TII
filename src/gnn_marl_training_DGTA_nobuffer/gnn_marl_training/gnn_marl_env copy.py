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
from gazebo_msgs.srv import SetEntityState, GetEntityState
from std_msgs.msg import Float32MultiArray
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

    # 文件日志级别：默认 INFO，滤掉每步 [graph]/[comm] 的 DEBUG 刷屏（曾累积出 56GB）。
    # 需要逐步排查时设 ENV_VERBOSE=1 落 DEBUG。
    _env_verbose = str(os.environ.get("ENV_VERBOSE", "0")).strip().lower() in ("1", "true", "yes", "on")
    _file_level = logging.DEBUG if _env_verbose else logging.INFO

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(_file_level)
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
# 奖励函数：6 项正交结构
# ---------------------------------------------------------------------
#   r_progress   : 朝目标前进 (path_delta + heading 塑形)   [-0.3, +0.3]
#   r_static     : 障碍避碰 (斥力 + 减速 + 前方预测风险)     [-1.0, 0]
#   r_social     : 邻居避碰 (机器人间 TTC)                  [-1.0, 0]
#   r_collision  : 碰撞终端                                 {-collision_penalty, 0}
#   r_goal       : 到达终端                                 {0, +goal_reward}
#   r_time       : 时间压力                                 -time_penalty
# 每项语义正交、互不重叠，便于做消融分析。
# =====================================================================

# 2026-06-30: 增强障碍物避碰——针对"容易向障碍物冲撞"的问题(用户反馈)
#   核心病灶:progress 信号(+0.30/步持续)盖过了障碍避碰惩罚(只在近距离触发封顶-1.0)
#   修复方案:(1) r_dynamic_obs权重 1.0→2.5+clip -1.0→-2.0,增强动态障碍专项惩罚
#            (2) 斥力场起作用距离 D0 0.5→0.75m,提前预警避让
#            (3) near_miss_penalty 起作用距离 0.25→0.30,增强擦肩惩罚
RWD_PROGRESS_CLIP        = 0.30   # r_progress 单步截断
RWD_STATIC_CLIP          = 1.00   # r_static 单步截断(保留1.0,由scale控制)
RWD_SOCIAL_CLIP          = 1.00   # r_social 单步截断
RWD_DYNAMIC_OBS_CLIP     = 2.00   # r_dynamic_obs 单步截断(新增:从 -1.0 提到 -2.0)
RWD_HEADING_COEF         = 0.10   # heading shaping 系数
RWD_HEADING_MIN_FWD_VEL  = 0.02   # 前向速度门控:必须真前进才给 heading
RWD_STATIC_D0            = 0.75   # 斥力场起作用距离(0.50→0.75m,提前预警)
RWD_STATIC_D_MIN         = 0.15   # 距离下界（防爆炸）
RWD_STATIC_SPEED_RISK_D0 = 0.70   # 前方障碍减速门控
RWD_NEAR_MISS_DIST       = 0.30   # near_miss_penalty 起作用距离(0.25→0.30)
RWD_SOCIAL_NEAR_DIST     = 1.5    # 邻居"近距离"判定（米）
RWD_SOCIAL_APPROACH_TH   = 0.05   # 接近速度门（m/s）
RWD_GOAL_REACH_RADIUS    = 0.35   # 到达目标半径(默认值;实际由 env_config goal_reach_radius 课程式覆盖)

# =====================================================================
# 运动预测与安全膨胀参数 (Sim2Real Friendly)
# =====================================================================
ROBOT_RADIUS             = 0.105  # TurtleBot3 半径 (m)
SAFETY_MARGIN            = 0.15   # 安全裕度 (m) - 考虑定位误差和传感器噪声
INFLATION_RADIUS         = ROBOT_RADIUS + SAFETY_MARGIN  # 总膨胀半径 = 0.255m
PREDICTION_DT            = 0.4    # 多步预测时间间隔 (s) - 匹配真实传感器频率
PREDICTION_STEPS         = 5      # 预测步数 (共2秒) - 平衡精度与计算开销
PREDICTION_WINDOW_FAST   = 2.0    # 快速运动物体预测窗口 (s)
PREDICTION_WINDOW_MED    = 1.5    # 中速运动物体预测窗口 (s)
PREDICTION_WINDOW_SLOW   = 1.0    # 慢速运动物体预测窗口 (s)
SPEED_THRESHOLD_FAST     = 0.5    # 快速运动阈值 (m/s)
SPEED_THRESHOLD_MED      = 0.3    # 中速运动阈值 (m/s)


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
        # 2026-06-29 修复: 早终止阈值改为按团队规模缩放（绝对值 2 对 8-agent 场景过严，导致 episode_len 仅占 max 的 9%）
        # 默认按 30% 团队规模 + 至少 3，避免少数 agent 早期碰撞就把整 episode 掐死
        _default_failed_cutoff = max(3, int(math.ceil(0.30 * self._num_agents)))
        self.max_failed_agents_before_cutoff = int(max(0, config.get('max_failed_agents_before_cutoff', _default_failed_cutoff)))
        # 高冲突训练模式：通过更激进的起终点路由采样制造会车/交叉冲突。
        self.high_conflict_mode = str(config.get('high_conflict_mode', 'off')).strip().lower()
        if self.high_conflict_mode not in ('off', 'mixed', 'aggressive'):
            self.high_conflict_mode = 'off'
        self.high_conflict_prob = float(np.clip(float(config.get('high_conflict_prob', 0.75)), 0.0, 1.0))
        # Circle swap arena (map 8): force aggressive conflict mode for antipodal assignment
        if self.map_number == 8:
            self.high_conflict_mode = 'aggressive'

        # ===== 密度自适应奖励调整 =====
        # 随着机器人数量增加，避障压力增大，需要动态调整惩罚权重
        # 以 4 车为基准，使用平方根缩放避免过度惩罚
        density_factor = math.sqrt(self._num_agents / 4.0)
        # Social reward scale: 机器人越多，邻居避碰越重要
        base_social_scale = float(config.get('social_scale', 1.0))
        adjusted_social_scale = base_social_scale * density_factor
        # Collision penalty: 高密度下碰撞风险增加，适度提高惩罚
        base_collision_penalty = float(config.get('collision_penalty', 20.0))
        adjusted_collision_penalty = base_collision_penalty * (1.0 + 0.2 * (density_factor - 1.0))

        # 创建独立环境实例
        self.agents = {}
        env_signature = inspect.signature(IndependentRobotEnv.__init__)
        for i in range(self._num_agents):
            candidate_kwargs = {
                'robot_id': i,
                'map_number': config.get('map_number', 3),
                'max_episode_steps': config.get('max_episode_steps', 1000),
                'collision_ends_episode': bool(config.get('collision_ends_episode', False)),
                'collision_hard_dist': float(config.get('collision_hard_dist', 0.20)),
                'collision_persist_dist': float(config.get('collision_persist_dist', 0.26)),
                'collision_persist_steps': int(config.get('collision_persist_steps', 2)),
                'collision_grace_steps': int(config.get('collision_grace_steps', 8)),
                'goal_reach_radius': float(config.get('goal_reach_radius', 0.35)),
                'near_wall_penalty_dist': float(config.get('near_wall_penalty_dist', 0.30)),
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
                'replan_distance_trigger': float(config.get('replan_distance_trigger', 2.5)),  # 新增:每移动X米触发重规划
                'dynamic_replan_neighbor_dist': float(config.get('dynamic_replan_neighbor_dist', 1.8)),
                'dynamic_replan_ttc': float(config.get('dynamic_replan_ttc', 2.6)),
                'dynamic_replan_block_radius': float(config.get('dynamic_replan_block_radius', 0.55)),
                'obs_target_dist_clip': float(config.get('obs_target_dist_clip', 6.0)),
                'obs_target_filter_alpha': float(config.get('obs_target_filter_alpha', 0.35)),
                'obs_target_max_step': float(config.get('obs_target_max_step', 0.45)),
                'guidance_lookahead_m': float(config.get('guidance_lookahead_m', 3.0)),
                'target_obs_dim': int(config.get('target_obs_dim', 6)),
                'progress_reward_scale': float(config.get('progress_reward_scale', 0.0)),
                'path_progress_reward_scale': float(config.get('path_progress_reward_scale', 0.0)),
                'goal_progress_reward_scale': float(config.get('goal_progress_reward_scale', 4.0)),
                'goal_reward': float(config.get('goal_reward', 20.0)),
                'collision_penalty': adjusted_collision_penalty,
                'time_penalty': float(config.get('time_penalty', 0.01)),
                'lateral_penalty_scale': float(config.get('lateral_penalty_scale', 0.0)),
                'heading_align_reward_scale': float(config.get('heading_align_reward_scale', 0.0)),
                'narrow_forward_penalty_scale': float(config.get('narrow_forward_penalty_scale', 0.0)),
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
                'use_gazebo_collision': bool(config.get('use_gazebo_collision', True)),
                'lidar_collision_fallback': bool(config.get('lidar_collision_fallback', True)),
                'obstacle_filter_range': float(config.get('obstacle_filter_range', 2.0)),
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
                'progress_scale': float(config.get('progress_scale', 1.5)),
                'static_scale': float(config.get('static_scale', 0.8)),
                'social_scale': adjusted_social_scale,
                'action_mode': config.get('action_mode', 'discrete_primitive'),
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
        _log_dir   = config.get('log_dir', os.path.expanduser('~/ray_results/gnn_marl_logs'))
        _log_file  = os.path.join(_log_dir, f'env_worker{_worker_id}.log')
        self.logger = _LogAdapter(
            _setup_env_logger(_log_file, _worker_id),
            {'worker_id': _worker_id}
        )
        self.log_every_n_steps = int(config.get('log_every_n_steps', 50))

        # ── 终端输出详略：默认安静（无 Episode 结束框 / graph / comm 详细日志）；设 ENV_VERBOSE=1 恢复。
        self.env_verbose_terminal = str(os.environ.get("ENV_VERBOSE", "0")).strip().lower() in ("1", "true", "yes", "on")

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

    # ===================================================================
    # 高冲突路由采样 —— 制造会车/交叉冲突的起终点路线
    # ===================================================================
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

        # Circle swap (map 8): routes are organized as antipodal PAIRS
        # [0,1] are a head-on pair, [2,3] are a pair, etc.
        # For 2/4/6 agents, sample complete pairs to guarantee head-on conflict.
        if master.map_number == 8 and len(route_lib) >= 2:
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
            # 【修复】最多邻居数 = min(其他机器人数量, 5)
            # Fixed 5 neighbor slots regardless of num_agents — ensures model is
            # deployable at any N (unused slots are zero-padded).
            max_neighbors = 5
            # 每个邻居: body-frame rel_pos(2) + rel_vel(2) + dist(1) + sin/cos rel_heading(2) = 7
            neighbor_dim = max_neighbors * 7
        else:
            neighbor_dim = 0
        
        # 可选：局部地图（32x32 ego-centric grid * 2 frames = 2048 flat）
        if self.enable_local_map:
            local_map_dim = 32 * 32 * 2
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

        route_plan = self._build_episode_route_plan()
        if route_plan:
            self.logger.info('[reset] 启用高冲突路线采样: mode=%s prob=%.2f',
                             self.high_conflict_mode, self.high_conflict_prob)

        obs_dict = {}
        info_dict = {}
        
        base_obs_dict = {}
        agent_starts = []  # list of (x, y)
        # 每个 episode 查询一次动态障碍物实时位置，供所有 agent 的 spawn 避让共享，
        # 避免每个 agent 重复查询 Gazebo 服务。障碍物由 obstacle_mover 随机游走驱动。
        master = self.agents.get('agent_0')
        live_obstacles = []
        if master is not None and hasattr(master, 'query_dynamic_obstacle_positions'):
            try:
                live_obstacles = master.query_dynamic_obstacle_positions(
                    getattr(master, 'num_dynamic_obstacles', 0)
                )
            except Exception:
                live_obstacles = []
        if live_obstacles:
            self.logger.info('[reset] 动态障碍物实时位置(%d): %s',
                             len(live_obstacles),
                             ', '.join(f'({x:.2f},{y:.2f})' for x, y in live_obstacles))
        for agent in self.agents.values():
            agent._live_obstacle_positions = live_obstacles
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
                self.logger.info('[step %d] %s 碰撞 (本 episode 第 %d 次, src=%s, min_dist=%.3f, agent_step=%s)',
                                 self.current_step_count, aid, self.episode_collisions[aid],
                                 info.get('collision_source', 'unknown'),
                                 float(info.get('collision_min_dist', -1.0)),
                                 info.get('collision_step', '?'))

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
            self.logger.info(
                '━━━ EPISODE END (%s, step=%d) dones=%d/%d '
                'successes=%s collisions=%s active_remaining=%d failed=%d ━━━',
                reason, self.current_step_count,
                len(self.dones), self._num_agents,
                {aid: self.episode_successes[aid]  for aid in self.agent_ids},
                {aid: self.episode_collisions[aid] for aid in self.agent_ids},
                active_remaining, failed_count,
            )
            print(
                f"\n{'='*60}\n"
                f"🏁 Episode 结束 ({reason})\n"
                f"   步数: {self.current_step_count}/{self.max_steps}\n"
                f"   完成: {len(self.dones)}/{self._num_agents}\n"
                f"   活跃: {active_remaining}  失败: {failed_count}\n"
                f"{'='*60}\n"
            ) if self.env_verbose_terminal else None
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

        # 控制台：只在前 3 步或 debug_comm=True 或 ENV_VERBOSE=1 时打印
        if (getattr(self, 'debug_comm', False) or self.current_step_count <= 3) and self.env_verbose_terminal:
            print('\n'.join(log_lines))

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

        if (getattr(self, 'debug_comm', False) or self.current_step_count <= 3) and self.env_verbose_terminal:
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
                 collision_ends_episode=True,
                 collision_hard_dist=0.05,
                 collision_persist_dist=0.15,
                 collision_persist_steps=3,
                 collision_grace_steps=8,
                 goal_reach_radius=0.35,
                 near_wall_penalty_dist=0.20,
                 waypoint_reach_radius=0.8,
                 waypoint_distance_threshold=1.2,
                 waypoint_min_clearance_m=0.40,
                 use_voronoi_planner=True,
                 voronoi_min_clearance_m=0.35,
                 num_dynamic_obstacles=8, obs_speed=0.3,
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
                 replan_cooldown_steps=25,
                 replan_distance_trigger=2.5,  # 新增:主动重规划间距(米)
                 dynamic_replan_neighbor_dist=1.8,
                 dynamic_replan_ttc=2.6,
                 dynamic_replan_block_radius=0.55,
                 obs_target_dist_clip=6.0,
                 obs_target_filter_alpha=0.35,
                 obs_target_max_step=0.45,
                 guidance_lookahead_m=3.0,
                 target_obs_dim=6,
                 progress_reward_scale=0.0,
                 path_progress_reward_scale=0.0,
                 goal_progress_reward_scale=4.0,
                 goal_reward=20.0,
                 collision_penalty=50.0,
                 time_penalty=0.01,
                 lateral_penalty_scale=0.0,
                 heading_align_reward_scale=0.0,
                 narrow_forward_penalty_scale=0.0,
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
                 use_gazebo_collision=True,
                 lidar_collision_fallback=False,
                 obstacle_filter_range=2.0,
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
                 progress_scale=1.5,
                 static_scale=0.8,
                 social_scale=0.4,
                 action_mode='discrete_primitive'):
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
        self.current_step = 0
        self.collision_hard_dist = float(collision_hard_dist)
        self.collision_persist_dist = float(collision_persist_dist)
        self.collision_persist_steps = int(max(1, collision_persist_steps))
        self.near_wall_penalty_dist = float(near_wall_penalty_dist)
        self.waypoint_reach_radius = float(waypoint_reach_radius)
        # 到达目标判定半径（可课程式配置：早期放宽让 agent 先尝到 goal 甜头，逐 stage 收紧）
        self.goal_reach_radius = float(goal_reach_radius)
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
        # 查询动态障碍物(dyn_obs_*)实时位置，用于 spawn 时避开随机游走的障碍物。
        # 障碍物由独立进程 obstacle_mover.py 随机游走驱动，env 不知道其位置，
        # 若机器人 spawn 在障碍物紧邻处会立即被撞（开局初速 0 来不及反应）。
        self.get_state_client = self.node.create_client(GetEntityState, '/get_entity_state')
        # spawn 时离每个动态障碍物的最小安全距离(m)。障碍物以 0.11m/s 随机游走,
        # grace 8 步(0.8s)内可移动 0.088m。初始距离 0.9m → grace 后最近 ~0.81m,
        # 结合机器人自己移动/转向,足够反应。过大的 clearance(如1.2m)会导致大量
        # forced route 非法回退随机采样,反而降低训练效率。
        self._live_obstacle_positions: list = []
        self._spawn_obstacle_clearance = 0.9

        self.latest_scan = None
        # spawn 后激光帧序号：用于 reset 等待"传送后的新激光帧"，并丢弃前若干瞬态帧。
        self._scan_seq = 0
        # 开局碰撞宽限步数：teleport 后头几步 lidar 读数可能为旧位姿残帧/瞬态，
        # 期间禁用 lidar_fallback 硬碰撞判定，避免一出生就被误判为碰撞而掐死 episode。
        self.collision_grace_steps = int(collision_grace_steps)
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
        self.scan_valid_min = 0.15
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
        self._local_map_history: deque = deque(maxlen=3)
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
        self.replan_distance_trigger = float(max(0.5, replan_distance_trigger))  # 新增:主动重规划间距
        self.dynamic_replan_neighbor_dist = max(0.5, float(dynamic_replan_neighbor_dist))
        self.dynamic_replan_ttc = max(0.5, float(dynamic_replan_ttc))
        self.dynamic_replan_block_radius = max(0.10, float(dynamic_replan_block_radius))
        self.obs_target_dist_clip = max(0.5, float(obs_target_dist_clip))
        self.obs_target_filter_alpha = float(np.clip(float(obs_target_filter_alpha), 0.0, 1.0))
        self.obs_target_max_step = max(0.05, float(obs_target_max_step))
        self.progress_reward_scale = float(progress_reward_scale)
        self.path_progress_reward_scale = float(path_progress_reward_scale)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        # ⚠️ 以下 4 个缩放参数保留在构造签名中以兼容外部传参，但当前奖励函数
        #    (get_step_result) 并未读取它们——调这些值不会改变奖励。lateral 惩罚目前
        #    在 get_step_result 中用硬编码 -0.05。若要启用，需在奖励函数里接回这些字段。
        self.goal_progress_reward_scale = float(goal_progress_reward_scale)   # 未使用
        self.lateral_penalty_scale = float(lateral_penalty_scale)            # 未使用
        self.heading_align_reward_scale = float(heading_align_reward_scale)  # 未使用
        self.narrow_forward_penalty_scale = float(narrow_forward_penalty_scale)  # 未使用
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
        self._last_replan_pos = None  # 新增:上次重规划的位置,用于距离触发
        self._last_subgoal_mode = 'nominal'

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
            map_mapping = {1: 'map1', 2: 'map2', 3: 'corridor_swap', 4: 'intersection', 5: 'warehouse_aisles', 6: 'interaction_hub', 7: 'interaction_hub_mini', 8: 'circle_swap_arena'}
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

        # 避开动态障碍物(dyn_obs_*)的实时位置：障碍物随机游走，机器人 spawn 在其
        # 紧邻处会开局即被撞死（初速 0 来不及反应）。要求起点离每个障碍物 ≥0.8m，
        # 给机器人在 grace 期内开始移动/转向的反应空间。
        live_obs = getattr(self, '_live_obstacle_positions', None)
        if live_obs:
            if any(math.hypot(wx - ox, wy - oy) < self._spawn_obstacle_clearance
                   for ox, oy in live_obs):
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

    # ===================================================================
    # ROS 回调：激光雷达 / 里程计 / 碰撞接触
    # ===================================================================
    def _scan_callback(self, msg):
        self.latest_scan = msg
        self._scan_seq += 1

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
                'left_min': self.scan_max_range,
                'right_min': self.scan_max_range,
            }

        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.scan_max_range, posinf=self.scan_max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        valid = ranges[(ranges > self.scan_valid_min)]
        min_dist = float(valid.min()) if valid.size else self.scan_max_range

        n = len(ranges)
        if n < 8:
            return {'min_dist': min_dist, 'front_min': min_dist, 'left_min': min_dist, 'right_min': min_dist}

        front_idx = np.r_[0:max(1, n // 18), n - max(1, n // 18):n]
        left_idx = np.arange(n // 6, n // 3)
        right_idx = np.arange(2 * n // 3, 5 * n // 6)

        def _sector_min(idx):
            vals = ranges[idx]
            vals = vals[(vals > self.scan_valid_min)]
            return float(vals.min()) if vals.size else self.scan_max_range

        return {
            'min_dist': min_dist,
            'front_min': _sector_min(front_idx),
            'left_min': _sector_min(left_idx),
            'right_min': _sector_min(right_idx),
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

    def _predict_trajectory(self, x, y, vx, vy, num_steps=PREDICTION_STEPS, dt=PREDICTION_DT):
        """
        多步线性运动预测 (Sim2Real: 简单线性模型，计算高效)

        Args:
            x, y: 当前位置 (body frame)
            vx, vy: 速度 (body frame)
            num_steps: 预测步数
            dt: 时间步长

        Returns:
            trajectory: [(x1, y1), (x2, y2), ...] 未来位置列表 (body frame)
        """
        trajectory = []
        for i in range(1, num_steps + 1):
            t = i * dt
            pred_x = x + vx * t
            pred_y = y + vy * t
            trajectory.append((pred_x, pred_y))
        return trajectory

    def _check_trajectory_collision_risk(self, trajectory, robot_path_waypoints=None):
        """
        检查预测轨迹的碰撞风险 (Sim2Real: 考虑膨胀半径)

        Args:
            trajectory: 障碍物预测轨迹 [(x1, y1), ...] (body frame)
            robot_path_waypoints: 机器人全局路径点 (可选)

        Returns:
            max_risk: 0.0-1.0 的风险值
        """
        max_risk = 0.0

        # 方法1: 检查轨迹点到机器人的最小距离
        for pred_x, pred_y in trajectory:
            pred_dist = math.hypot(pred_x, pred_y)
            effective_dist = max(0.0, pred_dist - INFLATION_RADIUS)

            # 风险随距离衰减
            if effective_dist < 1.0:
                risk = 1.0 - (effective_dist / 1.0)
                max_risk = max(max_risk, risk)

        # 方法2: 如果有机器人路径，检查轨迹是否与路径相交 (可选，更精确)
        if robot_path_waypoints and len(robot_path_waypoints) > 0:
            # 将路径转到 body frame 进行碰撞检测
            # 这里简化处理，只检查前方路径点
            pass

        return float(np.clip(max_risk, 0.0, 1.0))

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
        # 动态预测窗口：根据障碍物速度调整（Sim2Real: 快速物体需要更早预测）
        # predict_h 将在循环内根据每个障碍物的速度动态计算
        denom = max(self.obstacle_filter_range, 1e-6)
        corridor_half_width = max(self.predictive_min_sep, 0.45)

        for cluster in current_clusters:
            matched = self._match_previous_cluster(cluster, prev_clusters)
            vx_world = 0.0
            vy_world = 0.0
            if matched is not None:
                vx_world = float((float(cluster["xw"]) - float(matched["xw"])) / self.control_dt)
                vy_world = float((float(cluster["yw"]) - float(matched["yw"])) / self.control_dt)

            # 根据障碍物速度动态调整预测窗口
            obs_speed = math.hypot(vx_world, vy_world)
            if obs_speed > SPEED_THRESHOLD_FAST:
                predict_h = PREDICTION_WINDOW_FAST  # 2.0s
            elif obs_speed > SPEED_THRESHOLD_MED:
                predict_h = PREDICTION_WINDOW_MED   # 1.5s
            else:
                predict_h = PREDICTION_WINDOW_SLOW  # 1.0s

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

            # 应用安全膨胀半径 (Sim2Real: 考虑机器人尺寸和定位误差)
            effective_dist = max(0.0, dist - INFLATION_RADIUS)
            effective_future_dist = max(0.0, future_dist - INFLATION_RADIUS)

            close_risk = float(np.clip((self.close_obstacle_dist - effective_dist) / self.close_obstacle_dist, 0.0, 1.0))
            future_risk = float(np.clip((self.predictive_min_sep - effective_future_dist) / self.predictive_min_sep, 0.0, 1.0))
            crossing_gate = float(np.clip(1.0 - abs(future_y) / corridor_half_width, 0.0, 1.0))
            forward_gate = float(np.clip((future_x + 0.15) / max(self.obstacle_filter_range, 1e-6), 0.0, 1.0))
            transverse_speed = abs(vy)
            crossing_risk = crossing_gate * forward_gate * float(np.clip(transverse_speed / 0.6, 0.0, 1.0))
            closing_speed = float(max(0.0, -(x * vx + y * vy) / max(dist, 1e-6)))
            ttc = float(effective_dist / closing_speed) if closing_speed > 1e-3 else float("inf")
            ttc_risk = (
                float(np.clip((self.predictive_front_ttc_safe - ttc) / self.predictive_front_ttc_safe, 0.0, 1.0))
                if math.isfinite(ttc) else 0.0
            )

            # 多步轨迹预测风险 (Sim2Real: 检查整条未来轨迹)
            trajectory_risk = 0.0
            # 2026-06-29 bugfix: 此处原写 `speed`，但 `speed` 在下面第 ~2402 行才赋值，导致 UnboundLocalError
            # `obs_speed` 在本函数顶部已计算（用于选择 predict_h），语义一致，直接复用
            if obs_speed > 0.05:  # 只对运动物体预测轨迹
                trajectory = self._predict_trajectory(x, y, vx, vy)
                trajectory_risk = self._check_trajectory_collision_risk(trajectory)

            risk = max(close_risk, future_risk, crossing_risk, ttc_risk, trajectory_risk)
            if risk <= 1e-4:
                continue

            speed = obs_speed  # 保持下方 token 构造代码语义不变（speed 是 vx,vy 的模）
            is_dynamic = 1.0 if speed > 0.05 else 0.0
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
    # 子目标 / 重规划：A* 路径跟踪 + 死锁脱困 + 局部绕行
    # ===================================================================
    def _should_replan_by_distance(self) -> bool:
        """周期性距离触发重规划检查(每移动 replan_distance_trigger 米)"""
        if self._last_replan_pos is None:
            return True  # 首次,立即触发
        current_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
        dist_since_last = float(np.linalg.norm(current_pos - self._last_replan_pos))
        return dist_since_last >= self.replan_distance_trigger

    def _try_replan_due_to_deadlock(self) -> bool:
        """重规划执行(由死锁检测或距离触发调用)"""
        if not self.replan_on_deadlock or self.planner is None:
            return False
        if self.current_step < self._next_replan_step:
            return False

        try:
            start = (float(self.current_pose['x']), float(self.current_pose['y']))
            goal = (float(self.goal_pos[0]), float(self.goal_pos[1]))
            blocked = []
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
            path = self.planner.plan_with_dynamic_obstacles(
                start,
                goal,
                blocked_world_points=blocked,
                block_radius_m=self.dynamic_replan_block_radius,
            )
            if path is None:
                path = self.planner.plan(start, goal)
            self._next_replan_step = self.current_step + self.replan_cooldown_steps
            self._last_replan_pos = np.array([start[0], start[1]], dtype=np.float32)  # 记录重规划位置
            if not path:
                return False
            self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
            self.current_waypoint_index = 0
            self._subgoal_deadlock_streak = 0
            return True
        except Exception:
            self._next_replan_step = self.current_step + self.replan_cooldown_steps
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
        # 主动周期性重规划:每移动 replan_distance_trigger 米触发一次
        if self._should_replan_by_distance():
            self._try_replan_due_to_deadlock()

        pos = (self.current_pose['x'], self.current_pose['y'])
        path_points = self.global_waypoints if self.global_waypoints else [self.goal_pos]

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
            base = PathTrackingUtils.get_rolling_subgoal(pos, path_points, self.lookahead_dist)
            heading_error = self._get_target_angle(base['subgoal'])
            sectors = self._scan_sector_metrics()
            front_min = float(sectors['front_min'])
            left_min = float(sectors['left_min'])
            right_min = float(sectors['right_min'])
            sector_dists = self._get_current_sector_dists()
            adaptive = self._get_adaptive_lookahead(front_min, heading_error)
            info = PathTrackingUtils.get_rolling_subgoal(pos, path_points, adaptive)

            self.current_projection = tuple(info['projection'])
            chosen_subgoal, mode = self._select_local_detour_subgoal(
                tuple(info['subgoal']),
                adaptive,
                front_min,
                left_min,
                right_min,
                sector_dists=sector_dists,
            )
            self.current_subgoal = tuple(chosen_subgoal)
            self._last_subgoal_mode = mode
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
    def _apply_safety_shield(self, linear_vel, angular_vel, front_min, left_min, right_min, target_angle, neighbor_min_dist=None):
        if not self.shield_enable:
            return linear_vel, angular_vel, False

        turn_in_place = False
        if front_min < self.shield_front_slow_dist:
            linear_vel = min(linear_vel, self.shield_linear_slow)
        if front_min < self.shield_front_stop_dist:
            linear_vel = min(linear_vel, self.shield_linear_stop)
        if neighbor_min_dist is not None and neighbor_min_dist < self.shield_neighbor_slow_dist:
            linear_vel = min(linear_vel, self.shield_linear_slow)

        if front_min < self.turn_in_place_front_dist and abs(target_angle) > self.turn_in_place_angle_thresh:
            turn_in_place = True
            linear_vel = 0.0
            turn_dir = np.sign(target_angle)
            if abs(turn_dir) < 1e-6:
                turn_dir = 1.0 if left_min >= right_min else -1.0
            angular_vel = float(np.clip(self.turn_in_place_w * turn_dir, -1.2, 1.2))
        elif front_min < self.turn_in_place_front_dist:
            bias = self.shield_turn_bias if left_min >= right_min else -self.shield_turn_bias
            angular_vel = float(np.clip(angular_vel + bias, -1.2, 1.2))

        return float(linear_vel), float(angular_vel), turn_in_place

    def _check_collision_event(self, min_dist: float, info: Dict[str, Any]) -> bool:
        """碰撞判定：优先 Gazebo 硬碰撞事件，必要时回退雷达阈值。"""
        if self.use_gazebo_collision and self._gazebo_collision_active:
            info['collision_source'] = 'gazebo'
            # 消费事件，避免一次接触被重复多步计数
            self._gazebo_collision_active = False
            return True

        # 开局宽限：teleport 后头几步 lidar 可能是旧位姿残帧/瞬态，禁用 lidar_fallback
        # 误判（Gazebo 物理接触仍然有效，上面已处理）。期间保持 streak 清零，避免把
        # spawn 瞬态计入持续碰撞。grace 步内不触发 lidar_fallback 碰撞。
        if self.current_step <= self.collision_grace_steps:
            self._close_obstacle_streak = 0
            return False

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
    def reset(self, seed=None, options=None, other_agent_starts=None, forced_start_goal=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._publish_vel(0.0, 0.0)
        self._close_obstacle_streak = 0
        self._gazebo_collision_active = False
        self._front_min_history.clear()
        self._front_sector_dist_history.clear()
        self._obstacle_cluster_history.clear()
        self._last_motion_features = np.zeros(self.obstacle_motion_dim, dtype=np.float32)
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
        self._last_replan_pos = None  # 新增:上次重规划的位置,用于距离触发
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0

        if hasattr(self, 'vis') and self.vis:
            self.vis.clear_waypoints(namespace=self.vis_namespace)

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
        # 先推进仿真时间让 Gazebo 物理 settle，再主动等"传送后的新激光帧"。
        # 关键：_set_robot_pose 是异步传送，紧接着的首帧 scan 可能仍是旧位姿的残帧，
        # min_dist 会读到 <0.2m 的假障碍，被 lidar_fallback 判成 spawn 即碰撞。
        # 这里清零 latest_scan 后等待 _scan_seq 至少累计若干帧，丢弃瞬态，拿到稳定新帧。
        self._wait_for_sim_time(0.2)
        self._wait_for_fresh_scan(min_frames=5, timeout_sec=0.8)

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

        return self._get_obs(), {'start_xy': (start_x, start_y)}

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
        linear_vel, angular_vel, turn_in_place = self._apply_safety_shield(
            linear_vel,
            angular_vel,
            front_min,
            left_min,
            right_min,
            target_angle,
            neighbor_min_dist=neighbor_min_dist,
        )

        self._last_shield_info = {
            'front_min': front_min,
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
        }

        if debug and abs(linear_vel) < 0.01 and abs(angular_vel) < 0.01:
            if self.action_mode == 'continuous':
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                print(f"⚠️  Robot {self.robot_id}: 零动作! action=[{arr[0]:.3f}, {arr[1]:.3f}] -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")
            else:
                print(f"⚠️  Robot {self.robot_id}: 零动作! primitive={self._last_action_primitive} -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")

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

            # ==========================================
            # 1. r_progress: 朝目标前进 (goal_dist_delta + guidance heading)
            # ==========================================
            goal_dist_delta = 0.0
            if self.prev_dist_to_goal is not None:
                goal_dist_delta = float(self.prev_dist_to_goal - dist_to_goal)

            # heading shaping uses guidance direction (A* path direction)
            guidance_obs = self._get_guidance_obs()
            guidance_angle = math.atan2(float(guidance_obs[0]), float(guidance_obs[1]))
            heading_shaping = 0.0
            if forward_speed > RWD_HEADING_MIN_FWD_VEL:
                heading_shaping = RWD_HEADING_COEF * math.cos(guidance_angle)

            # 恢复 beifen 已验证的 r_progress 公式 (2026-06-30):
            # beifen 时代(STAGE_Cont_EnvStage1_o 等成功 run, reward +66/+145/+98) 用此公式:
            # progress_scale * (goal_dist_delta + heading_shaping),整体放大 + clip。
            # 配合 beifen 的 progress_scale=1.5, RWD_PROGRESS_CLIP=0.30, RWD_HEADING_COEF=0.10,
            # heading_shaping 单步最大 0.10,即使原地朝向也只 1.5*0.10=0.15,刷不出大分(对比 goal=60)。
            r_progress = self.progress_scale * (goal_dist_delta + heading_shaping)
            r_progress = float(np.clip(r_progress, -RWD_PROGRESS_CLIP, RWD_PROGRESS_CLIP))

            # ==========================================
            # 2. r_static: 障碍避碰（含动态障碍物）
            #    = 斥力势场 + 高风险减速 + 前方预测风险(含动态obs TTC)
            # ==========================================
            effective_min = min(min_dist, front_min)
            if effective_min < RWD_STATIC_D0:
                d_clamped = max(effective_min, RWD_STATIC_D_MIN)
                repulsive = -((1.0 / d_clamped) - (1.0 / RWD_STATIC_D0)) ** 2
            else:
                repulsive = 0.0

            speed_risk = 0.0
            if front_min < RWD_STATIC_SPEED_RISK_D0:
                risk_ratio = (RWD_STATIC_SPEED_RISK_D0 - front_min) / RWD_STATIC_SPEED_RISK_D0
                speed_risk = -forward_speed * (risk_ratio ** 2)

            front_risk = float(self._last_predictive_metrics.get('front_risk', 0.0))
            predictive_front_penalty = -(front_risk ** 2)

            # Near-miss penalty: 0.30m内擦肩而过强化惩罚(从0.25扩大,增强避碰)
            # 梯度平滑允许通过窄道(如1.2m走廊)
            near_miss_penalty = 0.0
            if effective_min < RWD_NEAR_MISS_DIST:
                near_miss_ratio = (RWD_NEAR_MISS_DIST - effective_min) / RWD_NEAR_MISS_DIST
                near_miss_penalty = -1.0 * (near_miss_ratio ** 2)

            r_static = self.static_scale * (repulsive + speed_risk + predictive_front_penalty + near_miss_penalty)
            r_static = float(max(-RWD_STATIC_CLIP, r_static))

            # ==========================================
            # 3. r_social: 动态避碰 (邻居机器人 TTC + 多步轨迹预测)
            # ==========================================
            worst_ttc_penalty = 0.0
            if hasattr(self, 'parent_env'):
                my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
                my_vel = np.array([
                    self.current_vel_x * math.cos(self.current_pose['yaw']),
                    self.current_vel_x * math.sin(self.current_pose['yaw'])
                ], dtype=np.float32)
                safe_ttc = float(self.predictive_social_ttc_safe)

                for aid, pos in self.parent_env.robot_positions.items():
                    if aid == f"agent_{self.robot_id}":
                        continue
                    rel_pos = pos - my_pos
                    dist = float(np.linalg.norm(rel_pos))
                    if dist >= RWD_SOCIAL_NEAR_DIST:
                        continue

                    # 应用安全膨胀半径 (Sim2Real: 两个机器人各有半径)
                    effective_dist = max(0.0, dist - 2 * INFLATION_RADIUS)

                    neighbor_vel = self.parent_env.robot_velocities[aid]
                    rel_vel = neighbor_vel - my_vel

                    # 相对速度在连线方向的投影
                    rel_pos_normalized = rel_pos / (dist + 1e-6)
                    approach_speed = float(-np.dot(rel_vel, rel_pos_normalized))

                    if approach_speed <= RWD_SOCIAL_APPROACH_TH:
                        continue

                    # TTC 基于有效距离
                    ttc = effective_dist / approach_speed if approach_speed > 1e-3 else float('inf')

                    # 多步轨迹预测：检查双方未来轨迹是否碰撞
                    trajectory_risk = 0.0
                    neighbor_speed = float(np.linalg.norm(neighbor_vel))
                    if neighbor_speed > 0.05:  # 邻居在运动
                        # 转到 body frame
                        rel_pos_body = self._world_to_body(rel_pos)
                        rel_vel_body = self._world_to_body(rel_vel)

                        # 预测邻居相对轨迹
                        neighbor_trajectory = self._predict_trajectory(
                            rel_pos_body[0], rel_pos_body[1],
                            rel_vel_body[0], rel_vel_body[1]
                        )
                        trajectory_risk = self._check_trajectory_collision_risk(neighbor_trajectory)

                    # 综合 TTC 和轨迹风险
                    if ttc < safe_ttc or trajectory_risk > 0.3:
                        ttc_penalty = -((safe_ttc - ttc) / safe_ttc) ** 2 if ttc < safe_ttc else 0.0
                        traj_penalty = -(trajectory_risk ** 2)
                        penalty = min(ttc_penalty, traj_penalty)
                        if penalty < worst_ttc_penalty:
                            worst_ttc_penalty = penalty

            r_social = self.social_scale * worst_ttc_penalty
            r_social = float(max(-RWD_SOCIAL_CLIP, r_social))

            # ==========================================
            # 3b. r_dynamic_obs: 动态障碍物避碰专项(多步轨迹预测+安全膨胀)
            # 2026-06-30 增强:权重 1.0→2.5、clip -1.0→-2.0,让障碍避碰惩罚量级能抗衡progress
            # ==========================================
            r_dynamic_obs = 0.0
            if self.obstacle_motion_dim > 0 and hasattr(self, '_last_motion_features'):
                motion_features = self._last_motion_features
                my_pos = np.array([self.current_pose['x'], self.current_pose['y']], dtype=np.float32)
                my_vel = np.array([
                    self.current_vel_x * math.cos(self.current_pose['yaw']),
                    self.current_vel_x * math.sin(self.current_pose['yaw'])
                ], dtype=np.float32)

                worst_penalty = 0.0
                for i in range(self.obstacle_motion_top_k):
                    start = i * 7
                    if start + 6 >= len(motion_features):
                        break

                    is_dynamic = float(motion_features[start + 6])
                    if is_dynamic < 0.5:
                        continue

                    obs_x = float(motion_features[start]) * 5.0
                    obs_y = float(motion_features[start + 1]) * 5.0
                    obs_vx = float(motion_features[start + 2]) * 0.8
                    obs_vy = float(motion_features[start + 3]) * 0.8

                    dist = float(math.hypot(obs_x, obs_y))
                    if dist > 2.0 or dist < 0.05:
                        continue

                    # 应用安全膨胀半径 (Sim2Real)
                    effective_dist = max(0.0, dist - INFLATION_RADIUS)

                    # 相对速度计算
                    rel_vel_x = obs_vx - self.current_vel_x
                    rel_vel_y = obs_vy - 0.0  # body frame
                    approach_speed = float(-(obs_x * rel_vel_x + obs_y * rel_vel_y) / (dist + 1e-6))

                    # TTC 计算
                    ttc_penalty = 0.0
                    if approach_speed > 0.05:
                        ttc = effective_dist / approach_speed
                        safe_ttc = 2.5
                        if ttc < safe_ttc:
                            ttc_penalty = -((safe_ttc - ttc) / safe_ttc) ** 2

                    # 多步轨迹预测风险
                    trajectory_risk = 0.0
                    obs_speed = math.hypot(obs_vx, obs_vy)
                    if obs_speed > 0.05:
                        trajectory = self._predict_trajectory(obs_x, obs_y, obs_vx, obs_vy)
                        trajectory_risk = self._check_trajectory_collision_risk(trajectory)
                        traj_penalty = -(trajectory_risk ** 2)
                    else:
                        traj_penalty = 0.0

                    # 综合惩罚
                    penalty = min(ttc_penalty, traj_penalty)
                    if penalty < worst_penalty:
                        worst_penalty = penalty

                r_dynamic_obs = 2.5 * worst_penalty  # 权重从1.0提到2.5,增强障碍避碰
                r_dynamic_obs = float(max(-RWD_DYNAMIC_OBS_CLIP, r_dynamic_obs))  # clip从-1.0扩到-2.0

            # ==========================================
            # 4-6. 终端 / 时间
            # ==========================================
            r_collision = 0.0
            r_goal = 0.0

            if self._check_collision_event(min_dist, info):
                r_collision = -float(self.collision_penalty)
                info['event'] = 'collision'
                info['collision_min_dist'] = float(min_dist)
                info['collision_step'] = int(self.current_step)
                if self.collision_ends_episode:
                    done = True

            if dist_to_goal < self.goal_reach_radius:
                r_goal = float(self.goal_reward)
                done = True
                info['event'] = 'goal'

            if self.current_step >= self.max_episode_steps:
                truncated = True

            r_time = -float(self.time_penalty)

            reward = r_progress + r_static + r_social + r_dynamic_obs + r_collision + r_goal + r_time

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
                'r_social': float(r_social),
                'r_collision': float(r_collision),
                'r_goal': float(r_goal),
                'r_time': float(r_time),
                'dist_to_goal': float(dist_to_goal),
                'min_dist': float(min_dist),
                'subgoal_mode': str(self._last_subgoal_mode),
                'shield_active': 1.0 if self._last_shield_info.get('shield_active', False) else 0.0,
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

    def _wait_for_fresh_scan(self, min_frames: int = 3, timeout_sec: float = 0.6):
        """等待传送后累计到 min_frames 帧新激光，丢弃 spawn 瞬态残帧。

        _scan_seq 在每次 _scan_callback 递增。reset 中调用 _set_robot_pose 前
        latest_scan 已被清零，这里从当前序号起等待若干新帧到达，确保返回首 obs
        时拿到的是机器人传送到位后的稳定激光，而非旧位姿残帧（否则首帧 min_dist
        可能 <0.2m，被 lidar_fallback 误判为 spawn 即碰撞）。带挂钟超时兜底，
        避免传感器异常时死等。
        """
        if not rclpy.ok():
            return
        start_seq = self._scan_seq
        deadline = time.time() + float(timeout_sec)
        while rclpy.ok() and time.time() < deadline:
            if (self._scan_seq - start_seq) >= int(min_frames):
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
        8: [
            ((2.5, 0.0), (-2.5, 0.0)),
            ((1.7678, 1.7678), (-1.7678, -1.7678)),
            ((0.0, 2.5), (0.0, -2.5)),
            ((-1.7678, 1.7678), (1.7678, -1.7678)),
            ((-2.5, 0.0), (2.5, 0.0)),
            ((-1.7678, -1.7678), (1.7678, 1.7678)),
            ((0.0, -2.5), (0.0, 2.5)),
            ((1.7678, -1.7678), (-1.7678, 1.7678)),
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
            ((2.5, 0.0), (-2.5, 0.0)),
            ((-2.5, 0.0), (2.5, 0.0)),
            ((1.7678, 1.7678), (-1.7678, -1.7678)),
            ((-1.7678, -1.7678), (1.7678, 1.7678)),
            ((0.0, 2.5), (0.0, -2.5)),
            ((0.0, -2.5), (0.0, 2.5)),
            ((-1.7678, 1.7678), (1.7678, -1.7678)),
            ((1.7678, -1.7678), (-1.7678, 1.7678)),
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
        # 动态障碍物由 Gazebo Actor 插件自驱动（见 world 文件 / obstacle_mover.py），
        # 环境侧无需主动摆放，保留此接口仅为兼容旧调用方。
        pass

    # ===================================================================
    # Guidance observation: A* 粗方向 + goal 并行信号
    # ===================================================================

    def _get_guidance_obs(self) -> np.ndarray:
        """6-dim guidance: [guidance_sin, guidance_cos, guidance_dist_norm, goal_sin, goal_cos, goal_dist_norm]"""
        px = self.current_pose['x']
        py = self.current_pose['y']
        yaw = self.current_pose['yaw']

        # guidance direction from A* path (lookahead ~2m)
        path_points = self.global_waypoints if getattr(self, 'global_waypoints', None) else None
        if path_points and len(path_points) >= 2:
            info = PathTrackingUtils.get_rolling_subgoal(
                (px, py), path_points, self.guidance_lookahead_m
            )
            gx, gy = info['subgoal']
        else:
            gx, gy = self.goal_pos

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
        """Build ego-centric 32x32 binary occupancy grid from lidar, stack 2 frames, return flat 2048-dim."""
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
        while len(frames) < 2:
            frames.insert(0, frames[0].copy())
        stacked = np.stack(frames[-2:], axis=0)
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
            return

        req = SetEntityState.Request()
        req.state.name = self.gazebo_model_name

        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        # z=0.01 与 launch 初次 spawn 对齐（spawn_robots.launch.py 默认 z=0.01）。
        # 此前用 0.1（高 10cm）导致 teleport 后机器人下落沉降，沉降期约 9 步内激光
        # 俯仰扫到地面，产生 0.12~0.21m 幻影近读数，被 lidar_fallback 误判为碰撞，
        # 整局在 grace 解除后立即 too_few_active 掐死（详见 spawn-lidar 误判分析）。
        req.state.pose.position.z = 0.01
        req.state.pose.orientation.w = math.cos(yaw / 2)
        req.state.pose.orientation.z = math.sin(yaw / 2)

        future = self.set_state_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

    def query_dynamic_obstacle_positions(self, num: int) -> list:
        """查询 dyn_obs_0~(num-1) 的实时世界坐标，用于 spawn 避让。

        返回 [(x,y), ...]。障碍物由 obstacle_mover.py 随机游走驱动，z=-10 表示未激活
        (已下沉)，跳过。服务不可用或超时则返回空列表（spawn 退化为不避让，但有
        grace + 等帧兜底）。
        """
        positions = []
        n = int(max(0, min(num, 8)))
        if n == 0:
            return positions
        if not self.get_state_client.wait_for_service(timeout_sec=0.5):
            return positions
        for i in range(n):
            try:
                req = GetEntityState.Request()
                req.name = f'dyn_obs_{i}'
                future = self.get_state_client.call_async(req)
                rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.3)
                res = future.result()
                if res is None or not getattr(res, 'success', False):
                    continue
                p = res.state.pose.position
                if float(p.z) < -1.0:  # 已下沉的未激活障碍物
                    continue
                positions.append((float(p.x), float(p.y)))
            except Exception:
                continue
        return positions

    def close(self):
        self.node.destroy_node()
