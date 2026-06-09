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
from gazebo_msgs.srv import SetEntityState
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
        # auto_reset_agents=True  : 连续任务流（GNN阶段，agent done后立刻在原episode重置）
        # auto_reset_agents=False : 单任务 episode（MLP基线，对齐 marl_training 语义）
        self.auto_reset_agents = bool(config.get('auto_reset_agents', False))
        # 是否将“collision 事件”作为局部重置触发条件（默认开启：判定为碰撞才重置）
        self.reset_on_collision_event = bool(config.get('reset_on_collision_event', True))

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
                'near_wall_penalty_dist': float(config.get('near_wall_penalty_dist', 0.30)),
                'waypoint_reach_radius': float(config.get('waypoint_reach_radius', 0.8)),
                'waypoint_distance_threshold': float(config.get('waypoint_distance_threshold', 1.2)),
                'waypoint_min_clearance_m': float(config.get('waypoint_min_clearance_m', 0.40)),
                'use_voronoi_planner': bool(config.get('use_voronoi_planner', False)),
                'voronoi_min_clearance_m': float(config.get('voronoi_min_clearance_m', 0.35)),
                'num_dynamic_obstacles': config.get('num_dynamic_obstacles', 8),
                'obs_speed': config.get('obs_speed', 0.3),
                'rolling_lookahead_dist': float(config.get('rolling_lookahead_dist', 0.8)),
                'progress_reward_scale': float(config.get('progress_reward_scale', 4.0)),
                'path_progress_reward_scale': float(config.get('path_progress_reward_scale', 3.0)),
                'goal_progress_reward_scale': float(config.get('goal_progress_reward_scale', 1.5)),
                'goal_reward': float(config.get('goal_reward', 40.0)),
                'collision_penalty': float(config.get('collision_penalty', 25.0)),
                'time_penalty': float(config.get('time_penalty', 0.002)),
                'lateral_penalty_scale': float(config.get('lateral_penalty_scale', 0.05)),
                'heading_align_reward_scale': float(config.get('heading_align_reward_scale', 0.15)),
                'narrow_forward_penalty_scale': float(config.get('narrow_forward_penalty_scale', 0.35)),
                'shield_enable': bool(config.get('shield_enable', True)),
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
            }
            supported_kwargs = {
                key: value for key, value in candidate_kwargs.items()
                if key in env_signature.parameters
            }
            self.agents[f"agent_{i}"] = IndependentRobotEnv(**supported_kwargs)
        
        self.agent_ids = list(self.agents.keys())
        self.current_step_count = 0
        self.max_steps = config.get('max_episode_steps', 1000)
        self.dones = set()  # 已完成的智能体
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

        # 定义增强后的观测空间
        self._define_observation_space()
        
        # 定义动作空间（与 IndependentRobotEnv 一致: [-1,1]x[-1,1]）
        # 底层环境自动映射: linear_vel=(a[0]+1)/2*0.22, angular_vel=a[1]*1.0
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

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
    
    def _define_observation_space(self):
        """定义观测空间"""
        # 从实际 agent 动态获取 base_obs_dim，避免与 IndependentRobotEnv.obs_dim 不一致
        # IndependentRobotEnv.obs_dim = scan_dim*scan_history_len + 2 + 2 + safety_feature_dim
        # 当前默认: 36*4 + 2 + 2 + 7 = 155
        base_obs_dim = self.agents['agent_0'].obs_dim
        
        # 可选：邻居状态（最多 K 个近邻）
        if self.enable_neighbor_obs:
            # 【修复】最多邻居数 = min(其他机器人数量, 5)
            max_neighbors = min(self._num_agents - 1, 5)  # 2个机器人时 = min(1, 5) = 1
            # 每个邻居: 相对位置(2) + 相对速度(2) + 距离(1) = 5
            neighbor_dim = max_neighbors * 5
        else:
            neighbor_dim = 0
        
        # 可选：局部地图（暂时不实现，预留）
        if self.enable_local_map:
            local_map_dim = 128  # CNN编码后的特征维度
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
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """重置环境"""
        self.current_step_count = 0
        self.dones = set()
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

        obs_dict = {}
        info_dict = {}
        
        base_obs_dict = {}
        agent_starts = []  # list of (x, y)
        for aid, agent in self.agents.items():
            obs, info = agent.reset(other_agent_starts=agent_starts)
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
        for aid in self.agent_ids:
            if aid in action_dict and aid not in self.dones:
                obs, rew, done, truncated, info = self.agents[aid].get_step_result()

                # 更新位置和速度缓存
                self.robot_positions[aid]  = self._get_robot_position(self.agents[aid])
                self.robot_velocities[aid] = self._get_robot_velocity(self.agents[aid])

                event = info.get('event', '')
                # 触发“局部重置/终止处理”的条件：
                # 1) 底层显式 need_reset；2) done=True；3) goal 事件；
                # 4) collision 事件且显式开启 reset_on_collision_event。
                terminal = bool(
                    info.get('need_reset', False)
                    or done
                    or event == 'goal'
                    or (event == 'collision' and self.reset_on_collision_event)
                )

                if terminal:
                    if event == 'goal':
                        self.episode_successes[aid] += 1
                        self.logger.info('[step %d] %s 到达目标 (本 episode 第 %d 次)',
                                         self.current_step_count, aid, self.episode_successes[aid])
                    elif event == 'collision':
                        self.episode_collisions[aid] += 1
                        self.logger.info('[step %d] %s 碰撞 (本 episode 第 %d 次)',
                                         self.current_step_count, aid, self.episode_collisions[aid])

                    if self.auto_reset_agents:
                        # ── 连续任务流模式（GNN 课程学习阶段）──
                        # agent 触发终止后立刻重置到新起点，继续在当前 episode 学习
                        other_starts = [
                            self.robot_positions[a].tolist()
                            for a in self.agent_ids if a != aid
                        ]
                        new_obs, new_info = self.agents[aid].reset(other_agent_starts=other_starts)
                        new_spawn = new_info.get('start_xy')
                        if new_spawn:
                            self.robot_positions[aid] = np.array(new_spawn, dtype=np.float32)
                        self.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
                        obs = new_obs
                        info['auto_reset'] = True
                        info['new_spawn']  = new_spawn
                        done_dict[aid]      = False   # 连续流：不向 RLlib 报告单 agent done
                        truncated_dict[aid] = False
                    else:
                        # ── 单任务 episode 模式（MLP 基线）──
                        # agent 触发终止后加入 dones 集合，后续步骤保持静止
                        # episode 整体终止条件：全部完成 or 超时
                        self.dones.add(aid)
                        done_dict[aid]      = False   # 等 episode 整体结束时统一标记
                        truncated_dict[aid] = False
                else:
                    done_dict[aid]      = False
                    truncated_dict[aid] = False

                obs_dict[aid]  = obs
                # ── NaN 守卫：reward NaN → 统计 episode_reward_mean = NaN
                if not np.isfinite(rew):
                    self.logger.warning('[step %d] %s reward=%.4f → 替换为 -0.01',
                                        self.current_step_count, aid, rew)
                    rew = -0.01
                rew_dict[aid]  = rew
                info_dict[aid] = info
            else:
                # 已完成 或 无动作的 agent：返回当前观测，奖励为 0
                obs_dict[aid]  = self.agents[aid]._get_obs()
                rew_dict[aid]  = 0.0
                done_dict[aid] = False
                truncated_dict[aid] = False
                info_dict[aid] = {
                    'status': 'done_waiting' if aid in self.dones else 'no_action_received'
                }
        
        # ── 构建通信图 & 更新状态历史 ────────────────────────────────────
        adjacency_matrix = self._build_communication_graph()
        if self.comm_mode == 'ros2_bridge':
            self._broadcast_ros2_states()
        else:
            self._push_state_snapshot()
        
        # ── 构建增强观测（邻居信息 + reset_flag + 全局状态）──────────────
        reset_flags = {aid: 0.0 for aid in self.agent_ids}
        for aid in self.agent_ids:
            if info_dict.get(aid, {}).get('auto_reset', False):
                reset_flags[aid] = 1.0

        enhanced_obs_dict = {}
        for aid in self.agent_ids:
            enhanced_obs = self._build_enhanced_observation(
                aid, obs_dict[aid], adjacency_matrix,
                all_base_obs=obs_dict,
                reset_flag=reset_flags[aid],
            )
            enhanced_obs_dict[aid] = enhanced_obs
        
        # ── 添加图信息到 info ─────────────────────────────────────────────
        for aid in self.agent_ids:
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

        if self.auto_reset_agents:
            # 连续任务流：只有超时结束
            episode_over = timeout
        else:
            # 单任务 episode：所有 agent 完成 or 超时，对齐 marl_training
            episode_over = all_done or timeout

        done_dict["__all__"]      = episode_over
        truncated_dict["__all__"] = timeout

        if episode_over:
            reason = 'timeout' if timeout else 'all_done'
            self.logger.info(
                '━━━ EPISODE END (%s, step=%d) dones=%d/%d '
                'successes=%s collisions=%s ━━━',
                reason, self.current_step_count,
                len(self.dones), self._num_agents,
                {aid: self.episode_successes[aid]  for aid in self.agent_ids},
                {aid: self.episode_collisions[aid] for aid in self.agent_ids},
            )
            print(
                f"\n{'='*60}\n"
                f"🏁 Episode 结束 ({reason})\n"
                f"   步数: {self.current_step_count}/{self.max_steps}\n"
                f"   完成: {len(self.dones)}/{self._num_agents}\n"
                f"{'='*60}\n"
            )
            for aid in self.agent_ids:
                # episode 结束时所有 agent 统一标记终止
                # auto_reset_agents=True 时 self.dones 始终为空，必须在此统一设置
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
        
        # 计算全量距离矩阵（用于日志）
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                if d < self.communication_range:
                    adjacency[i, j] = 1.0
                    adjacency[j, i] = 1.0
        
        np.fill_diagonal(adjacency, 1.0)

        # ── 日志：每步都记录到文件，控制台只在前3步或 debug_comm=True 时打印 ──
        log_lines = [f'[graph] step={self.current_step_count:4d}  comm_range={self.communication_range}m']
        for i in range(n):
            adj_row   = '  '.join(f'{adjacency[i,j]:.0f}' for j in range(n))
            dist_row  = '  '.join(f'{dist_matrix[i,j]:5.2f}' for j in range(n))
            neighbors = [j for j in range(n) if adjacency[i, j] > 0 and j != i]
            pos_i     = positions[i]
            log_lines.append(
                f'  robot_{i} pos=({pos_i[0]:6.3f},{pos_i[1]:6.3f})')
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
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
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
        agent_idx     = int(agent_id.split('_')[1])
        my_pos        = self.robot_positions[agent_id]
        my_vel        = self.robot_velocities[agent_id]
        max_neighbors = min(self._num_agents - 1, 5)

        candidate_indices = [
            i for i in np.where(adjacency_matrix[agent_idx] > 0)[0]
            if i != agent_idx
        ]

        received: List[Tuple[float, np.ndarray, np.ndarray]] = []

        if self.comm_mode == 'ros2_bridge':
            # ── ROS2 DDS 消息模式 ────────────────────────────────────────────
            # 消息年龄 = now - data[5]（send_wall_sec）
            # 只接受年龄 >= (latency + jitter) × 0.1s 的消息，模拟 WiFi 延迟
            now     = time.monotonic()
            step_dt = 0.1    # TurtleBot3 @ 10Hz，与 robot_policy_node 对齐
            jitter  = int(self.rng.integers(0, self.comm_jitter_steps + 1))
            min_age = (self.comm_latency_steps + jitter) * step_dt

            for n_idx in candidate_indices:
                # 丢包：模拟 WiFi UDP 丢帧
                if self.rng.random() < self.comm_dropout_prob:
                    continue

                n_id = f'agent_{n_idx}'
                buf  = self._ros2_neighbor_bufs[n_id]

                # 找最新的、但年龄 >= min_age 的消息（最近的延迟消息）
                selected_data = None
                for _recv_t, data in reversed(buf):    # buf 按接收顺序排列
                    if (now - data[5]) >= min_age:
                        selected_data = data
                        break

                if selected_data is None:
                    continue   # episode 初始 latency 步内：无满足延迟的消息，填零

                n_pos = np.array([selected_data[1], selected_data[2]], dtype=np.float32)
                n_vel = np.array([selected_data[3], selected_data[4]], dtype=np.float32)

                if self.comm_noise_std > 0.0:
                    n_pos = n_pos + self.rng.normal(0.0, self.comm_noise_std, 2).astype(np.float32)
                    n_vel = n_vel + self.rng.normal(0.0, self.comm_noise_std, 2).astype(np.float32)

                dist = float(np.linalg.norm(my_pos - n_pos))
                if dist <= self.communication_range:
                    received.append((dist, n_pos, n_vel))

        else:
            # ── Buffer 模式（centralized_oracle / decentralized）──────────────
            snapshot = self._get_delayed_snapshot()

            for n_idx in candidate_indices:
                if self.comm_mode == 'decentralized' and self.rng.random() < self.comm_dropout_prob:
                    continue

                n_id  = f'agent_{n_idx}'
                n_pos = snapshot['positions'][n_id].copy()
                n_vel = snapshot['velocities'][n_id].copy()

                if self.comm_mode == 'decentralized' and self.comm_noise_std > 0.0:
                    n_pos = n_pos + self.rng.normal(0.0, self.comm_noise_std, 2)
                    n_vel = n_vel + self.rng.normal(0.0, self.comm_noise_std, 2)

                dist = float(np.linalg.norm(my_pos - n_pos))
                if dist <= self.communication_range:
                    received.append((dist, n_pos, n_vel))

        received.sort(key=lambda x: x[0])   # 距离升序

        # 编码最近 K 个邻居（不足则填零，保持向量维度固定）
        features_list: List[np.ndarray] = []
        for k in range(max_neighbors):
            if k < len(received):
                dist, n_pos, n_vel = received[k]
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

        neighbor_vec = np.concatenate(features_list).astype(np.float32)

        # ── 日志：通信详情（每步写文件，前3步或 debug_comm=True 同时打印） ──
        comm_lines = [
            f'[comm] step={self.current_step_count:4d}  {agent_id}'
            f'  my_pos=({my_pos[0]:.3f},{my_pos[1]:.3f})'  
            f'  candidates={candidate_indices}  received={len(received)}/{len(candidate_indices)}'
        ]
        for k, (dist, n_pos, n_vel) in enumerate(received):
            comm_lines.append(
                f'    slot[{k}] dist={dist:.3f}m  '
                f'n_pos=({n_pos[0]:.3f},{n_pos[1]:.3f})  '
                f'rel_pos=({n_pos[0]-my_pos[0]:.3f},{n_pos[1]-my_pos[1]:.3f})'
            )
        if len(received) == 0:
            comm_lines.append(f'    (无有效邻居: 范围外/丢包/延迟填零)')
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
                 collision_ends_episode=True,
                 collision_hard_dist=0.05,
                 collision_persist_dist=0.15,
                 collision_persist_steps=3,
                 near_wall_penalty_dist=0.20,
                 waypoint_reach_radius=0.8,
                 waypoint_distance_threshold=1.2,
                 waypoint_min_clearance_m=0.40,
                 use_voronoi_planner=False,
                 voronoi_min_clearance_m=0.35,
                 num_dynamic_obstacles=8, obs_speed=0.3,
                 rolling_lookahead_dist=0.8,
                 progress_reward_scale=4.0,
                 path_progress_reward_scale=3.0,
                 goal_progress_reward_scale=1.5,
                 goal_reward=40.0,
                 collision_penalty=25.0,
                 time_penalty=0.002,
                 lateral_penalty_scale=0.05,
                 heading_align_reward_scale=0.15,
                 narrow_forward_penalty_scale=0.35,
                 shield_enable=True,
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
                 lidar_collision_fallback=False):
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

        self.scan_dim = 36
        self.scan_history_len = 4
        self._scan_history: deque = deque(maxlen=self.scan_history_len)
        self.safety_feature_dim = 7
        self.obs_dim = self.scan_dim * self.scan_history_len + 2 + 2 + self.safety_feature_dim
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        self.goal_pos = (0.0, 0.0)
        self.prev_dist_to_goal = None
        self.prev_dist_to_target = None
        self.prev_target_point = None

        self.num_dynamic_obstacles = max(0, min(int(num_dynamic_obstacles), 8))
        self.obs_speed = float(obs_speed)
        self.dynamic_obstacle_names: list = [f'dyn_obs_{i}' for i in range(8)]

        self.lookahead_dist = float(rolling_lookahead_dist)
        self.progress_reward_scale = float(progress_reward_scale)
        self.path_progress_reward_scale = float(path_progress_reward_scale)
        self.goal_progress_reward_scale = float(goal_progress_reward_scale)
        self.goal_reward = float(goal_reward)
        self.collision_penalty = float(collision_penalty)
        self.time_penalty = float(time_penalty)
        self.lateral_penalty_scale = float(lateral_penalty_scale)
        self.heading_align_reward_scale = float(heading_align_reward_scale)
        self.narrow_forward_penalty_scale = float(narrow_forward_penalty_scale)

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
        self._gazebo_collision_seen = False
        self._gazebo_collision_last_step = -10**9

        self.current_subgoal = None
        self.current_projection = None
        self.current_path_heading = 0.0
        self.path_progress = 0.0
        self.prev_path_progress = None
        self.current_lateral_error = 0.0

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
            map_mapping = {1: 'map1', 2: 'map2', 3: 'corridor_swap', 4: 'intersection', 5: 'warehouse_aisles'}
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
            return {'min_dist': 3.5, 'front_min': 3.5, 'left_min': 3.5, 'right_min': 3.5}

        ranges = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.0)
        ranges = np.clip(ranges, 0.0, 3.5)
        valid = ranges[(ranges > 0.10)]
        min_dist = float(valid.min()) if valid.size else 3.5

        n = len(ranges)
        if n < 8:
            return {'min_dist': min_dist, 'front_min': min_dist, 'left_min': min_dist, 'right_min': min_dist}

        front_idx = np.r_[0:max(1, n // 18), n - max(1, n // 18):n]
        left_idx = np.arange(n // 6, n // 3)
        right_idx = np.arange(2 * n // 3, 5 * n // 6)

        def _sector_min(idx):
            vals = ranges[idx]
            vals = vals[(vals > 0.10)]
            return float(vals.min()) if vals.size else 3.5

        return {
            'min_dist': min_dist,
            'front_min': _sector_min(front_idx),
            'left_min': _sector_min(left_idx),
            'right_min': _sector_min(right_idx),
        }

    def _wrap_angle(self, angle):
        return (float(angle) + math.pi) % (2 * math.pi) - math.pi

    def _get_target_angle(self, target):
        tgt_angle = math.atan2(target[1] - self.current_pose['y'], target[0] - self.current_pose['x'])
        return self._wrap_angle(tgt_angle - self.current_pose['yaw'])

    def _get_adaptive_lookahead(self, front_min, heading_error):
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
        return float(np.clip(lookahead, 0.25, self.lookahead_dist))

    def _get_tracking_target(self):
        pos = (self.current_pose['x'], self.current_pose['y'])
        path_points = self.global_waypoints if self.global_waypoints else [self.goal_pos]
        base = PathTrackingUtils.get_rolling_subgoal(pos, path_points, self.lookahead_dist)
        heading_error = self._get_target_angle(base['subgoal'])
        front_min = self._scan_sector_metrics()['front_min']
        adaptive = self._get_adaptive_lookahead(front_min, heading_error)
        info = PathTrackingUtils.get_rolling_subgoal(pos, path_points, adaptive)
        self.current_projection = tuple(info['projection'])
        self.current_subgoal = tuple(info['subgoal'])
        self.current_path_heading = float(info.get('path_heading', 0.0))
        self.path_progress = float(info.get('arc_progress', 0.0))
        self.current_lateral_error = float(info.get('lateral_error', 0.0))
        return self.current_subgoal

    def _publish_tracking_visuals(self, target_pos):
        if not (hasattr(self, 'vis') and self.vis):
            return
        try:
            self.vis.publish_tracking_state(
                robot_pos=(self.current_pose['x'], self.current_pose['y']),
                target_pos=target_pos,
                projection_pos=self.current_projection,
                robot_id=self.robot_id,
                namespace=self.vis_namespace,
                label=f'R{self.robot_id} rolling_subgoal',
            )
        except Exception as _vis_e:
            self.node.get_logger().warn(f'publish_tracking_state failed: {_vis_e}')

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

        side_min = min(left_min, right_min)
        if front_min < self.turn_in_place_front_dist and abs(target_angle) > self.turn_in_place_angle_thresh:
            turn_in_place = True
            linear_vel = 0.0
            turn_dir = np.sign(target_angle)
            if abs(turn_dir) < 1e-6:
                turn_dir = 1.0 if left_min >= right_min else -1.0
            if side_min < 0.22:
                side_ratio = float(np.clip((side_min - 0.10) / 0.12, 0.0, 1.0))
                effective_w = max(0.30, self.turn_in_place_w * side_ratio)
                angular_vel = float(np.clip(effective_w * turn_dir, -1.2, 1.2))
            else:
                angular_vel = float(np.clip(self.turn_in_place_w * turn_dir, -1.2, 1.2))
        elif front_min < self.turn_in_place_front_dist:
            bias = self.shield_turn_bias if left_min >= right_min else -self.shield_turn_bias
            angular_vel = float(np.clip(angular_vel + bias, -1.2, 1.2))

        if side_min < 0.20:
            side_ratio = float(np.clip((side_min - 0.10) / 0.10, 0.0, 1.0))
            effective_max_w = max(0.30, 1.2 * side_ratio)
            angular_vel = float(np.clip(angular_vel, -effective_max_w, effective_max_w))

        return float(linear_vel), float(angular_vel), turn_in_place

    def _check_collision_event(self, min_dist: floa