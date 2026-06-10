"""
GNN-MAPPO 环境包装器
基于动态图神经网络的多智能体强化学习环境
"""
import os
import time
import random
import logging
import inspect
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
from gymnasium import spaces
import rclpy
from std_msgs.msg import Float32MultiArray

# 使用包内独立环境实现
from gnn_marl_training.independent_env import IndependentRobotEnv


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


class GNNMARLEnv:
    """
    GNN-MAPPO 环境包装器
    
    核心特性：
    1. 动态图构建：只连接近距离机器人
    2. 消息传递：机器人间信息交换
    3. 增强观测：局部地图 + 邻居状态
    """
    
    def __init__(self, config: Dict):
        self.num_agents = config.get('num_agents', 3)
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

        # 创建独立环境实例
        self.agents = {}
        env_signature = inspect.signature(IndependentRobotEnv.__init__)
        for i in range(self.num_agents):
            candidate_kwargs = {
                'robot_id': i,
                'map_number': config.get('map_number', 3),
                'max_episode_steps': config.get('max_episode_steps', 1000),
                'collision_ends_episode': False,
                'num_dynamic_obstacles': config.get('num_dynamic_obstacles', 8),
                'obs_speed': config.get('obs_speed', 0.3),
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
        
        # 机器人位置缓存（用于构建图）
        self.robot_positions = {aid: np.zeros(2) for aid in self.agent_ids}
        # 机器人速度缓存（用于协作奖励和碰撞预测）
        self.robot_velocities = {aid: np.zeros(2) for aid in self.agent_ids}

        # episode 级累计统计（auto-reset 模式下每个 agent 在一个 episode 内的成功/碰撞次数）
        self.episode_successes  = {aid: 0 for aid in self.agent_ids}
        self.episode_collisions = {aid: 0 for aid in self.agent_ids}

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
            self.num_agents, self.comm_mode, self.communication_range,
            self.comm_latency_steps, self.comm_noise_std
        )

        if self.comm_mode == 'ros2_bridge':
            self._setup_ros2_comm_bridge()
    
    def _define_observation_space(self):
        """定义观测空间"""
        # 从实际 agent 动态获取 base_obs_dim，避免与 IndependentRobotEnv.obs_dim 不一致
        # IndependentRobotEnv.obs_dim = scan_dim * scan_history_len + 2 + 2
        # 例如 scan_history_len=4 时为 148，=1 时为 40
        base_obs_dim = self.agents['agent_0'].obs_dim
        
        # 可选：邻居状态（最多 K 个近邻）
        if self.enable_neighbor_obs:
            # 【修复】最多邻居数 = min(其他机器人数量, 5)
            max_neighbors = min(self.num_agents - 1, 5)  # 2个机器人时 = min(1, 5) = 1
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
        self.global_state_dim = self.num_agents * base_obs_dim

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
    
    def reset(self, seed=None) -> Tuple[Dict, Dict]:
        """重置环境"""
        self.current_step_count = 0
        self.dones = set()
        self.episode_successes  = {aid: 0 for aid in self.agent_ids}
        self.episode_collisions = {aid: 0 for aid in self.agent_ids}
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
        n = self.num_agents
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

                terminal = bool(info.get('need_reset', False) or done)

                if terminal:
                    event = info.get('event', '')
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
        all_done  = (len(self.dones) == self.num_agents)   # 仅 auto_reset=False 时有意义

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
                len(self.dones), self.num_agents,
                {aid: self.episode_successes[aid]  for aid in self.agent_ids},
                {aid: self.episode_collisions[aid] for aid in self.agent_ids},
            )
            print(
                f"\n{'='*60}\n"
                f"🏁 Episode 结束 ({reason})\n"
                f"   步数: {self.current_step_count}/{self.max_steps}\n"
                f"   完成: {len(self.dones)}/{self.num_agents}\n"
                f"{'='*60}\n"
            )
            for aid in self.agent_ids:
                # episode 结束时所有 agent 统一标记终止
                # auto_reset_agents=True 时 self.dones 始终为空，必须在此统一设置
                done_dict[aid]      = True
                truncated_dict[aid] = timeout
                info_dict[aid]['episode_successes']  = self.episode_successes[aid]
                info_dict[aid]['episode_collisions'] = self.episode_collisions[aid]
        
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
        n = self.num_agents
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
        print(f'[GNNMARLEnv] 话题: /gnn_swarm/robot_{{0..{self.num_agents - 1}}}/state')

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
                for i in range(self.num_agents)
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
        max_neighbors = min(self.num_agents - 1, 5)

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
