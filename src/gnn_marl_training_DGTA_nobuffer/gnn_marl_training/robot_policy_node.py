#!/usr/bin/env python3
"""
TurtleBot3 集群 GNN-MAPPO 部署节点
每台机器人独立运行此节点，通过 ROS2 topic 互相广播状态。

设计原则（对齐训练侧 gnn_marl_env.py 的 decentralized 模式）：
  - 每帧从本地传感器构造 local_obs（默认155维：Top-K障碍特征历史 + target + vel + safety）
  - 订阅其他机器人发布的 /gnn_swarm/robot_X/state，构造 neighbor_obs
  - 接收端用时间戳缓存消息，模拟延迟/抖动查询（与训练对齐）
  - 策略推理在本地 CPU/GPU 执行，输出 cmd_vel

运行方式（在每台 TB3 上）：
  ros2 run gnn_marl_training robot_policy_node \
      --ros-args \
      -p robot_id:=0 \
      -p num_agents:=3 \
      -p checkpoint_path:=/path/to/checkpoint/policy.pt \
      -p communication_range:=5.0 \
      -p max_neighbors:=2
"""

import math
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray

from gnn_marl_training.global_planner import PathTrackingUtils
from gnn_marl_training.counterfactual_ppo_policy import CounterfactualPPOTorchPolicy  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# 消息格式约定（与 _encode_neighbor_states 对齐）
# Float32MultiArray.data = [robot_id, x, y, vx, vy, timestamp_sec]
# 共 6 floats
# ──────────────────────────────────────────────────────────────────────────────
STATE_MSG_LEN = 6
NEIGHBOR_FEAT_DIM = 5   # rel_x, rel_y, rel_vx, rel_vy, dist
SCAN_HISTORY_LEN = 4
OBSTACLE_POINT_FEAT_DIM = 4  # [x_rel_norm, y_rel_norm, dist_norm, valid_flag]
OBSTACLE_TOP_K_DEFAULT = 9
SAFETY_FEAT_DIM = 7
BASE_OBS_DIM = SCAN_HISTORY_LEN * (OBSTACLE_TOP_K_DEFAULT * OBSTACLE_POINT_FEAT_DIM) + 2 + 2 + SAFETY_FEAT_DIM


class NeighborStateBuffer:
    """
    存储从 ROS2 topic 收到的邻居状态消息，
    按时间戳排序，支持"取最近有效消息"（对应训练时的 latency 查询）。
    """

    def __init__(self, max_age_sec: float = 0.5, maxlen: int = 32):
        self._buf: deque = deque(maxlen=maxlen)
        self.max_age_sec = max_age_sec   # 超过此时间的消息视为过期
        self._lock = threading.Lock()

    def push(self, x: float, y: float, vx: float, vy: float, ts: float):
        with self._lock:
            self._buf.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'ts': ts})

    def get_latest_valid(self, current_time: float) -> Optional[Dict]:
        """返回最新且未过期的消息，无则返回 None（视为丢包）。"""
        with self._lock:
            for entry in reversed(self._buf):
                if current_time - entry['ts'] <= self.max_age_sec:
                    return entry
        return None  # 过期/空缓冲 = 丢包


class GATInferenceWrapper:
    """
    轻量推理包装器：从 RLlib checkpoint 导出的策略权重，
    仅执行前向传播（无 RLlib 依赖），适合在机器人端独立运行。

    支持两种加载方式：
      1. RLlib checkpoint 目录（自动提取 policy 权重）
      2. 纯 PyTorch .pt 文件（torch.save 导出）
    """

    def __init__(
        self,
        checkpoint_path: str,
        obs_dim: int,
        num_outputs: int = 2,
        hidden_dim: int = 128,
        gat_hidden_dim: int = 128,
        lstm_hidden_dim: int = 256,
        n_gat_heads: int = 4,
        max_neighbors: int = 1,
        device: str = 'cpu'
    ):
        self.device = torch.device(device)
        self.lstm_hidden_dim = lstm_hidden_dim
        self.obs_dim = obs_dim

        # 动态导入（仅在有 RLlib 的机器上可用，机器人端可用纯 .pt 方式）
        try:
            from gnn_marl_training.gat_rllib_model import GATRLlibModel
            from ray.rllib.models.catalog import ModelCatalog
            self._load_from_rllib(
                checkpoint_path, GATRLlibModel,
                obs_dim, num_outputs, hidden_dim,
                gat_hidden_dim, lstm_hidden_dim, n_gat_heads, max_neighbors
            )
        except ImportError:
            self._load_from_pt(checkpoint_path)

        self.model.eval()
        self.lstm_state: Optional[List[torch.Tensor]] = None

    def _load_from_rllib(
        self, checkpoint_path, ModelClass,
        obs_dim, num_outputs, hidden_dim,
        gat_hidden_dim, lstm_hidden_dim, n_gat_heads, max_neighbors
    ):
        """从 RLlib checkpoint 加载权重。"""
        import pickle
        import os

        # 尝试找到 policy 权重文件
        policy_dir = os.path.join(checkpoint_path, 'policies', 'shared_policy')
        weights_file = os.path.join(policy_dir, 'model_weights.pkl')

        # 构建虚拟 obs/action space 用于模型初始化
        from gymnasium import spaces
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.model = ModelClass(
            obs_space=obs_space,
            action_space=act_space,
            num_outputs=num_outputs,
            model_config={},
            name='deploy_model',
            num_agents=max_neighbors + 1,
            max_neighbors=max_neighbors,
            hidden_dim=hidden_dim,
            gat_hidden_dim=gat_hidden_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            n_gat_heads=n_gat_heads,
        ).to(self.device)

        if os.path.exists(weights_file):
            with open(weights_file, 'rb') as f:
                weights = pickle.load(f)
            self.model.load_state_dict(weights, strict=False)
        else:
            # 尝试直接加载 checkpoint（新版 RLlib）
            import ray
            ray.init(ignore_reinit_error=True)
            from ray.rllib.algorithms.ppo import PPO
            algo = PPO.from_checkpoint(checkpoint_path)
            weights = algo.get_policy('shared_policy').get_weights()
            ray.shutdown()
            state_dict = {k: torch.tensor(v) for k, v in weights.items()}
            self.model.load_state_dict(state_dict, strict=False)

    def _load_from_pt(self, pt_path: str):
        """从纯 PyTorch .pt 文件加载（torch.save 导出，机器人端推荐）。"""
        self.model = torch.load(pt_path, map_location=self.device)

    def reset_lstm(self):
        """新 episode 开始时重置 LSTM 隐状态。"""
        self.lstm_state = [
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32, device=self.device),
            torch.zeros(self.lstm_hidden_dim, dtype=torch.float32, device=self.device),
        ]

    @torch.no_grad()
    def infer(self, obs: np.ndarray) -> np.ndarray:
        """
        推理单帧观测。
        Args:
            obs: float32 array, shape (obs_dim,)
        Returns:
            action: float32 array, shape (2,)  ← [linear_norm, angular_norm] ∈ [-1,1]
        """
        if self.lstm_state is None:
            self.reset_lstm()

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # [1, obs_dim]
        state = self.lstm_state

        logits, new_state = self.model(
            {'obs_flat': obs_t, 'obs': obs_t},
            state,
            torch.tensor([1])
        )
        self.lstm_state = new_state

        # PPO 高斯策略：logits 即 action_mean，确定性动作
        action = logits.squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class RobotPolicyNode(Node):
    """
    TurtleBot3 单机策略执行节点。

    Topics（发布）：
      /tb3_{id}/cmd_vel                    ← 速度指令
      /gnn_swarm/robot_{id}/state          ← 广播自身状态给集群

    Topics（订阅）：
      /tb3_{id}/scan                       ← 激光雷达
      /tb3_{id}/odom                       ← 里程计
      /gnn_swarm/robot_{j}/state (j≠id)   ← 邻居状态

    参数（ros2 param）：
      robot_id            int     机器人编号（0-based）
      num_agents          int     集群规模
      checkpoint_path     str     模型权重路径
      communication_range float   通信范围(m)
      max_neighbors       int     最大考虑邻居数
      control_freq_hz     float   控制频率(Hz)，默认 10Hz
      goal_x / goal_y     float   目标坐标（简单定点导航）
      waypoint_file       str     路点列表文件路径（逐行 "x,y"）
    """

    def __init__(self):
        super().__init__('robot_policy_node')

        # ── 参数 ──────────────────────────────────────────────────────────
        self.declare_parameter('robot_id',            0)
        self.declare_parameter('num_agents',          2)
        self.declare_parameter('checkpoint_path',     '')
        self.declare_parameter('communication_range', 5.0)
        self.declare_parameter('max_neighbors',       1)
        self.declare_parameter('control_freq_hz',     10.0)
        self.declare_parameter('goal_x',              5.0)
        self.declare_parameter('goal_y',              5.0)
        self.declare_parameter('waypoint_file',       '')
        self.declare_parameter('msg_max_age_sec',     0.3)   # 消息过期阈值
        self.declare_parameter('rolling_lookahead_dist', 0.8)
        self.declare_parameter('obs_target_dist_clip',   6.0)
        self.declare_parameter('obs_target_filter_alpha', 0.35)
        self.declare_parameter('obs_target_max_step',    0.45)
        self.declare_parameter('obstacle_filter_range',  2.0)
        self.declare_parameter('obstacle_filter_fov_deg', 360.0)
        self.declare_parameter('obstacle_top_k',         OBSTACLE_TOP_K_DEFAULT)
        self.declare_parameter('shield_enable',          True)
        self.declare_parameter('shield_front_slow_dist', 0.50)
        self.declare_parameter('shield_front_stop_dist', 0.20)
        self.declare_parameter('shield_neighbor_slow_dist', 0.35)
        self.declare_parameter('shield_linear_slow',     0.15)
        self.declare_parameter('shield_linear_stop',     0.05)
        self.declare_parameter('shield_turn_bias',       0.35)
        self.declare_parameter('turn_in_place_front_dist', 0.35)
        self.declare_parameter('turn_in_place_angle_thresh', 0.45)
        self.declare_parameter('turn_in_place_w', 0.90)
        self.declare_parameter('yielding_enable', True)
        self.declare_parameter('yielding_soft_dist', 0.90)
        self.declare_parameter('yielding_stop_dist', 0.50)
        self.declare_parameter('yielding_hard_stop_dist', 0.30)
        self.declare_parameter('yielding_ttc', 2.4)
        self.declare_parameter('yielding_commit_steps', 5)
        self.declare_parameter('subgoal_block_front_dist', 0.55)
        self.declare_parameter('subgoal_min_side_clearance', 0.20)
        self.declare_parameter('subgoal_detour_forward_gain', 0.55)
        self.declare_parameter('subgoal_detour_lateral_gain', 1.10)
        self.declare_parameter('subgoal_detour_hold_steps', 12)
        self.declare_parameter('subgoal_deadlock_front_dist', 0.60)
        self.declare_parameter('subgoal_deadlock_speed_thresh', 0.03)
        self.declare_parameter('subgoal_deadlock_steps', 6)
        self.declare_parameter('replan_on_deadlock', True)
        self.declare_parameter('replan_cooldown_steps', 25)
        self.declare_parameter('dynamic_replan_neighbor_dist', 1.8)
        self.declare_parameter('dynamic_replan_ttc', 2.6)
        self.declare_parameter('dynamic_replan_block_radius', 0.55)

        self.robot_id   = self.get_parameter('robot_id').value
        self.num_agents = self.get_parameter('num_agents').value
        self.comm_range = self.get_parameter('communication_range').value
        self.max_neighbors = min(
            self.get_parameter('max_neighbors').value,
            self.num_agents - 1,
            5
        )
        self.control_freq = self.get_parameter('control_freq_hz').value
        self.msg_max_age  = self.get_parameter('msg_max_age_sec').value
        self.lookahead_dist = float(self.get_parameter('rolling_lookahead_dist').value)
        self.obs_target_dist_clip = max(0.5, float(self.get_parameter('obs_target_dist_clip').value))
        self.obs_target_filter_alpha = float(np.clip(float(self.get_parameter('obs_target_filter_alpha').value), 0.0, 1.0))
        self.obs_target_max_step = max(0.05, float(self.get_parameter('obs_target_max_step').value))
        self.obstacle_filter_range = float(self.get_parameter('obstacle_filter_range').value)
        self.obstacle_filter_fov_deg = float(self.get_parameter('obstacle_filter_fov_deg').value)
        self.obstacle_top_k = int(self.get_parameter('obstacle_top_k').value)
        self.shield_enable = bool(self.get_parameter('shield_enable').value)
        self.shield_front_slow_dist = float(self.get_parameter('shield_front_slow_dist').value)
        self.shield_front_stop_dist = float(self.get_parameter('shield_front_stop_dist').value)
        self.shield_neighbor_slow_dist = float(self.get_parameter('shield_neighbor_slow_dist').value)
        self.shield_linear_slow = float(self.get_parameter('shield_linear_slow').value)
        self.shield_linear_stop = float(self.get_parameter('shield_linear_stop').value)
        self.shield_turn_bias = float(self.get_parameter('shield_turn_bias').value)
        self.turn_in_place_front_dist = float(self.get_parameter('turn_in_place_front_dist').value)
        self.turn_in_place_angle_thresh = float(self.get_parameter('turn_in_place_angle_thresh').value)
        self.turn_in_place_w = float(self.get_parameter('turn_in_place_w').value)
        self.yielding_enable = bool(self.get_parameter('yielding_enable').value)
        self.yielding_soft_dist = max(0.2, float(self.get_parameter('yielding_soft_dist').value))
        self.yielding_stop_dist = max(0.1, min(self.yielding_soft_dist, float(self.get_parameter('yielding_stop_dist').value)))
        self.yielding_hard_stop_dist = max(0.05, min(self.yielding_stop_dist, float(self.get_parameter('yielding_hard_stop_dist').value)))
        self.yielding_ttc = max(0.5, float(self.get_parameter('yielding_ttc').value))
        self.yielding_commit_steps = int(max(1, self.get_parameter('yielding_commit_steps').value))
        self.subgoal_block_front_dist = max(0.18, float(self.get_parameter('subgoal_block_front_dist').value))
        self.subgoal_min_side_clearance = max(0.10, float(self.get_parameter('subgoal_min_side_clearance').value))
        self.subgoal_detour_forward_gain = float(np.clip(float(self.get_parameter('subgoal_detour_forward_gain').value), 0.20, 1.20))
        self.subgoal_detour_lateral_gain = float(np.clip(float(self.get_parameter('subgoal_detour_lateral_gain').value), 0.20, 1.50))
        self.subgoal_detour_hold_steps = int(max(0, self.get_parameter('subgoal_detour_hold_steps').value))
        self.subgoal_deadlock_front_dist = max(0.20, float(self.get_parameter('subgoal_deadlock_front_dist').value))
        self.subgoal_deadlock_speed_thresh = max(0.0, float(self.get_parameter('subgoal_deadlock_speed_thresh').value))
        self.subgoal_deadlock_steps = int(max(1, self.get_parameter('subgoal_deadlock_steps').value))
        self.replan_on_deadlock = bool(self.get_parameter('replan_on_deadlock').value)
        self.replan_cooldown_steps = int(max(1, self.get_parameter('replan_cooldown_steps').value))
        self.dynamic_replan_neighbor_dist = max(0.5, float(self.get_parameter('dynamic_replan_neighbor_dist').value))
        self.dynamic_replan_ttc = max(0.5, float(self.get_parameter('dynamic_replan_ttc').value))
        self.dynamic_replan_block_radius = max(0.10, float(self.get_parameter('dynamic_replan_block_radius').value))
        self.control_dt = 1.0 / max(float(self.control_freq), 1e-6)

        self.ns = f'/tb3_{self.robot_id}'
        self.get_logger().info(
            f'🤖 PolicyNode | id={self.robot_id} | ns={self.ns} '
            f'| num_agents={self.num_agents} | max_neighbors={self.max_neighbors}'
        )

        # ── 传感器状态 ────────────────────────────────────────────────────
        self.latest_scan:     Optional[LaserScan] = None
        self.scan_max_range = 3.5
        self.scan_valid_min = 0.10
        self.scan_history_len = SCAN_HISTORY_LEN
        self.obstacle_point_feature_dim = OBSTACLE_POINT_FEAT_DIM
        self.obstacle_top_k = int(np.clip(int(self.obstacle_top_k), 1, 64))
        self.obstacle_filter_range = float(np.clip(float(self.obstacle_filter_range), 0.2, self.scan_max_range))
        self.obstacle_filter_fov_deg = float(np.clip(float(self.obstacle_filter_fov_deg), 10.0, 360.0))
        self.scan_dim = self.obstacle_top_k * self.obstacle_point_feature_dim
        self._scan_history: deque = deque(maxlen=self.scan_history_len)
        self.odom_x:          float = 0.0
        self.odom_y:          float = 0.0
        self.yaw:             float = 0.0
        self.linear_x:        float = 0.0
        self.angular_z:       float = 0.0
        self._sensor_lock     = threading.Lock()

        # ── 目标航点 ──────────────────────────────────────────────────────
        self.waypoints: List[Tuple[float, float]] = self._load_waypoints()
        self.wp_idx:    int = 0
        self.goal_threshold = 0.3   # 最终目标到达判定距离(m)
        self.current_subgoal: Optional[Tuple[float, float]] = None
        self._obs_target_state: Optional[np.ndarray] = None
        self.current_step = 0
        self._goal_reached_logged = False
        self._yield_hold_steps = 0
        self._yield_partner = ''
        self._yield_turn_sign = 0.0
        self._subgoal_detour_hold = 0
        self._subgoal_detour_side = 0
        self._subgoal_deadlock_streak = 0
        self._next_replan_step = 0
        self._replan_hold_steps = 0
        self._replan_target: Optional[Tuple[float, float]] = None
        self._last_subgoal_mode = 'nominal'
        self._last_conflict_state: Optional[Dict[str, float]] = None

        # ── 邻居状态缓冲 ──────────────────────────────────────────────────
        # key: robot_id(int), value: NeighborStateBuffer
        self._neighbor_bufs: Dict[int, NeighborStateBuffer] = {
            j: NeighborStateBuffer(max_age_sec=self.msg_max_age)
            for j in range(self.num_agents) if j != self.robot_id
        }

        # ── obs_dim ───────────────────────────────────────────────────────
        self.safety_feature_dim = SAFETY_FEAT_DIM
        self.base_obs_dim = self.scan_dim * self.scan_history_len + 2 + 2 + self.safety_feature_dim
        self.obs_dim = self.base_obs_dim + NEIGHBOR_FEAT_DIM * self.max_neighbors

        # ── ROS2 QoS ──────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        comm_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(
            Twist, f'{self.ns}/cmd_vel', 10
        )
        self.state_pub = self.create_publisher(
            Float32MultiArray,
            f'/gnn_swarm/robot_{self.robot_id}/state',
            comm_qos
        )

        # ── Subscribers ───────────────────────────────────────────────────
        self.create_subscription(
            LaserScan, f'{self.ns}/scan',
            self._scan_cb, sensor_qos
        )
        self.create_subscription(
            Odometry, f'{self.ns}/odom',
            self._odom_cb, sensor_qos
        )
        for j in range(self.num_agents):
            if j != self.robot_id:
                self.create_subscription(
                    Float32MultiArray,
                    f'/gnn_swarm/robot_{j}/state',
                    lambda msg, jj=j: self._neighbor_state_cb(msg, jj),
                    comm_qos
                )

        # ── 模型加载 ──────────────────────────────────────────────────────
        checkpoint_path = self.get_parameter('checkpoint_path').value
        self._policy: Optional[GATInferenceWrapper] = None
        if checkpoint_path:
            self.get_logger().info(f'📥 加载模型: {checkpoint_path}')
            try:
                self._policy = GATInferenceWrapper(
                    checkpoint_path=checkpoint_path,
                    obs_dim=self.obs_dim,
                    num_outputs=2,
                    max_neighbors=self.max_neighbors,
                )
                self._policy.reset_lstm()
                self.get_logger().info('✅ 模型加载成功')
            except Exception as e:
                self.get_logger().error(f'❌ 模型加载失败: {e}')
        else:
            self.get_logger().warn('⚠️  未提供 checkpoint_path，将使用随机动作')

        # ── 控制定时器 ────────────────────────────────────────────────────
        period = 1.0 / self.control_freq
        self.create_timer(period, self._control_loop)

        # ── 广播定时器（与控制频率相同）──────────────────────────────────
        self.create_timer(period, self._broadcast_state)

        self.get_logger().info(
            f'⚙️  控制频率 {self.control_freq:.1f} Hz | 观测维度 {self.obs_dim} '
            f'| obstacle_top_k={self.obstacle_top_k} | avoid_range={self.obstacle_filter_range:.2f}m'
        )

    # ── 传感器回调 ─────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        with self._sensor_lock:
            self.latest_scan = msg

    def _odom_cb(self, msg: Odometry):
        with self._sensor_lock:
            self.odom_x   = msg.pose.pose.position.x
            self.odom_y   = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.yaw      = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            self.linear_x  = msg.twist.twist.linear.x
            self.angular_z = msg.twist.twist.angular.z

    def _neighbor_state_cb(self, msg: Float32MultiArray, sender_id: int):
        """接收邻居广播的状态消息并存入缓冲。"""
        if len(msg.data) < STATE_MSG_LEN:
            return
        # data = [robot_id, x, y, vx, vy, timestamp_sec]
        _, x, y, vx, vy, ts = msg.data[:STATE_MSG_LEN]
        if sender_id in self._neighbor_bufs:
            self._neighbor_bufs[sender_id].push(x, y, vx, vy, ts)

    # ── 状态广播 ───────────────────────────────────────────────────────────

    def _broadcast_state(self):
        """向集群广播自身当前状态（位置 + 全局速度）。"""
        with self._sensor_lock:
            x  = self.odom_x
            y  = self.odom_y
            vx = self.linear_x * math.cos(self.yaw)
            vy = self.linear_x * math.sin(self.yaw)

        msg = Float32MultiArray()
        msg.data = [
            float(self.robot_id),
            float(x), float(y),
            float(vx), float(vy),
            float(self.get_clock().now().nanoseconds * 1e-9)
        ]
        self.state_pub.publish(msg)

    # ── 观测构造 ───────────────────────────────────────────────────────────

    def _extract_filtered_scan_features(self, scan: LaserScan) -> np.ndarray:
        feat = np.zeros((self.obstacle_top_k, self.obstacle_point_feature_dim), dtype=np.float32)
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.scan_max_range, posinf=self.scan_max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.scan_max_range)

        n = len(ranges)
        if n <= 0:
            return feat.reshape(-1)

        angle_min = float(getattr(scan, 'angle_min', -math.pi))
        angle_inc = float(getattr(scan, 'angle_increment', 0.0))
        if not math.isfinite(angle_min):
            angle_min = -math.pi
        if (not math.isfinite(angle_inc)) or abs(angle_inc) <= 1e-6:
            angle_inc = (2.0 * math.pi) / max(1, n)

        idx = np.arange(n, dtype=np.float32)
        angles = angle_min + idx * angle_inc
        angles = (angles + np.pi) % (2.0 * np.pi) - np.pi

        valid = (ranges > self.scan_valid_min) & (ranges <= self.obstacle_filter_range)
        if self.obstacle_filter_fov_deg < 359.9:
            half_fov = math.radians(self.obstacle_filter_fov_deg) * 0.5
            valid &= (np.abs(angles) <= half_fov)

        valid_idx = np.where(valid)[0]
        if valid_idx.size == 0:
            return feat.reshape(-1)

        ordered = valid_idx[np.argsort(ranges[valid_idx])]
        picked = ordered[:self.obstacle_top_k]

        d = ranges[picked]
        theta = angles[picked]
        denom = max(self.obstacle_filter_range, 1e-6)
        count = int(picked.size)

        feat[:count, 0] = (d * np.cos(theta)) / denom
        feat[:count, 1] = (d * np.sin(theta)) / denom
        feat[:count, 2] = d / denom
        feat[:count, 3] = 1.0
        return feat.reshape(-1)

    def _build_local_obs(self, target_override: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
        """
        构造本机局部观测（默认155维），对齐训练侧：
        Top-K障碍特征历史 + target(2) + vel(2) + safety(7)
        """
        with self._sensor_lock:
            scan   = self.latest_scan
            x, y   = self.odom_x, self.odom_y
            yaw    = self.yaw
            vel_x  = self.linear_x
            vel_w  = self.angular_z

        if scan is None:
            return None

        scan_feat = self._extract_filtered_scan_features(scan)

        self._scan_history.append(scan_feat.copy())
        history = list(self._scan_history)
        while len(history) < self.scan_history_len:
            history.insert(0, history[0].copy())
        stacked_scan = np.concatenate(history)

        raw_target = target_override if target_override is not None else self._get_tracking_target()
        if raw_target is None:
            target_x_body = 0.0
            target_y_body = 0.0
        else:
            raw_xy = np.array(raw_target, dtype=np.float32)
            if self._obs_target_state is None:
                self._obs_target_state = raw_xy.copy()
            else:
                prev_xy = self._obs_target_state
                delta = raw_xy - prev_xy
                step = float(np.linalg.norm(delta))
                if step > self.obs_target_max_step:
                    delta = delta / (step + 1e-8) * self.obs_target_max_step
                candidate = prev_xy + delta
                alpha = self.obs_target_filter_alpha
                self._obs_target_state = (1.0 - alpha) * prev_xy + alpha * candidate

            sx, sy = float(self._obs_target_state[0]), float(self._obs_target_state[1])
            tgt_dist = math.hypot(sx - x, sy - y)
            abs_angle = math.atan2(sy - y, sx - x)
            rel_angle = (abs_angle - yaw + math.pi) % (2 * math.pi) - math.pi
            dist_norm = float(np.clip(tgt_dist / self.obs_target_dist_clip, 0.0, 1.0))
            target_x_body = dist_norm * math.cos(rel_angle)
            target_y_body = dist_norm * math.sin(rel_angle)

        metrics = self._scan_sector_metrics() or {
            'min_dist': self.scan_max_range,
            'front_min': self.scan_max_range,
            'left_min': self.scan_max_range,
            'right_min': self.scan_max_range,
        }
        min_dist = float(metrics['min_dist'])
        front_min = float(metrics['front_min'])
        left_min = float(metrics['left_min'])
        right_min = float(metrics['right_min'])
        left_right_diff = left_min - right_min
        front_blocked = 1.0 if front_min < self.turn_in_place_front_dist else 0.0
        front_risk = max(0.0, 0.5 - front_min) * max(0.0, vel_x)

        obs = np.concatenate([
            stacked_scan,
            [target_x_body, target_y_body],
            [vel_x, vel_w],
            [min_dist, front_min, left_min, right_min,
             left_right_diff, front_blocked, front_risk],
        ]).astype(np.float32)

        return obs

    def _build_neighbor_obs(self) -> np.ndarray:
        """
        构造邻居特征（max_neighbors × 5 维）。
        使用接收缓冲中的最新有效消息，对应训练时的 decentralized 模式。
        """
        my_state = self._get_robot_state()
        valid_neighbors = self._get_valid_neighbor_entries(my_state)

        valid_neighbors.sort(key=lambda v: v[0])

        features: List[np.ndarray] = []
        for k in range(self.max_neighbors):
            if k < len(valid_neighbors):
                dist, nx, ny, nvx, nvy, _ = valid_neighbors[k]
                feat = np.array([
                    nx - my_state['x'], ny - my_state['y'],         # 相对位置
                    nvx - my_state['vx'], nvy - my_state['vy'],     # 相对速度（全局系）
                    dist
                ], dtype=np.float32)
            else:
                feat = np.zeros(NEIGHBOR_FEAT_DIM, dtype=np.float32)  # 槽位填零
            features.append(feat)

        return np.concatenate(features).astype(np.float32)

    def _get_robot_state(self) -> Dict[str, float]:
        with self._sensor_lock:
            yaw = float(self.yaw)
            linear_x = float(self.linear_x)
            return {
                'x': float(self.odom_x),
                'y': float(self.odom_y),
                'yaw': yaw,
                'linear_x': linear_x,
                'angular_z': float(self.angular_z),
                'vx': float(linear_x * math.cos(yaw)),
                'vy': float(linear_x * math.sin(yaw)),
            }

    def _wrap_angle(self, angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _world_to_body(self, vec_xy: np.ndarray, yaw: Optional[float] = None) -> np.ndarray:
        if yaw is None:
            yaw = self._get_robot_state()['yaw']
        c = math.cos(float(yaw))
        s = math.sin(float(yaw))
        return np.array([
            c * float(vec_xy[0]) + s * float(vec_xy[1]),
            -s * float(vec_xy[0]) + c * float(vec_xy[1]),
        ], dtype=np.float32)

    def _body_to_world_point(
        self,
        x_body: float,
        y_body: float,
        state: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, float]:
        state = state or self._get_robot_state()
        c = math.cos(float(state['yaw']))
        s = math.sin(float(state['yaw']))
        xw = float(state['x']) + c * float(x_body) - s * float(y_body)
        yw = float(state['y']) + s * float(x_body) + c * float(y_body)
        return (xw, yw)

    def _agent_rank(self, agent_id: int) -> int:
        try:
            return int(agent_id)
        except Exception:
            return int(self.robot_id)

    def _get_valid_neighbor_entries(
        self,
        my_state: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[float, float, float, float, float, float]]:
        my_state = my_state or self._get_robot_state()
        now = self.get_clock().now().nanoseconds * 1e-9
        valid_neighbors: List[Tuple[float, float, float, float, float, float]] = []
        for neighbor_id, buf in self._neighbor_bufs.items():
            entry = buf.get_latest_valid(now)
            if entry is None:
                continue
            nx = float(entry['x'])
            ny = float(entry['y'])
            nvx = float(entry['vx'])
            nvy = float(entry['vy'])
            dist = math.hypot(nx - my_state['x'], ny - my_state['y'])
            if dist <= self.comm_range:
                valid_neighbors.append((dist, nx, ny, nvx, nvy, float(neighbor_id)))
        valid_neighbors.sort(key=lambda item: item[0])
        return valid_neighbors

    # ── 航点管理 ───────────────────────────────────────────────────────────

    def _load_waypoints(self) -> List[Tuple[float, float]]:
        wf = self.get_parameter('waypoint_file').value
        if wf:
            try:
                wps = []
                with open(wf) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            x, y = line.split(',')
                            wps.append((float(x), float(y)))
                self.get_logger().info(f'📍 加载 {len(wps)} 个航点')
                return wps
            except Exception as e:
                self.get_logger().warn(f'航点文件加载失败: {e}')
        # 回退到单目标点
        gx = self.get_parameter('goal_x').value
        gy = self.get_parameter('goal_y').value
        return [(gx, gy)]

    def _get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        if not self.waypoints:
            return None
        return self.waypoints[-1]

    def _get_nominal_subgoal(self, lookahead_dist: Optional[float] = None) -> Optional[Tuple[float, float]]:
        if not self.waypoints:
            return None
        state = self._get_robot_state()
        if lookahead_dist is None:
            lookahead_dist = self.lookahead_dist
        info = PathTrackingUtils.get_rolling_subgoal(
            (state['x'], state['y']),
            self.waypoints,
            float(lookahead_dist),
        )
        return tuple(info['subgoal'])

    def _update_waypoint(self):
        """检查最终目标是否到达，避免规则状态机重复触发输出。"""
        final_goal = self._get_current_waypoint()
        if final_goal is None:
            return
        state = self._get_robot_state()
        dist_to_goal = math.hypot(final_goal[0] - state['x'], final_goal[1] - state['y'])
        if dist_to_goal < self.goal_threshold and not self._goal_reached_logged:
            self.get_logger().info('✅ 接近最终目标点')
            self._goal_reached_logged = True
        elif dist_to_goal >= self.goal_threshold:
            self._goal_reached_logged = False

    def _get_target_angle(
        self,
        target: Tuple[float, float],
        state: Optional[Dict[str, float]] = None,
    ) -> float:
        state = state or self._get_robot_state()
        tgt_angle = math.atan2(target[1] - state['y'], target[0] - state['x'])
        return self._wrap_angle(tgt_angle - state['yaw'])

    def _get_adaptive_lookahead(self, front_min: float, heading_error: float) -> float:
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

    def _get_current_sector_dists(self) -> np.ndarray:
        with self._sensor_lock:
            scan = self.latest_scan
        if scan is None or not getattr(scan, 'ranges', None):
            return np.full(self.obstacle_top_k, self.scan_max_range, dtype=np.float32)
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=self.scan_max_range,
            posinf=self.scan_max_range,
            neginf=0.0,
        )
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        return self._compute_front_sector_min_dists(ranges)

    def _get_dynamic_blocked_bearings(
        self,
        my_state: Dict[str, float],
        neighbors: List[Tuple[float, float, float, float, float, float]],
    ) -> List[Dict[str, float]]:
        blocked = []
        my_vel = np.array([my_state['vx'], my_state['vy']], dtype=np.float32)
        predict_h = min(self.dynamic_replan_ttc, 0.8)

        for dist, nx, ny, nvx, nvy, neighbor_id in neighbors:
            rel = np.array([nx - my_state['x'], ny - my_state['y']], dtype=np.float32)
            if dist < 1e-6:
                continue

            rel_body = self._world_to_body(rel, yaw=my_state['yaw'])
            if float(rel_body[0]) < -0.20:
                continue

            neighbor_vel = np.array([nvx, nvy], dtype=np.float32)
            rel_unit = rel / max(dist, 1e-6)
            closing_speed = float(-np.dot(neighbor_vel - my_vel, rel_unit))
            ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')
            if dist > self.dynamic_replan_neighbor_dist and (not math.isfinite(ttc) or ttc >= self.dynamic_replan_ttc):
                continue

            severity = max(
                float(np.clip((self.dynamic_replan_neighbor_dist - dist) / max(self.dynamic_replan_neighbor_dist, 1e-6), 0.0, 1.0)),
                float(np.clip((self.dynamic_replan_ttc - ttc) / max(self.dynamic_replan_ttc, 1e-6), 0.0, 1.0)) if math.isfinite(ttc) else 0.0,
            )
            rel_pred = rel + (neighbor_vel - my_vel) * predict_h

            for rel_point in (rel, rel_pred):
                body = self._world_to_body(rel_point, yaw=my_state['yaw'])
                body_dist = float(np.linalg.norm(body))
                if body_dist < 1e-6 or float(body[0]) < -0.20:
                    continue
                blocked.append({
                    'angle': float(math.atan2(body[1], body[0])),
                    'span': float(max(math.atan2(self.dynamic_replan_block_radius, max(body_dist, 1e-3)), math.radians(8.0))),
                    'weight': float(np.clip(0.45 + 0.55 * severity, 0.0, 1.0)),
                    'neighbor_id': float(neighbor_id),
                })

        return blocked

    def _compute_gap_metrics(
        self,
        sector_dists: np.ndarray,
        nominal_target_angle: float = 0.0,
        blocked_bearings: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        sector_arr = np.asarray(sector_dists, dtype=np.float32)
        if sector_arr.size <= 0:
            return {
                'best_gap_angle': 0.0,
                'best_gap_width': 0.0,
                'best_gap_clearance': 0.0,
                'best_gap_score': 0.0,
                'best_sector_idx': -1,
            }

        blocked_bearings = blocked_bearings or []
        open_thresh = max(self.subgoal_min_side_clearance, min(self.dynamic_replan_block_radius, 0.45))
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

            blocked_penalty = 0.0
            for block in blocked_bearings:
                delta = abs(self._wrap_angle(angle - float(block['angle'])))
                span = max(float(block['span']), math.radians(8.0))
                if delta <= span:
                    overlap = 1.0 - delta / span
                    blocked_penalty = max(blocked_penalty, float(block['weight']) * overlap)

            score = (
                0.50 * clearance_norm
                + 0.25 * width_norm
                + 0.15 * heading_align
                + 0.10 * forwardness
                - 0.35 * blocked_penalty
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

    def _get_head_on_conflict_state(
        self,
        my_state: Dict[str, float],
        neighbors: List[Tuple[float, float, float, float, float, float]],
    ) -> Optional[Dict[str, float]]:
        if not self.yielding_enable:
            return None

        my_pos = np.array([my_state['x'], my_state['y']], dtype=np.float32)
        my_vel = np.array([my_state['vx'], my_state['vy']], dtype=np.float32)
        my_forward = np.array([math.cos(my_state['yaw']), math.sin(my_state['yaw'])], dtype=np.float32)
        best = None

        for dist, nx, ny, nvx, nvy, neighbor_id in neighbors:
            if dist > self.dynamic_replan_neighbor_dist or dist < 1e-6:
                continue

            rel = np.array([nx, ny], dtype=np.float32) - my_pos
            body_rel = self._world_to_body(rel, yaw=my_state['yaw'])
            if float(body_rel[0]) < -0.15:
                continue

            rel_unit = rel / max(dist, 1e-6)
            neighbor_vel = np.array([nvx, nvy], dtype=np.float32)
            rel_vel = neighbor_vel - my_vel
            closing_speed = float(-np.dot(rel_vel, rel_unit))
            ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float('inf')
            bearing = float(math.atan2(rel[1], rel[0]))
            yaw_err = self._wrap_angle(bearing - my_state['yaw'])
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
            proximity_risk = float(
                np.clip((self.yielding_soft_dist - dist) / max(self.yielding_soft_dist, 1e-6), 0.0, 1.0)
            )
            severity = max(ttc_risk, proximity_risk)
            cand = {
                'partner': float(neighbor_id),
                'dist': float(dist),
                'closing_speed': float(max(closing_speed, 0.0)),
                'ttc': float(ttc),
                'turn_sign': float(turn_sign),
                'severity': float(severity),
                'should_yield': 1.0 if self._agent_rank(self.robot_id) > self._agent_rank(int(neighbor_id)) else 0.0,
            }
            if best is None or cand['severity'] > best['severity']:
                best = cand

        return best

    def _try_local_replan_due_to_deadlock(
        self,
        nominal_target_angle: float,
        adaptive_lookahead: float,
        sector_dists: np.ndarray,
        blocked_bearings: List[Dict[str, float]],
    ) -> Optional[Tuple[float, float]]:
        if not self.replan_on_deadlock:
            return None
        if self._replan_target is not None and self._replan_hold_steps > 0:
            self._replan_hold_steps -= 1
            return self._replan_target
        if self.current_step < self._next_replan_step:
            return None

        gap = self._compute_gap_metrics(
            sector_dists,
            nominal_target_angle=nominal_target_angle,
            blocked_bearings=blocked_bearings,
        )
        self._next_replan_step = self.current_step + self.replan_cooldown_steps
        if gap.get('best_sector_idx', -1) < 0 or float(gap.get('best_gap_score', 0.0)) <= 0.12:
            return None

        angle = float(gap['best_gap_angle'])
        lookahead = max(0.35, float(adaptive_lookahead))
        radial = lookahead * (0.85 + 0.55 * max(float(gap['best_gap_clearance']), float(gap['best_gap_width'])))
        radial = float(np.clip(radial, 0.45, max(self.lookahead_dist * 1.8, 1.4)))
        target = self._body_to_world_point(radial * math.cos(angle), radial * math.sin(angle))
        self._replan_target = target
        self._replan_hold_steps = max(1, self.subgoal_detour_hold_steps)
        self._subgoal_deadlock_streak = 0
        return target

    def _select_local_detour_subgoal(
        self,
        nominal_subgoal: Tuple[float, float],
        adaptive_lookahead: float,
        front_min: float,
        left_min: float,
        right_min: float,
        sector_dists: np.ndarray,
    ) -> Tuple[Tuple[float, float], str]:
        my_state = self._get_robot_state()
        pos = np.array([my_state['x'], my_state['y']], dtype=np.float32)
        nominal = np.asarray(nominal_subgoal, dtype=np.float32)
        rel_nominal = nominal - pos
        rel_body = self._world_to_body(rel_nominal, yaw=my_state['yaw'])
        nominal_target_angle = float(self._get_target_angle(nominal_subgoal, state=my_state))

        forward_speed = abs(float(my_state['linear_x']))
        if front_min < self.subgoal_deadlock_front_dist and forward_speed < self.subgoal_deadlock_speed_thresh:
            self._subgoal_deadlock_streak += 1
        else:
            self._subgoal_deadlock_streak = max(0, self._subgoal_deadlock_streak - 1)

        blocked = (front_min < self.subgoal_block_front_dist) and (float(rel_body[0]) > 0.12)
        force_detour = self._subgoal_deadlock_streak >= self.subgoal_deadlock_steps
        neighbors = self._get_valid_neighbor_entries(my_state)
        blocked_bearings = self._get_dynamic_blocked_bearings(my_state, neighbors)

        if force_detour:
            replan_target = self._try_local_replan_due_to_deadlock(
                nominal_target_angle,
                adaptive_lookahead,
                sector_dists,
                blocked_bearings,
            )
            if replan_target is not None:
                return tuple(replan_target), 'replan'

        conflict = self._get_head_on_conflict_state(my_state, neighbors)
        self._last_conflict_state = conflict
        if conflict is not None:
            same_partner = str(int(conflict['partner'])) == str(self._yield_partner)
            should_hold = (
                self._yield_hold_steps > 0
                and same_partner
                and float(conflict['dist']) < (self.yielding_soft_dist + 0.30)
            )
            if should_hold:
                self._yield_hold_steps = max(0, self._yield_hold_steps - 1)
                lookahead = max(0.25, float(adaptive_lookahead))
                x_body = max(0.08, 0.35 * lookahead)
                y_body = self._yield_turn_sign * max(0.18, 0.80 * lookahead)
                return self._body_to_world_point(x_body, y_body, state=my_state), 'yield'

            if (
                float(conflict['should_yield']) > 0.5
                and float(conflict['dist']) < self.yielding_soft_dist
                and (not math.isfinite(float(conflict['ttc'])) or float(conflict['ttc']) < self.yielding_ttc)
            ):
                self._yield_partner = str(int(conflict['partner']))
                self._yield_turn_sign = float(conflict['turn_sign'])
                self._yield_hold_steps = self.yielding_commit_steps
                lookahead = max(0.25, float(adaptive_lookahead))
                x_body = max(0.06, 0.30 * lookahead)
                y_body = self._yield_turn_sign * max(0.18, 0.90 * lookahead)
                return self._body_to_world_point(x_body, y_body, state=my_state), 'yield'
        else:
            self._yield_hold_steps = 0
            self._yield_partner = ''
            self._yield_turn_sign = 0.0

        if not blocked and not force_detour:
            self._subgoal_detour_hold = max(0, self._subgoal_detour_hold - 1)
            if self._subgoal_detour_hold == 0:
                self._subgoal_detour_side = 0
                if self._replan_hold_steps <= 0:
                    self._replan_target = None
            return nominal_subgoal, 'nominal'

        gap = self._compute_gap_metrics(
            sector_dists,
            nominal_target_angle=nominal_target_angle,
            blocked_bearings=blocked_bearings,
        )
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
                cand = self._body_to_world_point(x_body, y_body, state=my_state)
                if abs(self._get_target_angle(cand, state=my_state)) <= 1.52:
                    self._subgoal_detour_side = 1 if y_body >= 0.0 else -1
                    self._subgoal_detour_hold = self.subgoal_detour_hold_steps
                    return cand, 'gap_detour'

        if self._subgoal_detour_hold > 0 and self._subgoal_detour_side != 0:
            preferred_side = self._subgoal_detour_side
        else:
            preferred_side = 1 if left_min >= right_min else -1

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
            cand = self._body_to_world_point(forward_step, side * lateral_step, state=my_state)
            if abs(self._get_target_angle(cand, state=my_state)) > 1.45:
                continue
            self._subgoal_detour_side = int(side)
            self._subgoal_detour_hold = self.subgoal_detour_hold_steps
            return cand, 'detour'

        return nominal_subgoal, ('deadlock' if force_detour else 'blocked_nominal')

    def _get_tracking_target(self) -> Optional[Tuple[float, float]]:
        if not self.waypoints:
            return None

        state = self._get_robot_state()
        pos = (state['x'], state['y'])

        try:
            if self.lookahead_dist <= 0.0:
                self.current_subgoal = tuple(self.waypoints[-1])
                self._last_subgoal_mode = 'goal_only'
                return self.current_subgoal

            base = PathTrackingUtils.get_rolling_subgoal(pos, self.waypoints, self.lookahead_dist)
            heading_error = self._get_target_angle(tuple(base['subgoal']), state=state)
            sectors = self._scan_sector_metrics() or {
                'front_min': self.scan_max_range,
                'left_min': self.scan_max_range,
                'right_min': self.scan_max_range,
            }
            adaptive = self._get_adaptive_lookahead(float(sectors['front_min']), heading_error)
            info = PathTrackingUtils.get_rolling_subgoal(pos, self.waypoints, adaptive)
            chosen_subgoal, mode = self._select_local_detour_subgoal(
                tuple(info['subgoal']),
                adaptive,
                float(sectors['front_min']),
                float(sectors['left_min']),
                float(sectors['right_min']),
                self._get_current_sector_dists(),
            )
            self.current_subgoal = tuple(chosen_subgoal)
            self._last_subgoal_mode = mode
            return self.current_subgoal
        except Exception as exc:
            self.get_logger().warn(f'tracking_target fallback: {exc}')
            self.current_subgoal = tuple(self.waypoints[-1])
            self._last_subgoal_mode = 'tracking_fallback'
            return self.current_subgoal

    def _scan_sector_metrics(self) -> Optional[Dict[str, float]]:
        with self._sensor_lock:
            scan = self.latest_scan
        if scan is None:
            return None
        ranges = np.array(scan.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=self.scan_max_range, posinf=self.scan_max_range, neginf=0.0)
        ranges = np.clip(ranges, 0.0, self.scan_max_range)
        n = len(ranges)
        if n == 0:
            return {
                'min_dist': self.scan_max_range,
                'front_min': self.scan_max_range,
                'left_min': self.scan_max_range,
                'right_min': self.scan_max_range,
            }
        valid = ranges[ranges > self.scan_valid_min]
        min_dist = float(valid.min()) if valid.size > 0 else self.scan_max_range
        front_idxs = np.r_[0:max(1, n // 18), n - max(1, n // 18):n]
        front_vals = ranges[front_idxs]
        front_valid = front_vals[front_vals > self.scan_valid_min]
        front_min = float(front_valid.min()) if front_valid.size > 0 else self.scan_max_range
        left_slice = ranges[int(n * 60 / 360):int(n * 120 / 360)]
        right_slice = ranges[int(n * 240 / 360):int(n * 300 / 360)]
        left_valid = left_slice[left_slice > self.scan_valid_min]
        right_valid = right_slice[right_slice > self.scan_valid_min]
        left_min = float(left_valid.min()) if left_valid.size > 0 else self.scan_max_range
        right_min = float(right_valid.min()) if right_valid.size > 0 else self.scan_max_range
        return {'min_dist': min_dist, 'front_min': front_min, 'left_min': left_min, 'right_min': right_min}

    def _get_neighbor_min_dist(self) -> float:
        best = float('inf')
        for d, _, _, _, _, _ in self._get_valid_neighbor_entries():
            if d < best:
                best = d
        return best

    def _apply_safety_shield(
        self,
        linear_vel: float,
        angular_vel: float,
        target_angle: float,
        conflict: Optional[Dict[str, float]] = None,
    ):
        if not self.shield_enable:
            return float(linear_vel), float(angular_vel), None
        metrics = self._scan_sector_metrics()
        if metrics is None:
            return float(linear_vel), float(angular_vel), None
        neighbor_min_dist = self._get_neighbor_min_dist()
        front_min = metrics['front_min']
        left_min = metrics['left_min']
        right_min = metrics['right_min']
        v = float(linear_vel)
        w = float(angular_vel)
        reasons = []
        if front_min < self.shield_front_slow_dist:
            v = min(v, self.shield_linear_slow)
            reasons.append('front_slow')
        if neighbor_min_dist < self.shield_neighbor_slow_dist:
            v = min(v, self.shield_linear_slow)
            reasons.append('neighbor_slow')
        if conflict is not None and float(conflict.get('should_yield', 0.0)) > 0.5:
            conflict_dist = float(conflict.get('dist', float('inf')))
            if conflict_dist < self.yielding_soft_dist:
                v = min(v, self.shield_linear_slow)
                reasons.append('yield_slow')
            if conflict_dist < self.yielding_stop_dist:
                v = min(v, self.shield_linear_stop)
                reasons.append('yield_stop')
            if conflict_dist < self.yielding_hard_stop_dist:
                v = 0.0
                reasons.append('yield_hard_stop')
        if front_min < self.shield_front_stop_dist:
            v = min(v, self.shield_linear_stop)
            reasons.append('front_stop')
        if front_min < self.turn_in_place_front_dist and abs(target_angle) > self.turn_in_place_angle_thresh:
            v = 0.0
            turn_dir = np.sign(target_angle)
            if abs(turn_dir) < 1e-6:
                turn_dir = 1.0 if left_min >= right_min else -1.0
            w = float(np.clip(self.turn_in_place_w * turn_dir, -1.2, 1.2))
            reasons.append('turn_in_place')
        elif front_min < self.turn_in_place_front_dist:
            bias = self.shield_turn_bias if left_min >= right_min else -self.shield_turn_bias
            w = float(np.clip(w + bias, -1.2, 1.2))
            reasons.append('turn_bias')
        reason = '+'.join(reasons) if reasons else None
        return float(v), float(np.clip(w, -1.2, 1.2)), reason

    # ── 控制主循环 ─────────────────────────────────────────────────────────

    def _control_loop(self):
        self.current_step += 1

        # 1. 更新航点
        self._update_waypoint()

        tracking_target = self._get_tracking_target()

        # 2. 构造观测
        local_obs = self._build_local_obs(target_override=tracking_target)
        if local_obs is None:
            return  # 传感器尚未就绪，跳过

        neighbor_obs = self._build_neighbor_obs()
        obs = np.concatenate([local_obs, neighbor_obs]).astype(np.float32)

        # obs 维度对齐检查
        if obs.shape[0] != self.obs_dim:
            self.get_logger().warn(
                f'obs dim mismatch: got {obs.shape[0]}, expected {self.obs_dim}'
            )
            return

        # 3. 策略推理
        if self._policy is not None:
            try:
                action = self._policy.infer(obs)  # [linear_norm, angular_norm] ∈ [-1,1]
            except Exception as e:
                self.get_logger().error(f'推理异常: {e}')
                action = np.zeros(2, dtype=np.float32)
        else:
            # 无模型时使用随机动作（调试用）
            action = np.random.uniform(-0.1, 0.1, 2).astype(np.float32)

        # 4. 动作映射（与 IndependentRobotEnv 对齐）
        # linear_vel = (action[0] + 1) / 2 * 0.22   ∈ [0, 0.22] m/s
        # angular_vel = action[1] * 1.0              ∈ [-1.0, 1.0] rad/s
        linear_vel  = float((action[0] + 1.0) / 2.0 * 0.22)
        angular_vel = float(action[1] * 1.0)

        if tracking_target is not None:
            target_angle = self._get_target_angle(tracking_target)
            angular_vel = float(np.clip(0.75 * angular_vel + 0.45 * target_angle, -1.2, 1.2))
            if abs(target_angle) > 0.9:
                linear_vel = min(linear_vel, 0.10)
            elif abs(target_angle) > 0.45:
                linear_vel = min(linear_vel, 0.16)
        else:
            target_angle = 0.0

        linear_vel, angular_vel, shield_reason = self._apply_safety_shield(
            linear_vel,
            angular_vel,
            target_angle,
            conflict=self._last_conflict_state,
        )
        if shield_reason:
            self.get_logger().debug(f'shield={shield_reason} v={linear_vel:.3f} w={angular_vel:.3f}')

        # 5. 发布 cmd_vel
        cmd = Twist()
        cmd.linear.x  = float(np.clip(linear_vel,  0.0,  0.22))
        cmd.angular.z = float(np.clip(angular_vel, -1.2,  1.2))
        self.cmd_pub.publish(cmd)

    # ── 清理 ───────────────────────────────────────────────────────────────

    def destroy_node(self):
        # 停止机器人
        stop = Twist()
        self.cmd_pub.publish(stop)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 收到停止信号')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
