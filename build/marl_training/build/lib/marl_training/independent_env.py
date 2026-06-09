import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import random
import os
import yaml
from collections import deque
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.parameter import Parameter
from ament_index_python.packages import get_package_share_directory

# ROS 消息
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState

# 引入自定义工具
from marl_training.global_planner import AStarPlanner, WaypointExtractor
from marl_training.waypoint_visualizer import WaypointVisualizer 

class IndependentRobotEnv(gym.Env):
    def __init__(self, robot_id=0, map_number=3, max_episode_steps=500, use_random_mode=True,
                 collision_ends_episode=True, num_dynamic_obstacles=8, obs_speed=0.3):
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
        
        # 1. 初始化 ROS 节点
        if not rclpy.ok():
            rclpy.init()
            
        # 强制开启 use_sim_time
        self.node = rclpy.create_node(
            f'gym_env_robot_{robot_id}_{random.randint(0, 100000)}',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )
        
        # 2. 命名空间设置
        # 根据你的 launch 文件，前缀是 tb3_0
        self.ns = f"/tb3_{robot_id}" 
        self.gazebo_model_name = f"tb3_{robot_id}" 

        print(f"🤖 环境初始化: Namespace='{self.ns}', ModelName='{self.gazebo_model_name}'")

        # 3. 加载地图 & 初始化规划器
        self.map_image = None
        self.planner = None
        self._load_map_data(self.map_number)
        
        if self.map_image is not None:
            # 翻转逻辑
            map_data_inverted = 255 - self.map_image
            map_data_for_planner = np.flipud(map_data_inverted)
            
            self.planner = AStarPlanner(
                map_data_for_planner, 
                resolution=self.map_resolution, 
                origin=(self.map_origin[0], self.map_origin[1])
            )
            self.waypoint_extractor = WaypointExtractor()
            print(f"✅ Robot {robot_id}: A*规划器初始化完成")
        
        self.global_waypoints = []
        self.current_waypoint_index = 0

        self.vis_namespace = f"robot_{robot_id}_waypoints"
        # 可视化
        # vis_topic = f'{self.ns}/planned_path'
        # self.vis = WaypointVisualizer(self.node, topic_name=vis_topic)
        vis_topic = '/waypoint_markers' 
        self.vis = WaypointVisualizer(self.node, topic_name=vis_topic)

        # 4. ROS 接口
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.vel_pub = self.node.create_publisher(Twist, f'{self.ns}/cmd_vel', 10)
        self.scan_sub = self.node.create_subscription(LaserScan, f'{self.ns}/scan', self._scan_callback, qos)
        self.odom_sub = self.node.create_subscription(Odometry, f'{self.ns}/odom', self._odom_callback, qos)
        
        self.set_state_client = self.node.create_client(SetEntityState, '/set_entity_state')
        
        # 变量
        self.latest_scan = None
        self.current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_vel_x = 0.0
        self.current_vel_w = 0.0
        
        self.scan_dim = 36
        # obs = lidar×4帧(144) + target(2) + vel(2) = 148
        # 设计思路：叠加最近 4 帧 LiDAR 扫描，模型通过观察扫描值跟时变化自主
        # 学习动态障碍物的运动意图——无需任何特权信息，可直接迁移到真实机器人。
        # 帧差分含速度：frame[t]-frame[t-1]，4 帧覆盖 400ms @10Hz 历史。
        self.scan_history_len = 4
        self._scan_history: deque = deque(maxlen=self.scan_history_len)
        self.obs_dim = self.scan_dim * self.scan_history_len + 2 + 2  # 36×4+2+2 = 148
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.goal_pos = (0.0, 0.0)
        self.prev_dist_to_goal = None

        # ── 动态障碍物配置 ──────────────────────────────────────────────
        # 障碍物已改用 Gazebo <actor> 脚本轨迹驱动，仳真器自动平滑插值。
        # num_dynamic_obstacles / obs_speed 保留作接口参数，当前不对运行产生影响。
        self.num_dynamic_obstacles = max(0, min(int(num_dynamic_obstacles), 8))
        self.obs_speed = float(obs_speed)
        self.dynamic_obstacle_names: list = [f'dyn_obs_{i}' for i in range(8)]

    def _load_map_data(self, map_number):
        try:
            pkg_path = get_package_share_directory('start_rl_environment_tb3')
            map_mapping = {1: 'map1', 2: 'map2', 3: 'corridor_swap', 4: 'intersection', 5: 'warehouse_aisles'}
            map_name = map_mapping.get(map_number, 'map1')
            yaml_path = os.path.join(pkg_path, 'maps', f'{map_name}.yaml')
            if not os.path.exists(yaml_path): return

            with open(yaml_path, 'r') as f: map_info = yaml.safe_load(f)
            self.map_resolution = map_info['resolution']
            self.map_origin = map_info['origin']
            
            image_filename = map_info['image']
            image_path = os.path.join(os.path.dirname(yaml_path), image_filename)
            with Image.open(image_path) as img:
                self.map_image = np.array(img.convert('L'))
                self.map_height, self.map_width = self.map_image.shape
        except Exception as e:
            print(f"❌ 加载地图失败: {e}")
            self.map_image = None

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

    def reset(self, seed=None, options=None, other_agent_starts=None):
        """
        other_agent_starts: list of (x, y) — 已生成的其他机器人起始世界坐标。
        新起始点与这些点的距离不小于 min_agent_sep，防止多机器人重叠出生。
        """
        super().reset(seed=seed)
        self.current_step = 0
        self._publish_vel(0.0, 0.0)
        
        # 碰撞历史（用于持续性检测）
        self.collision_history = []

        # === 先清除上一回合的旧标记 ===
        # 必须传入当前机器人的专属 namespace
        if hasattr(self, 'vis') and self.vis:
             self.vis.clear_waypoints(namespace=self.vis_namespace)

        # 1. 寻找有效起终点（逐步放宽最小间距直到找到有效点）
        if self.use_random_mode and self.map_image is not None:
            found_path = False
            for min_sep in [1.5, 1.0, 0.5]:  # 依次放宽间距要求
              for _ in range(50):
                # 安全边距
                start_x, start_y = self._get_random_valid_point(
                    safe_margin=12,
                    other_agents=other_agent_starts,
                    min_agent_sep=min_sep,
                )
                goal_x, goal_y = self._get_random_valid_point(exclude=(start_x, start_y), safe_margin=10)
                
                if self.planner:
                     path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
                     if path:
                         found_path = True
                         break
              if found_path:
                  break  # 找到了，退出 min_sep 循环
            if not found_path:
                # 备用点：优先使用地图内部已知安全点，避免默认(0,0)在map1围墙外
                fallback = self._MAP_FALLBACK_POSES.get(self.map_number, ((0.0, 0.0), (2.0, 2.0)))
                (start_x, start_y), (goal_x, goal_y) = fallback
                print(f"⚠️ Reset robot_{self.robot_id}: 随机点生成失败，使用备用位置 {fallback}")
        else:
            # 非随机模式：同样使用地图内部备用点
            fallback = self._MAP_FALLBACK_POSES.get(self.map_number, ((0.0, 0.0), (5.0, 5.0)))
            (start_x, start_y), (goal_x, goal_y) = fallback

        # 记录实际起始点，供 GNNMARLEnv 收集已占用坐标
        self.last_spawn_pos = (start_x, start_y)

        self.goal_pos = (goal_x, goal_y)
        
        # 2. 规划路径
        if self.planner:
            path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
            if path:
                self.global_waypoints = self.waypoint_extractor.extract(path)
                self.current_waypoint_index = 0
                # self.vis.publish_waypoints(self.global_waypoints)
                self.vis.publish_waypoints(
                    self.global_waypoints, 
                    robot_id=self.robot_id, 
                    namespace=self.vis_namespace
                )
            else:
                self.global_waypoints = [self.goal_pos]
                # self.vis.publish_waypoints([self.goal_pos])

                self.vis.publish_waypoints(
                    [self.goal_pos], 
                    robot_id=self.robot_id, 
                    namespace=self.vis_namespace
                )
        
        # 3. 瞬移机器人
        yaw = random.uniform(-3.14, 3.14)
        self._set_robot_pose(start_x, start_y, yaw)
        
        # 4. 等待数据刷新
        self.latest_scan = None
        self._scan_history.clear()     # 重置：清除历史帧，新 episode 从单帧前向填充开始
        self._wait_for_sim_time(0.2)
        
        self.prev_dist_to_goal = math.hypot(
            self.goal_pos[0] - self.current_pose['x'],
            self.goal_pos[1] - self.current_pose['y']
        )
        return self._get_obs(), {'start_xy': (start_x, start_y)}
    
    def apply_action(self, action, debug=False):
        """
        [并行化第一步] 仅应用动作，不等待时间，不返回结果
        用于多智能体环境中的并行化执行
        """
        self.current_step += 1
        
        max_linear_vel = 0.22
        max_angular_vel = 1.0

        linear_vel = (action[0] + 1.0) / 2.0 * max_linear_vel
        angular_vel = action[1] * max_angular_vel
        
        # 【调试】检测是否为零动作
        if debug and abs(linear_vel) < 0.01 and abs(angular_vel) < 0.01:
            print(f"⚠️  Robot {self.robot_id}: 零动作! action=[{action[0]:.3f}, {action[1]:.3f}] -> vel=[{linear_vel:.3f}, {angular_vel:.3f}]")

        self._publish_vel(linear_vel, angular_vel)

    def get_step_result(self):
        """
        [并行化第二步] 在时间推进后，获取观测、奖励和状态
        """
        # 1. 获取最新观测 (此时回调函数应该已经更新了 self.latest_scan)
        obs = self._get_obs()
        
        # 【调试】检查步数
        # if self.current_step % 100 == 0:
        #     print(f"Robot {self.robot_id}: current_step={self.current_step}, max={self.max_episode_steps}")
        
        # 2. 计算奖励 (复用之前的逻辑)
        reward = 0.0
        done = False
        truncated = False
        info = {}
        
        # 目标跟踪
        current_target = self.goal_pos
        if self.global_waypoints and self.current_waypoint_index < len(self.global_waypoints):
            current_target = self.global_waypoints[self.current_waypoint_index]
        
        dist_to_final_goal = math.hypot(self.goal_pos[0] - self.current_pose['x'], self.goal_pos[1] - self.current_pose['y'])
        
        # 距离奖励
        if self.prev_dist_to_goal is not None:
            reward += (self.prev_dist_to_goal - dist_to_final_goal) * 10.0
        self.prev_dist_to_goal = dist_to_final_goal
        
        # 路点奖励
        dist_to_wp = math.hypot(current_target[0] - self.current_pose['x'], current_target[1] - self.current_pose['y'])
        if dist_to_wp < 0.5 and current_target != self.goal_pos:
            self.current_waypoint_index += 1
            reward += 1.0
            
        # 碰撞检测
        # 阈值说明：
        #   动态障碍物半径=0.1m，机器人底盘半径≈0.105m
        #   物理接触时 LiDAR 读数 ≈ 0.205 - 0.1 = 0.105m < 硬件 min_range(0.12m)
        #   → 传感器返回 inf，被过滤掉，导致碰撞对检测不可见
        #   修复：软件过滤阈值降到 0.10m，碰撞阈值提高到 0.35m
        if self.latest_scan and self.current_step > 5:
            # 过滤无效数据（仅去 nan/inf，保留硬件能感知的最近读数）
            valid_ranges = [r for r in self.latest_scan.ranges 
                            if not math.isnan(r) and not math.isinf(r) and r > 0.10]
            if len(valid_ranges) > 0:
                min_dist = min(valid_ranges)
                # 碰撞阈值：0.35m（动态障碍物半径0.1 + 机器人0.105 + 安全余量0.145）
                if min_dist < 0.35:
                    reward -= 20.0
                    done = True
                    info['event'] = 'collision'

        # 到达目标检测
        if dist_to_final_goal < 0.4:
            reward += 20.0
            done = True
            info['event'] = 'goal'

        # 时间惩罚
        reward -= 0.01

        # 超时截断
        if self.current_step >= self.max_episode_steps:
            truncated = True
            
        return obs, reward, done, truncated, info

    def step(self, action):
        """
        [保留兼容性] 如果是单智能体训练或测试，依然可以用这个旧接口
        """
        self.apply_action(action)
        self._wait_for_sim_time(0.1)
        return self.get_step_result()

    def _wait_for_sim_time(self, seconds):
        if not rclpy.ok(): return
        # 等待时钟启动
        while rclpy.ok() and self.node.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self.node, timeout_sec=0.01)

        start_time = self.node.get_clock().now().nanoseconds
        delta_ns = seconds * 1e9
        
        while rclpy.ok():
            now = self.node.get_clock().now().nanoseconds
            if now - start_time >= delta_ns:
                break
            rclpy.spin_once(self.node, timeout_sec=0.01)

    # 各地图封闭区域的世界坐标范围 (x_min, x_max, y_min, y_max)。
    # 根据围墙内边缘向内收缩 0.3m 得到，保证机器人不会生成在墙边处。
    # 数据依据：坯增内边缘 x≈-1.30m, 东墙内边缘 x≈2.40m,
    #          北墙内边缘 y≈-0.45m, 南边界(map底)≈-10.0m
    _MAP_WORLD_BOUNDS = {
        1: (-1.00, 2.10, -9.50, -0.75),  # map1: 封闭区域内部
        3: (-5.50, 5.50, -5.50, 5.50),   # map3 corridor_swap 12×12m
    }

    # 各地图大小不够时的 fallback 生成位置和目标位置
    _MAP_FALLBACK_POSES = {
        1: ((0.5, -5.0), (-0.3, -8.5)),   # map1
        3: ((-4.5, 0.0), ( 4.5,  0.0)),   # map3 corridor_swap
    }

    # 各地图动态障碍物初始 spawn 位置（与 obstacle_mover.py MAP_CONFIGS 保持同步）
    # 随机生成起点/终点时会排除距任意 spawn 点 < 1.0m 的候选位置
    _DYN_OBS_SPAWNS = {
        1: [(0.6,-1.2),(0.6,-2.4),(0.6,-3.6),(0.6,-4.8),
            (0.6,-6.0),(0.6,-7.2),(0.6,-8.4),(0.6,-9.6)],
        2: [(0.5,-1.5),(0.5,-4.0),(0.5,-6.5),(0.5,-9.5),
            (5.5,-1.5),(9.0,-2.0),(5.5,-7.0),(9.0,-9.5)],
        3: [(-4.5,-4.0),(-4.5, 4.0),(-2.0,-4.5),(-2.0, 4.5),
            ( 2.0,-4.5),( 2.0, 4.5),( 4.5,-4.0),( 4.5, 4.0)],
        4: [(-4.5, 0.8),(-4.5,-0.8),(4.5, 0.8),(4.5,-0.8),
            ( 0.8,-4.5),(-0.8,-4.5),(0.8, 4.5),(-0.8, 4.5)],
        5: [(-4.5,-6.0),(-4.5, 6.0),(-0.5,-5.5),(-0.5, 5.5),
            ( 3.5,-6.0),( 3.5, 5.0),( 7.5,-7.0),( 7.5, 7.0)],
    }

    def _get_map_world_bounds(self):
        """返回当前地图封闭区域的世界坐标范围 (x_min, x_max, y_min, y_max)。
        若无限制返回 None。"""
        return self._MAP_WORLD_BOUNDS.get(self.map_number, None)

    def _get_random_valid_point(self, exclude=None, safe_margin=10,
                                 other_agents=None, min_agent_sep=1.5):
        """
        other_agents : list of (x, y) world coords — 新点与这些点的距离需 >= min_agent_sep
        """
        if self.map_image is None: return 0.0, 0.0

        # 当前地图的世界坐标硬限制（封闭区域内部，应用后确保结果属于围墙内）
        world_bounds = self._get_map_world_bounds()  # (x_min, x_max, y_min, y_max) or None

        # 像素坐标范围：有世界坐标限制时根据其预计像素范围尾缩采样区间，加速收敛
        if world_bounds is not None:
            x_min, x_max, y_min, y_max = world_bounds
            # 世界坐标 → 像素坐标（留 safe_margin 预留量）
            px_lo = max(safe_margin, int((x_min - self.map_origin[0]) / self.map_resolution) - safe_margin)
            px_hi = min(self.map_width - 1 - safe_margin, int((x_max - self.map_origin[0]) / self.map_resolution) + safe_margin)
            py_lo = max(safe_margin, int(self.map_height - 1 - (y_max - self.map_origin[1]) / self.map_resolution) - safe_margin)
            py_hi = min(self.map_height - 1 - safe_margin, int(self.map_height - 1 - (y_min - self.map_origin[1]) / self.map_resolution) + safe_margin)
        else:
            px_lo, px_hi = safe_margin, self.map_width  - 1 - safe_margin
            py_lo, py_hi = safe_margin, self.map_height - 1 - safe_margin

        for _ in range(300):
            if px_lo >= px_hi or py_lo >= py_hi: return 0.0, 0.0

            px = random.randint(px_lo, px_hi)
            py = random.randint(py_lo, py_hi)
            
            # 1. 检查中心点
            if self.map_image[py, px] <= 200:
                continue
                
            # 2. 检查周围区域
            is_safe = True
            for dx in range(-safe_margin, safe_margin + 1, 2):
                for dy in range(-safe_margin, safe_margin + 1, 2):
                    ny, nx = py + dy, px + dx
                    if 0 <= ny < self.map_height and 0 <= nx < self.map_width:
                        if self.map_image[ny, nx] <= 200:
                            is_safe = False
                            break
                    # 超出地图边界的邻居点也认为不安全
                    else:
                        is_safe = False
                        break
                if not is_safe: break
            
            if not is_safe: continue

            # 3. 坐标转换
            grid_x = px
            grid_y = self.map_height - 1 - py
            world_x = self.map_origin[0] + (grid_x * self.map_resolution)
            world_y = self.map_origin[1] + (grid_y * self.map_resolution)

            # 4. 世界坐标硬限制（最终保随，防止像素边界计算小误差导致漏到围墙外）
            if world_bounds is not None:
                x_min, x_max, y_min, y_max = world_bounds
                if not (x_min <= world_x <= x_max and y_min <= world_y <= y_max):
                    continue
            
            # 5. 排除点距离检查（起/终点互斥）
            if exclude and math.hypot(world_x - exclude[0], world_y - exclude[1]) < 2.0:
                continue

            # 6. 多机器人间距检查（防止不同机器人在同一位置生成）
            if other_agents:
                too_close = any(
                    math.hypot(world_x - ax, world_y - ay) < min_agent_sep
                    for ax, ay in other_agents
                )
                if too_close:
                    continue

            # 7. 动态障碍物初始位置排除（距任意 spawn 点 < 1.0m 则跳过）
            dyn_obs_spawns = self._DYN_OBS_SPAWNS.get(self.map_number, [])
            if any(math.hypot(world_x - ox, world_y - oy) < 1.0
                   for ox, oy in dyn_obs_spawns):
                continue

            return world_x, world_y
            
        return 0.0, 0.0

    # ─────────────────────────────────────────────────────────────────
    # 动态障碍物管理
    # ─────────────────────────────────────────────────────────────────
    def randomize_obstacles(self, robot_positions: list):
        """障碍物已改用 Gazebo Actor 脚本轨迹，仳真器内部平滑持续运动。
        此方法保留以完成接口兼容，无需招行。"""
        pass

    def _set_obstacle_pose(self, model_name: str, x: float, y: float, z: float):
        """通过 /set_entity_state 将 Gazebo 中的模型瞬移到指定坐标。
        直接复用 self.set_state_client（与 _set_robot_pose 共享同一客户端）。"""
        if not self.set_state_client.wait_for_service(timeout_sec=2.0):
            return  # Gazebo 服务不可用时静默跳过

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

    def _get_obs(self):
        ranges = np.array(self.latest_scan.ranges if self.latest_scan else [3.5]*360)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.0)
        num_samples = self.scan_dim
        step = max(1, len(ranges) // num_samples)
        scan_obs = ranges[::step][:num_samples]
        if len(scan_obs) < num_samples:
            scan_obs = np.pad(scan_obs, (0, num_samples - len(scan_obs)), 'edge')
        scan_obs = np.clip(scan_obs, 0, 3.5) / 3.5   # 归一化到 [0, 1]

        # ── LiDAR 帧叠加 ─────────────────────────────────────────────────────
        # 模型通过对比连续帧的变化，自主学习动态障碍物的运动意图：
        #   静态障碍：各帧同一方向读数不变 → 学习识别为山墙/备柱
        #   动态障碍：某方向读数持续减小 → 学习识别为移动威胁并提前规避
        # 完全不依赖特权信息，可直接应用于真实机器人部署。
        self._scan_history.append(scan_obs.copy())

        # 历史帧堆叠（不足 scan_history_len 帧时用最旧帧前向填充）
        history = list(self._scan_history)
        while len(history) < self.scan_history_len:
            history.insert(0, history[0].copy())
        # 顺序：[最旧帧, ..., 次新帧, 最新帧]，模型通过帧序感知移动趋势
        stacked_scan = np.concatenate(history)   # (36 * scan_history_len,) = (144,)

        current_target = self.goal_pos
        if self.global_waypoints and self.current_waypoint_index < len(self.global_waypoints):
            current_target = self.global_waypoints[self.current_waypoint_index]

        tgt_dist = math.hypot(current_target[0] - self.current_pose['x'], current_target[1] - self.current_pose['y'])
        tgt_angle = math.atan2(current_target[1] - self.current_pose['y'], current_target[0] - self.current_pose['x'])
        rel_angle = (tgt_angle - self.current_pose['yaw'] + np.pi) % (2 * np.pi) - np.pi

        obs = np.concatenate([stacked_scan, [tgt_dist, rel_angle],
                               [self.current_vel_x, self.current_vel_w]])
        return obs.astype(np.float32)

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