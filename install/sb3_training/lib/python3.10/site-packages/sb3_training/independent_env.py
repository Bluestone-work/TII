import gymnasium as gym
import numpy as np
from gymnasium import spaces
import math
import random
import os
import yaml
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
from sb3_training.global_planner import AStarPlanner, WaypointExtractor
from sb3_training.waypoint_visualizer import WaypointVisualizer 

class IndependentRobotEnv(gym.Env):
    def __init__(self, robot_id=0, map_number=3, max_episode_steps=500, use_random_mode=True):
        super(IndependentRobotEnv, self).__init__()
        
        self.robot_id = robot_id
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
        self.obs_dim = self.scan_dim + 2 + 2 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        
        self.goal_pos = (0.0, 0.0)
        self.prev_dist_to_goal = None

    def _load_map_data(self, map_number):
        try:
            pkg_path = get_package_share_directory('start_rl_environment')
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self._publish_vel(0.0, 0.0)

        # === 先清除上一回合的旧标记 ===
        # 必须传入当前机器人的专属 namespace
        if hasattr(self, 'vis') and self.vis:
             self.vis.clear_waypoints(namespace=self.vis_namespace)

        # 1. 寻找有效起终点
        if self.use_random_mode and self.map_image is not None:
            found_path = False
            for _ in range(50):
                # 安全边距
                start_x, start_y = self._get_random_valid_point(safe_margin=12) 
                goal_x, goal_y = self._get_random_valid_point(exclude=(start_x, start_y), safe_margin=10)
                
                if self.planner:
                     path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
                     if path:
                         found_path = True
                         break
            if not found_path:
                print("⚠️ Reset: 随机点生成失败，使用默认点")
                start_x, start_y = 0.0, 0.0
                goal_x, goal_y = 2.0, 2.0
        else:
            start_x, start_y = 0.0, 0.0
            goal_x, goal_y = 5.0, 5.0

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
        self._wait_for_sim_time(0.2) 
        
        self.prev_dist_to_goal = math.hypot(
            self.goal_pos[0] - self.current_pose['x'],
            self.goal_pos[1] - self.current_pose['y']
        )
        return self._get_obs(), {}
    
    def apply_action(self, action):
        """
        [并行化第一步] 仅应用动作，不等待时间，不返回结果
        """
        self.current_step += 1
        
        max_linear_vel = 0.22
        max_angular_vel = 1.0

        linear_vel = (action[0] + 1.0) / 2.0 * max_linear_vel
        angular_vel = action[1] * max_angular_vel

        self._publish_vel(linear_vel, angular_vel)

    def get_step_result(self):
        """
        [并行化第二步] 在时间推进后，获取观测、奖励和状态
        """
        # 1. 获取最新观测 (此时回调函数应该已经更新了 self.latest_scan)
        obs = self._get_obs()
        
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
        if self.latest_scan and self.current_step > 5:
            # 过滤无效数据
            valid_ranges = [r for r in self.latest_scan.ranges 
                            if not math.isnan(r) and not math.isinf(r) and r > 0.15]
            if len(valid_ranges) > 0:
                min_dist = min(valid_ranges)
                # 碰撞阈值
                if min_dist < 0.20:
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
        [保留兼容性] 如果你是单智能体训练，依然可以用这个旧接口
        """
        self.apply_action(action)
        self._wait_for_sim_time(0.1)
        return self.get_step_result()
    
    # def step(self, action):
    #     self.current_step += 1

    #     # linear_vel = (action[0] + 1.0) / 10.0
    #     # angular_vel = action[1] * 1.0

    #     max_linear_vel = 0.22
    #     max_angular_vel = 1.0

    #     linear_vel = (action[0] + 1.0) / 2.0 * max_linear_vel
    #     angular_vel = action[1] * max_angular_vel

    #     self._publish_vel(linear_vel, angular_vel)
        
    #     # 等待仿真时间推进
    #     self._wait_for_sim_time(0.1)
            
    #     obs = self._get_obs()
    #     reward = 0.0
    #     done = False
    #     truncated = False
    #     info = {}
        
    #     # 目标跟踪
    #     current_target = self.goal_pos
    #     if self.global_waypoints and self.current_waypoint_index < len(self.global_waypoints):
    #         current_target = self.global_waypoints[self.current_waypoint_index]
        
    #     dist_to_final_goal = math.hypot(self.goal_pos[0] - self.current_pose['x'], self.goal_pos[1] - self.current_pose['y'])
    #     dist_to_wp = math.hypot(current_target[0] - self.current_pose['x'], current_target[1] - self.current_pose['y'])
        
    #     reward += (self.prev_dist_to_goal - dist_to_final_goal) * 10.0
    #     self.prev_dist_to_goal = dist_to_final_goal
        
    #     if dist_to_wp < 0.5 and current_target != self.goal_pos:
    #         self.current_waypoint_index += 1
    #         reward += 1.0
            
    #     # =========================================================
    #     # [核心修复] 
    #     # =========================================================
    #     if self.latest_scan and self.current_step > 5: # 给一点无敌时间
    #         valid_ranges = [r for r in self.latest_scan.ranges 
    #                         if not math.isnan(r) and not math.isinf(r) and r > 0.15]
    #         # print(f"距离: {valid_ranges}")
    #         if len(valid_ranges) > 0:
    #             min_dist = min(valid_ranges)
                
    #             # [关键] 阈值稍微放宽到 0.20
    #             if min_dist < 0.20:
    #                 reward -= 20.0
    #                 done = True
    #                 info['event'] = 'collision'
    #                 # print(f"💥 碰撞重置! 最小距离: {min_dist:.3f}")
        
    #     if dist_to_final_goal < 0.4:
    #         reward += 20.0
    #         done = True
    #         info['event'] = 'goal'
            
    #     reward -= 0.01 
        
    #     if self.current_step >= self.max_episode_steps:
    #         truncated = True
            
    #     # print(info.get('event', ''), f"Step: {self.current_step}, Reward: {reward:.3f}, DistToGoal: {dist_to_final_goal:.3f}")
    #     return obs, reward, done, truncated, info

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

    def _get_random_valid_point(self, exclude=None, safe_margin=10):
        if self.map_image is None: return 0.0, 0.0
        
        for _ in range(100):
            if self.map_width <= 2*safe_margin: return 0.0, 0.0

            px = random.randint(safe_margin, self.map_width - 1 - safe_margin)
            py = random.randint(safe_margin, self.map_height - 1 - safe_margin)
            
            # 1. 检查中心点
            if self.map_image[py, px] <= 200:
                continue
                
            # 2. 检查周围区域
            is_safe = True
            for dx in range(-safe_margin, safe_margin + 1, 2):
                for dy in range(-safe_margin, safe_margin + 1, 2):
                    if self.map_image[py + dy, px + dx] <= 200:
                        is_safe = False
                        break
                if not is_safe: break
            
            if not is_safe: continue

            # 3. 坐标转换
            grid_x = px
            grid_y = self.map_height - 1 - py
            world_x = self.map_origin[0] + (grid_x * self.map_resolution)
            world_y = self.map_origin[1] + (grid_y * self.map_resolution)
            
            # 4. 排除点距离检查
            if exclude and math.hypot(world_x - exclude[0], world_y - exclude[1]) < 2.0:
                continue
                
            return world_x, world_y
            
        return 0.0, 0.0

    def _get_obs(self):
        ranges = np.array(self.latest_scan.ranges if self.latest_scan else [3.5]*360)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=0.0)
        num_samples = self.scan_dim
        step = max(1, len(ranges) // num_samples)
        scan_obs = ranges[::step][:num_samples]
        if len(scan_obs) < num_samples:
            scan_obs = np.pad(scan_obs, (0, num_samples - len(scan_obs)), 'edge')
        scan_obs = np.clip(scan_obs, 0, 3.5) / 3.5 
        
        current_target = self.goal_pos
        if self.global_waypoints and self.current_waypoint_index < len(self.global_waypoints):
            current_target = self.global_waypoints[self.current_waypoint_index]
            
        tgt_dist = math.hypot(current_target[0] - self.current_pose['x'], current_target[1] - self.current_pose['y'])
        tgt_angle = math.atan2(current_target[1] - self.current_pose['y'], current_target[0] - self.current_pose['x'])
        rel_angle = (tgt_angle - self.current_pose['yaw'] + np.pi) % (2 * np.pi) - np.pi
        
        obs = np.concatenate([scan_obs, [tgt_dist, rel_angle], [self.current_vel_x, self.current_vel_w]])
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