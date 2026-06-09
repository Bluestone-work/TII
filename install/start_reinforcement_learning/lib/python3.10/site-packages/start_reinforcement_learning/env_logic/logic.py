import rclpy
from rclpy.node import Node

import numpy as np
import math
from math import pi
import time
import uuid
import json
import os
from datetime import datetime
from scipy.ndimage import distance_transform_edt

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

from rclpy.qos import qos_profile_sensor_data, QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
import tf2_ros

from start_reinforcement_learning.env_logic.restart_environment import RestartEnvironment

class Env():
    def __init__(
        self,
        number_of_robots=3,
        map_number=1,
        use_random_mode=False,
        goal_termination_mode: str = 'any',
        stuck_enabled: bool = True,
        stuck_min_progress: float = 0.02,
        stuck_max_steps: int = 40,
        stuck_check_after_steps: int = 20,
        stuck_penalty: float = -10.0,
    ):
        """
        初始化环境
        Args:
            number_of_robots: 机器人数量
            map_number: 地图编号 (1-5)
            use_random_mode: True=随机模式，False=固定模式
        """
        # 生成唯一 ID 避免节点名称冲突
        self.env_id = str(uuid.uuid4())[:8]
        
        self.number_of_robots = number_of_robots
        self.map_number = map_number
        self.use_random_mode = use_random_mode
        
        # ===== 距离场功能（禁用以简化） =====
        self.use_distance_field = False  # 禁用距离场
        self.distance_field_size = 7  # 7x7局部距离场
        self.distance_field_flat_size = self.distance_field_size * self.distance_field_size  # 49
        self.distance_field_cache = {}
        self.distance_field_warning_printed = set()
        self.cache_precision = 2  # 缓存精度（小数位）
        self.map_subscriber = None  # 地图订阅器（如果需要距离场）

        # 仿真时钟推进参数（使用ROS仿真时间）
        self.sim_step_sec = 0.1
        self.sim_clock = Clock(clock_type=ClockType.ROS_TIME)

        # 观测异常调试
        self.debug_obs_warnings = True
        self._warn_counts = {
            'lidar_missing': 0,
            'lidar_invalid': 0,
            'pose_invalid': 0,
            'obs_nan': 0,
        }
        self._warn_limit = 20
        self.odom_stale_warn_sec = 0.5
        self._last_pose = [(np.nan, np.nan)] * self.number_of_robots
        self._last_odom_stamp = [None] * self.number_of_robots
        self.map_frame = 'map'
        self.current_pose_x_map = np.full(self.number_of_robots, np.nan, dtype=np.float32)
        self.current_pose_y_map = np.full(self.number_of_robots, np.nan, dtype=np.float32)
        self.pose_in_map_valid = [False] * self.number_of_robots
        self.tf_node = Node(f"tf_listener_{self.env_id}")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.tf_node)
        self._tf_warn_count = 0
        self._tf_warn_limit = 10
        self.last_tf_info = {'tx': np.nan, 'ty': np.nan, 'yaw': np.nan, 'ok': False, 'frame': ''}
        
        # ===== 🔍 数据同步验证系统 =====
        self.enable_sync_validation = True  # 开启同步验证
        self.sync_validation_interval = 50  # 每50步详细验证一次
        # 改用相对时间戳验证（检查数据是否更新，而不是绝对年龄）
        self.last_validated_timestamps = {
            'odom': [None] * self.number_of_robots,
            'scan': [None] * self.number_of_robots,
        }
        self.stale_data_threshold_steps = 10  # 连续N步数据不更新视为过时
        self.stale_data_counters = {
            'odom': [0] * self.number_of_robots,
            'scan': [0] * self.number_of_robots,
        }
        self.data_timestamps = {
            'odom': [None] * self.number_of_robots,
            'scan': [None] * self.number_of_robots,
            'map': None,
            'action_executed': None,
            'reward_calculated': None
        }
        self.tf_frames_discovered = False
        self.available_tf_frames = set()
        print("\n" + "="*80)
        print("🔍 数据同步验证系统已启用（使用相对时间戳验证）")
        print(f"   - 数据更新检测: 每步检查时间戳是否变化")
        print(f"   - 详细报告间隔: 每{self.sync_validation_interval}步")
        print("="*80 + "\n")

        # ===== 终止/截断策略配置 =====
        # goal_termination_mode:
        #   - 'any': 任意机器人到达即结束（更适合多机器人训练）
        #   - 'all': 所有机器人到达才结束（更难，容易学会拖延/打转）
        self.goal_termination_mode = str(goal_termination_mode).lower()
        if self.goal_termination_mode not in ('any', 'all'):
            self.goal_termination_mode = 'any'

        # 卡住检测：连续多步“接近目标的进展”低于阈值，则提前截断
        self.stuck_enabled = bool(stuck_enabled)
        self.stuck_min_progress = float(stuck_min_progress)
        self.stuck_max_steps = int(stuck_max_steps)
        self.stuck_check_after_steps = int(stuck_check_after_steps)
        self.stuck_penalty = float(stuck_penalty)
        self._stuck_steps = np.zeros(self.number_of_robots, dtype=np.int32)
        self._stuck_last_distance = [None] * self.number_of_robots
        
        # ========== 分层强化学习：全局规划器 ==========
        from start_reinforcement_learning.env_logic.global_planner import AStarPlanner, WaypointExtractor
        from start_reinforcement_learning.env_logic.waypoint_visualizer import WaypointVisualizer
        
        self.use_global_planner = True  # 启用全局规划
        self.planner = None  # 等地图加载后初始化
        self.waypoint_extractor = WaypointExtractor(
            turning_threshold=0.3,  # 转角>17度算拐点
            distance_threshold=1.5  # 直线段每1.5米一个点
        )
        
        # 路径点管理
        self.global_waypoints = [None] * number_of_robots
        self.current_waypoint_index = [0] * number_of_robots  
        self.waypoint_reach_distance = 0.3  # 到达阈值
        
        # Gazebo可视化
        self.waypoint_visualizer = WaypointVisualizer()
        
        # 地图订阅器（用于A*规划）
        self.map_subscriber = MapSubscriber(env_id=self.env_id)
        
        print("🗺️ 分层RL已启用：A*全局 + RL局部")
        
        # ===== 交互数据记录 =
        self.enable_interaction_logging = True  # 设为False可关闭日志
        self.interaction_log_dir = '/home/wj/work/multi-robot-exploration-rl/interaction_logs'
        if self.enable_interaction_logging:
            os.makedirs(self.interaction_log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.interaction_log_file = os.path.join(
                self.interaction_log_dir, 
                f'interaction_log_map{map_number}_robots{number_of_robots}_{timestamp}.jsonl'
            )
            print("\n" + "="*80)
            print(f"[INFO] 📝 交互数据将记录到: {self.interaction_log_file}")
            print("="*80 + "\n")
        self.episode_counter = 0
        self.total_step_counter = 0
        self.restart_environment = RestartEnvironment(
            self.number_of_robots, 
            self.map_number, 
            self.use_random_mode
        )
        
        # Create a list of N nodes to publish velocities to robots where N is the number of robots
        self.cmd_vel_publisher_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.cmd_vel_publisher_list[i] = PublishCMD_VEL(i)
            
        # Create a list of N nodes to read odometry from each robot (get their positions) where N is the number of robots
        self.odometry_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.odometry_subscriber_list[i] = ReadOdom(i)
            
        # Create a list of N nodes to read laser scan information where N is the number of robots
        self.scan_subscriber_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.scan_subscriber_list[i] = ReadScan(i)
        
        # Create a node for logging output to terminal
        self.logger = Logger()
        
        # 观测空间维度计算：
        # - lidar: 38维
        # - 自身速度: 2维 (linear, angular)
        # - 目标信息: 3维 (dx_r, dy_r, distance)
        # - 最近机器人: 6维 (rel_x_r, rel_y_r, vx, vz, dist, angle)
        # 基础维度：38 + 2 + 3 + 6 = 49维
        # 如果启用距离场：49 + 49 = 98维
        base_obs_dim = 49
        if self.use_distance_field:
            self.single_robot_observation_space = base_obs_dim + self.distance_field_flat_size
        else:
            self.single_robot_observation_space = base_obs_dim
        
        self.individual_robot_action_space = 2
        self.total_robot_observation_space = []
        for _ in range(self.number_of_robots):
            self.total_robot_observation_space.append(self.single_robot_observation_space)
        self.initGoal = True
        self.current_goal_location = []
        self.current_scan_data = 0

        self.step_counter = 0
        self.reached_goal_counter = 0
        self.total_goal_counter= 0
        self.MAX_STEPS = 500
        self.first_reset = True  # 第一次reset标志
        
        # 目标位置管理
        self.current_goal_locations = []  # 当前目标位置列表
        self.previous_distance_to_goal = [None] * self.number_of_robots  # 上一步到目标的距离
        self.current_yaw = np.zeros(self.number_of_robots)  # 当前朝向
        
        # 机器人名称
        self.robot_names = [f'robot{i}' for i in range(self.number_of_robots)]
        
        # 奖励分量记录
        self.last_reward_components = []
        
        # List of robot properties
        self.current_angular_velocity = np.zeros(self.number_of_robots)
        self.current_linear_velocity = np.zeros(self.number_of_robots)
        self.current_pose_x = np.zeros(self.number_of_robots)
        self.current_pose_y = np.zeros(self.number_of_robots)
        self.current_observations = np.zeros(self.number_of_robots)

        # Robot velocity restraints 
        self.max_linear_vel = 0.6
        self.min_linear_vel = 0
        self.max_angular_vel = 0.5
        self.min_angular_vel = -0.5
        
        # Rewards
        self.goalReward = 200
        self.collisionReward = -200
        
        # Marker publishers for visualization
        self.reward_marker_publisher_list = [None] * self.number_of_robots
        self.goal_marker_publisher_list = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            self.reward_marker_publisher_list[i] = RewardMarkerPublisher(i, self.env_id)
            self.goal_marker_publisher_list[i] = GoalMarkerPublisher(i, self.env_id)
    
    # get obs space.  in future we will return proper box but for now just
    # return .shape E.G just a number
    def observation_space(self):
        return self.total_robot_observation_space

    def action_space(self):
        return self.individual_robot_action_space
    
    def cleanup(self):
        """清理资源"""
        # 停止所有机器人
        try:
            self.reset_cmd_vel()
        except:
            pass
        # 清理ROS节点
        if hasattr(self, 'tf_node'):
            try:
                self.tf_node.destroy_node()
            except:
                pass
        print("✅ 环境资源已清理")
    
    def reset_cmd_vel(self):
        """重置所有机器人的速度为0"""
        from geometry_msgs.msg import Twist
        for i in range(self.number_of_robots):
            self.current_linear_velocity[i] = 0
            self.current_angular_velocity[i] = 0
            cmd_vel_publisher = self.cmd_vel_publisher_list[i]
            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(0)
            desired_vel_cmd.angular.z = float(0)
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()
    
    def hasCollided(self, scan_range, robot_number=0):
        """检查是否碰撞"""
        min_range = 0.35
        if len(scan_range) > 35:
            if min_range > np.min(scan_range[:35]) > 0:
                return True
        return False
    
    def hasReachedGoal(self, scan_range, robot_number):
        """检查是否到达目标"""
        if robot_number >= len(self.current_goal_locations):
            return False
        goal_x, goal_y = self.current_goal_locations[robot_number]
        dis_to_goal = math.hypot(
            goal_x - self.current_pose_x[robot_number],
            goal_y - self.current_pose_y[robot_number]
        )
        return dis_to_goal < 0.50
    
    def updateRobotPosition(self):
        """更新所有机器人的位置信息"""
        for i in range(self.number_of_robots):
            odom_subscriber = self.odometry_subscriber_list[i]
            rclpy.spin_once(odom_subscriber, timeout_sec=0.01)
            
            if odom_subscriber.odom is not None:
                self.current_pose_x[i] = odom_subscriber.odom.pose.pose.position.x
                self.current_pose_y[i] = odom_subscriber.odom.pose.pose.position.y
                
                # 计算yaw角
                qz = odom_subscriber.odom.pose.pose.orientation.z
                qw = odom_subscriber.odom.pose.pose.orientation.w
                self.current_yaw[i] = 2 * math.atan2(qz, qw)
    
    def getGoalDistace(self):
        """计算所有机器人到各自目标的最短距离"""
        closest_distance_to_goal = math.inf
        for i in range(self.number_of_robots):
            if i >= len(self.current_goal_locations):
                continue
            goal_pos = self.current_goal_locations[i]
            ith_robots_distance = round(math.hypot(
                goal_pos[0] - self.current_pose_x[i], 
                goal_pos[1] - self.current_pose_y[i]), 2)
            if ith_robots_distance < closest_distance_to_goal:
                closest_distance_to_goal = ith_robots_distance
        return closest_distance_to_goal
    
    def resize_lidar(self, scan):
        """调整激光雷达数据为固定38维"""
        np_scan_range = np.zeros(38, dtype=np.float32)
        
        # 只取 scan 数据 和 38 中的较小值，防止越界
        num_points = min(len(scan.ranges), 38)
        
        for i in range(num_points):
            val = scan.ranges[i]
            if val == float('Inf'):
                np_scan_range[i] = 3.5
            elif np.isnan(val):
                np_scan_range[i] = 0
            else:
                np_scan_range[i] = val
        
        return np_scan_range
    
    def getRewards(self):
        """计算所有机器人的奖励"""
        robotRewards = []
        reward_components = []
        
        for i in range(self.number_of_robots):
            # 简单奖励：基于到目标的距离
            reward = 0.0
            
            if i < len(self.current_goal_locations):
                goal_x, goal_y = self.current_goal_locations[i]
                current_distance = math.hypot(
                    goal_x - self.current_pose_x[i],
                    goal_y - self.current_pose_y[i]
                )
                
                # 前进奖励
                if self.current_linear_velocity[i] > 0.1:
                    reward += 0.5
                
                # 距离进展奖励
                if self.previous_distance_to_goal[i] is not None:
                    delta = self.previous_distance_to_goal[i] - current_distance
                    reward += delta * 5.0
                
                self.previous_distance_to_goal[i] = current_distance
                
                # 到达奖励
                if current_distance < 0.5:
                    reward += 10.0
            
            robotRewards.append(reward)
            reward_components.append({
                'total': reward,
                'obs_distance': current_distance if i < len(self.current_goal_locations) else 0.0
            })
        
        self.last_reward_components = reward_components
        return robotRewards
    
    def publish_reward_marker(self, robot_idx, total_reward, r_action, 
                             r_obstacle, r_goal, v, w, min_dist):
        """
        发布奖励可视化标记 (适配新的动作驱动奖励函数)
        Args:
            r_action: 动作奖励 (速度 - 旋转)
            r_obstacle: 避障惩罚
            r_goal: 目标距离奖励
            v: 当前线速度
            w: 当前角速度
            min_dist: 最近障碍物距离
        """
        marker_publisher = self.reward_marker_publisher_list[robot_idx]
        
        # 创建文本标记
        marker = Marker()
        marker.header.frame_id = f"my_bot{robot_idx}/base_link"
        from builtin_interfaces.msg import Time
        marker.header.stamp = Time(sec=0, nanosec=0)
        marker.ns = f"robot_{robot_idx}_reward"
        marker.id = robot_idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置：机器人头顶上方0.5米
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        # 文本内容：显示详细的奖励分解
        # R: 总分
        # Act: 动作分 (线速度 v, 角速度 w)
        # Obs: 避障分 (最近距离 d)
        # Goal: 目标引导分
        marker.text = f"R: {total_reward:.2f}\n" \
                     f"Act: {r_action:.2f} (v={v:.2f}, w={w:.2f})\n" \
                     f"Obs: {r_obstacle:.2f} (d={min_dist:.2f})\n" \
                     f"Goal: {r_goal:.2f}"
        
        # 大小
        marker.scale.z = 0.15  # 文本高度
        
        # 根据总奖励设置颜色
        if total_reward > 0.1:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # 绿色 (表现好)
        elif total_reward < -0.1:
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # 红色 (表现差)
        else:
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0) # 黄色 (中庸)
        
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        # 发布标记
        marker_publisher.publish_marker(marker)
    
    def clear_markers(self, robot_idx):
        """清除指定机器人的所有marker"""
        # 清除奖励marker
        reward_publisher = self.reward_marker_publisher_list[robot_idx]
        delete_marker = Marker()
        delete_marker.header.frame_id = f"my_bot{robot_idx}/base_link"
        from builtin_interfaces.msg import Time
        delete_marker.header.stamp = Time(sec=0, nanosec=0)
        delete_marker.ns = f"robot_{robot_idx}_reward"
        delete_marker.id = robot_idx
        delete_marker.action = Marker.DELETE
        
        marker_array = MarkerArray()
        marker_array.markers = [delete_marker]
        reward_publisher.marker_publisher.publish(marker_array)
        rclpy.spin_once(reward_publisher, timeout_sec=0.0)
        
        # 清除目标marker（3个marker：球体、文本、箭头）
        goal_publisher = self.goal_marker_publisher_list[robot_idx]
        delete_markers = MarkerArray()
        
        for marker_id in [robot_idx * 10, robot_idx * 10 + 1, robot_idx * 10 + 2]:
            delete_marker = Marker()
            delete_marker.header.frame_id = "map"
            delete_marker.header.stamp = Time(sec=0, nanosec=0)
            delete_marker.id = marker_id
            delete_marker.action = Marker.DELETEALL  # 删除该namespace下的所有marker
            delete_markers.markers.append(delete_marker)
        
        goal_publisher.marker_publisher.publish(delete_markers)
        rclpy.spin_once(goal_publisher, timeout_sec=0.0)
    
    def publish_goal_marker(self, robot_idx, goal_x, goal_y):
        """发布目标点可视化标记"""
        marker_publisher = self.goal_marker_publisher_list[robot_idx]
        
        marker_array = MarkerArray()
        from builtin_interfaces.msg import Time
        # 使用Time(0,0)让TF自动使用最新变换，避免时间同步问题
        current_time = Time(sec=0, nanosec=0)
        
        # 目标点球体标记
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"
        sphere_marker.header.stamp = current_time
        sphere_marker.ns = f"robot_{robot_idx}_goal"
        sphere_marker.id = robot_idx * 10
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        
        # 位置：目标点位置
        sphere_marker.pose.position.x = goal_x
        sphere_marker.pose.position.y = goal_y
        sphere_marker.pose.position.z = 0.3  # 离地面0.3米
        sphere_marker.pose.orientation.w = 1.0
        
        # 大小
        sphere_marker.scale.x = 0.4
        sphere_marker.scale.y = 0.4
        sphere_marker.scale.z = 0.4
        
        # 颜色：根据机器人编号设置不同颜色
        colors = [
            (1.0, 0.0, 0.0),  # 红色 - Robot 0
            (0.0, 1.0, 0.0),  # 绿色 - Robot 1
            (0.0, 0.0, 1.0),  # 蓝色 - Robot 2
            (1.0, 1.0, 0.0),  # 黄色 - Robot 3
        ]
        color = colors[robot_idx % len(colors)]
        sphere_marker.color.r = color[0]
        sphere_marker.color.g = color[1]
        sphere_marker.color.b = color[2]
        sphere_marker.color.a = 0.7
        
        sphere_marker.lifetime.sec = 0  # 永久显示
        sphere_marker.lifetime.nanosec = 0
        marker_array.markers.append(sphere_marker)
        
        # 目标点文本标记（显示机器人编号）
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = current_time
        text_marker.ns = f"robot_{robot_idx}_goal_text"
        text_marker.id = robot_idx * 10 + 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # 位置：目标点上方
        text_marker.pose.position.x = goal_x
        text_marker.pose.position.y = goal_y
        text_marker.pose.position.z = 0.6
        text_marker.pose.orientation.w = 1.0
        
        # 文本内容
        text_marker.text = f"Goal {robot_idx}"
        text_marker.scale.z = 0.2
        
        # 白色文本
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        text_marker.lifetime.sec = 0  # 永久显示
        text_marker.lifetime.nanosec = 0
        marker_array.markers.append(text_marker)
        
        # 箭头标记（从机器人指向目标）
        arrow_marker = Marker()
        arrow_marker.header.frame_id = "map"
        arrow_marker.header.stamp = current_time
        arrow_marker.ns = f"robot_{robot_idx}_goal_arrow"
        arrow_marker.id = robot_idx * 10 + 2
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD
        
        # 箭头的起点和终点
        from geometry_msgs.msg import Point
        start_point = Point()
        start_point.x = self.current_pose_x[robot_idx]
        start_point.y = self.current_pose_y[robot_idx]
        start_point.z = 0.1
        
        end_point = Point()
        end_point.x = goal_x
        end_point.y = goal_y
        end_point.z = 0.1
        
        arrow_marker.points = [start_point, end_point]
        
        # 箭头大小
        arrow_marker.scale.x = 0.05  # 箭头轴直径
        arrow_marker.scale.y = 0.1   # 箭头头部直径
        arrow_marker.scale.z = 0.1   # 箭头头部长度
        
        # 使用与球体相同的颜色，但半透明
        arrow_marker.color.r = color[0]
        arrow_marker.color.g = color[1]
        arrow_marker.color.b = color[2]
        arrow_marker.color.a = 0.5
        
        arrow_marker.lifetime.sec = 0  # 永久显示，每step更新位置
        arrow_marker.lifetime.nanosec = 0
        marker_array.markers.append(arrow_marker)
        
        # 发布标记数组
        marker_publisher.publish_marker_array(marker_array)
    
    def compute_distance_field_for_goal(self, goal_x, goal_y):
        """
        计算到目标的距离场（模拟中央服务器计算并下发）
        
        现实对应：仓库顶部视觉系统实时维护全局占用地图，中央计算
        服务器为每个机器人计算到其目标的距离场并通过WiFi/5G下发。
        
        Args:
            goal_x, goal_y: 目标点在地图坐标系中的位置
            
        Returns:
            distance_field: 2D numpy array，每个格子存储到目标的欧氏距离
        """
        # 如果距离场功能未启用，直接返回None
        if not self.use_distance_field or self.map_subscriber is None:
            return None
        
        # 🔧 关键修复：spin一次确保地图数据更新
        rclpy.spin_once(self.map_subscriber, timeout_sec=0.001)
        
        # 检查缓存（对位置进行四舍五入以提高缓存命中率）
        cache_key = (
            round(goal_x, self.cache_precision),
            round(goal_y, self.cache_precision)
        )
        
        if cache_key in self.distance_field_cache:
            return self.distance_field_cache[cache_key]
        
        # 获取地图数据
        map_data = self.map_subscriber.map_data
        if map_data is None:
            # 如果地图还没有收到，返回零场
            return None
        
        # 将地图转换为二值图像（0=可通行，255=障碍物）
        width = self.map_subscriber.map_width
        height = self.map_subscriber.map_height
        resolution = self.map_subscriber.map_resolution
        origin_x = self.map_subscriber.map_origin_x
        origin_y = self.map_subscriber.map_origin_y
        
        # 地图数据：0-100，0=free, 100=occupied, -1=unknown
        # 转换为二值图像
        map_array = np.array(map_data).reshape(height, width)
        binary_map = np.zeros((height, width), dtype=np.uint8)
        binary_map[map_array > 70] = 1  # 🔧 放宽障碍物阈值：从50改为70
        binary_map[map_array < 0] = 1   # 未知区域也视为障碍物
        
        # 🔍 调试：统计地图占用情况
        if cache_key not in self.distance_field_warning_printed:
            total_cells = width * height
            free_cells = np.sum(binary_map == 0)
            occupied_cells = np.sum(binary_map == 1)
            print(f"[DEBUG] 地图统计: 自由={free_cells}/{total_cells} ({free_cells/total_cells*100:.1f}%), " 
                  f"障碍={occupied_cells}/{total_cells} ({occupied_cells/total_cells*100:.1f}%)", flush=True)
        
        # 将目标点坐标转换为栅格索引
        goal_grid_x = int((goal_x - origin_x) / resolution)
        goal_grid_y = int((goal_y - origin_y) / resolution)
        
        # 确保目标在地图范围内
        if not (0 <= goal_grid_x < width and 0 <= goal_grid_y < height):
            return None
        
        # 🔍 检查目标点是否在障碍物上
        goal_cell_value = map_array[goal_grid_y, goal_grid_x]
        is_goal_on_obstacle = binary_map[goal_grid_y, goal_grid_x] == 1
        
        if is_goal_on_obstacle:
            # 目标点在障碍物上，距离场全为inf（后续会被归一化为1.0）
            if cache_key not in self.distance_field_warning_printed:
                print(f"[WARNING] 目标点 ({goal_x:.2f}, {goal_y:.2f}) 在障碍物上！地图值={goal_cell_value}")
                self.distance_field_warning_printed.add(cache_key)
            return np.ones((height, width), dtype=np.float32) * np.inf
        
        # 🔧 关键修复：使用反向距离变换
        # 不是从目标点计算距离，而是计算每个自由空间点到目标的距离
        # 步骤：
        # 1. 创建一个只在可通行区域计算的掩码
        # 2. 使用距离变换计算到最近障碍物的距离
        # 3. 将目标点作为起点，使用BFS或波前算法计算到目标的距离
        
        # 简化方案：直接使用欧氏距离（忽略障碍物）
        # 创建距离场：每个格子存储到目标的欧氏距离
        y_coords, x_coords = np.ogrid[:height, :width]
        distance_field = np.sqrt((x_coords - goal_grid_x)**2 + (y_coords - goal_grid_y)**2) * resolution
        
        # 将障碍物位置设为无穷大
        distance_field[binary_map == 1] = np.inf
        
        # 🔍 调试：打印距离场统计信息（仅第一次）
        if cache_key not in self.distance_field_warning_printed:
            free_cells = distance_field[binary_map == 0]
            if len(free_cells) > 0:
                print(f"[DEBUG] 目标 ({goal_x:.2f}, {goal_y:.2f}):", flush=True)
                print(f"  - 自由空间距离: min={free_cells.min():.3f}, max={free_cells.max():.3f}, mean={free_cells.mean():.3f}", flush=True)
                print(f"  - 目标栅格: ({goal_grid_x}, {goal_grid_y})", flush=True)
                print(f"  - 目标处地图值: {goal_cell_value}", flush=True)
            self.distance_field_warning_printed.add(cache_key)
        
        # 注意：distance_field已经是实际米距离，不需要再乘resolution
        
        # 存入缓存
        self.distance_field_cache[cache_key] = distance_field
        
        return distance_field
    
    def extract_local_distance_field(self, robot_x, robot_y, distance_field):
        """
        从完整的距离场中提取机器人周围局部区域的距离场
        
        Args:
            robot_x, robot_y: 机器人当前位置（地图坐标系）
            distance_field: 完整的距离场
            
        Returns:
            local_field: 扁平化的局部距离场向量（size x size → flat）
        """
        # 如果距离场功能未启用或数据不可用，返回零向量
        if not self.use_distance_field or distance_field is None or self.map_subscriber is None:
            return np.zeros(self.distance_field_flat_size, dtype=np.float32)
        
        # 获取地图参数
        width = self.map_subscriber.map_width
        height = self.map_subscriber.map_height
        resolution = self.map_subscriber.map_resolution
        origin_x = self.map_subscriber.map_origin_x
        origin_y = self.map_subscriber.map_origin_y
        
        # 将机器人位置转换为栅格索引
        robot_grid_x = int((robot_x - origin_x) / resolution)
        robot_grid_y = int((robot_y - origin_y) / resolution)
        
        # 提取局部区域（例如 7x7）
        half_size = self.distance_field_size // 2
        local_field = np.zeros((self.distance_field_size, self.distance_field_size), dtype=np.float32)
        raw_values = []  # 用于调试
        inf_count = 0    # 无穷大数量
        
        for i in range(self.distance_field_size):
            for j in range(self.distance_field_size):
                # 计算全局栅格索引
                global_x = robot_grid_x + (i - half_size)
                global_y = robot_grid_y + (j - half_size)
                
                # 边界检查
                if 0 <= global_x < width and 0 <= global_y < height:
                    value = distance_field[global_y, global_x]
                    if np.isinf(value):
                        inf_count += 1
                        raw_values.append(999)
                        local_field[i, j] = 1.0  # 障碍物
                    else:
                        raw_values.append(value)
                        # 🔧 归一化到合理范围 [0, 1]，增大范围从10m到20m
                        local_field[i, j] = np.clip(value / 20.0, 0.0, 1.0)
                else:
                    # 边界外视为障碍物
                    local_field[i, j] = 1.0
                    raw_values.append(999)
        
        # 🔍 调试：如果局部距离场全为1.0，打印警告
        flat_field = local_field.flatten()
        if np.all(flat_field == 1.0):
            if not hasattr(self, '_local_field_warning_count'):
                self._local_field_warning_count = 0
            if self._local_field_warning_count < 5:  # 打印前5次
                raw_arr = np.array(raw_values)
                finite_vals = raw_arr[raw_arr < 999]
                print(f"[WARNING] 机器人位置 ({robot_x:.2f}, {robot_y:.2f}) 局部距离场全为1.0!", flush=True)
                print(f"  - 机器人栅格: ({robot_grid_x}, {robot_grid_y})", flush=True)
                print(f"  - 7x7区域中inf(障碍物)数量: {inf_count}/49", flush=True)
                if len(finite_vals) > 0:
                    print(f"  - 有限值距离范围: min={finite_vals.min():.3f}m, max={finite_vals.max():.3f}m, mean={finite_vals.mean():.3f}m", flush=True)
                    if finite_vals.min() > 10.0:
                        print(f"  - 原因：所有有限距离都 > 10m，归一化后全clip到1.0", flush=True)
                    else:
                        print(f"  - 原因未知：有有限值但结果仍为1.0", flush=True)
                else:
                    print(f"  - 原因：7x7区域全是障碍物(inf)或边界外", flush=True)
                self._local_field_warning_count += 1
        
        # 扁平化为一维向量
        return flat_field
    
    # Converts list of arrays to dictionary for MADDPG Algorithm
    def handleReturnValues(self, robotScans, robotRewards, robotDones, truncated, info):
        # Dict of each robots observation
        robot_observations = {}
        robot_rewards= {}
        robot_dones = {}
        robot_truncated = {}
        for i, val in enumerate(robotScans):
            robot_observations['robot'+str(i)] = val
        for i,val in enumerate(robotRewards):
            robot_rewards['robot'+str(i)] = val
        for i,val in enumerate(robotDones):
            robot_dones['robot'+str(i)] = val
        for i,val in enumerate(truncated): 
            robot_truncated['robot'+str(i)] = val
        return robot_observations, robot_rewards, robot_dones, robot_truncated, info
    
    # Updates the variables containing the robots position
    def updateRobotPosition(self):
        for i in range(self.number_of_robots):
            odom_data = None
            odom_subscriber = self.odometry_subscriber_list[i]
            while odom_data is None:
                rclpy.spin_once(odom_subscriber)
                odom_data = odom_subscriber.odom
            self.current_pose_x[i] = odom_data.pose.pose.position.x
            self.current_pose_y[i] = odom_data.pose.pose.position.y

            # 里程计数据有效性/时延检查
            if self.debug_obs_warnings and self._warn_counts['pose_invalid'] < self._warn_limit:
                if (not np.isfinite(self.current_pose_x[i])) or (not np.isfinite(self.current_pose_y[i])):
                    print(f"[WARN] Odom pose invalid for robot {i}: x={self.current_pose_x[i]}, y={self.current_pose_y[i]}")
                    self._warn_counts['pose_invalid'] += 1

            # 时间戳检查（Gazebo使用仿真时间，系统时钟与仿真时钟不同步）
            # 简化检查：只验证odom数据是否被更新（通过比较时间戳变化）
            try:
                stamp = odom_data.header.stamp
                msg_time_sec = stamp.sec + stamp.nanosec * 1e-9
                
                # 检查时间戳是否停滞（与上次相同 = 数据未更新）
                if hasattr(self, '_last_odom_stamp') and i < len(self._last_odom_stamp):
                    last_stamp = self._last_odom_stamp[i]
                    if last_stamp is not None and abs(msg_time_sec - last_stamp) < 1e-6:
                        # 时间戳完全相同，说明odom未更新
                        if self.debug_obs_warnings and self._warn_counts['pose_invalid'] < self._warn_limit:
                            print(f"[WARN] Odom数据未更新 robot {i}: 时间戳={msg_time_sec:.3f}s")
                            self._warn_counts['pose_invalid'] += 1
                
                self._last_odom_stamp[i] = msg_time_sec
            except Exception:
                pass

            # 跳变检查
            last_x, last_y = self._last_pose[i]
            if np.isfinite(last_x) and np.isfinite(last_y):
                jump = math.hypot(self.current_pose_x[i] - last_x, self.current_pose_y[i] - last_y)
                if self.debug_obs_warnings and jump > 1.0 and self._warn_counts['pose_invalid'] < self._warn_limit:
                    print(f"[WARN] Odom jump for robot {i}: Δ={jump:.2f}m")
                    self._warn_counts['pose_invalid'] += 1
            self._last_pose[i] = (self.current_pose_x[i], self.current_pose_y[i])

            # 从四元数计算 yaw（平面运动：只用 z/w）
            qz = odom_data.pose.pose.orientation.z
            qw = odom_data.pose.pose.orientation.w
            self.current_yaw[i] = 2.0 * math.atan2(qz, qw)

            # 🔧 TF变换处理：尝试转换到map坐标系，如果失败则直接使用odom坐标
            # 注意：如果你的目标点（goal）已经在odom坐标系下，可以直接使用odom坐标
            use_tf_transform = False  # 设为True启用TF变换，False则直接使用odom坐标
            
            if use_tf_transform:
                # 尝试将里程计坐标转换到map坐标
                odom_frame = odom_data.header.frame_id if odom_data.header.frame_id else 'odom'
                # 统一frame命名：去掉前导'/'，并处理未命名空间的odom
                if odom_frame.startswith('/'):
                    odom_frame = odom_frame[1:]
                if odom_frame in ('odom', 'base_odom'):
                    odom_frame = f"tb3_{i}/odom"
                try:
                    tf_msg = self.tf_buffer.lookup_transform(self.map_frame, odom_frame, rclpy.time.Time())
                    tx = tf_msg.transform.translation.x
                    ty = tf_msg.transform.translation.y
                    tqz = tf_msg.transform.rotation.z
                    tqw = tf_msg.transform.rotation.w
                    yaw_map_odom = 2.0 * math.atan2(tqz, tqw)

                    x_odom = self.current_pose_x[i]
                    y_odom = self.current_pose_y[i]
                    cos_t = math.cos(yaw_map_odom)
                    sin_t = math.sin(yaw_map_odom)
                    x_map = tx + cos_t * x_odom - sin_t * y_odom
                    y_map = ty + sin_t * x_odom + cos_t * y_odom

                    self.current_pose_x_map[i] = x_map
                    self.current_pose_y_map[i] = y_map
                    self.pose_in_map_valid[i] = True

                    # 使用map坐标进行后续计算（与目标点坐标系一致）
                    self.current_pose_x[i] = x_map
                    self.current_pose_y[i] = y_map
                    self.last_tf_info = {'tx': tx, 'ty': ty, 'yaw': yaw_map_odom, 'ok': True, 'frame': odom_frame}

                    # 打印TF差异信息（限频）
                    if self.debug_obs_warnings and self._tf_warn_count < self._tf_warn_limit:
                        print(f"[INFO] TF map<-{odom_frame}: tx={tx:.3f}, ty={ty:.3f}, yaw={yaw_map_odom:.3f}")
                        self._tf_warn_count += 1
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    # TF变换失败，使用odom坐标
                    if self.debug_obs_warnings and self._warn_counts['pose_invalid'] < self._warn_limit:
                        print(f"[INFO] TF变换不可用，直接使用odom坐标系（Robot {i}）")
                        self._warn_counts['pose_invalid'] += 1
                    self.pose_in_map_valid[i] = False
                    self.last_tf_info = {'tx': 0, 'ty': 0, 'yaw': 0, 'ok': False, 'frame': 'odom'}
            else:
                # 📍 直接使用odom坐标（无TF变换）
                # 这是最常见的情况：目标点和机器人位置都在同一个odom坐标系下
                self.current_pose_x_map[i] = self.current_pose_x[i]
                self.current_pose_y_map[i] = self.current_pose_y[i]
                self.pose_in_map_valid[i] = True
                self.last_tf_info = {'tx': 0, 'ty': 0, 'yaw': 0, 'ok': True, 'frame': 'odom_direct'}
                
                if self.debug_obs_warnings and self._tf_warn_count == 0:
                    print(f"[INFO] 使用odom坐标系（无TF变换）- 确保目标点也在odom坐标系下")
                    self._tf_warn_count += 1
            
            # 🔍 记录时间戳用于同步验证
            if self.enable_sync_validation:
                try:
                    stamp = odom_data.header.stamp
                    msg_time_ns = stamp.sec * 1e9 + stamp.nanosec
                    self.data_timestamps['odom'][i] = msg_time_ns
                except Exception:
                    self.data_timestamps['odom'][i] = None
    
    def _discover_tf_frames(self):
        """发现TF树中可用的frames"""
        try:
            # 给TF buffer一些时间积累数据
            time.sleep(0.5)
            
            # 尝试获取TF树的所有frames
            all_frames = self.tf_buffer.all_frames_as_string()
            print(f"\n{'='*80}")
            print("🔍 TF树诊断信息：")
            print(all_frames)
            print(f"{'='*80}\n")
            
            # 解析frames（简单版本）
            for line in all_frames.split('\n'):
                if 'Frame' in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        frame_name = parts[1].strip(':')
                        self.available_tf_frames.add(frame_name)
            
            self.tf_frames_discovered = True
            
        except Exception as e:
            print(f"[WARN] 无法发现TF frames: {e}")
    
    def _validate_data_sync(self, detailed=False):
        """验证数据同步性 - 使用相对时间戳验证"""
        if not self.enable_sync_validation:
            return True
        
        issues = []
        
        # 检查各传感器数据是否更新
        for i in range(self.number_of_robots):
            # Odom检查 - 时间戳是否变化
            if self.data_timestamps['odom'][i] is not None:
                last_ts = self.last_validated_timestamps['odom'][i]
                current_ts = self.data_timestamps['odom'][i]
                
                if last_ts is not None and current_ts == last_ts:
                    # 时间戳未变化
                    self.stale_data_counters['odom'][i] += 1
                    if self.stale_data_counters['odom'][i] >= self.stale_data_threshold_steps:
                        issues.append(f"Robot{i} odom未更新: 已{self.stale_data_counters['odom'][i]}步")
                else:
                    # 时间戳已变化，重置计数器
                    self.stale_data_counters['odom'][i] = 0
                
                self.last_validated_timestamps['odom'][i] = current_ts
            else:
                issues.append(f"Robot{i} odom无时间戳")
            
            # Scan检查 - 时间戳是否变化
            if self.data_timestamps['scan'][i] is not None:
                last_ts = self.last_validated_timestamps['scan'][i]
                current_ts = self.data_timestamps['scan'][i]
                
                if last_ts is not None and current_ts == last_ts:
                    # 时间戳未变化
                    self.stale_data_counters['scan'][i] += 1
                    if self.stale_data_counters['scan'][i] >= self.stale_data_threshold_steps:
                        issues.append(f"Robot{i} scan未更新: 已{self.stale_data_counters['scan'][i]}步")
                else:
                    # 时间戳已变化，重置计数器
                    self.stale_data_counters['scan'][i] = 0
                
                self.last_validated_timestamps['scan'][i] = current_ts
            else:
                issues.append(f"Robot{i} scan无时间戳")
        
        if detailed and len(issues) > 0:
            print(f"\n⚠️  数据同步问题 (Step {self.step_counter}):")
            for issue in issues:
                print(f"   - {issue}")
        
        return len(issues) == 0
    
    def _print_sync_report(self):
        """打印详细的同步报告"""
        print(f"\n{'='*80}")
        print(f"📊 数据同步报告 (Episode {self.episode_counter}, Step {self.step_counter})")
        print(f"{'='*80}")
        
        # Odom数据
        print(f"\n📍 里程计数据:")
        for i in range(self.number_of_robots):
            if self.data_timestamps['odom'][i] is not None:
                ts_sec = self.data_timestamps['odom'][i] / 1e9
                # 检查更新状态
                update_status = "✅ 正常更新" if self.stale_data_counters['odom'][i] == 0 else f"⚠️ {self.stale_data_counters['odom'][i]}步未更新"
                print(f"  Robot{i}: {update_status}, 时间戳={ts_sec:.3f}s, 位置=({self.current_pose_x[i]:.2f}, {self.current_pose_y[i]:.2f})")
            else:
                print(f"  Robot{i}: ❌ 无数据")
        
        # Scan数据
        print(f"\n📡 激光雷达数据:")
        for i in range(self.number_of_robots):
            if self.data_timestamps['scan'][i] is not None:
                ts_sec = self.data_timestamps['scan'][i] / 1e9
                scan_data = self.scan_subscriber_list[i]
                min_dist = np.min(scan_data.scan.ranges) if scan_data.scan is not None else np.nan
                # 检查更新状态
                update_status = "✅ 正常更新" if self.stale_data_counters['scan'][i] == 0 else f"⚠️ {self.stale_data_counters['scan'][i]}步未更新"
                print(f"  Robot{i}: {update_status}, 时间戳={ts_sec:.3f}s, 最近障碍={min_dist:.2f}m")
            else:
                print(f"  Robot{i}: ❌ 无数据")
        
        # 目标位置
        print(f"\n🎯 目标位置:")
        for i in range(min(self.number_of_robots, len(self.current_goal_locations))):
            goal_x, goal_y = self.current_goal_locations[i]
            dist = math.hypot(goal_x - self.current_pose_x[i], goal_y - self.current_pose_y[i])
            print(f"  Robot{i}: 目标=({goal_x:.2f}, {goal_y:.2f}), 距离={dist:.2f}m")
        
        # 速度命令
        print(f"\n🎮 当前速度命令:")
        for i in range(self.number_of_robots):
            print(f"  Robot{i}: v={self.current_linear_velocity[i]:.3f}m/s, w={self.current_angular_velocity[i]:.3f}rad/s")
        
        print(f"{'='*80}\n")
    
    def _verify_observation_reward_correspondence(self, robot_scans, full_observations, rewards):
        """验证观测与奖励的对应关系"""
        print(f"\n{'='*80}")
        print(f"🔍 观测-奖励对应验证 (Episode {self.episode_counter}, Step {self.step_counter})")
        print(f"{'='*80}")
        
        for i in range(self.number_of_robots):
            print(f"\n🤖 Robot {i}:")
            print(f"{'-'*70}")
            
            # ⚠️ 使用与奖励计算相同的数据源和过滤逻辑
            scan_subscriber = self.scan_subscriber_list[i]
            min_dist = 3.5
            valid_scan_count = 0
            scan_mean = 3.5
            
            if scan_subscriber.scan is not None:
                # 获取有效距离（过滤 0.0 和 inf）- 与 getRewards() 中逻辑一致
                valid_ranges = [r for r in scan_subscriber.scan.ranges if r > 0.05 and not math.isinf(r)]
                if len(valid_ranges) > 0:
                    min_dist = min(valid_ranges)
                    scan_mean = np.mean(valid_ranges)
                    valid_scan_count = len(valid_ranges)
            
            # 从full_observations获取速度和目标信息
            obs = full_observations[i]
            obs_linear_vel = obs[38]   # 线速度
            obs_angular_vel = obs[39]  # 角速度
            obs_goal_dx = obs[40]      # 目标相对x (机器人坐标系)
            obs_goal_dy = obs[41]      # 目标相对y
            obs_goal_dist = obs[42]    # 目标距离（归一化）
            
            # 获取奖励分量
            reward_comp = self.last_reward_components[i]
            
            print(f"📊 观测数据:")
            print(f"  激光雷达: 最小={min_dist:.2f}m, 平均={scan_mean:.2f}m, 有效点数={valid_scan_count}")
            if min_dist == 3.5:
                print(f"    ⚠️  使用默认值3.5m (无有效激光数据或scan为None)")
            print(f"  速度: v={obs_linear_vel:.3f}m/s, w={obs_angular_vel:.3f}rad/s")
            print(f"  目标: dx={obs_goal_dx:.2f}, dy={obs_goal_dy:.2f}, dist={obs_goal_dist:.2f}(归一化)")
            
            # 实际位置和目标
            if i < len(self.current_goal_locations):
                goal_x, goal_y = self.current_goal_locations[i]
                robot_x, robot_y = self.current_pose_x[i], self.current_pose_y[i]
                actual_dist = math.hypot(goal_x - robot_x, goal_y - robot_y)
                print(f"  实际目标距离: {actual_dist:.2f}m")
            
            print(f"\n💰 奖励分量:")
            # 适配简化的奖励函数（只有total和obs_distance）
            if 'r_base' in reward_comp:
                # 详细奖励分量（原始版本）
                print(f"  r_base    = {reward_comp['r_base']:+.3f}  (基础存活奖励)")
                print(f"  r_action  = {reward_comp['r_action']:+.3f}  (速度方向奖励)")
                print(f"  r_heading = {reward_comp['r_heading']:+.3f}  (朝向奖励)")
                print(f"  r_obstacle= {reward_comp['r_obstacle']:+.3f}  (障碍物惩罚)")
                print(f"  r_goal    = {reward_comp['r_goal']:+.3f}  (目标进展)")
                print(f"  r_time    = {reward_comp['r_time']:+.3f}  (时间惩罚)")
            else:
                # 简化奖励分量（当前版本）
                print(f"  使用简化奖励函数")
                print(f"  观测距离: {reward_comp.get('obs_distance', 0):.2f}m")
            print(f"  {'─'*50}")
            print(f"  Total     = {reward_comp['total']:+.3f}")
            
            # 🔍 逻辑一致性验证
            print(f"\n🔍 逻辑验证:")
            issues = []
            
            # 根据奖励函数类型进行不同的验证
            if 'r_obstacle' in reward_comp:
                # 详细奖励函数 - 检查各个分量
                # 1. 障碍物检测 vs 障碍物惩罚
                obstacle_threshold = 1.0
                if min_dist < obstacle_threshold:
                    expected_penalty = -2.0 * (obstacle_threshold - min_dist)
                    actual_penalty = reward_comp['r_obstacle']
                    if abs(expected_penalty - actual_penalty) > 0.01:
                        issues.append(f"⚠️  障碍距离{min_dist:.2f}m, 预期惩罚{expected_penalty:.2f}, 实际{actual_penalty:.2f}")
                    else:
                        print(f"  ✅ 障碍物惩罚正确: dist={min_dist:.2f}m → r_obs={actual_penalty:.2f}")
                else:
                    if reward_comp['r_obstacle'] != 0.0:
                        issues.append(f"⚠️  障碍物远({min_dist:.2f}m > {obstacle_threshold}m)但有惩罚({reward_comp['r_obstacle']:.2f})")
                    else:
                        print(f"  ✅ 远离障碍: dist={min_dist:.2f}m, 无惩罚")
                
                # 2. 速度 vs 动作奖励（检测原地转圈）
                norm_v = obs_linear_vel / 0.25
                norm_w = obs_angular_vel / 0.5
                
                if abs(norm_v) < 0.2 and abs(norm_w) > 0.1:
                    if reward_comp.get('r_action', 0) > -0.3:
                        issues.append(f"⚠️  原地转圈(v={obs_linear_vel:.2f}, w={obs_angular_vel:.2f})但惩罚不足({reward_comp['r_action']:.2f})")
                    else:
                        print(f"  ✅ 原地转圈惩罚: v={obs_linear_vel:.2f}, w={obs_angular_vel:.2f} → r={reward_comp['r_action']:.2f}")
                elif norm_v > 0.3:
                    print(f"  ✅ 前进运动: v={obs_linear_vel:.2f}m/s, r_action={reward_comp['r_action']:.2f}")
            else:
                # 简化奖励函数 - 只检查异常值
                print(f"  使用简化奖励函数")
                if reward_comp['total'] > 10.0:
                    issues.append(f"⚠️  奖励异常高: {reward_comp['total']:.3f}")
                elif reward_comp['total'] < -10.0:
                    issues.append(f"⚠️  奖励异常低: {reward_comp['total']:.3f}")
                else:
                    print(f"  ✅ 奖励在合理范围: {reward_comp['total']:.3f}")
            
            # 3. 目标距离变化 vs 目标奖励（仅详细奖励函数）
            if 'r_goal' in reward_comp and 'obs_distance' in reward_comp:
                obs_dist_raw = reward_comp['obs_distance']
                
                # 检测"刷分漏洞"：原地转圈但获得目标奖励
                if reward_comp['r_goal'] > 0.05 and abs(obs_linear_vel) < 0.05:
                    issues.append(f"🚨 刷分漏洞！原地转(v={obs_linear_vel:.3f})却获得目标奖励({reward_comp['r_goal']:+.2f})")
                elif reward_comp['r_goal'] > 0.05:
                    print(f"  ✅ 接近目标: v={obs_linear_vel:.2f}m/s, 距离={obs_dist_raw:.2f}m, r_goal={reward_comp['r_goal']:+.2f}")
                elif reward_comp['r_goal'] < -0.05:
                    issues.append(f"⚠️  远离目标: v={obs_linear_vel:.2f}m/s, 距离={obs_dist_raw:.2f}m, r_goal={reward_comp['r_goal']:+.2f}")
            
            # 4. 朝向一致性（仅详细奖励函数）
            if 'obs_goal_cos' in reward_comp and 'r_heading' in reward_comp:
                goal_cos = reward_comp['obs_goal_cos']
                if goal_cos > 0.7 and obs_linear_vel > 0.1:
                    print(f"  ✅ 朝向并前进: cos={goal_cos:.2f}, r_heading={reward_comp['r_heading']:.2f}")
                elif goal_cos < -0.3:
                    print(f"  ⚠️  背向目标: cos={goal_cos:.2f}, r_heading={reward_comp['r_heading']:.2f}")
                else:
                    print(f"  ➡️  朝向: cos={goal_cos:.2f}")
            
            # 打印所有异常
            if len(issues) > 0:
                print(f"\n  🚨 发现{len(issues)}个潜在问题:")
                for issue in issues:
                    print(f"     {issue}")
            else:
                print(f"  ✅ 所有逻辑验证通过！")
        
        print(f"\n{'='*80}\n")
    
    # Adds linear and angular velocities to the scan observation
    def addVelocitiesToObs(self, scans):
        # 添加速度、目标信息、其他机器人信息、距离场到观测（针对仓库协作导航）
        # scans 现在是38维的激光雷达数据，需要扩展到 98 维（如果使用距离场）
        full_observations = []
        
        for i in range(self.number_of_robots):
            # 位置有效性检查
            if self.debug_obs_warnings and self._warn_counts['pose_invalid'] < self._warn_limit:
                if (not np.isfinite(self.current_pose_x[i])) or (not np.isfinite(self.current_pose_y[i])):
                    print(f"[WARN] Pose invalid for robot {i}: x={self.current_pose_x[i]}, y={self.current_pose_y[i]}")
                    self._warn_counts['pose_invalid'] += 1

            robot_x = self.current_pose_x[i]
            robot_y = self.current_pose_y[i]
            yaw = float(self.current_yaw[i]) if i < len(self.current_yaw) else 0.0
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            # 创建完整观测向量
            full_obs = np.zeros(self.single_robot_observation_space, dtype=np.float32)
            
            # 1. 复制前38维的激光雷达数据
            full_obs[:38] = scans[i]

            # 雷达有效性检查
            if self.debug_obs_warnings and self._warn_counts['lidar_invalid'] < self._warn_limit:
                if (not np.isfinite(full_obs[:38]).all()) or (np.min(full_obs[:38]) < 0):
                    print(f"[WARN] Lidar invalid for robot {i}: min={np.min(full_obs[:38])}, max={np.max(full_obs[:38])}")
                    self._warn_counts['lidar_invalid'] += 1
            
            # 2. 添加自身速度（38-39维）
            full_obs[38] = self.current_linear_velocity[i]
            full_obs[39] = self.current_angular_velocity[i]
            
            # 3. 添加自己的目标信息（40-42维）
            if i < len(self.current_goal_locations):
                goal_x, goal_y = self.current_goal_locations[i]
                
                # 计算相对位置
                dx = goal_x - robot_x
                dy = goal_y - robot_y
                distance = math.sqrt(dx**2 + dy**2)

                # 把目标相对向量转到机器人坐标系，避免目标信息缺少朝向造成打转
                dx_r = cos_yaw * dx + sin_yaw * dy
                dy_r = -sin_yaw * dx + cos_yaw * dy
                
                # 归一化到合理范围 [-1, 1]（假设最大距离20米）
                full_obs[40] = np.clip(dx_r / 20.0, -1.0, 1.0)
                full_obs[41] = np.clip(dy_r / 20.0, -1.0, 1.0)
                full_obs[42] = np.clip(distance / 20.0, 0.0, 1.0)
            
            # 4. 添加最近其他机器人的信息（43-48维，共6维）
            # 找到距离当前机器人最近的其他机器人
            min_dist = float('inf')
            nearest_robot_idx = -1
            
            for j in range(self.number_of_robots):
                if i != j:  # 不包括自己
                    other_x = self.current_pose_x[j]
                    other_y = self.current_pose_y[j]
                    dist = math.sqrt((robot_x - other_x)**2 + (robot_y - other_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_robot_idx = j
            
            # 如果找到了其他机器人
            if nearest_robot_idx >= 0:
                other_x = self.current_pose_x[nearest_robot_idx]
                other_y = self.current_pose_y[nearest_robot_idx]
                other_vx = self.current_linear_velocity[nearest_robot_idx]
                other_vz = self.current_angular_velocity[nearest_robot_idx]
                
                # 相对位置
                rel_x = other_x - robot_x
                rel_y = other_y - robot_y
                rel_dist = math.sqrt(rel_x**2 + rel_y**2)
                rel_angle = math.atan2(rel_y, rel_x)

                # 转到机器人坐标系（与目标信息保持一致）
                rel_x_r = cos_yaw * rel_x + sin_yaw * rel_y
                rel_y_r = -sin_yaw * rel_x + cos_yaw * rel_y
                
                # 归一化并存储
                full_obs[43] = np.clip(rel_x_r / 10.0, -1.0, 1.0)  # 相对x(机器人系)
                full_obs[44] = np.clip(rel_y_r / 10.0, -1.0, 1.0)  # 相对y(机器人系)
                full_obs[45] = np.clip(other_vx / 0.5, -1.0, 1.0)  # 其他机器人线速度
                full_obs[46] = np.clip(other_vz / 2.0, -1.0, 1.0)  # 其他机器人角速度
                full_obs[47] = np.clip(rel_dist / 10.0, 0.0, 1.0)  # 距离
                full_obs[48] = rel_angle / math.pi  # 角度 [-1, 1]
            
            # 5. 添加局部距离场（49-97维，共49维 = 7x7）
            # 📡 模拟：中央服务器根据全局地图和机器人目标计算距离场并下发
            if self.use_distance_field and i < len(self.current_goal_locations):
                goal_x, goal_y = self.current_goal_locations[i]
                
                # 中央服务器计算该机器人到目标的距离场
                distance_field = self.compute_distance_field_for_goal(goal_x, goal_y)
                
                # 提取机器人周围7x7局部区域（减少通信量）
                local_field = self.extract_local_distance_field(robot_x, robot_y, distance_field)
                
                # 添加到观测向量中
                full_obs[49:49+self.distance_field_flat_size] = local_field
            
            full_observations.append(full_obs)

            # 观测向量有效性检查
            if self.debug_obs_warnings and self._warn_counts['obs_nan'] < self._warn_limit:
                if not np.isfinite(full_obs).all():
                    print(f"[WARN] Observation contains NaN/Inf for robot {i}")
                    self._warn_counts['obs_nan'] += 1
        
        return full_observations 
    
    def end_of_episode_functions(self, robot_scans):
        # Quickly update position variables of robots then reset velocities
        self.updateRobotPosition()
        self.reset_cmd_vel()
        # Add the velocities to the end of observation and return full observations
        full_obs = self.addVelocitiesToObs(robot_scans)
        return full_obs    
        
    # Steps the environment, (Reinforcement Learning term, it means - do this every time step)
    def step(self, action):
        # 第一次step时打印确认（使用flush确保立即输出）
        if self.total_step_counter == 0 and self.enable_interaction_logging:
            print(f"\n{'='*80}", flush=True)
            print(f"[DEBUG] 🎯 第一次step()调用！", flush=True)
            print(f"[DEBUG] 日志文件: {self.interaction_log_file}", flush=True)
            print(f"[DEBUG] enable_interaction_logging: {self.enable_interaction_logging}", flush=True)
            print(f"{'='*80}\n", flush=True)
        
        # 保存当前观测（作为state用于日志记录）
        if self.enable_interaction_logging:
            # 获取当前观测
            robot_scans_before = []
            for i in range(self.number_of_robots):
                scan_data = self.scan_subscriber_list[i]
                rclpy.spin_once(scan_data)
                if scan_data.scan is not None:
                    robot_scans_before.append(self.resize_lidar(scan_data.scan))
                else:
                    # 使用上一次有效scan，避免误判碰撞
                    if getattr(scan_data, 'last_scan', None) is not None:
                        robot_scans_before.append(self.resize_lidar(scan_data.last_scan))
                    else:
                        robot_scans_before.append(np.full(38, 3.5, dtype=np.float32))
                        if self.debug_obs_warnings and self._warn_counts['lidar_missing'] < self._warn_limit:
                            print(f"[WARN] LaserScan missing (pre-step) for robot {i}")
                            self._warn_counts['lidar_missing'] += 1
            full_obs_before_step = self.addVelocitiesToObs(robot_scans_before)
            # 转换为dict
            obs_dict_before = {}
            for idx, obs in enumerate(full_obs_before_step):
                obs_dict_before[f'robot{idx}'] = obs
            full_obs_before_step = obs_dict_before
        
        # 更新位置
        self.updateRobotPosition()

        # 执行动作：将网络输出映射到实际速度
        for i in range(self.number_of_robots):
            name = 'robot'+str(i)
            
            # 1. 获取原始动作 [-1, 1]
            raw_linear = float(action[name][0])   # 控制前后
            raw_angular = float(action[name][1])  # 控制左右

            # 允许倒车脱困：raw_linear∈[-1,1] -> [min_linear_vel, max_linear_vel]
            target_linear = ((raw_linear + 1.0) / 2.0) * (self.max_linear_vel - self.min_linear_vel) + self.min_linear_vel
            target_angular = raw_angular * self.max_angular_vel
            
            # 4. 平滑控制 (可选，防止动作突变导致“抽搐”)
            # 使用简单的加权平均：新速度 = 0.6 * 上次速度 + 0.4 * 目标速度
            self.current_linear_velocity[i] = 0.6 * self.current_linear_velocity[i] + 0.4 * target_linear
            self.current_angular_velocity[i] = 0.6 * self.current_angular_velocity[i] + 0.4 * target_angular

            # 发布给 Gazebo
            cmd_vel_publisher = self.cmd_vel_publisher_list[i]
            desired_vel_cmd = Twist()
            desired_vel_cmd.linear.x = float(self.current_linear_velocity[i])
            desired_vel_cmd.angular.z = float(self.current_angular_velocity[i])
            cmd_vel_publisher.cmd_vel = desired_vel_cmd
            cmd_vel_publisher.pub_vel()

        # 等待物理仿真更新（按仿真时钟节拍推进）
        try:
            self.sim_clock.sleep_for(Duration(seconds=self.sim_step_sec))
        except Exception:
            # 回退到墙钟，避免无仿真时钟时卡住
            time.sleep(self.sim_step_sec)
        
        # 更新位置
        self.updateRobotPosition()

        # --- 观测和奖励计算 ---

        # 读取最新的雷达数据
        robot_scans = []
        for i in range(self.number_of_robots):
            scan_data = self.scan_subscriber_list[i]
            rclpy.spin_once(scan_data)
            if scan_data.scan is not None:
                robot_scans.append(self.resize_lidar(scan_data.scan))
                
                # 🔍 记录scan时间戳
                if self.enable_sync_validation:
                    try:
                        stamp = scan_data.scan.header.stamp
                        msg_time_ns = stamp.sec * 1e9 + stamp.nanosec
                        self.data_timestamps['scan'][i] = msg_time_ns
                    except Exception:
                        self.data_timestamps['scan'][i] = None
            else:
                # 使用上一次有效scan，避免误判碰撞
                if getattr(scan_data, 'last_scan', None) is not None:
                    robot_scans.append(self.resize_lidar(scan_data.last_scan))
                else:
                    robot_scans.append(np.full(38, 3.5, dtype=np.float32))
                    if self.debug_obs_warnings and self._warn_counts['lidar_missing'] < self._warn_limit:
                        print(f"[WARN] LaserScan missing (step) for robot {i}")
                        self._warn_counts['lidar_missing'] += 1
        
        # 🔍 数据同步验证
        if self.enable_sync_validation:
            # 每N步做一次详细验证
            detailed = (self.step_counter % self.sync_validation_interval == 0)
            sync_ok = self._validate_data_sync(detailed=detailed)
            
            # 每N步打印完整报告
            if detailed:
                self._print_sync_report()

        # 检查终止条件
        truncated = [False] * self.number_of_robots
        if self.step_counter + 1 > self.MAX_STEPS:
            truncated = [True] * self.number_of_robots

        # 检查碰撞和到达
        collided = np.full(self.number_of_robots, False)
        reachedGoal = np.full(self.number_of_robots, False)
        for i in range(self.number_of_robots):
            collided[i] = self.hasCollided(robot_scans[i], robot_number=i)
            reachedGoal[i] = self.hasReachedGoal(robot_scans[i], i)

        # 计算奖励
        rewards = self.getRewards()

        # 检查并更新路径点，给予额外奖励
        if hasattr(self, 'use_global_planner') and self.use_global_planner:
            for i, robot_key in enumerate(self.robot_names):
                waypoint_reached = self._check_and_update_waypoint(i)
                if waypoint_reached and robot_key in rewards:
                    rewards[robot_key] += 0.5  # 到达路径点额外奖励

        dones = [False] * self.number_of_robots

        # 处理碰撞
        if any(collided):
            self.logger.log('Collision detected!')
            for idx in np.nonzero(collided)[0]:
                rewards[idx] += self.collisionReward
            dones = [True] * self.number_of_robots
            full_obs = self.end_of_episode_functions(robot_scans)
            if self.enable_interaction_logging:
                next_obs_dict = {}
                for idx, obs in enumerate(full_obs):
                    next_obs_dict[f'robot{idx}'] = obs
                self._log_interaction(action, full_obs_before_step, rewards, dones, truncated, 
                                    next_observation=next_obs_dict, event='collision')
            return self.handleReturnValues(full_obs, rewards, dones, truncated, {
                'event': 'collision',
                'reward_components': self.last_reward_components,
                'tf_info': self.last_tf_info
            })

        # 处理到达目标
        goal_reached = False
        if self.goal_termination_mode == 'all':
            goal_reached = all(reachedGoal)
        else:
            goal_reached = any(reachedGoal)

        if goal_reached:
            self.logger.log('Goal reached!')
            # 给真正到达的机器人加奖励（其余机器人不强行+200，避免奖励污染）
            for i in range(self.number_of_robots):
                if reachedGoal[i]:
                    rewards[i] += self.goalReward
            dones = [True] * self.number_of_robots
            full_obs = self.end_of_episode_functions(robot_scans)
            if self.enable_interaction_logging:
                next_obs_dict = {}
                for idx, obs in enumerate(full_obs):
                    next_obs_dict[f'robot{idx}'] = obs
                self._log_interaction(action, full_obs_before_step, rewards, dones, truncated, 
                                    next_observation=next_obs_dict, event='goal_reached')
            return self.handleReturnValues(full_obs, rewards, dones, truncated, {
                'event': 'goal_reached',
                'reward_components': self.last_reward_components,
                'tf_info': self.last_tf_info
            })

        # 处理卡住（无进展）：提前截断避免长时间打转
        if self.stuck_enabled and self.step_counter >= self.stuck_check_after_steps:
            stuck_flags = np.full(self.number_of_robots, False)
            for i in range(self.number_of_robots):
                if i >= len(self.current_goal_locations):
                    continue
                goal_x, goal_y = self.current_goal_locations[i]
                current_distance = math.hypot(goal_x - self.current_pose_x[i], goal_y - self.current_pose_y[i])

                last_d = self._stuck_last_distance[i]
                if last_d is None:
                    self._stuck_last_distance[i] = current_distance
                    self._stuck_steps[i] = 0
                    continue

                progress = last_d - current_distance
                if progress > self.stuck_min_progress:
                    self._stuck_steps[i] = 0
                else:
                    self._stuck_steps[i] += 1
                self._stuck_last_distance[i] = current_distance

                if self._stuck_steps[i] >= self.stuck_max_steps:
                    stuck_flags[i] = True

            if any(stuck_flags):
                for i in range(self.number_of_robots):
                    if stuck_flags[i]:
                        rewards[i] += self.stuck_penalty

                truncated = [True] * self.number_of_robots
                full_obs = self.end_of_episode_functions(robot_scans)
                if self.enable_interaction_logging:
                    next_obs_dict = {}
                    for idx, obs in enumerate(full_obs):
                        next_obs_dict[f'robot{idx}'] = obs
                    self._log_interaction(action, full_obs_before_step, rewards, dones, truncated,
                                        next_observation=next_obs_dict, event='stuck')
                return self.handleReturnValues(full_obs, rewards, dones, truncated, {
                    'event': 'stuck',
                    'reward_components': self.last_reward_components,
                    'tf_info': self.last_tf_info
                })
        
        # 处理超时
        if any(truncated):
            full_obs = self.end_of_episode_functions(robot_scans)
            if self.enable_interaction_logging:
                next_obs_dict = {}
                for idx, obs in enumerate(full_obs):
                    next_obs_dict[f'robot{idx}'] = obs
                self._log_interaction(action, full_obs_before_step, rewards, dones, truncated, 
                                    next_observation=next_obs_dict, event='timeout')
            return self.handleReturnValues(full_obs, rewards, dones, truncated, {
                'event': 'timeout',
                'reward_components': self.last_reward_components,
                'tf_info': self.last_tf_info
            })

        # 正常返回
        full_obs = self.addVelocitiesToObs(robot_scans)
        self.step_counter += 1
        
        # 🔍 观测-奖励对应验证（每10步打印一次）
        if self.step_counter % 10 == 0:
            self._verify_observation_reward_correspondence(robot_scans, full_obs, rewards)
        
        # ===== 记录交互数据（七元组） =====
        if self.enable_interaction_logging:
            # 将full_obs转为dict格式作next_observation
            next_obs_dict = {}
            for idx, obs in enumerate(full_obs):
                next_obs_dict[f'robot{idx}'] = obs
            self._log_interaction(action, full_obs_before_step, rewards, dones, truncated, 
                                next_observation=next_obs_dict, event='normal')
        
        return self.handleReturnValues(full_obs, rewards, dones, truncated, {
            'event': 'normal',
            'reward_components': self.last_reward_components,
            'tf_info': self.last_tf_info
        })
    
    # Resets the environment, gets initial observations and returns robots back to there original poses
    def reset(self):
        self.step_counter = 0
        self.episode_counter += 1  # 增加episode计数
        
        # ============ 清除旧的marker ============
        # 在每个episode开始时删除旧marker，确保显示最新状态
        for i in range(self.number_of_robots):
            self.clear_markers(i)
        
        # 第一次reset时不重置机器人位置，使用环境中已有的位置
        if not self.first_reset:
            self.restart_environment.reset_robots()
        else:
            # 第一次 reset：更新机器人位置信息
            self.updateRobotPosition()
            
            # 如果是固定模式且还没记录位置，记录当前位置
            if not self.use_random_mode:
                robot_positions = []
                for i in range(self.number_of_robots):
                    # 获取当前位置和朝向
                    odom_data = None
                    odom_subscriber = self.odometry_subscriber_list[i]
                    while odom_data is None:
                        rclpy.spin_once(odom_subscriber)
                        odom_data = odom_subscriber.odom
                    
                    x = odom_data.pose.pose.position.x
                    y = odom_data.pose.pose.position.y
                    # 从四元数计算 yaw
                    qz = odom_data.pose.pose.orientation.z
                    qw = odom_data.pose.pose.orientation.w
                    yaw = 2 * math.atan2(qz, qw)
                    
                    robot_positions.append((x, y, yaw))
                
                # 记录初始位置
                self.restart_environment.record_initial_positions(robot_positions)
            
            self.first_reset = False
            
        self.updateRobotPosition()

        # When reset function is first called, we need to initialise the goal entities
        if self.initGoal:
            print("[DEBUG] First reset - calling spawn_goals()")
            # spawn multiple goals (one for each robot)
            self.current_goal_locations = self.restart_environment.spawn_goals()
            print(f"[DEBUG] spawn_goals() returned {len(self.current_goal_locations)} locations")
            for idx, loc in enumerate(self.current_goal_locations):
                print(f"[DEBUG]   Goal {idx}: ({loc[0]:.2f}, {loc[1]:.2f})")
            self.initGoal = False
        else:
            # 非第一次reset：如果是随机模式，移动目标到新的随机位置
            if self.use_random_mode:
                print("[DEBUG] Random mode - calling move_goals()")
                self.current_goal_locations = self.restart_environment.move_goals()
                # 目标改变，清空距离场缓存
                self.distance_field_cache.clear()

        # 为每个机器人规划全局路径（必须在目标生成之后）
        if hasattr(self, 'use_global_planner') and self.use_global_planner:
            for i in range(self.number_of_robots):
                start_pos = (self.current_pose_x[i], self.current_pose_y[i])
                if i < len(self.current_goal_locations):
                    goal_pos = self.current_goal_locations[i]
                    print(f"[DEBUG] 规划路径: Robot {i} from {start_pos} to {goal_pos}")
                    self._plan_global_path(i, start_pos, goal_pos)
        
        # If robots have reached the goal node x times change location of goals
        if self.reached_goal_counter > 50:
            #time.sleep(2)
            msg = 'Found Goal, The robots have found the goal: ' + str(self.total_goal_counter) + ' times'
            self.logger.log(msg)

            self.current_goal_locations = self.restart_environment.move_goals()
            # 目标改变，清空距离场缓存
            self.distance_field_cache.clear()
            self.reached_goal_counter = 0
        
        # 初始化进步奖励所需的上一步距离
        for i in range(self.number_of_robots):
            if i < len(self.current_goal_locations):
                goal_pos = self.current_goal_locations[i]
                self.previous_distance_to_goal[i] = math.hypot(
                    goal_pos[0] - self.current_pose_x[i],
                    goal_pos[1] - self.current_pose_y[i]
                )

        # 初始化 stuck detector
        self._stuck_steps = np.zeros(self.number_of_robots, dtype=np.int32)
        self._stuck_last_distance = [None] * self.number_of_robots
        for i in range(self.number_of_robots):
            if i < len(self.current_goal_locations):
                goal_pos = self.current_goal_locations[i]
                self._stuck_last_distance[i] = math.hypot(
                    goal_pos[0] - self.current_pose_x[i],
                    goal_pos[1] - self.current_pose_y[i]
                )
            
        # This function wont make sense right now (could add this to observation space, or reward but it changes the whole
        # concept of only relying on lidar) ---- TODO 
        self.goal_distance = self.getGoalDistace()
        
        # Read lidar scans from all robots
        robot_scans = []
        for i in range(self.number_of_robots):
            data = None
            scan_data = self.scan_subscriber_list[i]

            while data is None:
                rclpy.spin_once(scan_data)
                #scan_data.get_logger().info("Reading data")
                data = scan_data.scan
            robot_scans.append(data)

        # Pass all lidar scans to getState function,  It resizes the lidar data and tells us if episode is done or truncated
        resized_scans = []
        for i in range(self.number_of_robots):
            rscan = robot_scans[i]
            lidar_data = self.resize_lidar(rscan)
            resized_scans.append(lidar_data)

        # Dict of each robots observation
        robot_observations = {}

        obs = self.addVelocitiesToObs(resized_scans)
        for i, val in enumerate(obs):
            robot_observations['robot'+str(i)] = val
        
        # ============ 重要：Reset后重新发布目标marker ============
        # 确保每个episode开始时目标marker都是最新的
        for i in range(self.number_of_robots):
            if i < len(self.current_goal_locations):
                goal_x, goal_y = self.current_goal_locations[i]
                self.publish_goal_marker(i, goal_x, goal_y)
        
        return robot_observations
    
    def _initialize_planner_after_map_loaded(self):
        """地图加载后初始化A*规划器"""
        if not hasattr(self, 'use_global_planner') or not self.use_global_planner:
            print("[DEBUG] 全局规划未启用")
            return
        
        if not self.map_subscriber:
            print("[ERROR] MapSubscriber未初始化！")
            return
            
        if self.map_subscriber.map_data is None:
            print("[WARN] 地图数据未接收，等待/map话题...")
            # 尝试spin等待地图
            for _ in range(10):
                rclpy.spin_once(self.map_subscriber, timeout_sec=0.1)
                if self.map_subscriber.map_data is not None:
                    print("[INFO] 地图数据已接收！")
                    break
        
        if self.map_subscriber.map_data is not None:
            map_data = self.map_subscriber.map_data
            resolution = self.map_subscriber.map_resolution
            origin = (self.map_subscriber.map_origin_x, self.map_subscriber.map_origin_y)
            
            print(f"[INFO] 初始化A*规划器: 地图大小={self.map_subscriber.map_width}x{self.map_subscriber.map_height}, 分辨率={resolution}m")
            
            from start_reinforcement_learning.env_logic.global_planner import AStarPlanner
            self.planner = AStarPlanner(map_data, resolution, origin)
            print("✅ A*规划器已就绪")
        else:
            print("[ERROR] 无法接收地图数据，A*规划器未初始化！")
            print("[HINT] 请确保/map话题正在发布: ros2 topic echo /map --once")
    
    def _plan_global_path(self, robot_id, start_pos, goal_pos):
        """为机器人规划全局路径"""
        print(f"[DEBUG] _plan_global_path called: robot={robot_id}, start={start_pos}, goal={goal_pos}")
        
        if not self.planner:
            print(f"[DEBUG] Planner为None，尝试初始化...")
            self._initialize_planner_after_map_loaded()
            if not self.planner:
                print(f"[WARN] ⚠️ A*规划器初始化失败，机器人{robot_id}直接使用目标点")
                self.global_waypoints[robot_id] = [goal_pos]
                self.current_waypoint_index[robot_id] = 0
                return
        
        print(f"[DEBUG] 开始A*路径规划...")
        path = self.planner.plan(start_pos, goal_pos)
        
        if path is None or len(path) < 2:
            print(f"⚠️ 机器人{robot_id}无法规划路径，直接使用目标点")
            self.global_waypoints[robot_id] = [goal_pos]
            self.current_waypoint_index[robot_id] = 0
            return
        
        print(f"[DEBUG] A*路径规划成功，路径长度={len(path)}")
        waypoints = self.waypoint_extractor.extract(path)
        self.global_waypoints[robot_id] = waypoints
        self.current_waypoint_index[robot_id] = 0
        
        print(f"✅ 机器人{robot_id}：{len(path)}点→{len(waypoints)}关键点")
        
        if self.waypoint_visualizer:
            print(f"[DEBUG] 正在发布路径点可视化...")
            self.waypoint_visualizer.publish_waypoints(waypoints, robot_id=robot_id)
            # 多次spin确保消息发送
            for _ in range(5):
                rclpy.spin_once(self.waypoint_visualizer, timeout_sec=0.01)
            
            if len(waypoints) > 0:
                print(f"[DEBUG] 正在高亮当前路径点: {waypoints[0]}")
                self.waypoint_visualizer.highlight_current_waypoint(waypoints[0], robot_id=robot_id)
                for _ in range(5):
                    rclpy.spin_once(self.waypoint_visualizer, timeout_sec=0.01)
            
            print(f"[DEBUG] 可视化发布完成，话题: /waypoint_markers")
        else:
            print(f"[WARN] waypoint_visualizer为None！")
    
    def _get_current_waypoint(self, robot_id):
        """获取当前应该前往的路径点"""
        if not self.use_global_planner or not self.global_waypoints[robot_id]:
            if robot_id < len(self.current_goal_locations):
                return self.current_goal_locations[robot_id]
            return None
        
        idx = self.current_waypoint_index[robot_id]
        waypoints = self.global_waypoints[robot_id]
        
        if idx < len(waypoints):
            return waypoints[idx]
        else:
            return waypoints[-1] if waypoints else None
    
    def _check_and_update_waypoint(self, robot_id):
        """检查是否到达路径点，到达则切换"""
        if not self.use_global_planner or not self.global_waypoints[robot_id]:
            return False
        
        current_wp = self._get_current_waypoint(robot_id)
        if current_wp is None:
            return False
        
        robot_x = self.current_pose_x[robot_id]
        robot_y = self.current_pose_y[robot_id]
        dist = math.hypot(current_wp[0] - robot_x, current_wp[1] - robot_y)
        
        if dist < self.waypoint_reach_distance:
            old_idx = self.current_waypoint_index[robot_id]
            self.current_waypoint_index[robot_id] += 1
            
            total_wps = len(self.global_waypoints[robot_id])
            new_idx = self.current_waypoint_index[robot_id]
            
            if new_idx < total_wps:
                print(f"🎯 机器人{robot_id}：路径点{old_idx}→{new_idx}/{total_wps}")
                if self.waypoint_visualizer:
                    next_wp = self.global_waypoints[robot_id][new_idx]
                    self.waypoint_visualizer.highlight_current_waypoint(next_wp, robot_id=robot_id)
                return True
            else:
                print(f"🏁 机器人{robot_id}到达最终目标！")
                return True
        
        return False
    
    def _log_interaction(self, action, observation, rewards, dones, truncated, next_observation=None, event='normal'):
        """记录环境交互的七元组数据 (state, action, reward, next_state, done, truncated, info)"""
        try:
            # 构建记录
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'episode': self.episode_counter,
                'step': self.step_counter,
                'total_step': self.total_step_counter,
                'event': event,
                'action_raw': {},
                'action_mapped': {},
                'observation': {},
                'next_observation': {},
                'rewards': {},
                'dones': {},
                'truncated': {},
                'robot_positions': {},
                'robot_velocities': {},
                'distance_field_info': {}
            }
            
            # 记录每个机器人的数据
            for i in range(self.number_of_robots):
                robot_name = f'robot{i}'
                
                if robot_name in action:
                    raw_action = action[robot_name]
                    log_entry['action_raw'][robot_name] = [float(a) for a in raw_action]
                    
                    if i < len(self.current_linear_velocity):
                        log_entry['action_mapped'][robot_name] = {
                            'linear': float(self.current_linear_velocity[i]),
                            'angular': float(self.current_angular_velocity[i]),
                            'raw_linear_action': float(raw_action[0]),
                            'raw_angular_action': float(raw_action[1])
                        }
                
                if robot_name in observation:
                    obs = observation[robot_name]
                    obs_info = {
                        'dim': len(obs),
                        'lidar_min': float(np.min(obs[:38])) if len(obs) >= 38 else None,
                        'lidar_max': float(np.max(obs[:38])) if len(obs) >= 38 else None,
                        'velocity_linear': float(obs[38]) if len(obs) > 38 else None,
                        'velocity_angular': float(obs[39]) if len(obs) > 39 else None,
                        'goal_dx': float(obs[40]) if len(obs) > 40 else None,
                        'goal_dy': float(obs[41]) if len(obs) > 41 else None,
                        'goal_distance': float(obs[42]) if len(obs) > 42 else None,
                    }
                    
                    if self.use_distance_field and len(obs) >= 98:
                        distance_field_data = obs[49:98]
                        obs_info['distance_field'] = {
                            'enabled': True,
                            'size': '7x7',
                            'min': float(np.min(distance_field_data)),
                            'max': float(np.max(distance_field_data)),
                            'mean': float(np.mean(distance_field_data)),
                            'center_value': float(distance_field_data[24])
                        }
                        log_entry['distance_field_info'][robot_name] = obs_info['distance_field']
                    else:
                        obs_info['distance_field'] = {'enabled': False}
                    
                    log_entry['observation'][robot_name] = obs_info
                
                if next_observation and robot_name in next_observation:
                    next_obs = next_observation[robot_name]
                    log_entry['next_observation'][robot_name] = {
                        'dim': len(next_obs),
                        'lidar_min': float(np.min(next_obs[:38])) if len(next_obs) >= 38 else None,
                        'goal_distance': float(next_obs[42]) if len(next_obs) > 42 else None
                    }
                
                log_entry['rewards'][robot_name] = float(rewards[i])
                log_entry['dones'][robot_name] = bool(dones[i])
                log_entry['truncated'][robot_name] = bool(truncated[i])
                
                if i < len(self.current_pose_x):
                    odom_subscriber = self.odometry_subscriber_list[i]
                    odom_data = odom_subscriber.odom
                    if odom_data:
                        qz = odom_data.pose.pose.orientation.z
                        qw = odom_data.pose.pose.orientation.w
                        yaw = 2 * math.atan2(qz, qw)
                    else:
                        yaw = 0.0
                    
                    log_entry['robot_positions'][robot_name] = {
                        'x': float(self.current_pose_x[i]),
                        'y': float(self.current_pose_y[i]),
                        'yaw': float(yaw)
                    }
                
                if i < len(self.current_linear_velocity):
                    log_entry['robot_velocities'][robot_name] = {
                        'linear': float(self.current_linear_velocity[i]),
                        'angular': float(self.current_angular_velocity[i])
                    }
            
            with open(self.interaction_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self.total_step_counter += 1
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 记录交互数据失败: {e}")
            traceback.print_exc()
            self.enable_interaction_logging = False

class ReadScan(Node):
    def __init__(self, robot_number, env_id='default'):
        super().__init__(f'ReadScan{robot_number}_{env_id}')
        topic_name = "/tb3_"+str(robot_number)+"/scan"
        self.subscriber = self.create_subscription(LaserScan, topic_name, self.scan_callback,
                                                   qos_profile=qos_profile_sensor_data)
        self.scan = None
        self.last_scan = None

    def scan_callback(self, data):
        self.scan = data
        self.last_scan = data

class PublishCMD_VEL(Node):
    def __init__(self, robot_number, env_id='default'):
        super().__init__(f'PublishCMD_VEL{robot_number}_{env_id}')
        topic_name = "/tb3_"+str(robot_number)+"/cmd_vel"

        self.cmd_vel_publisher = self.create_publisher(
            Twist, topic_name, 10)
        self.cmd_vel = ' '

    def pub_vel(self):
        self.cmd_vel_publisher.publish(self.cmd_vel)

class ReadOdom(Node):
    def __init__(self, robot_number, env_id='default'):
        super().__init__(f'ReadOdom{robot_number}_{env_id}')
        topic_name = "/tb3_"+str(robot_number)+"/odom"
        self.subscriber = self.create_subscription(
            Odometry, topic_name, self.odom_callback, 10)
        self.odom = None

    def odom_callback(self, data):
        self.odom = data

class Logger(Node):
    def __init__(self, env_id='default'):
        super().__init__(f'logger_{env_id}')
        
    def log(self, string):
        self.get_logger().info(string)

class RewardMarkerPublisher(Node):
    """发布奖励可视化Marker的节点"""
    def __init__(self, robot_number, env_id='default'):
        super().__init__(f'reward_marker_publisher_{robot_number}_{env_id}')
        
        topic_name = f"/robot_{robot_number}_reward_marker"
        
        # 使用可靠的QoS配置
        from rclpy.qos import QoSDurabilityPolicy
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray, topic_name, qos)
        
    def publish_marker(self, marker):
        # 将单个Marker包装成MarkerArray
        marker_array = MarkerArray()
        marker_array.markers = [marker]
        self.marker_publisher.publish(marker_array)

class GoalMarkerPublisher(Node):
    """发布目标点可视化Marker的节点"""
    def __init__(self, robot_number, env_id='default'):
        super().__init__(f'goal_marker_publisher_{robot_number}_{env_id}')
        
        topic_name = f"/robot_{robot_number}_goal_marker"
        
        # 使用可靠的QoS配置
        from rclpy.qos import QoSDurabilityPolicy
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray, topic_name, qos)
        
    def publish_marker_array(self, marker_array):
        self.marker_publisher.publish(marker_array)

class MapSubscriber(Node):
    """订阅地图数据的节点，用于距离场计算"""
    def __init__(self, env_id='default'):
        super().__init__(f'map_subscriber_{env_id}')
        
        # 地图话题通常为 /map
        topic_name = "/map"
        
        # 设置QoS配置：地图话题通常使用可靠传输、Transient Local耐久性和保留最后一条消息
        # 这样即使订阅者晚启动，也能接收到之前发布的地图数据
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # 关键：匹配 map_server 的 durability
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            topic_name,
            self.map_callback,
            qos_profile=qos_profile
        )
        
        # 存储地图数据
        self.map_data = None
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0.05  # 默认分辨率（米/格子）
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        
        self.get_logger().info("MapSubscriber initialized, waiting for /map topic...")
    
    def map_callback(self, msg):
        """接收地图数据并存储"""
        # 只在第一次接收时处理并打印信息
        if self.map_data is None:
            self.get_logger().info("🎉 Map callback triggered! Processing map data...")
            
            # 存储地图信息 - 转换为numpy数组！
            import numpy as np
            self.map_data = np.array(msg.data, dtype=np.int8).reshape((msg.info.height, msg.info.width))
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            self.map_resolution = msg.info.resolution
            self.map_origin_x = msg.info.origin.position.x
            self.map_origin_y = msg.info.origin.position.y
            
            self.get_logger().info(
                f"✅ Map received successfully:\n"
                f"   Size: {self.map_width}x{self.map_height}\n"
                f"   Resolution: {self.map_resolution}m\n"
                f"   Origin: ({self.map_origin_x:.2f}, {self.map_origin_y:.2f})\n"
                f"   Data shape: {self.map_data.shape}"
            )
