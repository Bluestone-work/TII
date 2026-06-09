#!/usr/bin/env python3
"""
Multi-Robot Navigation Node using Nav2 with ORCA collision avoidance

Three-layer architecture:
1. Nav2 Global Planner: Plans global path avoiding known static obstacles
2. ORCA Layer: Computes ideal velocity considering other robots (dynamic collision avoidance)
3. DWA Local Planner: Generates final control commands satisfying kinematic constraints and laser-based obstacle avoidance
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped, Pose, Point
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import numpy as np
from typing import List, Tuple
import math
from typing import Dict, List, Optional
import os

from .orca_algorithm import ORCAAgent, compute_preferred_velocity
from .dwa_planner import create_dwa_planner
from .global_planner import create_simple_planner


class ORCANavNode(Node):
    """
    Multi-robot navigation node with ORCA collision avoidance
    """
    
    def __init__(self):
        super().__init__('orca_nav_node')
        
        # Declare parameters
        self.declare_parameter('robot_number', 4)
        self.declare_parameter('robot_radius', 0.35)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 2.0)
        self.declare_parameter('neighbor_distance', 5.0)  # Distance to consider neighbors
        self.declare_parameter('time_horizon', 2.0)  # ORCA time horizon
        self.declare_parameter('navigation_mode', 'orca')  # 'orca' or 'nav2'
        self.declare_parameter('use_dwa', True)  # Whether to use DWA for obstacle avoidance
        self.declare_parameter('goal_tolerance', 0.3)  # Goal reached tolerance in meters
        
        # Get parameters
        self.robot_number = self.get_parameter('robot_number').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.neighbor_distance = self.get_parameter('neighbor_distance').value
        self.time_horizon = self.get_parameter('time_horizon').value
        self.navigation_mode = self.get_parameter('navigation_mode').value  # 'orca' or 'nav2'
        self.use_dwa = self.get_parameter('use_dwa').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        
        # 根据模式设置行为
        self.use_nav2_full = (self.navigation_mode == 'nav2')  # 完全使用Nav2控制
        self.use_orca = (self.navigation_mode == 'orca')  # 使用ORCA+DWA+Theta*
        
        # 创建DWA局部规划器（ORCA模式）
        if self.use_orca and self.use_dwa:
            self.dwa_planner = create_dwa_planner(
                max_speed=self.max_linear_speed,
                max_yaw_rate=self.max_angular_speed,
                robot_radius=self.robot_radius
            )
            self.get_logger().info('DWA local planner enabled')
        else:
            self.dwa_planner = None
        
        # 使用Nav2全局规划器（通过action获取路径）
        self.map_file = self.get_parameter('map_file').get_parameter_value().string_value
        self.use_nav2_global_planner = self.get_parameter('use_nav2_global_planner').get_parameter_value().bool_value
        
        # Nav2路径规划action clients
        self.nav2_planner_clients = {}  # ComputePathToPose action clients
        
        if self.use_nav2_full:
            self.get_logger().info(f'Initializing Nav2 FULL navigation for {self.robot_number} robots')
        else:
            planner_info = 'Nav2' if self.use_nav2_global_planner else 'Direct'
            self.get_logger().info(f'Initializing ORCA+DWA+{planner_info} navigation for {self.robot_number} robots')
        
        # Robot states
        self.robot_positions = {}  # {robot_id: np.array([x, y])}
        self.robot_velocities = {}  # {robot_id: np.array([vx, vy])}
        self.robot_yaws = {}  # {robot_id: yaw}
        self.robot_goals = {}  # {robot_id: np.array([x, y])}
        self.robot_goals_initialized = {}  # {robot_id: bool} 跟踪目标是否已初始化
        self.robot_goal_reached = {}  # {robot_id: bool} 跟踪是否已到达目标
        self.nav2_paths = {}  # {robot_id: Path} Nav2模式的路径
        self.theta_star_paths = {}  # {robot_id: List} ORCA模式的Theta*路径
        self.waypoint_distance = 1.5  # 切换waypoint的前瞻距离（增大防止振荡）
        self.current_waypoint_index = {}  # {robot_id: int} 当前waypoint索引（防抖）
        self.last_replan_position = {}  # {robot_id: np.array} 上次重规划时的位置
        self.replan_distance = 3.0  # 移动超过3米后触发重规划
        
        # 卡住检测机制
        self.stuck_detection_window = 5.0  # 检测卡住的时间窗口（秒）
        self.stuck_distance_threshold = 0.2  # 移动距离小于0.2米视为卡住
        self.stuck_check_time = {}  # {robot_id: float} 上次检测卡住的时间
        self.stuck_check_position = {}  # {robot_id: np.array} 检测卡住时的位置
        self.stuck_count = {}  # {robot_id: int} 连续卡住次数
        
        self.gazebo_goal_models = {}  # {robot_id: model_name} Gazebo中的目标模型
        self.gazebo_path_models = {}  # {robot_id: [model_names]} Gazebo中的路径标记模型
        self.laser_scans = {}  # {robot_id: LaserScan} 激光扫描数据
        self.robot_current_cmd_vel = {}  # {robot_id: (v, w)} 当前速度命令
        
        # Publishers and subscribers for each robot
        self.cmd_vel_publishers = {}
        self.odom_subscribers = {}
        self.goal_subscribers = {}
        self.goal_marker_publishers = {}  # 目标点可视化
        self.path_marker_publishers = {}  # 路径可视化（新增）
        self.laser_subscribers = {}  # 激光扫描订阅
        self.nav2_clients = {}
        
        for i in range(self.robot_number):
            robot_name = f'robot{i}'
            # 实际的 Gazebo 话题使用 my_bot{i} 命名
            gazebo_namespace = f'my_bot{i}'
            
            # Velocity command publisher - 发布到 my_bot{i}
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist, f'/{gazebo_namespace}/cmd_vel', 10
            )
            
            # Odometry subscriber - 订阅 my_bot{i}
            self.odom_subscribers[robot_name] = self.create_subscription(
                Odometry, f'/{gazebo_namespace}/odom',
                lambda msg, name=robot_name: self.odom_callback(msg, name),
                10
            )
            
            # Goal subscriber - 使用 robot{i} 方便外部发送目标
            self.goal_subscribers[robot_name] = self.create_subscription(
                PoseStamped, f'/robot{i}/goal_pose',
                lambda msg, name=robot_name: self.goal_callback(msg, name),
                10
            )
            
            # Nav2 ComputePathToPose action client
            if self.use_nav2_global_planner:
                self.nav2_planner_clients[robot_name] = ActionClient(
                    self, ComputePathToPose, f'/{gazebo_namespace}/compute_path_to_pose'
                )
            
            # Goal marker publisher - 可视化目标点
            self.goal_marker_publishers[robot_name] = self.create_publisher(
                Marker, f'/robot{i}/goal_marker', 10
            )
            
            # Path marker publisher - 可视化全局路径（新增）
            self.path_marker_publishers[robot_name] = self.create_publisher(
                Marker, f'/robot{i}/path_marker', 10
            )
            
            # Laser scan subscriber - 订阅激光数据（ORCA模式需要）
            if self.use_orca:
                self.laser_subscribers[robot_name] = self.create_subscription(
                    LaserScan, f'/{gazebo_namespace}/scan',
                    lambda msg, name=robot_name: self.laser_callback(msg, name),
                    10
                )
            
            # Nav2相关订阅和客户端
            if self.use_nav2_full:
                # 纯Nav2模式：只订阅路径，不创建action client（避免冲突）
                # Nav2会直接控制/cmd_vel
                self.create_subscription(
                    Path, f'/{gazebo_namespace}/plan',
                    lambda msg, name=robot_name: self.path_callback(msg, name),
                    10
                )
                # 创建action client用于发送导航目标
                self.nav2_clients[robot_name] = ActionClient(
                    self, NavigateToPose, f'/{gazebo_namespace}/navigate_to_pose'
                )
        
        # Control loop timer (20 Hz) - 只在ORCA模式下运行
        if self.use_orca:
            self.timer = self.create_timer(0.05, self.control_loop)
        # Nav2模式下，Nav2的控制器会接管cmd_vel，我们只需要监控状态
        
        # Marker republish timer (1 Hz) - 定期重新发布marker确保可视化
        self.marker_timer = self.create_timer(1.0, self.republish_markers)
        
        # Gazebo服务客户端 - 用于在Gazebo中生成目标点模型
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        
        self.get_logger().info('ORCA Navigation Node initialized')
    
    def odom_callback(self, msg: Odometry, robot_name: str):
        """Update robot position and velocity from odometry"""
        # Extract position
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_positions[robot_name] = np.array([x, y])
        
        # Extract velocity
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.robot_velocities[robot_name] = np.array([vx, vy])
        
        # Extract yaw from quaternion
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_yaws[robot_name] = yaw
        
        # 首次收到odom时记录日志（验证数据接收）
        if robot_name not in self.robot_goal_reached:
            self.get_logger().info(
                f'{robot_name} odom received: pos=[{x:.2f}, {y:.2f}], yaw={yaw:.2f}',
                throttle_duration_sec=5.0
            )
    
    def laser_callback(self, msg: LaserScan, robot_name: str):
        """\u5904\u7406\u6fc0\u5149\u626b\u63cf\u6570\u636e"""
        self.laser_scans[robot_name] = msg
    
    def _initialize_map_obstacles(self):
        """初始化地图障碍物（从map_config加载）"""
        if not self.global_planner:
            return
        
        obstacles = []
        
        # 方法1：从map_config加载实际障碍物
        if hasattr(self, 'map_config') and 'obstacles' in self.map_config:
            for obs in self.map_config['obstacles']:
                x, y = obs['position']
                obstacles.append((x, y))
        
        # 方法2：添加地图边界（兜底）
        if len(obstacles) == 0:
            # 四边墙壁（每隔0.5米设置一个障碍点）
            for i in range(-20, 21):
                pos = i * 0.5
                obstacles.append((pos, -10.0))  # 下边界
                obstacles.append((pos, 10.0))   # 上边界
                obstacles.append((-10.0, pos))  # 左边界
                obstacles.append((10.0, pos))   # 右边界
        
        # 设置到全局规划器（障碍物半径0.5米，考虑机器人半径）
        self.global_planner.set_map_obstacles(obstacles, radius=0.5)
        self.get_logger().info(f'Initialized {len(obstacles)} map obstacles')
    
    def goal_callback(self, msg: PoseStamped, robot_name: str):
        """Update robot goal"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.robot_goals[robot_name] = np.array([x, y])
        self.robot_goals_initialized[robot_name] = True
        self.robot_goal_reached[robot_name] = False  # 重置到达状态
        
        self.get_logger().info(f'{robot_name} received new goal: [{x:.2f}, {y:.2f}]')
        
        # 发布目标点可视化 marker (RViz)
        self.publish_goal_marker(robot_name, x, y)
        
        # 在Gazebo中生成目标点模型
        self.spawn_goal_in_gazebo(robot_name, x, y)
        
        # 根据导航模式处理
        if self.use_nav2_full:
            # 模式2：纯Nav2导航
            # 发送导航目标给Nav2，让Nav2完全接管控制
            if robot_name in self.nav2_clients:
                self.send_nav2_goal(robot_name, msg)
        else:
            # 模式1：ORCA+DWA+Nav2全局规划
            # 使用Nav2的ComputePathToPose获取全局路径
            if self.use_nav2_global_planner and robot_name in self.nav2_planner_clients:
                self._request_nav2_path(robot_name, msg)
            else:
                # 备用：直接使用目标点作为路径
                self.theta_star_paths[robot_name] = [(x, y)]
                self.current_waypoint_index[robot_name] = 0
                self.get_logger().info(f'{robot_name} Using direct path to goal')
    
    def _request_nav2_path(self, robot_name: str, goal_msg: PoseStamped):
        """请求Nav2规划全局路径"""
        if robot_name not in self.robot_positions:
            self.get_logger().warn(f'{robot_name} position unknown, cannot request path')
            return
        
        client = self.nav2_planner_clients[robot_name]
        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                f'{robot_name} Nav2 planner server not available, using direct path',
                throttle_duration_sec=5.0
            )
            # 备用：直接使用目标点
            x = goal_msg.pose.position.x
            y = goal_msg.pose.position.y
            self.theta_star_paths[robot_name] = [(x, y)]
            self.current_waypoint_index[robot_name] = 0
            return
        
        # 创建规划请求
        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg
        
        # 设置起点为当前位置
        goal.start = PoseStamped()
        goal.start.header.frame_id = 'map'
        goal.start.header.stamp = self.get_clock().now().to_msg()
        pos = self.robot_positions[robot_name]
        goal.start.pose.position.x = pos[0]
        goal.start.pose.position.y = pos[1]
        goal.start.pose.position.z = 0.0
        goal.start.pose.orientation.w = 1.0
        
        # 设置规划器ID（可选）
        goal.planner_id = ''  # 空字符串使用默认规划器
        goal.use_start = True
        
        # 发送异步请求
        self.get_logger().info(f'{robot_name} Requesting Nav2 path...')
        future = client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._nav2_path_response_callback(f, robot_name)
        )
    
    def _nav2_path_response_callback(self, future, robot_name: str):
        """处理Nav2路径规划响应"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn(f'{robot_name} Nav2 path request rejected')
            return
        
        # 等待结果
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._nav2_path_result_callback(f, robot_name)
        )
    
    def _nav2_path_result_callback(self, future, robot_name: str):
        """处理Nav2路径规划结果"""
        result = future.result().result
        path_msg = result.path
        
        if not path_msg.poses or len(path_msg.poses) == 0:
            self.get_logger().warn(f'{robot_name} Nav2 returned empty path')
            return
        
        # 转换Path消息为路径点列表
        path = [(pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses]
        
        # 保存路径
        self.theta_star_paths[robot_name] = path
        self.current_waypoint_index[robot_name] = 0
        self.last_replan_position[robot_name] = self.robot_positions[robot_name].copy()
        
        # 调试信息
        path_str = ' -> '.join([f'({p[0]:.2f},{p[1]:.2f})' for p in path[:5]])
        if len(path) > 5:
            path_str += f' ... ({path[-1][0]:.2f},{path[-1][1]:.2f})'
        self.get_logger().info(
            f'{robot_name} Nav2 path received with {len(path)} waypoints: {path_str}'
        )
        
        # 可视化路径
        self._publish_path_marker(robot_name, path)
        self._spawn_path_in_gazebo(robot_name, path)
    
    def publish_goal_marker(self, robot_name: str, x: float, y: float):
        """发布目标点可视化标记"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_goal"
        marker.id = int(robot_name.replace('robot', ''))
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 位置
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5  # 抬高一点方便观察
        marker.pose.orientation.w = 1.0
        
        # 大小
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # 颜色（与路径颜色一致）
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            (1.0, 0.0, 0.0),  # robot0 - 红色
            (0.0, 1.0, 0.0),  # robot1 - 绿色
            (0.0, 0.0, 1.0),  # robot2 - 蓝色
            (1.0, 1.0, 0.0),  # robot3 - 黄色
            (0.0, 1.0, 1.0),  # robot4 - 青色
            (1.0, 0.0, 1.0),  # robot5 - 品红
        ]
        color = colors[robot_id % len(colors)]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8  # 半透明
        
        marker.lifetime.sec = 0  # 永久显示
        
        # 发布
        if robot_name in self.goal_marker_publishers:
            self.goal_marker_publishers[robot_name].publish(marker)
            self.get_logger().info(f'Published goal marker for {robot_name} at [{x:.2f}, {y:.2f}]', throttle_duration_sec=5.0)
    
    def _publish_path_marker(self, robot_name: str, path: List[Tuple[float, float]]):
        """发布路径可视化marker
        
        Args:
            robot_name: 机器人名称
            path: 路径点列表 [(x1,y1), (x2,y2), ...]
        """
        if robot_name not in self.path_marker_publishers:
            return
        
        if not path or len(path) < 2:
            return
        
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_path"
        marker.id = int(robot_name.replace('robot', ''))
        marker.type = Marker.LINE_STRIP  # 使用线条类型
        marker.action = Marker.ADD
        
        # 设置位置和朝向
        marker.pose.orientation.w = 1.0
        
        # 设置线条属性
        marker.scale.x = 0.05  # 线条宽度
        
        # 根据机器人ID设置不同颜色
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            (0.0, 1.0, 0.0),  # 绿色 - robot0
            (0.0, 0.0, 1.0),  # 蓝色 - robot1
            (1.0, 1.0, 0.0),  # 黄色 - robot2
            (1.0, 0.0, 0.0),  # 红色 - robot3
            (0.0, 1.0, 1.0),  # 青色 - robot4
            (1.0, 0.0, 1.0),  # 品红 - robot5
        ]
        color = colors[robot_id % len(colors)]
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.8  # 半透明
        
        # 添加路径点
        from geometry_msgs.msg import Point
        for waypoint in path:
            point = Point()
            point.x = waypoint[0]
            point.y = waypoint[1]
            point.z = 0.1  # 稍微抬高避免与地面重叠
            marker.points.append(point)
        
        # 设置marker生命周期（0表示永久，直到被删除）
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0
        
        self.path_marker_publishers[robot_name].publish(marker)
        
        self.get_logger().info(
            f'{robot_name} 路径可视化已更新（{len(path)}个点）',
            throttle_duration_sec=2.0
        )
    
    def spawn_goal_in_gazebo(self, robot_name: str, x: float, y: float):
        """在Gazebo中生成目标点球体模型"""
        # 等待服务可用
        if not self.spawn_entity_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Gazebo spawn service not available')
            return
        
        # 如果已存在旧的目标点，先删除
        model_name = f'goal_{robot_name}'
        if robot_name in self.gazebo_goal_models:
            self.delete_goal_from_gazebo(robot_name)
        
        # 获取机器人ID用于颜色
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            ('1 0 0 1', 'Gazebo/Red'),      # 红色
            ('0 1 0 1', 'Gazebo/Green'),    # 绿色
            ('0 0 1 1', 'Gazebo/Blue'),     # 蓝色
            ('1 1 0 1', 'Gazebo/Yellow'),   # 黄色
        ]
        color_rgba, color_material = colors[robot_id % len(colors)]
        
        # SDF球体模型
        sdf = f'''<?xml version="1.0" ?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.06</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
        <transparency>0.3</transparency>
      </visual>
    </link>
  </model>
</sdf>'''
        
        # 创建spawn请求
        request = SpawnEntity.Request()
        request.name = model_name
        request.xml = sdf
        request.robot_namespace = ''
        request.initial_pose = Pose()
        request.initial_pose.position.x = x
        request.initial_pose.position.y = y
        request.initial_pose.position.z = 0.3  # 抬高一点
        request.initial_pose.orientation.w = 1.0
        request.reference_frame = 'world'
        
        # 异步调用
        future = self.spawn_entity_client.call_async(request)
        future.add_done_callback(
            lambda f: self._spawn_callback(f, robot_name, model_name)
        )
    
    def _spawn_callback(self, future, robot_name: str, model_name: str):
        """Spawn服务回调"""
        try:
            response = future.result()
            if response.success:
                self.gazebo_goal_models[robot_name] = model_name
                self.get_logger().info(f'Gazebo goal marker spawned for {robot_name}')
            else:
                self.get_logger().warn(f'Failed to spawn Gazebo goal: {response.status_message}')
        except Exception as e:
            self.get_logger().error(f'Spawn service call failed: {e}')
    
    def delete_goal_from_gazebo(self, robot_name: str):
        """从Gazebo中删除旧的目标点模型"""
        if robot_name not in self.gazebo_goal_models:
            return
        
        if not self.delete_entity_client.wait_for_service(timeout_sec=0.5):
            return
        
        request = DeleteEntity.Request()
        request.name = self.gazebo_goal_models[robot_name]
        
        future = self.delete_entity_client.call_async(request)
        future.add_done_callback(
            lambda f: self.get_logger().info(f'Deleted old goal for {robot_name}', throttle_duration_sec=5.0)
        )
        
        del self.gazebo_goal_models[robot_name]
    
    def _spawn_path_in_gazebo(self, robot_name: str, path: List[Tuple[float, float]]):
        """在Gazebo中显示路径（使用小球体链）
        
        Args:
            robot_name: 机器人名称
            path: 路径点列表
        """
        if not self.spawn_entity_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f'{robot_name} Gazebo spawn服务不可用，无法显示路径',
                throttle_duration_sec=5.0
            )
            return
        
        self.get_logger().info(f'{robot_name} 开始在Gazebo中生成路径...')
        
        # 删除旧路径
        self._delete_path_from_gazebo(robot_name)
        
        # 对完整路径进行密集采样显示（每0.3米一个点）
        display_points = self._sample_path_for_display(path, interval=0.3)
        
        if not display_points or len(display_points) == 0:
            self.get_logger().warn(f'{robot_name} 路径采样后为空')
            return
        
        self.get_logger().info(
            f'{robot_name} 路径采样: {len(path)}点 → {len(display_points)}显示点'
        )
        
        # 根据机器人ID选择颜色（RGBA格式，与目标点颜色一致）
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            (1.0, 0.0, 0.0, 0.9),  # robot0 - 红色
            (0.0, 1.0, 0.0, 0.9),  # robot1 - 绿色
            (0.0, 0.0, 1.0, 0.9),  # robot2 - 蓝色
            (1.0, 1.0, 0.0, 0.9),  # robot3 - 黄色
            (0.0, 1.0, 1.0, 0.9),  # robot4 - 青色
            (1.0, 0.0, 1.0, 0.9),  # robot5 - 品红
        ]
        color = colors[robot_id % len(colors)]
        
        # 为每个waypoint生成一个小球体
        model_names = []
        for i, point in enumerate(display_points):
            model_name = f'path_marker_{robot_name}_{i}'
            
            # 创建球体SDF模型（使用RGBA颜色）
            sdf = f"""<?xml version='1.0'?>
            <sdf version='1.6'>
              <model name='{model_name}'>
                <static>true</static>
                <link name='link'>
                  <visual name='visual'>
                    <geometry>
                      <sphere>
                        <radius>0.06</radius>
                      </sphere>
                    </geometry>
                    <material>
                      <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
                      <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
                      <specular>0.1 0.1 0.1 1</specular>
                      <emissive>{color[0]*0.3} {color[1]*0.3} {color[2]*0.3} 0</emissive>
                    </material>
                  </visual>
                </link>
              </model>
            </sdf>
            """
            
            request = SpawnEntity.Request()
            request.name = model_name
            request.xml = sdf
            request.robot_namespace = ''
            request.initial_pose = Pose()
            request.initial_pose.position.x = point[0]
            request.initial_pose.position.y = point[1]
            request.initial_pose.position.z = 0.08  # 略微抬高
            request.initial_pose.orientation.w = 1.0
            request.reference_frame = 'world'
            
            # 同步调用确保能看到结果
            try:
                future = self.spawn_entity_client.call_async(request)
                model_names.append(model_name)
            except Exception as e:
                self.get_logger().error(f'生成路径点{i}失败: {e}')
        
        # 记录生成的模型
        if model_names:
            self.gazebo_path_models[robot_name] = model_names
            self.get_logger().info(
                f'✓ {robot_name} 在Gazebo中生成了{len(model_names)}个路径标记（共{len(path)}个原始路径点）'
            )
        else:
            self.get_logger().warn(f'{robot_name} 未能生成任何路径标记')
    
    def _delete_path_from_gazebo(self, robot_name: str):
        """从Gazebo中删除旧的路径标记"""
        if robot_name not in self.gazebo_path_models:
            return
        
        if not self.delete_entity_client.wait_for_service(timeout_sec=0.5):
            return
        
        for model_name in self.gazebo_path_models[robot_name]:
            request = DeleteEntity.Request()
            request.name = model_name
            self.delete_entity_client.call_async(request)
        
        del self.gazebo_path_models[robot_name]
    
    def republish_markers(self):
        """定期重新发布所有目标点marker"""
        for robot_name, goal in self.robot_goals.items():
            if self.robot_goals_initialized.get(robot_name, False):
                self.publish_goal_marker(robot_name, goal[0], goal[1])
    
    def send_nav2_goal(self, robot_name: str, goal_pose: PoseStamped):
        """Send goal to Nav2 action server"""
        client = self.nav2_clients[robot_name]
        
        if not client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(f'Nav2 server not available for {robot_name}')
            return
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        # Send goal asynchronously
        future = client.send_goal_async(goal_msg)
        self.get_logger().info(f'Sent goal to Nav2 for {robot_name}')
    
    def path_callback(self, msg: Path, robot_name: str):
        """接收Nav2规划的全局路径"""
        if len(msg.poses) > 0:
            self.nav2_paths[robot_name] = msg
            self.get_logger().info(
                f'{robot_name} received Nav2 path with {len(msg.poses)} waypoints',
                throttle_duration_sec=2.0
            )
    
    def _sample_path_for_display(self, path: List[Tuple[float, float]], interval: float = 0.3) -> List[Tuple[float, float]]:
        """对路径进行等距采样用于显示
        
        Args:
            path: 完整路径
            interval: 采样间隔（米）
        
        Returns:
            采样后的路径点
        """
        if not path or len(path) == 0:
            return path
        
        if len(path) == 1:
            return path
        
        sampled_points = [path[0]]  # 起点
        
        for i in range(1, len(path)):
            p1 = path[i-1]
            p2 = path[i]
            segment_dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            if segment_dist < 1e-6:
                continue
            
            # 在这段线段上采样（即使只有2个点也要插值）
            num_samples = max(1, int(segment_dist / interval))
            for j in range(1, num_samples):
                t = j / num_samples
                sample = (
                    p1[0] + t * (p2[0] - p1[0]),
                    p1[1] + t * (p2[1] - p1[1])
                )
                sampled_points.append(sample)
            
            # 添加线段终点
            sampled_points.append(p2)
        
        return sampled_points
    
    def _extract_key_waypoints(self, path: List[Tuple[float, float]], min_distance: float = 0.3) -> List[Tuple[float, float]]:
        """从完整路径中提取关键waypoint（拐角点+等距采样）
        
        Args:
            path: 完整路径
            min_distance: 相邻waypoint的最小间距
        
        Returns:
            关键waypoint列表，保证包含所有拐角
        """
        if len(path) <= 2:
            return path
        
        key_points = [path[0]]  # 起点
        
        # 检测拐角：计算相邻线段的方向变化
        for i in range(1, len(path) - 1):
            # 前一段和后一段的方向向量
            prev_vec = np.array([path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]])
            next_vec = np.array([path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]])
            
            # 归一化
            prev_norm = np.linalg.norm(prev_vec)
            next_norm = np.linalg.norm(next_vec)
            
            if prev_norm > 1e-6 and next_norm > 1e-6:
                prev_vec = prev_vec / prev_norm
                next_vec = next_vec / next_norm
                
                # 计算角度变化（点积）
                dot_product = np.dot(prev_vec, next_vec)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                
                # 如果角度变化超过20度，认为是拐角（更敏感）
                if angle > np.pi / 9:  # 20度
                    # 检查与上一个关键点的距离
                    dist = math.sqrt(
                        (path[i][0] - key_points[-1][0])**2 + 
                        (path[i][1] - key_points[-1][1])**2
                    )
                    if dist > min_distance:
                        key_points.append(path[i])
        
        # 在长直线段上等距采样，防止waypoint间距过大
        final_points = [key_points[0]]
        for i in range(len(key_points) - 1):
            p1 = key_points[i]
            p2 = key_points[i + 1]
            dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # 如果两个关键点距离超过0.8米，在中间插入点（更密集）
            if dist > 0.8:
                num_intermediate = int(dist / 0.8)
                for j in range(1, num_intermediate + 1):
                    t = j / (num_intermediate + 1)
                    intermediate = (
                        p1[0] + t * (p2[0] - p1[0]),
                        p1[1] + t * (p2[1] - p1[1])
                    )
                    final_points.append(intermediate)
            
            final_points.append(p2)
        
        # 添加终点
        if path[-1] not in final_points:
            final_points.append(path[-1])
        
        return final_points
    
    def get_next_waypoint(self, robot_name: str) -> Optional[np.ndarray]:
        """从路径获取下一个waypoint（基于索引的waypoint跟踪）"""
        position = self.robot_positions.get(robot_name)
        if position is None:
            return None
        
        # 模式1：使用Theta*路径
        if robot_name in self.theta_star_paths:
            path = self.theta_star_paths[robot_name]
            if not path or len(path) == 0:
                return self.robot_goals.get(robot_name)
            
            # 初始化waypoint索引
            if robot_name not in self.current_waypoint_index:
                self.current_waypoint_index[robot_name] = 0
            
            # 获取当前waypoint索引
            idx = self.current_waypoint_index[robot_name]
            
            # 确保索引有效
            if idx >= len(path):
                idx = len(path) - 1
                self.current_waypoint_index[robot_name] = idx
            
            # 获取当前waypoint
            current_waypoint = path[idx]
            dist_to_current = math.sqrt(
                (current_waypoint[0] - position[0])**2 + 
                (current_waypoint[1] - position[1])**2
            )
            
            # 如果接近当前waypoint（0.5米内），切换到下一个
            if dist_to_current < 0.5 and idx < len(path) - 1:
                idx += 1
                self.current_waypoint_index[robot_name] = idx
                self.get_logger().info(
                    f'{robot_name} 切换到waypoint[{idx}]: {path[idx]}',
                    throttle_duration_sec=1.0
                )
                current_waypoint = path[idx]
            
            return np.array(current_waypoint)
        
        # 模式2：使用Nav2路径（仅用于可视化或监控）
        if robot_name in self.nav2_paths:
            path = self.nav2_paths[robot_name]
            if not path.poses:
                return self.robot_goals.get(robot_name)
            
            # 找到路径上第一个距离当前位置超过waypoint_distance的点
            for pose in path.poses:
                waypoint = np.array([
                    pose.pose.position.x,
                    pose.pose.position.y
                ])
                distance = np.linalg.norm(waypoint - position)
                
                if distance >= self.waypoint_distance:
                    return waypoint
            
            # 如果所有路径点都在前瞻距离内，返回最后一个点（目标点）
            last_pose = path.poses[-1]
            return np.array([
                last_pose.pose.position.x,
                last_pose.pose.position.y
            ])
        
        # 如果没有路径，直接返回最终目标
        return self.robot_goals.get(robot_name)
    
    def _is_stuck(self, robot_name: str, position: np.ndarray) -> bool:
        """检测机器人是否卡住
        
        在stuck_detection_window时间内，如果移动距离小于stuck_distance_threshold，
        则认为机器人卡住了。
        
        Args:
            robot_name: 机器人名称
            position: 当前位置
            
        Returns:
            True表示卡住
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # 初始化检测时间和位置
        if robot_name not in self.stuck_check_time:
            self.stuck_check_time[robot_name] = current_time
            self.stuck_check_position[robot_name] = position.copy()
            self.stuck_count[robot_name] = 0
            return False
        
        # 检查时间间隔
        time_elapsed = current_time - self.stuck_check_time[robot_name]
        
        if time_elapsed >= self.stuck_detection_window:
            # 计算移动距离
            last_pos = self.stuck_check_position[robot_name]
            distance_moved = np.linalg.norm(position - last_pos)
            
            # 更新检测时间和位置
            self.stuck_check_time[robot_name] = current_time
            self.stuck_check_position[robot_name] = position.copy()
            
            # 判断是否卡住
            if distance_moved < self.stuck_distance_threshold:
                self.stuck_count[robot_name] = self.stuck_count.get(robot_name, 0) + 1
                self.get_logger().info(
                    f'{robot_name} 在{self.stuck_detection_window}秒内仅移动{distance_moved:.2f}米（卡住次数：{self.stuck_count[robot_name]}）',
                    throttle_duration_sec=2.0
                )
                # 连续2次检测到卡住才触发重规划（减少误判）
                return self.stuck_count[robot_name] >= 2
            else:
                # 正常移动，重置计数
                self.stuck_count[robot_name] = 0
        
        return False
    
    def _should_replan(self, robot_name: str, position: np.ndarray, goal: np.ndarray) -> bool:
        """检查是否需要重新规划路径
        
        触发条件：
        1. 机器人移动超过replan_distance米（默认3米）
        2. 当前没有路径或路径为空
        
        Args:
            robot_name: 机器人名称
            position: 当前位置
            goal: 目标位置
            
        Returns:
            True表示需要重规划
        """
        # 如果没有路径，需要规划
        if robot_name not in self.theta_star_paths or not self.theta_star_paths[robot_name]:
            return True
        
        # 如果没有记录上次规划位置，需要规划
        if robot_name not in self.last_replan_position:
            return True
        
        # 检查移动距离
        last_pos = self.last_replan_position[robot_name]
        distance_moved = np.linalg.norm(position - last_pos)
        
        # 如果移动超过阈值，重新规划
        if distance_moved >= self.replan_distance:
            self.get_logger().info(
                f'{robot_name} 移动{distance_moved:.2f}米，触发重规划',
                throttle_duration_sec=2.0
            )
            return True
        
        return False
    
    def _replan_path(self, robot_name: str, position: np.ndarray, goal: np.ndarray):
        """重新规划路径
        
        Args:
            robot_name: 机器人名称
            position: 当前位置
            goal: 目标位置
        """
        if not self.global_planner:
            return
        
        # 使用Theta*重新规划
        path = self.global_planner.plan_path(
            (position[0], position[1]),
            (goal[0], goal[1])
        )
        
        if path:
            self.theta_star_paths[robot_name] = path
            self.current_waypoint_index[robot_name] = 0  # 重置waypoint索引
            self.last_replan_position[robot_name] = position.copy()  # 更新规划位置
            self.get_logger().info(
                f'{robot_name} 路径重规划完成，新路径{len(path)}个waypoint',
                throttle_duration_sec=1.0
            )
            # 可视化新路径（RViz）
            self._publish_path_marker(robot_name, path)
            # 在Gazebo中更新路径
            self._spawn_path_in_gazebo(robot_name, path)
        else:
            self.get_logger().warn(f'{robot_name} 路径重规划失败！')
    
    def control_loop(self):
        """Main control loop - compute and publish velocities (ORCA mode only)"""
        # 只在ORCA模式下运行，Nav2模式由Nav2控制器接管
        if not self.use_orca:
            return
        
        if len(self.robot_positions) < self.robot_number:
            # Not all robots have sent odometry yet
            self.get_logger().info(
                f'control_loop: 等待odom数据 ({len(self.robot_positions)}/{self.robot_number}) - '
                f'已接收: {list(self.robot_positions.keys())}',
                throttle_duration_sec=3.0
            )
            return
        
        # Create ORCA agents for all robots
        agents = {}
        for robot_name in self.robot_positions.keys():
            # 如果没有目标，将当前位置设为目标(停止)
            if robot_name not in self.robot_goals or not self.robot_goals_initialized.get(robot_name, False):
                # 设置当前位置为临时目标，让机器人停止
                self.robot_goals[robot_name] = self.robot_positions[robot_name].copy()
                self.get_logger().warn(f'{robot_name} has no goal, staying at current position', throttle_duration_sec=5.0)
                # 发布停止命令
                stop_cmd = Twist()
                self.cmd_vel_publishers[robot_name].publish(stop_cmd)
                continue
            
            # 检查是否已到达目标
            position = self.robot_positions[robot_name]
            goal = self.robot_goals[robot_name]
            distance_to_goal = np.linalg.norm(goal - position)
            
            if distance_to_goal < self.goal_tolerance:
                # 已到达目标，发布停止命令
                if not self.robot_goal_reached.get(robot_name, False):
                    self.get_logger().info(f'{robot_name} reached goal! Distance: {distance_to_goal:.3f}m')
                    self.robot_goal_reached[robot_name] = True
                
                # 持续发布停止命令
                stop_cmd = Twist()
                self.cmd_vel_publishers[robot_name].publish(stop_cmd)
                continue
            
            # position和goal已在上面定义过了
            velocity = self.robot_velocities.get(robot_name, np.zeros(2))
            
            # ========== 第1层：Theta*全局路径规划 ==========
            # 检查是否卡住
            if self._is_stuck(robot_name, position):
                self.get_logger().warn(
                    f'{robot_name} 检测到卡住！触发重规划',
                    throttle_duration_sec=3.0
                )
                # 卡住时强制重规划
                self._replan_path(robot_name, position, goal)
                # 重置卡住计数
                self.stuck_count[robot_name] = 0
            
            # 检查是否需要重规划（每移动3米触发一次）
            elif self._should_replan(robot_name, position, goal):
                self._replan_path(robot_name, position, goal)
            
            # 从Theta*路径获取下一个waypoint
            waypoint = self.get_next_waypoint(robot_name)
            if waypoint is not None:
                # 使用waypoint代替最终目标，实现路径跟踪
                local_goal = waypoint
                self.get_logger().info(
                    f'{robot_name}: waypoint=[{waypoint[0]:.2f}, {waypoint[1]:.2f}], '
                    f'pos=[{position[0]:.2f}, {position[1]:.2f}]',
                    throttle_duration_sec=1.0
                )
            else:
                local_goal = goal
                self.get_logger().info(
                    f'{robot_name}: no waypoint, using goal=[{goal[0]:.2f}, {goal[1]:.2f}]',
                    throttle_duration_sec=1.0
                )
            
            # 计算朝向waypoint的期望速度（作为ORCA的输入）
            pref_velocity = compute_preferred_velocity(
                position, local_goal, self.max_linear_speed
            )
            
            # Create ORCA agent
            agent = ORCAAgent(
                position=position,
                velocity=velocity,
                radius=self.robot_radius,
                max_speed=self.max_linear_speed,
                pref_velocity=pref_velocity,
                time_horizon=self.time_horizon
            )
            
            # 保存agent和local_goal用于后续处理
            agents[robot_name] = {
                'agent': agent,
                'local_goal': local_goal
            }
        
        # ========== 第2层：ORCA动态避碰 + 第3层：DWA局部控制 ==========
        for robot_name, agent_data in agents.items():
            agent = agent_data['agent']
            local_goal = agent_data['local_goal']
            
            # Find neighbors within range
            neighbors = []
            
            # 检查所有机器人（包括已到达目标的）
            for other_name in self.robot_positions.keys():
                if other_name == robot_name:
                    continue
                
                other_position = self.robot_positions[other_name]
                distance = np.linalg.norm(agent.position - other_position)
                
                if distance < self.neighbor_distance:
                    # 如果是正在移动的机器人（在agents中），使用其agent
                    if other_name in agents:
                        neighbors.append(agents[other_name]['agent'])
                    else:
                        # 如果是已停止的机器人，创建一个速度为零的agent
                        stopped_agent = ORCAAgent(
                            position=other_position,
                            velocity=np.zeros(2),
                            radius=self.robot_radius,
                            max_speed=0.0,
                            pref_velocity=np.zeros(2),
                            time_horizon=self.time_horizon
                        )
                        neighbors.append(stopped_agent)
            
            # ========== 第2层：ORCA计算理想避让速度 ==========
            orca_velocity = agent.compute_new_velocity(neighbors)
            
            # 调试输出 - 始终输出关键信息
            self.get_logger().info(
                f'{robot_name}: ORCA_vel=[{orca_velocity[0]:.3f}, {orca_velocity[1]:.3f}], '
                f'neighbors={len(neighbors)}',
                throttle_duration_sec=1.0
            )
            
            # ========== 第3层：DWA生成最终控制指令 ==========
            if self.use_dwa and robot_name in self.laser_scans:
                # 从激光数据提取静态障碍物
                obstacles = self._extract_obstacles_from_laser(
                    robot_name, 
                    self.laser_scans[robot_name]
                )
                
                # 添加其他机器人作为动态障碍物
                for other_name in self.robot_positions.keys():
                    if other_name != robot_name:
                        obstacles.append(self.robot_positions[other_name])
                
                # 获取当前速度命令
                current_vel = self.robot_current_cmd_vel.get(robot_name, (0.0, 0.0))
                
                # === [核心修复] DWA 使用实际waypoint而不是ORCA投射 ===
                # 问题：原代码使用ORCA速度投射虚拟目标，当ORCA方向错误时DWA也跟着错
                # 修复：DWA直接朝实际waypoint前进，保证导航方向正确
                
                # 直接使用当前waypoint作为DWA的目标
                dwa_target = local_goal
                
                self.get_logger().info(
                    f'{robot_name}: DWA target=[{dwa_target[0]:.2f}, {dwa_target[1]:.2f}]',
                    throttle_duration_sec=1.0
                )
                
                # DWA 规划：使用实际waypoint
                v, w = self.dwa_planner.plan(
                    agent.position,
                    current_vel,
                    self.robot_yaws.get(robot_name, 0.0),
                    dwa_target,
                    obstacles
                )
                
                # 速度平滑（避免突变）
                alpha = 0.6  # 稍微降低平滑系数 (原0.7)，让加减速反应更灵敏
                v = alpha * v + (1 - alpha) * current_vel[0]
                w = alpha * w + (1 - alpha) * current_vel[1]
                
                self.get_logger().info(
                    f'{robot_name}: cmd_vel=[{v:.3f}, {w:.3f}]',
                    throttle_duration_sec=1.0
                )

                # 创建Twist消息
                cmd_vel = Twist()
                cmd_vel.linear.x = float(v)
                cmd_vel.angular.z = float(w)
                
                # 保存当前命令
                self.robot_current_cmd_vel[robot_name] = (v, w)
                
            else:
                # 如果没有激光数据或未启用DWA，直接使用ORCA速度
                # 转换为Twist消息
                cmd_vel = self.velocity_to_twist(
                    orca_velocity, 
                    agent.position, 
                    local_goal,
                    self.robot_yaws.get(robot_name, 0.0)
                )
            
            # Publish
            self.cmd_vel_publishers[robot_name].publish(cmd_vel)
    
    def _extract_obstacles_from_laser(self, 
                                     robot_name: str, 
                                     scan: LaserScan) -> List[np.ndarray]:
        """从激光数据提取障碍物位置"""
        obstacles = []
        
        if robot_name not in self.robot_positions or robot_name not in self.robot_yaws:
            return obstacles
        
        pos = self.robot_positions[robot_name]
        yaw = self.robot_yaws[robot_name]
        
        angle = scan.angle_min
        for i, r in enumerate(scan.ranges):
            # \u8fc7\u6ee4\u65e0\u6548\u6570\u636e
            if r < scan.range_min or r > scan.range_max or math.isinf(r) or math.isnan(r):
                angle += scan.angle_increment
                continue
            
            # \u53ea\u5904\u7406\u8fd1\u8ddd\u79bb\u969c\u788d\u7269 (3\u7c73\u5185)
            if r > 3.0:
                angle += scan.angle_increment
                continue
            
            # \u8f6c\u6362\u5230\u4e16\u754c\u5750\u6807\u7cfb
            obstacle_x = pos[0] + r * math.cos(yaw + angle)
            obstacle_y = pos[1] + r * math.sin(yaw + angle)
            obstacles.append(np.array([obstacle_x, obstacle_y]))
            
            angle += scan.angle_increment
        
        return obstacles
    
    def velocity_to_twist(self, velocity: np.ndarray, position: np.ndarray, 
                         goal: np.ndarray, current_yaw: float) -> Twist:
        """
        Convert ORCA velocity to Twist command
        
        Args:
            velocity: Desired velocity [vx, vy] in world frame
            position: Current position
            goal: Goal position
            current_yaw: Current yaw angle
            
        Returns:
            Twist command
        """
        twist = Twist()
        
        # Check if reached goal
        distance_to_goal = np.linalg.norm(goal - position)
        if distance_to_goal < 0.2:  # 增加到达阈值
            # Reached goal, stop
            return twist
        
        # Desired heading from velocity
        desired_speed = np.linalg.norm(velocity)
        if desired_speed < 0.01:
            # Stop
            return twist
        
        # 计算期望朝向
        desired_heading = math.atan2(velocity[1], velocity[0])
        
        # Compute heading error
        heading_error = desired_heading - current_yaw
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
        
        # 改进的速度控制：当角度误差大时减速，优先转向
        if abs(heading_error) > math.pi / 4:  # 45度
            # 角度误差太大，主要转向，减速前进
            twist.linear.x = desired_speed * 0.3 * math.cos(heading_error)
        else:
            # 角度误差小，正常前进
            twist.linear.x = desired_speed * math.cos(heading_error)
        
        # 确保线速度为正
        twist.linear.x = max(0.0, twist.linear.x)
        
        # Angular velocity: proportional to heading error
        Kp_angular = 3.0  # 增加角速度增益
        twist.angular.z = np.clip(
            Kp_angular * heading_error, 
            -self.max_angular_speed, 
            self.max_angular_speed
        )
        
        return twist


def main(args=None):
    rclpy.init(args=args)
    node = ORCANavNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
