#!/usr/bin/env python3
"""
Multi-Robot Navigation Node using Nav2 Global Planner + ORCA/DWA Local Control

Architecture:
1. Nav2 Global Planner (ComputePathToPose): Plans global path using map
2. ORCA Layer: Computes ideal velocity avoiding other robots
3. DWA Local Planner: Generates final control commands with laser obstacle avoidance
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist, PoseStamped, Pose, Point
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import ComputePathToPose
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
import numpy as np
import math
import heapq
import yaml
from PIL import Image
from typing import Dict, List, Optional, Tuple

from start_orca_nav.orca_algorithm import ORCAAgent
from start_orca_nav.dwa_planner import DWAPlanner


class AStarPlanner:
    """Simple A* path planner for grid-based maps"""
    def __init__(self, map_data: np.ndarray, resolution: float, origin: List[float]):
        self.map_data = map_data  # 2D numpy array, 0=free, 100=occupied
        self.resolution = resolution
        self.origin = origin  # [x, y, z]
        self.width = map_data.shape[1]
        self.height = map_data.shape[0]
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        gx = int((x - self.origin[0]) / self.resolution)
        gy = int((y - self.origin[1]) / self.resolution)
        return (gx, gy)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates"""
        x = gx * self.resolution + self.origin[0]
        y = gy * self.resolution + self.origin[1]
        return (x, y)
    
    def is_valid(self, gx: int, gy: int, safety_radius: int = 2) -> bool:
        """Check if grid cell is valid and free with safety margin"""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            return False
        
        # Check the cell itself
        if self.map_data[gy, gx] >= 50:
            return False
        
        # Check safety radius around the cell (inflate obstacles)
        for dx in range(-safety_radius, safety_radius + 1):
            for dy in range(-safety_radius, safety_radius + 1):
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.map_data[ny, nx] >= 50:  # Obstacle nearby
                        return False
        
        return True
    
    def get_neighbors(self, gx: int, gy: int) -> List[Tuple[int, int]]:
        """Get valid neighbors (8-connected) with safety checking"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = gx + dx, gy + dy
                # Use safety_radius=2 to keep path away from walls (2*0.05m = 0.1m margin)
                if self.is_valid(nx, ny, safety_radius=2):
                    neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """Euclidean distance heuristic"""
        return math.sqrt((gx1 - gx2)**2 + (gy1 - gy2)**2)
    
    def plan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[List[float]]]:
        """A* path planning from start to goal (world coordinates)"""
        start_grid = self.world_to_grid(start[0], start[1])
        goal_grid = self.world_to_grid(goal[0], goal[1])
        
        if not self.is_valid(*start_grid) or not self.is_valid(*goal_grid):
            return None
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(*start_grid, *goal_grid)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # Reconstruct path
                path = []
                while current in came_from:
                    wx, wy = self.grid_to_world(*current)
                    path.append([wx, wy])
                    current = came_from[current]
                path.reverse()
                
                # Simplify path (keep every 5th waypoint + goal)
                if len(path) > 10:
                    simplified = [path[i] for i in range(0, len(path), 5)]
                    if simplified[-1] != path[-1]:
                        simplified.append(path[-1])
                    return simplified
                return path
            
            for neighbor in self.get_neighbors(*current):
                tentative_g = g_score[current] + self.heuristic(*current, *neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(*neighbor, *goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found


class ORCANavNode(Node):
    """ORCA navigation node with Nav2 global planning"""
    
    def __init__(self):
        super().__init__('orca_nav_node')
        
        # Parameters
        self.declare_parameter('robot_number', 4)
        self.declare_parameter('robot_radius', 0.35)
        self.declare_parameter('neighbor_distance', 5.0)
        self.declare_parameter('time_horizon', 2.0)
        self.declare_parameter('goal_tolerance', 0.3)
        self.declare_parameter('map_file', '')
        
        self.robot_number = self.get_parameter('robot_number').value
        self.robot_radius = self.get_parameter('robot_radius').value
        self.neighbor_distance = self.get_parameter('neighbor_distance').value
        self.time_horizon = self.get_parameter('time_horizon').value
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.map_file = self.get_parameter('map_file').value
        
        # Initialize DWA planner
        self.dwa_planner = DWAPlanner(robot_radius=self.robot_radius)
        self.get_logger().info('DWA local planner initialized')
        
        # Initialize A* planner with map
        self.astar_planner = None
        if self.map_file:
            self._load_map(self.map_file)
        
        # Robot states
        self.robot_positions = {}  # {robot_name: np.array([x, y])}
        self.robot_velocities = {}  # {robot_name: np.array([vx, vy])}
        self.robot_yaws = {}  # {robot_name: float}
        self.robot_goals = {}  # {robot_name: np.array([x, y])}
        self.robot_goal_reached = {}  # {robot_name: bool}
        
        # Nav2 paths
        self.nav2_paths = {}  # {robot_name: List[(x, y)]}
        self.current_waypoint_index = {}  # {robot_name: int}
        
        # ORCA agents
        self.orca_agents = {}  # {robot_name: ORCAAgent}
        
        # Laser scans
        self.laser_scans = {}  # {robot_name: LaserScan}
        
        # Current cmd_vel for smoothing
        self.robot_current_cmd_vel = {}  # {robot_name: (v, w)}
        
        # Publishers and subscribers
        self.cmd_vel_publishers = {}
        self.odom_subscribers = {}
        self.goal_subscribers = {}
        self.laser_subscribers = {}
        self.goal_marker_publishers = {}
        self.path_marker_publishers = {}
        self.nav2_planner_clients = {}  # ComputePathToPose action clients
        
        # Gazebo services
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        self.gazebo_goal_models = {}  # {robot_name: model_name}
        self.gazebo_path_models = {}  # {robot_name: [model_names]}
        
        for i in range(self.robot_number):
            robot_name = f'robot{i}'
            gazebo_namespace = f'my_bot{i}'
            
            # Publishers
            self.cmd_vel_publishers[robot_name] = self.create_publisher(
                Twist, f'/{gazebo_namespace}/cmd_vel', 10
            )
            self.goal_marker_publishers[robot_name] = self.create_publisher(
                Marker, f'/{robot_name}/goal_marker', 10
            )
            self.path_marker_publishers[robot_name] = self.create_publisher(
                Marker, f'/{robot_name}/path_marker', 10
            )
            
            # Subscribers
            self.odom_subscribers[robot_name] = self.create_subscription(
                Odometry, f'/{gazebo_namespace}/odom',
                lambda msg, name=robot_name: self.odom_callback(msg, name),
                10
            )
            self.goal_subscribers[robot_name] = self.create_subscription(
                PoseStamped, f'/robot{i}/goal_pose',
                lambda msg, name=robot_name: self.goal_callback(msg, name),
                10
            )
            self.laser_subscribers[robot_name] = self.create_subscription(
                LaserScan, f'/{gazebo_namespace}/scan',
                lambda msg, name=robot_name: self.laser_callback(msg, name),
                10
            )
            
            # Nav2 planner client
            self.nav2_planner_clients[robot_name] = ActionClient(
                self, ComputePathToPose, f'/{gazebo_namespace}/compute_path_to_pose'
            )
            
            # Initialize ORCA agent
            self.orca_agents[robot_name] = ORCAAgent(
                position=np.array([0.0, 0.0]),
                velocity=np.array([0.0, 0.0]),
                radius=self.robot_radius,
                max_speed=0.8,
                pref_velocity=np.array([0.0, 0.0]),
                time_horizon=self.time_horizon
            )
        
        # Control loop timer (20 Hz)
        self.timer = self.create_timer(0.05, self.control_loop)
        
        # Marker republish timer (1 Hz)
        self.marker_timer = self.create_timer(1.0, self.republish_markers)
        
        self.get_logger().info(f'ORCA Nav Node initialized with Nav2 global planning for {self.robot_number} robots')
        if self.map_file:
            self.get_logger().info(f'Using map file: {self.map_file}')
    
    def odom_callback(self, msg: Odometry, robot_name: str):
        """Update robot odometry"""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_positions[robot_name] = np.array([x, y])
        
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.robot_velocities[robot_name] = np.array([vx, vy])
        
        # Extract yaw
        quat = msg.pose.pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_yaws[robot_name] = yaw
        
        # Update ORCA agent
        if robot_name in self.orca_agents:
            self.orca_agents[robot_name].position = self.robot_positions[robot_name]
            self.orca_agents[robot_name].velocity = self.robot_velocities[robot_name]
    
    def laser_callback(self, msg: LaserScan, robot_name: str):
        """Store laser scan data"""
        self.laser_scans[robot_name] = msg
        # Debug: log laser reception with valid range count
        valid_ranges = sum(1 for r in msg.ranges if msg.range_min < r < msg.range_max)
        self.get_logger().info(
            f'{robot_name} laser: {len(msg.ranges)} points, {valid_ranges} valid (min={msg.range_min:.2f}, max={msg.range_max:.2f})',
            throttle_duration_sec=5.0
        )
    
    def goal_callback(self, msg: PoseStamped, robot_name: str):
        """Handle new goal"""
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.robot_goals[robot_name] = np.array([x, y])
        self.robot_goal_reached[robot_name] = False
        
        # Clear old path and waypoint index to avoid using stale navigation data
        if robot_name in self.nav2_paths:
            del self.nav2_paths[robot_name]
        if robot_name in self.current_waypoint_index:
            del self.current_waypoint_index[robot_name]
        
        self.get_logger().info(f'{robot_name} received new goal: [{x:.2f}, {y:.2f}], cleared old path')
        
        # Visualize goal
        self.publish_goal_marker(robot_name, x, y)
        self.spawn_goal_in_gazebo(robot_name, x, y)
        
        # Request Nav2 path
        self._request_nav2_path(robot_name, msg)
    
    def _request_nav2_path(self, robot_name: str, goal_msg: PoseStamped):
        """Request global path from Nav2"""
        if robot_name not in self.robot_positions:
            self.get_logger().warn(f'{robot_name} position unknown')
            return
        
        client = self.nav2_planner_clients[robot_name]
        if not client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn(
                f'{robot_name} Nav2 planner not available, using A* planning',
                throttle_duration_sec=5.0
            )
            # Fallback: Use A* planning with map
            x = goal_msg.pose.position.x
            y = goal_msg.pose.position.y
            goal_position = np.array([x, y])
            robot_position = self.robot_positions[robot_name]
            self.get_logger().info(
                f'{robot_name} A* planning from [{robot_position[0]:.2f}, {robot_position[1]:.2f}] '
                f'to [{x:.2f}, {y:.2f}]'
            )
            path = self._plan_path_astar(robot_position, goal_position)
            if path and len(path) > 0:
                self.nav2_paths[robot_name] = path
                self.get_logger().info(f'{robot_name} A* path with {len(path)} waypoints')
                # Visualize path in Gazebo
                self._spawn_path_in_gazebo(robot_name, path)
            else:
                # If A* fails, use direct path
                self.nav2_paths[robot_name] = [robot_position.tolist(), [x, y]]
                self.get_logger().warn(f'{robot_name} A* failed, using direct path')
            # Don't reset waypoint index here - let get_next_waypoint find closest point
            if robot_name in self.current_waypoint_index:
                del self.current_waypoint_index[robot_name]
            return
        
        # Create request
        goal = ComputePathToPose.Goal()
        goal.goal = goal_msg
        
        # Set start position
        goal.start = PoseStamped()
        goal.start.header.frame_id = 'map'
        goal.start.header.stamp = self.get_clock().now().to_msg()
        pos = self.robot_positions[robot_name]
        goal.start.pose.position.x = pos[0]
        goal.start.pose.position.y = pos[1]
        goal.start.pose.position.z = 0.0
        goal.start.pose.orientation.w = 1.0
        
        goal.planner_id = ''  # Use default planner
        goal.use_start = True
        
        # Send request
        self.get_logger().info(f'{robot_name} Requesting Nav2 path...')
        future = client.send_goal_async(goal)
        future.add_done_callback(
            lambda f: self._nav2_goal_response(f, robot_name)
        )
    
    def _nav2_goal_response(self, future, robot_name: str):
        """Handle Nav2 goal response"""
        try:
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().warn(f'{robot_name} Nav2 path request rejected')
                return
            
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(
                lambda f: self._nav2_path_result(f, robot_name)
            )
        except Exception as e:
            self.get_logger().error(f'{robot_name} Nav2 request error: {e}')
    
    def _nav2_path_result(self, future, robot_name: str):
        """Handle Nav2 path result"""
        try:
            result = future.result().result
            path_msg = result.path
            
            if not path_msg.poses or len(path_msg.poses) == 0:
                self.get_logger().warn(f'{robot_name} Nav2 returned empty path')
                return
            
            # Convert to waypoint list
            path = [(pose.pose.position.x, pose.pose.position.y) 
                    for pose in path_msg.poses]
            
            self.nav2_paths[robot_name] = path
            # Don't reset waypoint index here - let get_next_waypoint find closest point
            if robot_name in self.current_waypoint_index:
                del self.current_waypoint_index[robot_name]
            
            self.get_logger().info(
                f'{robot_name} Nav2 path received: {len(path)} waypoints'
            )
            
            # Visualize
            self._publish_path_marker(robot_name, path)
            self._spawn_path_in_gazebo(robot_name, path)
            
        except Exception as e:
            self.get_logger().error(f'{robot_name} Nav2 result error: {e}')
    
    def control_loop(self):
        """Main control loop"""
        for robot_name in self.robot_positions.keys():
            if robot_name not in self.robot_goals:
                continue
            
            if self.robot_goal_reached.get(robot_name, False):
                continue
            
            # Check if reached goal
            position = self.robot_positions[robot_name]
            goal = self.robot_goals[robot_name]
            dist_to_goal = np.linalg.norm(goal - position)
            
            if dist_to_goal < self.goal_tolerance:
                self.robot_goal_reached[robot_name] = True
                self.get_logger().info(f'{robot_name} reached goal!')
                # Stop robot
                cmd_vel = Twist()
                self.cmd_vel_publishers[robot_name].publish(cmd_vel)
                continue
            
            # Check if we have a valid path
            if robot_name not in self.nav2_paths:
                # No path yet, stop and wait for planning
                cmd_vel = Twist()
                self.cmd_vel_publishers[robot_name].publish(cmd_vel)
                self.get_logger().info(
                    f'{robot_name} waiting for path planning...',
                    throttle_duration_sec=1.0
                )
                continue
            
            # Get next waypoint
            waypoint = self.get_next_waypoint(robot_name)
            if waypoint is None:
                waypoint = goal
            
            # Debug: log distance to waypoint
            dist_to_waypoint = np.linalg.norm(waypoint - position)
            self.get_logger().debug(
                f'{robot_name} target waypoint at ({waypoint[0]:.2f}, {waypoint[1]:.2f}), '
                f'distance={dist_to_waypoint:.2f}m'
            )
            
            # Update ORCA agent preferred velocity
            agent = self.orca_agents[robot_name]
            direction = waypoint - position
            dist = np.linalg.norm(direction)
            if dist > 0.01:
                direction = direction / dist
                pref_speed = min(0.8, dist)
                agent.pref_velocity = direction * pref_speed
            else:
                agent.pref_velocity = np.array([0.0, 0.0])
            
            # Get neighbors
            neighbors = []
            for other_name, other_agent in self.orca_agents.items():
                if other_name != robot_name:
                    dist = np.linalg.norm(other_agent.position - agent.position)
                    if dist < self.neighbor_distance:
                        neighbors.append(other_agent)
            
            # ORCA velocity
            orca_velocity = agent.compute_new_velocity(neighbors)
            
            self.get_logger().info(
                f'{robot_name}: waypoint=[{waypoint[0]:.2f}, {waypoint[1]:.2f}], '
                f'ORCA_vel=[{orca_velocity[0]:.3f}, {orca_velocity[1]:.3f}]',
                throttle_duration_sec=1.0
            )
            
            # DWA local planning
            if robot_name in self.laser_scans:
                obstacles = self._extract_obstacles_from_laser(robot_name, self.laser_scans[robot_name])
                laser_obstacles_count = len(obstacles)
                
                # Add other robots as obstacles
                for other_name in self.robot_positions.keys():
                    if other_name != robot_name:
                        obstacles.append(self.robot_positions[other_name])
                
                current_vel = self.robot_current_cmd_vel.get(robot_name, (0.0, 0.0))
                
                v, w = self.dwa_planner.plan(
                    agent.position,
                    current_vel,
                    self.robot_yaws.get(robot_name, 0.0),
                    waypoint,
                    obstacles
                )
                
                # Velocity smoothing
                alpha = 0.6
                v = alpha * v + (1 - alpha) * current_vel[0]
                w = alpha * w + (1 - alpha) * current_vel[1]
                
                self.robot_current_cmd_vel[robot_name] = (v, w)
                
                cmd_vel = Twist()
                cmd_vel.linear.x = float(v)
                cmd_vel.angular.z = float(w)
                
                self.get_logger().info(
                    f'{robot_name} DWA: v={v:.3f}, w={w:.3f}, laser_obs={laser_obstacles_count}, robot_obs={len(obstacles)-laser_obstacles_count}, total={len(obstacles)}',
                    throttle_duration_sec=2.0
                )
            else:
                # No laser, use ORCA velocity directly
                self.get_logger().warn(
                    f'{robot_name} no laser data, using ORCA only',
                    throttle_duration_sec=5.0
                )
                cmd_vel = self._velocity_to_twist(
                    orca_velocity, agent.position, waypoint,
                    self.robot_yaws.get(robot_name, 0.0)
                )
            
            self.cmd_vel_publishers[robot_name].publish(cmd_vel)
    
    def get_next_waypoint(self, robot_name: str) -> Optional[np.ndarray]:
        """Get next waypoint from path with look-ahead"""
        if robot_name not in self.nav2_paths:
            return self.robot_goals.get(robot_name)
        
        path = self.nav2_paths[robot_name]
        if not path:
            return self.robot_goals.get(robot_name)
        
        position = self.robot_positions[robot_name]
        
        # Initialize: find closest waypoint as starting point
        if robot_name not in self.current_waypoint_index:
            min_dist = float('inf')
            closest_idx = 0
            for i, waypoint in enumerate(path):
                dist = math.sqrt(
                    (waypoint[0] - position[0])**2 + 
                    (waypoint[1] - position[1])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            self.current_waypoint_index[robot_name] = closest_idx
            self.get_logger().info(f'{robot_name} starting from waypoint {closest_idx}/{len(path)}')
        
        idx = self.current_waypoint_index[robot_name]
        if idx >= len(path):
            idx = len(path) - 1
            self.current_waypoint_index[robot_name] = idx
        
        current_waypoint = path[idx]
        dist_to_current = math.sqrt(
            (current_waypoint[0] - position[0])**2 + 
            (current_waypoint[1] - position[1])**2
        )
        
        # Switch to next waypoint if close enough (reduced threshold)
        if dist_to_current < 0.3 and idx < len(path) - 1:
            idx += 1
            self.current_waypoint_index[robot_name] = idx
            current_waypoint = path[idx]
            self.get_logger().info(
                f'{robot_name} switching to waypoint {idx}/{len(path)}',
                throttle_duration_sec=2.0
            )
        
        # Look-ahead: if current waypoint is close, target next one
        # This helps smooth navigation and reduces oscillation
        if dist_to_current < 1.0 and idx < len(path) - 1:
            # Look 1-2 waypoints ahead
            lookahead_idx = min(idx + 1, len(path) - 1)
            lookahead_waypoint = path[lookahead_idx]
            # Blend current and lookahead waypoint
            blend_factor = min(1.0, dist_to_current / 1.0)  # 0 when very close, 1 when far
            target = (
                blend_factor * np.array(current_waypoint) +
                (1 - blend_factor) * np.array(lookahead_waypoint)
            )
            return target
        
        return np.array(current_waypoint)
    
    def _extract_obstacles_from_laser(self, robot_name: str, scan: LaserScan) -> List[np.ndarray]:
        """Extract obstacles from laser scan"""
        obstacles = []
        
        if robot_name not in self.robot_positions or robot_name not in self.robot_yaws:
            return obstacles
        
        pos = self.robot_positions[robot_name]
        yaw = self.robot_yaws[robot_name]
        
        angle = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                # Transform to world coordinates
                world_angle = yaw + angle
                obs_x = pos[0] + r * math.cos(world_angle)
                obs_y = pos[1] + r * math.sin(world_angle)
                obstacles.append(np.array([obs_x, obs_y]))
            angle += scan.angle_increment
        
        return obstacles
    
    def _velocity_to_twist(self, velocity: np.ndarray, position: np.ndarray, 
                          goal: np.ndarray, current_yaw: float) -> Twist:
        """Convert 2D velocity to Twist command"""
        cmd_vel = Twist()
        
        # Linear velocity (magnitude)
        speed = np.linalg.norm(velocity)
        cmd_vel.linear.x = float(min(speed, 0.8))
        
        # Angular velocity (point towards goal)
        dx = goal[0] - position[0]
        dy = goal[1] - position[1]
        target_yaw = math.atan2(dy, dx)
        
        yaw_diff = target_yaw - current_yaw
        while yaw_diff > math.pi:
            yaw_diff -= 2 * math.pi
        while yaw_diff < -math.pi:
            yaw_diff += 2 * math.pi
        
        cmd_vel.angular.z = float(np.clip(yaw_diff * 2.0, -2.5, 2.5))
        
        return cmd_vel
    
    def _load_map(self, map_file: str):
        """Load map from YAML file and initialize A* planner"""
        try:
            with open(map_file, 'r') as f:
                map_info = yaml.safe_load(f)
            
            # Get map parameters
            resolution = map_info['resolution']
            origin = map_info['origin']
            image_file = map_info['image']
            
            # Load occupancy grid image
            import os
            map_dir = os.path.dirname(map_file)
            image_path = os.path.join(map_dir, image_file)
            
            img = Image.open(image_path).convert('L')
            map_array = np.array(img)
            
            # Convert to occupancy values (0=free, 100=occupied)
            # White (255) = free, Black (0) = occupied
            map_data = 100 - (map_array / 255.0 * 100)
            map_data = map_data.astype(np.int8)
            
            # Flip Y axis (image Y is down, map Y is up)
            map_data = np.flipud(map_data)
            
            self.astar_planner = AStarPlanner(map_data, resolution, origin)
            self.get_logger().info(f'Map loaded: {image_path}, size={map_data.shape}, resolution={resolution}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load map: {e}')
            self.astar_planner = None
    
    def _plan_path_astar(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[List[float]]]:
        """Plan path using A* planner"""
        if self.astar_planner is None:
            return None
        
        try:
            path = self.astar_planner.plan(start, goal)
            return path
        except Exception as e:
            self.get_logger().error(f'A* planning failed: {e}')
            return None
    
    def publish_goal_marker(self, robot_name: str, x: float, y: float):
        """Publish goal marker for RViz"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'{robot_name}_goal'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.3
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        
        # Color by robot ID
        robot_id = int(robot_name.replace('robot', ''))
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        color = colors[robot_id % len(colors)]
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 0.8
        
        self.goal_marker_publishers[robot_name].publish(marker)
        self.get_logger().info(f'Published goal marker for {robot_name}', throttle_duration_sec=5.0)
    
    def _publish_path_marker(self, robot_name: str, path: List[Tuple[float, float]]):
        """Publish path marker for RViz"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'{robot_name}_path'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        marker.scale.x = 0.05
        
        robot_id = int(robot_name.replace('robot', ''))
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        color = colors[robot_id % len(colors)]
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 0.6
        
        for x, y in path:
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.05
            marker.points.append(p)
        
        self.path_marker_publishers[robot_name].publish(marker)
    
    def spawn_goal_in_gazebo(self, robot_name: str, x: float, y: float):
        """Spawn goal marker in Gazebo"""
        # Delete old goal model if exists
        if robot_name in self.gazebo_goal_models:
            old_model_name = self.gazebo_goal_models[robot_name]
            delete_req = DeleteEntity.Request()
            delete_req.name = old_model_name
            self.delete_entity_client.call_async(delete_req)
        
        # Create new goal model
        model_name = f'{robot_name}_goal'
        
        # Color by robot ID
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            ('1 0 0 0.8', 'Gazebo/Red'),      # robot0: red
            ('0 1 0 0.8', 'Gazebo/Green'),    # robot1: green
            ('0 0 1 0.8', 'Gazebo/Blue'),     # robot2: blue
            ('1 1 0 0.8', 'Gazebo/Yellow'),   # robot3: yellow
        ]
        color_rgba, color_name = colors[robot_id % len(colors)]
        
        # SDF model for a sphere
        sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.2</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
          <emissive>0.2 0.2 0.2 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
        
        # Spawn entity request
        req = SpawnEntity.Request()
        req.name = model_name
        req.xml = sdf
        req.robot_namespace = ''
        req.initial_pose = Pose()
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = 0.2  # Slightly above ground
        req.initial_pose.orientation.w = 1.0
        req.reference_frame = 'world'
        
        # Call service
        future = self.spawn_entity_client.call_async(req)
        self.gazebo_goal_models[robot_name] = model_name
        
        self.get_logger().info(
            f'Spawned goal in Gazebo for {robot_name} at ({x:.2f}, {y:.2f})',
            throttle_duration_sec=5.0
        )
    
    def _spawn_path_in_gazebo(self, robot_name: str, path: List[List[float]]):
        """Spawn path markers in Gazebo"""
        # Delete old path models if exist
        if robot_name in self.gazebo_path_models:
            for old_model_name in self.gazebo_path_models[robot_name]:
                delete_req = DeleteEntity.Request()
                delete_req.name = old_model_name
                self.delete_entity_client.call_async(delete_req)
        
        self.gazebo_path_models[robot_name] = []
        
        # Color by robot ID (lighter version for path)
        robot_id = int(robot_name.replace('robot', ''))
        colors = [
            '1 0.5 0.5 0.6',    # robot0: light red
            '0.5 1 0.5 0.6',    # robot1: light green
            '0.5 0.5 1 0.6',    # robot2: light blue
            '1 1 0.5 0.6',      # robot3: light yellow
        ]
        color_rgba = colors[robot_id % len(colors)]
        
        # Spawn spheres for waypoints (skip every few to avoid clutter)
        step = max(1, len(path) // 10)  # Show max 10 waypoints
        for i in range(0, len(path), step):
            waypoint = path[i]
            x, y = waypoint[0], waypoint[1]
            
            model_name = f'{robot_name}_waypoint_{i}'
            
            # Smaller sphere for waypoints
            sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <sphere>
            <radius>0.1</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>{color_rgba}</ambient>
          <diffuse>{color_rgba}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
            
            req = SpawnEntity.Request()
            req.name = model_name
            req.xml = sdf
            req.robot_namespace = ''
            req.initial_pose = Pose()
            req.initial_pose.position.x = x
            req.initial_pose.position.y = y
            req.initial_pose.position.z = 0.1  # Just above ground
            req.initial_pose.orientation.w = 1.0
            req.reference_frame = 'world'
            
            self.spawn_entity_client.call_async(req)
            self.gazebo_path_models[robot_name].append(model_name)
        
        self.get_logger().info(
            f'Spawned {len(self.gazebo_path_models[robot_name])} path markers in Gazebo for {robot_name}',
            throttle_duration_sec=5.0
        )
    
    def republish_markers(self):
        """Republish markers periodically"""
        for robot_name, goal in self.robot_goals.items():
            if not self.robot_goal_reached.get(robot_name, False):
                self.publish_goal_marker(robot_name, goal[0], goal[1])


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
