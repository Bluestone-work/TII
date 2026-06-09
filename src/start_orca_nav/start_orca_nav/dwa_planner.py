"""
Dynamic Window Approach (DWA) Local Planner
用于局部避障和路径跟踪
"""

import numpy as np
import math
from typing import Tuple, List, Optional


class DWAConfig:
    """DWA配置参数"""
    def __init__(self):
        # 机器人参数
        self.max_speed = 0.8  # m/s（提高到0.8，原0.22太慢）
        self.min_speed = 0.0  # m/s (禁止倒车，机器人应该先转向再前进)
        self.max_yaw_rate = 2.5  # rad/s（提高转向速度）
        self.max_accel = 1.0  # m/s^2（提高加速度）
        self.max_delta_yaw_rate = 4.0  # rad/s^2
        
        # 速度采样（优化采样数量）
        self.v_resolution = 0.05  # m/s（从0.02增加到0.05，减少采样点）
        self.yaw_rate_resolution = 0.15  # rad/s（从0.1增加到0.15）
        
        # 预测参数（优化计算量）
        self.predict_time = 1.5  # 预测时间 s（从2.0减少到1.5）
        self.dt = 0.1  # 时间步长 s
        
        # 评价函数权重
        self.heading_weight = 2.0  # 朝向目标的权重
        self.dist_weight = 1.0  # 距离障碍物的权重（降低，避免过度保守）
        self.velocity_weight = 4.0  # 速度的权重（提高，鼓励前进）
        
        # 安全距离
        self.robot_radius = 0.25  # 机器人半径（实际约0.2米，稍微放大）
        self.safety_margin = 0.10  # 额外安全距离（减小到0.1米以适应狭窄通道）
        
        # 障碍物参数
        self.obstacle_check_distance = 2.0  # 检查障碍物的最大距离（减小以提高效率）
        self.trajectory_eval_stride = 2  # 轨迹碰撞检测采样步长


class DWAPlanner:
    """DWA局部路径规划器"""
    
    def __init__(self, config: Optional[DWAConfig] = None, robot_radius: Optional[float] = None):
        """
        初始化DWA规划器
        
        Args:
            config: DWA配置参数
            robot_radius: 机器人半径（如果提供，会覆盖config中的值）
        """
        self.config = config if config is not None else DWAConfig()
        
        # 如果提供了robot_radius，覆盖config中的值
        if robot_radius is not None:
            self.config.robot_radius = robot_radius
    
    def plan(self, 
             current_pos: np.ndarray,  # [x, y]
             current_vel: Tuple[float, float],  # (linear_v, angular_w)
             current_yaw: float,
             goal_pos: np.ndarray,  # [x, y]
             obstacles: List[np.ndarray]) -> Tuple[float, float]:
        """
        计算最优速度命令
        
        Args:
            current_pos: 当前位置 [x, y]
            current_vel: 当前速度 (linear_v, angular_w)
            current_yaw: 当前朝向 (rad)
            goal_pos: 目标位置 [x, y]
            obstacles: 障碍物位置列表 [[x1, y1], [x2, y2], ...]
            
        Returns:
            (linear_v, angular_w): 最优线速度和角速度
        """
        # 过滤和聚类障碍物，减少噪声
        filtered_obstacles = self._filter_obstacles(current_pos, obstacles)
        
        # 计算动态窗口
        dw = self._calculate_dynamic_window(current_vel)
        
        # 评估所有可能的速度
        best_v = 0.0
        best_w = 0.0
        best_score = -float('inf')
        
        for v in np.arange(dw[0], dw[1], self.config.v_resolution):
            for w in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):
                # 预测轨迹（包括朝向）
                trajectory, yaws = self._predict_trajectory(
                    current_pos, current_yaw, v, w
                )
                
                # 评估轨迹
                score = self._evaluate_trajectory(
                    trajectory, yaws, goal_pos, filtered_obstacles, v, w
                )
                
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
        
        # 如果没有找到任何可行速度（best_score仍为-inf），尝试慢速前进
        if best_score == -float('inf'):
            # 计算朝向目标的角度差
            dx = goal_pos[0] - current_pos[0]
            dy = goal_pos[1] - current_pos[1]
            goal_angle = math.atan2(dy, dx)
            angle_diff = goal_angle - current_yaw
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 如果角度差较大，优先转向
            if abs(angle_diff) > 0.5:  # 约30度
                best_v = 0.1  # 慢速前进
                best_w = 0.3 * (1.0 if angle_diff > 0 else -1.0)  # 转向
            else:
                # 角度差小，慢速直行
                best_v = 0.15
                best_w = 0.0
        
        return best_v, best_w
    
    def _calculate_dynamic_window(self, 
                                  current_vel: Tuple[float, float]) -> np.ndarray:
        """
        计算动态窗口
        
        Args:
            current_vel: 当前速度 (v, w)
            
        Returns:
            [v_min, v_max, w_min, w_max]
        """
        v, w = current_vel
        
        # 机器人性能限制
        Vs = [
            self.config.min_speed,
            self.config.max_speed,
            -self.config.max_yaw_rate,
            self.config.max_yaw_rate
        ]
        
        # 加速度限制 (动态窗口)
        Vd = [
            v - self.config.max_accel * self.config.dt,
            v + self.config.max_accel * self.config.dt,
            w - self.config.max_delta_yaw_rate * self.config.dt,
            w + self.config.max_delta_yaw_rate * self.config.dt
        ]
        
        # 取交集
        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3])
        ]
        
        return np.array(dw)
    
    def _predict_trajectory(self,
                           pos: np.ndarray,
                           yaw: float,
                           v: float,
                           w: float) -> Tuple[np.ndarray, List[float]]:
        """
        预测轨迹
        
        Args:
            pos: 起始位置 [x, y]
            yaw: 起始朝向
            v: 线速度
            w: 角速度
            
        Returns:
            (trajectory, yaws): 轨迹点数组和对应的朝向角列表
        """
        trajectory = [pos.copy()]
        yaws = [yaw]
        x, y = pos[0], pos[1]
        theta = yaw
        
        time = 0.0
        while time <= self.config.predict_time:
            x += v * math.cos(theta) * self.config.dt
            y += v * math.sin(theta) * self.config.dt
            theta += w * self.config.dt
            trajectory.append(np.array([x, y]))
            yaws.append(theta)
            time += self.config.dt
        
        return np.array(trajectory), yaws
    
    def _evaluate_trajectory(self,
                            trajectory: np.ndarray,
                            yaws: List[float],
                            goal: np.ndarray,
                            obstacles: List[np.ndarray],
                            velocity: float,
                            angular_velocity: float) -> float:
        """
        评估轨迹得分
        
        Args:
            trajectory: 轨迹点
            yaws: 轨迹各点的朝向角
            goal: 目标位置
            obstacles: 障碍物列表
            velocity: 线速度
            angular_velocity: 角速度
            
        Returns:
            得分 (越高越好)
        """
        # 1. 朝向目标得分（使用轨迹实际朝向）
        end_point = trajectory[-1]
        end_yaw = yaws[-1]
        
        # 计算目标方向
        dx = goal[0] - end_point[0]
        dy = goal[1] - end_point[1]
        goal_angle = math.atan2(dy, dx)
        goal_distance = math.sqrt(dx**2 + dy**2)
        
        # 角度差（归一化到[-pi, pi]）
        angle_diff = goal_angle - end_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # 朝向得分：角度差越小越好（0-1）
        heading_score = 1.0 - (abs(angle_diff) / math.pi)
        
        # 速度评分：鼓励前进而非原地转
        velocity_score = abs(velocity) / self.config.max_speed
        
        # 如果线速度太小，降低得分（避免DWA总是选v=0）
        if abs(velocity) < 0.05:
            velocity_score = -0.15 if goal_distance > 0.30 else 0.0
        
        # 2. 障碍物距离得分
        min_obstacle_dist = float('inf')
        collision = False
        
        # 如果没有障碍物，给满分
        if len(obstacles) == 0:
            dist_score = 1.0
        else:
            sample_stride = max(1, int(self.config.trajectory_eval_stride))
            sampled_points = list(trajectory[::sample_stride])
            if len(sampled_points) == 0 or not np.array_equal(sampled_points[-1], trajectory[-1]):
                sampled_points.append(trajectory[-1])

            for point in sampled_points:
                for obs in obstacles:
                    dist = np.linalg.norm(point - obs)
                    if dist < min_obstacle_dist:
                        min_obstacle_dist = dist
                    
                    # 检查碰撞（使用更小的安全距离）
                    collision_threshold = self.config.robot_radius + self.config.safety_margin
                    if dist < collision_threshold:
                        collision = True
                        break
                
                if collision:
                    break
            
            if collision:
                return -float('inf')  # 碰撞轨迹直接排除
            
            # 障碍物距离得分：距离越远越好（归一化到0-1）
            # 使用软饱和函数，避免过度惩罚接近障碍物的轨迹
            if min_obstacle_dist > self.config.obstacle_check_distance:
                dist_score = 1.0
            else:
                # 使用平方根函数，使得接近障碍物时得分下降更缓慢
                dist_score = math.sqrt(min_obstacle_dist / self.config.obstacle_check_distance)
        
        # 综合得分（所有得分都归一化到0-1）
        total_score = (
            self.config.heading_weight * heading_score +
            self.config.dist_weight * dist_score +
            self.config.velocity_weight * velocity_score
        )
        
        return total_score
    
    def _filter_obstacles(self, 
                         current_pos: np.ndarray,
                         obstacles: List[np.ndarray],
                         grid_size: float = 0.3) -> List[np.ndarray]:
        """
        过滤和聚类障碍物，减少噪声
        
        Args:
            current_pos: 当前位置
            obstacles: 原始障碍物列表
            grid_size: 网格大小，用于聚类
            
        Returns:
            过滤后的障碍物列表
        """
        if len(obstacles) == 0:
            return []
        
        close_obstacles = []
        for obs in obstacles:
            dist = np.linalg.norm(obs - current_pos)
            if dist < self.config.obstacle_check_distance:
                close_obstacles.append(obs)
        
        if len(close_obstacles) == 0:
            return []
        
        # 使用网格聚类减少点数
        grid_dict = {}
        for obs in close_obstacles:
            # 计算网格索引
            grid_x = int(obs[0] / grid_size)
            grid_y = int(obs[1] / grid_size)
            grid_key = (grid_x, grid_y)
            
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(obs)
        
        # 每个网格用中心点代表
        filtered = []
        for points in grid_dict.values():
            center = np.mean(points, axis=0)
            filtered.append(center)
        
        return filtered


def create_dwa_planner(max_speed: float = 0.22,
                       max_yaw_rate: float = 2.0,
                       robot_radius: float = 0.35) -> DWAPlanner:
    """
    创建DWA规划器的便捷函数
    
    Args:
        max_speed: 最大线速度
        max_yaw_rate: 最大角速度
        robot_radius: 机器人半径
        
    Returns:
        DWAPlanner实例
    """
    config = DWAConfig()
    config.max_speed = max_speed
    config.max_yaw_rate = max_yaw_rate
    config.robot_radius = robot_radius
    
    return DWAPlanner(config)
