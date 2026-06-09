"""
单机器人Gym环境包装器
将现有的多机器人环境包装成单个机器人的独立Gym环境
其他机器人被视为动态障碍物
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rclpy
from rclpy.node import Node
import math
import time
# 导入全局规划器
from sb3_training.global_planner import AStarPlanner, WaypointExtractor
from sb3_training.waypoint_visualizer import WaypointVisualizer

# 导入现有的环境逻辑
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../start_reinforcement_learning'))
from start_reinforcement_learning.env_logic.logic import Env as MultiRobotEnv


class SingleRobotGymEnv(gym.Env):
    """
    单机器人Gym环境
    
    - 每个机器人独立训练，有自己的RecurrentPPO策略
    - 将其他机器人视为动态障碍物（在激光雷达中可见）
    - 兼容Stable-Baselines3的RecurrentPPO
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, robot_id=0, total_robots=1, map_number=3, 
                 use_random_mode=False, max_episode_steps=300):
        """
        初始化单机器人环境
        
        Args:
            robot_id: 当前机器人的ID (0, 1, 2, ...)
            total_robots: 总机器人数量
            map_number: 地图编号
            use_random_mode: 是否使用随机起始位置
            max_episode_steps: 最大步数
        """
        super(SingleRobotGymEnv, self).__init__()
        
        self.robot_id = robot_id
        self.total_robots = total_robots
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # 初始化ROS2（如果尚未初始化）
        if not rclpy.ok():
            rclpy.init()
        
        # 创建底层多机器人环境
        self.multi_env = MultiRobotEnv(
            number_of_robots=total_robots,
            map_number=map_number,
            use_random_mode=use_random_mode
        )
        
        # 观测空间：单个机器人的观测维度
        # 98维: 38(lidar) + 2(velocity) + 3(goal) + 6(other_robots) + 49(distance_field)
        obs_dim = self.multi_env.single_robot_observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # 动作空间：连续动作 [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        # ========== 分层强化学习：全局规划器 ==========
        self.use_global_planner = True  # 启用全局规划
        self.planner = None  # 等地图加载后初始化
        self.waypoint_extractor = WaypointExtractor(
            turning_threshold=0.3,  # 转角>17度算拐点
            distance_threshold=1.5  # 直线段每1.5米一个点
        )
        
        # 路径点管理
        self.global_waypoints = None
        self.prev_dist_to_waypoint = None
        self.current_waypoint_index = 0
        self.waypoint_reach_distance = 0.3  # 到达阈值
        
        # Gazebo可视化
        self.waypoint_visualizer = WaypointVisualizer()
        
        print(f"\n{'='*80}")
        print(f"🤖 单机器人环境初始化 (Robot {robot_id}/{total_robots-1})")
        print(f"{'='*80}")
        print(f"观测空间: {self.observation_space.shape}")
        print(f"动作空间: {self.action_space.shape}")
        print(f"最大步数: {max_episode_steps}")
        print(f"🗺️ 分层RL已启用：A*全局 + RL局部")
        print(f"{'='*80}\n")
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.current_step = 0
        
        # 重置底层环境（所有机器人）
        multi_obs = self.multi_env.reset()
        for _ in range(10): 
            if hasattr(self.multi_env, 'node'):
                rclpy.spin_once(self.multi_env.node, timeout_sec=0.01)
            else:
                # 如果没有 node 属性，尝试打印一下属性列表找一找
                # print(dir(self.multi_env)) 
                pass
        # 为当前机器人规划全局路径
        if self.use_global_planner:
            self._plan_global_path()
        
        self.prev_dist_to_waypoint = None
        current_wp = self._get_current_waypoint()
        if current_wp:
            rx = self.multi_env.current_pose_x[self.robot_id]
            ry = self.multi_env.current_pose_y[self.robot_id]
            self.prev_dist_to_waypoint = math.hypot(current_wp[0] - rx, current_wp[1] - ry)

        # 提取当前机器人的观测
        robot_key = f'robot{self.robot_id}'
        obs = multi_obs[robot_key]
        
        info = {
            'episode': self.multi_env.episode_counter,
            'robot_id': self.robot_id
        }
        
        return obs, info
    
    def step(self, action):
        """执行一步"""
        self.current_step += 1
        
        # 构造所有机器人的动作（其他机器人用上一次的动作或0）
        multi_actions = {}
        for i in range(self.total_robots):
            robot_key = f'robot{i}'
            if i == self.robot_id:
                # 当前机器人使用给定动作
                multi_actions[robot_key] = action
            else:
                # 其他机器人使用默认动作（不动）
                # 实际上其他机器人也在训练，会有自己的策略
                multi_actions[robot_key] = np.array([0.0, 0.0], dtype=np.float32)
        
        # 执行环境步进
        multi_obs, multi_rewards, multi_dones, multi_truncated, info = self.multi_env.step(multi_actions)

        if self.use_global_planner and self.current_step % 20 == 0: # 每20步检查一次
             self._plan_global_path()


        # # 在 step 方法中添加
        # if self.use_global_planner and self.global_waypoints:
        #     current_wp = self._get_current_waypoint()
        #     if current_wp:
        #         robot_x = self.multi_env.current_pose_x[self.robot_id]
        #         robot_y = self.multi_env.current_pose_y[self.robot_id]
        #         dist = math.hypot(current_wp[0] - robot_x, current_wp[1] - robot_y)
        #         if dist > 2.0:
        #             # print(f"Robot {self.robot_id}: 严重偏离路径，触发重规划！")
        #             self._plan_global_path()
        
        # 提取当前机器人的结果
        robot_key = f'robot{self.robot_id}'
        obs = multi_obs[robot_key]
        reward = multi_rewards[robot_key]
        done = multi_dones[robot_key]
        truncated = multi_truncated[robot_key]
        
        # 检查并更新路径点，给予额外奖励
        if self.use_global_planner:
            waypoint_reached = self._check_and_update_waypoint()
            if waypoint_reached:
                reward += 0.5  # 到达路径点额外奖励
        
        # 超时截断
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        # 构造info
        step_info = {
            'robot_id': self.robot_id,
            'episode_step': self.current_step,
            'event': info.get('event', 'normal'),
        }
        
        # 添加奖励分量（用于调试）
        if 'reward_components' in info and self.robot_id < len(info['reward_components']):
            reward_comp = info['reward_components'][self.robot_id]
            step_info['reward_components'] = reward_comp
        
        return obs, reward, done, truncated, step_info
    
    def render(self, mode='human'):
        """渲染环境（Gazebo已经在渲染）"""
        pass
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'multi_env'):
            self.multi_env.cleanup()
    
    def _initialize_planner_after_map_loaded(self):
        """地图加载后初始化A*规划器"""
        if not self.use_global_planner:
            return
        
        # 从底层环境获取地图数据
        if hasattr(self.multi_env, 'map_subscriber') and self.multi_env.map_subscriber:
            map_sub = self.multi_env.map_subscriber
            if map_sub.map_data is not None:
                map_data = map_sub.map_data
                width = map_sub.map_width
                height = map_sub.map_height
                resolution = map_sub.map_resolution
                origin = (map_sub.map_origin_x, map_sub.map_origin_y)
                
                # 转换地图数据格式
                import numpy as np
                map_array = np.array(map_data).reshape(height, width)
                
                self.planner = AStarPlanner(map_array, resolution, origin)
                print(f"✅ Robot {self.robot_id}: A*规划器已就绪")
    
    def _plan_global_path(self):
        """为当前机器人规划全局路径"""
        if not self.planner:
            self._initialize_planner_after_map_loaded()
            if not self.planner:
                # 如果规划器还没准备好，使用目标点
                goal_pos = self.multi_env.current_goal_locations[self.robot_id]
                self.global_waypoints = [goal_pos]
                self.current_waypoint_index = 0
                return
        
        # 获取起点和终点
        start_pos = (
            self.multi_env.current_pose_x[self.robot_id],
            self.multi_env.current_pose_y[self.robot_id]
        )
        goal_pos = self.multi_env.current_goal_locations[self.robot_id]
        
        # 规划路径
        path = self.planner.plan(start_pos, goal_pos)
        
        if path is None or len(path) < 2:
            print(f"⚠️ Robot {self.robot_id}无法规划路径，直接使用目标点")
            self.global_waypoints = [goal_pos]
            self.current_waypoint_index = 0
            return
        
        # 提取关键路径点
        waypoints = self.waypoint_extractor.extract(path)
        self.global_waypoints = waypoints
        self.current_waypoint_index = 0
        
        # print(f"✅ Robot {self.robot_id}：{len(path)}点→{len(waypoints)}关键点")
        
        # 可视化
        if self.waypoint_visualizer:
            self.waypoint_visualizer.publish_waypoints(waypoints, robot_id=self.robot_id)
            # 多次spin确保消息发送
            for _ in range(5):
                rclpy.spin_once(self.waypoint_visualizer, timeout_sec=0.01)
            
            if len(waypoints) > 0:
                self.waypoint_visualizer.highlight_current_waypoint(
                    waypoints[0], robot_id=self.robot_id
                )
                # 再次spin确保高亮消息发送
                for _ in range(5):
                    rclpy.spin_once(self.waypoint_visualizer, timeout_sec=0.01)
    
    def _get_current_waypoint(self):
        """获取当前应该前往的路径点"""
        if not self.use_global_planner or not self.global_waypoints:
            return self.multi_env.current_goal_locations[self.robot_id]
        
        idx = self.current_waypoint_index
        if idx < len(self.global_waypoints):
            return self.global_waypoints[idx]
        else:
            return self.global_waypoints[-1] if self.global_waypoints else None
    
    def _check_and_update_waypoint(self):
        """检查是否到达路径点，到达则切换"""
        if not self.use_global_planner or not self.global_waypoints:
            return False
        
        current_wp = self._get_current_waypoint()
        if current_wp is None:
            return False
        
        robot_x = self.multi_env.current_pose_x[self.robot_id]
        robot_y = self.multi_env.current_pose_y[self.robot_id]
        dist = math.hypot(current_wp[0] - robot_x, current_wp[1] - robot_y)
        
        if dist < self.waypoint_reach_distance:
            old_idx = self.current_waypoint_index
            self.current_waypoint_index += 1
            
            total_wps = len(self.global_waypoints)
            new_idx = self.current_waypoint_index
            
            if new_idx < total_wps:
                print(f"🎯 Robot {self.robot_id}：路径点{old_idx}→{new_idx}/{total_wps}")
                if self.waypoint_visualizer:
                    next_wp = self.global_waypoints[new_idx]
                    self.waypoint_visualizer.highlight_current_waypoint(
                        next_wp, robot_id=self.robot_id
                    )
                    # spin确保高亮消息发送
                    for _ in range(3):
                        rclpy.spin_once(self.waypoint_visualizer, timeout_sec=0.01)
                return True
            else:
                print(f"🏁 Robot {self.robot_id}到达最终目标！")
                return True
        
        return False


class MultiAgentWrapper(gym.Env):
    """
    多智能体训练包装器
    管理多个独立的单机器人环境和策略
    兼容Gymnasium/Gym接口
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, total_robots=1, map_number=3, 
                 use_random_mode=False, max_episode_steps=300):
        """
        初始化多智能体包装器
        
        Args:
            total_robots: 机器人数量
            map_number: 地图编号
            use_random_mode: 是否随机起始位置
            max_episode_steps: 最大步数
        """
        super(MultiAgentWrapper, self).__init__()
        
        self.total_robots = total_robots
        self.max_episode_steps = max_episode_steps
        
        # 创建共享的底层环境
        if not rclpy.ok():
            rclpy.init()
        
        from start_reinforcement_learning.env_logic.logic import Env as MultiRobotEnv
        self.shared_env = MultiRobotEnv(
            number_of_robots=total_robots,
            map_number=map_number,
            use_random_mode=use_random_mode
        )
        
        # 观测和动作空间
        obs_dim = self.shared_env.single_robot_observation_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]), 
            dtype=np.float32
        )
        
        self.current_step = 0
        
        print(f"\n{'='*80}")
        print(f"🤖 多智能体训练环境初始化")
        print(f"{'='*80}")
        print(f"机器人数量: {total_robots}")
        print(f"观测空间: {self.observation_space.shape}")
        print(f"动作空间: {self.action_space.shape}")
        print(f"{'='*80}\n")
    
    def reset(self):
        """重置环境，返回所有机器人的观测"""
        self.current_step = 0
        multi_obs = self.shared_env.reset()
        
        # 返回列表形式的观测
        obs_list = [multi_obs[f'robot{i}'] for i in range(self.total_robots)]
        return obs_list
    
    def step(self, actions_list):
        """
        执行一步，所有机器人同时行动
        
        Args:
            actions_list: 所有机器人的动作列表
            
        Returns:
            obs_list, rewards_list, dones_list, truncated_list, infos_list
        """
        self.current_step += 1
        
        # 转换为字典格式
        multi_actions = {f'robot{i}': actions_list[i] 
                        for i in range(self.total_robots)}
        
        # 执行步进
        multi_obs, multi_rewards, multi_dones, multi_truncated, info = \
            self.shared_env.step(multi_actions)
        
        # 转换为列表格式
        obs_list = [multi_obs[f'robot{i}'] for i in range(self.total_robots)]
        rewards_list = [multi_rewards[f'robot{i}'] for i in range(self.total_robots)]
        dones_list = [multi_dones[f'robot{i}'] for i in range(self.total_robots)]
        truncated_list = [multi_truncated[f'robot{i}'] for i in range(self.total_robots)]
        
        # 超时截断
        if self.current_step >= self.max_episode_steps:
            truncated_list = [True] * self.total_robots
        
        # 构造info列表
        infos_list = []
        for i in range(self.total_robots):
            step_info = {
                'robot_id': i,
                'episode_step': self.current_step,
                'event': info.get('event', 'normal'),
            }
            if 'reward_components' in info and i < len(info['reward_components']):
                step_info['reward_components'] = info['reward_components'][i]
            infos_list.append(step_info)
        
        return obs_list, rewards_list, dones_list, truncated_list, infos_list
    
    def render(self, mode='human'):
        """渲染环境（Gazebo已经在渲染）"""
        pass
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'shared_env'):
            self.shared_env.cleanup()
