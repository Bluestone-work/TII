import os
import rclpy
from rclpy.node import Node
import yaml

import time
import math
import random
import numpy as np
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, SetEntityState
from geometry_msgs.msg import  Point, Pose, Quaternion
from .map_utils import MapCollisionChecker

class RestartEnvironment():
    def __init__(self, number_of_robots=3, map_number=1, use_random_mode=False):
        """
        初始化环境重置器
        Args:
            number_of_robots: 机器人数量
            map_number: 地图编号 (1: map1, 2: map2, 3: corridor_swap, 4: intersection, 5: warehouse_aisles)
            use_random_mode: True=随机模式，False=固定模式
        """
        self.number_of_robots = number_of_robots
        self.map_number = map_number
        self.use_random_mode = use_random_mode
        
        # 延迟初始化服务客户端（只在需要时创建）
        self._set_model_pose = None
        
        # 根据地图编号获取地图名称
        map_keys = ['map1', 'map2', 'corridor_swap', 'intersection', 'warehouse_aisles']
        if 1 <= map_number <= 5:
            self.current_map_name = map_keys[map_number - 1]
        else:
            print(f"[WARN] Invalid map_number {map_number}, defaulting to map1")
            self.current_map_name = 'map1'
        
        # 初始化地图碰撞检测器
        self.map_checker = MapCollisionChecker(self.current_map_name)
        
        # 从配置文件加载地图配置
        self._load_spawn_presets()
        
        # 根据模式选择机器人起始位置
        if use_random_mode:
            # 随机模式：从随机区域中生成位置（带碰撞检测）
            self.selected_robot_poses = self._generate_random_poses(
                self.current_map['start_regions'], 
                number_of_robots
            )
        else:
            # 固定模式：不预设位置，将在第一次 reset 时记录环境中的实际位置
            print("[INFO] Fixed mode: will use robot positions from environment")
            self.selected_robot_poses = []  # 空列表，将在第一次 reset 后填充
        
        # 设置安全位置（用于临时移动机器人）
        self.map_safe_spots = self.current_map.get('safe_spots', [])
        
        # 根据模式设置目标位置
        if use_random_mode:
            # 随机模式：在 goal_regions 中生成随机目标（带碰撞检测）
            print("[INFO] Random mode: generating random goal positions")
            self.selected_goal_poses = self._generate_random_poses(
                self.current_map['goal_regions'],
                number_of_robots
            )
            print(f"[DEBUG] Generated {len(self.selected_goal_poses)} random goal positions:")
            for i, pose in enumerate(self.selected_goal_poses):
                print(f"  Goal {i}: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
        else:
            # 固定模式：使用预设目标并进行碰撞检测
            raw_goal_poses = self.current_map['fixed_goals']
            self.selected_goal_poses = []
            
            # 对每个目标位置进行碰撞检测和调整
            for i, goal_pose in enumerate(raw_goal_poses):
                x, y = goal_pose.position.x, goal_pose.position.y
                # 检查位置是否在墙体内
                if not self.map_checker.is_position_free(x, y, radius=0.25):
                    print(f"[WARN] Goal {i} at ({x:.2f}, {y:.2f}) is in obstacle, searching for free position...")
                    x, y = self.map_checker.find_nearest_free_position(x, y, max_search_radius=3.0)
                    goal_pose = self._create_pose(x, y, goal_pose.orientation.z)
                
                self.selected_goal_poses.append(goal_pose)
        
        # 为每个机器人分配一个目标位置（修复：确保每个机器人有独立目标）
        self.current_goal_poses = []
        
        # 如果机器人数量 > 目标数量，需要扩展目标列表
        if number_of_robots > len(self.selected_goal_poses):
            # 复制目标列表直到足够
            extended_goals = list(self.selected_goal_poses)
            while len(extended_goals) < number_of_robots:
                # 在随机模式下生成新目标，在固定模式下重复现有目标
                if self.use_random_mode:
                    # 随机模式：生成新的随机目标
                    new_goals = self._generate_random_poses(
                        self.current_map['goal_regions'],
                        number_of_robots - len(extended_goals)
                    )
                    extended_goals.extend(new_goals)
                else:
                    # 固定模式：循环重复现有目标
                    extended_goals.extend(self.selected_goal_poses)
            
            # 取前 number_of_robots 个目标
            for i in range(number_of_robots):
                self.current_goal_poses.append(extended_goals[i])
        else:
            # 机器人数量 <= 目标数量，直接分配
            for i in range(number_of_robots):
                self.current_goal_poses.append(self.selected_goal_poses[i])
        
        print(f"[DEBUG] Final current_goal_poses ({len(self.current_goal_poses)} goals):")
        for i, pose in enumerate(self.current_goal_poses):
            print(f"  Robot {i} goal: x={pose.position.x:.2f}, y={pose.position.y:.2f}")
    
    
    def _load_spawn_presets(self):
        """从 YAML 文件加载地图配置"""
        try:
            pkg_share = get_package_share_directory('start_reinforcement_learning')
            yaml_path = os.path.join(pkg_share, 'config', 'spawn_presets.yaml')
            
            if not os.path.exists(yaml_path):
                print(f"[ERROR] Spawn presets file not found: {yaml_path}")
                print("[INFO] Using fallback hardcoded configs")
                self.current_map = self._get_fallback_config()
                return
            
            with open(yaml_path, 'r') as f:
                all_presets = yaml.safe_load(f)
            
            if self.current_map_name not in all_presets:
                print(f"[WARN] Map '{self.current_map_name}' not in presets, using fallback")
                self.current_map = self._get_fallback_config()
                return
            
            map_data = all_presets[self.current_map_name]
            
            # 转换 YAML 数据为 Pose 对象
            self.current_map = {
                'fixed_starts': [self._create_pose(p['x'], p['y'], p['yaw']) 
                                for p in map_data.get('fixed_starts', [])],
                'fixed_goals': [self._create_pose(p['x'], p['y'], p['yaw']) 
                               for p in map_data.get('fixed_goals', [])],
                'start_regions': map_data.get('start_regions', []),
                'goal_regions': map_data.get('goal_regions', []),
                'safe_spots': [self._create_pose(p['x'], p['y'], p['yaw']) 
                              for p in map_data.get('safe_spots', [])] 
                              if 'safe_spots' in map_data else []
            }
            
            print(f"[INFO] Loaded spawn presets for {self.current_map_name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load spawn presets: {e}")
            self.current_map = self._get_fallback_config()
    
    def _get_fallback_config(self):
        """获取回退配置（简单的默认值）"""
        return {
            'fixed_starts': [
                self._create_pose(-5.0, 0.0, 0.0),
                self._create_pose(-5.0, 1.0, 0.0),
                self._create_pose(-5.0, -1.0, 0.0),
            ],
            'fixed_goals': [
                self._create_pose(5.0, 0.0, math.pi),
                self._create_pose(5.0, 1.0, math.pi),
                self._create_pose(5.0, -1.0, math.pi),
            ],
            'start_regions': [[-8.0, -8.0, -2.0, 8.0]],
            'goal_regions': [[2.0, -8.0, 8.0, 8.0]],
            'safe_spots': []
        }
    
    def _init_map_configs(self):
        """初始化所有地图的配置"""
        configs = {}
        
        # ==================== Map 1: 原仓库走廊/房间组合 ====================
        configs['map1'] = {
            'fixed_starts': [
                self._create_pose(-8.0, -4.0, 0.0),
                self._create_pose(-8.0, 0.0, 0.0),
                self._create_pose(-8.0, 4.0, 0.0),
                self._create_pose(8.0, -4.0, math.pi),
                self._create_pose(8.0, 0.0, math.pi),
                self._create_pose(8.0, 4.0, math.pi),
            ],
            'fixed_goals': [
                self._create_pose(8.0, -4.0, 0.0),
                self._create_pose(8.0, 0.0, 0.0),
                self._create_pose(8.0, 4.0, 0.0),
                self._create_pose(-8.0, -4.0, math.pi),
                self._create_pose(-8.0, 0.0, math.pi),
                self._create_pose(-8.0, 4.0, math.pi),
            ],
            'start_regions': [
                [-9.0, -8.5, -6.0, 8.5],  # 左侧区域
                [6.0, -8.5, 9.0, 8.5],    # 右侧区域
            ],
            'goal_regions': [
                [6.0, -8.5, 9.0, 8.5],    # 右侧区域
                [-9.0, -8.5, -6.0, 8.5],  # 左侧区域
            ],
            'safe_spots': [
                self._create_pose(-9.5, -9.0, 0.0),
                self._create_pose(-9.5, -8.0, 0.0),
                self._create_pose(-9.5, -7.0, 0.0),
                self._create_pose(-9.5, -6.0, 0.0),
                self._create_pose(-9.5, -5.0, 0.0),
                self._create_pose(-9.5, -4.0, 0.0),
                self._create_pose(-9.5, -3.0, 0.0),
            ]
        }
        
        # ==================== Map 2: L型/走廊 ====================
        configs['map2'] = {
            'fixed_starts': [
                self._create_pose(-8.0, -3.5, 0.0),
                self._create_pose(-8.0, 0.0, 0.0),
                self._create_pose(-8.0, 3.5, 0.0),
                self._create_pose(3.5, -8.0, math.pi/2),
                self._create_pose(6.5, -8.0, math.pi/2),
            ],
            'fixed_goals': [
                self._create_pose(6.5, -8.0, -math.pi/2),
                self._create_pose(3.5, -8.0, -math.pi/2),
                self._create_pose(8.0, 3.5, math.pi),
                self._create_pose(-8.0, 0.0, 0.0),
                self._create_pose(-8.0, -3.5, 0.0),
            ],
            'start_regions': [
                [-9.0, -8.5, -6.0, 8.5],  # 左侧区域
                [2.0, -9.0, 9.0, -6.0],   # 下方区域
            ],
            'goal_regions': [
                [2.0, -9.0, 9.0, -6.0],   # 下方区域
                [-9.0, -8.5, -6.0, 8.5],  # 左侧区域
            ],
            'safe_spots': [
                self._create_pose(-9.5, 8.7, 0.0),
                self._create_pose(-9.5, 7.7, 0.0),
                self._create_pose(-9.5, 6.7, 0.0),
                self._create_pose(-9.5, 5.7, 0.0),
                self._create_pose(-9.5, 4.7, 0.0),
                self._create_pose(-9.5, 3.7, 0.0),
                self._create_pose(-9.5, 2.7, 0.0),
            ]
        }
        
        # ==================== Map 3: Corridor Swap (窄通道对撞测试) ====================
        configs['corridor_swap'] = {
            'fixed_starts': [
                self._create_pose(-7.5, -1.5, 0.0),
                self._create_pose(-7.5, 0.0, 0.0),
                self._create_pose(-7.5, 1.5, 0.0),
                self._create_pose(7.5, -1.5, math.pi),
                self._create_pose(7.5, 0.0, math.pi),
                self._create_pose(7.5, 1.5, math.pi),
            ],
            'fixed_goals': [
                self._create_pose(7.5, -1.5, 0.0),
                self._create_pose(7.5, 0.0, 0.0),
                self._create_pose(7.5, 1.5, 0.0),
                self._create_pose(-7.5, -1.5, math.pi),
                self._create_pose(-7.5, 0.0, math.pi),
                self._create_pose(-7.5, 1.5, math.pi),
            ],
            'start_regions': [
                [-9.0, -8.0, -5.5, 8.0],  # 左侧区域
                [5.5, -8.0, 9.0, 8.0],    # 右侧区域
            ],
            'goal_regions': [
                [5.5, -8.0, 9.0, 8.0],    # 右侧区域
                [-9.0, -8.0, -5.5, 8.0],  # 左侧区域
            ],
            'safe_spots': [
                self._create_pose(-9.5, -9.0, 0.0),
                self._create_pose(-9.5, -8.0, 0.0),
                self._create_pose(-9.5, -7.0, 0.0),
                self._create_pose(-9.5, -6.0, 0.0),
                self._create_pose(-9.5, -5.0, 0.0),
                self._create_pose(-9.5, -4.0, 0.0),
                self._create_pose(-9.5, -3.0, 0.0),
            ]
        }
        
        # ==================== Map 4: Intersection (十字路口) ====================
        configs['intersection'] = {
            'fixed_starts': [
                self._create_pose(-8.0, 0.0, 0.0),
                self._create_pose(8.0, 0.0, math.pi),
                self._create_pose(0.0, -8.0, math.pi/2),
                self._create_pose(0.0, 8.0, -math.pi/2),
                self._create_pose(-8.0, -1.5, 0.0),
                self._create_pose(8.0, 1.5, math.pi),
            ],
            'fixed_goals': [
                self._create_pose(8.0, 0.0, math.pi),
                self._create_pose(-8.0, 0.0, 0.0),
                self._create_pose(0.0, 8.0, -math.pi/2),
                self._create_pose(0.0, -8.0, math.pi/2),
                self._create_pose(8.0, -1.5, math.pi),
                self._create_pose(-8.0, 1.5, 0.0),
            ],
            'start_regions': [
                [-9.5, -2.0, -6.0, 2.0],  # 左侧
                [6.0, -2.0, 9.5, 2.0],    # 右侧
                [-2.0, -9.5, 2.0, -6.0],  # 下方
                [-2.0, 6.0, 2.0, 9.5],    # 上方
            ],
            'goal_regions': [
                [6.0, -2.0, 9.5, 2.0],    # 右侧
                [-9.5, -2.0, -6.0, 2.0],  # 左侧
                [-2.0, 6.0, 2.0, 9.5],    # 上方
                [-2.0, -9.5, 2.0, -6.0],  # 下方
            ],
            'safe_spots': [
                self._create_pose(-10.0, -10.0, 0.0),
                self._create_pose(-10.0, -9.0, 0.0),
                self._create_pose(-10.0, -8.0, 0.0),
                self._create_pose(-10.0, -7.0, 0.0),
                self._create_pose(-10.0, -6.0, 0.0),
                self._create_pose(-10.0, -5.0, 0.0),
                self._create_pose(-10.0, -4.0, 0.0),
            ]
        }
        
        # ==================== Map 5: Warehouse Aisles (货架通道) ====================
        configs['warehouse_aisles'] = {
            'fixed_starts': [
                self._create_pose(-8.5, -6.0, 0.0),
                self._create_pose(-8.5, -3.0, 0.0),
                self._create_pose(-8.5, 0.0, 0.0),
                self._create_pose(-8.5, 3.0, 0.0),
                self._create_pose(-8.5, 6.0, 0.0),
                self._create_pose(8.5, -4.5, math.pi),
                self._create_pose(8.5, 4.5, math.pi),
            ],
            'fixed_goals': [
                self._create_pose(8.5, -6.0, math.pi),
                self._create_pose(8.5, -3.0, math.pi),
                self._create_pose(8.5, 0.0, math.pi),
                self._create_pose(8.5, 3.0, math.pi),
                self._create_pose(8.5, 6.0, math.pi),
                self._create_pose(-8.5, -4.5, 0.0),
                self._create_pose(-8.5, 4.5, 0.0),
            ],
            'start_regions': [
                [-9.5, -8.5, -6.5, 8.5],  # 左侧区域
                [6.5, -8.5, 9.5, 8.5],    # 右侧区域
            ],
            'goal_regions': [
                [6.5, -8.5, 9.5, 8.5],    # 右侧区域
                [-9.5, -8.5, -6.5, 8.5],  # 左侧区域
            ],
            'safe_spots': [
                self._create_pose(-10.0, -9.0, 0.0),
                self._create_pose(-10.0, -8.0, 0.0),
                self._create_pose(-10.0, -7.0, 0.0),
                self._create_pose(-10.0, -6.0, 0.0),
                self._create_pose(-10.0, -5.0, 0.0),
                self._create_pose(-10.0, -4.0, 0.0),
                self._create_pose(-10.0, -3.0, 0.0),
            ]
        }
        
        return configs
    
    def _create_pose(self, x, y, yaw, z=0.1):
        """创建Pose对象，确保所有参数为 float"""
        return Pose(
            position=Point(x=float(x), y=float(y), z=float(z)),
            orientation=Quaternion(
                z=float(math.sin(yaw / 2)),
                w=float(math.cos(yaw / 2))
            )
        )
    
    def _generate_random_poses(self, regions, count):
        """在指定区域内生成随机位置（带地图碰撞检测和增强的回退机制）"""
        poses = []
        min_dist_sq = 0.8 ** 2  # 增加机器人间最小距离到0.8米，更安全
        max_attempts_per_robot = 1000
        
        for i in range(count):
            valid_pose = False
            attempts = 0
            x, y = 0.0, 0.0
            best_x, best_y = None, None
            best_clearance = 0.0
            
            while not valid_pose and attempts < max_attempts_per_robot:
                attempts += 1
                # 随机选择一个区域 [xmin, ymin, xmax, ymax]
                box = random.choice(regions)
                
                # 确保不会太靠近边界（留出0.5米的边距）
                margin = 0.5
                x_min = box[0] + margin
                x_max = box[2] - margin
                y_min = box[1] + margin
                y_max = box[3] - margin
                
                # 如果区域太小，使用原始边界
                if x_max <= x_min:
                    x_min, x_max = box[0], box[2]
                if y_max <= y_min:
                    y_min, y_max = box[1], box[3]
                
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                
                # 检查 1: 地图碰撞（是否在障碍物中或地图外，增大安全半径）
                clearance = self.map_checker.get_clearance(x, y, radius=0.35)
                if clearance < 0.35:  # 至少需要35cm的空闲空间
                    # 记录最好的位置（即使不够好）
                    if clearance > best_clearance:
                        best_clearance = clearance
                        best_x, best_y = x, y
                    continue
                
                # 检查 2: 与其他机器人的距离
                too_close = False
                for pose in poses:
                    ex, ey = pose.position.x, pose.position.y
                    dist_sq = (x - ex)**2 + (y - ey)**2
                    if dist_sq < min_dist_sq:
                        too_close = True
                        break

                if too_close:
                    continue

                # 检查 3: 动态障碍物初始 spawn 位置（距任意 spawn 点 < 1.0m 则跳过）
                # 与 obstacle_mover.py MAP_CONFIGS spawn_points 保持同步（×0.6 缩放后）
                _DYN_OBS_SPAWNS = {
                    'map1':            [(0.4,-0.7),(0.4,-1.4),(0.4,-2.2),(0.4,-2.9),
                                        (0.4,-3.6),(0.4,-4.3),(0.4,-5.0),(0.4,-5.8)],
                    'map2':            [(0.3,-0.9),(0.3,-2.4),(0.3,-3.9),(0.3,-5.7),
                                        (3.3,-0.9),(5.4,-1.2),(3.3,-4.2),(5.4,-5.7)],
                    'corridor_swap':   [(-4.5,-4.0),(-4.5, 4.0),(-2.0,-4.5),(-2.0, 4.5),
                                        ( 2.0,-4.5),( 2.0, 4.5),( 4.5,-4.0),( 4.5, 4.0)],
                    'intersection':    [(-2.7, 0.5),(-2.7,-0.5),( 2.7, 0.5),( 2.7,-0.5),
                                        ( 0.5,-2.7),(-0.5,-2.7),( 0.5, 2.7),(-0.5, 2.7)],
                    'warehouse_aisles':[(-2.7,-3.6),(-2.7, 3.6),(-0.3,-3.3),(-0.3, 3.3),
                                        ( 2.1,-3.6),( 2.1, 3.0),( 4.5,-4.2),( 4.5, 4.2)],
                }
                obs_spawns = _DYN_OBS_SPAWNS.get(self.current_map_name, [])
                near_obs = any(
                    math.sqrt((x - ox)**2 + (y - oy)**2) < 1.0
                    for ox, oy in obs_spawns
                )
                if near_obs:
                    continue

                valid_pose = True
            
            # 如果没找到完美位置，使用回退策略
            if not valid_pose:
                print(f"[WARN] Could not find ideal pose for robot/goal {i} after {attempts} attempts")
                
                # 策略1: 使用找到的最好位置
                if best_x is not None and best_clearance > 0.2:
                    print(f"[INFO] Using best found position with clearance {best_clearance:.2f}m")
                    x, y = best_x, best_y
                else:
                    # 策略2: 使用区域中心并搜索最近的空闲位置
                    box = regions[0] if regions else [-5, -5, 5, 5]
                    center_x = (box[0] + box[2]) / 2
                    center_y = (box[1] + box[3]) / 2
                    print(f"[INFO] Searching near region center ({center_x:.2f}, {center_y:.2f})")
                    x, y = self.map_checker.find_nearest_free_position(
                        center_x, center_y, 
                        max_search_radius=8.0,  # 扩大搜索半径
                        min_clearance=0.35
                    )
                    print(f"[INFO] Found fallback position at ({x:.2f}, {y:.2f})")
            
            # 随机朝向
            yaw = random.uniform(-math.pi, math.pi)
            pose = self._create_pose(x, y, yaw)
            poses.append(pose)
            print(f"[DEBUG] Generated pose {i}: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}")
        
        return poses
    
    # ==================== 公共方法 ====================
    
    def _get_set_model_pose(self):
        """获取SetModelPose服务客户端（延迟初始化）"""
        if self._set_model_pose is None:
            self._set_model_pose = SetModelPose()
        return self._set_model_pose
    
    def move_goals(self):
        """随机移动所有目标到新位置"""
        set_goal = self._get_set_model_pose()
        goal_locations = []
        
        # 如果是随机模式，重新生成随机目标位置
        if self.use_random_mode:
            print("[INFO] Regenerating random goal positions...")
            self.selected_goal_poses = self._generate_random_poses(
                self.current_map['goal_regions'],
                self.number_of_robots
            )
            self.current_goal_poses = self.selected_goal_poses[:]
        else:
            # 固定模式：从预设目标中选择不同位置
            for i in range(self.number_of_robots):
                # 随机选择一个不同的目标位置
                num_goal_poses = len(self.selected_goal_poses)
                goal_index = i % num_goal_poses  # 确保每个机器人有不同的目标
                
                # 随机选择但避免重复（如果目标位置足够多）
                if num_goal_poses >= self.number_of_robots:
                    available_indices = list(range(num_goal_poses))
                    # 移除已分配给其他机器人的索引
                    for j in range(i):
                        prev_index = j % num_goal_poses
                        if prev_index in available_indices:
                            available_indices.remove(prev_index)
                    if available_indices:
                        goal_index = random.choice(available_indices)
                
                self.current_goal_poses[i] = self.selected_goal_poses[goal_index]
        
        # 移动所有目标
        for i in range(self.number_of_robots):
            name = f'goal_box_{i}'
            set_goal.get_logger().info(f'%%%%%%%%% Moving Goal {i} %%%%%%%%%')
            set_goal.send_request(name, self.current_goal_poses[i])
            goal_locations.append((self.current_goal_poses[i].position.x, 
                                  self.current_goal_poses[i].position.y))
        
        return goal_locations
    
    def spawn_goals(self):
        """生成多个目标实体（每个机器人一个），使用不同颜色"""
        goal_locations = []
        
        print(f"[DEBUG] spawn_goals() called, spawning {self.number_of_robots} goals")
        print(f"[DEBUG] current_goal_poses has {len(self.current_goal_poses)} poses")
        
        # 定义不同颜色的目标
        colors = [
            ('Green', '0 1 0'),    # 绿色
            ('Blue', '0 0 1'),     # 蓝色
            ('Yellow', '1 1 0'),   # 黄色
            ('Red', '1 0 0'),      # 红色
            ('Cyan', '0 1 1'),     # 青色
            ('Magenta', '1 0 1'),  # 品红
        ]
        
        for i in range(self.number_of_robots):
            spawn_goal = Spawn_Entity()
            spawn_request = SpawnEntity.Request()
            
            goal_sdf_path = os.path.join(
                get_package_share_directory('start_rl_environment'),
                'models', 'goal_box', 'model.sdf'
            )
            
            # 读取并修改SDF以设置颜色
            sdf_content = open(goal_sdf_path, 'r').read()
            color_name, rgb = colors[i % len(colors)]
            
            # 替换颜色（使用Gazebo内置颜色或RGB）
            sdf_content = sdf_content.replace(
                '<name>Gazebo/Green</name>',
                f'<name>Gazebo/{color_name}</name>'
            )
            
            spawn_request.name = f'goal_box_{i}'
            spawn_request.xml = sdf_content
            spawn_request.robot_namespace = f'goal_box_{i}'
            spawn_request.initial_pose = self.current_goal_poses[i]
            
            print(f"[DEBUG] Spawning goal_box_{i} at x={self.current_goal_poses[i].position.x:.2f}, y={self.current_goal_poses[i].position.y:.2f}")
            spawn_goal.send_request(spawn_request)
            goal_locations.append((self.current_goal_poses[i].position.x, 
                                  self.current_goal_poses[i].position.y))
            
            # 添加小延迟，减少Gazebo负载
            if i < self.number_of_robots - 1:  # 最后一个不需要延迟
                time.sleep(0.05)  # 50ms延迟
        
        print(f"[DEBUG] spawn_goals() completed, returning {len(goal_locations)} locations")
        return goal_locations
    
    def delete_goal(self):
        """删除目标实体（当前未使用）"""
        delete_goal = Delete_Entity()
        delete_request = DeleteEntity.Request()
        delete_request.name = 'goal_box'
        delete_goal.send_request(delete_request)
    
    def record_initial_positions(self, robot_positions):
        """
        记录机器人在环境中的初始位置（仅用于固定模式）
        Args:
            robot_positions: List of (x, y, yaw) tuples
        """
        if not self.use_random_mode and len(self.selected_robot_poses) == 0:
            print("[INFO] Recording initial robot positions from environment")
            self.selected_robot_poses = []
            for i, (x, y, yaw) in enumerate(robot_positions):
                # 检查位置是否在障碍物中
                if not self.map_checker.is_position_free(x, y, radius=0.3):
                    print(f"[WARN] Robot {i} at ({x:.2f}, {y:.2f}) is in obstacle, adjusting...")
                    x, y = self.map_checker.find_nearest_free_position(x, y, max_search_radius=3.0)
                
                pose = self._create_pose(x, y, yaw)
                self.selected_robot_poses.append(pose)
                print(f"  Robot {i}: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})")
    
    def reset_robots(self):
        """重置所有机器人到初始位置"""
        # 如果是随机模式，每次 reset 都重新生成位置
        if self.use_random_mode:
            print("[INFO] Regenerating random robot positions...")
            self.selected_robot_poses = self._generate_random_poses(
                self.current_map['start_regions'], 
                self.number_of_robots
            )
        
        # 检查是否有有效的目标位置
        if len(self.selected_robot_poses) == 0:
            print("[ERROR] No robot poses defined! Cannot reset.")
            return
        
        set_bots = self._get_set_model_pose()
        
        # 直接移动到目标位置（不需要先移动到安全位置）
        # 之前的安全位置逻辑会导致机器人停留在错误的位置
        for i in range(self.number_of_robots):
            name = 'my_bot' + str(i)
            if i < len(self.selected_robot_poses):
                target_pose = self.selected_robot_poses[i]
                print(f"[INFO] Resetting {name} to ({target_pose.position.x:.2f}, {target_pose.position.y:.2f})")
                set_bots.send_request(name, target_pose)
            else:
                print(f"[WARN] No pose defined for robot {i}")


# ==================== ROS2 服务节点类 ====================

class SetModelPose(Node):
    def __init__(self):
        super().__init__('set_model_pose')
        self.cli = self.create_client(
            srv_type=SetEntityState, 
            srv_name="/set_entity_state"
        )
        
        # 等待服务，但有最大重试次数
        max_retries = 30
        retry_count = 0
        while not self.cli.wait_for_service(timeout_sec=1.0):
            retry_count += 1
            if retry_count >= max_retries:
                self.get_logger().error(
                    f'服务 /set_entity_state 在 {max_retries} 秒后仍未可用！'
                    '\n请检查：'
                    '\n1. Gazebo是否正确启动'
                    '\n2. 是否使用了正确的启动命令启动环境'
                    '\n3. map_number参数是否匹配'
                )
                raise RuntimeError('无法连接到 /set_entity_state 服务')
            self.get_logger().info(f'等待 /set_entity_state 服务... ({retry_count}/{max_retries})')
    
    def send_request(self, model_name, pose):
        req = SetEntityState.Request()
        req.state._name = model_name
        req.state._pose = pose
        self.future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)


class Spawn_Entity(Node):
    def __init__(self):
        super().__init__('Spawn_Entity', namespace='ss')
        self.cli = self.create_client(
            srv_type=SpawnEntity, 
            srv_name="/spawn_entity"
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
    
    def send_request(self, req):
        self.future = self.cli.call_async(req)
        
        # 等待响应完成，避免超时警告
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        
        if self.future.done():
            try:
                response = self.future.result()
                if response.success:
                    self.get_logger().info(f'%%%%%%%%%% Goal Spawned Successfully %%%%%%%%%%')
                else:
                    self.get_logger().warn(f'Goal spawn failed: {response.status_message}')
            except Exception as e:
                self.get_logger().error(f'Service call failed: {e}')
        else:
            self.get_logger().warn('Goal spawn request timed out')
        
        self.destroy_node()


class Delete_Entity(Node):
    def __init__(self):
        super().__init__('Delete_Entity')
        self.cli = self.create_client(
            srv_type=DeleteEntity, 
            srv_name="/delete_entity"
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
    
    def send_request(self, req):
        self.future = self.cli.call_async(req)
        
        # 等待响应完成，避免超时警告
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        
        if self.future.done():
            try:
                response = self.future.result()
                if response.success:
                    self.get_logger().info(f'Entity deleted successfully')
                else:
                    self.get_logger().warn(f'Entity deletion failed: {response.status_message}')
            except Exception as e:
                self.get_logger().error(f'Delete service call failed: {e}')
        else:
            self.get_logger().warn('Delete request timed out')
        
        self.destroy_node()
