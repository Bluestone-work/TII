"""
Traditional Global Path Planners (A*, Theta*, etc.)
用于ORCA+DWA模式的全局路径规划
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from collections import defaultdict
import heapq


class GridMap:
    """网格地图表示"""
    
    def __init__(self, width: float, height: float, resolution: float):
        """
        Args:
            width: 地图宽度（米）
            height: 地图高度（米）
            resolution: 网格分辨率（米/格）
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)
        
        # 0=自由，1=占用
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        
        self.origin_x = -width / 2  # 地图原点（世界坐标）
        self.origin_y = -height / 2
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """网格坐标转世界坐标"""
        x = self.origin_x + (grid_x + 0.5) * self.resolution
        y = self.origin_y + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid(self, grid_x: int, grid_y: int) -> bool:
        """检查网格坐标是否有效且自由"""
        if grid_x < 0 or grid_x >= self.grid_width:
            return False
        if grid_y < 0 or grid_y >= self.grid_height:
            return False
        return self.grid[grid_y, grid_x] == 0
    
    def set_obstacle(self, x: float, y: float, radius: float = 0.5):
        """在世界坐标处设置障碍物"""
        grid_x, grid_y = self.world_to_grid(x, y)
        grid_radius = int(radius / self.resolution)
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    if dx*dx + dy*dy <= grid_radius*grid_radius:
                        self.grid[gy, gx] = 1
    
    def has_line_of_sight(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Bresenham直线检查视线是否清晰（用于Theta*）"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        x, y = x1, y1
        
        while True:
            if not self.is_valid(x, y):
                return False
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True


class ThetaStarPlanner:
    """Theta* 全局路径规划器
    
    Theta*是A*的改进版本，支持any-angle路径，生成更平滑的路径
    """
    
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        
        # 8方向邻居
        self.neighbors_8 = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 4方向
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
    
    def heuristic(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """启发式函数：欧几里得距离"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """
        规划从start到goal的路径
        
        Args:
            start: 起点世界坐标 (x, y)
            goal: 终点世界坐标 (x, y)
            
        Returns:
            路径点列表 [(x1,y1), (x2,y2), ...] 或 None
        """
        # 转换为网格坐标
        start_grid = self.grid_map.world_to_grid(*start)
        goal_grid = self.grid_map.world_to_grid(*goal)
        
        # 检查起点和终点是否有效
        if not self.grid_map.is_valid(*start_grid):
            return None
        if not self.grid_map.is_valid(*goal_grid):
            return None
        
        # Theta* 算法
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_grid] = 0
        
        f_score = defaultdict(lambda: float('inf'))
        f_score[start_grid] = self.heuristic(*start_grid, *goal_grid)
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_grid:
                # 找到路径，重建
                return self._reconstruct_path(came_from, current)
            
            for dx, dy in self.neighbors_8:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self.grid_map.is_valid(*neighbor):
                    continue
                
                # Theta*的关键：尝试从parent直接连接到neighbor
                if current in came_from:
                    parent = came_from[current]
                    if self.grid_map.has_line_of_sight(*parent, *neighbor):
                        # Path 2: 从parent直接到neighbor
                        tentative_g = g_score[parent] + self.heuristic(*parent, *neighbor)
                        
                        if tentative_g < g_score[neighbor]:
                            came_from[neighbor] = parent
                            g_score[neighbor] = tentative_g
                            f_score[neighbor] = tentative_g + self.heuristic(*neighbor, *goal_grid)
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            continue
                
                # Path 1: 从current到neighbor（标准A*）
                cost = math.sqrt(dx*dx + dy*dy)  # 对角线代价更高
                tentative_g = g_score[current] + cost
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(*neighbor, *goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # 未找到路径
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[float, float]]:
        """重建路径"""
        path = []
        
        while current in came_from:
            x, y = self.grid_map.grid_to_world(*current)
            path.append((x, y))
            current = came_from[current]
        
        # 添加起点
        x, y = self.grid_map.grid_to_world(*current)
        path.append((x, y))
        
        path.reverse()
        return path


class SimpleGlobalPlanner:
    """简单的全局规划器（基于Theta*）
    
    用于没有Nav2时的全局路径规划
    """
    
    def __init__(self, map_width: float = 20.0, map_height: float = 20.0, resolution: float = 0.1):
        """
        Args:
            map_width: 地图宽度（米）
            map_height: 地图高度（米）
            resolution: 网格分辨率（米）
        """
        self.grid_map = GridMap(map_width, map_height, resolution)
        self.planner = ThetaStarPlanner(self.grid_map)
        
        # 缓存已规划的路径
        self.cached_paths = {}  # {robot_name: path}
    
    def set_map_obstacles(self, obstacles: List[Tuple[float, float]], radius: float = 0.5):
        """设置地图障碍物
        
        Args:
            obstacles: 障碍物位置列表 [(x1,y1), (x2,y2), ...]
            radius: 障碍物半径
        """
        for x, y in obstacles:
            self.grid_map.set_obstacle(x, y, radius)
    
    def plan_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        """规划路径
        
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            
        Returns:
            路径点列表或None
        """
        path = self.planner.plan(start, goal)
        return path
    
    def get_next_waypoint(self, robot_pos: Tuple[float, float], 
                         path: List[Tuple[float, float]], 
                         lookahead_distance: float = 1.0) -> Optional[Tuple[float, float]]:
        """从路径中获取下一个waypoint
        
        Args:
            robot_pos: 机器人当前位置
            path: 路径点列表
            lookahead_distance: 前瞻距离
            
        Returns:
            waypoint (x, y) 或 None
        """
        if not path:
            return None
        
        # 找到第一个距离超过lookahead_distance的点
        for waypoint in path:
            dx = waypoint[0] - robot_pos[0]
            dy = waypoint[1] - robot_pos[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance >= lookahead_distance:
                return waypoint
        
        # 如果所有点都在前瞻距离内，返回最后一个点
        return path[-1]


def create_simple_planner(map_width: float = 20.0, 
                         map_height: float = 20.0, 
                         resolution: float = 0.1) -> SimpleGlobalPlanner:
    """创建简单全局规划器的工厂函数"""
    return SimpleGlobalPlanner(map_width, map_height, resolution)
