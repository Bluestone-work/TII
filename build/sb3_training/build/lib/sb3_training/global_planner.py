"""
全局路径规划器 - 改进版
包含视线剪枝（Floyd算法简化）和更好的路径点提取
"""
import numpy as np
import heapq
import math
from typing import List, Tuple

class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, map_data, resolution=0.05, origin=(-10.0, -10.0)):
        self.map_data = map_data
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = map_data.shape
        # 膨胀障碍物
        self.inflated_map = self._inflate_obstacles(map_data, inflation_radius=4) # 稍微加大一点膨胀，保证安全
        
    def _inflate_obstacles(self, map_data, inflation_radius=3):
        from scipy.ndimage import binary_dilation
        obstacle_mask = (map_data > 50).astype(np.uint8)
        inflated = binary_dilation(obstacle_mask, iterations=inflation_radius)
        return inflated.astype(np.uint8) * 100
    
    def world_to_grid(self, x, y):
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y
    
    def is_valid(self, grid_x, grid_y):
        if grid_x < 0 or grid_x >= self.width or grid_y < 0 or grid_y >= self.height:
            return False
        if self.inflated_map[grid_y, grid_x] > 50:
            return False
        return True
    
    def heuristic(self, a, b):
        # 使用对角距离启发式，比欧氏距离在栅格中更好
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                cost = 1.414 if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
        return neighbors

    def check_line_of_sight(self, start_grid, end_grid):
        """
        Bresenham算法检查两点间是否有障碍物
        用于路径拉直（Pruning）
        """
        x0, y0 = start_grid
        x1, y1 = end_grid
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                if not self.is_valid(x, y): return False
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                if not self.is_valid(x, y): return False
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
                
        if not self.is_valid(x, y): return False
        return True

    def plan(self, start_pos, goal_pos):
        start = self.world_to_grid(*start_pos)
        goal = self.world_to_grid(*goal_pos)
        
        # 简单检查
        if not self.is_valid(*goal):
            # 如果终点无效，尝试找最近的有效点
            # print("终点无效，尝试寻找附近点...")
            return None # 简化处理，直接返回None由上层处理

        if not self.is_valid(*start):
             return None

        # A* 搜索
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                raw_path = self._reconstruct_path(came_from, current)
                # 关键步骤：路径平滑/剪枝
                pruned_path = self._prune_path(raw_path)
                return [self.grid_to_world(x, y) for x, y in pruned_path]
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _prune_path(self, path):
        """
        使用视线检查（Line of Sight）去除多余的路径点
        原理：如果 path[i] 能直接看到 path[i+2]，则 path[i+1] 是多余的
        """
        if len(path) < 3: return path
        
        pruned = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # 贪婪搜索：从当前点开始，尽可能往后找能直接连线的点
            # 限制搜索窗口，提高效率
            next_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                if self.check_line_of_sight(path[current_idx], path[i]):
                    next_idx = i
                    break
            
            pruned.append(path[next_idx])
            current_idx = next_idx
            
        return pruned

class WaypointExtractor:
    """
    现在Extractor主要负责距离过滤，因为主要的形状简化已经在Planner的prune_path中完成了
    """
    def __init__(self, distance_threshold=0.8):
        self.distance_threshold = distance_threshold # 增大点间距，减少点数
    
    def extract(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not path or len(path) < 2:
            return path
        
        # 1. 保留起点
        waypoints = [path[0]]
        
        # 2. 距离过滤：如果新点距离上一个点太近，且不是终点，则跳过
        for i in range(1, len(path) - 1):
            curr = path[i]
            prev = waypoints[-1]
            dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
            
            if dist >= self.distance_threshold:
                waypoints.append(curr)
        
        # 3. 智能处理终点
        goal = path[-1]
        last_added = waypoints[-1]
        dist_to_goal = math.hypot(goal[0] - last_added[0], goal[1] - last_added[1])
        
        # 如果最后一个添加的点距离终点太近（比如小于0.5米），
        # 且它不是起点，则替换它为终点，避免末端堆积
        if dist_to_goal < 0.5 and len(waypoints) > 1:
            waypoints.pop()
            waypoints.append(goal)
        else:
            waypoints.append(goal)
            
        return waypoints