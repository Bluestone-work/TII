"""
全局路径规划器 - 使用A*算法
为RL提供关键路径点，简化长距离导航任务
"""
import numpy as np
import heapq
import math
from typing import List, Tuple, Optional


class AStarPlanner:
    """A*路径规划器"""
    
    def __init__(self, map_data, resolution=0.05, origin=(-10.0, -10.0)):
        """
        初始化A*规划器
        
        Args:
            map_data: 占据栅格地图 (numpy array), 0=自由, 100=障碍
            resolution: 地图分辨率 (米/格子)
            origin: 地图原点 (x, y)
        """
        self.map_data = map_data
        self.resolution = resolution
        self.origin = origin
        self.height, self.width = map_data.shape
        
        # 膨胀障碍物，为机器人保留安全距离
        self.inflated_map = self._inflate_obstacles(map_data, inflation_radius=3)
        
    def _inflate_obstacles(self, map_data, inflation_radius=3):
        """膨胀障碍物，保证机器人安全"""
        from scipy.ndimage import binary_dilation
        obstacle_mask = (map_data > 50).astype(np.uint8)
        inflated = binary_dilation(obstacle_mask, iterations=inflation_radius)
        return inflated.astype(np.uint8) * 100
    
    def world_to_grid(self, x, y):
        """世界坐标转栅格坐标"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """栅格坐标转世界坐标"""
        x = grid_x * self.resolution + self.origin[0]
        y = grid_y * self.resolution + self.origin[1]
        return x, y
    
    def is_valid(self, grid_x, grid_y):
        """检查栅格点是否有效"""
        if grid_x < 0 or grid_x >= self.width:
            return False
        if grid_y < 0 or grid_y >= self.height:
            return False
        # 使用膨胀地图检查
        if self.inflated_map[grid_y, grid_x] > 50:
            return False
        return True
    
    def heuristic(self, a, b):
        """启发式函数：欧氏距离"""
        return math.hypot(b[0] - a[0], b[1] - a[1])
    
    def get_neighbors(self, node):
        """获取8连通邻居"""
        x, y = node
        neighbors = []
        # 8个方向
        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), 
                       (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                # 对角线距离为sqrt(2)
                cost = math.sqrt(2) if dx != 0 and dy != 0 else 1.0
                neighbors.append(((nx, ny), cost))
        return neighbors
    
    def plan(self, start_pos, goal_pos):
        """
        A*路径规划
        
        Args:
            start_pos: 起点世界坐标 (x, y)
            goal_pos: 终点世界坐标 (x, y)
            
        Returns:
            path: 世界坐标路径点列表 [(x1,y1), (x2,y2), ...]
        """
        # 转换为栅格坐标
        start = self.world_to_grid(*start_pos)
        goal = self.world_to_grid(*goal_pos)
        
        # 检查起点和终点是否有效
        if not self.is_valid(*start):
            print(f"⚠️ 起点 {start_pos} 无效（在障碍物中）")
            return None
        if not self.is_valid(*goal):
            print(f"⚠️ 终点 {goal_pos} 无效（在障碍物中）")
            return None
        
        # A*核心算法
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            # 到达目标
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                # 转换为世界坐标
                world_path = [self.grid_to_world(x, y) for x, y in path]
                return world_path
            
            # 扩展邻居
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        print(f"❌ 无法找到从 {start_pos} 到 {goal_pos} 的路径")
        return None
    
    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # 反转


class WaypointExtractor:
    """关键路径点提取器"""
    
    def __init__(self, turning_threshold=0.3, distance_threshold=1.0):
        """
        Args:
            turning_threshold: 转角阈值（弧度），超过此角度认为是拐点
            distance_threshold: 距离阈值（米），路径点间距超过此值则插入中间点
        """
        self.turning_threshold = turning_threshold
        self.distance_threshold = distance_threshold
    
    def extract(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        提取关键路径点
        
        策略：
        1. 检测转角超过阈值的拐点
        2. 在直线段每隔distance_threshold插入路径点
        3. 保留起点和终点
        
        Args:
            path: A*规划的完整路径
            
        Returns:
            waypoints: 关键路径点列表
        """
        if not path or len(path) < 2:
            return path
        
        waypoints = [path[0]]  # 起点
        
        i = 1
        while i < len(path) - 1:
            current = path[i]
            prev = path[i-1]
            next_point = path[i+1]
            
            # 计算转角
            angle = self._calculate_turn_angle(prev, current, next_point)
            
            # 是拐点：添加
            if abs(angle) > self.turning_threshold:
                waypoints.append(current)
                i += 1
            else:
                # 直线段：检查距离
                dist_from_last = math.hypot(
                    current[0] - waypoints[-1][0],
                    current[1] - waypoints[-1][1]
                )
                
                if dist_from_last >= self.distance_threshold:
                    waypoints.append(current)
                
                i += 1
        
        # 添加终点
        waypoints.append(path[-1])
        
        # 确保相邻路径点有一定距离（去除过密点）
        waypoints = self._remove_close_points(waypoints, min_dist=0.5)
        
        return waypoints
    
    def _calculate_turn_angle(self, p1, p2, p3):
        """计算三点之间的转角"""
        v1 = (p2[0] - p1[0], p2[1] - p1[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        
        # 计算向量夹角
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        det = v1[0]*v2[1] - v1[1]*v2[0]
        angle = math.atan2(det, dot)
        
        return angle
    
    def _remove_close_points(self, points, min_dist=0.5):
        """移除过于接近的路径点"""
        if len(points) <= 2:
            return points
        
        filtered = [points[0]]
        for i in range(1, len(points) - 1):
            dist = math.hypot(
                points[i][0] - filtered[-1][0],
                points[i][1] - filtered[-1][1]
            )
            if dist >= min_dist:
                filtered.append(points[i])
        
        # 始终保留终点
        filtered.append(points[-1])
        return filtered


def test_planner():
    """测试规划器"""
    # 创建简单测试地图
    map_data = np.zeros((100, 100), dtype=np.uint8)
    # 添加障碍物
    map_data[40:60, 30:35] = 100
    map_data[20:40, 60:65] = 100
    
    planner = AStarPlanner(map_data, resolution=0.1, origin=(0, 0))
    extractor = WaypointExtractor(turning_threshold=0.5, distance_threshold=1.0)
    
    # 规划路径
    start = (1.0, 1.0)
    goal = (8.0, 8.0)
    
    path = planner.plan(start, goal)
    if path:
        print(f"✅ 找到路径，共 {len(path)} 个点")
        waypoints = extractor.extract(path)
        print(f"✅ 提取关键点，共 {len(waypoints)} 个")
        print("关键路径点:")
        for i, wp in enumerate(waypoints):
            print(f"  {i}: ({wp[0]:.2f}, {wp[1]:.2f})")
    else:
        print("❌ 未找到路径")


if __name__ == '__main__':
    test_planner()
