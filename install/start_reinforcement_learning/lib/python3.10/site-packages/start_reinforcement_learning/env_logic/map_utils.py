"""
地图工具模块：用于读取 PGM 地图并进行碰撞检测
"""
import os
import yaml
from ament_index_python.packages import get_package_share_directory

class MapCollisionChecker:
    def __init__(self, map_name):
        """
        初始化地图碰撞检测器
        Args:
            map_name: 地图名称 (如 'map1', 'corridor_swap' 等)
        """
        self.map_name = map_name
        self.width = 0
        self.height = 0
        self.resolution = 0.05
        self.origin = [0.0, 0.0, 0.0]
        self.img_data = None
        
        self._load_map()
    
    def _load_map(self):
        """加载地图 YAML 和 PGM 文件"""
        try:
            pkg_share = get_package_share_directory('start_reinforcement_learning')
            
            # 读取 YAML 配置
            yaml_path = os.path.join(pkg_share, 'maps', f'{self.map_name}.yaml')
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    map_data = yaml.safe_load(f)
                    self.resolution = map_data.get('resolution', 0.05)
                    self.origin = map_data.get('origin', [0.0, 0.0, 0.0])
                    img_filename = map_data.get('image', f'{self.map_name}.pgm')
            else:
                print(f"[WARN] Map YAML not found: {yaml_path}, using defaults")
                img_filename = f'{self.map_name}.pgm'
            
            # 读取 PGM 图像
            # 处理 .png 引用但实际为 .pgm 的情况
            pgm_path = os.path.join(pkg_share, 'maps', os.path.basename(img_filename))
            if not pgm_path.endswith('.pgm'):
                pgm_path = pgm_path.replace('.png', '.pgm')
            
            if os.path.exists(pgm_path):
                self.width, self.height, self.img_data = self._load_pgm(pgm_path)
                print(f"[INFO] Loaded map {self.map_name}: {self.width}x{self.height}, resolution={self.resolution}")
            else:
                print(f"[WARN] PGM file not found: {pgm_path}")
        
        except Exception as e:
            print(f"[ERROR] Failed to load map {self.map_name}: {e}")
    
    def _load_pgm(self, path):
        """读取 PGM 文件 (P5 格式)"""
        try:
            with open(path, 'rb') as f:
                header = b''
                while True:
                    line = f.readline()
                    if line.startswith(b'#'):
                        continue
                    header += line
                    if len(header.split()) >= 4:
                        break
                
                parts = header.split()
                if parts[0] != b'P5':
                    print(f"[WARN] Not P5 format: {path}")
                    return 0, 0, None
                
                w, h = int(parts[1]), int(parts[2])
                data = f.read()
                return w, h, data
        except Exception as e:
            print(f"[ERROR] Failed to load PGM: {e}")
            return 0, 0, None
    
    def _world_to_pix(self, x, y):
        """世界坐标转像素坐标"""
        px = int((x - self.origin[0]) / self.resolution)
        py_map = int((y - self.origin[1]) / self.resolution)
        # Y 轴翻转 (地图坐标系原点在左下，图像坐标系在左上)
        py = (self.height - 1) - py_map
        return px, py
    
    def is_position_free(self, x, y, radius=0.3, threshold=200):
        """
        检查世界坐标 (x, y) 周围是否空闲
        Args:
            x, y: 世界坐标
            radius: 检查半径 (米)
            threshold: 像素值阈值，小于此值认为是障碍物
        Returns:
            True 如果空闲，False 如果碰撞或超出边界
        """
        if self.img_data is None:
            print("[WARN] Map data not loaded, skipping collision check")
            return True
        
        px, py = self._world_to_pix(x, y)
        
        # 检查是否在地图范围内
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return False
        
        # 计算像素空间的半径
        radius_px = max(1, int(radius / self.resolution))
        
        # 检查矩形区域
        x_start = max(0, px - radius_px)
        x_end = min(self.width - 1, px + radius_px)
        y_start = max(0, py - radius_px)
        y_end = min(self.height - 1, py + radius_px)
        
        for yi in range(y_start, y_end + 1):
            for xi in range(x_start, x_end + 1):
                idx = yi * self.width + xi
                if idx < len(self.img_data):
                    val = self.img_data[idx]
                    if val < threshold:  # 障碍物
                        return False
        
        return True
    
    def get_clearance(self, x, y, radius=0.3, threshold=200):
        """
        获取位置的空闲程度（clearance值）
        Args:
            x, y: 世界坐标
            radius: 检查半径 (米)
            threshold: 像素值阈值，小于此值认为是障碍物
        Returns:
            空闲半径（米），0表示在障碍物内，越大表示越空闲
        """
        if self.img_data is None:
            return radius  # 如果没有地图数据，假设空闲
        
        px, py = self._world_to_pix(x, y)
        
        # 检查是否在地图范围内
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return 0.0
        
        # 计算像素空间的半径
        radius_px = max(1, int(radius / self.resolution))
        
        # 寻找最近的障碍物距离
        min_obstacle_dist = radius  # 初始为最大半径
        
        for yi in range(max(0, py - radius_px), min(self.height, py + radius_px + 1)):
            for xi in range(max(0, px - radius_px), min(self.width, px + radius_px + 1)):
                idx = yi * self.width + xi
                if idx < len(self.img_data):
                    val = self.img_data[idx]
                    if val < threshold:  # 是障碍物
                        # 计算到障碍物的距离
                        dist = ((xi - px)**2 + (yi - py)**2)**0.5 * self.resolution
                        min_obstacle_dist = min(min_obstacle_dist, dist)
                        if min_obstacle_dist == 0:
                            return 0.0  # 在障碍物内
        
        return min_obstacle_dist
    
    def find_nearest_free_position(self, x, y, max_search_radius=2.0, step=0.1, min_clearance=0.3):
        """
        如果位置被占用，寻找最近的空闲位置
        Args:
            x, y: 目标世界坐标
            max_search_radius: 最大搜索半径 (米)
            step: 搜索步长 (米)
            min_clearance: 最小所需空闲半径 (米)
        Returns:
            (x, y) 最近的空闲位置，如果找不到返回原位置
        """
        if self.get_clearance(x, y, radius=min_clearance) >= min_clearance:
            return (x, y)
        
        # 螺旋搜索
        for r in range(1, int(max_search_radius / step) + 1):
            radius = r * step
            for angle in range(0, 360, 15):  # 每15度一个点
                import math
                dx = radius * math.cos(math.radians(angle))
                dy = radius * math.sin(math.radians(angle))
                test_x, test_y = x + dx, y + dy
                
                if self.get_clearance(test_x, test_y, radius=min_clearance) >= min_clearance:
                    print(f"[INFO] Adjusted position ({x:.2f}, {y:.2f}) -> ({test_x:.2f}, {test_y:.2f})")
                    return (test_x, test_y)
        
        print(f"[WARN] Could not find free position near ({x:.2f}, {y:.2f}) with clearance {min_clearance}m")
        return (x, y)
