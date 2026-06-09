import os
import math
import random
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node 

# ==========================================
# 1. 地图读取与碰撞检测工具函数
# ==========================================

def _load_pgm(path: str):
    """读取 PGM 地图文件 (P5格式)"""
    try:
        with open(path, 'rb') as f:
            header = b''
            while True:
                line = f.readline()
                # 过滤注释行
                if line.startswith(b'#'):
                    continue
                header += line
                if len(header.split()) >= 4: # P5 width height maxval
                    break
            
            parts = header.split()
            if parts[0] != b'P5':
                print(f"[WARN] Map file {path} is not P5 PGM. Collision check might fail.")
                return 0, 0, None
            
            w, h = int(parts[1]), int(parts[2])
            # 读取剩余的数据作为像素
            data = f.read()
            return w, h, data
    except Exception as e:
        print(f"[ERROR] Failed to load PGM: {e}")
        return 0, 0, None

def _is_free(img_data, width, height, px, py, radius_px=2, threshold=200):
    """检查像素点 (px, py) 周围 radius_px 范围内是否空闲"""
    if img_data is None: 
        return True # 如果地图加载失败，默认允许生成（避免崩溃）
        
    if px < 0 or py < 0 or px >= width or py >= height:
        return False
        
    # 简单的矩形区域检查
    x_start = max(0, px - radius_px)
    x_end = min(width - 1, px + radius_px)
    y_start = max(0, py - radius_px)
    y_end = min(height - 1, py + radius_px)
    
    for y in range(y_start, y_end + 1):
        for x in range(x_start, x_end + 1):
            idx = y * width + x
            if idx < len(img_data):
                # PGM: 0(黑/墙) -> 255(白/空)
                # 注意：Python bytes 取索引返回的是 int (0-255)
                val = img_data[idx]
                if val < threshold: # 小于阈值认为是障碍物
                    return False
    return True

def _world_to_pix(x, y, origin_x, origin_y, resolution, height_px):
    """世界坐标 -> 像素坐标"""
    px = int((x - origin_x) / resolution)
    # 地图坐标系原点通常在左下，图像坐标系在左上，需要 Y 轴翻转
    py_map = int((y - origin_y) / resolution) 
    py = (height_px - 1) - py_map
    return px, py

# ==========================================
# 2. 核心启动逻辑
# ==========================================

def launch_setup(context, *args, **kwargs):
    # 获取包路径 - 使用新的包名
    pkg_name = 'start_rl_environment_tb3'
    pkg_share = get_package_share_directory(pkg_name)
    
    # 获取 Launch 参数
    map_number = context.perform_substitution(LaunchConfiguration('map_number'))
    robot_number = int(context.perform_substitution(LaunchConfiguration('robot_number')))
    use_fixed_positions = context.perform_substitution(LaunchConfiguration('use_fixed_positions'))
    use_fixed = (use_fixed_positions.lower() == 'true')
    
    # 映射 map_number 到文件名
    map_mapping = {
        '1': 'map1', 
        '2': 'map2', 
        '3': 'corridor_swap', 
        '4': 'intersection',
        '5': 'warehouse_aisles',
        '6': 'interaction_hub',
        '7': 'interaction_hub_mini'
    }
    map_name = map_mapping.get(map_number, 'map1')
    
    print(f"[INFO] Spawning {robot_number} TurtleBot3 robots on {map_name}...")
    print(f"[INFO] Using {'fixed' if use_fixed else 'random'} positions")

    # 1. 读取 YAML 配置文件以获取分辨率和原点
    yaml_path = os.path.join(pkg_share, 'maps', f'{map_name}.yaml')
    origin = [0.0, 0.0, 0.0]
    resolution = 0.05
    img_filename = f'{map_name}.pgm'
    
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            map_data = yaml.safe_load(f)
            resolution = map_data['resolution']
            origin = map_data['origin'] # [x, y, yaw]
            img_filename = map_data['image']
    else:
        print(f"[WARN] Map YAML not found at {yaml_path}, using defaults.")

    # 2. 读取 PGM 图片用于碰撞检测
    # 处理 yaml 中可能指向同一目录下的图片
    pgm_path = os.path.join(pkg_share, 'maps', os.path.basename(img_filename))
    if not pgm_path.endswith('.pgm') and os.path.exists(pgm_path.replace('.png', '.pgm')):
         pgm_path = pgm_path.replace('.png', '.pgm')

    width, height, img_data = _load_pgm(pgm_path)

    # 3. 读取生成区域预设 (Spawn Presets)
    # 优先使用当前 TurtleBot3 包内的配置，缺失时再回退到旧包。
    preset_candidates = [os.path.join(pkg_share, 'config', 'spawn_presets.yaml')]
    try:
        orig_pkg_share = get_package_share_directory('start_rl_environment')
        preset_candidates.append(os.path.join(orig_pkg_share, 'config', 'spawn_presets.yaml'))
    except Exception:
        pass

    presets_path = next((p for p in preset_candidates if os.path.exists(p)), '')
    spawn_boxes = [[-2.0, -2.0, 2.0, 2.0]] # 默认的一个大框
    fixed_starts = []

    if presets_path:
        with open(presets_path, 'r') as f:
            presets = yaml.safe_load(f)
            if map_name in presets:
                # 获取固定起始点
                if 'fixed_starts' in presets[map_name]:
                    fixed_starts = presets[map_name]['fixed_starts']
                # 获取随机生成区域
                if 'start_regions' in presets[map_name]:
                    spawn_boxes = presets[map_name]['start_regions']
            else:
                # 尝试使用 'map1' 作为默认回退
                if 'map1' in presets and 'start_regions' in presets['map1']:
                     spawn_boxes = presets['map1']['start_regions']

    # 4. 生成坐标
    spawn_points = []
    
    # 判断是否使用固定位置
    # 对于map3以上（corridor_swap, intersection等）默认使用固定位置
    should_use_fixed = use_fixed or (int(map_number) >= 3)
    
    if should_use_fixed and len(fixed_starts) >= robot_number:
        # 使用固定位置
        print(f"[INFO] Using fixed positions for {robot_number} robots")
        for i in range(robot_number):
            start = fixed_starts[i]
            x, y, yaw = start['x'], start['y'], start['yaw']
            spawn_points.append((x, y, yaw))
            if 'description' in start:
                print(f"[INFO]   Robot {i}: {start['description']} at ({x:.2f}, {y:.2f})")
    else:
        # 使用随机位置生成
        print(f"[INFO] Using random positions for {robot_number} robots")
        min_dist_sq = 0.6 ** 2 # 机器人之间最小距离 (0.6m)
    
        for i in range(robot_number):
            valid_pose = False
            attempts = 0
            x, y, yaw = 0.0, 0.0, 0.0
            
            while not valid_pose and attempts < 1000:
                attempts += 1
                # 随机选择一个区域
                box = random.choice(spawn_boxes) # [xmin, ymin, xmax, ymax]
                x = random.uniform(box[0], box[2])
                y = random.uniform(box[1], box[3])
                yaw = random.uniform(-3.14, 3.14)
                
                # 检查 1: 与其他机器人的距离
                too_close = False
                for (ex, ey, _) in spawn_points:
                    if (x - ex)**2 + (y - ey)**2 < min_dist_sq:
                        too_close = True
                        break
                if too_close: continue
                
                # 检查 2: 地图碰撞
                px, py = _world_to_pix(x, y, origin[0], origin[1], resolution, height)
                if _is_free(img_data, width, height, px, py):
                    valid_pose = True
            
            if not valid_pose:
                print(f"[WARN] Could not find valid spawn for robot {i}, placing at (0,0)")
                x, y, yaw = 0.0, 0.0, 0.0
                
            spawn_points.append((x, y, yaw))

    # 5. 生成 Launch Actions 列表
    actions = []
    spawn_launch_file = os.path.join(pkg_share, 'launch', 'spawn_robots.launch.py')

    for i in range(robot_number):
        x, y, yaw = spawn_points[i]
        
        # 显式构建参数字典 - 使用 tb3 作为机器人名称前缀
        args_dict = {
            'robot_name': f'tb3_{i}',
            'robot_namespace': f'tb3_{i}',
            'robot_name_prefix': f'tb3_{i}/',
            'x': str(x),
            'y': str(y),
            'z': '0.01',  # TurtleBot3 更低矮，减少初始高度
            'rotation': str(yaw),
            'use_sim_time': 'true'
        }

        action = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(spawn_launch_file),
            # 关键修改：显式转换为 list，确保传入的是 [(key, val), ...] 格式
            launch_arguments=list(args_dict.items())
        )
        actions.append(action)

        # TF publisher: map -> tb3_X/odom
        tf_node = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'static_tf_tb3_{i}',
            arguments=['0', '0', '0', '0', '0', '0', 'map', f'tb3_{i}/odom'],
            output='screen'
        )
        actions.append(tf_node)

    return actions

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_number',
            default_value='1',
            description='Map number to select spawn regions'
        ),
        DeclareLaunchArgument(
            'robot_number',
            default_value='3',
            description='Number of TurtleBot3 robots to spawn'
        ),
        DeclareLaunchArgument(
            'use_fixed_positions',
            default_value='false',
            description='Use fixed positions (true) or random spawn (false). Map 3+ default to true.'
        ),
        # 使用 OpaqueFunction 执行 Python 逻辑
        OpaqueFunction(function=launch_setup)
    ])
