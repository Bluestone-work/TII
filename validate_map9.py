#!/usr/bin/env python3
"""
验证 Map 9 (warehouse_dynamic) 配置
检查边界、spawn点、障碍物位置是否合理
"""

# 地图边界（从 warehouse_dynamic.world）
BOUNDARY = {
    'north': 4.0,
    'south': -4.0,
    'east': 4.0,
    'west': -4.0,
}

# 静态障碍物位置
STATIC_OBSTACLES = [
    (2.0, 1.5),
    (-1.8, 2.2),
    (1.2, -2.0),
    (-2.5, -1.5),
    (0.5, 2.8),
    (-0.8, -2.5),
    (2.8, -0.5),
    (-2.2, 0.8),
]

# 动态障碍物初始位置
DYNAMIC_OBSTACLES = [
    (1.5, 1.0),
    (-1.5, 1.0),
    (1.0, -1.5),
    (-1.0, -1.0),
    (0.5, 0.0),
    (-0.5, 0.5),
    (0.0, -0.5),
    (-0.3, -0.8),
]

# Fallback spawn poses (从 gnn_marl_env.py)
FALLBACK_POSES = [
    ((2.0, 0.0), (-2.0, 0.0)),
    ((1.4142, 1.4142), (-1.4142, -1.4142)),
    ((0.0, 2.0), (0.0, -2.0)),
    ((-1.4142, 1.4142), (1.4142, -1.4142)),
    ((-2.0, 0.0), (2.0, 0.0)),
    ((-1.4142, -1.4142), (1.4142, 1.4142)),
    ((0.0, -2.0), (0.0, 2.0)),
    ((1.4142, -1.4142), (-1.4142, 1.4142)),
]

ROBOT_RADIUS = 0.105
SAFETY_MARGIN = 0.15
OBSTACLE_RADIUS_STATIC = 0.20
OBSTACLE_RADIUS_DYNAMIC = 0.22
MIN_CLEARANCE = ROBOT_RADIUS + SAFETY_MARGIN + OBSTACLE_RADIUS_DYNAMIC  # 0.475m

def check_point_in_bounds(x, y):
    """检查点是否在地图边界内"""
    margin = 0.5  # 给边界留出0.5m安全距离
    return (BOUNDARY['west'] + margin <= x <= BOUNDARY['east'] - margin and
            BOUNDARY['south'] + margin <= y <= BOUNDARY['north'] - margin)

def distance(p1, p2):
    """计算两点间距离"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def main():
    print("=" * 60)
    print("Map 9 (warehouse_dynamic) 配置验证")
    print("=" * 60)
    
    # 检查边界
    print(f"\n1. 地图边界:")
    print(f"   X: {BOUNDARY['west']:.1f} 到 {BOUNDARY['east']:.1f} ({BOUNDARY['east'] - BOUNDARY['west']:.1f}m)")
    print(f"   Y: {BOUNDARY['south']:.1f} 到 {BOUNDARY['north']:.1f} ({BOUNDARY['north'] - BOUNDARY['south']:.1f}m)")
    
    # 检查静态障碍物
    print(f"\n2. 静态障碍物 ({len(STATIC_OBSTACLES)}个):")
    for i, (x, y) in enumerate(STATIC_OBSTACLES):
        in_bounds = check_point_in_bounds(x, y)
        status = "✓" if in_bounds else "✗"
        print(f"   {status} static_obs_{i}: ({x:5.1f}, {y:5.1f})")
        if not in_bounds:
            print(f"      WARNING: 超出安全边界!")
    
    # 检查动态障碍物
    print(f"\n3. 动态障碍物 ({len(DYNAMIC_OBSTACLES)}个):")
    for i, (x, y) in enumerate(DYNAMIC_OBSTACLES):
        in_bounds = check_point_in_bounds(x, y)
        status = "✓" if in_bounds else "✗"
        print(f"   {status} dyn_obs_{i}: ({x:5.1f}, {y:5.1f})")
        if not in_bounds:
            print(f"      WARNING: 超出安全边界!")
    
    # 检查spawn点
    print(f"\n4. Fallback Spawn 点 ({len(FALLBACK_POSES)}对):")
    all_spawn_points = []
    for i, (start, goal) in enumerate(FALLBACK_POSES):
        all_spawn_points.extend([start, goal])
        start_ok = check_point_in_bounds(*start)
        goal_ok = check_point_in_bounds(*goal)
        status = "✓" if (start_ok and goal_ok) else "✗"
        print(f"   {status} Route {i}: ({start[0]:5.2f}, {start[1]:5.2f}) → ({goal[0]:5.2f}, {goal[1]:5.2f})")
    
    # 检查spawn点与障碍物的距离
    print(f"\n5. Spawn点与障碍物最小距离检查:")
    all_obstacles = STATIC_OBSTACLES + DYNAMIC_OBSTACLES
    min_dist_all = float('inf')
    worst_case = None
    
    for spawn_pt in all_spawn_points:
        for obs_pt in all_obstacles:
            dist = distance(spawn_pt, obs_pt)
            if dist < min_dist_all:
                min_dist_all = dist
                worst_case = (spawn_pt, obs_pt)
    
    print(f"   最小距离: {min_dist_all:.3f}m")
    print(f"   推荐最小距离: {MIN_CLEARANCE:.3f}m (机器人半径+安全裕度+障碍物半径)")
    if min_dist_all < MIN_CLEARANCE:
        print(f"   ⚠ WARNING: 最小距离不足! 可能导致spawn时碰撞")
        print(f"   问题位置: Spawn{worst_case[0]} <-> Obstacle{worst_case[1]}")
    else:
        print(f"   ✓ 距离充足，spawn安全")
    
    # 障碍物之间的距离
    print(f"\n6. 障碍物分布密度:")
    obstacle_distances = []
    for i in range(len(all_obstacles)):
        for j in range(i+1, len(all_obstacles)):
            obstacle_distances.append(distance(all_obstacles[i], all_obstacles[j]))
    
    if obstacle_distances:
        print(f"   障碍物间最小距离: {min(obstacle_distances):.3f}m")
        print(f"   障碍物间平均距离: {sum(obstacle_distances)/len(obstacle_distances):.3f}m")
        print(f"   障碍物间最大距离: {max(obstacle_distances):.3f}m")
    
    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
