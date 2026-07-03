#!/usr/bin/env python3
"""
诊断全局路径规划器的障碍物感知问题

检查点：
1. spawned_static_obstacles 是否被正确记录
2. blocked_points 是否被正确传递给 plan_with_dynamic_obstacles
3. _build_dynamic_free_mask 是否正确膨胀障碍物
4. 路径是否真的避开了障碍物
"""

import numpy as np
import math
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def visualize_planner_state(
    map_shape: Tuple[int, int],
    resolution: float,
    origin: Tuple[float, float],
    inflated_map: np.ndarray,
    blocked_points: List[Tuple[float, float]],
    block_radius_m: float,
    start_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    path: List[Tuple[float, float]] = None,
):
    """可视化规划器状态"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 原始地图 (inflated_map)
    ax = axes[0]
    ax.imshow(inflated_map, cmap='gray_r', origin='upper')
    ax.set_title('Original Inflated Map')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')

    # 2. 动态障碍物mask
    ax = axes[1]
    # 模拟 _build_dynamic_free_mask 逻辑
    free_mask = (inflated_map <= 50).astype(np.uint8)

    if blocked_points:
        radius_px = max(1, int(math.ceil(block_radius_m / resolution)))
        yy, xx = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
        disk = (xx * xx + yy * yy) <= (radius_px * radius_px)

        height, width = map_shape
        for wx, wy in blocked_points:
            gx = int((wx - origin[0]) / resolution)
            gy = int((wy - origin[1]) / resolution)

            x0 = max(0, gx - radius_px)
            x1 = min(width - 1, gx + radius_px)
            y0 = max(0, gy - radius_px)
            y1 = min(height - 1, gy + radius_px)

            if x0 <= x1 and y0 <= y1:
                mask_x0 = radius_px - (gx - x0)
                mask_x1 = radius_px + (x1 - gx)
                mask_y0 = radius_px - (gy - y0)
                mask_y1 = radius_px + (y1 - gy)
                free_mask[y0:y1 + 1, x0:x1 + 1] &= ~disk[mask_y0:mask_y1 + 1, mask_x0:mask_x1 + 1]

    ax.imshow(free_mask, cmap='RdYlGn', origin='upper', vmin=0, vmax=1)
    ax.set_title(f'Free Mask with Dynamic Obstacles (block_radius={block_radius_m}m)')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')

    # 标记障碍物中心
    for wx, wy in blocked_points:
        gx = int((wx - origin[0]) / resolution)
        gy = int((wy - origin[1]) / resolution)
        ax.plot(gx, gy, 'r*', markersize=10, label='Obstacle Center')

    # 3. 世界坐标系视图
    ax = axes[2]

    # 绘制自由空间
    world_map = np.flipud(free_mask)  # 翻转Y轴以匹配世界坐标
    extent = [
        origin[0],
        origin[0] + map_shape[1] * resolution,
        origin[1],
        origin[1] + map_shape[0] * resolution
    ]
    ax.imshow(world_map, cmap='RdYlGn', origin='lower', extent=extent, alpha=0.6, vmin=0, vmax=1)

    # 绘制障碍物
    for wx, wy in blocked_points:
        circle = Circle((wx, wy), block_radius_m, color='red', alpha=0.3, label='Block Zone')
        ax.add_patch(circle)
        ax.plot(wx, wy, 'rx', markersize=10, markeredgewidth=2)

    # 绘制起点和终点
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=12, label='Start', markeredgewidth=2)
    ax.plot(goal_pos[0], goal_pos[1], 'bs', markersize=12, label='Goal', markeredgewidth=2)

    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')

        # 检查路径是否穿过障碍物
        violations = []
        for px, py in path:
            for wx, wy in blocked_points:
                dist = math.hypot(px - wx, py - wy)
                if dist < block_radius_m:
                    violations.append((px, py, dist))

        if violations:
            print(f"\n⚠️  路径穿过障碍物！")
            for px, py, dist in violations[:5]:  # 只显示前5个
                print(f"    路径点 ({px:.2f}, {py:.2f}) 距离障碍物 {dist:.2f}m < {block_radius_m}m")
                ax.plot(px, py, 'mo', markersize=8, markeredgewidth=2)

    ax.set_xlabel('World X (m)')
    ax.set_ylabel('World Y (m)')
    ax.set_title('World Coordinate View')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def check_obstacle_blocking(
    blocked_points: List[Tuple[float, float]],
    block_radius_m: float,
    path: List[Tuple[float, float]]
) -> dict:
    """检查路径是否被障碍物阻挡"""

    results = {
        'total_points': len(path),
        'violated_points': 0,
        'min_clearance': float('inf'),
        'violations': []
    }

    for px, py in path:
        min_dist = float('inf')
        for wx, wy in blocked_points:
            dist = math.hypot(px - wx, py - wy)
            min_dist = min(min_dist, dist)

            if dist < block_radius_m:
                results['violated_points'] += 1
                results['violations'].append({
                    'path_point': (px, py),
                    'obstacle': (wx, wy),
                    'distance': dist,
                    'threshold': block_radius_m
                })

        results['min_clearance'] = min(results['min_clearance'], min_dist)

    return results


if __name__ == '__main__':
    print("=" * 70)
    print("全局路径规划器障碍物感知诊断工具")
    print("=" * 70)

    # 模拟参数
    resolution = 0.05  # 5cm/pixel
    origin = (-10.0, -10.0)
    map_size = (400, 400)  # 20m x 20m at 0.05m/pixel

    # 创建一个简单的测试地图（全部自由空间）
    inflated_map = np.zeros(map_size, dtype=np.uint8)

    # 模拟spawn的障碍物
    blocked_points = [
        (0.0, 0.0),
        (2.0, 1.0),
        (-1.5, 2.5),
    ]

    block_radius_m = 1.0

    # 起点和终点
    start_pos = (-3.0, -3.0)
    goal_pos = (3.0, 3.0)

    # 模拟一条直线路径（会穿过障碍物）
    num_waypoints = 20
    path = []
    for i in range(num_waypoints):
        t = i / (num_waypoints - 1)
        px = start_pos[0] + t * (goal_pos[0] - start_pos[0])
        py = start_pos[1] + t * (goal_pos[1] - start_pos[1])
        path.append((px, py))

    print(f"\n地图参数:")
    print(f"  分辨率: {resolution}m/pixel")
    print(f"  原点: {origin}")
    print(f"  地图尺寸: {map_size} ({map_size[1]*resolution}m x {map_size[0]*resolution}m)")

    print(f"\n障碍物信息:")
    print(f"  数量: {len(blocked_points)}")
    print(f"  膨胀半径: {block_radius_m}m = {int(block_radius_m/resolution)}px")
    for i, (wx, wy) in enumerate(blocked_points):
        print(f"  [{i}] 世界坐标: ({wx:.2f}, {wy:.2f})")

    print(f"\n路径信息:")
    print(f"  起点: {start_pos}")
    print(f"  终点: {goal_pos}")
    print(f"  路径点数: {len(path)}")

    # 检查碰撞
    check_results = check_obstacle_blocking(blocked_points, block_radius_m, path)

    print(f"\n碰撞检测结果:")
    print(f"  总路径点: {check_results['total_points']}")
    print(f"  违规点数: {check_results['violated_points']}")
    print(f"  最小间隙: {check_results['min_clearance']:.3f}m")

    if check_results['violations']:
        print(f"\n❌ 发现 {len(check_results['violations'])} 个路径点穿过障碍物:")
        for v in check_results['violations'][:5]:
            print(f"    路径点 {v['path_point']} -> 障碍物 {v['obstacle']}: "
                  f"距离 {v['distance']:.3f}m < 阈值 {v['threshold']:.3f}m")
    else:
        print(f"\n✅ 路径未穿过任何障碍物")

    # 可视化
    print(f"\n正在生成可视化...")
    fig = visualize_planner_state(
        map_size, resolution, origin, inflated_map,
        blocked_points, block_radius_m, start_pos, goal_pos, path
    )

    output_path = '/home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer/planner_diagnosis.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化已保存到: {output_path}")

    print("\n" + "=" * 70)
    print("诊断建议:")
    print("=" * 70)
    print("1. 检查 spawned_static_obstacles 是否被正确记录")
    print("2. 检查 blocked_points 是否正确传递给 plan_with_dynamic_obstacles")
    print("3. 检查 block_radius_m 是否足够大（当前: {}m）".format(block_radius_m))
    print("4. 检查 _build_dynamic_free_mask 的膨胀逻辑是否正确")
    print("5. 在 rviz 中对比: 障碍物实际位置 vs 路径规划结果")
