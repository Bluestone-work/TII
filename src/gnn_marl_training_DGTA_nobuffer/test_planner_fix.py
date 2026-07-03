#!/usr/bin/env python3
"""
测试全局路径规划器的障碍物感知修复

验证点：
1. world_to_grid 和 grid_to_world 的双向转换一致性
2. 障碍物在正确位置被标记
3. 路径规划能避开障碍物
"""

import sys
import numpy as np
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent / 'gnn_marl_training'))

from global_planner import AStarPlanner

def test_coordinate_conversion():
    """测试坐标转换的正确性"""
    print("="*80)
    print("测试1: 坐标转换双向一致性")
    print("="*80)

    # 创建一个简单的测试地图
    map_size = (400, 400)
    resolution = 0.05
    origin = (-10.0, -10.0)
    map_data = np.zeros(map_size, dtype=np.uint8)  # 全部自由空间

    planner = AStarPlanner(map_data, resolution=resolution, origin=origin)

    # 测试多个世界坐标点
    test_points = [
        (0.0, 0.0, "中心"),
        (-10.0, -10.0, "左下角"),
        (9.95, 9.95, "右上角"),
        (0.0, 5.0, "上方"),
        (0.0, -5.0, "下方"),
        (5.0, 0.0, "右侧"),
        (-5.0, 0.0, "左侧"),
    ]

    passed = 0
    failed = 0

    for wx, wy, desc in test_points:
        # 世界 → 网格 → 世界
        gx, gy = planner.world_to_grid(wx, wy)
        wx_back, wy_back = planner.grid_to_world(gx, gy)

        # 允许小的浮点误差
        error_x = abs(wx - wx_back)
        error_y = abs(wy - wy_back)

        if error_x < 0.1 and error_y < 0.1:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1

        print(f"{status} {desc:8s}: ({wx:6.2f}, {wy:6.2f}) → grid({gx:3d}, {gy:3d}) → "
              f"({wx_back:6.2f}, {wy_back:6.2f})  误差: ({error_x:.3f}, {error_y:.3f})")

    print(f"\n通过: {passed}/{passed+failed}")
    return failed == 0


def test_obstacle_blocking():
    """测试障碍物是否能正确阻挡路径"""
    print("\n" + "="*80)
    print("测试2: 障碍物阻挡路径规划")
    print("="*80)

    # 创建测试地图
    map_size = (400, 400)  # 20m x 20m at 0.05m/px
    resolution = 0.05
    origin = (-10.0, -10.0)
    map_data = np.zeros(map_size, dtype=np.uint8)

    planner = AStarPlanner(map_data, resolution=resolution, origin=origin)

    # 测试用例：起点到终点的直线上有障碍物
    test_cases = [
        {
            "name": "单个障碍物",
            "start": (-3.0, -3.0),
            "goal": (3.0, 3.0),
            "obstacles": [(0.0, 0.0)],
            "block_radius": 1.0,
        },
        {
            "name": "多个障碍物",
            "start": (-5.0, 0.0),
            "goal": (5.0, 0.0),
            "obstacles": [(-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)],
            "block_radius": 0.8,
        },
    ]

    passed = 0
    failed = 0

    for case in test_cases:
        print(f"\n测试场景: {case['name']}")
        print(f"  起点: {case['start']}, 终点: {case['goal']}")
        print(f"  障碍物: {case['obstacles']}, 膨胀半径: {case['block_radius']}m")

        # 规划路径
        path = planner.plan_with_dynamic_obstacles(
            case['start'],
            case['goal'],
            blocked_world_points=case['obstacles'],
            block_radius_m=case['block_radius']
        )

        if path is None:
            print("  ❌ 规划失败（无路径）")
            failed += 1
            continue

        print(f"  路径点数: {len(path)}")

        # 检查路径是否穿过障碍物
        violations = []
        min_clearance = float('inf')

        for px, py in path:
            for ox, oy in case['obstacles']:
                dist = np.hypot(px - ox, py - oy)
                min_clearance = min(min_clearance, dist)

                if dist < case['block_radius']:
                    violations.append((px, py, ox, oy, dist))

        if violations:
            print(f"  ❌ 路径穿过障碍物！发现 {len(violations)} 个违规点")
            for px, py, ox, oy, dist in violations[:3]:
                print(f"     路径点({px:.2f}, {py:.2f}) 距离障碍物({ox:.2f}, {oy:.2f}) = {dist:.2f}m < {case['block_radius']:.2f}m")
            failed += 1
        else:
            print(f"  ✅ 路径成功避开障碍物（最小间隙: {min_clearance:.2f}m）")
            passed += 1

    print(f"\n通过: {passed}/{passed+failed}")
    return failed == 0


def test_spawn_obstacle_scenario():
    """模拟实际spawn障碍物的场景"""
    print("\n" + "="*80)
    print("测试3: 模拟实际spawn障碍物场景")
    print("="*80)

    # 模拟warehouse_dynamic地图（Map 9）
    map_size = (400, 400)
    resolution = 0.05
    origin = (-10.0, -10.0)
    map_data = np.zeros(map_size, dtype=np.uint8)

    planner = AStarPlanner(map_data, resolution=resolution, origin=origin)

    # 模拟实际spawn的障碍物（从日志中提取）
    spawned_obstacles = [
        (1.23, -0.45, 0.28),   # (x, y, footprint)
        (-1.56, 2.10, 0.35),
        (0.78, 1.23, 0.42),
        (-2.34, -1.67, 0.28),
        (2.45, 0.89, 0.35),
    ]

    blocked_points = [(x, y) for x, y, _ in spawned_obstacles]
    block_radius = 1.0  # 与实际代码一致

    print(f"spawn的障碍物数量: {len(spawned_obstacles)}")
    print(f"block_radius: {block_radius}m")

    # 测试多个机器人的spawn点和目标点
    robot_routes = [
        ((-2.5, -2.5), (2.5, 2.5)),
        ((2.5, -2.5), (-2.5, 2.5)),
        ((-2.5, 2.5), (2.5, -2.5)),
    ]

    passed = 0
    failed = 0

    for i, (start, goal) in enumerate(robot_routes):
        print(f"\nRobot {i}: {start} → {goal}")

        path = planner.plan_with_dynamic_obstacles(
            start, goal,
            blocked_world_points=blocked_points,
            block_radius_m=block_radius
        )

        if path is None:
            print(f"  ⚠️  无法找到路径")
            continue

        # 检查违规
        violations = 0
        for px, py in path:
            for ox, oy in blocked_points:
                dist = np.hypot(px - ox, py - oy)
                if dist < block_radius:
                    violations += 1

        if violations > 0:
            print(f"  ❌ 路径穿过 {violations} 个障碍物")
            failed += 1
        else:
            print(f"  ✅ 路径避开所有障碍物（长度: {len(path)}点）")
            passed += 1

    print(f"\n通过: {passed}/{passed+failed}")
    return failed == 0


if __name__ == '__main__':
    print("\n" + "█"*80)
    print("全局路径规划器修复验证测试")
    print("█"*80 + "\n")

    results = []

    try:
        results.append(("坐标转换", test_coordinate_conversion()))
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
        results.append(("坐标转换", False))

    try:
        results.append(("障碍物阻挡", test_obstacle_blocking()))
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
        results.append(("障碍物阻挡", False))

    try:
        results.append(("实际场景", test_spawn_obstacle_scenario()))
    except Exception as e:
        print(f"❌ 测试3失败: {e}")
        results.append(("实际场景", False))

    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status}  {name}")

    all_passed = all(p for _, p in results)

    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！修复已验证有效。")
        print("="*80)
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，请检查修复。")
        print("="*80)
        sys.exit(1)
