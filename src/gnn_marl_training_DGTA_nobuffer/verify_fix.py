#!/usr/bin/env python3
"""
快速验证修复是否生效
"""
import sys
sys.path.insert(0, 'gnn_marl_training')

from global_planner import AStarPlanner
import numpy as np

print("="*80)
print("验证修复版本")
print("="*80)

# 创建测试地图
map_data = np.zeros((400, 400), dtype=np.uint8)
planner = AStarPlanner(map_data, resolution=0.05, origin=(-10.0, -10.0))

# 测试坐标转换
test_point = (0.0, 0.0)
gx, gy = planner.world_to_grid(*test_point)
wx, wy = planner.grid_to_world(gx, gy)

print(f"\n坐标转换测试:")
print(f"  世界坐标: {test_point}")
print(f"  → 网格坐标: ({gx}, {gy})")
print(f"  → 世界坐标: ({wx:.2f}, {wy:.2f})")
print(f"  误差: ({abs(test_point[0]-wx):.3f}, {abs(test_point[1]-wy):.3f})")

# 测试规划
print(f"\n规划测试:")
blocked_points = [(0.0, 0.0), (1.0, 1.0)]
path = planner.plan_with_dynamic_obstacles(
    (0.0, -2.0),
    (0.0, 2.0),
    blocked_world_points=blocked_points,
    block_radius_m=1.0
)

if path:
    print(f"✅ 规划成功，路径点数: {len(path)}")
else:
    print(f"❌ 规划失败（预期的，因为障碍物阻挡）")

print("\n" + "="*80)
print("如果看到上面的日志输出（🔧 🗺️ ⚠️），说明修复已生效")
print("="*80)
