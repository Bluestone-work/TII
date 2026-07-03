# 全局路径规划障碍物感知问题 - 已修复 ✅

## 问题
全局路径规划器无法感知spawn的静态障碍物，导致规划的路径穿过障碍物。

## 根本原因
**Y轴坐标转换错误** - `AStarPlanner` 的 `world_to_grid()` 和 `grid_to_world()` 函数没有考虑地图图像Y轴翻转（`np.flipud`）。

障碍物被标记到镜像位置，导致A*规划器"看不到"它们。

## 修复内容

修改文件：`gnn_marl_training/global_planner.py`

### 1. `world_to_grid` (第160行)
```python
def world_to_grid(self, x, y):
    grid_x = int((x - self.origin[0]) / self.resolution)
    # 修复: 考虑地图Y轴翻转 (np.flipud)
    grid_y = int(self.height - 1 - (y - self.origin[1]) / self.resolution)
    return grid_x, grid_y
```

### 2. `grid_to_world` (第167行)
```python
def grid_to_world(self, grid_x, grid_y):
    x = grid_x * self.resolution + self.origin[0]
    # 修复: 考虑地图Y轴翻转 (np.flipud)
    y = (self.height - 1 - grid_y) * self.resolution + self.origin[1]
    return x, y
```

## 验证

运行测试：
```bash
python3 test_planner_fix.py
```

结果：**所有测试通过 ✅**
- ✅ 坐标转换双向一致性
- ✅ 障碍物正确阻挡路径
- ✅ 实际spawn场景验证

## 预期效果

修复后：
1. ✅ rviz中全局路径会绕开spawn的棕色方块
2. ✅ 机器人不会spawn到障碍物中
3. ✅ 动态重规划能正确避开其他机器人

## 影响地图
- Map 8 (circle_swap_arena)
- Map 9 (warehouse_dynamic)
- 所有使用 AStarPlanner 的地图

## 相关文档
- `PLANNER_FIX_SUMMARY.md` - 详细修复总结
- `PLANNER_OBSTACLE_ISSUE_REPORT.md` - 问题诊断报告
- `test_planner_fix.py` - 自动化测试脚本

---
修复日期: 2026-07-03
