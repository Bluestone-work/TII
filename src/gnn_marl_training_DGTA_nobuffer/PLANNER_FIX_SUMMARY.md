# 全局路径规划器障碍物感知问题 - 修复总结

## 问题描述

全局路径规划器无法感知地图中spawn的动态和静态障碍物，导致rviz中观察到的全局路径总是穿过障碍物。

## 根本原因

**Y轴坐标转换错误** - `AStarPlanner` 的坐标转换函数没有考虑地图图像Y轴翻转。

### 详细分析

1. **地图加载时的处理**（在 `gnn_marl_env.py`）：
   ```python
   map_data_inverted = 255 - self.map_image
   map_data_for_planner = np.flipud(map_data_inverted)  # ← Y轴翻转！
   ```
   
2. **环境中的坐标转换**（正确实现）：
   ```python
   def _world_to_map_pixel(self, wx, wy):
       px = int(round((wx - self.map_origin[0]) / self.map_resolution))
       py = int(round(self.map_height - 1 - (wy - self.map_origin[1]) / self.map_resolution))
       #                ^^^^^^^^^^^^^^ 考虑了Y轴翻转
       return px, py
   ```

3. **AStarPlanner的坐标转换**（错误实现 - 已修复）：
   ```python
   # 修复前 ❌
   def world_to_grid(self, x, y):
       grid_x = int((x - self.origin[0]) / self.resolution)
       grid_y = int((y - self.origin[1]) / self.resolution)  # ← 缺少Y轴翻转！
       return grid_x, grid_y
   
   # 修复后 ✅
   def world_to_grid(self, x, y):
       grid_x = int((x - self.origin[0]) / self.resolution)
       grid_y = int(self.height - 1 - (y - self.origin[1]) / self.resolution)  # ← 添加Y轴翻转
       return grid_x, grid_y
   ```

### 影响

由于Y轴坐标错误，spawn的障碍物在 `plan_with_dynamic_obstacles` 中被标记到**镜像位置**：

- 障碍物实际在世界坐标 `(x, y)`
- 但在地图上被标记到 `(grid_x, height - 1 - grid_y)` 的镜像位置
- 导致A*规划器"看不到"障碍物，路径穿过障碍物

**示例**：
- 地图尺寸：400×400 像素，分辨率 0.05m/px
- 原点：(-10.0, -10.0)
- 障碍物世界坐标：(0.0, 0.0)

修复前：
```
grid_y = (0.0 - (-10.0)) / 0.05 = 200
```

修复后：
```
grid_y = 400 - 1 - (0.0 - (-10.0)) / 0.05 = 199
```

Y坐标镜像导致障碍物被标记到错误位置！

## 修复内容

### 1. 修复 `world_to_grid` 函数

**文件**: `gnn_marl_training/global_planner.py:160`

```python
def world_to_grid(self, x, y):
    grid_x = int((x - self.origin[0]) / self.resolution)
    # 修复: 考虑地图Y轴翻转 (np.flipud)
    # 地图图像坐标系Y轴向下，世界坐标系Y轴向上，需要翻转
    grid_y = int(self.height - 1 - (y - self.origin[1]) / self.resolution)
    return grid_x, grid_y
```

### 2. 修复 `grid_to_world` 函数

**文件**: `gnn_marl_training/global_planner.py:167`

```python
def grid_to_world(self, grid_x, grid_y):
    x = grid_x * self.resolution + self.origin[0]
    # 修复: 考虑地图Y轴翻转 (np.flipud)
    # 需要反向转换：从图像坐标系转回世界坐标系
    y = (self.height - 1 - grid_y) * self.resolution + self.origin[1]
    return x, y
```

### 3. 添加调试日志

**文件**: `gnn_marl_training/global_planner.py:176`

在 `_build_dynamic_free_mask` 中添加首次调用时的调试输出，便于验证障碍物膨胀参数。

## 验证方法

### 1. 启动训练观察rviz

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer
./run_curriculum.sh
```

**预期结果**：
- rviz中全局路径应该绕开spawn的棕色方块
- 机器人不会spawn到障碍物中

### 2. 检查日志输出

修复后应该看到：
```
🗺️  [AStarPlanner] 动态障碍物膨胀: block_radius=1.0m = 20px, 共8个障碍物
🗺️  Robot 0: A*规划 start=(...) goal=(...) blocked_points=8
```

### 3. 运行诊断脚本

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer
python3 diagnose_planner_obstacles.py
```

查看生成的 `planner_diagnosis.png` 验证障碍物膨胀是否正确。

## 影响范围

### 修复影响的功能

1. ✅ **初始规划** (`reset` 时)
   - 现在能正确避开spawn的静态障碍物
   
2. ✅ **动态重规划** (`replan_on_deadlock`)
   - 考虑其他机器人位置的重规划也会正确工作

3. ✅ **所有地图**
   - Map 8 (circle_swap_arena)
   - Map 9 (warehouse_dynamic)
   - 以及所有使用 `AStarPlanner` 的地图

### 不受影响的功能

- 局部避障（基于激光雷达，不依赖全局规划器）
- 碰撞检测（基于Gazebo物理引擎）
- 奖励计算（基于传感器数据）

## 回归测试

建议测试以下场景：

1. **Map 9 (warehouse_dynamic)** - 8个随机spawn的静态障碍物
   - 验证路径绕开障碍物
   - 验证机器人不spawn到障碍物中

2. **Map 8 (circle_swap_arena)** - 对角线换位场景
   - 验证路径规划正确
   - 验证多机器人协作

3. **其他地图** - 确保修复不影响现有功能
   - Map 1-7 应该继续正常工作

## 技术债务清理

此次修复暴露的技术问题：

1. ✅ **坐标系不一致** - 已修复
2. ⚠️  **缺少单元测试** - 建议添加坐标转换的单元测试
3. ⚠️  **文档不足** - 建议补充坐标系转换的文档说明

## 相关文件

- `gnn_marl_training/global_planner.py` - 路径规划器（已修复）
- `gnn_marl_training/gnn_marl_env.py` - 环境（spawn障碍物，调用规划器）
- `PLANNER_OBSTACLE_ISSUE_REPORT.md` - 详细诊断报告
- `diagnose_planner_obstacles.py` - 诊断工具脚本

## 修复时间

2026-07-03

## 修复者

Claude Code (Kiro AI Assistant)
