# 全局路径规划器障碍物感知问题诊断报告

## 问题描述

用户报告：全局路径规划器感知不到地图中放置的动态和静态障碍物位置，rviz中观察到全局路径总是穿过障碍物。从障碍物初始化时老是放置到障碍物中就能看出问题。

## 代码分析

### 当前实现流程

1. **障碍物Spawn** (`_spawn_random_obstacles`)
   - 只在 `robot_id==0` 时执行
   - 障碍物位置记录到 `spawned_static_obstacles` 列表
   - 格式: `[(x, y, footprint), ...]`

2. **路径规划** (reset时)
   - 获取 `blocked_points` 从 `spawned_static_obstacles`
   - 调用 `plan_with_dynamic_obstacles(blocked_points, block_radius_m=1.0)`
   - `_build_dynamic_free_mask` 在地图上膨胀障碍物区域

3. **地图数据流**
   ```
   原始地图图像 (map.pgm)
   → 反转颜色 (255 - map_image)
   → 翻转Y轴 (np.flipud)
   → AStarPlanner (inflated_map)
   → plan_with_dynamic_obstacles (动态膨胀blocked_points)
   ```

### 发现的潜在问题

#### 问题1: Map 8 没有启用随机障碍物

```python:gnn_marl_env.py:3902
if self.map_number == 9:  # warehouse_dynamic
    spawn_bounds = {'x_min': -3.0, 'x_max': 3.0, 'y_min': -3.0, 'y_max': 3.0}
elif self.map_number == 8:  # circle_swap_arena
    spawn_bounds = {'x_min': -2.5, 'x_max': 2.5, 'y_min': -2.5, 'y_max': 2.5}
else:
    # 其他地图暂不支持随机障碍物
    return
```

✅ Map 8 和 9 都支持，但其他地图不支持。

#### 问题2: spawned_static_obstacles 在多智能体环境下的同步

```python:gnn_marl_env.py:3040
if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
    blocked_points = [(x, y) for x, y, _ in self.parent_env.spawned_static_obstacles]
elif hasattr(self, 'spawned_static_obstacles'):
    blocked_points = [(x, y) for x, y, _ in self.spawned_static_obstacles]
```

**潜在问题**: 
- robot_id!=0 的机器人通过 `parent_env` 访问障碍物列表
- 如果 `parent_env` 没有正确设置或 `spawned_static_obstacles` 为空，则 `blocked_points` 为空
- 导致路径规划时看不到障碍物

#### 问题3: 地图坐标转换

```python:global_planner.py:160
def world_to_grid(self, x, y):
    grid_x = int((x - self.origin[0]) / self.resolution)
    grid_y = int((y - self.origin[1]) / self.resolution)
    return grid_x, grid_y
```

**潜在问题**:
- Y轴坐标可能需要翻转（地图图像坐标系 vs 世界坐标系）
- 当前实现**没有**翻转Y轴，可能导致障碍物位置错误

#### 问题4: 膨胀半径与障碍物实际尺寸不匹配

```python:gnn_marl_env.py:3060
block_radius = 1.0  # 增加到1.0m确保绕开
```

障碍物实际footprint:
- small_box: ~0.28m (0.4×0.4方块的对角线一半)
- medium_box: ~0.35m
- large_box: ~0.42m

`block_radius=1.0m` 应该足够，但如果障碍物位置记录错误或坐标转换有问题，仍会导致路径穿过障碍物。

## 根本原因推测

基于代码分析，最可能的原因是：

### **Y轴坐标不一致**

1. **地图加载时**: `map_data_for_planner = np.flipud(255 - self.map_image)`
   - Y轴被翻转，原点在图像左下角
   
2. **世界→网格转换**: `world_to_grid` 没有考虑Y轴翻转
   ```python
   grid_y = int((y - self.origin[1]) / self.resolution)  # ❌ 错误！
   ```
   
   应该是:
   ```python
   grid_y = int(self.height - 1 - (y - self.origin[1]) / self.resolution)  # ✅ 正确
   ```

3. **结果**: 障碍物在 `_build_dynamic_free_mask` 中被标记到错误的Y坐标位置，导致路径规划时"看不到"障碍物。

## 验证方法

添加调试输出，检查：

1. `spawned_static_obstacles` 是否非空
2. `blocked_points` 传递给规划器的值
3. 障碍物世界坐标 → 网格坐标的转换结果
4. `_build_dynamic_free_mask` 标记的网格位置

## 修复方案

### 方案1: 修复Y轴坐标转换（推荐）

**问题**: `world_to_grid` 函数没有考虑地图图像Y轴翻转。

**修复**: 在 `AStarPlanner.world_to_grid` 中添加Y轴翻转：

```python
def world_to_grid(self, x, y):
    grid_x = int((x - self.origin[0]) / self.resolution)
    # 修复: 考虑地图Y轴翻转 (flipud)
    grid_y = int(self.height - 1 - (y - self.origin[1]) / self.resolution)
    return grid_x, grid_y
```

**影响**: 需要同步修改 `grid_to_world`:

```python
def grid_to_world(self, grid_x, grid_y):
    x = grid_x * self.resolution + self.origin[0]
    # 修复: 考虑地图Y轴翻转
    y = (self.height - 1 - grid_y) * self.resolution + self.origin[1]
    return x, y
```

### 方案2: 增强调试日志

在 `_build_dynamic_free_mask` 中添加详细日志：

```python
def _build_dynamic_free_mask(self, blocked_world_points, block_radius_m, start=None, goal=None):
    free_mask = (self.inflated_map <= 50).copy()
    
    if not blocked_world_points:
        print(f"⚠️  [AStarPlanner] blocked_world_points 为空！")
        return free_mask
    
    radius_px = max(1, int(math.ceil(float(block_radius_m) / max(self.resolution, 1e-6))))
    print(f"🗺️  [AStarPlanner] 膨胀参数: block_radius={block_radius_m}m = {radius_px}px")
    
    for i, (wx, wy) in enumerate(blocked_world_points):
        gx, gy = self.world_to_grid(float(wx), float(wy))
        print(f"    障碍物[{i}] 世界({wx:.2f}, {wy:.2f}) → 网格({gx}, {gy})")
        
        # ... 原有膨胀逻辑 ...
```

### 方案3: 添加可视化验证

在规划后输出一张图片，显示：
- 原始地图
- 膨胀后的障碍物区域
- 规划的路径

用于在rviz之外独立验证。

## 下一步行动

1. **立即修复**: 应用方案1的Y轴坐标修复
2. **验证修复**: 运行训练，观察rviz中路径是否绕开障碍物
3. **增强日志**: 应用方案2，便于未来调试
4. **回归测试**: 确保修复不影响其他地图的路径规划

## 补充说明

如果Y轴坐标修复后问题仍存在，则需要检查：

1. `spawned_static_obstacles` 在 `parent_env` 中的同步机制
2. Gazebo中障碍物实际spawn的位置是否正确
3. 地图原点 `origin` 是否与world文件中的坐标系一致
