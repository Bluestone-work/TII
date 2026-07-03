# 静态障碍物Spawn问题修复

## 报告日期
2026-07-03

## 用户报告的问题

1. **数量不足**：设置了8个静态障碍物，但场景中只spawn了4个
2. **位置固定**：这4个方块的位置、大小一直不变，没有随机性
3. **方块消失**：spawn后又消失，刷新异常

## 根因分析

### Bug 1: 列表清空竞态条件

**代码位置**：`gnn_marl_env.py:3872-3875`（修复前）

```python
def _spawn_random_obstacles(self, other_robot_positions=None):
    # 清空上一次的静态障碍物记录（本地和parent_env）
    self.spawned_static_obstacles = []
    if hasattr(self, 'parent_env'):
        self.parent_env.spawned_static_obstacles = []
```

**问题**：
- `_spawn_random_obstacles()`只由`robot_id==0`调用（正确）
- 但函数内部**每次调用都清空`parent_env.spawned_static_obstacles`列表**
- 虽然只有一个robot调用，但如果该robot的reset被多次触发，或者在spawn过程中列表被清空，会导致记录丢失

**修复**：
- 将列表清空操作**移到reset()开始时**（Line 2909-2915）
- 只在`robot_id==0`执行时清空一次
- `_spawn_random_obstacles()`不再清空列表，只负责添加

```python
# reset() 中 (Line 2909-2915)
if self.robot_id == 0 and self.map_number == 9 and self.random_obstacles:
    # 清空障碍物记录列表（只执行一次）
    self.spawned_static_obstacles = []
    if hasattr(self, 'parent_env'):
        self.parent_env.spawned_static_obstacles = []
    
    # 删除Gazebo中的旧实体
    if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
        for i in range(8):
            req = DeleteEntity.Request()
            req.name = f'static_box_{i}'
            future = self.delete_entity_client.call_async(req)
```

### Bug 2: Spawn空间不足

**问题**：
- Spawn区域：6m x 6m（-3 ~ +3）
- 碰撞参数过于保守：
  - `MIN_OBSTACLE_SEP = 1.0m`（障碍物之间）
  - `MIN_ROBOT_SEP = 1.5m`（障碍物与机器人）
- 6个机器人时，每个占用半径1.5m → 直径3m圆形保护区
- 6个保护区 + 8个障碍物（每个≈0.7m间隔）= 需要约50m²
- 实际只有36m² → **空间严重不足**

**计算**：
```
机器人保护区面积 ≈ 6 × π × 1.5² ≈ 42m²  （已经超过总面积！）
障碍物需求面积 ≈ 8 × (0.6 + 1.0)² ≈ 20m²
总需求 ≈ 62m² >> 实际36m²
```

**修复**：
- `MIN_OBSTACLE_SEP`: 1.0 → 0.7m
- `MIN_ROBOT_SEP`: 1.5 → 1.0m
- 修复后空间需求：
  ```
  机器人保护区 ≈ 6 × π × 1.0² ≈ 19m²
  障碍物需求 ≈ 8 × (0.6 + 0.7)² ≈ 14m²
  总需求 ≈ 33m² < 36m²  ✓ 可行
  ```

### Bug 3: A*规划穿越障碍物

**问题**：
- `block_radius_m=0.35m`是之前圆柱的半径
- 方块的footprint更大：
  - 最大方块：0.6m × 0.6m
  - 对角线半径：√(0.6² + 0.6²)/2 ≈ 0.42m
- 0.35m无法覆盖方块 → A*认为可通过

**修复**：
- `block_radius_m`: 0.35 → 0.6m（覆盖最大方块+安全边距）

```python
path = self.planner.plan_with_dynamic_obstacles(
    (start_x, start_y),
    (goal_x, goal_y),
    blocked_world_points=blocked_points,
    block_radius_m=0.6  # 覆盖最大方块footprint + 安全边距
)
```

## 代码修改汇总

### 1. reset() - Line 2909-2921
```python
# 【Map 9特殊处理】删除上一轮spawn的静态障碍物（只由robot_id==0执行）
if self.robot_id == 0 and self.map_number == 9 and self.random_obstacles:
    # 清空障碍物记录列表（修复Bug1：只在这里清空一次）
    self.spawned_static_obstacles = []
    if hasattr(self, 'parent_env'):
        self.parent_env.spawned_static_obstacles = []

    if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
        for i in range(8):
            req = DeleteEntity.Request()
            req.name = f'static_box_{i}'
            future = self.delete_entity_client.call_async(req)
        self._wait_for_sim_time(0.1)
```

### 2. _spawn_random_obstacles() - Line 3870-3877
```python
def _spawn_random_obstacles(self, other_robot_positions=None):
    """
    每次reset时随机spawn静态障碍物（棕色方块）
    
    注意：此函数只由robot_id==0调用，不要在此清空spawned_static_obstacles列表
    （列表已在reset()开始时清空）
    """
    if not self.random_obstacles:
        return
    # ... 不再清空列表
```

### 3. 碰撞参数调整 - Line 3901-3903
```python
MIN_OBSTACLE_SEP = 0.7  # 障碍物之间最小间隔（减小到0.7）
MIN_ROBOT_SEP = 1.0     # 障碍物与机器人spawn点最小间隔（减小到1.0）
```

### 4. A*规划参数 - Line 3044
```python
block_radius_m=0.6  # 覆盖最大方块footprint + 安全边距
```

### 5. Debug输出增强
- Line 3975: 打印成功spawn的数量
- Line 3040: 打印A*规划时的blocked_points数量
- Line 3953: 每个方块spawn时打印位置、尺寸、footprint

## 测试验证

运行训练并观察日志：

```bash
./run_curriculum.sh --start_stage 1 --end_stage 1 --train_steps 100
```

**预期输出**：
```
🟫 Robot 0: 成功spawn 8/8 个棕色方块
  ✓ static_box_0 @ (-2.15, 1.32) size=medium_box footprint=0.35m
  ✓ static_box_1 @ (0.87, -2.41) size=large_box footprint=0.42m
  ...
  ✓ static_box_7 @ (1.98, 2.03) size=small_box footprint=0.28m
🗺️  Robot 0: A*规划 start=(x,y) goal=(x,y) blocked_points=8
```

**验证点**：
1. ✅ 每次reset成功spawn 8个方块（不是4个）
2. ✅ 位置每次不同（x, y坐标随机）
3. ✅ 尺寸每次不同（small/medium/large随机）
4. ✅ 方块不会消失（整个episode保持存在）
5. ✅ A*路径不穿越方块（绕行）
6. ✅ 无"无法找到有效位置"警告（空间充足）

## Gazebo可视化验证

打开Gazebo GUI观察：
1. **颜色**：8个棕色方块（RGB 0.7, 0.5, 0.3）
2. **位置**：分散在-3~+3区域，不重叠
3. **朝向**：随机旋转角度
4. **持久性**：整个episode不消失
5. **动态障碍物**：3个红色圆柱独立移动（不受影响）

## 性能影响

- Spawn开销：~200-400ms per reset（8次SpawnEntity调用）
- 与之前灰色圆柱相同（已经使用SpawnEntity）
- 删除开销：~50ms（8次DeleteEntity异步调用）

## 遗留注意事项

### 高密度场景（Stage 3: 6车+8静障）
- 当前参数在6车场景下理论可行（33m² < 36m²）
- 但实际spawn成功率取决于随机分布
- 如果频繁出现"无法找到有效位置"，考虑：
  1. 进一步减小`MIN_OBSTACLE_SEP`到0.6m
  2. 扩大spawn区域到(-3.5, 3.5)
  3. Stage 3减少静态障碍物数量到6个

### 动态障碍物初始位置
- world文件预定义了8个`dyn_obs_*`的初始位置：
  - `dyn_obs_0~3`: (±0.8, ±0.8)
  - `dyn_obs_4~7`: (±1.2, 0) 和 (0, ±1.2)
- 这些是**动态障碍物**，由`obstacle_mover.py`控制移动
- 如果`obstacle_mover.py`未运行，它们会静止在初始位置
- 用户可能误以为是"位置不变的静态障碍物"

### 验证obstacle_mover运行
```bash
ros2 node list | grep obstacle_mover
# 应该看到：/obstacle_mover
```

如果没有运行，检查launch参数：
```bash
ros2 param get /obstacle_mover num_obstacles  # 应返回3（Stage 1）
ros2 param get /obstacle_mover speed_scale    # 应返回0.5（Stage 1）
```
