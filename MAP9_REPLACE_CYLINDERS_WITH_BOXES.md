# Map 9静态障碍物：从灰色圆柱改为棕色方块

## 修改日期
2026-07-03

## 用户需求
- ❌ 移除灰色圆柱（`static_obs_*`）作为静态障碍物
- ✅ 保留红色圆柱（`dyn_obs_*`）作为动态障碍物
- ✅ 使用棕色方块作为静态障碍物，模拟仓库货物场景

## 实现方案

### 1. 静态障碍物类型变更

**之前**：灰色圆柱（RGB 0.6, 0.6, 0.6）
- 半径：0.20m
- 高度：0.8m
- 形状：Cylinder

**现在**：棕色方块（RGB 0.7, 0.5, 0.3）
- 尺寸：随机选择3种规格
  - 小箱子：0.4m × 0.4m × 0.5m
  - 中箱子：0.5m × 0.5m × 0.6m
  - 大箱子：0.6m × 0.6m × 0.8m
- 形状：Box
- 朝向：随机旋转（0-360°）

### 2. 代码修改

#### A. `_spawn_random_obstacles()` 重写
**文件**：`gnn_marl_training/gnn_marl_env.py`

主要变更：
1. 移除圆柱spawn逻辑
2. 添加3种box类型定义
3. 计算方块footprint（对角线半径）
4. 调用`_generate_box_sdf()`生成SDF
5. 支持随机yaw朝向

```python
box_types = [
    {'size': (0.4, 0.4, 0.5), 'name': 'small_box'},
    {'size': (0.5, 0.5, 0.6), 'name': 'medium_box'},
    {'size': (0.6, 0.6, 0.8), 'name': 'large_box'},
]

for i in range(self.num_static_obstacles):
    box_config = random.choice(box_types)
    spawn_box(f'static_box_{i}', box_config)
```

#### B. 新增`_generate_box_sdf()` 方法
生成棕色方块SDF模型：

```python
def _generate_box_sdf(self, name, size):
    """生成方块障碍物的SDF模型（棕色，模仿仓库货物）"""
    sx, sy, sz = size
    # ... SDF with <box><size>{sx} {sy} {sz}</size></box>
    # Material: ambient/diffuse = (0.7, 0.5, 0.3, 1)
```

#### C. Reset清理逻辑更新
**Line 2915**：`static_obs_` → `static_box_`

```python
for i in range(8):
    req = DeleteEntity.Request()
    req.name = f'static_box_{i}'  # 改名
    future = self.delete_entity_client.call_async(req)
```

### 3. 碰撞检测参数

**方块footprint计算**：
```python
def get_box_footprint(size_tuple):
    sx, sy, _ = size_tuple
    return math.sqrt(sx**2 + sy**2) / 2  # 对角线半径
```

**间隔参数**：
- `MIN_OBSTACLE_SEP = 0.8m`（障碍物之间）
- `MIN_ROBOT_SEP = 1.2m`（与机器人spawn点）

### 4. 工作流程

```
Episode N结束
    ↓
Reset() - robot_id==0 执行清理
    ↓
DeleteEntity(static_box_0~7)
    ↓
_spawn_random_obstacles()
    ↓
随机选择box类型 → SpawnEntity(static_box_i)
    ↓
Episode N+1 开始
```

## 效果对比

| 特性 | 灰色圆柱（旧） | 棕色方块（新） |
|------|--------------|--------------|
| 颜色 | 灰色 RGB(0.6,0.6,0.6) | 棕色 RGB(0.7,0.5,0.3) |
| 形状 | 圆柱 | 方块 |
| 尺寸 | 固定半径0.20m | 3种规格随机 |
| 朝向 | 无影响 | 随机旋转 |
| 视觉效果 | 通用障碍物 | 仓库货物感 |
| 多样性 | 单一 | 尺寸+朝向组合 |

## 参数控制

### run_curriculum.sh
```bash
STAGE_STATIC_OBS_NUM=( [1]=8  [2]=8  [3]=8 ... )
```
控制每个Stage的**棕色方块数量**（不再是灰色圆柱）

### 命令行覆盖
```bash
./run_curriculum.sh --num_static_obstacles 5
```

## 保留的动态障碍物

**红色圆柱**（`dyn_obs_*`）保持不变：
- 由`obstacle_mover.py`控制spawn和移动
- 数量由`STAGE_OBS_NUM`控制
- 颜色：RGB(0.8, 0.2, 0.2)
- 半径：0.22m

## 文件清单

修改的文件：
1. `gnn_marl_training/gnn_marl_env.py`
   - `_spawn_random_obstacles()`：重写spawn逻辑
   - `_generate_box_sdf()`：新增方法
   - `reset()`：更新清理逻辑（Line 2915）

参考文件（未使用但提供设计参考）：
- `start_rl_environment_tb3/scripts/warehouse_obstacle_spawner.py`

## 测试建议

1. **可视化验证**：
   ```bash
   ./run_curriculum.sh --start_stage 1 --end_stage 1 --train_steps 100
   ```
   打开Gazebo GUI观察：
   - 无灰色圆柱 ✓
   - 有棕色方块（3种尺寸混合） ✓
   - 有红色圆柱（动态移动） ✓

2. **Reset验证**：
   观察10个episode，确认：
   - 每次reset方块位置/尺寸/朝向随机
   - 旧方块正确删除
   - 无spawn失败警告

3. **碰撞检测验证**：
   - LiDAR能检测到方块
   - A*规划避开方块
   - 机器人不会穿过方块

## 注意事项

- ⚠️ **命名变更**：`static_obs_*` → `static_box_*`（如果有其他代码引用需同步修改）
- ⚠️ **footprint计算**：方块用对角线半径，比圆柱更保守
- ⚠️ **随机朝向**：方块旋转可能影响可通行空间，需测试
- ⚠️ **尺寸随机性**：训练多样性提升，但curriculum早期可能需要固定尺寸
