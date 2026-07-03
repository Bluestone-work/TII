# Map 9 (warehouse_dynamic) 修复说明

## 修复的问题

### 1. 地图映射缺失
**问题**: `gnn_marl_env.py` 中的 `map_mapping` 字典只定义到 map 8，但课程学习脚本（stages 1-3）尝试使用 map 9。

**修复**: 在 `gnn_marl_env.py:1654` 添加了 map 9 的映射：
```python
map_mapping = {..., 8: 'circle_swap_arena', 9: 'warehouse_dynamic'}
```

### 2. 动态障碍物被标记为静态
**问题**: `warehouse_dynamic.world` 中所有 8 个动态障碍物 (`dyn_obs_0` 到 `dyn_obs_7`) 都被错误地标记为 `<static>true</static>`，导致它们无法移动。

**修复**: 
- 将所有动态障碍物改为 `<static>false</static>`
- 添加了惯性属性 `<inertial><mass>5.0</mass>...`，使它们可以被物理引擎正确处理
- 添加了红色材质以区分动态障碍物

### 3. 障碍物初始位置不合理
**问题**: 原始动态障碍物位于地图边界（±3.5），距离边界墙（±4.0）只有 0.5m。更严重的是，某些动态障碍物(如1.5, 1.0)距离spawn点(1.41, 1.41)只有0.42m，远小于推荐的最小间隔0.475m，机器人spawn时容易发生碰撞。

**修复**:
- 动态障碍物全部移到地图中心区域
- 采用对称的8个位置：±0.8, ±0.8（对角线4个）和 ±1.2, 0 以及 0, ±1.2（轴向4个）
- 确保与所有spawn点的最小距离≥0.8m，远大于推荐的0.475m安全距离
- 障碍物不会阻塞主要通道，为机器人导航留出足够空间

### 4. 静态障碍物数量不足
**问题**: 原始地图只有动态障碍物，缺少静态障碍物，环境过于简单。

**修复**: 添加了 8 个静态障碍物（`static_obs_0` 到 `static_obs_7`），分布在地图各处：
- 位置: (2.0, 1.5), (-1.8, 2.2), (1.2, -2.0), (-2.5, -1.5), (0.5, 2.8), (-0.8, -2.5), (2.8, -0.5), (-2.2, 0.8)
- 半径: 0.20m (比动态障碍物稍小)
- 颜色: 灰色，区分于红色的动态障碍物

### 5. 缺少地图配置文件
**问题**: 缺少 `warehouse_dynamic.yaml` 和 `warehouse_dynamic.pgm` 文件。

**修复**:
- 创建了 `warehouse_dynamic.yaml`，配置与 `circle_swap_arena` 相同（8m×8m 区域）
- 从 `circle_swap_arena.pgm` 复制了地图图像文件

### 6. 安全Spawn配置缺失
**问题**: `_MAP_SAFE_MARGIN` 字典没有 map 9 的条目。

**修复**: 在 `gnn_marl_env.py:3447` 添加：
```python
_MAP_SAFE_MARGIN = {
    ...
    8: 8,  # circle_swap_arena
    9: 8,  # warehouse_dynamic
}
```

## 地图配置总结

### 边界
- 范围: -4.0m 到 +4.0m（X 和 Y）
- 总面积: 8m × 8m
- 墙体厚度: 0.2m

### 静态障碍物（8个）
- 半径: 0.20m
- 高度: 0.8m
- 分布: 均匀分布在地图内部

### 动态障碍物（8个）
- 半径: 0.22m（比静态障碍物略大）
- 高度: 0.8m
- 质量: 5.0kg
- 初始位置（对称分布，确保与spawn点安全距离≥0.8m）:
  - 对角线4个: (±0.8, ±0.8)
  - 轴向4个: (±1.2, 0), (0, ±1.2)
- 颜色: 红色（RGB: 0.8, 0.2, 0.2），便于区分
- 与spawn点最小距离: 0.800m（推荐值: 0.475m）✓

### 机器人Spawn区域
- Fallback poses 使用 circle_swap 模式（半径 ±2.0m 的圆形区域）
- 所有spawn点都在地图边界内，距离边界墙至少 2.0m
- 距离障碍物有足够安全距离（通过 erosion 算法保证）

## 文件变更清单

1. `/src/start_rl_environment_tb3/worlds/warehouse_dynamic.world` - 完全重写
2. `/src/start_rl_environment_tb3/maps/warehouse_dynamic.yaml` - 新建
3. `/src/start_rl_environment_tb3/maps/warehouse_dynamic.pgm` - 复制自 circle_swap_arena
4. `/src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/gnn_marl_env.py` - 两处修改:
   - Line 1654: 添加 map 9 到 map_mapping
   - Line 3447: 添加 map 9 到 _MAP_SAFE_MARGIN

## 验证

运行以下命令验证修复：
```bash
cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash
./run_curriculum.sh --run_suffix "map9_warehouse_dynamic_4frames" \
  --start_stage 1 --end_stage 4 \
  --gat_critic_mode gat --graph_ablation dual_graph
```

预期结果：
- 地图正常加载
- 8 个动态障碍物（红色）可以被 obstacle_mover 移动
- 8 个静态障碍物（灰色）固定在地图上
- 机器人spawn在安全区域内，不会立即碰撞
- 每回合随机初始化都在合理范围内（地图边界内，远离障碍物）
