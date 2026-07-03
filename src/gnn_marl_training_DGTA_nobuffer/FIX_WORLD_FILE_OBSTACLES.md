# 🎯 修复：World文件中的预置障碍物未被感知

## 问题根因

**真正的根本原因：** World文件中已经预置了8个`dyn_obs_X`障碍物，但全局规划器只能感知到后续spawn的4个`static_box`障碍物。

### 障碍物层次结构

```
总共的障碍物：
├── World文件预置（启动时就存在）
│   ├── Map 8: 8个圆柱 (dyn_obs_0 ~ dyn_obs_7) - 在场地边缘
│   └── Map 9: 8个圆柱 (dyn_obs_0 ~ dyn_obs_7) - 在场地内部
└── 环境spawn（reset时动态生成）
    └── 4个方块 (static_box_0 ~ static_box_3) - 随机位置
```

**结果：** 
- 实际有 **12个障碍物** (8个预置 + 4个spawn)
- 但规划器只看到 **4个** (只有spawn的)
- **8个预置障碍物完全被忽略！**

## 为什么之前没发现？

1. **删除逻辑只删除`static_box`**，不删除`dyn_obs`
2. **`_DYN_OBS_SPAWNS`定义错误**：
   - Map 8: ✅ 正确 - 匹配world文件
   - Map 9: ❌ 错误 - 位置完全不对

### Map 9的错误定义

**World文件实际位置 (warehouse_dynamic.world):**
```python
dyn_obs_0: (0.8, 0.8)
dyn_obs_1: (-0.8, 0.8)
dyn_obs_2: (0.8, -0.8)
dyn_obs_3: (-0.8, -0.8)
dyn_obs_4: (1.2, 0.0)
dyn_obs_5: (0.0, 1.2)
dyn_obs_6: (0.0, -1.2)
dyn_obs_7: (-1.2, 0.0)
```

**之前错误的定义:**
```python
9: [
    (2.5, 2.5), (-2.5, 2.5), (2.5, -2.5), (-2.5, -2.5),  # ❌ 完全不对
    (2.5, 0.0), (-2.5, 0.0), (0.0, 2.5), (0.0, -2.5),     # ❌ 完全不对
],
```

**位置偏差：** 1.7-3.5米！难怪路径会直接穿过真实障碍物。

## 修复内容

### 文件：gnn_marl_env.py:3756-3761

**修复前:**
```python
# Map 9 (warehouse_dynamic): 动态障碍物活动区域
9: [
    # 使用类似Map 8的配置
    (2.5, 2.5), (-2.5, 2.5), (2.5, -2.5), (-2.5, -2.5),
    (2.5, 0.0), (-2.5, 0.0), (0.0, 2.5), (0.0, -2.5),
],
```

**修复后:**
```python
# Map 9 (warehouse_dynamic): 动态障碍物初始位置（来自world文件）
9: [
    # 来自warehouse_dynamic.world文件中dyn_obs_0-7的实际初始位置
    (0.8, 0.8), (-0.8, 0.8), (0.8, -0.8), (-0.8, -0.8),
    (1.2, 0.0), (0.0, 1.2), (0.0, -1.2), (-1.2, 0.0),
],
```

## 验证方法

### 1. 检查blocked_points数量

**修复前:**
```
🔧 blocked_points=4  # 只有static_box
```

**修复后应该看到:**
```
🔧 blocked_points=12-20
# 4个static_box + 8个dyn_obs + 激光扫描聚类
```

### 2. 观察rviz中的路径

**修复前:**
- 路径直接穿过圆柱障碍物
- 尤其是Map 9最明显

**修复后:**
- 路径绕开所有障碍物
- 包括world文件中的8个圆柱

### 3. 查看spawn成功率

**修复前:**
- 机器人spawn后路径规划失败
- "No path found" 频繁出现

**修复后:**
- spawn位置远离所有障碍物
- 路径规划成功率提高

## World文件位置

- Map 8: `/src/start_rl_environment_tb3/worlds/circle_swap_arena.world`
- Map 9: `/src/start_rl_environment_tb3/worlds/warehouse_dynamic.world`

## 相关修复

这是**第6个修复**，与之前的修复协同工作：

1. ✅ Y轴坐标转换修复
2. ✅ MIN_ROBOT_SEP增加到1.8m
3. ✅ block_radius减小到0.8m
4. ✅ 增强debug日志
5. ✅ 扩展删除逻辑到Map 8
6. ✅ **修正Map 9的`_DYN_OBS_SPAWNS`位置** ← 当前修复

## 立即生效

✅ 无需重新编译
✅ 无需重启ROS
✅ 直接运行训练即可

```bash
./run_curriculum.sh 2>&1 | tee debug.log
```

## 预期结果

```
[Robot 0] 📍 Spawning 4/4 blocks at filtered positions
🔧 [AStarPlanner] blocked_points=16
   ├── 4个 static_box (随机spawn)
   ├── 8个 dyn_obs (world预置)
   └── 4个 激光聚类
🗑️  Robot 0: 已清理旧的static_box (0-15)
✅ Global path found: start=(x,y) -> goal=(x,y), 42 waypoints
```

路径不再穿过障碍物！ 🎉
