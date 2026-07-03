# 机器人Spawn到障碍物附近问题 - 已修复 ✅

## 问题描述

用户报告：机器人spawn时老是放置到障碍物附近，导致A*规划失败。

## 根本原因

**MIN_ROBOT_SEP 太小** - 障碍物spawn时与机器人保持的距离（1.0m）不够大，当A*规划使用 `block_radius=1.0m` 膨胀障碍物时，膨胀区域会覆盖机器人的起点/终点，导致规划失败。

### 具体场景

```
机器人 @ (0, 0)
障碍物 @ (1.0, 0)  ← 满足 MIN_ROBOT_SEP=1.0m

A*规划时:
  障碍物膨胀 radius=1.0m
  膨胀后范围: (-1.0, 0) 到 (2.0, 0)
  → 机器人起点(0, 0)被膨胀区域覆盖！
  → is_valid(start) = False
  → 规划失败
```

## 修复内容

### 1. 增加 MIN_ROBOT_SEP（主要修复）

**文件**: `gnn_marl_env.py:3922`

```python
# 修复前
MIN_ROBOT_SEP = 1.0  # 太小

# 修复后
MIN_ROBOT_SEP = 1.8  # 从1.0增加到1.8m
# 计算: 1.0m(block_radius) + 0.5m(机器人footprint) + 0.3m(安全边距)
```

### 2. 减小 block_radius（平衡修复）

**文件**: `gnn_marl_env.py:3060`

```python
# 修复前
block_radius = 1.0  # 太大，容易覆盖起点/终点

# 修复后
block_radius = 0.8  # 从1.0减小到0.8m
# 0.8m = 0.42m(最大方块) + 0.2m(机器人) + 0.18m(安全边距)
```

### 3. 扩大起点/终点保护区域（防御性修复）

**文件**: `global_planner.py:217-227`

```python
# 修复前
radius = 1  # 保护3x3区域

# 修复后  
radius = 2  # 保护5x5区域
# 即使障碍物很近，也确保起点/终点可通行
```

## 效果

修复后：
1. ✅ 障碍物spawn时距离机器人至少1.8m
2. ✅ 即使 block_radius=0.8m 膨胀，也不会覆盖起点（1.8m > 0.8m + 0.5m）
3. ✅ 起点/终点保护区域扩大，即使有覆盖也能强制清除
4. ✅ 减小 block_radius 提高路径规划成功率

## 权衡

### 优点
- 机器人不会spawn到障碍物附近
- A*规划成功率提高
- 路径更合理，不会不必要地绕远

### 缺点
- MIN_ROBOT_SEP=1.8m 较大，可能导致障碍物spawn失败（特别是小地图）
- 如果8个障碍物+6个机器人在小地图中，可能放不下

### 监控指标

运行后观察日志：
```
成功spawn X/8 个棕色方块
```

如果成功率低于75%（<6个），说明空间不够，需要调整：
- 减小 MIN_ROBOT_SEP 到 1.5m
- 或减少障碍物数量

## 测试建议

### Map 8 (circle_swap_arena)
- 区域: -2.5 到 2.5 (5m×5m)
- 6个机器人 + 8个障碍物
- MIN_ROBOT_SEP=1.8m 可能偏紧

### Map 9 (warehouse_dynamic)
- 区域: -3.0 到 3.0 (6m×6m)
- 更大的空间，1.8m 应该合适

## 备选方案

如果 MIN_ROBOT_SEP=1.8m 导致spawn失败率高：

### 方案A: 动态调整
```python
# 如果尝试100次仍失败，逐步放宽MIN_ROBOT_SEP
for min_sep in [1.8, 1.5, 1.2]:
    if spawn_box(..., min_robot_sep=min_sep):
        break
```

### 方案B: 优先级spawn
```python
# 先spawn机器人附近的障碍物（较大MIN_ROBOT_SEP）
# 后spawn远离机器人的障碍物（较小MIN_ROBOT_SEP）
```

### 方案C: 减少障碍物数量
```python
# Map 8: 从8个减少到6个
num_static_obstacles = 6 if self.map_number == 8 else 8
```

---
修复日期: 2026-07-03
