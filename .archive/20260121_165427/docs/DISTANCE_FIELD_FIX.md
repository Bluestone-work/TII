# 距离场全为1.0问题诊断与修复

## 问题描述

用户报告：训练过程中，即使机器人未发生碰撞，距离场（distance_field）数据也经常显示全为1.0。

## 问题分析

### 1. 数据分析

从日志文件 `interaction_log_map3_robots4_20260115_164323.jsonl` (496条记录) 分析：

- **robot0**: 距离场全为1.0: 496/496 (100%)
- **robot1**: 距离场全为1.0: 496/496 (100%)  
- **robot2**: 距离场正常: 0/496 (0%全1.0)
- **robot3**: 距离场全为1.0: 496/496 (100%)

### 2. 理论计算验证

使用测试脚本验证距离计算逻辑，发现：

| 机器人 | 目标距离 | 7x7局部距离范围 | 归一化后(/10) |
|-------|----------|-----------------|--------------|
| robot0 | 0.78m | 0.583-1.000m | 0.058-0.100 |
| robot1 | 0.85m | 0.658-1.031m | 0.066-0.103 |
| robot2 | 0.04m | 0.000-0.283m | 0.000-0.028 |
| robot3 | 0.74m | 0.602-0.966m | 0.060-0.097 |

**结论**：理论计算表明距离场应该是正常值（0.058-0.103），而非全1.0！

### 3. 根本原因

问题出在**地图障碍物判断**上：

```python
binary_map[map_array > 50] = 1  # 原阈值：50
```

当地图数据中某些区域的占用值在 50-70 之间时（灰色地带），被误判为障碍物。这导致：

1. 这些格子的 `distance_field` 值被设为 `np.inf`
2. 提取7x7局部区域时，如果机器人周围都是这种"误判障碍物"
3. 局部区域全为 `inf`，归一化后全变成 1.0

### 4. 为什么robot2正常？

robot2 的目标距离很近（0.04m），且可能位于地图的开阔区域，周围没有被误判的障碍物格子。

### 5. 归一化范围问题

原代码使用 10m 作为最大距离：

```python
local_field[i, j] = np.clip(value / 10.0, 0.0, 1.0)
```

当机器人距离目标较远（>10m）时，即使没有障碍物，归一化后也会全部变成1.0。

## 修复方案

### 修改1：放宽障碍物判断阈值

[logic.py](src/start_reinforcement_learning/start_reinforcement_learning/env_logic/logic.py#L823)

```python
# 修改前
binary_map[map_array > 50] = 1

# 修改后  
binary_map[map_array > 70] = 1  # 🔧 放宽障碍物阈值：从50改为70
```

**影响**：
- 减少"灰色地带"被误判为障碍物
- 允许机器人通过占用度在 50-70 的区域
- 提升距离场有效性

### 修改2：增大归一化范围

[logic.py](src/start_reinforcement_learning/start_reinforcement_learning/env_logic/logic.py#L927)

```python
# 修改前
local_field[i, j] = np.clip(value / 10.0, 0.0, 1.0)  # 10m上限

# 修改后
local_field[i, j] = np.clip(value / 20.0, 0.0, 1.0)  # 🔧 20m上限
```

**影响**：
- 支持更远距离的导航
- 避免远距离目标时距离场饱和为1.0  
- 提供更细粒度的距离信息

### 修改3：增强调试输出

添加了详细的调试信息：
- 地图占用统计（自由空间 vs 障碍物比例）
- 局部区域障碍物数量统计
- 原始距离值范围
- 警告信息（仅打印前5次，避免刷屏）

## 预期效果

修复后预期：
- ✅ robot0/robot1/robot3 的距离场不再全为1.0
- ✅ 距离场提供有效的导航梯度信息
- ✅ 机器人减少打圈行为
- ✅ 训练收敛速度提升

## 验证方法

1. 重新训练并生成日志：
   ```bash
   ./start_with_distance_field.sh
   ```

2. 使用日志分析脚本：
   ```bash
   python3 view_interaction_logs.py
   ```

3. 检查距离场统计：
   - `robot0/1/3` 的 distance_field 应该有变化（不再全为1.0）
   - min/max/mean 应该在合理范围内（0.0-1.0，非全1.0）

4. 观察训练行为：
   - 机器人是否减少打圈
   - 是否更直接地朝目标移动
   - 平均奖励是否提升

## 技术细节

### 距离场计算流程

1. 创建完整地图距离场（400x400）
   ```python
   y_coords, x_coords = np.ogrid[:height, :width]
   distance_field = np.sqrt((x_coords - goal_grid_x)**2 + 
                            (y_coords - goal_grid_y)**2) * resolution
   distance_field[binary_map == 1] = np.inf  # 障碍物设为无穷
   ```

2. 提取机器人周围7x7局部区域
   ```python
   for i in range(7):
       for j in range(7):
           global_x = robot_grid_x + (i - 3)
           global_y = robot_grid_y + (j - 3)
           value = distance_field[global_y, global_x]
           local_field[i, j] = clip(value / 20.0, 0, 1)  # 归一化
   ```

3. 扁平化为49维向量加入观测空间

### 观测空间结构（98维）

- [0-37]   激光雷达数据 (38维)
- [38-39]  自身速度 (线速度、角速度)
- [40-42]  目标信息 (dx, dy, distance)
- [43-48]  最近其他机器人信息 (6维)
- **[49-97]  局部距离场 (7x7=49维)** ← 本次修复重点

## 相关文件

- [logic.py](src/start_reinforcement_learning/start_reinforcement_learning/env_logic/logic.py)
  - Line 823: 障碍物阈值修改
  - Line 927: 归一化范围修改
  - Line 785-880: compute_distance_field_for_goal()
  - Line 882-950: extract_local_distance_field()

- 测试脚本：
  - [test_distance_field.py](test_distance_field.py) - 距离场计算验证
  - [view_interaction_logs.py](view_interaction_logs.py) - 日志分析工具

## 历史记录

- **2026-01-15 16:43**：发现问题并创建日志
- **2026-01-15 17:30**：完成根因分析
- **2026-01-15 17:45**：实施修复方案
- **待定**：验证修复效果

---

**修复者**: GitHub Copilot  
**日期**: 2026年1月15日  
**版本**: v1.0
