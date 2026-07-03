# Map9: warehouse_random 配置文档

## 📋 概述

**Map9 (warehouse_random)** 是基于 Map8 创建的新仓库环境，专门用于训练复杂环境中的多智能体协同导航和动态避碰。

---

## 🗺️ 环境特点

### 尺寸
- **12m × 12m** 仓库空间（与 Map3-7 相同）
- 边界：`(-5.7, 5.7, -5.7, 5.7)`
- 墙体高度：1.5m

### 障碍物配置

#### 静态障碍物（10个）
**5个方形货箱**：
- 尺寸：0.8m × 0.8m × 1.0m
- 位置：随机分布，带旋转角度
- 颜色：棕色/米色系（模拟货箱）
- AABB膨胀：+0.32m（安全裕度）

| ID | 位置 | 旋转 |
|----|------|------|
| box_0 | (-3.5, -3.5) | 0.3 rad |
| box_1 | (2.5, -4.0) | 0.8 rad |
| box_2 | (-1.5, 2.5) | 1.2 rad |
| box_3 | (4.0, 3.5) | 0.5 rad |
| box_4 | (0.5, -1.5) | 0.9 rad |

**5个圆柱货物**：
- 尺寸：半径 0.4m，高度 1.2m
- 位置：分散在场地内
- 颜色：蓝色/紫色系（模拟圆柱货物）
- AABB膨胀：半径 0.4 → 0.72m

| ID | 位置 |
|----|------|
| cyl_0 | (-4.5, 1.5) |
| cyl_1 | (3.0, -2.0) |
| cyl_2 | (-2.0, -1.0) |
| cyl_3 | (1.5, 4.0) |
| cyl_4 | (-0.5, 0.5) |

#### 动态障碍物（8个）
- **尺寸**：半径 0.22m，高度 0.8m（红色圆柱）
- **速度**：0.22 m/s（与机器人相同）
- **行为**：随机游走，避开静态障碍物和彼此
- **spawn点**：分布在四个象限边缘，避开静态障碍物

---

## 🎯 设计目标

### 1. 复杂导航场景
- **静态障碍物密度**：10个 / 144m² ≈ 0.069 个/m²
- **动态障碍物密度**：8个 / 144m² ≈ 0.056 个/m²
- **总障碍物**：18个（比 Map5 的 6 个墙体更复杂）

### 2. 多样化避碰训练
- **静态避碰**：方形和圆形障碍物混合
- **动态避碰**：8个移动障碍物随机游走
- **多智能体协同**：4-6个机器人同时导航

### 3. 泛化能力提升
- **不规则布局**：非网格化，更接近真实仓库
- **多种障碍物形状**：训练对不同几何形状的感知
- **随机旋转**：货箱带旋转，增加场景多样性

---

## 🚀 使用方法

### 1. 启动 Gazebo 环境
```bash
cd /home/wj/work/multi-robot-exploration-rl

ros2 launch start_rl_environment_tb3 start_multi_robot_gazebo.launch.py \
    world:=warehouse_random \
    num_robots:=4 \
    num_dynamic_obstacles:=5 \
    map_number:=9
```

### 2. 训练配置
在 `train_gnn_mappo_full.py` 中添加 Stage 配置：

```python
# Stage 配置示例
STAGE_CONFIGS = {
    # ... 现有 stages ...

    # Stage X: warehouse_random 复杂导航
    'X': {
        'map_number': 9,
        'num_agents': 4,
        'num_obstacles': 5,  # 5个动态障碍物（静态的10个在world中）
        'obs_speed_scale': 0.6,
        'max_episode_steps': 300,
    }
}
```

### 3. 运行训练
```bash
python3 gnn_marl_training/train_gnn_mappo_full.py \
    --env_stage X \
    --num_agents 4 \
    --num_obstacles 5 \
    --action_mode continuous \
    --num_train_iterations 100
```

---

## 📊 难度对比

| 地图 | 尺寸 | 静态障碍物 | 动态障碍物 | 难度 |
|------|------|-----------|-----------|------|
| Map1 | 3×6m | 0 | 可选 | ⭐ |
| Map3 | 12×12m | 4块墙+2柱 | 可选 | ⭐⭐ |
| Map5 | 12×12m | 4货架+2瓶颈 | 可选 | ⭐⭐⭐ |
| Map8 | 8×8m | 0 | 8个 | ⭐⭐⭐ |
| **Map9** | **12×12m** | **10个混合** | **8个** | **⭐⭐⭐⭐** |

**Map9 特点**：
- ✅ 最复杂的静态布局（10个不规则障碍物）
- ✅ 最多的动态障碍物（8个同时移动）
- ✅ 静态+动态混合，最接近真实仓库
- ⚠️ 难度最高，建议在 Stage3-4 使用

---

## 🔧 自定义静态障碍物

### 修改方法
编辑 `warehouse_random.world`，调整障碍物位置/数量：

```xml
<!-- 添加新的方形障碍物 -->
<model name="static_box_5">
  <static>true</static>
  <pose>X Y 0.5 0 0 ROTATION</pose>
  <link name="link">
    <collision name="collision">
      <geometry><box><size>0.8 0.8 1.0</size></box></geometry>
    </collision>
    <visual name="visual">
      <geometry><box><size>0.8 0.8 1.0</size></box></geometry>
      <material><ambient>R G B 1</ambient></material>
    </visual>
  </link>
</model>
```

### 同步更新 obstacle_mover.py
添加新障碍物的 AABB 到 Map9 配置：

```python
'aabbs': [
    # ... 现有 AABBs ...
    # 新增（box 0.8×0.8 + 膨胀 0.32）
    (X-0.72, X+0.72, Y-0.72, Y+0.72),  # static_box_5 @ (X, Y)
],
```

---

## 🧪 验证步骤

### 1. 检查 world 文件
```bash
# 检查语法
gazebo --verbose /home/wj/work/multi-robot-exploration-rl/src/start_rl_environment_tb3/worlds/warehouse_random.world
```

### 2. 验证 spawn 点距离
```bash
# 启动 obstacle_mover，查看日志
ros2 run start_rl_environment_tb3 obstacle_mover.py \
    map_number:=9 \
    num_obstacles:=5

# 期待看到：
# [obstacle_mover] ✅ Map9 spawn 点最小距离: 4.00m (OK)
```

### 3. 测试导航
```bash
# 启动完整环境，手动控制机器人测试导航
ros2 launch start_rl_environment_tb3 start_multi_robot_gazebo.launch.py \
    world:=warehouse_random \
    num_robots:=1 \
    num_dynamic_obstacles:=3 \
    map_number:=9
```

---

## 📝 文件清单

### 新增文件
- ✅ `worlds/warehouse_random.world` - Gazebo 世界文件
- ✅ `obstacle_mover.py` - Map9 配置（已添加）

### 修改文件
- ✅ `obstacle_mover.py:183-202` - 新增 Map9 配置
- ⏳ `train_gnn_mappo_full.py` - 需要添加 Stage 配置（用户自定义）

---

## 💡 训练建议

### 课程学习路径
```
Stage1 (Map1) → Stage2 (Map3) → Stage3 (Map5) → Stage4 (Map9)
    ↓              ↓                ↓                ↓
  空旷         简单墙体          货架走廊      复杂仓库
   ⭐            ⭐⭐              ⭐⭐⭐         ⭐⭐⭐⭐
```

### 超参数建议
```python
# Map9 推荐配置
{
    'num_agents': 4,           # 4个机器人协同
    'num_obstacles': 5,        # 5个动态障碍物（+10个静态）
    'obs_speed_scale': 0.6,    # 0.22 * 0.6 = 0.13 m/s（稍慢，降低难度）
    'max_episode_steps': 400,  # 增加步数（环境复杂，需要更多时间）
    'goal_reach_radius': 0.50, # 稍放宽到达半径
}
```

---

## 🎓 预期效果

训练完成后，智能体应该能够：
- ✅ 在复杂仓库环境中自主导航
- ✅ 同时避开静态障碍物（货箱/圆柱）和动态障碍物
- ✅ 与其他机器人协同，避免互相碰撞
- ✅ 泛化到不同形状的障碍物（方形/圆形）
- ✅ 处理不规则布局（非网格化）

---

**创建时间**: 2026-07-02  
**基于**: Map8 (circle_swap_arena)  
**难度等级**: ⭐⭐⭐⭐（最高）  
**推荐用于**: Stage 4-5 高级训练
