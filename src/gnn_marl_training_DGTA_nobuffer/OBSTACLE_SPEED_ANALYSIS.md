# 动态障碍物配置问题分析

## 🔍 你指出的两个问题

### 问题1: 初始化距离太近导致不移动

**现状**:
```python
# obstacle_mover.py
OBS_MIN_DIST = 0.60  # 两个障碍物最小安全距离（半径0.22×2 + 余量0.16）

def _obs_collides(x, y, others):
    for ox, oy in others:
        if math.hypot(x - ox, y - oy) < OBS_MIN_DIST:
            return True  # 碰撞
```

**问题**:
- spawn_points 是**手工固定位置**
- 没有在初始化时验证相互距离
- 如果两个 spawn 点距离 < 0.6m → 第二个障碍物spawn时检测到"碰撞" → 触发反弹逻辑
- 反弹算法假设"运动中碰撞"，对"初始就重叠"处理不好
- **结果：两个障碍物卡在原地互相阻挡**

**验证**:
```bash
# 检查各地图的 spawn_points 最小距离
Map1: min_dist = 0.7m ✓
Map2: min_dist = 1.2m ✓
Map3: min_dist = 8.0m ✓
Map4: min_dist = 1.0m ✓
Map5: min_dist = 2.4m ✓
Map6: min_dist = 0.8m ✓
Map7: min_dist = 0.8m ✓
Map8: min_dist = 2.04m ✓

# 理论上都 > 0.6m，但实际可能有问题？
```

**潜在原因**:
1. **Gazebo spawn 竞态**: 多个障碍物同时spawn，位置还没更新就互相检测
2. **坐标系偏差**: spawn_points 可能有小偏移
3. **物理碰撞**: Gazebo物理引擎检测到碰撞 → 位置被强制推开 → 进入"卡死"状态

---

### 问题2: 速度空间不匹配 ⚠️ **严重问题**

#### 当前配置对比

| 项目 | 速度 (m/s) | 说明 |
|------|-----------|------|
| **机器人最大速度** | **0.22** | max_forward_vel |
| **机器人实际速度** | **[-0.22, +0.22]** | 连续动作 action[0] ∈ [-1, 1] |
| **动态障碍物速度** | **0.11** | OBS_SPEED（固定） |
| **障碍物/机器人比** | **50%** | 0.11 / 0.22 |

#### 课程式训练配置

```python
# train_gnn_mappo_full.py
Stage1: obs_speed_scale = 0.5 → obs_speed = 0.3 * 0.5 = 0.15 m/s
Stage2: obs_speed_scale = 0.6 → obs_speed = 0.3 * 0.6 = 0.18 m/s
Stage3: obs_speed_scale = 0.7 → obs_speed = 0.3 * 0.7 = 0.21 m/s
Stage4: obs_speed_scale = 0.6 → obs_speed = 0.3 * 0.6 = 0.18 m/s
```

**等等！** 训练配置中的 `obs_speed` 实际上**没有被使用**！

```python
# env_config 只是传递参数，但 obstacle_mover.py 是独立节点
"obs_speed": 0.3 * stage_cfg['obs_speed_scale'],  # 没有实际作用！

# 实际速度由 obstacle_mover.py 硬编码
OBS_SPEED = 0.11  # m/s，固定值
```

---

### 🚨 核心问题：速度比例不合理

#### 问题分析

**假设**: 机器人以最大速度前进（0.22 m/s），障碍物迎面而来（0.11 m/s）

```python
# 相对速度
relative_speed = 0.22 + 0.11 = 0.33 m/s

# TTC 计算（距离 2.5m）
ttc = 2.5 / 0.33 = 7.6 秒

# 安全 TTC 阈值
RWD_TTC_SAFE = 2.5 秒

# 7.6s > 2.5s → 不触发惩罚！
```

**但如果障碍物速度是 0.22 m/s（与机器人相同）**:
```python
relative_speed = 0.22 + 0.22 = 0.44 m/s
ttc = 2.5 / 0.44 = 5.7 秒

# 仍然 > 2.5s，但更接近阈值
```

**当前 0.11 m/s 太慢的后果**:
1. **TTC 惩罚触发太晚** - 障碍物移动慢，TTC 长，惩罚不明显
2. **策略学不到紧迫性** - 障碍物"不够威胁"
3. **Sim2Real 差距大** - 真实行人速度 1.0-1.4 m/s，训练时只有 0.11 m/s

---

### 📊 速度对比（真实世界）

| 对象 | 速度 (m/s) | 训练中 | 比例 |
|------|-----------|--------|------|
| TurtleBot3 | 0.22 | 0.22 | 100% |
| 动态障碍物 | **0.11** | **0.11** | **50%** |
| 行人（慢走） | 1.0 | - | 455% ↑ |
| 行人（快走） | 1.4 | - | 636% ↑ |
| 自行车 | 3.0-5.0 | - | 1364-2273% ↑ |

**结论**: 训练时障碍物速度**远低于**真实场景，导致策略过度优化"慢速障碍物"场景。

---

## 🔧 建议修复方案

### 修复1: 初始化距离验证（高优先级）

```python
# obstacle_mover.py
def _validate_spawn_points(spawn_points, min_dist=OBS_MIN_DIST):
    """验证所有 spawn 点之间的距离"""
    for i, (x1, y1) in enumerate(spawn_points):
        for j, (x2, y2) in enumerate(spawn_points[i+1:], start=i+1):
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < min_dist:
                print(f"⚠️ Spawn point {i} and {j} too close: {dist:.2f}m < {min_dist}m")
                return False
    return True

# 在 MAP_CONFIGS 定义后验证
for map_id, cfg in MAP_CONFIGS.items():
    if not _validate_spawn_points(cfg['spawn_points']):
        print(f"❌ Map{map_id} spawn points validation failed!")
```

---

### 修复2: 提升障碍物速度（高优先级）

#### 方案A: 固定提速（简单）
```python
# obstacle_mover.py:34
OBS_SPEED = 0.22  # 从 0.11 提升到 0.22 (与机器人相同)
```

**预期效果**:
- 相对速度加倍 → TTC 减半
- 惩罚触发更频繁
- 策略学到更激进的避让

---

#### 方案B: 课程式速度（推荐）

```python
# 从 launch 参数读取速度
def __init__(self):
    self.declare_parameter('obs_speed', 0.11)  # 默认值
    obs_speed = self.get_parameter('obs_speed').value
    
    # 各 stage 不同速度
    # Stage1: 0.11 (慢速，易学习)
    # Stage2: 0.15 (中速)
    # Stage3: 0.22 (与机器人相同)
    # Stage4: 0.30 (超过机器人，更挑战)
```

**修改 launch 文件**:
```python
# 从 train_gnn_mappo_full.py 传递 obs_speed
Node(
    package='start_rl_environment_tb3',
    executable='obstacle_mover.py',
    parameters=[{'obs_speed': stage_cfg['obs_speed']}]
)
```

---

#### 方案C: 动态速度范围（最优）

```python
# 每个障碍物随机速度
self._speed = rng.uniform(0.15, 0.30)  # 范围 [0.15, 0.30] m/s

# 或者高斯分布
self._speed = rng.gauss(0.22, 0.05)  # 均值0.22，标准差0.05
```

**优势**:
- 模拟真实世界多样性
- 策略泛化能力更强
- 避免过拟合特定速度

---

### 修复3: 统一速度配置（中优先级）

**当前问题**: 
- `obstacle_mover.py` 硬编码 `OBS_SPEED = 0.11`
- `train_gnn_mappo_full.py` 计算 `obs_speed = 0.3 * obs_speed_scale`
- 两者**不同步**

**统一方案**:
```python
# train_gnn_mappo_full.py 传递真实速度到 launch
launch_cmd.append(f"obs_speed:={0.3 * stage_cfg['obs_speed_scale']:.2f}")

# obstacle_mover.py 从参数读取
self.declare_parameter('obs_speed', 0.11)
OBS_SPEED = self.get_parameter('obs_speed').value
```

---

## 🧪 验证方案

### 测试1: spawn 距离检查
```bash
# 打印所有地图的 spawn 点最小距离
python3 -c "
from obstacle_mover import MAP_CONFIGS
import math
for map_id, cfg in MAP_CONFIGS.items():
    pts = cfg['spawn_points']
    min_dist = min(math.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                   for i in range(len(pts)) for j in range(i+1, len(pts)))
    print(f'Map{map_id}: min_dist = {min_dist:.2f}m')
"
```

### 测试2: 速度提升效果
```bash
# 修改 OBS_SPEED = 0.22 后重新训练 Stage2
python3 train_gnn_mappo_full.py --env_stage 2 --num_train_iterations 50

# 对比指标
# - r_ttc_mean 应该更负（惩罚更频繁）
# - collision_rate 初期可能上升（难度增加）
# - 50 iters 后应该下降（学会应对）
```

---

## 💡 推荐实施顺序

### 阶段1: 验证问题（今日）
1. ✅ 运行 spawn 距离检查脚本
2. ✅ 查看训练日志，确认障碍物是否"不动"
3. ✅ 在 Gazebo 中观察障碍物行为

### 阶段2: 快速修复（明日）
1. ⭐ **方案A: 提升 OBS_SPEED 到 0.22**（1行代码）
2. ⭐ 重新训练 Stage2, 对比效果

### 阶段3: 完整优化（后续）
1. 🔬 实施方案B（课程式速度）
2. 🔬 统一速度配置（train ↔ launch）
3. 🔬 添加 spawn 距离验证

---

## 📚 相关文件

- `obstacle_mover.py` - 动态障碍物运动逻辑
- `train_gnn_mappo_full.py` - 训练配置
- `gnn_marl_env.py` - TTC 计算（已统一）

---

**分析时间**: 2026-07-02  
**下一步**: 验证 spawn 距离 + 提升障碍物速度到 0.22 m/s
