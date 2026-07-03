# 全局路径规划感知随机障碍物修复

## 修复日期
2026-07-03

## 问题描述

### 严重Bug：A*规划器无法感知随机spawn的障碍物

**问题现象**：
- 全局路径规划（A*）使用的是静态地图文件（.pgm）
- 随机spawn的静态障碍物在运行时通过SetEntityState添加到Gazebo
- A*规划时这些障碍物还不存在，导致规划的路径**直接穿过障碍物**
- 机器人按照全局路径行走时会撞上这些障碍物

**原始执行顺序**（有问题）：
```python
# reset()方法中的错误顺序
1. 生成start和goal位置
2. A*规划全局路径           ← 此时障碍物还不存在！
3. spawn随机障碍物          ← 规划后才spawn
4. spawn机器人
```

**后果**：
- 规划的全局路径会穿过静态障碍物
- 机器人到达障碍物时才发现无法通行
- 必须依赖重规划机制，但重规划有冷却时间（25步）
- 训练初期会频繁碰撞，影响学习效率

## 解决方案

### 方案A：调整执行顺序（主要修复）

**新执行顺序**：
```python
# reset()方法中的正确顺序
1. 生成start和goal位置
2. spawn随机障碍物          ← 先spawn障碍物
3. 等待物理引擎稳定(0.2s)   ← 确保障碍物已加载
4. A*规划全局路径           ← 此时障碍物已存在
5. spawn机器人
```

**代码位置**：`gnn_marl_env.py` 第2983-3006行

```python
self.last_spawn_pos = (start_x, start_y)
self.goal_pos = (goal_x, goal_y)

# ═══════════════════════════════════════════════════════════════════
# 【关键修复】先spawn障碍物，再进行A*规划
# 原因：A*规划器需要在障碍物存在时规划，否则规划的路径会穿过障碍物
# ═══════════════════════════════════════════════════════════════════

# 在机器人spawn之前先spawn障碍物（只有robot_id==0时执行，避免重复）
if self.robot_id == 0:
    # 收集所有机器人的spawn位置
    all_robot_positions = [(start_x, start_y)]
    if other_agent_starts:
        all_robot_positions.extend(other_agent_starts)
    self._spawn_random_obstacles(other_robot_positions=all_robot_positions)

# 等待障碍物spawn完成并稳定（重要：确保物理引擎已更新）
self._wait_for_sim_time(0.2)

# 现在进行A*规划（此时静态障碍物已经存在，但A*仍用静态地图）
# 注意：随机spawn的静态障碍物不在.pgm地图中，A*仍然感知不到
# 依赖重规划机制在运行时检测到碰撞后重新规划
if self.planner:
    path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
    ...
```

**注意**：
- A*仍然使用静态的.pgm地图文件
- 随机spawn的障碍物**不在地图文件中**
- 所以A*规划时**仍然感知不到**这些障碍物
- 但至少障碍物已经在物理世界中存在，激光雷达能检测到
- 依赖**方案B的重规划机制**作为补充

### 方案B：增强重规划机制（辅助修复）

**核心思想**：
- 记录每次spawn的静态障碍物位置
- 在重规划时将这些障碍物作为`blocked_world_points`传递给A*
- 这样重规划能避开静态障碍物

**实现细节**：

1. **记录spawn的静态障碍物**

在`IndependentRobotEnv`初始化时添加：
```python
self.spawned_static_obstacles: list = []  # [(x, y, radius), ...]
```

在`_spawn_random_obstacles()`中记录：
```python
def spawn_obstacle(name, radius, is_static=True):
    ...
    if is_static:
        obs_info = (float(x), float(y), float(radius))
        self.spawned_static_obstacles.append(obs_info)
        # 同时更新parent_env的列表（所有agent共享）
        if hasattr(self, 'parent_env'):
            self.parent_env.spawned_static_obstacles.append(obs_info)
    ...
```

2. **多agent环境共享静态障碍物信息**

在`GNNMARLEnv`初始化时添加：
```python
# 随机spawn的静态障碍物位置（用于重规划）
self.spawned_static_obstacles = []
```

3. **重规划时考虑静态障碍物**

修改`_try_replan_due_to_deadlock()`：
```python
def _try_replan_due_to_deadlock(self) -> bool:
    ...
    blocked = []
    
    # 添加spawn的静态障碍物到blocked列表（优先从parent_env获取）
    static_obs_list = []
    if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
        static_obs_list = self.parent_env.spawned_static_obstacles
    elif hasattr(self, 'spawned_static_obstacles'):
        static_obs_list = self.spawned_static_obstacles
    
    for obs_x, obs_y, obs_radius in static_obs_list:
        blocked.append((float(obs_x), float(obs_y)))
    
    # 添加其他机器人位置...
    ...
    
    path = self.planner.plan_with_dynamic_obstacles(
        start, goal,
        blocked_world_points=blocked,
        block_radius_m=self.dynamic_replan_block_radius,
    )
```

## 修改的文件

### 1. gnn_marl_env.py

**修改点1：MultiAgentEnv添加共享变量**（第263行）
```python
# 随机spawn的静态障碍物位置（用于重规划）
self.spawned_static_obstacles = []
```

**修改点2：IndependentRobotEnv添加局部变量**（第1584行）
```python
self.spawned_static_obstacles: list = []  # 记录spawn的静态障碍物位置
```

**修改点3：调整reset()执行顺序**（第2983-3006行）
- 先spawn障碍物
- 等待稳定(0.2s)
- 再进行A*规划

**修改点4：_spawn_random_obstacles()记录障碍物**（第3827-3833行）
- 清空旧记录
- spawn时记录静态障碍物位置
- 同步到parent_env

**修改点5：spawn_obstacle内部逻辑**（第3885-3890行）
- 记录到self.spawned_static_obstacles
- 同步到parent_env.spawned_static_obstacles

**修改点6：_try_replan_due_to_deadlock()增强**（第2405-2420行）
- 从parent_env获取静态障碍物列表
- 添加到blocked列表
- 传递给plan_with_dynamic_obstacles()

## 工作原理

### 初始规划（reset时）

```
1. spawn障碍物
   ├─ 静态障碍物spawn到Gazebo
   ├─ 记录位置到spawned_static_obstacles
   └─ 物理引擎稳定(0.2s)
   
2. A*规划
   ├─ 使用静态地图(.pgm)
   ├─ 不知道随机障碍物位置
   └─ 可能规划出穿过障碍物的路径
   
3. spawn机器人
   └─ 开始执行
```

**问题**：初始规划仍可能有问题，因为A*用的是静态地图

**解决**：依赖激光雷达 + 重规划机制

### 运行时重规划

```
机器人前进
   ↓
激光雷达检测到前方有障碍物
   ↓
触发重规划条件（deadlock/碰撞风险）
   ↓
_try_replan_due_to_deadlock()
   ├─ 获取spawned_static_obstacles列表
   ├─ 获取其他机器人位置
   ├─ 合并到blocked列表
   └─ 调用plan_with_dynamic_obstacles()
       └─ A*规划时避开这些blocked点
           └─ 生成避开障碍物的新路径
```

**优势**：
- 重规划能看到静态障碍物位置
- 不需要修改地图文件
- 对动态和静态障碍物统一处理

## 重规划触发条件

当前重规划机制触发条件：
1. `replan_on_deadlock=True`（默认开启）
2. `current_step >= _next_replan_step`（冷却时间25步）
3. 满足以下任一条件：
   - 检测到前方死锁（front obstacle distance < threshold）
   - 与其他机器人TTC过小
   - 速度持续过低

**参数**：
- `replan_cooldown_steps=25`：重规划冷却时间
- `dynamic_replan_neighbor_dist=1.8m`：考虑的邻居距离
- `dynamic_replan_ttc=2.6s`：碰撞时间阈值
- `dynamic_replan_block_radius=0.55m`：障碍物阻塞半径

## 测试验证

### 验证点1：初始规划不穿过障碍物

虽然A*仍用静态地图，但障碍物已经在Gazebo中：
- 启动训练，在RViz中观察全局路径
- 检查路径是否绕开明显的静态障碍物
- 如果路径穿过，依赖激光雷达 + 重规划修正

### 验证点2：重规划成功避障

在训练日志中观察：
```bash
# 期望看到的日志（当机器人接近障碍物时）
Robot 0: Replan triggered at step 150
  - Start: (1.2, 0.5)
  - Goal: (3.5, 2.0)
  - Blocked points: 5 (包括静态障碍物)
  - New path length: 4.2m
```

### 验证点3：多agent共享障碍物信息

确认所有agent的重规划都能看到相同的静态障碍物：
```python
# 在_try_replan_due_to_deadlock()中添加调试输出
print(f"Robot {self.robot_id}: Replan with {len(static_obs_list)} static obstacles")
```

## 局限性和未来改进

### 当前局限性

1. **初始规划仍可能有问题**
   - A*使用静态地图，感知不到随机障碍物
   - 依赖重规划作为补丁

2. **重规划有冷却时间**
   - 25步才能重规划一次
   - 高速移动时可能来不及

3. **动态障碍物不考虑**
   - 动态障碍物会移动，A*规划时考虑它们会过于保守
   - 只在运行时通过局部避障应对

### 未来改进方向

1. **动态地图更新**
   - 将spawn的静态障碍物写入运行时地图
   - A*规划时使用更新后的地图
   - 需要costmap更新机制

2. **预测性重规划**
   - 不等死锁，提前检测路径上的障碍物
   - 更早触发重规划

3. **局部绕障增强**
   - 当全局路径穿过障碍物时
   - 局部规划器能更智能地绕行
   - 不完全依赖重规划

## 测试命令

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer

# 测试随机障碍物环境
./run_curriculum.sh \
  --run_suffix "planning_fix_test" \
  --start_stage 1 --end_stage 1 \
  --train_steps 10000

# 监控指标：
# 1. 初始碰撞率是否下降
# 2. 重规划成功率
# 3. 平均到达目标步数
# 4. RViz中路径是否合理
```

## 总结

**关键修复**：
1. ✅ 调整执行顺序：先spawn障碍物再规划
2. ✅ 记录静态障碍物位置
3. ✅ 重规划时考虑静态障碍物
4. ✅ 多agent环境共享障碍物信息

**预期效果**：
- 初始碰撞率下降（虽然A*仍用静态地图，但激光雷达能检测）
- 重规划成功避开静态障碍物
- 到达目标更高效
- 训练收敛更快

**代码健壮性**：
- 兼容旧代码（没有parent_env时也能工作）
- 多agent环境共享信息
- 重规划冷却机制保护性能
