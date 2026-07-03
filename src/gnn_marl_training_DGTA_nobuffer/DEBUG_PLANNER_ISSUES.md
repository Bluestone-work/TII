# 全局路径规划问题调试指南

## 当前问题

用户报告：
1. rviz中观察到global waypoint话题**时不时报错，不显示全局路径**
2. **有的时候路径还是直接穿过静态障碍物**

## 已修复的问题

✅ **Y轴坐标转换错误** - 已在 `global_planner.py` 中修复

## 仍需排查的问题

### 问题1: A*规划失败导致退化为直线

**代码位置**: `gnn_marl_env.py:3081`

```python
if path:
    # 成功：使用规划的路径
    self.global_waypoints = self.waypoint_extractor.extract(path, planner=self.planner)
else:
    # 失败：退化为直线（会穿过障碍物！）
    print(f"❌ Robot {self.robot_id}: A* plan 失败，退化为直线路径（会穿过障碍物！）")
    self.global_waypoints = [self.goal_pos]  # ← 这就是问题所在！
```

**为什么会失败？**

可能原因：
1. **起点或终点无效** - 坐标越界或在障碍物中
2. **blocked_points为空** - 障碍物未正确spawn或同步
3. **障碍物完全阻挡** - block_radius太大，覆盖了所有可行路径
4. **终点被膨胀障碍物覆盖** - goal恰好在障碍物的膨胀范围内

### 问题2: blocked_points可能为空

**代码位置**: `gnn_marl_env.py:3039-3043`

```python
blocked_points = []
if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
    blocked_points = [(x, y) for x, y, _ in self.parent_env.spawned_static_obstacles]
elif hasattr(self, 'spawned_static_obstacles'):
    blocked_points = [(x, y) for x, y, _ in self.spawned_static_obstacles]
```

**何时为空？**

1. **robot_id != 0 且 parent_env 未设置**
2. **robot_id == 0 但 _spawn_random_obstacles 失败**
3. **地图不支持随机障碍物** (只有Map 8和9支持)
4. **random_obstacles=False**

### 问题3: 障碍物spawn时机

**代码位置**: `gnn_marl_env.py:3026-3034`

```python
if self.robot_id == 0:
    self._spawn_random_obstacles(other_robot_positions=all_robot_positions)
else:
    self._wait_for_sim_time(0.4)  # 等待robot_0完成spawn
```

**潜在问题**：
- 0.4秒可能不够，导致其他robot规划时障碍物还未完全spawn
- 多进程环境下，`spawned_static_obstacles` 列表可能不同步

## 增强的调试日志

现在添加了详细日志，可以通过以下输出诊断问题：

### 1. 障碍物spawn日志

```
🗺️  [AStarPlanner] 动态障碍物膨胀: block_radius=1.0m = 20px, 共8个障碍物
```

如果看不到这行 → `blocked_points` 为空

### 2. 规划失败日志

```
⚠️  [AStarPlanner] 规划失败: start_pos=(...) → grid=(...) valid=False
    起点网格(...) 越界或在障碍物中
```

如果看到这个 → 起点或终点坐标有问题

### 3. A*搜索失败日志

```
⚠️  [AStarPlanner] A*搜索失败: 无法找到从 (...) 到 (...) 的路径
    可能原因: 障碍物完全阻挡了路径，或终点被膨胀障碍物覆盖
```

如果看到这个 → 路径确实被障碍物阻挡

### 4. 规划成功日志

```
✅ Robot 0: A*规划成功，路径长度=25点 → waypoints=8点
```

## 诊断步骤

### 步骤1: 检查blocked_points

启动训练后，观察日志：

```bash
./run_curriculum.sh 2>&1 | tee debug.log
```

搜索关键字：
```bash
grep "blocked_points" debug.log
```

**预期**：
```
🗺️  Robot 0: A*规划 start=(...) goal=(...) blocked_points=8
🗺️  Robot 1: A*规划 start=(...) goal=(...) blocked_points=8
🗺️  Robot 2: A*规划 start=(...) goal=(...) blocked_points=8
```

**如果看到**：
```
⚠️  Robot 1: blocked_points为空！障碍物可能未spawn完成
```

→ 说明robot_id != 0的机器人没有获取到障碍物列表

### 步骤2: 检查A*失败原因

搜索：
```bash
grep "A\* plan 失败\|A\*搜索失败\|规划失败" debug.log
```

查看失败的具体原因

### 步骤3: 在rviz中验证

1. **订阅话题**：`/waypoint_markers`
2. **观察**：
   - 绿色球：起点
   - 蓝色方块：终点
   - 蓝色线：全局路径
   - 棕色方块：静态障碍物

3. **检查**：
   - 路径是否存在？（如果不存在 → A*失败）
   - 路径是否绕开障碍物？（如果穿过 → Y轴坐标问题未解决，或退化为直线）

## 临时解决方案

如果A*频繁失败，可以调整参数：

### 方案1: 减小block_radius

**文件**: `gnn_marl_env.py:3060`

```python
block_radius = 0.7  # 从1.0减小到0.7
```

**效果**: 减少障碍物膨胀范围，增加可行路径

### 方案2: 增加等待时间

**文件**: `gnn_marl_env.py:3034`

```python
self._wait_for_sim_time(0.8)  # 从0.4增加到0.8
```

**效果**: 确保障碍物完全spawn后再规划

### 方案3: 改进失败退化策略

**文件**: `gnn_marl_env.py:3080-3088`

当前退化为直线会穿过障碍物，改为：

```python
else:
    print(f"❌ Robot {self.robot_id}: A* plan 失败")
    # 尝试不考虑动态障碍物的规划
    fallback_path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
    if fallback_path:
        print(f"    → 使用fallback规划（不考虑动态障碍物）")
        self.global_waypoints = self.waypoint_extractor.extract(fallback_path, planner=self.planner)
    else:
        print(f"    → fallback也失败，使用直线（会穿过障碍物！）")
        self.global_waypoints = [self.goal_pos]
    
    self.vis.publish_waypoints(
        self.global_waypoints,
        robot_id=self.robot_id,
        namespace=self.vis_namespace
    )
```

## 根本解决方案

### 方案A: 确保parent_env正确设置

检查 `GNNMARLEnv.__init__` 中是否正确设置：

```python
for i in range(self._num_agents):
    env = IndependentRobotEnv(robot_id=i, ...)
    env.parent_env = self  # ← 确保这行存在
    self.agents[f"agent_{i}"] = env
```

### 方案B: 使用共享内存或消息传递

当前实现依赖 `parent_env` 属性访问，在多进程环境下可能失效。

改进：
1. 使用ROS话题发布障碍物位置
2. 所有robot订阅该话题获取障碍物列表
3. 或使用共享内存（如Redis）

### 方案C: 每个robot独立spawn障碍物记录

**文件**: `gnn_marl_env.py:3026-3043`

```python
# 不再依赖robot_id==0独占spawn
# 让每个robot在reset时都读取Gazebo中的实际障碍物位置

if self.robot_id == 0:
    # robot_0负责spawn
    self._spawn_random_obstacles(other_robot_positions=all_robot_positions)
    # 发布到ROS话题
    self._publish_obstacle_positions()
else:
    # 其他robot订阅话题获取
    self._wait_for_sim_time(0.4)
    blocked_points = self._subscribe_obstacle_positions()
```

## 下一步

1. **运行训练，收集完整日志**
2. **分析日志，确定失败的具体原因**
3. **根据原因选择对应的解决方案**

---
调试日期: 2026-07-03
