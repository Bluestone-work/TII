# 目标到达停止功能

## 功能说明

机器人到达目标点范围内（默认0.3米）会自动停止，不再继续移动。

## 实现逻辑

### 1. 状态跟踪

```python
self.robot_goal_reached = {}  # 跟踪每个机器人是否已到达目标
```

每个机器人维护一个到达状态标志。

### 2. 距离检测

```python
distance_to_goal = np.linalg.norm(goal - position)

if distance_to_goal < self.goal_tolerance:
    # 到达目标
    self.robot_goal_reached[robot_name] = True
    # 发布停止命令
    stop_cmd = Twist()  # 速度为0
    self.cmd_vel_publishers[robot_name].publish(stop_cmd)
```

### 3. 工作流程

```
┌─────────────────┐
│ 控制循环 (20Hz) │
└────────┬────────┘
         │
         v
  ┌──────────────────────┐
  │ 检查到达状态         │
  ├──────────────────────┤
  │ distance < tolerance?│
  └──────┬───────┬───────┘
         │ Yes   │ No
         v       v
   ┌─────────┐  ┌──────────┐
   │ 停止    │  │ 继续导航 │
   │ 发布v=0 │  │ 计算速度 │
   └─────────┘  └──────────┘
```

## 参数配置

### 默认值

```bash
GOAL_TOLERANCE=0.3  # 米
```

### 调整阈值

**增加阈值（停止更早）**：
```bash
./start_orca_nav.sh -m 1 -r 2 --goal-tolerance 0.5
```

**减小阈值（更精确到达）**：
```bash
./start_orca_nav.sh -m 1 -r 2 --goal-tolerance 0.15
```

### 建议值

| 场景 | 推荐阈值 | 原因 |
|------|---------|------|
| 快速导航 | 0.5m | 避免在目标附近徘徊 |
| 标准使用 | 0.3m | 平衡精度和效率 |
| 精确定位 | 0.15m | 高精度任务 |
| 狭窄空间 | 0.2m | 避免碰撞 |

## 行为特征

### 到达前

```
[INFO] robot0: neighbors detected, computing velocity
[INFO] robot0: distance to goal: 2.35m
[INFO] robot0: moving towards goal
```

### 到达时刻

```
[INFO] robot0 reached goal! Distance: 0.28m
```

**只显示一次**，避免日志刷屏。

### 到达后

- ✅ 持续发布 `Twist(linear.x=0, angular.z=0)`
- ✅ 不再计算DWA或ORCA
- ✅ 节省计算资源
- ✅ 机器人保持静止

## 新目标处理

### 接收新目标时

```python
def goal_callback(self, msg: PoseStamped, robot_name: str):
    self.robot_goals[robot_name] = new_goal
    self.robot_goal_reached[robot_name] = False  # 重置状态
```

**自动重置到达状态**，机器人开始向新目标移动。

## 日志查看

### 查看到达事件

```bash
tail -f orca_logs/navigation_*.log | grep "reached goal"
```

### 查看所有状态

```bash
ros2 topic echo /my_bot0/cmd_vel
```

到达后会看到：
```
linear:
  x: 0.0
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0
```

## 调试技巧

### 问题：机器人在目标附近抖动

**原因**：阈值太小，机器人来回穿过边界

**解决**：增加阈值
```bash
--goal-tolerance 0.4
```

### 问题：机器人停在离目标较远的地方

**原因**：阈值太大

**解决**：减小阈值
```bash
--goal-tolerance 0.2
```

### 问题：机器人停止后又开始移动

**原因**：里程计漂移导致位置估计超出阈值

**解决**：
1. 增加阈值：`--goal-tolerance 0.4`
2. 检查里程计质量
3. 考虑使用AMCL定位

## 与velocity_to_twist的关系

### 双重保护

```python
# 第1层：control_loop中
if distance_to_goal < self.goal_tolerance:
    stop_cmd = Twist()
    publish(stop_cmd)
    return  # 不进入后续计算

# 第2层：velocity_to_twist中（备用）
if distance_to_goal < 0.2:
    return Twist()  # 返回零速度
```

**优势**：即使第一层失效，第二层仍能保证停止。

## 性能影响

| 指标 | 到达前 | 到达后 |
|------|--------|--------|
| CPU使用 | 100% | ~5% |
| DWA计算 | ✅ | ❌ |
| ORCA计算 | ✅ | ❌ |
| 速度发布 | 变化 | 固定(0) |

**到达后CPU大幅降低**，只需发布停止命令。

## 扩展功能

### 未来可添加

1. **到达回调**
   ```python
   def on_goal_reached(self, robot_name):
       # 触发后续任务
       pass
   ```

2. **停留时间**
   ```python
   if distance < tolerance and time > dwell_time:
       # 确认到达
   ```

3. **姿态对齐**
   ```python
   if distance < tolerance:
       # 调整朝向
       align_orientation()
   ```

4. **多目标序列**
   ```python
   if reached(goal[i]):
       current_goal = goal[i+1]
   ```

## 总结

✅ **简单有效**：基于距离判断，逻辑清晰  
✅ **可配置**：通过参数调整阈值  
✅ **状态跟踪**：避免重复日志输出  
✅ **自动重置**：接收新目标时自动恢复  
✅ **节省资源**：到达后停止计算

---

**使用示例**：
```bash
# 标准使用（0.3米阈值）
./start_orca_nav.sh -m 1 -r 2

# 快速导航（0.5米阈值）
./start_orca_nav.sh -m 1 -r 2 --goal-tolerance 0.5

# 精确定位（0.15米阈值）
./start_orca_nav.sh -m 1 -r 2 --goal-tolerance 0.15
```
