# 🚀 快速开始 - 验证数据同步

## ⚡ 快速诊断TF问题（2分钟）

```bash
cd ~/work/multi-robot-exploration-rl

# 运行TF监控器（假设有3个机器人）
./scripts/run_sync_validation.sh 3

# 选择: 1 (TF树监控器 - 单次检查)
```

**期望输出**:
```
✅ map <- my_bot0/odom: 位移=(0.123, 0.456, 0.000)
✅ map <- my_bot1/odom: 位移=(0.234, 0.567, 0.000)
✅ map <- my_bot2/odom: 位移=(0.345, 0.678, 0.000)
```

**如果看到❌**: 说明TF有问题，参考下文"TF问题排查"。

---

## 🎯 验证观测与场景对应（5分钟）

### 步骤1: 启动模拟器

```bash
# 终端1: 启动Gazebo
cd ~/work/multi-robot-exploration-rl
ros2 launch your_launch_file simulation.launch.py
```

### 步骤2: 启动验证工具

```bash
# 终端2: 启动全套验证工具
cd ~/work/multi-robot-exploration-rl
./scripts/run_sync_validation.sh 3

# 选择: 4 (全套工具)
```

这会自动启动：
- TF监控器（新终端窗口）
- 数据同步可视化器（后台）
- RViz（可视化界面）

### 步骤3: 配置RViz

在RViz中:
1. 点击 **Add** → **By topic**
2. 添加以下话题（每个机器人一个）：
   - `/robot0/sync_visualization`
   - `/robot1/sync_visualization`
   - `/robot2/sync_visualization`
3. Fixed Frame 设置为 `map`

### 步骤4: 运行训练并观察

```bash
# 终端3: 启动训练
cd ~/work/multi-robot-exploration-rl
python3 src/start_reinforcement_learning/start_reinforcement_learning/matd3_main.py
```

**在RViz中观察**:
- 🟢 **绿色圆环**: 数据新鲜，一切正常
- 🟡 **黄色圆环**: 数据稍旧，可能有轻微延迟
- 🔴 **红色圆环**: 数据过时，需要检查！

**在终端输出中观察**（每50步）:
```
================================================================================
📊 数据同步报告 (Episode 1, Step 50)
================================================================================
📍 里程计数据:
  Robot0: ✅ 时间=123.455s, 年龄=12.3ms, 位置=(1.23, 4.56)
  
📡 激光雷达数据:
  Robot0: ✅ 时间=123.456s, 年龄=5.2ms, 最近障碍=0.85m
  
🎯 目标位置:
  Robot0: 目标=(5.00, 8.00), 距离=4.23m
================================================================================
```

---

## 🔍 TF问题排查

### 问题1: "Invalid frame ID 'odom'"

**原因**: Frame名称不匹配

**排查**:
```bash
# 查看所有可用的frames
ros2 run tf2_ros tf2_echo map my_bot0/odom

# 列出所有TF话题
ros2 topic list | grep tf
```

**解决**: 
修改 [logic.py](../src/start_reinforcement_learning/start_reinforcement_learning/env_logic/logic.py) 第1156行:

```python
# 旧代码
if odom_frame in ('odom', 'base_odom'):
    odom_frame = f"my_bot{i}/odom"

# 改为实际frame名（从tf2_echo输出获取）
if odom_frame in ('odom', 'base_odom'):
    odom_frame = f"robot{i}/odom"  # 或其他实际名称
```

### 问题2: TF监控器显示所有机器人都❌

**原因**: 
1. Gazebo未运行
2. 机器人未正确spawn
3. 状态发布器未配置

**排查**:
```bash
# 检查机器人话题
ros2 topic list | grep my_bot0

# 应该看到:
# /my_bot0/odom
# /my_bot0/scan
# /my_bot0/cmd_vel

# 检查TF发布
ros2 topic hz /tf
ros2 topic hz /tf_static
```

**解决**: 确保launch文件中包含状态发布器。

### 问题3: 数据年龄总是>500ms

**原因**: 
1. 仿真时间未启动
2. 传感器频率太低

**排查**:
```bash
# 检查仿真时间
ros2 param get /use_sim_time

# 检查话题频率
ros2 topic hz /my_bot0/odom
ros2 topic hz /my_bot0/scan
```

**解决**: 在launch文件中设置 `use_sim_time:=true`

---

## 📊 验证奖励与场景对应

### 方法1: 观察同步报告

在训练过程中，每50步会打印报告。**对比以下内容**:

| 场景 | 观测 | 奖励 | 是否对应 |
|------|------|------|---------|
| 机器人靠近障碍物 | 最近障碍=0.30m | r_obstacle=-1.4 | ✅ 对应（障碍近→大惩罚） |
| 机器人远离障碍 | 最近障碍=2.50m | r_obstacle=0.0 | ✅ 对应（障碍远→无惩罚） |
| 机器人接近目标 | 距离=1.23m | r_goal=0.15 | ✅ 对应（接近→正奖励） |
| 机器人远离目标 | 距离=8.45m | r_goal=-0.08 | ✅ 对应（远离→负奖励） |

### 方法2: 手动验证

在训练脚本中添加：

```python
# 在 step 后
obs, reward, done, truncated, info = env.step(action)

# 每N步打印一次
if env.step_counter % 50 == 0:
    env._print_sync_report()  # 打印详细报告
    
    # 手动验证
    for i in range(env.number_of_robots):
        print(f"\nRobot {i} 验证:")
        print(f"  观测中的激光雷达最小值: {obs[f'robot{i}'][:38].min():.2f}")
        print(f"  奖励中的障碍惩罚: {info['reward_components'][i]['r_obstacle']:.2f}")
        print(f"  实际机器人位置: ({env.current_pose_x[i]:.2f}, {env.current_pose_y[i]:.2f})")
```

---

## ✅ 成功标志

当一切正常时，你会看到：

1. **TF监控器**: 所有机器人显示 ✅
2. **RViz**: 所有圆环都是 🟢 绿色
3. **同步报告**: 所有数据年龄 < 200ms
4. **训练稳定**: 无"LaserScan missing"或"Odom stale"警告

---

## 🛑 关闭验证系统（生产环境）

训练稳定后，为了性能可以关闭验证：

```python
# 在创建环境时
env = Env(
    number_of_robots=3,
    map_number=1,
)

# 关闭验证
env.enable_sync_validation = False
env.debug_obs_warnings = False
```

---

## 📞 需要帮助？

1. 查看完整文档: [DATA_SYNC_VALIDATION_GUIDE.md](DATA_SYNC_VALIDATION_GUIDE.md)
2. 查看交互日志: `~/work/multi-robot-exploration-rl/interaction_logs/*.jsonl`
3. 检查终端输出中的警告信息

---

## 🎉 总结

通过这套验证系统，你可以：
- ✅ 确认TF变换正常工作
- ✅ 验证观测数据新鲜且准确
- ✅ 确保奖励与场景实时对应
- ✅ 可视化验证所有数据同步

**现在可以放心训练了！** 🚀
