# ORCA Multi-Robot Navigation

基于 Nav2 和 ORCA（Optimal Reciprocal Collision Avoidance）算法的多机器人导航包。

## 功能特性

- ✅ **ORCA 避碰算法**：多机器人局部避碰，无需通信即可协调
- ✅ **Nav2 集成**：可选使用 Nav2 进行全局路径规划
- ✅ **分布式架构**：每个机器人独立订阅目标和里程计
- ✅ **实时控制**：20Hz 控制频率，平滑速度输出

## 包结构

```
start_orca_nav/
├── launch/
│   └── start_orca_nav.launch.py    # 启动文件
├── start_orca_nav/
│   ├── orca_algorithm.py            # ORCA 核心算法实现
│   └── orca_nav_node.py             # 主导航节点
├── package.xml
└── setup.py
```

## 安装与编译

```bash
cd /home/wj/work/multi-robot-exploration-rl
colcon build --packages-select start_orca_nav
source install/setup.bash
```

## 使用方法

### 基本启动

```bash
ros2 launch start_orca_nav start_orca_nav.launch.py robot_number:=4
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `robot_number` | 4 | 机器人数量 |
| `robot_radius` | 0.35 | 机器人半径（米） |
| `max_linear_speed` | 0.22 | 最大线速度（米/秒） |
| `max_angular_speed` | 2.0 | 最大角速度（弧度/秒） |
| `neighbor_distance` | 5.0 | 邻居检测距离（米） |
| `time_horizon` | 2.0 | ORCA 时间范围（秒） |
| `use_nav2` | true | 是否使用 Nav2 全局规划 |

### 话题接口

**订阅的话题**（每个机器人）：
- `/{robot_name}/odom` (nav_msgs/Odometry)：里程计
- `/{robot_name}/goal_pose` (geometry_msgs/PoseStamped)：目标位置

**发布的话题**（每个机器人）：
- `/{robot_name}/cmd_vel` (geometry_msgs/Twist)：速度命令

### 发送目标示例

```bash
# 给 robot0 发送目标
ros2 topic pub --once /robot0/goal_pose geometry_msgs/PoseStamped '{
  header: {frame_id: "map"},
  pose: {
    position: {x: 5.0, y: 3.0, z: 0.0},
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}
  }
}'
```

## ORCA 算法原理

ORCA（Optimal Reciprocal Collision Avoidance）是一种分布式多智能体避碰算法：

1. **速度障碍**：为每个邻居机器人计算速度障碍锥
2. **共同责任**：假设所有机器人都会避让，分担避碰责任
3. **线性规划**：求解满足所有约束的最优速度

**优势**：
- 无需通信协调
- 计算效率高
- 保证无碰撞（理论上）
- 平滑轨迹

## 与强化学习方案对比

| 特性 | ORCA | 强化学习 |
|------|------|----------|
| 训练时间 | 无需训练 ✅ | 需要长时间训练 ❌ |
| 避碰保证 | 理论保证 ✅ | 不保证 ❌ |
| 适应性 | 固定算法 ⚠️ | 可学习复杂策略 ✅ |
| 计算开销 | 低 ✅ | 推理快，训练慢 ⚠️ |
| 可解释性 | 高 ✅ | 黑盒 ❌ |

## 调试技巧

1. **查看日志**：
   ```bash
   ros2 run start_orca_nav orca_nav_node --ros-args --log-level debug
   ```

2. **检查机器人状态**：
   ```bash
   ros2 topic echo /robot0/odom
   ```

3. **可视化（RViz）**：
   - 添加 `/robot*/odom` → Odometry
   - 添加 `/robot*/cmd_vel` → Twist

## 常见问题

**Q: 机器人不动？**
- 检查是否收到里程计数据
- 确认已发送目标位置
- 查看是否有其他节点在发布 cmd_vel

**Q: 机器人会碰撞？**
- 增大 `robot_radius` 参数
- 减小 `time_horizon`（更激进避让）
- 检查 `neighbor_distance` 是否足够大

**Q: 运动不平滑？**
- 增大 `time_horizon`
- 调整 `max_angular_speed`
- 可以在节点中添加速度滤波器

## 未来改进

- [ ] 添加静态障碍物避障（ORCA-Obstacles）
- [ ] 集成动态窗口法（DWA）进行运动学约束
- [ ] 添加编队控制模式
- [ ] 优化线性规划求解器
- [ ] 添加 RViz 可视化插件

## 许可证

Apache-2.0

## 参考文献

- Van den Berg, J., et al. "Reciprocal n-body collision avoidance." ISRR 2011.
