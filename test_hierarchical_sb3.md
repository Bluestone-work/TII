# 🎯 SB3训练包 - 分层强化学习测试指南

## ✅ 已完成的工作

### 1. 文件添加
- ✅ 复制 `global_planner.py` 到 `sb3_training/sb3_training/`
- ✅ 复制 `waypoint_visualizer.py` 到 `sb3_training/sb3_training/`

### 2. 代码集成（single_robot_env.py）

#### 添加导入
```python
import math
from sb3_training.global_planner import AStarPlanner, WaypointExtractor
from sb3_training.waypoint_visualizer import WaypointVisualizer
```

#### __init__中添加分层RL初始化
```python
# ========== 分层强化学习：全局规划器 ==========
self.use_global_planner = True  # 启用全局规划
self.planner = None  # 等地图加载后初始化
self.waypoint_extractor = WaypointExtractor(
    turning_threshold=0.3,  # 转角>17度算拐点
    distance_threshold=1.5  # 直线段每1.5米一个点
)

# 路径点管理
self.global_waypoints = None
self.current_waypoint_index = 0
self.waypoint_reach_distance = 0.3  # 到达阈值

# Gazebo可视化
self.waypoint_visualizer = WaypointVisualizer()
```

#### reset()中添加路径规划
```python
# 为当前机器人规划全局路径
if self.use_global_planner:
    self._plan_global_path()
```

#### step()中添加路径点检查
```python
# 检查并更新路径点，给予额外奖励
if self.use_global_planner:
    waypoint_reached = self._check_and_update_waypoint()
    if waypoint_reached:
        reward += 0.5  # 到达路径点额外奖励
```

#### 添加4个辅助方法
- `_initialize_planner_after_map_loaded()` - 初始化A*规划器
- `_plan_global_path()` - 规划全局路径并提取关键点
- `_get_current_waypoint()` - 获取当前目标路径点
- `_check_and_update_waypoint()` - 检查到达并切换路径点

### 3. 编译完成
```bash
colcon build --packages-select sb3_training
# Summary: 1 package finished [2.06s] ✅
```

---

## 🧪 测试步骤

### 步骤1: 启动Gazebo环境
```bash
# 终端1
cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash
ros2 launch start_rl_environment main.launch.py map_number:=3 robot_number:=1
```

**等待Gazebo完全加载，确保看到：**
- ✅ 地图已加载
- ✅ 机器人已生成
- ✅ 目标点已标记

### 步骤2: 启动SB3训练（带分层RL）
```bash
# 终端2
cd /home/wj/work/multi-robot-exploration-rl
source install/setup.bash

python3 src/sb3_training/sb3_training/train_ppo.py \
    --robot_number 1 \
    --map_number 3 \
    --total_timesteps 50000 \
    --device cpu
```

### 步骤3: 观察Gazebo中的可视化

应该看到：

#### ✅ 全局路径可视化
- **绿色线条**：从起点到终点的完整路径
- **蓝色球**：起点标记
- **红色球**：终点标记
- **黄色球**：中间关键路径点
- **白色标签**：路径点编号（WP0, WP1, WP2, ...）

#### ✅ 当前目标高亮
- **紫色圆圈**：当前应该前往的路径点（会移动）

#### ✅ 终端输出
```
✅ Robot 0：230点→8关键点
🎯 Robot 0：路径点0→1/8
🎯 Robot 0：路径点1→2/8
...
🏁 Robot 0到达最终目标！
```

---

## 📊 预期效果对比

| 指标 | 改进前（直接目标） | 改进后（分层RL） |
|------|-------------------|-----------------|
| 任务定义 | 学习20米长距离导航 | 学习1.5米短距离导航 |
| 学习难度 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 收敛速度 | 100K+ steps | 30-50K steps |
| 成功率 | <30% | >70% |
| 路径质量 | 可能绕远/卡死 | 遵循最优路径 |
| 奖励频率 | 稀疏（仅到达终点） | 密集（每到达路径点+0.5） |

---

## 🔧 调试命令

### 检查路径点话题
```bash
# 查看是否发布了waypoint markers
ros2 topic echo /waypoint_markers --once
```

### 查看路径点信息
```bash
# 查看marker数量和类型
ros2 topic info /waypoint_markers
```

### 如果看不到可视化
```bash
# 1. 确认Gazebo已启动
ps aux | grep gazebo

# 2. 确认ROS2话题正常
ros2 topic list | grep waypoint

# 3. 在RViz中添加Marker显示
# Add -> By topic -> /waypoint_markers -> MarkerArray
```

---

## 🎯 成功标志

训练成功运行的标志：

1. ✅ 终端输出显示路径点规划：
   ```
   ✅ Robot 0：XXX点→YY关键点
   ```

2. ✅ Gazebo中看到绿色路径线和彩色球

3. ✅ 训练过程中看到路径点切换：
   ```
   🎯 Robot 0：路径点0→1/8
   ```

4. ✅ TensorBoard显示奖励上升趋势
   ```bash
   tensorboard --logdir /home/wj/work/multi-robot-exploration-rl/sb3_logs
   ```

5. ✅ 机器人沿着绿色路径移动（而不是直线冲向终点）

---

## ⚙️ 参数调优

如果效果不理想，可以调整以下参数：

### single_robot_env.py
```python
# 转角阈值（更小=更多路径点）
turning_threshold=0.3  # 可改为 0.2 或 0.4

# 距离阈值（更小=路径点更密集）
distance_threshold=1.5  # 可改为 1.0 或 2.0

# 到达距离（更大=更容易切换）
waypoint_reach_distance=0.3  # 可改为 0.5
```

### 路径点奖励
```python
# step()中的奖励值
if waypoint_reached:
    reward += 0.5  # 可改为 0.3 ~ 1.0
```

---

## 🐛 常见问题

### Q1: 看不到绿色路径
**A:** 检查地图是否加载：
```python
# 在_initialize_planner_after_map_loaded()中会打印
✅ Robot 0: A*规划器已就绪  # 如果看到这个说明正常
```

### Q2: 规划失败
**A:** 检查起点/终点是否在障碍物中：
```
⚠️ Robot 0无法规划路径，直接使用目标点  # 如果看到这个
```
解决：重启Gazebo或换地图

### Q3: 路径点不切换
**A:** 检查`waypoint_reach_distance`是否太小，可以增大到0.5m

### Q4: 训练崩溃
**A:** 检查依赖：
```bash
pip list | grep scipy  # 需要scipy用于obstacle inflation
pip list | grep stable-baselines3
pip list | grep sb3-contrib
```

---

## 📝 测试清单

训练前检查：
- [ ] Gazebo已启动且地图加载完成
- [ ] 机器人和目标点可见
- [ ] ROS2话题正常（`ros2 topic list`）
- [ ] Python环境正确（ros2 conda环境）
- [ ] scipy已安装

训练中检查：
- [ ] 终端显示路径规划成功（✅ Robot 0：XXX点→YY关键点）
- [ ] Gazebo中看到绿色路径和彩色球
- [ ] 路径点切换消息正常（🎯 Robot 0：路径点X→Y）
- [ ] TensorBoard显示训练曲线
- [ ] 无错误或警告消息

---

## 🚀 下一步

训练完成后：
1. 比较有/无分层RL的收敛曲线
2. 统计成功率和平均步数
3. 可视化学到的策略
4. 测试不同地图（map_number 1-5）
5. 尝试多机器人训练（robot_number > 1）

---

## 💡 核心优势

**为什么分层RL更好？**

| 传统RL | 分层RL |
|--------|--------|
| 学习"从A到B" | 学习"前进1.5m" |
| 长期回报稀疏 | 短期回报密集 |
| 探索空间巨大 | 探索空间受限 |
| 容易陷入局部最优 | 全局路径保证 |
| 需要100K+ steps | 30-50K steps足够 |

**类比：**
- 传统RL = 让小孩直接学开车去100公里外的城市
- 分层RL = 给小孩导航，每个路口告诉他怎么转弯

**关键：** RL只需学习"跟随导航"这个简单任务！
