# start_rl_environment_tb3 - TurtleBot3强化学习环境包

## 📋 简介

这个包是 `start_rl_environment` 的TurtleBot3版本，使用TurtleBot3 Burger模型替代自定义机器人模型。

### ✅ TurtleBot3的优势

相比自定义机器人模型，TurtleBot3具有以下优点：

1. **更低的重心** - 底盘更低矮，不易倾斜
2. **标准化的传感器配置** - 激光雷达安装位置经过优化，离地高度合理
3. **更稳定的物理模型** - 经过广泛测试和验证的Gazebo模型
4. **减少误报碰撞** - 由于更稳定，激光雷达不易检测到地面，减少假阳性碰撞检测

### 🔄 与原包的区别

| 特性 | start_rl_environment | start_rl_environment_tb3 |
|-----|---------------------|-------------------------|
| 机器人模型 | 自定义box机器人 | TurtleBot3 Burger |
| 机器人名称 | my_bot0, my_bot1... | tb3_0, tb3_1... |
| 初始高度 | 0.1m | 0.01m |
| 地图/世界 | ✅ 完全相同 | ✅ 完全相同 |
| 激光雷达 | 自定义配置 | TurtleBot3标准LDS-01 |
| 稳定性 | ⚠️ 易倾斜 | ✅ 高稳定性 |

## 🚀 使用方法

### 0. 环境准备（重要！）

**在conda的ros2环境中启动：**

```bash
# 激活conda的ROS2环境
conda activate ros2

# 设置环境变量
export TURTLEBOT3_MODEL=burger
```

> ⚠️ **注意**: 必须在包含ROS2的conda环境中运行，否则无法启动。

### 0.1 解决Gazebo GUI加载缓慢问题

**首次启动Gazebo会卡在加载界面**，这是因为需要下载TurtleBot3模型资源。有以下解决方案：

**方案1：无GUI模式启动（推荐用于训练）**
```bash
# 只启动Gazebo服务器，不启动GUI（训练时推荐）
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
gzserver --verbose /path/to/world/file.world &
```

**方案2：等待首次加载完成**
- 首次启动需要3-5分钟下载模型
- 下载完成后会自动显示，后续启动会很快
- 从日志看到 "Successfully spawned entity" 说明机器人已经加载成功
- 即使GUI卡住，后台已经在正常工作

**方案3：预先下载模型**
```bash
# 确保模型路径正确
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models

# 手动下载模型（如果需要）
mkdir -p ~/.gazebo/models
cd ~/.gazebo/models
# TurtleBot3模型已经包含在安装的包中
```

**验证Gazebo是否正常工作（即使GUI卡住）：**
```bash
# 在另一个终端检查话题
ros2 topic list | grep tb3

# 检查机器人是否发布激光数据
ros2 topic echo /tb3_0/scan --once

# 发送测试速度命令
ros2 topic pub /tb3_0/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.1}, angular: {z: 0.0}}" --once
```

### 1. 启动仿真环境

**方式A：带GUI（可视化调试）**
```bash
# 启动3个TurtleBot3机器人，使用地图1
ros2 launch start_rl_environment_tb3 main.launch.py map_number:=1 robot_number:=3

# 启动4个TurtleBot3机器人，使用地图3 (corridor_swap)
ros2 launch start_rl_environment_tb3 main.launch.py map_number:=3 robot_number:=4
```

> ⚠️ **注意**: 首次启动GUI会卡在"Preparing your world"界面3-5分钟，这是正常的模型加载过程。后台已经在工作，可以直接开始训练。

**方式B：无GUI模式（推荐用于训练）⚡**
```bash
# 无GUI启动，速度更快，不会卡住
ros2 launch start_rl_environment_tb3 main_headless.launch.py map_number:=3 robot_number:=4
```

> ✅ **推荐**: 强化学习训练时使用无GUI模式，启动快且资源占用少。

### 2. 与强化学习训练集成

修改你的训练脚本中的包名和机器人名称：

**原来的配置 (使用自定义机器人):**
```python
# 在 marl_training/independent_env.py 或 sb3_training 中
namespace = f"my_bot{robot_id}"
```

**新的配置 (使用TurtleBot3):**
```python
# 在 marl_training/independent_env.py 或 sb3_training 中
namespace = f"tb3_{robot_id}"
```

### 3. 可用的地图

| map_number | 地图名称 | 描述 |
|-----------|---------|------|
| 1 | map1 | 基础地图 |
| 2 | map2 | 第二个地图 |
| 3 | corridor_swap | 走廊交换场景 |
| 4 | intersection | 十字路口场景 |
| 5 | warehouse_aisles | 仓库通道场景 |

## 📦 依赖项

```xml
<depend>turtlebot3_gazebo</depend>
<depend>turtlebot3_description</depend>
<depend>gazebo_ros_pkgs</depend>
<depend>simple_launch</depend>
```

确保已安装TurtleBot3包：

```bash
sudo apt install ros-humble-turtlebot3*
```

## 🔧 配置说明

### 机器人命名空间

- TurtleBot3机器人使用命名空间: `tb3_0`, `tb3_1`, `tb3_2`, ...
- 话题示例:
  - 激光: `/tb3_0/scan`
  - 速度命令: `/tb3_0/cmd_vel`
  - 里程计: `/tb3_0/odom`

### TF树结构

```
map
 └─ tb3_0/odom (static)
     └─ tb3_0/base_footprint
         └─ tb3_0/base_link
             ├─ tb3_0/wheel_left_link
             ├─ tb3_0/wheel_right_link
             ├─ tb3_0/caster_back_link
             └─ tb3_0/base_scan (LiDAR)
```

## 🎯 强化学习参数建议

使用TurtleBot3后，建议调整以下参数：

### 1. 角速度限制
```python
# TurtleBot3 更稳定，可以使用稍高的角速度
max_angular_vel = 1.5  # rad/s (原来是0.3)
```

### 2. 碰撞检测阈值
```python
# 可以使用更严格的阈值，因为误报更少
collision_threshold = 0.20  # m (保持不变)
```

### 3. 激光过滤
```python
# TurtleBot3的激光高度合理，可以减少过滤
valid_scan = scan_ranges > 0.10  # m (原来是0.15)
```

## 📝 Launch文件说明

### main.launch.py
主启动文件，同时启动Gazebo世界、机器人和地图服务器。

**参数:**
- `map_number`: 地图编号 (默认: 1)
- `robot_number`: 机器人数量 (默认: 3)
- `spawn_mode`: 生成模式 (默认: 'fixed')
- `seed`: 随机种子 (默认: 0)
- `min_separation`: 最小间隔距离 (默认: 0.8)

### start_world.launch.py
只启动Gazebo世界环境。

### start_robots.launch.py
只启动机器人（需要Gazebo已运行）。

### spawn_robots.launch.py
生成单个机器人的底层launch文件。

## 🐛 已知问题

1. **首次加载慢** - TurtleBot3模型首次加载需要下载Gazebo资源，可能较慢
2. **传感器频率** - TurtleBot3的LDS-01激光雷达频率为5Hz，低于某些自定义配置

## 🔗 相关资源

- [TurtleBot3官方文档](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)
- [Gazebo仿真教程](http://gazebosim.org/tutorials)
- [ROS2 Humble文档](https://docs.ros.org/en/humble/)

## 📄 许可证

Apache-2.0

## 👤 维护者

Theo Moore-Calters <nohacks2701@gmail.com>
