# ORCA Navigation - 修复版本说明

## 🔧 问题修复

### 修复的问题
1. ✅ **模块导入错误**：将`from start_orca_nav.orca`改为`from start_orca_nav.orca_algorithm`
2. ✅ **Gazebo未启动**：修改启动流程，先启动Gazebo再启动ORCA节点
3. ✅ **TF变换错误**：通过先启动完整的RL环境解决

### 新的启动流程

原来的问题是试图独立启动Nav2栈，但是：
- Nav2需要TF变换（base_link → map）
- TF变换来自Gazebo中的机器人
- 所以必须先启动Gazebo

**新方案**：
1. 先启动`start_rl_environment.launch.py`（包含Gazebo + 机器人 + Nav2栈）
2. 然后启动`orca_nav_node_nav2`（ORCA控制节点）

## 🚀 使用方法

### 方式1：一键启动（推荐）

```bash
./start_orca_nav2.sh -m corridor_swap -r 4
```

这个脚本会：
1. 启动Gazebo和RL环境（自动包含Nav2）
2. 等待15秒让Gazebo完全启动
3. 启动ORCA导航节点

### 方式2：分步启动

如果Gazebo已经在运行：

```bash
# 假设Gazebo已启动
./start_orca_nav2_simple.sh -m corridor_swap -r 4
```

### 方式3：完全手动

```bash
# Terminal 1: 启动Gazebo和RL环境
ros2 launch start_rl_environment start_rl_environment.launch.py \
    map_name:=corridor_swap robot_number:=4

# Terminal 2: 启动ORCA节点
ros2 launch start_orca_nav orca_nav2_simple.launch.py \
    robot_number:=4 use_rviz:=true
```

## 🏗️ 架构说明

```
start_rl_environment.launch.py
├─ Gazebo (世界、机器人模型)
├─ TF Publishers (机器人位姿)
└─ Nav2 Stack (每个机器人)
   ├─ map_server
   ├─ planner_server
   └─ (其他Nav2节点)

orca_nav_node_nav2
├─ 调用Nav2的ComputePathToPose获取全局路径
├─ ORCA计算避让其他机器人的速度
├─ DWA生成考虑障碍物的最终命令
└─ 发布cmd_vel到机器人
```

## 📁 文件清单

### 核心文件
- `orca_nav_node_nav2.py` - ORCA导航节点（修复了导入）
- `orca_nav2_simple.launch.py` - 简化的launch（只启动ORCA节点）
- `start_orca_nav2.sh` - 完整启动脚本
- `start_orca_nav2_simple.sh` - 简化启动脚本（需要Gazebo已运行）

### 废弃文件（暂时不用）
- `nav2_multi_robot.launch.py` - 独立Nav2栈启动（有TF问题）
- `orca_nav2.launch.py` - 完整launch（有TF问题）

## ✅ 测试步骤

1. **编译**
```bash
colcon build --packages-select start_orca_nav
source install/setup.bash
```

2. **启动系统**
```bash
./start_orca_nav2.sh -m corridor_swap -r 4
```

3. **等待启动完成**
- Gazebo窗口出现
- 机器人加载完成
- ORCA节点输出"initialized"

4. **发送目标**
```bash
ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, 
    pose: {position: {x: 5.0, y: 3.0}, orientation: {w: 1.0}}}'
```

5. **观察行为**
- 节点请求Nav2路径
- 打印收到的路径信息
- 机器人开始移动

## 🐛 故障排除

### 问题：orca_nav_node_nav2启动失败，找不到模块
**解决**：已修复，重新编译即可
```bash
colcon build --packages-select start_orca_nav
```

### 问题：TF变换错误
**原因**：Gazebo没有启动，没有机器人发布TF
**解决**：使用`start_orca_nav2.sh`，它会先启动Gazebo

### 问题：Nav2 planner服务不可用
**原因**：`start_rl_environment`的Nav2栈可能没启动
**检查**：
```bash
ros2 service list | grep compute_path
# 应该看到 /my_bot0/compute_path_to_pose 等
```

### 问题：机器人不动
**检查顺序**：
1. Gazebo是否在运行？`pgrep gzserver`
2. ORCA节点是否收到odom？看日志
3. 是否发送了goal？`ros2 topic echo /robot0/goal_pose`
4. Nav2是否返回了路径？看ORCA节点日志

## 📊 对比原版本

| 特性 | 原版 (start_orca_nav.sh) | 新版 (start_orca_nav2.sh) |
|------|--------------------------|---------------------------|
| 全局规划 | 自己的Theta* | Nav2 ComputePathToPose |
| Gazebo启动 | ✅ 包含 | ✅ 包含 |
| 地图支持 | 手动边界 | PGM/YAML文件 |
| ORCA避碰 | ✅ | ✅ |
| DWA控制 | ✅ | ✅ |
| 依赖 | 无 | 需要Nav2 |

## 🎯 当前状态

✅ **可以使用** - 修复了所有导入和启动问题
✅ **架构正确** - Gazebo → Nav2 → ORCA → DWA
✅ **文档完整** - 使用说明清晰

## 💡 使用建议

目前最稳定的方式是：
```bash
./start_orca_nav2.sh -m corridor_swap -r 4
```

如果想要更快的测试迭代（Gazebo已经在运行）：
```bash
./start_orca_nav2_simple.sh -m corridor_swap -r 4
```

## 📝 注意事项

1. **首次使用需要安装Nav2**（如果还没装）：
   ```bash
   sudo apt install ros-humble-nav2-bringup
   ```

2. **启动需要时间**：Gazebo启动大约需要15秒

3. **检查Nav2服务**：如果路径规划失败，检查Nav2服务是否可用

4. **查看日志**：ORCA节点会输出详细的路径请求和接收信息
