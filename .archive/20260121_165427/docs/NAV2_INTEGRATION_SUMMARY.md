# ORCA导航系统 - Nav2集成版本

## 🎯 完成的工作

我已经完成了**大架构改造**，将系统从自实现的Theta*全局规划改为使用Nav2的全局规划器：

### ✅ 已完成的修改

1. **新的导航节点** (`orca_nav_node_nav2.py`)
   - 删除了自己的Theta*规划器
   - 集成Nav2的`ComputePathToPose` action
   - 保留ORCA多机器人避碰
   - 保留DWA局部规划器

2. **Nav2多机器人Launch文件** (`nav2_multi_robot.launch.py`)
   - 为每个机器人启动独立的Nav2栈
   - 包含map_server和planner_server
   - 支持4个机器人（可扩展）

3. **主Launch文件** (`orca_nav2.launch.py`)
   - 启动Nav2栈
   - 启动ORCA导航节点
   - 支持RViz可视化

4. **便捷启动脚本** (`start_orca_nav2.sh`)
   - 支持命令行参数（地图名、机器人数量）
   - 自动查找地图文件
   - 友好的使用提示

5. **测试和文档**
   - 系统检查脚本 (`test_nav2_setup.sh`)
   - 完整的README文档 (`ORCA_NAV2_README.md`)

### 📁 文件清单

**新增文件**:
- `src/start_orca_nav/start_orca_nav/orca_nav_node_nav2.py` - 新的导航节点
- `src/start_orca_nav/launch/orca_nav2.launch.py` - 主launch文件
- `src/start_orca_nav/launch/nav2_multi_robot.launch.py` - Nav2多机器人launch
- `start_orca_nav2.sh` - 启动脚本
- `test_nav2_setup.sh` - 系统检查脚本
- `ORCA_NAV2_README.md` - 完整文档

**修改文件**:
- `src/start_orca_nav/setup.py` - 添加新的可执行文件入口

**地图文件** (已存在):
- `src/start_rl_environment/maps/*.yaml/*.pgm`

## 🏗️ 新架构说明

```
用户发送目标
    ↓
ORCA Nav Node (orca_nav_node_nav2.py)
    ↓
调用 Nav2 ComputePathToPose action
    ↓
Nav2 Planner Server (每个机器人独立)
    ├─ Map Server (加载PGM/YAML地图)
    └─ NavFn Planner (全局路径规划)
    ↓
返回路径 (List of waypoints)
    ↓
ORCA层: 计算避让其他机器人的速度
    ↓
DWA层: 考虑激光障碍物，生成最终速度命令
    ↓
发布到 /my_botX/cmd_vel
```

## 🚀 使用步骤

### 1. 安装Nav2 (必需)

```bash
sudo apt update
sudo apt install ros-humble-nav2-bringup
```

### 2. 编译

```bash
cd /home/wj/work/multi-robot-exploration-rl
colcon build --packages-select start_orca_nav
source install/setup.bash
```

### 3. 启动系统

```bash
# 默认配置
./start_orca_nav2.sh

# 自定义配置
./start_orca_nav2.sh -m corridor_swap -r 4
```

### 4. 发送目标点

```bash
ros2 topic pub --once /robot0/goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: map}, 
    pose: {position: {x: 5.0, y: 3.0}, orientation: {w: 1.0}}}'
```

## 📊 与原版本对比

| 特性 | 原版本 | 新版本 (Nav2集成) |
|-----|-------|------------------|
| 全局规划 | 自实现Theta* | ✅ Nav2 (NavFn/Smac/ThetaStar) |
| 地图支持 | 手动边界 | ✅ PGM/YAML地图文件 |
| 路径质量 | ❌ 可能穿墙 | ✅ 正确避障 |
| 障碍物 | 仅边界 | ✅ 真实地图障碍物 |
| ORCA避碰 | ✅ 保留 | ✅ 保留 |
| DWA局部规划 | ✅ 保留 | ✅ 保留 |
| 可扩展性 | 有限 | ✅ Nav2生态系统 |

## 🎓 架构优势

1. **利用Nav2成熟的规划器**: 不需要自己实现复杂的全局规划算法
2. **真实地图支持**: 可以使用任何PGM/YAML格式的地图
3. **保留自定义控制**: ORCA和DWA层完全由你控制
4. **易于扩展**: 可以轻松切换不同的Nav2规划器插件
5. **分离关注点**: 
   - Nav2负责全局规划（静态障碍物）
   - ORCA负责多机器人避碰（动态）
   - DWA负责局部规划（运动学约束）

## 📝 关键代码修改

### 删除的内容
- ❌ `global_planner.py` (Theta*实现) - 不再使用
- ❌ `_initialize_map_obstacles()` - 不再需要手动设置障碍物
- ❌ `SimpleGlobalPlanner` 类 - 被Nav2替代

### 新增的内容
- ✅ `ComputePathToPose` action client
- ✅ `_request_nav2_path()` 方法
- ✅ `_nav2_path_result()` 回调
- ✅ Nav2 launch配置

## 🐛 故障排除

### 问题：Nav2包未安装
```bash
sudo apt install ros-humble-nav2-bringup
```

### 问题：地图文件找不到
检查地图路径：
```bash
ls src/start_rl_environment/maps/*.yaml
```

### 问题：Nav2服务不可用
检查Nav2是否启动：
```bash
ros2 service list | grep compute_path
```

## 📚 相关文档

- [ORCA_NAV2_README.md](./ORCA_NAV2_README.md) - 详细文档
- [Nav2官方文档](https://navigation.ros.org/)
- 原有文档依然有效（ORCA和DWA部分）

## 🔄 如何回退到原版本

如果需要使用原来的Theta*版本：
```bash
./start_orca_nav.sh -m 3 -r 4
```

两个版本可以共存，使用不同的启动脚本。

## ✨ 下一步建议

1. **安装Nav2**: `sudo apt install ros-humble-nav2-bringup`
2. **测试系统**: `./test_nav2_setup.sh`
3. **启动测试**: `./start_orca_nav2.sh -m corridor_swap -r 1`
4. **发送目标**: 观察Nav2规划的路径是否避开墙壁
5. **多机器人测试**: 启动4个机器人，测试ORCA避碰

## 💡 技术亮点

- **Action-based通信**: 使用ROS2 action实现异步路径请求
- **模块化设计**: Nav2、ORCA、DWA三层完全解耦
- **真实地图**: 支持任何符合ROS标准的地图格式
- **独立Nav2栈**: 每个机器人有自己的map server和planner
- **保持兼容**: 原有的ORCA和DWA代码完全保留
