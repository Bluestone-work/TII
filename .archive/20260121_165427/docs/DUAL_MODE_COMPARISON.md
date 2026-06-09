# 双模式导航对比说明

## 问题背景

**原始Bug**：之前的代码同时使用了`NavigateToPose` action和ORCA节点控制机器人，导致两个"司机"同时向`/cmd_vel`发布命令，产生冲突。

**问题原因**：
- `NavigateToPose`动作会启动Nav2的Controller Server，直接控制`/cmd_vel`
- ORCA节点也在计算速度并发布到`/cmd_vel`
- 结果：两个控制器互相干扰，机器人行为异常

---

## 解决方案：双模式架构

现在实现了两种完全独立的导航模式，可以方便地进行对比测试。

### 🔵 模式1：ORCA + DWA + Theta* （自研算法）

**架构**：
```
Theta* 全局规划 → ORCA 动态避碰 → DWA 局部控制 → /cmd_vel
```

**特点**：
- ✅ **完全自主控制**：不依赖Nav2的控制器
- ✅ **三层分层设计**：职责清晰，易于调试
- ✅ **轻量级**：无需完整Nav2栈
- ✅ **可定制性强**：每层算法都可独立调整

**各层职责**：
1. **Theta***：规划全局路径，避开已知静态障碍
2. **ORCA**：计算多机器人避碰速度
3. **DWA**：生成满足运动学约束的控制命令

**启动命令**：
```bash
./start_orca_nav.sh -m 1 -r 2 --mode orca  # 默认就是orca
```

---

### 🟢 模式2：纯Nav2导航（对比基准）

**架构**：
```
Nav2 Planner → Nav2 Controller → /cmd_vel
```

**特点**：
- ✅ **成熟稳定**：Nav2是业界标准
- ✅ **功能完整**：包含恢复行为、代价地图等
- ✅ **社区支持**：文档丰富，问题易查

**工作流程**：
1. 节点接收目标点
2. 通过`NavigateToPose` action发送给Nav2
3. Nav2完全接管机器人控制
4. ORCA节点仅监控状态，不发布速度

**启动命令**：
```bash
./start_orca_nav.sh -m 1 -r 2 --mode nav2
```

---

## 代码实现细节

### 关键修改

#### 1. **参数系统**
```python
# 新参数
self.declare_parameter('navigation_mode', 'orca')  # 'orca' or 'nav2'

# 根据模式设置行为
self.use_nav2_full = (self.navigation_mode == 'nav2')  # 完全使用Nav2控制
self.use_orca = (self.navigation_mode == 'orca')      # 使用ORCA+DWA+Theta*
```

#### 2. **模式1：ORCA初始化**
```python
if self.use_orca:
    # 创建Theta*全局规划器
    self.global_planner = create_simple_planner(
        map_width=20.0,
        map_height=20.0,
        resolution=0.1
    )
    
    # 创建DWA局部规划器
    self.dwa_planner = create_dwa_planner(...)
    
    # 启动控制循环
    self.timer = self.create_timer(0.05, self.control_loop)
```

#### 3. **模式2：Nav2初始化**
```python
if self.use_nav2_full:
    # 创建action client
    self.nav2_clients[robot_name] = ActionClient(
        self, NavigateToPose, f'/{gazebo_namespace}/navigate_to_pose'
    )
    
    # 不启动控制循环！Nav2会接管cmd_vel
```

#### 4. **目标处理**
```python
def goal_callback(self, msg: PoseStamped, robot_name: str):
    if self.use_nav2_full:
        # 模式2：发送给Nav2，让Nav2控制
        self.send_nav2_goal(robot_name, msg)
    else:
        # 模式1：使用Theta*规划路径
        path = self.global_planner.plan_path(start, goal)
        self.theta_star_paths[robot_name] = path
```

#### 5. **控制循环**
```python
def control_loop(self):
    # 只在ORCA模式下运行
    if not self.use_orca:
        return
    
    # ... ORCA+DWA计算并发布cmd_vel ...
```

---

## 使用指南

### 快速测试

**测试ORCA模式**：
```bash
# 方式1：不指定（默认）
./start_orca_nav.sh -m 1 -r 2

# 方式2：明确指定
./start_orca_nav.sh -m 1 -r 2 --mode orca
```

**测试Nav2模式**：
```bash
./start_orca_nav.sh -m 1 -r 2 --mode nav2
```

### 对比测试流程

1. **测试场景1：开放空间**
```bash
# ORCA模式
./start_orca_nav.sh -m 1 -r 4 --mode orca
# 观察：路径平滑度、避碰效果、到达时间

# 清理
./kill_all_ros.sh

# Nav2模式
./start_orca_nav.sh -m 1 -r 4 --mode nav2
# 观察：对比ORCA模式的表现
```

2. **测试场景2：走廊对穿**
```bash
# ORCA模式
./start_orca_nav.sh -m 3 -r 2 --mode orca

# Nav2模式
./start_orca_nav.sh -m 3 -r 2 --mode nav2
```

3. **测试场景3：复杂环境**
```bash
# ORCA模式
./start_orca_nav.sh -m 2 -r 4 --mode orca

# Nav2模式
./start_orca_nav.sh -m 2 -r 4 --mode nav2
```

---

## 性能对比指标

### 建议对比维度

| 指标 | ORCA模式 | Nav2模式 | 说明 |
|------|---------|----------|------|
| **路径长度** | - | - | 从起点到终点的实际路径长度 |
| **运行时间** | - | - | 完成导航的总时间 |
| **路径平滑度** | - | - | 转向次数、速度变化 |
| **避碰效果** | - | - | 最小距离、碰撞次数 |
| **CPU占用** | - | - | 平均CPU使用率 |
| **内存占用** | - | - | 峰值内存 |
| **参数敏感度** | - | - | 参数调整难度 |
| **鲁棒性** | - | - | 失败率、恢复能力 |

### 监控命令

```bash
# 查看速度命令
ros2 topic echo /my_bot0/cmd_vel

# 查看ORCA日志
tail -f orca_logs/navigation_*.log | grep "ORCA_vel"

# 查看Nav2状态
ros2 topic echo /my_bot0/navigate_to_pose/_action/status

# 监控CPU/内存
htop  # 查找orca_nav_node或nav2进程
```

---

## 预期差异

### ORCA模式的优势场景

1. **多机器人密集环境**
   - ORCA的对称避让更自然
   - DWA响应更快

2. **动态障碍物多**
   - 激光雷达实时避障
   - 无需预先建图

3. **计算资源受限**
   - 轻量级实现
   - 可控的计算复杂度

### Nav2模式的优势场景

1. **单机器人导航**
   - 功能更完整
   - 恢复行为成熟

2. **静态地图明确**
   - 代价地图优化好
   - 路径规划更全局

3. **需要标准化**
   - 符合ROS2生态
   - 易于集成其他工具

---

## 故障排查

### ORCA模式

**问题：机器人不动**
```bash
# 检查Theta*路径
tail -f orca_logs/navigation_*.log | grep "Theta\*"

# 检查ORCA速度
tail -f orca_logs/navigation_*.log | grep "ORCA_vel"

# 检查DWA输出
ros2 topic echo /my_bot0/cmd_vel
```

**问题：路径规划失败**
```bash
# Theta*可能找不到路径
# 检查：起点/终点是否在障碍物内
# 调整：增大地图分辨率或检查障碍物设置
```

### Nav2模式

**问题：Nav2未响应**
```bash
# 检查action server
ros2 action list | grep navigate_to_pose

# 检查Nav2状态
ros2 topic echo /my_bot0/navigate_to_pose/_action/feedback
```

**问题：两个控制器冲突**
```bash
# 确认模式设置
# 应该看到：navigation_mode: nav2
ros2 param get /orca_nav_node navigation_mode
```

---

## 总结

### 修复内容

1. ✅ **移除NavigateToPose冲突**：ORCA模式不再使用Nav2控制器
2. ✅ **实现Theta*规划器**：替代Nav2的全局规划
3. ✅ **双模式参数化**：一个参数切换两种模式
4. ✅ **完全独立控制**：每种模式独占cmd_vel

### 使用建议

- **开发/研究**：使用ORCA模式，便于调试和定制
- **对比测试**：两种模式都运行，量化性能差异
- **生产部署**：根据场景选择最优模式

### 下一步

1. 收集对比数据
2. 优化表现较差的模式
3. 根据应用场景选择最终方案
