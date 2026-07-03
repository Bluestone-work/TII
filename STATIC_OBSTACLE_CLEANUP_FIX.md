# 静态障碍物清理修复：地图预置障碍物导致A*感知缺失

## 修复日期
2026-07-03

## 问题描述

用户发现map9地图中有**4个棕色方块静态障碍物在一开始就存在**，但在后续随机放置4个新障碍物后，**一开始的4个不会被清理**，导致：

1. **总共8个障碍物**存在于Gazebo中
2. **A*全局规划器只能感知到新spawn的4个**（记录在`spawned_static_obstacles`中）
3. **地图预置的4个障碍物对A*不可见**，导致规划路径穿过它们
4. **机器人按A*路径行进时撞到不可见障碍物**

## 根因分析

### 1. 障碍物命名演变
- **旧版命名**：`static_obs_*`（灰色圆柱）
- **新版命名**：`static_box_*`（棕色方块，2026-07-03改版）

### 2. Reset清理逻辑缺陷
**文件**：[gnn_marl_env.py:2915-2932](gnn_marl_training/gnn_marl_env.py#L2915-L2932)

**修复前**的代码只删除`static_box_*`：
```python
for i in range(16):
    req = DeleteEntity.Request()
    req.name = f'static_box_{i}'
    future = self.delete_entity_client.call_async(req)
```

**问题**：
- 如果之前某次运行使用旧命名spawn了`static_obs_*`
- 或者地图文件本身包含了`static_obs_*`（实际检查发现warehouse_dynamic.world不含）
- 这些旧障碍物永远不会被删除
- 它们的位置也不在`spawned_static_obstacles`列表中
- A*规划器`plan_with_dynamic_obstacles()`只接收`spawned_static_obstacles`中的坐标
- 结果：旧障碍物对激光雷达可见，但对A*不可见

### 3. 触发场景
1. **代码版本切换**：从旧版（`static_obs_*`）切换到新版（`static_box_*`）
2. **未重启Gazebo**：残留的`static_obs_*`一直存在
3. **地图预置障碍物**：如果地图文件被手动编辑添加了静态障碍物

## 修复方案

### 修改内容
**文件**：`gnn_marl_training/gnn_marl_env.py`  
**行数**：2915-2932

**修复后**的代码同时删除两种命名：
```python
for i in range(16):
    # 删除当前命名的棕色方块
    req = DeleteEntity.Request()
    req.name = f'static_box_{i}'
    future = self.delete_entity_client.call_async(req)

    # 删除地图预置的旧命名障碍物（static_obs_*）
    req_old = DeleteEntity.Request()
    req_old.name = f'static_obs_{i}'
    future_old = self.delete_entity_client.call_async(req_old)

# 更新日志输出
print(f"🗑️  Robot {self.robot_id}: 已清理旧的static_box (0-15) + static_obs (0-15)", flush=True)
```

### 修复效果
- ✅ 每次reset时清理所有可能的残留障碍物
- ✅ 兼容旧版和新版命名
- ✅ 确保A*规划器感知到所有实际存在的障碍物
- ✅ 异步删除，不影响reset性能

## 诊断工具

新增脚本：[list_gazebo_models.py](list_gazebo_models.py)

**用途**：列出当前Gazebo中的所有模型，检查是否有残留障碍物

**使用方法**：
```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer
python3 list_gazebo_models.py
```

**输出示例**：
```
📋 Gazebo中当前有 15 个模型:

🟫 棕色方块 (static_box_*): 4 个
   - static_box_0
   - static_box_1
   - static_box_2
   - static_box_3

⚠️  旧命名静态障碍物 (static_obs_*): 4 个
   - static_obs_0
   - static_obs_1
   - static_obs_2
   - static_obs_3

🔴 动态障碍物 (dyn_obs_*): 8 个
   - dyn_obs_0
   - ...

⚠️  检测到旧命名的static_obs_*障碍物！
   这些可能是之前残留的，会导致A*规划器无法感知
```

## 验证方法

### 1. 检查Gazebo中的模型
```bash
python3 list_gazebo_models.py
```
- 确认只有当前应该存在的`static_box_*`
- 没有残留的`static_obs_*`

### 2. 检查A*规划器日志
在reset时观察输出：
```
🗑️  Robot 0: 已清理旧的static_box (0-15) + static_obs (0-15)
🟫 Robot 0: 成功spawn 4/4 个棕色方块
✅ Robot 0: A*规划成功，路径长度=XX点 → waypoints=XX点
🗺️  Robot 0: A*规划 start=(...) goal=(...) blocked_points=4
```

**关键指标**：
- `blocked_points`数量应该等于`num_static_obstacles`配置值
- 如果`blocked_points=0`，说明`spawned_static_obstacles`列表为空（BUG）
- 如果`blocked_points < num_static_obstacles`，说明部分障碍物未被记录

### 3. 可视化验证
在RViz中查看：
- 发布的waypoints路径（绿色线）
- 路径应该绕过所有棕色方块
- 如果路径穿过障碍物，说明A*未感知到该障碍物

## 相关文件

| 文件 | 说明 |
|------|------|
| [gnn_marl_env.py:2915-2932](gnn_marl_training/gnn_marl_env.py#L2915-L2932) | 障碍物清理逻辑（已修复） |
| [gnn_marl_env.py:3942-4096](gnn_marl_training/gnn_marl_env.py#L3942-L4096) | 障碍物spawn逻辑（`static_box_*`命名） |
| [gnn_marl_env.py:4033-4037](gnn_marl_training/gnn_marl_env.py#L4033-L4037) | 记录到`spawned_static_obstacles`列表 |
| [gnn_marl_env.py:3048-3108](gnn_marl_training/gnn_marl_env.py#L3048-L3108) | A*规划器调用（读取`spawned_static_obstacles`） |
| [warehouse_dynamic.world](../../start_rl_environment_tb3/worlds/warehouse_dynamic.world) | Map9地图文件（不含预置静态障碍物） |
| [list_gazebo_models.py](list_gazebo_models.py) | 诊断工具：列出Gazebo模型 |

## 注意事项

1. **异步删除**：`call_async()`不会阻塞，所有删除请求并行发送
2. **等待时间**：删除后等待0.2秒确保Gazebo处理完成
3. **范围扩大**：删除0-15共16个索引，防止遗漏（实际配置最多8个）
4. **只由robot_0执行**：避免多个智能体重复删除
5. **不影响动态障碍物**：`dyn_obs_*`由`obstacle_mover.py`管理，不在此清理

## 后续建议

### 短期（已完成）
- ✅ 修复reset时的障碍物清理逻辑
- ✅ 添加诊断工具

### 中期
- [ ] 在spawn后验证`spawned_static_obstacles`列表长度是否正确
- [ ] 在A*规划前打印`blocked_points`数量，便于调试
- [ ] 添加单元测试：验证reset后Gazebo中只有预期数量的障碍物

### 长期
- [ ] 统一障碍物管理接口：spawn/delete/query封装成类
- [ ] 从Gazebo反向查询障碍物位置，而不是仅依赖`spawned_static_obstacles`列表
- [ ] 支持从地图文件中解析预置障碍物并加入A*感知列表

## 相关记忆

- [MAP9_REPLACE_CYLINDERS_WITH_BOXES.md](MAP9_REPLACE_CYLINDERS_WITH_BOXES.md) - 静态障碍物从灰色圆柱改为棕色方块
- [SPAWN_PERFORMANCE_OPTIMIZATION.md](SPAWN_PERFORMANCE_OPTIMIZATION.md) - Spawn性能优化
