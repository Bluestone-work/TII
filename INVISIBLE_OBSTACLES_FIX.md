# 隐形障碍物和全局规划修复

## 问题描述

### 问题1：隐形障碍物被雷达探测到
**现象**：
- Gazebo初始化时有8个白色柱子（world文件预定义）
- 机器人spawn后这些柱子从视觉上消失
- 但雷达仍能探测到它们（碰撞体仍存在）

**根本原因**：
1. warehouse_dynamic.world预定义了8个静态障碍物（static_obs_0~7）和8个动态障碍物（dyn_obs_0~7）
2. 环境reset时使用`SetEntityState`将未使用的障碍物移到(100, 100)地图外
3. **移动实体不会删除碰撞体**，雷达仍能探测到它们
4. 这些"隐形障碍物"会干扰机器人的避障决策

### 问题2：全局规划没考虑spawn的静态障碍物
**现象**：
- 虽然已经将障碍物spawn放在A*规划之前
- 但初始路径仍然会穿过随机spawn的静态障碍物

**根本原因**：
- `reset()`方法中使用的是`planner.plan()`
- 该方法只读取静态.pgm地图文件，无法感知运行时spawn的障碍物
- 虽然有`plan_with_dynamic_obstacles()`方法可以叠加动态障碍，但初始规划没有调用

---

## 解决方案

### 修复1：删除而非移动未使用障碍物

**修改文件**: `gnn_marl_env.py`

**变更1：添加DeleteEntity服务客户端**
```python
# Line 30: 添加导入
from gazebo_msgs.srv import SetEntityState, DeleteEntity

# Line 1430: 创建删除客户端
self.delete_entity_client = self.node.create_client(DeleteEntity, '/delete_entity')
```

**变更2：改用删除代替移动（Line 3937-3956）**
```python
# 动态障碍物由 obstacle_mover.py 负责spawn和移动
# 这里只负责删除未使用的动态障碍物（避免雷达误检测）
if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
    for i in range(self.num_dynamic_obstacles, 8):
        req = DeleteEntity.Request()
        req.name = f'dyn_obs_{i}'
        future = self.delete_entity_client.call_async(req)
        # 不等待完成，异步删除

# 等待物理引擎稳定（避免spawn后的碰撞检测误判）
self._wait_for_sim_time(0.15)

# 删除未使用的静态障碍物（避免遗留的隐形碰撞体被雷达探测）
if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
    for i in range(self.num_static_obstacles, 8):
        req = DeleteEntity.Request()
        req.name = f'static_obs_{i}'
        future = self.delete_entity_client.call_async(req)
        # 不等待完成，异步删除
```

**原理**：
- `DeleteEntity`从Gazebo场景中完全移除实体
- 包括视觉、碰撞体、物理属性等所有组件
- 雷达无法探测到已删除的实体

---

### 修复2：初始规划考虑spawn的障碍物

**修改文件**: `gnn_marl_env.py`

**变更：改用plan_with_dynamic_obstacles()（Line 3017-3033）**

**修改前**：
```python
# 现在进行A*规划（此时静态障碍物已经存在，但A*仍用静态地图）
# 注意：随机spawn的静态障碍物不在.pgm地图中，A*仍然感知不到
# 依赖重规划机制在运行时检测到碰撞后重新规划
if self.planner:
    path = self.planner.plan((start_x, start_y), (goal_x, goal_y))
```

**修改后**：
```python
# 现在进行A*规划（必须使用plan_with_dynamic_obstacles考虑spawn的静态障碍物）
if self.planner:
    # 获取spawn的静态障碍物位置
    blocked_points = []
    if hasattr(self, 'parent_env') and hasattr(self.parent_env, 'spawned_static_obstacles'):
        blocked_points = [(x, y) for x, y, _ in self.parent_env.spawned_static_obstacles]
    elif hasattr(self, 'spawned_static_obstacles'):
        blocked_points = [(x, y) for x, y, _ in self.spawned_static_obstacles]

    # 使用动态障碍物规划器（会在occupancy grid上叠加blocked区域）
    path = self.planner.plan_with_dynamic_obstacles(
        (start_x, start_y),
        (goal_x, goal_y),
        blocked_world_points=blocked_points,
        block_radius_m=0.35  # 略大于障碍物半径0.20，留安全边距
    )
```

**原理**：
1. 从`parent_env.spawned_static_obstacles`获取所有spawn的静态障碍物位置
2. 调用`plan_with_dynamic_obstacles()`而不是`plan()`
3. 规划器会在静态occupancy grid上叠加blocked区域
4. A*算法在叠加后的地图上规划，避开spawn的障碍物

**block_radius_m参数选择**：
- 静态障碍物实际半径：0.20m
- 设置为0.35m（1.75倍）
- 原因：留出安全边距，避免路径贴近障碍物边缘

---

## 预期效果

### 修复1效果
✅ **雷达不再探测到隐形障碍物**
- 未使用的障碍物被完全删除
- 雷达扫描数据更准确
- 避障决策不会受干扰

✅ **Gazebo场景更干净**
- 没有地图外(100,100)的遗留实体
- 物理引擎负担减轻
- 仿真性能可能略有提升

### 修复2效果
✅ **初始路径质量提升**
- 规划的全局路径避开spawn的静态障碍物
- 减少初始碰撞风险
- 到达目标更高效

✅ **减少重规划次数**
- 初始路径本身就是可行的
- 不需要等待碰撞后才触发重规划
- 训练初期episode存活时间更长

✅ **训练收敛更快**
- 智能体能更早学习到"沿路径到达目标"的策略
- 而不是一开始就陷入"反复碰撞-重规划"循环

---

## 两层防护机制

现在全局规划对spawn障碍物的感知有**两层保护**：

### 第一层：初始规划（本次修复）
- reset()时使用`plan_with_dynamic_obstacles()`
- 初始路径已经避开spawn的静态障碍物
- **防止问题发生**

### 第二层：运行时重规划（之前已实现）
- `_try_replan_due_to_deadlock()`中的增强重规划
- 考虑其他机器人位置和spawn的静态障碍物
- **问题发生后补救**

两层机制互补：
- 第一层减少问题发生概率
- 第二层处理动态变化（其他机器人移动、死锁等）

---

## 测试建议

### 测试1：雷达数据验证
```bash
# 启动训练
./run_curriculum.sh --start_stage 1 --end_stage 1 --train_steps 1000

# 另开终端，打印雷达数据
ros2 topic echo /robot_0/scan --once
```

**预期**：
- 雷达ranges数组中不应出现距离>10m的异常值
- 如果num_static_obstacles=4，场景中应只有4+4(墙)=8个障碍物被探测

### 测试2：初始路径可视化
**在RViz中观察**：
1. 机器人spawn后立即观察全局路径（绿色线）
2. 检查路径是否穿过灰色静态障碍物

**预期**：
- ✅ 路径绕开所有spawn的静态障碍物
- ✅ 路径与障碍物保持至少0.35m距离

### 测试3：初始碰撞率统计
**对比修复前后**：
```bash
# 运行1000步，统计前100 episode的初始碰撞率
grep "collision" train_*.log | head -100
```

**预期**：
- 修复后初始碰撞次数显著减少
- "卡在spawn点附近"的情况减少

---

## 技术细节

### DeleteEntity vs SetEntityState
| 方法 | 效果 | 碰撞体 | 物理计算 | 雷达可见 |
|------|------|--------|----------|----------|
| SetEntityState移到地图外 | 移动位置 | **保留** | 保留 | ✅ 可见 |
| DeleteEntity | **完全删除** | 删除 | 停止 | ❌ 不可见 |

### plan() vs plan_with_dynamic_obstacles()
| 方法 | 数据源 | 运行时障碍 | 适用场景 |
|------|--------|-----------|----------|
| plan() | 静态.pgm地图 | ❌ 感知不到 | 纯静态环境 |
| plan_with_dynamic_obstacles() | .pgm + blocked_points | ✅ 叠加计算 | spawn障碍物、其他机器人 |

### block_radius_m选择依据
```
障碍物实际半径: 0.20m
机器人半径: 0.105m
理论最小间隔: 0.20 + 0.105 = 0.305m
实际设置: 0.35m (增加0.045m安全边距)
安全边距占比: 0.045 / 0.305 = 14.75%
```

---

## 相关修复记录
- **OBSTACLE_FIXES_SUMMARY.md** - 5个障碍物系统修复
- **GLOBAL_PLANNING_FIX.md** - 规划顺序调整和重规划增强
- **本文档** - 隐形障碍物和初始规划修复

---

## 验证清单
- [x] 语法检查通过
- [ ] 启动训练无报错
- [ ] 雷达数据验证（无>10m异常值）
- [ ] RViz路径可视化（绕开障碍物）
- [ ] 初始碰撞率统计（对比修复前）
