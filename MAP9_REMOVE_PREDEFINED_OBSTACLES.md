# Map 9: 移除预定义灰色圆柱障碍物

## 问题分析

在warehouse_dynamic.world（Map 9）中，预定义了8个灰色圆柱障碍物（static_obs_0~7），导致：

1. **视觉混乱**：世界文件中的固定位置障碍物与代码动态spawn的冲突
2. **管理复杂**：需要用SetEntityState移动预定义实体到随机位置
3. **清理困难**：reset时需要删除旧障碍物，但world文件会在下次加载时重新创建

## 解决方案

### 方案：移除world预定义 + 代码动态spawn

**优点**：
- 彻底解耦world文件和代码逻辑
- 每次reset完全控制障碍物数量和位置
- 代码管理更清晰（spawn + delete）

**缺点**：
- 需要修改代码从SetEntityState改为SpawnEntity

---

## 实施细节

### 1. 修改world文件

**文件**：`src/start_rl_environment_tb3/worlds/warehouse_dynamic.world`

**变更**：删除第55-142行的所有static_obs定义

**保留内容**：
- 边界墙（b_north, b_south, b_east, b_west）
- 动态障碍物（dyn_obs_0~7）- 由obstacle_mover.py控制

**删除内容**：
- static_obs_0 ~ static_obs_7（灰色圆柱，RGB 0.6 0.6 0.6）

---

### 2. 修改环境代码

**文件**：`src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/gnn_marl_env.py`

#### 变更1：添加SpawnEntity服务（Line 30, 1432）

```python
# Line 30: 导入
from gazebo_msgs.srv import SetEntityState, DeleteEntity, SpawnEntity

# Line 1432: 创建客户端
self.spawn_entity_client = self.node.create_client(SpawnEntity, '/spawn_entity')
```

#### 变更2：重写spawn_obstacle函数（Line 3897-3941）

**修改前**：使用`SetEntityState`移动预定义实体位置

**修改后**：使用`SpawnEntity`创建新实体

```python
def spawn_obstacle(name, radius, is_static=True):
    """尝试spawn一个障碍物（使用SpawnEntity创建新实体）"""
    # ... 位置验证 ...
    
    # 生成SDF模型
    sdf_xml = self._generate_cylinder_sdf(
        name=name,
        radius=radius,
        height=0.8,
        is_static=is_static,
        color=(0.6, 0.6, 0.6, 1.0) if is_static else (0.8, 0.2, 0.2, 1.0)
    )

    # 使用SpawnEntity创建新实体
    req = SpawnEntity.Request()
    req.name = name
    req.xml = sdf_xml
    req.robot_namespace = ''
    req.initial_pose.position.x = float(x)
    req.initial_pose.position.y = float(y)
    req.initial_pose.position.z = 0.4
    req.initial_pose.orientation.w = 1.0

    if self.spawn_entity_client.wait_for_service(timeout_sec=0.5):
        future = self.spawn_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.5)
    return True
```

#### 变更3：添加SDF生成方法（Line 3959-4017）

新增`_generate_cylinder_sdf()`方法，动态生成圆柱体障碍物的SDF模型：

```python
def _generate_cylinder_sdf(self, name, radius, height, is_static=True, color=(0.6, 0.6, 0.6, 1.0)):
    """
    生成圆柱体障碍物的SDF模型
    
    Args:
        name: 模型名称
        radius: 圆柱半径
        height: 圆柱高度
        is_static: 是否为静态障碍物
        color: RGBA颜色元组
    
    Returns:
        str: SDF XML字符串
    """
    static_tag = 'true' if is_static else 'false'
    mass = 1.0 if is_static else 5.0
    inertial_block = '' if is_static else '''
      <inertial>
        <mass>{mass}</mass>
        <inertia>...</inertia>
      </inertial>'''
    
    sdf = f'''<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>{static_tag}</static>
    <link name="link">{inertial_block}
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>{radius}</radius>
            <length>{height}</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>{color[0]} {color[1]} {color[2]} {color[3]}</ambient>
          <diffuse>{color[0]} {color[1]} {color[2]} {color[3]}</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>'''
    return sdf
```

#### 变更4：reset时删除上一轮障碍物（Line 2908-2918）

在reset()方法开头添加清理逻辑：

```python
# 【Map 9特殊处理】删除上一轮spawn的静态障碍物（只由robot_id==0执行）
if self.robot_id == 0 and self.map_number == 9 and self.random_obstacles:
    if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
        # 删除上一轮的所有静态障碍物（最多8个）
        for i in range(8):
            req = DeleteEntity.Request()
            req.name = f'static_obs_{i}'
            future = self.delete_entity_client.call_async(req)
            # 不等待完成，异步删除
        # 等待删除操作完成
        self._wait_for_sim_time(0.1)
```

#### 变更5：简化_spawn_random_obstacles清理逻辑（Line 3943-3957）

**删除**：原来清理未使用static_obs的代码（因为world不再预定义）

**保留**：清理未使用动态障碍物的逻辑

```python
# Spawn静态障碍物（环境负责）
for i in range(self.num_static_obstacles):
    spawn_obstacle(f'static_obs_{i}', STATIC_OBS_RADIUS, is_static=True)

# 动态障碍物由 obstacle_mover.py 负责spawn和移动
# 这里只负责删除未使用的动态障碍物（避免雷达误检测）
if self.delete_entity_client.wait_for_service(timeout_sec=0.5):
    for i in range(self.num_dynamic_obstacles, 8):
        req = DeleteEntity.Request()
        req.name = f'dyn_obs_{i}'
        future = self.delete_entity_client.call_async(req)

# 等待物理引擎稳定
self._wait_for_sim_time(0.15)
```

---

## 工作流程

### Episode生命周期中的障碍物管理

```
┌─────────────────────────────────────────────────────────────┐
│ Episode N-1 结束                                              │
├─────────────────────────────────────────────────────────────┤
│ 1. reset() 开始                                               │
│    └─ 删除 static_obs_0~7（上一轮spawn的）                   │
│       [DeleteEntity × num_static_obstacles]                   │
├─────────────────────────────────────────────────────────────┤
│ 2. _spawn_random_obstacles()                                  │
│    └─ 创建 static_obs_0~3（假设num_static_obstacles=4）      │
│       [SpawnEntity × 4, 每个随机位置]                        │
│    └─ 删除 dyn_obs_4~7（假设num_dynamic_obstacles=4）        │
│       [DeleteEntity × 4]                                      │
├─────────────────────────────────────────────────────────────┤
│ 3. Episode N 运行                                             │
│    └─ 4个静态灰色圆柱在随机位置                              │
│    └─ 4个动态红色圆柱由obstacle_mover控制                    │
├─────────────────────────────────────────────────────────────┤
│ 4. Episode N 结束 → 循环到步骤1                              │
└─────────────────────────────────────────────────────────────┘
```

---

## SetEntityState vs SpawnEntity对比

| 特性 | SetEntityState | SpawnEntity |
|------|---------------|-------------|
| **前提** | 实体必须已存在 | 创建新实体 |
| **world文件依赖** | 依赖预定义 | 独立 |
| **灵活性** | 只能移动位置 | 完全自定义（几何、颜色、物理） |
| **清理** | 需要移到地图外或删除 | 直接删除 |
| **适用场景** | 微调已有实体 | 动态创建/销毁 |

---

## 预期效果

### ✅ 优点

1. **视觉清晰**
   - Gazebo启动后场景空旷，只有边界墙
   - 第一次reset后才出现障碍物

2. **代码可控**
   - 完全由代码控制障碍物数量和位置
   - 不依赖world文件预定义

3. **管理简单**
   - Reset清理：DeleteEntity删除旧的
   - Reset创建：SpawnEntity创建新的
   - 生命周期清晰

4. **扩展性强**
   - 可以轻松改变障碍物形状（圆柱→方块→任意）
   - 可以动态调整大小、颜色
   - 不需要编辑world文件

### ⚠️ 注意事项

1. **性能开销**
   - SpawnEntity比SetEntityState慢（~50-100ms）
   - 每个episode reset增加约200-400ms（spawn 4个障碍物）
   - 对于训练来说可接受

2. **首次spawn可能失败**
   - Gazebo服务需要时间初始化
   - 代码已有0.5s超时保护

3. **并发控制**
   - 只有robot_id==0执行spawn（避免重复）
   - 其他robot通过parent_env共享障碍物信息

---

## 测试验证

### 测试1：Gazebo启动验证

```bash
# 启动训练
./run_curriculum.sh --start_stage 1 --end_stage 1 --train_steps 100

# 观察Gazebo
# 预期：启动后场景只有边界墙和地面，无灰色圆柱
# 第一次reset后出现4个灰色圆柱（随机位置）
```

### 测试2：Reset清理验证

**观察要点**：
- 每次episode结束后，旧障碍物消失
- 新episode开始后，新障碍物出现在不同位置
- 障碍物数量=num_static_obstacles（配置值）

### 测试3：性能影响

```bash
# 对比reset时间（添加日志）
grep "reset完成" train_*.log | awk '{print $NF}' | sort -n | tail -100

# 预期：reset时间增加200-400ms（可接受）
```

---

## 相关文档

- **INVISIBLE_OBSTACLES_FIX.md** - 隐形障碍物和初始规划修复
- **OBSTACLE_FIXES_SUMMARY.md** - 5个障碍物系统修复总结
- **本文档** - Map 9预定义障碍物移除

---

## 回滚方案

如果发现问题需要回滚：

### 1. 恢复world文件

```bash
git checkout src/start_rl_environment_tb3/worlds/warehouse_dynamic.world
```

### 2. 恢复代码

在`_spawn_random_obstacles()`中改回SetEntityState：

```python
# 使用SetEntityState（旧方法）
req = SetEntityState.Request()
req.state.name = name
req.state.pose.position.x = float(x)
req.state.pose.position.y = float(y)
req.state.pose.position.z = 0.4
req.state.pose.orientation.w = 1.0

future = self.set_state_client.call_async(req)
rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.5)
```

并恢复清理未使用障碍物的代码。

---

## 验证清单

- [x] 语法检查通过
- [x] 编译成功
- [ ] Gazebo启动无报错
- [ ] 首次reset成功spawn障碍物
- [ ] 障碍物位置随机且不重叠
- [ ] Reset清理旧障碍物
- [ ] 训练运行稳定（至少100 episodes）
- [ ] 性能影响可接受（reset时间+200-400ms）
