# 障碍物系统修复总结

## 修复日期
2026-07-03

## 修复的问题

### 1. RViz全局路径点不清理
**问题**：训练过程中RViz显示的全局路径点（global waypoints）会累积，不会在reset时清除。

**根因**：`reset()`方法中只清理了RViz可视化，但没有清空`global_waypoints`列表。

**修复**：
- 位置：`gnn_marl_env.py` 第2887行
- 在调用`vis.clear_waypoints()`之前先执行`self.global_waypoints = []`

```python
# 清理全局路径点（RViz可视化 + 内部列表）
self.global_waypoints = []
if hasattr(self, 'vis') and self.vis:
    self.vis.clear_waypoints(namespace=self.vis_namespace)
```

### 2. 静态障碍物数量未纳入课程设置
**问题**：静态障碍物数量只能通过命令行参数`--num_static_obstacles`控制，没有在课程学习的stage配置中。

**修复**：
- 在`run_curriculum.sh`中添加`STAGE_STATIC_OBS_NUM`数组（第364行）
- 更新stage名称，明确显示动态和静态障碍物数量
- 在训练命令构建时自动传递静态障碍物数量

**配置示例**：
```bash
declare -A STAGE_STATIC_OBS_NUM=( [1]=4 [2]=5 [3]=5 [4]=6 [5]=4 [6]=3 )
```

### 3. 障碍物重叠问题
**问题**：
- 静态障碍物spawn时可能发生重叠
- 障碍物与机器人spawn点距离过近
- 多个机器人环境中，障碍物只考虑robot_id==0的位置

**根因**：
- `MIN_OBSTACLE_SEP`和`MIN_ROBOT_SEP`设置过小
- `_spawn_random_obstacles()`没有接收其他机器人的位置信息

**修复**：
- 增加安全间距：
  - `MIN_OBSTACLE_SEP`: 0.6m → 0.8m
  - `MIN_ROBOT_SEP`: 1.0m → 1.2m
- 修改方法签名：`_spawn_random_obstacles(other_robot_positions=None)`
- 在reset()中收集所有机器人位置并传递

```python
# 收集所有机器人的spawn位置
all_robot_positions = [(start_x, start_y)]
if other_agent_starts:
    all_robot_positions.extend(other_agent_starts)
self._spawn_random_obstacles(other_robot_positions=all_robot_positions)
```

### 4. 动态障碍物不移动
**问题**：部分动态障碍物spawn后不会移动。

**根因**：
- 环境的`_spawn_random_obstacles()`和`obstacle_mover.py`职责混乱
- 环境spawn动态障碍物到随机位置，但`obstacle_mover`使用固定spawn_points初始化状态
- 两者不同步导致obstacle_mover不知道环境spawn的位置

**修复策略**：
- **明确职责分离**：
  - **环境（`gnn_marl_env.py`）**：只负责spawn **静态障碍物**
  - **obstacle_mover.py**：负责spawn和移动所有 **动态障碍物**

**代码修改**：
```python
# Spawn静态障碍物（环境负责）
for i in range(self.num_static_obstacles):
    spawn_obstacle(f'static_obs_{i}', STATIC_OBS_RADIUS, is_static=True)

# 动态障碍物由 obstacle_mover.py 负责spawn和移动
# 这里只负责将未使用的动态障碍物移到地图外
off_map_x, off_map_y = 100.0, 100.0
for i in range(self.num_dynamic_obstacles, 8):
    req = SetEntityState.Request()
    req.state.name = f'dyn_obs_{i}'
    req.state.pose.position.x = off_map_x + i + 10
    req.state.pose.position.y = off_map_y
    req.state.pose.position.z = 0.4
    req.state.pose.orientation.w = 1.0
    future = self.set_state_client.call_async(req)
    rclpy.spin_until_future_complete(self.node, future, timeout_sec=0.1)
```

### 5. Map 9默认开启随机障碍物
**问题**：需要手动指定`--random_obstacles`才能开启随机spawn。

**修复**：
- Map 9（warehouse_dynamic）和Map 8（circle_swap_arena）默认开启`--random_obstacles`
- 其他地图保持原有行为（需手动指定）

```bash
# 随机障碍物开关：Map 9默认开启，Map 8默认开启
if [[ "$map_num" == "9" || "$map_num" == "8" ]]; then
    if script_supports_arg "--random_obstacles"; then
        cmd+=(--random_obstacles)
    fi
elif (( RANDOM_OBSTACLES == 1 )) && script_supports_arg "--random_obstacles"; then
    cmd+=(--random_obstacles)
fi
```

## 修改的文件

1. **gnn_marl_env.py**
   - 第2887行：清空global_waypoints列表
   - 第3805行：修改`_spawn_random_obstacles()`方法签名
   - 第3835-3836行：增加安全间距参数
   - 第3840-3843行：接收并使用other_robot_positions
   - 第3883-3901行：只spawn静态障碍物，移除动态障碍物spawn逻辑
   - 第3007-3010行：在reset()中传递all_robot_positions

2. **run_curriculum.sh**
   - 第364行：添加`STAGE_STATIC_OBS_NUM`配置
   - 第367-372行：更新stage名称描述
   - 第702行：添加static_obs_num局部变量
   - 第720-722行：添加NUM_STATIC_OBSTACLES_OVERRIDE处理
   - 第767-777行：修改随机障碍物开关逻辑（Map 8/9默认开启）
   - 第747行：更新info输出显示动态和静态障碍物数量

3. **obstacle_mover.py**
   - 第333行：添加`use_current_positions`参数（预留，未启用）

## 测试建议

### 基础功能测试
```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer

# 测试Map 9（warehouse_dynamic）- 2车 + 3动障 + 4静障
./run_curriculum.sh \
  --run_suffix "map9_fix_test" \
  --start_stage 1 --end_stage 1 \
  --train_steps 5000

# 监控指标：
# 1. RViz中路径点是否在每次reset后清空
# 2. 静态障碍物是否保持位置不变
# 3. 动态障碍物是否正常移动（不重叠、不与机器人太近）
# 4. 训练日志中是否有"无法找到有效位置"的警告
```

### 多agent测试
```bash
# 测试6车高密度场景
./run_curriculum.sh \
  --run_suffix "6agent_obstacle_test" \
  --start_stage 3 --end_stage 3 \
  --train_steps 3000
```

### 验证重叠问题
在Gazebo中观察：
1. 静态障碍物之间距离 ≥ 0.8m
2. 静态障碍物与机器人spawn点距离 ≥ 1.2m
3. 动态障碍物由obstacle_mover控制，不会与静态障碍物初始位置重叠

## 预期效果

1. ✅ RViz路径点在每次reset时正确清空
2. ✅ 静态障碍物数量可通过课程配置控制
3. ✅ 障碍物之间保持足够间距（≥0.8m）
4. ✅ 障碍物与机器人保持安全距离（≥1.2m）
5. ✅ 动态障碍物由obstacle_mover统一管理，正常移动
6. ✅ Map 8/9自动开启随机障碍物spawn
7. ✅ 多agent环境中考虑所有机器人位置

## 注意事项

1. **动态障碍物数量**：由`obstacle_mover.py`的`num_obstacles`参数控制
2. **静态障碍物数量**：由环境的`num_static_obstacles`参数控制
3. **职责分离**：
   - 环境只spawn静态障碍物
   - obstacle_mover负责所有动态障碍物的spawn和移动
4. **兼容性**：其他地图（如Map 3、4、5）保持原有行为不变
