# 项目训练监控与观察文档

## 1. 当前实验分支在做什么

当前长训练使用的是一条更接近 MSA3C 思路的分支，但不是论文原样复现。

这轮实验的核心特点：
- 多机器人训练框架仍然是 `MAPPO + CTDE`
- 策略网络采用社交编码 actor + 注意力 critic
- 动作采用 `msa3c_action_mode`，策略直接输出两维控制 `u, v`
  - `u`：线速度命令
  - `v`：角速度命令
- 运行时关闭了大部分底层动作接管
  - `base_shield_enable=OFF`
  - `base_tracking_assist_enable=OFF`
  - `base_hybrid_control_enable=OFF`
  - `local_executor_enable=OFF`
  - `base_zone_manager_enable=OFF`
  - `social_controller_enable=OFF`
  - `zone_reservation_enable=OFF`
  - `yield_action=OFF`
- 这轮训练更接近“纯 MARL 直接学控制”，而不是“规则控制器主导、RL 只做小修正”

## 2. 当前长训练的启动命令

运行名：`intent_mappo_map6_msa3c_uv_long`

核心参数：
- `map_number=6`
- `num_agents=4`
- `train_steps=500000`
- `train_batch_size=4000`
- `rollout_fragment_length=200`
- `communication_range=3.5`
- `msa3c_action_mode=ON`
- `msa3c_social_feature_enable=ON`
- `msa3c_lookahead_steps=4`
- `msa3c_lookahead_dt=0.20`
- `msa3c_lookahead_reward_scale=0.80`
- `msa3c_lookahead_collision_margin=0.34`
- `msa3c_lookahead_comfort_margin=0.65`
- `msa3c_freeze_penalty_scale=0.08`
- `visualization=OFF`
- `intent_visualization=OFF`

## 3. 要看哪些日志

### 3.1 包装日志
路径：
- `/home/wj/work/multi-robot-exploration-rl/train_logs/intent_train_20260323_msa3c_uv_long.log`

用途：
- 看 launcher 是否成功拉起 Gazebo 和 trainer
- 看真实训练日志和环境日志路径

### 3.2 真实训练日志
路径：
- `/home/wj/work/multi-robot-exploration-rl/train_logs/intent_train_20260323_140750.log`

用途：
- 看 Ray/RLlib 是否正常启动
- 看 iteration、reward、checkpoint
- 看有没有异常退出、NaN、worker restart

### 3.3 环境日志
路径：
- `/home/wj/work/multi-robot-exploration-rl/train_logs/intent_env_map6_20260323_140750.log`

用途：
- 看 Gazebo、map_server、robot spawn 是否正常
- 看 ROS/Gazebo warning 是否影响训练

### 3.4 Worker rollout 日志
路径：
- `/home/wj/ray_results/gnn_marl_logs/env_worker1.log`

用途：
- 这是最关键的训练行为日志
- 可以直接看到：
  - reset 是否卡住
  - step 是否在增长
  - 机器人是否真的在运动
  - goal/collision 是否频繁出现
  - episode end 汇总指标

## 4. 最应该盯的指标

### 4.1 训练侧指标
- `reward`：整体是否上升
- `iteration`：训练是否真正往前推进
- `env steps`：采样是否在持续积累
- `checkpoint`：是否按计划保存

### 4.2 行为侧指标
从 `env_worker1.log` 看：
- `goal_events`
- `collision_events`
- `avg_neighbors`
- `isolated_ratio`
- `connected_step_ratio`
- 每个 agent 的 `successes` 和 `collisions`

### 4.3 卡死/冻结相关指标
- `blocking_steps`
- `stall_steps`
- `deadlock_steps`
- `wait_steps`
- `yield_steps`

对当前这轮来说，这些值理论上应接近 0，因为规则式社交控制大多已关闭。如果这些值仍然很高，要检查是否还有残余的控制分支没有关掉，或 reward 设计本身在鼓励保守动作。

## 5. 如何判断训练是不是卡住

### 正常训练的特征
- `env_worker1.log` 不断出现 `step 50 / 100 / 150 ...`
- 训练日志会出现 iteration 和 reward
- Gazebo 中机器人位置会持续变化
- `rolling subgoal` 会随着 step 变化

### 伪运行、实际卡住的特征
- 训练进程还在，但 `env_worker1.log` 只停在 `EPISODE RESET`
- 没有新的 `step 50` 之后的日志
- `cmd_vel` 没有发布
- `rolling subgoal` 固定不动
- 训练日志没有 iteration 输出

## 6. 当前这轮实验已经暴露出的典型问题

### 6.1 训练可以 rollout，但早期碰撞仍然偏多
这是无强制底层控制下的典型现象。

### 6.2 奖励和“真实通行能力”不一定一致
如果 `goal_events` 增长但 `collision_events` 仍然高，说明 reward 还不足以把碰撞压低。

### 6.3 使用差速底盘时，纯 MARL 直接学 `u, v` 比学 holonomic `[vx, vy]` 更难
这也是为什么近年的很多工作会：
- 简化动作空间
- 或把安全约束交给 CBF/MPC/APF/优化器

## 7. 建议的监控命令

查看训练日志尾部：
```bash
tail -n 120 /home/wj/work/multi-robot-exploration-rl/train_logs/intent_train_20260323_140750.log
```

查看 worker 行为：
```bash
tail -n 200 /home/wj/ray_results/gnn_marl_logs/env_worker1.log
```

只看当前 run 的 step 和 episode 汇总：
```bash
rg -n "step|EPISODE END|碰撞|到达目标" /home/wj/ray_results/gnn_marl_logs/env_worker1.log | tail -n 120
```

查看环境启动情况：
```bash
tail -n 120 /home/wj/work/multi-robot-exploration-rl/train_logs/intent_env_map6_20260323_140750.log
```

## 8. 推荐的分析顺序

1. 先看环境日志，确认 spawn 和 Gazebo 正常。
2. 再看 worker 日志，确认 reset 后有没有进入 step。
3. 然后看 episode 汇总，判断 goal/collision 比例。
4. 最后才看 trainer reward，避免被 reward 单独误导。

## 9. 对这个项目下一阶段最重要的判断标准

当前项目不应该只问“reward 有没有涨”，而应该重点问：
- 是否能在交汇区持续通行，而不是冻结
- 是否能在不依赖强保护层的情况下控制碰撞
- 是否能在 map6 这种高交互地图上稳定得到正向通过行为

如果 `goal_events` 增加，但 `collision_events` 同时很高，或者大量 agent 只是撞了再自动 reset，这不应算训练成功。
