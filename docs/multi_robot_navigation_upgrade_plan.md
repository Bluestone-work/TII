# 多机器人避碰改造方案

生成时间：2026-03-22

## 1. 当前项目的核心问题

1. 当前系统的全局层和局部层职责混杂。
2. 强化学习、tracking assist、shield、yield、reservation 都在改动作，语义冲突。
3. 全局路径当前只在 reset 时规划，运行中不会因为动态障碍物或其他机器人重新规划。
4. 现有 PPO 更容易学到“靠近就减速/停住”，而不是稳定的绕行、通过、交汇礼让。

## 2. 总体目标架构

推荐改成三层：

- 全局路径层：负责从静态地图生成全局可行路径，并在局部被动态障碍堵塞时在线重规划。
- 局部避碰执行层：负责把 rolling subgoal 变成真正可执行的 v,w，并对静态/动态障碍进行解析式避让。
- 高层交互策略层：负责谁先走、谁让行、往哪边绕、何时请求通行权。

## 3. 推荐的改造顺序

### 阶段 A：先做可靠的动态全局重规划

目标：让 global planner 不再是一次性 A*，而是能在运行中根据 scan 和邻居位置重规划。

本轮已完成：

- 新增 D* Lite 风格的动态重规划后端。
- 当动态障碍落到当前未来路径上时，触发在线重规划。
- 动态障碍来源包括：激光雷达新出现的占据点、其他机器人当前位置。
- 如果增量解失败，自动回退到动态占据图上的 A* 搜索，保证可用性。

本轮涉及文件：

- `src/intent_marl_training/intent_marl_training/global_planner.py`
- `src/intent_marl_training/intent_marl_training/gnn_marl_env.py`

### 阶段 B：新增 zone / deadlock manager

目标：不要再让 pairwise yield 自己博弈路口通行权。

计划新增模块：

- `zone_manager.py`

职责：

- 识别 doorway、corridor、intersection、merge 区域
- 给出 owner / waiter / queue rank / hold point / release rule
- 统计 blocking、stall、deadlock、flow rate

### 阶段 C：新增 local avoidance executor

目标：把局部避碰从“纯 RL 学 v,w”改成“解析式执行器主导”。

计划新增模块：

- `local_avoidance.py`

建议包含：

- goal attract
- static obstacle repulse
- neighbor repulse
- tangential side-pass
- zone bias
- differential-drive projection

### 阶段 D：把 RL 改成高层交互策略

RL 不再直接输出最终 v,w，改成输出高层动作：

- speed_scale
- yield_commit
- side_bias
- assertiveness

或者离散动作：

- go
- yield
- hold
- pass_left
- pass_right

## 4. 建议的数据流

1. Global planner 给出 global path
2. Rolling subgoal 从 global path 提取局部目标
3. Zone manager 判断当前是否处在冲突区
4. Social policy 产生高层交互指令
5. Local avoidance executor 结合 subgoal、激光、邻居、zone 决策生成最终 v,w
6. Emergency brake 只做最后兜底

## 5. 需要新增的观测

建议 RL 增加以下高层交互特征：

- dist_to_conflict_point
- time_to_conflict_point
- zone_id
- queue_rank
- is_zone_owner
- reservation_age
- blocked_steps_norm
- stall_steps_norm
- left_free_space
- right_free_space
- neighbor_ttc_topk
- neighbor_heading_conflict_topk

## 6. 不建议继续走的方向

1. 不建议继续只调 reward 权重。
2. 不建议继续让一个 policy 同时面对 residual、tracking assist、yield override、shield override。
3. 不建议继续让 PPO 直接从 raw lidar 学出多机器人路口规则。

## 7. 推荐的训练路线

### 7.1 先做无 RL baseline

先用：

- 动态全局重规划
- zone manager
- local avoidance executor

目标是先让系统在 map6 上：

- 不全员卡死
- 遇到冲突能稳定排队或右绕通过
- collision 显著下降

### 7.2 再做 imitation

用解析式控制器生成数据，训练高层 social policy 做 imitation pretrain。

### 7.3 最后再 PPO fine-tune

PPO 微调的对象不再是低层控制，而是高层交互决策。

## 8. 推荐验收指标

不要只看 reward。每个 episode 建议强制记录：

- goal_events
- collision_events
- blocking_steps
- stall_steps
- deadlock_steps
- zone_entries
- queue_wait_time_mean
- handoff_count
- specific_flow_rate
- fairness_index
- owner_progress_mean
- global_replan_count
- global_last_replan_reason

## 9. 本轮代码改动摘要

### 已落地

1. `global_planner.py`
- 新增 `DStarLitePlanner`
- 支持动态障碍物占据叠加
- 支持在线重规划
- 增量解失败时回退到动态占据图上的 A*

2. `gnn_marl_env.py`
- 默认 planner 由一次性 A* 扩展为可选 `dstar_lite`
- 每个 agent 接收其他机器人位置，构建动态障碍输入
- 用激光雷达和邻居位置构建动态障碍世界坐标点
- 当动态障碍堵住未来路径段时触发重规划
- 在 info 中记录重规划计数和最近一次重规划原因

### 下一步立即建议做的事情

1. 新建 `zone_manager.py`
2. 新建 `local_avoidance.py`
3. 把 `intent_env_wrapper.py` 从“动作改写器”改成“高层社交策略器”
4. 做一个无 RL 的 map6 baseline

## 10. 结论

下一步最正确的方向不是继续调 PPO 直接学 v,w，而是：

- 先把全局层做成动态重规划
- 再把局部层做成解析式避碰执行器
- 最后再让 RL 学高层交互

这条路线更容易真正做出“多机避碰”，而不是“多机一起停住”。
