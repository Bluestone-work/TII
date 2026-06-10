# Interaction Protocol Redesign

## 这版做了什么

当前训练环境不再只靠连续速度动作和 reward shaping 自己“猜”谁该让、谁该过。

现在的训练链路是：

1. 父环境先同步所有机器人状态。
2. 统一计算局部冲突连通分量。
3. 在每个冲突分量里分配一个 `token owner`。
4. 给每个机器人一个协议模式：
   - `go`
   - `yield`
   - `wait`
   - `backoff`
5. 协议模式再影响：
   - 子目标选择
   - 动作下发前的 interaction shield
   - reward
   - observation

这版的目标不是“纯 end-to-end 学协议”，而是先把窄道会车这种离散协商问题结构化。

## 这算不算 CTDE

严格说，这一版 **不是标准 CTDE**。

标准 CTDE 的定义是：

- 训练时 critic 可以用全局信息。
- 执行时 actor 只能依赖局部观测或可通信信息。
- critic 不直接替 actor 做决策。

而当前版本里，父环境的 interaction manager 实际上已经在训练环境中做了部分高层决策：

- 谁先过
- 谁等待
- 谁后退

所以更准确的定义是：

**带显式协议层的混合系统**

这不是“作弊”，前提是你承认自己的系统是：

- 高层协议/协调器
- 低层 RL 执行器

如果你把它写成“纯去中心化 MAPPO 自发学会了窄道协商”，那就不严谨。

## 能不能往 CTDE 方向收敛

可以，而且这是下一步最合理的方向。

关键点是：

**critic 本身不会直接教会 actor 应该选什么协议模式。**

原因：

- centralized critic 主要学的是 `V(s)` 或 advantage baseline。
- 这类标量监督对“谁让谁、谁等、谁退”这种离散协议太间接。
- 窄道会车又高度对称，credit assignment 很差。

所以如果要做成更像 CTDE 的形式，建议这样走：

### 路线 A：teacher -> student

把当前 interaction manager 当训练期 teacher：

1. 用它生成 `go/yield/wait/backoff` 标签。
2. 给 actor 增加一个 interaction head。
3. 训练时增加辅助损失，让 actor 预测这些模式。
4. 后期逐步降低 teacher 对 subgoal/shield 的直接干预比例。
5. 最终只保留 actor 自己预测的模式。

这是最稳的路线。

### 路线 B：层级 CTDE

显式引入两层策略：

- 高层：离散协议 head，输出 `go/yield/wait/backoff`
- 低层：连续控制 head，执行速度

训练时：

- critic 看全局状态
- actor 看局部观测 + 可通信邻居信息

这样就更接近标准 CTDE。

### 路线 C：纯 critic shaping

只让 centralized critic 学全局交互价值，不增加协议 head。

这条路最不推荐，因为：

- 梯度太弱
- 不稳定
- 很容易重新退回“犹豫、镜像、双向死锁”

## 当前版本的优缺点

### 优点

- 窄道问题被显式建模，不再完全依赖连续动作自己发明协议。
- reward 不再只是在失败后惩罚，而是和协议模式挂钩。
- 会车时的行为可解释，可以直接监控。

### 风险

- 这版改变了 observation 结构，旧 checkpoint 不兼容。
- 当前主要改的是训练环境；部署节点 `robot_policy_node.py` 还没有同步这套协议层。
- 如果直接从高难阶段白板训练，仍然可能不稳定。

## 当前新增的关键监控项

训练时重点看这些：

- `interaction_mode_id`
- `interaction_in_conflict`
- `interaction_has_token`
- `interaction_wait_age_norm`
- `interaction_severity`
- `interaction_partner_dist`
- `interaction_mode_reward`
- `interaction_mode_penalty`
- `head_on_avoidance_reward`
- `corner_escape_reward`
- `reward_risk_gate`

理想现象：

- 冲突时 `interaction_mode_id` 不要每步来回切。
- `interaction_wait_age_norm` 有上升，但不是无限积累。
- `interaction_has_token` 和通过事件能对应起来。
- `interaction_mode_penalty` 不要长期很大。

## 推荐训练方式

这版不要直接 resume 老 checkpoint。

原因：

- `base_safety_feature_dim` 已从 `7` 变成 `14`
- observation 维度已经变了
- 协议逻辑也发生了根本变化

建议先从 **Stage 2** 做一轮新训练，验证协议层是否稳定，再决定是否推进到 Stage 3。

## 推荐启动命令

```bash
cd /home/wj/work/multi-robot-exploration-rl && \
./run_curriculum.sh \
  --model_type gat \
  --gat_actor_graph local_risk \
  --gat_critic_mode mlp \
  --num_agents 4 \
  --num_workers 1 \
  --start_stage 2 \
  --end_stage 2 \
  --train_steps 250000 \
  --train_batch_size 5000 \
  --checkpoint_freq 5000 \
  --action_mode continuous \
  --ppo_profile auto \
  --counterfactual_advantage_coef 0.10 \
  --collision_penalty 25.0 \
  --time_penalty 0.001 \
  --progress_reward_scale 1.20 \
  --path_progress_reward_scale 0.40 \
  --goal_progress_reward_scale 3.0 \
  --goal_reward 24.0 \
  --close_obstacle_penalty_scale 0.12 \
  --close_obstacle_dist 0.50 \
  --predictive_social_penalty_scale 0.08 \
  --predictive_front_penalty_scale 0.10 \
  --social_proximity_risk_scale 0.18 \
  --risk_aware_forward_penalty_scale 0.10 \
  --safe_turn_reward_scale 0.05 \
  --head_on_avoidance_reward_scale 0.25 \
  --team_reward_lambda 0.65 \
  --risk_gate_soft 0.18 \
  --risk_gate_hard 0.65 \
  --avoidance_low_risk_scale 0.12 \
  --navigation_high_risk_scale 0.92 \
  --time_penalty_risk_relax 0.85 \
  --yielding_enable 1 \
  --yielding_soft_dist 0.95 \
  --yielding_stop_dist 0.55 \
  --yielding_hard_stop_dist 0.32 \
  --yielding_ttc 2.6 \
  --yielding_commit_steps 8 \
  --failure_replay_enable 0 \
  --high_conflict_mode mixed \
  --high_conflict_prob 0.35 \
  --rolling_lookahead_dist 0.8 \
  --obstacle_filter_range 1.2 \
  --obstacle_filter_fov_deg 360 \
  --obstacle_top_k 9 \
  --headless_sim \
  --disable_rviz \
  --run_suffix interaction_protocol_v1_stage2
```

## 这条命令的用途

不是为了直接冲最终最优，而是先回答三个问题：

1. 协议模式会不会稳定下来。
2. token 分配会不会减少窄道死锁。
3. `yield / wait / backoff` 是否比旧版更少出现左右犹豫。

如果 Stage 2 曲线稳定，再推 Stage 3。

## 下一步建议

如果这版 Stage 2 验证有效，下一步建议不是继续堆 reward，而是：

1. 给 actor 增加一个显式 interaction head。
2. 让当前 manager 只做 teacher，不再长期直接接管执行。
3. 最终把训练系统从“混合协议层”逐步过渡到“更接近 CTDE 的层级策略”。
