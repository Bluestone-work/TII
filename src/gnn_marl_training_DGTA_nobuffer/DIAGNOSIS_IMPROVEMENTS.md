# 诊断代码改进方案

## 问题背景
当前诊断报告显示"碰撞率 0.05%"，但这可能是**假象**——障碍物稀疏导致没机会碰撞，不代表避障策略好。同时发现"不动比前进奖励高"的磨蹭问题，需要更细致的诊断维度。

## 改进维度

### 1. 风险暴露度量（Risk Exposure Metrics）
**目的**：即使不碰撞，也能量化"危险驾驶"行为。

**新增指标**：
- `risk_time_ratio`: min_dist < 0.3m 的时间占比
- `near_miss_count`: min_dist 从 >0.5m 突降到 <0.3m 的次数（危险接近事件）
- `close_call_severity`: 危险时刻的 min_dist 分布（越接近 0 越严重）
- `safety_margin_histogram`: min_dist 的时间分布直方图（0-0.2, 0.2-0.5, 0.5-1.0, >1.0）

**实现**：在 `collect_episode_diagnostics.py` 记录每步 min_dist，按 episode 聚合统计。

---

### 2. 目标导向行为的时间序列分析
**目的**：量化"磨蹭"问题——是在有效前进还是原地打转？

**新增指标**：
- `dist_to_goal_progress_rate`: Δdist_to_goal / Δt，负值=靠近目标，正值=远离
- `progress_stall_ratio`: |Δdist_to_goal| < 0.05m 的时间占比（卡住不动）
- `heading_alignment_ratio`: heading_error < 30° 的时间占比（朝向目标的时间）
- `velocity_heading_alignment`: v·cos(heading_error)，前进速度在目标方向的投影
- `episode_outcome_distribution`: 
  - 成功到达 (goal_reached)
  - 超时未到达 (timeout, dist_to_goal > threshold)
  - 碰撞失败 (collision)
  - 卡死 (stuck, progress < 0.1m for >100 steps)

**实现**：
- 记录每步的 `dist_to_goal`, `heading_error`（需要从 agent 读取）
- 计算相邻步的差分和相关性
- 按 episode 分类终止原因

---

### 3. 奖励分解的逐步追踪
**目的**：识别奖励函数哪些项在驱动"磨蹭"行为。

**前提**：环境需要在 `info_dict` 里输出奖励分解（当前是空的）。

**新增记录**（需修改 `gnn_marl_env.py`）：
```python
info_dict[aid]['reward_breakdown'] = {
    'progress_reward': ...,
    'heading_reward': ...,
    'goal_reward': ...,
    'collision_penalty': ...,
    'time_penalty': ...,
    'social_risk_penalty': ...,
    'front_risk_penalty': ...,
}
```

**诊断输出**：
- 各项奖励的时间序列轨迹图（看哪些项长期为负）
- 各项占总奖励的平均贡献（饼图）
- "卡住 episode" vs "成功 episode" 的奖励分解对比

---

### 4. 社交行为分析（多智能体交互）
**目的**：看是否存在"死锁"或"过度让行"导致磨蹭。

**新增指标**：
- `min_agent_distance`: 与最近其他 agent 的距离
- `agent_proximity_time`: min_agent_distance < 2.0m 的时间占比
- `yield_events`: 检测"速度突降 + 邻居靠近"的让行行为
- `deadlock_episodes`: 多个 agent 同时 `progress_stall` 的 episode

**实现**：从 `env.robot_positions` 计算 agent 间距离，记录到 JSONL。

---

### 5. 动作分布与策略退化检测
**目的**：看策略是否退化成"确定性不动"。

**新增指标**：
- `action_entropy`: 动作分布的熵（连续动作可用 (v, w) 的方差估计）
- `zero_action_ratio`: |v| < 0.02 且 |w| < 0.05 的时间占比
- `action_heatmap`: (v, w) 的 2D 联合分布热力图

**实现**：记录每步的 `(v, w)`，全局统计分布。

---

### 6. 场景难度分层统计
**目的**：看成功率是否与场景难度相关，还是所有场景都磨蹭。

**分层维度**：
- **初始距离**: start-goal 距离分 3 档（近/中/远）
- **初始拥挤度**: spawn 时 3m 内其他 agent 数量
- **初始障碍物密度**: spawn 时 min_dist（反映周围障碍物密集程度）

**输出**：
- 各层的成功率、平均奖励、平均耗时对比表
- 识别"简单场景也磨蹭" vs "只在难场景磨蹭"

---

## 实现优先级

### P0 - 立即可做（不改环境代码）
1. **风险暴露** - 只需聚合现有 `min_dist` 数据
2. **目标导向时间序列** - 已有 `dist_to_goal`，计算差分即可
3. **动作分布** - 已有 `vel_x, vel_w`
4. **场景难度分层** - 已有 spawn/goal 位置

**改动**：只需修改 `visualize_diagnostics.py`，读取现有 JSONL 做更多分析。

### P1 - 需改环境（提供更多数据）
1. **奖励分解** - 修改 `gnn_marl_env.py` 的 `step()` 在 `info_dict` 里加 `reward_breakdown`
2. **社交行为** - 记录 `min_agent_distance`（需从 `env.robot_positions` 读）
3. **heading_error** - 从 agent 读取朝向偏差（已计算但未记录）

---

## 具体改动文件

### 1. `collect_episode_diagnostics.py`（增强数据收集）
```python
# 在 record 里新增：
'heading_error': float(agent_obj._get_target_angle(agent_obj._get_tracking_target())),  # 已有方法
'min_agent_distance': min([math.hypot(p[0]-pos_x, p[1]-pos_y) 
                            for aid_other, p in env.robot_positions.items() 
                            if aid_other != aid] or [999.0]),
```

### 2. `visualize_diagnostics.py`（新增分析维度）
- 增加 `plot_risk_exposure_timeline()` - 风险暴露时间线
- 增加 `plot_progress_rate_distribution()` - 目标靠近速率分布
- 增加 `plot_action_heatmap()` - (v, w) 热力图
- 增加 `analyze_episode_outcomes()` - 终止原因分类统计
- 增加 `stratify_by_difficulty()` - 按场景难度分层对比

### 3. `gnn_marl_env.py`（奖励分解输出）
在 `step()` 返回前：
```python
info_dict[aid]['reward_breakdown'] = {
    'progress': r_progress,
    'heading': r_heading,
    'goal': r_goal,
    'collision': -penalty_collision,
    'time': -time_penalty,
    # ... 其他项
}
```

---

## 预期效果

改进后的诊断能回答：
1. **避障质量**：即使碰撞率低，风险暴露高 → 策略激进但运气好
2. **磨蹭根因**：progress_stall_ratio 高 + heading_alignment 低 → 原地打转；或 time_penalty 太小 → 不动划算
3. **策略退化**：action_entropy 低 + zero_action_ratio 高 → 退化成"不动"
4. **场景依赖**：只在难场景磨蹭 → 需要失败场景重采样；简单场景也磨蹭 → 奖励函数问题

---

## 下一步

你想：
1. **先做 P0**（不改环境，只改可视化脚本，立即能看到更多分析）
2. **直接上 P1**（同时改环境+采集+可视化，一次性加全）
3. **挑几个最关心的指标**（比如只加风险暴露+目标导向）
