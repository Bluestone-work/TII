# Critic 网络评判机制详解

## 🎯 Critic 的作用

在 PPO（Proximal Policy Optimization）算法中，Critic 网络的作用是：

**评估当前状态的价值** → 用于计算 Advantage（优势函数）→ 指导 Actor 网络学习

---

## 📐 Critic 是根据什么来评判的？

### 核心公式

```python
# Critic 网络输出
V(s) = Critic(global_state)  # 状态价值函数

# 用于计算 Advantage
Advantage = Q(s, a) - V(s)
          ≈ r + γ·V(s') - V(s)  # TD error
```

**Critic 评判依据**：
1. **当前全局状态** `global_state`
2. **未来累积奖励的期望**（通过 TD learning 学习）

---

## 🏗️ 当前 Critic 网络结构

### 输入：全局状态（集中式）

```python
# mappo_mlp_model.py:122-128
global_state_dim = num_agents × base_obs_dim
# 例如：4个机器人 × 每个观测维度 → 拼接成全局状态

# Critic 网络
self.critic_net = nn.Sequential(
    nn.Linear(global_state_dim, hidden_dim),  # 输入：全局状态
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
)
self.critic_head = nn.Linear(hidden_dim, 1)  # 输出：状态价值 V(s)
```

### 架构特点

**集中式 Critic（MAPPO 的核心）**：
- ✅ 输入：**所有智能体的观测拼接**
- ✅ 输出：**全局状态价值 V(s)**
- ✅ 训练时：可以看到全局信息（上帝视角）
- ✅ 执行时：Actor 只用自己的观测（分布式）

**网络深度**：
- 2层 MLP（hidden_dim = 256/512）
- Tanh 激活函数
- 输出层线性（回归问题）

---

## 🔍 Critic 评判的具体依据

### 1. 全局状态包含什么？

```python
global_state = [
    agent_0_obs,  # 机器人0的观测
    agent_1_obs,  # 机器人1的观测
    agent_2_obs,  # 机器人2的观测
    agent_3_obs,  # 机器人3的观测
]

# 每个 agent_obs 包含：
# - 目标方向和距离
# - 当前速度
# - 激光扫描（min_dist, front_min）
# - 邻居信息（其他机器人位置、速度）
# - 障碍物运动特征
# - Agent ID embedding
```

**Critic 能看到的信息**：
- ✅ 所有机器人的位置、速度、目标
- ✅ 所有机器人看到的障碍物
- ✅ 所有机器人之间的相对关系
- ✅ 整个系统的"全局态势"

---

### 2. Critic 如何学习评判？

**训练目标**：最小化 Value Function Loss

```python
# PPO 的 VF Loss
vf_loss = MSE(V(s), target_value)

# target_value 的计算（GAE - Generalized Advantage Estimation）
target_value = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... + γ^n·V(s_{t+n})

# 其中：
# - r_t, r_{t+1}, ... 是真实获得的奖励（环境反馈）
# - γ 是折扣因子（通常 0.99）
# - V(s_{t+n}) 是 n 步后的状态价值估计
```

**学习过程**：
1. Agent 执行动作 → 获得奖励 `r`
2. Critic 预测 `V(s)` 和 `V(s')`
3. 计算 TD error：`δ = r + γ·V(s') - V(s)`
4. 更新 Critic：让 `V(s)` 更接近 `r + γ·V(s')`
5. 随着训练，Critic 学会预测"这个状态未来能获得多少奖励"

---

## ⚠️ Critic 可能的问题

### 问题1: 全局状态维度过大

**当前配置**：
```python
global_state_dim = num_agents × base_obs_dim
# 4个机器人 × ~80维观测 = 320维输入
```

**问题**：
- 维度过高，Critic 难以学习
- 容易过拟合
- 梯度消失/爆炸

**解决方案**：
- ✅ 使用 GNN 聚合全局状态（而非简单拼接）
- ✅ 降低观测维度
- ✅ 使用更深的网络（当前只有2层）

---

### 问题2: 奖励信号稀疏/噪声

**现状**：
```python
# 你的奖励函数
reward = r_progress + r_static + r_ttc + r_collision + r_goal + r_time

# 典型场景：
# - 大部分时间 reward ≈ -0.01 ~ +0.1（稀疏）
# - 碰撞 reward = -60（突变）
# - 到达 reward = +60（突变）
```

**问题**：
- 大部分状态的价值差异很小 → Critic 学不到有效梯度
- 突然的大奖励 → Critic 预测不准 → Value Function Loss 很大

**从你的训练日志看**：
```csv
# GNN_MAPPO_Stage2_Cont_EnvStage2_o/training_monitor.csv
iteration,vf_loss
1,2.91  # 初始
50,~3.0  # 中期
180,9.94  # 最终反而更大！
```

**⚠️ VF Loss 不降反升 = Critic 学习困难！**

---

### 问题3: 集中式 Critic 的信息瓶颈

**当前架构**：
```
全局状态 [320维] → MLP [2层] → Value [1维]
```

**问题**：
- 2层 MLP 可能不足以捕捉 4 个机器人之间的复杂交互
- 简单拼接丢失了空间结构信息
- 无法区分"哪个机器人的状态对全局价值影响更大"

**改进方案**：
使用 **GNN-based Critic**（已有 GAT 模型）：
```
各 agent 观测 → GNN 消息传递 → 聚合全局特征 → Value
```

---

### 问题4: Critic 和 Actor 不匹配

**Actor**：
- ✅ 使用 LSTM（时序信息）
- ✅ 分布式决策（只看自己观测）

**Critic**：
- ❌ 不使用 LSTM（无时序）
- ❌ 简单 MLP（结构简单）

**问题**：
- Actor 能利用时序，Critic 不能 → 评价不准
- Actor 是分布式，Critic 是集中式 → 训练时信息不对称

---

## 🔧 诊断 Critic 问题的方法

### 1. 查看 Value Function Loss 趋势

```python
# 从训练日志中提取
import pandas as pd
df = pd.read_csv('training_monitor.csv')

# 正常情况：VF Loss 应该下降并稳定
# 异常情况：VF Loss 持续上升或剧烈波动

print(df[['iteration', 'vf_loss']].describe())
```

**你的情况**：
```
旧版 (_o):
  iter 1:   vf_loss = 2.91
  iter 180: vf_loss = 9.94  # ⚠️ 上升了 3.4 倍！
```

**结论**：**Critic 确实学习困难！**

---

### 2. 可视化 Value 预测

```python
# 记录一个 episode 的 Value 预测和真实 Return
episode_values = []  # Critic 预测的 V(s)
episode_returns = []  # 实际获得的累积奖励

# 理想情况：两者应该接近
# 实际：如果差距很大 → Critic 预测不准
```

---

### 3. 检查 Advantage 分布

```python
# PPO 使用 Advantage 来更新策略
# Advantage = 实际回报 - Critic 预测

# 正常：Advantage 应该接近 0 均值，小方差
# 异常：Advantage 方差很大 → Critic 预测偏差大
```

---

## 💡 改进 Critic 的建议

### 方案1: 使用 GNN-based Critic（推荐）

**切换到 GAT 模型**：
```python
# 当前使用的是 MLP 模型
MODEL_NAME_MLP = "mappo_mlp_lstm"

# 改用 GAT 模型
MODEL_NAME_GAT = "mappo_gat"

# GAT Critic 结构：
# 1. 各 agent 观测 → Node features
# 2. Graph Attention → 聚合邻居信息
# 3. Readout → 全局 Value
```

**优势**：
- ✅ 捕捉智能体间的交互
- ✅ 参数效率高（共享权重）
- ✅ 可扩展到不同数量的智能体

---

### 方案2: 增加 Critic 网络深度

```python
# 当前：2层 MLP
self.critic_net = nn.Sequential(
    nn.Linear(global_state_dim, hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
)

# 改进：4层 MLP + LayerNorm
self.critic_net = nn.Sequential(
    nn.Linear(global_state_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.Tanh(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh(),
)
```

---

### 方案3: 使用 Dual Critic（对抗训练）

```python
# 训练两个 Critic，取最小值（保守估计）
V(s) = min(Critic1(s), Critic2(s))

# 类似 TD3 的思想，减少过高估计
```

---

### 方案4: 调整 VF Loss 权重

```python
# PPO 的总 Loss
total_loss = policy_loss - entropy_bonus + vf_loss_coeff * vf_loss

# 当前可能 vf_loss_coeff 太小
# 增大它，让 Critic 学习更快

# 在 PPO config 中
"vf_loss_coeff": 1.0  # 默认值，可以尝试 2.0 或 5.0
```

---

### 方案5: 改进奖励塑形

**当前问题**：奖励信号稀疏且突变

**改进**：
```python
# 1. 平滑碰撞惩罚
if collision:
    r_collision = -60  # 太突然
    
# 改为渐进式
r_collision = -60 * collision_severity  # 根据碰撞速度调整

# 2. 增加稠密奖励
r_progress 权重加大（已实施：1.5 → 2.5）

# 3. Clip 奖励范围
reward = np.clip(reward, -10, +10)  # 防止极端值
```

---

## 📊 验证 Critic 改进效果

### 指标1: VF Loss 趋势
```python
# 改进后应该看到：
# - 初始 VF Loss 下降
# - 稳定在较低水平（< 5.0）
# - 不再持续上升
```

### 指标2: Value 预测准确度
```python
# 计算 Value RMSE
rmse = sqrt(mean((V_pred - V_target)²))

# 目标：RMSE < 平均奖励的 50%
```

### 指标3: 策略性能
```python
# 最终目标：更好的 Critic → 更好的 Policy
# 观察 episode_reward_mean 是否提升更快
```

---

## 🎯 总结

### Critic 评判依据
1. ✅ **全局状态**（所有智能体观测拼接）
2. ✅ **历史奖励**（通过 TD learning 学习未来累积奖励）
3. ✅ **训练目标**：最小化预测价值和真实回报的差距

### 你的 Critic 问题
1. ⚠️ **VF Loss 持续上升**（2.91 → 9.94）→ 学习困难
2. ⚠️ **网络太浅**（2层 MLP）→ 容量不足
3. ⚠️ **奖励信号稀疏**（大部分时间 ≈0）→ 梯度弱
4. ⚠️ **信息瓶颈**（简单拼接）→ 丢失结构

### 推荐改进
1. 🔥 **使用 GAT Critic**（最优先）
2. 🔥 **增加网络深度**（4层 + LayerNorm）
3. 🔥 **调整 vf_loss_coeff**（尝试 2.0-5.0）
4. ⭐ **改进奖励塑形**（已部分实施）

---

**分析时间**: 2026-07-02  
**关键发现**: VF Loss 上升 → Critic 学习困难 → 需要改进网络结构  
**下一步**: 切换到 GAT 模型 或 增加 MLP 深度
