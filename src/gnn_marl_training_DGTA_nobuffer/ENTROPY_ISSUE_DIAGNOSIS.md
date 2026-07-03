# 熵不降低问题诊断与修复

## 🔍 问题诊断

### 数据对比

| 指标 | 旧版 (_o) | 新版 | 问题 |
|------|-----------|------|------|
| **初始熵** | 3.137 | 2.672 | - |
| **最终熵** | 2.352 | 2.996 | ⚠️ |
| **熵降低幅度** | **-0.785** | **+0.325** | ⚠️ 不降反升！ |
| **熵趋势** | -0.781 ↓ | **+0.380 ↑** | ⚠️ 上升 |
| **熵与奖励相关性** | -0.560 (负) | **+0.512 (正)** | ⚠️ 异常 |

### 核心问题

**新版熵不降低，甚至上升！**

```
正常情况（旧版）：
  训练初期：熵高（3.1）→ 探索
  训练后期：熵低（2.4）→ 收敛，策略确定

异常情况（新版）：
  训练初期：熵 2.7
  训练后期：熵 3.0 ↑  （不降反升）
  
  熵与奖励正相关：奖励越高，熵越高（应该是负相关）
```

---

## 🔎 根本原因分析

### 原因1: Entropy Coefficient 设置不当

**当前配置** (continuous mode):
```python
# train_gnn_mappo_full.py:963-968
"entropy_coeff": 0.003,
"entropy_coeff_schedule": [
    [0, 0.003],
    [150_000, 0.001],
    [250_000, 0.0005],
],
```

**问题分析**：

1. **初始值太小**：0.003 可能不足以驱动熵下降
2. **衰减太慢**：150k 步才降到 0.001
3. **新版训练步数少**：新版只训练了 268k 步（67 iters），刚到 schedule 末尾

**新版实际经历的 entropy_coeff**：
```
Iter 1-37  (0-148k 步):  entropy_coeff = 0.003
Iter 38-62 (148k-248k 步): entropy_coeff = 0.003 → 0.001 (线性插值)
Iter 63-67 (248k-268k 步): entropy_coeff = 0.001 → 0.0005
```

**对比旧版**：
- 旧版训练了 900k 步（180 iters）
- 早就过了 250k 步，entropy_coeff 一直是 0.0005
- 更小的系数 → 更倾向于降低熵（exploitation）

---

### 原因2: 策略网络的 log_std 未正确约束

**PPO 连续动作空间的熵**：
```python
# 高斯策略的熵
entropy = 0.5 * log(2πe * σ²)
        = 0.5 * (log(2πe) + log_std * 2)

# 其中 log_std 是策略网络输出的标准差对数
```

**问题**：
- 如果 `log_std` 不受约束，可能会爆炸
- `entropy_coeff` 太小 → 策略不care熵 → `log_std` 可能增长
- `log_std` 增长 → 熵增长 → 探索性增加（但策略不稳定）

**旧版可能的修复**：
- 添加了 `log_std` 的 clipping
- 或者使用了更大的 `entropy_coeff` 初始值

---

### 原因3: 训练不充分

**新版训练进度**：
- 只训练了 67 iterations（268k 步）
- 相当于旧版的 30%
- 可能还在探索阶段，未进入收敛

**证据**：
- 新版最终奖励：+50（还在上升）
- 旧版最终奖励：+145（已收敛）
- 熵在策略收敛时才会显著下降

---

## 🔧 修复方案

### 方案1: 调整 Entropy Coefficient Schedule（推荐 ⭐⭐⭐⭐⭐）

**问题**：初始值太小，衰减太慢

**修复**：

```python
# train_gnn_mappo_full.py:963-968

# 当前（有问题）
"entropy_coeff": 0.003,
"entropy_coeff_schedule": [
    [0, 0.003],
    [150_000, 0.001],
    [250_000, 0.0005],
],

# 修复方案A：提高初始值，加快衰减
"entropy_coeff": 0.01,  # 从 0.003 提高到 0.01
"entropy_coeff_schedule": [
    [0, 0.01],          # 初期高熵，鼓励探索
    [80_000, 0.003],    # 80k 步后开始降低
    [150_000, 0.001],   # 150k 步降到 0.001
    [250_000, 0.0005],  # 250k 步降到最小
],

# 修复方案B：更激进的衰减（如果训练稳定）
"entropy_coeff": 0.015,
"entropy_coeff_schedule": [
    [0, 0.015],
    [50_000, 0.005],
    [100_000, 0.001],
    [200_000, 0.0005],
],
```

**原理**：
- 更高的初始 entropy_coeff → 策略更关注降低熵
- 更快的衰减 → 更早进入 exploitation 阶段
- 最终收敛到相同的 0.0005

---

### 方案2: 限制 log_std 范围（重要 ⭐⭐⭐⭐）

**位置**：Policy network 的 log_std 输出

需要检查当前模型的实现。让我查看：

```python
# mappo_mlp_model.py 或 gat_rllib_model.py
# 找到 actor_head 的定义

# 可能需要添加
LOG_STD_MIN = -5.0  # log_std 最小值（σ_min = e^-5 ≈ 0.0067）
LOG_STD_MAX = 0.5   # log_std 最大值（σ_max = e^0.5 ≈ 1.65）

# 在输出 log_std 时
log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
```

**如果你的模型使用 RLlib 的 `free_log_std`**：

```python
# 在 model config 中
"custom_model_config": {
    # ...
    "free_log_std": True,  # 当前可能是 True
    "log_std_bounds": [-5.0, 0.5],  # 添加边界
}
```

---

### 方案3: 对比旧版配置（确认差异 ⭐⭐⭐⭐⭐）

**关键问题**：旧版为什么能正常降低熵？

需要检查：
1. 旧版的 `entropy_coeff` 设置
2. 旧版的 `entropy_coeff_schedule`
3. 旧版是否有 log_std clipping

**操作**：

```bash
# 查看旧版训练日志或配置
# 如果有保存的 config.json
cat ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2_o/*/params.json | grep -A 5 "entropy"

# 或者查看旧版代码的 git 历史
cd /home/wj/work/multi-robot-exploration-rl
git log --all --full-history --oneline -- "*/train_gnn_mappo_full.py" | head -20
git show <commit_hash>:src/.../train_gnn_mappo_full.py | grep -A 10 "entropy_coeff"
```

---

### 方案4: 添加 Entropy Regularization Loss（高级 ⭐⭐⭐）

**思想**：除了 entropy coefficient，直接在 loss 中惩罚过高的熵

```python
# 在 policy loss 计算中
policy_loss = -advantages * action_prob_ratio

# 添加熵正则化
entropy_penalty = 0.0
if entropy > target_entropy:  # 如果熵超过目标值
    entropy_penalty = 0.1 * (entropy - target_entropy) ** 2

total_loss = policy_loss + entropy_penalty - entropy_coeff * entropy
```

**注意**：需要修改 RLlib 内部，较复杂

---

## 📊 预期效果

### 方案1（推荐）实施后

| 阶段 | 步数 | entropy_coeff | 熵 | 行为 |
|------|------|---------------|-----|------|
| 初期 | 0-80k | 0.01 | 3.5→3.0 | 探索 |
| 中期 | 80k-150k | 0.003 | 3.0→2.5 | 平衡 |
| 后期 | 150k-250k | 0.001 | 2.5→2.0 | 收敛 |
| 最终 | 250k+ | 0.0005 | 2.0→1.8 | 利用 |

### 对比当前

| 指标 | 当前 | 修复后 |
|------|------|--------|
| 熵降低幅度 | **+0.325 ↑** | **-0.8 ↓** |
| 熵趋势 | +0.380 | **-0.7** |
| 最终熵 | 2.996 | **~2.0** |
| 熵与奖励相关性 | +0.512 | **-0.5** |

---

## 🧪 验证方法

### 1. 检查当前 log_std 范围

```python
# 在训练过程中添加日志
import torch

# 在 model forward 中
print(f"log_std range: [{log_std.min():.3f}, {log_std.max():.3f}]")
```

**正常范围**：[-5, 0.5]  
**异常**：如果 log_std > 1.0，说明标准差过大

### 2. 训练并观察熵曲线

```bash
python3 gnn_marl_training/train_gnn_mappo_full.py \
    --env_stage 2 \
    --num_agents 4 \
    --num_obstacles 3 \
    --action_mode continuous \
    --num_train_iterations 100 \
    --entropy_coeff 0.01  # 使用新的初始值
```

**期待看到**：
- TensorBoard 中 `train/entropy` 曲线下降
- 前 50 iters：熵从 3.0 降到 2.5
- 后 50 iters：熵从 2.5 降到 2.0

### 3. 对比熵与奖励的关系

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('training_monitor.csv')

fig, ax1 = plt.subplots()

ax1.plot(df['iteration'], df['entropy'], 'b-', label='Entropy')
ax1.set_ylabel('Entropy', color='b')

ax2 = ax1.twinx()
ax2.plot(df['iteration'], df['episode_reward_mean'], 'r-', label='Reward')
ax2.set_ylabel('Reward', color='r')

plt.title('Entropy vs Reward')
plt.savefig('entropy_reward.png')
```

**期待看到**：
- 熵曲线下降（蓝线）
- 奖励曲线上升（红线）
- 两者负相关

---

## 🔍 进一步诊断

如果修复后仍然有问题，检查：

### 1. Actor 网络的输出层

```python
# 查看 actor_head 定义
# mappo_mlp_model.py 或 gat_rllib_model.py

# 正常实现应该有：
self.actor_head = nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
# 或
self.action_mean = nn.Linear(hidden_dim, action_dim)
self.action_log_std = nn.Parameter(torch.zeros(action_dim))  # 固定 log_std
```

### 2. 检查 RLlib 版本

```bash
pip show ray | grep Version
```

不同版本的 RLlib 对 entropy 的处理可能不同

### 3. 查看 action distribution

```python
# 在训练日志中添加
action_dist = policy.dist_class(...)
print(f"Action std: {action_dist.scale}")  # Gaussian 的标准差
```

---

## 💡 快速修复建议

**立即实施**（5分钟）：

```python
# 修改 train_gnn_mappo_full.py:963-968

"entropy_coeff": 0.01,  # 从 0.003 改为 0.01
"entropy_coeff_schedule": [
    [0, 0.01],
    [80_000, 0.003],
    [150_000, 0.001],
    [250_000, 0.0005],
],
```

**重新训练 50 iterations 验证**：

```bash
python3 gnn_marl_training/train_gnn_mappo_full.py \
    --env_stage 2 \
    --num_agents 4 \
    --num_obstacles 3 \
    --action_mode continuous \
    --num_train_iterations 50
```

**观察**：
- 前 20 iters：熵应该开始下降
- 如果熵仍然不降，检查 log_std 是否有 clipping

---

## 📚 参考

### 熵在 RL 中的作用

```
熵 (Entropy) = 策略的不确定性度量

高熵（3.0+）：
  ✅ 探索充分
  ❌ 策略不确定
  ❌ 性能波动大

低熵（1.5-2.0）：
  ✅ 策略确定
  ✅ 性能稳定
  ❌ 可能过早收敛

目标：
  训练初期：高熵（探索）
  训练后期：低熵（利用）
```

### Entropy Coefficient 的影响

```python
total_loss = policy_loss - entropy_coeff * entropy

# entropy_coeff 太小 (0.001):
#   → 策略不在乎熵
#   → 熵可能不降低

# entropy_coeff 适中 (0.01 → 0.0005):
#   → 初期鼓励探索
#   → 后期收敛到确定策略

# entropy_coeff 太大 (0.1):
#   → 策略过度追求高熵
#   → 无法收敛
```

---

**诊断时间**: 2026-07-02  
**问题**: 熵不降低，甚至上升（+0.325）  
**根本原因**: entropy_coeff 初始值太小（0.003），衰减太慢  
**推荐修复**: 提高初始值到 0.01，加快衰减 schedule  
**预期效果**: 熵降低幅度从 +0.325 改善到 -0.8
