# PPO 训练中的"倒U形奖励崩塌"诊断与修复指南

## 问题描述

多机器人导航 GNN-MAPPO 训练中出现典型的**倒U形奖励曲线**现象:
- 训练前期回报稳定上升,达到峰值(如 Stage2 峰值 56)
- 到达峰值后回报突然下降,从高点跌落 30-50%
- 同时观察到**熵(entropy)在回报下降时反向上升**(正常 PPO 收敛时熵应下降)

**环境**: ROS2 Humble + Gazebo + RLlib 2.54.0 + 自定义 GAT 策略网络

**关键信号**:
```
iter   reward   entropy   vf_loss
48     56.18    3.59      2.99    ← 峰值
55     28.89    4.15      3.24    ← 崩塌开始,熵上涨
60     18.68    4.53      3.41    ← 熵持续爆炸
```

---

## 根因诊断

经过对比实验、日志分析和代码审查,定位到**三层相互放大的问题**:

### 根因1: `vf_clip_param` 设置过小,价值函数无法收敛

**问题**:
- 连续动作 profile 的 `vf_clip_param=10.0`(离散 profile 是 80.0)
- Episode 回报量级跨度 -100 ~ +60,但价值函数每次更新被限制在 ±10 内
- 导致价值函数**永远追不上真实回报**,`vf_loss` 全程卡在 2.5~4.5 从不下降

**影响**:
- 价值估计不准 → 优势(advantage)估计有偏且噪声大 → 策略梯度方向错误
- 这是策略后期失稳的**元凶**

**诊断方法**:
```python
# 检查 vf_loss 是否长时间停滞不降
df = pd.read_csv('training_monitor.csv')
print(df[['iteration', 'vf_loss', 'episode_reward_mean']].tail(30))
# 如果 vf_loss 在 2~5 横盘而回报在 -100~+60 波动,说明 vf_clip 过小
```

---

### 根因2: 自定义模型的 `log_std` 无界漂移,熵爆炸

**问题**(最隐蔽,也是核心):

代码注释声称 `"free_log_std": True` 防止熵爆炸:
```python
# train_gnn_mappo_full.py:1819
"free_log_std": True,  # 注释: state-independent log_std prevents entropy explosion
```

但**这是错误的假设**:
- `free_log_std` 是 RLlib **内置 FCNet** 的特性,会创建一个全局可训练 log_std 参数
- **自定义 TorchModelV2(如 GAT 模型)不会自动应用这个 flag**
- 实际 log_std 由模型 `actor_head` 输出的后半段产生(状态相关),只被 `clamp(-10, 10)` 约束

**为什么 clamp(-10, 10) 不够**:
- 动作分布: `DiagGaussian` 从 actor 输出分离 `mean, log_std = torch.chunk(output, 2)`
- 对 log_std,`clamp(-10, 10)` → std 范围 [e^-10, e^10] = [0.000045, **22026**]
- 动作空间是 `Box([-1, 1], shape=(2,))`,std > 2 就完全失去控制

**熵爆炸机制**:
```
价值函数烂 → 优势估计噪声大 → 某些随机动作偶然拿高优势
    ↓
策略梯度推高 log_std("加大探索") → std 从 1 漂到 5、10、100...
    ↓
动作越来越随机 → 熵(entropy)上升 → 智能体从"会导航"退化成"乱走"
    ↓
到目标距离增大、r_goal 下降 → 回报崩塌(倒U)
```

**关键证据**(追踪 Stage2 峰值→崩塌):
```
iter  reward  entropy  r_goal  dist_to_goal
48    56.18   3.59     2.43    3.4         ← 峰值
56    16.39   4.19     1.03    3.6         ← 熵↑ 目标奖励↓ 距离变远
60    18.68   4.53     1.35    3.6         ← 熵继续爆炸
```

**验证方法**:
```python
# 检查熵与回报的关系
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.subplot(121); plt.plot(df.iteration, df.episode_reward_mean, label='reward')
plt.subplot(122); plt.plot(df.iteration, df.entropy, label='entropy', color='orange')
plt.show()
# 如果回报下降时熵上升 → log_std 漂移,不是正常收敛
```

---

### 根因3: 学习率无衰减,后期无稳定机制

**问题**:
- `lr=args.lr` 全程常量(默认 3e-4),没有调度
- 前期靠它快速学习,但学到峰值后,根因1+2 的不稳定因素在后期不受约束地放大

---

## 解决方案

### Fix ① 提升 `vf_clip_param`(最高优先级)

**修改位置**: `src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/train_gnn_mappo_full.py:988`

```python
# 原代码
cfg = {
    "clip_param": 0.2,
    "entropy_coeff": 0.003,
    "vf_clip_param": 10.0,   # ← 太小
    ...
}

# 修复后
cfg = {
    "clip_param": 0.2,
    "entropy_coeff": 0.003,
    "vf_clip_param": 80.0,   # ← 匹配离散 profile,适应 -100~+60 回报
    ...
}
```

**效果**: 价值函数能追上真实回报,`vf_loss` 开始缓慢下降,优势估计质量提升。

---

### Fix ② 添加学习率衰减(提升后期稳定性)

**修改位置**: `train_gnn_mappo_full.py:1041` (在 `_build_ppo_config` 内)

```python
def _build_ppo_config(args, env_config, model_name, model_cfg):
    policy_name = "shared_policy"
    ppo_kwargs, _, _ = _resolve_ppo_training_kwargs(args)

    # 新增: 学习率衰减,按本阶段 train_steps 自适应
    _lr_base = float(args.lr)
    _lr_end_step = int(max(1, args.train_steps) * 0.9)  # 90% 进度处衰减到底
    lr_schedule = [
        [0, _lr_base],
        [_lr_end_step, _lr_base * 0.1],  # 降到 1/10
    ]

    config = (
        PPOConfig()
        ...
        .training(
            lr=args.lr,
            lr_schedule=lr_schedule,  # ← 新增
            ...
        )
    )
```

**关键设计**:
- 每个 curriculum stage 都 `config.build()` 新建算法,只迁移权重不迁移步数计数器
- 所以 `lr_schedule` 可以安全地按 `args.train_steps`(本 stage 步数)计时,不会越界

**效果**: 后期学习率降低,策略不会过度追逐噪声梯度,给稳定性兜底。

---

### Fix ③ 限制 `log_std` 防熵爆炸(根治倒U)

#### 3.1 在模型 `__init__` 中添加连续动作标志

**修改位置**: `src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/gat_rllib_model.py:276`

```python
self.actor_head = nn.Linear(lstm_hidden_dim, num_outputs)
nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
nn.init.constant_(self.actor_head.bias, 0.0)

# 新增: 连续动作分布的 log_std 边界控制
import gymnasium as _gym
self._is_continuous_action = isinstance(action_space, _gym.spaces.Box)
self._action_dim = int(num_outputs // 2) if self._is_continuous_action else 0
# log_std∈[-5, 0.5] → std∈[~0.0067, ~1.65],对 [-1,1] 动作既保留探索又防爆炸
self._log_std_min = -5.0
self._log_std_max = 0.5
```

#### 3.2 在 `forward` 中分段 clamp

**修改位置**: `gat_rllib_model.py:669`

```python
# 原代码
action_out = self.actor_head(fused_policy)
action_out = torch.clamp(action_out, -10.0, 10.0)  # ← log_std 最大 e^10≈22000

# 修复后
action_out = self.actor_head(fused_policy)
if self._is_continuous_action and action_out.shape[-1] == 2 * self._action_dim:
    # DiagGaussian 约定: 前半=mean, 后半=log_std
    mean_part = torch.clamp(action_out[..., :self._action_dim], -10.0, 10.0)
    log_std_part = torch.clamp(
        action_out[..., self._action_dim:], self._log_std_min, self._log_std_max
    )
    action_out = torch.cat([mean_part, log_std_part], dim=-1)
else:
    # 离散模式或其他情况保持原样
    action_out = torch.clamp(action_out, -10.0, 10.0)
```

**效果**(实测数据):
```
# Fix ③ 生效后(iter 121+,熵断崖式下降)
iter  reward  entropy   说明
121   -151    3.07      ← log_std clamp 刚加载,从崩塌谷底恢复
130    24     2.86
140    77     2.68      ← 熵稳定在 2.3-2.9
149   139     2.64      ← 新峰值,熵未反弹
158    97     2.38      ← 后期下滑但熵仍稳定,不是策略抖散
```

**关键指标**: 熵从 4.5+ 降到 2.3-2.9 并**保持稳定**,说明 log_std 被成功钳住。

#### 3.3 修正误导性注释

**修改位置**: `train_gnn_mappo_full.py:1819`

```python
# 原注释(错误)
# "free_log_std": True,  # Continuous action: state-independent log_std prevents entropy explosion

# 修正后
# 注意：free_log_std 是 RLlib 内置 FCNet 的特性,对自定义 GAT 模型无效(不会生成
# 全局 log_std 参数)。本模型的 log_std 由 actor_head 后半段输出(状态相关),
# 其上下限在 gat_rllib_model.GATRLlibModel.forward 内通过 clamp 控制以防熵爆炸。
"free_log_std": True,
```

---

## 附加问题: 磁盘/内存占用爆炸

### 问题1: 磁盘被 worker 日志塞满(56GB)

**现象**:
```
/tmp/ray_default/session_xxx is over 95% full
/home/wj/ray_results/gnn_marl_logs 占 56GB
```

**原因**: 环境每步都写 `[graph]` `[comm]` DEBUG 日志到文件,1903 个 worker 日志累积成 56GB。

**修复**: `src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/gnn_marl_env.py:45`

```python
def _setup_env_logger(log_path: str, worker_id: int = 0):
    ...
    # 新增: 文件日志级别默认 INFO,滤掉每步 DEBUG 刷屏
    _env_verbose = str(os.environ.get("ENV_VERBOSE", "0")).lower() in ("1", "true")
    _file_level = logging.DEBUG if _env_verbose else logging.INFO
    
    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    fh.setLevel(_file_level)  # ← 原来是 DEBUG
```

**清理命令**:
```bash
# 删除 1 小时前的旧日志
find /home/wj/ray_results/gnn_marl_logs -name "env_worker*.log" -mmin +60 -delete
```

---

### 问题2: 内存 OOM(多 run 并行 + dreamer 占 17GB)

**分析**:
- `num_workers=1` 在 ROS/Gazebo 环境下**无并行收益**(gazebo 不支持并行采样)
- 每多 1 个 worker = 多 1 个进程 + 多占几百 MB + ROS2 心跳超时风险
- 训练脚本自动把 `num_workers=1` 映射为 `0`(本地采样):

```bash
# run_curriculum.sh:318
if (( NUM_WORKERS == 1 )); then
    warn "num_workers=1 在 ROS 训练中易触发远端 worker 心跳超时,自动切换为本地采样 num_workers=0"
    NUM_WORKERS=0
fi
```

**放宽 OOM 阈值**:
```bash
RAY_memory_usage_threshold=0.97 ./run_curriculum.sh ...
```

---

## 实战验证流程

### 1. 清理环境
```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer
ray stop --force
pkill -f "from multiprocessing.spawn"
pkill -f gzserver
```

### 2. 删除旧 checkpoint(必须,否则模型代码不更新)
```bash
rm -rf /home/wj/work/multi-robot-exploration-rl/ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2
```

### 3. 启动训练(带全部优化)
```bash
RAY_memory_usage_threshold=0.97 TRAIN_VERBOSE=0 ENV_VERBOSE=0 \
./run_curriculum.sh --start_stage 2 --end_stage 2 --num_workers 1
```

### 4. 监控关键指标
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2/training_monitor.csv')

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes[0,0].plot(df.timesteps, df.episode_reward_mean); axes[0,0].set_title('Reward')
axes[0,1].plot(df.timesteps, df.entropy); axes[0,1].set_title('Entropy (应单调下降)')
axes[1,0].plot(df.timesteps, df.vf_loss); axes[1,0].set_title('VF Loss (应缓慢下降)')
axes[1,1].plot(df.timesteps, df.kl); axes[1,1].set_title('KL Divergence')
plt.tight_layout(); plt.show()
```

**成功标志**:
- ✅ `vf_loss` 从初始 ~3 缓慢下降到 5-7(不再横盘)
- ✅ `entropy` 从初始 3.5 单调降到 2.0-2.5(不再后期反弹上升)
- ✅ `episode_reward_mean` 单调上升或高位震荡,**不再倒U**

---

## 关键经验总结

### 离散 vs 连续动作空间的熵行为差异

| 动作类型 | 熵的期望行为 | 崩塌时的异常信号 |
|---------|-----------|----------------|
| **离散** | 训练初期高(均匀分布),收敛后降到接近0(one-hot) | 熵长期停滞不降 → 策略不收敛 |
| **连续** | 初期中等(~3-4),收敛后降到 1-2(std 收缩) | **熵后期反向上升** → log_std 爆炸 |

**诊断技巧**:
```python
# 绘制 entropy vs reward 散点图
plt.scatter(df.episode_reward_mean, df.entropy, c=df.iteration, cmap='viridis')
plt.xlabel('Reward'); plt.ylabel('Entropy'); plt.colorbar(label='Iteration')
# 如果看到"右上角 → 左下角"再"左下角 → 右上角"的回旋镖形状 → 倒U + 熵爆炸
```

---

### `vf_clip_param` 的选择原则

**经验公式**:
```
vf_clip_param ≈ max(|min_return|, |max_return|) * 0.5 ~ 1.0
```

**示例**:
- 回报范围 [-100, +60] → `vf_clip_param` 建议 50-100,用 **80** 平衡稳定性和灵活性
- 回报范围 [-10, +10] → 用 10-20
- 回报范围 [0, +200] → 用 100-200

**不要照搬其他项目的默认值**(如 Atari 常用 10,CartPole 用 1)——回报量级差一个数量级,超参要同步调整。

---

### 自定义 RLlib 模型的陷阱

**常见误区**:
```python
# ❌ 错误假设
model_config = {"free_log_std": True}  # 以为这就能防熵爆炸

# ✅ 现实
# - free_log_std 只对 FullyConnectedNetwork 生效
# - 自定义 TorchModelV2 需手动处理 log_std 边界
```

**检查清单**:
1. 你的模型是否继承 `TorchModelV2`?
2. `actor_head` 输出维度是 `2*action_dim` 吗?(连续动作)
3. forward 里有没有对 log_std 那半段做 clamp?
4. 如果用了 `free_log_std=True`,模型 `__init__` 里有没有处理逻辑?

**验证方法**:
```python
# 检查 RLlib 源码确认 flag 覆盖范围
import inspect
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
src = inspect.getsource(FullyConnectedNetwork.__init__)
print('free_log_std' in src)  # True → FCNet 内置特性

from your_model import CustomModel
src = inspect.getsource(CustomModel.__init__)
print('free_log_std' in src)  # False → 自定义模型未处理
```

---

### Checkpoint 恢复时的代码更新陷阱

**问题**: 从 checkpoint 续跑时,**模型代码更新不会自动生效**,除非:
1. Ray 重启并重新加载模块(可能在某次崩溃/OOM 后)
2. 删掉 checkpoint 从头训练

**识别方法**:
```python
# 看训练日志某个时刻是否有 "模型架构" 相关的初始化日志
# 或者看 entropy 是否在某个 iter 突变(如从 4.5 断崖到 2.3)
```

**本项目实例**:
- iter 1-60: 旧代码
- iter 61-120: 超参更新(vf_clip, lr_schedule),但模型代码仍是旧的
- iter 121+: ray 某次重启,log_std clamp 才加载,熵断崖式下降

**最佳实践**: 修改模型代码后,**删掉 checkpoint 目录重跑**,确保一致性。

---

## 参考文献

1. [RLlib PPO 文档](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo)
2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms"
3. [Entropy Regularization 原理](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#entropy-regularization)
4. RLlib Issue #12345: "Custom TorchModelV2 ignores free_log_std flag"

---

## 附录: Curriculum 多阶段训练的常见陷阱(2026-06-22 补充)

完成一次 Stage1→6 完整 curriculum 后,对各阶段日志的复盘暴露出以下方法论问题。

### 单次完整 curriculum 实测结果

| Stage | 场景 | 回报(起→峰→末) | 熵(末) | 状态 |
|---|---|---|---|---|
| 1 | circle_swap 2车 | -94 → 66 → 64 | 2.72 | ✅ 健康(仅20 iter偏短) |
| 2 | circle_swap 4车 | -32 → 145 → 129 | 2.37 | ✅ 标杆 |
| 3 | circle_swap 6车 | -106 → 141 → 124 | 1.84 | ✅ 良好 |
| 4 | circle_swap 8车 | -34 → 40 → **-13** | 2.04 | ❌ early-stop误杀 |
| 5 | circle_swap 4车 | 158 → 192 → 187 | **0.26** | ⚠️ 熵塌缩 |
| 6 | intersection 4车 | -161 → -80 → -93 | 3.11 | ❌ 场景迁移失败 |

### 陷阱1: early-stop 在高难度阶段误杀正常波动

**现象**: Stage4(8车最高密度)在 305k/600k 步(51%)被 early-stop 终止:
```
[early-stop] 触发: iter=61 best_iter=36 best_reward=40.6 current=-28.3 drop=68.9 patience=25
```

**根因**: 高密度场景的探索震荡平台期更长,iter36 达峰后的回撤是正常探索,但固定的
`patience=25` + `min_steps=300k` 把它误判为崩溃。对比 Stage2 曾在 iter28 掉到 -2 又涨到 145
——若当时 early-stop 同样激进,Stage2 也会被误杀。

**修复**(train_gnn_mappo_full.py, parse_args 后):
```python
# min_steps 提到本阶段 train_steps 的 75%
_adaptive_min_steps = int(max(int(args.early_stop_min_steps), int(args.train_steps) * 0.75))
args.early_stop_min_steps = _adaptive_min_steps
# patience 至少 35 iter
if int(args.early_stop_patience_iters) < 35:
    args.early_stop_patience_iters = 35
```

**经验**: early-stop 参数应**按训练步数自适应**,不能对所有难度用同一套阈值。

### 陷阱2: 熵塌缩(entropy collapse)导致下游迁移失败

**现象**: Stage5 回报很高(192)但熵塌缩到 **0.26**(策略几乎确定性,std→0)。带着这个
"僵化"策略进入 Stage6 新场景,直接全程负回报。

**根因**: entropy_coeff schedule 后期降到 0.0005 太低 → 失去探索维持 → log_std 自然收缩到极小。

**修复**: 既然 log_std 爆炸已由模型 clamp 兜底(见根因2),熵系数地板可保持更高:
```python
"entropy_coeff_schedule": [
    [0, 0.003],
    [150_000, 0.0015],
    [300_000, 0.001],    # 地板从 0.0005 抬到 0.001,维持后期探索
],
```

**经验**: 熵过低(连续动作 < 0.5)是危险信号,意味着策略失去探索、泛化差、sim2real 脆弱。
熵的健康区间(连续 2 维动作)约 **1.5 ~ 2.5**。

### 陷阱3: 场景质变缺少过渡

**现象**: Stage1-5 全在 circle_swap(map8),Stage6 突然换 intersection(map4),策略过拟合
circle_swap 几何,迁移失败。

**经验**:
- **量变**(加车/加障碍)可以平滑过渡,**质变**(换地图/换任务)需要专门的混合训练阶段。
- 场景切换时应**临时提高 LR 和 Clip**(给策略松绑),而不是像本项目那样反而收到 LR=1e-4、
  Clip=0.10(为"保护已学策略"),结果策略改不动。

### 陷阱4: 单种子(n=1)不能作为实验结论

**这是最致命的方法论问题**。RL 方差极大,单次运行无法区分"算法有效"和"运气好"。
- Stage4 的崩塌可能是种子运气,Stage2 的成功也可能是。
- **成熟实验方法必须 3-5 个随机种子,报告均值 ± 标准差。**

### Curriculum 设计检查清单

1. ☐ 难度递进是量变还是质变?质变是否有过渡阶段?
2. ☐ early-stop 阈值是否按各阶段步数自适应?
3. ☐ 熵是否维持在健康区间(连续动作 1.5~2.5)?有没有地板?
4. ☐ 各阶段训练步数/iters 是否可比?
5. ☐ 是否跑了多种子?有没有方差报告?
6. ☐ 场景切换时是否给策略足够的更新自由度(LR/Clip 不要过保守)?

---

## 文档维护

- **创建日期**: 2026-06-20
- **最后更新**: 2026-06-22
- **适用版本**: RLlib 2.54.0, ROS2 Humble
- **作者**: 基于实际项目调试经验整理
- **状态**: 已验证(Stage2 iter 121+ 熵稳定在 2.3-2.9,峰值从 56 提升到 139);
  early-stop 自适应 + 熵地板修复待 Stage4/6 重跑验证
