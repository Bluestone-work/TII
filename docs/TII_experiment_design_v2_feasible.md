# TII 论文实验设计 v2.0 (基于真实可行性重构)

> **状态**: 第一版设计(TII_experiment_design.md)被发现严重不足——缺传统/SOTA对比、消融不到位、无统计分析、无测试集评估。
> 本版基于 agent 的基础设施盘点,给出**可执行、工作量透明、分优先级**的完整方案。

---

## 零、第一版的致命缺陷(已停止那个P0队列)

| 缺陷 | 第一版P0 | 应有标准 |
|---|---|---|
| **baseline对比** | 只有 MLP vs GAT(自己的两个版本) | 需 ORCA/DWA(传统)、IPPO/MAPPO(学习类)、Comm-GAT(同类) |
| **消融** | 只消图结构(3变体) | 5个创新点,应消融门控、风险偏置、Gap/Yield/Detour/Replan |
| **参数敏感性** | comm_range 3点×1种子 | 4参数×5点×3种子,含统计 |
| **评估** | 训练reward曲线 | held-out测试集的成功率/碰撞率/死锁率/makespan |
| **统计** | 单种子,看趋势 | 多种子均值±std、显著性检验(t-test/Wilcoxon)、置信区间 |

agent盘点结论:**ORCA/DWA/IPPO 能改造对接(7-10天),MAPPO几乎可用(2天),MATD3要重写(15天,可放弃)**。

---

## 一、重构后的实验优先级(P0→P2,按审稿人视角)

### P0 — 必做(审稿人拒稿门槛)
1. **对比实验**(Ours vs 传统+学习类baseline,5个方法)
2. **核心创新消融**(5个创新点逐个关,验证贡献)
3. **测试集评估**(held-out场景,成功率/碰撞率/makespan等6指标)
4. **多种子统计**(3种子均值±std,显著性检验)

### P1 — 推荐做(增强说服力,差异化)
5. **参数敏感性分析**(3-4关键参数,各5点×3种子)
6. **泛化性测试**(跨场景、跨密度、动态障碍鲁棒性)
7. **计算开销对比**(推理时延、参数量、训练时间)

### P2 — 加分项(锦上添花)
8. **SPL/路径效率**分析
9. **通信质量鲁棒性**(dropout/latency)
10. **真实机器人部署**(若有条件)

---

## 二、P0 对比实验设计(重构版)

### 2.1 对比方法清单(5个,覆盖传统+学习)

| 方法 | 类型 | 现状 | 改造工作量 | 优先级 |
|---|---|---|---|---|
| **ORCA** | 传统解析式 | 能改造对接 | 3天(包装+评估脚本) | P0 |
| **DWA** | 传统解析式 | 能改造对接 | 同ORCA | P0 |
| **IPPO** | 学习类(independent) | 能改造对接 | 5天(env对齐+训练) | P0 |
| **MAPPO** | 学习类(集中critic) | 几乎可用 | 2天(验证+测试) | P0 |
| ~~MATD3~~ | 学习类(off-policy) | 要重写 | 15天 | ❌ 放弃(性价比低) |
| **Ours(MLP)** | 主干策略 | 现成 | 0天 | P0 |
| **Ours(Full)** | 完整方法 | 现成 | 0天 | P0 |

**决策**: 放弃MATD3(重写成本15天,off-policy与我方法路线差异大,对比意义有限)。保留ORCA/DWA/IPPO/MAPPO,覆盖传统+学习两类,足够充分。

### 2.2 对比维度与指标

#### 定量指标(测试集,50 episodes × 3 seeds)
- **成功率** (goal reached without collision/timeout)
- **碰撞率** (collision with obstacles or agents)
- **死锁率** (no progress for N consecutive steps)
- **平均完成步数** (makespan,成功的episode)
- **平均最小间距** (safety,全episode)
- **推理时延** (ms/step,计算开销)

#### 定性分析
- 典型场景轨迹对比(会车、交叉、死锁恢复)
- 失败case分析(各方法死在哪)

### 2.3 测试场景(held-out,不用于训练)
- **circle_swap 4车**(训练见过的场景,测收敛性)
- **intersection 4车**(Stage6场景,测泛化)
- **fixed_benchmark_scenarios.py** 的 3 个场景(warehouse_aisles / hallway_cross / dense_6agents)

---

## 三、P0 核心创新消融设计(重构版)

5个创新点,逐个关掉,对比完整方法。**每个消融在 Stage2(4车) 训练到收敛,在测试集评估**。

| 消融ID | 创新点 | Full配置 | 消融配置(关掉该项) | 开关方式 | 改造成本 |
|---|---|---|---|---|---|
| **Abl-1** | 双图融合 | dual_graph | social_only / obstacle_only | `--graph_ablation` | 0天(现成) |
| **Abl-2** | 关系解耦架构 | GAT分支 | 纯MLP主干 | `--model_type mlp` | 0天(现成) |
| **Abl-3** | 后置残差门控 | residual_gating | direct_add / no_gate | `--fusion_mode` | 0.5天(加开关) |
| **Abl-4** | 风险偏置注意力 | gat_risk_bias_scale=2.5 | =0.0 | 现有参数 | 0天(现成) |
| **Abl-5a** | Gap特征 | gap_feature_enable=1 | =0 | 现有参数 | 0天(现成) |
| **Abl-5b** | Yielding让行 | yielding_enable=1 | =0 | 现有参数 | 0天(现成) |
| **Abl-5c** | Detour绕行 | (默认开) | detour_enable=0 | `--detour_enable` | 0.5天(加开关) |
| **Abl-5d** | 动态重规划 | (默认开) | dynamic_replan=0 | `--dynamic_replan_enable` | 0.5天(加开关) |

**注**: Abl-5(Gap-Yield-Detour-Replan)是4个子机制,可以分别关或合并为"w/o conflict resolution"一次性全关。

**消融表结构(TII论文)**:
```
Table: Ablation Study on Core Components
| 配置 | 成功率↑ | 碰撞率↓ | 步数↓ | 最小间距↑ |
| Full (Ours) | 92.3±1.2 | 3.1±0.8 | 245±12 | 0.68±0.03 |
| w/o Dual-graph | ... | ... | ... | ... |
| w/o Risk-biased Attn | ... | ... | ... | ... |
| w/o Residual Gate | ... | ... | ... | ... |
| w/o Conflict Resolve | ... | ... | ... | ... |
| w/o GNN (MLP-only) | ... | ... | ... | ... |
```

---

## 四、P0 测试集评估基础设施(需补)

agent盘点:**test_gnn_mappo.py 只打印摘要,无CSV聚合;training_monitor.csv 无成功率列**。

### 4.1 需补的评估脚本(2天工作量)

#### 脚本1: `evaluate_checkpoint.py`
```python
# 输入: checkpoint路径, 测试场景列表, num_episodes, num_seeds
# 输出: results/{method}_{scene}_seed{X}.csv (每episode一行,含success/collision/steps/min_dist等)
```

功能:
- 加载 checkpoint,在指定场景跑 N episodes
- 记录每个episode的6个指标(成功/碰撞/死锁/步数/最小间距/时延)
- 支持 ORCA/DWA/IPPO/MAPPO/Ours 统一接口

#### 脚本2: `aggregate_test_results.py`
```python
# 输入: results/*.csv
# 输出: 
#   - aggregated_table.csv (各方法×场景的均值±std)
#   - significance_test.txt (t-test p值,Ours vs 各baseline)
#   - latex_table.tex (直接贴论文)
```

功能:
- 多种子聚合(均值、std、置信区间)
- 统计显著性检验(scipy.stats.ttest_rel, Wilcoxon)
- 生成 LaTeX 表格代码

### 4.2 修改 training_monitor 回调(0.5天)
在 `MARLMetricsCallback` 里加成功率/碰撞率列,训练期也能看收敛。

---

## 五、P1 参数敏感性分析(重构版)

**目标**: 验证方法对关键超参的鲁棒性,给出调参指南。

### 5.1 参数选择(4个,各5点)

| 参数 | 默认值 | 扫描点 | 物理意义 | 预期影响 |
|---|---|---|---|---|
| `communication_range` | 3.5m | [2.0, 3.0, 3.5, 4.5, 6.0] | 社交图邻居范围 | 太小漏风险,太大引噪声 |
| `gat_risk_bias_scale` | 2.5 | [0, 1.0, 2.0, 2.5, 3.5] | 风险偏置注意力强度 | 0=无偏置,高值=强化高风险邻居 |
| `counterfactual_advantage_coef` | 0.15 | [0, 0.05, 0.10, 0.15, 0.25] | 反事实信用分配权重 | 0=关闭,高值=强调协作奖励 |
| `team_reward_lambda` | 0.7 | [0, 0.3, 0.5, 0.7, 0.9] | 团队/个体奖励混合比 | 0=纯个体,1=纯团队 |

**实验规模**: 4参数 × 5点 × 3种子 = 60个训练 × Stage2(30万步) = **~120小时串行** (5天满载)。

**可行性优化**:
- 把步数降到10万(而非30万),足够看趋势,缩短到 2天
- 或只做2个最关键参数(comm_range + risk_bias),30个训练,1天

### 5.2 输出
- **曲线图**: 4个子图,每个参数一张,横轴=参数值,纵轴=成功率(均值±std),含最优点标注
- **Table**: 各参数最优值 + 合理范围 + 性能敏感度评分

---

## 六、完整P0+P1 工作量估算(诚实版)

| 任务 | 工作量(天) | 机器时(小时) | 优先级 |
|---|---|---|---|
| **1. ORCA/DWA 对接+评估脚本** | 3 | 测试6h | P0 |
| **2. IPPO 环境对齐+训练** | 5 | 训练12h | P0 |
| **3. MAPPO 验证+测试** | 2 | 测试4h | P0 |
| **4. 评估基础设施(2脚本+回调)** | 2.5 | — | P0 |
| **5. 消融开关补齐(gate/detour/replan)** | 1.5 | — | P0 |
| **6. P0实验执行**(对比5方法+消融8配置,各3种子) | 1(并行监控) | 训练~200h | P0 |
| **7. 参数敏感性**(优化版,2参数×5点×3种子) | 0.5 | 训练60h | P1 |
| **8. 泛化测试**(跨场景/密度) | 1 | 测试12h | P1 |
| **9. 数据分析+表格生成** | 2 | — | P0+P1 |
| **小计 P0** | **15天人力** | **220小时机器** | 必做 |
| **小计 P1** | **+3.5天** | **+72小时机器** | 推荐 |

**关键路径**: ORCA/DWA/IPPO 对接(10天)是最长板,可并行做评估脚本(2.5天)。总wall-clock时间约 **12-15天**(若单人串行)。

---

## 七、执行建议(给你的决策参考)

### 方案A: 最小可发表集(P0,15天)
只做对比+消融+测试集评估+统计,不做参数敏感性和泛化测试。争取最快出能投的初稿。

### 方案B: 完整可信集(P0+P1精简,18天)
P0全做 + 2个参数敏感性 + 跨场景泛化。论文实验章节充实,审稿意见好应对。

### 方案C: 分阶段(先验证方向,再扩展)
- **阶段1(5天)**: 只做 ORCA/DWA 对接 + Ours(MLP/Full) 对比,出第一张表,**验证方法确实比传统方法强**。若结果不行,及早调整方法;若结果好,信心大增。
- **阶段2(+10天)**: 补 IPPO/MAPPO 对比 + 全消融。
- **阶段3(+3天)**: 参数敏感性和泛化(optional)。

我的建议:**方案C 分阶段**,因为你现在连"我的方法真比ORCA强多少"都不知道(之前只比了自己的MLP和GAT)。先用5天做对比验证方向,再决定是all-in消融还是调方法。

你要我现在做什么?
- **选A/B/C 哪个方案**(或自定义)
- 还是你想先看看现有那个跑到17万步的冒烟测试数据质量如何,再决定?
