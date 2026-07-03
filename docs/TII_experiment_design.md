# TII 论文实验章节设计方案

> **目标**: 为 Social-Risk GAT / Communication-Consistent Risk-Aware GNN-MAPPO 方法
> 设计完整的实验章节,达到 TII (Transactions on Industrial Informatics) 录用标准。
> 
> **方法核心创新**: ① 关系解耦架构(MLP-LSTM主干 + 轻量GAT分支) ② 无需通信的社交风险图
> ③ 后置残差门控融合 ④ Gap-Yield-Detour-Replan 局部冲突化解机制 ⑤ 训练-部署一致性

---

## 一、实验章节整体结构(符合 TII 范式)

### 1.1 实验设置 (Experimental Setup)
- 仿真平台:Gazebo + ROS2 + TurtleBot3
- 环境配置:circle_swap(8×8m)、intersection(12×12m)、走廊、十字路口
- 训练配置:6 阶段课程学习(2→4→6→8→4→4车泛化)
- 评估指标:成功率、碰撞率、平均完成步数、平均最小间距、推理时延
- 硬件:CPU/GPU 配置、训练时长

### 1.2 对比实验 (Comparative Experiments)
与已有方法对比,验证**整体性能优越性**:
- **Baseline 1**: MAPPO-MLP-LSTM(无图,主干策略)
- **Baseline 2**: ORCA(经典解析式避碰)
- **Baseline 3**: DWA(动态窗口法)
- **(可选)Baseline 4**: Communication-GAT MAPPO(直接学邻居通信图,旧版方法)
- **Ours**: Social-Risk GAT + MLP-LSTM(完整方法)

对比维度:
- **定量**: 4 车/6 车/8 车 circle_swap + 4 车 intersection,统计成功率/碰撞率/步数/最小间距
- **定性**: 可视化轨迹对比(会车、交叉、死锁场景)
- **计算开销**: 推理时延(ms/step)、参数量、训练时间

### 1.3 消融实验 (Ablation Studies)
逐个移除创新模块,验证**各组件的必要性和贡献**:

#### A. 图结构消融(核心创新点 ①②)
- **Full (Ours)**: Dual-graph(social + obstacle fusion)
- **Variant 1**: Social-only(仅社交风险图)
- **Variant 2**: Obstacle-only(仅环境障碍图)
- **Variant 3**: w/o GAT(纯 MLP-LSTM,即 Baseline 1)

对应代码:`--graph_ablation {dual_graph, social_only, obstacle_only}` + MLP baseline

#### B. 融合机制消融(核心创新点 ③)
- **Full (Ours)**: Post-backbone residual gating
- **w/o Gate**: 直接加权融合(无门控)
- **w/o Residual**: 非残差结构
- **Pre-backbone fusion**: GAT 在主干前融合(不解耦)

对应代码:修改 `GATRLlibModel.forward` 的融合逻辑(当前无直接开关,需改代码)

#### C. 风险偏置注意力消融(核心创新点 ⑤)
- **Full (Ours)**: Risk-biased attention(`gat_risk_bias_scale=2.5`)
- **w/o Risk Bias**: 标准 GAT(`gat_risk_bias_scale=0`)

对应代码:`--gat_risk_bias_scale {2.5, 0}`

#### D. 局部冲突化解机制消融(核心创新点 ④)
- **Full (Ours)**: Gap + Yield + Detour + Replan 全开
- **w/o Gap**: 无缝隙特征(`--gap_feature_enable 0`)
- **w/o Yield**: 无让行(`--yielding_enable 0`)
- **w/o Detour**: 无绕行子目标(修改环境代码关掉)
- **w/o Replan**: 无动态重规划(`--dynamic_replan_enable 0`,需确认该参数存在)

对应代码:`--gap_feature_enable`、`--yielding_enable` + 环境参数

#### E. Critic 结构消融
- **MLP Critic** (默认):`--gat_critic_mode mlp`
- **GAT Critic**:`--gat_critic_mode gat`

#### F. 反事实信用分配消融
- **With Counterfactual**:`--counterfactual_advantage_coef 0.15`
- **w/o Counterfactual**:`--counterfactual_advantage_coef 0`

### 1.4 参数敏感性分析 (Parameter Sensitivity Analysis)
验证方法对**关键超参数的鲁棒性**:

| 参数 | 默认值 | 扫描范围 | 固定其他 |
|---|---|---|---|
| `communication_range` | 3.5m | [2.0, 3.0, 3.5, 4.5, 6.0] | 4车 circle_swap |
| `counterfactual_advantage_coef` | 0.15 | [0, 0.05, 0.10, 0.15, 0.20, 0.30] | 同上 |
| `gat_risk_bias_scale` | 2.5 | [0, 1.0, 2.0, 2.5, 3.5, 5.0] | 同上 |
| `team_reward_lambda` | 0.7 | [0, 0.3, 0.5, 0.7, 0.9, 1.0] | 同上 |
| `lr` | 3e-4 | [1e-4, 2e-4, 3e-4, 5e-4, 1e-3] | 从头训练 |
| `entropy_coeff` 初始 | 0.003 | [0.001, 0.002, 0.003, 0.005, 0.01] | 同上 |

每组 3 个随机种子,绘制均值±标准差曲线。

### 1.5 泛化性测试 (Generalization Tests)
验证方法在**未见场景/密度/扰动**下的鲁棒性:

#### A. 跨场景泛化
- 训练:仅 circle_swap 6 阶段
- 测试:intersection、hallway、warehouse_aisles、四向交叉路口(来自 `fixed_benchmark_scenarios.py`)

#### B. 跨密度泛化
- 训练:2/4 车(低密度)
- 测试:6/8/10 车(训练时未见的高密度)

#### C. 动态障碍密度鲁棒性
- 训练:3 个动态障碍
- 测试:[0, 5, 8, 10, 15] 个动态障碍,速度 [0.3, 0.6, 1.0] m/s

#### D. 通信质量鲁棒性(如果方法宣称 communication-free,此项可选)
- 训练:完美通信(dropout=0, latency=0)
- 测试:dropout=[0, 0.2, 0.5, 0.8], latency=[0, 2, 5] steps

### 1.6 真实部署验证(可选,如有条件)
- 2-4 台实体 TurtleBot3 在实验室走廊/交叉口
- 指标:成功完成任务次数 / 总试次、人工干预次数、sim-to-real gap 分析

---

## 二、实验优先级分级(考虑时间/算力约束)

### P0 (必做,论文核心)
1. **对比实验 1.2**(Ours vs MAPPO-MLP vs ORCA vs DWA)
2. **图结构消融 1.3.A**(dual/social/obstacle/w/o GAT)
3. **参数敏感性 1.4**(至少 3 个关键参数:`communication_range`、`counterfactual_advantage_coef`、`gat_risk_bias_scale`)

### P1 (推荐做,增强说服力)
4. **融合机制消融 1.3.B**(w/o gate/residual)
5. **局部冲突化解机制消融 1.3.D**(w/o gap/yield)
6. **跨场景泛化 1.5.A**(intersection + fixed benchmark)

### P2 (锦上添花,时间充裕可做)
7. **反事实信用消融 1.3.F**
8. **Critic 结构消融 1.3.E**
9. **跨密度泛化 1.5.B**
10. **通信质量鲁棒性 1.5.D**

### P3 (加分项,非必需)
11. **真实部署验证 1.6**
12. **风险偏置消融 1.3.C**
13. **动态障碍鲁棒性 1.5.C**

---

## 三、实验执行方案(脚本化,可复现)

### 3.1 对比实验执行清单

| 实验 ID | 方法 | 命令 | Checkpoint 位置 | 说明 |
|---|---|---|---|---|
| **Baseline-MLP** | MAPPO-MLP-LSTM | `--model_type mlp --env_stage 2` | `ray_results/MAPPO_MLP_Stage2/best` | 无图主干 |
| **Baseline-ORCA** | ORCA | 调用 `src/start_orca_nav/` 节点 | N/A | 解析式,无训练 |
| **Baseline-DWA** | DWA | 同上 | N/A | 解析式 |
| **Ours-Full** | Social-Risk GAT | `--model_type gat --graph_ablation dual_graph` | `ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2/best` | 完整方法 |

测试场景:Stage2(4车)、Stage3(6车)、Stage4(8车)、Stage6(intersection 4车),各跑 **50 episodes × 3 seeds**。

**ORCA/DWA 测试**:需单独启动 ROS 节点,记录成功/碰撞/步数。参考 `src/gnn_bc_tools/run_orca_dwa_bc_pipeline.py` 的专家采集部分。

### 3.2 消融实验执行清单

#### A. 图结构消融(训练 3 个变体)
```bash
# Dual-graph (Full)
bash run_curriculum.sh --model_type gat --graph_ablation dual_graph --start_stage 2 --end_stage 2
# Social-only
bash run_curriculum.sh --model_type gat --graph_ablation social_only --start_stage 2 --end_stage 2
# Obstacle-only
bash run_curriculum.sh --model_type gat --graph_ablation obstacle_only --start_stage 2 --end_stage 2
# w/o GAT = MLP baseline(已有)
```
**注意 bug**:确认 `--gat_actor_graph local_risk`(默认),否则 `graph_ablation` 失效。

#### B. 融合机制消融(需修改代码)
在 `gat_rllib_model.py` 的 `forward` 中添加 `--fusion_mode` 参数控制:
- `residual_gating`(默认)
- `direct_add`(w/o gate)
- `concat`(w/o residual)

需要改代码 + 重训。

#### C. 风险偏置消融(单参数扫描)
```bash
for scale in 0 1.0 2.0 2.5 3.5 5.0; do
    bash run_curriculum.sh --model_type gat --gat_risk_bias_scale $scale --start_stage 2 --end_stage 2
done
```

#### D. 局部冲突化解机制消融
```bash
# Full(默认)
# w/o Gap
bash run_curriculum.sh --model_type gat --gap_feature_enable 0 --start_stage 2 --end_stage 2
# w/o Yield
bash run_curriculum.sh --model_type gat --yielding_enable 0 --start_stage 2 --end_stage 2
# w/o Detour + Replan(需确认环境参数,可能要改代码关掉)
```

#### E. Critic 结构消融
```bash
# MLP Critic(默认)
# GAT Critic
bash run_curriculum.sh --model_type gat --gat_critic_mode gat --start_stage 2 --end_stage 2
```

#### F. 反事实信用消融
```bash
# With(默认 0.15)
# w/o
bash run_curriculum.sh --model_type gat --counterfactual_advantage_coef 0 --start_stage 2 --end_stage 2
```

### 3.3 参数敏感性分析执行

为每个参数创建扫描脚本 `param_sweep_*.sh`,例如 `param_sweep_comm_range.sh`:
```bash
#!/bin/bash
for range in 2.0 3.0 3.5 4.5 6.0; do
    for seed in 42 123 456; do
        bash run_curriculum.sh \
            --model_type gat \
            --communication_range $range \
            --start_stage 2 --end_stage 2 \
            --seed $seed \
            --experiment_name "comm_range_${range}_seed${seed}"
    done
done
```

收集各实验的 `training_monitor.csv`,绘制 `episode_reward_mean` vs `communication_range` 曲线(均值±std)。

### 3.4 泛化性测试执行

#### A. 跨场景泛化
```bash
# 用 Stage2-5 训练的 checkpoint 测试 Stage6(intersection)
python3 test_gnn_mappo.py \
    --checkpoint_path ray_results/GNN_MAPPO_Stage2_Cont_EnvStage5/best \
    --map_number 4 \
    --num_episodes 50 \
    --num_agents 4
```

#### B. 跨密度泛化
```bash
# 用 4 车 checkpoint 测试 8 车
python3 test_gnn_mappo.py \
    --checkpoint_path ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2/best \
    --map_number 8 \
    --num_agents 8 \
    --num_episodes 50
```

---

## 四、实验输出物清单(论文素材)

### 4.1 表格
- **Table 1**: 对比实验定量结果(Ours vs Baselines,4 场景 × 5 指标)
- **Table 2**: 图结构消融结果(dual/social/obstacle/w/o GAT,4 指标)
- **Table 3**: 融合机制消融结果
- **Table 4**: 局部冲突化解消融结果(Full / w/o gap / w/o yield / ...)
- **Table 5**: 计算开销对比(参数量、推理时延、训练时间)
- **Table 6**: 跨场景泛化结果

### 4.2 曲线图
- **Fig 1**: 对比实验学习曲线(episode_reward vs timesteps,各方法)
- **Fig 2**: 参数敏感性曲线(3 个关键参数,均值±std)
- **Fig 3**: 消融实验学习曲线对比
- **Fig 4**: 跨密度泛化性能柱状图

### 4.3 可视化
- **Fig 5**: 典型场景轨迹对比(Ours vs ORCA vs MLP,会车/交叉/死锁)
- **Fig 6**: 注意力权重热力图(高风险邻居的注意力分布)
- **Fig 7**: 门控系数演化图(训练过程中 gate 的激活率)

---

## 五、实验执行时间估算(指导资源规划)

假设硬件:8 GPU × RTX 3090,每个实验独立占 1 GPU。

| 实验类型 | 实验数量 | 单实验时长 | 总时长(并行) |
|---|---|---|---|
| 对比实验(训练) | 2(MLP + Full) | ~8h | 1 天 |
| 对比实验(测试) | 4×4 场景×50 ep | ~6h | 1 天 |
| 图结构消融 | 3 | ~8h | 1 天 |
| 其他消融 | 5 | ~8h | 5 天 |
| 参数敏感性 | 3 参数×5 点×3 种子 | ~8h | 6 天 |
| 泛化测试 | 4 | 测试为主 | 1 天 |
| **总计** | — | — | **~15 天**(并行) |

**优化建议**:P0 实验(对比+图消融+3 参数敏感性)可在 **5 天**内完成,先出结果验证方法有效性,再补 P1/P2。

---

## 六、实验管理建议(保证可复现性)

### 6.1 实验命名规范
```
{experiment_type}_{variant}_{scene}_{seed}
例如: ablation_social_only_stage2_seed42
     comparison_orca_stage3
     param_sweep_comm_range_3.5_seed123
```

### 6.2 结果目录结构
```
experiments/
├── comparison/          # 对比实验
│   ├── mlp_stage2_seed42/
│   ├── ours_stage2_seed42/
│   └── results_summary.csv
├── ablation/            # 消融实验
│   ├── graph_dual/
│   ├── graph_social_only/
│   ├── graph_obstacle_only/
│   └── results_summary.csv
├── param_sensitivity/   # 参数敏感性
│   ├── comm_range/
│   ├── counterfactual_coef/
│   └── plots/
└── generalization/      # 泛化测试
    ├── cross_scene/
    ├── cross_density/
    └── results_summary.csv
```

### 6.3 自动化脚本模板
创建 `scripts/run_all_experiments.sh` 主控脚本,调用各子脚本:
```bash
#!/bin/bash
bash scripts/run_comparison.sh
bash scripts/run_ablation_graph.sh
bash scripts/run_param_sweep_comm_range.sh
# ...
python3 scripts/aggregate_results.py  # 汇总所有结果到 LaTeX 表格
```

---

## 七、论文写作对应关系

| 实验 | 对应论文章节 | 论证目标 |
|---|---|---|
| 对比实验 | Sec IV-B: Performance Comparison | 整体优于已有方法 |
| 图结构消融 | Sec IV-C: Ablation on Graph Structure | 双图融合的必要性 |
| 融合机制消融 | Sec IV-C: Ablation on Fusion Mechanism | 门控+残差的有效性 |
| 局部冲突消融 | Sec IV-C: Ablation on Conflict Resolution | Gap/Yield 的贡献 |
| 参数敏感性 | Sec IV-D: Sensitivity Analysis | 方法鲁棒性 |
| 泛化测试 | Sec IV-E: Generalization | 跨场景/密度能力 |
| 计算开销 | Sec IV-F: Computational Cost | 实用性 |

---

## 八、当前代码缺失的实验功能(需补充)

根据盘点结果,以下实验需要**额外开发**:

1. **ORCA/DWA 对比测试自动化**:当前 ORCA/DWA 在独立包 `start_orca_nav`,需编写批量测试脚本调用其 ROS 节点,记录指标。
2. **融合机制消融开关**:`gat_rllib_model.py` 无 `--fusion_mode` 参数,需加 argparse + 条件分支。
3. **Detour/Replan 消融开关**:环境代码可能无直接 CLI 参数关掉这两个,需确认或添加。
4. **固定 benchmark 场景测试集成**:`fixed_benchmark_scenarios.py` 在同级包,需导入并批量跑。
5. **结果聚合脚本**:`aggregate_results.py` 自动读取各实验 `training_monitor.csv`,生成 LaTeX 表格。
6. **注意力权重可视化**:当前 GAT 模型可能无保存注意力权重的钩子,需在 `forward` 里加 `return_attention=True` 分支。
7. **多种子并行管理**:当前 `run_curriculum.sh` 单线程,需改为并行 launch 多个实验(或用 Ray Tune)。

---

- **创建日期**: 2026-06-23
- **状态**: 设计完成,待执行
- **预计完成 P0 实验时间**: 5 天(8 GPU 并行)
- **下一步**: 先执行 P0 实验(对比+图消融+参数敏感性),验证方向正确后再补 P1/P2
