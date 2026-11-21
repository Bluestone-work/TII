# RADIUS Code/Theory Audit

审查基于 `TII-26-8823_Proof_hi.pdf`、`TII/tii-articles-template.tex` 和
`src/gnn_marl_training_DGTA_nobuffer/` 的当前代码。

## 初步结论

当前仓库不是论文 PDF 的一一对应实现。PDF 的题目是 RADIUS，而仓库中的
LaTeX 源文件仍是旧版本 `A Dual-Graph Counterfactual MAPPO Framework`。
训练代码实现了 dual graph、residual gate 和 leave-one-out critic，但不能直接
把当前仓库称为 PDF 中最终 RADIUS 版本的完整可复现实验代码。

## 已确认实现

- Actor 有独立 social GAT 和 obstacle GAT 分支，并通过拼接投影后融合。
- 融合形式为 `policy_feat + sigmoid(policy_feat, graph_feat) * graph_delta(graph_feat)`，
  与 residual gated fusion 的形式一致。
- Critic 使用全局状态；`compute_counterfactual_values` 对指定 agent 构造 leave-one-out
  mask，并计算 `V(full) - V(masked)`。
- Counterfactual 信号在 PPO trajectory post-processing 阶段加入 reward，随后由 GAE
  传播，而不是简单加到 advantage 上。
- Safety shield 在执行动作后进行减速、停车或原地转向；训练默认关闭 shield，避免
  修改 PPO 记录的行为动作。
- 动态障碍物驱动器采用独立随机游走、反弹和随机转向，不会根据机器人状态主动让行。

## 与论文/审稿意见的主要差异

1. **版本不一致**：PDF 的 RADIUS 叙述、仓库 LaTeX 源文件和训练代码的命名、场景及
   配置并未统一。修稿前应先确定唯一的代码-论文基线。
2. **消融不足以证明关系分离**：代码只有 `social_only`、`obstacle_only` 和 `dual_graph`
   三种开关，没有 shared graph + entity/relation embedding、参数匹配 shared graph、
   direct concatenation 或 ungated residual 对照。
3. **MASS 不是论文中完全相同的 zero-mask 实现**：MLP critic 会将对应全局节点置零，
   GAT critic 则使用 node mask 和邻接矩阵屏蔽。两种 critic 的 masking operator 不同，
   需要在论文中明确，并增加 masking validity/反事实回报验证。
4. **MASS 异常被静默吞掉**：counterfactual post-processing 的异常直接 `pass`，可能导致
   某些 rollout 实际没有使用 MASS，却没有留下可审计记录。
5. **gate 饱和没有监控**：代码定义了 sigmoid gate，但没有记录 gate 均值、分位数、饱和率
   或 graph residual 的相对范数，无法直接回应 reviewer 对 gate saturation 的质疑。
6. **动态障碍物是非反应式的**：当前 mover 只做随机游走和碰撞反弹，尚未覆盖会减速、让行、
   改变方向的 reactive pedestrian 行为。
7. **Safety shield 的影响可配置但尚未形成正式对照**：训练脚本提供 `safety_filter_enable`
   和 `allow_shield_during_training`，但需要输出 policy-only、policy+shield、shield
   intervention rate 和 fallback rate 的统一评估结果。
8. **训练接口支持离散和连续动作**：`action_mode` 默认仍是 `discrete_primitive`，连续动作
   需要显式设置为 `continuous`；因此 TIRL/GAR-CoNav 的连续控制适配不能只在论文中一句带过。

## 修稿优先级

1. 固定 PDF、LaTeX、训练代码的版本对应关系。
2. 新增 parameter-matched shared graph、relation embedding、ungated residual 消融。
3. 新增 zero/token/mean/node-removal masking 及真实 rollout counterfactual 对照。
4. 新增 Safety Filter on/off 与 intervention 统计。
5. 增加 gate 统计、动态障碍物行为说明和连续控制 baseline 配置记录。
6. 重新核对所有表格、图和实验场景是否由当前代码实际生成。
