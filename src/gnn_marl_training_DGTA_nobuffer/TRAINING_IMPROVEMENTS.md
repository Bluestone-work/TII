# 多机器人导航训练优化方案

## 问题诊断

### 原始训练效果分析

根据训练日志分析：

**Stage2 (4车)**: 
- episode_reward: -43 → -55~-70
- 训练184k steps收敛良好

**Stage3 (8车) - 效果差**:
- episode_reward: -130 → -65~-90  
- episode_length: 387 → 320-340
- 初期奖励下降严重，说明难度跳跃过大

**Stage4 (intersection) - 效果更差**:
- episode_reward: -133 → -75~-100
- episode_length: 170-220 不稳定
- 300k steps才勉强收敛

### 根本原因

1. **课程难度跳跃过大**
   - Stage2→3: 车辆从4→8翻倍，多体交互复杂度指数增加
   - Stage3→4: 切换到全新场景，之前的策略不适用

2. **超参数不合理**
   - Stage2→3: LR保持3e-4不变，未给予足够学习空间
   - Stage4: 新场景却用极保守的5e-5学习率

3. **奖励函数未考虑密度**
   - 8车场景下机器人间距更近，但避碰奖励权重未调整

## 优化方案

### 1. 渐进式课程学习 (6 Stages)

```
Stage 1: 2车 + 3动障 (circle_swap) - 100k steps  ← 基础避障
Stage 2: 4车 + 4动障 (circle_swap) - 300k steps  ← 多体交互
Stage 3: 6车 + 4动障 (circle_swap) - 400k steps  ← 渐进密度 (新增)
Stage 4: 8车 + 5动障 (circle_swap) - 600k steps  ← 高密度协调
Stage 5: 4车 + 3动障 (circle_swap) - 200k steps  ← 场景切换预热 (新增)
Stage 6: 4车 + 3动障 (intersection) - 400k steps ← 场景泛化
```

**改进点**:
- 添加Stage 3 (6车): 平滑4→8车的难度跳跃
- 添加Stage 5 (4车circle): 在同场景下降低车数，为场景切换做预热

### 2. 自适应超参数策略

#### 学习率 (LR)
```bash
Stage 1: 3e-4   # 白板训练，标准LR
Stage 2: 3e-4   # 4车多体，保持标准LR充分学习
Stage 3: 2.5e-4 # 6车缓冲，适度降低稳定过渡
Stage 4: 2e-4   # 8车高密度，降低LR精细调优
Stage 5: 1.5e-4 # 4车预热，微调LR巩固策略
Stage 6: 1e-4   # 新场景泛化，保守微调适应新环境
```

#### PPO Clip
```bash
Stage 1-2: 0.20  # 标准Clip，允许较大策略更新
Stage 3:   0.18  # 适度收紧，平滑过渡
Stage 4:   0.15  # 进一步收紧，保护高密度学习
Stage 5:   0.12  # 微调模式，保护已学策略
Stage 6:   0.10  # 保守Clip，适应新场景
```

### 3. 密度自适应奖励函数

```python
# 以4车为基准，使用平方根缩放避免过度惩罚
density_factor = sqrt(num_agents / 4.0)

# Social reward scale: 机器人越多，邻居避碰越重要
adjusted_social_scale = base_social_scale * density_factor

# Collision penalty: 高密度下碰撞风险增加，适度提高惩罚
adjusted_collision_penalty = base_collision_penalty * (1.0 + 0.2 * (density_factor - 1.0))
```

**缩放示例**:
- 2车: density_factor = 0.707, social_scale ↓ 29%, collision_penalty ↓ 6%
- 4车: density_factor = 1.000, 基准值
- 6车: density_factor = 1.225, social_scale ↑ 22%, collision_penalty ↑ 5%
- 8车: density_factor = 1.414, social_scale ↑ 41%, collision_penalty ↑ 8%

## 如何启动训练

### 完整课程训练 (推荐)

```bash
cd /home/wj/work/multi-robot-exploration-rl/src/gnn_marl_training_DGTA_nobuffer

# 从Stage 1开始完整训练
./run_curriculum.sh \
    --start_stage 1 \
    --end_stage 6 \
    --model_type gat \
    --gat_actor_graph neighbor \
    --action_mode continuous \
    --train_steps 400000 \
    --num_workers 1
```

### 从特定Stage恢复训练

```bash
# 从Stage 3恢复 (假设已有Stage 2的checkpoint)
./run_curriculum.sh \
    --start_stage 3 \
    --end_stage 6 \
    --resume /path/to/stage2/best/checkpoint
```

### 单独测试某个Stage

```bash
# 只训练Stage 3 (6车)
./run_curriculum.sh \
    --start_stage 3 \
    --end_stage 3 \
    --resume /path/to/stage2/checkpoint
```

## 预期效果

基于优化方案，预期改进：

1. **Stage 3 (6车)**: 
   - 初期reward: -90 (vs 原-130)
   - 收敛reward: -55~-70 (vs 原-65~-90)
   - 训练平滑性显著提升

2. **Stage 4 (8车)**:
   - 初期reward: -80 (vs 原-130)
   - 收敛reward: -60~-75 (vs 原-65~-90)
   - 收敛速度加快 ~30%

3. **Stage 6 (intersection)**:
   - 初期reward: -70 (vs 原-133)
   - 收敛reward: -40~-60 (vs 原-75~-100)
   - 场景泛化能力明显增强

## 监控指标

训练过程中重点关注：

```bash
# 实时监控日志
tail -f /home/wj/work/multi-robot-exploration-rl/curriculum_logs/stage*_train.log

# TensorBoard可视化
tensorboard --logdir=/home/wj/work/multi-robot-exploration-rl/ray_results
```

**关键指标**:
- `episode_reward_mean`: 应该平滑上升，不应有断崖式下跌
- `episode_len_mean`: Stage切换时允许短期波动，但应快速稳定
- `r_social`: 观察邻居避碰奖励是否随密度调整
- `r_collision`: 碰撞惩罚应该逐渐减少
- `policy_loss` / `vf_loss`: 应该保持在合理范围，不爆炸

## 部署建议

针对你提到的"难以迁移到turtlebot3"问题：

### Sim2Real策略

1. **降低观测维度**
   - 真实机器人算力有限，考虑降低激光雷达采样点
   - 可以在Stage 5/6用降采样的观测训练

2. **添加域随机化**
   - 训练时添加传感器噪声
   - 随机化机器人动力学参数
   - 地面摩擦力随机化

3. **两阶段训练**
   - 阶段1: 仿真中训练策略
   - 阶段2: 少量真实数据微调 (RL fine-tuning)

4. **降低控制频率**
   - 仿真: 10Hz
   - 真实机器人: 5Hz (更符合实际计算能力)

## 修改文件清单

```
modified: src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/train_gnn_mappo_full.py
    - 添加Stage 3 (6车) 和 Stage 5 (4车预热) 到ENV_CURRICULUM

modified: src/gnn_marl_training_DGTA_nobuffer/run_curriculum.sh  
    - 更新STAGE_MAP_NUM/STAGE_NUM_AGENTS等配置 (1-6)
    - 调整STAGE_LR和STAGE_CLIP超参数策略
    - 修改默认START_STAGE=1, END_STAGE=6

modified: src/gnn_marl_training_DGTA_nobuffer/gnn_marl_training/gnn_marl_env.py
    - 添加密度自适应因子计算 (density_factor)
    - 动态调整social_scale和collision_penalty
    - 传递adjusted参数到IndependentRobotEnv
```

## 下一步工作

1. **立即启动**: 运行上述完整课程训练命令
2. **监控收敛**: 每个Stage结束后检查checkpoint效果
3. **调优细节**: 如果某个Stage仍然效果差，可以:
   - 增加该Stage的训练步数
   - 调整该Stage的LR/Clip
   - 检查奖励函数是否合理
4. **Sim2Real**: 训练完成后，在Stage 6基础上添加域随机化重新训练

## 联系与反馈

如有问题，检查：
- 日志文件: `curriculum_logs/stage*_train.log`
- ROS日志: `curriculum_logs/stage*_ros.log`
- Ray输出: `ray_results/*/progress.csv`
