# 两个 Stage2 训练结果对比分析

## 📁 目录对比

### GNN_MAPPO_Stage2_Cont_EnvStage2_o（旧版，后缀 _o）
- **训练时间**: 2024年6月18日-19日
- **训练轮次**: 180 iterations
- **总步数**: 900,000 步
- **目录结构**: 包含 `best/`, `final/`, `tensorboard/`

### GNN_MAPPO_Stage2_Cont_EnvStage2（新版）
- **训练时间**: 2024年6月30日
- **训练轮次**: 67 iterations
- **总步数**: 268,000 步
- **目录结构**: 包含 `best/`, `tensorboard/`（无 `final/`）

---

## 📊 性能对比

### 初始表现

| 指标 | 旧版 (iter 1) | 新版 (iter 1) | 差异 |
|------|---------------|---------------|------|
| **episode_reward_mean** | **-31.63** | **-110.81** | 新版差79分 ⬇️ |
| **episode_len_mean** | **482.0** | **304.7** | 新版短177步 ⬇️ |
| **entropy** | 3.10 | 2.70 | 新版低0.4 ⬇️ |

### 最终表现

| 指标 | 旧版 (iter 180) | 新版 (iter 67) | 差异 |
|------|-----------------|----------------|------|
| **episode_reward_mean** | **+145.41** | **+50.11** | 新版低95分 ⬇️ |
| **episode_len_mean** | **390.63** | **409.95** | 新版长19步 ⬆️ |
| **entropy** | 1.90 | 2.93 | 新版高1.0 ⬆️ |

### 训练进度

| 项目 | 旧版 | 新版 | 说明 |
|------|------|------|------|
| **总轮次** | 180 | 67 | 新版训练不完整 |
| **总步数** | 900k | 268k | 新版只训练了30% |
| **每轮步数** | 5,000 | 4,000 | 新版配置不同 |
| **训练状态** | ✅ 完成 | ⏸️ 中断 |

---

## 🔍 关键差异分析

### 1. 训练完成度

**旧版 (_o)**:
- ✅ 完整训练 180 iterations
- ✅ 达到收敛（奖励 +145，稳定）
- ✅ 有 `final/` 目录（保存最终 checkpoint）
- ✅ 熵降到 1.90（策略收敛）

**新版**:
- ⏸️ 仅训练 67 iterations（37%）
- ⏸️ 未收敛（奖励 +50，仍在爬升）
- ❌ 无 `final/` 目录（训练中断）
- ⚠️ 熵仍在 2.93（策略不稳定）

---

### 2. 初始性能差异（重要）

**旧版初始奖励**: -31.63  
**新版初始奖励**: -110.81  

**差异原因可能是**:
1. **环境配置改变** - 障碍物数量、速度、地图不同
2. **奖励函数改变** - 如果新版修改了奖励权重
3. **随机种子不同** - 导致初始表现波动
4. **难度增加** - 新版可能用了更难的场景

---

### 3. 学习曲线对比

#### 旧版学习曲线（推测）
```
Iteration    Reward    说明
   1       -31.63     初始表现较好
  50        ~+50      中期快速提升
 100        ~+100     后期稳步爬升
 180       +145.41    收敛到高水平
```

#### 新版学习曲线（实际）
```
Iteration    Reward    说明
   1       -110.81    初始表现很差
  20        ~-50      缓慢改善
  50        ~+30      中期爬升
  67       +50.11     仍在提升中（未完成）
```

---

## 🤔 后缀 "_o" 的含义

根据 Linux 文件命名习惯，"_o" 通常表示：
- **"old"** - 旧版本
- **"original"** - 原始版本
- **备份标记** - 防止覆盖之前的训练结果

**推测场景**:
1. 2024年6月18-19日完成第一次 Stage2 训练 → `GNN_MAPPO_Stage2_Cont_EnvStage2`
2. 重新训练前，将其重命名为 `GNN_MAPPO_Stage2_Cont_EnvStage2_o`（备份）
3. 2024年6月30日开始新的 Stage2 训练 → `GNN_MAPPO_Stage2_Cont_EnvStage2`
4. 新训练在 67 iterations 时中断（未完成）

---

## 💡 建议

### 如果要继续训练
```bash
# 从新版的 best checkpoint 恢复
python3 train_gnn_mappo_full.py \
    --env_stage 2 \
    --restore /home/wj/work/multi-robot-exploration-rl/ray_results/GNN_MAPPO_Stage2_Cont_EnvStage2/best \
    --num_train_iterations 200  # 继续训练到 200
```

### 如果要用旧版结果
```bash
# 旧版已经收敛且表现更好
cp -r GNN_MAPPO_Stage2_Cont_EnvStage2_o/final GNN_MAPPO_Stage2_Cont_EnvStage2/
```

### 对比两个版本的配置
```bash
# 检查训练日志，看环境配置是否改变
cat GNN_MAPPO_Stage2_Cont_EnvStage2_o/tensorboard/events.out.tfevents.* | strings | grep "env_config"
cat GNN_MAPPO_Stage2_Cont_EnvStage2/tensorboard/events.out.tfevents.* | strings | grep "env_config"
```

---

## 📈 结论

| 比较项 | 旧版 (_o) | 新版 | 推荐 |
|--------|-----------|------|------|
| **训练完成度** | ✅ 100% | ⏸️ 37% | 旧版 |
| **最终性能** | ✅ +145 | ⏸️ +50 | 旧版 |
| **收敛状态** | ✅ 稳定 | ⏸️ 未收敛 | 旧版 |
| **时效性** | 6月18日 | 6月30日 | - |

**建议**:
1. **如果需要立即使用** → 用旧版 (`_o`) 的 `final/` checkpoint（已收敛）
2. **如果想要最新代码** → 继续训练新版到 180+ iterations
3. **如果想对比效果** → 用 TensorBoard 对比两个版本的学习曲线

---

**分析时间**: 2026-07-02  
**结论**: 旧版 (_o) 是完整训练的版本（180 iters，+145 reward），新版是中断的版本（67 iters，+50 reward）
