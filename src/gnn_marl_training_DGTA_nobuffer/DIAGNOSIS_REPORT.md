# 避碰策略训不好的量化诊断报告

**诊断时间**: 2026-06-30  
**诊断方法**: 静态代码分析 + 奖励-感知回路数值推算  
**配置**: beifen 基线 (progress_scale=1.5, static_scale=0.8, social_scale=0.4)

---

## 一、核心发现: 奖励失衡导致"不动"是最优策略

### 典型场景下的总奖励对比

| 场景 | r_progress | r_static | r_social | r_dynamic_obs | **总奖励** | 策略倾向 |
|---|---|---|---|---|---|---|
| 开阔前进 | +0.10 | 0.00 | 0.00 | 0.00 | **+0.09** | ✓ 可行 |
| 中密度避碰前进 (min_dist=0.6m, 1邻居) | +0.08 | -0.40 | -0.15 | 0.00 | **-0.48** | ✗ 负收益 |
| 高密度 (min_dist=0.4m, 2邻居, 1动态障碍) | +0.06 | -0.70 | -0.30 | -0.50 | **-1.45** | ✗ 强负 |
| 磨蹭不动 (v≈0) | 0.00 | 0.00 | 0.00 | 0.00 | **-0.01** | ✓ 几乎无罚 |

**结论**: 任何"动起来+靠近障碍"的动作都是负收益。在 4-8 车高密度场景,策略的最优解是**"不动"或"远离一切"**,而非"精确避碰同时到达目标"。

---

## 二、三项核心失配

### 1. 前进信号太弱 (r_progress)

- **设计值**: `progress_scale=1.5`, clip ±0.3
- **实测**: 典型前进速度 0.15 m/s → `goal_dist_delta ≈ 0.015m/step`
- **单步 r_progress**: ~**0.06–0.12** (heading_shaping 贡献 ~0.05)
- **问题**: 最优情况也只有 +0.18,实测更低,完全被避碰惩罚压制

### 2. 静态避碰惩罚太保守 (r_static)

| min_dist | r_static (static_scale=0.8) | 相对 r_progress 倍数 |
|---|---|---|
| 0.8m | **-0.07** | 约等于前进奖励 |
| 0.5m | **-0.65** | **6.5× 前进** |
| 0.3m (near-miss) | **-1.00** (clip) | **10× 前进** |

- **触发阈值**: `RWD_STATIC_D0=0.75m` 很宽,0.8m 时已开始惩罚
- **near_miss_penalty**: 0.30m 内触发,和碰撞硬阈值 0.22m 只差 **8cm**,但栅格分辨率 **12.5cm** — 感知精度不够
- **势场公式**: `-(1/d - 1/D0)²` 在近距离爆炸 (0.25m 时 raw repulsive=-7.1,被 clip 到 -1.0)

### 3. r_dynamic_obs 是"隐藏炸弹"

- **无 scale**: 直接 `1.0 * worst_penalty`,权重是 `r_social`(0.4×) 的 **2.5 倍**
- **速度估计不稳定**: 帧间差分,5mm 位置噪声 = 0.05 m/s 速度误差 = `is_dynamic` 阈值
- **触发不可预测**: 动态障碍被正确识别时 → -0.8 惩罚;速度估计失败时 → 0 惩罚
- **结果**: 策略无法可靠预测何时会吃惩罚,只能学"远离一切"

---

## 三、感知-奖励精度失配

| 奖励要求 | 需要的感知 | 实际感知能力 | 差距 |
|---|---|---|---|
| near_miss 0.30m 精确避让 | 知道 0.22~0.30m 范围内**哪个方向**有障碍 | 栅格 0.125m/格,只覆盖 2m 半径 | ✗ 分辨率不够 |
| 提前减速 (前方 0.70m) | 知道 0.70m 处障碍物相对速度 | `front_min` 标量 + `front_risk`(基于静态距离) | ✗ 无动态预判 |
| 避开动态障碍 TTC<2.5s | 准确的障碍物速度 | 帧间差分,±0.05m/s 噪声 | ✗ 速度不可信 |
| 识别左/右/前 障碍 | 稠密方向-距离信号 | 2 个标量 + 3 个 token + 粗栅格 | △ 信息稀疏 |

**例子**: 动态障碍物以 0.15 m/s 从左侧横穿,距离 1.2m。
1. 聚类匹配可能失败 → 速度估成 0 → `is_dynamic=0` → 被当静态
2. 即使估对速度,障碍物在左后方 → 不在 `front_min` 视野 → `speed_risk` 不触发
3. 即使进了 token,策略只有 2 个标量 + 32×32 粗栅格(每格 0.125m) → 提不出"左侧有运动物体"的细节
4. 输出随机转向 → 吃惩罚 → 策略学"别动"

---

## 四、改进方案(按性价比排序)

### 【高优先级】立即可改,收益明确

#### 1. 重平衡奖励尺度 (预计 1h, 收益 ★★★★)

**改动**:
```python
# train_gnn_mappo_full.py
_progress_scale = 2.5  # 1.5 → 2.5

# gnn_marl_env.py (或通过 env_config 传参)
"static_scale": 0.5,   # 0.8 → 0.5
"social_scale": 0.3,   # 0.4 → 0.3

# gnn_marl_env.py:3508 行
r_dynamic_obs = 0.3 * worst_penalty  # 加 scale,和 social 对齐

# gnn_marl_env.py 顶部常量
RWD_STATIC_D0 = 0.55  # 0.75 → 0.55m,减少"远处就开始罚"
```

**预期效果** (重算):
| 场景 | 改前总奖励 | 改后总奖励 | 改善 |
|---|---|---|---|
| 中密度避碰前进 | -0.48 | **+0.05** | 净收益 ✓ |
| 高密度 | -1.45 | **-0.35** | 从强负→可容忍 |

#### 2. 扇区距离拼回观测 (预计 2h, 收益 ★★★★)

**改动**:
```python
# gnn_marl_env.py:3962 _get_obs 拼接里
sector_dists_normed = np.clip(sector_dists / self.obstacle_filter_range, 0.0, 1.0)
obs = np.concatenate([
    target_features,
    [self.current_vel_x, self.current_vel_w],
    [front_min, min_dist],
    sector_dists_normed,  # <-- 新增,9 维 (obstacle_top_k=9)
    neighbor_prediction_features,
    obstacle_motion_features,
    self.agent_id_embedding,
])

# 同步修改 obs_dim (:1626)
self.obs_dim = (
    self.target_obs_dim
    + 2  # vel
    + self.safety_feature_dim  # [front_min, min_dist]
    + self.obstacle_top_k  # <-- 新增扇区距离
    + self.neighbor_prediction_dim
    + self.obstacle_motion_dim
    + self.max_agent_id_dim
)
```

**收益**: 给策略稠密的方向-距离信号 (9 个扇区,每个一个归一化距离),几乎零训练成本,但显著增强空间感知。

---

### 【中优先级】需重训,但改善避碰关键

#### 3. 动态速度估计加平滑 (预计 3h, 收益 ★★★)

**改动**:
```python
# gnn_marl_env.py IndependentRobotEnv.__init__ 加
self._cluster_velocity_ema = {}  # {cluster_id: (vx, vy)}

# _get_obstacle_motion_features :2396-2401 改
if matched is not None:
    raw_vx = (cluster["xw"] - matched["xw"]) / self.control_dt
    raw_vy = (cluster["yw"] - matched["yw"]) / self.control_dt
    cluster_id = (round(cluster["xw"], 1), round(cluster["yw"], 1))  # 粗 hash
    if cluster_id in self._cluster_velocity_ema:
        alpha = 0.3
        vx_world = alpha * raw_vx + (1-alpha) * self._cluster_velocity_ema[cluster_id][0]
        vy_world = alpha * raw_vy + (1-alpha) * self._cluster_velocity_ema[cluster_id][1]
    else:
        vx_world, vy_world = raw_vx, raw_vy
    self._cluster_velocity_ema[cluster_id] = (vx_world, vy_world)
```

**收益**: 降低速度估计噪声,`is_dynamic` 不再频繁跳变,`r_dynamic_obs` 变稳定。

#### 4. 去掉 token risk 门控 (预计 30min, 收益 ★★)

**改动**:
```python
# gnn_marl_env.py:2460
# 原: if risk <= 1e-4: continue
# 改为:
if dist > self.obstacle_filter_range:
    continue
# 只过滤超范围,不过滤"不危险"
```

**收益**: 让近处障碍物无论是否"危险"都进 token,给策略提前量。

---

### 【低优先级】改善明显但成本高

#### 5. 栅格提分辨率/历史帧 (预计 4h, 收益 ★★★)

- 32×32×2 → **64×64×4**, 或半径 2m → 3m
- 需重训,显存/计算增加 ~2-4×

#### 6. 增加"成功避让"正奖励 (预计 2h, 收益 ★★)

- 当前纯负向,加 `r_avoidance_bonus`
- 触发条件: 过去 N 步内 min_dist < 阈值,但未碰撞,且 dist_to_goal 减少
- 需精细设计,避免和 `r_progress` 重复

---

## 五、验证方案

### 最小验证 (推荐先做这个)

**改动**: 只做 1+2 (奖励重平衡 + 扇区距离), 预计 3 小时

**实验**:
```bash
# 从头训 Stage2, 100 iters
python3 train_gnn_mappo_full.py \
    --env_stage 2 \
    --num_agents 4 \
    --action_mode continuous \
    --num_workers 1 \
    --train_batch_size 8000 \
    --train_iterations 100
```

**对比指标** (用 training_monitor.jsonl 或 TensorBoard):
1. `episode_reward_mean`: 改前负值 → 改后正值?
2. `episode_collisions`: 改前高 → 改后降低?
3. `dist_to_goal` (final): 改前卡 4.4m → 改后下降?
4. `episode_len_mean`: 改前低(早终止) → 改后接近 max_steps?

**判断**:
- 如果 reward ↑ 且 collision ↓ → **确诊是奖励-感知失配**,继续 3+4
- 如果无改善 → 深入分析策略网络结构/学习率/PPO 超参

---

## 六、当前配置的量化问题总结

1. **奖励失衡**: 中密度场景净收益 -0.48 (避碰罚 > 前进奖 5×)
2. **感知稀疏**: 只有 2 个标量 + 3 个 token + 粗栅格,无稠密方向信息
3. **速度不稳定**: 帧间差分噪声 ±0.05m/s,动/静区分不可靠
4. **阈值过保守**: `RWD_STATIC_D0=0.75m` 在 0.8m 时已开始惩罚
5. **r_dynamic_obs 失控**: 无 scale,权重 2.5× r_social,触发不可预测

**根本矛盾**: 奖励要求的精细避碰行为(0.30m near-miss),感知精度(0.125m栅格)支撑不了 → 策略只能学粗略规避(不动/远离) → 到达目标失败。

---

## 附录: 诊断脚本

- **静态分析**: `static_reward_analysis.py` (已运行,输出见上)
- **在线采样**: `diagnose_reward_perception.py` (需 Gazebo,用于采集真实训练数据)

如需跑在线诊断:
```bash
# 先启动 Gazebo
ros2 launch ... spawn_robots:=4

# 然后跑诊断
python3 diagnose_reward_perception.py --num_episodes 10 --env_stage 2
# 会在 diagnosis_output/ 生成 .jsonl 和汇总统计
```
