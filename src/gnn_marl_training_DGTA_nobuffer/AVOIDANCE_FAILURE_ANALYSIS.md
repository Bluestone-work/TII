# 避碰失败根因分析

## 🔴 问题描述

训练后的智能体**有时**接近动态障碍物或其他智能体时不避碰，导致碰撞。

---

## 🔍 根本原因分析

### 1. **缺失动态障碍物奖励信号** ⚠️ 最严重

**现状**：
```python
# gnn_marl_env.py:3228
reward = r_progress + r_static + r_social + r_collision + r_goal + r_time
```

**问题**：
- ✅ `r_social` 只处理**其他智能体**（agent-to-agent）
- ❌ **动态障碍物**完全没有奖励惩罚！
- ❌ 虽然观测中有 `obstacle_motion_features`，但奖励函数不惩罚

**后果**：
- 智能体可以"看到"动态障碍物（观测中有）
- 但学不到要避开（奖励中没有）
- 导致策略对动态障碍物"视而不见"

**类比**：就像告诉学生"考试会考这个"（观测），但从不扣分（奖励），学生自然不重视。

---

### 2. **风险门控过滤掉了安全场景的学习** ⚠️ 中等

**现状**：
```python
# gnn_marl_env.py:2284
risk = max(close_risk, future_risk, crossing_risk, ttc_risk)
if risk <= 1e-4:
    continue  # 跳过这个障碍物，不加入 token
```

**问题**：
- "不危险"的障碍物被过滤，不进入观测 token
- 智能体无法学习**提前避让**
- 只有当障碍物已经很近（risk > 0）时才进入观测

**后果**：
- 智能体学会了"最后时刻才反应"
- 没有学到"看到远处障碍物就提前规避"

**改进方案**：
```python
# 只过滤超出范围的，不过滤"不危险"的
if dist > self.obstacle_filter_range:
    continue
```

---

### 3. **预测时间窗口太短** ⚠️ 中等

**现状**：
```python
# gnn_marl_env.py:2237
predict_h = min(max(self.predictive_horizon_sec, 0.3), 0.8)  # 最多 0.8 秒
```

**问题**：
- 快速移动的障碍物（0.6 m/s）在 0.8 秒内只移动 0.48m
- 机器人自己也在移动，相对预测距离更短
- 无法提前足够远规避

**后果**：
- 看不到足够远的未来
- 只能"被动反应"，无法"主动预判"

**改进方案**：
```python
# 根据障碍物速度动态调整
obs_speed = math.hypot(vx, vy)
if obs_speed > 0.5:  # 快速运动
    predict_h = 2.0
elif obs_speed > 0.3:  # 中速
    predict_h = 1.5
else:  # 慢速
    predict_h = 1.0
```

---

### 4. **没有安全膨胀半径** ⚠️ 轻度

**现状**：
```python
# gnn_marl_env.py:2271
close_risk = float(np.clip((self.close_obstacle_dist - dist) / self.close_obstacle_dist, 0.0, 1.0))
```

**问题**：
- 直接用物理距离计算风险
- 没有考虑机器人尺寸（半径 0.105m）
- 没有安全裕度（传感器误差、定位误差）

**后果**：
- 计算认为"0.3m 是安全的"
- 实际上 0.3m - 0.105m（机器人半径）= 0.195m 真实间隙
- 再考虑误差，可能已经接触

**改进方案**：
```python
ROBOT_RADIUS = 0.105  # TurtleBot3
SAFETY_MARGIN = 0.15  # 安全裕度
INFLATION_RADIUS = 0.255  # 总膨胀

effective_dist = max(0.0, dist - INFLATION_RADIUS)
close_risk = float(np.clip((self.close_obstacle_dist - effective_dist) / self.close_obstacle_dist, 0.0, 1.0))
```

---

### 5. **速度估计不稳定** ⚠️ 轻度

**现状**：
```python
# gnn_marl_env.py:2246
vx_world = float((float(cluster["xw"]) - float(matched["xw"])) / self.control_dt)
```

**问题**：
- 帧间差分估计速度
- 没有平滑滤波
- 激光噪声 ±5mm → 速度误差 ±0.05 m/s
- 导致 `is_dynamic` 频繁跳变（阈值 0.05 m/s）

**后果**：
- 同一个障碍物一会儿被识别为"动态"，一会儿"静态"
- 策略无法建立稳定的预测

**改进方案**：
```python
# 指数移动平均 (EMA)
self._cluster_velocity_ema = {}  # 初始化字典

# 在匹配后
if matched is not None:
    raw_vx = (cluster["xw"] - matched["xw"]) / self.control_dt
    raw_vy = (cluster["yw"] - matched["yw"]) / self.control_dt
    cluster_id = (round(cluster["xw"], 1), round(cluster["yw"], 1))
    
    if cluster_id in self._cluster_velocity_ema:
        alpha = 0.3
        vx_world = alpha * raw_vx + (1-alpha) * self._cluster_velocity_ema[cluster_id][0]
        vy_world = alpha * raw_vy + (1-alpha) * self._cluster_velocity_ema[cluster_id][1]
    else:
        vx_world, vy_world = raw_vx, raw_vy
    
    self._cluster_velocity_ema[cluster_id] = (vx_world, vy_world)
```

---

## 🎯 修复优先级（按影响排序）

### 🔴 【最高优先级】必须立即修复

#### ✅ 1. 新增 r_dynamic_obs 奖励项
**预计时间**: 30 分钟  
**预期效果**: 碰撞率降低 50%+  
**实施**: 在 `get_step_result()` 中仿照 `r_social` 添加动态障碍物 TTC 惩罚

---

### 🟡 【高优先级】显著改善

#### ✅ 2. 去掉风险门控
**预计时间**: 5 分钟  
**预期效果**: 学到提前避让  
**实施**: 只过滤超出范围的障碍物

#### ✅ 3. 延长预测窗口
**预计时间**: 10 分钟  
**预期效果**: 提前 1-2 米规避  
**实施**: 根据障碍物速度动态调整 0.8s → 1.5-2.0s

---

### 🟢 【中优先级】锦上添花

#### ✅ 4. 添加安全膨胀半径
**预计时间**: 15 分钟  
**预期效果**: 减少近距离擦碰  
**实施**: 所有距离计算减去膨胀半径

#### ✅ 5. 速度平滑滤波
**预计时间**: 20 分钟  
**预期效果**: 稳定 is_dynamic 识别  
**实施**: EMA 平滑速度估计

---

## 📊 预期改善对比

| 指标 | 当前 | 修复 1 | 修复 1-3 | 修复 1-5 |
|------|------|--------|---------|----------|
| 动态障碍物碰撞率 | **40-50%** | 20-25% | 10-15% | **5-10%** |
| 提前避让距离 | 0.3-0.5m | 0.8-1.0m | 1.5-2.0m | 1.5-2.5m |
| 急刹/急转次数 | 高 | 中 | 低 | **很低** |
| 轨迹平滑度 | 差 | 中 | 好 | **很好** |

---

## 🔬 验证方法

### 测试脚本
```bash
# 训练前测试（baseline）
python3 gnn_marl_training/test_gnn_mappo.py \
    --checkpoint_path <latest_ckpt> \
    --num_episodes 20 \
    --num_agents 4 \
    --num_dynamic_obstacles 5 \
    --obs_speed_scale 0.7 \
    --save_metrics baseline_metrics.json

# 每次修复后重新测试
# ...修复后重新训练 50-100 iters...
python3 gnn_marl_training/test_gnn_mappo.py \
    ... \
    --save_metrics after_fix_1.json
```

### 对比指标
```python
import json

baseline = json.load(open('baseline_metrics.json'))
after_fix = json.load(open('after_fix_1.json'))

print(f"碰撞率: {baseline['collision_rate']:.1%} → {after_fix['collision_rate']:.1%}")
print(f"平均距离: {baseline['avg_min_dist']:.2f}m → {after_fix['avg_min_dist']:.2f}m")
```

---

## 💡 为什么"有时"会碰撞？

结合上述 5 个原因，解释为何不是"总是"碰撞：

1. **场景依赖**：
   - 障碍物慢速/远距离 → 侥幸避开
   - 障碍物快速/近距离 → 碰撞

2. **随机性**：
   - spawn 位置好 → 没有危险接近场景
   - spawn 位置差 → 立即面临冲突

3. **r_static 间接保护**：
   - 虽然没有 r_dynamic_obs，但 r_static 会惩罚"接近任何障碍物"
   - 静止障碍物学到了避让
   - 动态障碍物部分受益（但不够）

4. **学习到的局部最优**：
   - 策略学会了"大部分时候避开"
   - 但没学到"提前预判快速障碍物"
   - 导致对慢速障碍物有效，对快速障碍物失效

---

## 📝 实施计划

### 第一阶段（今日）
- ✅ 新增 r_dynamic_obs 奖励项
- ✅ 去掉风险门控
- ✅ 延长预测窗口

### 第二阶段（明日）
- 重新训练 Stage2-3，100 iters
- 对比测试碰撞率

### 第三阶段（后续）
- 添加安全膨胀半径
- 速度平滑滤波
- 最终验证

---

## 🚨 注意事项

1. **必须重新训练**：修改观测/奖励后，旧 checkpoint 不兼容
2. **课程学习**：从 Stage1 慢速障碍物开始，逐步增加难度
3. **超参调整**：r_dynamic_obs 的 scale 需要与 r_social 平衡（建议都用 0.4）

---

**分析时间**: 2026-07-02  
**下一步**: 实施修复方案 1-3
