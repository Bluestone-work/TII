# 动态障碍物运动感知增强方案

## 你的想法分析

### ✅ 非常合理的两个核心点：

1. **障碍物膨胀（Safety Inflation）**
   - 将障碍物周围一定范围视为危险区域
   - 相当于给机器人一个"安全气囊"
   - 标准做法：膨胀半径 = 机器人半径 + 安全裕度

2. **线性运动预测（Linear Motion Prediction）**
   - 根据当前速度预测未来位置
   - 让机器人"看到"障碍物的运动趋势
   - 提前避让而不是被动反应

---

## 当前代码已有的功能

✅ **已实现的预测：**
```python
# 第2298行附近
predict_window = 0.5 到 1.0 秒
future_x = x + vx * predict_window
future_y = y + vy * predict_window
```

✅ **已有的风险评估：**
- `close_risk`: 当前距离风险
- `future_risk`: 预测位置风险
- `ttc_risk`: Time-To-Collision 风险
- `crossing_risk`: 横穿风险

⚠️ **不足之处：**
1. **预测时间太短**（0.5-1.0秒）
2. **没有显式的膨胀半径**
3. **风险计算没有考虑机器人尺寸**
4. **预测轨迹不可视**

---

## 改进方案

### 改进1: 增加安全膨胀半径

```python
# 在 gnn_marl_env.py 添加配置
ROBOT_RADIUS = 0.105  # TurtleBot3 半径 (m)
SAFETY_MARGIN = 0.15  # 额外安全裕度 (m)
INFLATION_RADIUS = ROBOT_RADIUS + SAFETY_MARGIN  # = 0.255m

# 修改风险计算（第2301行附近）
# 原来：
close_risk = float(np.clip((self.close_obstacle_dist - dist) / self.close_obstacle_dist, 0.0, 1.0))

# 改为：
effective_dist = max(0.0, dist - INFLATION_RADIUS)  # 膨胀后的有效距离
close_risk = float(np.clip((self.close_obstacle_dist - effective_dist) / self.close_obstacle_dist, 0.0, 1.0))
```

**效果：**
- 障碍物在 0.255m 范围内就触发高风险
- 相当于把障碍物"变大"了，增加安全裕度

---

### 改进2: 延长预测时间窗口

```python
# 第2287-2298行附近
# 原来：
predict_window = 0.5 if dist > 1.5 else 1.0

# 改为：根据障碍物速度动态调整
obs_speed = math.hypot(vx, vy)
if obs_speed > 0.5:  # 快速运动
    predict_window = 2.0  # 预测更远
elif obs_speed > 0.3:  # 中速
    predict_window = 1.5
else:  # 慢速/静止
    predict_window = 1.0
```

**效果：**
- 快速障碍物提前 2 秒预测
- 给机器人更多反应时间

---

### 改进3: 多步预测轨迹

不只预测 1 个点，预测一条轨迹：

```python
def _predict_obstacle_trajectory(self, x, y, vx, vy, num_steps=5, dt=0.4):
    """预测障碍物未来轨迹（多个时间步）"""
    trajectory = []
    for i in range(1, num_steps + 1):
        t = i * dt
        pred_x = x + vx * t
        pred_y = y + vy * t
        trajectory.append((pred_x, pred_y))
    return trajectory

# 在 _get_obstacle_motion_features 中使用
trajectory = self._predict_obstacle_trajectory(x, y, vx, vy)
# 检查轨迹是否与机器人路径相交
path_collision_risk = self._check_trajectory_collision(trajectory, my_path)
```

**效果：**
- 预测 5 个时间步，每步 0.4 秒（共 2 秒）
- 检测整条轨迹是否与机器人路径冲突
- 更准确的碰撞预测

---

### 改进4: 考虑机器人自身运动的相对风险

```python
# 在 r_dynamic_obs 计算中（第3229行之后）
# 当前只考虑障碍物接近，应该考虑相对运动

# 改进：计算相对速度在连线方向上的分量
rel_pos = obs_pos_world - my_pos
rel_vel = obs_vel_world - my_vel
dist = float(np.linalg.norm(rel_pos))

# 相对速度在连线方向的投影
rel_pos_normalized = rel_pos / (dist + 1e-6)
approach_speed = -float(np.dot(rel_vel, rel_pos_normalized))

# 只有正在接近才计算 TTC
if approach_speed > 0.05:
    # 考虑膨胀半径的有效距离
    effective_dist = max(0.0, dist - INFLATION_RADIUS)
    ttc = effective_dist / approach_speed
    safe_ttc = 2.5
    
    if ttc < safe_ttc:
        # 距离越近、速度越快，惩罚越大
        dist_factor = max(0.0, 1.0 - effective_dist / 2.0)
        speed_factor = min(1.0, approach_speed / 0.8)
        penalty = -(dist_factor * speed_factor * (safe_ttc - ttc) / safe_ttc) ** 2
```

**效果：**
- 同时考虑距离和相对速度
- 快速接近的障碍物惩罚更大
- 膨胀半径纳入计算

---

## 实施优先级

### 立即实施（最有效）
1. ✅ **改进1: 安全膨胀半径** - 5分钟，立竿见影
2. ✅ **改进2: 延长预测窗口** - 5分钟，显著提升

### 后续优化
3. ⭐ **改进4: 相对风险计算** - 10分钟，更精确
4. ⭐ **改进3: 多步预测** - 30分钟，最复杂但效果最好

---

## 预期效果对比

| 指标 | 当前 | 改进后 |
|------|------|--------|
| 动态障碍物碰撞率 | 30-40% | **15-25%** |
| 提前避让距离 | 0.5-1.0m | **1.5-2.0m** |
| 急刹/急转次数 | 高 | **显著降低** |
| 轨迹平滑度 | 中等 | **明显改善** |

---

## 可视化验证

在 RViz 中显示：
- 🔴 当前障碍物位置
- 🟡 预测未来位置（1s, 1.5s, 2s）
- ⭕ 膨胀安全区域
- 🟢 机器人预计路径

这样能直观看到预测是否准确。

---

## 需要我立即实施吗？

我可以现在就修改代码，实施改进1和改进2（10分钟完成），还是你想先看看方案？
