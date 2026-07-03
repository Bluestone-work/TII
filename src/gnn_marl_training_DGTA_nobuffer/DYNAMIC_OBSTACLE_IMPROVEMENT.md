# 动态障碍物避碰能力提升方案

## 问题分析

当前系统存在以下问题：

1. **r_social 只计算多机器人间的 TTC，不考虑动态障碍物**
2. **动态障碍物信息只在观测中，没有奖励信号引导**
3. **预测窗口较短**（0.5-1.0秒）
4. **动态障碍物数量和速度课程不足**

---

## 改进方案

### 1. 新增 r_dynamic_obstacle 奖励项

在 `get_step_result()` 中添加专门的动态障碍物避碰奖励：

```python
# ==========================================
# 新增: r_dynamic_obs: 动态障碍物 TTC 惩罚
# ==========================================
r_dynamic_obs = 0.0
if self.obstacle_motion_dim > 0:
    motion_features = self._get_obstacle_motion_features(sector_dists)
    # motion_features 格式: [x, y, vx, vy, future_x, future_y, is_dynamic] * top_k
    
    my_pos = np.array([self.current_pose['x'], self.current_pose['y']])
    my_vel = np.array([
        self.current_vel_x * math.cos(self.current_pose['yaw']),
        self.current_vel_x * math.sin(self.current_pose['yaw'])
    ])
    
    worst_penalty = 0.0
    for i in range(self.obstacle_motion_top_k):
        start = i * 7
        if start + 6 >= len(motion_features):
            break
        
        is_dynamic = motion_features[start + 6]
        if is_dynamic < 0.5:  # 静态障碍物跳过
            continue
        
        # 反归一化
        obs_x = motion_features[start] * 5.0  # body frame
        obs_y = motion_features[start + 1] * 5.0
        obs_vx = motion_features[start + 2] * 0.8
        obs_vy = motion_features[start + 3] * 0.8
        
        # 转到世界坐标
        cos_yaw = math.cos(self.current_pose['yaw'])
        sin_yaw = math.sin(self.current_pose['yaw'])
        obs_pos_world = my_pos + np.array([
            obs_x * cos_yaw - obs_y * sin_yaw,
            obs_x * sin_yaw + obs_y * cos_yaw
        ])
        obs_vel_world = np.array([
            obs_vx * cos_yaw - obs_vy * sin_yaw,
            obs_vx * sin_yaw + obs_vy * cos_yaw
        ])
        
        rel_pos = obs_pos_world - my_pos
        dist = float(np.linalg.norm(rel_pos))
        
        if dist > 2.0:  # 超过2米不考虑
            continue
        
        # 相对速度
        rel_vel = obs_vel_world - my_vel
        approach_speed = float(-np.dot(rel_pos, rel_vel) / (dist + 1e-6))
        
        if approach_speed > 0.05:  # 正在接近
            ttc = dist / approach_speed
            safe_ttc = 2.5  # 动态障碍物需要更长的安全时间
            if ttc < safe_ttc:
                penalty = -((safe_ttc - ttc) / safe_ttc) ** 2
                if penalty < worst_penalty:
                    worst_penalty = penalty
    
    r_dynamic_obs = 1.0 * worst_penalty  # 权重1.0
    r_dynamic_obs = float(max(-1.0, r_dynamic_obs))

# 总奖励
reward = r_progress + r_static + r_social + r_dynamic_obs + r_collision + r_goal + r_time
```

**效果:** 直接惩罚与动态障碍物的危险接近，训练会学会提前避让。

---

### 2. 增加动态障碍物预测窗口

修改 `_get_obstacle_motion_features()` 中的预测时间：

```python
# 第2298行附近
predict_window = 1.5  # 从 0.5-1.0 增加到 1.5 秒
future_x = float(x + vx * predict_window)
future_y = float(y + vy * predict_window)
```

**效果:** 更长的预测窗口让智能体看到更远的未来轨迹。

---

### 3. 调整训练课程（渐进式困难）

修改 `run_curriculum.sh` 中动态障碍物配置：

```bash
# Stage1: 慢速少量动态障碍物
STAGE_OBS_NUM[1]=2
STAGE_OBS_SPD[1]=0.3

# Stage2: 中速中量
STAGE_OBS_NUM[2]=3
STAGE_OBS_SPD[2]=0.5

# Stage3: 快速多量
STAGE_OBS_NUM[3]=4
STAGE_OBS_SPD[3]=0.65

# Stage4: 极限
STAGE_OBS_NUM[4]=5
STAGE_OBS_SPD[4]=0.75
```

**效果:** 逐步增加难度，避免一开始就过难导致学不到有效策略。

---

### 4. 增强观测特征（可选）

在 obstacle_motion_features 中添加**碰撞风险度**：

```python
# 第2312行附近，添加第8个特征
collision_risk = 0.0
if is_moving and dist > 0.05:
    ttc = dist / (np.linalg.norm([vx, vy]) + 1e-6)
    collision_risk = float(np.clip(1.0 - ttc / 3.0, 0.0, 1.0))

token = np.array([
    float(np.clip(x / denom, -1.0, 1.0)),
    float(np.clip(y / denom, -1.0, 1.0)),
    float(np.clip(vx / 0.8, -1.0, 1.0)),
    float(np.clip(vy / 0.8, -1.0, 1.0)),
    float(np.clip(future_x / denom, -1.0, 1.0)),
    float(np.clip(future_y / denom, -1.0, 1.0)),
    is_dynamic,
    collision_risk,  # 新增
], dtype=np.float32)
```

需要同步修改 `obstacle_motion_feature_dim = 8`。

---

## 实施优先级

### 立即实施（效果最明显）
1. ✅ **方案1: 新增 r_dynamic_obs** - 最直接有效
2. ✅ **方案3: 调整训练课程** - 提升收敛速度

### 后续优化
3. ⭐ **方案2: 增加预测窗口** - 提升前瞻能力
4. ⭐ **方案4: 增强观测特征** - 需要重新训练

---

## 预期效果

- **碰撞率降低**: 从当前 30-40% 降低到 15-25%
- **提前避让**: 智能体会在 1.5-2米外开始规避动作
- **平滑轨迹**: 减少急刹和急转，更像人类驾驶

---

## 测试方法

训练 Stage2 (4车) 完成后：

```bash
# 测试动态障碍物避碰
python test_gnn_mappo.py \
  --checkpoint_path <Stage2_ckpt> \
  --num_episodes 10 \
  --num_agents 4 \
  --num_dynamic_obstacles 5 \
  --obs_speed_scale 0.7 \
  --save_gif
```

观察：
- 机器人是否提前减速/绕行
- 碰撞次数
- 轨迹平滑度
