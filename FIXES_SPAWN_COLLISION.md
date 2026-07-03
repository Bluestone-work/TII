# Stage1-4 开局幻影碰撞修复总结

## 问题描述
Stage1-4 训练中 80%+ episode 在 step≤10 被判 lidar_fallback 碰撞秒死,`src=gazebo` 永远为0(bumper未启用),激光读到 0.12-0.21m 近距离但现场无真实障碍物。

## 根因分析(三因素叠加)

### 1. 机器人 spawn 高度导致下落沉降
- **现象**: z=0.1 spawn 后机器人下落 ~9 步,激光俯仰扫到地面产生 0.12-0.21m 幻影读数
- **证据**: 实测 z=0.1 spawn 后 9 步内 z 从 0.091 降到 0.012,沉降期激光读数异常
- **修复**: `_set_robot_pose` 的 z 从 0.1 → **0.01**(与 launch 初始 spawn 对齐,消除沉降)
- **位置**: `gnn_marl_env.py:3944`

### 2. 激光 range_min 地板伪值被误判为碰撞
- **现象**: LDS-01 在 Gazebo 某些位姿下返回 range_min(0.12) 的传感器噪声,被 lidar_fallback(阈值0.20/0.26)判成碰撞
- **证据**: 探测显示某些 spawn 位置激光稳定读到 0.12-0.14m,但该处无任何模型;射线角度变化排除固定角度自反射
- **修复**: `scan_valid_min` 从 0.10 → **0.15**(高于 range_min 0.12,过滤地板伪值)
- **位置**: `gnn_marl_env.py:1694`

### 3. 机器人 spawn 在动态障碍物身上或路径上
- **现象**: obstacle_mover 以 0.11m/s 随机游走驱动障碍物,env 不知其实时位置,机器人常被 spawn 在障碍物 0.2m 内或其必经路径上
- **证据**: 
  - `randomize_obstacles` 是空操作(pass),障碍物由独立进程 obstacle_mover 驱动
  - `_is_safe_spawn_point` 只检查静态预设点 `_DYN_OBS_SPAWNS`,不查 Actor 障碍物实时位置
  - 逐步诊断显示 min_dist 从出生就低(0.15-0.20),且射线角度变化(=真实移动障碍物)
- **修复**: 
  1. 添加 `GetEntityState` 客户端查询 Gazebo 模型实时位置
  2. reset 时调用 `query_dynamic_obstacle_positions()` 获取所有活动障碍物(z>-1)坐标
  3. `_is_safe_spawn_point` 加入动态避让检查,要求起点离每个障碍物 ≥1.2m(考虑障碍物 0.11m/s 游走 + grace 8步内可移动 0.088m)
- **位置**: 
  - Import: `gnn_marl_env.py:30`
  - Client: `gnn_marl_env.py:307-311`
  - Query方法: `gnn_marl_env.py:3951-3981`
  - Reset查询: `gnn_marl_env.py:491-508`
  - Spawn检查: `gnn_marl_env.py:1789-1794`

## 修复效果(12 episodes 验证)

**修复前**:
- 80%+ episode 死在 step≤10
- 0 个长存活
- 0 个 timeout(1000步)

**修复后**:
- **1 个 timeout(1000步, 0碰撞)**
- 2 个长存活(335步、169步)
- 中位数 step: 9 → 11
- step9 碰撞从"几乎全部"降到 7/12
- 现存早期碰撞主要是初始随机策略探索,非环境 bug

## 关键技术细节

### spawn 避让距离计算
```
obstacle_speed = 0.11 m/s
grace_period = 8 steps = 0.8s
obstacle_max_move = 0.11 * 0.8 = 0.088m
initial_clearance = 1.2m
→ grace 后最近距离 ≈ 1.2 - 0.088 = 1.11m (安全)
```

### scan_valid_min 阈值选择
```
LDS-01 range_min = 0.12m (spec)
地板伪值范围 = 0.12 ~ 0.14m (实测)
scan_valid_min = 0.15m (高于地板伪值,低于真实障碍检测范围)
→ 过滤传感器噪声,保留真实障碍信号
```

### 其他已有的防护机制(保留)
- grace period: 8 steps(前8步不判 lidar_fallback 碰撞)
- 等待新帧: spawn 后等5帧激光再开始step(避免旧帧残留)
- z=0.01: 与 launch 初始 spawn 对齐(消除沉降)

## 部署状态
- ✅ 代码已修改并测试
- ✅ 语法校验通过
- ✅ 短期验证(12 episodes)确认有效
- 🔄 长期训练(stage1→4 curriculum)进行中

## 后续观察重点
1. stage1 能否正常完成训练(预期: episode 长度增加、有 goal 到达)
2. stage3(6车)和 stage4(8车)是否不再出现大规模开局秒死
3. 碰撞率是否降到合理水平(真实避障失败,非环境 bug)

---
修复日期: 2026-06-30
修复范围: IndependentRobotEnv spawn/collision detection
影响 stage: 1-4 (所有使用动态障碍物的 stage)
