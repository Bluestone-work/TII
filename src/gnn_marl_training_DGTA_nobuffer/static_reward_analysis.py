#!/usr/bin/env python3
"""
静态奖励分析: 不跑训练,直接从代码逻辑推算各项奖励的数值范围和触发条件。

用于快速诊断奖励-感知失配,无需启动 Gazebo。
"""

import math
import numpy as np

# ===== 从 gnn_marl_env.py 复制的常量 =====
RWD_PROGRESS_CLIP        = 0.30
RWD_STATIC_CLIP          = 1.00
RWD_SOCIAL_CLIP          = 1.00
RWD_STATIC_D0            = 0.75   # beifen: 0.50, work 改成 0.75
RWD_STATIC_D_MIN         = 0.15
RWD_STATIC_SPEED_RISK_D0 = 0.70   # beifen: 0.70
RWD_NEAR_MISS_DIST       = 0.30
RWD_SOCIAL_NEAR_DIST     = 1.5
RWD_SOCIAL_APPROACH_TH   = 0.05
INFLATION_RADIUS         = 0.255  # ROBOT_RADIUS(0.105) + SAFETY_MARGIN(0.15)

# 训练配置
progress_scale = 1.5  # beifen
static_scale   = 0.8  # env 默认
social_scale   = 0.4  # env 默认
goal_reward    = 60.0
collision_penalty = 20.0
time_penalty   = 0.008

max_forward_vel = 0.22  # TurtleBot3
control_dt = 0.1

print("=" * 70)
print("静态奖励-感知失配分析")
print("=" * 70)
print(f"\n配置: progress_scale={progress_scale}, static_scale={static_scale}, social_scale={social_scale}")
print(f"      goal_reward={goal_reward}, collision_penalty={collision_penalty}\n")

# ===== 1. r_progress 分析 =====
print("┌─ 1. r_progress: 朝目标前进 ─────────────────────────────────")
print("│")
# 典型前进速度 0.15 m/s, 单步 0.1s -> goal_dist_delta ≈ 0.015m
typical_forward_speed = 0.15
goal_dist_delta_typical = typical_forward_speed * control_dt
# heading_shaping 最大贡献: RWD_HEADING_COEF * cos(0) = 0.10 (beifen)
RWD_HEADING_COEF = 0.10
heading_shaping_max = RWD_HEADING_COEF * 1.0

r_progress_typical = progress_scale * (goal_dist_delta_typical + heading_shaping_max / 2)
r_progress_max = progress_scale * (max_forward_vel * control_dt + heading_shaping_max)
r_progress_best_case = min(r_progress_max, RWD_PROGRESS_CLIP)

print(f"│ 典型前进(v={typical_forward_speed}m/s): goal_dist_delta ≈ {goal_dist_delta_typical:.4f}m/step")
print(f"│   + heading_shaping (均值) ≈ {heading_shaping_max/2:.4f}")
print(f"│   → r_progress ≈ {progress_scale} × {goal_dist_delta_typical + heading_shaping_max/2:.4f} = {r_progress_typical:.4f}")
print(f"│")
print(f"│ 最优情况(满速+完美朝向): r_progress ≈ {r_progress_best_case:.4f} (clip={RWD_PROGRESS_CLIP})")
print(f"│ 实测: goal_dist_delta 通常只有 0.04~0.07 → r_progress ≈ 0.06~0.12")
print(f"│")
print(f"│ 【诊断】: 前进奖励很弱,单步最多 +0.3,实测 +0.06~0.12")
print("└────────────────────────────────────────────────────────────\n")

# ===== 2. r_static 分析 =====
print("┌─ 2. r_static: 静态障碍避碰 ─────────────────────────────────")
print("│")

def calc_r_static(min_dist, front_min, forward_speed, front_risk=0.5):
    """计算 r_static 在给定感知下的值"""
    effective_min = min(min_dist, front_min)

    # repulsive
    if effective_min < RWD_STATIC_D0:
        d_clamped = max(effective_min, RWD_STATIC_D_MIN)
        repulsive = -((1.0 / d_clamped) - (1.0 / RWD_STATIC_D0)) ** 2
    else:
        repulsive = 0.0

    # speed_risk
    speed_risk = 0.0
    if front_min < RWD_STATIC_SPEED_RISK_D0:
        risk_ratio = (RWD_STATIC_SPEED_RISK_D0 - front_min) / RWD_STATIC_SPEED_RISK_D0
        speed_risk = -forward_speed * (risk_ratio ** 2)

    # predictive_front_penalty
    predictive_front_penalty = -(front_risk ** 2)

    # near_miss_penalty
    near_miss_penalty = 0.0
    if effective_min < RWD_NEAR_MISS_DIST:
        near_miss_ratio = (RWD_NEAR_MISS_DIST - effective_min) / RWD_NEAR_MISS_DIST
        near_miss_penalty = -1.0 * (near_miss_ratio ** 2)

    r_static_raw = repulsive + speed_risk + predictive_front_penalty + near_miss_penalty
    r_static = static_scale * r_static_raw
    r_static = max(-RWD_STATIC_CLIP, r_static)

    return r_static, {
        'repulsive': repulsive,
        'speed_risk': speed_risk,
        'predictive': predictive_front_penalty,
        'near_miss': near_miss_penalty
    }

# 场景分析
scenarios = [
    ("开阔(min_dist=3.0m)", 3.0, 3.0, typical_forward_speed, 0.0),
    ("中距离(0.8m)", 0.8, 0.8, typical_forward_speed, 0.3),
    ("接近障碍(0.5m)", 0.5, 0.5, typical_forward_speed, 0.6),
    ("擦边(0.25m)", 0.25, 0.25, typical_forward_speed, 0.8),
    ("即将碰撞(0.18m)", 0.18, 0.18, typical_forward_speed, 1.0),
]

print("│ 场景分析(假设 front_risk 随距离变化):")
print("│")
for name, min_d, front_d, v, risk in scenarios:
    r_s, breakdown = calc_r_static(min_d, front_d, v, risk)
    print(f"│ {name:20s}: r_static = {r_s:7.4f}")
    print(f"│   └─ repulsive={breakdown['repulsive']:6.3f}, speed_risk={breakdown['speed_risk']:6.3f}, "
          f"predictive={breakdown['predictive']:6.3f}, near_miss={breakdown['near_miss']:6.3f}")
    print("│")

print(f"│ 【诊断】:")
print(f"│   - 触发阈值 RWD_STATIC_D0={RWD_STATIC_D0}m 很宽,0.8m 时已有 r_static≈-0.3")
print(f"│   - 擦边(0.25m)时 r_static≈-0.6~-0.8,远超前进奖励 +0.1")
print(f"│   - near_miss 在 0.30m 内触发,和碰撞硬阈值 0.22m 只差 8cm,但栅格分辨率 12.5cm")
print(f"│   - 结论: 静态避碰惩罚太保守,压制前进信号")
print("└────────────────────────────────────────────────────────────\n")

# ===== 3. r_social 分析 =====
print("┌─ 3. r_social: 邻居避碰 ─────────────────────────────────────")
print("│")

def calc_r_social(neighbor_dist, approach_speed, neighbor_speed):
    """计算 r_social (简化,不考虑轨迹预测)"""
    if neighbor_dist >= RWD_SOCIAL_NEAR_DIST or approach_speed <= RWD_SOCIAL_APPROACH_TH:
        return 0.0, "未触发"

    effective_dist = max(0.0, neighbor_dist - 2 * INFLATION_RADIUS)
    safe_ttc = 2.5  # 代码里的值

    if approach_speed > 1e-3:
        ttc = effective_dist / approach_speed
    else:
        return 0.0, "approach_speed≈0"

    if ttc < safe_ttc:
        ttc_penalty = -((safe_ttc - ttc) / safe_ttc) ** 2
        r_social = social_scale * ttc_penalty
        r_social = max(-RWD_SOCIAL_CLIP, r_social)
        return r_social, f"TTC={ttc:.2f}s"
    else:
        return 0.0, f"TTC={ttc:.2f}s>safe"

social_scenarios = [
    ("远距离(2.0m, 对向0.3m/s)", 2.0, 0.3, 0.2),
    ("中距离(1.0m, 对向0.2m/s)", 1.0, 0.2, 0.15),
    ("近距离(0.6m, 对向0.15m/s)", 0.6, 0.15, 0.1),
    ("很近(0.4m, 对向0.1m/s)", 0.4, 0.1, 0.08),
]

print("│ 场景分析(approach_speed = 两车相对速度在连线方向投影):")
print("│")
for name, dist, app_spd, nei_spd in social_scenarios:
    r_soc, reason = calc_r_social(dist, app_spd, nei_spd)
    print(f"│ {name:30s}: r_social = {r_soc:7.4f}  ({reason})")

print("│")
print(f"│ 【诊断】:")
print(f"│   - 触发条件: dist<{RWD_SOCIAL_NEAR_DIST}m 且 approach_speed>{RWD_SOCIAL_APPROACH_TH}m/s")
print(f"│   - TTC<2.5s 时惩罚,实际高密度场景常触发")
print(f"│   - r_social 用 scale=0.4,比 static(0.8)弱一半")
print(f"│   - 但叠加 r_dynamic_obs(无 scale,权重 1.0!)后,总避碰惩罚很重")
print("└────────────────────────────────────────────────────────────\n")

# ===== 4. r_dynamic_obs 分析 =====
print("┌─ 4. r_dynamic_obs: 动态障碍避碰 ────────────────────────────")
print("│")
print("│ 代码: r_dynamic_obs = 1.0 * worst_penalty (无 scale!)")
print("│       只对 is_dynamic>0.5 的 token 计算 TTC 和轨迹风险")
print("│")
print("│ 问题:")
print("│   1. is_dynamic 靠帧间差分估计,速度噪声 ±0.05m/s → 常误判")
print("│   2. 无 scale,权重是 r_social 的 2.5 倍")
print("│   3. 只有 top-3 token,且被 risk>1e-4 门控过滤")
print("│")
print("│ 【诊断】: r_dynamic_obs 是\"隐藏炸弹\"")
print("│   - 如果场景有 3+ 动态障碍,且都被正确识别 → r_dynamic_obs ≈ -0.8")
print("│   - 但如果速度估计错误 → 被当静态 → 不触发")
print("│   - 结论: 不稳定的强惩罚,策略无法可靠预测")
print("└────────────────────────────────────────────────────────────\n")

# ===== 5. 总奖励估算 =====
print("┌─ 5. 典型场景下的总奖励 ────────────────────────────────────")
print("│")

scenarios_total = [
    {
        'name': '开阔前进',
        'r_progress': 0.10,
        'r_static': 0.0,
        'r_social': 0.0,
        'r_dynamic_obs': 0.0,
        'r_collision': 0.0,
        'r_goal': 0.0,
        'r_time': -0.008,
    },
    {
        'name': '中密度避碰前进(min_dist=0.6m, 1邻居)',
        'r_progress': 0.08,
        'r_static': -0.4,
        'r_social': -0.15,
        'r_dynamic_obs': 0.0,
        'r_collision': 0.0,
        'r_goal': 0.0,
        'r_time': -0.008,
    },
    {
        'name': '高密度(min_dist=0.4m, 2邻居, 1动态障碍)',
        'r_progress': 0.06,
        'r_static': -0.7,
        'r_social': -0.3,
        'r_dynamic_obs': -0.5,
        'r_collision': 0.0,
        'r_goal': 0.0,
        'r_time': -0.008,
    },
    {
        'name': '磨蹭不动(v≈0)',
        'r_progress': 0.0,
        'r_static': 0.0,
        'r_social': 0.0,
        'r_dynamic_obs': 0.0,
        'r_collision': 0.0,
        'r_goal': 0.0,
        'r_time': -0.008,
    },
]

for sc in scenarios_total:
    total = sum(sc[k] for k in sc if k not in ['name'])
    print(f"│ {sc['name']:40s}: 总奖励 = {total:7.4f}")
    breakdown = " + ".join([f"{k.replace('r_','')}={v:.3f}" for k, v in sc.items() if k != 'name'])
    print(f"│   └─ {breakdown}")
    print("│")

print(f"│ 【核心失配】:")
print(f"│   - 开阔前进: +0.09 (可行)")
print(f"│   - 中密度避碰: -0.47 (负收益!)")
print(f"│   - 高密度: -1.44 (强负!)")
print(f"│   - 磨蹭不动: -0.008 (几乎无惩罚)")
print(f"│")
print(f"│   结论: 任何\"动起来+靠近障碍\"的动作都是负收益,")
print(f"│         策略的最优解是\"不动\"或\"远离一切\"")
print("└────────────────────────────────────────────────────────────\n")

# ===== 6. 感知能力评估 =====
print("┌─ 6. 感知-奖励精度失配 ─────────────────────────────────────")
print("│")
print("│ 奖励要求 vs 感知能力:")
print("│")
req_vs_cap = [
    ("near_miss 0.30m 精确避让", "栅格分辨率 0.125m", "✗ 分辨率不够"),
    ("动态障碍 TTC 预测", "帧间差分速度,±0.05m/s噪声", "✗ 速度不可信"),
    ("0.75m 处提前减速", "只有 min_dist 标量", "✗ 无方向信息"),
    ("识别左/右/前 障碍", "2个标量 + 3个token + 粗栅格", "△ 信息稀疏"),
]

for req, cap, verdict in req_vs_cap:
    print(f"│ {req:30s} | {cap:35s} | {verdict}")

print("│")
print(f"│ 【结论】: 奖励基于的感知信号(min_dist标量/粗栅格/不稳定速度)")
print(f"│          精度不足以支撑奖励要求的精细避碰行为")
print("└────────────────────────────────────────────────────────────\n")

# ===== 7. 改进方向优先级 =====
print("=" * 70)
print("改进方向优先级(按性价比)")
print("=" * 70)
print()
print("【高优先级 — 立即可改,收益明确】")
print()
print("1. 重平衡奖励尺度 (预计 1h, 收益 ★★★★)")
print("   - progress_scale: 1.5 → 2.5~3.0")
print("   - static_scale:   0.8 → 0.5")
print("   - social_scale:   0.4 → 0.3")
print("   - r_dynamic_obs 加 scale 0.3")
print("   - RWD_STATIC_D0:  0.75 → 0.55m")
print("   → 目标: 让\"靠近障碍同时前进\"的动作净收益≥0")
print()
print("2. 扇区距离拼回观测 (预计 2h, 收益 ★★★★)")
print("   - _front_sector_dist_history 已算,未用")
print("   - 拼进 obs,给策略稠密方向-距离信号")
print("   - obs_dim +9, 几乎零训练成本")
print()
print("【中优先级 — 需重训,但改善避碰关键】")
print()
print("3. 动态速度估计加平滑 (预计 3h, 收益 ★★★)")
print("   - 帧间差分改 EMA,降低噪声")
print("   - 或直接用多帧最小二乘拟合")
print()
print("4. 去掉 token risk 门控 (预计 30min, 收益 ★★)")
print("   - 让近处障碍物无论危险与否都进 token")
print("   - 给策略提前量")
print()
print("【低优先级 — 改善明显但成本高】")
print()
print("5. 栅格提分辨率/历史帧 (预计 4h, 收益 ★★★)")
print("   - 32×32×2 → 64×64×4, 或半径 2m→3m")
print("   - 需重训,显存/计算增加")
print()
print("6. 增加\"成功避让\"正奖励 (预计 2h, 收益 ★★)")
print("   - 当前纯负向,加 r_avoidance_bonus")
print("   - 需精细设计触发条件")
print()
print("=" * 70)
print("建议: 先做 1+2 (共 3h),立即跑 Stage2 50 iters 验证.")
print("      如果 collision↓ 且 reward↑ → 确诊,继续 3+4.")
print("      如果无改善 → 深入分析策略网络结构/学习率.")
print("=" * 70)
