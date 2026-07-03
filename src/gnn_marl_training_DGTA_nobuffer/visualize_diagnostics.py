#!/usr/bin/env python3
"""
可视化诊断数据: 从 collect_episode_diagnostics.py 输出的 JSONL 生成:
  1. 奖励-距离相关性曲线
  2. 动态识别率统计
  3. 碰撞事件分布
  4. 感知-奖励失配的定量证据

输出: HTML 报告 + PNG 图表
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def load_data(jsonl_path):
    """加载 JSONL 数据到 pandas DataFrame"""
    records = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def analyze_reward_distance_correlation(df, output_dir):
    """分析 min_dist vs reward 的关系"""
    print("\n[分析 1] 奖励-距离相关性")

    # 按 min_dist 分箱
    bins = [0.0, 0.25, 0.35, 0.5, 0.75, 1.0, 2.0, 10.0]
    df['dist_bin'] = pd.cut(df['min_dist'], bins=bins)

    # 统计每个 bin 的奖励分布
    stats = df.groupby('dist_bin')['reward'].agg(['mean', 'std', 'count', 'min', 'max'])
    print("\n距离分段的奖励统计:")
    print(stats)

    # 画图: 箱线图
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df[df['min_dist'] < 2.5].copy()  # 只画 <2.5m 的
    sns.boxplot(data=df_plot, x='dist_bin', y='reward', ax=ax)
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='reward=0 基线')
    ax.set_xlabel('min_dist 分段 (m)', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('奖励 vs 最近障碍距离 (箱线图)', fontsize=14)
    ax.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(output_dir / 'reward_vs_distance_boxplot.png', dpi=150)
    plt.close(fig)
    print(f"  → 图表: {output_dir / 'reward_vs_distance_boxplot.png'}")

    # 散点图 (采样)
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(5000, len(df)))
    ax.scatter(sample['min_dist'], sample['reward'], alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('min_dist (m)', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('奖励 vs 距离 散点图 (采样)', fontsize=14)
    ax.set_xlim(0, 2.5)
    plt.tight_layout()
    fig.savefig(output_dir / 'reward_vs_distance_scatter.png', dpi=150)
    plt.close(fig)

    # 关键发现
    findings = []
    # 检查 0.5-0.75m 的奖励均值
    mask_mid = (df['min_dist'] >= 0.5) & (df['min_dist'] < 0.75)
    if mask_mid.sum() > 10:
        reward_mid = df.loc[mask_mid, 'reward'].mean()
        findings.append(f"中距离(0.5-0.75m): 奖励均值 {reward_mid:.4f}")
        if reward_mid < -0.2:
            findings.append("  ⚠️  中距离已有强负奖励,说明 static 阈值太保守")

    # 检查奖励转正的距离
    positive_mask = df['reward'] > 0.05
    if positive_mask.sum() > 0:
        min_dist_for_positive = df.loc[positive_mask, 'min_dist'].quantile(0.1)
        findings.append(f"奖励转正的距离: ≥{min_dist_for_positive:.2f}m (10% 分位)")
    else:
        findings.append("⚠️  几乎没有正奖励步 (奖励失衡严重)")

    return stats, findings


def analyze_dynamic_detection(df, output_dir):
    """分析动态障碍物识别率"""
    print("\n[分析 2] 动态障碍物识别")

    # 统计有多少步检测到动态 token
    has_dynamic = df['num_dynamic_tokens'] > 0
    detection_rate = has_dynamic.mean()
    print(f"  检测到动态 token 的步数占比: {detection_rate:.2%}")
    print(f"  动态 token 数量分布:")
    print(df['num_dynamic_tokens'].value_counts().sort_index())

    # 动态 token 距离分布
    token_dists = []
    for dists in df['token_distances']:
        if isinstance(dists, list):
            token_dists.extend(dists)

    if token_dists:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(token_dists, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('动态 token 距离 (m)', fontsize=12)
        ax.set_ylabel('频次', fontsize=12)
        ax.set_title('动态障碍物距离分布', fontsize=14)
        ax.axvline(np.median(token_dists), color='red', linestyle='--',
                   label=f'中位数={np.median(token_dists):.2f}m')
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / 'dynamic_token_distance_hist.png', dpi=150)
        plt.close(fig)

    findings = []
    findings.append(f"动态识别率: {detection_rate:.2%}")
    if detection_rate < 0.3:
        findings.append("  ⚠️  动态识别率低,速度估计可能不稳定")
    if token_dists:
        findings.append(f"动态 token 距离中位数: {np.median(token_dists):.2f}m")

    return findings


def analyze_collision_events(df, output_dir):
    """分析碰撞事件"""
    print("\n[分析 3] 碰撞事件")

    collision_steps = df[df['event'] == 'collision']
    collision_rate = len(collision_steps) / len(df)
    print(f"  碰撞步数占比: {collision_rate:.2%} ({len(collision_steps)}/{len(df)})")

    if len(collision_steps) > 0:
        print(f"  碰撞时 min_dist 分布:")
        print(collision_steps['min_dist'].describe())

        # 碰撞前若干步的轨迹
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左: 碰撞时的 min_dist 分布
        axes[0].hist(collision_steps['min_dist'], bins=20, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('min_dist at collision (m)', fontsize=12)
        axes[0].set_ylabel('频次', fontsize=12)
        axes[0].set_title('碰撞时的 min_dist 分布', fontsize=14)
        axes[0].axvline(0.22, color='red', linestyle='--', label='碰撞阈值 0.22m')
        axes[0].legend()

        # 右: 碰撞前 10 步的 min_dist 变化 (选几个样本)
        sample_collisions = collision_steps.head(min(5, len(collision_steps)))
        for _, row in sample_collisions.iterrows():
            ep_id = row['episode_id']
            step_id = row['step']
            aid = row['agent_id']
            # 找该 episode 该 agent 碰撞前 10 步
            pre_steps = df[
                (df['episode_id'] == ep_id) &
                (df['agent_id'] == aid) &
                (df['step'] >= step_id - 10) &
                (df['step'] <= step_id)
            ].sort_values('step')
            if len(pre_steps) > 1:
                axes[1].plot(pre_steps['step'] - step_id, pre_steps['min_dist'],
                            marker='o', alpha=0.6, label=f'ep{ep_id}_{aid}')

        axes[1].axhline(0.22, color='red', linestyle='--', linewidth=1, label='碰撞阈值')
        axes[1].set_xlabel('相对步数 (0=碰撞)', fontsize=12)
        axes[1].set_ylabel('min_dist (m)', fontsize=12)
        axes[1].set_title('碰撞前轨迹 (样本)', fontsize=14)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / 'collision_analysis.png', dpi=150)
        plt.close(fig)

    findings = []
    findings.append(f"碰撞率: {collision_rate:.2%}")
    if len(collision_steps) > 0:
        median_dist = collision_steps['min_dist'].median()
        findings.append(f"碰撞时 min_dist 中位数: {median_dist:.3f}m")
        if median_dist > 0.25:
            findings.append("  ⚠️  碰撞时 min_dist>0.25m,可能是激光读数滞后或栅格太粗")

    return findings


def analyze_velocity_reward_correlation(df, output_dir):
    """分析速度和奖励的关系"""
    print("\n[分析 4] 速度-奖励相关性")

    # 过滤掉 goal/collision 等终端步
    normal_steps = df[~df['event'].isin(['goal', 'collision'])]

    # 按速度分箱
    vel_bins = [0.0, 0.02, 0.05, 0.1, 0.15, 0.22]
    normal_steps['vel_bin'] = pd.cut(normal_steps['vel_x'].abs(), bins=vel_bins)

    stats = normal_steps.groupby('vel_bin')['reward'].agg(['mean', 'std', 'count'])
    print("\n速度分段的奖励统计:")
    print(stats)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=normal_steps, x='vel_bin', y='reward', ax=ax)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('|vel_x| 分段 (m/s)', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title('奖励 vs 速度', fontsize=14)
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(output_dir / 'reward_vs_velocity.png', dpi=150)
    plt.close(fig)

    findings = []
    # 检查 "不动" vs "前进" 的奖励差异
    still_mask = normal_steps['vel_x'].abs() < 0.02
    moving_mask = normal_steps['vel_x'].abs() > 0.1
    if still_mask.sum() > 10 and moving_mask.sum() > 10:
        reward_still = normal_steps.loc[still_mask, 'reward'].mean()
        reward_moving = normal_steps.loc[moving_mask, 'reward'].mean()
        findings.append(f"不动(v<0.02): 奖励均值 {reward_still:.4f}")
        findings.append(f"前进(v>0.1): 奖励均值 {reward_moving:.4f}")
        if reward_still > reward_moving:
            findings.append("  ⚠️  不动比前进奖励更高,奖励失衡确诊!")

    return findings


def generate_html_report(all_findings, output_dir):
    """生成 HTML 报告"""
    html_path = output_dir / 'diagnosis_report.html'

    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>避碰策略诊断报告</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; max-width: 1200px; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        .finding { background: #ecf0f1; padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; }
        .warning { background: #ffe6e6; border-left-color: #e74c3c; }
        img { max-width: 100%; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .summary { background: #d5f4e6; padding: 15px; margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🔍 避碰策略定量诊断报告</h1>
    <p><strong>生成时间:</strong> 2026-06-30</p>
"""

    for section, findings in all_findings.items():
        html_content += f"<h2>{section}</h2>\n"
        for finding in findings:
            css_class = 'warning' if '⚠️' in finding else 'finding'
            html_content += f'<div class="{css_class}">{finding}</div>\n'

    # 添加图表
    html_content += "<h2>📊 可视化分析</h2>\n"
    for img_file in sorted(output_dir.glob('*.png')):
        html_content += f'<h3>{img_file.stem.replace("_", " ").title()}</h3>\n'
        html_content += f'<img src="{img_file.name}" alt="{img_file.stem}">\n'

    # 总结
    html_content += """
    <div class="summary">
        <h2>💡 改进建议</h2>
        <p>基于上述量化分析,建议按优先级改进:</p>
        <ol>
            <li><strong>奖励重平衡</strong> (如果发现 "不动 > 前进" 或中距离强负)</li>
            <li><strong>扇区距离拼回观测</strong> (如果碰撞率高但 min_dist 在合理范围)</li>
            <li><strong>动态速度平滑</strong> (如果动态识别率 <30%)</li>
        </ol>
        <p>详见 <code>DIAGNOSIS_REPORT.md</code> 完整方案。</p>
    </div>
    </body>
    </html>
    """

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\n[报告] HTML 已生成: {html_path}")
    return html_path


def analyze_risk_exposure(df, output_dir):
    """分析风险暴露：即使不碰撞，也量化危险驾驶行为"""
    print("\n[分析 5] 风险暴露度量")

    findings = []

    # 风险暴露时间占比
    risk_threshold = 0.3  # <0.3m 视为危险
    close_threshold = 0.5  # <0.5m 视为接近

    total_steps = len(df)
    risk_steps = (df['min_dist'] < risk_threshold).sum()
    close_steps = (df['min_dist'] < close_threshold).sum()

    risk_ratio = risk_steps / max(1, total_steps)
    close_ratio = close_steps / max(1, total_steps)

    findings.append(f"危险区间(<0.3m)时间占比: {risk_ratio*100:.2f}%")
    findings.append(f"接近区间(<0.5m)时间占比: {close_ratio*100:.2f}%")

    # 危险接近事件：min_dist 从安全突降到危险
    df_sorted = df.sort_values(['episode_id', 'agent_id', 'step'])
    df_sorted['min_dist_prev'] = df_sorted.groupby(['episode_id', 'agent_id'])['min_dist'].shift(1)
    near_miss_mask = (df_sorted['min_dist_prev'] > close_threshold) & (df_sorted['min_dist'] < risk_threshold)
    near_miss_count = near_miss_mask.sum()

    findings.append(f"危险接近事件(0.5m→<0.3m突降): {near_miss_count} 次")

    # 安全边界分布直方图
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0]
    hist, bin_edges = np.histogram(df['min_dist'], bins=bins)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]

    ax.bar(range(len(hist)), hist, tick_label=bin_labels)
    ax.set_xlabel('min_dist 区间 (m)', fontsize=12)
    ax.set_ylabel('步数', fontsize=12)
    ax.set_title('安全边界分布 (风险暴露时间直方图)', fontsize=14)
    ax.axvline(1.5, color='green', linestyle='--', label='安全区(>0.5m)', alpha=0.7)
    ax.axvline(0.5, color='orange', linestyle='--', label='接近区(<0.5m)', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='危险区(<0.3m)', alpha=0.7)
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / 'risk_exposure_histogram.png', dpi=150)
    plt.close(fig)
    print(f"  → 图表: {output_dir / 'risk_exposure_histogram.png'}")

    if risk_ratio > 0.05:
        findings.append("⚠️  危险区时间占比>5%，避障策略过于激进或反应迟钝")

    return findings


def analyze_goal_oriented_behavior(df, output_dir):
    """分析目标导向行为：量化磨蹭问题"""
    print("\n[分析 6] 目标导向行为分析")

    findings = []

    # 按 episode 计算 dist_to_goal 的变化率
    df_sorted = df.sort_values(['episode_id', 'agent_id', 'step'])
    df_sorted['dist_to_goal_prev'] = df_sorted.groupby(['episode_id', 'agent_id'])['dist_to_goal'].shift(1)
    df_sorted['dist_progress'] = df_sorted['dist_to_goal_prev'] - df_sorted['dist_to_goal']  # 负值=远离

    # 过滤掉首步（没有 prev）
    df_progress = df_sorted[df_sorted['dist_to_goal_prev'].notna()].copy()

    # 统计
    approaching_ratio = (df_progress['dist_progress'] > 0.01).sum() / max(1, len(df_progress))
    stalling_ratio = (df_progress['dist_progress'].abs() < 0.01).sum() / max(1, len(df_progress))
    retreating_ratio = (df_progress['dist_progress'] < -0.01).sum() / max(1, len(df_progress))

    findings.append(f"靠近目标的时间占比: {approaching_ratio*100:.1f}%")
    findings.append(f"原地卡住(|Δd|<0.01m)时间占比: {stalling_ratio*100:.1f}%")
    findings.append(f"远离目标的时间占比: {retreating_ratio*100:.1f}%")

    # heading 对齐度（需要新采集的 heading_error）
    if 'heading_error' in df.columns:
        aligned_ratio = (df['heading_error'].abs() < np.deg2rad(30)).sum() / max(1, len(df))
        findings.append(f"朝向目标(|heading_err|<30°)时间占比: {aligned_ratio*100:.1f}%")

    # 画图：dist_progress 分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：dist_progress 直方图
    ax = axes[0]
    ax.hist(df_progress['dist_progress'].clip(-0.2, 0.2), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', label='无进展')
    ax.set_xlabel('Δdist_to_goal (m/step)', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('目标靠近速率分布', fontsize=14)
    ax.legend()

    # 子图2：速度 vs heading 对齐度
    ax = axes[1]
    if 'heading_error' in df.columns:
        sample = df.sample(min(3000, len(df)))
        scatter = ax.scatter(sample['heading_error'], sample['vel_x'],
                           c=sample['reward'], cmap='RdYlGn', alpha=0.5, s=10)
        plt.colorbar(scatter, ax=ax, label='Reward')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Heading Error (rad)', fontsize=12)
        ax.set_ylabel('Linear Velocity (m/s)', fontsize=12)
        ax.set_title('速度 vs 朝向对齐度', fontsize=14)

    plt.tight_layout()
    fig.savefig(output_dir / 'goal_oriented_behavior.png', dpi=150)
    plt.close(fig)
    print(f"  → 图表: {output_dir / 'goal_oriented_behavior.png'}")

    if stalling_ratio > 0.3:
        findings.append("⚠️  原地卡住时间>30%，存在严重磨蹭问题")
    if approaching_ratio < 0.4:
        findings.append("⚠️  靠近目标时间<40%，策略不够目标导向")

    return findings


def analyze_episode_outcomes(df, output_dir):
    """分析 episode 终止原因分布"""
    print("\n[分析 7] Episode 结果分析")

    findings = []

    # 按 episode 分组，取最后一步的 event
    episode_outcomes = df.groupby(['episode_id', 'agent_id']).last()['event']

    outcome_counts = episode_outcomes.value_counts()
    total_episodes = len(episode_outcomes)

    findings.append(f"总 episode 数: {total_episodes}")
    for event, count in outcome_counts.items():
        ratio = count / max(1, total_episodes)
        event_label = event if event else 'timeout/ongoing'
        findings.append(f"  {event_label}: {count} ({ratio*100:.1f}%)")

    # 画图：结果分布饼图
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = [e if e else 'timeout' for e in outcome_counts.index]
    ax.pie(outcome_counts.values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Episode 终止原因分布', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'episode_outcomes.png', dpi=150)
    plt.close(fig)
    print(f"  → 图表: {output_dir / 'episode_outcomes.png'}")

    # 成功率判断
    goal_count = outcome_counts.get('goal', 0)
    success_rate = goal_count / max(1, total_episodes)
    if success_rate < 0.3:
        findings.append(f"⚠️  到达率仅 {success_rate*100:.1f}%，远低于预期")

    return findings


def analyze_reward_breakdown_timeseries(df, output_dir):
    """分析奖励分解的时间序列：看哪些项在驱动磨蹭"""
    print("\n[分析 8] 奖励分解时间序列")

    findings = []

    # 检查是否有 reward_breakdown
    if 'reward_breakdown' not in df.columns or df['reward_breakdown'].isna().all():
        findings.append("⚠️  奖励分解数据缺失，跳过此分析")
        return findings

    # 展开 reward_breakdown
    breakdown_df = pd.json_normalize(df['reward_breakdown'])
    breakdown_df.index = df.index

    # 计算各项的平均贡献
    reward_components = ['r_progress', 'r_static', 'r_social', 'r_collision', 'r_goal', 'r_time']
    available_components = [c for c in reward_components if c in breakdown_df.columns]

    if not available_components:
        findings.append("⚠️  奖励分解字段缺失")
        return findings

    mean_contributions = breakdown_df[available_components].mean()
    findings.append("各项奖励的平均值:")
    for comp in available_components:
        findings.append(f"  {comp}: {mean_contributions[comp]:.4f}")

    # 画图：奖励分解饼图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 子图1：正负分开的柱状图
    ax = axes[0]
    ax.bar(range(len(mean_contributions)), mean_contributions.values)
    ax.set_xticks(range(len(mean_contributions)))
    ax.set_xticklabels(mean_contributions.index, rotation=45)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('平均贡献', fontsize=12)
    ax.set_title('奖励分解：各项平均贡献', fontsize=14)
    ax.grid(axis='y')

    # 子图2：时间序列（采样几个 episode）
    ax = axes[1]
    sample_episodes = df['episode_id'].unique()[:3]  # 前3个 episode
    for ep in sample_episodes:
        ep_df = df[df['episode_id'] == ep].copy()
        ep_breakdown = pd.json_normalize(ep_df['reward_breakdown'])
        if 'r_progress' in ep_breakdown.columns:
            ax.plot(ep_df['step'].values, ep_breakdown['r_progress'].values,
                   label=f'ep{ep}_r_progress', alpha=0.7)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('r_progress', fontsize=12)
    ax.set_title('r_progress 时间序列（采样）', fontsize=14)
    ax.legend(fontsize=8)
    ax.grid()

    plt.tight_layout()
    fig.savefig(output_dir / 'reward_breakdown_analysis.png', dpi=150)
    plt.close(fig)
    print(f"  → 图表: {output_dir / 'reward_breakdown_analysis.png'}")

    # 判断奖励失衡
    if mean_contributions.get('r_time', 0) > mean_contributions.get('r_progress', -999):
        findings.append("⚠️  r_time (惩罚) 绝对值小于 r_progress，不动比前进划算")

    return findings


def main():
    parser = argparse.ArgumentParser(description='可视化诊断数据')
    parser.add_argument('--input', type=str, required=True,
                        help='输入 JSONL 文件 (来自 collect_episode_diagnostics.py)')
    parser.add_argument('--output_dir', type=str, default='./diagnosis_output',
                        help='输出目录')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("避碰策略定量诊断")
    print("=" * 70)
    print(f"输入: {input_path}")
    print(f"输出: {output_dir}")

    # 加载数据
    print("\n[加载数据]")
    df = load_data(input_path)
    print(f"  总步数: {len(df)}")
    print(f"  Episodes: {df['episode_id'].nunique()}")
    print(f"  Agents: {df['agent_id'].nunique()}")

    # 各项分析
    all_findings = {}

    stats1, findings1 = analyze_reward_distance_correlation(df, output_dir)
    all_findings['1. 奖励-距离相关性'] = findings1

    findings2 = analyze_dynamic_detection(df, output_dir)
    all_findings['2. 动态障碍识别'] = findings2

    findings3 = analyze_collision_events(df, output_dir)
    all_findings['3. 碰撞分析'] = findings3

    findings4 = analyze_velocity_reward_correlation(df, output_dir)
    all_findings['4. 速度-奖励相关性'] = findings4

    # 新增诊断维度
    findings5 = analyze_risk_exposure(df, output_dir)
    all_findings['5. 风险暴露度量'] = findings5

    findings6 = analyze_goal_oriented_behavior(df, output_dir)
    all_findings['6. 目标导向行为'] = findings6

    findings7 = analyze_episode_outcomes(df, output_dir)
    all_findings['7. Episode 结果分析'] = findings7

    findings8 = analyze_reward_breakdown_timeseries(df, output_dir)
    all_findings['8. 奖励分解时间序列'] = findings8

    # 生成报告
    html_path = generate_html_report(all_findings, output_dir)

    print("\n" + "=" * 70)
    print("诊断完成!")
    print("=" * 70)
    print(f"\n查看报告: {html_path}")
    print(f"或直接打开浏览器: file://{html_path.absolute()}")
    print("\n关键发现:")
    for section, findings in all_findings.items():
        print(f"\n{section}:")
        for f in findings:
            print(f"  • {f}")


if __name__ == '__main__':
    main()
