#!/usr/bin/env python3
"""
诊断脚本: 在不改训练代码的前提下,单步采样一个 episode 并记录每步的:
  - 感知: min_dist, front_min, obstacle_motion_features (top-3 token)
  - 奖励分解: r_progress, r_static, r_social, r_dynamic_obs, r_collision, r_goal, r_time
  - 动作和碰撞事件

用法:
  python3 diagnose_reward_perception.py --checkpoint <path_to_checkpoint> --num_episodes 5

输出:
  - diagnosis_episodes.jsonl : 每 step 一行,可用 pandas 分析
  - diagnosis_summary.txt    : 汇总统计
"""

import os
import sys
import json
import argparse
import numpy as np
import math
from collections import defaultdict

# 添加当前目录到 path,以便 import gnn_marl_training
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def diagnose_episode(env, policy_fn=None, max_steps=500):
    """
    采样一个 episode,记录每步感知+奖励。

    Args:
        env: GNNMARLEnv 实例
        policy_fn: 如果提供,用策略采样动作;否则随机动作
        max_steps: 最大步数

    Returns:
        List[dict]: 每步的诊断数据
    """
    obs_dict, info_dict = env.reset()
    episode_data = []
    step = 0

    while step < max_steps:
        # 随机或策略动作
        if policy_fn is None:
            action_dict = {
                aid: env.action_space.sample()
                for aid in obs_dict.keys()
            }
        else:
            action_dict = policy_fn(obs_dict)

        # Step
        obs_dict, rew_dict, done_dict, truncated_dict, info_dict = env.step(action_dict)

        # 记录每个 agent 的详细数据
        for aid in list(rew_dict.keys()):
            agent_obj = env.agents.get(aid)
            if agent_obj is None:
                continue

            # 从 agent 读感知
            sectors = agent_obj._scan_sector_metrics()
            min_dist = float(sectors.get('min_dist', 10.0))
            front_min = float(sectors.get('front_min', 10.0))

            # obstacle_motion_features (top-3 token, 每个7维)
            motion_feat = getattr(agent_obj, '_last_motion_features', np.zeros(21))
            tokens = []
            for i in range(3):
                start = i * 7
                if start + 6 < len(motion_feat):
                    token = {
                        'x': float(motion_feat[start]),
                        'y': float(motion_feat[start+1]),
                        'vx': float(motion_feat[start+2]),
                        'vy': float(motion_feat[start+3]),
                        'future_x': float(motion_feat[start+4]),
                        'future_y': float(motion_feat[start+5]),
                        'is_dynamic': float(motion_feat[start+6]),
                    }
                    tokens.append(token)

            # 读奖励分解 (如果 env 有记录)
            # 由于当前代码没显式记录各项,我们用 info 里能拿到的
            info = info_dict.get(aid, {})

            record = {
                'step': step,
                'agent_id': aid,
                'reward': float(rew_dict.get(aid, 0.0)),
                'done': bool(done_dict.get(aid, False)),
                'event': info.get('event', ''),
                # 感知
                'min_dist': min_dist,
                'front_min': front_min,
                'obstacle_tokens': tokens,
                'num_dynamic_tokens': sum(1 for t in tokens if t['is_dynamic'] > 0.5),
                # 位置/速度
                'pos_x': float(agent_obj.current_pose.get('x', 0.0)),
                'pos_y': float(agent_obj.current_pose.get('y', 0.0)),
                'vel_x': float(getattr(agent_obj, 'current_vel_x', 0.0)),
                'vel_w': float(getattr(agent_obj, 'current_vel_w', 0.0)),
                # 如果有 shield info
                'shield_active': bool(getattr(agent_obj, '_last_shield_info', {}).get('shield_active', False)),
            }
            episode_data.append(record)

        step += 1
        if done_dict.get('__all__', False):
            break

    return episode_data


def analyze_diagnosis_data(all_episodes_data):
    """
    汇总分析多个 episode 的诊断数据。

    Returns:
        dict: 统计摘要
    """
    all_steps = []
    for ep_data in all_episodes_data:
        all_steps.extend(ep_data)

    if not all_steps:
        return {'error': 'No data collected'}

    # 提取各项指标
    rewards = [s['reward'] for s in all_steps]
    min_dists = [s['min_dist'] for s in all_steps]
    front_mins = [s['front_min'] for s in all_steps]
    num_dynamics = [s['num_dynamic_tokens'] for s in all_steps]

    collision_steps = [s for s in all_steps if s['event'] == 'collision']
    goal_steps = [s for s in all_steps if s['event'] == 'goal']

    # 分析 min_dist vs reward 分布
    bins = [0.0, 0.3, 0.5, 0.75, 1.0, 2.0, 10.0]
    dist_reward_bins = defaultdict(list)
    for s in all_steps:
        for i in range(len(bins)-1):
            if bins[i] <= s['min_dist'] < bins[i+1]:
                dist_reward_bins[f"{bins[i]:.1f}-{bins[i+1]:.1f}m"].append(s['reward'])
                break

    summary = {
        'total_steps': len(all_steps),
        'total_episodes': len(all_episodes_data),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'reward_min': float(np.min(rewards)),
        'reward_max': float(np.max(rewards)),
        'min_dist_mean': float(np.mean(min_dists)),
        'min_dist_std': float(np.std(min_dists)),
        'front_min_mean': float(np.mean(front_mins)),
        'num_dynamic_tokens_mean': float(np.mean(num_dynamics)),
        'collision_count': len(collision_steps),
        'collision_rate': len(collision_steps) / len(all_steps),
        'goal_count': len(goal_steps),
        'goal_rate': len(goal_steps) / len(all_steps),
        'shield_active_rate': sum(1 for s in all_steps if s['shield_active']) / len(all_steps),
        'dist_reward_correlation': {
            k: {
                'mean': float(np.mean(v)) if v else 0.0,
                'std': float(np.std(v)) if v else 0.0,
                'count': len(v)
            }
            for k, v in dist_reward_bins.items()
        }
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description='诊断避碰奖励-感知失配')
    parser.add_argument('--num_episodes', type=int, default=5,
                        help='采样 episode 数量')
    parser.add_argument('--max_steps', type=int, default=500,
                        help='单 episode 最大步数')
    parser.add_argument('--env_stage', type=int, default=2,
                        help='环境 stage (1-6)')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='机器人数量')
    parser.add_argument('--action_mode', type=str, default='continuous',
                        choices=['continuous', 'discrete_primitive'],
                        help='动作模式')
    parser.add_argument('--output_dir', type=str, default='./diagnosis_output',
                        help='输出目录')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 构建环境 (不启动 Gazebo,用本地模拟或跳过真实环境依赖)
    print(f"[诊断] 环境配置: stage={args.env_stage}, num_agents={args.num_agents}, action_mode={args.action_mode}")
    print("[诊断] 注意: 此脚本需要 Gazebo 已运行. 如果 Gazebo 未启动,会报错.")
    print("[诊断] 建议先手动启动训练环境,或修改脚本跳过 ROS2 依赖.\n")

    try:
        from gnn_marl_training.gnn_marl_env import GNNMARLEnv
        from gnn_marl_training.train_gnn_mappo_full import ENV_CURRICULUM

        stage_cfg = ENV_CURRICULUM[args.env_stage]

        # 简化配置,只用于诊断
        env_config = {
            "num_agents": args.num_agents,
            "map_number": stage_cfg['map_number'],
            "max_episode_steps": args.max_steps,
            "action_mode": args.action_mode,
            "enable_local_map": (args.action_mode == 'continuous'),
            "progress_scale": 1.5,
            "goal_reach_radius": 0.45,
            "collision_grace_steps": 8,
            "auto_reset_agents": True,
            "goal_reward": 60.0,
            "collision_penalty": 20.0,
            "time_penalty": 0.008,
            # 其他参数用默认
        }

        env = GNNMARLEnv(env_config)

    except Exception as e:
        print(f"[错误] 无法初始化环境: {e}")
        print("\n[建议] 此脚本需要完整的训练环境(Gazebo + ROS2).")
        print("如果只想分析已有数据,请跳到方案 B (分析现有 checkpoint 的 episode replay).\n")
        return

    # 采样 episodes
    print(f"[诊断] 开始采样 {args.num_episodes} 个 episodes (随机动作)...\n")
    all_episodes_data = []
    for ep_idx in range(args.num_episodes):
        print(f"  Episode {ep_idx+1}/{args.num_episodes}...", end='', flush=True)
        try:
            ep_data = diagnose_episode(env, policy_fn=None, max_steps=args.max_steps)
            all_episodes_data.append(ep_data)
            print(f" 完成 ({len(ep_data)} steps)")
        except KeyboardInterrupt:
            print("\n[中断] 用户取消")
            break
        except Exception as e:
            print(f" 失败: {e}")
            continue

    if not all_episodes_data:
        print("[错误] 没有采集到任何数据")
        return

    # 保存详细数据
    jsonl_path = os.path.join(args.output_dir, 'diagnosis_episodes.jsonl')
    with open(jsonl_path, 'w') as f:
        for ep_data in all_episodes_data:
            for record in ep_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"\n[输出] 详细数据已保存到: {jsonl_path}")

    # 分析汇总
    summary = analyze_diagnosis_data(all_episodes_data)
    summary_path = os.path.join(args.output_dir, 'diagnosis_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, indent=2, fp=f)
    print(f"[输出] 汇总统计已保存到: {summary_path}")

    # 打印关键发现
    print("\n" + "="*60)
    print("关键诊断结果:")
    print("="*60)
    print(f"总步数:         {summary['total_steps']}")
    print(f"总 episodes:    {summary['total_episodes']}")
    print(f"奖励均值:       {summary['reward_mean']:.4f} ± {summary['reward_std']:.4f}")
    print(f"奖励范围:       [{summary['reward_min']:.4f}, {summary['reward_max']:.4f}]")
    print(f"碰撞率:         {summary['collision_rate']:.2%} ({summary['collision_count']} 次)")
    print(f"到达率:         {summary['goal_rate']:.2%} ({summary['goal_count']} 次)")
    print(f"min_dist 均值:  {summary['min_dist_mean']:.3f}m ± {summary['min_dist_std']:.3f}")
    print(f"动态 token 均值: {summary['num_dynamic_tokens_mean']:.2f} / 3")
    print(f"Shield 激活率:  {summary['shield_active_rate']:.2%}")
    print("\n距离-奖励相关性 (按距离分段的奖励均值):")
    for dist_bin, stats in sorted(summary['dist_reward_correlation'].items()):
        if stats['count'] > 0:
            print(f"  {dist_bin:12s}: {stats['mean']:7.4f} ± {stats['std']:.4f}  (n={stats['count']})")

    print("\n[诊断] 完成. 请检查上述数据,重点关注:")
    print("  1. 奖励均值是否为负 (如果是,说明避碰惩罚压过了前进奖励)")
    print("  2. 距离 0.3-0.75m 时奖励是否已经很负 (如果是,说明 static 阈值太保守)")
    print("  3. 动态 token 均值是否 < 1.0 (如果是,说明动态障碍物常被识别为静态)")
    print("  4. 碰撞率 vs Shield 激活率 (如果 Shield 高但碰撞仍高,说明感知延迟)")


if __name__ == '__main__':
    main()
