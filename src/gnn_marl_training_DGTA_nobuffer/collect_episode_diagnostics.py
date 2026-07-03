#!/usr/bin/env python3
"""
收集 episode 级别的诊断数据: 在现有训练/测试脚本的基础上,
hook 每一步的感知+奖励分解,输出到 JSONL 用于离线分析。

用法 1: 从 checkpoint 跑 rollout
  python3 collect_episode_diagnostics.py \
      --checkpoint /path/to/checkpoint \
      --num_episodes 20 \
      --env_stage 2 \
      --num_agents 4 \
      --output diagnosis_data.jsonl

用法 2: 边训练边采集 (修改 train_gnn_mappo_full.py,在 callbacks 里调用)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import math
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def extract_reward_breakdown_from_agent(agent):
    """
    从 IndependentRobotEnv 实例读取奖励各项的值。

    注意: 当前代码没有显式记录各项奖励,需要在 get_step_result 里加 hook。
    这里提供一个"最小侵入"的方案: 读已有的 info/metrics。
    """
    # 如果 agent 有 _last_reward_breakdown (需在 env 里加),直接读
    if hasattr(agent, '_last_reward_breakdown'):
        return agent._last_reward_breakdown.copy()

    # 否则返回空,后续从 info 推断
    return {}


def collect_one_episode(env, policy_fn, max_steps=1500, out_f=None, episode_id=0,
                        progress_every=10):
    """
    采集一个 episode 的逐步数据。

    改动 (2026-07-01):
      - 逐步写盘 (out_f 传入文件句柄): 每步立即 flush,Ctrl+C 也不丢数据
      - 逐步打印进度: 每 progress_every 步打印一行,能看到卡在哪
      - reset 前后打印,定位是否卡在 reset

    Returns:
        int: 实际采集的 step 数
    """
    print(f"    [ep{episode_id}] reset 中 (等 Gazebo spawn + 首帧激光)...", flush=True)
    t0 = time.time()
    obs_dict, info_dict = env.reset()
    print(f"    [ep{episode_id}] reset 完成 (耗时 {time.time()-t0:.1f}s), 开始 step", flush=True)

    step = 0
    n_written = 0

    while step < max_steps:
        # 策略采样动作
        action_dict = policy_fn(obs_dict) if policy_fn else {
            aid: env.action_space.sample() for aid in obs_dict.keys()
        }

        # Step
        t_step = time.time()
        obs_dict, rew_dict, done_dict, truncated_dict, info_dict = env.step(action_dict)
        step_dur = time.time() - t_step

        # 记录每个 agent 的数据
        for aid in list(rew_dict.keys()):
            agent_obj = env.agents.get(aid)
            if agent_obj is None:
                continue

            # 感知
            sectors = agent_obj._scan_sector_metrics()
            min_dist = float(sectors.get('min_dist', 10.0))
            front_min = float(sectors.get('front_min', 10.0))

            # obstacle_motion_features (top-3)
            motion_feat = getattr(agent_obj, '_last_motion_features', np.zeros(21, dtype=np.float32))
            num_dynamic = 0
            token_distances = []
            for i in range(3):
                start = i * 7
                if start + 6 < len(motion_feat):
                    is_dyn = float(motion_feat[start + 6])
                    if is_dyn > 0.5:
                        num_dynamic += 1
                        x = float(motion_feat[start]) * 5.0
                        y = float(motion_feat[start + 1]) * 5.0
                        token_distances.append(math.hypot(x, y))

            # 位置/速度
            pos_x = float(agent_obj.current_pose.get('x', 0.0))
            pos_y = float(agent_obj.current_pose.get('y', 0.0))
            vel_x = float(getattr(agent_obj, 'current_vel_x', 0.0))
            vel_w = float(getattr(agent_obj, 'current_vel_w', 0.0))

            # 目标距离和朝向
            goal_x, goal_y = agent_obj.goal_pos
            dist_to_goal = math.hypot(goal_x - pos_x, goal_y - pos_y)

            # 目标朝向偏差（从 agent 的 tracking target 计算）
            current_target = agent_obj._get_tracking_target()
            heading_error = float(agent_obj._get_target_angle(current_target))

            # 多智能体交互：与最近其他 agent 的距离
            min_agent_distance = 999.0
            if hasattr(env, 'robot_positions'):
                other_distances = [
                    math.hypot(p[0] - pos_x, p[1] - pos_y)
                    for aid_other, p in env.robot_positions.items()
                    if aid_other != aid
                ]
                if other_distances:
                    min_agent_distance = min(other_distances)

            # 奖励分解（直接从 info_dict 读取，环境已输出）
            agent_info = info_dict.get(aid, {})
            reward_breakdown = {
                'r_progress': float(agent_info.get('r_progress', 0.0)),
                'r_static': float(agent_info.get('r_static', 0.0)),
                'r_social': float(agent_info.get('r_social', 0.0)),
                'r_collision': float(agent_info.get('r_collision', 0.0)),
                'r_goal': float(agent_info.get('r_goal', 0.0)),
                'r_time': float(agent_info.get('r_time', 0.0)),
            }

            # Shield info
            shield_info = getattr(agent_obj, '_last_shield_info', {})

            record = {
                'step': step,
                'agent_id': aid,
                'reward': float(rew_dict.get(aid, 0.0)),
                'done': bool(done_dict.get(aid, False)),
                'event': info_dict.get(aid, {}).get('event', ''),

                # 感知
                'min_dist': min_dist,
                'front_min': front_min,
                'num_dynamic_tokens': num_dynamic,
                'token_distances': token_distances,

                # 位置/速度/目标
                'pos_x': pos_x,
                'pos_y': pos_y,
                'vel_x': vel_x,
                'vel_w': vel_w,
                'dist_to_goal': dist_to_goal,
                'heading_error': heading_error,  # 新增

                # 多智能体交互
                'min_agent_distance': min_agent_distance,  # 新增

                # Shield
                'shield_active': bool(shield_info.get('shield_active', False)),
                'raw_linear_vel': float(shield_info.get('raw_linear_vel', vel_x)),
                'shielded_linear_vel': float(shield_info.get('shielded_linear_vel', vel_x)),

                # 奖励分解
                'reward_breakdown': reward_breakdown,
            }

            # 逐步写盘 + flush: Ctrl+C 也不丢已采集数据
            if out_f is not None:
                record['episode_id'] = episode_id
                out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                n_written += 1
        if out_f is not None:
            out_f.flush()

        # 逐步进度: 看是否推进 + 每步耗时(判断是否卡在 Gazebo 同步)
        if step % progress_every == 0:
            n_active = len(rew_dict)
            any_event = [info_dict.get(a, {}).get('event', '') for a in rew_dict]
            events = [e for e in any_event if e]
            print(f"    [ep{episode_id}] step={step:4d}  agents={n_active}  "
                  f"step_dur={step_dur*1000:.0f}ms  events={events}", flush=True)

        step += 1
        if done_dict.get('__all__', False):
            reason = info_dict.get(list(rew_dict.keys())[0], {}).get('episode_end_reason', '?') if rew_dict else '?'
            print(f"    [ep{episode_id}] episode 结束于 step={step} (reason={reason})", flush=True)
            break

    return n_written


def load_policy_from_checkpoint(checkpoint_path):
    """
    从 RLlib checkpoint 加载策略用于 rollout。

    这需要初始化 RLlib Algorithm,比较重。简化版:返回 None → 用随机策略。
    """
    # TODO: 如果需要真实策略,参考 test_gnn_mappo.py 的加载逻辑
    return None


def main():
    parser = argparse.ArgumentParser(description='收集 episode 诊断数据')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='RLlib checkpoint 路径 (可选,不提供则随机策略)')
    parser.add_argument('--num_episodes', type=int, default=20,
                        help='采集 episode 数')
    parser.add_argument('--env_stage', type=int, default=2,
                        help='环境 stage')
    parser.add_argument('--num_agents', type=int, default=4,
                        help='机器人数')
    parser.add_argument('--max_steps', type=int, default=1500,
                        help='单 episode 最大步数')
    parser.add_argument('--output', type=str, required=True,
                        help='输出 JSONL 文件路径')
    parser.add_argument('--action_mode', type=str, default='continuous',
                        choices=['continuous', 'discrete_primitive'])
    args = parser.parse_args()

    print(f"[采集] 配置:")
    print(f"  env_stage={args.env_stage}, num_agents={args.num_agents}")
    print(f"  num_episodes={args.num_episodes}, max_steps={args.max_steps}")
    print(f"  checkpoint={args.checkpoint or '(random policy)'}")
    print(f"  output={args.output}")
    print()

    # 加载策略 (如果提供)
    policy_fn = None
    if args.checkpoint:
        print("[采集] 加载 checkpoint...")
        policy_fn = load_policy_from_checkpoint(args.checkpoint)
        if policy_fn is None:
            print("[警告] Checkpoint 加载失败或未实现,使用随机策略")

    # 初始化环境
    try:
        from gnn_marl_training.gnn_marl_env import GNNMARLEnv
        from gnn_marl_training.train_gnn_mappo_full import ENV_CURRICULUM

        stage_cfg = ENV_CURRICULUM[args.env_stage]

        env_config = {
            "num_agents": args.num_agents,
            "map_number": stage_cfg['map_number'],
            "max_episode_steps": args.max_steps,
            "action_mode": args.action_mode,
            "enable_local_map": (args.action_mode == 'continuous'),
            "progress_scale": 1.5,
            "goal_reach_radius": 0.45,
            "collision_grace_steps": 8,
            # 诊断用: auto_reset=False,让 episode 在碰撞/到达后早结束,
            # 快速拿到多个短 episode 的数据,而不是傻等一个 1500 步的长局。
            "auto_reset_agents": False,
            "goal_reward": 60.0,
            "collision_penalty": 20.0,
            "time_penalty": 0.008,
        }

        print("[采集] 初始化环境...")
        env = GNNMARLEnv(env_config)

    except Exception as e:
        print(f"[错误] 环境初始化失败: {e}")
        print("\n需要 Gazebo 运行中. 如果是 headless,确保已启动 spawn_robots.launch.py")
        return

    # 采集 episodes
    print(f"\n[采集] 开始采集 {args.num_episodes} episodes...\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        total_written = 0
        for ep_idx in range(args.num_episodes):
            print(f"  Episode {ep_idx+1}/{args.num_episodes}:", flush=True)
            try:
                n = collect_one_episode(env, policy_fn, args.max_steps,
                                        out_f=f, episode_id=ep_idx)
                total_written += n
                print(f"  → Episode {ep_idx+1} 采集 {n} 条记录 (累计 {total_written})\n", flush=True)
            except KeyboardInterrupt:
                print("\n[中断] 用户取消 (已采集数据已写盘,不丢失)")
                break
            except Exception as e:
                import traceback
                print(f"  → Episode {ep_idx+1} 失败: {e}")
                traceback.print_exc()
                continue

    print(f"\n[完成] 数据已保存到: {output_path}")
    print(f"  总记录数: {sum(1 for _ in open(output_path))}")
    print(f"\n下一步: python3 visualize_diagnostics.py --input {output_path}")


if __name__ == '__main__':
    main()
