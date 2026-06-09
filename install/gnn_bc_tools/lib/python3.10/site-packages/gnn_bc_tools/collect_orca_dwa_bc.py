#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gnn_bc_tools.path_utils import ensure_runtime_modules, inject_workspace_paths


ENV_CURRICULUM = {
    1: {"name": "Stage 1", "map_number": 3, "max_episode_steps": 2000, "num_obstacles": 0, "obs_speed_scale": 0.0},
    2: {"name": "Stage 2", "map_number": 3, "max_episode_steps": 2000, "num_obstacles": 0, "obs_speed_scale": 0.0},
    3: {"name": "Stage 3", "map_number": 3, "max_episode_steps": 2500, "num_obstacles": 4, "obs_speed_scale": 0.5},
    4: {"name": "Stage 4", "map_number": 3, "max_episode_steps": 3000, "num_obstacles": 8, "obs_speed_scale": 1.0},
}


def build_env_config(args) -> Dict:
    stage_cfg = dict(ENV_CURRICULUM[args.env_stage])
    if args.map_number is not None:
        stage_cfg["map_number"] = int(args.map_number)

    return {
        "num_agents": args.num_agents,
        "map_number": stage_cfg["map_number"],
        "max_episode_steps": int(args.max_episode_steps or stage_cfg["max_episode_steps"]),
        "communication_range": float(args.communication_range),
        "enable_neighbor_obs": True,
        "enable_local_map": False,
        "comm_mode": args.comm_mode,
        "comm_dropout_prob": float(args.comm_dropout_prob),
        "comm_latency_steps": int(args.comm_latency_steps),
        "comm_jitter_steps": int(args.comm_jitter_steps),
        "comm_noise_std": float(args.comm_noise_std),
        "reset_on_collision_event": True,
        "collision_hard_dist": 0.20,
        "collision_persist_dist": 0.26,
        "collision_persist_steps": 2,
        "near_wall_penalty_dist": 0.30,
        "rolling_lookahead_dist": 0.8,
        "progress_reward_scale": 6.0,
        "path_progress_reward_scale": 3.0,
        "goal_progress_reward_scale": 1.5,
        "goal_reward": 40.0,
        "collision_penalty": 35.0,
        "time_penalty": 0.002,
        "lateral_penalty_scale": 0.05,
        "heading_align_reward_scale": 0.15,
        "narrow_forward_penalty_scale": 0.35,
        "near_collision_dist": 0.45,
        "near_collision_penalty_scale": 1.2,
        "front_safety_dist": 0.55,
        "front_safety_penalty_scale": 0.8,
        "neighbor_safety_dist": 0.45,
        "neighbor_safety_penalty_scale": 1.0,
        "shield_enable": True,
        "shield_front_slow_dist": 0.60,
        "shield_front_stop_dist": 0.26,
        "shield_neighbor_slow_dist": 0.45,
        "shield_linear_slow": 0.15,
        "shield_linear_stop": 0.05,
        "shield_turn_bias": 0.35,
        "turn_in_place_front_dist": 0.40,
        "enable_visualization": bool(args.enable_visualization),
        "tracking_viz_interval": int(args.tracking_viz_interval),
        "env_log_level": str(args.env_log_level),
        "sim_wait_wall_timeout": float(args.sim_wait_wall_timeout),
        "auto_reset_agents": bool(args.auto_reset_agents),
        "num_dynamic_obstacles": int(args.num_dynamic_obstacles) if args.num_dynamic_obstacles is not None else int(stage_cfg["num_obstacles"]),
        "obs_speed": 0.3 * float(args.obs_speed_scale) if args.obs_speed_scale is not None else 0.3 * float(stage_cfg["obs_speed_scale"]),
    }


def _append_agent_sequence(
    seq_obs: List[np.ndarray],
    seq_actions: List[np.ndarray],
    seq_agent_ids: List[int],
    per_agent_buf: Dict[str, Dict[str, List[np.ndarray]]],
) -> int:
    added = 0
    for aid, data in per_agent_buf.items():
        if len(data["obs"]) < 2:
            continue
        obs_arr = np.asarray(data["obs"], dtype=np.float32)
        act_arr = np.asarray(data["actions"], dtype=np.float32)
        if obs_arr.shape[0] != act_arr.shape[0]:
            continue
        seq_obs.append(obs_arr)
        seq_actions.append(act_arr)
        seq_agent_ids.append(int(aid.split("_")[1]))
        added += 1
    return added


def collect_dataset(args) -> Path:
    repo_root, _, _ = inject_workspace_paths()

    from gnn_marl_training.gnn_marl_env import env_creator
    from gnn_bc_tools.expert_apf import APFTeacher
    from gnn_bc_tools.expert_orca_dwa import ORCADWATeacher

    env_cfg = build_env_config(args)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_tag = args.dataset_name or f"orca_dwa_bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    npz_path = out_dir / f"{run_tag}.npz"
    meta_path = out_dir / f"{run_tag}.json"
    target_kept_episodes = int(args.target_kept_episodes) if args.target_kept_episodes is not None else int(args.episodes)
    if target_kept_episodes <= 0:
        raise ValueError("target_kept_episodes must be > 0")
    default_max_attempts = target_kept_episodes
    max_attempt_episodes = int(args.max_attempt_episodes) if args.max_attempt_episodes is not None else default_max_attempts
    max_attempt_episodes = max(max_attempt_episodes, target_kept_episodes)

    print("=" * 80)
    print("ORCA/DWA BC 数据采集")
    print(f"repo:        {repo_root}")
    print(f"output npz:  {npz_path}")
    print(
        f"target kept: {target_kept_episodes} "
        f"(max attempts: {max_attempt_episodes})"
    )
    print(f"env config:  map={env_cfg['map_number']} agents={env_cfg['num_agents']} max_steps={env_cfg['max_episode_steps']}")
    print("=" * 80)

    env = env_creator(env_cfg)
    teacher_common_kwargs = dict(
        communication_range=float(env_cfg["communication_range"]),
        max_linear_speed=0.22,
        max_angular_speed=1.2,
        robot_radius=float(args.robot_radius),
        time_horizon=float(args.time_horizon),
        laser_obstacle_max_dist=float(args.laser_obstacle_max_dist),
        velocity_smoothing_alpha=float(args.velocity_smoothing_alpha),
        neighbor_soft_dist=float(args.neighbor_soft_dist),
        neighbor_stop_dist=float(args.neighbor_stop_dist),
        neighbor_hard_stop_dist=float(args.neighbor_hard_stop_dist),
        orca_blend_max=float(args.orca_blend_max),
        dwa_heading_weight=float(args.dwa_heading_weight),
        dwa_dist_weight=float(args.dwa_dist_weight),
        dwa_velocity_weight=float(args.dwa_velocity_weight),
        dwa_safety_margin=float(args.dwa_safety_margin),
        intent_horizon_sec=float(args.intent_horizon_sec),
        intent_dt_sec=float(args.intent_dt_sec),
        intent_safe_margin=float(args.intent_safe_margin),
        intent_commit_steps=int(args.intent_commit_steps),
        intent_replan_interval_steps=int(args.intent_replan_interval_steps),
        intent_dropout_prob=float(args.comm_dropout_prob),
        intent_latency_steps=int(args.comm_latency_steps),
        intent_jitter_steps=int(args.comm_jitter_steps),
        intent_max_staleness_steps=int(args.intent_max_staleness_steps),
        intent_seed=args.seed,
    )

    if str(args.teacher).lower() == "apf":
        teacher = APFTeacher(
            **teacher_common_kwargs,
            apf_attract_gain=float(args.apf_attract_gain),
            apf_obstacle_gain=float(args.apf_obstacle_gain),
            apf_robot_gain=float(args.apf_robot_gain),
            apf_tangent_gain=float(args.apf_tangent_gain),
            apf_damping_gain=float(args.apf_damping_gain),
            apf_influence_radius=float(args.apf_influence_radius),
            apf_robot_influence_radius=float(args.apf_robot_influence_radius),
            apf_goal_slow_radius=float(args.apf_goal_slow_radius),
            apf_obstacle_top_k=int(args.apf_obstacle_top_k),
        )
    else:
        teacher = ORCADWATeacher(**teacher_common_kwargs)

    all_seq_obs: List[np.ndarray] = []
    all_seq_actions: List[np.ndarray] = []
    all_seq_agent_ids: List[int] = []

    total_steps = 0
    episode_lengths: List[int] = []
    kept_episodes = 0
    dropped_episodes = 0
    dropped_by_collision = 0
    dropped_by_timeout = 0
    dropped_by_other = 0
    attempted_episodes = 0
    start_wall = time.time()

    try:
        for ep in range(max_attempt_episodes):
            if kept_episodes >= target_kept_episodes:
                break

            attempted_episodes += 1
            obs_dict, _ = env.reset(seed=(None if args.seed is None else args.seed + ep))
            teacher.reset()

            per_agent_buf = {
                aid: {"obs": [], "actions": []}
                for aid in env.agent_ids
            }

            ep_steps = 0
            ep_goal_agents = set()
            ep_collision_agents = set()
            ep_timeout = False
            stop_reason = "unknown"
            while True:
                action_dict = teacher.compute_actions(env, obs_dict)

                for aid, action in action_dict.items():
                    if aid not in obs_dict:
                        continue
                    per_agent_buf[aid]["obs"].append(np.asarray(obs_dict[aid], dtype=np.float32))
                    per_agent_buf[aid]["actions"].append(np.asarray(action, dtype=np.float32))

                obs_dict, _, done_dict, trunc_dict, info_dict = env.step(action_dict)
                ep_steps += 1

                for aid in env.agent_ids:
                    event = str(info_dict.get(aid, {}).get("event", ""))
                    if event == "goal":
                        ep_goal_agents.add(aid)
                    elif event == "collision":
                        ep_collision_agents.add(aid)

                if bool(args.early_stop_on_collision) and ep_collision_agents:
                    stop_reason = "collision_early_stop"
                    break
                if done_dict.get("__all__", False) or trunc_dict.get("__all__", False):
                    ep_timeout = bool(trunc_dict.get("__all__", False))
                    stop_reason = "timeout" if ep_timeout else "all_done"
                    break
                if ep_steps >= int(env_cfg["max_episode_steps"]):
                    ep_timeout = True
                    stop_reason = "max_steps_guard"
                    break

            ep_success_count = len(ep_goal_agents)
            ep_collision_count = len(ep_collision_agents)
            if done_dict.get("__all__", False) or trunc_dict.get("__all__", False):
                ep_success_count = max(
                    ep_success_count,
                    sum(int(info_dict[aid].get("episode_successes", 0)) for aid in env.agent_ids),
                )
                ep_collision_count = max(
                    ep_collision_count,
                    sum(int(info_dict[aid].get("episode_collisions", 0)) for aid in env.agent_ids),
                )

            ep_success = (
                (not ep_timeout)
                and ep_collision_count == 0
                and ep_success_count >= int(env_cfg["num_agents"])
            )
            keep_episode = (not bool(args.discard_failure_episodes)) or ep_success

            if keep_episode:
                added = _append_agent_sequence(
                    seq_obs=all_seq_obs,
                    seq_actions=all_seq_actions,
                    seq_agent_ids=all_seq_agent_ids,
                    per_agent_buf=per_agent_buf,
                )
                kept_episodes += 1
            else:
                added = 0
                dropped_episodes += 1
                if ep_collision_count > 0:
                    dropped_by_collision += 1
                elif ep_timeout:
                    dropped_by_timeout += 1
                else:
                    dropped_by_other += 1

            episode_lengths.append(ep_steps)
            total_steps += ep_steps

            print(
                f"[collect] try={ep + 1:04d}/{max_attempt_episodes} "
                f"kept={kept_episodes:04d}/{target_kept_episodes} "
                f"steps={ep_steps:4d} reason={stop_reason:>20s} "
                f"succ={ep_success_count}/{env_cfg['num_agents']} coll={ep_collision_count} "
                f"{'keep' if keep_episode else 'drop'} "
                f"added_seq={added:2d} total_seq={len(all_seq_obs):5d}"
            )

    finally:
        env.close()

    if not all_seq_obs:
        raise RuntimeError(
            "没有采集到有效序列（可能失败 episode 被全部丢弃）。"
            f" kept={kept_episodes} dropped={dropped_episodes} attempts={attempted_episodes}/{max_attempt_episodes}"
        )
    if kept_episodes < target_kept_episodes:
        print(
            f"⚠️ 仅收集到 {kept_episodes}/{target_kept_episodes} 个有效 episode "
            f"(attempts={attempted_episodes}/{max_attempt_episodes})"
        )

    seq_lens = np.asarray([arr.shape[0] for arr in all_seq_obs], dtype=np.int32)
    obs_flat = np.concatenate(all_seq_obs, axis=0).astype(np.float32)
    actions_flat = np.concatenate(all_seq_actions, axis=0).astype(np.float32)
    seq_agent_ids = np.asarray(all_seq_agent_ids, dtype=np.int32)

    np.savez_compressed(
        npz_path,
        obs=obs_flat,
        actions=actions_flat,
        seq_lens=seq_lens,
        seq_agent_ids=seq_agent_ids,
        obs_dim=np.asarray([obs_flat.shape[1]], dtype=np.int32),
        action_dim=np.asarray([actions_flat.shape[1]], dtype=np.int32),
        num_agents=np.asarray([env_cfg["num_agents"]], dtype=np.int32),
        base_obs_dim=np.asarray([env.base_obs_dim], dtype=np.int32),
        neighbor_dim=np.asarray([env.neighbor_dim], dtype=np.int32),
        reset_flag_dim=np.asarray([env.reset_flag_dim], dtype=np.int32),
        global_state_dim=np.asarray([env.global_state_dim], dtype=np.int32),
        map_number=np.asarray([env_cfg["map_number"]], dtype=np.int32),
        communication_range=np.asarray([env_cfg["communication_range"]], dtype=np.float32),
    )

    elapsed = time.time() - start_wall
    metadata = {
        "dataset": str(npz_path),
        "episodes": int(args.episodes),
        "target_kept_episodes": int(target_kept_episodes),
        "max_attempt_episodes": int(max_attempt_episodes),
        "attempted_episodes": int(attempted_episodes),
        "total_env_steps": int(total_steps),
        "kept_episodes": int(kept_episodes),
        "dropped_episodes": int(dropped_episodes),
        "dropped_by_collision": int(dropped_by_collision),
        "dropped_by_timeout": int(dropped_by_timeout),
        "dropped_by_other": int(dropped_by_other),
        "num_sequences": int(len(seq_lens)),
        "mean_seq_len": float(np.mean(seq_lens)),
        "max_seq_len": int(np.max(seq_lens)),
        "min_seq_len": int(np.min(seq_lens)),
        "episode_mean_steps": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "obs_dim": int(obs_flat.shape[1]),
        "action_dim": int(actions_flat.shape[1]),
        "num_agents": int(env_cfg["num_agents"]),
        "map_number": int(env_cfg["map_number"]),
        "env_config": env_cfg,
        "collector_args": vars(args),
        "elapsed_sec": float(elapsed),
        "created_at": datetime.now().isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 80)
    print("采集完成")
    print(f"dataset:     {npz_path}")
    print(f"metadata:    {meta_path}")
    print(f"attempted:   {attempted_episodes}/{max_attempt_episodes}")
    print(f"kept/drop:   {kept_episodes}/{dropped_episodes}")
    print(f"sequences:   {len(seq_lens)}")
    print(f"mean_len:    {np.mean(seq_lens):.2f}")
    print(f"elapsed:     {elapsed:.1f}s")
    print("=" * 80)

    return npz_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect BC dataset with ORCA/DWA or APF teacher.")

    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--target_kept_episodes", type=int, default=None)
    p.add_argument("--max_attempt_episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="~/work/multi-robot-exploration-rl/bc_datasets")

    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--env_stage", type=int, default=1, choices=sorted(ENV_CURRICULUM.keys()))
    p.add_argument("--map_number", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument("--max_episode_steps", type=int, default=None)

    p.add_argument("--communication_range", type=float, default=3.5)
    p.add_argument("--comm_mode", type=str, default="decentralized", choices=["decentralized", "centralized_oracle", "ros2_bridge"])
    p.add_argument("--comm_dropout_prob", type=float, default=0.05)
    p.add_argument("--comm_latency_steps", type=int, default=1)
    p.add_argument("--comm_jitter_steps", type=int, default=1)
    p.add_argument("--comm_noise_std", type=float, default=0.05)

    p.add_argument("--num_dynamic_obstacles", type=int, default=None)
    p.add_argument("--obs_speed_scale", type=float, default=None)

    p.add_argument("--enable_visualization", action="store_true")
    p.add_argument("--disable_visualization", action="store_true")
    p.add_argument("--tracking_viz_interval", type=int, default=6)
    p.add_argument("--env_log_level", type=str, default="WARNING")
    p.add_argument("--sim_wait_wall_timeout", type=float, default=2.5)
    p.add_argument("--auto_reset_agents", action="store_true", default=False)
    p.add_argument("--early_stop_on_collision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--discard_failure_episodes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--intent_horizon_sec", type=float, default=1.8)
    p.add_argument("--intent_dt_sec", type=float, default=0.2)
    p.add_argument("--intent_safe_margin", type=float, default=0.12)
    p.add_argument("--intent_commit_steps", type=int, default=4)
    p.add_argument("--intent_replan_interval_steps", type=int, default=2)
    p.add_argument("--intent_max_staleness_steps", type=int, default=20)

    p.add_argument("--robot_radius", type=float, default=0.25)
    p.add_argument("--time_horizon", type=float, default=2.0)
    p.add_argument("--laser_obstacle_max_dist", type=float, default=2.0)
    p.add_argument("--velocity_smoothing_alpha", type=float, default=0.6)
    p.add_argument("--neighbor_soft_dist", type=float, default=0.72)
    p.add_argument("--neighbor_stop_dist", type=float, default=0.36)
    p.add_argument("--neighbor_hard_stop_dist", type=float, default=0.27)
    p.add_argument("--teacher", type=str, default="apf", choices=["apf", "orca_dwa"])
    p.add_argument("--orca_blend_max", type=float, default=0.78)
    p.add_argument("--dwa_heading_weight", type=float, default=2.0)
    p.add_argument("--dwa_dist_weight", type=float, default=2.8)
    p.add_argument("--dwa_velocity_weight", type=float, default=1.5)
    p.add_argument("--dwa_safety_margin", type=float, default=0.14)
    p.add_argument("--apf_attract_gain", type=float, default=0.85)
    p.add_argument("--apf_obstacle_gain", type=float, default=0.22)
    p.add_argument("--apf_robot_gain", type=float, default=0.42)
    p.add_argument("--apf_tangent_gain", type=float, default=0.18)
    p.add_argument("--apf_damping_gain", type=float, default=0.16)
    p.add_argument("--apf_influence_radius", type=float, default=1.15)
    p.add_argument("--apf_robot_influence_radius", type=float, default=1.45)
    p.add_argument("--apf_goal_slow_radius", type=float, default=0.70)
    p.add_argument("--apf_obstacle_top_k", type=int, default=28)

    return p


def main() -> None:
    ensure_runtime_modules(
        required_modules=["numpy", "gymnasium"],
        runner_module="gnn_bc_tools.collect_orca_dwa_bc",
    )

    parser = build_arg_parser()
    args = parser.parse_args()
    if args.enable_visualization and args.disable_visualization:
        parser.error("--enable_visualization 和 --disable_visualization 不能同时指定")
    if args.target_kept_episodes is not None and int(args.target_kept_episodes) <= 0:
        parser.error("--target_kept_episodes must be > 0")
    if args.max_attempt_episodes is not None and int(args.max_attempt_episodes) <= 0:
        parser.error("--max_attempt_episodes must be > 0")
    if not (0.05 <= float(args.intent_horizon_sec) <= 5.0):
        parser.error("--intent_horizon_sec must be in [0.05, 5.0]")
    if not (0.05 <= float(args.intent_dt_sec) <= 1.0):
        parser.error("--intent_dt_sec must be in [0.05, 1.0]")
    if int(args.intent_commit_steps) < 1:
        parser.error("--intent_commit_steps must be >= 1")
    if int(args.intent_replan_interval_steps) < 1:
        parser.error("--intent_replan_interval_steps must be >= 1")
    if int(args.intent_max_staleness_steps) < 1:
        parser.error("--intent_max_staleness_steps must be >= 1")
    if args.enable_visualization:
        args.enable_visualization = True
    elif args.disable_visualization:
        args.enable_visualization = False
    else:
        args.enable_visualization = False

    collect_dataset(args)


if __name__ == "__main__":
    main()
