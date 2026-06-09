#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from gnn_bc_tools.path_utils import ensure_runtime_modules, inject_workspace_paths


MAP1_OVERLAP_POSE_BANK: List[Tuple[Tuple[float, float], Tuple[float, float]]] = [
    ((-0.50, -5.00), (1.10, -1.80)),
    ((1.10, -5.00), (-0.50, -1.80)),
    ((0.30, -5.50), (0.30, -1.60)),
    ((0.80, -2.20), (-0.40, -4.80)),
    ((-0.40, -4.80), (0.80, -2.20)),
]


@dataclass
class TrialParams:
    neighbor_soft_dist: float
    neighbor_stop_dist: float
    neighbor_hard_stop_dist: float
    orca_blend_max: float
    time_horizon: float
    velocity_smoothing_alpha: float
    dwa_heading_weight: float
    dwa_dist_weight: float
    dwa_velocity_weight: float
    dwa_safety_margin: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "neighbor_soft_dist": float(self.neighbor_soft_dist),
            "neighbor_stop_dist": float(self.neighbor_stop_dist),
            "neighbor_hard_stop_dist": float(self.neighbor_hard_stop_dist),
            "orca_blend_max": float(self.orca_blend_max),
            "time_horizon": float(self.time_horizon),
            "velocity_smoothing_alpha": float(self.velocity_smoothing_alpha),
            "dwa_heading_weight": float(self.dwa_heading_weight),
            "dwa_dist_weight": float(self.dwa_dist_weight),
            "dwa_velocity_weight": float(self.dwa_velocity_weight),
            "dwa_safety_margin": float(self.dwa_safety_margin),
        }


def _pairwise_min_dist(positions: Dict[str, np.ndarray]) -> float:
    keys = list(positions.keys())
    if len(keys) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(keys)):
        pi = np.asarray(positions[keys[i]], dtype=np.float32)
        for j in range(i + 1, len(keys)):
            pj = np.asarray(positions[keys[j]], dtype=np.float32)
            d = float(np.linalg.norm(pi - pj))
            if d < best:
                best = d
    return best


def build_env_config(args) -> Dict:
    map_number = int(args.map_number)
    use_overlap_pairs = bool(args.use_overlap_pairs) and map_number == 1
    return {
        "num_agents": int(args.num_agents),
        "map_number": map_number,
        "use_random_mode": False,
        "fallback_pose_bank": MAP1_OVERLAP_POSE_BANK if use_overlap_pairs else None,
        "max_episode_steps": int(args.max_episode_steps),
        "communication_range": float(args.communication_range),
        "enable_neighbor_obs": True,
        "enable_local_map": False,
        "comm_mode": str(args.comm_mode),
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
        "auto_reset_agents": False,
        "num_dynamic_obstacles": 0,
        "obs_speed": 0.0,
        "log_every_n_steps": int(args.log_every_n_steps),
    }


def sample_trial_params(rng: np.random.Generator) -> TrialParams:
    hard = float(rng.uniform(0.20, 0.31))
    stop = float(hard + rng.uniform(0.06, 0.13))
    soft = float(stop + rng.uniform(0.20, 0.40))
    return TrialParams(
        neighbor_soft_dist=min(1.10, soft),
        neighbor_stop_dist=stop,
        neighbor_hard_stop_dist=hard,
        orca_blend_max=float(rng.uniform(0.58, 0.92)),
        time_horizon=float(rng.uniform(1.8, 3.0)),
        velocity_smoothing_alpha=float(rng.uniform(0.45, 0.78)),
        dwa_heading_weight=float(rng.uniform(1.6, 2.6)),
        dwa_dist_weight=float(rng.uniform(2.0, 4.2)),
        dwa_velocity_weight=float(rng.uniform(0.9, 2.2)),
        dwa_safety_margin=float(rng.uniform(0.08, 0.22)),
    )


def _meets_constraints(metrics: Dict[str, float], args) -> bool:
    return (
        float(metrics["success_rate"]) >= float(args.min_success_rate)
        and float(metrics["collision_rate"]) <= float(args.max_collision_rate)
        and float(metrics["timeout_rate"]) <= float(args.max_timeout_rate)
    )


def _rank_key(res: Dict, args) -> Tuple[float, ...]:
    m = res["metrics"]
    feasible = 1.0 if _meets_constraints(m, args) else 0.0
    return (
        feasible,
        float(m["success_rate"]),
        -float(m["collision_rate"]),
        -float(m["timeout_rate"]),
        -float(m["deadlock_rate"]),
        -float(m["near_rate"]),
        float(m["mean_speed"]),
        float(res["score"]),
    )


def evaluate_trial(
    trial_idx: int,
    params: TrialParams,
    args,
    env_cfg: Dict,
) -> Dict:
    from gnn_marl_training.gnn_marl_env import env_creator
    from gnn_bc_tools.expert_orca_dwa import ORCADWATeacher

    env = env_creator(env_cfg)
    teacher = ORCADWATeacher(
        communication_range=float(args.communication_range),
        max_linear_speed=0.22,
        max_angular_speed=1.2,
        robot_radius=float(args.robot_radius),
        time_horizon=float(params.time_horizon),
        laser_obstacle_max_dist=float(args.laser_obstacle_max_dist),
        velocity_smoothing_alpha=float(params.velocity_smoothing_alpha),
        neighbor_soft_dist=float(params.neighbor_soft_dist),
        neighbor_stop_dist=float(params.neighbor_stop_dist),
        neighbor_hard_stop_dist=float(params.neighbor_hard_stop_dist),
        orca_blend_max=float(params.orca_blend_max),
        dwa_heading_weight=float(params.dwa_heading_weight),
        dwa_dist_weight=float(params.dwa_dist_weight),
        dwa_velocity_weight=float(params.dwa_velocity_weight),
        dwa_safety_margin=float(params.dwa_safety_margin),
        intent_horizon_sec=float(args.intent_horizon_sec),
        intent_dt_sec=float(args.intent_dt_sec),
        intent_safe_margin=float(args.intent_safe_margin),
        intent_commit_steps=int(args.intent_commit_steps),
        intent_replan_interval_steps=int(args.intent_replan_interval_steps),
        intent_dropout_prob=float(args.comm_dropout_prob),
        intent_latency_steps=int(args.comm_latency_steps),
        intent_jitter_steps=int(args.comm_jitter_steps),
        intent_max_staleness_steps=int(args.intent_max_staleness_steps),
        intent_seed=(None if args.seed is None else int(args.seed) + trial_idx * 9973),
    )

    total_steps = 0
    total_interaction_steps = 0
    total_near_steps = 0
    total_collisions = 0
    total_successes = 0
    timeout_episodes = 0
    deadlock_episodes = 0
    mean_speed_acc = 0.0
    mean_min_dist_acc = 0.0

    try:
        for ep in range(int(args.episodes)):
            seed = None if args.seed is None else int(args.seed) + trial_idx * 10000 + ep
            obs_dict, _ = env.reset(seed=seed)
            teacher.reset()

            ep_steps = 0
            ep_interaction = 0
            ep_near = 0
            ep_speed_sum = 0.0
            ep_min_dist_sum = 0.0
            ep_goal_agents = set()
            ep_collision_agents = set()
            ep_timeout = False

            while True:
                action_dict = teacher.compute_actions(env, obs_dict)
                obs_dict, _, done_dict, trunc_dict, info_dict = env.step(action_dict)
                ep_steps += 1
                total_steps += 1

                for aid in env.agent_ids:
                    event = str(info_dict.get(aid, {}).get("event", ""))
                    if event == "goal":
                        ep_goal_agents.add(aid)
                    elif event == "collision":
                        ep_collision_agents.add(aid)

                min_dist = _pairwise_min_dist(env.robot_positions)
                ep_min_dist_sum += min_dist
                if min_dist < float(args.interaction_dist):
                    ep_interaction += 1
                if min_dist < float(args.near_dist):
                    ep_near += 1

                speeds = [
                    float(np.linalg.norm(np.asarray(env.robot_velocities[aid], dtype=np.float32)))
                    for aid in env.agent_ids
                ]
                ep_speed_sum += float(np.mean(speeds)) if speeds else 0.0

                if bool(args.stop_episode_on_collision) and ep_collision_agents:
                    break
                if done_dict.get("__all__", False) or trunc_dict.get("__all__", False):
                    ep_timeout = bool(trunc_dict.get("__all__", False))
                    break

            ep_collisions = len(ep_collision_agents)
            ep_successes = len(ep_goal_agents)
            if done_dict.get("__all__", False) or trunc_dict.get("__all__", False):
                ep_collisions = max(
                    ep_collisions,
                    sum(int(info_dict[aid].get("episode_collisions", 0)) for aid in env.agent_ids),
                )
                ep_successes = max(
                    ep_successes,
                    sum(int(info_dict[aid].get("episode_successes", 0)) for aid in env.agent_ids),
                )
            total_collisions += ep_collisions
            total_successes += ep_successes

            if ep_timeout:
                timeout_episodes += 1
                ep_mean_speed = ep_speed_sum / max(ep_steps, 1)
                if ep_mean_speed < float(args.deadlock_speed_thresh):
                    deadlock_episodes += 1

            total_interaction_steps += ep_interaction
            total_near_steps += ep_near
            mean_speed_acc += ep_speed_sum / max(ep_steps, 1)
            mean_min_dist_acc += ep_min_dist_sum / max(ep_steps, 1)

    finally:
        env.close()

    agent_tasks = max(1, int(args.episodes) * int(args.num_agents))
    total_steps_safe = max(1, total_steps)
    success_rate = float(total_successes) / agent_tasks
    collision_rate = float(total_collisions) / agent_tasks
    interaction_rate = float(total_interaction_steps) / total_steps_safe
    near_rate = float(total_near_steps) / total_steps_safe
    timeout_rate = float(timeout_episodes) / max(1, int(args.episodes))
    deadlock_rate = float(deadlock_episodes) / max(1, int(args.episodes))
    mean_speed = float(mean_speed_acc) / max(1, int(args.episodes))
    mean_min_dist = float(mean_min_dist_acc) / max(1, int(args.episodes))

    legacy_score = (
        3.0 * success_rate
        - 4.5 * collision_rate
        - 1.5 * timeout_rate
        - 1.2 * deadlock_rate
        + 1.0 * interaction_rate
        - 0.8 * near_rate
        + 0.3 * mean_speed
    )
    safety_score = (
        6.0 * success_rate
        - 6.2 * collision_rate
        - 2.4 * timeout_rate
        - 1.5 * deadlock_rate
        - 0.5 * near_rate
        + 0.2 * mean_speed
    )
    if success_rate < float(args.min_success_rate):
        safety_score -= 2.0 + 8.0 * (float(args.min_success_rate) - success_rate)
    if collision_rate > float(args.max_collision_rate):
        safety_score -= 2.0 + 8.0 * (collision_rate - float(args.max_collision_rate))
    if timeout_rate > float(args.max_timeout_rate):
        safety_score -= 1.0 + 4.0 * (timeout_rate - float(args.max_timeout_rate))
    score = float(safety_score if str(args.score_mode) == "safety_first" else legacy_score)
    metrics = {
        "success_rate": float(success_rate),
        "collision_rate": float(collision_rate),
        "interaction_rate": float(interaction_rate),
        "near_rate": float(near_rate),
        "timeout_rate": float(timeout_rate),
        "deadlock_rate": float(deadlock_rate),
        "mean_speed": float(mean_speed),
        "mean_pair_min_dist": float(mean_min_dist),
        "episodes": int(args.episodes),
        "total_steps": int(total_steps),
    }

    return {
        "trial_index": int(trial_idx),
        "score": float(score),
        "legacy_score": float(legacy_score),
        "safety_score": float(safety_score),
        "meets_constraints": bool(_meets_constraints(metrics, args)),
        "params": params.to_dict(),
        "metrics": metrics,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Small ORCA/DWA hyperparameter tuner with configurable test map."
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--num_trials", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num_agents", type=int, default=4)
    p.add_argument("--map_number", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--max_episode_steps", type=int, default=900)
    p.add_argument("--communication_range", type=float, default=3.5)
    p.add_argument("--comm_mode", type=str, default="decentralized", choices=["decentralized", "centralized_oracle", "ros2_bridge"])
    p.add_argument("--comm_dropout_prob", type=float, default=0.02)
    p.add_argument("--comm_latency_steps", type=int, default=0)
    p.add_argument("--comm_jitter_steps", type=int, default=0)
    p.add_argument("--comm_noise_std", type=float, default=0.02)
    p.add_argument("--use_overlap_pairs", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--robot_radius", type=float, default=0.25)
    p.add_argument("--laser_obstacle_max_dist", type=float, default=2.2)

    p.add_argument("--interaction_dist", type=float, default=1.2)
    p.add_argument("--near_dist", type=float, default=0.55)
    p.add_argument("--deadlock_speed_thresh", type=float, default=0.03)
    p.add_argument("--stop_episode_on_collision", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--intent_horizon_sec", type=float, default=1.8)
    p.add_argument("--intent_dt_sec", type=float, default=0.2)
    p.add_argument("--intent_safe_margin", type=float, default=0.12)
    p.add_argument("--intent_commit_steps", type=int, default=4)
    p.add_argument("--intent_replan_interval_steps", type=int, default=2)
    p.add_argument("--intent_max_staleness_steps", type=int, default=20)
    p.add_argument("--score_mode", type=str, default="safety_first", choices=["safety_first", "legacy"])
    p.add_argument("--rank_mode", type=str, default="lexicographic", choices=["lexicographic", "score"])
    p.add_argument("--min_success_rate", type=float, default=0.35)
    p.add_argument("--max_collision_rate", type=float, default=0.20)
    p.add_argument("--max_timeout_rate", type=float, default=0.35)

    p.add_argument("--enable_visualization", action="store_true")
    p.add_argument("--tracking_viz_interval", type=int, default=8)
    p.add_argument("--env_log_level", type=str, default="WARNING")
    p.add_argument("--sim_wait_wall_timeout", type=float, default=2.5)
    p.add_argument("--log_every_n_steps", type=int, default=120)

    p.add_argument("--output_dir", type=str, default="~/work/multi-robot-exploration-rl/bc_tune_results")
    p.add_argument("--tag", type=str, default=None)
    return p


def main() -> None:
    ensure_runtime_modules(
        required_modules=["numpy", "gymnasium", "ray"],
        runner_module="gnn_bc_tools.tune_orca_dwa_map1",
    )
    repo_root, _, _ = inject_workspace_paths()

    args = build_arg_parser().parse_args()
    if not (0.0 <= float(args.min_success_rate) <= 1.0):
        raise ValueError("min_success_rate must be in [0, 1]")
    if not (0.0 <= float(args.max_collision_rate) <= 1.0):
        raise ValueError("max_collision_rate must be in [0, 1]")
    if not (0.0 <= float(args.max_timeout_rate) <= 1.0):
        raise ValueError("max_timeout_rate must be in [0, 1]")
    if not (0.05 <= float(args.intent_horizon_sec) <= 5.0):
        raise ValueError("intent_horizon_sec must be in [0.05, 5.0]")
    if not (0.05 <= float(args.intent_dt_sec) <= 1.0):
        raise ValueError("intent_dt_sec must be in [0.05, 1.0]")
    if int(args.intent_commit_steps) < 1:
        raise ValueError("intent_commit_steps must be >= 1")
    if int(args.intent_replan_interval_steps) < 1:
        raise ValueError("intent_replan_interval_steps must be >= 1")
    if int(args.intent_max_staleness_steps) < 1:
        raise ValueError("intent_max_staleness_steps must be >= 1")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    env_cfg = build_env_config(args)

    tag = args.tag or f"orca_dwa_tune_map{int(args.map_number)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_json = out_dir / f"{tag}.json"

    print("=" * 90)
    print("ORCA/DWA small tuner")
    print(f"repo:        {repo_root}")
    print(f"output:      {out_json}")
    print(f"map_number:  {int(args.map_number)}")
    print(f"episodes:    {args.episodes} per trial")
    print(f"trials:      {args.num_trials}")
    overlap_effective = bool(args.use_overlap_pairs) and int(args.map_number) == 1
    print(f"overlap bank:{overlap_effective}")
    print(
        f"selection:   mode={args.rank_mode}, score={args.score_mode}, "
        f"min_succ={args.min_success_rate:.2f}, max_coll={args.max_collision_rate:.2f}, "
        f"max_timeout={args.max_timeout_rate:.2f}"
    )
    print(
        f"intent:      horizon={args.intent_horizon_sec:.2f}s, dt={args.intent_dt_sec:.2f}s, "
        f"commit={int(args.intent_commit_steps)} step, async_replan={int(args.intent_replan_interval_steps)} step"
    )
    if bool(args.use_overlap_pairs) and int(args.map_number) != 1:
        print("note: overlap pose bank is only applied on map_number=1; ignored for current map.")
    print("=" * 90)

    results = []
    for tid in range(int(args.num_trials)):
        params = sample_trial_params(rng)
        trial_res = evaluate_trial(tid, params, args, env_cfg)
        results.append(trial_res)
        m = trial_res["metrics"]
        feasible = "ok" if trial_res.get("meets_constraints", False) else "ng"
        print(
            f"[trial {tid + 1:02d}/{args.num_trials}] "
            f"{feasible} score={trial_res['score']:+.4f} "
            f"succ={m['success_rate']:.3f} coll={m['collision_rate']:.3f} "
            f"inter={m['interaction_rate']:.3f} timeout={m['timeout_rate']:.3f} dead={m['deadlock_rate']:.3f}"
        )

    if str(args.rank_mode) == "lexicographic":
        results = sorted(results, key=lambda x: _rank_key(x, args), reverse=True)
    else:
        results = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results[0]

    payload = {
        "created_at": datetime.now().isoformat(),
        "tag": tag,
        "env_config": env_cfg,
        "args": vars(args),
        "best": best,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    bp = best["params"]
    print("=" * 90)
    print("BEST PARAMS")
    print(json.dumps(bp, ensure_ascii=False, indent=2))
    print("BEST METRICS")
    print(json.dumps(best["metrics"], ensure_ascii=False, indent=2))
    print("-" * 90)
    print("Suggested collection command:")
    target_kept = int(args.episodes)
    max_attempt = max(target_kept * 4, target_kept)
    print(
        "ros2 run gnn_bc_tools collect_orca_dwa_bc "
        f"--env_stage 1 --map_number {int(args.map_number)} --episodes {args.episodes} "
        f"--target_kept_episodes {target_kept} --max_attempt_episodes {max_attempt} "
        f"--seed {int(args.seed) if args.seed is not None else 42} "
        f"--num_agents {int(args.num_agents)} "
        f"--max_episode_steps {int(args.max_episode_steps)} "
        f"--communication_range {float(args.communication_range):.3f} "
        f"--comm_mode {str(args.comm_mode)} "
        f"--comm_dropout_prob {float(args.comm_dropout_prob):.3f} "
        f"--comm_latency_steps {int(args.comm_latency_steps)} "
        f"--comm_jitter_steps {int(args.comm_jitter_steps)} "
        f"--comm_noise_std {float(args.comm_noise_std):.3f} "
        f"--sim_wait_wall_timeout {float(args.sim_wait_wall_timeout):.3f} "
        "--early_stop_on_collision --discard_failure_episodes "
        f"--intent_horizon_sec {float(args.intent_horizon_sec):.3f} "
        f"--intent_dt_sec {float(args.intent_dt_sec):.3f} "
        f"--intent_safe_margin {float(args.intent_safe_margin):.3f} "
        f"--intent_commit_steps {int(args.intent_commit_steps)} "
        f"--intent_replan_interval_steps {int(args.intent_replan_interval_steps)} "
        f"--intent_max_staleness_steps {int(args.intent_max_staleness_steps)} "
        f"--neighbor_soft_dist {bp['neighbor_soft_dist']:.3f} "
        f"--neighbor_stop_dist {bp['neighbor_stop_dist']:.3f} "
        f"--neighbor_hard_stop_dist {bp['neighbor_hard_stop_dist']:.3f} "
        f"--orca_blend_max {bp['orca_blend_max']:.3f} "
        f"--time_horizon {bp['time_horizon']:.3f} "
        f"--velocity_smoothing_alpha {bp['velocity_smoothing_alpha']:.3f} "
        f"--dwa_heading_weight {bp['dwa_heading_weight']:.3f} "
        f"--dwa_dist_weight {bp['dwa_dist_weight']:.3f} "
        f"--dwa_velocity_weight {bp['dwa_velocity_weight']:.3f} "
        f"--dwa_safety_margin {bp['dwa_safety_margin']:.3f}"
    )
    print(f"Saved: {out_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
