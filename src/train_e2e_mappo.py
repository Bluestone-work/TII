#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

try:
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: ray. "
        "Please activate the training environment (e.g. conda env with Ray/RLlib) before running."
    ) from exc


def _inject_workspace_paths() -> Path:
    # Works both inside the original workspace and after copying this file out for review.
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3] if len(this_file.parents) >= 4 else this_file.parent
    candidates = [
        repo_root / "src" / "intent_marl_training",
        repo_root / "build" / "intent_marl_training",
        repo_root / "build" / "intent_marl_training" / "build" / "lib",
        this_file.parent,
    ]
    existing = [str(p) for p in candidates if p.exists()]

    for p in existing:
        while p in sys.path:
            sys.path.remove(p)
    for p in reversed(existing):
        sys.path.insert(0, p)

    py_entries = existing + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    dedup = []
    for e in py_entries:
        if e not in dedup:
            dedup.append(e)
    os.environ["PYTHONPATH"] = os.pathsep.join(dedup)

    install_dir = repo_root / "install"
    if install_dir.exists():
        ws_prefixes = [
            str(p)
            for p in install_dir.iterdir()
            if p.is_dir() and not p.name.startswith("_") and p.name != "COLCON_IGNORE"
        ]
        old = [p for p in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if p]
        merged = ws_prefixes + [p for p in old if p not in ws_prefixes]
        os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join(merged)

    return repo_root


REPO_ROOT = _inject_workspace_paths()

logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)
os.environ.setdefault("RAY_DISABLE_METRICS_COLLECTION", "1")
os.environ.setdefault("RAY_DISABLE_IMPORT_METRICS_REPORTER", "1")
os.environ.setdefault("RAY_metrics_export_port", "0")

try:
    from intent_marl_training.e2e_env_wrapper import e2e_env_creator
except ModuleNotFoundError:
    from e2e_env_wrapper import e2e_env_creator  # type: ignore

try:
    from intent_marl_training.intent_mappo_model import MODEL_NAME_INTENT, IntentMAPPOMLPModel
except ModuleNotFoundError:
    from intent_mappo_model import MODEL_NAME_INTENT, IntentMAPPOMLPModel  # type: ignore


MAP6_HEAD_ON_POSE_BANK = [
    ((-5.1, -0.85), (5.1, -0.85)),
    ((5.1, -0.85), (-5.1, -0.85)),
]

MAP6_CROSS_POSE_BANK = [
    ((-5.1, 0.85), (5.1, 0.85)),
    ((0.85, -5.1), (0.85, 5.1)),
]


ENV_CURRICULUM = {
    1: {
        "name": "Stage 1 · static warmup",
        "map_number": 3,
        "max_episode_steps": 1500,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "use_random_mode": True,
    },
    6: {
        "name": "Stage 6 · two-robot head-on",
        "map_number": 6,
        "max_episode_steps": 900,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "use_random_mode": False,
        "fallback_pose_bank": MAP6_HEAD_ON_POSE_BANK,
    },
    7: {
        "name": "Stage 7 · two-robot crossing",
        "map_number": 6,
        "max_episode_steps": 1000,
        "num_obstacles": 0,
        "obs_speed_scale": 0.0,
        "use_random_mode": False,
        "fallback_pose_bank": MAP6_CROSS_POSE_BANK,
    },
}


E2E_BASE_OBS_DIM = 148  # 36*4 scan history + 2 target + 2 velocity, no extra social features


def _extract_steps(result: Dict) -> int:
    for key in (
        "timesteps_total",
        "num_env_steps_sampled_lifetime",
        "num_env_steps_sampled",
    ):
        value = result.get(key)
        if value is not None:
            return int(value)
    return 0


def _extract_reward(result: Dict) -> float:
    for value in (
        result.get("episode_reward_mean"),
        result.get("env_runners", {}).get("episode_reward_mean"),
        result.get("sampler_results", {}).get("episode_reward_mean"),
    ):
        if value is not None:
            return float(value)
    return float("nan")



def _build_base_env_config(args, stage_cfg: Dict) -> Dict:
    use_random_mode = stage_cfg.get("use_random_mode", True)
    if args.use_random_mode is not None:
        use_random_mode = bool(args.use_random_mode)

    fallback_pose_bank = stage_cfg.get("fallback_pose_bank")

    return {
        "num_agents": int(args.num_agents),
        "model_num_agents": int(max(args.model_num_agents, args.num_agents)),
        "map_number": int(stage_cfg["map_number"]),
        "use_random_mode": bool(use_random_mode),
        "fallback_pose_bank": fallback_pose_bank,
        "max_episode_steps": int(stage_cfg["max_episode_steps"]),
        "communication_range": float(args.communication_range),
        "enable_neighbor_obs": True,
        "enable_local_map": False,
        "comm_mode": str(args.comm_mode),
        "comm_dropout_prob": float(args.comm_dropout_prob),
        "comm_latency_steps": int(args.comm_latency_steps),
        "comm_jitter_steps": int(args.comm_jitter_steps),
        "comm_noise_std": float(args.comm_noise_std),
        "reset_on_collision_event": True,
        "collision_ends_episode": True,
        "collision_hard_dist": float(args.collision_hard_dist),
        "collision_persist_dist": float(args.collision_persist_dist),
        "collision_persist_steps": int(args.collision_persist_steps),
        "near_wall_penalty_dist": float(args.near_wall_penalty_dist),
        "rolling_lookahead_dist": 0.8,
        "progress_reward_scale": float(args.progress_reward_scale),
        "path_progress_reward_scale": float(args.path_progress_reward_scale),
        "goal_progress_reward_scale": 0.0,
        "goal_reward": float(args.goal_reward),
        "collision_penalty": float(args.collision_penalty),
        "time_penalty": float(args.time_penalty),
        "lateral_penalty_scale": 0.0,
        "heading_align_reward_scale": 0.0,
        "narrow_forward_penalty_scale": 0.0,
        "near_collision_dist": float(args.lidar_near_collision_dist),
        "near_collision_penalty_scale": float(args.lidar_near_collision_penalty_scale),
        "front_safety_dist": float(args.front_safety_dist),
        "front_safety_penalty_scale": float(args.front_safety_penalty_scale),
        "neighbor_safety_dist": float(args.neighbor_safety_dist),
        "neighbor_safety_penalty_scale": float(args.neighbor_safety_penalty_scale),
        # Hard-disable all helper controllers. RL owns the full action at close range.
        "shield_enable": False,
        "tracking_assist_enable": False,
        "local_executor_enable": False,
        "msa3c_action_mode": False,
        "msa3c_social_feature_enable": False,
        "base_zone_manager_enable": False,
        "hybrid_control_enable": False,
        "social_yield_reward_scale": 0.0,
        "social_passage_reward_scale": 0.0,
        "social_clear_reward_scale": 0.0,
        "max_reverse_speed": float(args.max_reverse_speed),
        "enable_visualization": bool(args.enable_visualization),
        "tracking_viz_interval": int(args.tracking_viz_interval),
        "env_log_level": str(args.env_log_level),
        "debug_comm": bool(args.debug_comm),
        "sim_wait_wall_timeout": float(args.sim_wait_wall_timeout),
        "auto_reset_agents": False,
        "num_dynamic_obstacles": int(stage_cfg["num_obstacles"]),
        "obs_speed": float(args.dynamic_obstacle_speed) * float(stage_cfg["obs_speed_scale"]),
    }



def _build_reward_config(args) -> Dict:
    return {
        "interaction_dist": float(args.e2e_interaction_dist),
        "neighbor_safe_dist": float(args.e2e_neighbor_safe_dist),
        "neighbor_penalty_scale": float(args.e2e_neighbor_penalty_scale),
        "escape_reward_scale": float(args.e2e_escape_reward_scale),
        "approach_penalty_scale": float(args.e2e_approach_penalty_scale),
        "clearance_reward_scale": float(args.e2e_clearance_reward_scale),
        "max_escape_delta": float(args.e2e_max_escape_delta),
    }



def _run_training(
    config: PPOConfig,
    train_steps: int,
    checkpoint_freq: int,
    storage_dir: str,
    run_name: str,
    restore_checkpoint: str | None = None,
):
    os.makedirs(storage_dir, exist_ok=True)
    run_dir = os.path.join(storage_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    algo = config.build()
    if restore_checkpoint:
        ckpt_path = os.path.expanduser(str(restore_checkpoint))
        print(f"restore_checkpoint: {ckpt_path}")
        algo.restore(ckpt_path)

    best_reward = float("-inf")
    best_ckpt = None
    last_ckpt = None
    start = time.time()
    iteration = 0

    while True:
        iteration += 1
        result = algo.train()
        steps = _extract_steps(result)
        rew = _extract_reward(result)
        pct = min(100.0, (steps / max(train_steps, 1)) * 100.0)
        elapsed = time.time() - start
        print(
            f"\r[iter {iteration:04d}] steps={steps:>9,}/{train_steps:,} "
            f"({pct:5.1f}%) reward={rew:8.3f} elapsed={elapsed/60.0:7.1f}m",
            end="",
            flush=True,
        )

        if rew > best_reward:
            best_reward = rew
            best_ckpt = algo.save(os.path.join(run_dir, "best"))

        if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
            last_ckpt = algo.save(run_dir)

        if steps >= train_steps:
            break

    print()
    last_ckpt = algo.save(run_dir)
    algo.stop()
    return best_ckpt or last_ckpt, best_reward



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a pure end-to-end MAPPO baseline (no yield / no shield / no social arbitration)."
    )

    parser.add_argument("--env_stage", type=int, default=6, choices=sorted(ENV_CURRICULUM.keys()))
    parser.add_argument("--map_number", type=int, default=None, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--use_random_mode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--model_num_agents", type=int, default=2)
    parser.add_argument("--communication_range", type=float, default=3.5)

    parser.add_argument(
        "--comm_mode",
        type=str,
        default="centralized_oracle",
        choices=["decentralized", "centralized_oracle", "ros2_bridge"],
    )
    parser.add_argument("--comm_dropout_prob", type=float, default=0.0)
    parser.add_argument("--comm_latency_steps", type=int, default=0)
    parser.add_argument("--comm_jitter_steps", type=int, default=0)
    parser.add_argument("--comm_noise_std", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train_batch_size", type=int, default=4000)
    parser.add_argument("--rollout_fragment_length", type=int, default=200)
    parser.add_argument("--sample_timeout_s", type=int, default=1200)
    parser.add_argument(
        "--batch_mode",
        type=str,
        default="truncate_episodes",
        choices=["truncate_episodes", "complete_episodes"],
    )

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=256)
    parser.add_argument("--max_seq_len", type=int, default=20)

    parser.add_argument("--progress_reward_scale", type=float, default=5.0)
    parser.add_argument("--path_progress_reward_scale", type=float, default=2.5)
    parser.add_argument("--goal_reward", type=float, default=35.0)
    parser.add_argument("--collision_penalty", type=float, default=90.0)
    parser.add_argument("--time_penalty", type=float, default=0.002)
    parser.add_argument("--idle_penalty_scale", type=float, default=0.02)
    parser.add_argument("--near_wall_penalty_dist", type=float, default=0.30)

    parser.add_argument("--collision_hard_dist", type=float, default=0.20)
    parser.add_argument("--collision_persist_dist", type=float, default=0.26)
    parser.add_argument("--collision_persist_steps", type=int, default=2)
    parser.add_argument("--lidar_near_collision_dist", type=float, default=0.45)
    parser.add_argument("--lidar_near_collision_penalty_scale", type=float, default=2.0)
    parser.add_argument("--front_safety_dist", type=float, default=0.55)
    parser.add_argument("--front_safety_penalty_scale", type=float, default=0.0)
    parser.add_argument("--neighbor_safety_dist", type=float, default=0.72)
    parser.add_argument("--neighbor_safety_penalty_scale", type=float, default=0.0)

    parser.add_argument("--e2e_interaction_dist", type=float, default=1.25)
    parser.add_argument("--e2e_neighbor_safe_dist", type=float, default=0.72)
    parser.add_argument("--e2e_neighbor_penalty_scale", type=float, default=1.25)
    parser.add_argument("--e2e_escape_reward_scale", type=float, default=0.45)
    parser.add_argument("--e2e_approach_penalty_scale", type=float, default=0.12)
    parser.add_argument("--e2e_clearance_reward_scale", type=float, default=0.10)
    parser.add_argument("--e2e_max_escape_delta", type=float, default=0.15)

    parser.add_argument("--max_reverse_speed", type=float, default=0.08)
    parser.add_argument("--dynamic_obstacle_speed", type=float, default=0.30)
    parser.add_argument("--enable_visualization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tracking_viz_interval", type=int, default=4)
    parser.add_argument("--env_log_level", type=str, default="WARNING")
    parser.add_argument("--debug_comm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sim_wait_wall_timeout", type=float, default=2.5)

    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/work/multi-robot-exploration-rl/ray_results",
    )
    parser.add_argument("--run_name", type=str, default="e2e_mappo")
    parser.add_argument("--restore_checkpoint", type=str, default=None)

    return parser



def main():
    args = build_parser().parse_args()

    if int(args.num_workers) > 1:
        print(
            "[WARN] num_workers > 1 with a shared ROS/Gazebo world can cause conflicting resets/actions. "
            "Recommended: --num_workers 1."
        )

    if int(args.env_stage) in {6, 7} and int(args.num_agents) != 2:
        raise SystemExit("For env_stage 6/7 this baseline assumes --num_agents 2.")

    stage_cfg = dict(ENV_CURRICULUM[int(args.env_stage)])
    if args.map_number is not None:
        stage_cfg["map_number"] = int(args.map_number)

    base_env_config = _build_base_env_config(args, stage_cfg)
    reward_config = _build_reward_config(args)

    model_num_agents = int(max(args.model_num_agents, args.num_agents))
    max_neighbors = min(model_num_agents - 1, 5)
    base_obs_dim = int(E2E_BASE_OBS_DIM)
    neighbor_dim = int(max_neighbors * 5)
    base_actor_obs_dim = int(base_obs_dim + neighbor_dim)
    actor_obs_dim = int(base_actor_obs_dim)
    critic_token_obs_dim = int(base_obs_dim)
    global_state_dim = int(model_num_agents * critic_token_obs_dim)

    register_env("e2e_marl", e2e_env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_INTENT, IntentMAPPOMLPModel)

    env_config = {
        "base_env_config": base_env_config,
        "reward_config": reward_config,
    }

    print("=" * 88)
    print("Pure End-to-End MAPPO Training")
    print(f"repo:             {REPO_ROOT}")
    print(f"stage:            {args.env_stage} ({stage_cfg['name']})")
    print(f"map:              {stage_cfg['map_number']}")
    print(f"num_agents:       {args.num_agents} (model_slots={model_num_agents})")
    print(
        f"spawn_mode:       {'random' if base_env_config.get('use_random_mode', True) else 'fixed_pose_bank'}"
    )
    print(f"comm_mode:        {args.comm_mode}")
    print("controller:       pure_rl_direct_velocity")
    print("helpers:          shield=OFF tracking=OFF local_executor=OFF hybrid=OFF social=OFF")
    print(
        f"reward(base):     progress={args.progress_reward_scale:.2f} path={args.path_progress_reward_scale:.2f} "
        f"goal={args.goal_reward:.1f} collision={args.collision_penalty:.1f} lidar_near={args.lidar_near_collision_penalty_scale:.2f}"
    )
    print(
        f"reward(e2e):      neighbor_safe={args.e2e_neighbor_safe_dist:.2f}m "
        f"neighbor_pen={args.e2e_neighbor_penalty_scale:.2f} escape={args.e2e_escape_reward_scale:.2f} "
        f"approach={args.e2e_approach_penalty_scale:.2f} clear={args.e2e_clearance_reward_scale:.2f}"
    )
    print(f"actor_obs_dim:    {actor_obs_dim} (base={base_obs_dim}, neighbor={neighbor_dim})")
    print(f"critic_obs_dim:   {global_state_dim} (token={critic_token_obs_dim})")
    print(f"train_steps:      {args.train_steps:,}")
    print("=" * 88)

    policy_name = "shared_policy"
    model_cfg = {
        "custom_model": MODEL_NAME_INTENT,
        "custom_model_config": {
            "actor_obs_dim": actor_obs_dim,
            "global_state_dim": global_state_dim,
            "critic_token_obs_dim": critic_token_obs_dim,
            "reset_flag_dim": 1,
            "hidden_dim": int(args.hidden_dim),
            "lstm_hidden_dim": int(args.lstm_hidden_dim),
            "num_agents": int(model_num_agents),
            "base_obs_dim": int(base_obs_dim),
            "neighbor_dim": int(neighbor_dim),
            "intent_dim": 0,
            "yield_obs_dim": 0,
            "base_actor_obs_dim": int(base_actor_obs_dim),
            "critic_attention_heads": 4,
        },
        "max_seq_len": int(args.max_seq_len),
    }

    config = (
        PPOConfig()
        .environment(env="e2e_marl", env_config=env_config, disable_env_checking=True)
        .framework("torch")
        .env_runners(
            num_env_runners=int(args.num_workers),
            num_envs_per_env_runner=1,
            sample_timeout_s=int(args.sample_timeout_s),
            batch_mode=str(args.batch_mode),
            rollout_fragment_length=int(args.rollout_fragment_length),
        )
        .training(
            lr=float(args.lr),
            gamma=0.99,
            lambda_=0.95,
            train_batch_size=int(args.train_batch_size),
            clip_param=0.2,
            entropy_coeff=0.01,
            vf_clip_param=50.0,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            minibatch_size=256,
            num_epochs=10,
            model=model_cfg,
        )
        .multi_agent(
            policies={policy_name: (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            policies_to_train=[policy_name],
        )
        .resources(num_gpus=int(__import__("torch").cuda.is_available()))
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    ray.init(ignore_reinit_error=True)
    out_dir = os.path.expanduser(args.output_dir)
    run_name = f"{args.run_name}_stage{args.env_stage}_n{args.num_agents}"

    try:
        checkpoint, best_reward = _run_training(
            config=config,
            train_steps=int(args.train_steps),
            checkpoint_freq=int(args.checkpoint_freq),
            storage_dir=out_dir,
            run_name=run_name,
            restore_checkpoint=args.restore_checkpoint,
        )
    finally:
        ray.shutdown()

    ckpt_path = checkpoint.path if hasattr(checkpoint, "path") else str(checkpoint)
    print("=" * 88)
    print("Training Finished")
    print(f"best_reward:      {best_reward:.4f}")
    print(f"checkpoint:       {ckpt_path}")
    print("=" * 88)


if __name__ == "__main__":
    main()
