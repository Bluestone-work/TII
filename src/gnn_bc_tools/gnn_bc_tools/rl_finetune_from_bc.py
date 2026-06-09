import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from gnn_bc_tools.path_utils import inject_workspace_paths


ENV_CURRICULUM = {
    1: {"map_number": 3, "max_episode_steps": 2000, "num_obstacles": 0, "obs_speed_scale": 0.0},
    2: {"map_number": 3, "max_episode_steps": 2000, "num_obstacles": 0, "obs_speed_scale": 0.0},
    3: {"map_number": 3, "max_episode_steps": 2500, "num_obstacles": 4, "obs_speed_scale": 0.5},
    4: {"map_number": 3, "max_episode_steps": 3000, "num_obstacles": 8, "obs_speed_scale": 1.0},
}


def _extract_reward(result: Dict[str, Any]) -> float:
    for val in [
        result.get("episode_reward_mean"),
        result.get("env_runners", {}).get("episode_reward_mean"),
        result.get("sampler_results", {}).get("episode_reward_mean"),
    ]:
        if val is not None:
            return float(val)
    return float("nan")


def _extract_timesteps(result: Dict[str, Any]) -> int:
    for val in [
        result.get("timesteps_total"),
        result.get("num_env_steps_sampled_lifetime"),
        result.get("info", {}).get("num_env_steps_sampled"),
    ]:
        if val is not None:
            return int(val)
    return 0


def build_rl_env_config(args) -> Dict[str, Any]:
    stage_cfg = dict(ENV_CURRICULUM[int(args.env_stage)])
    if args.map_number is not None:
        stage_cfg["map_number"] = int(args.map_number)

    return {
        "num_agents": int(args.num_agents),
        "map_number": int(stage_cfg["map_number"]),
        "max_episode_steps": int(args.max_episode_steps or stage_cfg["max_episode_steps"]),
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
        "auto_reset_agents": True,
        "num_dynamic_obstacles": int(args.num_dynamic_obstacles) if args.num_dynamic_obstacles is not None else int(stage_cfg["num_obstacles"]),
        "obs_speed": 0.3 * float(args.obs_speed_scale) if args.obs_speed_scale is not None else 0.3 * float(stage_cfg["obs_speed_scale"]),
    }


def run_rl_finetune(args, bc_weights_path: Path) -> Path:
    repo_root, workspace_pythonpath, workspace_ament = inject_workspace_paths()

    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env
    from ray.rllib.models import ModelCatalog

    from gnn_marl_training.gnn_marl_env import env_creator
    from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel, MODEL_NAME_MLP

    env_cfg = build_rl_env_config(args)
    max_neighbors = min(int(args.num_agents) - 1, 5)

    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        address="local",
        include_dashboard=False,
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": workspace_pythonpath,
                "AMENT_PREFIX_PATH": workspace_ament,
            }
        },
    )

    model_cfg = {
        "custom_model": MODEL_NAME_MLP,
        "custom_model_config": {
            "num_agents": int(args.num_agents),
            "max_neighbors": int(max_neighbors),
            "neighbor_feature_dim": 5,
            "use_neighbor_obs": True,
            "hidden_dim": int(args.hidden_dim),
            "lstm_hidden_dim": int(args.lstm_hidden_dim),
        },
        "max_seq_len": int(args.max_seq_len),
    }

    policy_name = "shared_policy"
    config = (
        PPOConfig()
        .environment(env="gnn_marl", env_config=env_cfg, disable_env_checking=True)
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
            entropy_coeff=float(args.entropy_coeff),
            vf_clip_param=50.0,
            grad_clip=0.5,
            grad_clip_by="global_norm",
            minibatch_size=int(args.minibatch_size),
            num_epochs=int(args.num_epochs),
            model=model_cfg,
        )
        .multi_agent(
            policies={policy_name: (None, None, None, {})},
            policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
            policies_to_train=[policy_name],
        )
        .resources(num_gpus=int(torch.cuda.is_available()))
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    algo = config.build()

    ckpt_obj = torch.load(bc_weights_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        state_dict = ckpt_obj["model_state_dict"]
    else:
        state_dict = ckpt_obj

    policy = algo.get_policy(policy_name)
    model = policy.model
    model.load_state_dict(state_dict, strict=False)
    print(f"✅ BC 权重已注入 policy model: {bc_weights_path}")

    results_root = Path(args.rl_results_dir).expanduser().resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_name = args.rl_run_name or f"MAPPO_MLP_BC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = results_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_reward = -float("inf")
    best_ckpt = None
    last_ckpt = None
    start_time = time.time()

    print("=" * 80)
    print("RL 微调（BC warm start）")
    print(f"repo:          {repo_root}")
    print(f"run_dir:       {run_dir}")
    print(f"train_steps:   {args.train_steps}")
    print(f"num_workers:   {args.num_workers}")
    print(f"lr:            {args.lr}")
    print("=" * 80)

    iteration = 0
    while True:
        iteration += 1
        result = algo.train()
        done_steps = _extract_timesteps(result)
        reward = _extract_reward(result)

        if reward > best_reward:
            best_reward = reward
            best_ckpt = algo.save(str(run_dir / "best"))

        if int(args.checkpoint_freq) > 0 and iteration % int(args.checkpoint_freq) == 0:
            last_ckpt = algo.save(str(run_dir))

        pct = min(100.0, done_steps / max(int(args.train_steps), 1) * 100.0)
        elapsed = time.time() - start_time
        eta_str = ""
        if done_steps > 0:
            eta_s = elapsed / done_steps * max(int(args.train_steps) - done_steps, 0)
            eta_str = f" ETA {int(eta_s//3600):02d}:{int(eta_s%3600//60):02d}:{int(eta_s%60):02d}"
        print(f"[rl] it={iteration:04d} steps={done_steps:>8,}/{int(args.train_steps):,} ({pct:5.1f}%) reward={reward:8.3f}{eta_str}")

        if done_steps >= int(args.train_steps):
            break

    last_ckpt = algo.save(str(run_dir))
    algo.stop()
    ray.shutdown()

    final_ckpt = best_ckpt or last_ckpt
    print("=" * 80)
    print("RL 微调完成")
    print(f"best_reward:   {best_reward:.4f}")
    print(f"checkpoint:    {final_ckpt}")
    print("=" * 80)

    return Path(str(final_ckpt))
