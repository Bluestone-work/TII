#!/usr/bin/env python3
import argparse
from pathlib import Path
from types import SimpleNamespace

from gnn_bc_tools.collect_orca_dwa_bc import collect_dataset
from gnn_bc_tools.path_utils import ensure_runtime_modules
from gnn_bc_tools.pretrain_mappo_bc import pretrain_bc
from gnn_bc_tools.rl_finetune_from_bc import run_rl_finetune


def _collector_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        episodes=args.episodes,
        seed=args.seed,
        dataset_name=args.dataset_name,
        output_dir=args.dataset_output_dir,
        num_agents=args.num_agents,
        env_stage=args.env_stage,
        map_number=args.map_number,
        max_episode_steps=args.max_episode_steps,
        communication_range=args.communication_range,
        comm_mode=args.comm_mode,
        comm_dropout_prob=args.comm_dropout_prob,
        comm_latency_steps=args.comm_latency_steps,
        comm_jitter_steps=args.comm_jitter_steps,
        comm_noise_std=args.comm_noise_std,
        num_dynamic_obstacles=args.num_dynamic_obstacles,
        obs_speed_scale=args.obs_speed_scale,
        enable_visualization=args.enable_visualization_collect,
        disable_visualization=(not args.enable_visualization_collect),
        tracking_viz_interval=args.tracking_viz_interval,
        env_log_level=args.env_log_level,
        sim_wait_wall_timeout=args.sim_wait_wall_timeout,
        auto_reset_agents=args.auto_reset_agents_collect,
        robot_radius=args.robot_radius,
        time_horizon=args.time_horizon,
        laser_obstacle_max_dist=args.laser_obstacle_max_dist,
        velocity_smoothing_alpha=args.velocity_smoothing_alpha,
        neighbor_soft_dist=args.neighbor_soft_dist,
        neighbor_stop_dist=args.neighbor_stop_dist,
        neighbor_hard_stop_dist=args.neighbor_hard_stop_dist,
        teacher=args.teacher,
        orca_blend_max=args.orca_blend_max,
        dwa_heading_weight=args.dwa_heading_weight,
        dwa_dist_weight=args.dwa_dist_weight,
        dwa_velocity_weight=args.dwa_velocity_weight,
        dwa_safety_margin=args.dwa_safety_margin,
        apf_attract_gain=args.apf_attract_gain,
        apf_obstacle_gain=args.apf_obstacle_gain,
        apf_robot_gain=args.apf_robot_gain,
        apf_tangent_gain=args.apf_tangent_gain,
        apf_damping_gain=args.apf_damping_gain,
        apf_influence_radius=args.apf_influence_radius,
        apf_robot_influence_radius=args.apf_robot_influence_radius,
        apf_goal_slow_radius=args.apf_goal_slow_radius,
        apf_obstacle_top_k=args.apf_obstacle_top_k,
    )


def _bc_args(args, dataset_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_path=str(dataset_path),
        output_dir=args.bc_output_dir,
        epochs=args.bc_epochs,
        batch_sequences=args.bc_batch_sequences,
        chunk_len=args.bc_chunk_len,
        lr=args.bc_lr,
        weight_decay=args.bc_weight_decay,
        val_ratio=args.bc_val_ratio,
        hidden_dim=args.hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        seed=args.seed if args.seed is not None else 42,
        device=args.bc_device,
    )


def _rl_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        env_stage=args.env_stage,
        map_number=args.map_number,
        max_episode_steps=args.max_episode_steps,
        num_agents=args.num_agents,
        communication_range=args.communication_range,
        comm_mode=args.comm_mode,
        comm_dropout_prob=args.comm_dropout_prob,
        comm_latency_steps=args.comm_latency_steps,
        comm_jitter_steps=args.comm_jitter_steps,
        comm_noise_std=args.comm_noise_std,
        num_dynamic_obstacles=args.num_dynamic_obstacles,
        obs_speed_scale=args.obs_speed_scale,
        enable_visualization=args.enable_visualization_rl,
        tracking_viz_interval=args.tracking_viz_interval,
        env_log_level=args.env_log_level,
        sim_wait_wall_timeout=args.sim_wait_wall_timeout,
        train_steps=args.rl_train_steps,
        num_workers=args.rl_num_workers,
        lr=args.rl_lr,
        train_batch_size=args.rl_train_batch_size,
        sample_timeout_s=args.rl_sample_timeout_s,
        rollout_fragment_length=args.rl_rollout_fragment_length,
        batch_mode=args.rl_batch_mode,
        entropy_coeff=args.rl_entropy_coeff,
        minibatch_size=args.rl_minibatch_size,
        num_epochs=args.rl_num_epochs,
        checkpoint_freq=args.rl_checkpoint_freq,
        max_seq_len=args.rl_max_seq_len,
        hidden_dim=args.hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        rl_results_dir=args.rl_results_dir,
        rl_run_name=args.rl_run_name,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ORCA/DWA BC pipeline: collect -> pretrain -> RL finetune")

    # shared env params
    p.add_argument("--num_agents", type=int, default=3)
    p.add_argument("--env_stage", type=int, default=1, choices=[1, 2, 3, 4])
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
    p.add_argument("--tracking_viz_interval", type=int, default=6)
    p.add_argument("--env_log_level", type=str, default="WARNING")
    p.add_argument("--sim_wait_wall_timeout", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)

    # collector
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--episodes", type=int, default=80)
    p.add_argument("--dataset_name", type=str, default=None)
    p.add_argument("--dataset_output_dir", type=str, default="~/work/multi-robot-exploration-rl/bc_datasets")
    p.add_argument("--enable_visualization_collect", action="store_true")
    p.add_argument("--auto_reset_agents_collect", action="store_true", default=False)
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

    # BC pretrain
    p.add_argument("--skip_bc", action="store_true")
    p.add_argument("--bc_weights_path", type=str, default=None)
    p.add_argument("--bc_output_dir", type=str, default="~/work/multi-robot-exploration-rl/bc_models")
    p.add_argument("--bc_epochs", type=int, default=30)
    p.add_argument("--bc_batch_sequences", type=int, default=32)
    p.add_argument("--bc_chunk_len", type=int, default=20)
    p.add_argument("--bc_lr", type=float, default=3e-4)
    p.add_argument("--bc_weight_decay", type=float, default=1e-5)
    p.add_argument("--bc_val_ratio", type=float, default=0.05)
    p.add_argument("--bc_device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # model dims (shared by BC + RL)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--lstm_hidden_dim", type=int, default=256)

    # RL
    p.add_argument("--skip_rl", action="store_true")
    p.add_argument("--enable_visualization_rl", action="store_true")
    p.add_argument("--rl_results_dir", type=str, default="~/work/multi-robot-exploration-rl/ray_results")
    p.add_argument("--rl_run_name", type=str, default=None)
    p.add_argument("--rl_train_steps", type=int, default=300000)
    p.add_argument("--rl_num_workers", type=int, default=2)
    p.add_argument("--rl_lr", type=float, default=3e-4)
    p.add_argument("--rl_train_batch_size", type=int, default=4000)
    p.add_argument("--rl_sample_timeout_s", type=int, default=1200)
    p.add_argument("--rl_rollout_fragment_length", type=int, default=200)
    p.add_argument("--rl_batch_mode", type=str, default="truncate_episodes", choices=["truncate_episodes", "complete_episodes"])
    p.add_argument("--rl_entropy_coeff", type=float, default=0.01)
    p.add_argument("--rl_minibatch_size", type=int, default=256)
    p.add_argument("--rl_num_epochs", type=int, default=10)
    p.add_argument("--rl_checkpoint_freq", type=int, default=20)
    p.add_argument("--rl_max_seq_len", type=int, default=20)

    return p


def main() -> None:
    ensure_runtime_modules(
        required_modules=["numpy", "gymnasium", "torch", "ray"],
        runner_module="gnn_bc_tools.run_orca_dwa_bc_pipeline",
    )

    args = build_arg_parser().parse_args()

    dataset_path = Path(args.dataset_path).expanduser().resolve() if args.dataset_path else None
    if not args.skip_collect:
        dataset_path = collect_dataset(_collector_args(args))
    elif dataset_path is None:
        raise ValueError("skip_collect=True 时必须提供 --dataset_path")

    bc_weights_path = Path(args.bc_weights_path).expanduser().resolve() if args.bc_weights_path else None
    if not args.skip_bc:
        bc_weights_path = pretrain_bc(_bc_args(args, dataset_path))
    elif bc_weights_path is None:
        raise ValueError("skip_bc=True 时必须提供 --bc_weights_path")

    if not args.skip_rl:
        run_rl_finetune(_rl_args(args), bc_weights_path)


if __name__ == "__main__":
    main()
