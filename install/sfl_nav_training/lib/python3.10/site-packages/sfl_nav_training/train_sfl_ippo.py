#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

try:
    import ray
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: ray. Please activate the training environment with Ray/RLlib."
    ) from exc


def _inject_workspace_paths() -> Path:
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[3] if len(this_file.parents) >= 4 else this_file.parent
    candidates = [
        repo_root / "src" / "sfl_nav_training",
        repo_root / "src" / "intent_marl_training_e2e",
        repo_root / "build" / "sfl_nav_training",
        repo_root / "build" / "intent_marl_training_e2e",
        repo_root / "build" / "sfl_nav_training" / "build" / "lib",
        repo_root / "build" / "intent_marl_training_e2e" / "build" / "lib",
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
    from sfl_nav_training.sfl_nav_env import sfl_nav_env_creator
    from sfl_nav_training.sfl_ippo_model import MODEL_NAME_SFL_IPPO, SFLIPPOGRUModel
except ModuleNotFoundError:
    from sfl_nav_env import sfl_nav_env_creator  # type: ignore
    from sfl_ippo_model import MODEL_NAME_SFL_IPPO, SFLIPPOGRUModel  # type: ignore


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
        "max_episode_steps": 500,
        "num_obstacles": 0,
        "use_random_mode": True,
    },
    6: {
        "name": "Stage 6 · two-robot head-on",
        "map_number": 6,
        "max_episode_steps": 500,
        "num_obstacles": 0,
        "use_random_mode": False,
        "fallback_pose_bank": MAP6_HEAD_ON_POSE_BANK,
    },
    7: {
        "name": "Stage 7 · two-robot crossing",
        "map_number": 6,
        "max_episode_steps": 500,
        "num_obstacles": 0,
        "use_random_mode": False,
        "fallback_pose_bank": MAP6_CROSS_POSE_BANK,
    },
}


class SFLMetricsCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        episode.user_data["GoalR_sum"] = 0.0
        episode.user_data["MapC_sum"] = 0.0
        episode.user_data["AgentC_sum"] = 0.0
        episode.user_data["TimeO_sum"] = 0.0
        episode.user_data["SFLSparse_sum"] = 0.0
        episode.user_data["SFLDense_sum"] = 0.0
        episode.user_data["SFLLidar_sum"] = 0.0
        episode.user_data["count"] = 0
        episode.user_data["episode_summary"] = {}

    def on_episode_step(self, *, worker, base_env, episode, env_index, **kwargs):
        agent_ids = []
        get_agents = getattr(episode, "get_agents", None)
        if callable(get_agents):
            try:
                agent_ids = list(get_agents())
            except Exception:
                agent_ids = []
        if not agent_ids:
            last_infos = getattr(episode, "_agent_to_last_info", None)
            if not isinstance(last_infos, dict):
                last_infos = getattr(episode, "_last_infos", None)
            if isinstance(last_infos, dict):
                agent_ids = list(last_infos.keys())

        for agent_id in agent_ids:
            info = episode.last_info_for(agent_id)
            if not isinstance(info, dict):
                continue

            summary = info.get("episode_summary")
            if isinstance(summary, dict):
                episode.user_data["episode_summary"] = dict(summary)

            episode.user_data["GoalR_sum"] += float(info.get("GoalR", 0.0))
            episode.user_data["MapC_sum"] += float(info.get("MapC", 0.0))
            episode.user_data["AgentC_sum"] += float(info.get("AgentC", 0.0))
            episode.user_data["TimeO_sum"] += float(info.get("TimeO", 0.0))
            episode.user_data["SFLSparse_sum"] += float(info.get("SFLSparse", 0.0))
            episode.user_data["SFLDense_sum"] += float(info.get("SFLDense", 0.0))
            episode.user_data["SFLLidar_sum"] += float(info.get("SFLLidar", 0.0))
            episode.user_data["count"] += 1

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        n = max(1, int(episode.user_data.get("count", 0)))
        goals = float(episode.user_data.get("GoalR_sum", 0.0))
        mapc = float(episode.user_data.get("MapC_sum", 0.0))
        agentc = float(episode.user_data.get("AgentC_sum", 0.0))
        timeo = float(episode.user_data.get("TimeO_sum", 0.0))

        summary = episode.user_data.get("episode_summary")
        if not isinstance(summary, dict):
            summary = {}
        reason = str(summary.get("reason", ""))

        episode.custom_metrics["goal_reached_count"] = goals
        episode.custom_metrics["success_rate"] = float(
            reason == "all_done" and (mapc + agentc) <= 0.0 and goals > 0.0
        ) if reason else float(goals > 0.0 and (mapc + agentc) <= 0.0 and timeo <= 0.0)
        episode.custom_metrics["collision_rate"] = float(
            reason == "collision_event" or (mapc + agentc) > 0.0
        )
        episode.custom_metrics["timeout_rate"] = float(reason == "timeout") if reason else float(timeo > 0.0)
        episode.custom_metrics["mean_sparse_reward"] = float(episode.user_data.get("SFLSparse_sum", 0.0) / n)
        episode.custom_metrics["mean_dense_reward"] = float(episode.user_data.get("SFLDense_sum", 0.0) / n)
        episode.custom_metrics["mean_lidar_penalty"] = float(episode.user_data.get("SFLLidar_sum", 0.0) / n)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _fmt_metric(value: float, precision: int = 3) -> str:
    v = _safe_float(value)
    if v is None:
        return "nan"
    return f"{v:.{precision}f}"


def _extract_steps(result: Dict) -> int:
    for key in ("timesteps_total", "num_env_steps_sampled_lifetime", "num_env_steps_sampled"):
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


def _extract_episode_len(result: Dict) -> float:
    for value in (
        result.get("episode_len_mean"),
        result.get("env_runners", {}).get("episode_len_mean"),
        result.get("sampler_results", {}).get("episode_len_mean"),
    ):
        v = _safe_float(value)
        if v is not None:
            return v
    return float("nan")


def _extract_custom_metric(result: Dict, metric_name: str) -> float:
    for metrics in (
        result.get("custom_metrics"),
        result.get("env_runners", {}).get("custom_metrics"),
        result.get("sampler_results", {}).get("custom_metrics"),
    ):
        if not isinstance(metrics, dict):
            continue
        for key in (f"{metric_name}_mean", metric_name):
            v = _safe_float(metrics.get(key))
            if v is not None:
                return v
    return float("nan")


def _extract_learner_stat(result: Dict, stat_key: str) -> float:
    learner = result.get("info", {}).get("learner")
    if isinstance(learner, dict):
        values = []
        for policy_stats in learner.values():
            if not isinstance(policy_stats, dict):
                continue
            learner_stats = policy_stats.get("learner_stats")
            if isinstance(learner_stats, dict):
                v = _safe_float(learner_stats.get(stat_key))
                if v is not None:
                    values.append(v)
        if values:
            return float(sum(values) / len(values))
    return float("nan")


def _build_base_env_config(args, stage_cfg: Dict) -> Dict:
    return {
        "num_agents": int(args.num_agents),
        "model_num_agents": int(args.num_agents),
        "map_number": int(stage_cfg["map_number"]),
        "use_random_mode": bool(stage_cfg.get("use_random_mode", True)),
        "fallback_pose_bank": stage_cfg.get("fallback_pose_bank"),
        "max_episode_steps": int(stage_cfg["max_episode_steps"]),
        "num_dynamic_obstacles": int(stage_cfg.get("num_obstacles", 0)),
        "obs_speed": 0.0,
        "communication_range": float(args.communication_range),
        "comm_mode": str(args.comm_mode),
        "collision_ends_episode": True,
        "end_episode_on_collision_event": bool(args.end_episode_on_collision_event),
        "reset_on_collision_event": False,
        "enable_visualization": bool(args.enable_visualization),
        "tracking_viz_interval": int(args.tracking_viz_interval),
        "env_log_level": str(args.env_log_level),
        "sim_wait_wall_timeout": 2.5,
        "end_to_end_rl": True,

        # Base env reward is ignored by wrapper, keep it neutral.
        "progress_reward_scale": 0.0,
        "path_progress_reward_scale": 0.0,
        "goal_reward": 0.0,
        "collision_penalty": 0.0,
        "time_penalty": 0.0,
        "idle_penalty_scale": 0.0,
        "near_collision_penalty_scale": 0.0,
        "front_safety_penalty_scale": 0.0,
        "neighbor_safety_penalty_scale": 0.0,
    }


def _build_reward_config(args) -> Dict:
    return {
        "rew_lambda": float(args.rew_lambda),
        "goal_rew": float(args.goal_rew),
        "dt_rew": float(args.dt_rew),
        "coll_rew": float(args.coll_rew),
        "lidar_thresh": float(args.lidar_thresh),
        "lidar_rew": float(args.lidar_rew),
        "agent_collision_dist": float(args.agent_collision_dist),
    }


def _build_obs_config(args) -> Dict:
    return {
        "lidar_num_beams": int(args.lidar_num_beams),
        "lidar_max_range": float(args.lidar_max_range),
        "lidar_min_range": float(args.lidar_min_range),
    }


def _parse_curriculum_stages(args) -> List[int]:
    raw = str(args.curriculum_stages or "").strip()
    if not raw:
        stages = [int(args.env_stage)]
    else:
        stages = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            stage_id = int(token)
            if stage_id not in ENV_CURRICULUM:
                raise SystemExit(f"Unsupported curriculum stage {stage_id}. Valid: {sorted(ENV_CURRICULUM.keys())}")
            stages.append(stage_id)
        if not stages:
            raise SystemExit("--curriculum_stages is empty after parsing")

    cycles = max(1, int(args.curriculum_cycles))
    return stages * cycles


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
        episode_reward_mean = _extract_reward(result)
        episode_len_mean = _extract_episode_len(result)
        success_rate = _extract_custom_metric(result, "success_rate")
        collision_rate = _extract_custom_metric(result, "collision_rate")
        timeout_rate = _extract_custom_metric(result, "timeout_rate")
        goal_reached_count = _extract_custom_metric(result, "goal_reached_count")
        policy_entropy = _extract_learner_stat(result, "entropy")
        vf_loss = _extract_learner_stat(result, "vf_loss")
        mean_sparse_reward = _extract_custom_metric(result, "mean_sparse_reward")
        mean_dense_reward = _extract_custom_metric(result, "mean_dense_reward")
        pct = min(100.0, (steps / max(train_steps, 1)) * 100.0)
        elapsed = time.time() - start
        print(
            f"[iter {iteration:04d}] steps={steps:>9,}/{train_steps:,} ({pct:5.1f}%) "
            f"episode_reward_mean={_fmt_metric(episode_reward_mean, 3)} "
            f"episode_len_mean={_fmt_metric(episode_len_mean, 1)} "
            f"success_rate={_fmt_metric(success_rate, 3)} "
            f"collision_rate={_fmt_metric(collision_rate, 3)} "
            f"timeout_rate={_fmt_metric(timeout_rate, 3)} "
            f"goal_reached_count={_fmt_metric(goal_reached_count, 2)} "
            f"mean_sparse_reward={_fmt_metric(mean_sparse_reward, 4)} "
            f"mean_dense_reward={_fmt_metric(mean_dense_reward, 4)} "
            f"policy_entropy={_fmt_metric(policy_entropy, 4)} "
            f"vf_loss={_fmt_metric(vf_loss, 4)} "
            f"elapsed={elapsed/60.0:7.1f}m",
            flush=True,
        )

        if episode_reward_mean > best_reward:
            best_reward = episode_reward_mean
            best_ckpt = algo.save(os.path.join(run_dir, "best"))

        if checkpoint_freq > 0 and iteration % checkpoint_freq == 0:
            last_ckpt = algo.save(run_dir)

        if steps >= train_steps:
            break

    last_ckpt = algo.save(run_dir)
    algo.stop()
    return best_ckpt or last_ckpt, best_reward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train sampling-style IPPO over Gazebo multi-robot nav")

    parser.add_argument("--env_stage", type=int, default=6, choices=sorted(ENV_CURRICULUM.keys()))
    parser.add_argument("--curriculum_stages", type=str, default="")
    parser.add_argument("--curriculum_cycles", type=int, default=1)
    parser.add_argument("--train_steps_per_stage", type=int, default=None)

    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--communication_range", type=float, default=3.5)
    parser.add_argument("--comm_mode", type=str, default="centralized_oracle", choices=["decentralized", "centralized_oracle"])
    parser.add_argument("--end_episode_on_collision_event", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=200000)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--rollout_fragment_length", type=int, default=256)
    parser.add_argument("--sample_timeout_s", type=int, default=1200)

    parser.add_argument("--fc_dim", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--use_layer_norm", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lidar_num_beams", type=int, default=200)
    parser.add_argument("--lidar_max_range", type=float, default=6.0)
    parser.add_argument("--lidar_min_range", type=float, default=0.0)

    parser.add_argument("--rew_lambda", type=float, default=0.5)
    parser.add_argument("--goal_rew", type=float, default=4.0)
    parser.add_argument("--dt_rew", type=float, default=-0.01)
    parser.add_argument("--coll_rew", type=float, default=-4.0)
    parser.add_argument("--lidar_thresh", type=float, default=0.1)
    parser.add_argument("--lidar_rew", type=float, default=-0.1)
    parser.add_argument("--agent_collision_dist", type=float, default=0.6)

    parser.add_argument("--enable_visualization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tracking_viz_interval", type=int, default=2)
    parser.add_argument("--env_log_level", type=str, default="INFO")

    parser.add_argument("--output_dir", type=str, default="~/work/multi-robot-exploration-rl/ray_results")
    parser.add_argument("--run_name", type=str, default="sfl_ippo_nav")
    parser.add_argument("--restore_checkpoint", type=str, default=None)

    return parser


def main():
    args = build_parser().parse_args()

    stage_sequence = _parse_curriculum_stages(args)
    if any(stage_id in {6, 7} for stage_id in stage_sequence) and int(args.num_agents) != 2:
        raise SystemExit("For stage 6/7 this setup assumes --num_agents 2.")

    steps_per_stage = int(args.train_steps_per_stage) if args.train_steps_per_stage is not None else int(args.train_steps)
    if steps_per_stage <= 0:
        raise SystemExit("train steps per stage must be > 0")

    ppo_minibatch_size = max(128, int(args.train_batch_size) // 4)

    register_env("sfl_nav_marl", sfl_nav_env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_SFL_IPPO, SFLIPPOGRUModel)

    policy_name = "shared_policy"
    model_cfg = {
        "custom_model": MODEL_NAME_SFL_IPPO,
        "custom_model_config": {
            "fc_dim": int(args.fc_dim),
            "hidden_size": int(args.hidden_size),
            "use_layer_norm": bool(args.use_layer_norm),
        },
        "max_seq_len": int(args.max_seq_len),
    }

    out_dir = os.path.expanduser(args.output_dir)
    use_gpu = int(__import__("torch").cuda.is_available())
    restore_checkpoint = args.restore_checkpoint
    stage_results = []

    ray.init(ignore_reinit_error=True)
    try:
        for idx, stage_id in enumerate(stage_sequence, start=1):
            stage_cfg = dict(ENV_CURRICULUM[int(stage_id)])

            env_config = {
                "base_env_config": _build_base_env_config(args, stage_cfg),
                "reward_config": _build_reward_config(args),
                "obs_config": _build_obs_config(args),
            }

            if len(stage_sequence) == 1:
                run_name = f"{args.run_name}_stage{stage_id}_n{args.num_agents}"
            else:
                run_name = f"{args.run_name}_phase{idx:02d}_stage{stage_id}_n{args.num_agents}"

            print("=" * 88)
            print("Sampling-Style IPPO Training (Gazebo + RViz)")
            print(f"repo:             {REPO_ROOT}")
            print(f"phase:            {idx}/{len(stage_sequence)}")
            print(f"stage:            {stage_id} ({stage_cfg['name']})")
            print(f"map:              {stage_cfg['map_number']}")
            print(f"num_agents:       {args.num_agents}")
            print(f"obs:              lidar_beams={args.lidar_num_beams} + 5")
            print("action:           continuous Box(2)")
            print(
                f"reward:           rew_lambda={args.rew_lambda:.2f} goal={args.goal_rew:.2f} "
                f"dt={args.dt_rew:.3f} coll={args.coll_rew:.2f} lidar={args.lidar_rew:.3f}@<{args.lidar_thresh:.2f}m"
            )
            print(
                f"ppo:              lr={args.lr:.2e} anneal_lr={bool(args.anneal_lr)} clip=0.04 ent=0.00 "
                f"vf_coef=0.50 epochs=4 minibatch={ppo_minibatch_size}"
            )
            print(f"visualization:    {'ON' if args.enable_visualization else 'OFF'}")
            print(f"train_steps:      {steps_per_stage:,} per stage")
            print("=" * 88)

            stage_lr_schedule = None
            if bool(args.anneal_lr):
                stage_lr_schedule = [[0, float(args.lr)], [int(max(1, steps_per_stage)), 0.0]]

            config = (
                PPOConfig()
                .environment(env="sfl_nav_marl", env_config=env_config, disable_env_checking=True)
                .framework("torch")
                .env_runners(
                    num_env_runners=int(args.num_workers),
                    num_envs_per_env_runner=1,
                    sample_timeout_s=int(args.sample_timeout_s),
                    batch_mode="truncate_episodes",
                    rollout_fragment_length=int(args.rollout_fragment_length),
                )
                .training(
                    lr=float(args.lr),
                    gamma=0.99,
                    lambda_=0.95,
                    train_batch_size=int(args.train_batch_size),
                    clip_param=0.04,
                    entropy_coeff=0.0,
                    vf_loss_coeff=0.5,
                    vf_clip_param=20.0,
                    lr_schedule=stage_lr_schedule,
                    grad_clip=0.5,
                    grad_clip_by="global_norm",
                    minibatch_size=int(ppo_minibatch_size),
                    num_epochs=4,
                    model=model_cfg,
                )
                .multi_agent(
                    policies={policy_name: (None, None, None, {})},
                    policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kwargs: policy_name,
                    policies_to_train=[policy_name],
                )
                .callbacks(SFLMetricsCallbacks)
                .resources(num_gpus=use_gpu)
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
            )

            checkpoint, best_reward = _run_training(
                config=config,
                train_steps=steps_per_stage,
                checkpoint_freq=int(args.checkpoint_freq),
                storage_dir=out_dir,
                run_name=run_name,
                restore_checkpoint=restore_checkpoint,
            )
            ckpt_path = checkpoint.path if hasattr(checkpoint, "path") else str(checkpoint)
            restore_checkpoint = ckpt_path
            stage_results.append((idx, stage_id, stage_cfg["name"], float(best_reward), ckpt_path))
    finally:
        ray.shutdown()

    final_ckpt = stage_results[-1][4] if stage_results else ""
    print("=" * 88)
    print("Training Finished")
    for idx, stage_id, stage_name, best_reward, ckpt_path in stage_results:
        print(
            f"phase {idx:02d} stage {stage_id} ({stage_name}) "
            f"best_reward={best_reward:.4f} checkpoint={ckpt_path}"
        )
    print(f"final_checkpoint: {final_ckpt}")
    print("=" * 88)


if __name__ == "__main__":
    main()
