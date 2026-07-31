#!/usr/bin/env python3
"""Profile one RLlib policy checkpoint without starting ROS or Gazebo."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import resource
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy

from gnn_marl_training.counterfactual_ppo_policy import register_counterfactual_policy
from gnn_marl_training.gat_rllib_model import GATRLlibModel, MODEL_NAME as MODEL_NAME_GAT
from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel, MODEL_NAME_MLP


warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray")
warnings.filterwarnings("ignore", category=UserWarning, module="ray")


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def checkpoint_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.uname().processor or platform.processor() or "CPU"


def unique_parameter_count(modules: Iterable[torch.nn.Module]) -> int:
    parameters = {
        id(parameter): parameter
        for module in modules if module is not None
        for parameter in module.parameters()
    }
    return sum(parameter.numel() for parameter in parameters.values())


def effective_actor_parameter_count(model: torch.nn.Module) -> int:
    """Count parameters that can influence the deployed actor output."""
    if not isinstance(model, GATRLlibModel):
        return sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "critic" not in name.lower() and "value" not in name.lower()
        )
    modules = [
        model.actor_input_norm,
        model.local_map_cnn,
        model.lstm_cell,
        model.lstm_norm,
        model.policy_fusion_norm,
        model.actor_head,
    ]
    if abs(model.graph_residual_scale) > 1e-12:
        modules.extend([
            model.local_input_norm,
            model.social_ego_encoder,
            model.graph_gate,
            model.graph_delta,
        ])
        if model.use_social_graph_tokens:
            if model.actor_graph_mode == "neighbor":
                modules.extend([model.neighbor_input_norm, model.neighbor_token_encoder])
            else:
                modules.extend([model.social_token_input_norm, model.social_token_encoder])
        if model.use_obstacle_graph_tokens:
            modules.extend([model.obstacle_token_input_norm, model.obstacle_token_encoder])
        if (
            model.graph_architecture == "single"
            and model.use_social_graph_tokens
            and model.use_obstacle_graph_tokens
        ):
            modules.extend([model.unified_gat, model.unified_proj])
        else:
            if model.use_social_graph_tokens:
                modules.extend([model.social_gat, model.social_proj])
            if model.use_obstacle_graph_tokens:
                modules.extend([model.obstacle_gat, model.obstacle_proj])
            if model.use_social_graph_tokens and model.use_obstacle_graph_tokens:
                modules.append(model.dual_graph_fusion)
    return unique_parameter_count(modules)


def disable_training_only_critic(model: torch.nn.Module) -> None:
    """Keep Policy.compute_single_action while excluding the centralized critic."""
    if hasattr(model, "_critic_from_global_obs"):
        model._critic_from_global_obs = (  # type: ignore[method-assign]
            lambda global_obs, node_mask=None, use_reference=False:
            global_obs.new_zeros(global_obs.shape[0])
        )
    if hasattr(model, "_critic_from_local_obs"):
        model._critic_from_local_obs = (  # type: ignore[method-assign]
            lambda actor_obs: actor_obs.new_zeros(actor_obs.shape[0])
        )


def team_actor_cycle(policy: Policy, observation: np.ndarray, states: list[list], team_size: int) -> None:
    for index in range(team_size):
        _action, states[index], _info = policy.compute_single_action(
            observation,
            state=states[index],
            explore=False,
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_config(checkpoint: Path) -> dict:
    with (checkpoint / "algorithm_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    return dict(state.get("config", {}))


def load_inference_policy(policy_path: Path) -> Policy:
    """Restore policy state without training-only optimizer slots."""
    with (policy_path / "policy_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    state = dict(state)
    state["_optimizer_variables"] = []
    return Policy.from_state(state)


def supported_operator_flops(policy: Policy, observation: np.ndarray, state: list) -> int:
    device = next(policy.model.parameters()).device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    synchronize(device)
    with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
        policy.compute_single_action(observation, state=state, explore=False)
        synchronize(device)
    return int(sum(int(event.flops or 0) for event in profiler.key_averages()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device-label", required=True, choices=("cpu", "gpu"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--team-size", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise SystemExit("P0 P1 latency protocol fixes actor micro-batch size to 1")
    torch.set_num_threads(max(1, args.torch_threads))

    checkpoint = args.checkpoint.resolve()
    policy_path = checkpoint / "policies" / "shared_policy"
    if not (checkpoint / "algorithm_state.pkl").is_file() or not policy_path.is_dir():
        raise SystemExit(f"invalid checkpoint: {checkpoint}")

    register_counterfactual_policy()
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)
    config = load_config(checkpoint)
    policy = load_inference_policy(policy_path)
    model = policy.model
    device = next(model.parameters()).device
    if args.device_label == "cpu" and device.type != "cpu":
        raise SystemExit(f"requested CPU profile but policy loaded on {device}")
    if args.device_label == "gpu" and device.type != "cuda":
        raise SystemExit(f"requested GPU profile but policy loaded on {device}")

    named_parameters = list(model.named_parameters())
    total_parameters = sum(parameter.numel() for _name, parameter in named_parameters)
    trainable_parameters = sum(
        parameter.numel() for _name, parameter in named_parameters if parameter.requires_grad
    )
    critic_named_parameters = sum(
        parameter.numel()
        for name, parameter in named_parameters
        if "critic" in name.lower() or "value" in name.lower()
    )
    actor_named_parameters = total_parameters - critic_named_parameters
    effective_actor_parameters = effective_actor_parameter_count(model)
    active_graph_parameters = (
        int(model.active_graph_parameter_count())
        if isinstance(model, GATRLlibModel) else 0
    )
    disable_training_only_critic(model)

    observation = np.zeros(policy.observation_space.shape, dtype=np.float32)
    states = [policy.get_initial_state() for _ in range(args.team_size)]
    for _ in range(max(1, args.warmup)):
        team_actor_cycle(policy, observation, states, args.team_size)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms = []
    for _ in range(max(1, args.repeats)):
        synchronize(device)
        started = time.perf_counter_ns()
        team_actor_cycle(policy, observation, states, args.team_size)
        synchronize(device)
        timings_ms.append((time.perf_counter_ns() - started) / 1e6)

    profiled_flops = supported_operator_flops(
        policy, observation, policy.get_initial_state()
    )
    custom = dict(config.get("model", {}).get("custom_model_config", {}))
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    latency_mean = float(np.mean(timings_ms))
    result = {
        "method": args.method,
        "device_label": args.device_label,
        "actual_device": str(device),
        "hardware": gpu_name or cpu_name(),
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint_bytes(policy_path),
        "model_class": type(model).__name__,
        "observation_dim": int(np.prod(policy.observation_space.shape)),
        "action_dim": int(np.prod(policy.action_space.shape)),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "actor_parameters_by_name": int(actor_named_parameters),
        "critic_parameters_by_name": int(critic_named_parameters),
        "effective_actor_parameters": int(effective_actor_parameters),
        "active_graph_parameters": int(active_graph_parameters),
        "profile_scope": "shared_actor_only_sequential_team_cycle",
        "profiled_team_size": int(args.team_size),
        "fixed_micro_batch_size": int(args.batch_size),
        "torch_threads": int(torch.get_num_threads()),
        "latency_repeats": int(len(timings_ms)),
        "latency_ms_mean": latency_mean,
        "latency_ms_std": float(np.std(timings_ms, ddof=1)),
        "latency_ms_p50": percentile(timings_ms, 50),
        "latency_ms_p95": percentile(timings_ms, 95),
        "latency_ms_p99": percentile(timings_ms, 99),
        "latency_ms_max": float(max(timings_ms)),
        "per_robot_latency_ms_mean": latency_mean / args.team_size,
        "team_cycles_per_second": 1000.0 / latency_mean,
        "actor_calls_per_second": args.team_size * 1000.0 / latency_mean,
        "decisions_per_second": 1000.0 / latency_mean,
        "meets_10hz_at_p99": bool(percentile(timings_ms, 99) < 100.0),
        "torch_profiled_supported_operator_flops": profiled_flops,
        "flop_coverage": "partial_supported_torch_ops_only",
        "process_max_rss_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0,
        "gpu_peak_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda" else None
        ),
        "num_agents": int(custom.get("num_agents", 0) or 0),
        "graph_architecture": custom.get("graph_architecture"),
        "graph_ablation": custom.get("graph_ablation"),
        "social_token_count": int(
            custom.get("neighbor_prediction_dim", 0)
            / max(1, custom.get("neighbor_prediction_feature_dim", 6))
        ),
        "obstacle_token_count": int(
            custom.get("obstacle_motion_dim", 0)
            / max(1, custom.get("obstacle_motion_feature_dim", 7))
        ),
        "token_token_edges": bool(custom.get("enable_token_token_edges", False)),
        "counterfactual_advantage_coef": float(
            custom.get("counterfactual_advantage_coef", 0.0) or 0.0
        ),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
