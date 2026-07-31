#!/usr/bin/env python3
"""Run strict finite-horizon leave-one-agent-out branches from selected states."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env

from gnn_marl_training.counterfactual_ppo_policy import register_counterfactual_policy
from gnn_marl_training.gat_rllib_model import GATRLlibModel, MODEL_NAME as MODEL_NAME_GAT
from gnn_marl_training.gnn_marl_env import GNNMARLEnv, env_creator
from gnn_marl_training.mappo_mlp_model import MAPPOMLPModel, MODEL_NAME_MLP
from gnn_marl_training.paired_branch_snapshot import BranchSnapshotManager
from gnn_marl_training.test_gnn_mappo import _compute_counterfactual_step


def load_plan(path: Path, map_number: int, num_agents: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["map_number"]) != int(map_number):
                continue
            if num_agents is not None and int(row["num_agents"]) != int(num_agents):
                continue
            rows.append(row)
    return rows


def completed_state_ids(path: Path, expected_agents: int) -> set[str]:
    counts = Counter()
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("valid") is True:
                        counts[str(row["state_id"])] += 1
    return {state_id for state_id, count in counts.items() if count >= expected_agents}


def append_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def invalid_records(spec: dict, reason: str, checkpoint: str, horizon: int, gamma: float) -> list[dict]:
    records = []
    for agent in spec["agents"]:
        records.append({
            "schema_version": "paired_intervention_result_v1",
            "state_id": spec["state_id"],
            "num_agents": int(spec["num_agents"]),
            "agent": agent["agent"],
            "agent_index": int(agent["agent_index"]),
            "estimated_credit": None,
            "baseline_return": None,
            "intervention_return": None,
            "marginal_return": None,
            "baseline_start_fingerprint": "",
            "intervention_start_fingerprint": "",
            "horizon_steps": int(horizon),
            "gamma": float(gamma),
            "valid": False,
            "invalid_reason": reason,
            "method": spec["method"],
            "scenario_bank": spec.get("scenario_bank"),
            "scenario_case": spec.get("scenario_case"),
            "training_seed": spec.get("training_seed"),
            "eval_seed": spec.get("eval_seed"),
            "episode_seed": spec["episode_seed"],
            "episode": spec["episode"],
            "step": spec["step"],
            "map_number": spec["map_number"],
            "checkpoint_path": checkpoint,
        })
    return records


def load_policy_and_config(checkpoint: Path):
    register_counterfactual_policy()
    register_env("gnn_marl", env_creator)
    ModelCatalog.register_custom_model(MODEL_NAME_MLP, MAPPOMLPModel)
    ModelCatalog.register_custom_model(MODEL_NAME_GAT, GATRLlibModel)
    with (checkpoint / "algorithm_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    policy = Policy.from_checkpoint(str(checkpoint / "policies" / "shared_policy"))
    return policy, dict(state["config"].get("env_config", {}))


def policy_adapter(policy, env):
    target_size = int(policy.observation_space.shape[0])
    prefix_size = int(env.base_obs_dim + env.local_map_dim + env.neighbor_dim + env.reset_flag_dim)

    def adapt(observation):
        source = np.asarray(observation, dtype=np.float32).reshape(-1)
        if source.size == target_size:
            return source
        if prefix_size > source.size or prefix_size > target_size:
            raise ValueError(
                f"cannot adapt observation source={source.size} target={target_size} prefix={prefix_size}")
        result = np.zeros(target_size, dtype=np.float32)
        result[:prefix_size] = source[:prefix_size]
        copied = min(source.size - prefix_size, target_size - prefix_size)
        result[prefix_size:prefix_size + copied] = source[prefix_size:prefix_size + copied]
        return result

    return adapt


def policy_step(policy, adapt, observations: dict, states: dict) -> tuple[dict, dict]:
    actions = {}
    next_states = {aid: state for aid, state in states.items()}
    for aid, observation in observations.items():
        action, state_out, _ = policy.compute_single_action(
            obs=adapt(observation), state=states[aid], explore=False)
        actions[aid] = action
        next_states[aid] = state_out
    return actions, next_states


def rollout_branch(
    env, policy, adapt, observations: dict, states: dict,
    horizon: int, gamma: float, original_team_size: int,
) -> tuple[float, int]:
    discounted_return = 0.0
    discount = 1.0
    steps = 0
    current_obs = observations
    current_states = states
    while steps < horizon and current_obs:
        actions, current_states = policy_step(policy, adapt, current_obs, current_states)
        current_obs, rewards, dones, _truncated, _infos = env.step(actions)
        discounted_return += discount * sum(float(value) for value in rewards.values()) / original_team_size
        discount *= gamma
        steps += 1
        if bool(dones.get("__all__", False)):
            break
    return float(discounted_return), steps


def evaluate_state(
    spec: dict, env, manager, policy, adapt, observations: dict, states: dict,
    horizon: int, gamma: float, checkpoint: str,
) -> tuple[dict, dict, list[dict]]:
    expected_agents = [f"agent_{index}" for index in range(int(spec["num_agents"]))]
    if sorted(observations) != expected_agents or env.dones:
        return observations, states, invalid_records(
            spec, "selected state is not a complete active joint state", checkpoint, horizon, gamma)

    wall_start = time.perf_counter()
    policy_observations = {aid: adapt(observations[aid]) for aid in expected_agents}
    credit = _compute_counterfactual_step(policy.model, policy_observations)
    snapshot = manager.capture(observations, states)
    baseline_return = None
    baseline_fingerprint = ""
    baseline_steps = 0
    results = []
    restored_obs = observations
    restored_states = states
    try:
        baseline_obs, baseline_states, baseline_fingerprint = manager.restore(snapshot)
        manager.resume()
        baseline_return, baseline_steps = rollout_branch(
            env, policy, adapt, baseline_obs, baseline_states,
            horizon, gamma, len(expected_agents))

        for agent_spec in spec["agents"]:
            aid = str(agent_spec["agent"])
            branch_obs, branch_states, branch_fingerprint = manager.restore(snapshot)
            fingerprint_match = branch_fingerprint == baseline_fingerprint
            manager.omit_agent(aid)
            branch_obs.pop(aid, None)
            branch_states.pop(aid, None)
            manager.resume()
            intervention_return, intervention_steps = rollout_branch(
                env, policy, adapt, branch_obs, branch_states,
                horizon, gamma, len(expected_agents))
            estimated_credit = float(credit[aid]["counterfactual_credit"])
            results.append({
                "schema_version": "paired_intervention_result_v1",
                "state_id": spec["state_id"],
                "num_agents": len(expected_agents),
                "agent": aid,
                "agent_index": int(agent_spec["agent_index"]),
                "estimated_credit": estimated_credit,
                "v_full": float(credit[aid]["v_full"]),
                "v_loo": float(credit[aid]["v_loo"]),
                "baseline_return": baseline_return,
                "intervention_return": intervention_return,
                "marginal_return": baseline_return - intervention_return,
                "baseline_start_fingerprint": baseline_fingerprint,
                "intervention_start_fingerprint": branch_fingerprint,
                "horizon_steps": int(horizon),
                "baseline_steps_executed": baseline_steps,
                "intervention_steps_executed": intervention_steps,
                "gamma": float(gamma),
                "valid": bool(fingerprint_match),
                "invalid_reason": "" if fingerprint_match else "branch start fingerprint mismatch",
                "method": spec["method"],
                "scenario_bank": spec.get("scenario_bank"),
                "scenario_case": spec.get("scenario_case"),
                "training_seed": spec.get("training_seed"),
                "eval_seed": spec.get("eval_seed"),
                "episode_seed": spec["episode_seed"],
                "episode": spec["episode"],
                "step": spec["step"],
                "map_number": spec["map_number"],
                "checkpoint_path": checkpoint,
            })
    finally:
        restored_obs, restored_states, parent_fingerprint = manager.restore(snapshot)
        manager.resume()
        if baseline_fingerprint and parent_fingerprint != baseline_fingerprint:
            for result in results:
                result["valid"] = False
                result["invalid_reason"] = "parent restore fingerprint mismatch"
    elapsed = time.perf_counter() - wall_start
    for result in results:
        result["state_wall_seconds"] = elapsed
    return restored_obs, restored_states, results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--map-number", required=True, type=int)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--num-dynamic-obstacles", required=True, type=int)
    parser.add_argument("--obs-speed-mps", type=float, default=0.055)
    parser.add_argument("--max-episode-steps", required=True, type=int)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--shield-enable", type=int, choices=(0, 1), default=0)
    parser.add_argument("--max-states", type=int, default=0)
    args = parser.parse_args()
    if args.horizon_steps < 1 or not 0.0 < args.gamma <= 1.0:
        parser.error("invalid horizon or gamma")
    if not (args.checkpoint_path / "algorithm_state.pkl").is_file():
        parser.error("invalid checkpoint")

    plan = load_plan(args.plan, args.map_number, args.num_agents)
    if not plan:
        raise SystemExit(f"no states for map {args.map_number}")
    declared_agent_counts = {int(row["num_agents"]) for row in plan}
    if len(declared_agent_counts) != 1:
        raise SystemExit(
            "a paired runner invocation requires one team size; "
            f"found {sorted(declared_agent_counts)}"
        )
    num_agents = next(iter(declared_agent_counts))
    already_done = completed_state_ids(args.output, num_agents)
    plan = [row for row in plan if row["state_id"] not in already_done]
    if args.max_states > 0:
        plan = plan[:args.max_states]
    if not plan:
        print(f"all map {args.map_number} states already complete")
        return 0

    ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
    policy, env_config = load_policy_and_config(args.checkpoint_path)
    env_config.update({
        "num_agents": num_agents,
        "map_number": args.map_number,
        "num_dynamic_obstacles": args.num_dynamic_obstacles,
        "num_static_obstacles": 0,
        "random_obstacles": False,
        "obs_speed": args.obs_speed_mps,
        "max_episode_steps": args.max_episode_steps,
        "auto_reset_agents": False,
        "min_active_agents_to_continue": 0,
        "max_failed_agents_before_cutoff": 0,
        "shield_enable": bool(args.shield_enable),
        "paired_dynamic_obstacles": True,
        "comm_dropout_prob": 0.0,
        "comm_jitter_steps": 0,
        "comm_noise_std": 0.0,
        "sensor_noise_std": 0.0,
    })
    env = GNNMARLEnv(env_config)
    adapt = policy_adapter(policy, env)
    manager = BranchSnapshotManager(env)
    by_episode = defaultdict(list)
    for spec in plan:
        episode_key = (
            int(spec["episode_seed"]),
            str(spec.get("scenario_case") or ""),
        )
        by_episode[episode_key].append(spec)
    run_start = time.perf_counter()
    completed = 0
    invalid = 0
    try:
        for (episode_seed, scenario_case), specs in sorted(by_episode.items()):
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            route_plan = specs[0].get("route_plan")
            reset_options = {"route_plan": route_plan} if route_plan is not None else None
            observations, _ = env.reset(seed=episode_seed, options=reset_options)
            states = {f"agent_{index}": policy.get_initial_state() for index in range(num_agents)}
            target_by_step = {int(spec["step"]): spec for spec in specs}
            last_target = max(target_by_step)
            step = 0
            dones = {"__all__": False}
            while step < last_target and not bool(dones.get("__all__", False)):
                target_step = step + 1
                if target_step in target_by_step:
                    spec = target_by_step[target_step]
                    observations, states, records = evaluate_state(
                        spec, env, manager, policy, adapt, observations, states,
                        args.horizon_steps, args.gamma, str(args.checkpoint_path.resolve()))
                    append_records(args.output, records)
                    completed += int(all(record["valid"] for record in records))
                    invalid += int(not all(record["valid"] for record in records))
                    print(
                        f"[paired] map={args.map_number} state={spec['state_id']} "
                        f"scenario={scenario_case or 'archived'} episode_seed={episode_seed} "
                        f"step={target_step} valid={all(r['valid'] for r in records)}",
                        flush=True,
                    )
                actions, states = policy_step(policy, adapt, observations, states)
                observations, _rewards, dones, _truncated, _infos = env.step(actions)
                step += 1
            for target_step, spec in target_by_step.items():
                if target_step > step:
                    records = invalid_records(
                        spec, "episode ended before selected state", str(args.checkpoint_path.resolve()),
                        args.horizon_steps, args.gamma)
                    append_records(args.output, records)
                    invalid += 1
    finally:
        env.close()
        ray.shutdown()

    summary = {
        "map_number": args.map_number,
        "num_agents": num_agents,
        "scenario_cases": sorted({str(row.get("scenario_case") or "") for row in plan}),
        "planned_states": len(plan),
        "completed_states": completed,
        "invalid_states": invalid,
        "wall_seconds": time.perf_counter() - run_start,
        "horizon_steps": args.horizon_steps,
        "gamma": args.gamma,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
