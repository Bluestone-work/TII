#!/usr/bin/env python3
"""Build deterministic paired-intervention states for the formal micro bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gnn_marl_training.formal_scenario_bank import get_scenario_bank


def state_id(case_id: str, episode_seed: int, step: int) -> str:
    payload = f"{case_id}:{episode_seed}:{step}".encode("ascii")
    return "micro_" + hashlib.sha256(payload).hexdigest()[:20]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenario-bank", default="paper_micro_v1")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--episode-seed-start", type=int, default=20260818)
    parser.add_argument("--target-step", type=int, default=200)
    parser.add_argument("--target-steps", nargs="+", type=int)
    parser.add_argument("--method", default="ours_dual_cf")
    parser.add_argument("--training-seed", type=int, default=1)
    parser.add_argument("--eval-seed", type=int, default=20260718)
    args = parser.parse_args()
    target_steps = args.target_steps or [args.target_step]
    if args.eval_seeds < 1 or any(step < 1 for step in target_steps):
        parser.error("eval-seeds and target steps must be positive")

    rows = []
    selected_cases = set(args.cases or ())
    scenarios = [
        scenario for scenario in get_scenario_bank(args.scenario_bank)
        if not selected_cases or scenario.case_id in selected_cases
    ]
    missing_cases = selected_cases - {scenario.case_id for scenario in scenarios}
    if missing_cases:
        parser.error(f"unknown scenario cases: {sorted(missing_cases)}")
    for scenario in scenarios:
        for episode in range(args.eval_seeds):
            for step_index, target_step in enumerate(target_steps):
                episode_index = episode * len(target_steps) + step_index
                episode_seed = args.episode_seed_start + episode_index
                rows.append({
                    "schema_version": "paired_intervention_plan_v1",
                    "state_id": state_id(scenario.case_id, episode_seed, target_step),
                    "method": args.method,
                    "training_seed": args.training_seed,
                    "eval_seed": args.eval_seed,
                    "episode_seed": episode_seed,
                    "episode": episode_index,
                    "step": target_step,
                    "map_number": scenario.map_number,
                    "num_agents": scenario.num_agents,
                    "agents": [
                        {"agent": f"agent_{index}", "agent_index": index}
                        for index in range(scenario.num_agents)
                    ],
                    "scenario_bank": args.scenario_bank,
                    "scenario_case": scenario.case_id,
                    "scenario_category": scenario.category,
                    "route_plan": scenario.route_plan(),
                    "max_episode_steps": scenario.max_episode_steps,
                    "num_dynamic_obstacles": scenario.num_dynamic_obstacles,
                    "obs_speed_mps": scenario.obstacle_speed_mps,
                    "source": "formal_scenario_bank",
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} fixed-micro states -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
