#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wj/work/multi-robot-exploration-rl")
TEST_SUMMARY_ROOT = REPO_ROOT / "train_logs" / "test_summaries" / "sensitivity_search"
TRAIN_OUTPUT_ROOT = REPO_ROOT / "ray_results" / "sensitivity_search"
CURRICULUM_SCRIPT = REPO_ROOT / "run_curriculum.sh"
TEST_SCRIPT = REPO_ROOT / "run_test.sh"


@dataclass(frozen=True)
class SweepCase:
    run_id: str
    run_suffix: str
    cli_overrides: dict[str, Any]
    reward_aggregation_overrides: dict[str, Any] | None = None
    interaction_potential_overrides: dict[str, Any] | None = None


BASE_ARGS = {
    "model_type": "gat",
    "gat_actor_graph": "neighbor",
    "gat_critic_mode": "mlp",
    "num_agents": 4,
    "num_workers": 1,
    "start_stage": 1,
    "end_stage": 1,
    "map_number": 6,
    "train_steps": 100000,
    "train_batch_size": 2000,
    "rollout_fragment_length": 200,
    "team_reward_lambda": 1.0,
    "interaction_reward_profile": "anti_reverse",
    "headless_sim": True,
    "disable_rviz": True,
    "rolling_lookahead_dist": 0.4,
}


SWEEP_CASES = [
    SweepCase(
        run_id="S1_BASE",
        run_suffix="sensitivity_base",
        cli_overrides={
            "risk_gate_soft": 0.08,
            "risk_gate_hard": 0.50,
            "navigation_high_risk_scale": 0.80,
            "safe_turn_reward_scale": 0.15,
        },
    ),
    SweepCase(
        run_id="S1_TURN_TIGHT",
        run_suffix="sensitivity_turn_tight",
        cli_overrides={
            "risk_gate_soft": 0.08,
            "risk_gate_hard": 0.50,
            "navigation_high_risk_scale": 0.80,
            "safe_turn_reward_scale": 0.24,
        },
    ),
    SweepCase(
        run_id="S1_RISK_SOFT",
        run_suffix="sensitivity_risk_soft",
        cli_overrides={
            "risk_gate_soft": 0.06,
            "risk_gate_hard": 0.42,
            "navigation_high_risk_scale": 0.74,
            "safe_turn_reward_scale": 0.20,
        },
    ),
    SweepCase(
        run_id="S1_RISK_STRONG",
        run_suffix="sensitivity_risk_strong",
        cli_overrides={
            "risk_gate_soft": 0.05,
            "risk_gate_hard": 0.35,
            "navigation_high_risk_scale": 0.68,
            "safe_turn_reward_scale": 0.22,
        },
    ),
    SweepCase(
        run_id="S1_BALANCED_CLIP",
        run_suffix="sensitivity_balanced_clip",
        cli_overrides={
            "risk_gate_soft": 0.07,
            "risk_gate_hard": 0.40,
            "navigation_high_risk_scale": 0.72,
            "safe_turn_reward_scale": 0.20,
        },
        reward_aggregation_overrides={
            "interaction_reward_clip": 0.14,
            "interaction_penalty_clip": 0.16,
            "interaction_base_weight": 0.90,
            "mode_reward_weight": 0.74,
            "mode_penalty_weight": 1.06,
        },
    ),
    SweepCase(
        run_id="S1_PROGRESS_BIAS",
        run_suffix="sensitivity_progress_bias",
        cli_overrides={
            "risk_gate_soft": 0.07,
            "risk_gate_hard": 0.45,
            "navigation_high_risk_scale": 0.76,
            "safe_turn_reward_scale": 0.18,
        },
        interaction_potential_overrides={
            "goal_drop_weight": 0.46,
            "obs_drop_weight": 0.58,
            "agent_drop_weight": 0.80,
            "reverse_penalty_scale": 2.8,
        },
    ),
]


def build_train_command(case: SweepCase) -> list[str]:
    cmd = [str(CURRICULUM_SCRIPT)]
    for key, value in BASE_ARGS.items():
        if key == "interaction_reward_profile":
            cmd.extend([f"--{key}", str(value)])
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    for key, value in case.cli_overrides.items():
        cmd.extend([f"--{key}", str(value)])
    cmd.extend(["--run_suffix", case.run_suffix, "--output_dir", str(TRAIN_OUTPUT_ROOT)])
    if case.reward_aggregation_overrides:
        cmd.extend(["--reward_aggregation_overrides_json", json.dumps(case.reward_aggregation_overrides, ensure_ascii=False, sort_keys=True)])
    if case.interaction_potential_overrides:
        cmd.extend(["--interaction_potential_overrides_json", json.dumps(case.interaction_potential_overrides, ensure_ascii=False, sort_keys=True)])
    return cmd


def build_test_command(checkpoint_path: str, case: SweepCase) -> list[str]:
    summary_dir = TEST_SUMMARY_ROOT / case.run_suffix
    return [
        str(TEST_SCRIPT),
        "-c",
        checkpoint_path,
        "--num_episodes",
        "10",
        "--test_stage",
        "1",
        "--map_number",
        "6",
        "--num_agents",
        "4",
        "--test_max_episode_steps",
        "1500",
        "--fixed_benchmark_set",
        "fixed10_classic_v1",
        "--rolling_lookahead_dist",
        "0.4",
        "--repeat_runs",
        "3",
        "--summary_dir",
        str(summary_dir),
    ]


def main() -> None:
    manifest_path = REPO_ROOT / "scripts" / "sensitivity_sweep_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_id", "run_suffix", "train_command", "test_command_template"
        ])
        writer.writeheader()
        for case in SWEEP_CASES:
            writer.writerow({
                "run_id": case.run_id,
                "run_suffix": case.run_suffix,
                "train_command": " ".join(build_train_command(case)),
                "test_command_template": " ".join(build_test_command("<checkpoint>", case)),
            })
    print(manifest_path)


if __name__ == "__main__":
    main()
