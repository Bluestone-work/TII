#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/wj/work/multi-robot-exploration-rl")
CONDA_SH = Path("/home/wj/anaconda3/etc/profile.d/conda.sh")
CONDA_ENV_NAME = "ros2"
CURRICULUM_SCRIPT = REPO_ROOT / "run_curriculum.sh"
TRAIN_SCRIPT = REPO_ROOT / "src/gnn_marl_training/gnn_marl_training/train_gnn_mappo_full.py"
TEST_SCRIPT = REPO_ROOT / "run_test.sh"
TRACKER_PATH = REPO_ROOT / "refine-logs/EXPERIMENT_TRACKER.md"
RESULTS_MD_PATH = REPO_ROOT / "refine-logs/EXPERIMENT_RESULTS.md"
RESULTS_JSON_PATH = REPO_ROOT / "scripts/reward_search_results.json"
RESULTS_CSV_PATH = REPO_ROOT / "scripts/reward_search_results.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "ray_results" / "reward_search"
DEFAULT_TEST_SUMMARY_ROOT = REPO_ROOT / "train_logs" / "test_summaries" / "reward_search"
BASELINE_PROXY_100K = 157.23134071222532


@dataclass
class TestSpec:
    num_episodes: int = 3
    test_stage: int = 1
    map_number: int = 6
    num_agents: int = 4
    test_max_episode_steps: int = 1500
    fixed_benchmark_set: str | None = None
    rolling_lookahead_dist: float | None = None


@dataclass
class RunSpec:
    run_id: str
    milestone: str
    variant: str
    train_steps: int
    cli_overrides: dict[str, Any]
    reward_aggregation_overrides: dict[str, Any] | None = None
    interaction_potential_overrides: dict[str, Any] | None = None
    interaction_reward_profile: str | None = None
    resume_checkpoint: str | None = None
    test_spec: TestSpec | None = None


@dataclass(frozen=True)
class RankingBreakdown:
    rank_key: tuple[float, float, float, float, float]
    collision_avoidance_score: float


def _json_arg(value: dict[str, Any] | None) -> str | None:
    if not value:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_run_specs() -> list[RunSpec]:
    test_spec = TestSpec(
        fixed_benchmark_set="fixed50_v1",
        rolling_lookahead_dist=0.4,
    )
    base = {
        "model_type": "gat",
        "gat_actor_graph": "neighbor",
        "gat_critic_mode": "mlp",
        "curriculum_stage": 1,
        "env_stage": 1,
        "num_agents": 4,
        "num_workers": 1,
        "map_number": 6,
        "train_batch_size": 2000,
        "rollout_fragment_length": 200,
        "team_reward_lambda": 1.0,
        "train_steps": 100000,
        "headless_sim": 1,
        "disable_rviz": 1,
        "rolling_lookahead_dist": 0.4,
    }
    return [
        RunSpec(
            run_id="M1_BASELINE",
            milestone="M1",
            variant="reward_baseline_no_detour_loop",
            train_steps=100000,
            cli_overrides={**base},
            interaction_reward_profile="no_detour_loop",
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C01",
            milestone="M2",
            variant="reward_progress_safe_profile",
            train_steps=100000,
            cli_overrides={**base},
            interaction_reward_profile="progress_safe",
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C02",
            milestone="M2",
            variant="reward_anti_reverse_profile",
            train_steps=100000,
            cli_overrides={**base},
            interaction_reward_profile="anti_reverse",
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C03",
            milestone="M2",
            variant="reward_event_light_profile",
            train_steps=100000,
            cli_overrides={**base},
            interaction_reward_profile="event_light",
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C04",
            milestone="M2",
            variant="reward_high_risk_avoidance",
            train_steps=100000,
            cli_overrides={
                **base,
                "risk_gate_soft": 0.05,
                "risk_gate_hard": 0.35,
                "navigation_high_risk_scale": 0.62,
                "risk_aware_forward_penalty_scale": 0.40,
                "safe_turn_reward_scale": 0.22,
            },
            interaction_reward_profile="no_detour_loop",
            interaction_potential_overrides={
                "obs_drop_weight": 0.55,
                "agent_drop_weight": 0.78,
                "spin_penalty_scale": 1.8,
                "reverse_penalty_scale": 2.4,
                "stuck_penalty_scale": 1.6,
            },
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C05",
            milestone="M2",
            variant="reward_safe_turn_ttc_focus",
            train_steps=100000,
            cli_overrides={
                **base,
                "safe_turn_reward_scale": 0.26,
                "risk_aware_forward_penalty_scale": 0.34,
                "avoidance_low_risk_scale": 0.40,
            },
            interaction_reward_profile="no_detour_loop",
            reward_aggregation_overrides={
                "safety_ttc_weight": 0.60,
                "baseline_ttc_weight": 1.30,
                "baseline_safe_turn_weight": 1.70,
                "baseline_head_on_weight": 1.95,
            },
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C06",
            milestone="M2",
            variant="reward_conflict_suppressed",
            train_steps=100000,
            cli_overrides={
                **base,
                "navigation_high_risk_scale": 0.70,
                "risk_gate_soft": 0.06,
            },
            interaction_reward_profile="no_detour_loop",
            reward_aggregation_overrides={
                "suppress_conflicting_interaction_shaping": True,
                "nav_progress_weight": 1.10,
                "nav_path_weight": 1.00,
                "nav_goal_weight": 0.72,
                "interaction_base_weight": 0.88,
                "mode_reward_weight": 0.75,
                "mode_penalty_weight": 1.10,
            },
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C07",
            milestone="M2",
            variant="reward_detour_loop_suppressed",
            train_steps=100000,
            cli_overrides={
                **base,
                "subgoal_progress_reward_scale": 1.10,
                "detour_progress_relax": 0.10,
                "risk_aware_forward_penalty_scale": 0.36,
            },
            interaction_reward_profile="no_detour_loop",
            interaction_potential_overrides={
                "detour_bonus_scale": 0.0,
                "detour_active_penalty_scale": 2.20,
                "corner_bonus_scale": 0.35,
                "path_drop_weight": 0.82,
                "event_reward_scale": 0.68,
            },
            test_spec=test_spec,
        ),
        RunSpec(
            run_id="M2_C08",
            milestone="M2",
            variant="reward_interaction_clipped",
            train_steps=100000,
            cli_overrides={
                **base,
                "risk_gate_soft": 0.07,
                "risk_gate_hard": 0.40,
                "navigation_high_risk_scale": 0.68,
            },
            interaction_reward_profile="no_detour_loop",
            reward_aggregation_overrides={
                "interaction_reward_clip": 0.12,
                "interaction_penalty_clip": 0.18,
                "interaction_base_weight": 0.82,
                "mode_reward_weight": 0.62,
                "mode_penalty_weight": 1.12,
            },
            interaction_potential_overrides={
                "goal_drop_weight": 0.42,
                "obs_drop_weight": 0.52,
                "agent_drop_weight": 0.72,
            },
            test_spec=test_spec,
        ),
    ]


def build_command(spec: RunSpec, output_dir: Path) -> list[str]:
    cmd = [
        str(CURRICULUM_SCRIPT),
        "--start_stage",
        "1",
        "--end_stage",
        "1",
        "--train_steps",
        str(spec.train_steps),
        "--run_suffix",
        spec.variant,
    ]
    if output_dir != DEFAULT_OUTPUT_DIR:
        cmd.extend(["--output_dir", str(output_dir)])
    else:
        cmd.extend(["--output_dir", str(output_dir)])
    if spec.interaction_reward_profile:
        cmd.extend(["--interaction_reward_profile", spec.interaction_reward_profile])
    for key, value in spec.cli_overrides.items():
        if key in {"train_steps", "curriculum_stage", "env_stage"}:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value in (0, 1) and key in {"headless_sim", "disable_rviz", "enable_rviz", "enable_visualization", "disable_visualization"}:
            if int(value) == 1:
                cmd.append(flag)
            continue
        cmd.extend([flag, str(value)])
    reward_json = _json_arg(spec.reward_aggregation_overrides)
    if reward_json:
        cmd.extend(["--reward_aggregation_overrides_json", reward_json])
    potential_json = _json_arg(spec.interaction_potential_overrides)
    if potential_json:
        cmd.extend(["--interaction_potential_overrides_json", potential_json])
    if spec.resume_checkpoint:
        cmd.extend(["--resume", spec.resume_checkpoint])
    return cmd


def run_dir_for_spec(spec: RunSpec, output_dir: Path) -> Path:
    mode_tag = "Interact"
    run_name = f"GNN_MAPPO_Stage1_{mode_tag}_{spec.variant}_EnvStage1"
    return output_dir / run_name


def monitor_csv_path(spec: RunSpec, output_dir: Path) -> Path:
    return run_dir_for_spec(spec, output_dir) / "training_monitor.csv"


def parse_monitor_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"training monitor not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in (None, ""):
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def compute_training_metrics(rows: list[dict[str, Any]], baseline_final: float) -> dict[str, Any]:
    rewards = [float(r["episode_reward_mean"]) for r in rows if r.get("episode_reward_mean") is not None]
    if not rewards:
        raise ValueError("no episode_reward_mean values found")
    tail_n = min(10, len(rewards))
    tail_rewards = rewards[-tail_n:]
    tail_var = statistics.pvariance(tail_rewards) if len(tail_rewards) > 1 else 0.0
    slope_window = rewards[max(0, len(rewards) - max(2, math.ceil(len(rewards) * 0.2))):]
    if len(slope_window) >= 2:
        slope = (slope_window[-1] - slope_window[0]) / max(1, len(slope_window) - 1)
    else:
        slope = 0.0
    final_reward = rewards[-1]
    max_reward = max(rewards)
    collapse_penalty = 0.0
    if final_reward < 0.25 * baseline_final:
        collapse_penalty += 0.5
    if max_reward <= 0.0:
        collapse_penalty += 0.5
    normalized_final = final_reward / baseline_final if baseline_final else final_reward
    stability_score = 1.0 / (1.0 + tail_var)
    slope_score = max(-1.0, min(1.0, slope / max(abs(baseline_final), 1.0)))
    combined_score = normalized_final + 0.15 * stability_score + 0.10 * slope_score - collapse_penalty
    return {
        "final_episode_reward_mean": final_reward,
        "max_episode_reward_mean": max_reward,
        "tail_variance": tail_var,
        "reward_slope_last_20pct": slope,
        "collapse_penalty": collapse_penalty,
        "normalized_final": normalized_final,
        "stability_score": stability_score,
        "combined_score": combined_score,
        "iterations": len(rows),
        "final_timesteps": rows[-1].get("timesteps"),
    }


def _find_latest_checkpoint_dir(root: Path) -> Path | None:
    if not root.exists() or not root.is_dir():
        return None
    if (root / "algorithm_state.pkl").exists() or (root / "rllib_checkpoint.json").exists():
        return root
    checkpoints = sorted(root.rglob("checkpoint_*"))
    checkpoint_dirs = [path for path in checkpoints if path.is_dir()]
    if checkpoint_dirs:
        return checkpoint_dirs[-1]
    if root.name in {"best", "final"}:
        return root
    return None


def resolve_checkpoint_path(spec: RunSpec, output_dir: Path) -> str | None:
    run_dir = run_dir_for_spec(spec, output_dir)
    best_candidate = _find_latest_checkpoint_dir(run_dir / "best")
    if best_candidate is not None:
        return str(best_candidate)
    final_candidate = _find_latest_checkpoint_dir(run_dir / "final")
    if final_candidate is not None:
        return str(final_candidate)
    run_candidate = _find_latest_checkpoint_dir(run_dir)
    if run_candidate is not None:
        return str(run_candidate)
    return None


def test_summary_dir_for_spec(spec: RunSpec, summary_root: Path) -> Path:
    return summary_root / spec.variant


def test_summary_csv_path(spec: RunSpec, summary_root: Path) -> Path:
    return test_summary_dir_for_spec(spec, summary_root) / "test_summary.csv"


def build_test_command(spec: RunSpec, checkpoint_path: str, summary_root: Path) -> list[str]:
    if spec.test_spec is None:
        raise ValueError(f"missing test_spec for {spec.run_id}")
    summary_dir = test_summary_dir_for_spec(spec, summary_root)
    return [
        str(TEST_SCRIPT),
        "-c",
        checkpoint_path,
        "--num_episodes",
        str(spec.test_spec.num_episodes),
        "--test_stage",
        str(spec.test_spec.test_stage),
        "--map_number",
        str(spec.test_spec.map_number),
        "--num_agents",
        str(spec.test_spec.num_agents),
        "--test_max_episode_steps",
        str(spec.test_spec.test_max_episode_steps),
        "--summary_dir",
        str(summary_dir),
        "--rolling_lookahead_dist",
        str(spec.test_spec.rolling_lookahead_dist if spec.test_spec.rolling_lookahead_dist is not None else 0.4),
        *( ["--fixed_benchmark_set", spec.test_spec.fixed_benchmark_set] if spec.test_spec.fixed_benchmark_set else [] ),
    ]


def parse_test_summary_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"test summary not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"test summary empty: {path}")
    row = rows[-1]
    parsed: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            parsed[key] = None
            continue
        try:
            numeric = float(value)
            if numeric.is_integer():
                parsed[key] = int(numeric)
            else:
                parsed[key] = numeric
        except ValueError:
            parsed[key] = value
    return parsed


def compute_collision_ranking(test_metrics: dict[str, Any]) -> RankingBreakdown:
    total_collision = float(test_metrics.get("total_collision") or 0.0)
    collided_agents = float(test_metrics.get("collided_agents") or 0.0)
    avg_min_dist = float(test_metrics.get("avg_min_dist") or 0.0)
    total_success = float(test_metrics.get("total_success") or 0.0)
    avg_reward = float(test_metrics.get("avg_reward") or 0.0)
    rank_key = (
        total_collision,
        collided_agents,
        -avg_min_dist,
        -total_success,
        -avg_reward,
    )
    collision_avoidance_score = (
        -10.0 * total_collision
        - 4.0 * collided_agents
        + 8.0 * avg_min_dist
        + 1.5 * total_success
        + 0.02 * avg_reward
    )
    return RankingBreakdown(rank_key=rank_key, collision_avoidance_score=collision_avoidance_score)


def update_tracker(results: dict[str, dict[str, Any]]) -> None:
    lines = TRACKER_PATH.read_text(encoding="utf-8").splitlines()
    updated: list[str] = []
    for line in lines:
        if not line.startswith("| M"):
            updated.append(line)
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 6:
            updated.append(line)
            continue
        run_id = parts[1]
        result = results.get(run_id)
        if not result:
            updated.append(line)
            continue
        updated.append(f"| {run_id} | {parts[2]} | {parts[3]} | {result['status']} | {result['notes']} |")
    TRACKER_PATH.write_text("\n".join(updated) + "\n", encoding="utf-8")


def _format_tracker_notes(result: dict[str, Any]) -> str:
    test_metrics = result.get("test_metrics") or {}
    if test_metrics:
        return (
            f"collision={test_metrics.get('total_collision', 'NA')} "
            f"agents={test_metrics.get('collided_agents', 'NA')} "
            f"min_dist={test_metrics.get('avg_min_dist', 'NA')} "
            f"success={test_metrics.get('total_success', 'NA')}"
        )
    training_metrics = result.get("training_metrics") or {}
    if training_metrics:
        return (
            f"train_final={training_metrics.get('final_episode_reward_mean', 'NA')} "
            f"train_score={training_metrics.get('combined_score', 'NA')}"
        )
    return str(result.get("notes", ""))


def write_results_files(summary: dict[str, Any]) -> None:
    RESULTS_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for item in summary["runs"]:
        test_metrics = item.get("test_metrics", {})
        training_metrics = item.get("training_metrics", {})
        ranking = item.get("ranking", {})
        row = {
            "run_id": item["run_id"],
            "milestone": item["milestone"],
            "variant": item["variant"],
            "status": item["status"],
            "total_collision": test_metrics.get("total_collision"),
            "collided_agents": test_metrics.get("collided_agents"),
            "avg_min_dist": test_metrics.get("avg_min_dist"),
            "total_success": test_metrics.get("total_success"),
            "avg_reward": test_metrics.get("avg_reward"),
            "collision_avoidance_score": ranking.get("collision_avoidance_score"),
            "train_final_reward": training_metrics.get("final_episode_reward_mean"),
            "run_dir": item.get("run_dir"),
            "checkpoint": item.get("checkpoint"),
            "test_summary_csv": item.get("test_summary_csv"),
        }
        rows.append(row)
    with RESULTS_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    ranked_runs = [r for r in summary["runs"] if r["status"] == "DONE" and r.get("test_metrics")]
    ranked_runs.sort(key=lambda r: tuple(r.get("ranking", {}).get("rank_key", (float("inf"),))))
    lines = [
        "# Experiment Results",
        "",
        "**Date**: 2026-05-21",
        "**Plan**: refine-logs/EXPERIMENT_PLAN.md",
        "",
        "## Summary",
        f"- Completed runs: {sum(1 for r in summary['runs'] if r['status'] == 'DONE')}/{len(summary['runs'])}",
        f"- Best candidate: {summary.get('best_candidate', 'N/A')}",
        f"- Ranking priority: total_collision asc → collided_agents asc → avg_min_dist desc → total_success desc → avg_reward desc",
        "",
        "## Collision-first ranking",
    ]
    if not ranked_runs:
        lines.append("- No completed tested runs yet.")
    for index, item in enumerate(ranked_runs, start=1):
        test_metrics = item.get("test_metrics", {})
        training_metrics = item.get("training_metrics", {})
        lines.append(
            f"- #{index} {item['run_id']} ({item['variant']}): "
            f"collision={test_metrics.get('total_collision', 'NA')} | "
            f"agents={test_metrics.get('collided_agents', 'NA')} | "
            f"min_dist={test_metrics.get('avg_min_dist', 'NA')} | "
            f"success={test_metrics.get('total_success', 'NA')} | "
            f"avg_reward={test_metrics.get('avg_reward', 'NA')} | "
            f"train_final={training_metrics.get('final_episode_reward_mean', 'NA')}"
        )
    lines.extend(["", "## All runs"])
    for item in summary["runs"]:
        lines.append(
            f"- {item['run_id']} ({item['variant']}): {item['status']}"
            f" | checkpoint={item.get('checkpoint', 'NA')}"
            f" | test_summary={item.get('test_summary_csv', 'NA')}"
        )
    RESULTS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _ensure_clean_summary_dir(summary_dir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if summary_dir.exists():
        shutil.rmtree(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)


def _run_in_ros2_env(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    shell_cmd = " ".join([shlex.quote(part) for part in cmd])
    return subprocess.run(
        [
            "bash",
            "-lc",
            f"source {shlex.quote(str(CONDA_SH))} && conda activate {shlex.quote(CONDA_ENV_NAME)} && {shell_cmd}",
        ],
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def execute_run(spec: RunSpec, output_dir: Path, summary_root: Path, dry_run: bool) -> dict[str, Any]:
    train_cmd = build_command(spec, output_dir)
    run_dir = run_dir_for_spec(spec, output_dir)
    summary_dir = test_summary_dir_for_spec(spec, summary_root)
    if dry_run:
        placeholder_checkpoint = str(run_dir / "best")
        test_cmd = build_test_command(spec, placeholder_checkpoint, summary_root)
        return {
            "run_id": spec.run_id,
            "milestone": spec.milestone,
            "variant": spec.variant,
            "status": "PLANNED",
            "train_command": train_cmd,
            "test_command": test_cmd,
            "run_dir": str(run_dir),
            "test_summary_dir": str(summary_dir),
            "notes": "dry-run",
        }

    completed = _run_in_ros2_env(train_cmd)
    if completed.returncode != 0:
        return {
            "run_id": spec.run_id,
            "milestone": spec.milestone,
            "variant": spec.variant,
            "status": "FAILED",
            "train_command": train_cmd,
            "run_dir": str(run_dir),
            "test_summary_dir": str(summary_dir),
            "stdout": completed.stdout,
            "notes": f"train exit={completed.returncode}",
        }

    training_metrics = compute_training_metrics(
        parse_monitor_csv(monitor_csv_path(spec, output_dir)),
        BASELINE_PROXY_100K,
    )
    checkpoint = resolve_checkpoint_path(spec, output_dir)
    if not checkpoint:
        return {
            "run_id": spec.run_id,
            "milestone": spec.milestone,
            "variant": spec.variant,
            "status": "FAILED",
            "train_command": train_cmd,
            "run_dir": str(run_dir),
            "training_metrics": training_metrics,
            "test_summary_dir": str(summary_dir),
            "notes": "checkpoint not found after training",
        }

    _ensure_clean_summary_dir(summary_dir, dry_run=False)
    test_cmd = build_test_command(spec, checkpoint, summary_root)
    test_completed = _run_in_ros2_env(test_cmd)
    if test_completed.returncode != 0:
        return {
            "run_id": spec.run_id,
            "milestone": spec.milestone,
            "variant": spec.variant,
            "status": "FAILED",
            "train_command": train_cmd,
            "test_command": test_cmd,
            "run_dir": str(run_dir),
            "checkpoint": checkpoint,
            "training_metrics": training_metrics,
            "test_summary_dir": str(summary_dir),
            "stdout": test_completed.stdout,
            "notes": f"test exit={test_completed.returncode}",
        }

    test_summary_csv = test_summary_csv_path(spec, summary_root)
    test_metrics = parse_test_summary_csv(test_summary_csv)
    ranking = compute_collision_ranking(test_metrics)
    return {
        "run_id": spec.run_id,
        "milestone": spec.milestone,
        "variant": spec.variant,
        "status": "DONE",
        "train_command": train_cmd,
        "test_command": test_cmd,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "training_metrics": training_metrics,
        "test_metrics": test_metrics,
        "ranking": {
            "rank_key": list(ranking.rank_key),
            "collision_avoidance_score": ranking.collision_avoidance_score,
        },
        "test_summary_dir": str(summary_dir),
        "test_summary_csv": str(test_summary_csv),
        "notes": _format_tracker_notes({"test_metrics": test_metrics}),
    }


def choose_best_candidate(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [r for r in results if r["status"] == "DONE" and r.get("test_metrics")]
    if not candidates:
        return None
    candidates.sort(key=lambda r: tuple(r["ranking"]["rank_key"]))
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential reward search with collision-first evaluation")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--test_summary_root", type=str, default=str(DEFAULT_TEST_SUMMARY_ROOT))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--start_from", type=str, default="M1_BASELINE")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_root = Path(args.test_summary_root).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    planned_runs = build_run_specs()
    all_results: list[dict[str, Any]] = []
    tracker_results: dict[str, dict[str, Any]] = {}

    run_ids = [spec.run_id for spec in planned_runs]
    if args.start_from not in run_ids:
        raise SystemExit(f"--start_from must be one of: {', '.join(run_ids)}")
    start_index = run_ids.index(args.start_from)

    for spec in planned_runs[start_index:]:
        result = execute_run(spec, output_dir, summary_root, args.dry_run)
        all_results.append(result)
        tracker_results[spec.run_id] = {
            "status": result["status"],
            "notes": _format_tracker_notes(result),
        }
        update_tracker(tracker_results)
        if result["status"] == "FAILED" and not args.dry_run:
            break

    best = choose_best_candidate(all_results) if not args.dry_run else None
    summary = {
        "output_dir": str(output_dir),
        "test_summary_root": str(summary_root),
        "best_candidate": best["run_id"] if best else None,
        "runs": all_results,
    }
    if all_results:
        write_results_files(summary)


if __name__ == "__main__":
    main()
