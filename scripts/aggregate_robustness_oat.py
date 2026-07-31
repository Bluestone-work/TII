#!/usr/bin/env python3
"""Aggregate one-at-a-time robustness and zero-shot team-size evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get("experiment_condition"):
                    records.append(record)
    return records


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def bootstrap_ci(values, rng: np.random.Generator, samples: int = 10000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan"), float("nan")
    indices = rng.integers(0, array.size, size=(samples, array.size))
    means = np.mean(array[indices], axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aggregate(records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[str(record["experiment_condition"])].append(record)
    rows = []
    rng = np.random.default_rng(20260713)
    for condition, items in sorted(groups.items()):
        first = items[0]
        completion = [
            float(item.get("reached_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        collision = [
            float(item.get("collided_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        if condition.startswith("obs_"):
            factor = "dynamic_obstacles"
            value = float(first.get("num_dynamic_obstacles") or 0)
        elif condition.startswith("speed_"):
            factor = "obstacle_speed"
            value = float(first.get("obs_speed") or 0.0)
        elif condition.startswith("agents_"):
            factor = "team_size"
            value = float(first.get("num_agents") or 0)
        else:
            factor = "baseline"
            value = float("nan")
        timeout = [float(bool(item.get("truncated"))) for item in items]
        min_lidar = [item.get("min_dist", math.nan) for item in items]
        min_pairwise = [item.get("min_agent_pairwise_dist", math.nan) for item in items]
        inference_mean = [item.get("inference_ms_mean", math.nan) for item in items]
        inference_p95 = [item.get("inference_ms_p95", math.nan) for item in items]
        row = {
            "condition": condition,
            "factor": factor,
            "factor_value": value,
            "n_episodes": len(items),
            "num_agents": int(first.get("num_agents") or 1),
            "trained_num_agents": int(first.get("trained_num_agents") or first.get("num_agents") or 1),
            "zero_shot_team_size": bool(first.get("zero_shot_team_size")),
            "num_dynamic_obstacles": int(first.get("num_dynamic_obstacles") or 0),
            "obs_speed_mps": float(first.get("obs_speed") or 0.0),
            "max_episode_steps": int(first.get("max_episode_steps") or 0),
            "agent_success_pct": 100.0 * finite_mean(completion),
            "team_success_pct": 100.0 * finite_mean(value >= 1.0 for value in completion),
            "agent_collision_pct": 100.0 * finite_mean(collision),
            "timeout_pct": 100.0 * finite_mean(timeout),
            "episode_steps_mean": finite_mean(item.get("steps", math.nan) for item in items),
            "min_lidar_m_mean": finite_mean(min_lidar),
            "min_agent_pairwise_m_mean": finite_mean(min_pairwise),
            "inference_ms_mean": finite_mean(inference_mean),
            "inference_ms_p95": finite_mean(inference_p95),
        }
        for name, values, scale in (
            ("agent_success_pct", completion, 100.0),
            ("team_success_pct", [float(value >= 1.0) for value in completion], 100.0),
            ("agent_collision_pct", collision, 100.0),
            ("timeout_pct", timeout, 100.0),
            ("min_lidar_m_mean", min_lidar, 1.0),
            ("min_agent_pairwise_m_mean", min_pairwise, 1.0),
            ("inference_ms_mean", inference_mean, 1.0),
            ("inference_ms_p95", inference_p95, 1.0),
        ):
            low, high = bootstrap_ci(values, rng)
            row[f"{name}_ci_low"] = scale * low
            row[f"{name}_ci_high"] = scale * high
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def factor_rows(rows: list[dict], factor: str) -> list[dict]:
    baseline = next(row for row in rows if row["factor"] == "baseline")
    value_key = {
        "dynamic_obstacles": float(baseline["num_dynamic_obstacles"]),
        "obstacle_speed": float(baseline["obs_speed_mps"]),
        "team_size": float(baseline["num_agents"]),
    }[factor]
    base_copy = dict(baseline, factor=factor, factor_value=value_key)
    selected = [base_copy] + [row for row in rows if row["factor"] == factor]
    return sorted(selected, key=lambda row: float(row["factor_value"]))


def plot(rows: list[dict], out_dir: Path) -> None:
    factors = [
        ("dynamic_obstacles", "Dynamic obstacles"),
        ("obstacle_speed", "Obstacle speed (m/s)"),
        ("team_size", "Robot team size (zero-shot)"),
    ]
    metrics = [
        ("agent_success_pct", "Agent success", "#2878B5"),
        ("agent_collision_pct", "Agent collision", "#C94C4C"),
        ("timeout_pct", "Timeout", "#8A6FB0"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220, sharey=True)
    for ax, (factor, xlabel) in zip(axes, factors):
        selected = factor_rows(rows, factor)
        x = [row["factor_value"] for row in selected]
        for metric, label, color in metrics:
            y = np.asarray([row[metric] for row in selected], dtype=float)
            low = np.asarray([row[f"{metric}_ci_low"] for row in selected], dtype=float)
            high = np.asarray([row[f"{metric}_ci_high"] for row in selected], dtype=float)
            ax.plot(x, y, marker="o", label=label, color=color)
            ax.fill_between(x, low, high, color=color, alpha=0.12)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0.0, 105.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rate (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_oat_outcomes.png", bbox_inches="tight")
    fig.savefig(out_dir / "robustness_oat_outcomes.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220)
    for ax, (factor, xlabel) in zip(axes, factors):
        selected = factor_rows(rows, factor)
        x = [row["factor_value"] for row in selected]
        for metric, label, marker in (
            ("min_lidar_m_mean", "Min LiDAR (m)", "o"),
            ("min_agent_pairwise_m_mean", "Min pairwise (m)", "s"),
        ):
            y = np.asarray([row[metric] for row in selected], dtype=float)
            low = np.asarray([row[f"{metric}_ci_low"] for row in selected], dtype=float)
            high = np.asarray([row[f"{metric}_ci_high"] for row in selected], dtype=float)
            ax.plot(x, y, marker=marker, label=label)
            ax.fill_between(x, low, high, alpha=0.12)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Distance (m)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_oat_clearance.png", bbox_inches="tight")
    fig.savefig(out_dir / "robustness_oat_clearance.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=220)
    for ax, (factor, xlabel) in zip(axes, factors):
        selected = factor_rows(rows, factor)
        x = [row["factor_value"] for row in selected]
        for metric, label, marker in (
            ("inference_ms_mean", "Mean inference", "o"),
            ("inference_ms_p95", "P95 inference", "s"),
        ):
            y = np.asarray([row[metric] for row in selected], dtype=float)
            low = np.asarray([row[f"{metric}_ci_low"] for row in selected], dtype=float)
            high = np.asarray([row[f"{metric}_ci_high"] for row in selected], dtype=float)
            ax.plot(x, y, marker=marker, label=label)
            ax.fill_between(x, low, high, alpha=0.12)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Inference latency (ms)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "robustness_oat_latency.png", bbox_inches="tight")
    fig.savefig(out_dir / "robustness_oat_latency.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = aggregate(load_records(args.episodes_jsonl))
    if not rows:
        raise SystemExit("No robustness records found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "robustness_oat_metrics.csv")
    plot(rows, args.out_dir)
    print(f"wrote {len(rows)} robustness conditions to {args.out_dir}")


if __name__ == "__main__":
    main()
