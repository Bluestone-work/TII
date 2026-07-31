#!/usr/bin/env python3
"""Aggregate episode records for map-specific timeout-boundary experiments."""

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


MAP_LABELS = {5: "Map 5 Warehouse", 9: "Map 9 Dynamic Warehouse"}


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
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
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for record in records:
        try:
            key = (int(record["map_number"]), int(record["max_episode_steps"]))
        except (KeyError, TypeError, ValueError):
            continue
        groups[key].append(record)

    rows = []
    rng = np.random.default_rng(20260713)
    for (map_number, max_steps), items in sorted(groups.items()):
        completion = [
            float(item.get("reached_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        team_success = [float(value >= 1.0) for value in completion]
        collision = [
            float(item.get("collided_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        timeout = [float(bool(item.get("truncated"))) for item in items]
        row = {
            "map_number": map_number,
            "map_name": MAP_LABELS.get(map_number, f"Map {map_number}"),
            "max_episode_steps": max_steps,
            "n_episodes": len(items),
            "num_agents": int(items[0].get("num_agents") or 1),
            "num_dynamic_obstacles": int(items[0].get("num_dynamic_obstacles") or 0),
            "obs_speed_mps": items[0].get("obs_speed"),
            "episode_steps_mean": finite_mean(item.get("steps", math.nan) for item in items),
            "min_lidar_m_mean": finite_mean(item.get("min_dist", math.nan) for item in items),
            "inference_ms_mean": finite_mean(item.get("inference_ms_mean", math.nan) for item in items),
            "inference_ms_p95": finite_mean(item.get("inference_ms_p95", math.nan) for item in items),
        }
        for name, values in (
            ("agent_success_pct", completion),
            ("team_success_pct", team_success),
            ("agent_collision_pct", collision),
            ("timeout_pct", timeout),
        ):
            low, high = bootstrap_ci(values, rng)
            row[name] = 100.0 * finite_mean(values)
            row[f"{name}_ci_low"] = 100.0 * low
            row[f"{name}_ci_high"] = 100.0 * high
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_recommendations(rows: list[dict], path: Path) -> None:
    recommendations = []
    for map_number in sorted({int(row["map_number"]) for row in rows}):
        map_rows = [row for row in rows if int(row["map_number"]) == map_number]
        best_agent = max(float(row["agent_success_pct"]) for row in map_rows)
        best_team = max(float(row["team_success_pct"]) for row in map_rows)
        eligible = [
            row for row in map_rows
            if float(row["agent_success_pct"]) >= best_agent - 2.0
            and float(row["team_success_pct"]) >= best_team - 5.0
        ]
        chosen = min(eligible or map_rows, key=lambda row: int(row["max_episode_steps"]))
        recommendations.append({
            "map_number": map_number,
            "recommended_max_episode_steps": int(chosen["max_episode_steps"]),
            "selection_rule": "smallest limit within 2pp agent and 5pp team of observed maxima",
            "agent_success_pct": chosen["agent_success_pct"],
            "team_success_pct": chosen["team_success_pct"],
            "timeout_pct": chosen["timeout_pct"],
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recommendations[0]))
        writer.writeheader()
        writer.writerows(recommendations)


def plot(rows: list[dict], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=220, sharey=True)
    metrics = [
        ("agent_success_pct", "Agent success", "#2878B5"),
        ("team_success_pct", "Full-team success", "#3FA65B"),
        ("agent_collision_pct", "Agent collision", "#C94C4C"),
        ("timeout_pct", "Timeout episodes", "#8A6FB0"),
    ]
    for ax, map_number in zip(axes, (5, 9)):
        map_rows = sorted(
            (row for row in rows if row["map_number"] == map_number),
            key=lambda row: row["max_episode_steps"],
        )
        x = np.asarray([row["max_episode_steps"] for row in map_rows], dtype=float)
        for metric, label, color in metrics:
            y = np.asarray([row[metric] for row in map_rows], dtype=float)
            low = np.asarray([row[f"{metric}_ci_low"] for row in map_rows], dtype=float)
            high = np.asarray([row[f"{metric}_ci_high"] for row in map_rows], dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.8, label=label, color=color)
            ax.fill_between(x, low, high, color=color, alpha=0.12)
        ax.set_title(MAP_LABELS[map_number])
        ax.set_xlabel("Episode step limit")
        ax.set_ylim(0.0, 105.0)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Rate (%)")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "timeout_boundary_curves.png", bbox_inches="tight")
    fig.savefig(out_dir / "timeout_boundary_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = aggregate(load_records(args.episodes_jsonl))
    if not rows:
        raise SystemExit("No timeout-boundary records with max_episode_steps found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out_dir / "timeout_boundary_metrics.csv")
    write_recommendations(rows, args.out_dir / "recommended_limits.csv")
    plot(rows, args.out_dir)
    print(f"wrote {len(rows)} timeout-boundary conditions to {args.out_dir}")


if __name__ == "__main__":
    main()
