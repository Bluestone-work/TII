#!/usr/bin/env python3
"""Aggregate zero-shot team-size limit tests and host resource telemetry."""

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


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def bootstrap_ci(values, rng, samples=10000):
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return float("nan"), float("nan")
    indices = rng.integers(0, array.size, size=(samples, array.size))
    boot = np.mean(array[indices], axis=1)
    return tuple(float(value) for value in np.percentile(boot, [2.5, 97.5]))


def load_records(root: Path):
    groups = defaultdict(list)
    for path in sorted(root.glob("agents_*/episodes.jsonl")):
        try:
            team_size = int(path.parent.name.split("_")[-1])
        except ValueError:
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    groups[team_size].append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return groups


def load_resources(path: Path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    return {
        "wall_time_s": float(rows[-1]["elapsed_s"]),
        "peak_process_rss_gb": max(float(row["process_rss_kb"]) for row in rows) / 1048576.0,
        "peak_process_cpu_pct": max(float(row["process_cpu_pct"]) for row in rows),
        "min_system_available_gb": min(float(row["system_available_kb"]) for row in rows) / 1048576.0,
        "peak_gpu_memory_mb": max(float(row["gpu_memory_mb"]) for row in rows),
    }


def aggregate(root: Path):
    rng = np.random.default_rng(20260714)
    output = []
    for team_size, items in sorted(load_records(root).items()):
        completion = [
            float(item.get("reached_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        collision = [
            float(item.get("collided_agents") or 0) / max(1, int(item.get("num_agents") or 1))
            for item in items
        ]
        timeout = [float(bool(item.get("truncated"))) for item in items]
        row = {
            "num_agents": team_size,
            "trained_num_agents": int(items[0].get("trained_num_agents") or 4),
            "zero_shot_team_size": True,
            "agent_id_aliasing": bool(items[0].get("agent_id_aliasing", team_size > 8)),
            "n_episodes": len(items),
            "agent_success_pct": 100.0 * finite_mean(completion),
            "team_success_pct": 100.0 * finite_mean(value >= 1.0 for value in completion),
            "agent_collision_pct": 100.0 * finite_mean(collision),
            "timeout_pct": 100.0 * finite_mean(timeout),
            "episode_steps_mean": finite_mean(item.get("steps", math.nan) for item in items),
            "min_lidar_m_mean": finite_mean(item.get("min_dist", math.nan) for item in items),
            "min_agent_pairwise_m_mean": finite_mean(
                item.get("min_agent_pairwise_dist", math.nan) for item in items
            ),
            "inference_ms_mean": finite_mean(item.get("inference_ms_mean", math.nan) for item in items),
            "inference_ms_p95": finite_mean(item.get("inference_ms_p95", math.nan) for item in items),
            **load_resources(root / f"agents_{team_size}" / "resources.csv"),
        }
        for name, values in (
            ("agent_success_pct", completion),
            ("team_success_pct", [float(value >= 1.0) for value in completion]),
            ("agent_collision_pct", collision),
            ("timeout_pct", timeout),
        ):
            low, high = bootstrap_ci(values, rng)
            row[f"{name}_ci_low"] = 100.0 * low
            row[f"{name}_ci_high"] = 100.0 * high
        output.append(row)
    return output


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, out_dir):
    x = np.asarray([row["num_agents"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=220)
    for metric, label, color in (
        ("agent_success_pct", "Agent success", "#2878B5"),
        ("agent_collision_pct", "Agent collision", "#C94C4C"),
        ("timeout_pct", "Timeout", "#8A6FB0"),
    ):
        y = np.asarray([row[metric] for row in rows])
        low = np.asarray([row[f"{metric}_ci_low"] for row in rows])
        high = np.asarray([row[f"{metric}_ci_high"] for row in rows])
        axes[0].plot(x, y, marker="o", label=label, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=0.12)
    axes[0].set(xlabel="Robot team size (zero-shot)", ylabel="Rate (%)", ylim=(0, 105))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, [row["inference_ms_mean"] for row in rows], marker="o", label="Mean inference")
    axes[1].plot(x, [row["inference_ms_p95"] for row in rows], marker="s", label="P95 inference")
    axes[1].set(xlabel="Robot team size (zero-shot)", ylabel="Latency (ms)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "team_size_limit.png", bbox_inches="tight")
    fig.savefig(out_dir / "team_size_limit.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    rows = aggregate(args.root)
    if not rows:
        raise SystemExit("No completed team-size limit records")
    write_csv(rows, args.root / "team_size_limit_metrics.csv")
    plot(rows, args.root)
    print(f"wrote {len(rows)} team-size conditions")


if __name__ == "__main__":
    main()
