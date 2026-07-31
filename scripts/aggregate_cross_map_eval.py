#!/usr/bin/env python3
"""Aggregate cross-map evaluation JSONL into paper-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


MAP_DISPLAY = {
    3: "Map 3 Corridor",
    4: "Map 4 Intersection",
    5: "Map 5 Warehouse",
    9: "Map 9 Dynamic Warehouse",
}


METRICS = [
    ("success_rate_mean", "Success Rate (%)"),
    ("collision_rate_mean", "Collision Rate (%)"),
    ("episode_length_mean", "Episode Length"),
    ("avg_reward_mean", "Average Reward"),
    ("min_dist_mean", "Min LiDAR (m)"),
    ("min_dynamic_obs_ttc_mean", "Min Dynamic TTC (s)"),
    ("avg_dynamic_obs_ttc_risk_mean", "Dynamic TTC Risk"),
    ("avg_shield_active_mean", "Shield Activation"),
    ("avg_motion_filter_active_mean", "Motion Filter Activation"),
    ("spl_mean", "SPL"),
    ("path_length_mean", "Path Length (m)"),
    ("clearance_p05_mean", "5th-pct Clearance (m)"),
    ("minimum_ttc_mean", "Minimum TTC (s)"),
    ("near_collision_count_mean", "Near Collisions"),
    ("deadlock_rate_mean", "Deadlock Rate (%)"),
    ("oscillation_count_mean", "Oscillations"),
    ("control_jerk_mean", "Control Jerk"),
    ("filter_action_difference_mean", "Filter Action Difference"),
    ("policy_only_inference_latency_ms_mean", "Policy Latency (ms)"),
]


def _finite(values):
    out = []
    for value in values:
        if value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def _mean_std(values):
    vals = _finite(values)
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def aggregate(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        map_number = int(record.get("map_number") or 0)
        scenario = record.get("scenario") or f"map{map_number}"
        groups[(map_number, scenario)].append(record)

    rows = []
    for (map_number, scenario), items in sorted(groups.items()):
        if not items:
            continue
        num_agents = int(items[0].get("num_agents") or 1)
        training_seeds = sorted({
            item.get("training_seed")
            for item in items
            if item.get("training_seed") is not None
        })
        aggregate_by_training_seed = len(training_seeds) >= 2

        def summarize(values):
            if not aggregate_by_training_seed:
                return _mean_std(values)
            by_seed = defaultdict(list)
            for item, value in zip(items, values):
                seed = item.get("training_seed")
                if seed is not None:
                    by_seed[seed].append(value)
            seed_means = []
            for seed in training_seeds:
                finite = _finite(by_seed[seed])
                if finite:
                    seed_means.append(float(np.mean(finite)))
            return _mean_std(seed_means)

        success_rates = [
            100.0 * float(item.get("reached_agents") or 0.0) / max(1, int(item.get("num_agents") or num_agents))
            for item in items
        ]
        collision_rates = [
            100.0 * float(item.get("collided_agents") or 0.0) / max(1, int(item.get("num_agents") or num_agents))
            for item in items
        ]

        row = {
            "map_number": map_number,
            "scenario": scenario,
            "map_name": MAP_DISPLAY.get(map_number, f"Map {map_number}"),
            "n_episodes": len(items),
            "num_agents": num_agents,
            "num_dynamic_obstacles": items[0].get("num_dynamic_obstacles"),
            "obs_speed": items[0].get("obs_speed"),
            "shield_enable": items[0].get("shield_enable"),
            "aggregation_level": "training_seed" if aggregate_by_training_seed else "episode",
            "seeds": ";".join(str(seed) for seed in sorted({item.get("seed") for item in items if item.get("seed") is not None})),
            "training_seeds": ";".join(str(seed) for seed in training_seeds),
        }

        fields = {
            "success_rate": success_rates,
            "collision_rate": collision_rates,
            "episode_length": [item.get("steps") for item in items],
            "avg_reward": [item.get("avg_reward") for item in items],
            "min_dist": [item.get("min_dist") for item in items],
            "avg_front_min": [item.get("avg_front_min") for item in items],
            "avg_r_ttc": [item.get("avg_r_ttc") for item in items],
            "avg_dynamic_obs_count": [item.get("avg_dynamic_obs_count") for item in items],
            "min_dynamic_obs_dist": [item.get("min_dynamic_obs_dist") for item in items],
            "min_dynamic_obs_ttc": [item.get("min_dynamic_obs_ttc") for item in items],
            "avg_dynamic_obs_ttc_risk": [item.get("avg_dynamic_obs_ttc_risk") for item in items],
            "max_dynamic_obs_ttc_risk": [item.get("max_dynamic_obs_ttc_risk") for item in items],
            "avg_shield_active": [item.get("avg_shield_active") for item in items],
            "avg_motion_filter_active": [item.get("avg_motion_filter_active") for item in items],
            "min_motion_safety_margin": [item.get("min_motion_safety_margin") for item in items],
            "path_length": [item.get("path_length") for item in items],
            "shortest_path_length": [item.get("shortest_path_length") for item in items],
            "spl": [item.get("spl") for item in items],
            "completion_time": [item.get("completion_time") for item in items],
            "minimum_clearance": [item.get("minimum_clearance") for item in items],
            "clearance_p05": [item.get("clearance_p05") for item in items],
            "minimum_ttc": [item.get("minimum_ttc") for item in items],
            "near_collision_count": [item.get("near_collision_count") for item in items],
            "deadlock_rate": [100.0 * float(bool(item.get("deadlock"))) for item in items],
            "oscillation_count": [item.get("oscillation_count") for item in items],
            "control_jerk": [item.get("control_jerk") for item in items],
            "filter_intervention_rate": [item.get("filter_intervention_rate") for item in items],
            "filter_action_difference": [item.get("filter_action_difference") for item in items],
            "policy_only_inference_latency_ms": [
                item.get("policy_only_inference_latency_ms_mean") for item in items
            ],
            "gate_mean": [item.get("gate_mean") for item in items],
            "gate_std": [item.get("gate_std") for item in items],
        }
        for name, values in fields.items():
            mean, std = summarize(values)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        truncated_values = [100.0 if item.get("truncated") else 0.0 for item in items]
        row["truncated_rate_mean"], row["truncated_rate_std"] = summarize(truncated_values)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    base_fields = [
        "map_number", "map_name", "scenario", "n_episodes", "num_agents",
        "num_dynamic_obstacles", "obs_speed", "truncated_rate_mean",
        "truncated_rate_std", "shield_enable", "aggregation_level",
        "seeds",
        "training_seeds",
    ]
    metric_fields = []
    for key in [
        "success_rate", "collision_rate", "episode_length", "avg_reward",
        "min_dist", "avg_front_min", "avg_r_ttc", "avg_dynamic_obs_count",
        "min_dynamic_obs_dist", "min_dynamic_obs_ttc",
        "avg_dynamic_obs_ttc_risk", "max_dynamic_obs_ttc_risk",
        "avg_shield_active",
        "avg_motion_filter_active", "min_motion_safety_margin",
    ]:
        metric_fields.extend([f"{key}_mean", f"{key}_std"])
    fields = base_fields + metric_fields
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_latex(rows: list[dict], out_path: Path) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cross-map evaluation of the trained GNN-MAPPO policy. With multiple training seeds, mean$\pm$std is computed over seed-level means; single-seed runs use episode-level values.}",
        r"\label{tab:cross_map_eval}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Map & Success (\%) $\uparrow$ & Collision (\%) $\downarrow$ & Steps $\downarrow$ & Min LiDAR (m) $\uparrow$ & Dyn. TTC (s) $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {:.1f}$\\pm${:.1f} & {:.1f}$\\pm${:.1f} & {:.0f}$\\pm${:.0f} & {:.2f}$\\pm${:.2f} & {:.2f}$\\pm${:.2f} \\\\".format(
                row["map_name"],
                row["success_rate_mean"], row["success_rate_std"],
                row["collision_rate_mean"], row["collision_rate_std"],
                row["episode_length_mean"], row["episode_length_std"],
                row["min_dist_mean"], row["min_dist_std"],
                row["min_dynamic_obs_ttc_mean"], row["min_dynamic_obs_ttc_std"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict], out_prefix: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [row["map_name"].replace(" ", "\n", 2) for row in rows]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), dpi=180)
    axes = axes.flatten()
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]

    for ax, (key, title) in zip(axes, METRICS):
        means = [row.get(key, float("nan")) for row in rows]
        std_key = key.replace("_mean", "_std")
        stds = [row.get(std_key, 0.0) for row in rows]
        ax.bar(range(len(rows)), means, yerr=stds, capsize=4, color=colors[: len(rows)], edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(out_prefix) + ".png", bbox_inches="tight")
    fig.savefig(str(out_prefix) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    outdir = args.outdir or args.jsonl.parent
    outdir.mkdir(parents=True, exist_ok=True)
    rows = aggregate(load_records(args.jsonl))
    write_csv(rows, outdir / "cross_map_metrics.csv")
    write_latex(rows, outdir / "cross_map_table.tex")
    plot(rows, outdir / "cross_map_metrics")
    print(f"wrote {len(rows)} map summaries to {outdir}")


if __name__ == "__main__":
    main()
