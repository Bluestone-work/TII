#!/usr/bin/env python3
"""Generate the required paper tables and figures from formal experiment data."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.formal_statistics import (
    DEFAULT_METRICS,
    finite_float,
    load_records,
    summarize_across_seeds,
)


TABLE_METRICS = (
    "agent_success_rate", "team_success", "collision_rate", "timeout_rate",
    "spl", "path_length", "completion_time", "minimum_clearance",
    "clearance_p05", "minimum_ttc", "deadlock", "control_jerk",
)

MAIN_METHODS = ("full", "ippo", "mlp_lstm", "orca_dwa")


def save_figure(fig, out_dir, stem):
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def seed_summary_frame(records, metrics):
    frame = pd.DataFrame(records)
    if frame.empty or not {"method", "training_seed"}.issubset(frame.columns):
        return pd.DataFrame()
    available = [metric for metric in metrics if metric in frame.columns]
    rows = summarize_across_seeds(records, available)
    return pd.DataFrame(rows).rename(columns={
        "mean_across_seeds": "mean",
        "std_across_seeds": "std",
    })


def write_tables(records, out_dir):
    summary = seed_summary_frame(records, TABLE_METRICS)
    if summary.empty:
        return False
    table = summary.pivot(index="method", columns="metric", values="mean")
    std = summary.pivot(index="method", columns="metric", values="std")
    display = table.copy().astype(object)
    for method in table.index:
        for metric in table.columns:
            mean = table.loc[method, metric]
            deviation = std.loc[method, metric]
            display.loc[method, metric] = (
                f"{mean:.4f} +/- {deviation:.4f}"
                if math.isfinite(float(deviation)) else f"{mean:.4f}"
            )
    main_methods = [method for method in MAIN_METHODS if method in display.index]
    main_display = display.loc[main_methods] if main_methods else display
    main_display.to_csv(out_dir / "main_performance_table.csv")
    (out_dir / "main_performance_table.tex").write_text(
        main_display.to_latex(escape=True), encoding="utf-8"
    )
    ablation = display.loc[
        [index for index in display.index if str(index) not in {"ippo", "mlp_lstm", "orca_dwa", "full_filter_off"}]
    ]
    if not ablation.empty:
        ablation.to_csv(out_dir / "ablation_table.csv")
        (out_dir / "ablation_table.tex").write_text(
            ablation.to_latex(escape=True), encoding="utf-8"
        )
    return True


def parse_monitor_identity(path):
    text = path.as_posix()
    seed_match = re.search(r"/seed(\d+)(?:/|$)", text)
    map_match = re.search(r"/map(\d+)(?:/|$)", text)
    if seed_match:
        prefix = text[:seed_match.start()]
        method = prefix.rstrip("/").rsplit("/", 1)[-1]
        return method, int(seed_match.group(1)), int(map_match.group(1)) if map_match else 0
    name = path.parent.name
    seed_match = re.search(r"seed[_-]?(\d+)", text)
    return name, int(seed_match.group(1)) if seed_match else 0, 0


def training_curve_data(training_root):
    frames = []
    map_order = {3: 0, 4: 1, 5: 2, 9: 3}
    for path in sorted(training_root.rglob("training_monitor.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        method, seed, map_number = parse_monitor_identity(path)
        steps_column = "environment_steps" if "environment_steps" in frame else "timesteps"
        reward_column = "episode_reward" if "episode_reward" in frame else "episode_reward_mean"
        if steps_column not in frame or reward_column not in frame:
            continue
        frame = frame.copy()
        frame["environment_steps"] = pd.to_numeric(frame[steps_column], errors="coerce")
        if map_number in map_order:
            stage_span = float(frame["environment_steps"].max())
            frame["environment_steps"] += map_order[map_number] * stage_span
        frame["episode_reward"] = pd.to_numeric(frame[reward_column], errors="coerce")
        frame["method"] = method
        frame["training_seed"] = seed
        frames.append(frame[["method", "training_seed", "environment_steps", "episode_reward"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_curves(training_root, out_dir):
    frame = training_curve_data(training_root)
    if frame.empty or frame["training_seed"].nunique() < 2:
        return False
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    for method, method_frame in frame.groupby("method"):
        max_step = int(method_frame["environment_steps"].max())
        grid = np.linspace(0, max_step, 200)
        curves = []
        for _seed, seed_frame in method_frame.groupby("training_seed"):
            seed_frame = seed_frame.dropna().sort_values("environment_steps")
            if len(seed_frame) >= 2:
                curves.append(np.interp(grid, seed_frame["environment_steps"], seed_frame["episode_reward"]))
        if len(curves) < 2:
            continue
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=1)
        ax.plot(grid, mean, label=method)
        ax.fill_between(grid, mean - std, mean + std, alpha=0.15)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, ncol=2)
    save_figure(fig, out_dir, "training_curves_seed_bands")
    return True


def plot_scenario_graph(records, out_dir):
    frame = pd.DataFrame(records)
    required = {"graph_architecture", "scenario", "agent_success_rate", "training_seed"}
    if frame.empty or not required.issubset(frame.columns) or frame["graph_architecture"].nunique() < 2:
        return False
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str).str.startswith("micro_")]
    if "method" in frame:
        frame = frame[frame["method"].isin(["full", "single_graph"])]
    frame["agent_success_rate"] = frame["agent_success_rate"].map(finite_float)
    per_seed = frame.groupby(
        ["graph_architecture", "scenario", "training_seed"], as_index=False
    )["agent_success_rate"].mean()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    sns.barplot(data=per_seed, x="scenario", y="agent_success_rate", hue="graph_architecture", ax=ax)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Agent success rate")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out_dir, "single_vs_dual_by_scenario")
    return True


def plot_lambda(records, out_dir):
    frame = pd.DataFrame(records)
    required = {"counterfactual_advantage_coef", "agent_success_rate", "training_seed"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    if "method" in frame:
        frame = frame[
            frame["method"].astype(str).str.startswith("lambda_")
            | frame["method"].isin(["full", "no_cf"])
        ]
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str) == "core"]
    frame["lambda_cf"] = frame["counterfactual_advantage_coef"].map(finite_float)
    frame["agent_success_rate"] = frame["agent_success_rate"].map(finite_float)
    frame = frame.dropna(subset=["lambda_cf", "agent_success_rate"])
    if frame["lambda_cf"].nunique() < 3:
        return False
    per_seed = frame.groupby(["lambda_cf", "training_seed"], as_index=False)["agent_success_rate"].mean()
    summary = per_seed.groupby("lambda_cf")["agent_success_rate"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(summary["lambda_cf"], summary["mean"], yerr=summary["std"], marker="o", capsize=3)
    ax.set_xlabel("lambda_cf")
    ax.set_ylabel("Agent success rate")
    ax.grid(alpha=0.2)
    save_figure(fig, out_dir, "lambda_cf_sensitivity")
    return True


def plot_filter(records, out_dir):
    frame = pd.DataFrame(records)
    required = {
        "shield_enable", "agent_success_rate", "collision_rate",
        "filter_action_difference", "training_seed",
    }
    if "method" in frame:
        frame = frame[frame["method"].isin(["full", "full_filter_off"])]
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str) == "core"]
    if frame.empty or not required.issubset(frame.columns) or frame["shield_enable"].nunique() < 2:
        return False
    melted = frame.melt(
        id_vars=["shield_enable", "training_seed"],
        value_vars=["agent_success_rate", "collision_rate", "filter_action_difference"],
        var_name="metric", value_name="value",
    )
    melted["value"] = melted["value"].map(finite_float)
    per_seed = melted.groupby(["shield_enable", "training_seed", "metric"], as_index=False)["value"].mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    sns.barplot(data=per_seed, x="metric", y="value", hue="shield_enable", ax=ax)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, out_dir, "safety_filter_intervention_analysis")
    return True


def factor_plot(records, out_dir, factor, stem, x_label):
    frame = pd.DataFrame(records)
    required = {factor, "agent_success_rate", "collision_rate", "training_seed"}
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str).str.startswith("team_n")]
    if frame.empty or not required.issubset(frame.columns) or frame[factor].nunique() < 2:
        return False
    melted = frame.melt(
        id_vars=[factor, "training_seed"],
        value_vars=["agent_success_rate", "collision_rate"],
        var_name="metric", value_name="value",
    )
    melted[factor] = pd.to_numeric(melted[factor], errors="coerce")
    melted["value"] = melted["value"].map(finite_float)
    per_seed = melted.groupby([factor, "training_seed", "metric"], as_index=False)["value"].mean()
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    sns.lineplot(data=per_seed, x=factor, y="value", hue="metric", marker="o", errorbar="sd", ax=ax)
    ax.set_xlabel(x_label)
    ax.grid(alpha=0.2)
    save_figure(fig, out_dir, stem)
    return True


def plot_speed_count_heatmap(records, out_dir):
    frame = pd.DataFrame(records)
    required = {"obs_speed", "num_dynamic_obstacles", "agent_success_rate", "training_seed"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str).str.startswith("count")]
    for column in ("obs_speed", "num_dynamic_obstacles", "agent_success_rate"):
        frame[column] = frame[column].map(finite_float)
    frame = frame.dropna(subset=list(required - {"training_seed"}))
    if frame["obs_speed"].nunique() < 2 or frame["num_dynamic_obstacles"].nunique() < 2:
        return False
    per_seed = frame.groupby(
        ["obs_speed", "num_dynamic_obstacles", "training_seed"], as_index=False
    )["agent_success_rate"].mean()
    matrix = per_seed.groupby(["obs_speed", "num_dynamic_obstacles"])["agent_success_rate"].mean().unstack()
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_xlabel("Dynamic obstacle count")
    ax.set_ylabel("Obstacle speed (m/s)")
    save_figure(fig, out_dir, "speed_count_interaction_heatmap")
    return True


def load_resource_scaling(paths):
    rows = []
    for path in paths or []:
        path = Path(path)
        team_match = re.search(r"team_n(\d+)", path.as_posix())
        seed_match = re.search(r"/seed(\d+)(?:/|$)", path.as_posix())
        if not team_match or not seed_match:
            continue
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty:
            continue
        if "process_rss_kb" not in frame:
            continue
        gpu_column = (
            "process_gpu_memory_mb" if "process_gpu_memory_mb" in frame
            else "gpu_memory_total_mb" if "gpu_memory_total_mb" in frame
            else None
        )
        rows.append({
            "num_agents": int(team_match.group(1)),
            "training_seed": int(seed_match.group(1)),
            "peak_process_rss_mb": pd.to_numeric(
                frame.get("process_rss_kb"), errors="coerce"
            ).max() / 1024.0,
            "peak_process_gpu_memory_mb": (
                pd.to_numeric(frame[gpu_column], errors="coerce").max()
                if gpu_column else float("nan")
            ),
        })
    return pd.DataFrame(rows)


def plot_runtime(records, resource_files, out_dir):
    frame = pd.DataFrame(records)
    factor = "num_agents"
    policy_latency = "policy_only_inference_latency_ms_mean"
    filter_latency = "safety_filter_latency_ms_mean"
    rtf_metric = "real_time_factor"
    latency_metrics = [policy_latency, filter_latency]
    required = {factor, *latency_metrics, rtf_metric, "training_seed"}
    if "experiment_condition" in frame:
        frame = frame[frame["experiment_condition"].astype(str).str.startswith("team_n")]
    if frame.empty or not required.issubset(frame.columns) or frame[factor].nunique() < 3:
        return False
    for metric in latency_metrics:
        frame[metric] = frame[metric].map(finite_float)
    frame[rtf_metric] = frame[rtf_metric].map(finite_float)
    per_seed = frame.groupby([factor, "training_seed"], as_index=False)[
        [*latency_metrics, rtf_metric]
    ].mean()
    resources = load_resource_scaling(resource_files)
    summary = per_seed.copy()
    if not resources.empty:
        summary = summary.merge(resources, on=[factor, "training_seed"], how="left")
    summary.to_csv(out_dir / "runtime_memory_scaling.csv", index=False)

    panels = 2 + int(not resources.empty)
    fig, axes = plt.subplots(1, panels, figsize=(6.8 * panels, 4.2), squeeze=False)
    ax = axes[0, 0]
    latency_long = per_seed.melt(
        id_vars=[factor, "training_seed"],
        value_vars=latency_metrics,
        var_name="latency_layer", value_name="latency_ms",
    )
    sns.lineplot(
        data=latency_long, x=factor, y="latency_ms", hue="latency_layer",
        marker="o", errorbar="sd", ax=ax,
    )
    ax.set_xlabel("Team size")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.2)
    ax = axes[0, 1]
    sns.lineplot(
        data=per_seed, x=factor, y=rtf_metric, marker="o", errorbar="sd", ax=ax
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Team size")
    ax.set_ylabel("Real-time factor")
    ax.grid(alpha=0.2)
    if not resources.empty:
        memory = resources.melt(
            id_vars=[factor, "training_seed"],
            value_vars=["peak_process_rss_mb", "peak_process_gpu_memory_mb"],
            var_name="memory_type", value_name="memory_mb",
        ).dropna(subset=["memory_mb"])
        ax = axes[0, 2]
        sns.lineplot(
            data=memory, x=factor, y="memory_mb", hue="memory_type",
            marker="o", errorbar="sd", ax=ax,
        )
        ax.set_xlabel("Team size")
        ax.set_ylabel("Peak memory (MB)")
        ax.grid(alpha=0.2)
    save_figure(fig, out_dir, "runtime_scaling")
    return True


def open_trace(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else path.open("r", encoding="utf-8")


def plot_failures(trace_paths, out_dir):
    trajectories = defaultdict(list)
    failed = set()
    for path in trace_paths:
        with open_trace(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                x, y = finite_float(row.get("x")), finite_float(row.get("y"))
                if x is None or y is None:
                    continue
                key = (
                    str(row.get("method")), int(row.get("map_number", -1)),
                    int(row.get("episode", -1)), str(row.get("agent")),
                )
                trajectories[key].append((int(row.get("step", 0)), x, y))
                if str(row.get("event")) == "collision":
                    failed.add(key[:3])
    selected = sorted(failed)[:6]
    if not selected:
        return False
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), squeeze=False)
    for ax, episode_key in zip(axes.flat, selected):
        for key, points in trajectories.items():
            if key[:3] != episode_key:
                continue
            points.sort()
            ax.plot([point[1] for point in points], [point[2] for point in points], label=key[3])
        ax.set_title(f"{episode_key[0]} M{episode_key[1]} ep{episode_key[2]}", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)
    save_figure(fig, out_dir, "failure_case_trajectories")
    return True


def copy_cf_figures(cf_dir, out_dir):
    found = set()
    if not cf_dir:
        return found
    for name, stem in (
        ("gate_activation_vs_risk", "gate_activation_vs_risk"),
        ("cf_credit_distribution", "credit_distribution_and_attribution"),
    ):
        for suffix in ("png", "pdf"):
            source = cf_dir / f"{name}.{suffix}"
            if source.is_file():
                shutil.copy2(source, out_dir / f"{stem}.{suffix}")
                found.add(stem)
    ranking = cf_dir / "cf_credit_attribution_ranking.csv"
    if ranking.is_file():
        shutil.copy2(ranking, out_dir / ranking.name)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--core-episodes", nargs="*", type=Path, default=[])
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--cf-diagnosis-dir", type=Path)
    parser.add_argument("--cf-traces", nargs="*", type=Path, default=[])
    parser.add_argument("--resource-files", nargs="*", type=Path, default=[])
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.episodes)
    core_records = load_records(args.core_episodes) if args.core_episodes else records

    status = {
        "main_performance_table": write_tables(core_records, args.out_dir),
        "ablation_table": (args.out_dir / "ablation_table.csv").is_file(),
        "training_curves_seed_bands": (
            plot_training_curves(args.training_root, args.out_dir) if args.training_root else False
        ),
        "single_vs_dual_by_scenario": plot_scenario_graph(records, args.out_dir),
        "lambda_cf_sensitivity": plot_lambda(records, args.out_dir),
        "safety_filter_intervention_analysis": plot_filter(records, args.out_dir),
        "team_size_generalization": factor_plot(
            records, args.out_dir, "num_agents", "team_size_generalization", "Team size"
        ),
        "speed_count_interaction_heatmap": plot_speed_count_heatmap(records, args.out_dir),
        "runtime_scaling": plot_runtime(records, args.resource_files, args.out_dir),
        "failure_case_trajectories": plot_failures(args.cf_traces, args.out_dir),
    }
    copied = copy_cf_figures(args.cf_diagnosis_dir, args.out_dir)
    status["gate_activation_vs_risk"] = "gate_activation_vs_risk" in copied
    status["credit_distribution_and_attribution"] = "credit_distribution_and_attribution" in copied
    with (args.out_dir / "artifact_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["artifact", "generated"])
        writer.writerows(sorted((name, int(value)) for name, value in status.items()))
    missing = [name for name, generated in status.items() if not generated]
    print(f"generated {sum(status.values())}/{len(status)} required artifacts; missing={missing}")


if __name__ == "__main__":
    main()
