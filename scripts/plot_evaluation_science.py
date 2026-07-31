#!/usr/bin/env python3
"""Create distribution, outcome, trajectory, and density figures from cross-map evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MAP_LABELS = {
    3: "Map 3\nCorridor",
    4: "Map 4\nIntersection",
    5: "Map 5\nWarehouse",
    9: "Map 9\nDynamic warehouse",
}
MAP_ORDER = [3, 4, 5, 9]
COLORS = ["#2878B5", "#E88B2D", "#3FA65B", "#C94C4C"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def finite(value, default=np.nan):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def records_by_map(records: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        try:
            groups[int(record.get("map_number"))].append(record)
        except (TypeError, ValueError):
            continue
    return groups


def completion(record: dict) -> float:
    return finite(record.get("reached_agents"), 0.0) / max(1, int(record.get("num_agents") or 1))


def collision_fraction(record: dict) -> float:
    return finite(record.get("collided_agents"), 0.0) / max(1, int(record.get("num_agents") or 1))


def plot_distributions(groups: dict[int, list[dict]], out_dir: Path) -> None:
    specs = [
        ("Agent completion fraction", completion, (0.0, 1.05)),
        ("Agent collision fraction", collision_fraction, (0.0, 1.05)),
        ("Episode steps", lambda r: finite(r.get("steps")), None),
        ("Mean team reward", lambda r: finite(r.get("avg_reward")), None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200)
    for ax, (title, fn, ylim) in zip(axes.flat, specs):
        data = [np.asarray([fn(r) for r in groups.get(m, [])], dtype=float) for m in MAP_ORDER]
        data = [v[np.isfinite(v)] if np.any(np.isfinite(v)) else np.array([np.nan]) for v in data]
        box = ax.boxplot(data, patch_artist=True, tick_labels=[MAP_LABELS[m] for m in MAP_ORDER], showmeans=True)
        for patch, color in zip(box["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        if ylim:
            ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(out_dir / "episode_metric_distributions.png", bbox_inches="tight")
    fig.savefig(out_dir / "episode_metric_distributions.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_outcomes(groups: dict[int, list[dict]], out_dir: Path) -> None:
    agent_success, team_success, collision, timeout = [], [], [], []
    rows = []
    for map_number in MAP_ORDER:
        items = groups.get(map_number, [])
        n = max(1, len(items))
        agent = 100.0 * float(np.mean([completion(r) for r in items])) if items else 0.0
        team = 100.0 * float(np.mean([completion(r) >= 1.0 for r in items])) if items else 0.0
        coll = 100.0 * float(np.mean([collision_fraction(r) for r in items])) if items else 0.0
        tout = 100.0 * float(np.mean([bool(r.get("truncated")) for r in items])) if items else 0.0
        agent_success.append(agent)
        team_success.append(team)
        collision.append(coll)
        timeout.append(tout)
        rows.append((map_number, n, agent, team, coll, tout))
    x = np.arange(len(MAP_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    for offset, values, label, color in [
        (-1.5 * width, agent_success, "Agent success", "#2878B5"),
        (-0.5 * width, team_success, "Full-team success", "#3FA65B"),
        (0.5 * width, collision, "Agent collision", "#C94C4C"),
        (1.5 * width, timeout, "Timeout episodes", "#8A6FB0"),
    ]:
        ax.bar(x + offset, values, width, label=label, color=color, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x, [MAP_LABELS[m] for m in MAP_ORDER])
    ax.set_ylim(0.0, 105.0)
    ax.set_ylabel("Rate (%)")
    ax.set_title("Outcome rates across the specified test environments")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "outcome_rates.png", bbox_inches="tight")
    fig.savefig(out_dir / "outcome_rates.pdf", bbox_inches="tight")
    plt.close(fig)
    with (out_dir / "outcome_rates.csv").open("w", encoding="utf-8") as handle:
        handle.write("map_number,n_episodes,agent_success_pct,team_success_pct,agent_collision_pct,timeout_episode_pct\n")
        for row in rows:
            handle.write("{},{},{:.6f},{:.6f},{:.6f},{:.6f}\n".format(*row))


def load_risk(npz_path: Path) -> list[dict]:
    try:
        data = np.load(npz_path, allow_pickle=False)
        return json.loads(str(data["risk_records"])) if "risk_records" in data else []
    except Exception:
        return []


def in_bounds(records: list[dict]) -> list[dict]:
    out = []
    for record in records:
        x, y = finite(record.get("x")), finite(record.get("y"))
        if math.isfinite(x) and math.isfinite(y) and abs(x) < 20.0 and abs(y) < 20.0:
            out.append(record)
    return out


def plot_density_and_trajectories(groups: dict[int, list[dict]], risk_dir: Path, out_dir: Path) -> None:
    fig_density, axes_density = plt.subplots(2, 2, figsize=(10, 9), dpi=200)
    fig_paths, axes_paths = plt.subplots(2, 2, figsize=(10, 9), dpi=200)
    for ax_density, ax_paths, map_number in zip(axes_density.flat, axes_paths.flat, MAP_ORDER):
        label = {3: "map3_corridor_swap", 4: "map4_intersection", 5: "map5_warehouse_aisles", 9: "map9_warehouse_dynamic"}[map_number]
        risk = in_bounds(load_risk(risk_dir / f"{label}_attn_risk.npz"))
        if not risk:
            for ax in (ax_density, ax_paths):
                ax.set_title(f"{MAP_LABELS[map_number]}: no spatial samples")
                ax.set_axis_off()
            continue
        xs = np.asarray([finite(r.get("x")) for r in risk])
        ys = np.asarray([finite(r.get("y")) for r in risk])
        pad = 0.45
        extent = (xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad)
        hist, xedges, yedges = np.histogram2d(xs, ys, bins=45, range=[[extent[0], extent[1]], [extent[2], extent[3]]])
        image = ax_density.imshow(np.log1p(hist.T), origin="lower", extent=extent, cmap="magma", aspect="equal")
        ax_density.set_title(f"{MAP_LABELS[map_number]} trajectory density")
        ax_density.set_xlabel("map x (m)")
        ax_density.set_ylabel("map y (m)")
        fig_density.colorbar(image, ax=ax_density, fraction=0.046, pad=0.04, label="log(1 + samples)")

        items = groups.get(map_number, [])
        target_ep = 0
        if items:
            comps = np.asarray([completion(r) for r in items])
            target_ep = int(items[int(np.argmin(np.abs(comps - np.median(comps))))].get("episode_index", 0))
        ep_rows = [r for r in risk if int(r.get("episode", -1)) == target_ep]
        for idx, agent in enumerate(sorted({str(r.get("agent")) for r in ep_rows})):
            path = sorted((r for r in ep_rows if str(r.get("agent")) == agent), key=lambda r: int(r.get("step", 0)))
            px = [finite(r.get("x")) for r in path]
            py = [finite(r.get("y")) for r in path]
            if px:
                ax_paths.plot(px, py, linewidth=1.4, color=COLORS[idx % len(COLORS)], label=agent)
                ax_paths.scatter(px[0], py[0], marker="o", s=22, color=COLORS[idx % len(COLORS)])
                ax_paths.scatter(px[-1], py[-1], marker="x", s=28, color=COLORS[idx % len(COLORS)])
        ax_paths.set_title(f"{MAP_LABELS[map_number]} representative episode {target_ep}")
        ax_paths.set_xlabel("map x (m)")
        ax_paths.set_ylabel("map y (m)")
        ax_paths.set_aspect("equal", adjustable="box")
        ax_paths.grid(alpha=0.25)
        if ep_rows:
            ax_paths.legend(fontsize=7, ncol=2)
    fig_density.tight_layout()
    fig_density.savefig(out_dir / "trajectory_density_heatmaps.png", bbox_inches="tight")
    fig_density.savefig(out_dir / "trajectory_density_heatmaps.pdf", bbox_inches="tight")
    plt.close(fig_density)
    fig_paths.tight_layout()
    fig_paths.savefig(out_dir / "representative_trajectories.png", bbox_inches="tight")
    fig_paths.savefig(out_dir / "representative_trajectories.pdf", bbox_inches="tight")
    plt.close(fig_paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episodes_jsonl", type=Path)
    parser.add_argument("--risk-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    records = load_jsonl(args.episodes_jsonl)
    if not records:
        raise SystemExit("No valid episode records found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    groups = records_by_map(records)
    plot_distributions(groups, args.out_dir)
    plot_outcomes(groups, args.out_dir)
    plot_density_and_trajectories(groups, args.risk_dir, args.out_dir)
    print(f"wrote scientific evaluation figures to {args.out_dir}")


if __name__ == "__main__":
    main()
