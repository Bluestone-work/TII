#!/usr/bin/env python3
"""Plot local occupancy maps used by the GNN-MAPPO observation.

The training environment builds a 32x32 ego-centric occupancy grid from each
LaserScan and stacks the latest 4 frames. This script mirrors that logic and
exports publication-friendly figures:

  1. single-frame local occupancy heatmap
  2. four-frame temporal sequence
  3. accumulated obstacle-density heatmap

It can subscribe to a live ROS2 LaserScan topic or render from a previously
saved .npz capture.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def scan_to_grid(
    ranges: np.ndarray,
    angles: np.ndarray,
    *,
    grid_size: int = 32,
    grid_range: float = 4.0,
    scan_valid_min: float = 0.05,
) -> np.ndarray:
    """Convert LaserScan ranges to a binary ego-centric occupancy grid.

    This intentionally follows IndependentRobotEnv._build_local_map_obs:
    valid lidar endpoints are written into a 32x32 grid in the robot body frame.
    The model observes only obstacle endpoints, not a ray-traced free-space map.
    """

    cell_size = grid_range / grid_size
    half = grid_size // 2
    max_r = grid_range / 2.0

    ranges = np.asarray(ranges, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    ranges = np.nan_to_num(ranges, nan=max_r, posinf=max_r, neginf=0.0)

    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    valid = (ranges > scan_valid_min) & (ranges <= max_r)
    if not np.any(valid):
        return grid

    xs = ranges[valid] * np.cos(angles[valid])
    ys = ranges[valid] * np.sin(angles[valid])
    gxs = (xs / cell_size + half).astype(np.int32)
    gys = (ys / cell_size + half).astype(np.int32)
    in_bounds = (gxs >= 0) & (gxs < grid_size) & (gys >= 0) & (gys < grid_size)
    grid[gxs[in_bounds], gys[in_bounds]] = 1.0
    return grid


def grid_for_display(grid: np.ndarray) -> np.ndarray:
    """Return image array with forward direction upward and left to the left."""

    # Internal indexing is grid[x_forward, y_left]. Display rows should increase
    # upward in x, columns should increase leftward in y.
    return np.flipud(grid.T)


def draw_robot_marker(ax, grid_range: float, *, show_forward_label: bool = True) -> None:
    ax.scatter([0.0], [0.0], marker="o", s=45, c="#111111", edgecolors="white", linewidths=0.8, zorder=5)
    ax.annotate(
        "",
        xy=(0.0, 0.55),
        xytext=(0.0, 0.10),
        arrowprops={"arrowstyle": "-|>", "lw": 1.4, "color": "#111111"},
        zorder=6,
    )
    if show_forward_label:
        ax.text(0.08, 0.56, "forward", fontsize=8, color="#111111", va="center")
    ax.set_xlim(-grid_range / 2.0, grid_range / 2.0)
    ax.set_ylim(-grid_range / 2.0, grid_range / 2.0)


def style_axis(
    ax,
    title: str,
    grid_range: float,
    *,
    show_ylabel: bool = True,
    show_forward_label: bool = True,
) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("lateral y (m)")
    if show_ylabel:
        ax.set_ylabel("forward x (m)")
    else:
        ax.set_ylabel("")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d8d8d8", linewidth=0.35, alpha=0.45)
    draw_robot_marker(ax, grid_range, show_forward_label=show_forward_label)


def plot_single(grid: np.ndarray, out_path: Path, *, grid_range: float, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=220)
    im = ax.imshow(
        grid_for_display(grid),
        origin="lower",
        extent=(-grid_range / 2.0, grid_range / 2.0, -grid_range / 2.0, grid_range / 2.0),
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    style_axis(ax, title, grid_range)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("occupied endpoint")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sequence(frames: np.ndarray, out_path: Path, *, grid_range: float, title: str) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(3.2 * len(frames), 3.3), dpi=220)
    if len(frames) == 1:
        axes = [axes]

    last_im = None
    for i, (ax, grid) in enumerate(zip(axes, frames)):
        last_im = ax.imshow(
            grid_for_display(grid),
            origin="lower",
            extent=(-grid_range / 2.0, grid_range / 2.0, -grid_range / 2.0, grid_range / 2.0),
        cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        style_axis(
            ax,
            f"t-{len(frames) - 1 - i}",
            grid_range,
            show_ylabel=(i == 0),
            show_forward_label=(i == 0),
        )

    fig.suptitle(title, fontsize=12, y=1.02)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=list(axes), fraction=0.025, pad=0.025)
        cbar.set_label("occupied endpoint")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_density(grids: np.ndarray, out_path: Path, *, grid_range: float, title: str) -> np.ndarray:
    density = np.mean(grids.astype(np.float32), axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=220)
    im = ax.imshow(
        grid_for_display(density),
        origin="lower",
        extent=(-grid_range / 2.0, grid_range / 2.0, -grid_range / 2.0, grid_range / 2.0),
        cmap="inferno",
        vmin=0.0,
        vmax=max(float(np.percentile(density, 99.0)), 1e-6),
        interpolation="nearest",
    )
    style_axis(ax, title, grid_range)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("occupancy frequency")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return density


def plot_temporal_montage(
    grids: np.ndarray,
    out_path: Path,
    *,
    grid_range: float,
    title: str,
    n_frames: int = 16,
) -> np.ndarray:
    n = min(int(n_frames), len(grids))
    indices = np.linspace(0, len(grids) - 1, n, dtype=np.int32) if n > 1 else np.array([len(grids) - 1])
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), dpi=180)
    axes = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n:
            ax.axis("off")
            continue
        grid = grids[indices[ax_idx]]
        ax.imshow(
            grid_for_display(grid),
            origin="lower",
            extent=(-grid_range / 2.0, grid_range / 2.0, -grid_range / 2.0, grid_range / 2.0),
            cmap="gray_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        style_axis(
            ax,
            f"frame {int(indices[ax_idx])}",
            grid_range,
            show_ylabel=(ax_idx % cols == 0),
            show_forward_label=False,
        )
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return indices


def plot_change_density(grids: np.ndarray, out_path: Path, *, grid_range: float, title: str) -> np.ndarray:
    if len(grids) < 2:
        change = np.zeros_like(grids[0], dtype=np.float32)
    else:
        change = np.mean(np.abs(np.diff(grids.astype(np.float32), axis=0)), axis=0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=220)
    im = ax.imshow(
        grid_for_display(change),
        origin="lower",
        extent=(-grid_range / 2.0, grid_range / 2.0, -grid_range / 2.0, grid_range / 2.0),
        cmap="magma",
        vmin=0.0,
        vmax=max(float(np.percentile(change, 99.0)), 1e-6),
        interpolation="nearest",
    )
    style_axis(ax, title, grid_range)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("mean frame-to-frame change")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return change


def plot_occupancy_timeline(grids: np.ndarray, out_path: Path, *, title: str) -> np.ndarray:
    occ_counts = grids.reshape(len(grids), -1).sum(axis=1).astype(np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=220)
    ax.plot(np.arange(len(occ_counts)), occ_counts, color="#3b6ea8", linewidth=1.4)
    ax.fill_between(np.arange(len(occ_counts)), occ_counts, color="#3b6ea8", alpha=0.18)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("captured scan frame")
    ax.set_ylabel("occupied cells")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return occ_counts


def select_consecutive_window(grids: np.ndarray, frames: int) -> np.ndarray:
    """Pick `frames` CONSECUTIVE indices representing the model's stacked observation.

    The training env stacks the latest 4 consecutive lidar frames in a
    deque(maxlen=4), so the temporal-sequence figure must show consecutive
    frames — not frames sampled across the whole rollout. To make the figure
    informative (and to avoid landing on the static tail after the robot has
    stopped), we choose the consecutive window with the largest frame-to-frame
    motion.
    """
    n = len(grids)
    if n <= frames:
        return np.arange(n, dtype=np.int32)

    flat = grids.reshape(n, -1).astype(np.float32)
    # Per-transition change magnitude between consecutive captured frames.
    per_step_change = np.abs(np.diff(flat, axis=0)).sum(axis=1)  # length n-1
    # Cumulative change over each length-(frames-1) transition window; the
    # window covering [start, start+frames-1] uses transitions [start, start+frames-2].
    win_trans = frames - 1
    if win_trans <= 0:
        return np.array([n - 1], dtype=np.int32)
    csum = np.concatenate([[0.0], np.cumsum(per_step_change)])
    window_scores = csum[win_trans:] - csum[:len(csum) - win_trans]  # length n-frames+1
    start = int(np.argmax(window_scores))
    return np.arange(start, start + frames, dtype=np.int32)


def save_figures(
    grids: np.ndarray,
    out_dir: Path,
    *,
    robot: str,
    grid_range: float,
    frames: int,
    prefix: str,
    source_topic: str,
    sequence_mode: str = "consecutive",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if grids.ndim != 3 or grids.shape[0] == 0:
        raise ValueError("expected grids with shape [N, H, W]")

    if frames <= 1 or len(grids) <= frames:
        frame_indices = np.arange(len(grids), dtype=np.int32)
        selected_frames = grids
    elif sequence_mode == "spread":
        # Legacy behaviour: sample evenly across the whole rollout. NOTE: this
        # does NOT match the model's stacked observation and the later frames
        # can look identical if the robot stops moving mid-capture.
        frame_indices = np.linspace(0, len(grids) - 1, frames, dtype=np.int32)
        selected_frames = grids[frame_indices]
    else:
        # Default: consecutive frames = what the model actually stacks.
        frame_indices = select_consecutive_window(grids, frames)
        selected_frames = grids[frame_indices]
    single_path = out_dir / f"{prefix}_{robot}_local_map_single.png"
    sequence_path = out_dir / f"{prefix}_{robot}_local_map_4frame_sequence.png"
    density_path = out_dir / f"{prefix}_{robot}_local_map_density.png"
    montage_path = out_dir / f"{prefix}_{robot}_local_map_montage16.png"
    change_path = out_dir / f"{prefix}_{robot}_local_map_change_density.png"
    timeline_path = out_dir / f"{prefix}_{robot}_local_map_occupancy_timeline.png"
    npz_path = out_dir / f"{prefix}_{robot}_local_map_capture.npz"

    plot_single(grids[-1], single_path, grid_range=grid_range, title=f"{robot} local occupancy grid")
    seq_span = f"frames {int(frame_indices[0])}-{int(frame_indices[-1])}" if len(frame_indices) else ""
    plot_sequence(
        selected_frames,
        sequence_path,
        grid_range=grid_range,
        title=f"{robot} stacked local occupancy observations ({sequence_mode}, {seq_span})",
    )
    density = plot_density(
        grids,
        density_path,
        grid_range=grid_range,
        title=f"{robot} accumulated local obstacle density (N={len(grids)})",
    )
    montage_indices = plot_temporal_montage(
        grids,
        montage_path,
        grid_range=grid_range,
        title=f"{robot} rollout local occupancy montage",
        n_frames=16,
    )
    change_density = plot_change_density(
        grids,
        change_path,
        grid_range=grid_range,
        title=f"{robot} local occupancy temporal change",
    )
    occ_counts = plot_occupancy_timeline(
        grids,
        timeline_path,
        title=f"{robot} local occupancy count over rollout",
    )

    np.savez_compressed(
        npz_path,
        grids=grids.astype(np.float32),
        density=density.astype(np.float32),
        change_density=change_density.astype(np.float32),
        occupancy_counts=occ_counts.astype(np.float32),
        sequence_indices=frame_indices.astype(np.int32),
        montage_indices=montage_indices.astype(np.int32),
        robot=robot,
        source_topic=source_topic,
        grid_range=float(grid_range),
        grid_size=int(grids.shape[-1]),
    )

    print(f"[OK] saved {single_path}")
    print(f"[OK] saved {sequence_path}")
    print(f"[OK] saved {density_path}")
    print(f"[OK] saved {montage_path}")
    print(f"[OK] saved {change_path}")
    print(f"[OK] saved {timeline_path}")
    print(f"[OK] saved raw capture {npz_path}")


def collect_live_scans(args: argparse.Namespace) -> Tuple[np.ndarray, str]:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan

    topic = args.topic or f"/{args.robot}/scan"
    grids: List[np.ndarray] = []
    frame_counter = 0
    start = time.monotonic()

    class ScanCollector(Node):
        def __init__(self) -> None:
            super().__init__("local_map_plot_collector")
            self.subscription = self.create_subscription(LaserScan, topic, self.on_scan, 10)

        def on_scan(self, msg: LaserScan) -> None:
            nonlocal frame_counter
            frame_counter += 1
            if frame_counter % max(1, args.stride) != 0:
                return
            ranges = np.asarray(msg.ranges, dtype=np.float32)
            angles = msg.angle_min + np.arange(len(ranges), dtype=np.float32) * msg.angle_increment
            grids.append(
                scan_to_grid(
                    ranges,
                    angles,
                    grid_size=args.grid_size,
                    grid_range=args.grid_range,
                    scan_valid_min=args.scan_valid_min,
                )
            )
            if len(grids) % 25 == 0:
                self.get_logger().info(f"collected {len(grids)}/{args.samples} grids from {topic}")

    rclpy.init()
    node = ScanCollector()
    try:
        while rclpy.ok() and len(grids) < args.samples:
            rclpy.spin_once(node, timeout_sec=0.2)
            if args.timeout_s > 0 and time.monotonic() - start > args.timeout_s:
                if grids:
                    node.get_logger().warning(
                        f"timed out after {args.timeout_s}s; saving partial capture "
                        f"with {len(grids)} grids from {topic}"
                    )
                    break
                raise TimeoutError(f"timed out after {args.timeout_s}s while collecting 0 grids from {topic}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return np.stack(grids, axis=0), topic


def load_npz(path: Path) -> Tuple[np.ndarray, str, str]:
    data = np.load(path, allow_pickle=False)
    grids = np.asarray(data["grids"], dtype=np.float32)
    robot = str(data["robot"]) if "robot" in data else "robot"
    topic = str(data["source_topic"]) if "source_topic" in data else str(path)
    return grids, robot, topic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="tb3_0", help="Robot namespace, e.g. tb3_0")
    parser.add_argument("--topic", default="", help="LaserScan topic. Defaults to /<robot>/scan")
    parser.add_argument("--samples", type=int, default=160, help="Number of grids to collect")
    parser.add_argument("--stride", type=int, default=3, help="Use every Nth LaserScan message")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Live collection timeout")
    parser.add_argument("--grid-size", type=int, default=32, help="Occupancy grid width/height")
    parser.add_argument("--grid-range", type=float, default=4.0, help="Grid side length in meters")
    parser.add_argument("--scan-valid-min", type=float, default=0.05, help="Minimum valid laser range")
    parser.add_argument("--frames", type=int, default=4, help="Number of frames in temporal sequence figure")
    parser.add_argument(
        "--sequence-mode",
        choices=["consecutive", "spread"],
        default="consecutive",
        help="consecutive: 4 adjacent frames = the model's stacked observation (default). "
        "spread: legacy even sampling across the whole rollout (can look static).",
    )
    parser.add_argument("--out-dir", default="artifacts/local_map_obs", help="Output directory")
    parser.add_argument("--prefix", default="", help="Filename prefix. Defaults to timestamp")
    parser.add_argument("--from-npz", default="", help="Render figures from a saved capture instead of live ROS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = args.prefix or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)

    if args.from_npz:
        grids, robot, topic = load_npz(Path(args.from_npz))
        if args.grid_range == 4.0:
            try:
                args.grid_range = float(np.load(args.from_npz, allow_pickle=False)["grid_range"])
            except Exception:
                pass
    else:
        grids, topic = collect_live_scans(args)
        robot = args.robot

    save_figures(
        grids,
        out_dir,
        robot=robot,
        grid_range=args.grid_range,
        frames=args.frames,
        prefix=prefix,
        source_topic=topic,
        sequence_mode=args.sequence_mode,
    )


if __name__ == "__main__":
    main()
