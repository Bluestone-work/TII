#!/usr/bin/env python3
"""Plot spatial risk-field heatmaps from per-step evaluation logs.

Reads the .npz written by test_gnn_mappo.py --capture_attention (which also
stores per-step robot positions + risk metrics) and bins them onto a spatial
grid over the map, producing "where on the map is it dangerous" heatmaps:

  - dyn_ttc_risk   : mean dynamic-obstacle TTC risk per cell (higher = riskier)
  - min_dist       : mean min separation to any agent/obstacle (lower = riskier)
  - front_min      : mean forward clearance per cell (lower = riskier)
  - visitation     : sample count per cell (exposure / congestion)

Grid extent is derived from the observed positions so it works on any map.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_risk(npz_path: Path):
    data = np.load(npz_path, allow_pickle=False)
    risk = json.loads(str(data["risk_records"])) if "risk_records" in data else []
    meta = {
        "map_number": int(data["map_number"]) if "map_number" in data else -1,
        "num_agents": int(data["num_agents"]) if "num_agents" in data else 0,
        "scenario": str(data["scenario"]) if "scenario" in data else "",
    }
    return risk, meta


def inlier_mask(xs, ys, k=3.0):
    """IQR-based outlier rejection per axis. Drops sentinel/parked positions
    (e.g. done agents parked at (100,100)) that would blow up the map extent.
    Robust because the real trajectory bulk dominates Q1..Q3 regardless of how
    many sentinels there are."""
    mask = np.ones(len(xs), dtype=bool)
    for arr in (xs, ys):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = max(q3 - q1, 1e-6)
        lo, hi = q1 - k * iqr, q3 + k * iqr
        mask &= (arr >= lo) & (arr <= hi)
    return mask


def _extent(xs, ys, pad=0.5):
    xmin, xmax = float(np.min(xs)), float(np.max(xs))
    ymin, ymax = float(np.min(ys)), float(np.max(ys))
    if xmax - xmin < 1e-3:
        xmin, xmax = xmin - 1.0, xmax + 1.0
    if ymax - ymin < 1e-3:
        ymin, ymax = ymin - 1.0, ymax + 1.0
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def bin_field(xs, ys, vals, extent, bins, *, reduce="mean"):
    """Bin (x,y,val) onto a grid. reduce='mean'|'min'|'count'. Empty cells -> nan (mean/min)."""
    xmin, xmax, ymin, ymax = extent
    xedges = np.linspace(xmin, xmax, bins + 1)
    yedges = np.linspace(ymin, ymax, bins + 1)
    ix = np.clip(np.digitize(xs, xedges) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(ys, yedges) - 1, 0, bins - 1)

    count = np.zeros((bins, bins), dtype=np.float64)
    for a, b in zip(ix, iy):
        count[a, b] += 1.0
    if reduce == "count":
        return count.T, xedges, yedges  # transpose so imshow rows=y

    acc = np.full((bins, bins), np.nan, dtype=np.float64)
    for a, b, v in zip(ix, iy, vals):
        if not np.isfinite(v):
            continue
        if reduce == "mean":
            acc[a, b] = v if np.isnan(acc[a, b]) else acc[a, b] + v
        elif reduce == "min":
            acc[a, b] = v if np.isnan(acc[a, b]) else min(acc[a, b], v)
    if reduce == "mean":
        with np.errstate(invalid="ignore"):
            nz = count > 0
            acc[nz] = acc[nz] / count[nz]
    return acc.T, xedges, yedges


def plot_field(field, extent, out_path: Path, *, title: str, cbar_label: str,
               cmap: str, vmax=None, vmin=0.0) -> None:
    xmin, xmax, ymin, ymax = extent
    masked = np.ma.masked_invalid(field)
    if masked.count() == 0:
        return
    hi = vmax if vmax is not None else float(np.nanpercentile(field, 99.0))
    hi = max(hi, 1e-6)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#eeeeee")  # empty cells = light gray (never visited)
    im = ax.imshow(masked, origin="lower", extent=(xmin, xmax, ymin, ymax),
                   cmap=cm, vmin=vmin, vmax=hi, aspect="equal", interpolation="nearest")
    ax.set_xlabel("map x (m)")
    ax.set_ylabel("map y (m)")
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="risk .npz from test_gnn_mappo.py --capture_attention")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--bins", type=int, default=40, help="grid resolution per axis")
    args = parser.parse_args()

    risk, meta = load_risk(args.npz)
    if not risk:
        print(f"[plot_risk_field] no risk records in {args.npz}")
        return
    out_dir = args.out_dir or args.npz.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (meta["scenario"] or f"map{meta['map_number']}")

    xs = np.array([r["x"] for r in risk], dtype=np.float64)
    ys = np.array([r["y"] for r in risk], dtype=np.float64)
    keep = inlier_mask(xs, ys)
    dropped = int((~keep).sum())
    if dropped:
        print(f"[plot_risk_field] dropped {dropped}/{len(xs)} outlier/parked positions "
              f"({100.0 * dropped / len(xs):.1f}%) before binning")
    xs, ys = xs[keep], ys[keep]
    extent = _extent(xs, ys)

    def col(name):
        vals = np.array([r.get(name, np.nan) for r in risk], dtype=np.float64)
        return vals[keep]

    # 1. dynamic-obstacle TTC risk (higher = more dangerous)
    f, _, _ = bin_field(xs, ys, col("dyn_ttc_risk"), extent, args.bins, reduce="mean")
    plot_field(f, extent, out_dir / f"{prefix}_riskfield_dyn_ttc.png",
               title=f"{prefix} dynamic-obstacle TTC risk field",
               cbar_label="mean dyn TTC risk", cmap="inferno")

    # 2. min separation (lower = more dangerous) -> min reduce
    f, _, _ = bin_field(xs, ys, col("min_dist"), extent, args.bins, reduce="min")
    plot_field(f, extent, out_dir / f"{prefix}_riskfield_min_sep.png",
               title=f"{prefix} minimum separation field",
               cbar_label="min separation (m)", cmap="viridis")

    # 3. forward clearance (lower = tighter)
    f, _, _ = bin_field(xs, ys, col("front_min"), extent, args.bins, reduce="mean")
    plot_field(f, extent, out_dir / f"{prefix}_riskfield_front_clearance.png",
               title=f"{prefix} forward clearance field",
               cbar_label="mean front clearance (m)", cmap="viridis")

    # 4. visitation density (exposure / congestion)
    f, _, _ = bin_field(xs, ys, col("dyn_ttc_risk"), extent, args.bins, reduce="count")
    plot_field(f, extent, out_dir / f"{prefix}_riskfield_visitation.png",
               title=f"{prefix} visitation density",
               cbar_label="samples per cell", cmap="magma")

    print(f"[OK] wrote 4 risk-field heatmaps for {prefix} to {out_dir}")


if __name__ == "__main__":
    main()
