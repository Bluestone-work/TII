#!/usr/bin/env python3
"""Compute paired safety-filter effects with training seed as the repeat unit."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = ("agent_success_rate", "team_success", "collision_rate", "timeout_rate", "spl", "minimum_clearance")


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    paired = defaultdict(dict)
    for row in rows:
        key = (str(row["method"]), int(row["training_seed"]), str(row["scenario_case"]), int(row["episode_index"]))
        paired[key][str(row["experiment_condition"])] = row
    seed_rows = []
    by_seed = defaultdict(lambda: defaultdict(list))
    for (method, seed, _case, _episode), pair in paired.items():
        if set(pair) != {"filter_off", "filter_on"}:
            continue
        for metric in METRICS:
            off, on = finite(pair["filter_off"].get(metric)), finite(pair["filter_on"].get(metric))
            if off is not None and on is not None:
                by_seed[(method, seed)][metric].append(on - off)
    for (method, seed), metrics in sorted(by_seed.items()):
        for metric, values in metrics.items():
            seed_rows.append({"method": method, "training_seed": seed, "metric": metric, "paired_delta_on_minus_off": float(np.mean(values)), "n_paired_episodes": len(values)})
    summary = []
    grouped = defaultdict(list)
    for row in seed_rows:
        grouped[(row["method"], row["metric"])].append(row["paired_delta_on_minus_off"])
    for (method, metric), values in sorted(grouped.items()):
        summary.append({"method": method, "metric": metric, "n_training_seeds": len(values), "mean_paired_delta_on_minus_off": float(np.mean(values)), "std_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "filter_paired_by_seed.csv", seed_rows)
    write_csv(args.out_dir / "filter_paired_summary.csv", summary)
    print(f"wrote safety-filter paired analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
