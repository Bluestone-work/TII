#!/usr/bin/env python3
"""Summarize fixed-weight, fixed-state critic interventions from evaluation traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np


STRATEGIES = ("zero_agent", "mean_agent", "zero_social", "zero_obstacle")


def lines(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        yield from handle


def write_csv(path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    grouped = defaultdict(lambda: defaultdict(list))
    identity_errors = 0
    for path in args.inputs:
        for line in lines(path):
            if not line.strip():
                continue
            row = json.loads(line)
            method, seed = str(row.get("method")), int(row.get("training_seed"))
            v_full = row.get("v_full")
            if v_full is None:
                continue
            for strategy in STRATEGIES:
                value, delta = row.get(f"v_{strategy}"), row.get(f"delta_v_{strategy}")
                if value is None or delta is None:
                    continue
                if not math.isclose(float(delta), float(v_full) - float(value), abs_tol=1e-5):
                    identity_errors += 1
                grouped[(method, seed)][strategy].append(float(delta))
    if identity_errors:
        raise SystemExit(f"value-intervention identity failures={identity_errors}")
    seed_rows = []
    for (method, seed), strategies in sorted(grouped.items()):
        for strategy, values in strategies.items():
            array = np.asarray(values, dtype=float)
            seed_rows.append({
                "method": method, "training_seed": seed, "intervention": strategy,
                "n_states": len(array), "delta_v_mean": float(array.mean()),
                "delta_v_abs_mean": float(np.abs(array).mean()),
                "delta_v_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
                "positive_fraction": float(np.mean(array > 0.0)),
            })
    summary = []
    across = defaultdict(list)
    for row in seed_rows:
        across[(row["method"], row["intervention"])].append(row["delta_v_abs_mean"])
    for (method, strategy), values in sorted(across.items()):
        summary.append({
            "method": method, "intervention": strategy, "n_training_seeds": len(values),
            "mean_abs_delta_v_across_seeds": float(np.mean(values)),
            "std_abs_delta_v_across_seeds": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "value_intervention_by_seed.csv", seed_rows)
    write_csv(args.out_dir / "value_intervention_summary.csv", summary)
    (args.out_dir / "INTERPRETATION.md").write_text(
        "# Controlled value intervention\n\n"
        "Each delta is recomputed on the same observed state and the same frozen critic weights; "
        "only the named global-state component is changed. This supports a model-sensitivity claim, "
        "not an unrestricted causal claim about navigation outcomes.\n",
        encoding="utf-8",
    )
    print(f"wrote value-intervention analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
