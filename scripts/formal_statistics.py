#!/usr/bin/env python3
"""Seed-aware statistics for paired multi-method navigation evaluations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


DEFAULT_METRICS = (
    "agent_success_rate",
    "team_success",
    "collision_rate",
    "timeout_rate",
    "spl",
    "path_length",
    "completion_time",
    "minimum_clearance",
    "clearance_p05",
    "minimum_ttc",
    "near_collision_count",
    "deadlock",
    "oscillation_count",
    "control_jerk",
    "filter_intervention_rate",
    "filter_action_difference",
    "policy_only_inference_latency_ms_mean",
    "safety_filter_latency_ms_mean",
    "real_time_factor",
)

NON_LEARNING_METHODS = frozenset({"orca_dwa"})
PAIRING_FIELDS = frozenset({
    "initial_state_fingerprint",
    "paired_scenario_eligible",
    "pairing_basis",
})


def finite_float(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return float(value.strip().lower() == "true")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_records(paths):
    records = []
    for path in paths:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            with path.open("r", newline="", encoding="utf-8") as handle:
                records.extend(csv.DictReader(handle))
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def validate_records(records, *, require_pairing_metadata=False):
    missing = []
    missing_pairing = []
    for index, record in enumerate(records):
        if record.get("method") in (None, "") or record.get("training_seed") in (None, ""):
            missing.append(index)
        if require_pairing_metadata and not PAIRING_FIELDS.issubset(record):
            missing_pairing.append(index)
    if missing:
        preview = ",".join(str(index) for index in missing[:10])
        raise ValueError(
            "formal statistics require method and training_seed on every record; "
            f"missing rows={preview}"
        )
    if missing_pairing:
        preview = ",".join(str(index) for index in missing_pairing[:10])
        raise ValueError(
            "paired analysis requires explicit initial-state and eligibility metadata; "
            f"missing rows={preview}"
        )


def scenario_key(record):
    return (
        str(record.get("scenario") or f"map{record.get('map_number')}"),
        int(record.get("map_number") or -1),
        int(record.get("episode_index") or 0),
    )


def _truthy(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def pairing_fingerprint(record):
    fingerprint = str(record.get("initial_state_fingerprint") or "").strip()
    if not fingerprint or not _truthy(record.get("paired_scenario_eligible")):
        return None
    trajectory = str(record.get("dynamic_obstacle_trajectory_fingerprint") or "").strip()
    if int(record.get("num_dynamic_obstacles") or 0) > 0:
        if not trajectory:
            return None
        return f"{fingerprint}:{trajectory}"
    return fingerprint


def seed_metric_values(records, metric):
    grouped = defaultdict(list)
    for record in records:
        value = finite_float(record.get(metric))
        if value is None:
            continue
        grouped[(str(record["method"]), int(record["training_seed"]))].append(value)
    return {
        key: float(np.mean(values))
        for key, values in grouped.items()
        if values
    }


def summarize_across_seeds(records, metrics):
    rows = []
    for metric in metrics:
        seed_values = seed_metric_values(records, metric)
        methods = sorted({method for method, _seed in seed_values})
        for method in methods:
            if method in NON_LEARNING_METHODS:
                unique_units = {}
                for record in records:
                    if str(record.get("method")) != method:
                        continue
                    value = finite_float(record.get(metric))
                    if value is None:
                        continue
                    key = (scenario_key(record), pairing_fingerprint(record))
                    unique_units.setdefault(key, value)
                values = list(unique_units.values())
                rows.append({
                    "metric": metric,
                    "method": method,
                    "n_training_seeds": 0,
                    "mean_across_seeds": float(np.mean(values)) if values else float("nan"),
                    "std_across_seeds": float("nan"),
                })
                continue
            values = [
                value for (candidate, _seed), value in seed_values.items()
                if candidate == method
            ]
            rows.append({
                "metric": metric,
                "method": method,
                "n_training_seeds": len(values),
                "mean_across_seeds": float(np.mean(values)) if values else float("nan"),
                "std_across_seeds": float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan"),
            })
    return rows


def paired_unit_values(records, metric, method):
    grouped = defaultdict(lambda: {"values": [], "fingerprints": set()})
    for record in records:
        if str(record.get("method")) != method:
            continue
        value = finite_float(record.get(metric))
        fingerprint = pairing_fingerprint(record)
        if value is None or fingerprint is None:
            continue
        group = grouped[(int(record["training_seed"]), scenario_key(record))]
        group["values"].append(value)
        group["fingerprints"].add(fingerprint)
    return {
        key: (float(np.mean(group["values"])), next(iter(group["fingerprints"])))
        for key, group in grouped.items()
        if group["values"] and len(group["fingerprints"]) == 1
    }


def hierarchical_paired_bootstrap(
    records,
    metric,
    method,
    reference,
    *,
    samples=10000,
    seed=20260715,
):
    candidate = paired_unit_values(records, metric, method)
    baseline = paired_unit_values(records, metric, reference)
    common = sorted(
        key for key in set(candidate) & set(baseline)
        if candidate[key][1] == baseline[key][1]
    )
    by_seed = defaultdict(list)
    for training_seed, unit in common:
        by_seed[training_seed].append(
            candidate[(training_seed, unit)][0] - baseline[(training_seed, unit)][0]
        )
    seeds = sorted(seed for seed, values in by_seed.items() if values)
    if not seeds:
        return float("nan"), float("nan"), float("nan"), 0, 0
    observed = float(np.mean([np.mean(by_seed[item]) for item in seeds]))
    rng = np.random.default_rng(seed)
    draws = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        seed_differences = []
        for sampled_seed in sampled_seeds:
            values = np.asarray(by_seed[int(sampled_seed)], dtype=np.float64)
            sampled_values = rng.choice(values, size=len(values), replace=True)
            seed_differences.append(float(np.mean(sampled_values)))
        draws[index] = float(np.mean(seed_differences))
    low, high = np.percentile(draws, [2.5, 97.5])
    return observed, float(low), float(high), len(seeds), len(common)


def rank_biserial_paired(differences):
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0.0)]
    if not values.size:
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(np.sum(ranks[values > 0.0]))
    negative = float(np.sum(ranks[values < 0.0]))
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def holm_adjust(p_values):
    count = len(p_values)
    if not count:
        return []
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.ones(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def inferential_tests(records, metrics, reference, bootstrap_samples):
    methods = sorted({str(record["method"]) for record in records})
    friedman_rows = []
    pairwise_rows = []
    bootstrap_rows = []
    for metric in metrics:
        seed_values = seed_metric_values(records, metric)
        common_seeds = sorted(set.intersection(*[
            {seed for method, seed in seed_values if method == candidate}
            for candidate in methods
        ])) if methods else []
        if len(methods) >= 3 and len(common_seeds) >= 2:
            arrays = [
                [seed_values[(method, seed)] for seed in common_seeds]
                for method in methods
            ]
            statistic, p_value = stats.friedmanchisquare(*arrays)
            friedman_rows.append({
                "metric": metric,
                "methods": ";".join(methods),
                "n_paired_training_seeds": len(common_seeds),
                "statistic": float(statistic),
                "p_value": float(p_value),
            })

        metric_pair_rows = []
        if reference in methods:
            for method in methods:
                if method == reference:
                    continue
                paired_seeds = sorted(
                    {seed for candidate, seed in seed_values if candidate == method}
                    & {seed for candidate, seed in seed_values if candidate == reference}
                )
                differences = [
                    seed_values[(method, seed)] - seed_values[(reference, seed)]
                    for seed in paired_seeds
                ]
                if differences and any(value != 0.0 for value in differences):
                    statistic, p_value = stats.wilcoxon(differences, zero_method="wilcox")
                else:
                    statistic, p_value = 0.0, 1.0
                metric_pair_rows.append({
                    "metric": metric,
                    "method": method,
                    "reference": reference,
                    "n_paired_training_seeds": len(paired_seeds),
                    "mean_paired_difference": float(np.mean(differences)) if differences else float("nan"),
                    "wilcoxon_statistic": float(statistic),
                    "p_value": float(p_value),
                    "rank_biserial_effect_size": rank_biserial_paired(differences),
                })
                observed, low, high, n_seeds, n_units = hierarchical_paired_bootstrap(
                    records,
                    metric,
                    method,
                    reference,
                    samples=bootstrap_samples,
                )
                bootstrap_rows.append({
                    "metric": metric,
                    "method": method,
                    "reference": reference,
                    "paired_difference": observed,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_training_seeds": n_seeds,
                    "n_paired_scenario_units": n_units,
                })
        adjusted = holm_adjust([row["p_value"] for row in metric_pair_rows])
        for row, corrected in zip(metric_pair_rows, adjusted):
            row["holm_adjusted_p"] = corrected
        pairwise_rows.extend(metric_pair_rows)
    return friedman_rows, pairwise_rows, bootstrap_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--require-pairing-metadata", action="store_true")
    args = parser.parse_args()

    records = load_records(args.inputs)
    validate_records(records, require_pairing_metadata=args.require_pairing_metadata)
    summaries = summarize_across_seeds(records, args.metrics)
    friedman, pairwise, bootstrap = inferential_tests(
        records,
        args.metrics,
        args.reference,
        max(1000, int(args.bootstrap_samples)),
    )
    write_csv(args.out_dir / "seed_summary.csv", summaries)
    write_csv(args.out_dir / "friedman_tests.csv", friedman)
    write_csv(args.out_dir / "wilcoxon_holm_effect_sizes.csv", pairwise)
    write_csv(args.out_dir / "paired_scenario_bootstrap.csv", bootstrap)
    print(f"wrote formal statistics to {args.out_dir}")


if __name__ == "__main__":
    main()
