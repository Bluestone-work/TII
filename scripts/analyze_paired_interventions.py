#!/usr/bin/env python3
"""Validate and analyze paired intervention rollout records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Iterable, List

from paired_intervention_metrics import (
    cluster_bootstrap,
    compute_metrics,
    random_mask_distribution,
)


REQUIRED_FIELDS = {
    "state_id",
    "num_agents",
    "agent",
    "agent_index",
    "estimated_credit",
    "baseline_return",
    "intervention_return",
    "baseline_start_fingerprint",
    "intervention_start_fingerprint",
}


def iter_jsonl(path: Path) -> Iterable[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def finite_float(row: dict, field: str, source: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {field} is not numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{source}: {field} is not finite")
    return value


def validate_rows(inputs: List[Path], identity_tolerance: float) -> tuple[List[dict], dict]:
    rows = []
    rejected = {"fingerprint_mismatch": 0, "invalid_flag": 0}
    seen = set()
    state_baselines = {}
    state_agents = {}
    state_expected_agents = {}
    declared_agent_counts = set()

    for path in inputs:
        for line_number, row in enumerate(iter_jsonl(path), 1):
            source = f"{path}:{line_number}"
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(f"{source}: missing fields {missing}")
            if str(row["baseline_start_fingerprint"]) != str(row["intervention_start_fingerprint"]):
                rejected["fingerprint_mismatch"] += 1
                continue
            if row.get("valid") is False:
                rejected["invalid_flag"] += 1
                continue

            state_id = str(row["state_id"])
            agent_index = int(row["agent_index"])
            if agent_index < 0:
                raise ValueError(f"{source}: agent_index must be non-negative")
            expected_agents = int(row["num_agents"])
            if expected_agents < 1:
                raise ValueError(f"{source}: num_agents must be positive")
            if agent_index >= expected_agents:
                raise ValueError(f"{source}: agent_index exceeds declared num_agents")
            declared_agent_counts.add(expected_agents)
            previous_expected = state_expected_agents.setdefault(state_id, expected_agents)
            if previous_expected != expected_agents:
                raise ValueError(f"{source}: num_agents differs within state {state_id}")
            key = (state_id, agent_index)
            if key in seen:
                raise ValueError(f"{source}: duplicate state/agent pair {key}")
            seen.add(key)

            estimated = finite_float(row, "estimated_credit", source)
            baseline = finite_float(row, "baseline_return", source)
            intervention = finite_float(row, "intervention_return", source)
            marginal = baseline - intervention
            if "marginal_return" in row:
                recorded = finite_float(row, "marginal_return", source)
                if not math.isclose(recorded, marginal, rel_tol=0.0, abs_tol=identity_tolerance):
                    raise ValueError(
                        f"{source}: marginal identity failed: {recorded} != {baseline} - {intervention}"
                    )
            previous = state_baselines.setdefault(state_id, baseline)
            if not math.isclose(previous, baseline, rel_tol=0.0, abs_tol=identity_tolerance):
                raise ValueError(f"{source}: baseline_return differs within state {state_id}")
            agents = state_agents.setdefault(state_id, set())
            if agent_index in agents:
                raise ValueError(f"{source}: duplicate agent index in state {state_id}")
            agents.add(agent_index)

            clean = dict(row)
            clean.update({
                "state_id": state_id,
                "agent_index": agent_index,
                "estimated_credit": estimated,
                "baseline_return": baseline,
                "intervention_return": intervention,
                "marginal_return": marginal,
            })
            rows.append(clean)

    complete_states = {
        state_id for state_id, indices in state_agents.items()
        if len(indices) == state_expected_agents[state_id]
        and indices == set(range(state_expected_agents[state_id]))
    }
    incomplete = len(state_agents) - len(complete_states)
    rows = [row for row in rows if row["state_id"] in complete_states]
    rejected["incomplete_states"] = incomplete
    rejected["accepted_states"] = len(complete_states)
    rejected["accepted_pairs"] = len(rows)
    rejected["expected_agents_per_state"] = (
        next(iter(declared_agent_counts))
        if len(declared_agent_counts) == 1
        else sorted(declared_agent_counts)
    )
    return rows, rejected


def write_csv(path: Path, rows: List[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def metric_row(estimator: str, result, intervals=None) -> dict:
    intervals = intervals or {}
    row = {"estimator": estimator, **result.__dict__}
    for name in ("spearman", "sign_agreement", "top1_accuracy"):
        low, high = intervals.get(name, (float("nan"), float("nan")))
        row[f"{name}_ci_low"] = low
        row[f"{name}_ci_high"] = high
    return row


def grouped_metric_rows(rows: List[dict], repeats: int, seed: int, sign_epsilon: float) -> List[dict]:
    output = []
    group_repeats = min(repeats, 200)
    for field in ("scenario_case", "map_number", "step"):
        values = sorted({str(row[field]) for row in rows if row.get(field) is not None})
        for group_index, value in enumerate(values):
            group = [row for row in rows if str(row.get(field)) == value]
            actual = compute_metrics(group, sign_epsilon=sign_epsilon)
            _random_rows, random_summary = random_mask_distribution(
                group, group_repeats, seed + 1000 + group_index, sign_epsilon=sign_epsilon)
            output.append({
                "group_type": field,
                "group": value,
                "n_states": actual.n_states,
                "n_pairs": actual.n_pairs,
                "spearman": actual.spearman,
                "sign_agreement": actual.sign_agreement,
                "top1_accuracy": actual.top1_accuracy,
                "random_spearman": random_summary["spearman"][0],
                "random_sign_agreement": random_summary["sign_agreement"][0],
                "random_top1_accuracy": random_summary["top1_accuracy"][0],
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--random-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--sign-epsilon", type=float, default=1e-8)
    parser.add_argument("--identity-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    if args.random_repeats < 1 or args.bootstrap_repeats < 0:
        parser.error("repeat counts must be positive (bootstrap may be zero)")
    rows, quality = validate_rows(args.inputs, args.identity_tolerance)
    if not rows:
        raise SystemExit("no complete, fingerprint-matched paired states")

    actual = compute_metrics(rows, sign_epsilon=args.sign_epsilon)
    intervals = cluster_bootstrap(
        rows, args.bootstrap_repeats, args.seed + 1, sign_epsilon=args.sign_epsilon)
    random_rows, random_summary = random_mask_distribution(
        rows, args.random_repeats, args.seed, sign_epsilon=args.sign_epsilon)
    random_result = compute_metrics(rows, sign_epsilon=args.sign_epsilon)
    random_result = random_result.__class__(
        **{
            **random_result.__dict__,
            "spearman": random_summary["spearman"][0],
            "sign_agreement": random_summary["sign_agreement"][0],
            "top1_accuracy": random_summary["top1_accuracy"][0],
            "mean_state_spearman": random_summary["mean_state_spearman"][0],
        }
    )
    random_intervals = {
        name: (values[1], values[2])
        for name, values in random_summary.items()
        if name in {"spearman", "sign_agreement", "top1_accuracy"}
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        metric_row("value_difference", actual, intervals),
        metric_row("random_mask", random_result, random_intervals),
    ]
    randomization_p = {}
    for name in ("spearman", "sign_agreement", "top1_accuracy"):
        actual_value = float(getattr(actual, name))
        null_values = [
            float(row[name]) for row in random_rows
            if math.isfinite(float(row[name]))
        ]
        exceedances = sum(value >= actual_value for value in null_values)
        randomization_p[name] = (1 + exceedances) / (1 + len(null_values))
        summary_rows[0][f"{name}_random_mask_p_one_sided"] = randomization_p[name]
        summary_rows[1][f"{name}_random_mask_p_one_sided"] = float("nan")
    write_csv(args.out_dir / "paired_intervention_metrics.csv", summary_rows)
    write_csv(args.out_dir / "random_mask_repeats.csv", random_rows)
    write_csv(
        args.out_dir / "paired_intervention_by_group.csv",
        grouped_metric_rows(rows, args.random_repeats, args.seed, args.sign_epsilon),
    )
    (args.out_dir / "quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def display(value: float) -> str:
        return "NA" if not math.isfinite(value) else f"{value:.4f}"

    report = [
        "# Paired intervention validation",
        "",
        f"Accepted {actual.n_states} joint states and {actual.n_pairs} state-agent interventions.",
        "Branches with mismatched start fingerprints and incomplete joint states were excluded.",
        "",
        "| Estimator | Spearman | Sign agreement | Top-1 responsible-agent accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report.append(
            f"| {row['estimator']} | {display(row['spearman'])} "
            f"[{display(row['spearman_ci_low'])}, {display(row['spearman_ci_high'])}] | "
            f"{display(row['sign_agreement'])} "
            f"[{display(row['sign_agreement_ci_low'])}, {display(row['sign_agreement_ci_high'])}] | "
            f"{display(row['top1_accuracy'])} "
            f"[{display(row['top1_accuracy_ci_low'])}, {display(row['top1_accuracy_ci_high'])}] |"
        )
    report.extend([
        "",
        "The primary Spearman ranks agents within each joint state before pooling; raw-value pooled Spearman is diagnostic only.",
        "Value-difference intervals use a joint-state cluster bootstrap.",
        "Sign agreement compares ternary signs (-1, 0, +1) using the configured epsilon.",
        "Top-1 ties are resolved by the lowest agent index and tie counts are retained in the CSV.",
        "The random-mask baseline permutes observed mask scores across agent identities within each state.",
        "One-sided randomization p-values for value difference versus random mask: "
        f"Spearman={randomization_p['spearman']:.4f}, "
        f"sign={randomization_p['sign_agreement']:.4f}, "
        f"Top-1={randomization_p['top1_accuracy']:.4f}.",
    ])
    (args.out_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote paired intervention analysis to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
