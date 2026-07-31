#!/usr/bin/env python3
"""Metrics for validating value-difference credit against paired rollouts."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MetricResult:
    n_states: int
    n_pairs: int
    spearman: float
    pooled_raw_spearman: float
    sign_agreement: float
    top1_accuracy: float
    n_spearman_states: int
    mean_state_spearman: float
    n_oracle_top1_ties: int
    n_estimate_top1_ties: int


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Return one-based average ranks, including deterministic tie handling."""
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute Spearman's rho without requiring scipy."""
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have the same length")
    if len(left) < 2:
        return float("nan")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.ptp(left_ranks) == 0.0 or np.ptp(right_ranks) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _sign(values: np.ndarray, epsilon: float) -> np.ndarray:
    signs = np.zeros(values.shape, dtype=np.int8)
    signs[values > epsilon] = 1
    signs[values < -epsilon] = -1
    return signs


def _argmax_with_tie(values: np.ndarray, agent_indices: np.ndarray, epsilon: float) -> Tuple[int, bool]:
    maximum = float(np.max(values))
    tied = agent_indices[np.abs(values - maximum) <= epsilon]
    return int(np.min(tied)), len(tied) > 1


def compute_metrics(rows: Sequence[Mapping[str, object]], sign_epsilon: float = 1e-8) -> MetricResult:
    """Compute pooled credit metrics and joint-state Top-1 accuracy."""
    if not rows:
        return MetricResult(0, 0, *(float("nan"),) * 4, 0, float("nan"), 0, 0)

    estimates = np.asarray([float(row["estimated_credit"]) for row in rows], dtype=np.float64)
    marginals = np.asarray([float(row["marginal_return"]) for row in rows], dtype=np.float64)
    pooled_raw_rho = spearman_correlation(estimates, marginals)
    sign_agreement = float(np.mean(_sign(estimates, sign_epsilon) == _sign(marginals, sign_epsilon)))

    by_state: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)

    correct = 0
    oracle_ties = 0
    estimate_ties = 0
    state_rhos = []
    estimate_state_ranks = []
    marginal_state_ranks = []
    for state_rows in by_state.values():
        ordered = sorted(state_rows, key=lambda row: int(row["agent_index"]))
        agent_indices = np.asarray([int(row["agent_index"]) for row in ordered], dtype=np.int64)
        state_estimates = np.asarray([float(row["estimated_credit"]) for row in ordered])
        state_marginals = np.asarray([float(row["marginal_return"]) for row in ordered])
        estimate_state_ranks.extend(_average_ranks(state_estimates))
        marginal_state_ranks.extend(_average_ranks(state_marginals))
        predicted, estimate_tie = _argmax_with_tie(state_estimates, agent_indices, sign_epsilon)
        responsible, oracle_tie = _argmax_with_tie(state_marginals, agent_indices, sign_epsilon)
        correct += int(predicted == responsible)
        oracle_ties += int(oracle_tie)
        estimate_ties += int(estimate_tie)
        rho = spearman_correlation(state_estimates, state_marginals)
        if math.isfinite(rho):
            state_rhos.append(rho)

    stratified_rho = spearman_correlation(estimate_state_ranks, marginal_state_ranks)

    return MetricResult(
        n_states=len(by_state),
        n_pairs=len(rows),
        spearman=stratified_rho,
        pooled_raw_spearman=pooled_raw_rho,
        sign_agreement=sign_agreement,
        top1_accuracy=correct / len(by_state),
        n_spearman_states=len(state_rhos),
        mean_state_spearman=float(np.mean(state_rhos)) if state_rhos else float("nan"),
        n_oracle_top1_ties=oracle_ties,
        n_estimate_top1_ties=estimate_ties,
    )


def random_mask_rows(
    rows: Sequence[Mapping[str, object]], rng: np.random.Generator
) -> List[dict]:
    """Randomly reassign actual agent-mask scores within every joint state.

    A permutation preserves the mask-score distribution and the number of masks,
    while breaking the association between a masked slot and the credited agent.
    """
    by_state: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)

    randomized = []
    for state_rows in by_state.values():
        ordered = sorted(state_rows, key=lambda row: int(row["agent_index"]))
        credits = np.asarray([float(row["estimated_credit"]) for row in ordered])
        permuted = rng.permutation(credits)
        for row, credit in zip(ordered, permuted):
            item = dict(row)
            item["estimated_credit"] = float(credit)
            randomized.append(item)
    return randomized


def cluster_bootstrap(
    rows: Sequence[Mapping[str, object]],
    repeats: int,
    seed: int,
    sign_epsilon: float = 1e-8,
) -> Dict[str, Tuple[float, float]]:
    """Return state-cluster bootstrap 95% intervals for the three primary metrics."""
    by_state: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    state_ids = sorted(by_state)
    if not state_ids or repeats <= 0:
        return {}

    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(repeats):
        sampled = rng.choice(state_ids, size=len(state_ids), replace=True)
        boot_rows = []
        for sample_index, state_id in enumerate(sampled):
            for row in by_state[str(state_id)]:
                item = dict(row)
                item["state_id"] = f"bootstrap_{sample_index}"
                boot_rows.append(item)
        result = compute_metrics(boot_rows, sign_epsilon=sign_epsilon)
        samples["spearman"].append(result.spearman)
        samples["sign_agreement"].append(result.sign_agreement)
        samples["top1_accuracy"].append(result.top1_accuracy)

    intervals = {}
    for name, values in samples.items():
        finite = np.asarray([value for value in values if math.isfinite(value)])
        intervals[name] = (
            float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
            float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        )
    return intervals


def random_mask_distribution(
    rows: Sequence[Mapping[str, object]],
    repeats: int,
    seed: int,
    sign_epsilon: float = 1e-8,
) -> Tuple[List[dict], Dict[str, Tuple[float, float, float]]]:
    """Evaluate repeated random-mask assignments and summarize mean/95% interval."""
    rng = np.random.default_rng(seed)
    repeat_rows = []
    for repeat in range(repeats):
        result = compute_metrics(
            random_mask_rows(rows, rng), sign_epsilon=sign_epsilon)
        repeat_rows.append({"repeat": repeat, **result.__dict__})

    summary = {}
    for name in ("spearman", "sign_agreement", "top1_accuracy", "mean_state_spearman"):
        values = np.asarray([
            float(row[name]) for row in repeat_rows if math.isfinite(float(row[name]))
        ])
        summary[name] = (
            float(np.mean(values)) if values.size else float("nan"),
            float(np.quantile(values, 0.025)) if values.size else float("nan"),
            float(np.quantile(values, 0.975)) if values.size else float("nan"),
        )
    return repeat_rows, summary
