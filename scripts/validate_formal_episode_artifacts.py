#!/usr/bin/env python3
"""Fail closed when a formal episode artifact is incomplete or inconsistent."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path


REQUIRED_FIELDS = {
    "steps", "reached_agents", "collided_agents", "truncated",
    "agent_success_rate", "team_success", "collision_rate", "timeout_rate",
    "SR", "TSR", "CR", "spl", "path_length", "completion_time", "minimum_clearance",
    "clearance_p05", "near_collision_count", "deadlock", "oscillation_count",
    "control_jerk", "filter_intervention_rate", "filter_action_difference",
    "successful_agent_completion_time_mean", "successful_agent_completion_time_sum",
    "successful_agent_completion_time_count", "filter_intervention_count",
    "filter_action_samples",
    "policy_only_inference_latency_ms_mean", "safety_filter_latency_ms_mean",
    "safety_filter_latency_samples", "episode_wall_time_s",
    "real_time_factor", "control_dt", "episode_index",
    "episode_seed", "num_agents", "map_number", "num_dynamic_obstacles",
    "max_episode_steps", "shield_enable", "evaluation_system", "controller",
    "seed", "training_seed", "method", "scenario", "initial_start_goal",
    "initial_state_fingerprint", "paired_scenario_eligible", "pairing_basis",
    "dynamic_obstacle_trajectory_fingerprint", "dynamic_obstacle_trajectory_spec",
}
FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")


def scalar(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return float(value.strip().lower() == "true")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def boolean(value):
    parsed = scalar(value)
    return bool(parsed) if parsed in (0.0, 1.0) else None


def close(left, right, tolerance=1e-8):
    left_value, right_value = scalar(left), scalar(right)
    return (
        left_value is not None
        and right_value is not None
        and abs(left_value - right_value) <= tolerance
    )


def record_key(record):
    return (
        int(record["map_number"]), int(record["episode_index"]),
        int(record["training_seed"]), str(record["method"]), str(record["scenario"]),
    )


def parse_routes(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_mapping(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def route_fingerprint(routes):
    payload = json.dumps(routes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def validate_record(record, map_limits, scenario_cases):
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        return [f"missing fields: {sorted(missing)}"]
    errors = []
    map_number = int(record["map_number"])
    episode_index = int(record["episode_index"])
    num_agents = int(record["num_agents"])
    reached = int(record["reached_agents"])
    collided = int(record["collided_agents"])
    steps = int(record["steps"])
    max_steps = int(record["max_episode_steps"])
    truncated = boolean(record["truncated"])

    if map_number in map_limits and max_steps != map_limits[map_number]:
        errors.append(f"max_episode_steps={max_steps}, expected={map_limits[map_number]}")
    scenario_case = str(record.get("scenario_case") or "")
    expected_scenario = None
    if scenario_cases:
        expected_scenario = scenario_cases.get(scenario_case)
        if expected_scenario is None:
            errors.append(f"scenario_case={scenario_case!r} absent from manifest")
        else:
            for field in ("map_number", "num_agents", "max_episode_steps", "num_dynamic_obstacles"):
                if int(record[field]) != int(expected_scenario[field]):
                    errors.append(f"{field}={record[field]}, expected={expected_scenario[field]}")
    if not 0 <= reached <= num_agents or not 0 <= collided <= num_agents:
        errors.append("reached/collided agents outside [0,num_agents]")
    if not 0 <= steps <= max_steps:
        errors.append("steps outside [0,max_episode_steps]")
    if not close(record["control_dt"], 0.1):
        errors.append("control_dt must equal 0.1 s")
    if not close(record["completion_time"], steps * 0.1, tolerance=1e-6):
        errors.append("completion_time != steps * 0.1")
    success_time_sum = scalar(record["successful_agent_completion_time_sum"])
    success_time_count = scalar(record["successful_agent_completion_time_count"])
    success_time_mean = scalar(record["successful_agent_completion_time_mean"])
    if success_time_sum is None or success_time_sum < 0.0:
        errors.append("successful-agent completion-time sum is invalid")
    if success_time_count is None or int(success_time_count) != reached:
        errors.append("successful-agent completion-time count != reached_agents")
    elif reached > 0 and (
        success_time_mean is None
        or not close(success_time_mean, success_time_sum / reached, tolerance=1e-6)
    ):
        errors.append("successful-agent completion-time mean is inconsistent")
    wall_time = scalar(record["episode_wall_time_s"])
    real_time_factor = scalar(record["real_time_factor"])
    completion_time = scalar(record["completion_time"])
    if wall_time is None or wall_time <= 0.0:
        errors.append("episode_wall_time_s must be finite and positive")
    elif completion_time is None or real_time_factor is None or not close(
        real_time_factor,
        completion_time / wall_time,
        tolerance=1e-6,
    ):
        errors.append("real_time_factor != completion_time / episode_wall_time_s")
    if int(record["episode_seed"]) != int(record["seed"]) + episode_index:
        errors.append("episode_seed != seed + episode_index")

    expected_values = {
        "agent_success_rate": reached / max(1, num_agents),
        "SR": reached / max(1, num_agents),
        "collision_rate": collided / max(1, num_agents),
        "CR": collided / max(1, num_agents),
        "team_success": float(reached == num_agents),
        "TSR": float(reached == num_agents),
        "timeout_rate": float(bool(truncated)),
    }
    for field, expected in expected_values.items():
        if not close(record[field], expected):
            errors.append(f"{field} inconsistent with episode outcome")

    for field in (
        "agent_success_rate", "team_success", "collision_rate", "timeout_rate",
        "spl", "filter_intervention_rate",
    ):
        value = scalar(record[field])
        if value is None or not 0.0 <= value <= 1.0:
            errors.append(f"{field} outside [0,1] or non-finite")
    for field in (
        "path_length", "completion_time", "minimum_clearance", "clearance_p05",
        "near_collision_count", "oscillation_count", "control_jerk",
        "filter_intervention_rate", "filter_action_difference",
    ):
        value = scalar(record[field])
        if value is None or value < 0.0:
            errors.append(f"{field} negative or non-finite")
    intervention_count = scalar(record["filter_intervention_count"])
    filter_samples = scalar(record["filter_action_samples"])
    intervention_rate = scalar(record["filter_intervention_rate"])
    if (
        intervention_count is None or filter_samples is None
        or int(intervention_count) != intervention_count
        or int(filter_samples) != filter_samples
        or not 0 <= intervention_count <= filter_samples
        or filter_samples <= 0
    ):
        errors.append("invalid filter intervention count or denominator")
    elif intervention_rate is None or not close(
        intervention_rate, intervention_count / filter_samples, tolerance=1e-6
    ):
        errors.append("filter_intervention_rate != count / samples")
    clearance_p05 = scalar(record["clearance_p05"])
    minimum_clearance = scalar(record["minimum_clearance"])
    if (
        clearance_p05 is not None
        and minimum_clearance is not None
        and clearance_p05 + 1e-9 < minimum_clearance
    ):
        errors.append("clearance_p05 below minimum_clearance")
    if str(record["controller"]) == "policy":
        latency = scalar(record["policy_only_inference_latency_ms_mean"])
        if latency is None or latency <= 0.0:
            errors.append("policy-only latency missing or non-positive")
    filter_latency = scalar(record["safety_filter_latency_ms_mean"])
    filter_samples = scalar(record["safety_filter_latency_samples"])
    if filter_latency is None or filter_latency < 0.0 or filter_samples is None or filter_samples <= 0.0:
        errors.append("safety-filter latency missing, negative, or unsampled")
    if boolean(record["shield_enable"]) and str(record["controller"]) == "policy":
        if str(record["evaluation_system"]) != "policy_plus_safety_filter":
            errors.append("shield-on policy lacks policy_plus_safety_filter attribution")
    if boolean(record["shield_enable"]) is False and str(record["controller"]) == "policy":
        if str(record["evaluation_system"]) != "policy_only":
            errors.append("shield-off policy lacks policy_only attribution")

    routes = parse_routes(record["initial_start_goal"])
    fingerprint = str(record["initial_state_fingerprint"] or "")
    if routes is None or not FINGERPRINT_RE.fullmatch(fingerprint):
        errors.append("invalid initial start--goal record or SHA-256 fingerprint")
    elif route_fingerprint(routes) != fingerprint:
        errors.append("initial_state_fingerprint does not match initial_start_goal")
    if (
        routes is not None
        and expected_scenario is not None
        and str(expected_scenario.get("route_mode", "fixed")) == "fixed"
    ):
        expected_routes = {
            str(agent): {
                "start": [round(float(value), 3) for value in route[0]],
                "goal": [round(float(value), 3) for value in route[1]],
            }
            for agent, route in dict(expected_scenario.get("route_plan", {})).items()
        }
        if routes != expected_routes:
            errors.append("initial_start_goal does not match the frozen scenario route plan")
    eligible = boolean(record["paired_scenario_eligible"])
    basis = str(record["pairing_basis"])
    if eligible is None:
        errors.append("paired_scenario_eligible must be boolean")
    dynamic_count = int(record["num_dynamic_obstacles"])
    trajectory_fingerprint = str(record.get("dynamic_obstacle_trajectory_fingerprint") or "")
    if dynamic_count > 0 and eligible:
        if basis != "fixed_start_goal_and_dynamic_trajectory":
            errors.append("paired dynamic episode lacks fixed trajectory basis")
        if not FINGERPRINT_RE.fullmatch(trajectory_fingerprint):
            errors.append("paired dynamic episode lacks trajectory SHA-256")
        trajectory_spec = parse_mapping(record.get("dynamic_obstacle_trajectory_spec"))
        if not trajectory_spec:
            errors.append("paired dynamic episode lacks trajectory manifest")
        else:
            expected_trajectory_values = {
                "episode_seed": int(record["episode_seed"]),
                "map_number": map_number,
                "num_dynamic_obstacles": dynamic_count,
                "max_episode_steps": max_steps,
                "frame_count": max_steps + 1,
            }
            for field, expected in expected_trajectory_values.items():
                try:
                    actual = int(trajectory_spec[field])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"trajectory manifest has invalid {field}")
                    continue
                if actual != expected:
                    errors.append(
                        f"trajectory manifest {field}={actual}, expected={expected}"
                    )
            if not close(trajectory_spec.get("control_dt"), record["control_dt"]):
                errors.append("trajectory manifest control_dt mismatch")
            if "obs_speed" in record and not close(
                trajectory_spec.get("obstacle_speed_mps"), record["obs_speed"], tolerance=1e-6
            ):
                errors.append("trajectory manifest obstacle_speed_mps mismatch")
    if eligible and basis not in {
        "fixed_scenario_bank", "start_goal_static_environment",
        "fixed_start_goal_and_dynamic_trajectory",
    }:
        errors.append(f"unsupported eligible pairing_basis={basis}")
    return errors


def load_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
    return records


def parse_map(spec):
    try:
        map_number, episodes, limit = (int(item) for item in spec.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected MAP:EPISODES:LIMIT") from exc
    return map_number, episodes, limit


def load_scenario_cases(path):
    if path is None:
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["case_id"]): case for case in document.get("cases", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--jsonl", required=True, type=Path)
    parser.add_argument("--map", action="append", type=parse_map, default=[])
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--scenario-manifest", type=Path)
    args = parser.parse_args()
    if not args.csv.is_file() or not args.jsonl.is_file():
        parser.error("CSV and JSONL inputs must exist")

    expected_counts = {map_number: count for map_number, count, _limit in args.map}
    map_limits = {map_number: limit for map_number, _count, limit in args.map}
    scenario_cases = load_scenario_cases(args.scenario_manifest)
    csv_records, json_records = load_csv(args.csv), load_jsonl(args.jsonl)
    errors, csv_keys = [], []
    for index, record in enumerate(csv_records, 2):
        key = record_key(record)
        csv_keys.append(key)
        for issue in validate_record(record, map_limits, scenario_cases):
            errors.append({"row": index, "key": key, "error": issue})
    json_keys = [record_key(record) for record in json_records]
    duplicate_csv = len(csv_keys) - len(set(csv_keys))
    duplicate_json = len(json_keys) - len(set(json_keys))
    if set(csv_keys) != set(json_keys):
        errors.append({"error": "CSV and JSONL key sets differ"})
    if duplicate_csv or duplicate_json:
        errors.append({"error": f"duplicate keys csv={duplicate_csv} jsonl={duplicate_json}"})
    actual_counts = Counter(key[0] for key in csv_keys)
    if expected_counts and dict(actual_counts) != expected_counts:
        errors.append({"error": f"map counts {dict(actual_counts)} != {expected_counts}"})
    if args.expected_records is not None and len(csv_records) != args.expected_records:
        errors.append({"error": f"record count {len(csv_records)} != {args.expected_records}"})
    summary = {
        "valid": not errors,
        "csv_rows": len(csv_records),
        "jsonl_rows": len(json_records),
        "counts_by_map": dict(sorted(actual_counts.items())),
        "duplicate_csv_keys": duplicate_csv,
        "duplicate_jsonl_keys": duplicate_json,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
