from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


OPTION_NAMES: tuple[str, ...] = (
    "follow_path",
    "slow_follow",
    "stop_wait",
    "backoff",
    "detour_left",
    "detour_right",
    "replan",
)

OPTION_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(OPTION_NAMES)}

CANONICAL_MODE_BY_OPTION: Dict[str, str] = {
    "follow_path": "go",
    "slow_follow": "go",
    "stop_wait": "wait",
    "backoff": "backoff",
    "detour_left": "detour",
    "detour_right": "detour",
    "replan": "replan",
}


@dataclass
class LocalOptionObservation:
    min_dist: float
    front_min: float
    left_min: float
    right_min: float
    rear_min: float
    front_left_min: float
    front_center_min: float
    front_right_min: float
    clearance_asymmetry: float
    social_risk_max: float
    ttc_min: float
    front_risk: float
    left_risk: float
    right_risk: float
    nearest_neighbor_dist: float
    nearest_neighbor_bearing: float
    closing_speed: float
    stuck_score: float
    current_vel_x: float
    current_vel_w: float
    local_target_direction: float
    rolling_subgoal_direction: float
    option_state: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptionFeasibilityResult:
    local_metrics: LocalOptionObservation
    feasible_by_option: Dict[str, bool]
    action_mask: np.ndarray
    infeasible_reason_by_option: Dict[str, List[str]]

    def is_feasible(self, option_name: str) -> bool:
        return bool(self.feasible_by_option.get(str(option_name), False))

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "local_metrics": self.local_metrics.to_dict(),
            "feasible_by_option": dict(self.feasible_by_option),
            "action_mask": self.action_mask.astype(np.int32).tolist(),
            "infeasible_reason_by_option": {
                name: list(reasons)
                for name, reasons in self.infeasible_reason_by_option.items()
            },
        }
        for option_name in OPTION_NAMES:
            payload[f"feasible_{option_name}"] = bool(self.feasible_by_option.get(option_name, False))
        return payload


def _finite_min(values: Sequence[float], default: float = float("inf")) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return min(finite) if finite else float(default)


def _angle_mask(angles: np.ndarray, start: float, end: float) -> np.ndarray:
    start = float(start)
    end = float(end)
    if start <= end:
        return (angles >= start) & (angles <= end)
    return (angles >= start) | (angles <= end)


def _scan_ranges_and_angles(agent: Any) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.array(
        agent.latest_scan.ranges if getattr(agent, "latest_scan", None) else [agent.scan_max_range] * 360,
        dtype=np.float32,
    )
    ranges = np.nan_to_num(
        ranges,
        nan=float(agent.scan_max_range),
        posinf=float(agent.scan_max_range),
        neginf=0.0,
    )
    ranges = np.clip(ranges, 0.0, float(agent.scan_max_range))

    n = int(ranges.size)
    if getattr(agent, "latest_scan", None) is not None and getattr(agent.latest_scan, "ranges", None):
        angle_min = float(getattr(agent.latest_scan, "angle_min", -math.pi))
        angle_inc = float(getattr(agent.latest_scan, "angle_increment", (2.0 * math.pi) / max(1, n)))
        if not math.isfinite(angle_inc) or abs(angle_inc) < 1e-6:
            angle_inc = (2.0 * math.pi) / max(1, n)
        angles = angle_min + np.arange(n, dtype=np.float32) * angle_inc
    else:
        angles = np.linspace(-math.pi, math.pi, num=n, endpoint=False, dtype=np.float32)
    angles = (angles + math.pi) % (2.0 * math.pi) - math.pi
    return ranges, angles


def _sector_min(agent: Any, ranges: np.ndarray, angles: np.ndarray, start_deg: float, end_deg: float) -> float:
    mask = _angle_mask(angles, math.radians(start_deg), math.radians(end_deg))
    vals = ranges[mask]
    vals = vals[(vals > float(agent.scan_valid_min))]
    if vals.size <= 0:
        return float(agent.scan_max_range)
    return float(vals.min())


def _nearest_neighbor_metrics(agent: Any) -> tuple[float, float, float, float]:
    parent = getattr(agent, "parent_env", None)
    if parent is None or not hasattr(parent, "_get_perceived_neighbor_samples"):
        return float("inf"), 0.0, 0.0, float("inf")

    samples = parent._get_perceived_neighbor_samples(f"agent_{agent.robot_id}")
    if not samples:
        return float("inf"), 0.0, 0.0, float("inf")

    _, dist, n_pos, n_vel = samples[0]
    my_pos = np.array([float(agent.current_pose["x"]), float(agent.current_pose["y"])], dtype=np.float32)
    rel = np.asarray(n_pos, dtype=np.float32) - my_pos
    body_rel = agent._world_to_body(rel)
    bearing = float(math.atan2(float(body_rel[1]), float(body_rel[0])))

    yaw = float(agent.current_pose["yaw"])
    my_vel = np.array(
        [
            float(agent.current_vel_x) * math.cos(yaw),
            float(agent.current_vel_x) * math.sin(yaw),
        ],
        dtype=np.float32,
    )
    rel_unit = rel / max(float(np.linalg.norm(rel)), 1e-6)
    rel_vel = np.asarray(n_vel, dtype=np.float32) - my_vel
    closing_speed = float(max(0.0, -np.dot(rel_vel, rel_unit)))
    ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 1e-3 else float("inf")
    return float(dist), bearing, closing_speed, ttc


def extract_local_option_observation(agent: Any, option_state: str = "idle") -> LocalOptionObservation:
    sectors = agent._scan_sector_metrics()
    ranges, angles = _scan_ranges_and_angles(agent)

    front_min = float(sectors.get("front_min", float(agent.scan_max_range)))
    left_min = float(sectors.get("left_min", float(agent.scan_max_range)))
    right_min = float(sectors.get("right_min", float(agent.scan_max_range)))
    min_dist = float(sectors.get("min_dist", float(agent.scan_max_range)))

    front_left_min = _sector_min(agent, ranges, angles, 10.0, 40.0)
    front_center_min = _sector_min(agent, ranges, angles, -12.0, 12.0)
    front_right_min = _sector_min(agent, ranges, angles, -40.0, -10.0)
    rear_min = _sector_min(agent, ranges, angles, 150.0, -150.0)

    predictive = dict(getattr(agent, "_last_predictive_metrics", {}))
    interaction_ctx = dict(agent._get_interaction_context())
    social_summary = agent._compute_social_risk_summary()
    nearest_dist, nearest_bearing, closing_speed, neighbor_ttc = _nearest_neighbor_metrics(agent)

    nominal_target = tuple(agent._compute_nominal_tracking_info()["subgoal"])
    current_target = tuple(getattr(agent, "current_subgoal", None) or nominal_target)

    social_risk = float(
        max(
            0.0,
            float(getattr(agent, "_last_social_risk", 0.0)),
            float(social_summary.get("social_risk", 0.0)),
            float(predictive.get("social_risk", 0.0)),
        )
    )

    close_front_ratio = float(
        np.clip(
            (float(getattr(agent, "close_obstacle_dist", 0.55)) - front_center_min)
            / max(float(getattr(agent, "close_obstacle_dist", 0.55)), 1e-6),
            0.0,
            1.0,
        )
    )
    side_close_dist = float(getattr(agent, "side_close_dist", 0.22))
    left_risk = float(
        max(
            np.clip((side_close_dist - min(left_min, front_left_min)) / max(side_close_dist, 1e-6), 0.0, 1.0),
            social_risk if nearest_bearing > 0.10 else 0.0,
        )
    )
    right_risk = float(
        max(
            np.clip((side_close_dist - min(right_min, front_right_min)) / max(side_close_dist, 1e-6), 0.0, 1.0),
            social_risk if nearest_bearing < -0.10 else 0.0,
        )
    )
    front_risk = float(max(close_front_ratio, float(predictive.get("front_risk", 0.0))))

    front_blocked_ratio = float(
        np.clip(
            (float(getattr(agent, "subgoal_block_front_dist", 0.42)) - front_center_min)
            / max(float(getattr(agent, "subgoal_block_front_dist", 0.42)), 1e-6),
            0.0,
            1.0,
        )
    )
    stuck_score = float(agent._compute_stuck_score(front_blocked_ratio))

    ttc_min = _finite_min(
        [
            float(interaction_ctx.get("ttc", float("inf"))),
            float(predictive.get("social_ttc", float("inf"))),
            float(predictive.get("front_ttc", float("inf"))),
            float(neighbor_ttc),
        ]
    )

    return LocalOptionObservation(
        min_dist=min_dist,
        front_min=front_min,
        left_min=left_min,
        right_min=right_min,
        rear_min=rear_min,
        front_left_min=front_left_min,
        front_center_min=front_center_min,
        front_right_min=front_right_min,
        clearance_asymmetry=float(left_min - right_min),
        social_risk_max=social_risk,
        ttc_min=ttc_min,
        front_risk=front_risk,
        left_risk=left_risk,
        right_risk=right_risk,
        nearest_neighbor_dist=float(nearest_dist),
        nearest_neighbor_bearing=float(nearest_bearing),
        closing_speed=float(closing_speed),
        stuck_score=stuck_score,
        current_vel_x=float(getattr(agent, "current_vel_x", 0.0)),
        current_vel_w=float(getattr(agent, "current_vel_w", 0.0)),
        local_target_direction=float(agent._get_target_angle(current_target)),
        rolling_subgoal_direction=float(agent._get_target_angle(nominal_target)),
        option_state=str(option_state),
    )


def evaluate_option_feasibility(
    agent: Any,
    option_state: str = "idle",
    *,
    include_replan: bool = True,
) -> OptionFeasibilityResult:
    obs = extract_local_option_observation(agent, option_state=option_state)

    severe_head_on = bool(
        obs.social_risk_max >= 0.62
        and obs.nearest_neighbor_dist < 1.35
        and abs(obs.nearest_neighbor_bearing) < 0.40
        and obs.closing_speed > 0.08
        and (not math.isfinite(obs.ttc_min) or obs.ttc_min < 1.10)
    )
    front_choked = bool(obs.front_min < 0.30 or obs.front_center_min < 0.28)
    both_sides_narrow = bool(obs.left_min < 0.22 and obs.right_min < 0.22)
    rear_blocked = bool(obs.rear_min < 0.24)
    target_far_off_heading = bool(abs(obs.local_target_direction) > 1.45)
    front_blocked_ratio = float(
        np.clip(
            (float(getattr(agent, "subgoal_block_front_dist", 0.42)) - obs.front_center_min)
            / max(float(getattr(agent, "subgoal_block_front_dist", 0.42)), 1e-6),
            0.0,
            1.0,
        )
    )
    ttc_risk = 0.0
    if math.isfinite(obs.ttc_min):
        ttc_risk = float(np.clip((2.0 - obs.ttc_min) / 2.0, 0.0, 1.0))
    detour_need = bool(
        front_blocked_ratio > 0.30
        or obs.front_min < 0.38
        or obs.front_center_min < 0.36
        or obs.front_risk > 0.35
        or severe_head_on
        or (obs.social_risk_max > 0.50 and obs.nearest_neighbor_dist < 1.45 and ttc_risk > 0.15)
        or obs.stuck_score > 0.48
    )
    left_detour_clearance = float(min(obs.left_min, obs.front_left_min))
    right_detour_clearance = float(min(obs.right_min, obs.front_right_min))
    side_margin = 0.06
    side_clear_floor = 0.30

    reasons: Dict[str, List[str]] = {name: [] for name in OPTION_NAMES}

    if obs.front_min <= 0.30:
        reasons["follow_path"].append("front_min<=0.30")
    if obs.front_center_min <= 0.28:
        reasons["follow_path"].append("front_center_min<=0.28")
    if severe_head_on:
        reasons["follow_path"].append("severe_head_on")
    if target_far_off_heading:
        reasons["follow_path"].append("target_heading_too_large")

    if obs.front_min <= 0.20:
        reasons["slow_follow"].append("front_min<=0.20")
    if obs.front_center_min <= 0.18:
        reasons["slow_follow"].append("front_center_min<=0.18")
    if math.isfinite(obs.ttc_min) and obs.ttc_min < 0.45:
        reasons["slow_follow"].append("ttc_min<0.45")
    if abs(obs.local_target_direction) > 1.52:
        reasons["slow_follow"].append("target_heading_too_large")

    if (
        obs.front_center_min < float(getattr(agent, "collision_hard_dist", 0.20))
        and rear_blocked
        and both_sides_narrow
    ):
        reasons["stop_wait"].append("fully_boxed")

    if rear_blocked:
        reasons["backoff"].append("rear_min<=0.24")
    if obs.nearest_neighbor_dist < 0.45 and abs(obs.nearest_neighbor_bearing) > 2.40:
        reasons["backoff"].append("rear_neighbor_too_close")
    if obs.right_risk > 0.95 and obs.left_risk > 0.95 and rear_blocked:
        reasons["backoff"].append("rear_escape_corridor_unavailable")

    if obs.left_min <= 0.24:
        reasons["detour_left"].append("left_min<=0.24")
    if obs.front_left_min <= 0.22:
        reasons["detour_left"].append("front_left_min<=0.22")
    if both_sides_narrow:
        reasons["detour_left"].append("both_sides_narrow")
    if obs.left_risk >= 0.95:
        reasons["detour_left"].append("left_risk>=0.95")
    if not detour_need:
        reasons["detour_left"].append("no_blocked_or_conflict_need")
    if left_detour_clearance < side_clear_floor:
        reasons["detour_left"].append("left_detour_clearance<0.30")
    if left_detour_clearance + side_margin < right_detour_clearance:
        reasons["detour_left"].append("right_side_clearer")

    if obs.right_min <= 0.24:
        reasons["detour_right"].append("right_min<=0.24")
    if obs.front_right_min <= 0.22:
        reasons["detour_right"].append("front_right_min<=0.22")
    if both_sides_narrow:
        reasons["detour_right"].append("both_sides_narrow")
    if obs.right_risk >= 0.95:
        reasons["detour_right"].append("right_risk>=0.95")
    if not detour_need:
        reasons["detour_right"].append("no_blocked_or_conflict_need")
    if right_detour_clearance < side_clear_floor:
        reasons["detour_right"].append("right_detour_clearance<0.30")
    if right_detour_clearance + side_margin < left_detour_clearance:
        reasons["detour_right"].append("left_side_clearer")

    if include_replan:
        if not (
            obs.stuck_score > 0.58
            or (front_choked and both_sides_narrow)
            or (severe_head_on and rear_blocked and obs.left_min < 0.28 and obs.right_min < 0.28)
        ):
            reasons["replan"].append("no_deadlock_or_stall_signal")
    else:
        reasons["replan"].append("replan_disabled")

    feasible_by_option = {
        option_name: len(option_reasons) == 0
        for option_name, option_reasons in reasons.items()
    }
    if not include_replan:
        feasible_by_option["replan"] = False

    action_mask = np.zeros(len(OPTION_NAMES), dtype=np.int32)
    for option_name, idx in OPTION_INDEX.items():
        action_mask[idx] = 1 if feasible_by_option.get(option_name, False) else 0

    return OptionFeasibilityResult(
        local_metrics=obs,
        feasible_by_option=feasible_by_option,
        action_mask=action_mask,
        infeasible_reason_by_option=reasons,
    )


# ── 6-action mask builder for interaction_mode training ──
def build_interaction_action_mask(
    agent: Any,
    *,
    option_state: str = "go",
    include_replan: bool = False,
) -> "Tuple[np.ndarray, OptionFeasibilityResult]":
    """Build a 6-action feasibility mask for the training action space.

    Uses the 7-option feasibility evaluator and maps results to the
    6-action space (go, wait, backoff, detour_left, detour_right, slow_follow).

    """
    from gnn_marl_training.interaction_option_definitions import (
        TRAINING_OPTION_NAMES,
        TRAINING_OPTION_INDEX,
        TRAINING_TO_FEASIBILITY_OPTION,
        NUM_TRAINING_OPTIONS,
    )

    # Evaluate against the 7-option feasibility rules (replan always disabled)
    feasibility = evaluate_option_feasibility(
        agent,
        option_state=option_state,
        include_replan=False,
    )

    # Map 7-option feasibility → 6-action mask
    action_mask = np.ones(NUM_TRAINING_OPTIONS, dtype=np.int32)
    for action_name, idx in TRAINING_OPTION_INDEX.items():
        opt_name = TRAINING_TO_FEASIBILITY_OPTION.get(action_name, action_name)
        action_mask[idx] = 1 if feasibility.is_feasible(opt_name) else 0

    # Build per-action feasibility dict for the 6-action space
    feasible_by_action: Dict[str, bool] = {}
    infeasible_reason_by_action: Dict[str, List[str]] = {}
    for action_name in TRAINING_OPTION_NAMES:
        opt_name = TRAINING_TO_FEASIBILITY_OPTION.get(action_name, action_name)
        feasible_by_action[action_name] = bool(action_mask[TRAINING_OPTION_INDEX[action_name]])
        infeasible_reason_by_action[action_name] = list(
            feasibility.infeasible_reason_by_option.get(opt_name, [])
        )

    result = OptionFeasibilityResult(
        local_metrics=feasibility.local_metrics,
        feasible_by_option=feasible_by_action,
        action_mask=action_mask,
        infeasible_reason_by_option=infeasible_reason_by_action,
    )
    return action_mask, result


def summarize_mask(mask: Sequence[int]) -> Dict[str, int]:
    return {
        option_name: int(mask[OPTION_INDEX[option_name]])
        for option_name in OPTION_NAMES
    }


def feasibility_to_row(
    feasibility: OptionFeasibilityResult,
    *,
    prefix: str = "",
) -> Dict[str, Any]:
    row = {}
    for key, value in feasibility.local_metrics.to_dict().items():
        row[f"{prefix}{key}"] = value
    for option_name, feasible in feasibility.feasible_by_option.items():
        row[f"{prefix}feasible_{option_name}"] = int(bool(feasible))
        row[f"{prefix}reasons_{option_name}"] = "|".join(
            feasibility.infeasible_reason_by_option.get(option_name, [])
        )
    return row
