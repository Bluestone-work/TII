from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class SocialRiskSummary:
    social_risk: float
    distance_risk: float
    ttc_risk: float
    rel_dist: float
    rel_bearing: float
    closing_speed: float
    ttc: float
    comm_valid: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "social_risk": self.social_risk,
            "distance_risk": self.distance_risk,
            "ttc_risk": self.ttc_risk,
            "rel_dist": self.rel_dist,
            "rel_bearing": self.rel_bearing,
            "closing_speed": self.closing_speed,
            "ttc": self.ttc,
            "comm_valid": self.comm_valid,
        }


def compute_social_risk_summary(
    *,
    current_pose: Dict[str, float],
    current_vel_x: float,
    self_agent_id: str,
    robot_positions: Dict[str, np.ndarray],
    robot_velocities: Dict[str, np.ndarray],
    communication_range: float,
    scan_max_range: float,
    predictive_social_range: float,
    predictive_social_ttc_safe: float,
    yielding_ttc: float,
) -> Dict[str, float]:
    my_pos = np.array([current_pose["x"], current_pose["y"]], dtype=np.float32)
    yaw = float(current_pose["yaw"])
    my_vel = np.array(
        [current_vel_x * math.cos(yaw), current_vel_x * math.sin(yaw)],
        dtype=np.float32,
    )

    comm_range = float(communication_range if communication_range > 0.0 else scan_max_range)
    best = SocialRiskSummary(
        social_risk=0.0,
        distance_risk=0.0,
        ttc_risk=0.0,
        rel_dist=1.0,
        rel_bearing=0.0,
        closing_speed=0.0,
        ttc=1.0,
        comm_valid=0.0,
    )
    best_key = (-1.0, float("inf"))

    for aid, pos in robot_positions.items():
        if aid == self_agent_id:
            continue
        rel = np.asarray(pos, dtype=np.float32) - my_pos
        dist = float(np.linalg.norm(rel))
        if dist < 1e-6:
            continue

        rel_unit = rel / max(dist, 1e-6)
        n_vel = np.asarray(
            robot_velocities.get(aid, np.zeros(2, dtype=np.float32)),
            dtype=np.float32,
        )
        rel_vel = n_vel - my_vel
        closing_speed = float(max(0.0, -np.dot(rel_vel, rel_unit)))
        ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 0.05 else float("inf")
        distance_risk = _clamp01((predictive_social_range - dist) / max(predictive_social_range, 1e-6))
        ttc_risk = (
            _clamp01((predictive_social_ttc_safe - ttc) / max(predictive_social_ttc_safe, 1e-6))
            if math.isfinite(ttc) else 0.0
        )
        social_risk = max(distance_risk, ttc_risk)
        key = (social_risk, -dist)
        if key <= best_key:
            continue
        bearing = _wrap_angle(math.atan2(rel[1], rel[0]) - yaw)
        best_key = key
        best = SocialRiskSummary(
            social_risk=_clamp01(social_risk),
            distance_risk=distance_risk,
            ttc_risk=ttc_risk,
            rel_dist=_clamp01(dist / max(comm_range, 1e-6)),
            rel_bearing=float(np.clip(bearing / math.pi, -1.0, 1.0)),
            closing_speed=_clamp01(closing_speed / 0.6),
            ttc=1.0 if not math.isfinite(ttc) else _clamp01(ttc / max(yielding_ttc, 1e-6)),
            comm_valid=1.0 if dist <= comm_range else 0.0,
        )
    return best.as_dict()


def compute_progress_delta_signal(
    *,
    path_progress: float,
    prev_path_progress: Optional[float],
    lookahead_dist: float,
) -> float:
    if prev_path_progress is not None:
        delta = float(path_progress - prev_path_progress)
    else:
        delta = 0.0
    return float(np.clip(delta / max(lookahead_dist, 0.25), -1.0, 1.0))


def compute_stuck_score(
    *,
    current_vel_x: float,
    progress_delta: float,
    stall_elapsed_sec: float,
    stall_global_replan_sec: float,
    front_blocked_ratio: float,
) -> float:
    low_speed = _clamp01((0.05 - abs(float(current_vel_x))) / 0.05)
    progress_stall = _clamp01((0.02 - max(progress_delta, 0.0)) / 0.02)
    stall_time_ratio = _clamp01(stall_elapsed_sec / max(stall_global_replan_sec, 1e-6))
    return _clamp01(
        0.40 * front_blocked_ratio
        + 0.25 * low_speed
        + 0.20 * progress_stall
        + 0.15 * stall_time_ratio
    )


def build_high_level_policy_features(
    *,
    stuck_score: float,
    front_blocked_ratio: float,
    wait_age_norm: float,
    progress_delta: float,
    social_risk: float,
    hold_fraction: float,
    target_x_body: float,
    target_y_body: float,
) -> np.ndarray:
    return np.array(
        [
            float(stuck_score),
            float(front_blocked_ratio),
            _clamp01(wait_age_norm),
            float(progress_delta),
            _clamp01(social_risk),
            _clamp01(hold_fraction),
            float(np.clip(target_x_body, -1.0, 1.0)),
            float(np.clip(target_y_body, -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_interaction_neighbor_token(
    *,
    my_pos: np.ndarray,
    my_vel: np.ndarray,
    my_yaw: float,
    neighbor_pos: np.ndarray,
    neighbor_vel: np.ndarray,
    perception_range: float,
    yielding_ttc: float,
) -> np.ndarray:
    rel_pos = np.asarray(neighbor_pos, dtype=np.float32) - np.asarray(my_pos, dtype=np.float32)
    dist = float(np.linalg.norm(rel_pos))
    rel_bearing = _wrap_angle(math.atan2(float(rel_pos[1]), float(rel_pos[0])) - float(my_yaw))
    rel_unit = rel_pos / max(dist, 1e-6)
    rel_vel = np.asarray(neighbor_vel, dtype=np.float32) - np.asarray(my_vel, dtype=np.float32)
    closing_speed = float(max(0.0, -np.dot(rel_vel, rel_unit)))
    ttc = float(dist / max(closing_speed, 1e-6)) if closing_speed > 0.05 else float("inf")
    return np.array(
        [
            float(np.clip(dist / max(perception_range, 1e-6), 0.0, 1.0)),
            float(np.clip(rel_bearing / math.pi, -1.0, 1.0)),
            float(np.clip(closing_speed / 0.6, 0.0, 1.0)),
            1.0 if not math.isfinite(ttc) else float(np.clip(ttc / max(yielding_ttc, 1e-6), 0.0, 1.0)),
            1.0,
        ],
        dtype=np.float32,
    )


# ── Path projection progress (avoids Euclidean-goal-only progress trap) ───

def project_point_to_polyline_arclength(
    point_xy: "Tuple[float, float]",
    path_points_xy: "List[Tuple[float, float]]",
) -> "Tuple[float, float, int, Tuple[float, float]]":
    """Project a 2-D point onto a polyline path and return arc-length progress.

    Args:
        point_xy: current robot position (x, y).
        path_points_xy: planned path waypoints for THIS robot only.

    Returns:
        s_proj: cumulative arc-length from path start to projection point.
        closest_dist: lateral distance from point to path (cross-track).
        segment_idx: index of the segment containing the projection.
        projection_xy: world coordinates of the projection point.
    """
    if not path_points_xy or len(path_points_xy) < 2:
        return 0.0, 0.0, 0, point_xy

    px, py = float(point_xy[0]), float(point_xy[1])
    best_s = 0.0
    best_dist = float("inf")
    best_seg = 0
    best_proj = point_xy
    cumulative_s = 0.0

    for i in range(len(path_points_xy) - 1):
        x0, y0 = float(path_points_xy[i][0]), float(path_points_xy[i][1])
        x1, y1 = float(path_points_xy[i + 1][0]), float(path_points_xy[i + 1][1])
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            cumulative_s += math.hypot(dx, dy)
            continue

        u = ((px - x0) * dx + (py - y0) * dy) / seg_len_sq
        u = max(0.0, min(1.0, u))
        proj_x = x0 + u * dx
        proj_y = y0 + u * dy
        dist = math.hypot(px - proj_x, py - proj_y)

        if dist < best_dist:
            best_dist = dist
            best_seg = i
            best_s = cumulative_s + u * math.sqrt(seg_len_sq)
            best_proj = (proj_x, proj_y)

        cumulative_s += math.sqrt(seg_len_sq)

    return float(best_s), float(best_dist), int(best_seg), best_proj
