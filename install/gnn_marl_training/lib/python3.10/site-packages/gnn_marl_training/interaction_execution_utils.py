from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def build_interaction_subgoal_offset(
    *,
    mode: str,
    adaptive_lookahead: float,
    turn_sign: float,
    fallback_turn_sign: float,
    gap_angle: float,
    prev_turn_sign: float = 0.0,
    turn_flip_hysteresis: float = 0.18,
) -> Optional[Tuple[float, float]]:
    sign = float(turn_sign)
    if abs(sign) < 1e-6:
        sign = float(fallback_turn_sign)
    if abs(float(prev_turn_sign)) > 1e-6 and sign * float(prev_turn_sign) < 0.0:
        if abs(float(gap_angle)) < float(turn_flip_hysteresis):
            sign = float(prev_turn_sign)
    lookahead = max(0.20, float(adaptive_lookahead))

    if mode == "yield":
        return (
            max(0.05, 0.30 * lookahead),
            sign * max(0.18, 0.85 * lookahead),
        )
    if mode == "wait":
        return (0.03, sign * 0.10)
    if mode == "backoff":
        return (
            -max(0.16, 0.55 * lookahead),
            sign * max(0.14, 0.55 * lookahead),
        )
    if mode == "detour":
        if abs(float(gap_angle)) > 0.08:
            gap_sign = math.copysign(1.0, float(gap_angle))
            if abs(float(prev_turn_sign)) > 1e-6 and gap_sign * float(prev_turn_sign) < 0.0 and abs(float(gap_angle)) < float(turn_flip_hysteresis):
                sign = float(prev_turn_sign)
            else:
                sign = gap_sign
        return (
            max(0.16, 0.65 * lookahead),
            sign * max(0.22, 0.95 * lookahead),
        )
    return None


def compute_tracking_controller_cmd(
    *,
    tracking_target: Tuple[float, float],
    current_pose: Dict[str, float],
    max_forward_vel: float,
    max_reverse_vel: float,
    max_angular_vel: float,
    corner_escape_front_dist: float,
    yielding_stop_dist: float,
    yielding_ttc: float,
    front_min: float,
    left_min: float,
    right_min: float,
    turn_sign: float,
    severity: float,
    ttc: float,
    gap_angle: float,
    behavior_mode: str,
) -> Tuple[float, float]:
    dx = float(tracking_target[0] - current_pose["x"])
    dy = float(tracking_target[1] - current_pose["y"])
    yaw = float(current_pose["yaw"])
    target_angle = _wrap_angle(math.atan2(dy, dx) - yaw)
    target_dist = float(math.hypot(dx, dy))

    if abs(turn_sign) < 1e-6:
        turn_sign = 1.0 if left_min >= right_min else -1.0
    short_ttc = math.isfinite(ttc) and ttc < (0.50 * yielding_ttc)
    front_constrained = front_min < corner_escape_front_dist

    base_speed = 0.18 * float(np.clip(target_dist / 0.65, 0.35, 1.0))
    angular_vel = float(np.clip(1.15 * target_angle, -max_angular_vel, max_angular_vel))
    if abs(target_angle) > 0.95:
        base_speed = min(base_speed, 0.05)
    elif abs(target_angle) > 0.55:
        base_speed = min(base_speed, 0.10)
    elif abs(target_angle) > 0.30:
        base_speed = min(base_speed, 0.14)
    linear_vel = min(base_speed, max_forward_vel)

    if behavior_mode == "yield":
        target_speed = 0.12 if (not front_constrained and not short_ttc and severity < 0.55) else 0.08
        linear_vel = min(linear_vel, target_speed)
        if front_min < yielding_stop_dist or short_ttc:
            linear_vel = min(linear_vel, 0.04)
        angular_vel = float(np.clip(
            angular_vel + (0.22 + 0.16 * severity) * turn_sign,
            -max_angular_vel,
            max_angular_vel,
        ))
    elif behavior_mode == "wait":
        linear_vel = 0.0 if (front_constrained or short_ttc) else min(linear_vel, 0.05)
        if front_constrained or short_ttc or abs(angular_vel) < 0.20:
            angular_vel = float(np.clip(
                (0.30 + 0.25 * severity) * turn_sign,
                -max_angular_vel,
                max_angular_vel,
            ))
    elif behavior_mode == "backoff":
        linear_vel = -min(max_reverse_vel, 0.07 + 0.04 * severity)
        angular_vel = float(np.clip(
            (0.42 + 0.28 * severity) * turn_sign,
            -max_angular_vel,
            max_angular_vel,
        ))
    elif behavior_mode == "detour":
        detour_sign = float(turn_sign)
        if abs(float(gap_angle)) > 0.08:
            detour_sign = math.copysign(1.0, float(gap_angle))
        linear_vel = min(linear_vel, 0.12 if front_constrained else 0.15)
        angular_vel = float(np.clip(
            angular_vel + 0.28 * detour_sign,
            -max_angular_vel,
            max_angular_vel,
        ))
    elif behavior_mode == "replan":
        linear_vel = 0.0 if front_constrained or short_ttc else min(linear_vel, 0.05)
        angular_vel = float(np.clip(
            0.18 * turn_sign if abs(target_angle) > 0.20 else 0.0,
            -max_angular_vel,
            max_angular_vel,
        ))

    return (
        float(np.clip(linear_vel, -max_reverse_vel, max_forward_vel)),
        float(np.clip(angular_vel, -max_angular_vel, max_angular_vel)),
    )
