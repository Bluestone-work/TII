"""Paper evaluation metrics with explicit, versioned definitions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


METRIC_DEFINITION_VERSION = "paper_metrics_v2"


def _finite_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _valid_position(value):
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size < 2 or not np.all(np.isfinite(array[:2])):
        return None
    if np.any(np.abs(array[:2]) >= 20.0):
        return None
    return array[:2]


def polyline_length(points: Sequence, start=None) -> float:
    """Return a finite 2-D polyline length, optionally prefixed by ``start``."""
    valid = []
    if start is not None:
        point = _valid_position(start)
        if point is not None:
            valid.append(point)
    for value in points or ():
        point = _valid_position(value)
        if point is not None:
            valid.append(point)
    if len(valid) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(valid, valid[1:])))


def initial_state_description(
    agent_ids,
    positions: Mapping,
    goals: Mapping,
    reference_routes: Mapping | None = None,
    reference_tolerance_m: float = 0.02,
):
    """Return the accepted start--goal record and its stable SHA-256 fingerprint."""
    def coordinate(value):
        rounded = round(float(value), 3)
        return 0.0 if rounded == 0.0 else rounded

    routes = {}
    for agent_id in sorted(agent_ids):
        start = _valid_position(positions.get(agent_id))
        goal = _valid_position(goals.get(agent_id))
        if start is None or goal is None:
            return {}, None
        reference = (reference_routes or {}).get(agent_id)
        if reference is not None:
            reference_start = _valid_position(reference[0])
            reference_goal = _valid_position(reference[1])
            tolerance = max(0.0, float(reference_tolerance_m))
            if (
                reference_start is not None
                and reference_goal is not None
                and float(np.linalg.norm(start - reference_start)) <= tolerance
                and float(np.linalg.norm(goal - reference_goal)) <= tolerance
            ):
                start = reference_start
                goal = reference_goal
        routes[str(agent_id)] = {
            "start": [coordinate(value) for value in start],
            "goal": [coordinate(value) for value in goal],
        }
    payload = json.dumps(routes, sort_keys=True, separators=(",", ":"))
    return routes, hashlib.sha256(payload.encode("ascii")).hexdigest()


def pair_time_to_collision(position_a, velocity_a, position_b, velocity_b, collision_distance):
    """Constant-velocity time until two discs first reach ``collision_distance``."""
    pa = _valid_position(position_a)
    pb = _valid_position(position_b)
    va = _valid_position(velocity_a)
    vb = _valid_position(velocity_b)
    if pa is None or pb is None or va is None or vb is None:
        return None
    relative_position = pa - pb
    relative_velocity = va - vb
    radius = max(0.0, float(collision_distance))
    c = float(np.dot(relative_position, relative_position) - radius * radius)
    if c <= 0.0:
        return 0.0
    a = float(np.dot(relative_velocity, relative_velocity))
    b = 2.0 * float(np.dot(relative_position, relative_velocity))
    if a <= 1e-12 or b >= 0.0:
        return None
    discriminant = b * b - 4.0 * a * c
    if discriminant < 0.0:
        return None
    ttc = (-b - math.sqrt(discriminant)) / (2.0 * a)
    return float(ttc) if ttc >= 0.0 and math.isfinite(ttc) else None


@dataclass(frozen=True)
class MetricThresholds:
    control_dt: float = 0.1
    near_collision_clearance_m: float = 0.45
    robot_collision_distance_m: float = 0.24
    deadlock_speed_mps: float = 0.03
    deadlock_progress_m: float = 0.005
    deadlock_window_steps: int = 50
    oscillation_angular_speed_rps: float = 0.20
    teleport_reject_distance_m: float = 1.0


class EpisodeMetricTracker:
    """Accumulate navigation, safety, smoothness, and intervention metrics."""

    def __init__(
        self,
        agent_ids,
        initial_positions: Mapping,
        shortest_path_lengths: Mapping,
        thresholds: MetricThresholds,
    ):
        self.agent_ids = tuple(agent_ids)
        self.thresholds = thresholds
        self.previous_positions = {
            aid: _valid_position(initial_positions.get(aid)) for aid in self.agent_ids
        }
        self.previous_goal_distances = {}
        self.path_lengths = {aid: 0.0 for aid in self.agent_ids}
        self.shortest_path_lengths = {
            aid: max(0.0, float(shortest_path_lengths.get(aid, 0.0)))
            for aid in self.agent_ids
        }
        self.completion_steps = {}
        self.clearances = []
        self.ttcs = []
        self.near_active = {aid: False for aid in self.agent_ids}
        self.near_collision_count = 0
        self.deadlock_streak = 0
        self.deadlock_detected = False
        self.deadlock_event_count = 0
        self.oscillation_count = 0
        self.previous_turn_sign = {aid: 0 for aid in self.agent_ids}
        self.previous_commands = {}
        self.previous_accelerations = {}
        self.linear_jerks = []
        self.angular_jerks = []
        self.filter_action_differences = []
        self.filter_interventions = []
        self.gate_means = []
        self.gate_stds = []

    def mark_event(self, agent_id: str, event: str, step: int):
        if event == "goal" and agent_id not in self.completion_steps:
            self.completion_steps[agent_id] = int(step)

    def _record_path(self, positions):
        max_segment = max(0.0, self.thresholds.teleport_reject_distance_m)
        for aid in self.agent_ids:
            current = _valid_position(positions.get(aid))
            previous = self.previous_positions.get(aid)
            if current is None:
                continue
            if previous is not None:
                segment = float(np.linalg.norm(current - previous))
                if segment <= max_segment:
                    self.path_lengths[aid] += segment
            self.previous_positions[aid] = current

    def _record_pair_ttc(self, positions, velocities, active_ids):
        active = sorted(set(active_ids))
        for index, aid in enumerate(active):
            for other in active[index + 1:]:
                ttc = pair_time_to_collision(
                    positions.get(aid), velocities.get(aid),
                    positions.get(other), velocities.get(other),
                    self.thresholds.robot_collision_distance_m,
                )
                if ttc is not None:
                    self.ttcs.append(ttc)

    def _record_deadlock(self, infos, commands, active_ids):
        active = [aid for aid in active_ids if aid in infos]
        if not active:
            self.deadlock_streak = 0
            return
        all_slow = True
        any_progress = False
        for aid in active:
            command = commands.get(aid, {})
            speed = abs(float(command.get("executed_linear_vel") or 0.0))
            all_slow = all_slow and speed < self.thresholds.deadlock_speed_mps
            current_distance = _finite_float(infos.get(aid, {}).get("dist_to_goal"))
            previous_distance = self.previous_goal_distances.get(aid)
            if current_distance is not None:
                if previous_distance is not None:
                    any_progress = any_progress or (
                        previous_distance - current_distance
                        >= self.thresholds.deadlock_progress_m
                    )
                self.previous_goal_distances[aid] = current_distance
        was_deadlocked = self.deadlock_streak >= self.thresholds.deadlock_window_steps
        self.deadlock_streak = self.deadlock_streak + 1 if all_slow and not any_progress else 0
        is_deadlocked = self.deadlock_streak >= self.thresholds.deadlock_window_steps
        if is_deadlocked and not was_deadlocked:
            self.deadlock_event_count += 1
        self.deadlock_detected = self.deadlock_detected or is_deadlocked

    def _record_commands(self, commands):
        dt = max(1e-6, float(self.thresholds.control_dt))
        turn_threshold = max(0.0, self.thresholds.oscillation_angular_speed_rps)
        for aid, command in commands.items():
            raw_v = _finite_float(command.get("raw_linear_vel"))
            raw_w = _finite_float(command.get("raw_angular_vel"))
            executed_v = _finite_float(command.get("executed_linear_vel"))
            executed_w = _finite_float(command.get("executed_angular_vel"))
            if executed_v is None or executed_w is None:
                continue

            if abs(executed_w) >= turn_threshold:
                sign = 1 if executed_w > 0.0 else -1
                if self.previous_turn_sign.get(aid, 0) not in (0, sign):
                    self.oscillation_count += 1
                self.previous_turn_sign[aid] = sign

            previous_command = self.previous_commands.get(aid)
            if previous_command is not None:
                acceleration = (
                    (executed_v - previous_command[0]) / dt,
                    (executed_w - previous_command[1]) / dt,
                )
                previous_acceleration = self.previous_accelerations.get(aid)
                if previous_acceleration is not None:
                    self.linear_jerks.append(abs((acceleration[0] - previous_acceleration[0]) / dt))
                    self.angular_jerks.append(abs((acceleration[1] - previous_acceleration[1]) / dt))
                self.previous_accelerations[aid] = acceleration
            self.previous_commands[aid] = (executed_v, executed_w)

            if raw_v is not None and raw_w is not None:
                difference = math.hypot(raw_v - executed_v, raw_w - executed_w)
                self.filter_action_differences.append(difference)
                self.filter_interventions.append(float(difference > 1e-6))

            gate_mean = _finite_float(command.get("gate_mean"))
            gate_std = _finite_float(command.get("gate_std"))
            if gate_mean is not None:
                self.gate_means.append(gate_mean)
            if gate_std is not None:
                self.gate_stds.append(gate_std)

    def record_step(self, *, positions, velocities, infos, commands, active_ids):
        self._record_path(positions)
        self._record_pair_ttc(positions, velocities, active_ids)
        self._record_commands(commands)
        self._record_deadlock(infos, commands, active_ids)

        for aid in active_ids:
            info = infos.get(aid, {})
            clearance = _finite_float(info.get("min_dist"))
            if clearance is not None:
                self.clearances.append(clearance)
                near = clearance < self.thresholds.near_collision_clearance_m
                if near and not self.near_active.get(aid, False):
                    self.near_collision_count += 1
                self.near_active[aid] = near
            dynamic_ttc = _finite_float(info.get("dynamic_obs_min_ttc"))
            if dynamic_ttc is not None and dynamic_ttc >= 0.0:
                self.ttcs.append(dynamic_ttc)

    @staticmethod
    def _mean(values):
        return float(np.mean(values)) if values else float("nan")

    def summary(self, *, reached_agents, steps: int):
        reached = set(reached_agents)
        spl_values = []
        for aid in self.agent_ids:
            shortest = self.shortest_path_lengths[aid]
            actual = self.path_lengths[aid]
            if aid in reached and shortest > 0.0:
                spl_values.append(shortest / max(shortest, actual, 1e-9))
            else:
                spl_values.append(0.0)
        completion_times = [
            step * self.thresholds.control_dt for step in self.completion_steps.values()
        ]
        filter_intervention_count = int(sum(self.filter_interventions))
        filter_action_samples = len(self.filter_interventions)
        team_success = len(reached) == len(self.agent_ids)
        linear_jerk = self._mean(self.linear_jerks)
        angular_jerk = self._mean(self.angular_jerks)
        combined_jerks = [
            math.hypot(linear, angular)
            for linear, angular in zip(self.linear_jerks, self.angular_jerks)
        ]
        return {
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "path_length": self._mean(list(self.path_lengths.values())),
            "path_length_sum": float(sum(self.path_lengths.values())),
            "shortest_path_length": self._mean(list(self.shortest_path_lengths.values())),
            "shortest_path_length_sum": float(sum(self.shortest_path_lengths.values())),
            "spl": float(np.mean(spl_values)),
            "completion_time": float(steps) * self.thresholds.control_dt,
            "successful_agent_completion_time_mean": self._mean(completion_times),
            "successful_agent_completion_time_sum": float(sum(completion_times)),
            "successful_agent_completion_time_count": len(completion_times),
            "team_completion_time": (
                max(completion_times) if team_success and completion_times else float("nan")
            ),
            "minimum_clearance": min(self.clearances) if self.clearances else float("nan"),
            "clearance_p05": (
                float(np.percentile(self.clearances, 5)) if self.clearances else float("nan")
            ),
            "minimum_ttc": min(self.ttcs) if self.ttcs else float("nan"),
            "near_collision_count": int(self.near_collision_count),
            "deadlock": bool(self.deadlock_detected),
            "deadlock_event_count": int(self.deadlock_event_count),
            "oscillation_count": int(self.oscillation_count),
            "control_jerk": self._mean(combined_jerks),
            "control_jerk_linear": linear_jerk,
            "control_jerk_angular": angular_jerk,
            "filter_intervention_count": filter_intervention_count,
            "filter_action_samples": filter_action_samples,
            "filter_intervention_rate": (
                filter_intervention_count / filter_action_samples
                if filter_action_samples else float("nan")
            ),
            "filter_action_difference": self._mean(self.filter_action_differences),
            "filter_action_difference_max": (
                max(self.filter_action_differences)
                if self.filter_action_differences else float("nan")
            ),
            "gate_mean": self._mean(self.gate_means),
            "gate_std": self._mean(self.gate_stds),
            "near_collision_clearance_m": self.thresholds.near_collision_clearance_m,
            "deadlock_window_steps": self.thresholds.deadlock_window_steps,
            "deadlock_speed_mps": self.thresholds.deadlock_speed_mps,
            "oscillation_angular_speed_rps": self.thresholds.oscillation_angular_speed_rps,
            "control_dt": self.thresholds.control_dt,
        }
