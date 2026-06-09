from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from gnn_marl_training.interaction_execution_utils import build_interaction_subgoal_offset
from gnn_marl_training.option_feasibility import (
    CANONICAL_MODE_BY_OPTION,
    LocalOptionObservation,
    OPTION_NAMES,
    OptionFeasibilityResult,
)


@dataclass
class PrimitiveCommand:
    option_name: str
    canonical_mode: str
    policy_mode: str
    executed_mode: str
    cmd_vel: Tuple[float, float]
    tracking_target: Tuple[float, float]
    nominal_target: Tuple[float, float]
    option_phase: str
    option_done: bool = False
    option_success: bool = False
    option_failed: bool = False
    failure_reason: str = ""
    safety_override: bool = False
    emergency_override: bool = False
    action_mask_allow: bool = True
    metadata: Dict[str, float] = field(default_factory=dict)


class AtomicOptionPrimitive:
    def __init__(
        self,
        option_name: str,
        *,
        max_steps: int = 12,
        min_steps: Optional[int] = None,
        terminate_on_success: bool = True,
        enable_safety_overrides: bool = True,
    ) -> None:
        if option_name not in OPTION_NAMES:
            raise ValueError(f"Unsupported option primitive: {option_name}")
        self.option_name = option_name
        self.canonical_mode = CANONICAL_MODE_BY_OPTION[option_name]
        self.max_steps = int(max(1, max_steps))
        self.min_steps = int(max(1, min_steps if min_steps is not None else self._default_min_steps()))
        self.terminate_on_success = bool(terminate_on_success)
        self.enable_safety_overrides = bool(enable_safety_overrides)
        self.reset()

    def _default_min_steps(self) -> int:
        if self.option_name in {"follow_path", "slow_follow"}:
            return 2
        if self.option_name == "stop_wait":
            return 4
        if self.option_name == "replan":
            return 1
        return 3

    def reset(self) -> None:
        self.started = False
        self.done = False
        self.success = False
        self.failed = False
        self.failure_reason = ""
        self.success_reason = ""
        self.option_phase = "idle"
        self.step_count = 0
        self.initial_feasible = True
        self.mask_allow_last = True
        self.start_pos: Optional[Tuple[float, float]] = None
        self.start_yaw: float = 0.0
        self.start_goal_dist = 0.0
        self.start_path_progress = 0.0
        self.start_front_min = 0.0
        self.start_rear_min = 0.0
        self.start_social_risk = 0.0
        self.start_ttc = float("inf")
        self.start_neighbor_dist = float("inf")
        self.last_goal_dist = 0.0
        self.last_path_progress = 0.0
        self.last_front_min = 0.0
        self.last_rear_min = 0.0
        self.last_social_risk = 0.0
        self.last_ttc = float("inf")
        self.last_neighbor_dist = float("inf")
        self.progress_gain = 0.0
        self.goal_distance_drop = 0.0
        self.front_clearance_gain = 0.0
        self.social_risk_drop = 0.0
        self.ttc_gain = 0.0
        self.lateral_displacement = 0.0
        self.backward_distance = 0.0
        self.rolling_pullback_count = 0
        self.safety_override_count = 0
        self.emergency_override_count = 0
        self.near_miss_count = 0
        self.wall_scrape_count = 0
        self._persistent_target: Optional[Tuple[float, float]] = None
        self._persistent_nominal_target: Optional[Tuple[float, float]] = None
        self._last_tracking_target: Optional[Tuple[float, float]] = None
        self._last_nominal_target: Optional[Tuple[float, float]] = None
        self._replan_attempted_internal = False

    def status(self) -> Dict[str, float | str | bool]:
        return {
            "option_name": self.option_name,
            "canonical_mode": self.canonical_mode,
            "started": self.started,
            "done": self.done,
            "success": self.success,
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "success_reason": self.success_reason,
            "option_phase": self.option_phase,
            "steps_executed": self.step_count,
            "initial_feasible": self.initial_feasible,
            "mask_allow_last": self.mask_allow_last,
            "progress_gain": self.progress_gain,
            "goal_distance_drop": self.goal_distance_drop,
            "front_clearance_gain": self.front_clearance_gain,
            "social_risk_drop": self.social_risk_drop,
            "ttc_gain": self.ttc_gain,
            "lateral_displacement": self.lateral_displacement,
            "backward_distance": self.backward_distance,
            "rolling_pullback_count": self.rolling_pullback_count,
            "safety_override_count": self.safety_override_count,
            "emergency_override_count": self.emergency_override_count,
            "near_miss_count": self.near_miss_count,
            "wall_scrape_count": self.wall_scrape_count,
        }

    def start(self, agent: Any, feasibility: OptionFeasibilityResult) -> None:
        if self.started:
            return
        obs = feasibility.local_metrics
        self.started = True
        self.option_phase = "start"
        self.initial_feasible = bool(feasibility.is_feasible(self.option_name))
        self.mask_allow_last = self.initial_feasible
        self.start_pos = (
            float(agent.current_pose["x"]),
            float(agent.current_pose["y"]),
        )
        self.start_yaw = float(agent.current_pose["yaw"])
        self.start_goal_dist = self._goal_distance(agent)
        self.start_path_progress = float(getattr(agent, "path_progress", 0.0))
        self.start_front_min = float(obs.front_min)
        self.start_rear_min = float(obs.rear_min)
        self.start_social_risk = float(obs.social_risk_max)
        self.start_ttc = float(obs.ttc_min)
        self.start_neighbor_dist = float(obs.nearest_neighbor_dist)
        self.last_goal_dist = self.start_goal_dist
        self.last_path_progress = self.start_path_progress
        self.last_front_min = self.start_front_min
        self.last_rear_min = self.start_rear_min
        self.last_social_risk = self.start_social_risk
        self.last_ttc = self.start_ttc
        self.last_neighbor_dist = self.start_neighbor_dist

    def terminate(
        self,
        *,
        success: bool,
        failed: bool,
        reason: str,
    ) -> None:
        if self.done:
            return
        self.done = True
        self.success = bool(success)
        self.failed = bool(failed)
        self.option_phase = "terminal"
        if success:
            self.success_reason = str(reason)
        if failed:
            self.failure_reason = str(reason)

    def _goal_distance(self, agent: Any) -> float:
        return float(
            math.hypot(
                float(agent.goal_pos[0]) - float(agent.current_pose["x"]),
                float(agent.goal_pos[1]) - float(agent.current_pose["y"]),
            )
        )

    def _preferred_turn_sign(self, agent: Any, obs: LocalOptionObservation) -> float:
        if self.option_name == "detour_left":
            return 1.0
        if self.option_name == "detour_right":
            return -1.0
        interaction = agent._get_interaction_context()
        sign = float(interaction.get("turn_sign", 0.0))
        if abs(sign) > 1e-6:
            return math.copysign(1.0, sign)
        return 1.0 if obs.left_min >= obs.right_min else -1.0

    def _nominal_info(self, agent: Any) -> Dict[str, Any]:
        return dict(agent._compute_nominal_tracking_info())

    def _build_persistent_protocol_target(
        self,
        agent: Any,
        mode: str,
        nominal_info: Dict[str, Any],
        turn_sign: float,
    ) -> Tuple[float, float]:
        adaptive = float(nominal_info.get("adaptive_lookahead", max(getattr(agent, "lookahead_dist", 0.0), 0.30)))
        if mode in {"wait", "backoff"} and hasattr(agent, "_build_interaction_subgoal"):
            target = agent._build_interaction_subgoal(mode, adaptive, turn_sign)
            if target is not None:
                return tuple(target)
        offset = build_interaction_subgoal_offset(
            mode="detour" if mode == "detour" else mode,
            adaptive_lookahead=adaptive,
            turn_sign=turn_sign,
            fallback_turn_sign=turn_sign,
            gap_angle=0.0,
        )
        if offset is None:
            if mode == "wait":
                offset = (0.03, 0.10 * turn_sign)
            elif mode == "backoff":
                offset = (-0.22, 0.16 * turn_sign)
            else:
                offset = (0.28, 0.24 * turn_sign)
        return tuple(agent._body_to_world_point(float(offset[0]), float(offset[1])))

    def _build_tracking_target(
        self,
        agent: Any,
        obs: LocalOptionObservation,
        nominal_info: Dict[str, Any],
    ) -> Tuple[Tuple[float, float], str, Dict[str, float]]:
        nominal_target = tuple(nominal_info["subgoal"])
        adaptive = float(nominal_info.get("adaptive_lookahead", 0.0))
        turn_sign = self._preferred_turn_sign(agent, obs)
        meta = {
            "adaptive_lookahead": adaptive,
            "turn_sign": turn_sign,
        }

        if self.option_name in {"follow_path", "slow_follow"}:
            return nominal_target, "go", meta

        if self.option_name == "stop_wait":
            if self._persistent_target is None:
                self._persistent_target = self._build_persistent_protocol_target(
                    agent,
                    "wait",
                    nominal_info,
                    turn_sign,
                )
                self._persistent_nominal_target = nominal_target
            return self._persistent_target, "wait", meta

        if self.option_name == "backoff":
            if self._persistent_target is None:
                self._persistent_target = self._build_persistent_protocol_target(
                    agent,
                    "backoff",
                    nominal_info,
                    turn_sign,
                )
                self._persistent_nominal_target = nominal_target
            return self._persistent_target, "backoff", meta

        if self.option_name in {"detour_left", "detour_right"}:
            if self._persistent_target is None:
                self._persistent_target = self._build_persistent_protocol_target(
                    agent,
                    "detour",
                    nominal_info,
                    turn_sign,
                )
                self._persistent_nominal_target = nominal_target
            return self._persistent_target, "detour", meta

        if self.option_name == "replan":
            if not self._replan_attempted_internal:
                self._replan_attempted_internal = True
                agent._try_replan_due_to_deadlock()
                nominal_info = self._nominal_info(agent)
                nominal_target = tuple(nominal_info["subgoal"])
            self._persistent_nominal_target = nominal_target
            return nominal_target, "replan", meta

        return nominal_target, self.canonical_mode, meta

    def _compute_cmd(
        self,
        agent: Any,
        target: Tuple[float, float],
        executed_mode: str,
    ) -> Tuple[float, float]:
        if self.option_name == "slow_follow":
            linear_vel, angular_vel = agent._compute_tracking_controller_cmd(target, "go")
            linear_vel = min(float(linear_vel), 0.08)
            angular_vel = float(np.clip(float(angular_vel), -0.95, 0.95))
            return float(linear_vel), float(angular_vel)
        return agent._compute_tracking_controller_cmd(target, executed_mode)

    def _override_target(
        self,
        agent: Any,
        mode: str,
        nominal_target: Tuple[float, float],
        obs: LocalOptionObservation,
    ) -> Tuple[float, float]:
        nominal_info = {"subgoal": nominal_target, "adaptive_lookahead": max(0.25, getattr(agent, "lookahead_dist", 0.0))}
        return self._build_persistent_protocol_target(
            agent,
            mode,
            nominal_info,
            self._preferred_turn_sign(agent, obs),
        )

    def _apply_safety_override(
        self,
        agent: Any,
        obs: LocalOptionObservation,
        target: Tuple[float, float],
        nominal_target: Tuple[float, float],
        executed_mode: str,
        cmd_vel: Tuple[float, float],
    ) -> Tuple[Tuple[float, float], str, Tuple[float, float], bool, bool]:
        if not self.enable_safety_overrides:
            return cmd_vel, executed_mode, target, False, False

        collision_hard = float(getattr(agent, "collision_hard_dist", 0.20))
        emergency = bool(
            obs.min_dist <= (collision_hard + 0.02)
            or (math.isfinite(obs.ttc_min) and obs.ttc_min < 0.18)
        )
        safety = bool(
            not emergency
            and (
                obs.front_center_min <= float(getattr(agent, "yielding_hard_stop_dist", 0.30))
                or obs.front_risk >= 0.92
            )
        )
        if not emergency and not safety:
            return cmd_vel, executed_mode, target, False, False

        override_mode = "wait"
        if emergency and obs.rear_min > max(collision_hard + 0.08, 0.24):
            override_mode = "backoff"

        override_target = self._override_target(agent, override_mode, nominal_target, obs)
        override_cmd = agent._compute_tracking_controller_cmd(override_target, override_mode)
        return override_cmd, override_mode, override_target, True, emergency

    def step(
        self,
        agent: Any,
        feasibility: OptionFeasibilityResult,
        *,
        force_execute: bool = True,
        enable_safety_overrides: Optional[bool] = None,
    ) -> PrimitiveCommand:
        if enable_safety_overrides is not None:
            self.enable_safety_overrides = bool(enable_safety_overrides)
        if not self.started:
            self.start(agent, feasibility)

        agent._last_replan_attempted = False
        agent._last_replan_success = False
        agent._last_replan_wall_time_sec = 0.0

        self.mask_allow_last = bool(feasibility.is_feasible(self.option_name))
        if not force_execute and not self.mask_allow_last:
            self.terminate(success=False, failed=True, reason="masked_infeasible")
            zero_cmd = PrimitiveCommand(
                option_name=self.option_name,
                canonical_mode=self.canonical_mode,
                policy_mode=self.canonical_mode,
                executed_mode="stop",
                cmd_vel=(0.0, 0.0),
                tracking_target=(float(agent.current_pose["x"]), float(agent.current_pose["y"])),
                nominal_target=(float(agent.current_pose["x"]), float(agent.current_pose["y"])),
                option_phase="masked",
                option_done=True,
                option_failed=True,
                failure_reason="masked_infeasible",
                action_mask_allow=False,
            )
            return zero_cmd

        obs = feasibility.local_metrics
        nominal_info = self._nominal_info(agent)
        target, executed_mode, meta = self._build_tracking_target(agent, obs, nominal_info)
        nominal_target = tuple(nominal_info["subgoal"])
        cmd_vel = self._compute_cmd(agent, target, executed_mode)
        cmd_vel, final_mode, final_target, safety_override, emergency_override = self._apply_safety_override(
            agent,
            obs,
            target,
            nominal_target,
            executed_mode,
            cmd_vel,
        )

        self.step_count += 1
        self.option_phase = "execute"
        if safety_override:
            self.safety_override_count += 1
        if emergency_override:
            self.emergency_override_count += 1

        tracking_angle = float(agent._get_target_angle(final_target))
        pullback_angle = abs(float(obs.rolling_subgoal_direction) - tracking_angle)
        pullback_angle = abs((pullback_angle + math.pi) % (2.0 * math.pi) - math.pi)
        if pullback_angle > 0.65:
            self.rolling_pullback_count += 1

        self._last_tracking_target = tuple(final_target)
        self._last_nominal_target = nominal_target
        meta.update(
            {
                "tracking_angle": tracking_angle,
                "rolling_pullback_angle": pullback_angle,
            }
        )

        return PrimitiveCommand(
            option_name=self.option_name,
            canonical_mode=self.canonical_mode,
            policy_mode=self.canonical_mode,
            executed_mode=final_mode,
            cmd_vel=(float(cmd_vel[0]), float(cmd_vel[1])),
            tracking_target=tuple(final_target),
            nominal_target=nominal_target,
            option_phase=self.option_phase,
            action_mask_allow=self.mask_allow_last,
            safety_override=safety_override,
            emergency_override=emergency_override,
            metadata=meta,
        )

    def _update_motion_deltas(self, agent: Any, obs: LocalOptionObservation) -> None:
        self.last_goal_dist = self._goal_distance(agent)
        self.last_path_progress = float(getattr(agent, "path_progress", 0.0))
        self.last_front_min = float(obs.front_min)
        self.last_rear_min = float(obs.rear_min)
        self.last_social_risk = float(obs.social_risk_max)
        self.last_ttc = float(obs.ttc_min)
        self.last_neighbor_dist = float(obs.nearest_neighbor_dist)

        self.goal_distance_drop = float(self.start_goal_dist - self.last_goal_dist)
        self.progress_gain = float(self.last_path_progress - self.start_path_progress)
        self.front_clearance_gain = float(self.last_front_min - self.start_front_min)
        self.social_risk_drop = float(self.start_social_risk - self.last_social_risk)
        if math.isfinite(self.start_ttc) and math.isfinite(self.last_ttc):
            self.ttc_gain = float(self.last_ttc - self.start_ttc)
        elif not math.isfinite(self.start_ttc) and math.isfinite(self.last_ttc):
            self.ttc_gain = 0.0
        elif math.isfinite(self.start_ttc) and not math.isfinite(self.last_ttc):
            self.ttc_gain = float(getattr(agent, "yielding_ttc", 2.4))

        if self.start_pos is not None:
            dx = float(agent.current_pose["x"]) - self.start_pos[0]
            dy = float(agent.current_pose["y"]) - self.start_pos[1]
            c = math.cos(self.start_yaw)
            s = math.sin(self.start_yaw)
            self.lateral_displacement = float((-s * dx) + (c * dy))
            forward_delta = float((c * dx) + (s * dy))
            self.backward_distance = float(max(0.0, -forward_delta))

    def _evaluate_success(self, agent: Any, obs: LocalOptionObservation) -> tuple[bool, str]:
        if self.step_count < self.min_steps:
            return False, ""

        if self.option_name == "follow_path":
            if self.progress_gain > 0.18 or self.goal_distance_drop > 0.22:
                return True, "progress_recovered"
        elif self.option_name == "slow_follow":
            if (
                (self.progress_gain > 0.10 or self.goal_distance_drop > 0.14)
                and float(abs(getattr(agent, "current_vel_x", 0.0))) <= 0.12
            ):
                return True, "slow_progress_recovered"
        elif self.option_name == "stop_wait":
            if (
                self.social_risk_drop > 0.10
                or self.ttc_gain > 0.50
                or self.last_neighbor_dist > (self.start_neighbor_dist + 0.12)
            ):
                return True, "risk_dropped"
        elif self.option_name == "backoff":
            if self.backward_distance > 0.06 and (
                self.front_clearance_gain > 0.15 or self.social_risk_drop > 0.08
            ):
                return True, "clearance_recovered"
        elif self.option_name == "detour_left":
            if self.lateral_displacement > 0.06 and (
                self.front_clearance_gain > 0.12
                or self.progress_gain > 0.10
                or self.social_risk_drop > 0.08
            ):
                return True, "left_detour_recovered"
        elif self.option_name == "detour_right":
            if self.lateral_displacement < -0.06 and (
                self.front_clearance_gain > 0.12
                or self.progress_gain > 0.10
                or self.social_risk_drop > 0.08
            ):
                return True, "right_detour_recovered"
        elif self.option_name == "replan":
            if bool(getattr(agent, "_last_replan_success", False)):
                return True, "replan_succeeded"
        return False, ""

    def _timeout_failure_reason(self, obs: LocalOptionObservation) -> str:
        if not self.initial_feasible:
            return "infeasible_on_start"
        if self.option_name == "backoff" and self.start_rear_min < 0.24:
            return "rear_blocked"
        if self.option_name == "detour_left" and self.start_front_min < 0.36 and obs.left_min < 0.24:
            return "left_detour_blocked"
        if self.option_name == "detour_right" and self.start_front_min < 0.36 and obs.right_min < 0.24:
            return "right_detour_blocked"
        if self.option_name == "stop_wait" and self.social_risk_drop < 0.02 and self.ttc_gain < 0.10:
            return "mutual_wait_no_risk_drop"
        if self.rolling_pullback_count >= max(2, self.step_count // 2) and self.progress_gain <= 0.05:
            return "rolling_subgoal_pullback"
        if self.safety_override_count >= max(2, self.step_count // 2):
            return "repeated_safety_override"
        return "option_timeout"

    def observe_transition(
        self,
        agent: Any,
        info: Dict[str, Any],
        obs: LocalOptionObservation,
        command: PrimitiveCommand,
    ) -> None:
        if self.done:
            return

        self.option_phase = "observe"
        self._update_motion_deltas(agent, obs)

        near_miss_thresh = float(getattr(agent, "collision_hard_dist", 0.20)) + 0.08
        if obs.min_dist <= near_miss_thresh or obs.social_risk_max >= 0.92:
            self.near_miss_count += 1
        if min(obs.left_min, obs.right_min) <= 0.10:
            self.wall_scrape_count += 1

        event = str(info.get("event", ""))
        if event == "collision":
            self.terminate(success=False, failed=True, reason="collision")
            return
        if event == "goal":
            self.terminate(success=True, failed=False, reason="goal_reached")
            return

        if self.option_name == "replan" and self.step_count >= 1:
            if bool(getattr(agent, "_last_replan_success", False)):
                self.terminate(success=True, failed=False, reason="replan_succeeded")
            elif self.step_count >= self.min_steps:
                self.terminate(success=False, failed=True, reason="replan_failed")
            return

        success, success_reason = self._evaluate_success(agent, obs)
        if success and self.terminate_on_success:
            self.terminate(success=True, failed=False, reason=success_reason)
            return

        if self.step_count >= self.max_steps:
            self.terminate(
                success=False,
                failed=True,
                reason=self._timeout_failure_reason(obs),
            )


def create_option_primitive(
    option_name: str,
    *,
    max_steps: int = 12,
    min_steps: Optional[int] = None,
    terminate_on_success: bool = True,
    enable_safety_overrides: bool = True,
) -> AtomicOptionPrimitive:
    return AtomicOptionPrimitive(
        option_name,
        max_steps=max_steps,
        min_steps=min_steps,
        terminate_on_success=terminate_on_success,
        enable_safety_overrides=enable_safety_overrides,
    )


def apply_primitive_command(
    agent: Any,
    command: PrimitiveCommand,
    *,
    global_step: int,
) -> None:
    agent.current_step += 1
    agent._policy_interaction_mode = str(command.policy_mode)
    agent._effective_interaction_mode = str(command.executed_mode)
    agent._executed_behavior_mode = str(command.executed_mode)
    agent._cached_step_tracking_target = tuple(command.tracking_target)
    agent._cached_step_tracking_mode = str(command.executed_mode)
    agent._cached_step_tracking_step = int(agent.current_step)
    agent._last_subgoal_mode = str(command.executed_mode)
    agent._last_nominal_subgoal = tuple(command.nominal_target)

    sectors = agent._scan_sector_metrics()
    front_min = float(sectors.get("front_min", getattr(agent, "scan_max_range", 3.5)))
    left_min = float(sectors.get("left_min", getattr(agent, "scan_max_range", 3.5)))
    right_min = float(sectors.get("right_min", getattr(agent, "scan_max_range", 3.5)))
    linear_vel = float(command.cmd_vel[0])
    angular_vel = float(command.cmd_vel[1])
    agent._last_control_info = {
        "front_min": front_min,
        "left_min": left_min,
        "right_min": right_min,
        "raw_linear_vel": linear_vel,
        "raw_angular_vel": angular_vel,
        "applied_linear_vel": linear_vel,
        "applied_angular_vel": angular_vel,
        "interaction_reason": f"option_tester:{command.option_name}",
        "global_step": int(global_step),
    }
    agent._publish_vel(linear_vel, angular_vel)
