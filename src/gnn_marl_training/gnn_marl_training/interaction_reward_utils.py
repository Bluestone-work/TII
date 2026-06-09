from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Method3RewardTerms:
    interaction_mode_reward: float
    interaction_mode_penalty: float
    high_level_interaction_reward: float
    high_level_safety_reward: float
    high_level_efficiency_penalty: float
    high_level_policy_penalty: float
    progress_delta_window: float
    path_progress_delta_window: float
    goal_progress_delta_window: float
    social_risk_delta: float
    clear_reward: float
    blocked_score: float
    stuck_score: float
    replan_cost: float
    replan_freq_penalty: float
    replan_time_penalty: float


def compute_method3_reward_terms(
    *,
    policy_mode: str,
    interaction_social_risk: float,
    progress_delta_window: float,
    path_progress_delta_window: float,
    goal_progress_delta_window: float,
    social_risk_delta_window: float,
    clear_reward_window: float,
    blocked_score: float,
    stuck_score: float,
    wait_age_norm: float,
    front_close_ratio: float,
    side_close_ratio: float,
    stall_elapsed_sec: float,
    stall_global_replan_sec: float,
    last_subgoal_mode: str,
    replan_attempted: bool,
    replan_recent_count: int,
    replan_wall_time_sec: float,
    replan_time_budget_sec: float,
    replan_fixed_cost: float,
    replan_freq_cost: float,
    replan_time_cost: float,
) -> Method3RewardTerms:
    progress_delta_window = float(progress_delta_window)
    path_progress_delta_window = float(path_progress_delta_window)
    goal_progress_delta_window = float(goal_progress_delta_window)
    social_risk_delta = float(social_risk_delta_window)
    clear_reward = float(clear_reward_window)
    near_penalty = -0.16 * (front_close_ratio + 0.50 * side_close_ratio)
    replan_needed = bool(
        stuck_score > 0.55 or stall_elapsed_sec >= stall_global_replan_sec
    )

    interaction_mode_reward = 0.0
    interaction_mode_penalty = 0.0
    high_level_efficiency_penalty = 0.0
    high_level_policy_penalty = 0.0
    replan_freq_penalty = 0.0
    replan_time_penalty = 0.0
    replan_cost = 0.0
    real_risk_need = bool(
        interaction_social_risk > 0.35
        or blocked_score > 0.25
        or stuck_score > 0.45
        or front_close_ratio > 0.25
    )
    useful_relief = bool(
        social_risk_delta > 0.025
        or clear_reward > 0.015
        or progress_delta_window > 0.008
        or path_progress_delta_window > 0.008
        or goal_progress_delta_window > 0.008
    )

    if policy_mode == "go":
        if interaction_social_risk < 0.25 and blocked_score < 0.20:
            interaction_mode_reward += 0.02
            interaction_mode_reward += 0.05 * max(path_progress_delta_window, 0.0)
            interaction_mode_reward += 0.03 * max(goal_progress_delta_window, 0.0)
        if interaction_social_risk > 0.55:
            interaction_mode_penalty -= 0.10 * interaction_social_risk
        if stuck_score > 0.45 and blocked_score > 0.30:
            interaction_mode_penalty -= 0.05 * stuck_score
    elif policy_mode == "wait":
        relief_signal = max(0.0, social_risk_delta, clear_reward)
        if real_risk_need and useful_relief:
            interaction_mode_reward += 0.035 * max(social_risk_delta, 0.0)
            interaction_mode_reward += 0.025 * clear_reward
        elif not real_risk_need:
            interaction_mode_penalty -= 0.09 + 0.05 * wait_age_norm
        else:
            interaction_mode_penalty -= 0.055 * (0.50 + 0.50 * wait_age_norm)
    elif policy_mode == "backoff":
        if real_risk_need and useful_relief:
            interaction_mode_reward += 0.035 * max(social_risk_delta, 0.0)
            interaction_mode_reward += 0.030 * clear_reward
        else:
            interaction_mode_penalty -= 0.10 if not real_risk_need else 0.06
    elif policy_mode == "detour":
        if blocked_score > 0.25 or stuck_score > 0.35:
            interaction_mode_reward += 0.04 * max(progress_delta_window, 0.0)
            interaction_mode_reward += 0.05 * max(path_progress_delta_window, 0.0)
            interaction_mode_reward += 0.04 * max(goal_progress_delta_window, 0.0)
            interaction_mode_reward += 0.07 * clear_reward
        else:
            interaction_mode_penalty -= 0.04
    elif policy_mode == "replan":
        replan_freq_penalty = max(0.0, float(replan_recent_count)) * max(0.0, float(replan_freq_cost))
        if replan_attempted:
            replan_time_penalty = max(0.0, float(replan_time_cost)) * min(
                float(replan_wall_time_sec) / max(float(replan_time_budget_sec), 1e-6),
                1.0,
            )
        replan_cost = max(0.0, float(replan_fixed_cost)) + replan_freq_penalty + replan_time_penalty
        interaction_mode_penalty -= replan_cost
        if replan_needed:
            interaction_mode_reward += 0.04
            if last_subgoal_mode == "replan":
                interaction_mode_reward += 0.05 * clear_reward
                interaction_mode_reward += 0.03 * max(goal_progress_delta_window, 0.0)
        else:
            interaction_mode_penalty -= 0.05

    high_level_interaction_reward = 0.60 * social_risk_delta + 0.40 * clear_reward
    high_level_safety_reward = near_penalty - 0.25 * interaction_social_risk
    if policy_mode in {"wait", "backoff"} and not useful_relief:
        high_level_efficiency_penalty -= 0.04 * (0.50 + 0.50 * wait_age_norm)

    return Method3RewardTerms(
        interaction_mode_reward=float(interaction_mode_reward),
        interaction_mode_penalty=float(interaction_mode_penalty),
        high_level_interaction_reward=float(high_level_interaction_reward),
        high_level_safety_reward=float(high_level_safety_reward),
        high_level_efficiency_penalty=float(high_level_efficiency_penalty),
        high_level_policy_penalty=float(high_level_policy_penalty),
        progress_delta_window=float(progress_delta_window),
        path_progress_delta_window=float(path_progress_delta_window),
        goal_progress_delta_window=float(goal_progress_delta_window),
        social_risk_delta=float(social_risk_delta),
        clear_reward=float(clear_reward),
        blocked_score=float(blocked_score),
        stuck_score=float(stuck_score),
        replan_cost=float(replan_cost),
        replan_freq_penalty=float(replan_freq_penalty),
        replan_time_penalty=float(replan_time_penalty),
    )


# ── New: Option Outcome Reward (Phase 6) ───────────────────────────────────

@dataclass(frozen=True)
class OptionOutcomeRewardTerms:
    """Reward components for feasibility-aware closed-loop option policy."""
    # Option outcome
    option_progress_reward: float = 0.0
    option_clearance_reward: float = 0.0
    option_safety_reward: float = 0.0
    option_completion_bonus: float = 0.0
    option_failure_penalty: float = 0.0
    option_timeout_penalty: float = 0.0
    # Safe turn
    safe_turn_reward: float = 0.0
    wrong_turn_penalty: float = 0.0
    random_turn_penalty: float = 0.0
    # Stability
    spin_without_progress_penalty: float = 0.0
    idle_without_progress_penalty: float = 0.0
    conservative_mode_penalty: float = 0.0
    option_switch_penalty: float = 0.0
    infeasible_action_penalty: float = 0.0
    # Pair event
    pair_cooperative_reward: float = 0.0
    pair_competitive_penalty: float = 0.0
    # Obstacle proximity
    obstacle_proximity_penalty: float = 0.0
    cross_track_penalty: float = 0.0
    backoff_release_reward: float = 0.0
    detour_loop_penalty: float = 0.0
    # Progress diagnostics
    positive_path_projection_progress: float = 0.0
    negative_path_projection_progress: float = 0.0
    positive_goal_progress: float = 0.0
    positive_local_goal_progress: float = 0.0
    positive_guide_target_progress: float = 0.0
    progress_positive: float = 0.0
    progress_source_id: float = 0.0
    obstacle_risk_drop: float = 0.0
    ttc_improvement: float = 0.0
    front_blocked_ratio_delta: float = 0.0
    # Safe-turn diagnostics
    left_safety_score: float = 0.0
    right_safety_score: float = 0.0
    ttc_risk: float = 0.0
    risk_gate: float = 0.0
    correct_turn: float = 0.0
    wrong_turn: float = 0.0
    risk_reduced: float = 0.0
    # Legacy-compatible aggregates
    interaction_mode_reward: float = 0.0
    interaction_mode_penalty: float = 0.0
    social_risk_delta: float = 0.0
    clear_reward: float = 0.0
    blocked_score: float = 0.0
    stuck_score: float = 0.0


def compute_option_outcome_reward(
    *,
    effective_mode: str,
    policy_mode: str,
    # Progress deltas
    progress_delta: float,
    path_progress_delta: float,
    goal_progress_delta: float,
    local_goal_progress_delta: float = 0.0,
    path_projection_progress_delta: float = 0.0,
    path_projection_progress_window: float = 0.0,
    guide_target_progress_delta: float = 0.0,
    closest_dist_to_path: float = 0.0,
    cross_track_error: float = 0.0,
    # Risk
    social_risk: float,
    social_risk_delta: float,
    front_risk: float,
    ttc_min: float,
    ttc_delta: float,
    # Clearance
    front_min: float,
    left_min: float,
    right_min: float,
    front_blocked_ratio: float,
    blocked_score: float,
    stuck_score: float,
    # Option state
    option_elapsed: int,
    option_duration_steps: int,
    option_just_completed: bool,
    option_success: bool,
    option_failed: bool,
    # Detour
    detour_lateral_displacement: float = 0.0,
    # Turn
    applied_angular_vel: float = 0.0,
    applied_linear_vel: float = 0.0,
    # Feasibility
    action_was_feasible: bool = True,
    policy_vs_effective_mismatch: bool = False,
    # Pair
    pair_partner_id: str = "",
    pair_partner_mode: str = "",
    pair_dist: float = float("inf"),
    pair_closing_speed: float = 0.0,
    pair_ttc: float = float("inf"),
    pair_mode_complementary: bool = False,
    # Progress gate
    progress_positive: bool = False,
    progress_source: str = "none",
    risk_reduced: bool = False,
    front_blocked_ratio_delta: float = 0.0,
    obstacle_risk_drop: float = 0.0,
    # Constants
    safe_turn_reward_scale: float = 0.15,
    collision_penalty_base: float = 20.0,
) -> OptionOutcomeRewardTerms:
    """Compute reward for feasibility-aware closed-loop option policy.

    Rewards are attributed to the *effective_mode* (the option that actually
    executed), not the raw policy output.  This keeps credit assignment clean.
    """
    mode = str(effective_mode)
    # ── 7.1 R_progress (weighted path-projection progress) ──
    progress_clip = 0.08
    _proj_prog = float(path_projection_progress_delta)
    _proj_pos = float(np.clip(_proj_prog, 0.0, progress_clip))
    _proj_neg = float(np.clip(-_proj_prog, 0.0, progress_clip))
    _goal_pos = float(np.clip(goal_progress_delta, 0.0, progress_clip))
    _guide_pos = float(np.clip(guide_target_progress_delta, 0.0, progress_clip))
    _local_goal_delta = float(local_goal_progress_delta if math.isfinite(float(local_goal_progress_delta)) else path_progress_delta)
    _local_pos = float(np.clip(_local_goal_delta, 0.0, progress_clip))

    w_proj = 0.45
    w_goal = 0.25
    w_local = 0.20
    w_guide = 0.20
    w_backtrack = 0.35

    # Mode-specific adjustments
    risk_gate_val = max(float(social_risk), float(front_risk), float(front_blocked_ratio))
    if math.isfinite(float(ttc_min)) and float(ttc_min) < 2.0:
        risk_gate_val = max(risk_gate_val, float(np.clip((2.0 - float(ttc_min)) / 2.0, 0.0, 1.0)))

    # backoff: soften backtrack penalty in high risk
    if mode == "backoff" and risk_gate_val > 0.30:
        w_backtrack *= 0.35
        w_guide = 0.30
    # detour: allow lateral deviation, but do not reward circular guide chasing.
    if mode in ("detour_left", "detour_right"):
        w_guide = 0.12
        w_local = 0.25

    option_progress_reward = (
        w_proj * _proj_pos
        + w_goal * _goal_pos
        + w_local * _local_pos
        + w_guide * _guide_pos
        - w_backtrack * _proj_neg
    )
    # ── Cross-track penalty (light, mode-aware) ──
    allowed_band = 0.50
    if mode in ("detour_left", "detour_right", "backoff"):
        allowed_band *= 1.5
    k_cross = 0.05
    cross_track_penalty = 0.0
    _cte = float(cross_track_error)
    if _cte > allowed_band:
        cross_track_penalty = -k_cross * float(np.clip((_cte - allowed_band) / allowed_band, 0.0, 1.0))
    option_progress_reward += float(cross_track_penalty)

    # ── 7.2 R_safety (obstacle proximity) ──
    front_safe = 0.40
    side_safe = 0.35
    corner_safe = 0.30
    front_obstacle_risk = float(np.clip((front_safe - max(0.0, float(front_min))) / front_safe, 0.0, 1.0))
    side_obstacle_risk = float(np.clip(
        (side_safe - max(0.0, min(float(left_min), float(right_min)))) / side_safe, 0.0, 1.0
    ))
    obstacle_proximity_penalty = -0.10 * front_obstacle_risk - 0.08 * side_obstacle_risk
    obstacle_proximity_penalty = float(max(obstacle_proximity_penalty, -0.35))

    # ── 7.3 R_interaction ──
    social_risk_delta_out = float(social_risk_delta)
    clear_reward_out = max(0.0, float(_proj_pos)) * (1.0 - float(blocked_score))

    # ── 7.4 R_option_outcome (completion bonus) ──
    option_completion_bonus = 0.0
    option_failure_penalty = 0.0
    option_timeout_penalty = 0.0
    backoff_release_reward = 0.0
    if option_just_completed:
        if option_success:
            decay = 1.0 - 0.02 * min(float(option_elapsed), 50.0)
            option_completion_bonus = 0.15 * decay
        elif option_failed:
            option_failure_penalty = -0.10
    elif option_elapsed >= option_duration_steps and not option_success:
        option_timeout_penalty = -0.05

    if mode == "backoff" and risk_gate_val > 0.30:
        release_signal = max(
            0.0,
            float(social_risk_delta),
            float(obstacle_risk_drop),
            float(-front_blocked_ratio_delta),
        )
        if release_signal > 0.0:
            backoff_release_reward = 0.05 * float(np.clip(release_signal, 0.0, 1.0))

    # ── 7.5 R_safe_turn ──
    safe_turn_reward = 0.0
    wrong_turn_penalty = 0.0
    random_turn_penalty = 0.0
    spin_without_progress_penalty = 0.0
    idle_without_progress_penalty = 0.0
    conservative_mode_penalty = 0.0

    ttc_risk = 0.0
    if math.isfinite(float(ttc_min)) and float(ttc_min) < 2.0:
        ttc_risk = float(np.clip((2.0 - float(ttc_min)) / 2.0, 0.0, 1.0))
    risk_gate = max(float(social_risk), float(front_risk), float(ttc_risk), float(front_blocked_ratio))

    ttc_improvement = max(0.0, float(ttc_delta))
    risk_reduced_effective = bool(
        risk_reduced
        or float(social_risk_delta) > 0.02
        or ttc_improvement > 0.10
        or float(front_blocked_ratio_delta) < -0.05
        or float(obstacle_risk_drop) > 0.05
    )

    turning = abs(float(applied_angular_vel)) > 0.15
    low_forward = abs(float(applied_linear_vel)) < 0.03
    no_progress = not bool(progress_positive)
    no_relief = not bool(risk_reduced_effective)
    no_useful_outcome = no_progress and no_relief
    corner_escape_active = bool(float(env_snapshot.get("corner_escape_active", 0.0)) > 0.5)
    subgoal_deadlock_streak = float(env_snapshot.get("subgoal_deadlock_streak", 0.0))
    deadlock_escape_active = corner_escape_active or float(stuck_score) > 0.58 or subgoal_deadlock_streak >= 3.0
    escape_relax = 0.45 if deadlock_escape_active else 1.0
    turn_sign = 1.0 if float(applied_angular_vel) > 0.0 else (-1.0 if float(applied_angular_vel) < 0.0 else 0.0)
    left_score = 0.0
    right_score = 0.0
    correct_turn = 0.0
    wrong_turn = 0.0

    if no_useful_outcome:
        if low_forward:
            idle_without_progress_penalty = -0.05 * escape_relax
            if mode in ("go", "slow_follow"):
                idle_without_progress_penalty -= 0.04 * escape_relax
            elif mode == "wait":
                idle_without_progress_penalty -= (0.03 if risk_gate < 0.35 else 0.01) * escape_relax
            elif mode in ("detour_left", "detour_right"):
                idle_without_progress_penalty -= 0.02 * escape_relax
        elif float(applied_linear_vel) < 0.0 and mode not in ("backoff",) and not deadlock_escape_active:
            idle_without_progress_penalty -= 0.04

    if turning:
        left_clear = max(0.0, float(left_min))
        right_clear = max(0.0, float(right_min))
        clear_ref = 0.80
        left_score = float(np.clip(left_clear / clear_ref, 0.0, 1.0))
        right_score = float(np.clip(right_clear / clear_ref, 0.0, 1.0))

        is_spinning = abs(float(applied_angular_vel)) > 0.30

        if is_spinning and low_forward and no_useful_outcome:
            spin_without_progress_penalty = -0.10 * escape_relax
            if risk_gate > 0.30:
                spin_without_progress_penalty -= 0.05 * escape_relax

        if risk_gate > 0.30:
            if turn_sign > 0 and left_score > right_score + 0.05:
                correct_turn = 1.0
                margin = left_score - right_score
                if progress_positive or risk_reduced_effective:
                    safe_turn_reward = float(safe_turn_reward_scale) * risk_gate * margin
                else:
                    spin_without_progress_penalty = min(spin_without_progress_penalty, -0.04)
            elif turn_sign < 0 and right_score > left_score + 0.05:
                correct_turn = 1.0
                margin = right_score - left_score
                if progress_positive or risk_reduced_effective:
                    safe_turn_reward = float(safe_turn_reward_scale) * risk_gate * margin
                else:
                    spin_without_progress_penalty = min(spin_without_progress_penalty, -0.04)
            elif turn_sign > 0 and right_score > left_score + 0.05:
                wrong_turn = 1.0
                wrong_turn_penalty = -0.08 * risk_gate
            elif turn_sign < 0 and left_score > right_score + 0.05:
                wrong_turn = 1.0
                wrong_turn_penalty = -0.08 * risk_gate

        elif risk_gate < 0.15:
            random_turn_penalty = -0.03

        # Mode-specific turn handling
        if mode == "wait" and abs(float(applied_angular_vel)) > 0.35:
            if no_useful_outcome:
                spin_without_progress_penalty = min(spin_without_progress_penalty, -0.08)
        if mode in ("detour_left", "detour_right"):
            detour_side = -1.0 if mode == "detour_right" else 1.0
            if turn_sign != 0 and turn_sign != detour_side and risk_gate > 0.30:
                wrong_turn_penalty = min(wrong_turn_penalty, -0.06)

    if mode == "wait" and no_useful_outcome and option_elapsed >= max(2, option_duration_steps // 2):
        idle_without_progress_penalty -= 0.03 * escape_relax
    if mode in ("go", "slow_follow") and no_useful_outcome and blocked_score < 0.25 and risk_gate < 0.30:
        idle_without_progress_penalty -= 0.03 * escape_relax

    if mode in ("wait", "backoff"):
        escape_need = max(float(risk_gate), float(blocked_score), float(stuck_score))
        if escape_need < 0.25:
            conservative_mode_penalty -= 0.08
        elif not risk_reduced_effective and not progress_positive:
            conservative_mode_penalty -= 0.05 * escape_relax
        if mode == "backoff" and float(applied_linear_vel) < -0.03:
            if escape_need < 0.35 and not deadlock_escape_active:
                conservative_mode_penalty -= 0.08
            if (_proj_neg > 0.004 or float(goal_progress_delta) < -0.004) and not deadlock_escape_active:
                conservative_mode_penalty -= 0.06
        if option_elapsed >= max(4, option_duration_steps // 2) and not risk_reduced_effective:
            conservative_mode_penalty -= 0.03 * escape_relax

    detour_loop_penalty = 0.0
    if mode in ("detour_left", "detour_right"):
        detour_need = max(float(risk_gate), float(blocked_score), float(stuck_score))
        useful_progress = bool(
            _proj_pos > 0.004
            or _goal_pos > 0.004
            or _local_pos > 0.004
            or _guide_pos > 0.008
        )
        if detour_need < 0.25:
            detour_loop_penalty -= 0.06
        if not useful_progress and not risk_reduced_effective:
            detour_loop_penalty -= 0.05
            if option_elapsed >= max(3, option_duration_steps // 3):
                detour_loop_penalty -= 0.04
        if option_elapsed >= max(6, option_duration_steps // 2) and _proj_pos <= 0.002 and _goal_pos <= 0.002:
            detour_loop_penalty -= 0.04
        if float(detour_lateral_displacement) > 0.55 and _proj_pos <= 0.002 and _goal_pos <= 0.002:
            detour_loop_penalty -= 0.03

    # ── 7.6 R_stability ──
    option_switch_penalty = 0.0
    infeasible_action_penalty = 0.0
    if policy_vs_effective_mismatch:
        option_switch_penalty = -0.03
    if not action_was_feasible:
        infeasible_action_penalty = -0.08

    # ── 7.7 R_pair_event ──
    pair_cooperative_reward = 0.0
    pair_competitive_penalty = 0.0
    if pair_partner_id:
        if pair_mode_complementary:
            pair_cooperative_reward = 0.08 * max(0.0, 1.0 - float(pair_dist) / 3.5)
        elif mode == pair_partner_mode and mode in ("go",):
            pair_competitive_penalty = -0.06

    # ── Aggregate ──
    interaction_mode_reward = float(
        + option_completion_bonus
        + safe_turn_reward
        + backoff_release_reward
        + pair_cooperative_reward
    )
    interaction_mode_penalty = float(
        option_failure_penalty
        + option_timeout_penalty
        + wrong_turn_penalty
        + random_turn_penalty
        + spin_without_progress_penalty
        + idle_without_progress_penalty
        + conservative_mode_penalty
        + option_switch_penalty
        + infeasible_action_penalty
        + pair_competitive_penalty
        + obstacle_proximity_penalty
        + detour_loop_penalty
    )

    progress_source_id = {
        "none": 0.0,
        "projection": 1.0,
        "goal": 2.0,
        "local_goal": 3.0,
        "guide_target": 4.0,
    }.get(str(progress_source), 0.0)

    return OptionOutcomeRewardTerms(
        option_progress_reward=float(option_progress_reward),
        option_clearance_reward=float(clear_reward_out),
        option_safety_reward=float(-front_obstacle_risk),
        option_completion_bonus=float(option_completion_bonus),
        option_failure_penalty=float(option_failure_penalty),
        option_timeout_penalty=float(option_timeout_penalty),
        safe_turn_reward=float(safe_turn_reward),
        wrong_turn_penalty=float(wrong_turn_penalty),
        random_turn_penalty=float(random_turn_penalty),
        spin_without_progress_penalty=float(spin_without_progress_penalty),
        idle_without_progress_penalty=float(idle_without_progress_penalty),
        conservative_mode_penalty=float(conservative_mode_penalty),
        option_switch_penalty=float(option_switch_penalty),
        infeasible_action_penalty=float(infeasible_action_penalty),
        pair_cooperative_reward=float(pair_cooperative_reward),
        pair_competitive_penalty=float(pair_competitive_penalty),
        obstacle_proximity_penalty=float(obstacle_proximity_penalty),
        cross_track_penalty=float(cross_track_penalty),
        backoff_release_reward=float(backoff_release_reward),
        detour_loop_penalty=float(detour_loop_penalty),
        positive_path_projection_progress=float(_proj_pos),
        negative_path_projection_progress=float(_proj_neg),
        positive_goal_progress=float(_goal_pos),
        positive_local_goal_progress=float(_local_pos),
        positive_guide_target_progress=float(_guide_pos),
        progress_positive=1.0 if progress_positive else 0.0,
        progress_source_id=float(progress_source_id),
        obstacle_risk_drop=float(obstacle_risk_drop),
        ttc_improvement=float(ttc_improvement),
        front_blocked_ratio_delta=float(front_blocked_ratio_delta),
        left_safety_score=float(left_score),
        right_safety_score=float(right_score),
        ttc_risk=float(ttc_risk),
        risk_gate=float(risk_gate),
        correct_turn=float(correct_turn),
        wrong_turn=float(wrong_turn),
        risk_reduced=1.0 if risk_reduced_effective else 0.0,
        interaction_mode_reward=float(interaction_mode_reward),
        interaction_mode_penalty=float(interaction_mode_penalty),
        social_risk_delta=float(social_risk_delta_out),
        clear_reward=float(clear_reward_out),
        blocked_score=float(blocked_score),
        stuck_score=float(stuck_score),
    )
