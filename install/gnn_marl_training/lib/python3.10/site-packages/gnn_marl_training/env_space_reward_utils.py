from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from gymnasium import spaces

from gnn_marl_training.interaction_option_definitions import (
    DETOUR_PHASE_INDEX,
    DetourPhase,
    NUM_TRAINING_OPTIONS,
    TRAINING_OPTION_INDEX,
    TRAINING_OPTION_NAMES,
)
from gnn_marl_training.interaction_observation_utils import (
    build_minimal_risk_summary,
    build_temporal_delta_summary,
)


@dataclass(frozen=True)
class PotentialRewardTerms:
    phi_goal_prev: float
    phi_goal_curr: float
    phi_goal_drop: float
    phi_obs_prev: float
    phi_obs_curr: float
    phi_obs_drop: float
    phi_agent_prev: float
    phi_agent_curr: float
    phi_agent_drop: float
    phi_path_prev: float
    phi_path_curr: float
    phi_path_drop: float
    front_obstacle_potential: float
    side_obstacle_potential: float
    corner_obstacle_potential: float
    ttc_risk: float
    r_potential: float
    r_event: float
    r_terminal: float
    time_penalty: float
    spin_without_progress_penalty: float
    reverse_without_risk_penalty: float
    stuck_long_penalty: float
    detour_active_penalty: float
    detour_success_bonus: float
    corner_clear_bonus: float
    no_progress: float
    stuck_long: float
    progress_positive: float




@dataclass(frozen=True)
class ClassicNavigationRewardTerms:
    progress_reward: float
    heading_reward: float
    obstacle_penalty: float
    predictive_penalty: float
    time_penalty: float
    terminal_reward: float
    total_reward: float


def compute_classic_navigation_reward(
    *,
    path_projection_progress_delta: float,
    target_angle: float,
    forward_speed: float,
    front_potential_penalty: float,
    side_wall_penalty: float,
    close_obstacle_penalty: float,
    predictive_social_penalty: float,
    predictive_front_penalty: float,
    time_penalty: float,
    goal_reached: bool,
    collision: bool,
    timeout: bool,
    goal_reward: float,
    collision_penalty: float,
    timeout_penalty: float = 0.0,
    progress_weight: float = 1.0,
    heading_weight: float = 0.08,
    obstacle_weight: float = 1.35,
    predictive_weight: float = 1.50,
) -> ClassicNavigationRewardTerms:
    clipped_progress = float(np.clip(float(path_projection_progress_delta), -0.10, 0.10))
    progress_reward = float(progress_weight * clipped_progress)

    heading_reward = 0.0
    if float(forward_speed) > 0.02 and abs(float(target_angle)) <= math.pi / 2.0:
        heading_reward = float(heading_weight * math.cos(float(target_angle)))

    obstacle_penalty = float(obstacle_weight * (
        front_potential_penalty + side_wall_penalty + close_obstacle_penalty
    ))
    predictive_penalty = float(predictive_weight * (predictive_social_penalty + predictive_front_penalty))
    time_penalty_term = -float(time_penalty)

    terminal_reward = 0.0
    if goal_reached:
        terminal_reward += float(goal_reward)
    if collision:
        terminal_reward -= float(collision_penalty)
    if timeout:
        terminal_reward -= float(timeout_penalty)

    total_reward = float(
        progress_reward
        + heading_reward
        + obstacle_penalty
        + predictive_penalty
        + time_penalty_term
        + terminal_reward
    )
    return ClassicNavigationRewardTerms(
        progress_reward=float(progress_reward),
        heading_reward=float(heading_reward),
        obstacle_penalty=float(obstacle_penalty),
        predictive_penalty=float(predictive_penalty),
        time_penalty=float(time_penalty_term),
        terminal_reward=float(terminal_reward),
        total_reward=float(total_reward),
    )

@dataclass(frozen=True)
class PairRewardSummary:
    rewards: Dict[str, float]
    metrics: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class RewardAggregationOverrides:
    nav_progress_weight: float = 1.05
    nav_path_weight: float = 0.95
    nav_goal_weight: float = 0.80
    nav_option_weight: float = 1.00
    interaction_base_weight: float = 1.00
    safety_ttc_weight: float = 0.35
    mode_reward_weight: float = 1.00
    mode_penalty_weight: float = 1.00
    efficiency_weight: float = 1.00
    policy_penalty_weight: float = 1.00
    baseline_progress_weight: float = 0.70
    baseline_path_weight: float = 0.55
    baseline_goal_weight: float = 0.30
    baseline_subgoal_weight: float = 0.75
    baseline_heading_weight: float = 0.55
    baseline_lateral_weight: float = 0.40
    baseline_keep_right_weight: float = 0.25
    baseline_wrong_direction_weight: float = 0.35
    baseline_turn_alignment_weight: float = 0.35
    baseline_turn_escape_weight: float = 1.10
    baseline_corner_escape_weight: float = 1.25
    baseline_ttc_weight: float = 1.10
    baseline_obstacle_weight: float = 1.15
    baseline_close_obstacle_weight: float = 1.00
    baseline_predictive_weight: float = 1.25
    baseline_yield_weight: float = 1.05
    baseline_interaction_reward_weight: float = 1.10
    baseline_interaction_penalty_weight: float = 1.00
    baseline_risk_forward_weight: float = 1.20
    baseline_safe_turn_weight: float = 1.45
    baseline_head_on_weight: float = 1.70
    interaction_penalty_clip: float = 0.0
    interaction_reward_clip: float = 0.0
    suppress_conflicting_interaction_shaping: bool = False


@dataclass(frozen=True)
class PotentialRewardConfig:
    goal_drop_weight: float = 0.75
    obs_drop_weight: float = 0.55
    agent_drop_weight: float = 0.60
    path_drop_weight: float = 0.40
    spin_penalty_scale: float = 1.00
    reverse_penalty_scale: float = 1.00
    stuck_penalty_scale: float = 1.00
    event_reward_scale: float = 1.00
    terminal_reward_scale: float = 1.00
    time_penalty_scale: float = 1.00
    detour_bonus_scale: float = 1.00
    detour_active_penalty_scale: float = 1.00
    corner_bonus_scale: float = 1.00
    use_path_potential: bool = True


def configure_multi_agent_observation_space(env: Any) -> None:
    """Define the wrapper observation space from the per-agent base observation."""
    base_obs_dim = env.agents["agent_0"].obs_dim

    if env.enable_neighbor_obs:
        max_neighbors = min(env._num_agents - 1, 5)
        neighbor_dim = max_neighbors * 5
    else:
        neighbor_dim = 0

    local_map_dim = 128 if env.enable_local_map else 0
    env.reset_flag_dim = 1
    env.global_state_dim = env._num_agents * base_obs_dim

    total_dim = (
        base_obs_dim
        + neighbor_dim
        + local_map_dim
        + env.reset_flag_dim
        + env.global_state_dim
    )

    env.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(total_dim,),
        dtype=np.float32,
    )
    env.base_obs_dim = base_obs_dim
    env.neighbor_dim = neighbor_dim
    env.local_map_dim = local_map_dim


def configure_independent_env_action_observation_spaces(
    env: Any,
    *,
    obstacle_top_k: int,
    obstacle_filter_range: float,
    obstacle_filter_fov_deg: float,
    predictive_feature_enable: bool,
    predictive_horizon_sec: float,
    predictive_social_ttc_safe: float,
    predictive_front_ttc_safe: float,
    predictive_min_sep: float,
    predictive_social_range: float,
    interaction_neighbor_perception_range: float,
    communication_range: float,
    predictive_social_penalty_scale: float,
    predictive_front_penalty_scale: float,
    social_proximity_risk_scale: float,
    gap_feature_enable: bool,
    neighbor_prediction_top_k: int,
    obstacle_motion_feature_enable: bool,
    obstacle_motion_top_k: int,
    angular_bins: int = 0,
) -> None:
    env.learned_interaction_modes = tuple(TRAINING_OPTION_NAMES)
    env.learned_interaction_mode_to_id = dict(TRAINING_OPTION_INDEX)
    env.action_mode = "interaction_mode"

    env.scan_max_range = 3.5
    env.scan_valid_min = 0.10
    env.obstacle_point_feature_dim = 4
    env.obstacle_top_k = int(np.clip(int(obstacle_top_k), 1, 64))
    env.obstacle_filter_range = float(
        np.clip(float(obstacle_filter_range), 0.2, env.scan_max_range)
    )
    env.obstacle_filter_fov_deg = float(
        np.clip(float(obstacle_filter_fov_deg), 10.0, 360.0)
    )
    env.angular_bins = int(max(0, int(angular_bins)))
    if env.angular_bins < 8:
        env.angular_bins = env.obstacle_top_k
    env.scan_dim = env.angular_bins

    env.interaction_ego_state_dim = 8
    env.base_safety_feature_dim = 8
    env.temporal_delta_dim = 4
    env.option_state_dim = 0
    env.action_mask_dim = 0
    env.tracking_target_dim = 0

    env.predictive_feature_enable = bool(predictive_feature_enable)
    env.predictive_feature_dim = 6 if env.predictive_feature_enable else 0
    env.predictive_horizon_sec = max(0.2, float(predictive_horizon_sec))
    env.predictive_social_ttc_safe = max(0.2, float(predictive_social_ttc_safe))
    env.predictive_front_ttc_safe = max(0.2, float(predictive_front_ttc_safe))
    env.predictive_min_sep = max(0.15, float(predictive_min_sep))
    env.predictive_social_range = max(
        env.predictive_min_sep,
        float(predictive_social_range),
    )

    perception_range_cfg = float(interaction_neighbor_perception_range)
    if perception_range_cfg > 0.0:
        env.interaction_neighbor_perception_range = max(0.5, perception_range_cfg)
    else:
        env.interaction_neighbor_perception_range = max(
            env.scan_max_range,
            env.predictive_social_range,
            float(communication_range),
        )

    env.predictive_social_penalty_scale = max(
        0.0,
        float(predictive_social_penalty_scale),
    )
    env.predictive_front_penalty_scale = max(
        0.0,
        float(predictive_front_penalty_scale),
    )
    env.social_proximity_risk_scale = max(
        0.0,
        float(social_proximity_risk_scale),
    )
    env.gap_feature_enable = False
    env.gap_feature_dim = 0
    env.neighbor_prediction_top_k = 0
    env.neighbor_prediction_feature_dim = 6
    env.neighbor_prediction_dim = 0

    env.obstacle_motion_feature_enable = False
    env.obstacle_motion_top_k = 0
    env.obstacle_motion_feature_dim = 6
    env.obstacle_motion_dim = 0

    env.safety_feature_dim = env.base_safety_feature_dim
    env.obs_dim = (
        env.scan_dim * env.scan_history_len
        + 2
        + 2
        + env.safety_feature_dim
        + env.temporal_delta_dim
    )

    env.observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(env.obs_dim,),
        dtype=np.float32,
    )
    env.action_space = spaces.Discrete(len(env.learned_interaction_modes))


def build_option_state_features(env: Any) -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def build_action_mask_features(env: Any) -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def build_tracking_target_features(env: Any) -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def build_independent_env_observation(
    env: Any,
    target_override: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    ranges = np.array(
        env.latest_scan.ranges if env.latest_scan else [env.scan_max_range] * 360,
        dtype=np.float32,
    )
    ranges = np.nan_to_num(
        ranges,
        nan=env.scan_max_range,
        posinf=env.scan_max_range,
        neginf=0.0,
    )
    sector_dists = env._compute_front_sector_min_dists(ranges)
    scan_obs = env._extract_filtered_scan_features(ranges)

    env._scan_history.append(scan_obs.copy())
    env._front_sector_dist_history.append(sector_dists.copy())
    history = list(env._scan_history)
    while len(history) < env.scan_history_len:
        history.insert(0, history[0].copy())
    stacked_scan = np.concatenate(history)

    if target_override is not None:
        raw_target = target_override
    else:
        nominal_info = env._compute_nominal_tracking_info()
        raw_target = tuple(nominal_info["subgoal"])

    raw_target_xy = np.array(raw_target, dtype=np.float32)
    if env._obs_target_state is None:
        env._obs_target_state = raw_target_xy.copy()
    else:
        prev_xy = env._obs_target_state
        delta = raw_target_xy - prev_xy
        step = float(np.linalg.norm(delta))
        if step > env.obs_target_max_step:
            delta = delta / (step + 1e-8) * env.obs_target_max_step
        candidate = prev_xy + delta
        alpha = env.obs_target_filter_alpha
        env._obs_target_state = (1.0 - alpha) * prev_xy + alpha * candidate

    obs_target = env._obs_target_state
    tgt_dist = math.hypot(
        float(obs_target[0]) - env.current_pose["x"],
        float(obs_target[1]) - env.current_pose["y"],
    )
    tgt_angle = math.atan2(
        float(obs_target[1]) - env.current_pose["y"],
        float(obs_target[0]) - env.current_pose["x"],
    )
    rel_angle = (tgt_angle - env.current_pose["yaw"] + np.pi) % (2 * np.pi) - np.pi
    dist_norm = float(np.clip(tgt_dist / env.obs_target_dist_clip, 0.0, 1.0))
    target_x_body = dist_norm * math.cos(rel_angle)
    target_y_body = dist_norm * math.sin(rel_angle)

    sectors = env._scan_sector_metrics()
    front_min = float(sectors["front_min"])
    left_min = float(sectors.get("left_min", front_min))
    right_min = float(sectors.get("right_min", front_min))
    predictive_features = env._get_predictive_obs_features(front_min)
    predictive_social_risk = float(env._last_predictive_metrics.get('social_risk', 0.0))
    predictive_front_risk = float(env._last_predictive_metrics.get('front_risk', 0.0))
    interaction_ctx = env._get_interaction_context()
    ttc_min = float(interaction_ctx.get('ttc', float('inf')))
    front_close_ratio = float(np.clip(
        (env.close_obstacle_dist - front_min) / max(env.close_obstacle_dist, 1e-6),
        0.0,
        1.0,
    ))
    side_min = float(min(left_min, right_min))
    side_close_ratio = float(np.clip(
        (env.side_close_dist - side_min) / max(env.side_close_dist, 1e-6),
        0.0,
        1.0,
    ))
    risk_summary = build_minimal_risk_summary(
        front_min=front_min,
        left_min=left_min,
        right_min=right_min,
        front_close_ratio=front_close_ratio,
        side_close_ratio=side_close_ratio,
        predictive_social_risk=predictive_social_risk,
        predictive_front_risk=predictive_front_risk,
        ttc_min=ttc_min,
        predictive_social_ttc_safe=float(env.predictive_social_ttc_safe),
    )

    prev_front = float(getattr(env, '_prev_obs_front_min', front_min))
    prev_ttc = float(getattr(env, '_prev_obs_ttc_min', ttc_min))
    prev_social = float(getattr(env, '_prev_obs_social_risk', predictive_social_risk))
    delta_front_min = float(front_min - prev_front)
    delta_ttc = 0.0 if not (math.isfinite(prev_ttc) and math.isfinite(ttc_min)) else float(ttc_min - prev_ttc)
    delta_social_risk = float(predictive_social_risk - prev_social)
    delta_path_progress = float(getattr(env, '_path_projection_progress_delta', 0.0))
    temporal_delta = build_temporal_delta_summary(
        delta_front_min=delta_front_min,
        delta_ttc=delta_ttc,
        delta_social_risk=delta_social_risk,
        delta_path_progress=delta_path_progress,
    )
    env._prev_obs_front_min = float(front_min)
    env._prev_obs_ttc_min = float(ttc_min if math.isfinite(ttc_min) else prev_ttc)
    env._prev_obs_social_risk = float(predictive_social_risk)

    obs = np.concatenate(
        [
            stacked_scan,
            [target_x_body, target_y_body],
            [env.current_vel_x, env.current_vel_w],
            risk_summary,
            temporal_delta,
        ]
    ).astype(np.float32)

    if obs.shape != env.observation_space.shape:
        raise ValueError(
            f"[IndependentRobotEnv] _get_obs shape mismatch: got={obs.shape}, "
            f"expected={env.observation_space.shape}, scan_dim={env.scan_dim}, "
            f"scan_history_len={env.scan_history_len}, obstacle_top_k={env.obstacle_top_k}"
        )
    return obs


def _smooth_repulsive_potential(distance: float, sigma: float, amplitude: float = 3.0) -> float:
    """Smooth exponential repulsive potential — continuous everywhere, no hard cutoffs.

    V(d) = A · exp(−d / σ)

    At d=0:        V = A           (maximum repulsion)
    At d=σ:        V ≈ 0.37·A      (moderate)
    At d=3σ:       V ≈ 0.05·A      (nearly zero)
    """
    if not math.isfinite(float(distance)):
        return float(amplitude)
    d = max(float(distance), 0.02)
    s = max(float(sigma), 1e-6)
    return float(amplitude) * math.exp(-d / s)


def compute_interaction_potential_reward(
    env: Any,
    *,
    dist_to_target: float,
    dist_to_goal: float,
    front_min: float,
    left_min: float,
    right_min: float,
    front_left_min: float,
    front_right_min: float,
    social_risk_max: float,
    ttc_min: float,
    front_risk: float,
    cross_track_error: float,
    effective_mode: str,
    applied_linear_vel: float,
    applied_angular_vel: float,
    local_goal_progress_delta: float,
    goal_progress_delta: float,
    path_projection_progress_delta: float,
    stuck_score: float,
    detour_active: bool,
    detour_done: bool,
    head_on_pass_event: bool,
    collision: bool,
    timeout: bool,
    goal_reached: bool,
    config: PotentialRewardConfig | None = None,
) -> PotentialRewardTerms:
    config = config or PotentialRewardConfig()
    corridor_span = float(left_min + right_min)
    narrow_span_ref = max(
        0.45,
        2.4 * float(getattr(env, "subgoal_min_side_clearance", 0.20)),
    )
    corridor_narrow_ratio = float(np.clip(
        (narrow_span_ref - corridor_span) / max(narrow_span_ref, 1e-6),
        0.0,
        1.0,
    ))

    sigma_front = 0.22
    sigma_side = 0.16
    sigma_corner = 0.18
    front_obstacle_potential = _smooth_repulsive_potential(front_min, sigma_front, amplitude=2.8)
    side_obstacle_potential = _smooth_repulsive_potential(
        min(left_min, right_min),
        sigma_side,
        amplitude=1.8,
    )
    corner_obstacle_potential = _smooth_repulsive_potential(
        min(front_left_min, front_right_min),
        sigma_corner,
        amplitude=2.2,
    )
    phi_obs_curr = float(
        0.50 * front_obstacle_potential
        + 0.20 * side_obstacle_potential
        + 0.30 * corner_obstacle_potential
    )

    ttc_risk = 0.0
    if math.isfinite(float(ttc_min)):
        ttc_risk = float(np.clip((2.0 - float(ttc_min)) / 2.0, 0.0, 1.0))
    phi_agent_curr = float(
        np.clip(
            max(float(social_risk_max), float(ttc_risk), float(front_risk)),
            0.0,
            1.0,
        )
    )

    phi_goal_curr = float(dist_to_target)
    phi_path_curr = 0.0

    phi_goal_prev = float(
        getattr(env, "_prev_phi_goal", phi_goal_curr)
        if getattr(env, "_prev_phi_goal", None) is not None else phi_goal_curr
    )
    phi_obs_prev = float(
        getattr(env, "_prev_phi_obs", phi_obs_curr)
        if getattr(env, "_prev_phi_obs", None) is not None else phi_obs_curr
    )
    phi_agent_prev = float(
        getattr(env, "_prev_phi_agent", phi_agent_curr)
        if getattr(env, "_prev_phi_agent", None) is not None else phi_agent_curr
    )
    phi_path_prev = float(
        getattr(env, "_prev_phi_path", phi_path_curr)
        if getattr(env, "_prev_phi_path", None) is not None else phi_path_curr
    )

    phi_total_prev = (
        config.obs_drop_weight * phi_obs_prev
        + config.agent_drop_weight * phi_agent_prev
        + config.path_drop_weight * phi_path_prev
    )
    phi_total_curr = (
        config.obs_drop_weight * phi_obs_curr
        + config.agent_drop_weight * phi_agent_curr
        + config.path_drop_weight * phi_path_curr
    )
    raw_potential_delta = float(phi_total_prev - phi_total_curr)
    r_potential = config.goal_drop_weight * float(np.clip(raw_potential_delta, -0.20, 0.20))

    progress_delta = max(
        float(path_projection_progress_delta),
        float(goal_progress_delta),
        float(local_goal_progress_delta),
    )
    r_progress = float(np.clip(progress_delta, -0.05, 0.05))

    path_progress_small = abs(float(path_projection_progress_delta)) < 0.006
    goal_progress_small = abs(float(goal_progress_delta)) < 0.006
    local_progress_small = abs(float(local_goal_progress_delta)) < 0.006
    no_progress = bool(path_progress_small and goal_progress_small and local_progress_small)
    progress_positive = 0.0 if no_progress else 1.0

    gap_angle = float(getattr(env, "_last_gap_metrics", {}).get("best_gap_angle", 0.0))
    risk_signal = float(max(phi_obs_curr, phi_agent_curr))
    forward_active = bool(float(applied_linear_vel) > 0.03)
    turn_rate_mag = float(abs(applied_angular_vel))
    turn_direction = 1.0 if float(applied_angular_vel) > 0.0 else (-1.0 if float(applied_angular_vel) < 0.0 else 0.0)
    gap_direction = 1.0 if gap_angle > 0.08 else (-1.0 if gap_angle < -0.08 else 0.0)
    turn_gate = bool(
        risk_signal > 0.18
        and forward_active
        and float(progress_delta) > 0.0
        and gap_direction != 0.0
        and turn_rate_mag > 0.05
    )
    r_turn = 0.0
    if turn_gate:
        turn_align = 1.0 if turn_direction == gap_direction else -1.0
        r_turn = 0.08 * turn_align * min(
            1.0,
            turn_rate_mag / max(float(getattr(env, "max_angular_vel", 1.0)), 1e-6),
        )

    time_penalty = -0.003 * float(config.time_penalty_scale)
    corner_escape_active = bool(str(getattr(env, "_last_subgoal_mode", "")) == "corner_escape")
    subgoal_deadlock_streak = float(getattr(env, "_subgoal_deadlock_streak", 0.0))
    deadlock_escape_active = corner_escape_active or float(stuck_score) > 0.60 or subgoal_deadlock_streak >= 3.0
    escape_relax = 0.40 if deadlock_escape_active else 1.0
    spin_without_progress_penalty = 0.0
    if turn_rate_mag > 0.20 and abs(float(applied_linear_vel)) < 0.05 and no_progress:
        spin_without_progress_penalty = -0.08 * float(config.spin_penalty_scale) * escape_relax
        if risk_signal < 0.18:
            spin_without_progress_penalty -= 0.02 * float(config.spin_penalty_scale) * escape_relax

    reverse_without_risk_penalty = 0.0
    if float(applied_linear_vel) < -0.03 and risk_signal < 0.25 and no_progress and not deadlock_escape_active:
        reverse_without_risk_penalty = -0.08 * float(config.reverse_penalty_scale)

    stuck_long = bool(float(stuck_score) > 0.65 and no_progress)
    stuck_long_penalty = ((-0.08 * float(config.stuck_penalty_scale)) * (0.35 if deadlock_escape_active else 1.0)) if stuck_long else 0.0

    detour_success_bonus = 0.0
    detour_active_penalty = 0.0
    corner_clear_bonus = 0.0
    phi_agent_drop = float(phi_agent_prev - phi_agent_curr)
    phi_obs_drop = float(phi_obs_prev - phi_obs_curr)
    detour_useful = bool(
        progress_delta > 0.006
        or phi_agent_drop > 0.03
        or phi_obs_drop > 0.03
        or risk_signal > 0.35
        or float(stuck_score) > 0.55
    )

    if detour_active and no_progress and risk_signal < 0.25:
        detour_active_penalty = -0.05 * float(config.detour_active_penalty_scale)
    if detour_done and detour_useful:
        detour_success_bonus = 0.04 * float(config.detour_bonus_scale)
    if phi_obs_drop > 0.03 and risk_signal < 0.20 and float(cross_track_error) < 0.35:
        corner_clear_bonus = 0.06 * float(config.corner_bonus_scale)

    r_event = float(
        config.event_reward_scale
        * (
            r_progress
            + r_turn
            + time_penalty
            + spin_without_progress_penalty
            + reverse_without_risk_penalty
            + stuck_long_penalty
            + detour_active_penalty
            + detour_success_bonus
            + corner_clear_bonus
        )
    )

    r_terminal = 0.0
    if goal_reached:
        r_terminal += 8.0
    if collision:
        r_terminal -= 10.0
    if timeout:
        r_terminal -= 0.5
    r_terminal *= float(config.terminal_reward_scale)

    env._prev_phi_goal = float(phi_goal_curr)
    env._prev_phi_obs = float(phi_obs_curr)
    env._prev_phi_agent = float(phi_agent_curr)
    env._prev_phi_path = float(phi_path_curr)
    env._prev_corner_obstacle_potential = float(corner_obstacle_potential)

    return PotentialRewardTerms(
        phi_goal_prev=float(phi_goal_prev),
        phi_goal_curr=float(phi_goal_curr),
        phi_goal_drop=float(phi_goal_prev - phi_goal_curr),
        phi_obs_prev=float(phi_obs_prev),
        phi_obs_curr=float(phi_obs_curr),
        phi_obs_drop=float(phi_obs_prev - phi_obs_curr),
        phi_agent_prev=float(phi_agent_prev),
        phi_agent_curr=float(phi_agent_curr),
        phi_agent_drop=float(phi_agent_prev - phi_agent_curr),
        phi_path_prev=float(phi_path_prev),
        phi_path_curr=float(phi_path_curr),
        phi_path_drop=float(phi_path_prev - phi_path_curr),
        front_obstacle_potential=float(front_obstacle_potential),
        side_obstacle_potential=float(side_obstacle_potential),
        corner_obstacle_potential=float(corner_obstacle_potential),
        ttc_risk=float(ttc_risk),
        r_potential=float(r_potential),
        r_event=float(r_event),
        r_terminal=float(r_terminal),
        time_penalty=float(time_penalty),
        spin_without_progress_penalty=float(spin_without_progress_penalty),
        reverse_without_risk_penalty=float(reverse_without_risk_penalty),
        stuck_long_penalty=float(stuck_long_penalty),
        detour_active_penalty=float(detour_active_penalty),
        detour_success_bonus=float(detour_success_bonus),
        corner_clear_bonus=float(corner_clear_bonus),
        no_progress=1.0 if no_progress else 0.0,
        stuck_long=1.0 if stuck_long else 0.0,
        progress_positive=float(progress_positive),
    )


def compute_pairwise_local_rewards(env: Any, info_dict: Dict[str, Dict[str, Any]]) -> PairRewardSummary:
    rewards: Dict[str, float] = {aid: 0.0 for aid in info_dict.keys()}
    metrics: Dict[str, Dict[str, float]] = {
        aid: {
            "pair_collision_penalty": 0.0,
            "pair_near_miss_penalty": 0.0,
            "local_head_on_pass_event": 0.0,
            "mutual_yield_penalty": 0.0,
            "yield_pass_credit": 0.0,
            "pair_event_reward": 0.0,
        }
        for aid in info_dict.keys()
    }

    pair_memory = getattr(env, "_pair_event_memory", {})
    new_memory: Dict[tuple[str, str], Dict[str, float]] = {}
    near_miss_dist = 0.24
    collision_sync_dist = 0.20

    active_aids = [aid for aid in env.agent_ids if aid in info_dict]
    for i in range(len(active_aids)):
        for j in range(i + 1, len(active_aids)):
            ai = active_aids[i]
            aj = active_aids[j]
            pos_i = np.asarray(env.robot_positions.get(ai, np.zeros(2)), dtype=np.float32)
            pos_j = np.asarray(env.robot_positions.get(aj, np.zeros(2)), dtype=np.float32)
            dist = float(np.linalg.norm(pos_j - pos_i))
            pair = tuple(sorted((ai, aj)))
            prev = pair_memory.get(pair, {})
            info_i = info_dict[ai]
            info_j = info_dict[aj]
            ctx_i = env.get_agent_interaction_context(ai)
            ctx_j = env.get_agent_interaction_context(aj)
            active_pair = (
                str(ctx_i.get("partner", "")) == aj
                or str(ctx_j.get("partner", "")) == ai
            )
            risk_i = float(info_i.get("social_risk", 0.0))
            risk_j = float(info_j.get("social_risk", 0.0))
            low_speed_i = abs(float(getattr(env.agents.get(ai), "current_vel_x", 0.0))) < 0.04
            low_speed_j = abs(float(getattr(env.agents.get(aj), "current_vel_x", 0.0))) < 0.04
            progress_i = max(
                float(info_i.get("goal_progress_delta", 0.0)),
                float(info_i.get("local_goal_progress_delta", 0.0)),
                float(info_i.get("path_projection_progress_delta", 0.0)),
            )
            progress_j = max(
                float(info_j.get("goal_progress_delta", 0.0)),
                float(info_j.get("local_goal_progress_delta", 0.0)),
                float(info_j.get("path_projection_progress_delta", 0.0)),
            )

            collision_i = (
                info_i.get("synced_collision_with") == aj
                or (
                    info_i.get("event") == "collision"
                    and dist <= collision_sync_dist
                )
            )
            collision_j = (
                info_j.get("synced_collision_with") == ai
                or (
                    info_j.get("event") == "collision"
                    and dist <= collision_sync_dist
                )
            )
            if collision_i or collision_j:
                rewards[ai] -= 4.0
                rewards[aj] -= 4.0
                metrics[ai]["pair_collision_penalty"] -= 4.0
                metrics[aj]["pair_collision_penalty"] -= 4.0
            elif active_pair and dist < near_miss_dist:
                rewards[ai] -= 0.15
                rewards[aj] -= 0.15
                metrics[ai]["pair_near_miss_penalty"] -= 0.15
                metrics[aj]["pair_near_miss_penalty"] -= 0.15

            both_waiting = (
                active_pair
                and low_speed_i
                and low_speed_j
                and max(risk_i, risk_j) > 0.20
                and progress_i < 0.003
                and progress_j < 0.003
            )
            if both_waiting:
                rewards[ai] -= 0.10
                rewards[aj] -= 0.10
                metrics[ai]["mutual_yield_penalty"] -= 0.10
                metrics[aj]["mutual_yield_penalty"] -= 0.10

            pass_event = False
            if prev.get("active", 0.0) > 0.5 and not active_pair and dist > float(prev.get("dist", dist)):
                if max(risk_i, risk_j) < 0.18:
                    pass_event = True
            if pass_event:
                rewards[ai] += 0.20
                rewards[aj] += 0.20
                metrics[ai]["local_head_on_pass_event"] = 1.0
                metrics[aj]["local_head_on_pass_event"] = 1.0

            if active_pair:
                if float(ctx_i.get("should_yield", 0.0)) > 0.5 and low_speed_i and progress_j > 0.01:
                    rewards[ai] += 0.10
                    rewards[aj] += 0.10
                    metrics[ai]["yield_pass_credit"] += 0.10
                    metrics[aj]["yield_pass_credit"] += 0.10
                elif float(ctx_j.get("should_yield", 0.0)) > 0.5 and low_speed_j and progress_i > 0.01:
                    rewards[ai] += 0.10
                    rewards[aj] += 0.10
                    metrics[ai]["yield_pass_credit"] += 0.10
                    metrics[aj]["yield_pass_credit"] += 0.10

            new_memory[pair] = {
                "active": 1.0 if active_pair else 0.0,
                "dist": float(dist),
            }

    env._pair_event_memory = new_memory
    for aid in metrics:
        metrics[aid]["pair_event_reward"] = float(rewards.get(aid, 0.0))
    return PairRewardSummary(rewards=rewards, metrics=metrics)
