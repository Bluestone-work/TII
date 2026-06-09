"""Feasibility-aware closed-loop option policy — shared definitions.

This module exists to avoid circular imports between gnn_marl_env.py,
option_feasibility.py, option_primitives.py, and interaction_reward_utils.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ── 6-action space (replan disabled in main training, slot reserved but unused) ──
TRAINING_OPTION_NAMES: Tuple[str, ...] = (
    "go",            # 0
    "wait",          # 1
    "backoff",       # 2
    "detour_left",   # 3
    "detour_right",  # 4
    "slow_follow",   # 5
)
NUM_TRAINING_OPTIONS: int = len(TRAINING_OPTION_NAMES)

TRAINING_OPTION_INDEX: Dict[str, int] = {
    name: idx for idx, name in enumerate(TRAINING_OPTION_NAMES)
}

# Mapping from 6-action names → 7-option names used by option_feasibility.py
TRAINING_TO_FEASIBILITY_OPTION: Dict[str, str] = {
    "go":             "follow_path",
    "wait":           "stop_wait",
    "backoff":        "backoff",
    "detour_left":    "detour_left",
    "detour_right":   "detour_right",
    "slow_follow":    "slow_follow",
}

# Canonical execution mode for each training option (used by tracking controller)
CANONICAL_MODE_BY_TRAINING_OPTION: Dict[str, str] = {
    "go":             "go",
    "wait":           "wait",
    "backoff":        "backoff",
    "detour_left":    "detour",
    "detour_right":   "detour",
    "slow_follow":    "go",       # slow_follow uses "go" tracking but with reduced speed
}

# ── Detour phases ──
class DetourPhase:
    ENTER = "enter"
    PASS = "pass"
    MERGE = "merge"
    DONE = "done"

    _ALL = frozenset({ENTER, PASS, MERGE, DONE})


DETOUR_PHASE_INDEX: Dict[str, int] = {
    DetourPhase.ENTER: 0,
    DetourPhase.PASS: 1,
    DetourPhase.MERGE: 2,
}


# ── Option outcome ──
@dataclass
class OptionOutcome:
    option_name: str = ""
    steps_executed: int = 0
    success: bool = False
    failed: bool = False
    failure_reason: str = ""
    progress_gain: float = 0.0
    goal_distance_drop: float = 0.0
    front_clearance_gain: float = 0.0
    social_risk_drop: float = 0.0
    ttc_gain: float = 0.0
    lateral_displacement: float = 0.0
    backward_distance: float = 0.0
    near_miss_count: int = 0
    safety_override_count: int = 0
    emergency_override_count: int = 0
    wall_scrape_count: int = 0
    exit_feasible: bool = True


# ── Collision responsibility ──
@dataclass
class CollisionAttribution:
    self_at_fault: float = 0.5
    partner_at_fault: float = 0.5
    self_speed_toward_partner: float = 0.0
    partner_speed_toward_self: float = 0.0
    self_deviation_from_path: float = 0.0
    partner_deviation_from_path: float = 0.0
    self_had_token: bool = False
    partner_had_token: bool = False
    self_mode_at_impact: str = "go"
    partner_mode_at_impact: str = "go"


# ── Safe-turn risk gate ──
SAFE_TURN_RISK_GATE = 0.30
SAFE_TURN_RISK_LOW = 0.15
SAFE_TURN_CLEAR_REF = 0.80       # reference distance for clearance score normalization
SAFE_TURN_DIRECTION_MARGIN = 0.05  # minimum difference for "turning toward safer side"

# Detour phase transition thresholds
DETOUR_ENTER_MIN_STEPS = 3
DETOUR_PASS_MIN_STEPS = 4
DETOUR_LATERAL_DISPLACEMENT_THRESH = 0.08   # m, enter→pass
DETOUR_FRONT_CLEAR_THRESH = 0.40            # m, pass→merge
DETOUR_FRONT_RISK_THRESH = 0.35             # pass→merge
