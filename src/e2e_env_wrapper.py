from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from gymnasium import spaces
except ModuleNotFoundError:
    from gym import spaces  # type: ignore

from ray.rllib.env.multi_agent_env import MultiAgentEnv

try:
    from intent_marl_training.gnn_marl_env import env_creator as base_env_creator
except ModuleNotFoundError:
    from gnn_marl_env import env_creator as base_env_creator  # type: ignore


class EndToEndMAPPOEnv(MultiAgentEnv):
    """
    Lightweight end-to-end RL wrapper over GNNMARLEnv.

    Goals:
    - keep the original multi-robot environment and observation pipeline;
    - remove all action arbitration / shield / yielding / social-controller logic;
    - keep actions fully under RL control;
    - add only a small amount of *reward shaping* that encourages
      near-range avoidance instead of freezing.

    This wrapper does **not** modify actions. It only forwards actions directly to the
    base env and adjusts rewards using pairwise robot distances after the step.
    """

    def __init__(self, env_config: Dict):
        super().__init__()

        cfg = dict(env_config)
        base_env_config = dict(cfg.get("base_env_config", {}))
        reward_cfg = dict(cfg.get("reward_config", {}))

        # Keep RL in full control even if some caller forgot to disable a helper.
        forced_base_flags = {
            "shield_enable": False,
            "tracking_assist_enable": False,
            "local_executor_enable": False,
            "msa3c_action_mode": False,
            "msa3c_social_feature_enable": False,
            "hybrid_control_enable": False,
            "base_zone_manager_enable": False,
            "auto_reset_agents": False,
            "social_yield_reward_scale": 0.0,
            "social_passage_reward_scale": 0.0,
            "social_clear_reward_scale": 0.0,
        }
        base_env_config.update(forced_base_flags)

        # Preserve RLlib worker metadata for logging.
        for key in ("worker_index", "vector_index", "num_workers", "remote"):
            if key not in base_env_config and key in cfg:
                base_env_config[key] = cfg[key]

        self.base_env = base_env_creator(base_env_config)
        self.agent_ids = list(getattr(self.base_env, "agent_ids", []))
        self._agent_ids = set(self.agent_ids)
        self.possible_agents = list(getattr(self.base_env, "possible_agents", self.agent_ids))
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space

        # Distance-based shaping. These are *training signals*, not control logic.
        self.interaction_dist = float(reward_cfg.get("interaction_dist", 1.25))
        self.neighbor_safe_dist = float(reward_cfg.get("neighbor_safe_dist", 0.72))
        self.neighbor_penalty_scale = float(reward_cfg.get("neighbor_penalty_scale", 1.25))
        self.escape_reward_scale = float(reward_cfg.get("escape_reward_scale", 0.45))
        self.approach_penalty_scale = float(reward_cfg.get("approach_penalty_scale", 0.12))
        self.clearance_reward_scale = float(reward_cfg.get("clearance_reward_scale", 0.10))
        self.max_escape_delta = float(reward_cfg.get("max_escape_delta", 0.15))

        self._prev_neighbor_min_dist: Dict[str, float] = {
            aid: float("inf") for aid in self.agent_ids
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._prev_neighbor_min_dist = self._compute_neighbor_min_distances()
        return obs, info

    def step(self, action_dict: Dict[str, np.ndarray]):
        obs, rewards, dones, truncated, infos = self.base_env.step(action_dict)
        curr_neighbor_min_dist = self._compute_neighbor_min_distances()

        for aid in self.agent_ids:
            if aid not in rewards:
                continue

            info = infos.setdefault(aid, {})
            status = str(info.get("status", ""))
            if status in {"done_waiting", "no_action_received"}:
                self._prev_neighbor_min_dist[aid] = curr_neighbor_min_dist.get(aid, float("inf"))
                continue

            prev_d = float(self._prev_neighbor_min_dist.get(aid, float("inf")))
            curr_d = float(curr_neighbor_min_dist.get(aid, float("inf")))
            shaping = 0.0
            neighbor_penalty = 0.0
            escape_reward = 0.0
            approach_penalty = 0.0
            clearance_bonus = 0.0

            if np.isfinite(curr_d):
                if curr_d < self.neighbor_safe_dist:
                    ratio = (self.neighbor_safe_dist - curr_d) / max(self.neighbor_safe_dist, 1e-6)
                    neighbor_penalty = -self.neighbor_penalty_scale * ratio * ratio

                interaction_active = np.isfinite(prev_d) and min(prev_d, curr_d) < self.interaction_dist
                if interaction_active:
                    delta = float(np.clip(curr_d - prev_d, -self.max_escape_delta, self.max_escape_delta))
                    if delta > 1e-3:
                        escape_reward = self.escape_reward_scale * (delta / max(self.max_escape_delta, 1e-6))
                    elif delta < -1e-3 and curr_d < self.neighbor_safe_dist:
                        approach_penalty = -self.approach_penalty_scale * (
                            abs(delta) / max(self.max_escape_delta, 1e-6)
                        )

                if np.isfinite(prev_d) and prev_d < self.neighbor_safe_dist <= curr_d:
                    clearance_bonus = self.clearance_reward_scale

            shaping = neighbor_penalty + escape_reward + approach_penalty + clearance_bonus
            rewards[aid] = float(rewards[aid] + shaping)

            info.update(
                {
                    "e2e_neighbor_min_dist": curr_d,
                    "e2e_prev_neighbor_min_dist": prev_d,
                    "e2e_neighbor_penalty": float(neighbor_penalty),
                    "e2e_escape_reward": float(escape_reward),
                    "e2e_approach_penalty": float(approach_penalty),
                    "e2e_clearance_bonus": float(clearance_bonus),
                    "e2e_reward_shaping": float(shaping),
                    "e2e_reward_total": float(rewards[aid]),
                }
            )
            self._prev_neighbor_min_dist[aid] = curr_d

        return obs, rewards, dones, truncated, infos

    def _compute_neighbor_min_distances(self) -> Dict[str, float]:
        base_fn = getattr(self.base_env, "_compute_neighbor_min_distances", None)
        if callable(base_fn):
            return {
                aid: float(dist)
                for aid, dist in base_fn().items()
            }

        min_dists = {aid: float("inf") for aid in self.agent_ids}
        if len(self.agent_ids) <= 1:
            return min_dists

        positions = getattr(self.base_env, "robot_positions", {})
        for i, aid in enumerate(self.agent_ids):
            pos_i = np.asarray(positions.get(aid, np.zeros(2, dtype=np.float32)), dtype=np.float32)
            best = float("inf")
            for j, other in enumerate(self.agent_ids):
                if i == j:
                    continue
                pos_j = np.asarray(positions.get(other, np.zeros(2, dtype=np.float32)), dtype=np.float32)
                dist = float(np.linalg.norm(pos_i - pos_j))
                if dist < best:
                    best = dist
            min_dists[aid] = best
        return min_dists

    def close(self):
        close_fn = getattr(self.base_env, "close", None)
        if callable(close_fn):
            close_fn()


# RLlib env creator ---------------------------------------------------------

def e2e_env_creator(env_config):
    cfg = dict(env_config)
    for key in ("worker_index", "vector_index", "num_workers", "remote"):
        if key not in cfg and hasattr(env_config, key):
            cfg[key] = getattr(env_config, key)
    return EndToEndMAPPOEnv(cfg)
