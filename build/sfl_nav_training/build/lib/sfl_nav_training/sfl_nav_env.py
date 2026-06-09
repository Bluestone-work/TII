from __future__ import annotations

import math
from typing import Dict

import gymnasium as gym
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv

try:
    from intent_marl_training_e2e.gnn_marl_env import env_creator as base_env_creator
except ModuleNotFoundError:
    from gnn_marl_env import env_creator as base_env_creator  # type: ignore


class SamplingStyleNavEnv(MultiAgentEnv):
    """
    Gazebo-backed multi-agent env wrapper with sampling-for-learnability style I/O:
    - action space: continuous 2D velocity command (preserved from base env)
    - observation: lidar_num_beams + 5
    - reward: rew_lambda * sparse + (1-rew_lambda) * dense

    Path/guidance visualization is preserved by enabling base env visualization.
    """

    def __init__(self, env_config: Dict):
        super().__init__()

        cfg = dict(env_config)
        base_env_config = dict(cfg.get("base_env_config", {}))
        reward_cfg = dict(cfg.get("reward_config", {}))
        obs_cfg = dict(cfg.get("obs_config", {}))

        # Keep direct RL control, and keep RViz guidance/path visualization available.
        base_env_config.update(
            {
                "end_to_end_rl": True,
                "auto_reset_agents": False,
                "enable_visualization": bool(base_env_config.get("enable_visualization", True)),
                "tracking_viz_interval": int(base_env_config.get("tracking_viz_interval", 2)),
            }
        )

        for key in ("worker_index", "vector_index", "num_workers", "remote"):
            if key not in base_env_config and key in cfg:
                base_env_config[key] = cfg[key]

        self.base_env = base_env_creator(base_env_config)
        self.agent_ids = list(getattr(self.base_env, "agent_ids", []))
        self._agent_ids = set(self.agent_ids)
        self.possible_agents = list(getattr(self.base_env, "possible_agents", self.agent_ids))

        self.action_space = self.base_env.action_space

        self.lidar_num_beams = int(obs_cfg.get("lidar_num_beams", 200))
        self.lidar_max_range = float(obs_cfg.get("lidar_max_range", 6.0))
        self.lidar_min_range = float(obs_cfg.get("lidar_min_range", 0.0))
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.lidar_num_beams + 5,),
            dtype=np.float32,
        )

        # sampling-for-learnability / JaxNav-style reward params.
        self.rew_lambda = float(reward_cfg.get("rew_lambda", 0.5))
        self.goal_rew = float(reward_cfg.get("goal_rew", 4.0))
        self.dt_rew = float(reward_cfg.get("dt_rew", -0.01))
        self.coll_rew = float(reward_cfg.get("coll_rew", -4.0))
        self.lidar_thresh = float(reward_cfg.get("lidar_thresh", 0.1))
        self.lidar_rew = float(reward_cfg.get("lidar_rew", -0.1))
        self.agent_collision_dist = float(reward_cfg.get("agent_collision_dist", 0.6))

    def reset(self, *, seed=None, options=None):
        _, info = self.base_env.reset(seed=seed, options=options)
        obs = {aid: self._build_agent_obs(aid) for aid in self.agent_ids}
        for aid in self.agent_ids:
            info.setdefault(aid, {})
            info[aid].update(
                {
                    "GoalR": 0.0,
                    "MapC": 0.0,
                    "AgentC": 0.0,
                    "TimeO": 0.0,
                    "terminated": 0.0,
                    "SFLSparse": 0.0,
                    "SFLDense": 0.0,
                    "SFLLidar": 0.0,
                }
            )
        return obs, info

    def step(self, action_dict: Dict[str, np.ndarray]):
        _, _, dones, truncated, infos = self.base_env.step(action_dict)

        obs = {}
        rewards = {}
        for aid in self.agent_ids:
            obs[aid] = self._build_agent_obs(aid)
            info = infos.setdefault(aid, {})
            status = str(info.get("status", ""))
            if status in {"done_waiting", "no_action_received"}:
                info.update(
                    {
                        "GoalR": 0.0,
                        "MapC": 0.0,
                        "AgentC": 0.0,
                        "TimeO": 0.0,
                        "terminated": 0.0,
                        "SFLSparse": 0.0,
                        "SFLDense": 0.0,
                        "SFLLidar": 0.0,
                        "SFLReward": 0.0,
                    }
                )
                rewards[aid] = 0.0
                continue

            if "neighbor_min_dist" not in info:
                neighbor_fn = getattr(self.base_env, "_compute_neighbor_min_distances", None)
                if callable(neighbor_fn):
                    try:
                        info["neighbor_min_dist"] = float(neighbor_fn().get(aid, float("inf")))
                    except Exception:
                        pass

            rewards[aid] = float(self._compute_sampling_reward(info, truncated=bool(truncated.get(aid, False))))

        return obs, rewards, dones, truncated, infos

    def _build_agent_obs(self, agent_id: str) -> np.ndarray:
        agent = self.base_env.agents[agent_id]

        scan = getattr(agent, "latest_scan", None)
        ranges = None
        if scan is not None and getattr(scan, "ranges", None):
            ranges = np.asarray(scan.ranges, dtype=np.float32)

        if ranges is None or ranges.size == 0:
            ranges = np.full((360,), self.lidar_max_range, dtype=np.float32)

        ranges = np.nan_to_num(
            ranges,
            nan=self.lidar_max_range,
            posinf=self.lidar_max_range,
            neginf=self.lidar_min_range,
        )
        ranges = np.clip(ranges, self.lidar_min_range, self.lidar_max_range)

        idx = np.linspace(0, max(0, ranges.size - 1), num=self.lidar_num_beams, dtype=np.int32)
        scan_obs = ranges[idx] / max(self.lidar_max_range, 1e-6)
        scan_obs = np.clip(scan_obs, 0.0, 1.0)

        target = (
            agent._get_tracking_target()
            if hasattr(agent, "_get_tracking_target")
            else getattr(agent, "goal_pos", (0.0, 0.0))
        )
        dx = float(target[0]) - float(agent.current_pose.get("x", 0.0))
        dy = float(target[1]) - float(agent.current_pose.get("y", 0.0))
        dist = math.hypot(dx, dy)
        tgt_angle = math.atan2(dy, dx)
        yaw = float(agent.current_pose.get("yaw", 0.0))
        rel_angle = (tgt_angle - yaw + math.pi) % (2.0 * math.pi) - math.pi

        max_lin = max(float(getattr(agent, "max_linear_vel", 0.22)), 1e-6)
        max_ang = max(float(getattr(agent, "max_angular_vel", 1.2)), 1e-6)

        extra = np.array(
            [
                np.clip(dist / max(self.lidar_max_range, 1e-6), 0.0, 1.0),
                float(np.sin(rel_angle)),
                float(np.cos(rel_angle)),
                np.clip(float(getattr(agent, "current_vel_x", 0.0)) / max_lin, -1.0, 1.0),
                np.clip(float(getattr(agent, "current_vel_w", 0.0)) / max_ang, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate([scan_obs.astype(np.float32), extra], axis=0)

    def _compute_sampling_reward(self, info: Dict, truncated: bool) -> float:
        event = str(info.get("event", ""))
        goal = event == "goal"
        collision = event == "collision"
        timeout = bool(truncated and not goal and not collision)

        sparse = 0.0
        if goal:
            sparse += self.goal_rew
        if collision:
            sparse += self.coll_rew

        dense = self.dt_rew
        lidar_term = 0.0
        min_dist = info.get("min_dist")
        if min_dist is not None:
            try:
                min_dist_f = float(min_dist)
            except (TypeError, ValueError):
                min_dist_f = float("inf")
            if np.isfinite(min_dist_f) and min_dist_f < self.lidar_thresh:
                ratio = (self.lidar_thresh - min_dist_f) / max(self.lidar_thresh, 1e-6)
                lidar_term = self.lidar_rew * ratio
                dense += lidar_term

        reward = self.rew_lambda * sparse + (1.0 - self.rew_lambda) * dense

        neighbor_min = info.get("neighbor_min_dist", float("inf"))
        try:
            neighbor_min = float(neighbor_min)
        except (TypeError, ValueError):
            neighbor_min = float("inf")
        agent_collision = float(
            collision and np.isfinite(neighbor_min) and neighbor_min < self.agent_collision_dist
        )
        map_collision = float(collision and not agent_collision)

        info.update(
            {
                "GoalR": float(goal),
                "MapC": float(map_collision),
                "AgentC": float(agent_collision),
                "TimeO": float(timeout),
                "terminated": float(goal or collision or timeout),
                "SFLSparse": float(sparse),
                "SFLDense": float(dense),
                "SFLLidar": float(lidar_term),
                "SFLReward": float(reward),
            }
        )
        return float(reward)

    def close(self):
        close_fn = getattr(self.base_env, "close", None)
        if callable(close_fn):
            close_fn()


# RLlib env creator ---------------------------------------------------------

def sfl_nav_env_creator(env_config):
    cfg = dict(env_config)
    for key in ("worker_index", "vector_index", "num_workers", "remote"):
        if key not in cfg and hasattr(env_config, key):
            cfg[key] = getattr(env_config, key)
    return SamplingStyleNavEnv(cfg)
