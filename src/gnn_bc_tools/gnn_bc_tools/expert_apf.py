from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from gnn_bc_tools.expert_orca_dwa import ORCADWATeacher, _wrap_angle


class APFTeacher(ORCADWATeacher):
    """Global waypoint + APF local avoidance teacher.

    This keeps the same outer collection loop / safety wrappers as the existing
    BC teacher, but replaces the low-level DWA local planner with an APF
    controller. The goal is not a textbook pure potential field; it is a
    pragmatic local controller with:
    - attractive pull to the current local goal,
    - repulsion from fused laser obstacles,
    - stronger repulsion from neighboring robots,
    - a tangential term to reduce sticking / head-on deadlock,
    - the existing intent-commitment and close-range safety hold logic.
    """

    def __init__(
        self,
        *,
        apf_attract_gain: float = 0.85,
        apf_obstacle_gain: float = 0.22,
        apf_robot_gain: float = 0.42,
        apf_tangent_gain: float = 0.18,
        apf_damping_gain: float = 0.16,
        apf_influence_radius: float = 1.15,
        apf_robot_influence_radius: float = 1.45,
        apf_goal_slow_radius: float = 0.70,
        apf_obstacle_top_k: int = 28,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.apf_attract_gain = max(0.0, float(apf_attract_gain))
        self.apf_obstacle_gain = max(0.0, float(apf_obstacle_gain))
        self.apf_robot_gain = max(0.0, float(apf_robot_gain))
        self.apf_tangent_gain = max(0.0, float(apf_tangent_gain))
        self.apf_damping_gain = max(0.0, float(apf_damping_gain))
        self.apf_influence_radius = max(0.10, float(apf_influence_radius))
        self.apf_robot_influence_radius = max(0.10, float(apf_robot_influence_radius))
        self.apf_goal_slow_radius = max(0.05, float(apf_goal_slow_radius))
        self.apf_obstacle_top_k = max(4, int(apf_obstacle_top_k))

    def _compute_nominal_plan(self, env, aid: str, step_idx: int) -> Dict[str, Any]:
        agent = env.agents[aid]
        pos = np.asarray(env.robot_positions[aid], dtype=np.float32)
        vel_xy = np.asarray(env.robot_velocities[aid], dtype=np.float32)
        yaw = float(agent.current_pose.get("yaw", 0.0))
        local_goal = self._get_local_goal(agent)
        neighbor_states = self._collect_neighbor_states(env, aid, pos)

        obstacle_points = self.obstacle_fuser.build_obstacles(
            agent_id=aid,
            agent=agent,
            pos=pos,
            yaw=yaw,
            neighbor_states=neighbor_states,
            step_idx=int(step_idx),
            include_robot_obstacles=False,
        )

        force = np.zeros(2, dtype=np.float32)

        goal_vec = np.asarray(local_goal, dtype=np.float32) - pos
        goal_dist = float(np.linalg.norm(goal_vec))
        if goal_dist > 1e-6:
            goal_dir = goal_vec / goal_dist
            attract_scale = min(goal_dist, self.apf_goal_slow_radius) / max(self.apf_goal_slow_radius, 1e-6)
            force += self.apf_attract_gain * attract_scale * goal_dir
        else:
            goal_dir = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float32)

        force += self._repulsive_force_from_points(
            origin=pos,
            target_dir=goal_dir,
            points=obstacle_points,
            influence_radius=self.apf_influence_radius,
            gain=self.apf_obstacle_gain,
            tangent_gain=self.apf_tangent_gain,
            top_k=self.apf_obstacle_top_k,
        )

        robot_points = self._predict_neighbor_points(neighbor_states)
        force += self._repulsive_force_from_points(
            origin=pos,
            target_dir=goal_dir,
            points=robot_points,
            influence_radius=self.apf_robot_influence_radius,
            gain=self.apf_robot_gain,
            tangent_gain=0.5 * self.apf_tangent_gain,
            top_k=max(8, self.apf_obstacle_top_k // 2),
        )

        force += -self.apf_damping_gain * np.asarray(vel_xy, dtype=np.float32)

        v_cmd, w_cmd = self._force_to_unicycle(
            force=force,
            yaw=yaw,
            goal_dist=goal_dist,
        )
        v_cmd, w_cmd = self._enforce_neighbor_safety(
            aid,
            v_cmd,
            w_cmd,
            yaw,
            pos,
            vel_xy,
            neighbor_states,
        )

        return {
            "v_cmd": float(np.clip(v_cmd, 0.0, self.max_linear_speed)),
            "w_cmd": float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed)),
            "pos": pos,
            "yaw": yaw,
        }

    def _predict_neighbor_points(self, neighbor_states: List[Dict[str, Any]]) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        horizon_steps = max(1, int(round(0.6 / max(self.obstacle_fuser.control_dt, 1e-3))))
        for ns in neighbor_states:
            pos = np.asarray(ns["pos"], dtype=np.float32)
            vel = np.asarray(ns.get("vel", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            points.append(pos.copy())
            speed = float(np.linalg.norm(vel))
            if speed < 0.02:
                continue
            for k in range(1, horizon_steps + 1):
                points.append(pos + vel * (self.obstacle_fuser.control_dt * float(k)))
        return points

    def _repulsive_force_from_points(
        self,
        *,
        origin: np.ndarray,
        target_dir: np.ndarray,
        points: List[np.ndarray],
        influence_radius: float,
        gain: float,
        tangent_gain: float,
        top_k: int,
    ) -> np.ndarray:
        if not points or gain <= 0.0:
            return np.zeros(2, dtype=np.float32)

        ranked: List[Tuple[float, np.ndarray]] = []
        for point in points:
            delta = origin - np.asarray(point, dtype=np.float32)
            dist = float(np.linalg.norm(delta))
            if not np.isfinite(dist) or dist < 1e-4 or dist > influence_radius:
                continue
            ranked.append((dist, delta))

        if not ranked:
            return np.zeros(2, dtype=np.float32)

        ranked.sort(key=lambda item: item[0])
        repulsive = np.zeros(2, dtype=np.float32)
        q = max(influence_radius, 1e-3)

        for dist, delta in ranked[:top_k]:
            away = delta / max(dist, 1e-6)
            scale = gain * ((1.0 / dist) - (1.0 / q)) / max(dist * dist, 1e-6)
            repulsive += np.asarray(scale * away, dtype=np.float32)

            if tangent_gain > 0.0:
                tangent = np.array([-away[1], away[0]], dtype=np.float32)
                if float(np.dot(tangent, target_dir)) < 0.0:
                    tangent = -tangent
                tangent_scale = tangent_gain * max(0.0, 1.0 - dist / q)
                repulsive += np.asarray(tangent_scale * tangent, dtype=np.float32)

        return repulsive

    def _force_to_unicycle(
        self,
        *,
        force: np.ndarray,
        yaw: float,
        goal_dist: float,
    ) -> Tuple[float, float]:
        norm = float(np.linalg.norm(force))
        if norm < 1e-5 or goal_dist < 0.08:
            return 0.0, 0.0

        desired_heading = float(math.atan2(force[1], force[0]))
        yaw_error = _wrap_angle(desired_heading - float(yaw))
        w_cmd = float(np.clip(2.2 * yaw_error, -self.max_angular_speed, self.max_angular_speed))

        desired_speed = min(self.max_linear_speed, norm)
        heading_gate = max(0.0, math.cos(yaw_error))
        if abs(yaw_error) > 1.15:
            heading_gate = 0.0

        goal_gate = min(1.0, goal_dist / max(self.apf_goal_slow_radius, 1e-6))
        v_cmd = desired_speed * heading_gate * max(0.20, goal_gate)
        return float(v_cmd), float(w_cmd)
