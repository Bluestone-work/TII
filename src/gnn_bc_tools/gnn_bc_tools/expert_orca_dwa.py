import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from start_orca_nav.orca_algorithm import ORCAAgent, compute_preferred_velocity
from start_orca_nav.dwa_planner import create_dwa_planner
from gnn_bc_tools.orca_dwa_scene_utils import DWAObstacleFuser


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class ORCADWATeacher:
    """ORCA + DWA teacher policy that outputs normalized RL actions in [-1, 1]."""

    def __init__(
        self,
        communication_range: float,
        max_linear_speed: float = 0.22,
        max_angular_speed: float = 1.2,
        robot_radius: float = 0.25,
        time_horizon: float = 2.0,
        laser_obstacle_max_dist: float = 2.0,
        velocity_smoothing_alpha: float = 0.6,
        neighbor_soft_dist: float = 0.72,
        neighbor_stop_dist: float = 0.36,
        neighbor_hard_stop_dist: float = 0.27,
        orca_blend_max: float = 0.78,
        dwa_heading_weight: float = 2.0,
        dwa_dist_weight: float = 2.8,
        dwa_velocity_weight: float = 1.5,
        dwa_safety_margin: float = 0.14,
        intent_horizon_sec: float = 1.8,
        intent_dt_sec: float = 0.2,
        intent_safe_margin: float = 0.12,
        intent_commit_steps: int = 4,
        intent_replan_interval_steps: int = 2,
        intent_dropout_prob: float = 0.05,
        intent_latency_steps: int = 1,
        intent_jitter_steps: int = 1,
        intent_max_staleness_steps: int = 20,
        intent_seed: Optional[int] = None,
    ):
        self.communication_range = float(communication_range)
        self.max_linear_speed = float(max_linear_speed)
        self.max_angular_speed = float(max_angular_speed)
        self.robot_radius = float(robot_radius)
        self.time_horizon = float(time_horizon)
        self.laser_obstacle_max_dist = float(laser_obstacle_max_dist)
        self.alpha = float(velocity_smoothing_alpha)
        self.neighbor_soft_dist = float(neighbor_soft_dist)
        self.neighbor_stop_dist = float(neighbor_stop_dist)
        self.neighbor_hard_stop_dist = float(neighbor_hard_stop_dist)
        self.orca_blend_max = float(np.clip(orca_blend_max, 0.0, 1.0))

        self.intent_horizon_sec = float(np.clip(float(intent_horizon_sec), 1.0, 3.0))
        self.intent_dt_sec = max(0.05, float(intent_dt_sec))
        self.intent_points = max(2, int(round(self.intent_horizon_sec / self.intent_dt_sec)))
        self.intent_safe_dist = 2.0 * self.robot_radius + max(0.0, float(intent_safe_margin))
        self.intent_commit_steps = max(1, int(intent_commit_steps))
        self.intent_replan_interval_steps = max(1, int(intent_replan_interval_steps))
        self.intent_dropout_prob = float(np.clip(intent_dropout_prob, 0.0, 0.95))
        self.intent_latency_steps = max(0, int(intent_latency_steps))
        self.intent_jitter_steps = max(0, int(intent_jitter_steps))
        self.intent_max_staleness_steps = max(1, int(intent_max_staleness_steps))
        self.intent_history_len = max(
            32,
            self.intent_latency_steps + self.intent_jitter_steps + self.intent_max_staleness_steps + 8,
        )
        self._intent_rng = np.random.default_rng(intent_seed)

        self.dwa_planner = create_dwa_planner(
            max_speed=self.max_linear_speed,
            max_yaw_rate=self.max_angular_speed,
            robot_radius=self.robot_radius,
        )
        # Tune DWA to be less speed-greedy and more conservative around obstacles.
        self.dwa_planner.config.heading_weight = float(dwa_heading_weight)
        self.dwa_planner.config.dist_weight = float(dwa_dist_weight)
        self.dwa_planner.config.velocity_weight = float(dwa_velocity_weight)
        self.dwa_planner.config.safety_margin = max(float(dwa_safety_margin), 0.0)
        self.dwa_planner.config.predict_time = max(1.8, float(self.dwa_planner.config.predict_time))
        self.dwa_planner.config.obstacle_check_distance = max(
            2.2,
            float(self.dwa_planner.config.obstacle_check_distance),
        )
        self.obstacle_fuser = DWAObstacleFuser(
            robot_radius=self.robot_radius,
            laser_obstacle_max_dist=self.laser_obstacle_max_dist,
            control_dt=float(self.dwa_planner.config.dt),
        )
        self._current_cmd_vel: Dict[str, Tuple[float, float]] = {}
        # Existing close-range anti-oscillation for immediate neighbor safety.
        self._yield_state: Dict[str, Dict[str, float]] = {}
        # Intent-sharing state for short-horizon trajectory commitments.
        self._intent_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self._intent_commit_state: Dict[str, Dict[str, float]] = {}
        self._intent_replan_phase: Dict[str, int] = {}

    def reset(self) -> None:
        self.obstacle_fuser.reset()
        self._current_cmd_vel.clear()
        self._yield_state.clear()
        self._intent_history.clear()
        self._intent_commit_state.clear()
        self._intent_replan_phase.clear()

    def compute_actions(self, env, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute teacher actions for active agents in env.

        Args:
            env: GNNMARLEnv instance.
            obs_dict: Current observations from env.

        Returns:
            action_dict for env.step(...), each action is normalized to [-1, 1].
        """
        action_dict: Dict[str, np.ndarray] = {}
        active_ids = [aid for aid in env.agent_ids if aid not in getattr(env, "dones", set())]
        if not active_ids:
            return action_dict

        step_idx = self._get_env_step(env)
        history_snapshot = self._snapshot_intent_history()
        nominal_plans: Dict[str, Dict[str, Any]] = {}

        for aid in active_ids:
            if aid not in obs_dict:
                continue
            nominal_plans[aid] = self._compute_nominal_plan(env, aid, step_idx)

        publish_packets: Dict[str, Dict[str, Any]] = {}
        for aid in active_ids:
            plan = nominal_plans.get(aid)
            if plan is None:
                continue

            v_cmd = float(plan["v_cmd"])
            w_cmd = float(plan["w_cmd"])
            pos = np.asarray(plan["pos"], dtype=np.float32)
            yaw = float(plan["yaw"])

            # Asynchronous replanning: each agent has a fixed phase and softly
            # holds the previous command between replan ticks.
            v_cmd, w_cmd = self._apply_async_replan_gate(aid, step_idx, v_cmd, w_cmd)

            my_traj = self._predict_constant_trajectory(
                pos=pos,
                yaw=yaw,
                v_cmd=v_cmd,
                w_cmd=w_cmd,
            )
            received_intents = self._collect_received_intents(
                env=env,
                aid=aid,
                my_pos=pos,
                step_idx=step_idx,
                history_snapshot=history_snapshot,
            )
            v_cmd, w_cmd = self._apply_intent_conflict_resolution(
                aid=aid,
                pos=pos,
                yaw=yaw,
                v_cmd=v_cmd,
                w_cmd=w_cmd,
                my_traj=my_traj,
                received_intents=received_intents,
            )
            v_cmd, w_cmd = self._apply_intent_commitment(aid, v_cmd, w_cmd)

            # Match ORCA nav node behavior: smooth command changes.
            current_vw = self._current_cmd_vel.get(aid, (0.0, 0.0))
            v_cmd = self.alpha * float(v_cmd) + (1.0 - self.alpha) * current_vw[0]
            w_cmd = self.alpha * float(w_cmd) + (1.0 - self.alpha) * current_vw[1]

            v_cmd = float(np.clip(v_cmd, 0.0, self.max_linear_speed))
            w_cmd = float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed))
            self._current_cmd_vel[aid] = (v_cmd, w_cmd)
            action_dict[aid] = self._vw_to_normalized_action(v_cmd, w_cmd)
            publish_packets[aid] = {"pos": pos, "yaw": yaw, "v": v_cmd, "w": w_cmd}

        # Publish intent after all actions are finalized so this step only uses
        # delayed/stale intents from previous cycles.
        for aid, pkt in publish_packets.items():
            traj = self._predict_constant_trajectory(
                pos=pkt["pos"],
                yaw=float(pkt["yaw"]),
                v_cmd=float(pkt["v"]),
                w_cmd=float(pkt["w"]),
            )
            self._publish_intent(aid=aid, step_idx=step_idx, pos=pkt["pos"], traj=traj)

        self._cleanup_inactive_state(active_ids)
        return action_dict

    def _compute_nominal_plan(self, env, aid: str, step_idx: int) -> Dict[str, Any]:
        agent = env.agents[aid]

        pos = np.asarray(env.robot_positions[aid], dtype=np.float32)
        vel_xy = np.asarray(env.robot_velocities[aid], dtype=np.float32)
        yaw = float(agent.current_pose.get("yaw", 0.0))

        local_goal = self._get_local_goal(agent)
        pref_velocity = compute_preferred_velocity(
            current_pos=pos,
            goal_pos=local_goal,
            max_speed=self.max_linear_speed,
        )

        self_agent = ORCAAgent(
            position=pos,
            velocity=vel_xy,
            radius=self.robot_radius,
            max_speed=self.max_linear_speed,
            pref_velocity=pref_velocity,
            time_horizon=self.time_horizon,
        )

        neighbor_states = self._collect_neighbor_states(env, aid, pos)
        neighbors = [
            ORCAAgent(
                position=ns["pos"],
                velocity=ns["vel"],
                radius=self.robot_radius,
                max_speed=self.max_linear_speed,
                pref_velocity=np.zeros(2, dtype=np.float32),
                time_horizon=self.time_horizon,
            )
            for ns in neighbor_states
        ]
        orca_velocity = self_agent.compute_new_velocity(neighbors)

        dwa_obstacles = self.obstacle_fuser.build_obstacles(
            agent_id=aid,
            agent=agent,
            pos=pos,
            yaw=yaw,
            neighbor_states=neighbor_states,
            step_idx=int(step_idx),
        )
        current_vw = self._current_cmd_vel.get(aid, (0.0, 0.0))
        # Use DWA as the base local controller with a fused obstacle cloud:
        # laser obstacles, predicted laser motion, and explicit robot occupancy.
        try:
            base_v, base_w = self.dwa_planner.plan(
                current_pos=pos,
                current_vel=current_vw,
                current_yaw=yaw,
                goal_pos=np.asarray(local_goal, dtype=np.float32),
                obstacles=dwa_obstacles,
            )
        except Exception:
            base_v, base_w = self._orca_velocity_to_unicycle(
                velocity_xy=orca_velocity,
                yaw=yaw,
            )

        # Keep DWA as the final local controller once it has seen the fused
        # obstacle cloud. Mixing ORCA velocity back into the safe DWA command
        # was reintroducing collisions around static/dynamic obstacles.
        v_cmd = float(base_v)
        w_cmd = float(base_w)
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

    def _get_local_goal(self, agent) -> np.ndarray:
        if getattr(agent, "current_subgoal", None) is not None:
            return np.asarray(agent.current_subgoal, dtype=np.float32)
        goal = getattr(agent, "goal_pos", (0.0, 0.0))
        return np.asarray(goal, dtype=np.float32)

    def _collect_neighbor_states(self, env, aid: str, my_pos: np.ndarray) -> List[Dict]:
        neighbors: List[Dict] = []
        detect_range = max(self.communication_range, self.neighbor_soft_dist * 2.0)

        for other_aid in env.agent_ids:
            if other_aid == aid:
                continue
            other_pos = np.asarray(env.robot_positions[other_aid], dtype=np.float32)
            dist = float(np.linalg.norm(other_pos - my_pos))
            if not np.isfinite(dist) or dist > detect_range:
                continue
            other_vel = np.asarray(env.robot_velocities[other_aid], dtype=np.float32)
            neighbors.append(
                {
                    "aid": other_aid,
                    "pos": other_pos,
                    "vel": other_vel,
                    "dist": float(dist),
                }
            )

        return neighbors

    def _enforce_neighbor_safety(
        self,
        aid: str,
        v_cmd: float,
        w_cmd: float,
        yaw: float,
        pos: np.ndarray,
        vel_xy: np.ndarray,
        neighbor_states: List[Dict],
    ) -> Tuple[float, float]:
        if not neighbor_states:
            self._yield_state.pop(aid, None)
            return float(v_cmd), float(w_cmd)

        hold = self._yield_state.get(aid)
        if hold is not None:
            partner = str(hold.get("partner", ""))
            partner_state = next((ns for ns in neighbor_states if str(ns.get("aid", "")) == partner), None)
            if partner_state is not None and float(partner_state["dist"]) < self.neighbor_soft_dist + 0.25:
                steps_left = int(hold.get("steps_left", 0))
                if steps_left > 0:
                    hold["steps_left"] = float(steps_left - 1)
                    self._yield_state[aid] = hold
                    v_hold = min(float(v_cmd), float(hold.get("v_cap", 0.08)))
                    w_hold = float(hold.get("turn_sign", 1.0)) * (0.75 * self.max_angular_speed)
                    return (
                        float(np.clip(v_hold, 0.0, self.max_linear_speed)),
                        float(np.clip(w_hold, -self.max_angular_speed, self.max_angular_speed)),
                    )
            self._yield_state.pop(aid, None)

        nearest = min(neighbor_states, key=lambda x: float(x["dist"]))
        dist = float(nearest["dist"])
        rel = np.asarray(nearest["pos"] - pos, dtype=np.float32)
        rel_norm = float(np.linalg.norm(rel))
        if rel_norm < 1e-6:
            rel = np.array([1.0, 0.0], dtype=np.float32)
            rel_norm = 1.0
        rel_unit = rel / rel_norm

        rel_vel = np.asarray(nearest["vel"] - vel_xy, dtype=np.float32)
        closing_speed = float(-np.dot(rel_vel, rel_unit))

        bearing = float(math.atan2(rel[1], rel[0]))
        yaw_err = _wrap_angle(bearing - float(yaw))
        turn_away_sign = -1.0 if yaw_err > 0.0 else 1.0
        avoid_turn = turn_away_sign * (0.85 * self.max_angular_speed)

        ttc = float("inf")
        if closing_speed > 1e-3:
            ttc = dist / max(closing_speed, 1e-6)

        my_forward = np.array([math.cos(float(yaw)), math.sin(float(yaw))], dtype=np.float32)
        my_toward_neighbor = float(np.dot(my_forward, rel_unit))
        other_vel = np.asarray(nearest["vel"], dtype=np.float32)
        other_speed = float(np.linalg.norm(other_vel))
        other_toward_me = 0.0
        if other_speed > 0.05:
            other_toward_me = float(np.dot(other_vel / max(other_speed, 1e-6), -rel_unit))
        head_on_like = (my_toward_neighbor > 0.25) and (other_toward_me > 0.20)

        if (
            head_on_like
            and dist < max(self.neighbor_soft_dist + 0.05, 0.75)
            and closing_speed > 0.04
            and ttc < 2.2
        ):
            # Deterministic right-of-way: higher index yields to lower index.
            if self._agent_rank(aid) > self._agent_rank(str(nearest["aid"])):
                yield_v_cap = 0.04 if dist <= self.neighbor_stop_dist + 0.05 else 0.08
                self._yield_state[aid] = {
                    "partner": str(nearest["aid"]),
                    "steps_left": 4.0,
                    "turn_sign": float(turn_away_sign),
                    "v_cap": float(yield_v_cap),
                }
                v_cmd = min(float(v_cmd), float(yield_v_cap))
                w_cmd = float(turn_away_sign) * (0.85 * self.max_angular_speed)
            else:
                # Priority side avoids over-yielding that often leads to mutual deadlock.
                if dist > self.neighbor_hard_stop_dist + 0.05:
                    v_cmd = max(float(v_cmd), 0.09)

        if dist <= self.neighbor_hard_stop_dist:
            v_cmd = 0.0
            w_cmd = avoid_turn
        elif dist <= self.neighbor_stop_dist:
            v_cmd = min(v_cmd, 0.03 if closing_speed > 0.0 else 0.06)
            w_cmd = avoid_turn if abs(w_cmd) < 0.6 * self.max_angular_speed else w_cmd
        elif dist < self.neighbor_soft_dist:
            span = max(self.neighbor_soft_dist - self.neighbor_stop_dist, 1e-6)
            ratio = (self.neighbor_soft_dist - dist) / span
            v_cap = 0.18 - 0.10 * float(np.clip(ratio, 0.0, 1.0))
            v_cmd = min(v_cmd, max(0.05, v_cap))
            if closing_speed > 0.05:
                w_cmd = (1.0 - ratio) * w_cmd + ratio * avoid_turn

        return (
            float(np.clip(v_cmd, 0.0, self.max_linear_speed)),
            float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed)),
        )

    def _get_env_step(self, env) -> int:
        if hasattr(env, "current_step_count"):
            try:
                return int(getattr(env, "current_step_count"))
            except Exception:
                pass
        if hasattr(env, "current_step"):
            try:
                return int(getattr(env, "current_step"))
            except Exception:
                pass
        return 0

    def _apply_async_replan_gate(
        self,
        aid: str,
        step_idx: int,
        v_cmd: float,
        w_cmd: float,
    ) -> Tuple[float, float]:
        if self.intent_replan_interval_steps <= 1:
            return float(v_cmd), float(w_cmd)

        phase = self._intent_replan_phase.get(aid)
        if phase is None:
            phase = self._agent_rank(aid) % self.intent_replan_interval_steps
            self._intent_replan_phase[aid] = int(phase)

        should_replan = ((int(step_idx) + int(phase)) % self.intent_replan_interval_steps) == 0
        if should_replan:
            return float(v_cmd), float(w_cmd)

        prev = self._current_cmd_vel.get(aid)
        if prev is None:
            return float(v_cmd), float(w_cmd)

        hold_mix = 0.82
        return (
            float(hold_mix * float(prev[0]) + (1.0 - hold_mix) * float(v_cmd)),
            float(hold_mix * float(prev[1]) + (1.0 - hold_mix) * float(w_cmd)),
        )

    def _predict_constant_trajectory(
        self,
        pos: np.ndarray,
        yaw: float,
        v_cmd: float,
        w_cmd: float,
    ) -> np.ndarray:
        x = float(pos[0])
        y = float(pos[1])
        th = float(yaw)
        v = float(np.clip(v_cmd, 0.0, self.max_linear_speed))
        w = float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed))
        dt = float(self.intent_dt_sec)
        traj = np.zeros((self.intent_points, 2), dtype=np.float32)
        for i in range(self.intent_points):
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            th = _wrap_angle(th + w * dt)
            traj[i, 0] = x
            traj[i, 1] = y
        return traj

    def _snapshot_intent_history(self) -> Dict[str, List[Dict[str, Any]]]:
        snap: Dict[str, List[Dict[str, Any]]] = {}
        for aid, hist in self._intent_history.items():
            snap[aid] = list(hist)
        return snap

    def _publish_intent(
        self,
        aid: str,
        step_idx: int,
        pos: np.ndarray,
        traj: np.ndarray,
    ) -> None:
        hist = self._intent_history.get(aid)
        if hist is None:
            hist = deque(maxlen=self.intent_history_len)
            self._intent_history[aid] = hist
        hist.append(
            {
                "step": int(step_idx),
                "pos": np.asarray(pos, dtype=np.float32).copy(),
                "traj": np.asarray(traj, dtype=np.float32).copy(),
            }
        )

    def _collect_received_intents(
        self,
        env,
        aid: str,
        my_pos: np.ndarray,
        step_idx: int,
        history_snapshot: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        received: List[Dict[str, Any]] = []
        detect_range = max(self.communication_range, self.neighbor_soft_dist * 2.0) + 0.3

        for other_aid in env.agent_ids:
            if other_aid == aid:
                continue

            hist = history_snapshot.get(other_aid)
            if not hist:
                continue

            other_pos = np.asarray(env.robot_positions[other_aid], dtype=np.float32)
            dist = float(np.linalg.norm(other_pos - my_pos))
            if not np.isfinite(dist) or dist > detect_range:
                continue

            if self._intent_rng.random() < self.intent_dropout_prob:
                continue

            jitter = int(self._intent_rng.integers(0, self.intent_jitter_steps + 1))
            delay = self.intent_latency_steps + jitter
            target_step = int(step_idx) - int(delay)

            msg = None
            for cand in reversed(hist):
                if int(cand.get("step", -10**9)) <= target_step:
                    msg = cand
                    break
            if msg is None:
                continue

            msg_step = int(msg.get("step", -10**9))
            age = int(step_idx) - msg_step
            if age > self.intent_max_staleness_steps:
                continue

            traj = np.asarray(msg.get("traj", []), dtype=np.float32)
            if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] != 2:
                continue

            received.append(
                {
                    "aid": str(other_aid),
                    "traj": traj,
                    "dist": float(dist),
                    "age_steps": int(age),
                }
            )

        received.sort(key=lambda d: float(d["dist"]))
        return received

    def _trajectory_min_sep(self, traj_a: np.ndarray, traj_b: np.ndarray) -> Tuple[float, int]:
        n = int(min(traj_a.shape[0], traj_b.shape[0]))
        if n <= 0:
            return float("inf"), -1
        d = np.linalg.norm(traj_a[:n] - traj_b[:n], axis=1)
        idx = int(np.argmin(d))
        return float(d[idx]), idx

    def _apply_intent_conflict_resolution(
        self,
        aid: str,
        pos: np.ndarray,
        yaw: float,
        v_cmd: float,
        w_cmd: float,
        my_traj: np.ndarray,
        received_intents: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        if not received_intents:
            return float(v_cmd), float(w_cmd)

        worst: Optional[Dict[str, Any]] = None
        for msg in received_intents:
            min_sep, idx = self._trajectory_min_sep(my_traj, np.asarray(msg["traj"], dtype=np.float32))
            if not np.isfinite(min_sep):
                continue
            if min_sep >= self.intent_safe_dist:
                continue
            if worst is None or float(min_sep) < float(worst["min_sep"]):
                worst = {
                    "aid": str(msg["aid"]),
                    "traj": np.asarray(msg["traj"], dtype=np.float32),
                    "min_sep": float(min_sep),
                    "idx": int(idx),
                }

        if worst is None:
            return float(v_cmd), float(w_cmd)

        other_aid = str(worst["aid"])
        other_rank = self._agent_rank(other_aid)
        my_rank = self._agent_rank(aid)
        other_traj = np.asarray(worst["traj"], dtype=np.float32)
        other_idx = int(np.clip(worst["idx"], 0, max(0, other_traj.shape[0] - 1)))
        other_pt = other_traj[other_idx]

        rel = np.asarray(other_pt - pos, dtype=np.float32)
        rel_norm = float(np.linalg.norm(rel))
        if rel_norm < 1e-6:
            rel = np.array([1.0, 0.0], dtype=np.float32)
            rel_norm = 1.0
        bearing = float(math.atan2(rel[1], rel[0]))
        yaw_err = _wrap_angle(bearing - float(yaw))
        turn_away_sign = -1.0 if yaw_err > 0.0 else 1.0
        severity = float(
            np.clip(
                (self.intent_safe_dist - float(worst["min_sep"])) / max(self.intent_safe_dist, 1e-6),
                0.0,
                1.0,
            )
        )

        if my_rank > other_rank:
            # Higher rank yields and commits briefly, reducing reciprocal re-yield.
            v_cap = float(0.02 + 0.08 * (1.0 - severity))
            target_w = float(turn_away_sign * (0.65 + 0.25 * severity) * self.max_angular_speed)
            v_cmd = min(float(v_cmd), v_cap)
            if abs(float(w_cmd)) < abs(target_w):
                w_cmd = target_w
            hold_steps = max(
                1,
                int(self.intent_commit_steps + math.ceil(0.5 * self.intent_latency_steps)),
            )
            self._intent_commit_state[aid] = {
                "v": float(v_cmd),
                "w": float(w_cmd),
                "steps_left": float(hold_steps),
                "partner": str(other_aid),
            }
        else:
            # Priority side avoids over-yielding, but keeps speed modest near conflict.
            if float(worst["min_sep"]) > (self.intent_safe_dist * 0.75):
                v_cmd = max(float(v_cmd), min(0.10, 0.55 * self.max_linear_speed))

        return (
            float(np.clip(v_cmd, 0.0, self.max_linear_speed)),
            float(np.clip(w_cmd, -self.max_angular_speed, self.max_angular_speed)),
        )

    def _apply_intent_commitment(
        self,
        aid: str,
        v_cmd: float,
        w_cmd: float,
    ) -> Tuple[float, float]:
        hold = self._intent_commit_state.get(aid)
        if hold is None:
            return float(v_cmd), float(w_cmd)

        steps_left = int(hold.get("steps_left", 0))
        if steps_left <= 0:
            self._intent_commit_state.pop(aid, None)
            return float(v_cmd), float(w_cmd)

        hold["steps_left"] = float(steps_left - 1)
        self._intent_commit_state[aid] = hold
        mix = 0.80
        v_out = mix * float(hold.get("v", v_cmd)) + (1.0 - mix) * float(v_cmd)
        w_out = mix * float(hold.get("w", w_cmd)) + (1.0 - mix) * float(w_cmd)

        if int(hold["steps_left"]) <= 0:
            self._intent_commit_state.pop(aid, None)

        return (
            float(np.clip(v_out, 0.0, self.max_linear_speed)),
            float(np.clip(w_out, -self.max_angular_speed, self.max_angular_speed)),
        )

    def _cleanup_inactive_state(self, active_ids: List[str]) -> None:
        active = set(active_ids)
        for aid in list(self._current_cmd_vel.keys()):
            if aid not in active:
                self._current_cmd_vel.pop(aid, None)
        for aid in list(self._yield_state.keys()):
            if aid not in active:
                self._yield_state.pop(aid, None)
        for aid in list(self._intent_commit_state.keys()):
            if aid not in active:
                self._intent_commit_state.pop(aid, None)

    def _agent_rank(self, aid: str) -> int:
        try:
            return int(str(aid).split("_")[-1])
        except Exception:
            return abs(hash(str(aid))) % 10000

    def _orca_velocity_to_unicycle(self, velocity_xy: np.ndarray, yaw: float) -> Tuple[float, float]:
        speed = float(np.linalg.norm(velocity_xy))
        if speed < 1e-6:
            return 0.0, 0.0

        desired_yaw = float(math.atan2(velocity_xy[1], velocity_xy[0]))
        yaw_error = _wrap_angle(desired_yaw - yaw)

        v_cmd = min(speed, self.max_linear_speed)
        if abs(yaw_error) > 0.7:
            v_cmd *= 0.35

        w_cmd = np.clip(2.0 * yaw_error, -self.max_angular_speed, self.max_angular_speed)
        return float(v_cmd), float(w_cmd)

    def _vw_to_normalized_action(self, v: float, w: float) -> np.ndarray:
        # In env: v = (a0 + 1) / 2 * max_linear_speed; w = a1 * max_angular_speed
        a0 = 2.0 * float(v) / max(self.max_linear_speed, 1e-6) - 1.0
        a1 = float(w) / max(self.max_angular_speed, 1e-6)
        return np.array(
            [
                np.clip(a0, -1.0, 1.0),
                np.clip(a1, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
