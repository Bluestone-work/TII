"""Strict branch snapshots for paired finite-horizon Gazebo interventions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import random
import time
from typing import Any, Dict

import numpy as np
import rclpy
from example_interfaces.srv import AddTwoInts
from gazebo_msgs.srv import GetEntityState, SetEntityState
from std_srvs.srv import Empty, Trigger


ENV_STATE_FIELDS = (
    "current_step_count", "dones", "failed_agents", "robot_positions",
    "robot_velocities", "spawned_static_obstacles", "episode_successes",
    "episode_collisions", "episode_stats", "_state_history", "rng",
    "_last_adj_matrix", "_paired_trajectory_pending_start",
    "dynamic_obstacle_trajectory_fingerprint", "dynamic_obstacle_trajectory_spec",
    "_ros2_neighbor_bufs",
)

AGENT_STATE_FIELDS = (
    "current_step", "current_pose", "current_vel_x", "current_vel_w",
    "_close_obstacle_streak", "_gazebo_collision_active",
    "_last_predictive_metrics", "_last_gap_metrics", "_subgoal_detour_hold",
    "_subgoal_detour_side", "_subgoal_deadlock_streak", "_next_replan_step",
    "_progress_replan_window_start_step", "_progress_replan_window_start_progress",
    "_yield_hold_steps", "_yield_partner", "_yield_turn_sign",
    "global_waypoints", "_global_path_valid", "current_waypoint_index",
    "current_subgoal", "current_projection", "current_path_heading",
    "path_progress", "prev_path_progress", "current_lateral_error",
    "_obs_target_state", "_last_subgoal_mode", "_last_guidance_near_weight",
    "_last_path_curvature", "_last_guidance_lookahead",
    "_last_allowed_lateral_error", "_last_path_clearance",
    "_last_swept_clearance", "_last_swept_filter_active",
    "_last_motion_safety_margin", "_last_motion_safety_source",
    "_last_motion_filter_active", "_last_motion_features",
    "prev_dist_to_goal", "prev_target_point", "prev_dist_to_target",
    "prev_abs_target_angle", "_last_action_primitive", "_last_shield_info",
    "_front_min_history", "_front_sector_dist_history",
    "_obstacle_cluster_history", "_scan_history", "_local_map_history",
    "goal_pos", "last_spawn_pos", "spawned_static_obstacles",
    "_static_obstacles_spawned",
)


def _clone(value: Any) -> Any:
    if isinstance(value, np.random.Generator):
        cloned = np.random.default_rng()
        cloned.bit_generator.state = copy.deepcopy(value.bit_generator.state)
        return cloned
    return copy.deepcopy(value)


def _capture_fields(obj: Any, names: tuple[str, ...]) -> dict:
    return {name: _clone(getattr(obj, name)) for name in names if hasattr(obj, name)}


def _restore_fields(obj: Any, values: dict) -> None:
    for name, value in values.items():
        setattr(obj, name, _clone(value))


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, deque, set)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.round(value.astype(np.float64), 6).tolist()
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, np.random.Generator):
        return _canonical(value.bit_generator.state)
    return repr(value)


@dataclass
class BranchSnapshot:
    obstacle_snapshot_id: int
    env_state: dict
    agent_states: Dict[str, dict]
    robot_states: Dict[str, dict]
    observations: Dict[str, np.ndarray]
    policy_states: Dict[str, list]
    python_random_state: object
    numpy_random_state: tuple


class BranchSnapshotManager:
    """Capture and restore simulator, environment, communication, and RNN state."""

    def __init__(self, env, timeout: float = 5.0):
        self.env = env
        self.timeout = float(timeout)
        self.master = env.agents[env.agent_ids[0]].node
        self.pause_client = self.master.create_client(Empty, "/pause_physics")
        self.unpause_client = self.master.create_client(Empty, "/unpause_physics")
        self.get_state_client = self.master.create_client(GetEntityState, "/get_entity_state")
        self.set_state_client = self.master.create_client(SetEntityState, "/set_entity_state")
        self.snapshot_obstacle_client = self.master.create_client(
            AddTwoInts, "/snapshot_dynamic_obstacles")
        self.restore_obstacle_client = self.master.create_client(
            AddTwoInts, "/restore_dynamic_obstacles")
        self.start_obstacle_client = self.master.create_client(
            Trigger, "/start_dynamic_obstacles")

    def _call(self, client, request, name: str):
        if not client.wait_for_service(timeout_sec=self.timeout):
            raise RuntimeError(f"required branch service unavailable: {name}")
        future = client.call_async(request)
        deadline = time.time() + self.timeout
        while rclpy.ok() and not future.done() and time.time() < deadline:
            rclpy.spin_once(self.master, timeout_sec=0.01)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"branch service timed out: {name}")
        return future.result()

    def _get_entity_state(self, name: str) -> dict:
        request = GetEntityState.Request()
        request.name = name
        request.reference_frame = "world"
        response = self._call(self.get_state_client, request, "/get_entity_state")
        if not getattr(response, "success", True):
            raise RuntimeError(f"failed to read Gazebo state for {name}")
        state = response.state
        return {
            "name": state.name or request.name,
            "position": [state.pose.position.x, state.pose.position.y, state.pose.position.z],
            "orientation": [
                state.pose.orientation.x, state.pose.orientation.y,
                state.pose.orientation.z, state.pose.orientation.w,
            ],
            "linear": [state.twist.linear.x, state.twist.linear.y, state.twist.linear.z],
            "angular": [state.twist.angular.x, state.twist.angular.y, state.twist.angular.z],
        }

    def _get_robot_state(self, aid: str) -> dict:
        return self._get_entity_state(self.env.agents[aid].gazebo_model_name)

    def _set_robot_state(self, aid: str, values: dict) -> None:
        request = SetEntityState.Request()
        request.state.name = values["name"]
        request.state.reference_frame = "world"
        position = values["position"]
        orientation = values["orientation"]
        linear = values["linear"]
        angular = values["angular"]
        request.state.pose.position.x, request.state.pose.position.y, request.state.pose.position.z = position
        (
            request.state.pose.orientation.x, request.state.pose.orientation.y,
            request.state.pose.orientation.z, request.state.pose.orientation.w,
        ) = orientation
        request.state.twist.linear.x, request.state.twist.linear.y, request.state.twist.linear.z = linear
        request.state.twist.angular.x, request.state.twist.angular.y, request.state.twist.angular.z = angular
        response = self._call(
            self.set_state_client, request, "/set_entity_state")
        if not getattr(response, "success", True):
            raise RuntimeError(f"failed to restore Gazebo state for {aid}")

    def capture(self, observations: dict, policy_states: dict) -> BranchSnapshot:
        obstacle_request = AddTwoInts.Request()
        obstacle_request.a = 0
        obstacle_request.b = 0
        obstacle_response = self._call(
            self.snapshot_obstacle_client, obstacle_request, "/snapshot_dynamic_obstacles")
        obstacle_snapshot_id = int(obstacle_response.sum)
        if obstacle_snapshot_id <= 0:
            raise RuntimeError("dynamic-obstacle snapshot failed")
        self._call(self.pause_client, Empty.Request(), "/pause_physics")
        return BranchSnapshot(
            obstacle_snapshot_id=obstacle_snapshot_id,
            env_state=_capture_fields(self.env, ENV_STATE_FIELDS),
            agent_states={
                aid: _capture_fields(agent, AGENT_STATE_FIELDS)
                for aid, agent in self.env.agents.items()
            },
            robot_states={aid: self._get_robot_state(aid) for aid in self.env.agent_ids},
            observations={aid: np.asarray(obs).copy() for aid, obs in observations.items()},
            policy_states={aid: _clone(state) for aid, state in policy_states.items()},
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
        )

    def restore(self, snapshot: BranchSnapshot) -> tuple[dict, dict, str]:
        self._call(self.pause_client, Empty.Request(), "/pause_physics")
        obstacle_request = AddTwoInts.Request()
        obstacle_request.a = int(snapshot.obstacle_snapshot_id)
        obstacle_request.b = 0
        response = self._call(
            self.restore_obstacle_client, obstacle_request, "/restore_dynamic_obstacles")
        if int(response.sum) != int(snapshot.obstacle_snapshot_id):
            raise RuntimeError("dynamic-obstacle restore failed")
        time.sleep(0.05)
        for aid, values in snapshot.robot_states.items():
            self._set_robot_state(aid, values)
        _restore_fields(self.env, snapshot.env_state)
        for aid, values in snapshot.agent_states.items():
            _restore_fields(self.env.agents[aid], values)
        random.setstate(snapshot.python_random_state)
        np.random.set_state(snapshot.numpy_random_state)
        observations = {aid: obs.copy() for aid, obs in snapshot.observations.items()}
        policy_states = {aid: _clone(state) for aid, state in snapshot.policy_states.items()}
        fingerprint = self.fingerprint(policy_states)
        return observations, policy_states, fingerprint

    def omit_agent(self, aid: str) -> None:
        park_x, park_y, park_yaw = self.env._completed_agent_parking_pose(aid)
        values = {
            "name": self.env.agents[aid].gazebo_model_name,
            "position": [park_x, park_y, 0.1],
            "orientation": [0.0, 0.0, math.sin(park_yaw / 2.0), math.cos(park_yaw / 2.0)],
            "linear": [0.0, 0.0, 0.0],
            "angular": [0.0, 0.0, 0.0],
        }
        self._set_robot_state(aid, values)
        agent = self.env.agents[aid]
        agent.current_pose = {"x": park_x, "y": park_y, "yaw": park_yaw}
        agent.current_vel_x = 0.0
        agent.current_vel_w = 0.0
        self.env.robot_positions[aid] = np.asarray([park_x, park_y], dtype=np.float32)
        self.env.robot_velocities[aid] = np.zeros(2, dtype=np.float32)
        self.env.dones.add(aid)

    def resume(self) -> None:
        self._call(self.unpause_client, Empty.Request(), "/unpause_physics")
        self._call(self.start_obstacle_client, Trigger.Request(), "/start_dynamic_obstacles")

    def fingerprint(self, policy_states: dict) -> str:
        physical = {aid: self._get_robot_state(aid) for aid in self.env.agent_ids}
        obstacle_count = int(getattr(self.env.agents[self.env.agent_ids[0]], "num_dynamic_obstacles", 0))
        dynamic_obstacles = {
            f"dyn_obs_{index}": self._get_entity_state(f"dyn_obs_{index}")
            for index in range(obstacle_count)
        }
        payload = {
            "physical": physical,
            "dynamic_obstacles": dynamic_obstacles,
            "env": {
                "step": self.env.current_step_count,
                "dones": sorted(self.env.dones),
                "failed": sorted(self.env.failed_agents),
                "positions": self.env.robot_positions,
                "velocities": self.env.robot_velocities,
            },
            "agents": {
                aid: {
                    "step": agent.current_step,
                    "pose": agent.current_pose,
                    "velocity": [agent.current_vel_x, agent.current_vel_w],
                    "path_progress": agent.path_progress,
                    "prev_path_progress": agent.prev_path_progress,
                    "yield": [agent._yield_hold_steps, agent._yield_partner, agent._yield_turn_sign],
                }
                for aid, agent in self.env.agents.items()
            },
            "policy_states": policy_states,
        }
        encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
