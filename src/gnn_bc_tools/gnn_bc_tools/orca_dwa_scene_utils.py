from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


@dataclass
class _ObstacleTrack:
    center: np.ndarray
    velocity: np.ndarray
    step_idx: int


class DWAObstacleFuser:
    """Build a DWA obstacle cloud from laser scans and nearby robot states.

    The BC teacher previously fed DWA only sparse laser points, which meant:
    1. other robots were not explicitly modeled as DWA obstacles, and
    2. moving obstacles were only represented at their instantaneous scan points.

    This helper keeps that geometry handling out of the teacher policy itself and
    fuses three signals into a single obstacle set:
    - clustered laser obstacles,
    - short-horizon predicted laser obstacle motion,
    - explicit nearby robot occupancy / short-horizon robot prediction.
    """

    def __init__(
        self,
        *,
        robot_radius: float,
        laser_obstacle_max_dist: float,
        control_dt: float,
        laser_sample_target: int = 180,
        laser_cluster_radius: float = 0.16,
        laser_inflation_radius: float = 0.12,
        laser_prediction_horizon_sec: float = 0.6,
        laser_match_radius: float = 0.45,
        robot_exclusion_radius: float = 0.45,
        robot_prediction_horizon_sec: float = 0.6,
        robot_obstacle_margin: float = 0.08,
        dedupe_grid_size: float = 0.10,
    ) -> None:
        self.robot_radius = float(robot_radius)
        self.laser_obstacle_max_dist = float(laser_obstacle_max_dist)
        self.control_dt = max(1e-3, float(control_dt))
        self.laser_sample_target = max(24, int(laser_sample_target))
        self.laser_cluster_radius = max(0.05, float(laser_cluster_radius))
        self.laser_inflation_radius = max(0.02, float(laser_inflation_radius))
        self.laser_prediction_horizon_sec = max(0.0, float(laser_prediction_horizon_sec))
        self.laser_match_radius = max(0.05, float(laser_match_radius))
        self.robot_exclusion_radius = max(0.05, float(robot_exclusion_radius))
        self.robot_prediction_horizon_sec = max(0.0, float(robot_prediction_horizon_sec))
        self.robot_obstacle_margin = max(0.0, float(robot_obstacle_margin))
        self.dedupe_grid_size = max(0.02, float(dedupe_grid_size))
        self._laser_tracks: Dict[str, List[_ObstacleTrack]] = {}

    def reset(self) -> None:
        self._laser_tracks.clear()

    def build_obstacles(
        self,
        *,
        agent_id: str,
        agent: Any,
        pos: np.ndarray,
        yaw: float,
        neighbor_states: Sequence[Mapping[str, Any]],
        step_idx: int,
        include_robot_obstacles: bool = True,
    ) -> List[np.ndarray]:
        laser_points = self._extract_laser_points(agent=agent, pos=pos, yaw=yaw)
        neighbor_positions = [
            np.asarray(ns["pos"], dtype=np.float32)
            for ns in neighbor_states
            if "pos" in ns
        ]
        static_laser_points = self._exclude_points_near_neighbors(
            points=laser_points,
            neighbor_positions=neighbor_positions,
        )
        laser_centers = self._cluster_points(static_laser_points, self.laser_cluster_radius)
        predicted_laser_centers = self._predict_laser_centers(
            agent_id=agent_id,
            centers=laser_centers,
            step_idx=int(step_idx),
        )
        laser_obstacles = self._inflate_centers(
            centers=predicted_laser_centers,
            radius=self.laser_inflation_radius,
            spokes=6,
        )
        robot_obstacles = self._build_robot_obstacles(neighbor_states) if include_robot_obstacles else []
        return self._dedupe_points(laser_obstacles + robot_obstacles)

    def _extract_laser_points(self, *, agent: Any, pos: np.ndarray, yaw: float) -> List[np.ndarray]:
        points: List[np.ndarray] = []
        scan = getattr(agent, "latest_scan", None)
        if scan is None or not getattr(scan, "ranges", None):
            return points

        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return points

        stride = max(1, int(math.ceil(ranges.size / float(self.laser_sample_target))))
        range_min = float(getattr(scan, "range_min", 0.0))
        range_max = float(getattr(scan, "range_max", 3.5))
        angle_min = float(getattr(scan, "angle_min", -math.pi))
        angle_inc = float(getattr(scan, "angle_increment", 2.0 * math.pi / max(1, ranges.size)))

        for idx in range(0, ranges.size, stride):
            r = float(ranges[idx])
            if not np.isfinite(r):
                continue
            if r < range_min or r > range_max or r > self.laser_obstacle_max_dist:
                continue
            angle = angle_min + idx * angle_inc
            world_angle = float(yaw) + angle
            points.append(
                np.array(
                    [
                        float(pos[0]) + r * math.cos(world_angle),
                        float(pos[1]) + r * math.sin(world_angle),
                    ],
                    dtype=np.float32,
                )
            )
        return points

    def _exclude_points_near_neighbors(
        self,
        *,
        points: Sequence[np.ndarray],
        neighbor_positions: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        if not points or not neighbor_positions:
            return [np.asarray(p, dtype=np.float32) for p in points]

        filtered: List[np.ndarray] = []
        for point in points:
            keep = True
            for neighbor_pos in neighbor_positions:
                if float(np.linalg.norm(point - neighbor_pos)) <= self.robot_exclusion_radius:
                    keep = False
                    break
            if keep:
                filtered.append(np.asarray(point, dtype=np.float32))
        return filtered

    def _cluster_points(self, points: Sequence[np.ndarray], cluster_radius: float) -> List[np.ndarray]:
        if not points:
            return []

        remaining = [np.asarray(p, dtype=np.float32) for p in points]
        clusters: List[np.ndarray] = []
        while remaining:
            seed = remaining.pop()
            members = [seed]
            kept: List[np.ndarray] = []
            for point in remaining:
                if float(np.linalg.norm(point - seed)) <= cluster_radius:
                    members.append(point)
                else:
                    kept.append(point)
            remaining = kept
            clusters.append(np.mean(np.stack(members, axis=0), axis=0).astype(np.float32))
        return clusters

    def _predict_laser_centers(
        self,
        *,
        agent_id: str,
        centers: Sequence[np.ndarray],
        step_idx: int,
    ) -> List[np.ndarray]:
        current_tracks: List[_ObstacleTrack] = []
        prev_tracks = list(self._laser_tracks.get(agent_id, []))
        remaining_prev = set(range(len(prev_tracks)))
        predicted: List[np.ndarray] = []

        for center in centers:
            center = np.asarray(center, dtype=np.float32)
            best_idx = None
            best_dist = float("inf")
            for idx in remaining_prev:
                dist = float(np.linalg.norm(center - prev_tracks[idx].center))
                if dist < self.laser_match_radius and dist < best_dist:
                    best_idx = idx
                    best_dist = dist

            velocity = np.zeros(2, dtype=np.float32)
            if best_idx is not None:
                prev = prev_tracks[best_idx]
                dt = max(1, step_idx - int(prev.step_idx)) * self.control_dt
                velocity = ((center - prev.center) / max(dt, 1e-3)).astype(np.float32)
                speed = float(np.linalg.norm(velocity))
                if speed > 0.8:
                    velocity *= 0.8 / max(speed, 1e-6)
                remaining_prev.discard(best_idx)

            track = _ObstacleTrack(center=center.copy(), velocity=velocity.copy(), step_idx=int(step_idx))
            current_tracks.append(track)
            predicted.append(center.copy())

            speed = float(np.linalg.norm(velocity))
            if speed < 0.05 or self.laser_prediction_horizon_sec <= 0.0:
                continue
            horizon_steps = max(1, int(round(self.laser_prediction_horizon_sec / self.control_dt)))
            for k in range(1, horizon_steps + 1):
                future_center = center + velocity * (self.control_dt * float(k))
                predicted.append(np.asarray(future_center, dtype=np.float32))

        self._laser_tracks[agent_id] = current_tracks
        return predicted

    def _build_robot_obstacles(self, neighbor_states: Sequence[Mapping[str, Any]]) -> List[np.ndarray]:
        obstacles: List[np.ndarray] = []
        inflated_radius = self.robot_radius + self.robot_obstacle_margin
        horizon_steps = max(1, int(round(self.robot_prediction_horizon_sec / self.control_dt)))

        for ns in neighbor_states:
            pos = np.asarray(ns["pos"], dtype=np.float32)
            vel = np.asarray(ns.get("vel", np.zeros(2, dtype=np.float32)), dtype=np.float32)
            sample_centers = [pos]
            speed = float(np.linalg.norm(vel))
            if speed > 0.02 and self.robot_prediction_horizon_sec > 0.0:
                for k in range(1, horizon_steps + 1):
                    sample_centers.append(pos + vel * (self.control_dt * float(k)))
            obstacles.extend(self._inflate_centers(sample_centers, inflated_radius, spokes=8))
        return obstacles

    def _inflate_centers(
        self,
        centers: Sequence[np.ndarray],
        radius: float,
        spokes: int,
    ) -> List[np.ndarray]:
        if radius <= 1e-6:
            return [np.asarray(c, dtype=np.float32) for c in centers]

        obstacles: List[np.ndarray] = []
        for center in centers:
            center = np.asarray(center, dtype=np.float32)
            obstacles.append(center.copy())
            for i in range(max(4, int(spokes))):
                angle = 2.0 * math.pi * float(i) / float(max(4, int(spokes)))
                obstacles.append(
                    np.array(
                        [
                            float(center[0]) + radius * math.cos(angle),
                            float(center[1]) + radius * math.sin(angle),
                        ],
                        dtype=np.float32,
                    )
                )
        return obstacles

    def _dedupe_points(self, points: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not points:
            return []

        buckets: Dict[tuple[int, int], np.ndarray] = {}
        for point in points:
            point = np.asarray(point, dtype=np.float32)
            key = (
                int(round(float(point[0]) / self.dedupe_grid_size)),
                int(round(float(point[1]) / self.dedupe_grid_size)),
            )
            buckets[key] = point
        return list(buckets.values())
