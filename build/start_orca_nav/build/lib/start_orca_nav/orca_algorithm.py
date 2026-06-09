"""
ORCA (Optimal Reciprocal Collision Avoidance) Algorithm
Implementation for multi-robot local collision avoidance
"""

import numpy as np
import math
from typing import List, Tuple


class ORCAAgent:
    """
    ORCA agent for computing collision-free velocities
    """
    
    def __init__(self, position: np.ndarray, velocity: np.ndarray, radius: float, 
                 max_speed: float, pref_velocity: np.ndarray, time_horizon: float = 2.0):
        """
        Initialize ORCA agent
        
        Args:
            position: 2D position [x, y]
            velocity: Current velocity [vx, vy]
            radius: Robot radius
            max_speed: Maximum speed
            pref_velocity: Preferred velocity (towards goal)
            time_horizon: Time horizon for collision avoidance
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = np.array(pref_velocity, dtype=float)
        self.time_horizon = time_horizon
        
    def compute_new_velocity(self, neighbors: List['ORCAAgent'], 
                            obstacles: List = None) -> np.ndarray:
        """
        Compute collision-free velocity using ORCA
        
        Args:
            neighbors: List of neighboring agents
            obstacles: List of obstacles (not implemented yet)
            
        Returns:
            New velocity [vx, vy]
        """
        orca_lines = []
        
        # Compute ORCA constraints for each neighbor
        for neighbor in neighbors:
            orca_line = self._compute_orca_line(neighbor)
            if orca_line is not None:
                orca_lines.append(orca_line)
        
        # Find optimal velocity that satisfies all ORCA constraints
        new_velocity = self._linear_program(orca_lines, self.pref_velocity)
        
        return new_velocity
    
    def _compute_orca_line(self, other: 'ORCAAgent') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ORCA half-plane constraint for one neighbor
        
        Returns:
            (direction, point): Half-plane defined by direction · (v - point) >= 0
        """
        relative_position = other.position - self.position
        relative_velocity = self.velocity - other.velocity
        dist_sq = np.dot(relative_position, relative_position)
        combined_radius = self.radius + other.radius
        combined_radius_sq = combined_radius ** 2
        
        dist = np.sqrt(dist_sq)
        
        # Check if agents are too close (collision)
        if dist_sq > combined_radius_sq:
            # No collision currently
            # Compute ORCA line
            w = relative_velocity - relative_position / self.time_horizon
            w_length = np.linalg.norm(w)
            
            if w_length < 1e-6:
                # Velocities are similar, use direction from relative position
                if dist > 1e-6:
                    unit_w = relative_position / dist
                else:
                    # Too close, emergency
                    unit_w = np.array([1.0, 0.0])
            else:
                unit_w = w / w_length
            
            # Direction of ORCA half-plane (perpendicular to unit_w)
            direction = np.array([unit_w[1], -unit_w[0]])
            
            # Point on ORCA line (velocity obstacle boundary)
            u = (combined_radius / self.time_horizon - w_length) * unit_w
            point = self.velocity + 0.5 * u  # Shared responsibility
            
            return (direction, point)
        else:
            # Collision! Need immediate avoidance
            # Use direction away from other agent
            if dist > 1e-6:
                collision_dir = relative_position / dist
            else:
                # Exact same position, use random direction
                collision_dir = np.array([1.0, 0.0])
            
            # Direction perpendicular to collision direction
            direction = np.array([collision_dir[1], -collision_dir[0]])
            
            # Emergency avoidance - push away from collision
            # 增加紧急避让的力度
            penetration = combined_radius - dist
            w = relative_velocity - collision_dir * penetration / (self.time_horizon * 0.5)  # 更短的时间范围
            point = self.velocity + 0.5 * w
            
            return (direction, point)
    
    def _linear_program(self, orca_lines: List[Tuple[np.ndarray, np.ndarray]], 
                       pref_velocity: np.ndarray) -> np.ndarray:
        """
        Solve linear program to find optimal velocity
        
        This is a simplified implementation using projection.
        Full ORCA uses a proper LP solver.
        """
        if len(orca_lines) == 0:
            # No constraints, return preferred velocity clipped to max speed
            return self._clip_velocity(pref_velocity)
        
        # Start with preferred velocity
        result = pref_velocity.copy()
        
        # Iteratively project onto half-planes
        for i, (direction, point) in enumerate(orca_lines):
            # Check if result violates this constraint
            # The constraint is: direction · (v - point) >= 0
            violation = np.dot(direction, result - point)
            if violation < 0:
                # Project result onto the half-plane boundary
                # The corrected velocity is: result - violation * direction
                result = result - violation * direction
        
        # Clip to maximum speed
        result = self._clip_velocity(result)
        
        return result
    
    def _project_onto_line(self, vector: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """
        Project vector onto line defined by direction
        """
        # Project onto the line parallel to direction
        parallel = np.dot(vector, direction) * direction
        return vector - parallel
    
    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Clip velocity to maximum speed
        """
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        return velocity


def compute_preferred_velocity(current_pos: np.ndarray, goal_pos: np.ndarray, 
                               max_speed: float) -> np.ndarray:
    """
    Compute preferred velocity towards goal
    
    Args:
        current_pos: Current position [x, y]
        goal_pos: Goal position [x, y]
        max_speed: Maximum speed
        
    Returns:
        Preferred velocity [vx, vy]
    """
    direction = goal_pos - current_pos
    distance = np.linalg.norm(direction)
    
    if distance < 1e-6:
        return np.zeros(2)
    
    # Move towards goal at max speed (or slower if close)
    desired_speed = min(max_speed, distance)
    return (direction / distance) * desired_speed
