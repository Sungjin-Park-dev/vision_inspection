#!/usr/bin/env python3
"""
Trajectory interpolation utilities for Vision Inspection project

This module provides interpolation functions for generating smooth trajectories
between waypoints. Used by both run_app_v3.py (simulation) and coal_check.py
(collision validation).

All interpolation is linear in joint space.
"""

import numpy as np
from typing import List, Union


def generate_interpolated_path(
    start: np.ndarray,
    end: np.ndarray,
    num_steps: int
) -> List[np.ndarray]:
    """
    Generate linear interpolation between two joint configurations

    Creates intermediate waypoints between start and end configurations.
    The returned path does NOT include the start configuration, but moves
    progressively toward the end configuration.

    Args:
        start: Starting configuration (typically 6 joints for UR robot)
        end: Ending configuration
        num_steps: Number of intermediate steps to generate

    Returns:
        path: List of interpolated configurations (does not include start, ends at end)

    Examples:
        >>> start = np.array([0, 0, 0, 0, 0, 0])
        >>> end = np.array([1, 1, 1, 1, 1, 1])
        >>> path = generate_interpolated_path(start, end, 2)
        >>> len(path)
        2
        >>> np.allclose(path[0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        True
        >>> np.allclose(path[1], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        True
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)

    if start.shape != end.shape:
        raise ValueError(
            f"Start and end configurations must have the same shape. "
            f"Got start: {start.shape}, end: {end.shape}"
        )

    if num_steps <= 0:
        return [end]

    # Generate interpolation parameters from 0.0 to 1.0, excluding 0.0
    # This gives us num_steps+1 values [0, α₁, α₂, ..., αₙ, 1], then we take [1:]
    alphas = np.linspace(0.0, 1.0, num_steps + 1, endpoint=True)[1:]

    # Linear interpolation: q(α) = start + α * (end - start)
    path = [start + alpha * (end - start) for alpha in alphas]

    return path


def generate_multi_segment_path(
    waypoints: List[np.ndarray],
    steps_per_segment: int
) -> List[np.ndarray]:
    """
    Generate interpolated path through multiple waypoints

    Args:
        waypoints: List of configurations to visit in order
        steps_per_segment: Number of interpolation steps between each pair

    Returns:
        path: Complete interpolated path through all waypoints

    Examples:
        >>> wp1 = np.array([0, 0, 0])
        >>> wp2 = np.array([1, 1, 1])
        >>> wp3 = np.array([2, 2, 2])
        >>> path = generate_multi_segment_path([wp1, wp2, wp3], 1)
        >>> len(path)
        2  # wp2 and wp3 (wp1 is start, not included)
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least 2 waypoints to generate a path")

    full_path = []

    for i in range(len(waypoints) - 1):
        segment = generate_interpolated_path(
            waypoints[i],
            waypoints[i + 1],
            steps_per_segment
        )
        full_path.extend(segment)

    return full_path


def compute_path_length(
    configurations: List[np.ndarray],
    weights: np.ndarray = None
) -> float:
    """
    Compute total path length in configuration space

    Args:
        configurations: List of joint configurations
        weights: Optional weights for each joint dimension

    Returns:
        total_length: Sum of Euclidean distances between consecutive configs

    Examples:
        >>> configs = [np.array([0, 0]), np.array([1, 0]), np.array([1, 1])]
        >>> compute_path_length(configs)
        2.0
        >>> compute_path_length(configs, weights=np.array([2.0, 1.0]))
        3.0
    """
    if len(configurations) < 2:
        return 0.0

    configs = [np.asarray(c, dtype=np.float64) for c in configurations]

    if weights is None:
        weights = np.ones(configs[0].shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    total_length = 0.0
    for i in range(len(configs) - 1):
        diff = configs[i + 1] - configs[i]
        weighted_diff = weights * diff ** 2
        distance = np.sqrt(np.sum(weighted_diff))
        total_length += distance

    return total_length


def validate_trajectory_continuity(
    configurations: List[np.ndarray],
    max_step: float = None
) -> bool:
    """
    Check if trajectory has no discontinuous jumps

    Args:
        configurations: List of joint configurations
        max_step: Maximum allowed step size (optional)

    Returns:
        True if trajectory is continuous (no large jumps)

    Examples:
        >>> configs = [np.array([0, 0]), np.array([0.1, 0.1]), np.array([0.2, 0.2])]
        >>> validate_trajectory_continuity(configs, max_step=0.2)
        True
        >>> configs = [np.array([0, 0]), np.array([10, 10])]
        >>> validate_trajectory_continuity(configs, max_step=0.5)
        False
    """
    if len(configurations) < 2:
        return True

    configs = [np.asarray(c, dtype=np.float64) for c in configurations]

    for i in range(len(configs) - 1):
        diff = configs[i + 1] - configs[i]
        step_size = np.linalg.norm(diff)

        if max_step is not None and step_size > max_step:
            return False

    return True


def resample_trajectory(
    configurations: List[np.ndarray],
    target_num_points: int
) -> List[np.ndarray]:
    """
    Resample trajectory to have a specific number of points

    Uses linear interpolation to generate a uniformly sampled trajectory.

    Args:
        configurations: Original trajectory
        target_num_points: Desired number of points in resampled trajectory

    Returns:
        resampled: Trajectory with target_num_points configurations

    Raises:
        ValueError: If configurations is empty or target_num_points < 2

    Examples:
        >>> configs = [np.array([0.0]), np.array([1.0])]
        >>> resampled = resample_trajectory(configs, 5)
        >>> len(resampled)
        5
        >>> np.allclose(resampled[0], [0.0])
        True
        >>> np.allclose(resampled[-1], [1.0])
        True
    """
    if len(configurations) == 0:
        raise ValueError("Cannot resample empty trajectory")

    if target_num_points < 2:
        raise ValueError("target_num_points must be at least 2")

    configs = [np.asarray(c, dtype=np.float64) for c in configurations]

    if len(configs) == target_num_points:
        return configs

    # Compute cumulative arc length along original trajectory
    arc_lengths = [0.0]
    for i in range(len(configs) - 1):
        diff = configs[i + 1] - configs[i]
        distance = np.linalg.norm(diff)
        arc_lengths.append(arc_lengths[-1] + distance)

    total_length = arc_lengths[-1]

    # Generate target arc lengths for resampled points
    target_lengths = np.linspace(0, total_length, target_num_points)

    # Interpolate configurations at target arc lengths
    resampled = []
    for target_s in target_lengths:
        # Find the segment containing this arc length
        for i in range(len(arc_lengths) - 1):
            if arc_lengths[i] <= target_s <= arc_lengths[i + 1]:
                # Interpolate within this segment
                segment_length = arc_lengths[i + 1] - arc_lengths[i]
                if segment_length > 0:
                    alpha = (target_s - arc_lengths[i]) / segment_length
                else:
                    alpha = 0.0

                config = configs[i] + alpha * (configs[i + 1] - configs[i])
                resampled.append(config)
                break
        else:
            # Target length is at or beyond the end
            resampled.append(configs[-1].copy())

    return resampled


if __name__ == "__main__":
    # Run doctests
    import doctest
    doctest.testmod()

    print("interpolation_utils.py: All doctests passed!")
