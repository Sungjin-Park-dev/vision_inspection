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

if __name__ == "__main__":
    # Run doctests
    import doctest
    doctest.testmod()

    print("interpolation_utils.py: All doctests passed!")
