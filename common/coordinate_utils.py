#!/usr/bin/env python3
"""
Coordinate and geometry utilities for Vision Inspection project

This module provides common geometric operations used across the pipeline,
particularly for viewpoint generation and coordinate transformations.

All functions assume Z-up coordinate system (Isaac Sim / URDF / Pinocchio convention).
"""

import numpy as np
from typing import Tuple


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length

    Args:
        vectors: (N, 3) array of vectors

    Returns:
        normalized: (N, 3) array of unit vectors

    Examples:
        >>> v = np.array([[1, 0, 0], [3, 4, 0]])
        >>> normalize_vectors(v)
        array([[1., 0., 0.],
               [0.6, 0.8, 0.]])
    """
    if vectors.size == 0:
        return vectors

    # Handle both 1D and 2D arrays
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / np.maximum(norm, 1e-9)

    # 2D array: normalize each row
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)  # Avoid division by zero
    return vectors / norms


def offset_points_along_normals(
    points: np.ndarray,
    normals: np.ndarray,
    offset: float
) -> np.ndarray:
    """
    Offset points along their normals by a given distance

    This is commonly used to compute camera positions from surface points:
    camera_position = surface_position + surface_normal * working_distance

    Args:
        points: (N, 3) array of 3D points
        normals: (N, 3) array of normal vectors (will be normalized)
        offset: Distance to offset along normals (in meters)

    Returns:
        offset_points: (N, 3) array of offset points

    Raises:
        ValueError: If points and normals have different shapes

    Examples:
        >>> points = np.array([[0, 0, 0], [1, 0, 0]])
        >>> normals = np.array([[0, 0, 1], [0, 0, 1]])
        >>> offset_points_along_normals(points, normals, 0.1)
        array([[0. , 0. , 0.1],
               [1. , 0. , 0.1]])
    """
    if points.size == 0:
        return points

    if points.shape != normals.shape:
        raise ValueError(
            f"Points and normals must have the same shape. "
            f"Got points: {points.shape}, normals: {normals.shape}"
        )

    # Ensure normals are unit vectors
    safe_normals = normalize_vectors(normals)

    return points + safe_normals * offset


# ============================================================================
# Pose Transformations
# ============================================================================

def transform_pose_to_world(
    local_pose: np.ndarray,
    object_world_pose: np.ndarray,
    debug: bool = False
) -> np.ndarray:
    """Transform local pose to world frame

    Transforms a pose defined in an object's local coordinate frame to the
    world coordinate frame using the object's world pose.

    Args:
        local_pose: 4x4 pose matrix in object's local frame
        object_world_pose: 4x4 transformation of object in world frame
        debug: Print debug information

    Returns:
        4x4 pose matrix in world frame

    Raises:
        ValueError: If input matrices are not 4x4

    Example:
        >>> # Object at position (1, 0, 0)
        >>> object_world = np.eye(4)
        >>> object_world[:3, 3] = [1, 0, 0]
        >>> # Local pose at (0.1, 0, 0) relative to object
        >>> local = np.eye(4)
        >>> local[:3, 3] = [0.1, 0, 0]
        >>> # Result should be at (1.1, 0, 0) in world
        >>> world = transform_pose_to_world(local, object_world)
        >>> np.allclose(world[:3, 3], [1.1, 0, 0])
        True
    """
    if local_pose.shape != (4, 4):
        raise ValueError("local_pose must be 4x4")
    if object_world_pose.shape != (4, 4):
        raise ValueError("object_world_pose must be 4x4")

    if debug:
        print(f"\n=== Coordinate Transform Debug ===")
        print(f"Object world pose:\n{object_world_pose}")
        print(f"Local pose (Z-up):\n{local_pose}")

    # Simple matrix multiplication: world_pose = object_world_pose @ local_pose
    world_pose = object_world_pose @ local_pose

    if debug:
        print(f"World pose result:\n{world_pose}")
        print(f"===================================\n")

    return world_pose


if __name__ == "__main__":
    # Run doctests
    import doctest
    doctest.testmod()

    print("coordinate_utils.py: All doctests passed!")
