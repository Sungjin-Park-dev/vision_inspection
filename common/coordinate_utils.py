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


def compute_surface_to_camera_transform(
    surface_position: np.ndarray,
    surface_normal: np.ndarray,
    working_distance: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute camera position and direction from surface point and normal

    Args:
        surface_position: (3,) position on surface
        surface_normal: (3,) outward-pointing surface normal
        working_distance: Distance from surface to camera (meters)

    Returns:
        camera_position: (3,) camera position in space
        camera_direction: (3,) camera viewing direction (normalized)

    Examples:
        >>> surf_pos = np.array([0, 0, 0])
        >>> surf_normal = np.array([0, 0, 1])
        >>> cam_pos, cam_dir = compute_surface_to_camera_transform(surf_pos, surf_normal, 0.11)
        >>> cam_pos
        array([0.  , 0.  , 0.11])
        >>> cam_dir
        array([ 0.,  0., -1.])
    """
    # Camera position: offset from surface along normal
    camera_position = offset_points_along_normals(
        surface_position.reshape(1, 3),
        surface_normal.reshape(1, 3),
        working_distance
    ).flatten()

    # Camera direction: points back toward surface (negative of normal)
    camera_direction = -normalize_vectors(surface_normal.reshape(1, 3)).flatten()

    return camera_position, camera_direction


def build_pose_matrix(
    position: np.ndarray,
    direction: np.ndarray,
    up_hint: np.ndarray = None
) -> np.ndarray:
    """
    Build 4x4 pose matrix from position and viewing direction

    Creates an orthogonal coordinate frame with:
    - Z-axis aligned with direction vector
    - X-axis and Y-axis orthogonal to Z

    Args:
        position: (3,) position vector
        direction: (3,) direction vector (will be normalized to Z-axis)
        up_hint: (3,) optional hint for up direction (default: [0, 0, 1])

    Returns:
        pose_matrix: (4, 4) transformation matrix

    Examples:
        >>> pos = np.array([1, 2, 3])
        >>> direction = np.array([0, 0, 1])
        >>> pose = build_pose_matrix(pos, direction)
        >>> pose[:3, 3]  # Translation
        array([1., 2., 3.])
        >>> pose[:3, 2]  # Z-axis (direction)
        array([0., 0., 1.])
    """
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    # Normalize direction to get Z-axis
    z_axis = normalize_vectors(direction.reshape(1, 3)).flatten()

    # Choose helper vector for cross product
    # If Z-axis is nearly aligned with up_hint, use different helper
    if np.abs(np.dot(z_axis, up_hint)) > 0.99:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        helper = up_hint

    # Compute X-axis (orthogonal to Z)
    x_axis = np.cross(helper, z_axis)
    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-6:
        # If first cross product failed, try different helper
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x_axis = np.cross(helper, z_axis)
        x_norm = np.linalg.norm(x_axis)

        if x_norm < 1e-6:
            raise ValueError("Failed to construct orthogonal frame from direction vector")

    x_axis = x_axis / x_norm

    # Compute Y-axis (orthogonal to both X and Z)
    y_axis = np.cross(z_axis, x_axis)

    # Build 4x4 matrix
    pose_matrix = np.eye(4, dtype=np.float64)
    pose_matrix[:3, 0] = x_axis
    pose_matrix[:3, 1] = y_axis
    pose_matrix[:3, 2] = z_axis
    pose_matrix[:3, 3] = position

    return pose_matrix


def extract_pose_components(pose_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract position and rotation from 4x4 pose matrix

    Args:
        pose_matrix: (4, 4) transformation matrix

    Returns:
        position: (3,) position vector
        rotation: (3, 3) rotation matrix

    Raises:
        ValueError: If pose_matrix is not 4x4

    Examples:
        >>> pose = np.eye(4)
        >>> pose[:3, 3] = [1, 2, 3]
        >>> pos, rot = extract_pose_components(pose)
        >>> pos
        array([1., 2., 3.])
        >>> rot
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    if pose_matrix.shape != (4, 4):
        raise ValueError(f"Pose matrix must be 4x4, got {pose_matrix.shape}")

    position = pose_matrix[:3, 3]
    rotation = pose_matrix[:3, :3]

    return position.copy(), rotation.copy()


# ============================================================================
# Validation utilities
# ============================================================================

def validate_unit_vectors(vectors: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Check if vectors are unit length (normalized)

    Args:
        vectors: (N, 3) array of vectors
        tolerance: Acceptable deviation from unit length

    Returns:
        True if all vectors are unit length within tolerance

    Examples:
        >>> v = np.array([[1, 0, 0], [0, 1, 0]])
        >>> validate_unit_vectors(v)
        True
        >>> v = np.array([[2, 0, 0], [0, 1, 0]])
        >>> validate_unit_vectors(v)
        False
    """
    if vectors.size == 0:
        return True

    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return abs(norm - 1.0) < tolerance

    norms = np.linalg.norm(vectors, axis=1)
    return np.all(np.abs(norms - 1.0) < tolerance)


def validate_coordinate_range(
    points: np.ndarray,
    expected_range: Tuple[float, float] = (0.0, 2.0)
) -> bool:
    """
    Validate that point coordinates are within expected range

    Useful for detecting unit conversion errors (mm vs m)

    Args:
        points: (N, 3) array of points
        expected_range: (min, max) expected coordinate range in meters

    Returns:
        True if all coordinates are within range

    Examples:
        >>> points = np.array([[0.5, 0.5, 0.5]])
        >>> validate_coordinate_range(points, (0.0, 2.0))
        True
        >>> points = np.array([[500, 500, 500]])  # Likely in mm, not m
        >>> validate_coordinate_range(points, (0.0, 2.0))
        False
    """
    if points.size == 0:
        return True

    min_val, max_val = expected_range
    return np.all(points >= min_val) and np.all(points <= max_val)


if __name__ == "__main__":
    # Run doctests
    import doctest
    doctest.testmod()

    print("coordinate_utils.py: All doctests passed!")
