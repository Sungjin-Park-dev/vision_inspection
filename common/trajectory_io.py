#!/usr/bin/env python3
"""
Trajectory I/O Utilities for CSV and HDF5 Files

Provides unified functions for loading and saving robot trajectories
in CSV and HDF5 formats with consistent error handling and validation.
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import pandas as pd


def load_trajectory_csv(
    csv_path: str,
    joint_prefix: Optional[str] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Load joint trajectory from CSV file

    Args:
        csv_path: Path to CSV file
        joint_prefix: Optional prefix to filter joint columns (e.g., "ur20-")
                     If None, uses all columns with "joint" in name

    Returns:
        trajectory: (N, n_joints) array of joint angles
        joint_names: List of joint column names

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If no joint columns found or invalid data

    Example:
        >>> trajectory, joint_names = load_trajectory_csv(
        ...     "data/trajectory/path.csv",
        ...     joint_prefix="ur20-"
        ... )
        >>> print(trajectory.shape)
        (150, 6)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Trajectory file not found: {csv_path}")

    joint_angles = []
    joint_names = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if headers is None:
            raise ValueError(f"CSV file has no headers: {csv_path}")

        # Extract joint column names
        if joint_prefix:
            joint_names = [h for h in headers if h.startswith(joint_prefix)]
        else:
            joint_names = [h for h in headers if 'joint' in h.lower()]

        if not joint_names:
            raise ValueError(
                f"No joint columns found in CSV. Headers: {headers}\n"
                f"Use joint_prefix parameter to specify joint column prefix."
            )

        # Read trajectory data
        for row in reader:
            try:
                config = [float(row[joint_name]) for joint_name in joint_names]
                joint_angles.append(config)
            except (ValueError, KeyError) as e:
                raise ValueError(
                    f"Error parsing row in {csv_path}: {e}\n"
                    f"Joint names: {joint_names}"
                )

    if not joint_angles:
        raise ValueError(f"No trajectory data found in {csv_path}")

    trajectory = np.array(joint_angles, dtype=np.float64)

    print(f"Loaded trajectory: {len(trajectory)} waypoints, {len(joint_names)} joints")
    print(f"Joint names: {joint_names}")

    return trajectory, joint_names


def save_trajectory_csv(
    trajectory: np.ndarray,
    output_path: str,
    joint_names: Optional[List[str]] = None,
    include_time: bool = True
) -> Path:
    """
    Save trajectory to CSV file

    Args:
        trajectory: (N, n_joints) trajectory array
        output_path: Path to output CSV file
        joint_names: List of joint names (default: joint_0, joint_1, ...)
        include_time: Add time column with sequential values (default: True)

    Returns:
        Path to saved file

    Raises:
        ValueError: If trajectory shape is invalid

    Example:
        >>> trajectory = np.random.rand(100, 6)
        >>> save_trajectory_csv(
        ...     trajectory,
        ...     "output.csv",
        ...     joint_names=["joint1", "joint2", ...]
        ... )
    """
    if trajectory.ndim != 2:
        raise ValueError(
            f"Trajectory must be 2D array (N, n_joints), got shape {trajectory.shape}"
        )

    n_waypoints, n_joints = trajectory.shape

    # Generate joint names if not provided
    if joint_names is None:
        joint_names = [f"joint_{i}" for i in range(n_joints)]
    elif len(joint_names) != n_joints:
        raise ValueError(
            f"Number of joint names ({len(joint_names)}) must match "
            f"number of joints ({n_joints})"
        )

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    df = pd.DataFrame(trajectory, columns=joint_names)

    # Add time column if requested
    if include_time:
        time_values = np.arange(n_waypoints, dtype=np.float64)
        df.insert(0, 'time', time_values)

    # Save to CSV
    df.to_csv(output_path, index=False)

    print(f"Saved trajectory: {output_path}")
    print(f"  Waypoints: {n_waypoints}, Joints: {n_joints}")

    return output_path


def load_viewpoints_hdf5(
    hdf5_path: str
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load viewpoints from HDF5 file

    Args:
        hdf5_path: Path to HDF5 file

    Returns:
        positions: (N, 3) array of surface positions in meters
        normals: (N, 3) array of surface normals (unit vectors)
        metadata: Dictionary containing metadata and camera_spec

    Raises:
        FileNotFoundError: If HDF5 file doesn't exist
        ValueError: If file format is invalid

    Example:
        >>> positions, normals, metadata = load_viewpoints_hdf5(
        ...     "data/viewpoint/100/viewpoints.h5"
        ... )
        >>> print(positions.shape)
        (100, 3)
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Viewpoints file not found: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        # Load required datasets
        if 'viewpoints' not in f:
            raise ValueError(
                f"Invalid HDF5 format: missing 'viewpoints' group in {hdf5_path}"
            )

        viewpoints_grp = f['viewpoints']

        if 'positions' not in viewpoints_grp:
            raise ValueError(
                f"Invalid HDF5 format: missing 'positions' dataset in {hdf5_path}"
            )
        if 'normals' not in viewpoints_grp:
            raise ValueError(
                f"Invalid HDF5 format: missing 'normals' dataset in {hdf5_path}"
            )

        positions = np.array(viewpoints_grp['positions'], dtype=np.float64)
        normals = np.array(viewpoints_grp['normals'], dtype=np.float64)

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata_grp = f['metadata']
            # Load all metadata attributes
            for key in metadata_grp.attrs:
                metadata[key] = metadata_grp.attrs[key]

            # Load camera_spec if present
            if 'camera_spec' in metadata_grp:
                camera_spec = {}
                for key in metadata_grp['camera_spec'].attrs:
                    camera_spec[key] = metadata_grp['camera_spec'].attrs[key]
                metadata['camera_spec'] = camera_spec

    print(f"Loaded {len(positions)} viewpoints from {hdf5_path}")

    return positions, normals, metadata


def save_viewpoints_hdf5(
    positions: np.ndarray,
    normals: np.ndarray,
    output_path: str,
    metadata: Optional[dict] = None,
    camera_spec: Optional[dict] = None
) -> Path:
    """
    Save viewpoints to HDF5 file

    Args:
        positions: (N, 3) array of surface positions in meters
        normals: (N, 3) array of surface normals (unit vectors)
        output_path: Path to output HDF5 file
        metadata: Optional metadata dictionary
        camera_spec: Optional camera specification dictionary

    Returns:
        Path to saved file

    Raises:
        ValueError: If positions/normals have invalid shapes

    Example:
        >>> save_viewpoints_hdf5(
        ...     positions,
        ...     normals,
        ...     "data/viewpoint/100/viewpoints.h5",
        ...     metadata={"num_viewpoints": 100},
        ...     camera_spec={"fov_width_mm": 10.6}
        ... )
    """
    if positions.shape != normals.shape:
        raise ValueError(
            f"Positions and normals must have same shape, "
            f"got {positions.shape} and {normals.shape}"
        )
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"Positions must be (N, 3) array, got shape {positions.shape}"
        )

    # Create output directory if needed
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Create viewpoints group
        viewpoints_grp = f.create_group('viewpoints')
        viewpoints_grp.create_dataset('positions', data=positions.astype(np.float32))
        viewpoints_grp.create_dataset('normals', data=normals.astype(np.float32))

        # Create metadata group
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(positions)

        # Add additional metadata
        if metadata:
            for key, value in metadata.items():
                if key != 'camera_spec':  # Handle separately
                    metadata_grp.attrs[key] = value

        # Add camera spec
        if camera_spec:
            camera_spec_grp = metadata_grp.create_group('camera_spec')
            for key, value in camera_spec.items():
                camera_spec_grp.attrs[key] = value

    print(f"Saved {len(positions)} viewpoints to {output_path}")

    return output_path


def validate_trajectory(
    trajectory: np.ndarray,
    joint_limits: Optional[np.ndarray] = None,
    joint_names: Optional[List[str]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate trajectory for common issues

    Args:
        trajectory: (N, n_joints) trajectory array
        joint_limits: Optional (n_joints, 2) array of [min, max] limits
        joint_names: Optional list of joint names for error messages

    Returns:
        is_valid: True if trajectory is valid
        errors: List of error messages (empty if valid)

    Example:
        >>> is_valid, errors = validate_trajectory(trajectory)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"Validation error: {error}")
    """
    errors = []

    # Check shape
    if trajectory.ndim != 2:
        errors.append(f"Trajectory must be 2D array, got shape {trajectory.shape}")
        return False, errors

    n_waypoints, n_joints = trajectory.shape

    if n_waypoints == 0:
        errors.append("Trajectory is empty (0 waypoints)")

    if n_joints == 0:
        errors.append("Trajectory has 0 joints")

    # Check for NaN or Inf
    if np.any(np.isnan(trajectory)):
        errors.append("Trajectory contains NaN values")

    if np.any(np.isinf(trajectory)):
        errors.append("Trajectory contains Inf values")

    # Check joint limits if provided
    if joint_limits is not None:
        if joint_limits.shape != (n_joints, 2):
            errors.append(
                f"Joint limits shape must be ({n_joints}, 2), "
                f"got {joint_limits.shape}"
            )
        else:
            for i in range(n_joints):
                joint_min, joint_max = joint_limits[i]
                violations = np.where(
                    (trajectory[:, i] < joint_min) | (trajectory[:, i] > joint_max)
                )[0]

                if len(violations) > 0:
                    joint_name = joint_names[i] if joint_names else f"joint_{i}"
                    errors.append(
                        f"{joint_name} exceeds limits [{joint_min}, {joint_max}] "
                        f"at {len(violations)} waypoint(s)"
                    )

    is_valid = len(errors) == 0
    return is_valid, errors
