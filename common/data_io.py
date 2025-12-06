#!/usr/bin/env python3
"""
Data I/O Utilities for Viewpoints, Trajectories, and IK Solutions

Provides unified functions for loading and saving robot data in CSV and HDF5 formats.
Consolidates I/O operations from multiple modules for consistency and maintainability.

Functions consolidated from:
- trajectory_io.py: Viewpoint and trajectory I/O
- compute_ik_solutions.py: IK solutions I/O
- fk_gtsp_gpu_claude2.py: HDF5 cluster loading
"""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import h5py
import numpy as np
import pandas as pd


# ============================================================================
# Viewpoint I/O (HDF5)
# ============================================================================

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


# ============================================================================
# Trajectory I/O (CSV)
# ============================================================================

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


# ============================================================================
# IK Solutions I/O (HDF5)
# ============================================================================

def save_ik_solutions_hdf5(
    viewpoints: List,
    save_path: str,
    viewpoints_path: str
):
    """
    Save all IK solutions to HDF5 file

    Args:
        viewpoints: List of Viewpoint objects with IK solutions
        save_path: Path to output HDF5 file
        viewpoints_path: Path to original viewpoints file (for metadata)

    Note:
        Viewpoint objects must have attributes:
        - index: int
        - world_pose: (4,4) array or None
        - all_ik_solutions: List of joint configurations
        - safe_ik_solutions: List of collision-free joint configurations
    """
    num_with_solutions = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    num_with_safe = sum(1 for vp in viewpoints if len(vp.safe_ik_solutions) > 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(viewpoints)
        metadata_grp.attrs['num_viewpoints_with_solutions'] = num_with_solutions
        metadata_grp.attrs['num_viewpoints_with_safe_solutions'] = num_with_safe
        metadata_grp.attrs['timestamp'] = datetime.now().isoformat()
        metadata_grp.attrs['viewpoints_file'] = viewpoints_path

        for vp in viewpoints:
            vp_grp_name = f'viewpoint_{vp.index:04d}'
            vp_grp = f.create_group(vp_grp_name)
            vp_grp.attrs['original_index'] = vp.index

            if vp.world_pose is not None:
                vp_grp.create_dataset('world_pose', data=vp.world_pose.astype(np.float32))
            else:
                vp_grp.create_dataset('world_pose', data=np.zeros((4, 4), dtype=np.float32))

            if len(vp.all_ik_solutions) > 0:
                all_sols = np.stack([np.asarray(sol, dtype=np.float64)
                                    for sol in vp.all_ik_solutions])
                vp_grp.create_dataset('all_ik_solutions', data=all_sols.astype(np.float32))
            else:
                vp_grp.create_dataset('all_ik_solutions', data=np.zeros((0, 6), dtype=np.float32))

            collision_free_mask = np.zeros(len(vp.all_ik_solutions), dtype=bool)
            for i, sol in enumerate(vp.all_ik_solutions):
                sol_array = np.asarray(sol, dtype=np.float64)
                for safe_sol in vp.safe_ik_solutions:
                    if np.allclose(sol_array, safe_sol, atol=1e-6):
                        collision_free_mask[i] = True
                        break
            vp_grp.create_dataset('collision_free_mask', data=collision_free_mask)

            vp_grp.attrs['num_all_solutions'] = len(vp.all_ik_solutions)
            vp_grp.attrs['num_safe_solutions'] = len(vp.safe_ik_solutions)

    print(f"\n{'='*60}")
    print("IK SOLUTIONS SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total viewpoints: {len(viewpoints)}")
    print(f"With any solutions: {num_with_solutions}")
    print(f"With safe solutions: {num_with_safe}")
    print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"{'='*60}\n")


def build_clusters_from_h5(
    h5_path: str,
    use_safe_only: bool,
    tool_z: float,
    group_prefix: str = "viewpoint_",
) -> Tuple[List[Dict], np.ndarray, List[int]]:
    """
    Build clusters from IK solutions HDF5 file for GTSP trajectory optimization

    Args:
        h5_path: Path to IK solutions HDF5 file
        use_safe_only: If True, use only collision-free IK solutions
        tool_z: Tool length offset in meters
        group_prefix: Prefix for viewpoint groups in HDF5

    Returns:
        clusters: List of cluster dictionaries, each containing:
            - "q": (S, DOF) joint configurations
            - "R": (S, 3, 3) end-effector rotations from FK
            - "p": (S, 3) end-effector positions from FK
            - "Q": (S, 4) end-effector quaternions from FK
            - "target": (3,) original camera position (from world_pose)
            - "target_Q": (4,) original camera orientation quaternion
        target_coords: (M, 3) array of target camera positions
        nonempty_map: List mapping original viewpoint index to cluster index (-1 if empty)

    Note:
        Requires kinematics_utils for FK and quaternion conversions.
    """
    # Import kinematics utilities
    from common.kinematics_utils import fk_batch, rot_to_quat_batch

    clusters: List[Dict] = []
    target_coords_list = []
    nonempty_map: List[int] = []

    with h5py.File(h5_path, "r") as f:
        # List viewpoint_* groups (sorted)
        keys = sorted([k for k in f.keys() if k.startswith(group_prefix)])

        for gi, gname in enumerate(keys):
            g = f[gname]
            # Required: world_pose(4x4), all_ik_solutions(S,DOF)
            if "world_pose" not in g or "all_ik_solutions" not in g:
                nonempty_map.append(-1)
                continue

            world_pose = np.array(g["world_pose"], dtype=np.float64)
            target = world_pose[:3, 3].astype(np.float64)

            # Extract original camera orientation (rotation matrix → quaternion)
            target_R = world_pose[:3, :3].astype(np.float64)
            target_Q = rot_to_quat_batch(target_R[None, ...])[0]  # (1,3,3) → (1,4) → (4,)

            q_all = np.array(g["all_ik_solutions"], dtype=np.float64)  # (S, DOF)
            if q_all.ndim != 2 or q_all.shape[0] == 0:
                # No IK solutions
                nonempty_map.append(-1)
                continue

            # Filter by collision-free mask if requested
            if use_safe_only and "collision_free_mask" in g:
                m = np.array(g["collision_free_mask"], dtype=bool).reshape(-1)
                if m.shape[0] == q_all.shape[0]:
                    q_all = q_all[m]
                    if q_all.shape[0] == 0:
                        nonempty_map.append(-1)
                        continue

            # Pre-compute end-effector FK
            R, p = fk_batch(q_all, tool_z)
            Q = rot_to_quat_batch(R)

            clusters.append({
                "q": q_all,           # (S, DOF)
                "R": R,               # (S, 3, 3)
                "p": p,               # (S, 3)
                "Q": Q,               # (S, 4)
                "target": target,     # (3,) - original camera position
                "target_Q": target_Q, # (4,) - original camera orientation (w,x,y,z)
            })
            target_coords_list.append(target)
            nonempty_map.append(len(clusters) - 1)

    if len(clusters) == 0:
        raise RuntimeError("No non-empty viewpoints found in HDF5.")

    target_coords = np.stack(target_coords_list, axis=0)  # (M, 3)
    return clusters, target_coords, nonempty_map
