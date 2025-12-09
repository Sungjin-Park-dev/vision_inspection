#!/usr/bin/env python3
"""Camera configuration for vision inspection pipeline

All measurements in millimeters (mm) unless otherwise specified.
"""

import numpy as np
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# ============================================================================
# Camera Specifications
# ============================================================================

# Camera Field of View (mm)
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0

# Working distance (mm) - distance from camera to surface
CAMERA_WORKING_DISTANCE_MM = 110.0

# Overlap ratio between adjacent viewpoints (0.5 = 50% overlap)
CAMERA_OVERLAP_RATIO = 0.5


# ============================================================================
# World Configuration (Isaac Sim coordinates, meters)
# ============================================================================

# Target object position in world frame (x, y, z)
TARGET_OBJECT_POSITION = np.array([1.00, 0.0, -0.172], dtype=np.float64)

# Target object orientation in world frame (quaternion: w, x, y, z)
TARGET_OBJECT_ROTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Table cuboid position in world frame (x, y, z)
TABLE_POSITION = np.array([1.0, 0.0, -0.425], dtype=np.float64)

# Table cuboid dimensions (x, y, z) in meters
TABLE_DIMENSIONS = np.array([0.6, 1.0, 0.5], dtype=np.float64)

# Wall (Fence) cuboid position in world frame (x, y, z)
WALL_POSITION = np.array([-1.1, 0.0, 0.5], dtype=np.float64)

# Wall cuboid dimensions (x, y, z) in meters
WALL_DIMENSIONS = np.array([0.1, 2.2, 1.0], dtype=np.float64)

# Workbench cuboid position in world frame (x, y, z)
WORKBENCH_POSITION = np.array([0.35, -1.1, 0.5], dtype=np.float64)

# Workbench cuboid dimensions (x, y, z) in meters
WORKBENCH_DIMENSIONS = np.array([3.0, 0.1, 1.0], dtype=np.float64)

# Robot mount (base) cuboid position in world frame (x, y, z)
ROBOT_MOUNT_POSITION = np.array([0.0, 0.0, -0.25], dtype=np.float64)

# Robot mount cuboid dimensions (x, y, z) in meters
ROBOT_MOUNT_DIMENSIONS = np.array([0.3, 0.3, 0.5], dtype=np.float64)


# ============================================================================
# Robot Configuration
# ============================================================================

DEFAULT_ROBOT_CONFIG = "ur20.yml"
DEFAULT_URDF_PATH = "/isaac-sim/curobo/src/curobo/content/assets/robot/ur_description/ur20.urdf"


# ============================================================================
# IK Solver Parameters
# ============================================================================

IK_ROTATION_THRESHOLD = 0.05  # radians
IK_POSITION_THRESHOLD = 0.005  # meters
IK_NUM_SEEDS = 32


# ============================================================================
# Collision Checking Parameters
# ============================================================================

COLLISION_MARGIN = 0.0
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 1.0
COLLISION_INTERP_EXCLUDE_LAST_JOINT = True


# ============================================================================
# Replanning Parameters
# ============================================================================

REPLAN_ENABLED = True
REPLAN_MAX_ATTEMPTS = 60
REPLAN_TIMEOUT = 10.0  # seconds
REPLAN_INTERP_DT = 0.01
REPLAN_TRAJOPT_TSTEPS = 32


# ============================================================================
# Object-Based Data Path Helpers
# ============================================================================

def get_mesh_path(object_name: str, filename: str = None, mesh_type: str = "target") -> Path:
    """
    Get path to object mesh file

    Args:
        object_name: Name of the object (e.g., "glass", "phone")
        filename: Explicit mesh filename (overrides mesh_type if provided)
        mesh_type: Type of mesh file (default: "target")
            - "source": source.obj (full multi-material mesh for collision checking)
            - "target": target.ply (inspection surface for viewpoint sampling)

    Returns:
        Path to mesh file: data/{object_name}/mesh/{filename}

    Examples:
        >>> get_mesh_path("glass")  # Default: target mesh
        PosixPath('data/glass/mesh/target.ply')  # or target.obj if .ply doesn't exist

        >>> get_mesh_path("glass", mesh_type="source")  # Full mesh for collision
        PosixPath('data/glass/mesh/source.obj')

        >>> get_mesh_path("glass", filename="custom.obj")  # Explicit filename
        PosixPath('data/glass/mesh/custom.obj')
    """
    if filename is None:
        # Auto-determine filename based on mesh_type
        if mesh_type == "source":
            filename = "source.obj"
        elif mesh_type == "target":
            # Try target.ply first (preferred for inspection), fallback to target.obj
            target_ply = DATA_ROOT / object_name / "mesh" / "target.ply"
            if target_ply.exists():
                return target_ply
            filename = "target.obj"
        else:
            raise ValueError(f"Invalid mesh_type: '{mesh_type}'. Must be 'source' or 'target'")

    return DATA_ROOT / object_name / "mesh" / filename


def get_viewpoint_path(object_name: str, num_viewpoints: int, filename: str = "viewpoints.h5") -> Path:
    """
    Get path to viewpoints file

    Args:
        object_name: Name of the object (e.g., "glass")
        num_viewpoints: Number of viewpoints
        filename: Filename (default: "viewpoints.h5")

    Returns:
        Path to viewpoints: data/{object_name}/viewpoint/{num_viewpoints}/{filename}

    Example:
        >>> get_viewpoint_path("glass", 500)
        PosixPath('data/glass/viewpoint/500/viewpoints.h5')
    """
    return DATA_ROOT / object_name / "viewpoint" / str(num_viewpoints) / filename


def get_ik_path(object_name: str, num_viewpoints: int, filename: str = "ik_solutions.h5") -> Path:
    """
    Get path to IK solutions file

    Args:
        object_name: Name of the object (e.g., "glass")
        num_viewpoints: Number of viewpoints
        filename: Filename (default: "ik_solutions.h5")

    Returns:
        Path to IK solutions: data/{object_name}/ik/{num_viewpoints}/{filename}

    Example:
        >>> get_ik_path("glass", 500)
        PosixPath('data/glass/ik/500/ik_solutions.h5')
    """
    return DATA_ROOT / object_name / "ik" / str(num_viewpoints) / filename


def get_trajectory_path(object_name: str, num_viewpoints: int, filename: str = "gtsp.csv") -> Path:
    """
    Get path to trajectory file

    Args:
        object_name: Name of the object (e.g., "glass")
        num_viewpoints: Number of viewpoints
        filename: Filename (default: "gtsp.csv", can also be "gtsp_final.csv")

    Returns:
        Path to trajectory: data/{object_name}/trajectory/{num_viewpoints}/{filename}

    Example:
        >>> get_trajectory_path("glass", 500)
        PosixPath('data/glass/trajectory/500/gtsp.csv')
        >>> get_trajectory_path("glass", 500, "gtsp_final.csv")
        PosixPath('data/glass/trajectory/500/gtsp_final.csv')
    """
    return DATA_ROOT / object_name / "trajectory" / str(num_viewpoints) / filename
