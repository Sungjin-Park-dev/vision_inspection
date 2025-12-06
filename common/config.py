#!/usr/bin/env python3
"""
Central configuration file for Vision Inspection project

This file consolidates all configuration values used across the pipeline:
- mesh_to_viewpoints.py
- viewpoints_to_tsp.py
- run_app_v3.py
- coal_check.py

All values use SI units (meters) unless otherwise specified.
Coordinate system: Z-up (Isaac Sim / URDF / Pinocchio convention)
"""

import numpy as np
from pathlib import Path

# ============================================================================
# Project Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

# ============================================================================
# Camera Specifications
# ============================================================================

# Camera sensor
CAMERA_SENSOR_WIDTH_PX = 4096
CAMERA_SENSOR_HEIGHT_PX = 3000
CAMERA_PIXEL_SIZE_UM = 3.45

# Field of View (mm)
# CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0

# Working distance (mm) - distance from camera to object surface
CAMERA_WORKING_DISTANCE_MM = 110.0

# Depth of field (mm) - acceptable depth variation
CAMERA_DEPTH_OF_FIELD_MM = 0.5

# Overlap ratio between adjacent viewpoints (0.25 = 25% overlap)
CAMERA_OVERLAP_RATIO = 0.5


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix

    Args:
        quat: Quaternion in (w, x, y, z) format

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)


def get_camera_working_distance_m() -> float:
    """Get camera working distance in meters"""
    return CAMERA_WORKING_DISTANCE_MM / 1000.0


# ============================================================================
# World Configuration (Isaac Sim coordinates, meters)
# ============================================================================

# Glass object position in world frame (x, y, z)
# Glass -0.13
# Phone -0.17
# TV -0.145

GLASS_POSITION = np.array([1.00, 0.0, -0.172], dtype=np.float64)

# Glass object orientation in world frame (quaternion: w, x, y, z)
# Identity quaternion = no rotation
GLASS_ROTATION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Table cuboid position in world frame (x, y, z)
TABLE_POSITION = np.array([1.0, 0.0, -0.425], dtype=np.float64)

# Table cuboid dimensions (x, y, z) in meters
TABLE_DIMENSIONS = np.array([0.6, 1.0, 0.5], dtype=np.float64)

# Wall (Fence) cuboid position in world frame (x, y, z)
# Positioned behind the robot as a safety barrier
WALL_POSITION = np.array([-1.1, 0.0, 0.5], dtype=np.float64)

# Wall cuboid dimensions (x, y, z) in meters
# Thin wall (x=thickness, y=width, z=height)
WALL_DIMENSIONS = np.array([0.1, 2.2, 1.0], dtype=np.float64)

# Workbench cuboid position in world frame (x, y, z)
# Additional work surface next to main table
WORKBENCH_POSITION = np.array([0.35, -1.1, 0.5], dtype=np.float64)

# Workbench cuboid dimensions (x, y, z) in meters
WORKBENCH_DIMENSIONS = np.array([3.0, 0.1, 1.0], dtype=np.float64)

# Robot mount (base) cuboid position in world frame (x, y, z)
# Platform underneath the robot base
ROBOT_MOUNT_POSITION = np.array([0.0, 0.0, -0.25], dtype=np.float64)

# Robot mount cuboid dimensions (x, y, z) in meters
ROBOT_MOUNT_DIMENSIONS = np.array([0.3, 0.3, 0.5], dtype=np.float64)


# ============================================================================
# File Paths
# ============================================================================

# Mesh files (Z-up coordinate system)
DEFAULT_MESH_FILE = str(DATA_ROOT / "object" / "glass.obj")

# Robot files
DEFAULT_ROBOT_URDF = "ur_description/ur20.urdf"
DEFAULT_ROBOT_CONFIG = "ur20_safe.yml"
DEFAULT_ROBOT_CONFIG_YAML = "ur_description/ur20_safe.yml"  # For collision spheres

# Mesh base path for URDF collision meshes
MESH_BASE_PATH = "ur_description"


# ============================================================================
# Object-Based Data Path Helpers
# ============================================================================

def get_mesh_path(object_name: str, filename: str = "target.obj") -> Path:
    """
    Get path to object mesh file

    Args:
        object_name: Name of the object (e.g., "glass", "phone")
        filename: Mesh filename (default: "target.obj")

    Returns:
        Path to mesh file: data/{object_name}/mesh/{filename}

    Example:
        >>> get_mesh_path("glass")
        PosixPath('data/glass/mesh/target.obj')
    """
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


# ============================================================================
# Algorithm Parameters
# ============================================================================

# Trajectory interpolation
INTERPOLATION_STEPS = 60  # Number of steps between waypoints

# IK Solver
IK_NUM_SEEDS = 20  # Number of random seeds for IK solver
IK_ROTATION_THRESHOLD = 0.05  # Rotation error threshold (radians)
IK_POSITION_THRESHOLD = 0.005  # Position error threshold (meters)

# Collision checker cache sizes
N_OBSTACLE_CUBOIDS = 30  # Maximum number of cuboid obstacles for collision cache
N_OBSTACLE_MESH = 10  # Maximum number of mesh obstacles for collision cache

# Joint selection (for DP algorithm in run_app_v3.py)
JOINT_WEIGHTS = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
RECONFIGURATION_THRESHOLD = 1.0  # radians (deprecated, use RECONFIG_JOINT_THRESHOLD)
RECONFIGURATION_PENALTY = 10.0
MAX_MOVE_WEIGHT = 5.0

# Enhanced reconfiguration detection (4-condition logic)
RECONFIG_EE_FRAME_NAME = "tool0"  # End-effector frame for FK computation
RECONFIG_POSITION_THRESHOLD = 0.2  # meters - max EE position change
RECONFIG_ANGLE_THRESHOLD = 0.5  # radians (~28.6 degrees) - max EE z-axis angle change
RECONFIG_JOINT_THRESHOLD = 1.0  # radians - max single joint change
RECONFIG_DEVIATION_THRESHOLD = 0.1  # meters - max deviation from straight line
RECONFIG_INTERPOLATION_SAMPLES = 2  # number of interpolation samples to check
RECONFIG_FACTOR = 1.0  # global scaling factor for all thresholds

# Collision checking
COLLISION_MARGIN = 0.0  # Safety margin in meters (0 = no margin)
COLLISION_INTERP_STEPS = 10  # Interpolation steps for collision checking (CPU linear only)
COLLISION_USE_LINK_MESHES = False  # Use actual collision meshes from URDF
COLLISION_SHOW_LINK_DETAILS = False  # Show detailed link collision info
COLLISION_VERBOSE = True  # Print detailed progress
COLLISION_PARALLEL = True  # Enable parallel collision checking
COLLISION_NUM_WORKERS = None  # Number of workers for parallel (None = auto)
COLLISION_ADAPTIVE_INTERP = True  # Vary interpolation density per segment based on joint deltas
COLLISION_INTERP_EXCLUDE_LAST_JOINT = True  # Exclude last joint when computing max delta for interpolation

# interpolation할때 최소 단위
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 1.0  # Max joint delta (deg) allowed per interpolation gap
COLLISION_ADAPTIVE_LAST_JOINT_STEP_DEG = 5.0  # Max step for last joint (wrist_3) - less critical for collisions
COLLISION_ADAPTIVE_MIN_STEPS = 0  # Minimum interpolation steps per segment in adaptive mode
COLLISION_ADAPTIVE_MAX_STEPS = None  # Optional cap on interpolation steps per segment (None = no cap)

# Replanning parameters
REPLAN_ENABLED = True  # Attempt replanning for collisions/reconfigurations
REPLAN_TIMEOUT = 8.0  # Timeout in seconds for each planning query
REPLAN_MAX_ATTEMPTS = 3  # Maximum attempts for each planning request
REPLAN_INTERP_DT = 0.02  # Interpolation dt for trajectories
REPLAN_INTERP_STEPS = 5000  # Interpolation steps for trajectories
REPLAN_TRAJOPT_TSTEPS = 32  # Trajectory optimization timesteps


# ============================================================================
# TSP Parameters
# ============================================================================

# TSP algorithm defaults
TSP_NUM_STARTS = 10  # Number of random starting points for heuristics
TSP_MAX_2OPT_ITERATIONS = 100  # Max iterations for 2-opt refinement


# ============================================================================
# Viewpoint Sampling Parameters
# ============================================================================

# Auto-calculation defaults
VIEWPOINT_TARGET_COVERAGE = 1.0  # Target coverage ratio (1.0 = 100%)
VIEWPOINT_CURVATURE_FACTOR = 1.5  # Multiplier for high-curvature regions

# Voxel-based coverage calculation
VOXEL_SIZE_MM = 2.0  # Voxel size in mm for coverage analysis


# ============================================================================
# Coordinate System Notes
# ============================================================================
"""
COORDINATE SYSTEM UNIFICATION (Post-Refactoring):

All components now use Z-up coordinate system:
- Mesh files: glass_zup.obj (Z-up)
- Isaac Sim: Z-up (native)
- Pinocchio/URDF: Z-up (native)
- COAL collision checker: Z-up (native)

Surface → Camera transformation:
- Surface position: Point on mesh surface
- Surface normal: Outward-pointing normal vector (unit length)
- Camera position: surface_position + surface_normal * WORKING_DISTANCE
- Camera direction: -surface_normal (camera looks toward surface)

HDF5 file storage:
- Stores SURFACE positions (not camera positions)
- Camera position = surface + normal * working_distance
- This allows changing working distance without regenerating viewpoints
"""


