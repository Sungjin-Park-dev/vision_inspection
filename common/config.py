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
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0

# Working distance (mm) - distance from camera to object surface
CAMERA_WORKING_DISTANCE_MM = 110.0

# Depth of field (mm) - acceptable depth variation
CAMERA_DEPTH_OF_FIELD_MM = 0.5

# Overlap ratio between adjacent viewpoints (0.25 = 25% overlap)
CAMERA_OVERLAP_RATIO = 0.25


def get_camera_working_distance_m() -> float:
    """Get camera working distance in meters"""
    return CAMERA_WORKING_DISTANCE_MM / 1000.0


def get_camera_fov_m() -> tuple:
    """Get camera FOV in meters (width, height)"""
    return CAMERA_FOV_WIDTH_MM / 1000.0, CAMERA_FOV_HEIGHT_MM / 1000.0


def get_camera_dof_m() -> float:
    """Get camera depth of field in meters"""
    return CAMERA_DEPTH_OF_FIELD_MM / 1000.0


def get_effective_coverage_mm() -> tuple:
    """Get effective coverage per viewpoint considering overlap (width, height in mm)"""
    effective_width = CAMERA_FOV_WIDTH_MM * (1.0 - CAMERA_OVERLAP_RATIO)
    effective_height = CAMERA_FOV_HEIGHT_MM * (1.0 - CAMERA_OVERLAP_RATIO)
    return effective_width, effective_height


# ============================================================================
# World Configuration (Isaac Sim coordinates, meters)
# ============================================================================

# Glass object position in world frame (x, y, z)
GLASS_POSITION = np.array([1.3, 0.0, -0.125], dtype=np.float64)

# Table cuboid position in world frame (x, y, z)
TABLE_POSITION = np.array([1.3, 0.0, -0.425], dtype=np.float64)

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
DEFAULT_MESH_FILE = str(DATA_ROOT / "object" / "glass_zup.obj")

# Robot files
DEFAULT_ROBOT_URDF = "ur_description/ur20.urdf"
DEFAULT_ROBOT_CONFIG = "ur20_safe.yml"
DEFAULT_ROBOT_CONFIG_YAML = "ur_description/ur20_safe.yml"  # For collision spheres

# Mesh base path for URDF collision meshes
MESH_BASE_PATH = "ur_description"


# ============================================================================
# Algorithm Parameters
# ============================================================================

# Trajectory interpolation
INTERPOLATION_STEPS = 60  # Number of steps between waypoints

# IK Solver
IK_NUM_SEEDS = 20  # Number of random seeds for IK solver
IK_ROTATION_THRESHOLD = 0.05  # Rotation error threshold (radians)
IK_POSITION_THRESHOLD = 0.005  # Position error threshold (meters)

# Joint selection (for DP algorithm in run_app_v3.py)
JOINT_WEIGHTS = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
RECONFIGURATION_THRESHOLD = 1.0  # radians
RECONFIGURATION_PENALTY = 10.0
MAX_MOVE_WEIGHT = 5.0

# Collision checking
COLLISION_MARGIN = 0.0  # Safety margin in meters (0 = no margin)
COLLISION_INTERP_STEPS = 10  # Interpolation steps for collision checking


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


# ============================================================================
# Configuration Export
# ============================================================================

def get_camera_spec_dict() -> dict:
    """
    Get camera specification as dictionary for HDF5 storage

    Returns:
        Dictionary with all camera specifications
    """
    return {
        'sensor_width_px': CAMERA_SENSOR_WIDTH_PX,
        'sensor_height_px': CAMERA_SENSOR_HEIGHT_PX,
        'pixel_size_um': CAMERA_PIXEL_SIZE_UM,
        'fov_width_mm': CAMERA_FOV_WIDTH_MM,
        'fov_height_mm': CAMERA_FOV_HEIGHT_MM,
        'working_distance_mm': CAMERA_WORKING_DISTANCE_MM,
        'depth_of_field_mm': CAMERA_DEPTH_OF_FIELD_MM,
        'overlap_ratio': CAMERA_OVERLAP_RATIO,
    }


def print_config_summary():
    """Print configuration summary for debugging"""
    print("=" * 70)
    print("VISION INSPECTION CONFIGURATION")
    print("=" * 70)
    print("\nCamera Specifications:")
    print(f"  Sensor: {CAMERA_SENSOR_WIDTH_PX} x {CAMERA_SENSOR_HEIGHT_PX} px")
    print(f"  Pixel size: {CAMERA_PIXEL_SIZE_UM} μm")
    print(f"  FOV: {CAMERA_FOV_WIDTH_MM} x {CAMERA_FOV_HEIGHT_MM} mm")
    print(f"  Working Distance: {CAMERA_WORKING_DISTANCE_MM} mm")
    print(f"  Depth of Field: {CAMERA_DEPTH_OF_FIELD_MM} mm")
    print(f"  Overlap: {CAMERA_OVERLAP_RATIO * 100:.1f}%")
    eff_w, eff_h = get_effective_coverage_mm()
    print(f"  Effective coverage: {eff_w:.2f} x {eff_h:.2f} mm")

    print("\nWorld Configuration:")
    print(f"  Glass position: {GLASS_POSITION}")
    print(f"  Table position: {TABLE_POSITION}")
    print(f"  Table dimensions: {TABLE_DIMENSIONS}")
    print(f"  Wall position: {WALL_POSITION}")
    print(f"  Wall dimensions: {WALL_DIMENSIONS}")
    print(f"  Workbench position: {WORKBENCH_POSITION}")
    print(f"  Workbench dimensions: {WORKBENCH_DIMENSIONS}")
    print(f"  Robot mount position: {ROBOT_MOUNT_POSITION}")
    print(f"  Robot mount dimensions: {ROBOT_MOUNT_DIMENSIONS}")

    print("\nAlgorithm Parameters:")
    print(f"  Interpolation steps: {INTERPOLATION_STEPS}")
    print(f"  IK seeds: {IK_NUM_SEEDS}")
    print(f"  Joint weights: {JOINT_WEIGHTS}")

    print("\nFile Paths:")
    print(f"  Default mesh: {DEFAULT_MESH_FILE}")
    print(f"  Robot URDF: {DEFAULT_ROBOT_URDF}")
    print(f"  Robot config: {DEFAULT_ROBOT_CONFIG}")
    print("=" * 70)


if __name__ == "__main__":
    print_config_summary()
