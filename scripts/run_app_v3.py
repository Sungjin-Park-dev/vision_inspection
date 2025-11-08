#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
Refactored vision inspection robot trajectory planner (v3)

⚠️  DEPRECATED - This script is deprecated as of 2025-11-08
===========================================================

This monolithic script has been split into modular components for better maintainability:

NEW RECOMMENDED WORKFLOW:
1. Compute IK solutions:
   omni_python scripts/compute_ik_solutions.py --tsp_tour data/tour/tour_3000.h5

2. Plan trajectory (no Isaac Sim required):
   python scripts/plan_trajectory.py --ik_solutions data/ik/ik_solutions_3000.h5 --method dp

3. Simulate trajectory:
   omni_python scripts/simulate_trajectory.py --trajectory data/trajectory/3000/joint_trajectory_dp.csv

OR run full pipeline at once:
   omni_python scripts/run_full_pipeline.py --tsp_tour data/tour/tour_3000.h5 --method dp --simulate

Benefits of new workflow:
- ✓ Compute IK once, try multiple planning methods without re-running IK
- ✓ Plan trajectory without Isaac Sim (faster iteration)
- ✓ Each stage can be tested independently
- ✓ Better code organization and maintainability

This file is kept for backward compatibility only.
===========================================================

Original Description:
This version improves upon run_app_v2.py by:
- Removing global variables and encapsulating state in classes
- Breaking down the large main() function into smaller, testable functions
- Removing unused code
- Making configuration values explicit and configurable
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import csv
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# ============================================================================
# Third Party Imports
# ============================================================================
import carb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from urchin import URDF

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Isaac Sim Imports
# ============================================================================
try:
    import isaacsim
except ImportError:
    pass

from isaacsim.simulation_app import SimulationApp

# Parse arguments before SimulationApp initialization
parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument("--robot", type=str, default="ur20.yml", help="robot configuration to load")
parser.add_argument(
    "--tsp_tour_path",
    type=str,
    required=True,
    help="Path to TSP tour result file (.h5). Required for robot trajectory planning with TSP-optimized visit order.",
)
parser.add_argument(
    "--save_plot",
    action="store_true",
    help="When True, saves joint trajectory plot as PNG file",
    default=False,
)
parser.add_argument(
    "--selection_method",
    type=str,
    default="dp",
    choices=["random", "greedy", "dp"],
    help="Method to select IK solutions: 'random' (random selection), 'greedy' (nearest neighbor), 'dp' (dynamic programming, default)",
)
parser.add_argument(
    "--no_sim",
    action="store_true",
    help="Without Isaac Sim",
    default=False,
)
args = parser.parse_args()

# Initialize SimulationApp
simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1280",
        "height": "720",
    }
)

# ============================================================================
# Isaac Sim Component Imports (after SimulationApp)
# ============================================================================
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from omni.isaac.core.prims import XFormPrim

try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    from isaacsim.util.debug_draw import _debug_draw

from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.api.materials import OmniGlass
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

# ============================================================================
# CuRobo Imports
# ============================================================================
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.geom.transform import pose_to_matrix, matrix_to_quaternion
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose, quat_multiply
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import (
    MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
)
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# ============================================================================
# Local Imports
# ============================================================================
# Common utilities
from common import config
from common.coordinate_utils import normalize_vectors, offset_points_along_normals
from common.interpolation_utils import generate_interpolated_path

# Project modules
from utilss.simulation_helper import add_extensions, add_robot_to_scene
from analyze_joint_reconfigurations import (
    load_joint_trajectory,
    analyze_joint_reconfigurations,
    print_analysis_results
)
from eaik.IK_URDF import UrdfRobot
from eaik.IK_DH import DhRobot


# ============================================================================
# Constants and Coordinate Transformations
# ============================================================================
# Note: Z-up coordinate system is now used throughout the pipeline
# No Y-up → Z-up conversion needed (mesh files are already Z-up)

CUROBO_TO_EAIK_TOOL = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class SimulationConfig:
    """Central configuration for all simulation parameters"""

    # Command line arguments
    robot_config_file: str
    tsp_tour_path: str
    selection_method: str
    headless_mode: Optional[str]
    visualize_spheres: bool
    save_plot: bool
    no_sim: bool

    # Robot and trajectory parameters (defaults from common.config)
    normal_sample_offset: float = config.get_camera_working_distance_m()  # meters (working distance)
    interpolation_steps: int = config.INTERPOLATION_STEPS

    # Joint selection cost function parameters (defaults from common.config)
    joint_weights: np.ndarray = field(
        default_factory=lambda: config.JOINT_WEIGHTS.copy()
    )
    reconfig_threshold: float = config.RECONFIGURATION_THRESHOLD  # radians
    reconfig_penalty: float = config.RECONFIGURATION_PENALTY
    max_move_weight: float = config.MAX_MOVE_WEIGHT

    # Visualization parameters
    plot_interval: int = 5  # Update plot every N viewpoints

    # World configuration (defaults from common.config)
    table_position: np.ndarray = field(default_factory=lambda: config.TABLE_POSITION.copy())
    table_dimensions: np.ndarray = field(default_factory=lambda: config.TABLE_DIMENSIONS.copy())
    glass_position: np.ndarray = field(default_factory=lambda: config.GLASS_POSITION.copy())

    # IK Solver configuration (defaults from common.config)
    ik_rotation_threshold: float = config.IK_ROTATION_THRESHOLD
    ik_position_threshold: float = config.IK_POSITION_THRESHOLD
    ik_num_seeds: int = config.IK_NUM_SEEDS

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SimulationConfig':
        """Create configuration from command line arguments"""
        return cls(
            robot_config_file=args.robot,
            tsp_tour_path=args.tsp_tour_path,
            selection_method=args.selection_method,
            headless_mode=args.headless_mode,
            visualize_spheres=args.visualize_spheres,
            save_plot=args.save_plot,
            no_sim=args.no_sim,
        )


@dataclass
class WorldState:
    """Encapsulates Isaac Sim world state"""
    world: World
    glass_prim: XFormPrim
    robot: any  # Robot articulation
    idx_list: List[int]  # Active joint indices
    ik_solver: IKSolver
    default_config: np.ndarray  # Retract configuration


@dataclass
class Viewpoint:
    """Represents a camera viewpoint with IK solutions"""
    index: int
    local_pose: Optional[np.ndarray] = None  # 4x4 pose in object frame
    world_pose: Optional[np.ndarray] = None  # 4x4 pose in world frame
    all_ik_solutions: List[np.ndarray] = field(default_factory=list)
    safe_ik_solutions: List[np.ndarray] = field(default_factory=list)


# ============================================================================
# State Management Classes
# ============================================================================
class ViewpointManager:
    """Manages viewpoint data and operations"""

    def __init__(
        self,
        viewpoints: List[Viewpoint],
        local_points: np.ndarray,
        local_normals: np.ndarray
    ):
        self.viewpoints = viewpoints
        self.local_points = local_points
        self.local_normals = local_normals

    def filter_with_safe_ik(self) -> List[Viewpoint]:
        """Return only viewpoints that have safe IK solutions"""
        return [vp for vp in self.viewpoints if len(vp.safe_ik_solutions) > 0]

    def count_with_all_ik(self) -> int:
        """Count viewpoints with any IK solutions"""
        return sum(1 for vp in self.viewpoints if len(vp.all_ik_solutions) > 0)

    def count_with_safe_ik(self) -> int:
        """Count viewpoints with collision-free IK solutions"""
        return sum(1 for vp in self.viewpoints if len(vp.safe_ik_solutions) > 0)

    def update_world_poses(self, reference_prim: XFormPrim, debug_first: bool = True):
        """Update world poses for all viewpoints"""
        for i, viewpoint in enumerate(self.viewpoints):
            if viewpoint.local_pose is None:
                continue
            debug = debug_first and i == 0
            viewpoint.world_pose = open3d_pose_to_world(
                viewpoint.local_pose, reference_prim, debug=debug
            )

    def collect_world_matrices(self) -> Tuple[np.ndarray, List[int]]:
        """Collect world pose matrices from viewpoints"""
        matrices: List[np.ndarray] = []
        indices: List[int] = []

        for idx, viewpoint in enumerate(self.viewpoints):
            if viewpoint.world_pose is None:
                continue
            matrices.append(np.asarray(viewpoint.world_pose, dtype=np.float64))
            indices.append(idx)

        if matrices:
            stacked = np.stack(matrices, axis=0)
        else:
            stacked = np.empty((0, 4, 4), dtype=np.float64)

        return stacked, indices


class JointHistoryTracker:
    """Tracks joint trajectory history during simulation"""

    def __init__(self):
        self.timestamps: List[int] = []
        self.joint_values: List[np.ndarray] = []
        self.viewpoint_markers: List[int] = []

    def record_step(self, timestamp: int, joints: np.ndarray):
        """Record joint values at a timestep"""
        self.timestamps.append(timestamp)
        self.joint_values.append(joints)

    def mark_viewpoint(self, timestamp: int):
        """Mark that a viewpoint was reached"""
        self.viewpoint_markers.append(timestamp)

    def has_data(self) -> bool:
        """Check if any data has been recorded"""
        return len(self.timestamps) > 0

    def get_joint_array(self) -> np.ndarray:
        """Get joint values as (N, 6) array"""
        if not self.joint_values:
            return np.empty((0, 6))
        return np.array(self.joint_values)


# ============================================================================
# Utility Functions
# ============================================================================
# normalize_vectors() and offset_points_along_normals() are now imported from common.coordinate_utils


# open3d_to_isaac_coords() removed - Z-up coordinate system used throughout (no conversion needed)


def open3d_pose_to_world(
    pose_matrix: np.ndarray, reference_prim: XFormPrim, debug: bool = False
) -> np.ndarray:
    """
    Transform local pose to Isaac Sim world pose

    Note: Z-up coordinate system is used throughout, so no Open3D→Isaac conversion needed.
    """
    if pose_matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4")
    if reference_prim is None:
        raise ValueError("reference_prim is required to map pose into world coordinates")

    world_position, world_orientation = reference_prim.get_world_pose()
    local_scale = np.asarray(reference_prim.get_local_scale(), dtype=np.float64)
    rotation_matrix = quat_to_rot_matrix(np.asarray(world_orientation, dtype=np.float64))

    if debug:
        print(f"\n=== Coordinate Transform Debug ===")
        print(f"Reference object world position: {world_position}")
        print(f"Reference object world orientation (quat): {world_orientation}")
        print(f"Reference object local scale: {local_scale}")
        print(f"Original pose position (Z-up): {pose_matrix[:3, 3]}")

    # Extract local pose components (already in Z-up)
    local_rot = pose_matrix[:3, :3]
    local_pos = pose_matrix[:3, 3]

    # Apply scale, rotation, translation
    scaled_pos = local_pos * local_scale
    if debug:
        print(f"After scaling: {scaled_pos}")

    rotated_pos = rotation_matrix @ scaled_pos
    if debug:
        print(f"After world rotation: {rotated_pos}")

    world_pos = rotated_pos + np.asarray(world_position, dtype=np.float64)

    if debug:
        print(f"Final world position: {world_pos}")
        print(f"===================================\n")

    world_rot = rotation_matrix @ local_rot

    world_pose = np.eye(4, dtype=np.float64)
    world_pose[:3, :3] = world_rot
    world_pose[:3, 3] = world_pos
    return world_pose


def extract_pose_from_matrix(
    pose_matrix: np.ndarray
) -> Tuple[float, float, float, float, float, float, float]:
    """Extract position and quaternion from 4x4 transformation matrix

    Returns:
        Tuple of (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
    """
    if pose_matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4")

    # Extract position
    position = pose_matrix[:3, 3]
    pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])

    # Extract rotation and convert to quaternion
    rotation_matrix = pose_matrix[:3, :3]
    rot_tensor = torch.from_numpy(rotation_matrix.astype(np.float32)).unsqueeze(0)
    quat_tensor = matrix_to_quaternion(rot_tensor)  # Returns (w, x, y, z)

    # Convert to (x, y, z, w) order
    quat_w = float(quat_tensor[0, 0].item())
    quat_x = float(quat_tensor[0, 1].item())
    quat_y = float(quat_tensor[0, 2].item())
    quat_z = float(quat_tensor[0, 3].item())

    return pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w


# ============================================================================
# TSP Tour Loading
# ============================================================================
def load_tsp_tour(tsp_tour_path: str) -> dict:
    """Load TSP tour result from HDF5 file"""
    from tsp_utils import load_tsp_result
    return load_tsp_result(tsp_tour_path)


def create_viewpoints_from_tsp(
    tsp_result: dict, config: SimulationConfig
) -> ViewpointManager:
    """
    Create ViewpointManager from TSP tour result

    The TSP tour file stores surface positions and normals in Open3D coordinates.
    This function:
    1. Extracts points and normals in TSP visit order
    2. Determines working distance from camera_spec or config
    3. Offsets surface positions to create camera viewpoints
    4. Creates local pose matrices for each viewpoint

    Returns:
        ViewpointManager with viewpoints in TSP order
    """
    # Extract tour data
    tour_coords = tsp_result["tour"]["coordinates"]
    tour_indices = tsp_result["tour"]["indices"]
    normals = tsp_result["normals"]
    tour_normals = normals[tour_indices]

    print(f"\n{'='*60}")
    print("TSP TOUR DATA LOADING")
    print(f"{'='*60}")
    print(f"Loaded {len(tour_coords)} points in TSP-optimized visit order")
    print(f"Coordinate system: Open3D (Y-up)")
    print(f"\nOriginal coordinate ranges (Open3D):")
    print(f"  X: [{tour_coords[:, 0].min():.4f}, {tour_coords[:, 0].max():.4f}]")
    print(f"  Y: [{tour_coords[:, 1].min():.4f}, {tour_coords[:, 1].max():.4f}]")
    print(f"  Z: [{tour_coords[:, 2].min():.4f}, {tour_coords[:, 2].max():.4f}]")
    print(f"{'='*60}\n")

    # Determine working distance
    working_distance_m = config.normal_sample_offset  # Default

    if 'metadata' in tsp_result and 'camera_spec' in tsp_result['metadata']:
        camera_spec = tsp_result['metadata']['camera_spec']
        if 'working_distance_mm' in camera_spec:
            working_distance_m = camera_spec['working_distance_mm'] / 1000.0
            print(f"Using working distance from HDF5 camera_spec: "
                  f"{camera_spec['working_distance_mm']} mm = {working_distance_m} m")
        else:
            print(f"⚠️  WARNING: camera_spec found but no working_distance_mm, "
                  f"using default {config.normal_sample_offset} m")
    else:
        print(f"No camera_spec in HDF5, using default working distance: "
              f"{config.normal_sample_offset} m (100mm)")
        print(f"⚠️  For FOV-based viewpoints, this may not match the intended working distance!")

    # Generate viewpoint poses
    print(f"\nGenerating viewpoint poses from surface positions...")
    print(f"  Surface positions: {len(tour_coords)} points")
    print(f"  Offsetting by {working_distance_m*1000:.1f} mm along surface normals")

    # Offset surface points to camera positions
    offset_points = offset_points_along_normals(tour_coords, tour_normals, working_distance_m)

    # Camera looks toward surface (negative of surface normal)
    approach_normals = -normalize_vectors(tour_normals)

    # Create viewpoints with local poses
    helper_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    helper_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    viewpoints: List[Viewpoint] = []

    for point_idx, (position, normal) in enumerate(zip(offset_points, approach_normals)):
        # Build orthogonal frame with normal as Z-axis
        z_axis = normal / np.linalg.norm(normal)

        helper = helper_z if np.abs(np.dot(z_axis, helper_z)) <= 0.99 else helper_y
        x_axis = np.cross(helper, z_axis)
        norm_x = np.linalg.norm(x_axis)

        if norm_x < 1e-6:
            helper = helper_y if np.abs(np.dot(z_axis, helper_z)) > 0.99 else helper_z
            x_axis = np.cross(helper, z_axis)
            norm_x = np.linalg.norm(x_axis)
            if norm_x < 1e-6:
                raise ValueError("Failed to construct orthogonal frame from normal vector")

        x_axis /= norm_x
        y_axis = np.cross(z_axis, x_axis)

        # Create 4x4 pose matrix
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        pose_matrix[:3, 3] = position.astype(np.float64)

        viewpoints.append(Viewpoint(index=int(point_idx), local_pose=pose_matrix))

    print(f"Generated {len(viewpoints)} viewpoints in TSP order")

    return ViewpointManager(
        viewpoints=viewpoints,
        local_points=offset_points,
        local_normals=approach_normals
    )


# ============================================================================
# World Initialization
# ============================================================================
def create_world() -> World:
    """Create Isaac Sim world"""
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    return my_world


def setup_robot(my_world: World, config: SimulationConfig) -> dict:
    """Setup robot in the world

    Returns:
        dict with keys: robot, idx_list, default_config, robot_prim_path, robot_cfg
    """
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), config.robot_config_file))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(
        robot_config=robot_cfg,
        my_world=my_world,
        position=np.array([0.0, 0.0, 0.0]),
    )

    idx_list = [robot.get_dof_index(x) for x in j_names]
    robot.set_joint_positions(default_config, idx_list)

    return {
        'robot': robot,
        'idx_list': idx_list,
        'default_config': default_config,
        'robot_prim_path': robot_prim_path,
        'robot_cfg': robot_cfg,
    }


def setup_glass_object(my_world: World, config: SimulationConfig) -> XFormPrim:
    """Setup glass object in the world"""
    asset_path = "/isaac-sim/curobo/vision_inspection/data/input/object/glass.usdc"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/glass_usd")

    glass_prim = XFormPrim(
        prim_path="/World/glass_usd",
        position=config.glass_position,
    )

    # Print transform for verification
    glass_world_pos, glass_world_rot = glass_prim.get_world_pose()
    glass_local_scale = glass_prim.get_local_scale()
    print(f"\n{'='*60}")
    print("GLASS OBJECT TRANSFORM (Isaac Sim)")
    print(f"{'='*60}")
    print(f"World position: {glass_world_pos} (meters)")
    print(f"World rotation (quat): {glass_world_rot}")
    print(f"Local scale: {glass_local_scale}")
    print(f"Note: Isaac Sim uses meters (stage_units_in_meters=1.0)")
    print(f"{'='*60}\n")

    # Apply glass material
    glass_material = OmniGlass(
        prim_path="/World/Looks/glass_mat",
        color=np.array([0.7, 0.85, 0.9]),
        ior=1.52,
        depth=0.01,
        thin_walled=False,
    )
    glass_prim.apply_visual_material(glass_material)

    return glass_prim


def setup_glass_object_from_mesh(my_world: World, config: SimulationConfig, usd_helper: UsdHelper) -> XFormPrim:
    """Setup glass object in the world using mesh file

    Args:
        my_world: Isaac Sim world
        config: Simulation configuration
        usd_helper: UsdHelper for adding mesh to stage

    Returns:
        XFormPrim: Glass object prim
    """
    # Default mesh path (can be overridden in config)
    mesh_file_path = getattr(config, 'glass_mesh_path',
                              "/isaac-sim/curobo/vision_inspection/data/object/glass_zup.obj")

    print(f"\n{'='*60}")
    print("ADDING GLASS MESH TO STAGE")
    print(f"{'='*60}")
    print(f"Mesh file: {mesh_file_path}")
    print(f"Position: {config.glass_position}")
    print(f"{'='*60}\n")

    # Load stage
    usd_helper.load_stage(my_world.stage)

    # Create glass mesh with position and orientation
    glass_mesh = Mesh(
        name="glass",
        file_path=mesh_file_path,
        pose=list(config.glass_position) + [1, 0, 0, 0],  # position + quaternion (w, x, y, z)
        color=[0.7, 0.85, 0.9, 0.3]  # Light blue with transparency
    )

    # Add mesh to stage
    glass_path = usd_helper.add_mesh_to_stage(
        obstacle=glass_mesh,
        base_frame="/World"
    )

    print(f"Glass mesh added at prim path: {glass_path}")

    # Wrap with XFormPrim for control
    glass_prim = XFormPrim(glass_path)

    # Print transform for verification
    glass_world_pos, glass_world_rot = glass_prim.get_world_pose()
    glass_local_scale = glass_prim.get_local_scale()
    print(f"\n{'='*60}")
    print("GLASS OBJECT TRANSFORM (Isaac Sim)")
    print(f"{'='*60}")
    print(f"World position: {glass_world_pos} (meters)")
    print(f"World rotation (quat): {glass_world_rot}")
    print(f"Local scale: {glass_local_scale}")
    print(f"Note: Isaac Sim uses meters (stage_units_in_meters=1.0)")
    print(f"{'='*60}\n")

    # Apply glass material (optional)
    try:
        glass_material = OmniGlass(
            prim_path="/World/Looks/glass_mat",
            color=np.array([0.7, 0.85, 0.9]),
            ior=1.52,
            depth=0.01,
            thin_walled=False,
        )
        glass_prim.apply_visual_material(glass_material)
        print("Applied OmniGlass material")
    except Exception as e:
        print(f"Warning: Could not apply glass material: {e}")
        print("Continuing with default mesh material...")

    return glass_prim


def setup_camera(robot_prim_path: str, my_world: World):
    """Setup camera mounted on robot end-effector"""
    tool_prim_path = robot_prim_path + "/tool0"
    camera_prim_path = tool_prim_path + "/mounted_camera"

    camera = Camera(
        prim_path=camera_prim_path,
        frequency=20,
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1, 0, 0, 0]),
        resolution=(256, 256),
    )

    # Camera specifications
    camera.set_focal_length(38.0 / 1e3)
    camera.set_focus_distance(110.0 / 1e3)
    camera.set_horizontal_aperture(14.13 / 1e3)
    camera.set_vertical_aperture(10.35 / 1e3)
    camera.set_clipping_range(10/1e3, 100/1e3)
    camera.set_local_pose(
        np.array([0.0, 0.0, 0.0]),
        euler_angles_to_quats(np.array([0, 180, 0]), degrees=True),
        camera_axes="usd"
    )
    my_world.scene.add(camera)

    return camera


def setup_collision_checker(
    my_world: World,
    robot_state: dict,
    config: SimulationConfig
) -> IKSolver:
    """Setup collision checker and IK solver"""
    usd_helper = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = robot_state['robot_cfg']
    robot_prim_path = robot_state['robot_prim_path']

    # Setup world collision configuration
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[:3] = config.table_position
    world_cfg_table.cuboid[0].dims[:3] = config.table_dimensions

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # Create IK solver
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=config.ik_rotation_threshold,
        position_threshold=config.ik_position_threshold,
        num_seeds=config.ik_num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    )
    ik_solver = IKSolver(ik_config)

    # Setup world in USD
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane(z_position=-0.5)

    # Get obstacles from stage
    obstacles = usd_helper.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/mount",
        ],
    ).get_collision_check_world()

    print("Object list:")
    print([x.name for x in obstacles.objects])
    print("=" * 60 + "\n")

    ik_solver.update_world(obstacles)

    return ik_solver


def initialize_simulation(config: SimulationConfig) -> WorldState:
    """Initialize Isaac Sim world and all components

    Returns:
        WorldState with all initialized components
    """
    print(f"\n{'='*60}")
    print("INITIALIZING SIMULATION")
    print(f"{'='*60}\n")

    my_world = create_world()
    robot_state = setup_robot(my_world, config)
    # glass_prim = setup_glass_object(my_world, config)
    
    usd_helper = UsdHelper()
    glass_prim = setup_glass_object_from_mesh(my_world, config, usd_helper)

    camera = setup_camera(robot_state['robot_prim_path'], my_world)
    ik_solver = setup_collision_checker(my_world, robot_state, config)

    return WorldState(
        world=my_world,
        glass_prim=glass_prim,
        robot=robot_state['robot'],
        idx_list=robot_state['idx_list'],
        ik_solver=ik_solver,
        default_config=robot_state['default_config'],
    )


# ============================================================================
# IK Computation
# ============================================================================
def compute_ik_eaik(
    world_matrices: np.ndarray,
    urdf_path: str = "/isaac-sim/curobo/src/curobo/content/assets/robot/ur_description/ur20.urdf"
):
    """Compute IK solutions using EAIK (analytical IK solver)

    Args:
        world_matrices: (N, 4, 4) array of world pose matrices
        urdf_path: Path to robot URDF file

    Returns:
        List of EAIK solution objects
    """
    if isinstance(world_matrices, torch.Tensor):
        mats_np = world_matrices.detach().cpu().numpy()
    else:
        mats_np = np.asarray(world_matrices)

    if mats_np.ndim != 3 or mats_np.shape[1:] != (4, 4):
        raise ValueError("Expected poses shaped (batch, 4, 4)")

    # Load URDF robot
    bot = UrdfRobot(urdf_path)

    # Transform from CuRobo to EAIK tool frame
    curobo_to_eaik_tool = CUROBO_TO_EAIK_TOOL.astype(mats_np.dtype, copy=False)
    mats_eaik = mats_np @ curobo_to_eaik_tool

    # Compute IK solutions
    solutions = bot.IK_batched(mats_eaik)

    return solutions


def assign_ik_solutions_to_viewpoints(
    viewpoints: List[Viewpoint],
    eaik_results,
    indices: Optional[List[int]] = None,
):
    """Assign EAIK IK solutions to viewpoints

    Args:
        viewpoints: List of Viewpoint objects
        eaik_results: Results from EAIK solver
        indices: Indices of viewpoints that have results
    """
    # Clear existing solutions
    for viewpoint in viewpoints:
        viewpoint.all_ik_solutions = []
        viewpoint.safe_ik_solutions = []

    if eaik_results is None:
        return

    try:
        eaik_list = list(eaik_results)
    except TypeError:
        eaik_list = [eaik_results]

    if indices is None:
        indices = list(range(len(viewpoints)))

    max_assignable = min(len(indices), len(eaik_list))

    for result_idx in range(max_assignable):
        viewpoint_idx = indices[result_idx]
        result = eaik_list[result_idx]

        if result is None:
            continue

        q_candidates = getattr(result, "Q", None)
        if q_candidates is None or len(q_candidates) == 0:
            continue

        solutions = [np.asarray(q, dtype=np.float64) for q in q_candidates]
        viewpoints[viewpoint_idx].all_ik_solutions = solutions


def check_ik_solutions_collision(
    viewpoints: List[Viewpoint],
    ik_solver: IKSolver,
):
    """Check which IK solutions are collision-free and update safe_ik_solutions

    Args:
        viewpoints: List of Viewpoint objects with all_ik_solutions
        ik_solver: IK solver with collision checker
    """
    # Clear safe solutions
    for viewpoint in viewpoints:
        viewpoint.safe_ik_solutions = []

    # Batch all solutions
    batched_q: List[np.ndarray] = []
    index_map: List[Tuple[int, int]] = []  # (viewpoint_idx, solution_idx)

    for vp_idx, viewpoint in enumerate(viewpoints):
        for sol_idx, solution in enumerate(viewpoint.all_ik_solutions):
            batched_q.append(np.asarray(solution, dtype=np.float64))
            index_map.append((vp_idx, sol_idx))

    if not batched_q:
        return

    # Convert to tensor
    batched_array = np.stack(batched_q, axis=0)
    tensor_args = getattr(ik_solver, "tensor_args", TensorDeviceType())
    q_tensor = tensor_args.to_device(torch.from_numpy(batched_array))

    # Create joint state
    zeros = torch.zeros_like(q_tensor)
    joint_state = JointState(
        position=q_tensor,
        velocity=zeros,
        acceleration=zeros,
        jerk=zeros,
        joint_names=ik_solver.kinematics.joint_names,
    )

    # Check constraints
    metrics = ik_solver.check_constraints(joint_state)
    feasible = getattr(metrics, "feasible", None)

    if feasible is None:
        feasibility = torch.ones(len(index_map), dtype=torch.bool)
    else:
        feasibility = feasible.detach()
        if feasibility.is_cuda:
            feasibility = feasibility.cpu()
        feasibility = feasibility.flatten().to(dtype=torch.bool)

    # Update safe solutions
    for batch_idx, ((vp_idx, _), is_feasible) in enumerate(zip(index_map, feasibility)):
        if not bool(is_feasible):
            continue
        solution = batched_q[batch_idx]
        viewpoints[vp_idx].safe_ik_solutions.append(solution)


def process_viewpoints(
    config: SimulationConfig,
    world_state: WorldState
) -> ViewpointManager:
    """Process viewpoints: load TSP, compute IK, check collisions

    Returns:
        ViewpointManager with processed viewpoints
    """
    print(f"\n{'='*60}")
    print("PROCESSING VIEWPOINTS")
    print(f"{'='*60}\n")

    # Load TSP tour
    print("Loading TSP tour...")
    tsp_result = load_tsp_tour(config.tsp_tour_path)

    # Create viewpoints from TSP
    viewpoint_mgr = create_viewpoints_from_tsp(tsp_result, config)

    # Update world poses
    viewpoint_mgr.update_world_poses(world_state.glass_prim)

    # Collect world matrices
    world_mats, used_indices = viewpoint_mgr.collect_world_matrices()

    if world_mats.size == 0:
        raise ValueError("Error: No valid world poses found for sampled viewpoints")

    # Compute IK solutions
    print(f"\nComputing IK solutions for {len(world_mats)} viewpoints...")
    assign_start = perf_counter()
    ik_results = compute_ik_eaik(world_mats)
    assign_ik_solutions_to_viewpoints(viewpoint_mgr.viewpoints, ik_results, used_indices)
    assign_elapsed = perf_counter() - assign_start

    # Check collisions
    print("Checking collision constraints...")
    safe_start = perf_counter()
    check_ik_solutions_collision(viewpoint_mgr.viewpoints, world_state.ik_solver)
    safe_elapsed = perf_counter() - safe_start

    print(f"IK computation time: {assign_elapsed * 1000.0:.2f} ms")
    print(f"Collision checking time: {safe_elapsed * 1000.0:.2f} ms")

    # Print statistics
    total = len(viewpoint_mgr.viewpoints)
    with_all = viewpoint_mgr.count_with_all_ik()
    with_safe = viewpoint_mgr.count_with_safe_ik()
    print(f"\nIK Statistics:")
    print(f"  Total viewpoints: {total}")
    print(f"  With any IK solutions: {with_all}/{total}")
    print(f"  With safe IK solutions: {with_safe}/{total}")
    print(f"{'='*60}\n")

    return viewpoint_mgr


# ============================================================================
# Joint Trajectory Planning
# ============================================================================
def compute_weighted_joint_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    joint_weights: Optional[np.ndarray] = None
) -> float:
    """Compute weighted Euclidean distance between two joint configurations"""
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    weighted_diff = joint_weights * (q1_array - q2_array) ** 2
    return np.sqrt(np.sum(weighted_diff))


def compute_reconfiguration_cost(
    q1: np.ndarray,
    q2: np.ndarray,
    joint_weights: Optional[np.ndarray] = None,
    reconfig_threshold: float = 1.0,
    reconfig_penalty: float = 10.0,
    max_move_weight: float = 5.0
) -> Tuple[float, dict]:
    """
    Compute reconfiguration-aware cost that penalizes large joint movements

    Returns:
        Tuple of (total_cost, cost_breakdown_dict)
    """
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    # Base cost: weighted Euclidean distance
    base_cost = compute_weighted_joint_distance(q1_array, q2_array, joint_weights)

    # Reconfiguration penalty
    joint_deltas = np.abs(q1_array - q2_array)
    reconfig_count = np.sum(joint_deltas > reconfig_threshold)
    reconfig_cost = reconfig_count * reconfig_penalty

    # Max joint movement penalty
    max_joint_move = np.max(joint_deltas)
    max_move_cost = max_joint_move * max_move_weight

    total_cost = base_cost  # Currently using only base cost

    cost_breakdown = {
        'base_cost': base_cost,
        'reconfig_count': int(reconfig_count),
        'reconfig_cost': reconfig_cost,
        'max_joint_move': max_joint_move,
        'max_move_cost': max_move_cost,
        'total_cost': total_cost
    }

    return total_cost, cost_breakdown


def select_ik_random(viewpoints: List[Viewpoint]) -> Tuple[List[np.ndarray], List[int]]:
    """Select first safe IK solution for each viewpoint (RANDOM/FIRST method)"""
    print(f"\nSelecting joint targets (RANDOM/FIRST method):")

    targets: List[np.ndarray] = []
    solution_indices: List[int] = []

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} has no safe IK solutions, skipping")
            continue

        solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)
        targets.append(solution)
        solution_indices.append(0)

    print(f"  → Collected {len(targets)} targets")
    return targets, solution_indices


def select_ik_greedy(viewpoints: List[Viewpoint]) -> Tuple[List[np.ndarray], List[int]]:
    """Select IK solutions using greedy nearest neighbor (GREEDY method)"""
    print(f"\nSelecting joint targets (GREEDY method):")

    targets: List[np.ndarray] = []
    solution_indices: List[int] = []
    previous_joints = None

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} has no safe IK solutions, skipping")
            continue

        if previous_joints is None:
            # First viewpoint: use first solution
            solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)
            selected_idx = 0
        else:
            # Find closest solution to previous configuration
            best_solution = None
            min_distance = float('inf')
            selected_idx = 0

            for sol_idx, candidate in enumerate(viewpoint.safe_ik_solutions):
                candidate_joints = np.asarray(candidate, dtype=np.float64)
                distance = np.linalg.norm(candidate_joints - previous_joints)
                if distance < min_distance:
                    min_distance = distance
                    best_solution = candidate_joints
                    selected_idx = sol_idx

            solution = best_solution

        targets.append(solution)
        solution_indices.append(selected_idx)
        previous_joints = solution

    print(f"  → Collected {len(targets)} targets")
    return targets, solution_indices


def select_ik_dp(
    viewpoints: List[Viewpoint],
    initial_config: np.ndarray,
    config: SimulationConfig
) -> Tuple[List[np.ndarray], float, List[int]]:
    """Select IK solutions using Dynamic Programming (DP method)"""
    print(f"\nSelecting joint targets (DP method):")

    initial_joints = np.asarray(initial_config, dtype=np.float64)

    # Filter viewpoints with safe solutions
    valid_viewpoints = [vp for vp in viewpoints if vp.safe_ik_solutions]
    n_viewpoints = len(valid_viewpoints)

    if not valid_viewpoints:
        print("  Error: No valid viewpoints with safe IK solutions!")
        return [], 0.0, []

    print(f"  Processing {n_viewpoints} viewpoints")
    print(f"  Joint weights: {config.joint_weights}")
    print(f"  Reconfiguration threshold: {config.reconfig_threshold} rad")

    # DP table: dp[i][sol_idx] = (min_cost, prev_solution_idx)
    dp = [dict() for _ in range(n_viewpoints)]

    # Initialize first viewpoint
    first_vp = valid_viewpoints[0]
    for sol_idx, solution in enumerate(first_vp.safe_ik_solutions):
        sol_array = np.asarray(solution, dtype=np.float64)
        init_cost, _ = compute_reconfiguration_cost(
            sol_array, initial_joints, config.joint_weights,
            config.reconfig_threshold, config.reconfig_penalty, config.max_move_weight
        )
        dp[0][sol_idx] = (init_cost, -1)

    # Forward pass
    for i in range(1, n_viewpoints):
        current_vp = valid_viewpoints[i]
        prev_dp = dp[i - 1]

        if not prev_dp:
            print(f"  Error: No valid path to viewpoint {i}")
            break

        for curr_sol_idx, curr_solution in enumerate(current_vp.safe_ik_solutions):
            curr_joints = np.asarray(curr_solution, dtype=np.float64)
            min_cost = float('inf')
            best_prev_idx = -1

            for prev_sol_idx, (prev_cost, _) in prev_dp.items():
                prev_solution = valid_viewpoints[i - 1].safe_ik_solutions[prev_sol_idx]
                prev_joints = np.asarray(prev_solution, dtype=np.float64)

                transition_cost, _ = compute_reconfiguration_cost(
                    curr_joints, prev_joints, config.joint_weights,
                    config.reconfig_threshold, config.reconfig_penalty, config.max_move_weight
                )
                total_cost = prev_cost + transition_cost

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_prev_idx = prev_sol_idx

            dp[i][curr_sol_idx] = (min_cost, best_prev_idx)

    # Find best final solution
    last_dp = dp[n_viewpoints - 1]
    if not last_dp:
        print("  Error: No valid complete path found!")
        return [], 0.0, []

    best_final_idx = min(last_dp.keys(), key=lambda idx: last_dp[idx][0])
    total_distance = last_dp[best_final_idx][0]

    # Backward pass: reconstruct path
    solution_indices = [0] * n_viewpoints
    solution_indices[n_viewpoints - 1] = best_final_idx

    for i in range(n_viewpoints - 1, 0, -1):
        current_sol_idx = solution_indices[i]
        _, prev_sol_idx = dp[i][current_sol_idx]
        solution_indices[i - 1] = prev_sol_idx

    # Build target list
    targets = []
    for i, sol_idx in enumerate(solution_indices):
        solution = valid_viewpoints[i].safe_ik_solutions[sol_idx]
        targets.append(np.asarray(solution, dtype=np.float64))

    print(f"  → Collected {len(targets)} targets with total cost {total_distance:.4f}")
    return targets, total_distance, solution_indices


def plan_trajectory(
    viewpoint_mgr: ViewpointManager,
    config: SimulationConfig,
    world_state: WorldState
) -> Tuple[List[np.ndarray], List[int]]:
    """Plan joint trajectory using selected method

    Returns:
        Tuple of (joint_targets, solution_indices)
    """
    print(f"\n{'='*60}")
    print(f"TRAJECTORY PLANNING ({config.selection_method.upper()})")
    print(f"{'='*60}\n")

    # Get viewpoints with safe IK
    viewpoints_with_safe = viewpoint_mgr.filter_with_safe_ik()

    if not viewpoints_with_safe:
        raise ValueError("No viewpoints with safe IK solutions!")

    # Select method
    if config.selection_method == "random":
        targets, solution_indices = select_ik_random(viewpoints_with_safe)
    elif config.selection_method == "greedy":
        targets, solution_indices = select_ik_greedy(viewpoints_with_safe)
    elif config.selection_method == "dp":
        targets, _, solution_indices = select_ik_dp(
            viewpoints_with_safe,
            world_state.default_config,
            config
        )
    else:
        raise ValueError(f"Unknown selection method: {config.selection_method}")

    print(f"\n✓ Trajectory planned with {len(targets)} waypoints")
    print(f"{'='*60}\n")

    return targets, solution_indices


# ============================================================================
# File I/O and Analysis
# ============================================================================
def save_joint_trajectory_csv(
    viewpoints: List[Viewpoint],
    joint_targets: List[np.ndarray],
    save_path: str
):
    """Save joint trajectory and target poses to CSV file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    csv_rows = []
    skipped_count = 0

    count = min(len(viewpoints), len(joint_targets))

    for idx in range(count):
        viewpoint = viewpoints[idx]
        joint_config = joint_targets[idx]

        if viewpoint.world_pose is None:
            print(f"Warning: Viewpoint {viewpoint.index} has no world_pose, skipping")
            skipped_count += 1
            continue

        time = float(idx)
        joints = joint_config.flatten().tolist()

        if len(joints) != 6:
            print(f"Warning: Expected 6 joints but got {len(joints)}, skipping")
            skipped_count += 1
            continue

        try:
            pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w = extract_pose_from_matrix(
                viewpoint.world_pose
            )
        except Exception as e:
            print(f"Warning: Failed to extract pose: {e}, skipping")
            skipped_count += 1
            continue

        row = [time, *joints, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        csv_rows.append(row)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'time',
            'ur20-shoulder_pan_joint',
            'ur20-shoulder_lift_joint',
            'ur20-elbow_joint',
            'ur20-wrist_1_joint',
            'ur20-wrist_2_joint',
            'ur20-wrist_3_joint',
            'target-POS_X',
            'target-POS_Y',
            'target-POS_Z',
            'target-ROT_X',
            'target-ROT_Y',
            'target-ROT_Z',
            'target-ROT_W'
        ]
        writer.writerow(header)
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print("JOINT TRAJECTORY CSV SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total rows: {len(csv_rows)}")
    if skipped_count > 0:
        print(f"Rows skipped: {skipped_count}")
    print(f"{'='*60}\n")


def analyze_reconfigurations(csv_path: str, threshold: float, output_dir: str):
    """Analyze joint reconfigurations from CSV file"""
    try:
        joint_data, joint_names = load_joint_trajectory(csv_path)
        results = analyze_joint_reconfigurations(joint_data, threshold=threshold)
        print_analysis_results(results, joint_names)

        input_basename = os.path.splitext(os.path.basename(csv_path))[0]
        results_file = os.path.join(output_dir, f"{input_basename}_reconfig.txt")

        os.makedirs(output_dir, exist_ok=True)

        with open(results_file, 'w') as f:
            f.write("Joint Reconfiguration Analysis Results\n")
            f.write("="*60 + "\n\n")
            f.write(f"Input file: {csv_path}\n")
            f.write(f"Threshold: {threshold:.3f} radians\n")
            f.write(f"Total timesteps: {results['total_timesteps']}\n")
            f.write(f"Total reconfigurations: {results['total_reconfigurations']}\n")
            f.write(f"Reconfiguration rate: {results['reconfiguration_rate']:.1%}\n\n")

            f.write("Per-joint statistics:\n")
            for i, joint_name in enumerate(joint_names):
                f.write(f"{joint_name}: {results['reconfigurations_per_joint'][i]} reconfigurations, "
                       f"max change: {results['max_changes_per_joint'][i]:.3f} rad\n")

            f.write(f"\nAll Reconfiguration Timesteps:\n")
            f.write(f"{'Timestep':<10} {'Max Change':<12} {'Joints Involved':<30}\n")
            f.write("-" * 60 + "\n")
            for reconfig in results['large_reconfig_timesteps']:
                joint_indices = reconfig['joints_involved']
                involved_joints = [joint_names[i].replace('ur20-', '').replace('_joint', '')
                                 for i in joint_indices]
                f.write(f"{reconfig['timestep']:<10} "
                       f"{reconfig['max_change']:<12.3f} "
                       f"{', '.join(involved_joints)}\n")

        print(f"\n{'='*60}")
        print("RECONFIGURATION ANALYSIS SAVED")
        print(f"{'='*60}")
        print(f"Output file: {results_file}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Warning: Failed to analyze reconfigurations: {e}")


def save_ik_solutions_hdf5(
    viewpoints: List[Viewpoint],
    selected_indices: List[int],
    selection_method: str,
    save_path: str,
    tsp_tour_path: str
):
    """Save all IK solutions to HDF5 file"""
    import h5py

    viewpoints_with_safe = [vp for vp in viewpoints if len(vp.safe_ik_solutions) > 0]

    # Create mapping
    selected_map = {}
    if len(selected_indices) == len(viewpoints_with_safe):
        for vp, sol_idx in zip(viewpoints_with_safe, selected_indices):
            selected_map[vp.index] = sol_idx

    num_with_solutions = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    num_with_safe = len(viewpoints_with_safe)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(viewpoints)
        metadata_grp.attrs['num_viewpoints_with_solutions'] = num_with_solutions
        metadata_grp.attrs['num_viewpoints_with_safe_solutions'] = num_with_safe
        metadata_grp.attrs['selection_method'] = selection_method
        metadata_grp.attrs['timestamp'] = datetime.now().isoformat()
        metadata_grp.attrs['tsp_tour_file'] = tsp_tour_path

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

            if vp.index in selected_map:
                vp_grp.attrs['selected_solution_index'] = selected_map[vp.index]
            else:
                vp_grp.attrs['selected_solution_index'] = -1

    print(f"\n{'='*60}")
    print("IK SOLUTIONS HDF5 SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total viewpoints: {len(viewpoints)}")
    print(f"With any solutions: {num_with_solutions}")
    print(f"With safe solutions: {num_with_safe}")
    print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"{'='*60}\n")


def save_results(
    trajectory: Tuple[List[np.ndarray], List[int]],
    viewpoint_mgr: ViewpointManager,
    config: SimulationConfig,
    tsp_result: dict
):
    """Save all analysis results to files"""
    joint_targets, solution_indices = trajectory

    # Determine output directory
    num_points = tsp_result['metadata']['num_points']
    output_dir = f'data/trajectory/{num_points}'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save CSV
    viewpoints_with_safe = viewpoint_mgr.filter_with_safe_ik()
    csv_path = f'{output_dir}/joint_trajectory_{config.selection_method}.csv'
    save_joint_trajectory_csv(viewpoints_with_safe, joint_targets, csv_path)

    # Analyze reconfigurations
    analyze_reconfigurations(csv_path, config.reconfig_threshold, output_dir)

    # Save HDF5
    h5_path = f'{output_dir}/ik_solutions_all_{timestamp}.h5'
    save_ik_solutions_hdf5(
        viewpoint_mgr.viewpoints,
        solution_indices,
        config.selection_method,
        h5_path,
        config.tsp_tour_path
    )


# ============================================================================
# Simulation Loop
# ============================================================================
# generate_interpolated_path() is now imported from common.interpolation_utils


def get_active_joint_positions(robot, idx_list: List[int]) -> np.ndarray:
    """Get current joint positions for active joints"""
    all_positions = robot.get_joint_positions()
    return np.asarray([all_positions[i] for i in idx_list], dtype=np.float64)


def run_simulation(
    world_state: WorldState,
    trajectory: Tuple[List[np.ndarray], List[int]],
    viewpoint_mgr: ViewpointManager,
    config: SimulationConfig,
    tsp_result: dict
):
    """Run Isaac Sim simulation with planned trajectory"""
    joint_targets, _ = trajectory

    print(f"\n{'='*60}")
    print("STARTING SIMULATION")
    print(f"{'='*60}")
    print(f"Total waypoints: {len(joint_targets)}")
    print(f"Interpolation steps: {config.interpolation_steps}")
    print(f"{'='*60}\n")

    # Setup trajectory queue
    target_queue: Deque[np.ndarray] = deque(joint_targets)
    active_trajectory: List[np.ndarray] = []
    trajectory_step = 0

    # History tracking
    history = JointHistoryTracker()

    step_counter = 0
    idle_counter = 0
    viewpoint_counter = 0

    # Sphere visualization
    spheres = None
    tensor_args = TensorDeviceType()

    # Output directory
    num_points = tsp_result['metadata']['num_points']
    output_dir = f'data/output/{num_points}'

    # Main simulation loop
    while simulation_app.is_running():
        world_state.world.step(render=True)

        if not world_state.world.is_playing():
            if idle_counter % 100 == 0:
                print("**** Click Play to start simulation *****")
            idle_counter += 1
            continue

        idle_counter = 0
        step_counter += 1

        # Record current state
        current_joints = get_active_joint_positions(world_state.robot, world_state.idx_list)
        history.record_step(step_counter, current_joints)

        # Visualize robot spheres
        if config.visualize_spheres and step_counter % 2 == 0:
            # Get current joint state from simulator
            sim_js = world_state.robot.get_joints_state()
            sim_js_names = world_state.robot.dof_names

            # Convert to CuRobo joint state
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(world_state.ik_solver.kinematics.joint_names)

            # Get sphere representation
            sph_list = world_state.ik_solver.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # Create spheres
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                # Update sphere positions and radii
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))

        # Execute trajectory
        if active_trajectory and trajectory_step < len(active_trajectory):
            joint_cmd = active_trajectory[trajectory_step]
            world_state.robot.set_joint_positions(joint_cmd.tolist(), world_state.idx_list)
            trajectory_step += 1

            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0

        elif target_queue:
            # Reached viewpoint
            viewpoint_counter += 1
            history.mark_viewpoint(step_counter)

            # Get next target
            next_target = target_queue.popleft()
            current_state = get_active_joint_positions(world_state.robot, world_state.idx_list)

            active_trajectory = generate_interpolated_path(
                current_state,
                next_target,
                config.interpolation_steps
            )
            trajectory_step = 0

            if not active_trajectory:
                active_trajectory = [next_target]

            joint_cmd = active_trajectory[trajectory_step]
            world_state.robot.set_joint_positions(joint_cmd.tolist(), world_state.idx_list)
            trajectory_step += 1

            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0

    print(f"\nSimulation completed!")
    print(f"Viewpoints reached: {viewpoint_counter}")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    # Create configuration
    config = SimulationConfig.from_args(args)

    print(f"\n{'='*60}")
    print("VISION INSPECTION - ROBOT TRAJECTORY PLANNER V3")
    print(f"{'='*60}")
    print(f"TSP tour: {config.tsp_tour_path}")
    print(f"Selection method: {config.selection_method}")
    print(f"Robot config: {config.robot_config_file}")
    print(f"{'='*60}\n")

    # Load TSP tour data (needed for output directory)
    tsp_result = load_tsp_tour(config.tsp_tour_path)

    # Initialize simulation
    world_state = initialize_simulation(config)

    # Process viewpoints (load, IK, collision check)
    viewpoint_mgr = process_viewpoints(config, world_state)

    # Plan trajectory (select IK solutions)
    trajectory = plan_trajectory(viewpoint_mgr, config, world_state)

    # Save results
    save_results(trajectory, viewpoint_mgr, config, tsp_result)

    # Run simulation (if requested)
    if not config.no_sim:
        run_simulation(world_state, trajectory, viewpoint_mgr, config, tsp_result)

    simulation_app.close()


if __name__ == "__main__":
    main()
