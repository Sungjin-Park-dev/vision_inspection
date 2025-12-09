#!/usr/bin/env python3
"""
Compute IK Solutions and Check Collisions (CuRobo only)

This script:
1. Loads viewpoints file (surface positions and normals)
2. Initializes CuRobo IK solver and collision checker (no Isaac Sim needed)
3. Computes IK solutions for each viewpoint using EAIK
4. Checks collision constraints for each IK solution
5. Saves all IK solutions with collision-free flags to HDF5

Usage:
   omni_python scripts/compute_ik_solutions.py --viewpoints data/viewpoint/675/viewpoints.h5
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Third Party Imports
# ============================================================================
import numpy as np
import torch

# ============================================================================
# CuRobo Imports
# ============================================================================
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# ============================================================================
# Local Imports
# ============================================================================
from common import config
from common.cli_utils import print_section_header, print_key_value, print_success
from common.coordinate_utils import normalize_vectors, offset_points_along_normals, transform_pose_to_world
from common.kinematics_utils import quaternion_to_rotation_matrix
from common.world_setup import setup_collision_world
from common.data_io import load_viewpoints_hdf5, save_ik_solutions_hdf5
from common.ik_utils import (
    Viewpoint,
    compute_ik_eaik,
    assign_ik_solutions_to_viewpoints,
    check_ik_solutions_collision,
)


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class ComputeConfig:
    """Configuration for IK computation"""
    # Input/Output
    object_name: Optional[str]
    num_viewpoints: Optional[int]
    viewpoints_path: str
    output_path: Optional[str]
    robot_config_file: str

    # World configuration (consolidated)
    obstacles: config.WorldObstacleConfig = field(default_factory=config.WorldObstacleConfig)

    # IK solver configuration
    ik_rotation_threshold: float = config.IK_ROTATION_THRESHOLD
    ik_position_threshold: float = config.IK_POSITION_THRESHOLD
    ik_num_seeds: int = config.IK_NUM_SEEDS

    # Camera configuration
    normal_sample_offset: float = config.get_camera_working_distance_m()

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'ComputeConfig':
        """Create configuration from command line arguments"""
        return cls(
            object_name=args.object_name,
            num_viewpoints=args.num_viewpoints,
            viewpoints_path=args.viewpoints,
            output_path=args.output,
            robot_config_file=args.robot,
        )


# ============================================================================
# Viewpoint Manager
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

    def count_with_all_ik(self) -> int:
        """Count viewpoints with any IK solutions"""
        return sum(1 for vp in self.viewpoints if len(vp.all_ik_solutions) > 0)

    def count_with_safe_ik(self) -> int:
        """Count viewpoints with collision-free IK solutions"""
        return sum(1 for vp in self.viewpoints if len(vp.safe_ik_solutions) > 0)

    def update_world_poses(self, glass_pose: np.ndarray, debug_first: bool = False):
        """Update world poses for all viewpoints

        Args:
            glass_pose: 4x4 transformation matrix of glass object in world frame
            debug_first: Print debug info for first viewpoint
        """
        for i, viewpoint in enumerate(self.viewpoints):
            if viewpoint.local_pose is None:
                continue
            debug = debug_first and i == 0

            # Transform local pose to world frame
            viewpoint.world_pose = transform_pose_to_world(
                viewpoint.local_pose, glass_pose, debug=debug
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


# ============================================================================
# Viewpoints Loading
# ============================================================================
def load_viewpoints_file(viewpoints_path: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load viewpoints from HDF5 file

    Returns:
        surface_positions: (N, 3) array of surface positions in meters
        surface_normals: (N, 3) array of surface normals (unit vectors)
        metadata: Dictionary with metadata and camera_spec
    """
    return load_viewpoints_hdf5(viewpoints_path)


def create_viewpoints_from_file(
    surface_positions: np.ndarray,
    surface_normals: np.ndarray,
    metadata: dict,
    cfg: ComputeConfig
) -> ViewpointManager:
    """Create ViewpointManager from viewpoints file

    The viewpoints file stores surface positions and normals.
    This function:
    1. Extracts surface points and normals
    2. Determines working distance from camera_spec or config
    3. Offsets surface positions to create camera viewpoints
    4. Creates local pose matrices for each viewpoint

    Args:
        surface_positions: (N, 3) array of surface positions
        surface_normals: (N, 3) array of surface normals
        metadata: Metadata dictionary from viewpoints file
        cfg: Computation configuration

    Returns:
        ViewpointManager with viewpoints
    """
    print_section_header("VIEWPOINTS DATA LOADING", width=60)
    print_key_value("Loaded viewpoints", len(surface_positions))
    print_key_value("Coordinate system", "Z-up")
    print("\nCoordinate ranges:")
    print_key_value("X range", f"[{surface_positions[:, 0].min():.4f}, {surface_positions[:, 0].max():.4f}]")
    print_key_value("Y range", f"[{surface_positions[:, 1].min():.4f}, {surface_positions[:, 1].max():.4f}]")
    print_key_value("Z range", f"[{surface_positions[:, 2].min():.4f}, {surface_positions[:, 2].max():.4f}]")
    print()

    # Determine working distance
    working_distance_m = cfg.normal_sample_offset

    if 'camera_spec' in metadata:
        camera_spec = metadata['camera_spec']
        if 'working_distance_mm' in camera_spec:
            working_distance_m = camera_spec['working_distance_mm'] / 1000.0
            print(f"Using working distance from HDF5: {camera_spec['working_distance_mm']} mm")
        else:
            print(f"⚠️  No working_distance_mm in camera_spec, using default {cfg.normal_sample_offset} m")
    else:
        print(f"Using default working distance: {cfg.normal_sample_offset} m")

    # Generate viewpoint poses
    print(f"\nGenerating viewpoint poses...")
    print(f"  Offsetting by {working_distance_m*1000:.1f} mm along surface normals")

    # Offset surface points to camera positions
    offset_points = offset_points_along_normals(surface_positions, surface_normals, working_distance_m)

    # Camera looks toward surface (negative of surface normal)
    approach_normals = -normalize_vectors(surface_normals)

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
                raise ValueError("Failed to construct orthogonal frame")

        x_axis /= norm_x
        y_axis = np.cross(z_axis, x_axis)

        # Create 4x4 pose matrix
        pose_matrix = np.eye(4, dtype=np.float64)
        pose_matrix[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
        pose_matrix[:3, 3] = position.astype(np.float64)

        viewpoints.append(Viewpoint(index=int(point_idx), local_pose=pose_matrix))

    print(f"Generated {len(viewpoints)} viewpoints")

    return ViewpointManager(
        viewpoints=viewpoints,
        local_points=offset_points,
        local_normals=approach_normals
    )


# ============================================================================
# CuRobo IK Solver Setup
# ============================================================================
def setup_collision_world_for_ik(cfg: ComputeConfig) -> WorldConfig:
    """Setup collision world configuration using common utility

    Args:
        cfg: Computation configuration

    Returns:
        WorldConfig with table, target object, and additional obstacles
    """
    print_section_header("SETTING UP COLLISION WORLD", width=60)

    world_cfg = setup_collision_world(
        **cfg.obstacles.to_world_setup_kwargs(),
        verbose=True
    )

    return world_cfg


def setup_ik_solver(cfg: ComputeConfig, world_cfg: WorldConfig) -> IKSolver:
    """Setup CuRobo IK solver with collision checking

    Args:
        cfg: Computation configuration
        world_cfg: World configuration with obstacles

    Returns:
        Configured IK solver
    """
    print_section_header("INITIALIZING IK SOLVER", width=60)

    # Load robot configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), cfg.robot_config_file))["robot_cfg"]
    print_key_value("Robot config", cfg.robot_config_file)

    # Create tensor device
    tensor_args = TensorDeviceType()

    # Create IK solver config
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=cfg.ik_rotation_threshold,
        position_threshold=cfg.ik_position_threshold,
        num_seeds=cfg.ik_num_seeds,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": config.N_OBSTACLE_CUBOIDS, "mesh": config.N_OBSTACLE_MESH},
    )

    print("\nIK solver configuration:")
    print_key_value("Rotation threshold", f"{cfg.ik_rotation_threshold} rad")
    print_key_value("Position threshold", f"{cfg.ik_position_threshold} m")
    print_key_value("Number of seeds", cfg.ik_num_seeds)
    print_key_value("Self collision check", True)
    print_key_value("Collision checker", "MESH")
    print_key_value("Using CUDA graph", True)

    # Create IK solver
    ik_solver = IKSolver(ik_config)

    print_success("IK solver initialized successfully")
    print()

    return ik_solver


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point - performs same tasks as notebook Section 2"""
    parser = argparse.ArgumentParser(
        description="Compute IK solutions and check collisions for viewpoints"
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Object name for automatic path generation (e.g., 'glass', 'phone'). "
             "If provided with --num_viewpoints, paths will be auto-generated."
    )
    parser.add_argument(
        "--num_viewpoints",
        type=int,
        default=None,
        help="Number of viewpoints (used with --object_name for path generation)"
    )
    parser.add_argument(
        "--viewpoints",
        type=str,
        default=None,
        help="Path to viewpoints HDF5 file (e.g., data/glass/viewpoint/500/viewpoints.h5). "
             "Required if --object_name is not provided."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save IK solutions HDF5 file (default: auto-generate in data/{object_name}/ik/)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="ur20.yml",
        help="Robot configuration file name (default: ur20.yml)"
    )
    args = parser.parse_args()

    # Validate and resolve paths
    if args.object_name is None and args.viewpoints is None:
        parser.error("Either --object_name (with --num_viewpoints) or --viewpoints must be provided")

    if args.object_name and args.num_viewpoints is None:
        parser.error("--num_viewpoints is required when using --object_name")

    # Determine viewpoints path
    if args.object_name:
        if args.viewpoints is None:
            args.viewpoints = str(config.get_viewpoint_path(args.object_name, args.num_viewpoints))
            print(f"Using auto-generated viewpoints path: {args.viewpoints}")

    # Step 1: Create configuration
    cfg = ComputeConfig.from_args(args)

    # Auto-generate mesh file path if using object_name (use source mesh for collision)
    if cfg.obstacles.target_object_mesh_file is None:
        if args.object_name:
            # Use source mesh (full geometry) for collision checking
            cfg.obstacles.target_object_mesh_file = str(config.get_mesh_path(args.object_name, mesh_type="source"))
            print(f"Using auto-generated collision mesh: {cfg.obstacles.target_object_mesh_file}")
            print(f"  → Using 'source' mesh (full geometry for collision checking)")
        else:
            # Fallback to default for backward compatibility
            cfg.obstacles.target_object_mesh_file = config.DEFAULT_MESH_FILE
            print(f"Using default collision mesh: {cfg.obstacles.target_object_mesh_file}")

    # Auto-generate output path if not provided
    if cfg.output_path is None:
        if args.object_name:
            # New structure: data/{object_name}/ik/{num_viewpoints}/ik_solutions.h5
            cfg.output_path = str(config.get_ik_path(args.object_name, args.num_viewpoints))
        else:
            # Fallback to old structure for backward compatibility
            # Extract num_viewpoints from path
            viewpoints_dir = os.path.dirname(cfg.viewpoints_path)
            dataset_name = os.path.basename(viewpoints_dir)
            cfg.output_path = f"data/ik/{dataset_name}/ik_solutions.h5"

    print(f"\n{'='*60}")
    print("COMPUTE IK SOLUTIONS (Section 2)")
    print(f"{'='*60}")
    print_key_value("Viewpoints file", cfg.viewpoints_path)
    print_key_value("Output file", cfg.output_path)
    print_key_value("Robot config", cfg.robot_config_file)
    print(f"{'='*60}\n")

    # Step 2: Load viewpoints from HDF5
    surface_positions, surface_normals, metadata = load_viewpoints_file(cfg.viewpoints_path)

    # Step 3: Setup collision world for IK
    world_cfg = setup_collision_world_for_ik(cfg)

    # Step 4: Setup IK solver
    ik_solver = setup_ik_solver(cfg, world_cfg)

    # Step 5: Create viewpoint manager
    viewpoint_mgr = create_viewpoints_from_file(
        surface_positions, surface_normals, metadata, cfg
    )

    # Step 6: Update world poses (target object pose)
    target_object_world_pose = np.eye(4, dtype=np.float64)
    target_object_world_pose[:3, :3] = quaternion_to_rotation_matrix(cfg.obstacles.target_object_rotation)
    target_object_world_pose[:3, 3] = cfg.obstacles.target_object_position
    viewpoint_mgr.update_world_poses(target_object_world_pose)

    # Step 7: Collect world matrices and compute IK
    print_section_header("COMPUTING IK SOLUTIONS", width=60)
    world_mats, used_indices = viewpoint_mgr.collect_world_matrices()
    print_key_value("Valid viewpoints for IK", len(used_indices))

    ik_results = compute_ik_eaik(world_mats)
    assign_ik_solutions_to_viewpoints(viewpoint_mgr.viewpoints, ik_results, used_indices)

    print_key_value("Viewpoints with solutions", viewpoint_mgr.count_with_all_ik())

    # Step 8: Check collision for IK solutions
    print_section_header("CHECKING COLLISIONS", width=60)
    check_ik_solutions_collision(viewpoint_mgr.viewpoints, ik_solver)
    print_key_value("Viewpoints with safe solutions", viewpoint_mgr.count_with_safe_ik())
    print()

    # Step 9: Save IK solutions to HDF5
    save_ik_solutions_hdf5(viewpoint_mgr.viewpoints, cfg.output_path, cfg.viewpoints_path)

    print_success("✓ Section 2 완료!")


if __name__ == "__main__":
    main()
