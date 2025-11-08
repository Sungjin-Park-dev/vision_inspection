#!/usr/bin/env python3
"""
Compute IK Solutions and Check Collisions (CuRobo only)

This script:
1. Loads TSP tour result (viewpoints in optimized order)
2. Initializes CuRobo IK solver and collision checker (no Isaac Sim needed)
3. Computes IK solutions for each viewpoint using EAIK
4. Checks collision constraints for each IK solution
5. Saves all IK solutions with collision-free flags to HDF5

Usage:
    python scripts/compute_ik_solutions.py \\
        --tsp_tour data/tour/tour_3000.h5 \\
        --output data/ik/ik_solutions_3000.h5 \\
        --robot ur20.yml
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter
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
from common.coordinate_utils import normalize_vectors, offset_points_along_normals
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
    tsp_tour_path: str
    output_path: Optional[str]
    robot_config_file: str

    # World configuration
    table_position: np.ndarray = field(default_factory=lambda: config.TABLE_POSITION.copy())
    table_dimensions: np.ndarray = field(default_factory=lambda: config.TABLE_DIMENSIONS.copy())
    glass_position: np.ndarray = field(default_factory=lambda: config.GLASS_POSITION.copy())
    glass_mesh_file: str = config.DEFAULT_MESH_FILE

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
            tsp_tour_path=args.tsp_tour,
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

    def update_world_poses(self, glass_pose: np.ndarray, debug_first: bool = True):
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
# Utility Functions
# ============================================================================
def transform_pose_to_world(
    local_pose: np.ndarray,
    object_world_pose: np.ndarray,
    debug: bool = False
) -> np.ndarray:
    """Transform local pose to world frame

    Args:
        local_pose: 4x4 pose matrix in object's local frame
        object_world_pose: 4x4 transformation of object in world frame
        debug: Print debug information

    Returns:
        4x4 pose matrix in world frame
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


# ============================================================================
# TSP Tour Loading
# ============================================================================
def load_tsp_tour(tsp_tour_path: str) -> dict:
    """Load TSP tour result from HDF5 file"""
    from tsp_utils import load_tsp_result
    return load_tsp_result(tsp_tour_path)


def create_viewpoints_from_tsp(
    tsp_result: dict, cfg: ComputeConfig
) -> ViewpointManager:
    """Create ViewpointManager from TSP tour result

    The TSP tour file stores surface positions and normals.
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
    print(f"Coordinate system: Z-up")
    print(f"\nCoordinate ranges:")
    print(f"  X: [{tour_coords[:, 0].min():.4f}, {tour_coords[:, 0].max():.4f}]")
    print(f"  Y: [{tour_coords[:, 1].min():.4f}, {tour_coords[:, 1].max():.4f}]")
    print(f"  Z: [{tour_coords[:, 2].min():.4f}, {tour_coords[:, 2].max():.4f}]")
    print(f"{'='*60}\n")

    # Determine working distance
    working_distance_m = cfg.normal_sample_offset

    if 'metadata' in tsp_result and 'camera_spec' in tsp_result['metadata']:
        camera_spec = tsp_result['metadata']['camera_spec']
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
                raise ValueError("Failed to construct orthogonal frame")

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
# CuRobo IK Solver Setup
# ============================================================================
def setup_collision_world(cfg: ComputeConfig) -> WorldConfig:
    """Setup collision world configuration

    Args:
        cfg: Computation configuration

    Returns:
        WorldConfig with table and glass obstacles
    """
    print(f"\n{'='*60}")
    print("SETTING UP COLLISION WORLD")
    print(f"{'='*60}")

    # Load base world config (table)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[:3] = cfg.table_position
    world_cfg_table.cuboid[0].dims[:3] = cfg.table_dimensions

    print(f"Table position: {cfg.table_position}")
    print(f"Table dimensions: {cfg.table_dimensions}")

    # Add glass mesh
    glass_mesh = Mesh(
        name="glass",
        file_path=cfg.glass_mesh_file,
        pose=list(cfg.glass_position) + [1, 0, 0, 0],  # position + quat (w,x,y,z)
    )

    print(f"Glass mesh: {cfg.glass_mesh_file}")
    print(f"Glass position: {cfg.glass_position}")
    print(f"{'='*60}\n")

    # Combine table cuboid and glass mesh
    world_cfg = WorldConfig(
        cuboid=world_cfg_table.cuboid,
        mesh=[glass_mesh]
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
    print(f"\n{'='*60}")
    print("INITIALIZING IK SOLVER")
    print(f"{'='*60}")

    # Load robot configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), cfg.robot_config_file))["robot_cfg"]
    print(f"Robot: {cfg.robot_config_file}")

    # Create tensor device
    tensor_args = TensorDeviceType()

    # Create IK solver config
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

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
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    )

    print(f"IK solver configuration:")
    print(f"  Rotation threshold: {cfg.ik_rotation_threshold}")
    print(f"  Position threshold: {cfg.ik_position_threshold}")
    print(f"  Number of seeds: {cfg.ik_num_seeds}")
    print(f"  Self collision check: True")
    print(f"  Collision checker: MESH")
    print(f"  Using CUDA graph: True")

    # Create IK solver
    ik_solver = IKSolver(ik_config)

    print(f"✓ IK solver initialized successfully")
    print(f"{'='*60}\n")

    return ik_solver


# ============================================================================
# IK Processing
# ============================================================================
def process_viewpoints(
    cfg: ComputeConfig,
    ik_solver: IKSolver
) -> Tuple[ViewpointManager, dict]:
    """Process viewpoints: load TSP, compute IK, check collisions

    Returns:
        Tuple of (viewpoint_manager, tsp_result)
    """
    print(f"\n{'='*60}")
    print("PROCESSING VIEWPOINTS")
    print(f"{'='*60}\n")

    # Load TSP tour
    print("Loading TSP tour...")
    tsp_result = load_tsp_tour(cfg.tsp_tour_path)

    # Create viewpoints from TSP
    viewpoint_mgr = create_viewpoints_from_tsp(tsp_result, cfg)

    # Glass object pose in world frame (identity rotation at glass_position)
    glass_world_pose = np.eye(4, dtype=np.float64)
    glass_world_pose[:3, 3] = cfg.glass_position

    # Update world poses
    viewpoint_mgr.update_world_poses(glass_world_pose)

    # Collect world matrices
    world_mats, used_indices = viewpoint_mgr.collect_world_matrices()

    if world_mats.size == 0:
        raise ValueError("No valid world poses found for viewpoints")

    # Compute IK solutions
    print(f"\nComputing IK solutions for {len(world_mats)} viewpoints...")
    assign_start = perf_counter()
    ik_results = compute_ik_eaik(world_mats)
    assign_ik_solutions_to_viewpoints(viewpoint_mgr.viewpoints, ik_results, used_indices)
    assign_elapsed = perf_counter() - assign_start

    # Check collisions
    print("Checking collision constraints...")
    safe_start = perf_counter()
    check_ik_solutions_collision(viewpoint_mgr.viewpoints, ik_solver)
    safe_elapsed = perf_counter() - safe_start

    print(f"\nTiming:")
    print(f"  IK computation: {assign_elapsed * 1000.0:.2f} ms")
    print(f"  Collision checking: {safe_elapsed * 1000.0:.2f} ms")

    # Print statistics
    total = len(viewpoint_mgr.viewpoints)
    with_all = viewpoint_mgr.count_with_all_ik()
    with_safe = viewpoint_mgr.count_with_safe_ik()
    print(f"\nIK Statistics:")
    print(f"  Total viewpoints: {total}")
    print(f"  With any IK solutions: {with_all}/{total}")
    print(f"  With safe IK solutions: {with_safe}/{total}")
    print(f"{'='*60}\n")

    return viewpoint_mgr, tsp_result


# ============================================================================
# File I/O
# ============================================================================
def save_ik_solutions_hdf5(
    viewpoints: List[Viewpoint],
    save_path: str,
    tsp_tour_path: str
):
    """Save all IK solutions to HDF5 file"""
    import h5py

    num_with_solutions = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    num_with_safe = sum(1 for vp in viewpoints if len(vp.safe_ik_solutions) > 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(viewpoints)
        metadata_grp.attrs['num_viewpoints_with_solutions'] = num_with_solutions
        metadata_grp.attrs['num_viewpoints_with_safe_solutions'] = num_with_safe
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

    print(f"\n{'='*60}")
    print("IK SOLUTIONS SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total viewpoints: {len(viewpoints)}")
    print(f"With any solutions: {num_with_solutions}")
    print(f"With safe solutions: {num_with_safe}")
    print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"{'='*60}\n")


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Compute IK solutions (CuRobo only, no Isaac Sim)")
    parser.add_argument(
        "--tsp_tour",
        type=str,
        required=True,
        help="Path to TSP tour result file (.h5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for IK solutions HDF5 file (default: auto-generated)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="ur20.yml",
        help="Robot configuration file (default: ur20.yml)"
    )
    args = parser.parse_args()

    cfg = ComputeConfig.from_args(args)

    print(f"\n{'='*60}")
    print("COMPUTE IK SOLUTIONS (CuRobo only)")
    print(f"{'='*60}")
    print(f"TSP tour: {cfg.tsp_tour_path}")
    print(f"Robot config: {cfg.robot_config_file}")
    print(f"Mode: CuRobo only (no Isaac Sim simulation)")
    print(f"{'='*60}\n")

    # Setup collision world
    world_cfg = setup_collision_world(cfg)

    # Setup IK solver
    ik_solver = setup_ik_solver(cfg, world_cfg)

    # Process viewpoints
    viewpoint_mgr, tsp_result = process_viewpoints(cfg, ik_solver)

    # Determine output path
    if cfg.output_path is None:
        num_points = tsp_result['metadata']['num_points']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = 'data/ik'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/ik_solutions_{num_points}_{timestamp}.h5'
    else:
        output_path = cfg.output_path

    # Save IK solutions
    save_ik_solutions_hdf5(
        viewpoint_mgr.viewpoints,
        output_path,
        cfg.tsp_tour_path
    )

    print("\n✓ IK computation complete!")


if __name__ == "__main__":
    main()
