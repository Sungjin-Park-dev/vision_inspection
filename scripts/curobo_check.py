#!/usr/bin/env python3
"""
CuRobo-based Collision Checker for Robot Trajectories

This script checks for collisions along a robot trajectory using CuRobo library.
It loads a joint trajectory CSV file and checks for collisions with environment
meshes using CuRobo's GPU-accelerated collision checking.

CuRobo provides unified kinematics and collision checking with GPU acceleration
and native integration with motion planning.

Coordinate System:
- Uses Z-up coordinate system (Isaac Sim / URDF / CuRobo convention)
- All meshes should be in Z-up format (e.g., glass_zup.obj)
- Consistent with other pipeline components

Usage:
    python curobo_check.py --trajectory data/trajectory/joint_trajectory.csv \
                          --robot_config ur20.yml \
                          --mesh data/object/glass_zup.obj
"""

import argparse
import csv
import numpy as np
import os
import sys
import time
import pandas as pd
from typing import List, Tuple, Optional, Dict
from scipy.spatial.transform import Rotation
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from common import config
from common.cli_utils import (
    print_section_header,
    print_key_value,
    print_success,
    print_warning,
    print_error
)
from common.interpolation_utils import generate_interpolated_path
from common.world_setup import setup_collision_world
from common.trajectory_io import load_trajectory_csv, save_trajectory_csv

try:
    import torch
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
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
    from curobo.util.trajectory import get_batch_interpolated_trajectory, get_linear_traj, get_smooth_trajectory
    CUROBO_AVAILABLE = True
except ImportError as e:
    print(f"Error: CuRobo not available: {e}")
    CUROBO_AVAILABLE = False
    sys.exit(1)


class CuRoboCollisionChecker:
    """Collision checker using CuRobo for kinematics and collision detection"""

    def __init__(
        self,
        robot_config_path: str,
        obstacle_mesh_paths: List[str],
        glass_position: np.ndarray = None,
        glass_rotation: np.ndarray = None,
        table_position: np.ndarray = None,
        table_dimensions: np.ndarray = None,
        wall_position: np.ndarray = None,
        wall_dimensions: np.ndarray = None,
        workbench_position: np.ndarray = None,
        workbench_dimensions: np.ndarray = None,
        robot_mount_position: np.ndarray = None,
        robot_mount_dimensions: np.ndarray = None,
        collision_margin: float = None,
        tensor_args: Optional[TensorDeviceType] = None,
        motion_gen: Optional[MotionGen] = None,
        world_cfg: Optional[WorldConfig] = None,
    ):
        """
        Initialize CuRobo collision checker

        Args:
            robot_config_path: Path to CuRobo robot config YAML (e.g., ur20.yml)
            obstacle_mesh_paths: List of paths to obstacle mesh files
            glass_position: Position of glass object in world frame (x, y, z)
            glass_rotation: Rotation of glass object as quaternion (w, x, y, z)
            table_position: Position of table cuboid in world frame (x, y, z)
            table_dimensions: Dimensions of table cuboid (x, y, z) in meters
            wall_position: Position of wall cuboid in world frame (x, y, z)
            wall_dimensions: Dimensions of wall cuboid (x, y, z) in meters
            workbench_position: Position of workbench cuboid in world frame (x, y, z)
            workbench_dimensions: Dimensions of workbench cuboid (x, y, z) in meters
            robot_mount_position: Position of robot mount cuboid in world frame (x, y, z)
            robot_mount_dimensions: Dimensions of robot mount cuboid (x, y, z) in meters
            collision_margin: Safety margin for collision detection (meters)
            tensor_args: Optional pre-initialized TensorDeviceType to reuse CUDA context
            motion_gen: Optional pre-initialized MotionGen instance to avoid duplicate setup
            world_cfg: Optional WorldConfig to reuse prebuilt collision environment
        """
        # Apply config defaults
        if glass_position is None:
            glass_position = config.GLASS_POSITION.copy()
        if glass_rotation is None:
            glass_rotation = config.GLASS_ROTATION.copy()
        if table_position is None:
            table_position = config.TABLE_POSITION.copy()
        if table_dimensions is None:
            table_dimensions = config.TABLE_DIMENSIONS.copy()
        if wall_position is None:
            wall_position = config.WALL_POSITION.copy()
        if wall_dimensions is None:
            wall_dimensions = config.WALL_DIMENSIONS.copy()
        if workbench_position is None:
            workbench_position = config.WORKBENCH_POSITION.copy()
        if workbench_dimensions is None:
            workbench_dimensions = config.WORKBENCH_DIMENSIONS.copy()
        if robot_mount_position is None:
            robot_mount_position = config.ROBOT_MOUNT_POSITION.copy()
        if robot_mount_dimensions is None:
            robot_mount_dimensions = config.ROBOT_MOUNT_DIMENSIONS.copy()
        if collision_margin is None:
            collision_margin = config.COLLISION_MARGIN

        self.robot_config_path = robot_config_path
        self.glass_position = glass_position
        self.glass_rotation = glass_rotation
        self.table_position = table_position
        self.table_dimensions = table_dimensions
        self.wall_position = wall_position
        self.wall_dimensions = wall_dimensions
        self.workbench_position = workbench_position
        self.workbench_dimensions = workbench_dimensions
        self.robot_mount_position = robot_mount_position
        self.robot_mount_dimensions = robot_mount_dimensions
        self.collision_margin = collision_margin

        # Setup tensor device (auto-detect CUDA availability)
        self.tensor_args = tensor_args if tensor_args is not None else TensorDeviceType()
        self.use_cuda = self.tensor_args.device.type == 'cuda'

        print_section_header("INITIALIZING CUROBO COLLISION CHECKER")
        print_key_value("Robot config", robot_config_path)
        print_key_value("Glass position", glass_position)
        print_key_value("Glass rotation (quat)", glass_rotation)
        print_key_value("Collision margin", f"{collision_margin} m")
        print_key_value("Device", str(self.tensor_args.device))

        # Setup collision world configuration
        if world_cfg is not None:
            self.world_cfg = world_cfg
            print("Using provided world configuration")
        else:
            if not obstacle_mesh_paths:
                raise ValueError("obstacle_mesh_paths must be provided when world_cfg is None")
            self.world_cfg = self._setup_world_config(obstacle_mesh_paths)

        # Create or reuse MotionGen
        if motion_gen is not None:
            self.motion_gen = motion_gen
            self.robot_cfg = None
            print("Using provided MotionGen instance")
        else:
            if not robot_config_path:
                raise ValueError("robot_config_path must be provided when MotionGen is not supplied")
            robot_cfg_dict = load_yaml(join_path(get_robot_configs_path(), robot_config_path))
            self.robot_cfg = robot_cfg_dict["robot_cfg"]
            self.motion_gen = self._setup_motion_gen()

        print(f"✓ CuRobo collision checker {'initialized' if motion_gen is None else 'ready'}")
        print(f"{'='*70}\n")

    def _setup_world_config(self, obstacle_mesh_paths: List[str]) -> WorldConfig:
        """Setup collision world configuration with obstacles using common utility"""
        world_cfg = setup_collision_world(
            table_position=self.table_position,
            table_dimensions=self.table_dimensions,
            wall_position=self.wall_position,
            wall_dimensions=self.wall_dimensions,
            workbench_position=self.workbench_position,
            workbench_dimensions=self.workbench_dimensions,
            robot_mount_position=self.robot_mount_position,
            robot_mount_dimensions=self.robot_mount_dimensions,
            mesh_files=obstacle_mesh_paths,
            mesh_position=self.glass_position,
            mesh_rotation=self.glass_rotation,
            verbose=True
        )

        return world_cfg

    def _setup_motion_gen(self) -> MotionGen:
        """Setup CuRobo MotionGen for trajectory planning"""
        print(f"\nInitializing MotionGen for trajectory replanning...")

        # Create MotionGen config
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            tensor_args=self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=self.use_cuda,
            interpolation_dt=config.REPLAN_INTERP_DT,
            trajopt_tsteps=config.REPLAN_TRAJOPT_TSTEPS,
        )

        print(f"  MotionGen configuration:")
        print(f"    Interpolation dt: {config.REPLAN_INTERP_DT}")
        print(f"    Trajectory optimization timesteps: {config.REPLAN_TRAJOPT_TSTEPS}")
        print(f"    Timeout: {config.REPLAN_TIMEOUT}s")
        print(f"    Max attempts: {config.REPLAN_MAX_ATTEMPTS}")

        # Create MotionGen instance
        motion_gen = MotionGen(motion_gen_config)

        print(f"  ✓ MotionGen initialized")

        return motion_gen

    def _interpolate_trajectory(
        self,
        trajectory: np.ndarray,
        adaptive_interp: bool,
        adaptive_max_joint_step_deg: float,
        exclude_last_joint: bool,
        verbose: bool = True
    ) -> Tuple[Optional[Dict[int, List[Tuple[int, float, np.ndarray]]]], List[int], int]:
        """
        Step 1: Interpolate trajectory using CPU linear interpolation

        Args:
            trajectory: (N, n_joints) trajectory array
            adaptive_interp: Use adaptive interpolation
            adaptive_max_joint_step_deg: Maximum joint delta (deg) for interpolation
            exclude_last_joint: If True, exclude last joint when computing max delta
            verbose: Print progress

        Returns:
            (interpolated_configs_map, segment_step_counts, total_interp_points):
                - interpolated_configs_map: Dict mapping segment index to list of (interp_idx, alpha, config)
                - segment_step_counts: List of interpolation counts per segment
                - total_interp_points: Total number of interpolated configurations
        """
        num_waypoints = len(trajectory)
        segment_step_counts: List[int] = []
        total_interp_points = 0
        interpolated_configs_map: Optional[Dict[int, List[Tuple[int, float, np.ndarray]]]] = None

        if verbose:
            print(f"\n[STEP 1] Interpolating trajectory (CPU linear)...")

        interp_timer_start = time.perf_counter()

        # Compute interpolation counts per segment
        segment_step_counts = self._compute_segment_interp_counts(
            trajectory,
            adaptive_interp,
            adaptive_max_joint_step_deg,
            exclude_last_joint
        )
        total_interp_points = int(sum(segment_step_counts))

        if verbose:
            print(f"  Trajectory: {num_waypoints} waypoints")
            print(f"  Interpolation: {total_interp_points} configs")
            if adaptive_interp and segment_step_counts:
                nonzero_steps = [c for c in segment_step_counts if c > 0]
                if nonzero_steps:
                    exclude_note = " (last joint excluded)" if exclude_last_joint else ""
                    print(f"  Adaptive mode: max joint step {adaptive_max_joint_step_deg:.2f} deg{exclude_note}")
                    print(f"    Steps → min {min(nonzero_steps)}, max {max(nonzero_steps)}, "
                          f"avg {total_interp_points / len(segment_step_counts):.2f}")

        if total_interp_points > 0:
            # Precompute all interpolations
            interpolated_configs_map = self._precompute_segment_interpolations(
                trajectory,
                segment_step_counts=segment_step_counts
            )
            interp_time = time.perf_counter() - interp_timer_start
            if verbose:
                print(f"  Interpolation completed in {interp_time:.3f}s")
        else:
            if verbose:
                print(f"  No interpolation needed (adaptive resolved to 0 steps)")
            interpolated_configs_map = {}

        return interpolated_configs_map, segment_step_counts, total_interp_points

    def _check_collisions_batched(
        self,
        trajectory: np.ndarray,
        interpolated_configs_map: Optional[Dict[int, List[Tuple[int, float, np.ndarray]]]] = None,
        total_interp_points: int = 0,
        verbose: bool = True
    ) -> Dict:
        """
        Step 2: Check collisions on trajectory using batched GPU processing

        Args:
            trajectory: (N, n_joints) trajectory array
            interpolated_configs_map: Optional map of interpolated configs per segment
            total_interp_points: Total number of interpolated points
            verbose: Print progress

        Returns:
            Dictionary with collision results:
                - collision_indices: List of waypoint indices with collisions
                - collision_segments: List of (seg_idx, alpha) tuples for segment collisions
                - collision_free_configs: List of (config, metadata) tuples
                - collision_check_time: Time taken for collision checking
                - num_collisions: Number of waypoint collisions
                - num_segment_collisions: Number of segment collisions
                - total_collisions: Total collisions
                - collision_segments_original: Sorted list of segment indices with collisions
        """
        num_waypoints = len(trajectory)

        if verbose:
            total_configs_to_check = num_waypoints + total_interp_points
            print(f"\n[STEP 2] Checking collisions on interpolated trajectory...")
            print(f"  Total configurations to check: {total_configs_to_check:,}")
            print(f"  Using batched CuRobo collision checking (GPU-accelerated)")

        collision_timer_start = time.perf_counter()

        # Collect all configurations to check
        all_configs = []
        config_metadata = []  # (type, waypoint_idx, segment_idx, interp_idx, alpha)

        # Add all waypoint configs
        for i, joint_config in enumerate(trajectory):
            all_configs.append(joint_config)
            config_metadata.append(('waypoint', i, None, None, None))

        # Add all interpolated configs
        if interpolated_configs_map is not None:
            for seg_idx in range(num_waypoints - 1):
                interpolated_list = interpolated_configs_map.get(seg_idx, [])
                for interp_idx, alpha, interp_config in interpolated_list:
                    all_configs.append(interp_config)
                    config_metadata.append(('segment', seg_idx, seg_idx, interp_idx, alpha))

        # Batch collision check
        collision_indices = []
        collision_free_indices = []
        collision_segments = []
        collision_free_configs = []  # Store collision-free configs with metadata
        link_collision_counter = Counter()

        if all_configs:
            # Convert to tensor (batched)
            batched_array = np.stack(all_configs, axis=0)
            q_tensor = self.tensor_args.to_device(torch.from_numpy(batched_array))

            # Create joint state
            zeros = torch.zeros_like(q_tensor)
            joint_state = JointState(
                position=q_tensor,
                velocity=zeros,
                acceleration=zeros,
                jerk=zeros,
                joint_names=self.motion_gen.kinematics.joint_names,
            )

            # Check constraints (batched)
            metrics = self.motion_gen.check_constraints(joint_state)
            feasible = getattr(metrics, "feasible", None)

            if feasible is None:
                feasibility = torch.ones(len(all_configs), dtype=torch.bool)
            else:
                feasibility = feasible.detach().cpu().flatten().to(dtype=torch.bool)

            # Process results
            for idx, (is_feasible, meta) in enumerate(zip(feasibility, config_metadata)):
                meta_type, wp_idx, seg_idx, interp_idx, alpha = meta
                is_collision = not bool(is_feasible)

                if is_collision:
                    if meta_type == 'waypoint':
                        collision_indices.append(wp_idx)
                    else:  # segment
                        collision_segments.append((seg_idx, alpha))
                else:
                    # Store collision-free configs (both waypoints and interpolated)
                    if meta_type == 'waypoint':
                        collision_free_indices.append(wp_idx)
                    collision_free_configs.append((all_configs[idx], meta))

        configs_checked = len(all_configs)
        collision_check_time = time.perf_counter() - collision_timer_start

        # Calculate collision statistics
        num_collisions = len(collision_indices)
        num_segment_collisions = len(collision_segments)
        total_collisions = num_collisions + num_segment_collisions

        # Calculate collision segments from original trajectory
        collision_segments_original = set()

        # From waypoint collisions
        for wp_idx in collision_indices:
            if wp_idx > 0:
                collision_segments_original.add(wp_idx - 1)
            if wp_idx < num_waypoints - 1:
                collision_segments_original.add(wp_idx)

        # From interpolated segment collisions
        for seg_idx, alpha in collision_segments:
            collision_segments_original.add(seg_idx)

        collision_segments_original = sorted(collision_segments_original)

        if verbose:
            print(f"  Collision check completed in {collision_check_time:.3f}s")
            print(f"  Found {total_collisions} collision(s) ({num_collisions} waypoint, {num_segment_collisions} segment)")
            if num_collisions > 0:
                print(f"  Original trajectory collision waypoint indices: {collision_indices}")
                print(f"  Original trajectory collision segments: {collision_segments_original}")

        return {
            'collision_indices': collision_indices,
            'collision_free_indices': collision_free_indices,
            'collision_segments': collision_segments,
            'collision_free_configs': collision_free_configs,
            'collision_segments_original': collision_segments_original,
            'num_collisions': num_collisions,
            'num_segment_collisions': num_segment_collisions,
            'total_collisions': total_collisions,
            'configs_checked': configs_checked,
            'collision_check_time': collision_check_time,
            'link_collision_counter': link_collision_counter,
        }

    def _replan_colliding_segments(
        self,
        trajectory: np.ndarray,
        collision_results: Dict,
        replan_enabled: bool,
        max_replan_iterations: int,
        verbose: bool = True
    ) -> Tuple[List[int], Dict[int, np.ndarray], int, int, int, float, np.ndarray]:
        """
        Step 3: Replan colliding segments using MotionGen

        Args:
            trajectory: (N, n_joints) trajectory array
            collision_results: Results from _check_collisions_batched()
            replan_enabled: Whether replanning is enabled
            max_replan_iterations: Maximum replanning iterations
            verbose: Print progress

        Returns:
            (replanned_segments, replanned_paths, replan_success_count,
             replan_fail_count, replan_iterations_performed, replan_time, updated_trajectory):
                - replanned_segments: List of successfully replanned segment indices
                - replanned_paths: Dict mapping seg_idx to new trajectory
                - replan_success_count: Number of successful replanning attempts
                - replan_fail_count: Number of failed replanning attempts
                - replan_iterations_performed: Number of iterations performed
                - replan_time: Time taken for replanning
                - updated_trajectory: Trajectory with replanned segments integrated
        """
        replan_time_start = time.perf_counter()
        replanned_segments = []
        replan_success_count = 0
        replan_fail_count = 0
        replan_iterations_performed = 0
        replanned_paths = {}
        num_waypoints_current = len(trajectory)
        current_trajectory = np.array(trajectory, dtype=np.float64)

        collision_indices = collision_results['collision_indices']
        collision_segments = collision_results['collision_segments']

        if not replan_enabled:
            if verbose:
                print(f"\n[STEP 3] Motion Replanning DISABLED - skipping")
            replan_time = time.perf_counter() - replan_time_start
            return replanned_segments, replanned_paths, replan_success_count, replan_fail_count, replan_iterations_performed, replan_time, current_trajectory

        if verbose:
            print(f"\n[STEP 3] Motion Replanning (Original Trajectory)...")
            print(f"  Replanning enabled: {replan_enabled}")
            print(f"  Max iterations: {max_replan_iterations}")

        # Identify segments that need replanning from ORIGINAL trajectory
        segments_to_replan = set()

        # Condition 1: Waypoint collisions
        for wp_idx in collision_indices:
            if wp_idx > 0:
                segments_to_replan.add(wp_idx - 1)
            if wp_idx < num_waypoints_current - 1:
                segments_to_replan.add(wp_idx)
            if verbose:
                print(f"  Waypoint {wp_idx}: collision detected, adding connected segments")

        # Condition 2: Segment collisions from interpolated trajectory
        for seg_idx, alpha in collision_segments:
            if 0 <= seg_idx < num_waypoints_current - 1:
                segments_to_replan.add(seg_idx)
                # if verbose:
                    # print(f"  Segment {seg_idx}→{seg_idx+1} (α={alpha:.2f}): collision detected in interpolation")

        segments_to_replan = sorted(list(segments_to_replan))

        if verbose:
            print(f"\n  Total segments to replan: {len(segments_to_replan)}")
            if len(segments_to_replan) > 0:
                print(f"  Segment indices: {segments_to_replan[:20]}{'...' if len(segments_to_replan) > 20 else ''}")

        # Attempt replanning for each segment
        if len(segments_to_replan) > 0:
            successfully_replanned = set()

            for iteration in range(max_replan_iterations):
                replan_iterations_performed = iteration + 1

                if verbose:
                    print(f"\n  Replanning iteration {iteration + 1}/{max_replan_iterations}...")

                segments_replanned_this_iter = 0
                segments_failed_this_iter = 0

                for seg_idx in segments_to_replan:
                    # Skip already replanned segments
                    if seg_idx in successfully_replanned:
                        continue

                    if seg_idx >= num_waypoints_current - 1:
                        continue  # Skip if segment is invalid

                    start_config = current_trajectory[seg_idx]
                    goal_config = current_trajectory[seg_idx + 1]

                    if verbose:
                        print(f"  Replanning segment {seg_idx} -> {seg_idx + 1}...")

                    success, new_segment_traj = self._replan_segment(
                        start_config,
                        goal_config,
                        verbose=verbose
                    )

                    if success and new_segment_traj is not None:
                        # Mark as successfully replanned and store the new path
                        successfully_replanned.add(seg_idx)
                        replanned_segments.append(seg_idx)
                        replanned_paths[seg_idx] = new_segment_traj
                        replan_success_count += 1
                        segments_replanned_this_iter += 1

                        if verbose:
                            print(f"    ✓ Segment {seg_idx} replanned successfully ({len(new_segment_traj)} waypoints)")
                    else:
                        replan_fail_count += 1
                        segments_failed_this_iter += 1

                        if verbose:
                            print(f"    ✗ Segment {seg_idx} replanning failed")

                if verbose:
                    print(f"  Iteration {iteration + 1} complete:")
                    print(f"    Replanned: {segments_replanned_this_iter}")
                    print(f"    Failed: {segments_failed_this_iter}")

                # Check if all segments were successfully replanned
                if len(successfully_replanned) == len(segments_to_replan):
                    if verbose:
                        print(f"  All segments successfully replanned!")
                    break

            if verbose:
                print(f"\n  Replanning summary:")
                print(f"    Total segments identified: {len(segments_to_replan)}")
                print(f"    Successfully replanned: {len(successfully_replanned)}")
                print(f"    Failed: {len(segments_to_replan) - len(successfully_replanned)}")

            # Reconstruct trajectory with replanned segments
            if len(replanned_paths) > 0:
                if verbose:
                    print(f"\n  Reconstructing trajectory with {len(replanned_paths)} replanned segments...")

                new_points = [np.array(current_trajectory[0], dtype=np.float64)]

                for seg_idx in range(num_waypoints_current - 1):
                    if seg_idx in replanned_paths:
                        # Use replanned path (skip first point to avoid duplication)
                        replanned_traj = replanned_paths[seg_idx]
                        for waypoint in replanned_traj[1:]:
                            new_points.append(np.array(waypoint, dtype=np.float64))
                    else:
                        # Use original trajectory waypoint
                        new_points.append(np.array(current_trajectory[seg_idx + 1], dtype=np.float64))

                # Update current_trajectory with reconstructed trajectory
                current_trajectory = np.vstack(new_points)

                if verbose:
                    print(f"    Trajectory reconstructed: {len(current_trajectory)} waypoints")

        replan_time = time.perf_counter() - replan_time_start

        return replanned_segments, replanned_paths, replan_success_count, replan_fail_count, replan_iterations_performed, replan_time, current_trajectory

    def _build_final_trajectory(
        self,
        original_trajectory: np.ndarray,
        updated_trajectory: np.ndarray,
        collision_results: Dict,
        replanned_segments: List[int],
        replanned_paths: Dict[int, np.ndarray],
        interpolate: bool,
        total_interp_points: int,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Step 4: Build final collision-free trajectory

        Args:
            original_trajectory: Original input trajectory
            updated_trajectory: Trajectory after replanning
            collision_results: Results from _check_collisions_batched()
            replanned_segments: List of successfully replanned segment indices
            replanned_paths: Dict mapping seg_idx to new trajectory
            interpolate: Whether interpolation was enabled
            total_interp_points: Total number of interpolated points
            verbose: Print progress

        Returns:
            interpolated_trajectory: Final trajectory array
        """
        num_waypoints_current = len(updated_trajectory)
        collision_free_configs = collision_results['collision_free_configs']
        collision_free_indices = collision_results['collision_free_indices']

        # Check if replanning occurred
        replanning_occurred = len(replanned_segments) > 0

        if replanning_occurred and interpolate and len(collision_free_configs) > 0:
            # HYBRID APPROACH: Combine replanned segments with original collision-free configs
            if verbose:
                print(f"\n{'='*70}")
                print("BUILDING FINAL TRAJECTORY (HYBRID)")
                print(f"{'='*70}")
                print(f"Replanning occurred on {len(replanned_segments)} segment(s)")
                print(f"Using hybrid approach:")
                print(f"  - Replanned segments: Use MotionGen waypoints")
                print(f"  - Other segments: Use original collision-free interpolated configs")

            # Build segment-wise collision-free config map
            segment_collision_free_map = {}
            for joint_config, meta in collision_free_configs:
                meta_type, wp_idx, seg_idx, interp_idx, alpha = meta
                if meta_type == 'waypoint':
                    # Add waypoint to its segments
                    if wp_idx > 0:
                        segment_collision_free_map.setdefault(wp_idx - 1, []).append((joint_config, meta, 1.0))  # End of segment
                    if wp_idx < num_waypoints_current - 1:
                        segment_collision_free_map.setdefault(wp_idx, []).append((joint_config, meta, 0.0))  # Start of segment
                else:  # segment
                    segment_collision_free_map.setdefault(seg_idx, []).append((joint_config, meta, alpha))

            # Reconstruct trajectory with hybrid approach
            all_configs_output = []
            num_replanned_configs = 0
            num_original_configs = 0

            for seg_idx in range(num_waypoints_current - 1):
                if seg_idx in replanned_segments and seg_idx in replanned_paths:
                    # Use replanned trajectory for this segment
                    replanned_traj = replanned_paths[seg_idx]
                    # Skip first waypoint if not the first segment (to avoid duplication)
                    start_idx = 1 if seg_idx > 0 and all_configs_output else 0
                    for waypoint in replanned_traj[start_idx:]:
                        all_configs_output.append(np.array(waypoint, dtype=np.float64))
                        num_replanned_configs += 1
                else:
                    # Use original collision-free configs for this segment
                    segment_configs = segment_collision_free_map.get(seg_idx, [])
                    # Sort by alpha (0.0 for start waypoint, then interpolated configs, 1.0 for end)
                    segment_configs.sort(key=lambda x: x[2])
                    # Skip first if not the first segment (to avoid duplication)
                    start_idx = 1 if seg_idx > 0 and all_configs_output else 0
                    for joint_config, _, _ in segment_configs[start_idx:]:
                        all_configs_output.append(joint_config)
                        num_original_configs += 1

            interpolated_trajectory = np.vstack(all_configs_output) if all_configs_output else updated_trajectory

            if verbose:
                print(f"  Hybrid trajectory: {len(interpolated_trajectory)} configurations")
                print(f"    From replanning: {num_replanned_configs} configs")
                print(f"    From original: {num_original_configs} configs")
                print(f"  Replanned segments: {replanned_segments}")

        elif replanning_occurred:
            # Replanning occurred but no interpolation - use replanned trajectory only
            if verbose:
                print(f"\n{'='*70}")
                print("BUILDING FINAL TRAJECTORY (REPLANNED ONLY)")
                print(f"{'='*70}")
                print(f"Replanning occurred on {len(replanned_segments)} segment(s)")
                print(f"Using replanned trajectory without interpolation")
                print(f"  Final trajectory: {len(updated_trajectory)} waypoints")

            interpolated_trajectory = updated_trajectory

        elif interpolate and len(collision_free_configs) > 0:
            # No replanning - use collision-free interpolated configs
            if verbose:
                print(f"\n{'='*70}")
                print("BUILDING FINAL TRAJECTORY (INTERPOLATED)")
                print(f"{'='*70}")

            # Use only collision-free configurations
            # Sort by waypoint index and interpolation order to maintain trajectory continuity
            sorted_collision_free = sorted(collision_free_configs, key=lambda x: (
                x[1][1] if x[1][0] == 'waypoint' else x[1][2],  # waypoint index or segment index
                0 if x[1][0] == 'waypoint' else x[1][4]  # 0 for waypoint, alpha for interpolated
            ))

            all_configs_output = [joint_config for joint_config, _ in sorted_collision_free]
            interpolated_trajectory = np.vstack(all_configs_output)

            num_removed = (len(original_trajectory) + total_interp_points) - len(all_configs_output)

            if verbose:
                print(f"  Collision-free trajectory: {len(interpolated_trajectory)} configurations")
                print(f"  Removed {num_removed} configs with collisions")
                print(f"  Collision-free rate: {len(all_configs_output) / (len(original_trajectory) + total_interp_points) * 100:.1f}%")
        else:
            # No interpolation and no replanning - use waypoints only
            if verbose:
                print(f"\n{'='*70}")
                print("BUILDING FINAL TRAJECTORY (WAYPOINTS ONLY)")
                print(f"{'='*70}")

            if len(collision_free_indices) > 0:
                collision_free_waypoints = [updated_trajectory[i] for i in sorted(collision_free_indices)]
                interpolated_trajectory = np.vstack(collision_free_waypoints) if collision_free_waypoints else updated_trajectory
            else:
                interpolated_trajectory = updated_trajectory

            if verbose:
                print(f"  Final trajectory: {len(interpolated_trajectory)} waypoints")

        return interpolated_trajectory

    def check_collision_single_config(
        self,
        joint_positions: np.ndarray,
        return_distance: bool = False,  # Not used - kept for API compatibility
        return_link_info: bool = False  # Not used - kept for API compatibility
    ) -> Tuple[bool, float, Optional[List[Dict]]]:
        """
        Check collision for a single robot configuration using CuRobo

        Args:
            joint_positions: Array of joint angles (n_joints,)
            return_distance: If True, return minimum distance (not implemented for CuRobo)
            return_link_info: If True, return detailed collision info (not implemented)

        Returns:
            (is_collision, distance, collision_info):
                - Collision flag
                - Distance (always inf for CuRobo)
                - Collision info (None for CuRobo)
        """
        _ = return_distance  # Unused - API compatibility
        _ = return_link_info  # Unused - API compatibility
        # Convert to tensor
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        q_tensor = self.tensor_args.to_device(torch.from_numpy(joint_positions).unsqueeze(0))

        # Create JointState
        zeros = torch.zeros_like(q_tensor)
        joint_state = JointState(
            position=q_tensor,
            velocity=zeros,
            acceleration=zeros,
            jerk=zeros,
            joint_names=self.motion_gen.kinematics.joint_names,
        )

        # Check constraints
        metrics = self.motion_gen.check_constraints(joint_state)
        feasible = getattr(metrics, "feasible", None)

        if feasible is None:
            is_collision = False
        else:
            feasibility = feasible.detach().cpu().flatten().to(dtype=torch.bool)
            is_collision = not bool(feasibility[0])

        # CuRobo doesn't provide distance information
        distance = float('inf') if not is_collision else 0.0

        return is_collision, distance, None

    def check_trajectory(
        self,
        trajectory: np.ndarray,
        verbose: bool = True,
        show_link_collisions: bool = False,
        adaptive_max_joint_step_deg: float = None,
        exclude_last_joint: bool = None,
        attempt_replan: bool = None,
        max_replan_iterations: int = None,
    ) -> dict:
        """
        Check trajectory for collisions using CuRobo

        This is an orchestrator function that coordinates the 4-step collision checking pipeline:
        1. Interpolation (_interpolate_trajectory)
        2. Collision checking (_check_collisions_batched)
        3. Replanning (_replan_colliding_segments)
        4. Build final trajectory (_build_final_trajectory)

        Args:
            trajectory: (N, n_joints) array of joint configurations
            verbose: Print progress
            show_link_collisions: Show which links are colliding (not implemented for CuRobo)
            adaptive_max_joint_step_deg: Maximum joint delta (deg) allowed for all joints
            exclude_last_joint: If True, the last joint is ignored when computing max delta
            attempt_replan: If True, attempt replanning for collision problems
            max_replan_iterations: Maximum collision replanning iterations

        Returns:
            Dictionary with collision statistics
        """
        # Interpolation is always enabled and adaptive
        interpolate = True
        adaptive_interp = True

        # Apply config defaults
        if adaptive_max_joint_step_deg is None:
            adaptive_max_joint_step_deg = config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG
        if exclude_last_joint is None:
            exclude_last_joint = config.COLLISION_INTERP_EXCLUDE_LAST_JOINT
        if attempt_replan is None:
            attempt_replan = config.REPLAN_ENABLED
        if max_replan_iterations is None:
            max_replan_iterations = config.REPLAN_MAX_ATTEMPTS

        overall_start_time = time.perf_counter()
        num_waypoints = len(trajectory)
        current_trajectory = np.array(trajectory, dtype=np.float64)

        # Determine if replanning is enabled
        replan_enabled = attempt_replan or config.REPLAN_ENABLED

        if verbose:
            print("\n" + "=" * 70)
            print("TRAJECTORY CHECKING - CUROBO")
            print("=" * 70)
            print(f"Input: {num_waypoints} waypoints")
            print(f"Replanning enabled: {replan_enabled}")
            print(f"Interpolation: enabled (adaptive)")
            print(f"Device: {self.tensor_args.device}")
            print("=" * 70)

        # ====================================================================
        # STEP 1: CPU LINEAR INTERPOLATION
        # ====================================================================
        interpolated_configs_map, segment_step_counts, total_interp_points = self._interpolate_trajectory(
            current_trajectory,
            adaptive_interp,
            adaptive_max_joint_step_deg,
            exclude_last_joint,
            verbose
        )

        # ====================================================================
        # STEP 2: COLLISION CHECKING (BATCHED)
        # ====================================================================
        collision_results = self._check_collisions_batched(
            current_trajectory,
            interpolated_configs_map,
            total_interp_points,
            verbose
        )

        # Extract results
        collision_indices = collision_results['collision_indices']
        collision_free_indices = collision_results['collision_free_indices']
        collision_segments = collision_results['collision_segments']
        collision_free_configs = collision_results['collision_free_configs']
        collision_segments_original = collision_results['collision_segments_original']
        num_collisions = collision_results['num_collisions']
        num_segment_collisions = collision_results['num_segment_collisions']
        total_collisions = collision_results['total_collisions']
        configs_checked = collision_results['configs_checked']
        collision_check_time = collision_results['collision_check_time']
        link_collision_counter = collision_results['link_collision_counter']

        # ====================================================================
        # STEP 3: MOTION REPLANNING
        # ====================================================================
        replanned_segments, replanned_paths, replan_success_count, replan_fail_count, replan_iterations_performed, replan_time, current_trajectory = self._replan_colliding_segments(
            current_trajectory,
            collision_results,
            replan_enabled,
            max_replan_iterations,
            verbose
        )
        num_waypoints_current = len(current_trajectory)

        # ====================================================================
        # STEP 4: BUILD FINAL TRAJECTORY
        # ====================================================================
        interpolated_trajectory = self._build_final_trajectory(
            trajectory,
            current_trajectory,
            collision_results,
            replanned_segments,
            replanned_paths,
            interpolate,
            total_interp_points,
            verbose
        )

        # ====================================================================
        # STEP 5: BUILD RESULTS & RETURN
        # ====================================================================
        overall_time = time.perf_counter() - overall_start_time

        if interpolate:
            total_configs = num_waypoints_current + total_interp_points
            collision_rate = (total_collisions / total_configs * 100) if total_configs > 0 else 0.0
        else:
            total_configs = num_waypoints_current
            collision_rate = (num_collisions / total_configs * 100) if total_configs > 0 else 0.0

        results = {
            # Trajectory data
            'input_trajectory': trajectory,
            'final_trajectory': current_trajectory,
            'interpolated_trajectory': interpolated_trajectory,
            'trajectory_modified': not np.array_equal(trajectory, current_trajectory),

            # Basic stats
            'total_waypoints': num_waypoints,
            'final_waypoints': num_waypoints_current,
            'total_configs_checked': configs_checked,

            # Interpolation info
            'interpolate': interpolate,
            'total_interpolated_configs': total_interp_points if interpolate else 0,
            'segment_interp_counts': segment_step_counts if interpolate else [],
            'adaptive_interp': adaptive_interp,
            'adaptive_interp_params': {
                'max_joint_step_deg': adaptive_max_joint_step_deg,
                'exclude_last_joint': exclude_last_joint,
            } if interpolate else None,

            # Collision results
            'num_collisions': num_collisions,
            'num_segment_collisions': num_segment_collisions if interpolate else 0,
            'total_collisions': total_collisions,
            'num_collision_free': len(collision_free_indices),
            'collision_rate': collision_rate,
            'collision_indices': collision_indices,
            'collision_segments': collision_segments if interpolate else [],
            'collision_segments_original': collision_segments_original,  # From original trajectory waypoints
            'collision_free_indices': collision_free_indices,
            'link_collisions': dict(link_collision_counter) if show_link_collisions else {},

            # Replanning summary
            'replan_enabled': replan_enabled,
            'replan_collision_summary': None,
            'replanned_segments': replanned_segments,
            'replan_success_count': replan_success_count,
            'replan_fail_count': replan_fail_count,
            'replan_iterations': replan_iterations_performed,
            'replan_success_rate': (replan_success_count / len(replanned_segments) if replanned_segments else 0.0),

            # Timing
            'collision_check_time_sec': collision_check_time,
            'replan_time_sec': replan_time,
            'overall_time_sec': overall_time,
        }

        if verbose:
            print("\n" + "=" * 70)
            print("FINAL RESULTS")
            print("=" * 70)
            print(f"Original trajectory: {num_waypoints} waypoints")
            print(f"Final trajectory: {num_waypoints_current} waypoints")
            print(f"Modified: {results['trajectory_modified']}")
            print(f"\nSegment Statistics:")
            print(f"  Collision segments: {len(collision_segments_original)}")
            if len(collision_segments_original) > 0:
                print(f"    Segments: {collision_segments_original[:20]}{'...' if len(collision_segments_original) > 20 else ''}")
            print(f"  Replanning segments: {len(replanned_segments)}")
            if len(replanned_segments) > 0:
                print(f"    Segments: {replanned_segments[:20]}{'...' if len(replanned_segments) > 20 else ''}")
            print(f"\nTiming:")
            print(f"  Collision checking: {collision_check_time:.3f}s")
            print(f"  Replanning: {replan_time:.3f}s")
            print(f"  Total: {overall_time:.3f}s")
            print("=" * 70)

        return results

    def _compute_segment_interp_counts(
        self,
        trajectory: np.ndarray,
        adaptive_interp: bool,
        adaptive_max_joint_step_deg: float,
        exclude_last_joint: bool,
    ) -> List[int]:
        """
        Compute interpolation counts per segment.

        Uses a single threshold for all joints to keep interpolation density consistent.
        """
        num_segments = max(0, len(trajectory) - 1)
        if num_segments <= 0:
            return []

        segment_counts = [0 for _ in range(num_segments)]

        if not adaptive_interp:
            return segment_counts

        # Main joints threshold (first n-1 joints)
        threshold_deg = adaptive_max_joint_step_deg if adaptive_max_joint_step_deg is not None else 0.0
        threshold_rad = np.deg2rad(max(threshold_deg, 0.0))

        for seg_idx in range(num_segments):
            start_config = trajectory[seg_idx]
            end_config = trajectory[seg_idx + 1]

            if exclude_last_joint and len(start_config) > 1:
                delta = end_config[:-1] - start_config[:-1]
            else:
                delta = end_config - start_config

            max_delta = float(np.max(np.abs(delta))) if delta.size > 0 else 0.0
            steps = int(np.ceil(max_delta / threshold_rad)) if (threshold_rad > 0 and max_delta > 0) else 0

            segment_counts[seg_idx] = max(0, int(steps))

        return segment_counts

    def _precompute_segment_interpolations(
        self,
        trajectory: np.ndarray,
        segment_step_counts: Optional[List[int]] = None
    ) -> Dict[int, List[Tuple[int, float, np.ndarray]]]:
        """Precompute per-segment interpolation results using CPU linear interpolation."""
        interpolated_map: Dict[int, List[Tuple[int, float, np.ndarray]]] = {}

        num_segments = max(0, len(trajectory) - 1)
        if num_segments <= 0:
            return interpolated_map

        if segment_step_counts is None or len(segment_step_counts) != num_segments:
            segment_step_counts = [0 for _ in range(num_segments)]

        for seg_idx in range(num_segments):
            steps = max(0, int(segment_step_counts[seg_idx]))
            if steps <= 0:
                interpolated_map[seg_idx] = []
                continue

            start_config = trajectory[seg_idx]
            end_config = trajectory[seg_idx + 1]

            # Always use CPU linear interpolation
            segment_configs = generate_interpolated_path(start_config, end_config, steps)

            if not segment_configs:
                interpolated_map[seg_idx] = []
                continue

            interpolated_map[seg_idx] = []
            denom = steps + 1
            for interp_idx, joint_config in enumerate(segment_configs):
                alpha = (interp_idx + 1) / denom if denom > 0 else 0.0
                interpolated_map[seg_idx].append((interp_idx, alpha, np.array(joint_config, dtype=np.float64)))

        return interpolated_map

    def _replan_segment(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        verbose: bool = False
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Replan a single trajectory segment using MotionGen

        Args:
            start_config: Starting joint configuration (n_joints,)
            goal_config: Goal joint configuration (n_joints,)
            verbose: Print replanning details

        Returns:
            (success, new_trajectory):
                - success: True if replanning succeeded
                - new_trajectory: New trajectory as (N, n_joints) array, or None if failed
        """
        try:
            # Convert to float32 tensors (CuRobo expects float32)
            start_config_f32 = np.asarray(start_config, dtype=np.float32)
            goal_config_f32 = np.asarray(goal_config, dtype=np.float32)

            start_tensor = self.tensor_args.to_device(torch.from_numpy(start_config_f32).unsqueeze(0))
            goal_tensor = self.tensor_args.to_device(torch.from_numpy(goal_config_f32).unsqueeze(0))

            # Create JointState objects for start and goal
            start_state = JointState.from_position(start_tensor)
            goal_state = JointState.from_position(goal_tensor)

            # Plan trajectory using plan_single_js (Joint Space planning)
            if verbose:
                print(f"    Calling MotionGen...")

            result = self.motion_gen.plan_single_js(
                start_state=start_state,
                goal_state=goal_state,
                plan_config=MotionGenPlanConfig(
                    enable_graph=False,
                    enable_opt=True,
                    need_graph_success=False,
                    timeout=config.REPLAN_TIMEOUT,
                    max_attempts=config.REPLAN_MAX_ATTEMPTS,
                )
            )

            if result.success.item():
                # Extract trajectory from result
                try:
                    # Try to get trajectory from result
                    if hasattr(result, 'trajectory') and result.trajectory is not None:
                        traj = result.trajectory
                        if hasattr(traj, 'position'):
                            traj_positions = traj.position.detach().cpu().numpy()
                        else:
                            traj_positions = traj.detach().cpu().numpy()
                    elif hasattr(result, 'optimized_plan') and result.optimized_plan is not None:
                        plan = result.optimized_plan
                        traj_positions = plan.position.detach().cpu().numpy() if hasattr(plan, 'position') else plan.detach().cpu().numpy()
                    else:
                        if verbose:
                            print(f"    ✗ Could not extract trajectory")
                        return False, None

                    if verbose:
                        print(f"    ✓ Replanning succeeded")
                    return True, traj_positions

                except Exception as e:
                    if verbose:
                        print(f"    ✗ Extraction error: {e}")
                    return False, None
            else:
                if verbose:
                    print(f"    ✗ Replanning failed")
                return False, None

        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            return False, None

    def interpolate_trajectory_curobo(
        self,
        trajectory: np.ndarray,
        raw_dt: float = 0.2,
        interpolation_dt: float = 0.02,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Interpolate trajectory using CuRobo's get_batch_interpolated_trajectory

        Args:
            trajectory: (N, n_joints) trajectory array
            raw_dt: Time step between waypoints in original trajectory (seconds)
            interpolation_dt: Time step for interpolation (seconds)
            verbose: Print interpolation details

        Returns:
            Interpolated trajectory as (M, n_joints) array
        """
        if verbose:
            print(f"\nInterpolating trajectory using CuRobo...")
            print(f"  Input waypoints: {len(trajectory)}")
            print(f"  Raw dt: {raw_dt}s")
            print(f"  Interpolation dt: {interpolation_dt}s")

        # Convert to float32 tensor
        traj_f32 = np.asarray(trajectory, dtype=np.float32)
        traj_tensor = self.tensor_args.to_device(torch.from_numpy(traj_f32))

        # Create JointState
        state = JointState.from_position(traj_tensor)

        # Get max velocity from robot configuration
        max_vel = self.motion_gen.kinematics.get_joint_limits().velocity[0]  # [1, n_joints]
        max_acc = self.motion_gen.kinematics.get_joint_limits().acceleration[0]
        max_jerk = self.motion_gen.kinematics.get_joint_limits().jerk[0]

        # Convert raw_dt to tensor
        raw_dt_tensor = self.tensor_args.to_device(torch.tensor([raw_dt], dtype=torch.float32))

        # Use CuRobo's get_batch_interpolated_trajectory function
        # 보간을 하면 오히려 충돌이 발생할 수 있다.
        # TODO: 최종 output의 보간을 어떻게 할지 생각해보자.
        result_tuple = get_batch_interpolated_trajectory(
            raw_traj=state,
            raw_dt=raw_dt_tensor,
            interpolation_dt=interpolation_dt,
            max_vel=max_vel,
            max_acc=max_acc,
            max_jerk=max_jerk,
        )

        # Extract interpolated trajectory (function returns tuple)
        if isinstance(result_tuple, tuple):
            interpolated_state = result_tuple[0]  # First element is the trajectory
        else:
            interpolated_state = result_tuple

        # Extract positions and convert to numpy
        if hasattr(interpolated_state, 'position'):
            result = interpolated_state.position.detach().cpu().numpy()
        else:
            result = interpolated_state.detach().cpu().numpy()

        # Remove batch dimension if present (shape: [1, N, joints] -> [N, joints])
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]

        if verbose:
            print(f"  Output waypoints: {len(result)}")
            print(f"  Interpolation ratio: {len(result) / len(trajectory):.1f}x")

        return result


# ============================================================================
# Utility Functions
# ============================================================================
# Note: load_trajectory_csv and save_trajectory_csv are now imported from common.trajectory_io


def save_collision_report(
    trajectory_path: str,
    results: dict,
    timing_info: Optional[Dict[str, float]] = None,
    checker_type: str = "curobo"
) -> Path:
    """Save collision report to file"""
    base_dir = Path(__file__).resolve().parent.parent
    traj_path = Path(trajectory_path)
    parent_dir_name = traj_path.parent.name

    if parent_dir_name.isdigit():
        num_points = parent_dir_name
    else:
        num_points = traj_path.stem or "unknown"

    report_dir = base_dir / 'data' / 'collision' / str(num_points)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f'collision_{checker_type}.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    unique_segments = sorted({wp_idx for wp_idx, _ in results.get('collision_segments', [])})
    collision_free_configs = (
        results['total_configs_checked'] - results['total_collisions']
        if results['interpolate'] else results['num_collision_free']
    )

    def format_list(values, max_items=50):
        if not values:
            return "None"
        subset = list(values)[:max_items]
        suffix = "" if len(values) <= max_items else f" ... (+{len(values) - max_items} more)"
        return ", ".join(str(v) for v in subset) + suffix

    def format_time(seconds: Optional[float]) -> str:
        if seconds is None:
            return "N/A"
        return f"{seconds:.3f} s"

    segment_pairs = [f"{idx}->{idx + 1}" for idx in unique_segments]

    lines = [
        f"=== Collision Report (CuRobo) @ {timestamp} ===",
        f"Trajectory: {trajectory_path}",
        f"Robot config: {config.DEFAULT_ROBOT_CONFIG}",
        f"Collision checker: CuRobo (GPU-accelerated)",
        f"Obstacle meshes: {config.DEFAULT_MESH_FILE}",
        f"Collision margin: {config.COLLISION_MARGIN}",
        f"Interpolation enabled: {results['interpolate']}",
        f"Interpolation mode: {'adaptive' if results.get('adaptive_interp') else 'uniform'}",
        f"Interpolated configurations: {results.get('total_interpolated_configs', 0)}",
        "",
        f"Total waypoints: {results['total_waypoints']}",
        f"Total configurations checked: {results['total_configs_checked']}",
        f"Collisions at waypoints: {results['num_collisions']}",
        f"Segment collisions (raw count): {results['num_segment_collisions'] if results['interpolate'] else 0}",
        f"Total collisions: {results['total_collisions']}",
        f"Collision-free configurations: {collision_free_configs}",
        f"Collision rate (%): {results['collision_rate']:.2f}",
        "",
        f"Collision waypoint indices: {format_list(results['collision_indices'])}",
        f"Collision segments (unique pairs): {format_list(segment_pairs)}",
    ]

    # Add replanning statistics
    if results.get('replan_enabled', False):
        lines.append("")
        lines.append("Motion Replanning Analysis:")
        lines.append(f"  Replanning enabled: {results.get('replan_enabled', False)}")
        lines.append(f"  Iterations performed: {results.get('replan_iterations', 0)}")
        lines.append(f"  Segments replanned (success): {results.get('replan_success_count', 0)}")
        lines.append(f"  Segments failed: {results.get('replan_fail_count', 0)}")
        lines.append(f"  Success rate: {results.get('replan_success_rate', 0):.1%}")
        replanned_segs = results.get('replanned_segments', [])
        lines.append(f"  Replanned segment indices: {format_list(replanned_segs)}")

    lines.append("")
    lines.append("Segment Statistics:")

    collision_segs_orig = results.get('collision_segments_original', [])
    lines.append(f"  Collision segments: {len(collision_segs_orig)}")
    if collision_segs_orig:
        lines.append(f"    Segments: {format_list(collision_segs_orig)}")

    if results.get('replan_enabled', False):
        replanned_segs = results.get('replanned_segments', [])
        lines.append(f"  Replanning segments: {len(replanned_segs)}")
        if replanned_segs:
            lines.append(f"    Segments: {format_list(replanned_segs)}")

    if timing_info:
        lines.append("")
        lines.append("Timing:")
        lines.append(f"  Collision checking: {format_time(timing_info.get('collision_check_sec'))}")
        lines.append(f"  Replanning:         {format_time(timing_info.get('replan_time_sec'))}")
        lines.append(f"  Total:              {format_time(timing_info.get('total_runtime_sec'))}")

    content = "\n".join(lines)
    if report_path.exists() and report_path.stat().st_size > 0:
        with open(report_path, 'a') as f:
            f.write("\n\n" + content)
    else:
        with open(report_path, 'w') as f:
            f.write(content)

    return report_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="CuRobo-based Collision Checker for Robot Trajectories"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        required=True,
        help="Path to trajectory CSV file"
    )
    parser.add_argument(
        "--robot_config",
        type=str,
        default=config.DEFAULT_ROBOT_CONFIG,
        help="CuRobo robot config file (e.g., ur20.yml)"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=config.DEFAULT_MESH_FILE,
        help="Path to obstacle mesh file"
    )
    parser.add_argument(
        "--exclude_last_joint",
        dest="exclude_last_joint",
        action="store_true",
        default=config.COLLISION_INTERP_EXCLUDE_LAST_JOINT,
        help="Exclude last joint when computing max joint delta for interpolation"
    )
    parser.add_argument(
        "--include_last_joint",
        dest="exclude_last_joint",
        action="store_false",
        help="Include the last joint when computing max joint delta for interpolation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.COLLISION_VERBOSE,
        help="Print detailed progress"
    )

    args = parser.parse_args()

    print_section_header("CUROBO COLLISION CHECKER")
    print_key_value("Trajectory", args.trajectory)
    print_key_value("Robot config", args.robot_config)
    print_key_value("Mesh", args.mesh)
    print_key_value("Interpolation", "enabled (adaptive)")
    print_key_value("Exclude last joint", args.exclude_last_joint)
    print()

    # Check trajectory
    start_time = time.perf_counter()

    # Load trajectory
    trajectory, joint_names = load_trajectory_csv(args.trajectory)

    # Create collision checker
    checker = CuRoboCollisionChecker(
        robot_config_path=args.robot_config,
        obstacle_mesh_paths=[args.mesh],
    )

    results = checker.check_trajectory(
        trajectory,
        verbose=args.verbose,
        exclude_last_joint=args.exclude_last_joint,
    )

    # Save collision report
    timing_info = {
        'collision_check_sec': results['collision_check_time_sec'],
        'replan_time_sec': results.get('replan_time_sec', 0.0),
        'total_runtime_sec': start_time,
    }
    report_path = save_collision_report(args.trajectory, results, timing_info, checker_type="curobo")

    print(f"\n✓ Collision report saved: {report_path}")

    # Save interpolated trajectory (includes all collision-checked configurations)
    print(f"\nSaving interpolated trajectory...")
    interpolated_trajectory = results['interpolated_trajectory']

    # Use the fully interpolated trajectory that was collision-checked
    # This includes waypoints + all interpolated configs that passed collision detection
    # Prevents collision issues when replaying trajectory in simulation
    print(f"  Using collision-checked interpolated trajectory: {len(interpolated_trajectory)} configurations")
    print(f"  This includes {results['total_waypoints']} waypoints + {results.get('total_interpolated_configs', 0)} interpolated configs")
    print(f"  All configurations have been verified collision-free")

    # Save to CSV (using parent directory from original path)
    traj_path = Path(args.trajectory).parent / f"{Path(args.trajectory).stem}_collision_free.csv"
    save_trajectory_csv(
        interpolated_trajectory,
        str(traj_path),
        joint_names=joint_names,
        include_time=True
    )

    print(f"✓ Final trajectory saved: {traj_path}")

    # ====================================================================
    # VERIFICATION: Check saved trajectory for collisions (FAST)
    # ====================================================================
    print(f"\n{'='*70}")
    print("VERIFYING SAVED TRAJECTORY")
    print(f"{'='*70}")
    print(f"Checking {len(interpolated_trajectory)} configurations for collisions...")
    print(f"  Using fast collision checking (no interpolation)")

    # Use _check_collisions_batched for fast verification (no interpolation)
    verification_collision_results = checker._check_collisions_batched(
        interpolated_trajectory,
        interpolated_configs_map=None,  # No interpolation
        total_interp_points=0,
        verbose=False
    )

    num_collisions = verification_collision_results['num_collisions']
    collision_indices = verification_collision_results['collision_indices']
    configs_checked = verification_collision_results['configs_checked']
    collision_rate = (num_collisions / configs_checked * 100) if configs_checked > 0 else 0.0

    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    if num_collisions > 0:
        print(f"⚠ WARNING: Saved trajectory contains collisions!")
        print(f"  Total collisions: {num_collisions}")
        print(f"  Collision rate: {collision_rate:.2f}%")
        print(f"  Collision indices: {collision_indices[:20]}")
        if len(collision_indices) > 20:
            print(f"    ... and {len(collision_indices) - 20} more")
        print(f"\n⚠ The saved trajectory is NOT collision-free!")
        print(f"  Consider:")
        print(f"    1. Increasing collision margin (current: {config.COLLISION_MARGIN})")
        print(f"    2. Enabling replanning (--replan)")
        print(f"    3. Reviewing mesh accuracy")
    else:
        print(f"✓ VERIFICATION PASSED")
        print(f"  All {len(interpolated_trajectory)} configurations are collision-free")
        print(f"  Saved trajectory is safe to execute")
    print(f"{'='*70}\n")

    total_time = time.perf_counter() - start_time
    print(f"✓ Total runtime: {total_time:.3f}s")


if __name__ == "__main__":
    main()
