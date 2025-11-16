#!/usr/bin/env python3
"""
COAL-based Collision Checker for Robot Trajectories

This script checks for collisions along a robot trajectory using COAL library.
It loads a joint trajectory CSV file, computes forward kinematics for each
configuration, and checks for collisions with environment meshes.

COAL (Collision and Occupancy Algorithms Library) is an improved version of FCL
with 5-15x better performance and native Pinocchio integration.

Coordinate System:
- Uses Z-up coordinate system (Isaac Sim / URDF / Pinocchio convention)
- All meshes should be in Z-up format (e.g., glass_zup.obj)
- Consistent with other pipeline components

Usage:
    omni_python coal_check.py --trajectory data/trajectory/joint_trajectory_dp.csv \
                               --robot_urdf ur_description/ur20.urdf \
                               --mesh data/object/glass_zup.obj
"""

import argparse
import csv
import coal
import numpy as np
import os
import sys
import time
from typing import List, Tuple, Optional, Dict
import trimesh
from scipy.spatial.transform import Rotation
import pinocchio as pin
from pathlib import Path
from datetime import datetime
import yaml
from multiprocessing import Pool, cpu_count
from functools import partial

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
    from curobo.util.trajectory import get_interpolated_trajectory, get_batch_interpolated_trajectory, InterpolateType
    CUROBO_AVAILABLE = True
except ImportError:
    torch = None
    WorldConfig = None
    Mesh = None
    TensorDeviceType = None
    JointState = None
    MotionGen = None
    MotionGenConfig = None
    MotionGenPlanConfig = None
    CollisionCheckerType = None
    get_robot_configs_path = None
    get_world_configs_path = None
    join_path = None
    load_yaml = None
    get_interpolated_trajectory = None
    get_batch_interpolated_trajectory = None
    InterpolateType = None
    CUROBO_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from common import config
from common.interpolation_utils import generate_interpolated_path


# Global variable for worker processes (to avoid pickling large objects)
_worker_checker = None


def _init_worker(checker_args):
    """
    Initialize collision checker in each worker process.

    This is called once per worker to set up the collision checker instance.
    Avoids pickling the large COALCollisionChecker object.

    Args:
        checker_args: Dictionary with initialization arguments for COALCollisionChecker
    """
    global _worker_checker
    _worker_checker = COALCollisionChecker(**checker_args)


def _check_collision_worker(config_data):
    """
    Worker function for parallel collision checking.

    Args:
        config_data: Tuple of (index, joint_config, show_link_collisions)

    Returns:
        Tuple of (index, is_collision, distance, link_info)
    """
    global _worker_checker
    idx, joint_config, show_link_collisions = config_data

    is_collision, dist, link_info = _worker_checker.check_collision_single_config(
        joint_config,
        return_distance=True,
        return_link_info=show_link_collisions
    )

    return (idx, is_collision, dist, link_info)


class COALCollisionChecker:
    """Collision checker using COAL and pinocchio for FK"""

    def __init__(self, robot_urdf_path: str, obstacle_mesh_paths: List[str],
                 glass_position: np.ndarray = None,
                 table_position: np.ndarray = None,
                 table_dimensions: np.ndarray = None,
                 wall_position: np.ndarray = None,
                 wall_dimensions: np.ndarray = None,
                 workbench_position: np.ndarray = None,
                 workbench_dimensions: np.ndarray = None,
                 robot_mount_position: np.ndarray = None,
                 robot_mount_dimensions: np.ndarray = None,
                 robot_config_path: Optional[str] = None,
                 use_link_meshes: bool = False,
                 mesh_base_path: str = None,
                 collision_margin: float = None,
                 use_curobo_interpolation: bool = True):
        """
        Initialize collision checker

        Args:
            robot_urdf_path: Path to robot URDF file
            obstacle_mesh_paths: List of paths to obstacle mesh files
            glass_position: Position of glass object in world frame (x, y, z)
            table_position: Position of table cuboid in world frame (x, y, z)
            table_dimensions: Dimensions of table cuboid (x, y, z) in meters
            wall_position: Position of wall cuboid in world frame (x, y, z)
            wall_dimensions: Dimensions of wall cuboid (x, y, z) in meters
            workbench_position: Position of workbench cuboid in world frame (x, y, z)
            workbench_dimensions: Dimensions of workbench cuboid (x, y, z) in meters
            robot_mount_position: Position of robot mount cuboid in world frame (x, y, z)
            robot_mount_dimensions: Dimensions of robot mount cuboid (x, y, z) in meters
            robot_config_path: Path to CuRobo robot config YAML (e.g., ur20.yml)
            use_link_meshes: If True, use actual URDF collision meshes for robot links
            mesh_base_path: Base path for robot mesh files
            collision_margin: Safety margin for collision detection (meters).
                            Positive = more conservative, Negative = less conservative
            use_curobo_interpolation: If True, use CuRobo GPU interpolation helpers
        """
        # Apply config defaults
        if glass_position is None:
            glass_position = config.GLASS_POSITION.copy()
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
        if robot_config_path is None:
            robot_config_path = config.DEFAULT_ROBOT_CONFIG_YAML
        if mesh_base_path is None:
            mesh_base_path = config.MESH_BASE_PATH
        if collision_margin is None:
            collision_margin = config.COLLISION_MARGIN

        self.robot_urdf_path = robot_urdf_path
        self.obstacle_meshes = []
        self.obstacle_collision_objects = []
        self.glass_position = glass_position
        self.table_position = table_position
        self.table_dimensions = table_dimensions
        self.wall_position = wall_position
        self.wall_dimensions = wall_dimensions
        self.workbench_position = workbench_position
        self.workbench_dimensions = workbench_dimensions
        self.robot_mount_position = robot_mount_position
        self.robot_mount_dimensions = robot_mount_dimensions
        self.robot_config_path = robot_config_path
        self.use_link_meshes = use_link_meshes
        self.mesh_base_path = mesh_base_path
        self.collision_margin = collision_margin
        self.collision_spheres = {}
        self.link_meshes = {}
        self.request_curobo_interpolation = bool(use_curobo_interpolation)
        self.use_curobo_interpolation = (
            self.request_curobo_interpolation
            and CUROBO_AVAILABLE
            and torch is not None
            and TensorDeviceType is not None
            and JointState is not None
        )
        self.curobo_tensor_args = TensorDeviceType() if self.use_curobo_interpolation else None
        self.curobo_interp_kind = (
            InterpolateType.CUBIC if self.use_curobo_interpolation and InterpolateType is not None else None
        )

        # Load obstacle meshes
        print(f"Loading obstacle meshes...")
        print(f"  Glass position: {glass_position}")

        for mesh_path in obstacle_mesh_paths:
            mesh = self._load_mesh(mesh_path)
            if mesh is not None:
                self.obstacle_meshes.append(mesh)

                # Create COAL transform for glass position
                glass_transform = self._create_transform(
                    rotation=np.eye(3),
                    translation=glass_position
                )

                # Create COAL collision object with transform
                col_obj = self._create_coal_collision_object(mesh, glass_transform)
                self.obstacle_collision_objects.append(col_obj)
                print(f"  Loaded: {mesh_path} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
                print(f"    Applied transform: translation = {glass_position}")

        print(f"Total obstacles loaded: {len(self.obstacle_meshes)}")

        # Add table cuboid
        print(f"\nAdding table cuboid...")
        print(f"  Position: {table_position}")
        print(f"  Dimensions (x, y, z): {table_dimensions}")

        # Create COAL Box
        table_box = coal.Box(table_dimensions[0], table_dimensions[1], table_dimensions[2])

        # Create table transform
        table_transform = self._create_transform(
            rotation=np.eye(3),
            translation=table_position
        )

        # Create collision object
        table_obj = coal.CollisionObject(table_box, table_transform)
        self.obstacle_collision_objects.append(table_obj)
        print(f"  Table cuboid added successfully")

        # Add wall cuboid
        print(f"\nAdding wall cuboid...")
        print(f"  Position: {wall_position}")
        print(f"  Dimensions (x, y, z): {wall_dimensions}")
        wall_box = coal.Box(wall_dimensions[0], wall_dimensions[1], wall_dimensions[2])
        wall_transform = self._create_transform(
            rotation=np.eye(3),
            translation=wall_position
        )
        wall_obj = coal.CollisionObject(wall_box, wall_transform)
        self.obstacle_collision_objects.append(wall_obj)
        print(f"  Wall cuboid added successfully")

        # Add workbench cuboid
        print(f"\nAdding workbench cuboid...")
        print(f"  Position: {workbench_position}")
        print(f"  Dimensions (x, y, z): {workbench_dimensions}")
        workbench_box = coal.Box(workbench_dimensions[0], workbench_dimensions[1], workbench_dimensions[2])
        workbench_transform = self._create_transform(
            rotation=np.eye(3),
            translation=workbench_position
        )
        workbench_obj = coal.CollisionObject(workbench_box, workbench_transform)
        self.obstacle_collision_objects.append(workbench_obj)
        print(f"  Workbench cuboid added successfully")

        # Add robot mount cuboid
        print(f"\nAdding robot mount cuboid...")
        print(f"  Position: {robot_mount_position}")
        print(f"  Dimensions (x, y, z): {robot_mount_dimensions}")
        robot_mount_box = coal.Box(robot_mount_dimensions[0], robot_mount_dimensions[1], robot_mount_dimensions[2])
        robot_mount_transform = self._create_transform(
            rotation=np.eye(3),
            translation=robot_mount_position
        )
        robot_mount_obj = coal.CollisionObject(robot_mount_box, robot_mount_transform)
        self.obstacle_collision_objects.append(robot_mount_obj)
        print(f"  Robot mount cuboid added successfully")

        # Load robot model with pinocchio
        print(f"\nLoading robot model with Pinocchio...")
        print(f"  URDF: {robot_urdf_path}")
        self.robot_model = pin.buildModelFromUrdf(robot_urdf_path)
        self.robot_data = self.robot_model.createData()
        print(f"  Robot loaded: {self.robot_model.nq} DOF, {self.robot_model.njoints} joints")
        print(f"  Joint names: {[self.robot_model.names[i] for i in range(1, self.robot_model.njoints)]}")

        # Load collision geometry
        if use_link_meshes:
            # Load actual mesh geometry from URDF
            self._load_link_meshes_from_urdf()
            print(f"  Using actual collision meshes from URDF")
            print(f"  Loaded {len(self.link_meshes)} link meshes")
        elif robot_config_path:
            # Load collision spheres from CuRobo config
            self._load_collision_spheres_from_yaml(robot_config_path)
            print(f"  Using collision spheres from: {robot_config_path}")
            total_spheres = sum(len(spheres) for spheres in self.collision_spheres.values())
            print(f"  Total collision spheres: {total_spheres}")

    def _create_transform(
        self,
        rotation: np.ndarray = None,
        translation: np.ndarray = None
    ) -> coal.Transform3s:
        """
        Helper to create COAL Transform3s

        Args:
            rotation: 3x3 rotation matrix (optional)
            translation: 3D translation vector (optional)

        Returns:
            COAL Transform3s object
        """
        transform = coal.Transform3s()

        if rotation is not None:
            transform.setRotation(rotation)

        if translation is not None:
            transform.setTranslation(translation)

        return transform

    def _load_link_meshes_from_urdf(self):
        """
        Load collision meshes from URDF file

        Parses URDF to extract collision mesh file paths for each link
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.robot_urdf_path)
        root = tree.getroot()

        # Map of standard UR link names to mesh files
        # Based on standard UR URDF structure
        mesh_map = {
            'base_link': 'meshes/ur20/collision/base.stl',
            'shoulder_link': 'meshes/ur20/collision/shoulder.stl',
            'upper_arm_link': 'meshes/ur20/collision/upperarm.stl',
            'forearm_link': 'meshes/ur20/collision/forearm.stl',
            'wrist_1_link': 'meshes/ur20/collision/wrist1.stl',
            'wrist_2_link': 'meshes/ur20/collision/wrist2.stl',
            'wrist_3_link': 'meshes/ur20/collision/wrist3.stl',
        }

        for link_name, mesh_file in mesh_map.items():
            mesh_path = Path(self.mesh_base_path) / mesh_file
            if mesh_path.exists():
                try:
                    mesh = trimesh.load(str(mesh_path), force='mesh')
                    self.link_meshes[link_name] = mesh
                    print(f"    Loaded mesh for {link_name}: {mesh.vertices.shape[0]} vertices")
                except Exception as e:
                    print(f"    Warning: Could not load mesh for {link_name}: {e}")

    def _load_collision_spheres_from_yaml(self, yaml_path: str):
        """
        Load collision spheres from CuRobo robot config YAML

        Args:
            yaml_path: Path to robot config YAML file
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        # Extract collision_spheres from robot_cfg
        robot_cfg = config.get('robot_cfg', {})
        kinematics = robot_cfg.get('kinematics', {})
        collision_spheres = kinematics.get('collision_spheres', {})

        # Store collision spheres
        for link_name, spheres in collision_spheres.items():
            self.collision_spheres[link_name] = []
            for sphere in spheres:
                center = np.array(sphere['center'])
                radius = sphere['radius']
                if radius > 0:  # Skip spheres with negative radius
                    self.collision_spheres[link_name].append({
                        'center': center,
                        'radius': radius
                    })

    def _load_mesh(self, mesh_path: str) -> Optional[trimesh.Trimesh]:
        """Load mesh file using trimesh"""
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            return mesh
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            return None

    def _create_coal_collision_object(
        self,
        mesh: trimesh.Trimesh,
        transform: Optional[coal.Transform3s] = None
    ) -> coal.CollisionObject:
        """Create COAL collision object from trimesh"""
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        triangles = np.asarray(mesh.faces, dtype=np.int32)

        # Create COAL BVHModel (using OBBRSS type for better performance)
        bvh = coal.BVHModelOBBRSS()
        bvh.beginModel(len(vertices), len(triangles))

        # Add each triangle using explicit vertex coordinates (API expects Vec3s)
        for triangle in triangles:
            v0 = np.asarray(vertices[triangle[0]], dtype=np.float64)
            v1 = np.asarray(vertices[triangle[1]], dtype=np.float64)
            v2 = np.asarray(vertices[triangle[2]], dtype=np.float64)
            bvh.addTriangle(v0, v1, v2)

        bvh.endModel()

        # Create collision object
        if transform is None:
            transform = coal.Transform3s()

        return coal.CollisionObject(bvh, transform)

    def _create_robot_collision_geometry(
        self,
        joint_positions: np.ndarray
    ) -> List[coal.CollisionObject]:
        """
        Create robot collision geometry for given joint configuration using FK

        Args:
            joint_positions: Array of joint angles (6 values for UR robot)

        Returns:
            List of COAL collision objects representing robot links
        """
        # Compute forward kinematics for all joints
        pin.forwardKinematics(self.robot_model, self.robot_data, joint_positions)
        pin.updateFramePlacements(self.robot_model, self.robot_data)

        # Use actual link meshes if available
        if self.link_meshes:
            return self._create_mesh_collision_objects()
        # Use collision spheres if available
        elif self.collision_spheres:
            return self._create_sphere_collision_objects()

    def _create_mesh_collision_objects(self) -> List[coal.CollisionObject]:
        """Create COAL collision objects using actual link meshes"""
        robot_collision_objects = []

        for link_name, mesh in self.link_meshes.items():
            # Find the frame/joint for this link
            try:
                frame_id = self.robot_model.getFrameId(link_name)
                transform_matrix = self.robot_data.oMf[frame_id]
            except:
                try:
                    joint_id = self.robot_model.getJointId(link_name)
                    transform_matrix = self.robot_data.oMi[joint_id]
                except:
                    continue

            # Get position and rotation from pinocchio
            position = transform_matrix.translation
            rotation = transform_matrix.rotation

            # Create COAL mesh collision object
            vertices = np.array(mesh.vertices, dtype=np.float64)
            triangles = np.array(mesh.faces, dtype=np.int32)

            # Create BVH model (using OBBRSS type for better performance)
            bvh = coal.BVHModelOBBRSS()
            bvh.beginModel(len(vertices), len(triangles))
            bvh.addSubModel(vertices, triangles)
            bvh.endModel()

            # Create transform using helper method
            coal_transform = self._create_transform(rotation, position)

            # Create collision object
            col_obj = coal.CollisionObject(bvh, coal_transform)
            robot_collision_objects.append(col_obj)

        return robot_collision_objects

    def _create_sphere_collision_objects(self) -> List[coal.CollisionObject]:
        """Create COAL collision objects using spheres from YAML config"""
        robot_collision_objects = []

        for link_name, spheres in self.collision_spheres.items():
            # Find the frame/joint for this link
            try:
                frame_id = self.robot_model.getFrameId(link_name)
                transform_matrix = self.robot_data.oMf[frame_id]
            except:
                try:
                    joint_id = self.robot_model.getJointId(link_name)
                    transform_matrix = self.robot_data.oMi[joint_id]
                except:
                    continue

            # Create a sphere for each defined collision sphere
            for sphere_def in spheres:
                center_local = sphere_def['center']
                radius = sphere_def['radius']

                # Transform sphere center to world frame
                center_world = transform_matrix.translation + transform_matrix.rotation @ center_local

                # Create COAL sphere
                coal_transform = self._create_transform(translation=center_world)
                sphere = coal.Sphere(radius)
                col_obj = coal.CollisionObject(sphere, coal_transform)
                robot_collision_objects.append(col_obj)

        return robot_collision_objects

    def _generate_segment_interpolation(
        self,
        start_config: np.ndarray,
        end_config: np.ndarray,
        num_steps: int
    ) -> List[np.ndarray]:
        """Interpolate between two configs using CuRobo trajectory util when available."""
        if num_steps <= 0:
            return [np.array(end_config, dtype=np.float64)]

        if (
            self.use_curobo_interpolation
            and CUROBO_AVAILABLE
            and self.curobo_tensor_args
            and JointState
            and get_interpolated_trajectory
        ):
            try:
                return self._curobo_interpolate_segment(start_config, end_config, num_steps)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Warning: CuRobo interpolation failed ({exc}), falling back to linear path.")

        return generate_interpolated_path(start_config, end_config, num_steps)

    def _curobo_interpolate_segment(
        self,
        start_config: np.ndarray,
        end_config: np.ndarray,
        num_steps: int
    ) -> List[np.ndarray]:
        """Use curobo.util.trajectory.get_interpolated_trajectory for segment interpolation."""
        if torch is None or self.curobo_tensor_args is None:
            raise RuntimeError("CuRobo interpolation is unavailable (torch not initialized).")

        horizon = num_steps + 1
        dtype = getattr(self.curobo_tensor_args, "dtype", torch.float32)
        device = getattr(self.curobo_tensor_args, "device", torch.device("cpu"))

        start_tensor = torch.as_tensor(start_config, dtype=dtype, device=device)
        end_tensor = torch.as_tensor(end_config, dtype=dtype, device=device)
        traj_tensor = torch.stack([start_tensor, end_tensor]).unsqueeze(0)

        dof = start_tensor.shape[-1]
        out_state = JointState.zeros([1, horizon, dof], self.curobo_tensor_args)

        interp_dt = max(1.0 / max(num_steps, 1), 1e-3)
        interpolated_state, last_tsteps, _ = get_interpolated_trajectory(
            [traj_tensor],
            out_state,
            des_horizon=horizon,
            interpolation_dt=interp_dt,
            kind=self.curobo_interp_kind or InterpolateType.LINEAR,
            tensor_args=self.curobo_tensor_args,
        )

        steps = last_tsteps[0] if last_tsteps else horizon
        steps = min(steps, horizon)
        data = interpolated_state.position[0, :steps, :].detach().cpu().numpy()

        if data.shape[0] <= 1:
            return [np.array(end_config, dtype=np.float64)]

        return [np.array(row, dtype=np.float64) for row in data[1:]]

    def _batch_interpolate_trajectory(
        self,
        trajectory: np.ndarray,
        num_interp_steps: int
    ) -> List[Tuple[int, int, np.ndarray]]:
        """
        Batch interpolate entire trajectory using CuRobo's get_batch_interpolated_trajectory.

        This is much faster than per-segment interpolation as it:
        - Makes a single GPU/tensor call instead of N-1 calls
        - Better utilizes GPU parallelism
        - Reduces memory allocation overhead

        Args:
            trajectory: (N, dof) array of joint configurations
            num_interp_steps: Number of interpolation steps between waypoints

        Returns:
            List of (segment_idx, interp_idx, config) tuples for all interpolated configurations.
            segment_idx: which segment (0 to N-2)
            interp_idx: which interpolation point within segment (0 to num_interp_steps-1)
            config: the interpolated joint configuration
        """
        if not self.use_curobo_interpolation:
            raise RuntimeError("CuRobo batch interpolation disabled via configuration.")

        if not CUROBO_AVAILABLE or torch is None or self.curobo_tensor_args is None:
            raise RuntimeError("CuRobo batch interpolation unavailable (torch not initialized).")

        if get_batch_interpolated_trajectory is None:
            raise RuntimeError("get_batch_interpolated_trajectory not available in CuRobo version.")

        num_waypoints = len(trajectory)
        if num_waypoints < 2:
            return []

        dtype = getattr(self.curobo_tensor_args, "dtype", torch.float32)
        device = getattr(self.curobo_tensor_args, "device", torch.device("cpu"))

        # Convert trajectory to torch tensor
        traj_tensor = torch.as_tensor(trajectory, dtype=dtype, device=device)

        # Create JointState from trajectory
        raw_traj = JointState.from_position(traj_tensor.unsqueeze(0))  # Shape: (1, N, dof)

        # For batch interpolation:
        # - raw_dt: time between waypoints (uniform, scalar for batch)
        # - interpolation_dt: desired time resolution for interpolation
        # - Result: ~(raw_dt / interpolation_dt) steps between each waypoint

        # Set raw_dt to 1.0 (1 second between waypoints) for simplicity
        raw_dt = torch.ones(1, dtype=dtype, device=device)

        # Calculate interpolation_dt to get approximately num_interp_steps between waypoints
        # interpolation_dt = raw_dt / num_interp_steps
        interpolation_dt = 1.0 / max(num_interp_steps, 1)

        # Call batch interpolation
        # Using LINEAR_CUDA for speed (can be changed to CUBIC if smoothness is critical)
        interp_kind = InterpolateType.LINEAR_CUDA if hasattr(InterpolateType, 'LINEAR_CUDA') else InterpolateType.LINEAR

        interpolated_traj, traj_steps, opt_dt = get_batch_interpolated_trajectory(
            raw_traj=raw_traj,
            raw_dt=raw_dt,
            interpolation_dt=interpolation_dt,
            kind=interp_kind,
            tensor_args=self.curobo_tensor_args,
            optimize_dt=False,  # Keep fixed dt for predictable results
        )

        # Extract interpolated positions
        # Shape: (1, total_interpolated_points, dof)
        interp_positions = interpolated_traj.position[0].detach().cpu().numpy()
        actual_steps = traj_steps[0].item()  # Number of actual interpolated steps

        # Parse interpolated trajectory back into per-segment structure
        # get_batch_interpolated_trajectory returns all waypoints + interpolated points
        # We need to identify which points belong to which segment

        result = []
        point_idx = 0

        # Skip first waypoint (already in original trajectory)
        point_idx = 1

        for segment_idx in range(num_waypoints - 1):
            # For each segment, extract num_interp_steps interpolated points
            for interp_idx in range(num_interp_steps):
                if point_idx < len(interp_positions):
                    config = np.array(interp_positions[point_idx], dtype=np.float64)
                    result.append((segment_idx, interp_idx, config))
                    point_idx += 1

            # Skip the next waypoint
            point_idx += 1

        return result

    def _precompute_segment_interpolations(
        self,
        trajectory: np.ndarray,
        num_interp_steps: int,
        force_cpu: bool = False,
        segment_step_counts: Optional[List[int]] = None
    ) -> Dict[int, List[Tuple[int, float, np.ndarray]]]:
        """Precompute per-segment interpolation results for reuse."""
        interpolated_map: Dict[int, List[Tuple[int, float, np.ndarray]]] = {}

        num_segments = max(0, len(trajectory) - 1)
        if num_segments <= 0:
            return interpolated_map

        if segment_step_counts is None or len(segment_step_counts) != num_segments:
            base_steps = max(0, int(num_interp_steps))
            segment_step_counts = [base_steps for _ in range(num_segments)]

        for seg_idx in range(num_segments):
            steps = max(0, int(segment_step_counts[seg_idx]))
            if steps <= 0:
                interpolated_map[seg_idx] = []
                continue

            start_config = trajectory[seg_idx]
            end_config = trajectory[seg_idx + 1]

            if force_cpu or not self.use_curobo_interpolation:
                segment_configs = generate_interpolated_path(start_config, end_config, steps)
            else:
                segment_configs = self._generate_segment_interpolation(start_config, end_config, steps)

            if not segment_configs:
                interpolated_map[seg_idx] = []
                continue

            interpolated_map[seg_idx] = []
            denom = steps + 1
            for interp_idx, config in enumerate(segment_configs):
                alpha = (interp_idx + 1) / denom if denom > 0 else 0.0
                interpolated_map[seg_idx].append((interp_idx, alpha, np.array(config, dtype=np.float64)))

        return interpolated_map

    def _compute_segment_interp_counts(
        self,
        trajectory: np.ndarray,
        default_steps: int,
        adaptive_interp: bool,
        adaptive_max_joint_step_deg: float,
        adaptive_min_steps: int,
        adaptive_max_steps: Optional[int]
    ) -> List[int]:
        """Compute interpolation counts per segment."""
        num_segments = max(0, len(trajectory) - 1)
        if num_segments <= 0:
            return []

        base_steps = max(0, int(default_steps))
        segment_counts = [base_steps for _ in range(num_segments)]

        if not adaptive_interp:
            return segment_counts

        threshold_deg = adaptive_max_joint_step_deg if adaptive_max_joint_step_deg is not None else 0.0
        threshold_rad = np.deg2rad(max(threshold_deg, 0.0))

        for seg_idx in range(num_segments):
            start_config = trajectory[seg_idx]
            end_config = trajectory[seg_idx + 1]
            max_delta = float(np.max(np.abs(end_config - start_config)))

            if threshold_rad <= 0:
                steps = base_steps
            else:
                steps = int(np.ceil(max_delta / threshold_rad)) if max_delta > 0 else 0

            if adaptive_min_steps is not None:
                steps = max(adaptive_min_steps, steps)

            if adaptive_max_steps is not None:
                steps = min(adaptive_max_steps, steps)

            segment_counts[seg_idx] = max(0, int(steps))

        return segment_counts

    def check_collision_single_config(
        self,
        joint_positions: np.ndarray,
        return_distance: bool = False,
        return_link_info: bool = False
    ) -> Tuple[bool, float, Optional[List[Dict]]]:
        """
        Check collision for a single robot configuration

        Args:
            joint_positions: Array of joint angles
            return_distance: If True, return minimum distance
            return_link_info: If True, return detailed collision info per link

        Returns:
            (is_collision, distance, collision_info):
                - Collision flag
                - Distance (if requested)
                - List of collision info dicts (if return_link_info=True)
        """
        # Create robot collision geometry using FK
        robot_objects = self._create_robot_collision_geometry(joint_positions)

        if len(robot_objects) == 0:
            # No collision geometry created (could happen if link names don't match)
            return False, float('inf'), []

        # Check collision with each obstacle
        min_distance = float('inf')
        collision_detected = False
        collision_info = []

        # Get link names for reporting
        link_names = list(self.link_meshes.keys()) if self.link_meshes else \
                     list(self.collision_spheres.keys()) if self.collision_spheres else \
                     [cap['link_name'] for cap in self.link_capsules]

        for idx, robot_obj in enumerate(robot_objects):
            # Determine link name for this collision object
            if self.link_meshes:
                link_name = link_names[idx] if idx < len(link_names) else f"link_{idx}"
            elif self.collision_spheres:
                # For spheres, need to map back to link
                sphere_count = 0
                link_name = "unknown"
                for ln, spheres in self.collision_spheres.items():
                    if idx < sphere_count + len(spheres):
                        link_name = f"{ln}_sphere_{idx - sphere_count}"
                        break
                    sphere_count += len(spheres)

            for obstacle_obj in self.obstacle_collision_objects:
                # Collision check
                request = coal.CollisionRequest()
                request.security_margin = self.collision_margin
                result = coal.CollisionResult()

                ret = coal.collide(robot_obj, obstacle_obj, request, result)

                # Distance check
                dist_request = coal.DistanceRequest()
                dist_request.enable_signed_distance = True
                dist_result = coal.DistanceResult()
                dist = coal.distance(robot_obj, obstacle_obj, dist_request, dist_result)

                link_collision = False
                if ret > 0:  # Collision detected
                    link_collision = True
                    collision_detected = True
                    min_distance = 0.0
                elif dist < self.collision_margin:
                    # Within margin - treat as collision
                    link_collision = True
                    collision_detected = True

                # Store link collision info
                if return_link_info:
                    collision_info.append({
                        'link_name': link_name,
                        'collision': link_collision,
                        'distance': dist
                    })

                if link_collision and not return_distance and not return_link_info:
                    return True, 0.0, []

                min_distance = min(min_distance, dist)

        return collision_detected, min_distance, collision_info if return_link_info else None

    def check_trajectory(
        self,
        trajectory: np.ndarray,
        verbose: bool = True,
        show_link_collisions: bool = False,
        max_show: int = 10,
        interpolate: bool = True,
        num_interp_steps: int = 10,
        check_reconfig: bool = True,
        reconfig_threshold: float = 1.0,
        parallel: bool = False,
        num_workers: Optional[int] = None,
        adaptive_interp: bool = False,
        adaptive_max_joint_step_deg: float = 2.0,
        adaptive_min_steps: int = 0,
        adaptive_max_steps: Optional[int] = None,
    ) -> dict:
        """
        Check collisions and reconfigurations along entire trajectory

        Args:
            trajectory: (N, 6) array of joint configurations
            verbose: Print progress
            show_link_collisions: Show which links are colliding
            max_show: Maximum number of collision details to show
            interpolate: If True, check intermediate configurations between waypoints
            num_interp_steps: Number of interpolation steps between waypoints
            check_reconfig: If True, also check for joint reconfigurations
            reconfig_threshold: Threshold for joint reconfigurations in radians
            parallel: If True, use multiprocessing for collision checking
            num_workers: Number of worker processes (default: cpu_count() - 2)
            adaptive_interp: Use adaptive interpolation density based on joint deltas
            adaptive_max_joint_step_deg: Maximum joint delta (deg) allowed between samples
            adaptive_min_steps: Minimum interpolation steps per segment when adaptive mode is on
            adaptive_max_steps: Optional cap on interpolation steps per segment

        Returns:
            Dictionary with collision and reconfiguration statistics
        """
        from collections import Counter

        num_waypoints = len(trajectory)
        collision_indices = []
        collision_free_indices = []
        collision_segments = []  # List of (waypoint_idx, alpha) tuples for interpolated collisions
        link_collision_counter = Counter()
        collision_timer_start = time.perf_counter()

        # Determine if parallel processing should be used
        use_parallel = parallel
        if use_parallel:
            # Safety checks for parallel mode
            if num_waypoints < 500:
                if verbose:
                    print(f"  Note: Trajectory too small ({num_waypoints} waypoints) for parallel speedup")
                    print(f"  Using sequential processing instead")
                use_parallel = False
            elif num_workers is None:
                num_workers = max(1, cpu_count() - 2)
                if verbose:
                    print(f"  Parallel mode: Using {num_workers} workers (auto-detected)")
            else:
                if verbose:
                    print(f"  Parallel mode: Using {num_workers} workers")

        # Pre-compute all interpolations using batch processing if enabled
        interpolated_configs_map: Optional[Dict[int, List[Tuple[int, float, np.ndarray]]]] = None
        segment_step_counts: List[int] = []
        total_interp_points = 0
        if interpolate:
            segment_step_counts = self._compute_segment_interp_counts(
                trajectory,
                num_interp_steps,
                adaptive_interp,
                adaptive_max_joint_step_deg,
                adaptive_min_steps,
                adaptive_max_steps
            )
            total_interp_points = int(sum(segment_step_counts))

            print(f"\nChecking {num_waypoints} waypoints with interpolation "
                  f"(default {num_interp_steps} steps between waypoints)...")
            print(f"Total configurations to check: {num_waypoints + total_interp_points:,}")

            if adaptive_interp and segment_step_counts:
                nonzero_steps = [c for c in segment_step_counts if c > 0]
                if nonzero_steps:
                    print(f"  Adaptive interpolation: max joint step {adaptive_max_joint_step_deg:.2f} deg")
                    print(f"  Segment steps → min {min(nonzero_steps)}, max {max(nonzero_steps)}, "
                          f"avg {total_interp_points / len(segment_step_counts):.2f}")
                else:
                    print("  Adaptive interpolation resolved to 0 steps for all segments")

            if total_interp_points <= 0:
                print("  No interpolated configurations required; only waypoints will be checked.")
                interpolated_configs_map = {}
            else:
                batch_success = False

                if self.use_curobo_interpolation and not adaptive_interp and num_interp_steps > 0:
                    try:
                        batch_start = time.perf_counter()
                        batch_results = self._batch_interpolate_trajectory(trajectory, num_interp_steps)
                        batch_time = time.perf_counter() - batch_start

                        interpolated_configs_map = {}
                        for seg_idx, interp_idx, config in batch_results:
                            if seg_idx not in interpolated_configs_map:
                                interpolated_configs_map[seg_idx] = []
                            denom = segment_step_counts[seg_idx] + 1 if seg_idx < len(segment_step_counts) else (num_interp_steps + 1)
                            alpha = (interp_idx + 1) / denom if denom > 0 else 0.0
                            interpolated_configs_map[seg_idx].append((interp_idx, alpha, config))

                        print(f"  Batch interpolation completed in {batch_time:.3f}s (optimized)")
                        batch_success = True
                    except Exception as exc:
                        print(f"  Batch interpolation failed ({exc}), using segment-by-segment fallback")
                        interpolated_configs_map = None

                if not batch_success:
                    cpu_start = time.perf_counter()
                    if not self.use_curobo_interpolation:
                        print("  CuRobo interpolation disabled; using CPU linear interpolation")
                    interpolated_configs_map = self._precompute_segment_interpolations(
                        trajectory,
                        num_interp_steps,
                        force_cpu=not self.use_curobo_interpolation,
                        segment_step_counts=segment_step_counts
                    )
                    cpu_time = time.perf_counter() - cpu_start
                    print(f"  Linear interpolation precomputed in {cpu_time:.3f}s")
        else:
            print(f"\nChecking {num_waypoints} waypoints for collisions (no interpolation)...")

        shown_count = 0
        configs_checked = 0

        # === PARALLEL COLLISION CHECKING ===
        if use_parallel:
            # Collect all configurations to check
            all_configs = []
            config_metadata = []  # (type, waypoint_idx, segment_idx, interp_idx, alpha)

            # Add all waypoint configs
            for i, joint_config in enumerate(trajectory):
                all_configs.append(joint_config)
                config_metadata.append(('waypoint', i, None, None, None))

            # Add all interpolated configs
            if interpolate and interpolated_configs_map is not None:
                for seg_idx in range(num_waypoints - 1):
                    interpolated_list = interpolated_configs_map.get(seg_idx, [])
                    for interp_idx, alpha, interp_config in interpolated_list:
                        all_configs.append(interp_config)
                        config_metadata.append(('segment', seg_idx, seg_idx, interp_idx, alpha))

            # Prepare worker arguments (avoid pickling self)
            checker_args = {
                'robot_urdf_path': self.robot_urdf_path,
                'obstacle_mesh_paths': [str(p) for p in self.obstacle_meshes] if hasattr(self.obstacle_meshes[0], '__fspath__') else self.obstacle_meshes,
                'glass_position': self.glass_position,
                'table_position': self.table_position,
                'table_dimensions': self.table_dimensions,
                'wall_position': self.wall_position,
                'wall_dimensions': self.wall_dimensions,
                'workbench_position': self.workbench_position,
                'workbench_dimensions': self.workbench_dimensions,
                'robot_mount_position': self.robot_mount_position,
                'robot_mount_dimensions': self.robot_mount_dimensions,
                'robot_config_path': self.robot_config_path,
                'use_link_meshes': self.use_link_meshes,
                'mesh_base_path': self.mesh_base_path,
                'collision_margin': self.collision_margin,
                'use_curobo_interpolation': self.request_curobo_interpolation,
            }

            # Create work items
            work_items = [(i, config, show_link_collisions) for i, config in enumerate(all_configs)]

            # Process in parallel
            if verbose:
                print(f"  Checking {len(all_configs)} configurations in parallel...")

            try:
                with Pool(processes=num_workers, initializer=_init_worker, initargs=(checker_args,)) as pool:
                    results_list = pool.map(_check_collision_worker, work_items, chunksize=max(1, len(work_items) // (num_workers * 4)))

                configs_checked = len(results_list)

                # Process results
                for (idx, is_collision, dist, link_info) in results_list:
                    meta_type, wp_idx, seg_idx, interp_idx, alpha = config_metadata[idx]

                    if is_collision:
                        if meta_type == 'waypoint':
                            collision_indices.append(wp_idx)
                        else:  # segment
                            collision_segments.append((seg_idx, alpha))

                        # Collect link collision statistics
                        if show_link_collisions and link_info:
                            colliding_links = [info['link_name'] for info in link_info if info['collision']]
                            for link_name in colliding_links:
                                link_collision_counter[link_name] += 1
                    else:
                        if meta_type == 'waypoint':
                            collision_free_indices.append(wp_idx)

                if verbose:
                    print(f"  Parallel checking completed: {configs_checked} configurations")

            except Exception as e:
                print(f"  Warning: Parallel processing failed ({e})")
                print(f"  Falling back to sequential processing...")
                use_parallel = False

        # === SEQUENTIAL COLLISION CHECKING ===
        if not use_parallel:
            # Check all waypoints
            for i, joint_config in enumerate(trajectory):
                is_collision, dist, link_info = self.check_collision_single_config(
                    joint_config,
                    return_distance=True,
                    return_link_info=show_link_collisions
                )
                configs_checked += 1

                if is_collision:
                    collision_indices.append(i)

                    # Collect link collision statistics
                    if show_link_collisions and link_info:
                        colliding_links = [info['link_name'] for info in link_info if info['collision']]
                        for link_name in colliding_links:
                            link_collision_counter[link_name] += 1

                        # Show detailed info for first few collisions
                        if shown_count < max_show:
                            print(f"\n  Waypoint {i}: COLLISION")
                            print(f"    Colliding links: {', '.join(colliding_links)}")
                            shown_count += 1
                else:
                    collision_free_indices.append(i)

                # Check interpolated segments if enabled
                if interpolate and i < num_waypoints - 1:
                    # Use pre-computed batch results if available, otherwise fallback to per-segment
                    if interpolated_configs_map is not None:
                        # Fast path: use pre-computed interpolations
                        interpolated_list = interpolated_configs_map.get(i, [])
                        for interp_idx, alpha, interp_config in interpolated_list:
                            is_collision, dist, link_info = self.check_collision_single_config(
                                interp_config,
                                return_distance=True,
                                return_link_info=show_link_collisions
                            )
                            configs_checked += 1

                            if is_collision:
                                # Calculate alpha value for this interpolation point
                                collision_segments.append((i, alpha))

                                # Collect link collision statistics
                                if show_link_collisions and link_info:
                                    colliding_links = [info['link_name'] for info in link_info if info['collision']]
                                    for link_name in colliding_links:
                                        link_collision_counter[link_name] += 1

                                    # Show detailed info for first few collisions
                                    if shown_count < max_show:
                                        print(f"\n  Segment {i}→{i+1} (α={alpha:.2f}): COLLISION")
                                        print(f"    Colliding links: {', '.join(colliding_links)}")
                                        shown_count += 1
                    else:
                        # Fallback path: compute interpolations per-segment
                        start_config = trajectory[i]
                        end_config = trajectory[i + 1]
                        segment_steps = segment_step_counts[i] if i < len(segment_step_counts) else num_interp_steps
                        segment_steps = max(0, int(segment_steps))

                        interpolated_configs = []
                        if segment_steps > 0:
                            interpolated_configs = self._generate_segment_interpolation(
                                start_config, end_config, segment_steps
                            )

                        # Check each interpolated configuration
                        for interp_idx, interp_config in enumerate(interpolated_configs):
                            is_collision, dist, link_info = self.check_collision_single_config(
                                interp_config,
                                return_distance=True,
                                return_link_info=show_link_collisions
                            )
                            configs_checked += 1

                            if is_collision:
                                # Calculate alpha value for this interpolation point
                                denom = segment_steps + 1 if segment_steps > 0 else 1
                                alpha = (interp_idx + 1) / denom
                                collision_segments.append((i, alpha))

                                # Collect link collision statistics
                                if show_link_collisions and link_info:
                                    colliding_links = [info['link_name'] for info in link_info if info['collision']]
                                    for link_name in colliding_links:
                                        link_collision_counter[link_name] += 1

                                    # Show detailed info for first few collisions
                                    if shown_count < max_show:
                                        print(f"\n  Segment {i}→{i+1} (α={alpha:.2f}): COLLISION")
                                        print(f"    Colliding links: {', '.join(colliding_links)}")
                                        shown_count += 1

                # Progress reporting
                if verbose and interpolate:
                    if configs_checked % 500 == 0:
                        total_to_check = num_waypoints + total_interp_points
                        print(f"  Progress: {configs_checked}/{total_to_check} configurations checked")
                elif verbose and not interpolate:
                    if (i + 1) % 100 == 0:
                        print(f"  Progress: {i+1}/{num_waypoints} waypoints checked")

        # Calculate statistics
        num_collisions = len(collision_indices)
        num_segment_collisions = len(collision_segments)
        total_collisions = num_collisions + num_segment_collisions

        if interpolate:
            total_configs = num_waypoints + total_interp_points
            collision_rate = (total_collisions / total_configs * 100) if total_configs > 0 else 0.0
        else:
            total_configs = num_waypoints
            collision_rate = (num_collisions / total_configs * 100) if total_configs > 0 else 0.0

        results = {
            'total_waypoints': num_waypoints,
            'total_configs_checked': configs_checked,
            'interpolate': interpolate,
            'num_interp_steps': num_interp_steps if interpolate else 0,
            'total_interpolated_configs': total_interp_points if interpolate else 0,
            'num_collisions': num_collisions,
            'num_segment_collisions': num_segment_collisions if interpolate else 0,
            'total_collisions': total_collisions,
            'num_collision_free': len(collision_free_indices),
            'collision_rate': collision_rate,
            'collision_indices': collision_indices,
            'collision_segments': collision_segments if interpolate else [],
            'collision_free_indices': collision_free_indices,
            'link_collisions': dict(link_collision_counter) if show_link_collisions else {},
            'segment_interp_counts': segment_step_counts if interpolate else [],
            'adaptive_interp': adaptive_interp,
            'adaptive_interp_params': {
                'max_joint_step_deg': adaptive_max_joint_step_deg,
                'min_steps': adaptive_min_steps,
                'max_steps': adaptive_max_steps,
            } if interpolate else None
        }
        results['collision_check_time_sec'] = time.perf_counter() - collision_timer_start
        results['reconfig_check_time_sec'] = 0.0

        # Check for joint reconfigurations if requested
        if check_reconfig:
            if verbose:
                print(f"\nChecking for joint reconfigurations...")
            reconfig_timer_start = time.perf_counter()
            reconfig_results = self.detect_joint_reconfigurations(
                trajectory,
                threshold=reconfig_threshold,
                exclude_last_joint=True,
            )
            results['reconfig_check_time_sec'] = time.perf_counter() - reconfig_timer_start
            results.update({
                'reconfiguration_segments': reconfig_results['reconfiguration_segments'],
                'num_reconfigurations': reconfig_results['num_reconfigurations'],
                'reconfiguration_rate': reconfig_results['reconfiguration_rate'],
                'reconfigurations_per_joint': reconfig_results['reconfigurations_per_joint'],
                'max_changes_per_joint': reconfig_results['max_changes_per_joint'],
                'mean_changes_per_joint': reconfig_results['mean_changes_per_joint'],
                'max_changes_per_segment': reconfig_results['max_changes_per_segment'],
                'reconfig_threshold': reconfig_results['threshold_used'],
                'excluded_last_joint': reconfig_results['excluded_last_joint']
            })
            if verbose:
                print(f"  Found {reconfig_results['num_reconfigurations']} reconfigurations")
                print(f"  Reconfiguration rate: {reconfig_results['reconfiguration_rate']:.1%}")
        else:
            # Add empty reconfiguration data
            results.update({
                'reconfiguration_segments': [],
                'num_reconfigurations': 0,
                'reconfiguration_rate': 0.0,
                'reconfigurations_per_joint': [],
                'max_changes_per_joint': [],
                'mean_changes_per_joint': [],
                'max_changes_per_segment': [],
                'reconfig_threshold': 0.0,
                'excluded_last_joint': False
            })

        return results

    def check_trajectory_segments(
        self,
        trajectory: np.ndarray,
        segment_indices: List[int],
        verbose: bool = False,
        show_link_collisions: bool = False,
        max_show: int = 10,
        interpolate: bool = True,
        num_interp_steps: int = 10,
        adaptive_interp: bool = False,
        adaptive_max_joint_step_deg: float = 2.0,
        adaptive_min_steps: int = 0,
        adaptive_max_steps: Optional[int] = None
    ) -> dict:
        """
        Check collisions only for specific segments of the trajectory

        Args:
            trajectory: (N, 6) array of joint configurations
            segment_indices: List of segment indices to check (segment i is between waypoint i and i+1)
            verbose: Print progress
            show_link_collisions: Show which links are colliding
            max_show: Maximum number of collision details to show
            interpolate: If True, check intermediate configurations between waypoints
            num_interp_steps: Number of interpolation steps between waypoints (baseline)
            adaptive_interp: Use adaptive interpolation density based on joint deltas
            adaptive_max_joint_step_deg: Maximum joint delta (deg) between interpolated samples
            adaptive_min_steps: Minimum interpolation steps per segment when adaptive mode is on
            adaptive_max_steps: Optional cap on interpolation steps per segment

        Returns:
            Dictionary with collision statistics for the checked segments
        """
        from collections import Counter

        num_waypoints = len(trajectory)
        collision_indices = []
        collision_free_indices = []
        collision_segments = []
        link_collision_counter = Counter()

        # Create a set of waypoints to check (endpoints of segments)
        waypoints_to_check = set()
        valid_segments = [seg_idx for seg_idx in segment_indices if 0 <= seg_idx < num_waypoints - 1]
        for seg_idx in valid_segments:
            waypoints_to_check.add(seg_idx)
            waypoints_to_check.add(seg_idx + 1)

        segment_step_counts = []
        total_interp_points = 0
        if interpolate and num_waypoints > 1:
            segment_step_counts = self._compute_segment_interp_counts(
                trajectory,
                num_interp_steps,
                adaptive_interp,
                adaptive_max_joint_step_deg,
                adaptive_min_steps,
                adaptive_max_steps
            )
            total_interp_points = int(sum(segment_step_counts[seg] for seg in valid_segments))

        if verbose:
            print(f"\nChecking {len(segment_indices)} segments (waypoint pairs: {len(waypoints_to_check)})...")
            if interpolate:
                total_configs = len(waypoints_to_check) + total_interp_points
                print(f"Total configurations to check: {total_configs:,}")
                if adaptive_interp and segment_step_counts:
                    nonzero_steps = [segment_step_counts[seg] for seg in valid_segments if segment_step_counts[seg] > 0]
                    if nonzero_steps:
                        print(f"  Adaptive interpolation: max joint step {adaptive_max_joint_step_deg:.2f} deg")
                        print(f"  Segment steps for selection → min {min(nonzero_steps)}, max {max(nonzero_steps)}")

        shown_count = 0
        configs_checked = 0

        # Check waypoints that are endpoints of segments to check
        for waypoint_idx in sorted(waypoints_to_check):
            joint_config = trajectory[waypoint_idx]
            is_collision, dist, link_info = self.check_collision_single_config(
                joint_config,
                return_distance=True,
                return_link_info=show_link_collisions
            )
            configs_checked += 1

            if is_collision:
                collision_indices.append(waypoint_idx)

                if show_link_collisions and link_info:
                    colliding_links = [info['link_name'] for info in link_info if info['collision']]
                    for link_name in colliding_links:
                        link_collision_counter[link_name] += 1

                    if shown_count < max_show:
                        print(f"\n  Waypoint {waypoint_idx}: COLLISION")
                        print(f"    Colliding links: {', '.join(colliding_links)}")
                        shown_count += 1
            else:
                collision_free_indices.append(waypoint_idx)

        # Check interpolated segments
        if interpolate:
            for seg_idx in valid_segments:
                start_config = trajectory[seg_idx]
                end_config = trajectory[seg_idx + 1]
                steps = segment_step_counts[seg_idx] if seg_idx < len(segment_step_counts) else num_interp_steps
                steps = max(0, int(steps))

                if steps <= 0:
                    continue

                interpolated_configs = self._generate_segment_interpolation(
                    start_config, end_config, steps
                )

                for interp_idx, interp_config in enumerate(interpolated_configs):
                    is_collision, dist, link_info = self.check_collision_single_config(
                        interp_config,
                        return_distance=True,
                        return_link_info=show_link_collisions
                    )
                    configs_checked += 1

                    if is_collision:
                        alpha = (interp_idx + 1) / (steps + 1)
                        collision_segments.append((seg_idx, alpha))

                        if show_link_collisions and link_info:
                            colliding_links = [info['link_name'] for info in link_info if info['collision']]
                            for link_name in colliding_links:
                                link_collision_counter[link_name] += 1

                            if shown_count < max_show:
                                print(f"\n  Segment {seg_idx}→{seg_idx+1} (α={alpha:.2f}): COLLISION")
                                print(f"    Colliding links: {', '.join(colliding_links)}")
                                shown_count += 1

        # Calculate statistics
        num_collisions = len(collision_indices)
        num_segment_collisions = len(collision_segments)
        total_collisions = num_collisions + num_segment_collisions

        results = {
            'configs_checked': configs_checked,
            'total_configs_checked': configs_checked,
            'num_collisions': num_collisions,
            'num_segment_collisions': num_segment_collisions if interpolate else 0,
            'total_collisions': total_collisions,
            'num_collision_free': len(collision_free_indices),
            'collision_indices': collision_indices,
            'collision_segments': collision_segments if interpolate else [],
            'collision_free_indices': collision_free_indices,
            'link_collisions': dict(link_collision_counter) if show_link_collisions else {},
            'interpolate': interpolate,
            'num_interp_steps': num_interp_steps if interpolate else 0,
            'total_interpolated_configs': total_interp_points if interpolate else 0,
            'segment_interp_counts': segment_step_counts if interpolate else [],
            'adaptive_interp': adaptive_interp,
            'adaptive_interp_params': {
                'max_joint_step_deg': adaptive_max_joint_step_deg,
                'min_steps': adaptive_min_steps,
                'max_steps': adaptive_max_steps,
            } if interpolate else None
        }

        return results

    def detect_joint_reconfigurations(
        self,
        trajectory: np.ndarray,
        threshold: float = 1.0,
        exclude_last_joint: bool = True
    ) -> dict:
        """
        Detect joint reconfigurations (sudden large joint movements) in trajectory

        Args:
            trajectory: (N, n_joints) array of joint configurations
            threshold: Minimum joint change (radians) to count as reconfiguration
            exclude_last_joint: If True, exclude the last joint from reconfiguration analysis

        Returns:
            Dictionary with reconfiguration statistics
        """
        n_timesteps, n_joints = trajectory.shape

        # Calculate joint differences between consecutive waypoints
        joint_diffs = np.diff(trajectory, axis=0)  # Shape: (n_timesteps-1, n_joints)
        joint_changes = np.abs(joint_diffs)

        # Create mask to exclude last joint if requested
        if exclude_last_joint:
            joint_mask = np.ones(n_joints, dtype=bool)
            joint_mask[-1] = False  # Exclude last joint
            joint_changes_for_reconfig = joint_changes[:, joint_mask]
        else:
            joint_mask = np.ones(n_joints, dtype=bool)
            joint_changes_for_reconfig = joint_changes

        # Count reconfigurations per joint (for statistics)
        reconfigurations_per_joint = np.sum(joint_changes > threshold, axis=0)

        # Count total reconfigurations (any joint exceeding threshold, excluding last if requested)
        total_reconfigurations = np.sum(np.any(joint_changes_for_reconfig > threshold, axis=1))

        # Calculate movement statistics
        max_changes_per_joint = np.max(joint_changes, axis=0)
        mean_changes_per_joint = np.mean(joint_changes, axis=0)

        # Find segments with large reconfigurations
        reconfiguration_segments = []
        max_changes_per_segment = []

        for i in range(joint_changes.shape[0]):
            if np.any(joint_changes_for_reconfig[i] > threshold):
                max_change = np.max(joint_changes_for_reconfig[i])
                # Segment i is between waypoint i and waypoint i+1
                reconfiguration_segments.append(i)
                max_changes_per_segment.append(max_change)

        return {
            'reconfiguration_segments': reconfiguration_segments,
            'num_reconfigurations': int(total_reconfigurations),
            'reconfiguration_rate': float(total_reconfigurations) / (n_timesteps - 1) if n_timesteps > 1 else 0.0,
            'reconfigurations_per_joint': reconfigurations_per_joint.tolist(),
            'max_changes_per_joint': max_changes_per_joint.tolist(),
            'mean_changes_per_joint': mean_changes_per_joint.tolist(),
            'max_changes_per_segment': max_changes_per_segment,
            'threshold_used': threshold,
            'excluded_last_joint': exclude_last_joint
        }


def load_trajectory_csv(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load joint trajectory from CSV file

    Returns:
        (trajectory, joint_names): Joint angles array and joint names
    """
    joint_angles = []
    joint_names = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        # Extract joint column names (assuming they contain 'joint')
        joint_names = [h for h in headers if 'joint' in h.lower()]

        for row in reader:
            config = [float(row[joint_name]) for joint_name in joint_names]
            joint_angles.append(config)

    trajectory = np.array(joint_angles)
    print(f"Loaded trajectory: {len(trajectory)} waypoints, {len(joint_names)} joints")
    print(f"Joint names: {joint_names}")

    return trajectory, joint_names


def save_trajectory_csv(
    trajectory: np.ndarray,
    joint_names: List[str],
    output_path: Path,
    time_step: float = 1.0
) -> Path:
    """
    Save joint trajectory to CSV using provided joint names.

    Uses an explicit 'time' column patterned after plan_trajectory.py where time is an
    incrementing multiple of `time_step` (default: 1.0 second per row).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not joint_names:
        joint_names = [f"joint_{idx}" for idx in range(trajectory.shape[1])]

    fieldnames = ['time'] + joint_names

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, config in enumerate(trajectory):
            row = {'time': float(idx) * time_step}
            row.update({name: float(config[i]) for i, name in enumerate(joint_names)})
            writer.writerow(row)

    return output_path


def save_collision_report(
    trajectory_path: str,
    results,
    replan_summary: Optional[dict] = None,
    timing_info: Optional[Dict[str, float]] = None
) -> Path:
    """
    Persist collision summary to data/collision/{num_points}/collision.txt
    (appends to the file so multiple runs are logged sequentially).
    """
    base_dir = Path(__file__).resolve().parent.parent
    # Extract directory number from trajectory path (e.g., data/trajectory/1158/... -> 1158)
    traj_path = Path(trajectory_path)
    parent_dir_name = traj_path.parent.name
    # Use parent directory name if it looks like a number, otherwise use filename stem
    if parent_dir_name.isdigit():
        num_points = parent_dir_name
    else:
        num_points = traj_path.stem or "unknown"
    report_dir = base_dir / 'data' / 'collision' / str(num_points)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / 'collision.txt'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    unique_segments = sorted({wp_idx for wp_idx, _ in results.get('collision_segments', [])})
    mesh_info = config.DEFAULT_MESH_FILE
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
        f"=== Collision Report @ {timestamp} ===",
        f"Trajectory: {trajectory_path}",
        f"Robot URDF: {config.DEFAULT_ROBOT_URDF}",
        f"Robot config: {config.DEFAULT_ROBOT_CONFIG_YAML}",
        f"Obstacle meshes: {mesh_info}",
        f"Collision margin: {config.COLLISION_MARGIN}",
        f"Interpolation enabled: {results['interpolate']}",
        f"Interpolation steps (default): {results['num_interp_steps'] if results['interpolate'] else 0}",
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

    # Add reconfiguration statistics if available
    if results.get('reconfig_threshold', 0) > 0:
        lines.append("")
        lines.append("Joint Reconfiguration Analysis:")
        lines.append(f"  Threshold: {results.get('reconfig_threshold', 0):.2f} rad")
        lines.append(f"  Excluded last joint: {results.get('excluded_last_joint', False)}")
        lines.append(f"  Total reconfigurations: {results.get('num_reconfigurations', 0)}")
        lines.append(f"  Reconfiguration rate: {results.get('reconfiguration_rate', 0):.1%}")
        reconfig_segs = results.get('reconfiguration_segments', [])
        lines.append(f"  Reconfiguration segments: {format_list(reconfig_segs)}")

    link_collisions = results.get('link_collisions', {})
    if link_collisions:
        lines.append("")
        lines.append("Collisions by link:")
        for link_name, count in sorted(link_collisions.items(), key=lambda item: item[1], reverse=True):
            lines.append(f"  - {link_name}: {count}")
    else:
        lines.append("")
        lines.append("Collisions by link: None")

    if replan_summary:
        lines.append("")
        lines.append("Replanning Summary:")
        attempted = replan_summary.get('attempted', False)
        lines.append(f"  Attempted: {attempted}")
        lines.append(f"  Success: {replan_summary.get('success', False)}")
        segments_requested = replan_summary.get('segments_requested') or []
        collision_segs_requested = replan_summary.get('collision_segments_requested') or []
        reconfig_segs_requested = replan_summary.get('reconfiguration_segments_requested') or []
        segments_replanned = replan_summary.get('segments_replanned') or []
        segments_failed = replan_summary.get('segments_failed') or []
        lines.append(f"  Total segments requested: {format_list(segments_requested)}")
        lines.append(f"    - Collision segments: {format_list(collision_segs_requested)}")
        lines.append(f"    - Reconfiguration segments: {format_list(reconfig_segs_requested)}")
        lines.append(f"  Segments replanned: {format_list(segments_replanned)}")
        if segments_failed:
            failure_str = ", ".join(
                f"{item.get('segment')} ({item.get('status', 'failed')})" for item in segments_failed[:50]
            )
        else:
            failure_str = "None"
        lines.append(f"  Failed segments: {failure_str}")
        if replan_summary.get('output_path'):
            lines.append(f"  Collision-free CSV: {replan_summary['output_path']}")
        if replan_summary.get('message'):
            lines.append(f"  Notes: {replan_summary['message']}")

    lines.append("")
    lines.append("Occurrence Counts:")
    lines.append(f"  Collisions: {results.get('total_collisions', 0)}")
    lines.append(f"  Reconfigurations: {results.get('num_reconfigurations', 0)}")

    if timing_info:
        lines.append("")
        lines.append("Timing (wall-clock):")
        lines.append(f"  Collision check:         {format_time(timing_info.get('collision_check_sec'))}")
        lines.append(f"  Reconfiguration check:  {format_time(timing_info.get('reconfig_check_sec'))}")
        lines.append(f"  CuRobo replanning:      {format_time(timing_info.get('replan_sec'))}")
        lines.append(f"  Total runtime:          {format_time(timing_info.get('total_runtime_sec'))}")

    content = "\n".join(lines)
    if report_path.exists() and report_path.stat().st_size > 0:
        with open(report_path, 'a') as f:
            f.write("\n\n" + content)
    else:
        with open(report_path, 'w') as f:
            f.write(content)

    return report_path


def determine_segments_to_replan(num_segments: int, results: dict) -> dict:
    """
    Map waypoint/segment collision and reconfiguration info to concrete segment indices.

    Args:
        num_segments: Total number of segments in trajectory
        results: Dictionary containing collision and reconfiguration results

    Returns:
        Dictionary with keys:
            - 'all': All segments to replan (collision + reconfiguration)
            - 'collision': Segments with collisions only
            - 'reconfiguration': Segments with reconfigurations only
    """
    collision_segments = set()
    reconfig_segments = set()

    # Add collision segments from waypoint collisions
    for idx in results.get('collision_indices', []):
        if idx > 0:
            collision_segments.add(idx - 1)
        if idx < num_segments:
            collision_segments.add(idx)

    # Add collision segments from interpolated segment collisions
    for seg_idx, _ in results.get('collision_segments', []):
        if 0 <= seg_idx < num_segments:
            collision_segments.add(seg_idx)

    # Add reconfiguration segments
    for seg_idx in results.get('reconfiguration_segments', []):
        if 0 <= seg_idx < num_segments:
            reconfig_segments.add(seg_idx)

    # Combine all segments (union)
    all_segments = collision_segments | reconfig_segments

    return {
        'all': sorted(all_segments),
        'collision': sorted(collision_segments),
        'reconfiguration': sorted(reconfig_segments)
    }


if CUROBO_AVAILABLE:

    def build_motion_world_config(
        glass_position: np.ndarray,
        table_position: np.ndarray,
        table_dimensions: np.ndarray,
        wall_position: np.ndarray,
        wall_dimensions: np.ndarray,
        workbench_position: np.ndarray,
        workbench_dimensions: np.ndarray,
        robot_mount_position: np.ndarray,
        robot_mount_dimensions: np.ndarray,
        mesh_paths: List[str],
    ) -> WorldConfig:
        """Create a CuRobo WorldConfig mirroring the COAL environment."""
        base_world = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        base_world.cuboid[0].pose[:3] = table_position.tolist()
        base_world.cuboid[0].dims[:3] = table_dimensions.tolist()
        base_world.cuboid[0].name = "table"

        def cuboid_from_pose(name: str, position: np.ndarray, dims: np.ndarray) -> WorldConfig:
            cuboid_dict = {
                name: {
                    "dims": dims.tolist(),
                    "pose": list(position) + [1, 0, 0, 0],
                }
            }
            cfg = WorldConfig.from_dict({"cuboid": cuboid_dict})
            cfg.cuboid[0].name = name
            return cfg

        wall_cfg = cuboid_from_pose("wall", wall_position, wall_dimensions)
        workbench_cfg = cuboid_from_pose("workbench", workbench_position, workbench_dimensions)
        robot_mount_cfg = cuboid_from_pose("robot_mount", robot_mount_position, robot_mount_dimensions)

        mesh_list = []
        base_pose = list(glass_position) + [1, 0, 0, 0]
        for mesh_path in mesh_paths:
            mesh_path = str(Path(mesh_path).resolve())
            mesh_list.append(
                Mesh(
                    name=Path(mesh_path).stem,
                    file_path=mesh_path,
                    pose=base_pose,
                )
            )

        all_cuboids = (
            base_world.cuboid +
            wall_cfg.cuboid +
            workbench_cfg.cuboid +
            robot_mount_cfg.cuboid
        )

        return WorldConfig(
            cuboid=all_cuboids,
            mesh=mesh_list
        )


    class CuRoboMotionPlanner:
        """Wrapper around CuRobo MotionGen for segment replanning."""

        def __init__(
            self,
            robot_config_path: str,
            world_config: WorldConfig,
            trajopt_tsteps: int = 32,
            interpolation_dt: float = 0.01,
            interpolation_steps: int = 5000,
            collision_checker_type=CollisionCheckerType.MESH,
        ):
            self.ready = False
            self.motion_gen = None
            self.init_error = None
            self.tensor_args = TensorDeviceType() if TensorDeviceType else None

            try:
                robot_cfg = self._load_robot_config(robot_config_path)
                self.dof = len(robot_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"])

                motion_gen_cfg = MotionGenConfig.load_from_robot_config(
                    robot_cfg,
                    world_config,
                    self.tensor_args,
                    trajopt_tsteps=trajopt_tsteps,
                    interpolation_dt=interpolation_dt,
                    interpolation_steps=interpolation_steps,
                    num_graph_seeds=4,
                    num_trajopt_seeds=4,
                    num_trajopt_noisy_seeds=2,
                    num_ik_seeds=16,
                    collision_checker_type=collision_checker_type,
                    evaluate_interpolated_trajectory=True,
                    use_cuda_graph=True,
                )
                self.motion_gen = MotionGen(motion_gen_cfg)
                self.motion_gen.warmup(n_goalset=1)
                self.ready = True
            except Exception as exc:  # pylint: disable=broad-except
                self.init_error = str(exc)
                self.motion_gen = None

        def _load_robot_config(self, robot_config_path: str) -> dict:
            abs_path = Path(robot_config_path)
            if not abs_path.is_absolute():
                abs_path = Path(__file__).resolve().parent.parent / abs_path
            with open(abs_path, 'r') as f:
                data = yaml.safe_load(f)
            if "robot_cfg" not in data:
                data = {"robot_cfg": data}
            return data

        def _to_joint_state(self, config: np.ndarray) -> JointState:
            tensor = torch.as_tensor(
                config,
                dtype=self.tensor_args.dtype,
                device=self.tensor_args.device
            ).view(1, -1)
            return JointState.from_position(tensor)

        def _joint_state_to_numpy(self, joint_state: JointState) -> Optional[np.ndarray]:
            if joint_state is None:
                return None
            positions = joint_state.position
            if isinstance(positions, torch.Tensor):
                arr = positions.detach().cpu().numpy()
            else:
                arr = np.asarray(positions)
            if arr.ndim == 3:
                arr = arr[0]
            return np.array(arr, dtype=np.float64)

        def plan_segment(
            self,
            start_config: np.ndarray,
            goal_config: np.ndarray,
            timeout: float,
            max_attempts: int,
        ) -> Dict:
            if not self.ready or self.motion_gen is None:
                return {
                    'success': False,
                    'status': self.init_error or 'planner_not_initialized'
                }

            plan_cfg = MotionGenPlanConfig(
                enable_graph=True,
                timeout=timeout,
                max_attempts=max_attempts,
                enable_graph_attempt=5,
                need_graph_success=False,
                use_start_state_as_retract=True,
            )

            try:
                start_state = self._to_joint_state(start_config)
                goal_state = self._to_joint_state(goal_config)
                result = self.motion_gen.plan_single_js(start_state, goal_state, plan_cfg)
            except Exception as exc:  # pylint: disable=broad-except
                return {'success': False, 'status': str(exc)}

            success_tensor = getattr(result, "success", None)
            success = bool(success_tensor.detach().cpu().numpy()[0]) if success_tensor is not None else False

            path = None
            if success:
                joint_state = result.get_interpolated_plan() or result.optimized_plan
                path = self._joint_state_to_numpy(joint_state)
                success = path is not None

            return {
                'success': success,
                'path': path,
                'status': str(getattr(result, "status", "success")),
                'solve_time': getattr(result, "total_time", None),
            }

        def plan_segments_batch(
            self,
            start_configs: List[np.ndarray],
            goal_configs: List[np.ndarray],
            timeout: float,
            max_attempts: int,
        ) -> List[Dict]:
            """
            Plan multiple segments sequentially (optimized for CuRobo warmup)

            Note: True batch planning with plan_batch() requires specific CuRobo setup.
            This method uses sequential planning but benefits from GPU warmup and
            CUDA graph optimization after the first call.

            Args:
                start_configs: List of start configurations
                goal_configs: List of goal configurations
                timeout: Timeout for each planning query
                max_attempts: Maximum attempts for each query

            Returns:
                List of dictionaries with planning results for each segment
            """
            if not self.ready or self.motion_gen is None:
                return [{
                    'success': False,
                    'status': self.init_error or 'planner_not_initialized',
                    'path': None
                } for _ in range(len(start_configs))]

            if len(start_configs) != len(goal_configs):
                raise ValueError("start_configs and goal_configs must have same length")

            batch_size = len(start_configs)
            if batch_size == 0:
                return []

            # Sequential planning with optimized execution
            # CuRobo's CUDA graphs make subsequent calls much faster
            results = []
            for start_config, goal_config in zip(start_configs, goal_configs):
                result = self.plan_segment(start_config, goal_config, timeout, max_attempts)
                results.append(result)

            return results


def merge_collision_results(
    original_results: dict,
    segment_results: dict,
    replanned_segments: List[int],
    new_num_waypoints: int
) -> dict:
    """
    Merge original collision results with results from rechecked segments.

    Args:
        original_results: Original collision check results for entire trajectory
        segment_results: Collision check results for replanned segments only
        replanned_segments: List of segment indices that were replanned
        new_num_waypoints: Total number of waypoints in new trajectory

    Returns:
        Merged collision results dictionary
    """
    from collections import Counter

    # Start with collision-free assumption for all waypoints
    all_collision_indices = set(original_results['collision_indices'])
    all_collision_free_indices = set(original_results['collision_free_indices'])
    all_collision_segments = list(original_results['collision_segments'])

    # Remove old collision data for replanned segments
    replanned_set = set(replanned_segments)

    # Remove waypoint collisions at replanned segment endpoints
    waypoints_to_remove = set()
    for seg_idx in replanned_set:
        waypoints_to_remove.add(seg_idx)
        waypoints_to_remove.add(seg_idx + 1)

    all_collision_indices -= waypoints_to_remove
    all_collision_free_indices -= waypoints_to_remove

    # Remove segment collisions for replanned segments
    all_collision_segments = [
        (seg_idx, alpha) for seg_idx, alpha in all_collision_segments
        if seg_idx not in replanned_set
    ]

    # Add new results from segment recheck
    for idx in segment_results['collision_indices']:
        all_collision_indices.add(idx)
        all_collision_free_indices.discard(idx)

    for idx in segment_results['collision_free_indices']:
        all_collision_free_indices.add(idx)
        all_collision_indices.discard(idx)

    all_collision_segments.extend(segment_results['collision_segments'])

    # Merge link collisions
    link_collisions = Counter(original_results.get('link_collisions', {}))
    link_collisions.update(segment_results.get('link_collisions', {}))

    # Calculate final statistics
    num_collisions = len(all_collision_indices)
    num_segment_collisions = len(all_collision_segments)
    total_collisions = num_collisions + num_segment_collisions

    # Calculate total configs checked
    original_configs = original_results['total_configs_checked']
    segment_configs = segment_results['configs_checked']

    orig_segment_counts = original_results.get('segment_interp_counts', [])
    new_segment_counts = segment_results.get('segment_interp_counts', [])

    old_segment_interp = 0
    for seg_idx in replanned_segments:
        if seg_idx < len(orig_segment_counts):
            old_segment_interp += orig_segment_counts[seg_idx]

    if original_results['interpolate']:
        old_segment_configs = len(waypoints_to_remove) + old_segment_interp
    else:
        old_segment_configs = len(waypoints_to_remove)

    total_configs_checked = original_configs - old_segment_configs + segment_configs

    # Merge segment interpolation counts
    if original_results['interpolate']:
        if len(new_segment_counts) == len(orig_segment_counts) and len(orig_segment_counts) > 0:
            merged_segment_counts = list(orig_segment_counts)
            for seg_idx in replanned_segments:
                if seg_idx < len(merged_segment_counts):
                    merged_segment_counts[seg_idx] = new_segment_counts[seg_idx] if seg_idx < len(new_segment_counts) else 0
        else:
            # Length mismatch or missing data – fall back to new counts if available
            merged_segment_counts = list(new_segment_counts) if new_segment_counts else list(orig_segment_counts)
    else:
        merged_segment_counts = []

    new_total_interp = sum(merged_segment_counts)

    # Calculate collision rate
    if original_results['interpolate']:
        total_possible = new_num_waypoints + new_total_interp
        collision_rate = total_collisions / total_possible * 100 if total_possible > 0 else 0
    else:
        collision_rate = num_collisions / new_num_waypoints * 100 if new_num_waypoints > 0 else 0

    # Merge reconfiguration results if available
    merged_results = {
        'total_waypoints': new_num_waypoints,
        'total_configs_checked': total_configs_checked,
        'interpolate': original_results['interpolate'],
        'num_interp_steps': original_results['num_interp_steps'],
        'num_collisions': num_collisions,
        'num_segment_collisions': num_segment_collisions,
        'total_collisions': total_collisions,
        'num_collision_free': len(all_collision_free_indices),
        'collision_rate': collision_rate,
        'collision_indices': sorted(all_collision_indices),
        'collision_segments': sorted(all_collision_segments),
        'collision_free_indices': sorted(all_collision_free_indices),
        'link_collisions': dict(link_collisions),
        'segment_interp_counts': merged_segment_counts,
        'total_interpolated_configs': new_total_interp if original_results['interpolate'] else 0,
        'adaptive_interp': original_results.get('adaptive_interp', False),
        'adaptive_interp_params': original_results.get('adaptive_interp_params')
    }

    # Add reconfiguration data from segment_results if available
    if 'reconfiguration_segments' in segment_results:
        # Get original reconfiguration segments (excluding replanned ones)
        all_reconfig_segments = set(original_results.get('reconfiguration_segments', []))
        all_reconfig_segments -= replanned_set  # Remove replanned segments

        # Add new reconfiguration segments from recheck
        all_reconfig_segments.update(segment_results.get('reconfiguration_segments', []))

        merged_results.update({
            'reconfiguration_segments': sorted(all_reconfig_segments),
            'num_reconfigurations': len(all_reconfig_segments),
            'reconfiguration_rate': len(all_reconfig_segments) / (new_num_waypoints - 1) if new_num_waypoints > 1 else 0.0,
            'reconfigurations_per_joint': segment_results.get('reconfigurations_per_joint', []),
            'max_changes_per_joint': segment_results.get('max_changes_per_joint', []),
            'mean_changes_per_joint': segment_results.get('mean_changes_per_joint', []),
            'max_changes_per_segment': segment_results.get('max_changes_per_segment', []),
            'reconfig_threshold': segment_results.get('reconfig_threshold', 0.0),
            'excluded_last_joint': segment_results.get('excluded_last_joint', False)
        })
    else:
        # Preserve original reconfiguration data if not rechecking
        merged_results.update({
            'reconfiguration_segments': original_results.get('reconfiguration_segments', []),
            'num_reconfigurations': original_results.get('num_reconfigurations', 0),
            'reconfiguration_rate': original_results.get('reconfiguration_rate', 0.0),
            'reconfigurations_per_joint': original_results.get('reconfigurations_per_joint', []),
            'max_changes_per_joint': original_results.get('max_changes_per_joint', []),
            'mean_changes_per_joint': original_results.get('mean_changes_per_joint', []),
            'max_changes_per_segment': original_results.get('max_changes_per_segment', []),
            'reconfig_threshold': original_results.get('reconfig_threshold', 0.0),
            'excluded_last_joint': original_results.get('excluded_last_joint', False)
        })

    return merged_results


def attempt_motion_replan(
    trajectory: np.ndarray,
    checker: COALCollisionChecker,
    initial_results: dict,
) -> dict:
    """Attempt to repair colliding segments using CuRobo MotionGen."""
    summary = {
        'attempted': False,
        'success': False,
        'message': None,
        'segments_requested': [],
        'segments_replanned': [],
        'segments_failed': [],
        'trajectory': None,
        'collision_results': None,
    }

    if not CUROBO_AVAILABLE:
        summary['message'] = "CuRobo dependencies are not available."
        return summary

    if len(trajectory) < 2:
        summary['message'] = "Trajectory must contain at least two waypoints for replanning."
        return summary

    num_segments = len(trajectory) - 1
    segment_info = determine_segments_to_replan(num_segments, initial_results)
    segments_to_replan = segment_info['all']  # collision + reconfiguration union

    if not segments_to_replan:
        summary['message'] = "No specific segments identified for replanning."
        return summary

    # Print segment breakdown
    print(f"  Segments requiring replanning:")
    print(f"    - Collision segments: {len(segment_info['collision'])}")
    print(f"    - Reconfiguration segments: {len(segment_info['reconfiguration'])}")
    print(f"    - Total (union): {len(segments_to_replan)}")

    world_cfg = build_motion_world_config(
        checker.glass_position,
        checker.table_position,
        checker.table_dimensions,
        checker.wall_position,
        checker.wall_dimensions,
        checker.workbench_position,
        checker.workbench_dimensions,
        checker.robot_mount_position,
        checker.robot_mount_dimensions,
        [config.DEFAULT_MESH_FILE],
    )

    planner = CuRoboMotionPlanner(
        config.DEFAULT_ROBOT_CONFIG_YAML,
        world_cfg,
        trajopt_tsteps=config.REPLAN_TRAJOPT_TSTEPS,
        interpolation_dt=config.REPLAN_INTERP_DT,
        interpolation_steps=config.REPLAN_INTERP_STEPS,
    )

    summary['attempted'] = True
    summary['segments_requested'] = segments_to_replan
    summary['collision_segments_requested'] = segment_info['collision']
    summary['reconfiguration_segments_requested'] = segment_info['reconfiguration']

    if not planner.ready:
        summary['message'] = planner.init_error or "Failed to initialize MotionGen planner."
        return summary

    # Batch planning: collect all segments to replan
    print(f"  Planning {len(segments_to_replan)} segments in batch...")

    batch_start_configs = []
    batch_goal_configs = []
    batch_segment_indices = []

    for seg_idx in segments_to_replan:
        start_config = np.array(trajectory[seg_idx], dtype=np.float64)
        goal_config = np.array(trajectory[seg_idx + 1], dtype=np.float64)
        batch_start_configs.append(start_config)
        batch_goal_configs.append(goal_config)
        batch_segment_indices.append(seg_idx)

    # Plan all segments in batch
    batch_start_time = time.time()
    batch_results = planner.plan_segments_batch(
        batch_start_configs,
        batch_goal_configs,
        timeout=config.REPLAN_TIMEOUT,
        max_attempts=config.REPLAN_MAX_ATTEMPTS,
    )
    batch_end_time = time.time()

    print(f"  Batch planning completed in {batch_end_time - batch_start_time:.2f}s")

    # Create mapping from segment index to planning result
    segment_to_plan = {}
    for seg_idx, plan_res in zip(batch_segment_indices, batch_results):
        segment_to_plan[seg_idx] = plan_res

    # Build new trajectory by stitching segments together
    new_points: List[np.ndarray] = [np.array(trajectory[0], dtype=np.float64)]
    successful_segments = []
    failed_segments = []
    segment_ranges = {}  # Maps original seg_idx -> (new_start_idx, new_end_idx) in new trajectory

    for seg_idx in range(num_segments):
        start_idx = len(new_points) - 1  # Current position in new trajectory
        goal = np.array(trajectory[seg_idx + 1], dtype=np.float64)

        if seg_idx in segment_to_plan:
            # This segment was replanned
            plan_res = segment_to_plan[seg_idx]

            if plan_res['success'] and plan_res['path'] is not None:
                replanned_path = plan_res['path']
                # Adjust start point to match current trajectory endpoint
                # Use replanned path from second point onwards
                for waypoint in replanned_path[1:]:
                    new_points.append(np.array(waypoint, dtype=np.float64))
                successful_segments.append(seg_idx)
                end_idx = len(new_points) - 1
                segment_ranges[seg_idx] = (start_idx, end_idx)
                print(f"    Segment {seg_idx}: SUCCESS ({len(replanned_path)} waypoints)")
                continue
            else:
                failed_segments.append({
                    'segment': seg_idx,
                    'status': plan_res.get('status', 'plan_failed')
                })
                print(f"    Segment {seg_idx}: FAILED ({plan_res.get('status', 'unknown')})")

        # Use original trajectory for this segment
        new_points.append(goal)
        end_idx = len(new_points) - 1
        segment_ranges[seg_idx] = (start_idx, end_idx)

    new_traj = np.vstack(new_points)

    print(f"  Replanning summary: {len(successful_segments)}/{len(segments_to_replan)} segments successful")

    # Only recheck replanned segments instead of entire trajectory
    print(f"  Rechecking {len(successful_segments)} replanned segments (optimized)...")

    # Build list of new trajectory segment indices to recheck
    new_segments_to_check = []
    for orig_seg_idx in successful_segments:
        start_idx, end_idx = segment_ranges[orig_seg_idx]
        # Add all segment indices in this range
        for i in range(start_idx, end_idx):
            new_segments_to_check.append(i)

    if new_segments_to_check:
        # Recheck only the replanned segments for collisions
        segment_recheck_results = checker.check_trajectory_segments(
            new_traj,
            segment_indices=new_segments_to_check,
            verbose=False,
            show_link_collisions=config.COLLISION_SHOW_LINK_DETAILS,
            interpolate=True,
            num_interp_steps=config.COLLISION_INTERP_STEPS,
            adaptive_interp=config.COLLISION_ADAPTIVE_INTERP,
            adaptive_max_joint_step_deg=config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG,
            adaptive_min_steps=config.COLLISION_ADAPTIVE_MIN_STEPS,
            adaptive_max_steps=config.COLLISION_ADAPTIVE_MAX_STEPS
        )

        # Also recheck joint reconfigurations for the full new trajectory
        if initial_results.get('reconfig_threshold', 0) > 0:
            reconfig_recheck_results = checker.detect_joint_reconfigurations(
                new_traj,
                threshold=initial_results.get('reconfig_threshold', 1.0),
                exclude_last_joint=True
            )
            segment_recheck_results.update(reconfig_recheck_results)

        # Merge with original results
        recheck_results = merge_collision_results(
            original_results=initial_results,
            segment_results=segment_recheck_results,
            replanned_segments=segments_to_replan,
            new_num_waypoints=len(new_traj)
        )

        print(f"  Checked {segment_recheck_results['configs_checked']} configurations "
              f"(vs {initial_results['total_configs_checked']} for full trajectory)")
    else:
        # No successful replanning, use original results
        recheck_results = initial_results.copy()
        recheck_results['total_waypoints'] = len(new_traj)

    summary.update({
        'trajectory': new_traj,
        'collision_results': recheck_results,
        'segments_replanned': successful_segments,
        'segments_failed': failed_segments,
    })

    # Check if both collisions and reconfigurations are resolved
    collisions_resolved = recheck_results['total_collisions'] == 0
    reconfigs_resolved = recheck_results.get('num_reconfigurations', 0) == 0

    if collisions_resolved and reconfigs_resolved:
        summary['success'] = True
        summary['message'] = "All collisions and reconfigurations resolved via replanning."
    elif collisions_resolved:
        summary['success'] = False
        summary['message'] = f"Collisions resolved, but {recheck_results.get('num_reconfigurations', 0)} reconfigurations remain."
    elif reconfigs_resolved:
        summary['success'] = False
        summary['message'] = f"Reconfigurations resolved, but {recheck_results['total_collisions']} collisions remain."
    else:
        summary['success'] = False
        summary['message'] = f"Replanning completed but {recheck_results['total_collisions']} collisions and {recheck_results.get('num_reconfigurations', 0)} reconfigurations remain."

    return summary


def print_collision_summary(results: dict, title: str):
    """Pretty-print collision statistics."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"Total waypoints:        {results['total_waypoints']}")

    if results['interpolate']:
        total_interp = results.get('total_interpolated_configs', 0)
        interp_label = f"default {results['num_interp_steps']} steps"
        if results.get('adaptive_interp'):
            interp_label += ", adaptive"
        print(f"Total configurations:   {results['total_configs_checked']:,} "
              f"(interpolated: {total_interp:,}, {interp_label})")
        print(f"Collisions at waypoints:    {results['num_collisions']}")
        print(f"Collisions in segments:     {results['num_segment_collisions']}")
        print(f"Total collisions:           {results['total_collisions']}")
        print(f"Collision-free configs:     "
              f"{results['total_configs_checked'] - results['total_collisions']:,}")
    else:
        print(f"Collision-free:         {results['num_collision_free']}")
        print(f"Collisions detected:    {results['num_collisions']}")

    print(f"Collision rate:         {results['collision_rate']:.2f}%")

    # Print reconfiguration statistics if available
    if results.get('reconfig_threshold', 0) > 0:
        print(f"\nJoint Reconfigurations:")
        print(f"  Threshold:            {results.get('reconfig_threshold', 0):.2f} rad")
        print(f"  Excluded last joint:  {results.get('excluded_last_joint', False)}")
        print(f"  Total reconfigurations: {results.get('num_reconfigurations', 0)}")
        print(f"  Reconfiguration rate:   {results.get('reconfiguration_rate', 0):.1%}")
        if results.get('num_reconfigurations', 0) > 0:
            reconfig_segs = results.get('reconfiguration_segments', [])
            print(f"  Reconfiguration segments (first 50): {reconfig_segs[:50]}")

    if results['num_collisions'] > 0:
        print(f"\nCollision waypoint indices (first 50):")
        print(f"  {results['collision_indices'][:50]}")

    if results['interpolate'] and results['collision_segments']:
        unique_segments = sorted({wp_idx for wp_idx, _ in results['collision_segments']})
        print(f"\nCollision segments (first 50 waypoint pairs):")
        for wp_idx in unique_segments[:50]:
            print(f"  Waypoint {wp_idx}→{wp_idx+1}")

    if config.COLLISION_SHOW_LINK_DETAILS and results.get('link_collisions'):
        print(f"\nCollisions by link:")
        sorted_links = sorted(results['link_collisions'].items(), key=lambda x: x[1], reverse=True)
        total_link_collisions = results['total_collisions'] if results['interpolate'] else results['num_collisions']
        for link_name, count in sorted_links:
            percentage = (count / total_link_collisions) * 100 if total_link_collisions > 0 else 0
            print(f"  {link_name}: {count} ({percentage:.1f}% of collisions)")


def main():
    parser = argparse.ArgumentParser(
        description='Check trajectory collisions using COAL (all config from config.py)'
    )
    parser.add_argument(
        '--trajectory',
        type=str,
        required=True,
        help='Path to trajectory CSV file'
    )

    args = parser.parse_args()
    script_start_time = time.perf_counter()

    print("=" * 70)
    print("COAL Collision Checker for Robot Trajectories")
    print("=" * 70)
    print("NOTE: Using COAL library (improved FCL) for better performance")
    print("Execute with: omni_python coal_check.py --trajectory <path>")
    print("=" * 70)

    # Load trajectory
    print(f"\n1. Loading trajectory from: {args.trajectory}")
    trajectory, joint_names = load_trajectory_csv(args.trajectory)

    # Initialize collision checker
    print(f"\n2. Initializing COAL collision checker")
    print(f"   Robot URDF: {config.DEFAULT_ROBOT_URDF}")
    print(f"   Robot config: {config.DEFAULT_ROBOT_CONFIG_YAML}")
    print(f"   Obstacle meshes: {[config.DEFAULT_MESH_FILE]}")
    print(f"   Glass position: {config.GLASS_POSITION}")
    print(f"   Table position: {config.TABLE_POSITION}")
    print(f"   Table dimensions: {config.TABLE_DIMENSIONS}")
    print(f"   Use link meshes: {config.COLLISION_USE_LINK_MESHES}")
    print(f"   Collision margin: {config.COLLISION_MARGIN}")

    checker = COALCollisionChecker(
        robot_urdf_path=config.DEFAULT_ROBOT_URDF,
        obstacle_mesh_paths=[config.DEFAULT_MESH_FILE],
        glass_position=config.GLASS_POSITION.copy(),
        table_position=config.TABLE_POSITION.copy(),
        table_dimensions=config.TABLE_DIMENSIONS.copy(),
        wall_position=config.WALL_POSITION.copy(),
        wall_dimensions=config.WALL_DIMENSIONS.copy(),
        workbench_position=config.WORKBENCH_POSITION.copy(),
        workbench_dimensions=config.WORKBENCH_DIMENSIONS.copy(),
        robot_mount_position=config.ROBOT_MOUNT_POSITION.copy(),
        robot_mount_dimensions=config.ROBOT_MOUNT_DIMENSIONS.copy(),
        robot_config_path=config.DEFAULT_ROBOT_CONFIG_YAML,
        use_link_meshes=config.COLLISION_USE_LINK_MESHES,
        mesh_base_path=config.MESH_BASE_PATH,
        collision_margin=config.COLLISION_MARGIN,
        use_curobo_interpolation=config.COLLISION_USE_CUROBO_INTERP,
    )

    # Check trajectory
    print(f"\n3. Running collision detection with COAL")
    if config.COLLISION_SHOW_LINK_DETAILS:
        print(f"   Link collision analysis enabled (showing first 10)")
    print(f"   Interpolation enabled: {config.COLLISION_INTERP_STEPS} steps between waypoints")
    interp_mode = "CuRobo GPU" if config.COLLISION_USE_CUROBO_INTERP else "CPU linear"
    print(f"   Interpolation method: {interp_mode}")
    if config.COLLISION_ADAPTIVE_INTERP:
        print(f"   Adaptive interpolation enabled (max joint delta {config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG} deg)")
    print(f"   Joint reconfiguration checking enabled (threshold: {config.RECONFIGURATION_THRESHOLD} rad)")
    if config.COLLISION_PARALLEL:
        print(f"   Parallel processing enabled")

    results = checker.check_trajectory(
        trajectory,
        verbose=config.COLLISION_VERBOSE,
        show_link_collisions=config.COLLISION_SHOW_LINK_DETAILS,
        interpolate=True,
        num_interp_steps=config.COLLISION_INTERP_STEPS,
        check_reconfig=True,
        reconfig_threshold=config.RECONFIGURATION_THRESHOLD,
        parallel=config.COLLISION_PARALLEL,
        num_workers=config.COLLISION_NUM_WORKERS,
        adaptive_interp=config.COLLISION_ADAPTIVE_INTERP,
        adaptive_max_joint_step_deg=config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG,
        adaptive_min_steps=config.COLLISION_ADAPTIVE_MIN_STEPS,
        adaptive_max_steps=config.COLLISION_ADAPTIVE_MAX_STEPS,
    )

    print_collision_summary(results, "COLLISION CHECK RESULTS")

    final_results = results
    replan_summary: Dict = {'attempted': False}
    collision_free_csv_path = None
    timing_info: Dict[str, float] = {
        'collision_check_sec': results.get('collision_check_time_sec', 0.0),
        'reconfig_check_sec': results.get('reconfig_check_time_sec', 0.0),
        'replan_sec': 0.0,
        'total_runtime_sec': 0.0,
    }

    # Attempt replanning if there are collisions OR reconfigurations
    needs_replan = results['total_collisions'] > 0 or results.get('num_reconfigurations', 0) > 0

    if config.REPLAN_ENABLED and needs_replan:
        print("\n4. Attempting CuRobo replanning for problematic segments...")
        replan_start_time = time.perf_counter()
        replan_summary = attempt_motion_replan(trajectory, checker, results)
        timing_info['replan_sec'] = time.perf_counter() - replan_start_time
        replan_summary['replan_time_sec'] = timing_info['replan_sec']

        # if replan_summary.get('success'):
        final_results = replan_summary['collision_results']
        trajectory = replan_summary['trajectory']
        output_path = Path(args.trajectory).parent / 'collision_free_trajectory.csv'
        collision_free_csv_path = save_trajectory_csv(trajectory, joint_names, output_path)
        replan_summary['output_path'] = str(collision_free_csv_path)
        print("\n✓ Collision-free trajectory generated via CuRobo.")
        print(f"Saved to: {collision_free_csv_path}")
        print_collision_summary(final_results, "REPLANNED COLLISION CHECK RESULTS")
        # else:
        #     print("\nCuRobo replanning unsuccessful.")
        #     if replan_summary.get('message'):
        #         print(f"Reason: {replan_summary['message']}")
    elif config.REPLAN_ENABLED:
        print("\nNo collisions detected — skipping CuRobo replanning.")
        replan_summary['replan_time_sec'] = timing_info['replan_sec']

    timing_info['total_runtime_sec'] = time.perf_counter() - script_start_time

    report_path = save_collision_report(
        args.trajectory,
        final_results,
        replan_summary if replan_summary.get('attempted') else None,
        timing_info
    )
    print(f"\nCollision report saved to: {report_path}")

    if collision_free_csv_path:
        print(f"- Collision-free trajectory saved to {collision_free_csv_path}")


if __name__ == "__main__":
    main()
