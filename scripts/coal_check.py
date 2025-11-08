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
from typing import List, Tuple, Optional, Dict
import trimesh
from scipy.spatial.transform import Rotation
import pinocchio as pin
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common utilities
from common import config
from common.interpolation_utils import generate_interpolated_path


class COALCollisionChecker:
    """Collision checker using COAL and pinocchio for FK"""

    def __init__(self, robot_urdf_path: str, obstacle_mesh_paths: List[str],
                 glass_position: np.ndarray = None,
                 table_position: np.ndarray = None,
                 table_dimensions: np.ndarray = None,
                 robot_config_path: Optional[str] = None,
                 use_capsules: bool = False, capsule_radius: float = 0.05,
                 use_link_meshes: bool = False,
                 mesh_base_path: str = None,
                 collision_margin: float = None):
        """
        Initialize collision checker

        Args:
            robot_urdf_path: Path to robot URDF file
            obstacle_mesh_paths: List of paths to obstacle mesh files
            glass_position: Position of glass object in world frame (x, y, z)
            table_position: Position of table cuboid in world frame (x, y, z)
            table_dimensions: Dimensions of table cuboid (x, y, z) in meters
            robot_config_path: Path to CuRobo robot config YAML (e.g., ur20.yml)
            use_capsules: If True, use capsule approximations instead of spheres
            capsule_radius: Radius for capsule collision geometry (meters)
            use_link_meshes: If True, use actual URDF collision meshes for robot links
            mesh_base_path: Base path for robot mesh files
            collision_margin: Safety margin for collision detection (meters).
                            Positive = more conservative, Negative = less conservative
        """
        # Apply config defaults
        if glass_position is None:
            glass_position = config.GLASS_POSITION.copy()
        if table_position is None:
            table_position = config.TABLE_POSITION.copy()
        if table_dimensions is None:
            table_dimensions = config.TABLE_DIMENSIONS.copy()
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
        self.robot_config_path = robot_config_path
        self.use_capsules = use_capsules
        self.capsule_radius = capsule_radius
        self.use_link_meshes = use_link_meshes
        self.mesh_base_path = mesh_base_path
        self.collision_margin = collision_margin
        self.collision_spheres = {}
        self.link_meshes = {}

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
        elif robot_config_path and not use_capsules:
            # Load collision spheres from CuRobo config
            self._load_collision_spheres_from_yaml(robot_config_path)
            print(f"  Using collision spheres from: {robot_config_path}")
            total_spheres = sum(len(spheres) for spheres in self.collision_spheres.values())
            print(f"  Total collision spheres: {total_spheres}")
        else:
            # Use capsule approximations
            self.link_capsules = self._define_robot_capsules()
            print(f"  Using {len(self.link_capsules)} capsule collision geometries")

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

    def _define_robot_capsules(self) -> List[dict]:
        """
        Define capsule approximations for robot links

        Returns:
            List of dictionaries with link collision info
        """
        # Approximate UR20 link lengths (you may need to adjust these)
        capsules = [
            {'link_name': 'shoulder_link', 'length': 0.15, 'offset': np.array([0, 0, 0.075])},
            {'link_name': 'upper_arm_link', 'length': 0.80, 'offset': np.array([0, 0, 0.40])},
            {'link_name': 'forearm_link', 'length': 0.80, 'offset': np.array([0, 0, 0.40])},
            {'link_name': 'wrist_1_link', 'length': 0.15, 'offset': np.array([0, 0, 0.075])},
            {'link_name': 'wrist_2_link', 'length': 0.15, 'offset': np.array([0, 0, 0.075])},
            {'link_name': 'wrist_3_link', 'length': 0.10, 'offset': np.array([0, 0, 0.05])},
        ]
        return capsules

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
        # Otherwise use capsules
        else:
            return self._create_capsule_collision_objects()

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

    def _create_capsule_collision_objects(self) -> List[coal.CollisionObject]:
        """Create COAL collision objects using capsule approximations"""
        robot_collision_objects = []

        for capsule_def in self.link_capsules:
            link_name = capsule_def['link_name']
            length = capsule_def['length']
            offset = capsule_def['offset']

            # Find the joint/frame index
            try:
                frame_id = self.robot_model.getFrameId(link_name)
                transform_matrix = self.robot_data.oMf[frame_id]
            except:
                # If frame not found, try with joint name
                try:
                    joint_id = self.robot_model.getJointId(link_name)
                    transform_matrix = self.robot_data.oMi[joint_id]
                except:
                    continue

            # Extract position and rotation from pinocchio transform
            position = transform_matrix.translation + transform_matrix.rotation @ offset
            rotation = transform_matrix.rotation

            # Create COAL transform
            coal_transform = self._create_transform(rotation, position)

            # Create capsule collision object
            capsule = coal.Capsule(self.capsule_radius, length)
            col_obj = coal.CollisionObject(capsule, coal_transform)
            robot_collision_objects.append(col_obj)

        return robot_collision_objects

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
            else:
                # Capsules
                link_name = link_names[idx] if idx < len(link_names) else f"link_{idx}"

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
        num_interp_steps: int = 10
    ) -> dict:
        """
        Check collisions along entire trajectory

        Args:
            trajectory: (N, 6) array of joint configurations
            verbose: Print progress
            show_link_collisions: Show which links are colliding
            max_show: Maximum number of collision details to show
            interpolate: If True, check intermediate configurations between waypoints
            num_interp_steps: Number of interpolation steps between waypoints

        Returns:
            Dictionary with collision statistics
        """
        from collections import Counter

        num_waypoints = len(trajectory)
        collision_indices = []
        collision_free_indices = []
        collision_segments = []  # List of (waypoint_idx, alpha) tuples for interpolated collisions
        link_collision_counter = Counter()

        if interpolate:
            # Calculate total configurations to check
            total_configs = num_waypoints + (num_waypoints - 1) * num_interp_steps
            print(f"\nChecking {num_waypoints} waypoints with interpolation "
                  f"({num_interp_steps} steps between waypoints)...")
            print(f"Total configurations to check: {total_configs:,}")
        else:
            print(f"\nChecking {num_waypoints} waypoints for collisions (no interpolation)...")

        shown_count = 0
        configs_checked = 0

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
                start_config = trajectory[i]
                end_config = trajectory[i + 1]

                # Generate interpolated path
                interpolated_configs = generate_interpolated_path(
                    start_config, end_config, num_interp_steps
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
                        alpha = (interp_idx + 1) / (num_interp_steps + 1)
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
                    total_to_check = num_waypoints + (num_waypoints - 1) * num_interp_steps
                    print(f"  Progress: {configs_checked}/{total_to_check} configurations checked")
            elif verbose and not interpolate:
                if (i + 1) % 100 == 0:
                    print(f"  Progress: {i+1}/{num_waypoints} waypoints checked")

        # Calculate statistics
        num_collisions = len(collision_indices)
        num_segment_collisions = len(collision_segments)
        total_collisions = num_collisions + num_segment_collisions

        if interpolate:
            total_configs = num_waypoints + (num_waypoints - 1) * num_interp_steps
            collision_rate = total_collisions / total_configs * 100
        else:
            total_configs = num_waypoints
            collision_rate = num_collisions / num_waypoints * 100

        results = {
            'total_waypoints': num_waypoints,
            'total_configs_checked': configs_checked,
            'interpolate': interpolate,
            'num_interp_steps': num_interp_steps if interpolate else 0,
            'num_collisions': num_collisions,
            'num_segment_collisions': num_segment_collisions if interpolate else 0,
            'total_collisions': total_collisions,
            'num_collision_free': len(collision_free_indices),
            'collision_rate': collision_rate,
            'collision_indices': collision_indices,
            'collision_segments': collision_segments if interpolate else [],
            'collision_free_indices': collision_free_indices,
            'link_collisions': dict(link_collision_counter) if show_link_collisions else {}
        }

        return results


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


def main():
    parser = argparse.ArgumentParser(description='Check trajectory collisions using COAL')
    parser.add_argument(
        '--trajectory',
        type=str,
        default='data/trajectory/joint_trajectory_dp_5000_base.csv',
        help='Path to trajectory CSV file'
    )
    parser.add_argument(
        '--robot_urdf',
        type=str,
        default=config.DEFAULT_ROBOT_URDF,
        help='Path to robot URDF file'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        nargs='+',
        default=[config.DEFAULT_MESH_FILE],
        help='Paths to obstacle mesh files (Z-up coordinates)'
    )
    parser.add_argument(
        '--robot_config',
        type=str,
        default=config.DEFAULT_ROBOT_CONFIG_YAML,
        help='Path to CuRobo robot config YAML file (for collision spheres)'
    )
    parser.add_argument(
        '--glass_position',
        type=float,
        nargs=3,
        default=config.GLASS_POSITION.tolist(),
        help='Glass object position in world frame (x y z)'
    )
    parser.add_argument(
        '--table_position',
        type=float,
        nargs=3,
        default=config.TABLE_POSITION.tolist(),
        help='Table cuboid position in world frame (x y z)'
    )
    parser.add_argument(
        '--table_dimensions',
        type=float,
        nargs=3,
        default=config.TABLE_DIMENSIONS.tolist(),
        help='Table cuboid dimensions (x y z) in meters'
    )
    parser.add_argument(
        '--use_link_meshes',
        action='store_true',
        help='Use actual collision meshes from URDF (most accurate)'
    )
    parser.add_argument(
        '--use_capsules',
        action='store_true',
        help='Use capsule approximations instead of spheres'
    )
    parser.add_argument(
        '--mesh_base_path',
        type=str,
        default=config.MESH_BASE_PATH,
        help='Base path for robot mesh files'
    )
    parser.add_argument(
        '--collision_margin',
        type=float,
        default=config.COLLISION_MARGIN,
        help='Safety margin for collision detection in meters (e.g., -0.05 for 5cm tolerance)'
    )
    parser.add_argument(
        '--show_link_collisions',
        action='store_true',
        help='Show which links are colliding (detailed analysis)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress'
    )
    parser.add_argument(
        '--no-interpolate',
        action='store_true',
        help='Disable interpolation between waypoints (only check discrete waypoints)'
    )
    parser.add_argument(
        '--interp-steps',
        type=int,
        default=config.COLLISION_INTERP_STEPS,
        help=f'Number of interpolation steps between waypoints (default: {config.COLLISION_INTERP_STEPS})'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("COAL Collision Checker for Robot Trajectories")
    print("=" * 70)
    print("NOTE: Using COAL library (improved FCL) for better performance")
    print("Execute with: omni_python coal_check.py [options]")
    print("=" * 70)

    # Load trajectory
    print(f"\n1. Loading trajectory from: {args.trajectory}")
    trajectory, _ = load_trajectory_csv(args.trajectory)

    # Initialize collision checker
    print(f"\n2. Initializing COAL collision checker")
    print(f"   Robot URDF: {args.robot_urdf}")
    print(f"   Robot config: {args.robot_config}")
    print(f"   Obstacle meshes: {args.mesh}")
    print(f"   Glass position: {args.glass_position}")
    print(f"   Table position: {args.table_position}")
    print(f"   Table dimensions: {args.table_dimensions}")
    print(f"   Use link meshes: {args.use_link_meshes}")
    print(f"   Use capsules: {args.use_capsules}")

    checker = COALCollisionChecker(
        robot_urdf_path=args.robot_urdf,
        obstacle_mesh_paths=args.mesh,
        glass_position=np.array(args.glass_position),
        table_position=np.array(args.table_position),
        table_dimensions=np.array(args.table_dimensions),
        robot_config_path=args.robot_config,
        use_capsules=args.use_capsules,
        use_link_meshes=args.use_link_meshes,
        mesh_base_path=args.mesh_base_path,
        collision_margin=args.collision_margin
    )

    # Check trajectory
    print(f"\n3. Running collision detection with COAL")
    if args.show_link_collisions:
        print(f"   Link collision analysis enabled (showing first 10)")
    if not args.no_interpolate:
        print(f"   Interpolation enabled: {args.interp_steps} steps between waypoints")
    else:
        print(f"   Interpolation disabled: checking waypoints only")

    results = checker.check_trajectory(
        trajectory,
        verbose=args.verbose,
        show_link_collisions=args.show_link_collisions,
        interpolate=not args.no_interpolate,
        num_interp_steps=args.interp_steps
    )

    # Print results
    print("\n" + "=" * 70)
    print("COLLISION CHECK RESULTS")
    print("=" * 70)
    print(f"Total waypoints:        {results['total_waypoints']}")

    if results['interpolate']:
        print(f"Total configurations:   {results['total_configs_checked']:,} "
              f"(with {results['num_interp_steps']} interpolation steps)")
        print(f"Collisions at waypoints:    {results['num_collisions']}")
        print(f"Collisions in segments:     {results['num_segment_collisions']}")
        print(f"Total collisions:           {results['total_collisions']}")
        print(f"Collision-free configs:     {results['total_configs_checked'] - results['total_collisions']:,}")
    else:
        print(f"Collision-free:         {results['num_collision_free']}")
        print(f"Collisions detected:    {results['num_collisions']}")

    print(f"Collision rate:         {results['collision_rate']:.2f}%")

    if results['num_collisions'] > 0:
        print(f"\nCollision waypoint indices (first 50):")
        print(f"  {results['collision_indices'][:50]}")

    # Print segment collisions if interpolation was used
    if results['interpolate'] and results['num_segment_collisions'] > 0:
        print(f"\nCollision segments (first 50):")
        for wp_idx, alpha in results['collision_segments'][:50]:
            print(f"  Waypoint {wp_idx}→{wp_idx+1} (α={alpha:.2f})")

    # Print link collision statistics
    if args.show_link_collisions and results.get('link_collisions'):
        print(f"\nCollisions by link:")
        sorted_links = sorted(results['link_collisions'].items(), key=lambda x: x[1], reverse=True)
        total_link_collisions = results['total_collisions'] if results['interpolate'] else results['num_collisions']
        for link_name, count in sorted_links:
            percentage = (count / total_link_collisions) * 100 if total_link_collisions > 0 else 0
            print(f"  {link_name}: {count} ({percentage:.1f}% of collisions)")

    print("\n" + "=" * 70)
    print("\nNOTE:")
    print("- Using COAL (Collision and Occupancy Algorithms Library)")
    print("  COAL is 5-15x faster than FCL with improved numerical stability")
    print("- Using Pinocchio for Forward Kinematics")
    if results['interpolate']:
        print(f"- Interpolation ENABLED: Checking {results['num_interp_steps']} intermediate "
              f"configurations between each waypoint pair")
        print("  This detects collisions that occur during motion between waypoints")
    else:
        print("- Interpolation DISABLED: Only checking discrete waypoints")
        print("  WARNING: May miss collisions that occur between waypoints!")
    print("- Robot collision geometry:")
    if args.use_link_meshes:
        print("  Using actual mesh geometries from URDF (most accurate)")
    elif not args.use_capsules:
        print("  Using collision spheres from CuRobo config")
    else:
        print("  Using capsule approximations")
    print("- To improve accuracy:")
    print("  1. Use --use_link_meshes for most accurate collision checking")
    print("  2. Adjust --interp-steps to control interpolation density (default: 10)")
    print("  3. Use --collision_margin to add safety margins (e.g., 0.01 for 1cm)")
    print("=" * 70)


if __name__ == "__main__":
    main()
