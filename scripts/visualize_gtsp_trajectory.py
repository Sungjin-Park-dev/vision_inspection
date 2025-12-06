#!/usr/bin/env python3
"""
Visualize GTSP Trajectory with Glass Mesh

This script visualizes:
1. Glass mesh object
2. Target poses from GTSP CSV trajectory
3. Optional: trajectory path connecting waypoints

Usage:
    python scripts/visualize_gtsp_trajectory.py \
        --csv data/trajectory/2/gtsp.csv \
        --mesh data/object/glass.obj \
        --show_frames  # Optional: show coordinate frames at each pose
        --frame_size 0.02  # Frame size in meters
"""

import argparse
import csv
import os
import sys
from typing import List, Tuple

import numpy as np
import open3d as o3d

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.cli_utils import print_section_header, print_key_value
from common import config


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        3x3 rotation matrix
    """
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def load_trajectory_poses(csv_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load target poses from GTSP CSV file

    Returns:
        positions: List of (x, y, z) positions
        rotations: List of 3x3 rotation matrices
    """
    print_section_header("LOADING TRAJECTORY", width=60)
    print_key_value("CSV file", csv_path)

    positions = []
    rotations = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Position
            pos = np.array([
                float(row['target-POS_X']),
                float(row['target-POS_Y']),
                float(row['target-POS_Z'])
            ])
            positions.append(pos)

            # Rotation (quaternion: X, Y, Z, W in CSV)
            quat = np.array([
                float(row['target-ROT_X']),
                float(row['target-ROT_Y']),
                float(row['target-ROT_Z']),
                float(row['target-ROT_W'])
            ])
            rot_mat = quaternion_to_rotation_matrix(quat)
            rotations.append(rot_mat)

    print_key_value("Loaded waypoints", len(positions))
    print()

    return positions, rotations


def load_mesh(mesh_path: str, apply_world_transform: bool = True) -> o3d.geometry.TriangleMesh:
    """Load mesh file and apply world transformation

    Args:
        mesh_path: Path to mesh file
        apply_world_transform: If True, apply GLASS_POSITION and GLASS_ROTATION from config

    Returns:
        Transformed mesh
    """
    print_section_header("LOADING MESH", width=60)
    print_key_value("Mesh file", mesh_path)

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print_key_value("Vertices", len(mesh.vertices))
    print_key_value("Triangles", len(mesh.triangles))

    if apply_world_transform:
        # Apply glass position and rotation from config
        glass_pos = config.GLASS_POSITION
        glass_quat = config.GLASS_ROTATION  # (w, x, y, z)

        # Convert quaternion to rotation matrix
        rot_mat = config.quaternion_to_rotation_matrix(glass_quat)

        print()
        print("Applying world transformation:")
        print_key_value("Position", f"[{glass_pos[0]:.3f}, {glass_pos[1]:.3f}, {glass_pos[2]:.3f}]")
        print_key_value("Rotation (quat)", f"[{glass_quat[0]:.3f}, {glass_quat[1]:.3f}, {glass_quat[2]:.3f}, {glass_quat[3]:.3f}]")

        # Apply transformation: first rotate, then translate
        mesh.rotate(rot_mat, center=[0, 0, 0])
        mesh.translate(glass_pos)

    print()

    return mesh


def create_coordinate_frame(position: np.ndarray, rotation: np.ndarray, size: float = 0.02):
    """Create coordinate frame geometry

    Args:
        position: (3,) position
        rotation: (3, 3) rotation matrix
        size: Frame size in meters
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # Apply rotation and translation
    frame.rotate(rotation, center=[0, 0, 0])
    frame.translate(position)

    return frame


def create_trajectory_path(positions: List[np.ndarray], color=[1, 0, 0]):
    """Create line set connecting trajectory waypoints

    Args:
        positions: List of waypoint positions
        color: RGB color for lines
    """
    if len(positions) < 2:
        return None

    points = np.array(positions)
    lines = [[i, i+1] for i in range(len(positions) - 1)]
    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_viewing_direction_arrows(
    positions: List[np.ndarray],
    rotations: List[np.ndarray],
    arrow_length: float = 0.05,
    color=[1, 0, 0]
):
    """Create arrows showing viewing directions (camera Z-axis)

    Args:
        positions: List of camera positions
        rotations: List of rotation matrices
        arrow_length: Length of arrows in meters
        color: RGB color for arrows
    """
    all_points = []
    all_lines = []
    all_colors = []

    point_idx = 0
    for pos, rot in zip(positions, rotations):
        # Camera viewing direction is typically the Z-axis of the rotation matrix
        # rot[:, 2] gives the Z-axis direction
        view_dir = rot[:, 2]

        # Create arrow from position along viewing direction
        start = pos
        end = pos + view_dir * arrow_length

        all_points.extend([start, end])
        all_lines.append([point_idx, point_idx + 1])
        all_colors.append(color)

        point_idx += 2

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(all_lines)
    line_set.colors = o3d.utility.Vector3dVector(all_colors)

    return line_set


def visualize_trajectory(
    mesh: o3d.geometry.TriangleMesh,
    positions: List[np.ndarray],
    rotations: List[np.ndarray],
    show_frames: bool = False,
    frame_size: float = 0.02,
    show_path: bool = True,
    show_normals: bool = True,
    normal_length: float = 0.05
):
    """Visualize mesh with trajectory poses"""
    print_section_header("VISUALIZATION", width=60)

    geometries = []

    # Add mesh (semi-transparent gray)
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(mesh_vis)

    # Add coordinate frame at origin
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,
        origin=[0, 0, 0]
    )
    geometries.append(origin_frame)

    # Add target poses
    if show_frames:
        print(f"Adding {len(positions)} coordinate frames...")
        for pos, rot in zip(positions, rotations):
            frame = create_coordinate_frame(pos, rot, size=frame_size)
            geometries.append(frame)
    else:
        # Show as points
        print(f"Adding {len(positions)} waypoint markers...")
        points = np.array(positions)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 1, 0])  # Green points
        geometries.append(pcd)

    # Add viewing direction arrows
    if show_normals:
        print(f"Adding viewing direction arrows...")
        arrows = create_viewing_direction_arrows(
            positions, rotations,
            arrow_length=normal_length,
            color=[1, 0, 0]
        )
        geometries.append(arrows)

    # Add trajectory path
    if show_path and len(positions) > 1:
        print("Adding trajectory path...")
        path = create_trajectory_path(positions, color=[0, 0, 1])  # Blue for path to distinguish from arrows
        if path:
            geometries.append(path)

    # Print legend
    print()
    print("Legend:")
    print("  Gray mesh: Glass object")
    print("  Origin frame (large): World coordinate system")
    if show_frames:
        print(f"  Small frames: Target poses (X=red, Y=green, Z=blue)")
    else:
        print("  Green points: Target positions")
    if show_normals:
        print("  Red arrows: Camera viewing directions")
    if show_path:
        print("  Blue lines: Trajectory path")
    print()

    # Visualize
    print(f"Opening visualization window...")
    print("  → Use mouse to rotate/zoom")
    print("  → Press 'Q' or close window to exit")
    print()

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GTSP Trajectory Visualization",
        width=1280,
        height=720
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GTSP trajectory with glass mesh"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to GTSP trajectory CSV file"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="data/object/glass.obj",
        help="Path to glass mesh file (default: data/object/glass.obj)"
    )
    parser.add_argument(
        "--show_frames",
        action="store_true",
        help="Show coordinate frames at each pose (default: show points)"
    )
    parser.add_argument(
        "--frame_size",
        type=float,
        default=0.02,
        help="Size of coordinate frames in meters (default: 0.02)"
    )
    parser.add_argument(
        "--no_path",
        action="store_true",
        help="Don't show trajectory path lines"
    )
    parser.add_argument(
        "--no_transform",
        action="store_true",
        help="Don't apply world transformation to mesh (show at origin)"
    )
    parser.add_argument(
        "--no_normals",
        action="store_true",
        help="Don't show viewing direction arrows"
    )
    parser.add_argument(
        "--normal_length",
        type=float,
        default=0.05,
        help="Length of viewing direction arrows in meters (default: 0.05)"
    )

    args = parser.parse_args()

    # Load data
    positions, rotations = load_trajectory_poses(args.csv)
    mesh = load_mesh(args.mesh, apply_world_transform=not args.no_transform)

    # Visualize
    visualize_trajectory(
        mesh=mesh,
        positions=positions,
        rotations=rotations,
        show_frames=args.show_frames,
        frame_size=args.frame_size,
        show_path=not args.no_path,
        show_normals=not args.no_normals,
        normal_length=args.normal_length
    )

    print("Done!")


if __name__ == "__main__":
    main()
