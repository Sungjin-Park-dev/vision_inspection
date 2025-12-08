#!/usr/bin/env python3
"""
Visualization Utilities for Vision Inspection Pipeline

Provides visualization functions for debugging and analysis:
1. Isaac Sim visualization (debug_draw)
2. Open3D trajectory visualization
"""

import csv
from typing import List, Optional, Tuple

import numpy as np

# Import guard for Isaac Sim dependencies (optional)
try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        _debug_draw = None

try:
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver
except ImportError:
    TensorDeviceType = None
    IKSolver = None

# Import guard for Open3D (optional dependency)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None


# ============================================================================
# Isaac Sim Debug Visualization
# ============================================================================

def visualize_target_positions(
    target_positions: List[np.ndarray],
    draw,
    joint_targets: List[np.ndarray] = None,
    ik_solver = None
):
    """Visualize target waypoint positions as green points in Isaac Sim

    Args:
        target_positions: List of target positions (from CSV), or None to compute from FK
        draw: Debug draw interface (omni.isaac.debug_draw._debug_draw)
        joint_targets: List of joint configurations (used if target_positions is None)
        ik_solver: IK solver with kinematics (used if target_positions is None)

    Raises:
        ValueError: If target_positions is None but joint_targets or ik_solver is missing

    Example:
        >>> # In Isaac Sim context
        >>> from omni.isaac.debug_draw import _debug_draw
        >>> draw = _debug_draw.acquire_debug_draw_interface()
        >>> positions = [np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.6])]
        >>> visualize_target_positions(positions, draw)
        # Draws green points at the target positions
    """
    from common.cli_utils import print_section_header, print_key_value

    print_section_header("DEBUG: VISUALIZING TARGET POSITIONS", width=60)

    # If target_positions not provided, compute from FK
    if target_positions is None:
        if joint_targets is None or ik_solver is None:
            print("Error: Cannot visualize - no target positions or FK solver provided")
            return

        if TensorDeviceType is None:
            print("Error: CuRobo not available - cannot compute FK")
            return

        print("Computing target positions from forward kinematics...")
        tensor_args = TensorDeviceType()
        target_positions = []

        for joint_config in joint_targets:
            # Convert to tensor (batch size 1)
            q_tensor = tensor_args.to_device([joint_config])

            # Compute forward kinematics
            ee_pose = ik_solver.fk(q_tensor)
            ee_position = ee_pose.position[0].cpu().numpy()

            target_positions.append(ee_position)
    else:
        print("Using target positions from CSV file...")

    target_positions = np.array(target_positions)

    # Draw green points at target positions
    point_sizes = [10.0] * len(target_positions)
    colors = [(0.0, 1.0, 0.0, 1.0)] * len(target_positions)  # Green with alpha

    draw.draw_points(
        target_positions.tolist(),
        colors,
        point_sizes
    )

    print_key_value("Target waypoints visualized", len(target_positions))
    print_key_value("Color", "Green")
    print_key_value("Point size", 10.0)
    print()


# ============================================================================
# Open3D Trajectory Visualization
# ============================================================================

def quaternion_to_rotation_matrix_o3d(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to rotation matrix for Open3D

    Args:
        q: Quaternion [x, y, z, w]

    Returns:
        3x3 rotation matrix

    Example:
        >>> q = np.array([0.0, 0.0, 0.0, 1.0])  # Identity quaternion
        >>> R = quaternion_to_rotation_matrix_o3d(q)
        >>> np.allclose(R, np.eye(3))
        True
    """
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def load_trajectory_poses_from_csv(csv_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load target poses from GTSP CSV file

    Args:
        csv_path: Path to CSV file with columns:
                  target-POS_X, target-POS_Y, target-POS_Z,
                  target-ROT_X, target-ROT_Y, target-ROT_Z, target-ROT_W

    Returns:
        positions: List of (x, y, z) positions
        rotations: List of 3x3 rotation matrices

    Example:
        >>> positions, rotations = load_trajectory_poses_from_csv('trajectory.csv')
        >>> len(positions) == len(rotations)
        True
    """
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
            rot_mat = quaternion_to_rotation_matrix_o3d(quat)
            rotations.append(rot_mat)

    return positions, rotations


def load_mesh_for_visualization(
    mesh_path: str,
    apply_world_transform: bool = True,
    glass_position: Optional[np.ndarray] = None,
    glass_rotation: Optional[np.ndarray] = None
) -> 'o3d.geometry.TriangleMesh':
    """Load mesh file and optionally apply world transformation

    Args:
        mesh_path: Path to mesh file (OBJ, PLY, STL, etc.)
        apply_world_transform: If True, apply glass position/rotation
        glass_position: Glass position [x, y, z] (default: None = no transform)
        glass_rotation: Glass rotation quaternion [w, x, y, z] (default: None = no transform)

    Returns:
        Transformed Open3D triangle mesh

    Raises:
        ImportError: If Open3D is not available
        FileNotFoundError: If mesh file doesn't exist

    Example:
        >>> mesh = load_mesh_for_visualization('glass.obj', apply_world_transform=False)
        >>> mesh.has_vertex_normals()
        True
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available. Install with: pip install open3d")

    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    if apply_world_transform and glass_position is not None:
        # Apply glass position and rotation
        if glass_rotation is not None:
            # Convert quaternion (w, x, y, z) to rotation matrix
            # Note: config.quaternion_to_rotation_matrix expects (w, x, y, z)
            from common import config
            rot_mat = config.quaternion_to_rotation_matrix(glass_rotation)

            # Apply transformation: first rotate, then translate
            mesh.rotate(rot_mat, center=[0, 0, 0])

        mesh.translate(glass_position)

    return mesh


def create_coordinate_frame_o3d(
    position: np.ndarray,
    rotation: np.ndarray,
    size: float = 0.02
) -> 'o3d.geometry.TriangleMesh':
    """Create coordinate frame geometry for Open3D

    Args:
        position: (3,) position [x, y, z]
        rotation: (3, 3) rotation matrix
        size: Frame size in meters

    Returns:
        Coordinate frame mesh (X=red, Y=green, Z=blue)

    Raises:
        ImportError: If Open3D is not available

    Example:
        >>> pos = np.array([1.0, 0.0, 0.5])
        >>> rot = np.eye(3)
        >>> frame = create_coordinate_frame_o3d(pos, rot, size=0.05)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available. Install with: pip install open3d")

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

    # Apply rotation and translation
    frame.rotate(rotation, center=[0, 0, 0])
    frame.translate(position)

    return frame


def create_trajectory_path_lineset(
    positions: List[np.ndarray],
    color: List[float] = [1, 0, 0]
) -> Optional['o3d.geometry.LineSet']:
    """Create line set connecting trajectory waypoints

    Args:
        positions: List of waypoint positions
        color: RGB color for lines [0-1 range] (default: red)

    Returns:
        Line set or None if less than 2 positions

    Raises:
        ImportError: If Open3D is not available

    Example:
        >>> positions = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        >>> lineset = create_trajectory_path_lineset(positions, color=[0, 0, 1])
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available. Install with: pip install open3d")

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
    color: List[float] = [1, 0, 0]
) -> 'o3d.geometry.LineSet':
    """Create arrows showing viewing directions (camera Z-axis)

    Args:
        positions: List of camera positions
        rotations: List of rotation matrices
        arrow_length: Length of arrows in meters
        color: RGB color for arrows [0-1 range] (default: red)

    Returns:
        Line set representing arrows

    Raises:
        ImportError: If Open3D is not available

    Example:
        >>> positions = [np.array([1.0, 0.0, 0.5])]
        >>> rotations = [np.eye(3)]
        >>> arrows = create_viewing_direction_arrows(positions, rotations, arrow_length=0.1)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available. Install with: pip install open3d")

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


def visualize_trajectory_with_mesh(
    mesh: 'o3d.geometry.TriangleMesh',
    positions: List[np.ndarray],
    rotations: List[np.ndarray],
    show_frames: bool = False,
    frame_size: float = 0.02,
    show_path: bool = True,
    show_normals: bool = True,
    normal_length: float = 0.05,
    window_name: str = "GTSP Trajectory Visualization",
    window_width: int = 1280,
    window_height: int = 720
) -> None:
    """Visualize mesh with trajectory poses using Open3D

    Args:
        mesh: Open3D mesh to visualize
        positions: List of waypoint positions
        rotations: List of rotation matrices
        show_frames: Show coordinate frames instead of points
        frame_size: Size of coordinate frames in meters
        show_path: Show trajectory path lines
        show_normals: Show viewing direction arrows
        normal_length: Length of viewing arrows in meters
        window_name: Visualization window title
        window_width: Window width in pixels
        window_height: Window height in pixels

    Raises:
        ImportError: If Open3D is not available

    Example:
        >>> mesh = o3d.io.read_triangle_mesh('glass.obj')
        >>> positions = [np.array([1.0, 0.0, 0.5])]
        >>> rotations = [np.eye(3)]
        >>> visualize_trajectory_with_mesh(mesh, positions, rotations)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D not available. Install with: pip install open3d")

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
        for pos, rot in zip(positions, rotations):
            frame = create_coordinate_frame_o3d(pos, rot, size=frame_size)
            geometries.append(frame)
    else:
        # Show as points
        points = np.array(positions)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0, 1, 0])  # Green points
        geometries.append(pcd)

    # Add viewing direction arrows
    if show_normals:
        arrows = create_viewing_direction_arrows(
            positions, rotations,
            arrow_length=normal_length,
            color=[1, 0, 0]
        )
        geometries.append(arrows)

    # Add trajectory path
    if show_path and len(positions) > 1:
        path = create_trajectory_path_lineset(positions, color=[0, 0, 1])  # Blue for path
        if path:
            geometries.append(path)

    # Visualize
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=window_width,
        height=window_height
    )
