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

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import os
import torch

# Standard Library
import argparse
from time import perf_counter

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
args = parser.parse_args()

############################################################

# Third Party
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1280",
        "height": "720",
    }
)

# # Anti_aliasing mode 변경!!
# import omni.replicator.core as rep
# rep.settings.set_render_rtx_realtime(antialiasing="TAA")

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# Third Party
import carb
import numpy as np
from urchin import URDF

from utilss.simulation_helper import add_extensions, add_robot_to_scene
from curobo.util.usd_helper import UsdHelper

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.rotations import quat_to_rot_matrix

try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    from isaacsim.util.debug_draw import _debug_draw

# Camera
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from omni.isaac.core.prims import XFormPrim
from isaacsim.core.api.materials import OmniGlass

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.geom.types import Mesh
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
from curobo.types.math import quat_multiply
from curobo.geom.transform import pose_to_matrix, matrix_to_quaternion

# EAIK
from eaik.IK_URDF import UrdfRobot
from eaik.IK_DH import DhRobot

SAMPLED_LOCAL_POINTS: Optional[np.ndarray] = None
SAMPLED_LOCAL_NORMALS: Optional[np.ndarray] = None

CUROBO_TO_EAIK_TOOL = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

NORMAL_SAMPLE_OFFSET = 0.1  # meters
OPEN3D_TO_ISAAC_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass
class Viewpoint:
    index: int
    local_pose: Optional[np.ndarray] = None
    world_pose: Optional[np.ndarray] = None
    all_ik_solutions: List[np.ndarray] = field(default_factory=list)
    safe_ik_solutions: List[np.ndarray] = field(default_factory=list)


class ViewpointList:
    def __init__(self, viewpoints: Optional[Iterable[Viewpoint]] = None) -> None:
        self.viewpoints: List[Viewpoint] = list(viewpoints) if viewpoints else []


SAMPLED_VIEWPOINTS: List[Viewpoint] = []

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.clip(norms, 1e-9, None))


def offset_points_along_normals(
    points: np.ndarray, normals: np.ndarray, offset: float
) -> np.ndarray:
    if points.size == 0:
        return points
    if points.shape != normals.shape:
        raise ValueError("Points and normals must have the same shape")
    safe_normals = normalize_vectors(normals)
    return points + safe_normals * offset

def open3d_to_isaac_coords(points: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Open3D coordinates (Y-up) to Isaac Sim coordinates (Z-up).

    Open3D typically uses Y-up coordinate system while Isaac Sim uses Z-up.
    This function applies the necessary rotation to align the coordinate systems.
    """
    # Apply rotation to points and normals
    isaac_points = (OPEN3D_TO_ISAAC_ROT @ points.T).T
    isaac_normals = normalize_vectors((OPEN3D_TO_ISAAC_ROT @ normals.T).T)

    return isaac_points, isaac_normals


def open3d_pose_to_world(pose_matrix: np.ndarray, reference_prim: XFormPrim) -> np.ndarray:
    if pose_matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4")
    if reference_prim is None:
        raise ValueError("reference_prim is required to map pose into world coordinates")

    world_position, world_orientation = reference_prim.get_world_pose()
    local_scale = np.asarray(reference_prim.get_local_scale(), dtype=np.float64)
    rotation_matrix = quat_to_rot_matrix(np.asarray(world_orientation, dtype=np.float64))

    local_rot = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, :3]
    local_pos = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, 3]

    scaled_pos = local_pos * local_scale
    rotated_pos = rotation_matrix @ scaled_pos
    world_pos = rotated_pos + np.asarray(world_position, dtype=np.float64)

    world_rot = rotation_matrix @ local_rot

    world_pose = np.eye(4, dtype=np.float64)
    world_pose[:3, :3] = world_rot
    world_pose[:3, 3] = world_pos
    return world_pose


def update_viewpoints_world_pose(viewpoints: Iterable[Viewpoint], reference_prim: Optional[XFormPrim]) -> None:
    if reference_prim is None:
        return
    for viewpoint in viewpoints:
        if viewpoint.local_pose is None:
            continue
        viewpoint.world_pose = open3d_pose_to_world(viewpoint.local_pose, reference_prim)


def collect_viewpoint_world_matrices(
    viewpoints: Iterable[Viewpoint],
) -> Tuple[np.ndarray, List[int]]:
    matrices: List[np.ndarray] = []
    indices: List[int] = []

    for idx, viewpoint in enumerate(viewpoints):
        if viewpoint.world_pose is None:
            continue
        matrices.append(np.asarray(viewpoint.world_pose, dtype=np.float64))
        indices.append(idx)

    if matrices:
        stacked = np.stack(matrices, axis=0)
    else:
        stacked = np.empty((0, 4, 4), dtype=np.float64)

    return stacked, indices


def assign_eaik_solutions(
    viewpoints: Iterable[Viewpoint],
    eaik_results,
    indices: Optional[List[int]] = None,
) -> None:
    viewpoint_list = list(viewpoints)
    for viewpoint in viewpoint_list:
        viewpoint.all_ik_solutions = []
        viewpoint.safe_ik_solutions = []

    if eaik_results is None:
        return

    try:
        eaik_list = list(eaik_results)
    except TypeError:
        eaik_list = [eaik_results]

    if indices is None:
        indices = list(range(len(viewpoint_list)))

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
        viewpoint_list[viewpoint_idx].all_ik_solutions = solutions


def update_safe_ik_solutions(
    viewpoints: Iterable[Viewpoint],
    ik_solver: IKSolver,
) -> List[np.ndarray]:
    safe_targets: List[np.ndarray] = []
    for viewpoint in viewpoints:
        safe_solutions: List[np.ndarray] = []
        for solution in viewpoint.all_ik_solutions:
            q_candidate = np.asarray(solution, dtype=np.float64)
            if collision_checking(ik_solver, q_candidate):
                safe_solutions.append(q_candidate)
                safe_targets.append(q_candidate)
        viewpoint.safe_ik_solutions = safe_solutions
    return safe_targets


def log_viewpoint_ik_stats(viewpoints: Iterable[Viewpoint]) -> None:
    viewpoint_list = list(viewpoints)
    total = len(viewpoint_list)
    if total == 0:
        print("No sampled viewpoints available for statistics.")
        return

    solved = sum(1 for vp in viewpoint_list if len(vp.all_ik_solutions) > 0)
    collision_free = sum(1 for vp in viewpoint_list if len(vp.safe_ik_solutions) > 0)
    print(
        "EAIK IK solutions: "
        f"{solved}/{total} total, {collision_free}/{total} collision-free"
    )


def initialize_world():
    # collision checking
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    usd_helper = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[:3] = [0.7, 0.0, 0.0]
    world_cfg_table.cuboid[0].dims[:3] = [0.6, 1.0, 1.1]

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
    )
    ik_solver = IKSolver(ik_config)

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(
        robot_config=robot_cfg, 
        my_world=my_world,
        position=np.array([0.0, 0.0, 0.0]),
    )

    articulation_controller = robot.get_articulation_controller()
    idx_list = [robot.get_dof_index(x) for x in j_names]
    robot.set_joint_positions(default_config, idx_list)
    # mount_prim = add_robot_mount()

    # 유리 추가
    asset_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_isaac_ori.usdc"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/glass_usd")

    glass_prim = XFormPrim(
        prim_path="/World/glass_usd",
        position=np.array([0.7, 0.0, 0.6]),
    )

    glass_material = OmniGlass(
        prim_path="/World/Looks/glass_mat",
        color=np.array([0.7, 0.85, 0.9]),
        ior=1.52,
        depth=0.01,
        thin_walled=False,
    )
    glass_prim.apply_visual_material(glass_material)


    # Setup world configuration
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane(z_position=-0.5)

    # 장애물 업데이트
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

    return my_world, glass_prim, robot, idx_list, ik_solver


def load_mesh():
    import open3d as o3d
    global SAMPLED_LOCAL_POINTS, SAMPLED_LOCAL_NORMALS, SAMPLED_VIEWPOINTS

    SAMPLED_VIEWPOINTS = []
    test_obj_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_o3d.obj"
    mesh = o3d.io.read_triangle_mesh(test_obj_path)
    mesh.compute_vertex_normals()

    print("Visualizing original mesh...")
    # o3d.visualization.draw_geometries([mesh])

    print("Converting mesh to point cloud using Poisson sampling...")
    pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=20000)

    # Estimate normals for point cloud
    pcd_poisson.estimate_normals()
    # Set uniform color (gray) to prevent rainbow coloring
    pcd_poisson.paint_uniform_color([0.7, 0.7, 0.7])
    # print(f"Point cloud has {len(pcd_poisson.points)} points and {len(pcd_poisson.normals)} normals")

    # Save point cloud to file
    output_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_pointcloud.pcd"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    o3d.io.write_point_cloud(output_path, pcd_poisson)
    # print(f"Point cloud saved to: {output_path}")

    # print("Visualizing Poisson sampled point cloud...")
    # o3d.visualization.draw_geometries([pcd_poisson])

    points = np.asarray(pcd_poisson.points)
    normals = np.asarray(pcd_poisson.normals)

    valid_indices = [idx for idx in range(len(points))]
    sample_indices = np.asarray(valid_indices, dtype=np.int64)

    if sample_indices.size > 0:
        sampled_cloud = o3d.geometry.PointCloud()
        sampled_points = points[sample_indices]
        sampled_normals = normals[sample_indices]
        offset_points = offset_points_along_normals(
            sampled_points, sampled_normals, NORMAL_SAMPLE_OFFSET
        )
        approach_normals = -normalize_vectors(sampled_normals)

        sampled_cloud.points = o3d.utility.Vector3dVector(offset_points)
        sampled_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        print("Visualizing preset point samples in Open3D alongside the full cloud...")
        o3d.visualization.draw_geometries([pcd_poisson, sampled_cloud])

        helper_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        helper_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        sampled_viewpoints: List[Viewpoint] = []

        for point_idx, position, normal in zip(sample_indices, offset_points, approach_normals):
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

            pose_matrix = np.eye(4, dtype=np.float64)
            pose_matrix[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
            pose_matrix[:3, 3] = position.astype(np.float64)

            sampled_viewpoints.append(Viewpoint(index=int(point_idx), local_pose=pose_matrix))

        SAMPLED_LOCAL_POINTS = offset_points
        SAMPLED_LOCAL_NORMALS = approach_normals
        SAMPLED_VIEWPOINTS = sampled_viewpoints
    else:
        SAMPLED_LOCAL_POINTS = None
        SAMPLED_LOCAL_NORMALS = None
        SAMPLED_VIEWPOINTS = []

    return points, normals


def load_pcd():
    import open3d as o3d
    global SAMPLED_LOCAL_POINTS, SAMPLED_LOCAL_NORMALS, SAMPLED_VIEWPOINTS

    SAMPLED_VIEWPOINTS = []
    pcd_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_pointcloud.pcd"

    print(f"Loading point cloud from: {pcd_path}")
    pcd_poisson = o3d.io.read_point_cloud(pcd_path)

    if len(pcd_poisson.points) == 0:
        print("Warning: No points loaded from PCD file")
        return np.array([]), np.array([])

    # Estimate normals if not present
    if not pcd_poisson.has_normals():
        print("Estimating normals for loaded point cloud...")
        pcd_poisson.estimate_normals()

    # Set uniform color (gray) to prevent rainbow coloring
    pcd_poisson.paint_uniform_color([0.7, 0.7, 0.7])
    print(f"Point cloud has {len(pcd_poisson.points)} points and {len(pcd_poisson.normals)} normals")

    points = np.asarray(pcd_poisson.points)
    normals = np.asarray(pcd_poisson.normals)

    # valid_indices = [idx for idx in range(len(points))]
    valid_indices = [idx for idx in range(len(points))]

    sample_indices = np.asarray(valid_indices, dtype=np.int64)

    if sample_indices.size > 0:
        sampled_cloud = o3d.geometry.PointCloud()
        sampled_points = points[sample_indices]
        sampled_normals = normals[sample_indices]
        offset_points = offset_points_along_normals(
            sampled_points, sampled_normals, NORMAL_SAMPLE_OFFSET
        )
        approach_normals = -normalize_vectors(sampled_normals)

        sampled_cloud.points = o3d.utility.Vector3dVector(offset_points)
        sampled_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        print("Visualizing preset point samples in Open3D alongside the full cloud...")
        o3d.visualization.draw_geometries([pcd_poisson, sampled_cloud])

        helper_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        helper_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        sampled_viewpoints: List[Viewpoint] = []

        for point_idx, position, normal in zip(sample_indices, offset_points, approach_normals):
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

            pose_matrix = np.eye(4, dtype=np.float64)
            pose_matrix[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
            pose_matrix[:3, 3] = position.astype(np.float64)

            sampled_viewpoints.append(Viewpoint(index=int(point_idx), local_pose=pose_matrix))

        SAMPLED_LOCAL_POINTS = offset_points
        SAMPLED_LOCAL_NORMALS = approach_normals
        SAMPLED_VIEWPOINTS = sampled_viewpoints
    else:
        SAMPLED_LOCAL_POINTS = None
        SAMPLED_LOCAL_NORMALS = None

    return points, normals


def convert_points(
    points: np.ndarray,
    normals: np.ndarray,
    reference_prim: Optional[XFormPrim] = None,
    debug: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Open3D point cloud data into Isaac Sim world coordinates.

    First applies coordinate system conversion (Y-up to Z-up), then applies
    the object's world transform (scale, rotation, translation).
    """
    if points.size == 0 or normals.size == 0:
        return points, normals

    # Step 1: Convert coordinate system from Open3D (Y-up) to Isaac Sim (Z-up)
    isaac_points, isaac_normals = open3d_to_isaac_coords(
        np.asarray(points, dtype=np.float64),
        np.asarray(normals, dtype=np.float64)
    )

    if debug:
        print(f"Original point range: X[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
              f"Y[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
              f"Z[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
        print(f"After coord conversion: X[{isaac_points[:, 0].min():.3f}, {isaac_points[:, 0].max():.3f}], "
              f"Y[{isaac_points[:, 1].min():.3f}, {isaac_points[:, 1].max():.3f}], "
              f"Z[{isaac_points[:, 2].min():.3f}, {isaac_points[:, 2].max():.3f}]")

    if reference_prim is None:
        return isaac_points, isaac_normals

    # Step 2: Get object's world transform
    world_position, world_orientation = reference_prim.get_world_pose()
    local_scale = reference_prim.get_local_scale()

    if debug:
        print(f"Object world position: {world_position}")
        print(f"Object world orientation: {world_orientation}")
        print(f"Object local scale: {local_scale}")

    # Step 3: Apply proper transformation order: Scale → Rotate → Translate
    rotation_matrix = quat_to_rot_matrix(np.asarray(world_orientation, dtype=np.float64))

    # Scale first
    scaled_points = isaac_points * local_scale

    # Then rotate
    rotated_points = (rotation_matrix @ scaled_points.T).T

    # Finally translate
    world_points = rotated_points + np.asarray(world_position, dtype=np.float64)

    # For normals: only apply rotation (inverse transpose of scale and rotation)
    # For uniform scaling, normal transformation is just rotation
    safe_scale = np.where(local_scale == 0.0, 1.0, local_scale)
    scale_matrix = np.diag(1.0 / safe_scale)  # Inverse scale
    normal_transform = rotation_matrix @ scale_matrix

    world_normals = normalize_vectors((normal_transform @ isaac_normals.T).T)

    if debug:
        print(f"Final world points range: X[{world_points[:, 0].min():.3f}, {world_points[:, 0].max():.3f}], "
              f"Y[{world_points[:, 1].min():.3f}, {world_points[:, 1].max():.3f}], "
              f"Z[{world_points[:, 2].min():.3f}, {world_points[:, 2].max():.3f}]")

    return world_points, world_normals


def visualize_sampled_points_in_isaac(
    world_points: np.ndarray,
    world_normals: np.ndarray,
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    normal_scale: float = 0.05,
) -> None:
    """Draw sampled points (and optional normal rays) in Isaac Sim using debug_draw."""

    world_points = np.asarray(world_points, dtype=np.float32)
    world_normals = np.asarray(world_normals, dtype=np.float32)

    if world_points.size == 0:
        return

    drawer = _debug_draw.acquire_debug_draw_interface()

    drawer.clear_points()
    drawer.clear_lines()

    point_list = [tuple(p) for p in world_points]
    point_colors = [(*color, 1.0)] * len(point_list)
    point_sizes = [6.0] * len(point_list)

    drawer.draw_points(point_list, point_colors, point_sizes)

    if world_normals.size:
        start_points = []
        end_points = []
        line_colors = []
        thicknesses = []
        rgba = (*color, 1.0)

        for point, normal in zip(world_points, world_normals):
            start = tuple(point)
            end = tuple(point + normal * normal_scale)
            start_points.append(start)
            end_points.append(end)
            line_colors.append(rgba)
            thicknesses.append(1.5)

        drawer.draw_lines(start_points, end_points, line_colors, thicknesses)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compare_ik_solution(eaik_solution, curobo_solution: torch.Tensor, tolerance: float = 1e-3) -> bool:
    if eaik_solution is None:
        return False

    curobo_np = curobo_solution.detach().cpu().numpy().flatten()

    q_list = getattr(eaik_solution, "Q", [])
    for q in q_list:
        diff = _wrap_to_pi(curobo_np - np.asarray(q))
        if np.all(np.abs(diff) < tolerance):
            return True
    return False


def compare_ik_curobo(
    goal_pose: Pose,
    ik_solver: IKSolver,
    eaik_results,
    tolerance: float = 1e-3,
):
    solver_result = ik_solver.solve_batch(goal_pose)

    success_flags = solver_result.success.detach().cpu().numpy()
    comparison_summary = []

    for idx, success in enumerate(success_flags):
        if not success:
            comparison_summary.append((idx, False, "curobo failed"))
            continue

        js_solution = solver_result.js_solution[idx].position
        eaik_entry = eaik_results[idx] if idx < len(eaik_results) else None
        match = compare_ik_solution(eaik_entry, js_solution, tolerance=tolerance)
        reason = "match" if match else "no matching EAIK solution"
        comparison_summary.append((idx, match, reason))

    return solver_result, comparison_summary

def compute_ik(mats: torch.Tensor, urdf_path: Optional[str] = None):
    if isinstance(mats, torch.Tensor):
        mats_np = mats.detach().cpu().numpy()
    else:
        mats_np = np.asarray(mats)

    if mats_np.ndim != 3 or mats_np.shape[1:] != (4, 4):
        raise ValueError("Expected poses shaped (batch, 4, 4)")

    if urdf_path is None:
        script_dir = os.path.dirname(__file__)
        urdf_path = os.path.abspath(
            os.path.join(script_dir, "..", "data", "input", "ur20.urdf")
        )

    bot = UrdfRobot(urdf_path)
    curobo_to_eaik_tool = CUROBO_TO_EAIK_TOOL.astype(mats_np.dtype, copy=False)

    mats_eaik = mats_np @ curobo_to_eaik_tool

    solutions = bot.IK_batched(mats_eaik)
    
    return solutions

def collision_checking(ik_solver: IKSolver, q_values) -> bool:
    tensor_args = getattr(ik_solver, "tensor_args", TensorDeviceType())
    q_tensor = tensor_args.to_device(np.asarray(q_values, dtype=np.float64))
    if q_tensor.ndim == 1:
        q_tensor = q_tensor.unsqueeze(0)

    zeros = torch.zeros_like(q_tensor)
    joint_state = JointState(
        position=q_tensor,
        velocity=zeros,
        acceleration=zeros,
        jerk=zeros,
        joint_names=ik_solver.kinematics.joint_names,
    )

    metrics = ik_solver.check_constraints(joint_state)
    # return True
    feasible = getattr(metrics, "feasible", None)
    if feasible is None:
        return True

    feasible_tensor = feasible.detach()
    if feasible_tensor.is_cuda:
        feasible_tensor = feasible_tensor.cpu()
    return bool(feasible_tensor.flatten().bool().all().item())

def set_robot_joint(robot, idx_list, joint_values) -> bool:
    if joint_values is None:
        return False

    if isinstance(joint_values, torch.Tensor):
        cmd = joint_values.detach().cpu().numpy()
    else:
        cmd = np.asarray(joint_values)

    if cmd.ndim != 1 or len(cmd) != len(idx_list):
        raise ValueError("Joint command must match controlled DOF length")

    robot.set_joint_positions(cmd.tolist(), idx_list)
    return True
    
def normals_to_quats(
    normals: np.ndarray,
    tensor_args: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    """Convert surface normals into quaternions that align the tool +Z axis with each normal."""

    if normals.size == 0:
        dtype = tensor_args.dtype if tensor_args is not None else torch.float32
        device = tensor_args.device if tensor_args is not None else "cpu"
        return torch.empty((0, 4), dtype=dtype, device=device)

    normals = np.asarray(normals, dtype=np.float64)
    helper_z = np.array([0.0, 0.0, 1.0])
    helper_y = np.array([0.0, 1.0, 0.0])
    rot_mats: List[np.ndarray] = []

    for normal in normals:
        z_axis = normal / np.linalg.norm(normal)

        # 보조 축이 법선과 너무 평행하면 다른 축을 사용해 수치 불안정을 방지한다.
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
        rot_mats.append(np.stack([x_axis, y_axis, z_axis], axis=1))

    rot_stack = np.stack(rot_mats, axis=0)
    dtype = tensor_args.dtype if tensor_args is not None else torch.float32
    device = tensor_args.device if tensor_args is not None else "cpu"
    rot_tensor = torch.from_numpy(rot_stack).to(device=device, dtype=dtype)
    return matrix_to_quaternion(rot_tensor)
    

def main():
    tensor_args = TensorDeviceType()

    # Check if PCD file exists, if not generate it from mesh first
    pcd_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_pointcloud.pcd"

    if os.path.exists(pcd_path):
        print("Loading existing point cloud from PCD file...")
        o3d_points, o3d_normals = load_pcd()
    else:
        print("PCD file not found. Generating point cloud from mesh...")
        o3d_points, o3d_normals = load_mesh()

        # If mesh loading failed, try to load PCD again in case it was just created
        if o3d_points.size == 0:
            print("Mesh loading failed. Checking for PCD file again...")
            if os.path.exists(pcd_path):
                o3d_points, o3d_normals = load_pcd()

    if o3d_points.size == 0:
        print("Error: Could not load any point cloud data!")
        return

    print(f"Loaded {len(o3d_points)} points with {len(o3d_normals)} normals")

    my_world, glass_prim, robot, idx_list, ik_solver = initialize_world()

    ik_joint_targets: List[np.ndarray] = []
    target_idx = 0

    # Also visualize sampled points if available
    if SAMPLED_LOCAL_POINTS is not None and SAMPLED_LOCAL_NORMALS is not None:
        update_viewpoints_world_pose(SAMPLED_VIEWPOINTS, glass_prim)

        world_mats, used_indices = collect_viewpoint_world_matrices(SAMPLED_VIEWPOINTS)
        if world_mats.size == 0:
            print("Warning: No valid world poses found for sampled viewpoints")
        else:
            ik_results = compute_ik(
                mats=world_mats,
                urdf_path="/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf"
            )

            assign_start = perf_counter()
            assign_eaik_solutions(SAMPLED_VIEWPOINTS, ik_results, used_indices)
            assign_elapsed = perf_counter() - assign_start

            safe_start = perf_counter()
            ik_joint_targets = update_safe_ik_solutions(SAMPLED_VIEWPOINTS, ik_solver)
            safe_elapsed = perf_counter() - safe_start

            print(
                f"assign_eaik_solutions duration: {assign_elapsed * 1000.0:.2f} ms"
            )
            print(
                f"update_safe_ik_solutions duration: {safe_elapsed * 1000.0:.2f} ms"
            )
            log_viewpoint_ik_stats(SAMPLED_VIEWPOINTS)


    step_counter = 0
    idle_counter = 0

    # Main simulation loop
    while simulation_app.is_running():
        my_world.step(render=True)
        
        if not my_world.is_playing():
            if idle_counter % 100 == 0:
                print("**** Click Play to start simulation *****")
            idle_counter += 1
            continue

        idle_counter = 0
        step_counter += 1

        if ik_joint_targets and step_counter % 100 == 0:
            joint_cmd = ik_joint_targets[target_idx % len(ik_joint_targets)]
            if set_robot_joint(robot, idx_list, joint_cmd):
                target_idx = (target_idx + 1) % len(ik_joint_targets)
        
        # if ik_joint_targets and step_counter % 100 == 0:
        #     joint_cmd = curobo_results.js_solution[target_idx % len(ik_joint_targets)].position

        #     print(joint_cmd)
        #     if set_robot_joint(robot, idx_list, joint_cmd):
        #         target_idx = (target_idx + 1) % len(ik_joint_targets)
            
        # Update simulation step
        # my_world.current_time_step_index

    simulation_app.close()


if __name__ == "__main__":
    main()
