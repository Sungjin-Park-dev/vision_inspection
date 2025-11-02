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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt

# Standard Library
import argparse
import csv
from time import perf_counter
from datetime import datetime

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

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

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
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.sdf.world import CollisionCheckerType

# Joint reconfiguration analysis
from analyze_joint_reconfigurations import (
    load_joint_trajectory,
    analyze_joint_reconfigurations,
    print_analysis_results
)
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
INTERPOLATION_STEPS = 60


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
TSP_TOUR_RESULT: Optional[dict] = None  # Global variable to store TSP tour result

# Joint trajectory data collection
JOINT_HISTORY = {
    'timestamps': [],
    'joint_values': [],  # List of arrays with shape (6,) for UR20
    'viewpoint_markers': []  # List of timestamps when reaching new viewpoints
}

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


def open3d_pose_to_world(pose_matrix: np.ndarray, reference_prim: XFormPrim, debug: bool = False) -> np.ndarray:
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
        print(f"Original pose position (Open3D): {pose_matrix[:3, 3]}")

    local_rot = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, :3]
    local_pos = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, 3]

    if debug:
        print(f"After Open3D->Isaac rotation: {local_pos}")

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


def update_viewpoints_world_pose(viewpoints: Iterable[Viewpoint], reference_prim: Optional[XFormPrim], debug_first: bool = True) -> None:
    if reference_prim is None:
        return
    for i, viewpoint in enumerate(viewpoints):
        if viewpoint.local_pose is None:
            continue
        # Debug first viewpoint to see transformation
        debug = debug_first and i == 0
        viewpoint.world_pose = open3d_pose_to_world(viewpoint.local_pose, reference_prim, debug=debug)


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


def get_active_joint_positions(robot, idx_list: List[int]) -> np.ndarray:
    all_positions = robot.get_joint_positions()
    return np.asarray([all_positions[i] for i in idx_list], dtype=np.float64)


def generate_interpolated_joint_path(
    start: np.ndarray,
    target: np.ndarray,
    num_steps: int = INTERPOLATION_STEPS,
) -> List[np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if start.shape != target.shape:
        raise ValueError("Start and target joint vectors must have the same shape")
    if num_steps <= 0:
        return [target]

    alphas = np.linspace(0.0, 1.0, num_steps + 1, endpoint=True)[1:]
    path = [start + alpha * (target - start) for alpha in alphas]
    return path if path else [target]


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
) -> None:
    viewpoint_list = list(viewpoints)
    for viewpoint in viewpoint_list:
        viewpoint.safe_ik_solutions = []

    batched_q: List[np.ndarray] = []
    index_map: List[Tuple[int, int]] = []

    for vp_idx, viewpoint in enumerate(viewpoint_list):
        for sol_idx, solution in enumerate(viewpoint.all_ik_solutions):
            batched_q.append(np.asarray(solution, dtype=np.float64))
            index_map.append((vp_idx, sol_idx))

    if not batched_q:
        return

    batched_array = np.stack(batched_q, axis=0)
    tensor_args = getattr(ik_solver, "tensor_args", TensorDeviceType())
    q_tensor = tensor_args.to_device(torch.from_numpy(batched_array))

    zeros = torch.zeros_like(q_tensor)
    joint_state = JointState(
        position=q_tensor,
        velocity=zeros,
        acceleration=zeros,
        jerk=zeros,
        joint_names=ik_solver.kinematics.joint_names,
    )

    metrics = ik_solver.check_constraints(joint_state)
    feasible = getattr(metrics, "feasible", None)
    if feasible is None:
        feasibility = torch.ones(len(index_map), dtype=torch.bool)
    else:
        feasibility = feasible.detach()
        if feasibility.is_cuda:
            feasibility = feasibility.cpu()
        feasibility = feasibility.flatten().to(dtype=torch.bool)

    for batch_idx, ((vp_idx, _), is_feasible) in enumerate(zip(index_map, feasibility)):
        if not bool(is_feasible):
            continue
        solution = batched_q[batch_idx]
        viewpoint_list[vp_idx].safe_ik_solutions.append(solution)
    return


def collect_random_joint_targets(viewpoints: Iterable[Viewpoint]) -> List[np.ndarray]:
    """
    Select first safe IK solution for each viewpoint (FIRST approach)

    Args:
        viewpoints: Iterable of Viewpoint objects (must be in TSP order)

    Returns:
        List of first safe IK solutions in TSP visit order
    """
    print(f"\nCollecting joint targets in TSP order (FIRST):")

    targets: List[np.ndarray] = []
    viewpoint_indices_collected = []

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} (position {idx}) has no safe IK solutions, skipping")
            continue

        # Select first solution
        solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)

        targets.append(solution)
        viewpoint_indices_collected.append(viewpoint.index)

    print(f"  → Collected {len(targets)} targets in order: {viewpoint_indices_collected[:10]}{'...' if len(viewpoint_indices_collected) > 10 else ''}")
    return targets


def collect_sorted_joint_safe_targets(viewpoints: Iterable[Viewpoint]) -> List[np.ndarray]:
    """
    Collect safe IK solutions from viewpoints in TSP tour order (GREEDY approach)

    Args:
        viewpoints: Iterable of Viewpoint objects (must be in TSP order)

    Returns:
        List of joint solutions in TSP visit order
    """
    print(f"\nCollecting joint targets in TSP order (GREEDY):")

    # Preserve TSP order: return viewpoints in the order they appear
    targets: List[np.ndarray] = []
    viewpoint_indices_collected = []
    previous_joints = None

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} (position {idx}) has no safe IK solutions, skipping")
            continue

        if previous_joints is None:
            # For the first viewpoint, just use the first safe solution
            # solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)  # Original line
            solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)
        else:
            # Find the IK solution closest to the previous joint configuration
            best_solution = None
            min_distance = float('inf')

            for candidate in viewpoint.safe_ik_solutions:
                candidate_joints = np.asarray(candidate, dtype=np.float64)
                # Calculate Euclidean distance in joint space
                distance = np.linalg.norm(candidate_joints - previous_joints)
                if distance < min_distance:
                    min_distance = distance
                    best_solution = candidate_joints

            solution = best_solution

        targets.append(solution)
        viewpoint_indices_collected.append(viewpoint.index)
        previous_joints = solution

    print(f"  → Collected {len(targets)} targets in order: {viewpoint_indices_collected[:10]}{'...' if len(viewpoint_indices_collected) > 10 else ''}")
    return targets


def compute_weighted_joint_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    joint_weights: Optional[np.ndarray] = None
) -> float:
    """
    Compute weighted Euclidean distance between two joint configurations.

    Args:
        q1: First joint configuration (6 DOF)
        q2: Second joint configuration (6 DOF)
        joint_weights: Weight for each joint (default: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
                      Higher weights for base joints (1, 2, 3)

    Returns:
        Weighted Euclidean distance
    """
    if joint_weights is None:
        # Default: higher weights for joints 1, 2, 3 (indices 0, 1, 2)
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    # Weighted squared differences
    weighted_diff = joint_weights * (q1_array - q2_array) ** 2

    # Return weighted L2 norm
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
    Compute reconfiguration-aware cost that penalizes large joint movements.

    This hybrid cost function combines:
    1. Weighted Euclidean distance (base cost)
    2. Reconfiguration count penalty (penalize joints that move > threshold)
    3. Max joint movement penalty (penalize the largest single joint movement)

    Args:
        q1: First joint configuration (6 DOF)
        q2: Second joint configuration (6 DOF)
        joint_weights: Weight for each joint (default: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        reconfig_threshold: Threshold (radians) for considering a joint as "reconfigured"
        reconfig_penalty: Penalty cost per reconfigured joint
        max_move_weight: Weight for maximum joint movement penalty

    Returns:
        Tuple of:
        - total_cost: Combined cost value
        - cost_breakdown: Dictionary with cost components
    """
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    # 1. Weighted Euclidean distance (base cost)
    base_cost = compute_weighted_joint_distance(q1_array, q2_array, joint_weights)

    # 2. Reconfiguration penalty: count joints that moved significantly
    joint_deltas = np.abs(q1_array - q2_array)
    reconfig_count = np.sum(joint_deltas > reconfig_threshold)
    reconfig_cost = reconfig_count * reconfig_penalty

    # 3. Max joint movement penalty: penalize the largest movement
    max_joint_move = np.max(joint_deltas)
    max_move_cost = max_joint_move * max_move_weight

    # Total cost
    # total_cost = base_cost + reconfig_cost + max_move_cost
    total_cost = base_cost


    # Cost breakdown for analysis
    cost_breakdown = {
        'base_cost': base_cost,
        'reconfig_count': int(reconfig_count),
        'reconfig_cost': reconfig_cost,
        'max_joint_move': max_joint_move,
        'max_move_cost': max_move_cost,
        'total_cost': total_cost
    }

    return total_cost, cost_breakdown


def collect_optimal_joint_targets_dp(
    viewpoints: Iterable[Viewpoint],
    initial_joint_config: np.ndarray,
    joint_weights: Optional[np.ndarray] = None,
    reconfig_threshold: float = 1.0,
    reconfig_penalty: float = 100.0,
    max_move_weight: float = 5.0,
) -> Tuple[List[np.ndarray], float, List[int], List[int]]:
    """
    Collect optimal IK solutions using Dynamic Programming with reconfiguration-aware cost.

    Args:
        viewpoints: Iterable of Viewpoint objects (must be in TSP order)
        initial_joint_config: Starting joint configuration (e.g., retract_config)
        joint_weights: Weight for each joint (default: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
        reconfig_threshold: Threshold (radians) for considering a joint as "reconfigured"
        reconfig_penalty: Penalty cost per reconfigured joint
        max_move_weight: Weight for maximum joint movement penalty

    Returns:
        Tuple of:
        - targets: List of optimal joint solutions in TSP visit order
        - total_cost: Total reconfiguration-aware cost
        - solution_indices: List of selected solution indices for each viewpoint
        - valid_viewpoint_indices: List of viewpoint indices that have safe solutions
    """
    print(f"\nCollecting joint targets using Dynamic Programming:")

    viewpoint_list = list(viewpoints)
    initial_joints = np.asarray(initial_joint_config, dtype=np.float64)

    # Filter viewpoints with safe solutions
    valid_viewpoints = []
    valid_indices = []
    for idx, vp in enumerate(viewpoint_list):
        if vp.safe_ik_solutions:
            valid_viewpoints.append(vp)
            valid_indices.append(idx)
        else:
            print(f"  Warning: Viewpoint {vp.index} (position {idx}) has no safe IK solutions, skipping")

    if not valid_viewpoints:
        print("  Error: No valid viewpoints with safe IK solutions!")
        return [], 0.0, [], []

    n_viewpoints = len(valid_viewpoints)

    # Print cost function parameters
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)
    print(f"  Processing {n_viewpoints} viewpoints with safe solutions")
    print(f"  Joint weights: {joint_weights}")
    print(f"  Reconfiguration threshold: {reconfig_threshold} rad")
    print(f"  Reconfiguration penalty: {reconfig_penalty}")
    print(f"  Max move weight: {max_move_weight}")

    # DP table: dp[i] = dict mapping solution_idx -> (min_cost, prev_solution_idx)
    dp = [dict() for _ in range(n_viewpoints)]

    # Initialize first viewpoint: use reconfiguration cost from initial config
    first_vp = valid_viewpoints[0]
    for sol_idx, solution in enumerate(first_vp.safe_ik_solutions):
        sol_array = np.asarray(solution, dtype=np.float64)
        init_cost, _ = compute_reconfiguration_cost(
            sol_array, initial_joints, joint_weights,
            reconfig_threshold, reconfig_penalty, max_move_weight
        )
        dp[0][sol_idx] = (init_cost, -1)  # -1 indicates no previous solution

    print(f"  Initialized first viewpoint with {len(dp[0])} solutions")

    # Forward pass: fill DP table using reconfiguration-aware cost
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

            # Try all solutions from previous viewpoint
            for prev_sol_idx, (prev_cost, _) in prev_dp.items():
                prev_solution = valid_viewpoints[i - 1].safe_ik_solutions[prev_sol_idx]
                prev_joints = np.asarray(prev_solution, dtype=np.float64)

                # Calculate reconfiguration-aware transition cost
                transition_cost, _ = compute_reconfiguration_cost(
                    curr_joints, prev_joints, joint_weights,
                    reconfig_threshold, reconfig_penalty, max_move_weight
                )
                total_cost = prev_cost + transition_cost

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_prev_idx = prev_sol_idx

            dp[i][curr_sol_idx] = (min_cost, best_prev_idx)

    # Find best solution at last viewpoint
    last_dp = dp[n_viewpoints - 1]
    if not last_dp:
        print("  Error: No valid complete path found!")
        return [], 0.0, [], []

    best_final_idx = min(last_dp.keys(), key=lambda idx: last_dp[idx][0])
    total_distance = last_dp[best_final_idx][0]

    print(f"  DP forward pass complete. Total distance: {total_distance:.4f}")

    # Backward pass: reconstruct optimal path
    solution_indices = [0] * n_viewpoints
    solution_indices[n_viewpoints - 1] = best_final_idx

    for i in range(n_viewpoints - 1, 0, -1):
        current_sol_idx = solution_indices[i]
        _, prev_sol_idx = dp[i][current_sol_idx]
        solution_indices[i - 1] = prev_sol_idx

    # Build final target list
    targets = []
    for i, sol_idx in enumerate(solution_indices):
        solution = valid_viewpoints[i].safe_ik_solutions[sol_idx]
        targets.append(np.asarray(solution, dtype=np.float64))

    viewpoint_indices_collected = [vp.index for vp in valid_viewpoints]
    print(f"  → Collected {len(targets)} targets with total distance {total_distance:.4f}")
    print(f"  → Viewpoint order: {viewpoint_indices_collected[:10]}{'...' if len(viewpoint_indices_collected) > 10 else ''}")

    return targets, total_distance, solution_indices, valid_indices


def analyze_and_save_reconfigurations(
    csv_path: str,
    threshold: float = 1.0,
    output_dir: Optional[str] = None
) -> None:
    """
    Automatically analyze joint reconfigurations from CSV trajectory file.

    Args:
        csv_path: Path to joint trajectory CSV file
        threshold: Threshold for joint change to count as reconfiguration (radians)
        output_dir: Directory to save analysis results (default: same as CSV)
    """
    try:
        # Load trajectory data
        joint_data, joint_names = load_joint_trajectory(csv_path)

        # Analyze reconfigurations
        results = analyze_joint_reconfigurations(joint_data, threshold=threshold)

        # Print results to console
        print_analysis_results(results, joint_names)

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(csv_path)

        # Generate output filename from input filename
        input_basename = os.path.splitext(os.path.basename(csv_path))[0]
        results_file = os.path.join(output_dir, f"{input_basename}_reconfig.txt")

        # Save detailed results to file
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
                involved_joints = [joint_names[i].replace('ur20-', '').replace('_joint', '') for i in joint_indices]
                f.write(f"{reconfig['timestep']:<10} "
                       f"{reconfig['max_change']:<12.3f} "
                       f"{', '.join(involved_joints)}\n")

        print(f"\n{'='*60}")
        print("RECONFIGURATION ANALYSIS SAVED")
        print(f"{'='*60}")
        print(f"Output file: {results_file}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Warning: Failed to analyze joint reconfigurations: {e}")
        print("Continuing without reconfiguration analysis...")


def save_ik_solutions_hdf5(
    viewpoints: List[Viewpoint],
    selected_indices: Optional[List[int]],
    selection_method: str,
    save_path: str,
    tsp_tour_path: str,
) -> None:
    """
    Save all IK solutions and collision information to HDF5 file.

    Args:
        viewpoints: List of Viewpoint objects (in TSP order)
        selected_indices: List of solution indices selected by DP/Greedy/Random
                         Length should match viewpoints with safe solutions
                         None if no selection was made
        selection_method: Method used for selection ("dp", "greedy", "random")
        save_path: Output HDF5 file path
        tsp_tour_path: Path to TSP tour file (for metadata)
    """
    import h5py

    # Filter viewpoints with safe solutions to match selected_indices
    viewpoints_with_safe = [vp for vp in viewpoints if len(vp.safe_ik_solutions) > 0]

    # Create mapping from viewpoint index to selected solution index
    selected_map = {}
    if selected_indices is not None:
        if len(selected_indices) != len(viewpoints_with_safe):
            print(f"Warning: selected_indices length ({len(selected_indices)}) != "
                  f"viewpoints with safe solutions ({len(viewpoints_with_safe)})")
        else:
            for vp, sol_idx in zip(viewpoints_with_safe, selected_indices):
                selected_map[vp.index] = sol_idx

    # Count statistics
    num_with_solutions = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    num_with_safe = len(viewpoints_with_safe)

    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save to HDF5
    with h5py.File(save_path, 'w') as f:
        # Create metadata group
        metadata_grp = f.create_group('metadata')
        metadata_grp.attrs['num_viewpoints'] = len(viewpoints)
        metadata_grp.attrs['num_viewpoints_with_solutions'] = num_with_solutions
        metadata_grp.attrs['num_viewpoints_with_safe_solutions'] = num_with_safe
        metadata_grp.attrs['selection_method'] = selection_method
        metadata_grp.attrs['timestamp'] = datetime.now().isoformat()
        metadata_grp.attrs['tsp_tour_file'] = tsp_tour_path

        # Save each viewpoint's data
        for vp in viewpoints:
            vp_grp_name = f'viewpoint_{vp.index:04d}'
            vp_grp = f.create_group(vp_grp_name)

            # Save original index
            vp_grp.attrs['original_index'] = vp.index

            # Save world pose
            if vp.world_pose is not None:
                vp_grp.create_dataset('world_pose', data=vp.world_pose.astype(np.float32))
            else:
                vp_grp.create_dataset('world_pose', data=np.zeros((4, 4), dtype=np.float32))

            # Save all IK solutions
            if len(vp.all_ik_solutions) > 0:
                all_sols = np.stack([np.asarray(sol, dtype=np.float64) for sol in vp.all_ik_solutions])
                vp_grp.create_dataset('all_ik_solutions', data=all_sols.astype(np.float32))
            else:
                vp_grp.create_dataset('all_ik_solutions', data=np.zeros((0, 6), dtype=np.float32))

            # Create collision free mask
            collision_free_mask = np.zeros(len(vp.all_ik_solutions), dtype=bool)
            for i, sol in enumerate(vp.all_ik_solutions):
                # Check if this solution is in safe_ik_solutions
                sol_array = np.asarray(sol, dtype=np.float64)
                for safe_sol in vp.safe_ik_solutions:
                    if np.allclose(sol_array, safe_sol, atol=1e-6):
                        collision_free_mask[i] = True
                        break
            vp_grp.create_dataset('collision_free_mask', data=collision_free_mask)

            # Save counts
            vp_grp.attrs['num_all_solutions'] = len(vp.all_ik_solutions)
            vp_grp.attrs['num_safe_solutions'] = len(vp.safe_ik_solutions)

            # Save selected solution index
            if vp.index in selected_map:
                vp_grp.attrs['selected_solution_index'] = selected_map[vp.index]
            else:
                vp_grp.attrs['selected_solution_index'] = -1  # Not selected

    print(f"\n{'='*60}")
    print("IK SOLUTIONS HDF5 SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total viewpoints: {len(viewpoints)}")
    print(f"Viewpoints with any IK solutions: {num_with_solutions}")
    print(f"Viewpoints with safe IK solutions: {num_with_safe}")
    print(f"Selection method: {selection_method}")
    print(f"File size: {os.path.getsize(save_path) / 1024:.2f} KB")
    print(f"{'='*60}\n")


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


def verify_tsp_tour_mapping(viewpoints: List[Viewpoint], tsp_result: dict) -> None:
    """
    Verify that TSP tour is correctly mapped to Isaac Sim viewpoints.

    Checks:
    1. Viewpoint indices match TSP tour order (should be 0, 1, 2, ...)
    2. Number of viewpoints matches tour length
    3. World poses are computed correctly
    """
    print(f"\n{'='*60}")
    print("TSP TOUR MAPPING VERIFICATION")
    print(f"{'='*60}")

    tour_indices = tsp_result["tour"]["indices"]
    tour_length = len(tour_indices)

    print(f"TSP tour length: {tour_length}")
    print(f"Number of viewpoints: {len(viewpoints)}")

    if len(viewpoints) != tour_length:
        print(f"⚠ WARNING: Mismatch in counts!")
        return

    # Check if viewpoint indices are sequential (0, 1, 2, ...)
    expected_indices = list(range(tour_length))
    actual_indices = [vp.index for vp in viewpoints]

    if actual_indices == expected_indices:
        print(f"✓ Viewpoint indices are correctly ordered: 0 → {tour_length-1}")
    else:
        print(f"⚠ WARNING: Viewpoint indices are NOT sequential!")
        print(f"  Expected: {expected_indices[:10]}...")
        print(f"  Actual:   {actual_indices[:10]}...")

    # Check world poses
    poses_computed = sum(1 for vp in viewpoints if vp.world_pose is not None)
    print(f"World poses computed: {poses_computed}/{tour_length}")

    # Check IK solutions
    with_all_ik = sum(1 for vp in viewpoints if len(vp.all_ik_solutions) > 0)
    with_safe_ik = sum(1 for vp in viewpoints if len(vp.safe_ik_solutions) > 0)
    print(f"Viewpoints with all IK solutions: {with_all_ik}/{tour_length}")
    print(f"Viewpoints with safe IK solutions: {with_safe_ik}/{tour_length}")

    # Sample position check (first, middle, last)
    sample_indices = [0, tour_length // 2, tour_length - 1]
    print(f"\nSample position verification:")
    for idx in sample_indices:
        if idx < len(viewpoints):
            vp = viewpoints[idx]
            if vp.world_pose is not None:
                pos = vp.world_pose[:3, 3]
                print(f"  Viewpoint {idx}: position = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            else:
                print(f"  Viewpoint {idx}: No world pose")

    print(f"{'='*60}\n")


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


    # 카메라 추가
    tool_prim_path = robot_prim_path + "/tool0"
    camera_prim_path = tool_prim_path + "/mounted_camera"

    camera = Camera(
        prim_path=camera_prim_path,
        frequency=20,
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1, 0, 0, 0]),
        resolution=(256, 256),
    )
    
    # camera.set_focal_length(1.93)
    # camera.set_clipping_range(0.01, 10.0)
    
    camera.set_focal_length(38.0 / 1e3)
    camera.set_focus_distance(110.0 / 1e3)
    camera.set_horizontal_aperture(14.13 / 1e3)
    camera.set_vertical_aperture(10.35 / 1e3)
    camera.set_clipping_range(10/1e3, 100/1e3)
    camera.set_local_pose(np.array([0.0, 0.0, 0.0]), euler_angles_to_quats(np.array([0, 180, 0]), degrees=True), camera_axes="usd")
    my_world.scene.add(camera)

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

    return my_world, glass_prim, robot, idx_list, ik_solver, default_config


def load_tsp_tour(tsp_tour_path: str) -> dict:
    """
    Load TSP tour result from HDF5 file

    Args:
        tsp_tour_path: Path to TSP tour HDF5 file

    Returns:
        Dictionary containing TSP tour data
    """
    # Use the utility function from tsp_utils
    from tsp_utils import load_tsp_result

    return load_tsp_result(tsp_tour_path)


def load_tsp_tour_points_and_normals(tsp_result: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract points and normals from TSP tour result in tour visit order
    Also generates SAMPLED_VIEWPOINTS in TSP order

    IMPORTANT: Coordinate System Handling
    --------------------------------------
    The TSP tour file stores coordinates in Open3D coordinate system (Y-up).
    These coordinates will later be transformed to Isaac Sim coordinate system (Z-up)
    using the open3d_to_isaac_coords() function and the glass object's world transform.

    The transformation happens in convert_points() which is called from main() to
    visualize the sampled points. The same transformation is applied to viewpoint
    world poses via open3d_pose_to_world().

    Args:
        tsp_result: TSP tour result dictionary

    Returns:
        Tuple of (points, normals) in tour visit order (Open3D coordinate system)
    """
    global SAMPLED_LOCAL_POINTS, SAMPLED_LOCAL_NORMALS, SAMPLED_VIEWPOINTS

    # Use pre-ordered tour coordinates directly
    tour_coords = tsp_result["tour"]["coordinates"]
    tour_indices = tsp_result["tour"]["indices"]
    normals = tsp_result["normals"]

    # Get normals in tour order
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

    # Generate viewpoints in TSP order (same logic as load_mesh/load_pcd)
    # Apply offset along normals
    offset_points = offset_points_along_normals(
        tour_coords, tour_normals, NORMAL_SAMPLE_OFFSET
    )
    approach_normals = -normalize_vectors(tour_normals)

    # Create viewpoints with local poses
    helper_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    helper_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    sampled_viewpoints: List[Viewpoint] = []

    for point_idx, (position, normal) in enumerate(zip(offset_points, approach_normals)):
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

        # Use point_idx as index (TSP tour order: 0, 1, 2, ...)
        sampled_viewpoints.append(Viewpoint(index=int(point_idx), local_pose=pose_matrix))

    SAMPLED_LOCAL_POINTS = offset_points
    SAMPLED_LOCAL_NORMALS = approach_normals
    SAMPLED_VIEWPOINTS = sampled_viewpoints

    print(f"Generated {len(SAMPLED_VIEWPOINTS)} viewpoints in TSP order")

    return tour_coords, tour_normals


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


def visualize_tsp_tour_path(viewpoints: List[Viewpoint], glass_prim: XFormPrim) -> None:
    """
    Visualize TSP tour points colored by IK solution availability.

    Green: Viewpoint has safe IK solutions (collision-free)
    Red: Viewpoint has no safe IK solutions
    """
    if not viewpoints:
        return

    drawer = _debug_draw.acquire_debug_draw_interface()

    # Separate points by IK solution availability
    points_with_ik = []
    points_without_ik = []

    for vp in viewpoints:
        if vp.world_pose is not None:
            world_pos = tuple(vp.world_pose[:3, 3].astype(np.float32))

            if len(vp.safe_ik_solutions) > 0:
                points_with_ik.append(world_pos)
            else:
                points_without_ik.append(world_pos)

    # Draw points with IK solutions in green
    if points_with_ik:
        green_colors = [(0.0, 1.0, 0.0, 1.0)] * len(points_with_ik)
        green_sizes = [10.0] * len(points_with_ik)
        drawer.draw_points(points_with_ik, green_colors, green_sizes)

    # Draw points without IK solutions in red
    if points_without_ik:
        red_colors = [(1.0, 0.0, 0.0, 1.0)] * len(points_without_ik)
        red_sizes = [10.0] * len(points_without_ik)
        drawer.draw_points(points_without_ik, red_colors, red_sizes)

    print(f"\n{'='*60}")
    print("TSP TOUR POINT VISUALIZATION")
    print(f"{'='*60}")
    print(f"Total viewpoints: {len(viewpoints)}")
    print(f"Points with safe IK (green): {len(points_with_ik)}")
    print(f"Points without safe IK (red): {len(points_without_ik)}")
    print(f"Success rate: {len(points_with_ik)}/{len(viewpoints)} ({100*len(points_with_ik)/max(1,len(viewpoints)):.1f}%)")
    print(f"{'='*60}\n")


def plot_joint_trajectories(save_plot: bool = False, save_path: str = 'data/output/joint_trajectory.png') -> None:
    """
    Plot joint trajectories over time with viewpoint markers.

    Args:
        save_plot: If True, saves the plot to a PNG file. If False, skips plotting.
        save_path: Path to save the plot image (only used if save_plot is True)
    """
    if not save_plot:
        return

    if not JOINT_HISTORY['timestamps']:
        print("No joint data to plot")
        return

    timestamps = np.array(JOINT_HISTORY['timestamps'])
    joint_values = np.array(JOINT_HISTORY['joint_values'])  # Shape: (N, 6)
    viewpoint_markers = JOINT_HISTORY['viewpoint_markers']

    # Create figure
    plt.figure(figsize=(14, 8))

    # Joint names for UR20
    joint_names = ['Shoulder Pan', 'Shoulder Lift', 'Elbow', 'Wrist 1', 'Wrist 2', 'Wrist 3']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot each joint with different color
    for i in range(6):
        plt.plot(timestamps, joint_values[:, i],
                label=joint_names[i],
                color=colors[i],
                linewidth=2,
                alpha=0.8)

    # Add viewpoint markers as vertical lines
    for marker_time in viewpoint_markers:
        plt.axvline(x=marker_time, color='red', linestyle='--', alpha=0.3, linewidth=1)

    # Formatting
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Joint Value (radians)', fontsize=12)
    plt.title('Robot Joint Trajectories Over Time', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print("JOINT TRAJECTORY PLOT SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total time steps: {len(timestamps)}")
    print(f"Viewpoints reached: {len(viewpoint_markers)}")
    print(f"{'='*60}\n")


def extract_pose_from_matrix(pose_matrix: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """Extract position and quaternion from 4x4 transformation matrix.

    Args:
        pose_matrix: 4x4 transformation matrix

    Returns:
        Tuple of (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
    """
    if pose_matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4")

    # Extract position from translation component
    position = pose_matrix[:3, 3]
    pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])

    # Extract rotation matrix and convert to quaternion
    rotation_matrix = pose_matrix[:3, :3]
    # Convert to torch tensor for matrix_to_quaternion
    rot_tensor = torch.from_numpy(rotation_matrix.astype(np.float32)).unsqueeze(0)  # Add batch dimension
    quat_tensor = matrix_to_quaternion(rot_tensor)  # Returns (batch, 4) with (w, x, y, z) order

    # CuRobo returns quaternion in (w, x, y, z) order, but CSV needs (x, y, z, w)
    quat_w = float(quat_tensor[0, 0].item())
    quat_x = float(quat_tensor[0, 1].item())
    quat_y = float(quat_tensor[0, 2].item())
    quat_z = float(quat_tensor[0, 3].item())

    return pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w


def save_joint_trajectory_csv(
    viewpoints: List[Viewpoint],
    joint_targets: List[np.ndarray],
    save_path: str = 'data/output/joint_trajectory.csv'
) -> None:
    """Save joint trajectory and target poses to CSV file.

    Args:
        viewpoints: List of Viewpoint objects (TSP-ordered) with world_pose information
        joint_targets: List of joint configurations (TSP-ordered, matching viewpoints)
        save_path: Output CSV file path
    """
    if len(viewpoints) != len(joint_targets):
        print(f"Warning: Viewpoint count ({len(viewpoints)}) != joint target count ({len(joint_targets)})")
        # Use minimum length to avoid index errors
        count = min(len(viewpoints), len(joint_targets))
    else:
        count = len(viewpoints)

    # Prepare CSV data
    csv_rows = []
    skipped_count = 0

    for idx in range(count):
        viewpoint = viewpoints[idx]
        joint_config = joint_targets[idx]

        # Skip if world pose is not available
        if viewpoint.world_pose is None:
            print(f"Warning: Viewpoint {viewpoint.index} has no world_pose, skipping CSV row")
            skipped_count += 1
            continue

        # Time column (viewpoint index as float)
        time = float(idx)

        # Joint values (6 joints for UR20)
        joints = joint_config.flatten().tolist()
        if len(joints) != 6:
            print(f"Warning: Expected 6 joints but got {len(joints)} for viewpoint {idx}, skipping")
            skipped_count += 1
            continue

        # Extract target pose
        try:
            pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w = extract_pose_from_matrix(viewpoint.world_pose)
        except Exception as e:
            print(f"Warning: Failed to extract pose for viewpoint {idx}: {e}, skipping")
            skipped_count += 1
            continue

        # Create CSV row
        row = [time, *joints, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        csv_rows.append(row)

    # Write CSV file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
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

        # Write data rows
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print("JOINT TRAJECTORY CSV SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total rows written: {len(csv_rows)}")
    if skipped_count > 0:
        print(f"Rows skipped (missing data): {skipped_count}")
    print(f"{'='*60}\n")


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
    global TSP_TOUR_RESULT
    tensor_args = TensorDeviceType()

    print(f"\n{'='*60}")
    print("INITIALIZATION")
    print(f"{'='*60}")
    print(f"TSP tour path: {args.tsp_tour_path}")
    print(f"{'='*60}\n")

    # Load TSP tour (required)
    if args.tsp_tour_path is None:
        raise ValueError("TSP tour path is required. Use --tsp_tour_path to specify the HDF5 tour file.")

    print("Loading TSP tour...")
    TSP_TOUR_RESULT = load_tsp_tour(args.tsp_tour_path)
    o3d_points, o3d_normals = load_tsp_tour_points_and_normals(TSP_TOUR_RESULT)

    print(f"✓ Using TSP-optimized visit order from: {args.tsp_tour_path}")
    print(f"✓ Loaded {len(o3d_points)} points from TSP tour")

    if o3d_points.size == 0:
        raise ValueError("Error: No points loaded from TSP tour!")

    print(f"Loaded {len(o3d_points)} points with {len(o3d_normals)} normals")

    # Extract number of points for output directory structure
    num_points = TSP_TOUR_RESULT['metadata']['num_points']
    output_dir = f'data/output/{num_points}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    my_world, glass_prim, robot, idx_list, ik_solver, default_config = initialize_world()

    ik_joint_targets: List[np.ndarray] = []
    target_queue: Deque[np.ndarray] = deque()
    active_trajectory: List[np.ndarray] = []
    trajectory_step = 0

    # Process TSP tour viewpoints
    if SAMPLED_LOCAL_POINTS is None or SAMPLED_LOCAL_NORMALS is None:
        raise ValueError("Error: Sampled points/normals not available after loading TSP tour!")

    update_viewpoints_world_pose(SAMPLED_VIEWPOINTS, glass_prim)

    world_mats, used_indices = collect_viewpoint_world_matrices(SAMPLED_VIEWPOINTS)
    if world_mats.size == 0:
        raise ValueError("Error: No valid world poses found for sampled viewpoints")

    ik_results = compute_ik(
        mats=world_mats,
        urdf_path="/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf"
    )

    assign_start = perf_counter()
    assign_eaik_solutions(SAMPLED_VIEWPOINTS, ik_results, used_indices)
    assign_elapsed = perf_counter() - assign_start

    safe_start = perf_counter()
    update_safe_ik_solutions(SAMPLED_VIEWPOINTS, ik_solver)
    safe_elapsed = perf_counter() - safe_start

    print(
        f"assign_eaik_solutions duration: {assign_elapsed * 1000.0:.2f} ms"
    )
    print(
        f"update_safe_ik_solutions duration: {safe_elapsed * 1000.0:.2f} ms"
    )
    log_viewpoint_ik_stats(SAMPLED_VIEWPOINTS)

    # Verify TSP tour mapping
    verify_tsp_tour_mapping(SAMPLED_VIEWPOINTS, TSP_TOUR_RESULT)

    # Collect joint targets in TSP order
    print(f"\nNumber of viewpoints = {len(SAMPLED_VIEWPOINTS)}")
    if len(SAMPLED_VIEWPOINTS) > 0:
        print(f"  First viewpoint index = {SAMPLED_VIEWPOINTS[0].index}")
        print(f"  Last viewpoint index = {SAMPLED_VIEWPOINTS[-1].index}")

    # Define cost function parameters
    # Joint weights: higher weights for base joints (1, 2, 3)
    joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    # Reconfiguration penalty parameters (for DP method)
    reconfig_threshold = 1.0  # radians (~17 degrees)
    reconfig_penalty = 10.0   # penalty per reconfigured joint
    max_move_weight = 5.0     # weight for max joint movement

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Select IK solution method based on command line argument
    selection_method = args.selection_method
    print(f"\n{'='*60}")
    print(f"USING SELECTION METHOD: {selection_method.upper()}")
    print(f"{'='*60}\n")

    # Track selected solution indices for HDF5 save
    selected_solution_indices = None

    if selection_method == "random":
        # First solution selection (baseline method)
        ik_joint_targets = collect_random_joint_targets(SAMPLED_VIEWPOINTS)
        # Random/First always selects index 0 for each viewpoint with safe solutions
        viewpoints_with_safe = [vp for vp in SAMPLED_VIEWPOINTS if len(vp.safe_ik_solutions) > 0]
        selected_solution_indices = [0] * len(viewpoints_with_safe)

    elif selection_method == "greedy":
        # Greedy nearest neighbor selection
        ik_joint_targets = collect_sorted_joint_safe_targets(SAMPLED_VIEWPOINTS)
        # Greedy needs to track which solution index was selected
        # For now, we'll compute this by matching the selected joint configs
        viewpoints_with_safe = [vp for vp in SAMPLED_VIEWPOINTS if len(vp.safe_ik_solutions) > 0]
        selected_solution_indices = []
        for vp, target in zip(viewpoints_with_safe, ik_joint_targets):
            # Find which safe solution matches the target
            for idx, safe_sol in enumerate(vp.safe_ik_solutions):
                if np.allclose(safe_sol, target, atol=1e-6):
                    selected_solution_indices.append(idx)
                    break

    elif selection_method == "dp":
        # Dynamic programming selection
        ik_joint_targets, total_cost, solution_indices, valid_indices = collect_optimal_joint_targets_dp(
            SAMPLED_VIEWPOINTS,
            initial_joint_config=default_config,
            joint_weights=joint_weights,
            reconfig_threshold=reconfig_threshold,
            reconfig_penalty=reconfig_penalty,
            max_move_weight=max_move_weight
        )
        print(f"DP optimization complete. Total cost: {total_cost:.4f}")
        selected_solution_indices = solution_indices
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    # Setup target queue for robot trajectory
    target_queue.clear()
    target_queue.extend(ik_joint_targets)

    print(f"✓ Robot will visit {len(ik_joint_targets)} points using {selection_method.upper()} joint selection")
    print(f"  Expected order: viewpoint indices 0 → 1 → 2 → ... → {len(ik_joint_targets)-1}")

    # Generate CSV file with joint trajectories and target poses
    # Note: We need to filter viewpoints to only those with safe IK solutions
    viewpoints_with_safe_ik = [vp for vp in SAMPLED_VIEWPOINTS if len(vp.safe_ik_solutions) > 0]
    csv_save_path = f'{output_dir}/joint_trajectory_{selection_method}_{timestamp}.csv'
    save_joint_trajectory_csv(
        viewpoints=viewpoints_with_safe_ik,
        joint_targets=ik_joint_targets,
        save_path=csv_save_path
    )

    # Automatically analyze joint reconfigurations from the saved CSV
    analyze_and_save_reconfigurations(
        csv_path=csv_save_path,
        threshold=reconfig_threshold,  # Use same threshold as DP cost function
        output_dir=output_dir
    )

    # Save all IK solutions to HDF5
    save_ik_solutions_hdf5(
        viewpoints=SAMPLED_VIEWPOINTS,
        selected_indices=selected_solution_indices,
        selection_method=selection_method,
        save_path=f'{output_dir}/ik_solutions_all_{timestamp}.h5',
        tsp_tour_path=args.tsp_tour_path
    )

    if args.no_sim:
        simulation_app.close()
        return

    # Visualize TSP tour path in Isaac Sim
    # visualize_tsp_tour_path(SAMPLED_VIEWPOINTS, glass_prim)


    step_counter = 0
    idle_counter = 0
    viewpoint_counter = 0  # Track number of viewpoints reached
    plot_interval = 5  # Update plot every N viewpoints

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

        # Collect current joint values
        current_joints = get_active_joint_positions(robot, idx_list)
        JOINT_HISTORY['timestamps'].append(step_counter)
        JOINT_HISTORY['joint_values'].append(current_joints)

        if active_trajectory and trajectory_step < len(active_trajectory):
            joint_cmd = active_trajectory[trajectory_step]
            robot.set_joint_positions(joint_cmd.tolist(), idx_list)
            trajectory_step += 1
            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0
        elif target_queue:
            # Mark viewpoint reached
            viewpoint_counter += 1
            JOINT_HISTORY['viewpoint_markers'].append(step_counter)

            # Update plot periodically
            if viewpoint_counter % plot_interval == 0:
                plot_joint_trajectories(save_plot=args.save_plot, save_path=f'{output_dir}/joint_trajectory.png')

            next_target = target_queue.popleft()
            current_state = get_active_joint_positions(robot, idx_list)
            active_trajectory = generate_interpolated_joint_path(
                current_state,
                next_target,
                num_steps=INTERPOLATION_STEPS,
            )
            trajectory_step = 0

            if not active_trajectory:
                active_trajectory = [next_target]

            joint_cmd = active_trajectory[trajectory_step]
            robot.set_joint_positions(joint_cmd.tolist(), idx_list)
            trajectory_step += 1
            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0


        # Update simulation step
        # my_world.current_time_step_index

    # Final plot after simulation ends
    if JOINT_HISTORY['timestamps']:
        plot_joint_trajectories(save_plot=args.save_plot, save_path=f'{output_dir}/joint_trajectory_final.png')

    simulation_app.close()


if __name__ == "__main__":
    main()
