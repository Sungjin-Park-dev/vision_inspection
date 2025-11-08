"""
IK (Inverse Kinematics) computation and collision checking utilities

This module provides functions for:
- Computing IK solutions using EAIK (analytical IK solver)
- Assigning IK solutions to viewpoints
- Checking collision constraints on IK solutions
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState


# ============================================================================
# Constants
# ============================================================================
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
# Data Classes
# ============================================================================
@dataclass
class Viewpoint:
    """Represents a camera viewpoint with IK solutions"""
    index: int
    local_pose: Optional[np.ndarray] = None  # 4x4 pose in object frame
    world_pose: Optional[np.ndarray] = None  # 4x4 pose in world frame
    all_ik_solutions: List[np.ndarray] = field(default_factory=list)
    safe_ik_solutions: List[np.ndarray] = field(default_factory=list)


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
    # Import here to avoid circular dependency
    from eaik.IK_URDF import UrdfRobot

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
