"""
Trajectory planning utilities for joint configuration selection

This module provides functions for:
- Computing weighted joint distance and reconfiguration costs
- Selecting IK solutions using different strategies (random, greedy, DP)
"""

from typing import List, Optional, Tuple

import numpy as np

from .ik_utils import Viewpoint
from . import config


# ============================================================================
# Cost Functions
# ============================================================================
def compute_weighted_joint_distance(
    q1: np.ndarray,
    q2: np.ndarray,
    joint_weights: Optional[np.ndarray] = None
) -> float:
    """Compute weighted Euclidean distance between two joint configurations"""
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 0.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    weighted_diff = joint_weights * (q1_array - q2_array) ** 2
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
    Compute reconfiguration-aware cost that penalizes large joint movements

    Returns:
        Tuple of (total_cost, cost_breakdown_dict)
    """
    if joint_weights is None:
        joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    q1_array = np.asarray(q1, dtype=np.float64)
    q2_array = np.asarray(q2, dtype=np.float64)

    # Base cost: weighted Euclidean distance
    base_cost = compute_weighted_joint_distance(q1_array, q2_array, joint_weights)

    # Reconfiguration penalty
    joint_deltas = np.abs(q1_array - q2_array)
    reconfig_count = np.sum(joint_deltas > reconfig_threshold)
    reconfig_cost = reconfig_count * reconfig_penalty

    # Max joint movement penalty
    max_joint_move = np.max(joint_deltas)
    max_move_cost = max_joint_move * max_move_weight

    total_cost = base_cost  # Currently using only base cost

    cost_breakdown = {
        'base_cost': base_cost,
        'reconfig_count': int(reconfig_count),
        'reconfig_cost': reconfig_cost,
        'max_joint_move': max_joint_move,
        'max_move_cost': max_move_cost,
        'total_cost': total_cost
    }

    return total_cost, cost_breakdown


# ============================================================================
# IK Solution Selection Strategies
# ============================================================================
def select_ik_random(viewpoints: List[Viewpoint]) -> Tuple[List[np.ndarray], List[int]]:
    """Select first safe IK solution for each viewpoint (RANDOM/FIRST method)"""
    print(f"\nSelecting joint targets (RANDOM/FIRST method):")

    targets: List[np.ndarray] = []
    solution_indices: List[int] = []

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} has no safe IK solutions, skipping")
            continue

        solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)
        targets.append(solution)
        solution_indices.append(0)

    print(f"  → Collected {len(targets)} targets")
    return targets, solution_indices


def select_ik_greedy(viewpoints: List[Viewpoint]) -> Tuple[List[np.ndarray], List[int]]:
    """Select IK solutions using greedy nearest neighbor (GREEDY method)"""
    print(f"\nSelecting joint targets (GREEDY method):")

    targets: List[np.ndarray] = []
    solution_indices: List[int] = []
    previous_joints = None

    for idx, viewpoint in enumerate(viewpoints):
        if not viewpoint.safe_ik_solutions:
            print(f"  Warning: Viewpoint {viewpoint.index} has no safe IK solutions, skipping")
            continue

        if previous_joints is None:
            # First viewpoint: use first solution
            solution = np.asarray(viewpoint.safe_ik_solutions[0], dtype=np.float64)
            selected_idx = 0
        else:
            # Find closest solution to previous configuration
            best_solution = None
            min_distance = float('inf')
            selected_idx = 0

            for sol_idx, candidate in enumerate(viewpoint.safe_ik_solutions):
                candidate_joints = np.asarray(candidate, dtype=np.float64)
                distance = np.linalg.norm(candidate_joints - previous_joints)
                if distance < min_distance:
                    min_distance = distance
                    best_solution = candidate_joints
                    selected_idx = sol_idx

            solution = best_solution

        targets.append(solution)
        solution_indices.append(selected_idx)
        previous_joints = solution

    print(f"  → Collected {len(targets)} targets")
    return targets, solution_indices


def select_ik_dp(
    viewpoints: List[Viewpoint],
    initial_config: np.ndarray,
    joint_weights: Optional[np.ndarray] = None,
    reconfig_threshold: float = None,
    reconfig_penalty: float = None,
    max_move_weight: float = None
) -> Tuple[List[np.ndarray], float, List[int]]:
    """Select IK solutions using Dynamic Programming (DP method)

    Args:
        viewpoints: List of viewpoints with safe IK solutions
        initial_config: Initial joint configuration
        joint_weights: Weights for joint distance calculation (default: from config)
        reconfig_threshold: Threshold for reconfiguration detection (default: from config)
        reconfig_penalty: Penalty for reconfigurations (default: from config)
        max_move_weight: Weight for max joint movement (default: from config)

    Returns:
        Tuple of (joint_targets, total_cost, solution_indices)
    """
    print(f"\nSelecting joint targets (DP method):")

    # Use default values from config if not provided
    if joint_weights is None:
        joint_weights = config.JOINT_WEIGHTS.copy()
    if reconfig_threshold is None:
        reconfig_threshold = config.RECONFIGURATION_THRESHOLD
    if reconfig_penalty is None:
        reconfig_penalty = config.RECONFIGURATION_PENALTY
    if max_move_weight is None:
        max_move_weight = config.MAX_MOVE_WEIGHT

    initial_joints = np.asarray(initial_config, dtype=np.float64)

    # Filter viewpoints with safe solutions
    valid_viewpoints = [vp for vp in viewpoints if vp.safe_ik_solutions]
    n_viewpoints = len(valid_viewpoints)

    if not valid_viewpoints:
        print("  Error: No valid viewpoints with safe IK solutions!")
        return [], 0.0, []

    print(f"  Processing {n_viewpoints} viewpoints")
    print(f"  Joint weights: {joint_weights}")
    print(f"  Reconfiguration threshold: {reconfig_threshold} rad")

    # DP table: dp[i][sol_idx] = (min_cost, prev_solution_idx)
    dp = [dict() for _ in range(n_viewpoints)]

    # Initialize first viewpoint
    first_vp = valid_viewpoints[0]
    for sol_idx, solution in enumerate(first_vp.safe_ik_solutions):
        sol_array = np.asarray(solution, dtype=np.float64)
        init_cost, _ = compute_reconfiguration_cost(
            sol_array, initial_joints, joint_weights,
            reconfig_threshold, reconfig_penalty, max_move_weight
        )
        dp[0][sol_idx] = (init_cost, -1)

    # Forward pass
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

            for prev_sol_idx, (prev_cost, _) in prev_dp.items():
                prev_solution = valid_viewpoints[i - 1].safe_ik_solutions[prev_sol_idx]
                prev_joints = np.asarray(prev_solution, dtype=np.float64)

                transition_cost, _ = compute_reconfiguration_cost(
                    curr_joints, prev_joints, joint_weights,
                    reconfig_threshold, reconfig_penalty, max_move_weight
                )
                total_cost = prev_cost + transition_cost

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_prev_idx = prev_sol_idx

            dp[i][curr_sol_idx] = (min_cost, best_prev_idx)

    # Find best final solution
    last_dp = dp[n_viewpoints - 1]
    if not last_dp:
        print("  Error: No valid complete path found!")
        return [], 0.0, []

    best_final_idx = min(last_dp.keys(), key=lambda idx: last_dp[idx][0])
    total_distance = last_dp[best_final_idx][0]

    # Backward pass: reconstruct path
    solution_indices = [0] * n_viewpoints
    solution_indices[n_viewpoints - 1] = best_final_idx

    for i in range(n_viewpoints - 1, 0, -1):
        current_sol_idx = solution_indices[i]
        _, prev_sol_idx = dp[i][current_sol_idx]
        solution_indices[i - 1] = prev_sol_idx

    # Build target list
    targets = []
    for i, sol_idx in enumerate(solution_indices):
        solution = valid_viewpoints[i].safe_ik_solutions[sol_idx]
        targets.append(np.asarray(solution, dtype=np.float64))

    print(f"  → Collected {len(targets)} targets with total cost {total_distance:.4f}")
    return targets, total_distance, solution_indices
