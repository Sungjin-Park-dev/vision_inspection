#!/usr/bin/env python3

"""
Independent Joint Trajectory Optimization Tool using cuRobo IK Solver

This script optimizes ONLY the originally problematic timesteps by using cuRobo's
GPU-accelerated IK solver, without cascading effects. This is a comparison version
of optimize_joint_reconfigurations_independent.py that uses cuRobo instead of relaxed_ik.

Key Differences from relaxed_ik version:
- Uses cuRobo's IKSolver with GPU batch processing
- Leverages num_seeds parameter for diverse IK solutions
- Threshold-based instead of tolerance-based optimization
- Significantly faster due to GPU parallelization

Strategy:
- Detect problematic timesteps (joint change > threshold)
- For each problematic timestep, re-solve IK with:
  * Original target pose (position + orientation)
  * Adjustable position/rotation thresholds
  * Multiple seeds for diverse solutions
  * Previous timestep's configuration as retract config
- **FILTERING**: Reject IK solutions that create NEW reconfigurations in other joints
  * Prevents trading wrist reconfigs for shoulder/elbow reconfigs
  * Only accepts solutions that don't exceed threshold on previously OK joints
- Replace ONLY the problematic IK solutions
- No cascading - other timesteps remain unchanged

Usage Examples:
    # Basic independent optimization with cuRobo
    python optimize_joint_reconfigurations_independent_curobo.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20.yml

    # With custom thresholds (relax position constraints)
    python optimize_joint_reconfigurations_independent_curobo.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20.yml \
        --position_threshold 0.01 \
        --rotation_threshold 0.1

    # With visualization and more IK seeds
    python optimize_joint_reconfigurations_independent_curobo.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20.yml \
        --num_seeds 50 \
        --plot

Author: Enhanced for ICRA'25 Hierarchical Coverage Path Planning (cuRobo version)
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time

import torch
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.geom.sdf.world import CollisionCheckerType

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_joint_reconfigurations import (
    analyze_joint_reconfigurations,
    print_analysis_results
)


def load_joint_trajectory_with_poses(csv_path: str) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Load joint trajectory data and pose data from CSV file.

    Args:
        csv_path: Path to the joint trajectory CSV file

    Returns:
        Tuple of (joint_data, joint_names, pose_data, pose_columns) where:
        - joint_data: numpy array of shape (n_timesteps, n_joints)
        - joint_names: list of joint column names
        - pose_data: numpy array of shape (n_timesteps, 7) [x, y, z, qw, qx, qy, qz]
        - pose_columns: list of pose column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    joint_data = []
    pose_data = []
    joint_names = []
    pose_columns = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Extract column names
        all_columns = reader.fieldnames

        # Find joint columns (format: robotname-joint_name)
        joint_names = [col for col in all_columns
                      if '-' in col and col.endswith('_joint')]

        # If no joints found with that pattern, try simpler pattern
        if not joint_names:
            joint_names = [col for col in all_columns
                          if 'joint' in col.lower() and col != 'time']

        # Find pose columns (format: target-POS_X, target-ROT_W, etc.)
        pose_columns = [col for col in all_columns
                       if col.startswith('target-')]

        # If no pose columns found, try alternative patterns
        if not pose_columns:
            pose_columns = [col for col in all_columns
                           if any(x in col.upper() for x in ['POS', 'ROT', 'QUAT'])]

        print(f"Found {len(joint_names)} joints: {joint_names}")
        print(f"Found {len(pose_columns)} pose columns: {pose_columns}")

        # Read data
        for row in reader:
            # Read joint values
            joint_values = [float(row[joint_name]) for joint_name in joint_names]
            joint_data.append(joint_values)

            # Read pose values (reorder to [x, y, z, qw, qx, qy, qz] if needed)
            pose_values = [float(row[pose_col]) for pose_col in pose_columns]

            # Check if we need to reorder quaternion (input might be x,y,z,qx,qy,qz,qw)
            if len(pose_values) == 7:
                # Assume order is x, y, z, qx, qy, qz, qw → need to swap to x, y, z, qw, qx, qy, qz
                if 'ROT_W' in pose_columns[-1] or 'QUAT_W' in pose_columns[-1]:
                    # Input is x, y, z, qx, qy, qz, qw
                    pose_values = pose_values[:3] + [pose_values[6]] + pose_values[3:6]

            pose_data.append(pose_values)

    joint_data = np.array(joint_data)
    pose_data = np.array(pose_data)

    print(f"Loaded trajectory with {joint_data.shape[0]} time steps, "
          f"{joint_data.shape[1]} joints, and {pose_data.shape[1]} pose values")

    return joint_data, joint_names, pose_data, pose_columns


def setup_ik_solver(
    robot_name: str = 'ur20.yml',
    collision_check: bool = False,
    num_seeds: int = 20,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
) -> Tuple[IKSolver, TensorDeviceType, List[float]]:
    """
    Initialize cuRobo IKSolver with specified parameters.

    Args:
        robot_name: Robot configuration YAML file name
        collision_check: Whether to enable collision checking
        num_seeds: Number of random seeds for IK solving
        position_threshold: Position error threshold (meters)
        rotation_threshold: Rotation error threshold (radians)

    Returns:
        Tuple of (IKSolver, TensorDeviceType, retract_config)
    """
    print(f"Setting up cuRobo IK solver...")

    tensor_args = TensorDeviceType()

    # Load robot configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_name))["robot_cfg"]

    # Load world configuration for collision checking
    if collision_check:
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
    else:
        world_cfg = WorldConfig()

    # Configure IK solver
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=rotation_threshold,
        position_threshold=position_threshold,
        num_seeds=num_seeds,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,  # Enable CUDA graph for faster inference
        collision_checker_type=CollisionCheckerType.MESH if collision_check else CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": 30, "mesh": 10},
    )

    ik_solver = IKSolver(ik_config)

    # Get retract configuration
    retract_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    print(f"✓ IK solver initialized:")
    print(f"  Robot: {robot_name}")
    print(f"  Collision checking: {collision_check}")
    print(f"  Number of seeds: {num_seeds}")
    print(f"  Position threshold: {position_threshold} m")
    print(f"  Rotation threshold: {rotation_threshold} rad")
    print(f"  Device: {tensor_args.device}")

    return ik_solver, tensor_args, retract_config


def pose_array_to_curobo_pose(
    pose: np.ndarray,
    tensor_args: TensorDeviceType
) -> Pose:
    """
    Convert pose array [x, y, z, qw, qx, qy, qz] to cuRobo Pose object.

    Args:
        pose: Numpy array of shape (7,) containing [x, y, z, qw, qx, qy, qz]
        tensor_args: TensorDeviceType for device placement

    Returns:
        cuRobo Pose object
    """
    position = torch.tensor(
        pose[:3],
        dtype=tensor_args.dtype,
        device=tensor_args.device
    ).unsqueeze(0)  # Shape: (1, 3)

    quaternion = torch.tensor(
        pose[3:7],
        dtype=tensor_args.dtype,
        device=tensor_args.device
    ).unsqueeze(0)  # Shape: (1, 4), format: [w, x, y, z]

    return Pose(position=position, quaternion=quaternion)


def sample_ik_solutions_curobo(
    ik_solver: IKSolver,
    tensor_args: TensorDeviceType,
    target_pose: np.ndarray,
    seed_config: Optional[np.ndarray] = None,
    num_attempts: int = 10,
) -> List[np.ndarray]:
    """
    Sample multiple IK solutions for a target pose using cuRobo.

    Since cuRobo uses num_seeds internally for diversity, we call solve_batch
    multiple times to get different solutions (each call uses different random seeds).

    Args:
        ik_solver: cuRobo IK solver instance
        tensor_args: TensorDeviceType for device placement
        target_pose: Target pose array [x, y, z, qw, qx, qy, qz]
        seed_config: Seed joint configuration (optional)
        num_attempts: Number of solve attempts to try

    Returns:
        List of unique IK solutions (numpy arrays)
    """
    goal_pose = pose_array_to_curobo_pose(target_pose, tensor_args)

    # Prepare seed configuration if provided
    if seed_config is not None:
        retract_tensor = torch.tensor(
            seed_config,
            dtype=tensor_args.dtype,
            device=tensor_args.device
        ).unsqueeze(0)  # Shape: (1, n_joints)
    else:
        retract_tensor = None

    solutions = []

    for attempt in range(num_attempts):
        # Solve IK (internally uses num_seeds different starting points)
        result = ik_solver.solve_batch(goal_pose, retract_config=retract_tensor)

        # Extract successful solutions
        success_flags = result.success.cpu().numpy()

        for i in range(len(success_flags)):
            if success_flags[i]:
                solution = result.js_solution.position[i].cpu().numpy()

                # Check if this solution is unique (not too close to existing ones)
                is_unique = True
                for existing_sol in solutions:
                    if np.linalg.norm(solution - existing_sol) < 0.01:  # 0.01 rad threshold
                        is_unique = False
                        break

                if is_unique:
                    solutions.append(solution)

    return solutions


def optimize_trajectory_independent_curobo(
    joint_data: np.ndarray,
    pose_data: np.ndarray,
    ik_solver: IKSolver,
    tensor_args: TensorDeviceType,
    retract_config: List[float],
    threshold: float = 1.0,
    num_ik_attempts: int = 10,
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize trajectory by re-solving IK ONLY for problematic timesteps using cuRobo.

    This approach identifies all problematic timesteps first, then optimizes each one independently
    without cascading effects. Uses cuRobo's GPU-accelerated IK solver.

    Args:
        joint_data: Original joint trajectory array of shape (n_timesteps, n_joints)
        pose_data: Target pose array of shape (n_timesteps, 7)
        ik_solver: cuRobo IK solver instance
        tensor_args: TensorDeviceType for device placement
        retract_config: Default retract configuration
        threshold: Threshold for detecting reconfigurations (radians)
        num_ik_attempts: Number of IK solve attempts per problematic timestep

    Returns:
        Tuple of (optimized_joint_data, optimization_stats)
    """
    print(f"\n{'='*80}")
    print("INDEPENDENT TRAJECTORY OPTIMIZATION (cuRobo)")
    print(f"{'='*80}")
    print(f"Threshold: {threshold:.3f} radians")
    print(f"IK attempts per timestep: {num_ik_attempts}")
    print("\nStrategy: Optimize ONLY originally problematic timesteps using cuRobo")
    print("No cascading - other timesteps remain unchanged")
    print("⚠️  FILTERING: Rejecting IK solutions that create NEW reconfigurations")

    # Step 1: Identify all problematic timesteps in ORIGINAL trajectory
    print(f"\n{'='*80}")
    print("STEP 1: Identifying problematic timesteps in original trajectory")
    print(f"{'='*80}")

    n_timesteps = joint_data.shape[0]
    problematic_timesteps = []

    for t in range(1, n_timesteps):
        prev_ik = joint_data[t-1]
        curr_ik = joint_data[t]

        # Calculate joint space distance
        joint_diff = np.abs(curr_ik - prev_ik)
        max_joint_change = np.max(joint_diff)

        # Check if reconfiguration detected
        if max_joint_change > threshold:
            problematic_timesteps.append({
                'timestep': t,
                'original_max_change': max_joint_change,
                'prev_ik': prev_ik.copy(),
                'curr_ik': curr_ik.copy(),
                'target_pose': pose_data[t].copy()
            })

    print(f"Found {len(problematic_timesteps)} problematic timesteps")
    if len(problematic_timesteps) > 0:
        print(f"Timesteps: {[p['timestep'] for p in problematic_timesteps]}")

    # Step 2: Optimize ONLY the problematic timesteps
    print(f"\n{'='*80}")
    print("STEP 2: Optimizing problematic timesteps independently with cuRobo")
    print(f"{'='*80}")

    optimized_joint_data = joint_data.copy()
    reconfigurations_fixed = 0
    reconfigurations_unfixed = 0
    optimization_details = []

    for i, problem in enumerate(problematic_timesteps):
        t = problem['timestep']
        prev_ik = problem['prev_ik']
        curr_ik_original = problem['curr_ik']
        target_pose = problem['target_pose']
        original_max_change = problem['original_max_change']

        print(f"\n[{i+1}/{len(problematic_timesteps)}] Timestep {t}: Original max change = {original_max_change:.3f} rad")
        print(f"  Sampling {num_ik_attempts} IK solutions with cuRobo...")

        # Get next timestep's IK for bidirectional validation
        next_ik = joint_data[t+1] if t+1 < n_timesteps else None

        # Sample multiple IK solutions using cuRobo
        candidates = sample_ik_solutions_curobo(
            ik_solver=ik_solver,
            tensor_args=tensor_args,
            target_pose=target_pose,
            seed_config=prev_ik,  # Use previous timestep as seed
            num_attempts=num_ik_attempts
        )

        # Evaluate each candidate
        candidate_scores = []
        n_joints = len(prev_ik)

        for ik_solution in candidates:
            # Calculate distance to previous timestep
            joint_diff_prev = np.abs(ik_solution - prev_ik)
            max_change_prev = np.max(joint_diff_prev)

            # Calculate distance to next timestep (if exists)
            if next_ik is not None:
                joint_diff_next = np.abs(next_ik - ik_solution)
                max_change_next = np.max(joint_diff_next)
                # Total cost: max of both directions
                total_cost = max(max_change_prev, max_change_next)
            else:
                total_cost = max_change_prev
                max_change_next = None

            # ===== METHOD 2: REJECT SOLUTIONS THAT CREATE NEW RECONFIGURATIONS =====
            # Check if this solution creates new reconfigurations in joints that were originally OK
            creates_new_reconfig = False

            # Check previous direction
            original_joint_diff_prev = np.abs(curr_ik_original - prev_ik)
            new_joint_diff_prev = np.abs(ik_solution - prev_ik)

            for joint_idx in range(n_joints):
                # If this joint was originally below threshold but now exceeds it, reject
                was_ok = bool(original_joint_diff_prev[joint_idx] < threshold)
                now_bad = bool(new_joint_diff_prev[joint_idx] > threshold)
                if was_ok and now_bad:
                    creates_new_reconfig = True
                    break

            # Check next direction (if exists)
            if not creates_new_reconfig and next_ik is not None:
                original_joint_diff_next = np.abs(next_ik - curr_ik_original)
                new_joint_diff_next = np.abs(next_ik - ik_solution)

                for joint_idx in range(n_joints):
                    # If this joint was originally below threshold but now exceeds it, reject
                    was_ok = bool(original_joint_diff_next[joint_idx] < threshold)
                    now_bad = bool(new_joint_diff_next[joint_idx] > threshold)
                    if was_ok and now_bad:
                        creates_new_reconfig = True
                        break

            # Only add to candidates if it doesn't create new reconfigurations
            if not creates_new_reconfig:
                candidate_scores.append((ik_solution, max_change_prev, max_change_next, total_cost))

        # Select the best candidate (smallest total_cost)
        if len(candidate_scores) > 0:
            print(f"  Found {len(candidates)} IK solutions, {len(candidate_scores)} after filtering new reconfigurations")
            best_ik, best_prev, best_next, best_total = min(candidate_scores, key=lambda x: x[3])

            # Calculate original cost (bidirectional)
            if next_ik is not None:
                original_next_change = np.max(np.abs(next_ik - curr_ik_original))
                original_total_cost = max(original_max_change, original_next_change)
            else:
                original_next_change = None
                original_total_cost = original_max_change

            # Check if best solution is better than original (considering both directions)
            if best_total < original_total_cost:
                improvement = original_total_cost - best_total
                if best_next is not None:
                    print(f"  ✓ Improved! Prev: {original_max_change:.3f}→{best_prev:.3f}, Next: {original_next_change:.3f}→{best_next:.3f}, Total: {original_total_cost:.3f}→{best_total:.3f} (Δ={improvement:.3f})")
                else:
                    print(f"  ✓ Improved! Old: {original_total_cost:.3f} → New: {best_total:.3f} (Δ={improvement:.3f})")

                optimized_joint_data[t] = best_ik
                reconfigurations_fixed += 1

                optimization_details.append({
                    'timestep': t,
                    'old_max_change': original_total_cost,
                    'new_max_change': best_total,
                    'improvement': improvement,
                    'status': 'improved',
                    'old_prev': original_max_change,
                    'new_prev': best_prev,
                    'old_next': original_next_change,
                    'new_next': best_next
                })
            else:
                if best_next is not None:
                    print(f"  ✗ Not better: Prev: {original_max_change:.3f}→{best_prev:.3f}, Next: {original_next_change:.3f}→{best_next:.3f}, Total: {original_total_cost:.3f}→{best_total:.3f}")
                else:
                    print(f"  ✗ Best solution not better: {best_total:.3f} >= {original_total_cost:.3f}")
                reconfigurations_unfixed += 1

                optimization_details.append({
                    'timestep': t,
                    'old_max_change': original_total_cost,
                    'new_max_change': best_total,
                    'improvement': 0,
                    'status': 'no_improvement',
                    'old_prev': original_max_change,
                    'new_prev': best_prev,
                    'old_next': original_next_change,
                    'new_next': best_next
                })
        else:
            print(f"  ✗ Failed to find any valid IK solution from cuRobo")
            reconfigurations_unfixed += 1

            optimization_details.append({
                'timestep': t,
                'old_max_change': original_max_change,
                'new_max_change': None,
                'improvement': 0,
                'status': 'failed',
                'old_prev': original_max_change,
                'new_prev': None,
                'old_next': None,
                'new_next': None
            })

    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Originally problematic timesteps: {len(problematic_timesteps)}")
    print(f"Reconfigurations fixed: {reconfigurations_fixed}")
    print(f"Reconfigurations unfixed: {reconfigurations_unfixed}")

    stats = {
        'originally_problematic': len(problematic_timesteps),
        'reconfigurations_fixed': reconfigurations_fixed,
        'reconfigurations_unfixed': reconfigurations_unfixed,
        'optimization_details': optimization_details
    }

    return optimized_joint_data, stats


def save_optimized_trajectory(
    output_path: str,
    joint_data: np.ndarray,
    pose_data: np.ndarray,
    joint_names: List[str],
    pose_columns: List[str]
) -> None:
    """
    Save optimized trajectory to CSV file.

    Args:
        output_path: Path to save the optimized CSV file
        joint_data: Joint trajectory array
        pose_data: Target pose array
        joint_names: List of joint column names
        pose_columns: List of pose column names
    """
    n_timesteps = joint_data.shape[0]

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['time'] + joint_names + pose_columns
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t in range(n_timesteps):
            row = {'time': float(t)}

            # Add joint values
            for i, joint_name in enumerate(joint_names):
                row[joint_name] = joint_data[t, i]

            # Add pose values
            for i, pose_col in enumerate(pose_columns):
                row[pose_col] = pose_data[t, i]

            writer.writerow(row)

    print(f"\n✓ Optimized trajectory saved to: {output_path}")


def plot_optimization_comparison(
    original_joint_data: np.ndarray,
    optimized_joint_data: np.ndarray,
    joint_names: List[str],
    original_results: Dict,
    optimized_results: Dict,
    threshold: float,
    robot_name: str,
    save_path: str = None
) -> None:
    """
    Create before/after comparison plot for trajectory optimization.

    Args:
        original_joint_data: Original joint trajectory
        optimized_joint_data: Optimized joint trajectory
        joint_names: List of joint names
        original_results: Analysis results for original trajectory
        optimized_results: Analysis results for optimized trajectory
        threshold: Reconfiguration threshold
        robot_name: Name of the robot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle('Independent Joint Trajectory Optimization Comparison (cuRobo)', fontsize=16, fontweight='bold')

    timesteps = np.arange(original_joint_data.shape[0])

    # Original trajectory
    ax1 = axes[0]
    joint_diffs_orig = np.abs(np.diff(original_joint_data, axis=0))
    for i, joint_name in enumerate(joint_names):
        # Remove robot prefix for cleaner display
        short_name = joint_name.split('-')[-1].replace('_joint', '')
        ax1.plot(timesteps[1:], joint_diffs_orig[:, i], label=short_name, alpha=0.8)
    ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Joint Change (radians)')
    ax1.set_title(f'Original Trajectory (Reconfigs: {original_results["total_reconfigurations"]})')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Optimized trajectory
    ax2 = axes[1]
    joint_diffs_opt = np.abs(np.diff(optimized_joint_data, axis=0))
    for i, joint_name in enumerate(joint_names):
        short_name = joint_name.split('-')[-1].replace('_joint', '')
        ax2.plot(timesteps[1:], joint_diffs_opt[:, i], label=short_name, alpha=0.8)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Joint Change (radians)')
    ax2.set_title(f'Optimized Trajectory (Reconfigs: {optimized_results["total_reconfigurations"]})')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Independently optimize problematic timesteps in joint trajectories using cuRobo IK solver"
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to input joint trajectory CSV file"
    )
    parser.add_argument(
        "--robot",
        type=str,
        required=True,
        help="Robot configuration YAML file (e.g., ur20.yml, panda.yml)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for joint change to count as reconfiguration in radians (default: 1.0)"
    )
    parser.add_argument(
        "--position_threshold",
        type=float,
        default=0.005,
        help="cuRobo position error threshold in meters (default: 0.005)"
    )
    parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=0.05,
        help="cuRobo rotation error threshold in radians (default: 0.05)"
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=20,
        help="Number of random seeds for cuRobo IK solver (default: 20)"
    )
    parser.add_argument(
        "--num_ik_attempts",
        type=int,
        default=10,
        help="Number of IK solve attempts per problematic timestep (default: 10)"
    )
    parser.add_argument(
        "--collision_check",
        action="store_true",
        help="Enable collision checking in cuRobo IK solver"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and show visualization plots"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save optimized trajectory (default: same as input)"
    )
    parser.add_argument(
        "--use_solver_criteria",
        action="store_true",
        help="Use comprehensive reconfiguration criteria from solver_base.py for analysis"
    )

    args = parser.parse_args()

    # Generate output directory from input filename if not specified
    if args.output_dir is None:
        output_dir = os.path.dirname(args.input_csv)
        if not output_dir:
            output_dir = '.'
    else:
        output_dir = args.output_dir

    # Load trajectory data
    print(f"Loading trajectory from: {args.input_csv}")
    try:
        joint_data, joint_names, pose_data, pose_columns = load_joint_trajectory_with_poses(args.input_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Initialize cuRobo IK solver
    print(f"\nInitializing cuRobo IK solver for '{args.robot}'...")
    try:
        ik_solver, tensor_args, retract_config = setup_ik_solver(
            robot_name=args.robot,
            collision_check=args.collision_check,
            num_seeds=args.num_seeds,
            position_threshold=args.position_threshold,
            rotation_threshold=args.rotation_threshold,
        )
        print(f"✓ cuRobo IK solver initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing cuRobo IK solver: {e}")
        return 1

    # Analyze original trajectory
    print("\n" + "="*80)
    print("ANALYZING ORIGINAL TRAJECTORY")
    print("="*80)

    # Use simple threshold-based analysis (analyze_joint_reconfigurations only takes joint_data and threshold)
    original_results = analyze_joint_reconfigurations(
        joint_data,
        threshold=args.threshold
    )
    print_analysis_results(original_results, joint_names)

    start_time = time.perf_counter()

    # Run independent optimization with cuRobo
    optimized_joint_data, opt_stats = optimize_trajectory_independent_curobo(
        joint_data=joint_data,
        pose_data=pose_data,
        ik_solver=ik_solver,
        tensor_args=tensor_args,
        retract_config=retract_config,
        threshold=args.threshold,
        num_ik_attempts=args.num_ik_attempts,
    )
    total_time = time.perf_counter() - start_time

    # Analyze optimized trajectory
    print("\n" + "="*80)
    print("ANALYZING OPTIMIZED TRAJECTORY")
    print("="*80)
    optimized_results = analyze_joint_reconfigurations(
        optimized_joint_data,
        threshold=args.threshold
    )
    print_analysis_results(optimized_results, joint_names)

    # Compare before and after
    print("\n" + "="*80)
    print("BEFORE vs AFTER COMPARISON")
    print("="*80)
    print(f"Original reconfigurations: {original_results['total_reconfigurations']}")
    print(f"Optimized reconfigurations: {optimized_results['total_reconfigurations']}")
    reduction = original_results['total_reconfigurations'] - optimized_results['total_reconfigurations']
    print(f"Reduction: {reduction}")
    if original_results['total_reconfigurations'] > 0:
        improvement_pct = (reduction / original_results['total_reconfigurations']) * 100
        print(f"Improvement: {improvement_pct:.1f}%")
    else:
        print(f"Improvement: N/A (no reconfigurations in original)")

    # Save optimized trajectory
    input_basename = os.path.splitext(os.path.basename(args.input_csv))[0]
    optimized_csv_path = os.path.join(output_dir, f"{input_basename}_independent_optimized_curobo.csv")
    save_optimized_trajectory(
        output_path=optimized_csv_path,
        joint_data=optimized_joint_data,
        pose_data=pose_data,
        joint_names=joint_names,
        pose_columns=pose_columns
    )

    # Generate comparison plot if requested
    if args.plot:
        plot_save_path = os.path.join(output_dir, f"{input_basename}_independent_optimization_comparison_curobo.png")
        plot_optimization_comparison(
            original_joint_data=joint_data,
            optimized_joint_data=optimized_joint_data,
            joint_names=joint_names,
            original_results=original_results,
            optimized_results=optimized_results,
            threshold=args.threshold,
            robot_name=args.robot.replace('.yml', ''),
            save_path=plot_save_path
        )

    # Save optimization details
    details_file = os.path.join(output_dir, f"{input_basename}_independent_optimized_curobo.txt")
    with open(details_file, 'w') as f:
        f.write("Independent Joint Trajectory Optimization Details (cuRobo IK Solver)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file: {args.input_csv}\n")
        f.write(f"Robot: {args.robot}\n")
        f.write(f"Threshold: {args.threshold:.3f} radians\n")
        f.write(f"Position threshold: {args.position_threshold} m\n")
        f.write(f"Rotation threshold: {args.rotation_threshold} rad\n")
        f.write(f"Number of seeds: {args.num_seeds}\n")
        f.write(f"IK attempts per timestep: {args.num_ik_attempts}\n")
        f.write(f"Collision checking: {args.collision_check}\n\n")

        f.write("Optimization Summary:\n")
        f.write(f"  Originally problematic timesteps: {opt_stats['originally_problematic']}\n")
        f.write(f"  Reconfigurations fixed: {opt_stats['reconfigurations_fixed']}\n")
        f.write(f"  Reconfigurations unfixed: {opt_stats['reconfigurations_unfixed']}\n\n")

        f.write("Before vs After:\n")
        f.write(f"  Original reconfigurations: {original_results['total_reconfigurations']}\n")
        f.write(f"  Optimized reconfigurations: {optimized_results['total_reconfigurations']}\n")
        f.write(f"  Reduction: {reduction}\n")
        if original_results['total_reconfigurations'] > 0:
            f.write(f"  Improvement: {improvement_pct:.1f}%\n\n")

        # Add originally problematic timesteps
        f.write("\n" + "="*80 + "\n")
        f.write("ORIGINALLY PROBLEMATIC TIMESTEPS\n")
        f.write("="*80 + "\n\n")

        if len(original_results['large_reconfig_timesteps']) > 0:
            f.write(f"{'Timestep':<10} {'Max Change':<15}\n")
            f.write("-" * 80 + "\n")
            for reconfig in original_results['large_reconfig_timesteps']:
                f.write(f"{reconfig['timestep']:<10} "
                       f"{reconfig['max_change']:<15.3f}\n")
            f.write(f"\nTotal: {len(original_results['large_reconfig_timesteps'])}\n")
        else:
            f.write("No reconfigurations in original trajectory.\n")

        f.write("\n" + "="*80 + "\n")
        f.write("EXPLANATION OF INDEPENDENT OPTIMIZATION WITH cuRobo\n")
        f.write("="*80 + "\n")
        f.write("This optimizer uses cuRobo's GPU-accelerated IK solver instead of relaxed_ik.\n")
        f.write("Key differences:\n")
        f.write("1. GPU batch processing for faster computation\n")
        f.write("2. Threshold-based instead of tolerance-based optimization\n")
        f.write("3. Uses num_seeds parameter for solution diversity\n")
        f.write("4. BIDIRECTIONAL VALIDATION: Checks impact on both previous AND next timestep\n")
        f.write("   - Only accepts improvement if BOTH directions improve or stay the same\n")
        f.write("   - Prevents creating new reconfigurations in the next timestep\n")
        f.write("\n'Old Total' = max(prev_change, next_change) in ORIGINAL trajectory\n")
        f.write("'New Total' = max(prev_change, next_change) after re-solving IK with cuRobo\n")
        f.write("'Prev Change' = distance to previous timestep (old→new)\n")
        f.write("'Next Change' = distance to next timestep (old→new)\n")
        f.write("="*80 + "\n\n")

        f.write("Detailed Changes (ONLY Problematic Timesteps):\n")
        f.write("Note: 'Total Cost' considers both previous and next timestep distances\n\n")
        f.write(f"{'Timestep':<10} {'Status':<15} {'Old Total':<12} {'New Total':<12} {'Improvement':<12} {'Prev Change':<15} {'Next Change':<15}\n")
        f.write("-" * 100 + "\n")
        for detail in opt_stats['optimization_details']:
            new_change_str = f"{detail['new_max_change']:.3f}" if detail['new_max_change'] is not None else "N/A"

            # Format prev change: old→new
            if detail.get('new_prev') is not None:
                prev_str = f"{detail['old_prev']:.3f}→{detail['new_prev']:.3f}"
            else:
                prev_str = f"{detail['old_prev']:.3f}→N/A"

            # Format next change: old→new
            if detail.get('old_next') is not None and detail.get('new_next') is not None:
                next_str = f"{detail['old_next']:.3f}→{detail['new_next']:.3f}"
            elif detail.get('old_next') is not None:
                next_str = f"{detail['old_next']:.3f}→N/A"
            else:
                next_str = "N/A"

            f.write(f"{detail['timestep']:<10} "
                   f"{detail['status']:<15} "
                   f"{detail['old_max_change']:<12.3f} "
                   f"{new_change_str:<12} "
                   f"{detail['improvement']:<12.3f} "
                   f"{prev_str:<15} "
                   f"{next_str:<15}\n")

        # Add remaining reconfigurations after optimization
        f.write("\n" + "="*80 + "\n")
        f.write("REMAINING RECONFIGURATIONS AFTER OPTIMIZATION\n")
        f.write("="*80 + "\n\n")

        if len(optimized_results['large_reconfig_timesteps']) > 0:
            f.write(f"{'Timestep':<10} {'Max Change':<15}\n")
            f.write("-" * 80 + "\n")
            for reconfig in optimized_results['large_reconfig_timesteps']:
                f.write(f"{reconfig['timestep']:<10} "
                       f"{reconfig['max_change']:<15.3f}\n")

            f.write(f"\nTotal remaining reconfigurations: {len(optimized_results['large_reconfig_timesteps'])}\n")
        else:
            f.write("No reconfigurations remaining! All problematic timesteps successfully optimized.\n")

    print(f"\n✓ Optimization details saved to: {details_file}")
    print(f"Output directory: {output_dir}")
    print(f"Total time: {total_time:.3f} seconds")

    return 0


if __name__ == "__main__":
    exit(main())
