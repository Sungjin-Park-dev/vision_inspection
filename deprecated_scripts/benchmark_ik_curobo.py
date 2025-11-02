#!/usr/bin/env python3
"""
Benchmark script for measuring IK solving performance with cuRobo IKSolver.
Generates 100 random target poses and solves IK 10 times for each pose.

This script is equivalent to benchmark_ik.py but uses cuRobo's GPU-accelerated
IK solver instead of relaxed_ik.
"""

import numpy as np
import time
import os
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional

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


def setup_ik_solver(
    robot_name: str = 'ur20.yml',
    collision_check: bool = True,
    num_seeds: int = 20,
    position_threshold: float = 0.005,
    rotation_threshold: float = 0.05,
) -> IKSolver:
    """
    Initialize cuRobo IKSolver with specified parameters.

    Args:
        robot_name: Robot configuration YAML file name
        collision_check: Whether to enable collision checking
        num_seeds: Number of random seeds for IK solving
        position_threshold: Position error threshold (meters)
        rotation_threshold: Rotation error threshold (radians)

    Returns:
        IKSolver: Configured cuRobo IK solver
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

    print(f"IK solver initialized:")
    print(f"  Robot: {robot_name}")
    print(f"  Collision checking: {collision_check}")
    print(f"  Number of seeds: {num_seeds}")
    print(f"  Position threshold: {position_threshold} m")
    print(f"  Rotation threshold: {rotation_threshold} rad")
    print(f"  Device: {tensor_args.device}")

    return ik_solver


def generate_random_pose_in_workspace(
    ik_solver: IKSolver,
    retract_config: List[float],
    num_attempts: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a random pose within the robot's reachable workspace.

    Uses forward kinematics on random joint configurations to ensure
    the pose is reachable.

    Args:
        ik_solver: cuRobo IK solver instance
        retract_config: Default/retract joint configuration
        num_attempts: Number of attempts to generate valid pose

    Returns:
        Tuple of (position, quaternion) in numpy arrays, or None if failed
    """
    # Get robot joint limits
    joint_names = ik_solver.kinematics.joint_names
    n_joints = len(joint_names)

    for _ in range(num_attempts):
        # Generate random joint configuration within limits
        # Use retract config as base and add random perturbation
        random_joints = np.array(retract_config) + np.random.uniform(-np.pi/2, np.pi/2, n_joints)

        # Clip to joint limits (approximate)
        random_joints = np.clip(random_joints, -2*np.pi, 2*np.pi)

        # Use forward kinematics to get pose
        random_joints_tensor = torch.tensor(
            random_joints,
            dtype=torch.float32,
            device=ik_solver.tensor_args.device
        ).unsqueeze(0)

        # Get FK result
        state = ik_solver.fk(random_joints_tensor)

        if state is not None:
            # Extract position and quaternion
            position = state.ee_position[0].cpu().numpy()
            quaternion = state.ee_quaternion[0].cpu().numpy()  # (w, x, y, z)

            return position, quaternion

    return None


def solve_ik_curobo_batch(
    ik_solver: IKSolver,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    retract_config: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]], float]:
    """
    Solve IK for multiple target poses using cuRobo batch processing.

    Args:
        ik_solver: cuRobo IK solver instance
        target_positions: Target positions array of shape (batch_size, 3)
        target_quaternions: Target quaternions array of shape (batch_size, 4) [w, x, y, z]
        retract_config: Retract joint configuration to use as seed (optional)

    Returns:
        Tuple of (success_flags, solutions, solve_time)
        - success_flags: Boolean array of shape (batch_size,)
        - solutions: List of joint solutions (None for failed solves)
        - solve_time: Total time for batch solve
    """
    # Convert to torch tensors
    device = ik_solver.tensor_args.device
    dtype = ik_solver.tensor_args.dtype

    # Create goal pose batch
    goal_position = torch.tensor(
        target_positions,
        dtype=dtype,
        device=device
    )

    goal_quaternion = torch.tensor(
        target_quaternions,
        dtype=dtype,
        device=device
    )

    goal_pose = Pose(position=goal_position, quaternion=goal_quaternion)

    # Prepare retract config as seed if provided
    if retract_config is not None:
        # Repeat retract config for each pose in batch
        batch_size = len(target_positions)
        retract_tensor = torch.tensor(
            retract_config,
            dtype=dtype,
            device=device
        ).unsqueeze(0).repeat(batch_size, 1)
    else:
        retract_tensor = None

    # Solve IK for entire batch
    start_time = time.time()
    result = ik_solver.solve_batch(goal_pose, retract_config=retract_tensor)
    solve_time = time.time() - start_time

    # Extract results
    success_flags = result.success.cpu().numpy()

    solutions = []
    for i in range(len(success_flags)):
        if success_flags[i]:
            solution = result.js_solution.position[i].cpu().numpy()
            solutions.append(solution)
        else:
            solutions.append(None)

    return success_flags, solutions, solve_time


def benchmark_ik_solving(
    robot_name: str = 'ur20.yml',
    num_poses: int = 100,
    num_samples: int = 10,
    collision_check: bool = False,
    num_seeds: int = 20,
    position_threshold: float = 0.0,
    rotation_threshold: float = 0.05,
    batch_size: int = 32,
) -> dict:
    """
    Benchmark IK solving performance using cuRobo with batch processing.

    Args:
        robot_name: Robot configuration YAML file name
        num_poses: Number of random poses to test
        num_samples: Number of IK solutions to generate per pose
        collision_check: Whether to enable collision checking
        num_seeds: Number of random seeds for IK solving
        position_threshold: Position error threshold (meters)
        rotation_threshold: Rotation error threshold (radians)
        batch_size: Batch size for GPU processing (default: 32)

    Returns:
        dict: Benchmark results including timing information
    """
    print(f"=== cuRobo IK Solving Benchmark (Batch Mode) ===")
    print(f"Robot: {robot_name}")
    print(f"Number of poses: {num_poses}")
    print(f"Samples per pose: {num_samples}")
    print(f"Total IK solves: {num_poses * num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Collision checking: {collision_check}")
    print()

    # Initialize IK solver
    print("Initializing cuRobo IK solver...")
    start_init = time.time()
    ik_solver = setup_ik_solver(
        robot_name=robot_name,
        collision_check=collision_check,
        num_seeds=num_seeds,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
    )
    init_time = time.time() - start_init
    print(f"IK solver initialization time: {init_time:.3f} seconds")
    print()

    # Get retract configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_name))["robot_cfg"]
    retract_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # Generate random target poses
    print("Generating random target poses...")
    start_gen = time.time()
    target_poses = []

    for i in range(num_poses):
        pose = generate_random_pose_in_workspace(ik_solver, retract_config)
        if pose is not None:
            target_poses.append(pose)

        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_poses} poses...")

    gen_time = time.time() - start_gen
    print(f"Pose generation time: {gen_time:.3f} seconds")
    print(f"Successfully generated {len(target_poses)} valid poses")
    print()

    if len(target_poses) == 0:
        print("ERROR: Failed to generate any valid poses!")
        return {}

    # Benchmark IK solving with batch processing
    print(f"Starting IK solving benchmark with batch size {batch_size}...")
    ik_solve_times = []
    success_count = 0
    failure_count = 0

    total_start = time.time()

    # Prepare batches: each pose is solved num_samples times
    # Total number of solves = len(target_poses) * num_samples
    total_solves = len(target_poses) * num_samples

    # Create batched data by repeating each pose num_samples times
    all_positions = []
    all_quaternions = []
    for target_position, target_quaternion in target_poses:
        for _ in range(num_samples):
            all_positions.append(target_position)
            all_quaternions.append(target_quaternion)

    all_positions = np.array(all_positions)
    all_quaternions = np.array(all_quaternions)

    # Process in batches
    num_batches = (total_solves + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_solves)
        current_batch_size = end_idx - start_idx

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"Progress: Batch {batch_idx + 1}/{num_batches} "
                  f"({end_idx}/{total_solves} solves processed)")

        # Extract batch
        batch_positions = all_positions[start_idx:end_idx]
        batch_quaternions = all_quaternions[start_idx:end_idx]

        # Solve IK for batch with retract config as seed
        success_flags, _, batch_solve_time = solve_ik_curobo_batch(
            ik_solver,
            batch_positions,
            batch_quaternions,
            retract_config=np.array(retract_config),
        )

        # Record time per solve (amortized)
        time_per_solve = batch_solve_time / current_batch_size
        ik_solve_times.extend([time_per_solve] * current_batch_size)

        # Count successes
        success_count += int(np.sum(success_flags))
        failure_count += int(current_batch_size - np.sum(success_flags))

    total_time = time.time() - total_start

    print()
    print("=== Benchmark Results ===")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Total IK solves: {len(ik_solve_times)}")
    print(f"Successful solves: {success_count} ({100*success_count/len(ik_solve_times):.1f}%)")
    print(f"Failed solves: {failure_count} ({100*failure_count/len(ik_solve_times):.1f}%)")
    print()

    # Calculate statistics
    ik_solve_times = np.array(ik_solve_times)
    print("=== Timing Statistics (per IK solve) ===")
    print(f"Mean time: {np.mean(ik_solve_times):.6f} seconds")
    print(f"Median time: {np.median(ik_solve_times):.6f} seconds")
    print(f"Min time: {np.min(ik_solve_times):.6f} seconds")
    print(f"Max time: {np.max(ik_solve_times):.6f} seconds")
    print(f"Std dev: {np.std(ik_solve_times):.6f} seconds")
    print()

    print("=== Throughput ===")
    print(f"IK solves per second: {len(ik_solve_times)/total_time:.2f}")
    print(f"Average time per pose (10 samples): {total_time/len(target_poses):.3f} seconds")
    print()

    # Return results
    results = {
        'robot_name': robot_name,
        'num_poses': len(target_poses),
        'num_samples': num_samples,
        'total_solves': len(ik_solve_times),
        'success_count': success_count,
        'failure_count': failure_count,
        'success_rate': success_count / len(ik_solve_times),
        'total_time': total_time,
        'init_time': init_time,
        'gen_time': gen_time,
        'mean_time': float(np.mean(ik_solve_times)),
        'median_time': float(np.median(ik_solve_times)),
        'min_time': float(np.min(ik_solve_times)),
        'max_time': float(np.max(ik_solve_times)),
        'std_time': float(np.std(ik_solve_times)),
        'throughput': len(ik_solve_times) / total_time,
        'solve_times': ik_solve_times.tolist(),
        'batch_size': batch_size,
        'collision_check': collision_check,
        'num_seeds': num_seeds,
        'position_threshold': position_threshold,
        'rotation_threshold': rotation_threshold,
    }

    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark cuRobo IK solver performance')
    parser.add_argument('--robot', type=str, default='ur20.yml',
                       help='Robot configuration YAML file (default: ur20.yml)')
    parser.add_argument('--num_poses', type=int, default=100,
                       help='Number of random poses to test (default: 100)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of IK samples per pose (default: 10)')
    parser.add_argument('--collision_check', action='store_true',
                       help='Enable collision checking (default: False)')
    parser.add_argument('--num_seeds', type=int, default=20,
                       help='Number of random seeds for IK (default: 20)')
    parser.add_argument('--position_threshold', type=float, default=0.005,
                       help='Position error threshold in meters (default: 0.005)')
    parser.add_argument('--rotation_threshold', type=float, default=0.05,
                       help='Rotation error threshold in radians (default: 0.05)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for GPU processing (default: 32)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_ik_solving(
        robot_name=args.robot,
        num_poses=args.num_poses,
        num_samples=args.num_samples,
        collision_check=args.collision_check,
        num_seeds=args.num_seeds,
        position_threshold=args.position_threshold,
        rotation_threshold=args.rotation_threshold,
        batch_size=args.batch_size,
    )

    if results:
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        robot_basename = os.path.splitext(args.robot)[0]
        output_file = os.path.join(
            args.output_dir,
            f"benchmark_curobo_{robot_basename}_{timestamp}.json"
        )

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")
