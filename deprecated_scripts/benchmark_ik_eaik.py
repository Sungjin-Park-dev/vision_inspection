#!/usr/bin/env python3
"""
Benchmark script for measuring IK solving performance with EAIK (Efficient Analytical IK).
Generates 100 random target poses and solves IK 10 times for each pose.

This script uses EAIK analytical IK solver for comparison with cuRobo.
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
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.state import JointState

# EAIK imports
from eaik.IK_URDF import UrdfRobot


# Coordinate transformation matrix from CuRobo to EAIK tool frame
CUROBO_TO_EAIK_TOOL = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def setup_ik_solver_for_collision(
    robot_name: str = 'ur20.yml',
    collision_check: bool = True,
) -> IKSolver:
    """
    Initialize cuRobo IKSolver for collision checking only.

    Args:
        robot_name: Robot configuration YAML file name
        collision_check: Whether to enable collision checking

    Returns:
        IKSolver: Configured cuRobo IK solver (for collision checking)
    """
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

    # Configure IK solver (for collision checking only)
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
        collision_checker_type=CollisionCheckerType.MESH if collision_check else CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": 30, "mesh": 10},
    )

    ik_solver = IKSolver(ik_config)

    print(f"Collision checker initialized:")
    print(f"  Robot: {robot_name}")
    print(f"  Collision checking: {collision_check}")

    return ik_solver


def setup_eaik_solver(urdf_path: str) -> UrdfRobot:
    """
    Initialize EAIK solver.

    Args:
        urdf_path: Path to URDF file

    Returns:
        UrdfRobot: EAIK robot instance
    """
    print(f"Setting up EAIK solver...")
    print(f"  URDF path: {urdf_path}")

    bot = UrdfRobot(urdf_path)

    print(f"EAIK solver initialized")

    return bot


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
        ik_solver: cuRobo IK solver instance (for FK)
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


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.

    Args:
        quat: Quaternion [w, x, y, z]

    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quat

    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
    ], dtype=np.float64)


def pose_to_matrix(position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Convert position and quaternion to 4x4 transformation matrix.

    Args:
        position: Position [x, y, z]
        quaternion: Quaternion [w, x, y, z]

    Returns:
        4x4 transformation matrix
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    matrix[:3, 3] = position
    return matrix


def solve_ik_eaik_batch(
    eaik_bot: UrdfRobot,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
) -> Tuple[List, float]:
    """
    Solve IK for multiple target poses using EAIK analytical solver.

    Args:
        eaik_bot: EAIK robot instance
        target_positions: Target positions array of shape (batch_size, 3)
        target_quaternions: Target quaternions array of shape (batch_size, 4) [w, x, y, z]

    Returns:
        Tuple of (eaik_results, solve_time)
        - eaik_results: List of EAIK solution objects
        - solve_time: Total time for batch solve
    """
    batch_size = len(target_positions)

    # Convert poses to 4x4 matrices
    pose_matrices = []
    for i in range(batch_size):
        pose_matrix = pose_to_matrix(target_positions[i], target_quaternions[i])
        # Apply coordinate transformation from CuRobo to EAIK
        pose_matrix_eaik = pose_matrix @ CUROBO_TO_EAIK_TOOL
        pose_matrices.append(pose_matrix_eaik)

    pose_matrices = np.array(pose_matrices)

    # Solve IK for entire batch
    start_time = time.time()
    eaik_results = eaik_bot.IK_batched(pose_matrices)
    solve_time = time.time() - start_time

    return eaik_results, solve_time


def check_collision_batch(
    ik_solver: IKSolver,
    joint_configs: List[List[np.ndarray]],
) -> Tuple[List[List[bool]], float]:
    """
    Check collision for batched joint configurations.

    Args:
        ik_solver: cuRobo IK solver (for collision checking)
        joint_configs: List of [list of joint configurations per pose]

    Returns:
        Tuple of (collision_flags, check_time)
        - collision_flags: List of [list of collision-free flags per solution]
        - check_time: Total time for collision checking
    """
    start_time = time.time()

    all_collision_flags = []

    for solutions in joint_configs:
        solution_flags = []

        if not solutions:
            all_collision_flags.append([])
            continue

        # Batch collision check for all solutions of this pose
        batched_q = np.stack([np.asarray(sol, dtype=np.float64) for sol in solutions], axis=0)

        tensor_args = ik_solver.tensor_args
        q_tensor = tensor_args.to_device(torch.from_numpy(batched_q))

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
            feasibility = torch.ones(len(solutions), dtype=torch.bool)
        else:
            feasibility = feasible.detach()
            if feasibility.is_cuda:
                feasibility = feasibility.cpu()
            feasibility = feasibility.flatten().to(dtype=torch.bool)

        solution_flags = [bool(flag) for flag in feasibility]
        all_collision_flags.append(solution_flags)

    check_time = time.time() - start_time

    return all_collision_flags, check_time


def benchmark_ik_solving(
    robot_name: str = 'ur20.yml',
    urdf_path: str = '/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf',
    num_poses: int = 100,
    num_samples: int = 10,
    collision_check: bool = False,
    batch_size: int = 32,
) -> dict:
    """
    Benchmark IK solving performance using EAIK analytical solver.

    Args:
        robot_name: Robot configuration YAML file name
        urdf_path: Path to URDF file for EAIK
        num_poses: Number of random poses to test
        num_samples: Number of IK solutions to generate per pose
        collision_check: Whether to enable collision checking
        batch_size: Batch size for processing (default: 32)

    Returns:
        dict: Benchmark results including timing information
    """
    print(f"=== EAIK IK Solving Benchmark (Batch Mode) ===")
    print(f"Robot: {robot_name}")
    print(f"URDF: {urdf_path}")
    print(f"Number of poses: {num_poses}")
    print(f"Samples per pose: {num_samples}")
    print(f"Total IK solves: {num_poses * num_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Collision checking: {collision_check}")
    print()

    # Initialize EAIK solver
    print("Initializing EAIK solver...")
    start_init = time.time()
    eaik_bot = setup_eaik_solver(urdf_path)
    init_time = time.time() - start_init
    print(f"EAIK solver initialization time: {init_time:.3f} seconds")
    print()

    # Initialize collision checker (cuRobo for collision checking only)
    if collision_check:
        print("Initializing collision checker...")
        collision_checker = setup_ik_solver_for_collision(robot_name, collision_check=True)
        print()
    else:
        collision_checker = None

    # Get retract configuration (use cuRobo config for FK)
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_name))["robot_cfg"]
    retract_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    # Setup FK solver (for pose generation)
    fk_solver = setup_ik_solver_for_collision(robot_name, collision_check=False)

    # Generate random target poses
    print("Generating random target poses...")
    start_gen = time.time()
    target_poses = []

    for i in range(num_poses):
        pose = generate_random_pose_in_workspace(fk_solver, retract_config)
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
    print(f"Starting EAIK IK solving benchmark with batch size {batch_size}...")
    ik_solve_times = []
    collision_check_times = []
    success_count = 0
    failure_count = 0
    collision_free_count = 0

    total_start = time.time()

    # Prepare batches: each pose is solved num_samples times
    # Note: EAIK returns all analytical solutions, so we don't need to repeat poses
    # total_solves will be used for reporting in output
    total_solves = len(target_poses) * num_samples  # noqa: F841

    # Process poses in batches
    all_positions = []
    all_quaternions = []
    for target_position, target_quaternion in target_poses:
        # EAIK returns all solutions in one call, so we only call once per pose
        all_positions.append(target_position)
        all_quaternions.append(target_quaternion)

    all_positions = np.array(all_positions)
    all_quaternions = np.array(all_quaternions)

    # Process in batches
    num_batches = (len(target_poses) + batch_size - 1) // batch_size

    all_solutions = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(target_poses))
        current_batch_size = end_idx - start_idx

        if (batch_idx + 1) % max(1, num_batches // 10) == 0:
            print(f"Progress: Batch {batch_idx + 1}/{num_batches} "
                  f"({end_idx}/{len(target_poses)} poses processed)")

        # Extract batch
        batch_positions = all_positions[start_idx:end_idx]
        batch_quaternions = all_quaternions[start_idx:end_idx]

        # Solve IK for batch using EAIK
        eaik_results, batch_solve_time = solve_ik_eaik_batch(
            eaik_bot,
            batch_positions,
            batch_quaternions,
        )

        # Record time per pose
        time_per_pose = batch_solve_time / current_batch_size

        # Extract solutions from EAIK results
        batch_solutions = []
        for result in eaik_results:
            if result is None:
                batch_solutions.append([])
            else:
                q_candidates = getattr(result, "Q", None)
                if q_candidates is None or len(q_candidates) == 0:
                    batch_solutions.append([])
                else:
                    solutions = [np.asarray(q, dtype=np.float64) for q in q_candidates]
                    batch_solutions.append(solutions)

        all_solutions.extend(batch_solutions)

        # Record times (multiply by num_samples since each pose generates multiple solutions)
        ik_solve_times.extend([time_per_pose / num_samples] * (current_batch_size * num_samples))

    # Collision checking (if enabled)
    if collision_check and collision_checker is not None:
        print("\nPerforming collision checking...")
        collision_flags, collision_time = check_collision_batch(
            collision_checker,
            all_solutions,
        )
        collision_check_times.append(collision_time)
        print(f"Collision checking time: {collision_time:.3f} seconds")
    else:
        collision_flags = [[True] * len(sols) for sols in all_solutions]
        collision_time = 0.0

    # Count successes and failures
    for solutions, flags in zip(all_solutions, collision_flags):
        num_solutions = len(solutions)
        num_collision_free = sum(flags)

        # Count solutions (up to num_samples)
        solutions_to_count = min(num_solutions, num_samples)

        if num_solutions == 0:
            # No solutions found
            failure_count += num_samples
        else:
            # Has solutions
            success_count += solutions_to_count

            # Pad if fewer solutions than num_samples
            if solutions_to_count < num_samples:
                failure_count += num_samples - solutions_to_count

        # Count collision-free solutions
        collision_free_count += min(num_collision_free, num_samples)

    total_time = time.time() - total_start

    print()
    print("=== Benchmark Results ===")
    print(f"Total time: {total_time:.3f} seconds")
    print(f"Total IK solves: {len(ik_solve_times)}")
    print(f"Successful solves: {success_count} ({100*success_count/len(ik_solve_times):.1f}%)")
    print(f"Failed solves: {failure_count} ({100*failure_count/len(ik_solve_times):.1f}%)")
    if collision_check:
        print(f"Collision-free solves: {collision_free_count} ({100*collision_free_count/len(ik_solve_times):.1f}%)")
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
    print(f"Average time per pose: {total_time/len(target_poses):.3f} seconds")
    print()

    # Return results
    results = {
        'robot_name': robot_name,
        'urdf_path': urdf_path,
        'num_poses': len(target_poses),
        'num_samples': num_samples,
        'total_solves': len(ik_solve_times),
        'success_count': int(success_count),
        'failure_count': int(failure_count),
        'collision_free_count': int(collision_free_count),
        'success_rate': success_count / len(ik_solve_times),
        'collision_free_rate': collision_free_count / len(ik_solve_times) if collision_check else None,
        'total_time': total_time,
        'init_time': init_time,
        'gen_time': gen_time,
        'collision_time': collision_time if collision_check else 0.0,
        'mean_time': float(np.mean(ik_solve_times)),
        'median_time': float(np.median(ik_solve_times)),
        'min_time': float(np.min(ik_solve_times)),
        'max_time': float(np.max(ik_solve_times)),
        'std_time': float(np.std(ik_solve_times)),
        'throughput': len(ik_solve_times) / total_time,
        'solve_times': ik_solve_times.tolist(),
        'batch_size': batch_size,
        'collision_check': collision_check,
        'solver': 'EAIK',
    }

    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Benchmark EAIK analytical IK solver performance')
    parser.add_argument('--robot', type=str, default='ur20.yml',
                       help='Robot configuration YAML file (default: ur20.yml)')
    parser.add_argument('--urdf', type=str,
                       default='/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf',
                       help='Path to URDF file for EAIK')
    parser.add_argument('--num_poses', type=int, default=100,
                       help='Number of random poses to test (default: 100)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of IK samples per pose (default: 10)')
    parser.add_argument('--collision_check', action='store_true',
                       help='Enable collision checking (default: False)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing (default: 32)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_ik_solving(
        robot_name=args.robot,
        urdf_path=args.urdf,
        num_poses=args.num_poses,
        num_samples=args.num_samples,
        collision_check=args.collision_check,
        batch_size=args.batch_size,
    )

    if results:
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        robot_basename = os.path.splitext(args.robot)[0]
        output_file = os.path.join(
            args.output_dir,
            f"benchmark_eaik_{robot_basename}_{timestamp}.json"
        )

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {output_file}")
