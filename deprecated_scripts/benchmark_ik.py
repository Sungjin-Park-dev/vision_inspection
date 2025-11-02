#!/usr/bin/env python3
"""
Benchmark script for measuring IK solving performance with relaxed_ik.
Generates 100 random target poses and solves IK 10 times for each pose.
"""

import numpy as np
import time
import sys
import os
from robot import Robot
from robot_kinematics import RobotKinematics

def generate_random_pose_in_workspace(robot):
    """Generate a random pose within the robot's reachable workspace"""
    # Generate random joint configuration
    random_joints = robot.generate_random_angle()

    # Use forward kinematics to get a reachable pose
    pose = robot.rk.forward_position_kinematics(random_joints)

    return pose

def benchmark_ik_solving(robot_name='panda', num_poses=100, num_samples_per_pose=10):
    """
    Benchmark IK solving performance

    Args:
        robot_name: Name of the robot ('panda', 'ur5', etc.)
        num_poses: Number of random poses to test
        num_samples_per_pose: Number of IK solutions to generate per pose

    Returns:
        dict: Benchmark results including timing information
    """
    print(f"=== IK Solving Benchmark ===")
    print(f"Robot: {robot_name}")
    print(f"Number of poses: {num_poses}")
    print(f"Samples per pose: {num_samples_per_pose}")
    print(f"Total IK solves: {num_poses * num_samples_per_pose}")
    print()

    # Initialize robot
    print("Initializing robot...")
    start_init = time.time()
    robot = Robot(robot_name)
    init_time = time.time() - start_init
    print(f"Robot initialization time: {init_time:.3f} seconds")
    print()

    # Set relaxed IK parameters
    # Use tolerances that are supported by check_pose function
    # Format: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
    # This configuration allows x,y tolerance, strict z, strict x,y rotation, free z rotation
    tolerances = [0., 0., 0., 0., 0., 999.]
    constrain_velocity = False
    max_iter = 10

    # Generate random target poses
    print("Generating random target poses...")
    start_gen = time.time()
    target_poses = []
    for i in range(num_poses):
        pose = generate_random_pose_in_workspace(robot)
        target_poses.append(pose)
    gen_time = time.time() - start_gen
    print(f"Pose generation time: {gen_time:.3f} seconds")
    print()

    # Benchmark IK solving
    print("Starting IK solving benchmark...")
    ik_solve_times = []
    success_count = 0
    failure_count = 0

    total_start = time.time()

    for pose_idx, target_pose in enumerate(target_poses):
        if (pose_idx + 1) % 10 == 0:
            print(f"Progress: {pose_idx + 1}/{num_poses} poses processed...")

        for sample_idx in range(num_samples_per_pose):
            # Generate random start configuration
            start_config = robot.generate_random_angle()

            # Measure IK solving time
            ik_start = time.time()
            ik_solution = robot.reach_with_relaxed_ik(
                target_pose,
                constrain_velocity,
                tolerances,
                start_config=start_config,
                max_iter=max_iter
            )
            ik_time = time.time() - ik_start

            ik_solve_times.append(ik_time)

            if len(ik_solution) > 0:
                success_count += 1
            else:
                failure_count += 1

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
    print(f"Average time per pose (10 samples): {total_time/num_poses:.3f} seconds")
    print()

    # Return results
    results = {
        'robot_name': robot_name,
        'num_poses': num_poses,
        'num_samples_per_pose': num_samples_per_pose,
        'total_solves': len(ik_solve_times),
        'success_count': success_count,
        'failure_count': failure_count,
        'success_rate': success_count / len(ik_solve_times),
        'total_time': total_time,
        'init_time': init_time,
        'gen_time': gen_time,
        'mean_time': np.mean(ik_solve_times),
        'median_time': np.median(ik_solve_times),
        'min_time': np.min(ik_solve_times),
        'max_time': np.max(ik_solve_times),
        'std_time': np.std(ik_solve_times),
        'throughput': len(ik_solve_times) / total_time,
        'solve_times': ik_solve_times.tolist()
    }

    return results

if __name__ == "__main__":
    # Parse command line arguments
    robot_name = sys.argv[1] if len(sys.argv) > 1 else 'panda'
    num_poses = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    num_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Run benchmark
    results = benchmark_ik_solving(robot_name, num_poses, num_samples)

    # Save results to file
    import json
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{robot_name}_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
