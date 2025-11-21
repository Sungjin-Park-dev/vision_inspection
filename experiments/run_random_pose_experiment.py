#!/usr/bin/env python3
"""
Random Pose Experiment Runner

This script automates the full pipeline for multiple random glass poses:
1. Generate random poses using random_pose.py
2. For each pose, run:
   - compute_ik_solutions.py (IK computation)
   - plan_trajectory.py (trajectory planning)
   - coal_check.py (collision and reconfiguration checking)
3. Collect metrics and save to CSV

Usage:
    python experiments/run_random_pose_experiment.py --num_poses 10 --method dp
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add parent directory to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Add scripts directory to path for analyze_joint_reconfigurations import
scripts_dir = os.path.join(project_root, 'scripts')
sys.path.insert(0, scripts_dir)

# Import the random pose generator
from experiments.random_pose import random_pose

# Import common config
from common import config


# ============================================================================
# Configuration
# ============================================================================

class ExperimentConfig:
    """Configuration for random pose experiment"""
    def __init__(
        self,
        num_poses: int = 10,
        selection_method: str = "dp",
        base_mesh_file: str = None,
        tsp_tour_path: str = None,
        output_dir: str = None,
    ):
        self.num_poses = num_poses
        self.selection_method = selection_method
        self.base_mesh_file = base_mesh_file or config.DEFAULT_MESH_FILE
        self.tsp_tour_path = tsp_tour_path
        self.output_dir = output_dir or "experiments/results"

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# Metric Collection
# ============================================================================

class PoseMetrics:
    """Container for metrics from a single pose experiment"""
    def __init__(self, pose_id: int, position: np.ndarray, rotation: np.ndarray):
        self.pose_id = pose_id
        self.glass_pos = position
        self.glass_quat = rotation

        # Timing metrics
        self.ik_time_sec = 0.0  # IK computation + CuRobo collision filtering
        self.plan_time_sec = 0.0
        self.ik_plan_total_sec = 0.0  # Combined IK + planning time
        self.collision_check_time_sec = 0.0  # Trajectory collision checking (COAL)
        self.replan_time_sec = 0.0  # Time spent on motion replanning

        # IK metrics
        self.ik_solutions_all = 0
        self.ik_solutions_safe = 0

        # Collision metrics
        self.waypoint_collisions = 0
        self.segment_collisions = 0
        self.total_collisions = 0

        # Reconfiguration metrics
        self.reconfigurations = 0

        # Replanning metrics
        self.replan_attempts = 0  # Number of replanning attempts
        self.trajectory_modified = False  # Whether trajectory was modified by replanning

        # Status
        self.success = False
        self.error_message = ""

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for CSV writing"""
        return {
            'pose_id': self.pose_id,
            'glass_pos_x': float(self.glass_pos[0]),
            'glass_pos_y': float(self.glass_pos[1]),
            'glass_pos_z': float(self.glass_pos[2]),
            'glass_quat_w': float(self.glass_quat[0]),
            'glass_quat_x': float(self.glass_quat[1]),
            'glass_quat_y': float(self.glass_quat[2]),
            'glass_quat_z': float(self.glass_quat[3]),
            'ik_time_sec': self.ik_time_sec,
            'plan_time_sec': self.plan_time_sec,
            'ik_plan_total_sec': self.ik_plan_total_sec,
            'ik_solutions_all': self.ik_solutions_all,
            'ik_solutions_safe': self.ik_solutions_safe,
            'collision_check_time_sec': self.collision_check_time_sec,
            'replan_time_sec': self.replan_time_sec,
            'waypoint_collisions': self.waypoint_collisions,
            'segment_collisions': self.segment_collisions,
            'total_collisions': self.total_collisions,
            'reconfigurations': self.reconfigurations,
            'replan_attempts': self.replan_attempts,
            'trajectory_modified': self.trajectory_modified,
            'success': self.success,
            'error_message': self.error_message,
        }


# ============================================================================
# Pipeline Execution Functions
# ============================================================================

def setup_pose_directories(pose_id: int) -> Dict[str, str]:
    """Create directory structure for a pose experiment"""
    base_dir = Path("data")

    dirs = {
        'viewpoint': base_dir / "viewpoint" / f"pose_{pose_id:02d}",
        'ik': base_dir / "ik" / f"pose_{pose_id:02d}",
        'trajectory': base_dir / "trajectory" / f"pose_{pose_id:02d}",
        'collision': base_dir / "collision" / f"pose_{pose_id:02d}",
    }

    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return {key: str(val) for key, val in dirs.items()}


def run_ik_computation(
    pose_id: int,
    glass_position: np.ndarray,
    glass_rotation: np.ndarray,
    exp_config: ExperimentConfig,
    dirs: Dict[str, str]
) -> Tuple[Optional[str], float, int, int]:
    """
    Run IK computation for a pose

    Returns:
        (ik_solutions_path, elapsed_time, num_all_solutions, num_safe_solutions)
    """
    print(f"\n{'='*70}")
    print(f"POSE {pose_id}: COMPUTING IK SOLUTIONS")
    print(f"{'='*70}")
    print(f"Position: {glass_position}")
    print(f"Rotation (quat): {glass_rotation}")

    # Import here to avoid module loading issues
    import sys
    import importlib.util

    # Load compute_ik_solutions module
    spec = importlib.util.spec_from_file_location(
        "compute_ik_solutions",
        os.path.join(scripts_dir, "compute_ik_solutions.py")
    )
    compute_ik_module = importlib.util.module_from_spec(spec)
    sys.modules["compute_ik_solutions"] = compute_ik_module
    spec.loader.exec_module(compute_ik_module)

    ComputeConfig = compute_ik_module.ComputeConfig
    setup_collision_world = compute_ik_module.setup_collision_world
    setup_ik_solver = compute_ik_module.setup_ik_solver
    process_viewpoints = compute_ik_module.process_viewpoints
    save_ik_solutions_hdf5 = compute_ik_module.save_ik_solutions_hdf5

    # Determine TSP tour path
    if exp_config.tsp_tour_path:
        tsp_tour_path = exp_config.tsp_tour_path
    else:
        # Use a default TSP tour - you may need to adjust this
        # For now, assume the TSP tour already exists
        tsp_tour_path = "data/tsp/tsp_tour.csv"  # Adjust as needed

    # Setup output paths
    ik_output_path = os.path.join(dirs['ik'], 'ik_solutions.h5')

    # Create compute config with custom glass pose
    compute_cfg = ComputeConfig(
        tsp_tour_path=tsp_tour_path,
        output_path=ik_output_path,
        robot_config_file=config.DEFAULT_ROBOT_CONFIG,
        glass_position=glass_position.copy(),
        glass_rotation=glass_rotation.copy(),
    )

    start_time = time.perf_counter()

    try:
        # Setup collision world
        world_cfg = setup_collision_world(compute_cfg)

        # Setup IK solver
        ik_solver = setup_ik_solver(compute_cfg, world_cfg)

        # Process viewpoints (load TSP, compute IK, check collisions)
        viewpoint_mgr, tsp_result = process_viewpoints(compute_cfg, ik_solver)

        # Save IK solutions
        save_ik_solutions_hdf5(
            viewpoint_mgr.viewpoints,
            ik_output_path,
            tsp_tour_path
        )

        elapsed = time.perf_counter() - start_time

        # Count solutions
        num_all = viewpoint_mgr.count_with_all_ik()
        num_safe = viewpoint_mgr.count_with_safe_ik()

        print(f"\n✓ IK computation completed in {elapsed:.2f}s")
        print(f"  All solutions: {num_all}/{len(viewpoint_mgr.viewpoints)}")
        print(f"  Safe solutions: {num_safe}/{len(viewpoint_mgr.viewpoints)}")

        return ik_output_path, elapsed, num_all, num_safe

    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - start_time
        print(f"\n✗ IK computation failed after {elapsed:.2f}s: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return None, elapsed, 0, 0


def run_trajectory_planning(
    pose_id: int,
    ik_solutions_path: str,
    exp_config: ExperimentConfig,
    dirs: Dict[str, str]
) -> Tuple[Optional[str], float]:
    """
    Run trajectory planning

    Returns:
        (trajectory_csv_path, elapsed_time)
    """
    print(f"\n{'='*70}")
    print(f"POSE {pose_id}: PLANNING TRAJECTORY")
    print(f"{'='*70}")

    # Import here to avoid module loading issues
    import sys
    import importlib.util

    # Load plan_trajectory module
    spec = importlib.util.spec_from_file_location(
        "plan_trajectory",
        os.path.join(scripts_dir, "plan_trajectory.py")
    )
    plan_module = importlib.util.module_from_spec(spec)
    sys.modules["plan_trajectory"] = plan_module
    spec.loader.exec_module(plan_module)

    PlanConfig = plan_module.PlanConfig
    load_ik_solutions = plan_module.load_ik_solutions
    plan_trajectory = plan_module.plan_trajectory
    save_joint_trajectory_csv = plan_module.save_joint_trajectory_csv

    trajectory_output_path = os.path.join(
        dirs['trajectory'],
        f'trajectory_{exp_config.selection_method}.csv'
    )

    plan_cfg = PlanConfig(
        ik_solutions_path=ik_solutions_path,
        output_path=trajectory_output_path,
        selection_method=exp_config.selection_method,
    )

    start_time = time.perf_counter()

    try:
        # Load IK solutions
        viewpoints, metadata, tsp_tour_path = load_ik_solutions(ik_solutions_path)

        # Plan trajectory
        joint_targets, solution_indices, viewpoints_with_safe = plan_trajectory(
            viewpoints,
            plan_cfg
        )

        # Save trajectory
        save_joint_trajectory_csv(
            viewpoints_with_safe,
            joint_targets,
            trajectory_output_path
        )

        elapsed = time.perf_counter() - start_time

        print(f"\n✓ Trajectory planning completed in {elapsed:.2f}s")
        print(f"  Waypoints: {len(joint_targets)}")

        return trajectory_output_path, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        print(f"\n✗ Trajectory planning failed after {elapsed:.2f}s: {e}")
        return None, elapsed


def run_collision_checking(
    pose_id: int,
    trajectory_path: str,
    glass_position: np.ndarray,
    glass_rotation: np.ndarray,
    dirs: Dict[str, str]
) -> Tuple[float, int, int, int, int, float, int, bool]:
    """
    Run collision and reconfiguration checking

    Returns:
        (elapsed_time, waypoint_collisions, segment_collisions, total_collisions, reconfigurations,
         replan_time, replan_attempts, trajectory_modified)
    """
    print(f"\n{'='*70}")
    print(f"POSE {pose_id}: CHECKING COLLISIONS")
    print(f"{'='*70}")

    # Import here to avoid module loading issues
    import sys
    import importlib.util

    # Load coal_check module
    spec = importlib.util.spec_from_file_location(
        "coal_check",
        os.path.join(scripts_dir, "coal_check.py")
    )
    coal_module = importlib.util.module_from_spec(spec)
    sys.modules["coal_check"] = coal_module
    spec.loader.exec_module(coal_module)

    COALCollisionChecker = coal_module.COALCollisionChecker
    load_trajectory_csv = coal_module.load_trajectory_csv

    start_time = time.perf_counter()

    try:
        # Load trajectory
        trajectory, joint_names = load_trajectory_csv(trajectory_path)

        # Create collision checker with custom glass pose
        checker = COALCollisionChecker(
            robot_urdf_path=config.DEFAULT_ROBOT_URDF,
            obstacle_mesh_paths=[config.DEFAULT_MESH_FILE],
            glass_position=glass_position.copy(),
            glass_rotation=glass_rotation.copy(),
            table_position=config.TABLE_POSITION.copy(),
            table_dimensions=config.TABLE_DIMENSIONS.copy(),
            wall_position=config.WALL_POSITION.copy(),
            wall_dimensions=config.WALL_DIMENSIONS.copy(),
            workbench_position=config.WORKBENCH_POSITION.copy(),
            workbench_dimensions=config.WORKBENCH_DIMENSIONS.copy(),
            robot_mount_position=config.ROBOT_MOUNT_POSITION.copy(),
            robot_mount_dimensions=config.ROBOT_MOUNT_DIMENSIONS.copy(),
            robot_config_path=config.DEFAULT_ROBOT_CONFIG_YAML,
            use_link_meshes=config.COLLISION_USE_LINK_MESHES,
            mesh_base_path=config.MESH_BASE_PATH,
            collision_margin=config.COLLISION_MARGIN,
        )

        # Check trajectory
        results = checker.check_trajectory(
            trajectory,
            interpolate=True,
            num_interp_steps=config.COLLISION_INTERP_STEPS,
            check_reconfig=True,
            verbose=config.COLLISION_VERBOSE,
            show_link_collisions=config.COLLISION_SHOW_LINK_DETAILS,
            parallel=config.COLLISION_PARALLEL,
            num_workers=config.COLLISION_NUM_WORKERS,
            adaptive_interp=config.COLLISION_ADAPTIVE_INTERP,
            adaptive_max_joint_step_deg=config.COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG,
            adaptive_min_steps=config.COLLISION_ADAPTIVE_MIN_STEPS,
            adaptive_max_steps=config.COLLISION_ADAPTIVE_MAX_STEPS,
        )

        elapsed = time.perf_counter() - start_time

        # Extract metrics
        waypoint_coll = results.get('num_collisions', 0)
        segment_coll = results.get('num_segment_collisions', 0)
        total_coll = results.get('total_collisions', 0)
        reconfigs = results.get('num_reconfigurations', 0)

        # Extract replanning metrics
        traj_modified = results.get('trajectory_modified', False)
        replan_reconfig_summary = results.get('replan_reconfig_summary', {})
        replan_collision_summary = results.get('replan_collision_summary', {})

        # Calculate total replanning attempts and time
        reconfig_attempts = replan_reconfig_summary.get('total_replanned', 0)
        collision_iterations = results.get('collision_replan_iterations', 0)
        replan_attempts = reconfig_attempts + collision_iterations

        # Estimate replanning time (from overall time minus check times)
        overall_time = results.get('overall_time_sec', elapsed)
        reconfig_check_time = results.get('reconfig_check_time_sec', 0)
        collision_check_time = results.get('collision_check_time_sec', 0)
        replan_time = max(0, overall_time - reconfig_check_time - collision_check_time)

        print(f"\n✓ Collision checking completed in {elapsed:.2f}s")
        print(f"  Waypoint collisions: {waypoint_coll}")
        print(f"  Segment collisions: {segment_coll}")
        print(f"  Total collisions: {total_coll}")
        print(f"  Reconfigurations: {reconfigs}")
        print(f"  Trajectory modified: {traj_modified}")
        print(f"  Replanning attempts: {replan_attempts}")
        print(f"  Replanning time: {replan_time:.2f}s")

        # Save collision results to text file
        collision_output_path = os.path.join(dirs['collision'], 'collision_results.txt')
        with open(collision_output_path, 'w') as f:
            f.write(f"Collision Check Results for Pose {pose_id}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Glass position: {glass_position}\n")
            f.write(f"Glass rotation: {glass_rotation}\n\n")
            f.write(f"Waypoint collisions: {waypoint_coll}\n")
            f.write(f"Segment collisions: {segment_coll}\n")
            f.write(f"Total collisions: {total_coll}\n")
            f.write(f"Reconfigurations: {reconfigs}\n")
            f.write(f"Trajectory modified: {traj_modified}\n")
            f.write(f"Replanning attempts: {replan_attempts}\n")
            f.write(f"Check time: {elapsed:.2f}s\n")
            f.write(f"Replanning time: {replan_time:.2f}s\n")

        return elapsed, waypoint_coll, segment_coll, total_coll, reconfigs, replan_time, replan_attempts, traj_modified

    except Exception as e:
        import traceback
        elapsed = time.perf_counter() - start_time
        print(f"\n✗ Collision checking failed after {elapsed:.2f}s: {e}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return elapsed, 0, 0, 0, 0, 0.0, 0, False


# ============================================================================
# Experiment Runner
# ============================================================================

def run_single_pose_experiment(
    pose_id: int,
    glass_position: np.ndarray,
    glass_rotation: np.ndarray,
    exp_config: ExperimentConfig
) -> PoseMetrics:
    """Run complete pipeline for a single pose"""
    metrics = PoseMetrics(pose_id, glass_position, glass_rotation)

    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT: POSE {pose_id}/{exp_config.num_poses}")
    print(f"{'#'*70}")

    # Setup directories
    dirs = setup_pose_directories(pose_id)

    try:
        # Step 1: IK Computation
        ik_path, ik_time, num_all, num_safe = run_ik_computation(
            pose_id, glass_position, glass_rotation, exp_config, dirs
        )
        metrics.ik_time_sec = ik_time
        metrics.ik_solutions_all = num_all
        metrics.ik_solutions_safe = num_safe

        if ik_path is None or num_safe == 0:
            metrics.error_message = "IK computation failed or no safe solutions"
            return metrics

        # Step 2: Trajectory Planning
        traj_path, plan_time = run_trajectory_planning(
            pose_id, ik_path, exp_config, dirs
        )
        metrics.plan_time_sec = plan_time
        metrics.ik_plan_total_sec = metrics.ik_time_sec + metrics.plan_time_sec

        if traj_path is None:
            metrics.error_message = "Trajectory planning failed"
            return metrics

        # Step 3: Collision Checking
        (coll_time, wp_coll, seg_coll, total_coll, reconfigs,
         replan_time, replan_attempts, traj_modified) = run_collision_checking(
            pose_id, traj_path, glass_position, glass_rotation, dirs
        )
        metrics.collision_check_time_sec = coll_time
        metrics.waypoint_collisions = wp_coll
        metrics.segment_collisions = seg_coll
        metrics.total_collisions = total_coll
        metrics.reconfigurations = reconfigs
        metrics.replan_time_sec = replan_time
        metrics.replan_attempts = replan_attempts
        metrics.trajectory_modified = traj_modified

        # Success!
        metrics.success = True

    except Exception as e:
        metrics.error_message = f"Unexpected error: {str(e)}"
        print(f"\n✗ Pose {pose_id} failed: {e}")

    return metrics


def run_experiment(exp_config: ExperimentConfig) -> List[PoseMetrics]:
    """Run complete experiment for all poses"""
    print(f"\n{'='*70}")
    print("RANDOM POSE EXPERIMENT")
    print(f"{'='*70}")
    print(f"Number of poses: {exp_config.num_poses}")
    print(f"Selection method: {exp_config.selection_method}")
    print(f"Output directory: {exp_config.output_dir}")
    print(f"{'='*70}\n")

    # Generate random poses
    print("Generating random poses...")
    poses = [random_pose() for _ in range(exp_config.num_poses)]

    print("\nGenerated poses:")
    for i, (pos, quat) in enumerate(poses, 1):
        print(f"  Pose {i}: pos={pos}, quat={quat}")

    # Run experiments
    all_metrics = []
    start_time = time.time()

    for i, (position, rotation) in enumerate(poses, 1):
        metrics = run_single_pose_experiment(i, position, rotation, exp_config)
        all_metrics.append(metrics)

        # Save intermediate results
        save_metrics_csv(all_metrics, exp_config)

    total_time = time.time() - start_time

    # Print summary
    print_experiment_summary(all_metrics, total_time)

    return all_metrics


def save_metrics_csv(all_metrics: List[PoseMetrics], exp_config: ExperimentConfig):
    """Save metrics to CSV file with mean and std rows"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(
        exp_config.output_dir,
        f"experiment_{exp_config.selection_method}_{timestamp}.csv"
    )

    # Convert metrics to dictionaries
    rows = [m.to_dict() for m in all_metrics]

    if not rows:
        return

    # Calculate statistics for successful runs only
    success_metrics = [m for m in all_metrics if m.success]

    if success_metrics:
        # Create mean row
        mean_row = {'pose_id': 'MEAN'}
        mean_row['glass_pos_x'] = np.mean([m.glass_pos[0] for m in success_metrics])
        mean_row['glass_pos_y'] = np.mean([m.glass_pos[1] for m in success_metrics])
        mean_row['glass_pos_z'] = np.mean([m.glass_pos[2] for m in success_metrics])
        mean_row['glass_quat_w'] = np.mean([m.glass_quat[0] for m in success_metrics])
        mean_row['glass_quat_x'] = np.mean([m.glass_quat[1] for m in success_metrics])
        mean_row['glass_quat_y'] = np.mean([m.glass_quat[2] for m in success_metrics])
        mean_row['glass_quat_z'] = np.mean([m.glass_quat[3] for m in success_metrics])
        mean_row['ik_time_sec'] = np.mean([m.ik_time_sec for m in success_metrics])
        mean_row['plan_time_sec'] = np.mean([m.plan_time_sec for m in success_metrics])
        mean_row['ik_plan_total_sec'] = np.mean([m.ik_plan_total_sec for m in success_metrics])
        mean_row['ik_solutions_all'] = np.mean([m.ik_solutions_all for m in success_metrics])
        mean_row['ik_solutions_safe'] = np.mean([m.ik_solutions_safe for m in success_metrics])
        mean_row['collision_check_time_sec'] = np.mean([m.collision_check_time_sec for m in success_metrics])
        mean_row['replan_time_sec'] = np.mean([m.replan_time_sec for m in success_metrics])
        mean_row['waypoint_collisions'] = np.mean([m.waypoint_collisions for m in success_metrics])
        mean_row['segment_collisions'] = np.mean([m.segment_collisions for m in success_metrics])
        mean_row['total_collisions'] = np.mean([m.total_collisions for m in success_metrics])
        mean_row['reconfigurations'] = np.mean([m.reconfigurations for m in success_metrics])
        mean_row['replan_attempts'] = np.mean([m.replan_attempts for m in success_metrics])
        mean_row['trajectory_modified'] = sum([m.trajectory_modified for m in success_metrics])
        mean_row['success'] = f"{len(success_metrics)}/{len(all_metrics)}"
        mean_row['error_message'] = ''

        # Create std row
        std_row = {'pose_id': 'STD'}
        std_row['glass_pos_x'] = np.std([m.glass_pos[0] for m in success_metrics])
        std_row['glass_pos_y'] = np.std([m.glass_pos[1] for m in success_metrics])
        std_row['glass_pos_z'] = np.std([m.glass_pos[2] for m in success_metrics])
        std_row['glass_quat_w'] = np.std([m.glass_quat[0] for m in success_metrics])
        std_row['glass_quat_x'] = np.std([m.glass_quat[1] for m in success_metrics])
        std_row['glass_quat_y'] = np.std([m.glass_quat[2] for m in success_metrics])
        std_row['glass_quat_z'] = np.std([m.glass_quat[3] for m in success_metrics])
        std_row['ik_time_sec'] = np.std([m.ik_time_sec for m in success_metrics])
        std_row['plan_time_sec'] = np.std([m.plan_time_sec for m in success_metrics])
        std_row['ik_plan_total_sec'] = np.std([m.ik_plan_total_sec for m in success_metrics])
        std_row['ik_solutions_all'] = np.std([m.ik_solutions_all for m in success_metrics])
        std_row['ik_solutions_safe'] = np.std([m.ik_solutions_safe for m in success_metrics])
        std_row['collision_check_time_sec'] = np.std([m.collision_check_time_sec for m in success_metrics])
        std_row['replan_time_sec'] = np.std([m.replan_time_sec for m in success_metrics])
        std_row['waypoint_collisions'] = np.std([m.waypoint_collisions for m in success_metrics])
        std_row['segment_collisions'] = np.std([m.segment_collisions for m in success_metrics])
        std_row['total_collisions'] = np.std([m.total_collisions for m in success_metrics])
        std_row['reconfigurations'] = np.std([m.reconfigurations for m in success_metrics])
        std_row['replan_attempts'] = np.std([m.replan_attempts for m in success_metrics])
        std_row['trajectory_modified'] = ''
        std_row['success'] = ''
        std_row['error_message'] = ''

        # Add separator, mean and std rows
        separator_row = {key: '---' for key in rows[0].keys()}
        rows.append(separator_row)
        rows.append(mean_row)
        rows.append(std_row)

    # Write CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Metrics saved to: {csv_path}")


def print_experiment_summary(all_metrics: List[PoseMetrics], total_time: float):
    """Print summary statistics"""
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")

    num_success = sum(1 for m in all_metrics if m.success)
    num_total = len(all_metrics)

    print(f"Total poses: {num_total}")
    print(f"Successful: {num_success}")
    print(f"Failed: {num_total - num_success}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Calculate statistics for successful runs
    success_metrics = [m for m in all_metrics if m.success]

    if success_metrics:
        print(f"\nStatistics (successful runs only):")

        # Timing statistics
        avg_ik_time = np.mean([m.ik_time_sec for m in success_metrics])
        std_ik_time = np.std([m.ik_time_sec for m in success_metrics])

        avg_plan_time = np.mean([m.plan_time_sec for m in success_metrics])
        std_plan_time = np.std([m.plan_time_sec for m in success_metrics])

        avg_ik_plan_total = np.mean([m.ik_plan_total_sec for m in success_metrics])
        std_ik_plan_total = np.std([m.ik_plan_total_sec for m in success_metrics])

        avg_coll_time = np.mean([m.collision_check_time_sec for m in success_metrics])
        std_coll_time = np.std([m.collision_check_time_sec for m in success_metrics])

        avg_replan_time = np.mean([m.replan_time_sec for m in success_metrics])
        std_replan_time = np.std([m.replan_time_sec for m in success_metrics])

        # IK solution statistics
        avg_ik_all = np.mean([m.ik_solutions_all for m in success_metrics])
        std_ik_all = np.std([m.ik_solutions_all for m in success_metrics])

        avg_ik_safe = np.mean([m.ik_solutions_safe for m in success_metrics])
        std_ik_safe = np.std([m.ik_solutions_safe for m in success_metrics])

        # Collision and reconfiguration statistics
        avg_collisions = np.mean([m.total_collisions for m in success_metrics])
        std_collisions = np.std([m.total_collisions for m in success_metrics])

        avg_reconfigs = np.mean([m.reconfigurations for m in success_metrics])
        std_reconfigs = np.std([m.reconfigurations for m in success_metrics])

        avg_replan_attempts = np.mean([m.replan_attempts for m in success_metrics])
        std_replan_attempts = np.std([m.replan_attempts for m in success_metrics])

        num_traj_modified = sum([m.trajectory_modified for m in success_metrics])

        print(f"\n  Timing:")
        print(f"    IK time:                 {avg_ik_time:6.2f} ± {std_ik_time:5.2f} s")
        print(f"    Planning time:           {avg_plan_time:6.2f} ± {std_plan_time:5.2f} s")
        print(f"    IK + Planning total:     {avg_ik_plan_total:6.2f} ± {std_ik_plan_total:5.2f} s")
        print(f"    Collision check time:    {avg_coll_time:6.2f} ± {std_coll_time:5.2f} s")
        print(f"    Replanning time:         {avg_replan_time:6.2f} ± {std_replan_time:5.2f} s")

        print(f"\n  IK Solutions:")
        print(f"    All solutions:           {avg_ik_all:6.1f} ± {std_ik_all:5.1f}")
        print(f"    Safe solutions:          {avg_ik_safe:6.1f} ± {std_ik_safe:5.1f}")

        print(f"\n  Collisions & Reconfigurations:")
        print(f"    Total collisions:        {avg_collisions:6.1f} ± {std_collisions:5.1f}")
        print(f"    Reconfigurations:        {avg_reconfigs:6.1f} ± {std_reconfigs:5.1f}")

        print(f"\n  Replanning:")
        print(f"    Replanning attempts:     {avg_replan_attempts:6.1f} ± {std_replan_attempts:5.1f}")
        print(f"    Trajectories modified:   {num_traj_modified}/{len(success_metrics)}")

    # Print failed cases
    failed_metrics = [m for m in all_metrics if not m.success]
    if failed_metrics:
        print(f"\nFailed poses:")
        for m in failed_metrics:
            print(f"  Pose {m.pose_id}: {m.error_message}")

    print(f"{'='*70}\n")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run random pose experiment with full pipeline"
    )
    parser.add_argument(
        "--num_poses",
        type=int,
        default=10,
        help="Number of random poses to generate (default: 10)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dp",
        choices=["random", "greedy", "dp"],
        help="Trajectory planning method (default: dp)"
    )
    parser.add_argument(
        "--tsp_tour",
        type=str,
        default=None,
        help="Path to TSP tour file (if not specified, uses default)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Output directory for results (default: experiments/results)"
    )

    args = parser.parse_args()

    # Create experiment config
    exp_config = ExperimentConfig(
        num_poses=args.num_poses,
        selection_method=args.method,
        tsp_tour_path=args.tsp_tour,
        output_dir=args.output_dir,
    )

    # Run experiment
    all_metrics = run_experiment(exp_config)

    print("\n✓ Experiment complete!")


if __name__ == "__main__":
    main()
