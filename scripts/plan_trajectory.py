#!/usr/bin/env python3
"""
Plan Robot Trajectory from IK Solutions

This script:
1. Loads IK solutions from HDF5 file
2. Selects optimal joint configurations using DP/greedy/random methods
3. Saves joint trajectory to CSV file
4. Analyzes joint reconfigurations

This script does NOT require Isaac Sim - it's pure computation.

Usage:
    python scripts/plan_trajectory.py \\
        --ik_solutions data/ik/ik_solutions_3000.h5 \\
        --method dp \\
        --output data/trajectory/3000/joint_trajectory_dp.csv
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import csv
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Third Party Imports
# ============================================================================
import h5py
import numpy as np
import torch

# ============================================================================
# Local Imports
# ============================================================================
from common import config
from common.ik_utils import Viewpoint
from common.trajectory_planning import (
    select_ik_random,
    select_ik_greedy,
    select_ik_dp,
)
from analyze_joint_reconfigurations import (
    load_joint_trajectory,
    analyze_joint_reconfigurations,
    print_analysis_results,
)


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class PlanConfig:
    """Configuration for trajectory planning"""
    ik_solutions_path: str
    output_path: Optional[str]
    selection_method: str

    # Joint selection cost function parameters
    joint_weights: np.ndarray = field(default_factory=lambda: config.JOINT_WEIGHTS.copy())
    reconfig_threshold: float = config.RECONFIGURATION_THRESHOLD
    reconfig_penalty: float = config.RECONFIGURATION_PENALTY
    max_move_weight: float = config.MAX_MOVE_WEIGHT

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PlanConfig':
        """Create configuration from command line arguments"""
        return cls(
            ik_solutions_path=args.ik_solutions,
            output_path=args.output,
            selection_method=args.method,
        )


# ============================================================================
# File I/O
# ============================================================================
def load_ik_solutions(ik_solutions_path: str) -> tuple:
    """Load IK solutions from HDF5 file

    Returns:
        tuple of (viewpoints, metadata, tsp_tour_path)
    """
    print(f"\n{'='*60}")
    print("LOADING IK SOLUTIONS")
    print(f"{'='*60}")
    print(f"Input file: {ik_solutions_path}")

    viewpoints: List[Viewpoint] = []

    with h5py.File(ik_solutions_path, 'r') as f:
        # Load metadata
        metadata = dict(f['metadata'].attrs)
        tsp_tour_path = metadata.get('tsp_tour_file', 'unknown')

        print(f"Total viewpoints: {metadata['num_viewpoints']}")
        print(f"Viewpoints with solutions: {metadata['num_viewpoints_with_solutions']}")
        print(f"Viewpoints with safe solutions: {metadata['num_viewpoints_with_safe_solutions']}")
        print(f"Original TSP tour: {tsp_tour_path}")

        # Load viewpoints
        for key in f.keys():
            if not key.startswith('viewpoint_'):
                continue

            vp_grp = f[key]
            vp_index = vp_grp.attrs['original_index']

            # Load world pose
            world_pose = np.array(vp_grp['world_pose'])

            # Load all IK solutions
            all_ik_solutions = []
            if 'all_ik_solutions' in vp_grp:
                all_sols = np.array(vp_grp['all_ik_solutions'])
                if all_sols.shape[0] > 0:
                    all_ik_solutions = [sol for sol in all_sols]

            # Load collision-free mask and extract safe solutions
            safe_ik_solutions = []
            if 'collision_free_mask' in vp_grp:
                collision_free_mask = np.array(vp_grp['collision_free_mask'])
                safe_ik_solutions = [all_ik_solutions[i] for i in range(len(collision_free_mask))
                                    if collision_free_mask[i]]

            viewpoint = Viewpoint(
                index=vp_index,
                world_pose=world_pose if world_pose.shape == (4, 4) else None,
                all_ik_solutions=all_ik_solutions,
                safe_ik_solutions=safe_ik_solutions,
            )
            viewpoints.append(viewpoint)

    # Sort viewpoints by index
    viewpoints.sort(key=lambda vp: vp.index)

    print(f"Loaded {len(viewpoints)} viewpoints")
    print(f"{'='*60}\n")

    return viewpoints, metadata, tsp_tour_path


def extract_pose_from_matrix(
    pose_matrix: np.ndarray
) -> tuple:
    """Extract position and quaternion from 4x4 transformation matrix

    Returns:
        Tuple of (pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w)
    """
    from curobo.geom.transform import matrix_to_quaternion

    if pose_matrix.shape != (4, 4):
        raise ValueError("Pose matrix must be 4x4")

    # Extract position
    position = pose_matrix[:3, 3]
    pos_x, pos_y, pos_z = float(position[0]), float(position[1]), float(position[2])

    # Extract rotation and convert to quaternion
    rotation_matrix = pose_matrix[:3, :3]
    rot_tensor = torch.from_numpy(rotation_matrix.astype(np.float32)).unsqueeze(0)
    quat_tensor = matrix_to_quaternion(rot_tensor)  # Returns (w, x, y, z)

    # Convert to (x, y, z, w) order
    quat_w = float(quat_tensor[0, 0].item())
    quat_x = float(quat_tensor[0, 1].item())
    quat_y = float(quat_tensor[0, 2].item())
    quat_z = float(quat_tensor[0, 3].item())

    return pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w


def save_joint_trajectory_csv(
    viewpoints: List[Viewpoint],
    joint_targets: List[np.ndarray],
    save_path: str
):
    """Save joint trajectory and target poses to CSV file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    csv_rows = []
    skipped_count = 0

    count = min(len(viewpoints), len(joint_targets))

    for idx in range(count):
        viewpoint = viewpoints[idx]
        joint_config = joint_targets[idx]

        if viewpoint.world_pose is None:
            print(f"Warning: Viewpoint {viewpoint.index} has no world_pose, skipping")
            skipped_count += 1
            continue

        time = float(idx)
        joints = joint_config.flatten().tolist()

        if len(joints) != 6:
            print(f"Warning: Expected 6 joints but got {len(joints)}, skipping")
            skipped_count += 1
            continue

        try:
            pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w = extract_pose_from_matrix(
                viewpoint.world_pose
            )
        except Exception as e:
            print(f"Warning: Failed to extract pose: {e}, skipping")
            skipped_count += 1
            continue

        row = [time, *joints, pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        csv_rows.append(row)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
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
        writer.writerows(csv_rows)

    print(f"\n{'='*60}")
    print("JOINT TRAJECTORY CSV SAVED")
    print(f"{'='*60}")
    print(f"Output path: {save_path}")
    print(f"Total rows: {len(csv_rows)}")
    if skipped_count > 0:
        print(f"Rows skipped: {skipped_count}")
    print(f"{'='*60}\n")


def analyze_reconfigurations(csv_path: str, threshold: float, output_dir: str):
    """Analyze joint reconfigurations from CSV file"""
    try:
        joint_data, joint_names = load_joint_trajectory(csv_path)
        results = analyze_joint_reconfigurations(joint_data, threshold=threshold)
        print_analysis_results(results, joint_names)

        input_basename = os.path.splitext(os.path.basename(csv_path))[0]
        results_file = os.path.join(output_dir, f"{input_basename}_reconfig.txt")

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
                involved_joints = [joint_names[i].replace('ur20-', '').replace('_joint', '')
                                 for i in joint_indices]
                f.write(f"{reconfig['timestep']:<10} "
                       f"{reconfig['max_change']:<12.3f} "
                       f"{', '.join(involved_joints)}\n")

        print(f"\n{'='*60}")
        print("RECONFIGURATION ANALYSIS SAVED")
        print(f"{'='*60}")
        print(f"Output file: {results_file}")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"Warning: Failed to analyze reconfigurations: {e}")


# ============================================================================
# Trajectory Planning
# ============================================================================
def plan_trajectory(
    viewpoints: List[Viewpoint],
    cfg: PlanConfig,
    default_config: np.ndarray = None
) -> tuple:
    """Plan joint trajectory using selected method

    Returns:
        Tuple of (joint_targets, solution_indices)
    """
    print(f"\n{'='*60}")
    print(f"TRAJECTORY PLANNING ({cfg.selection_method.upper()})")
    print(f"{'='*60}\n")

    # Filter viewpoints with safe IK solutions
    viewpoints_with_safe = [vp for vp in viewpoints if len(vp.safe_ik_solutions) > 0]

    if not viewpoints_with_safe:
        raise ValueError("No viewpoints with safe IK solutions!")

    print(f"Planning with {len(viewpoints_with_safe)} viewpoints")

    # Select method
    if cfg.selection_method == "random":
        targets, solution_indices = select_ik_random(viewpoints_with_safe)
    elif cfg.selection_method == "greedy":
        targets, solution_indices = select_ik_greedy(viewpoints_with_safe)
    elif cfg.selection_method == "dp":
        # Use default retract config if not provided
        if default_config is None:
            default_config = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64)

        targets, _, solution_indices = select_ik_dp(
            viewpoints_with_safe,
            default_config,
            cfg.joint_weights,
            cfg.reconfig_threshold,
            cfg.reconfig_penalty,
            cfg.max_move_weight
        )
    else:
        raise ValueError(f"Unknown selection method: {cfg.selection_method}")

    print(f"\n✓ Trajectory planned with {len(targets)} waypoints")
    print(f"{'='*60}\n")

    return targets, solution_indices, viewpoints_with_safe


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Plan robot trajectory from IK solutions")
    parser.add_argument(
        "--ik_solutions",
        type=str,
        required=True,
        help="Path to IK solutions HDF5 file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for joint trajectory CSV (default: auto-generated)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dp",
        choices=["random", "greedy", "dp"],
        help="Selection method: 'random', 'greedy', or 'dp' (default: dp)"
    )
    args = parser.parse_args()

    cfg = PlanConfig.from_args(args)

    print(f"\n{'='*60}")
    print("PLAN TRAJECTORY")
    print(f"{'='*60}")
    print(f"IK solutions: {cfg.ik_solutions_path}")
    print(f"Selection method: {cfg.selection_method}")
    print(f"{'='*60}\n")

    # Load IK solutions
    viewpoints, metadata, tsp_tour_path = load_ik_solutions(cfg.ik_solutions_path)

    # Plan trajectory
    joint_targets, solution_indices, viewpoints_with_safe = plan_trajectory(viewpoints, cfg)

    # Determine output path
    if cfg.output_path is None:
        # Extract number of points from metadata or filename
        num_points = metadata.get('num_viewpoints', len(viewpoints))
        output_dir = f'data/trajectory/{num_points}'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/joint_trajectory_{cfg.selection_method}.csv'
    else:
        output_path = cfg.output_path
        output_dir = os.path.dirname(output_path)

    # Save joint trajectory CSV
    save_joint_trajectory_csv(viewpoints_with_safe, joint_targets, output_path)

    # Analyze reconfigurations
    analyze_reconfigurations(output_path, cfg.reconfig_threshold, output_dir)

    print("\n✓ Trajectory planning complete!")


if __name__ == "__main__":
    main()
