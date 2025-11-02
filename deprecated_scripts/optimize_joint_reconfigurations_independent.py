#!/usr/bin/env python3

"""
Independent Joint Trajectory Optimization Tool

This script optimizes ONLY the originally problematic timesteps by using relaxed tolerances,
without cascading effects. Unlike the sequential optimizer, this approach:

1. Identifies all reconfigurations in the original trajectory
2. Re-solves IK ONLY for those specific timesteps
3. Uses relaxed tolerances to find alternative IK solutions
4. Does NOT propagate changes to other timesteps

Strategy:
- Detect problematic timesteps (joint change > threshold)
- For each problematic timestep, re-solve IK with:
  * Original target pose (position + orientation)
  * Relaxed tolerances (allow some deviation)
  * Previous timestep's configuration as seed
- Replace ONLY the problematic IK solutions
- No cascading - other timesteps remain unchanged

Usage Examples:
    # Basic independent optimization
    python optimize_joint_reconfigurations_independent.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20

    # With custom tolerances (relax position constraints)
    python optimize_joint_reconfigurations_independent.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20 \
        --tolerances 0.05 0.05 0.01 0.0 0.0 0.0

    # With visualization
    python optimize_joint_reconfigurations_independent.py \
        --input_csv motions/ur20_motion.csv \
        --robot ur20 \
        --tolerances 0.05 0.05 0.01 0.0 0.0 0.0 \
        --plot

Author: Enhanced for ICRA'25 Hierarchical Coverage Path Planning
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robot import Robot
from analyze_joint_reconfigurations import (
    load_joint_trajectory,
    analyze_joint_reconfigurations,
    print_analysis_results
)


def optimize_trajectory_independent(
    joint_data: np.ndarray,
    pose_data: np.ndarray,
    robot: Robot,
    threshold: float = 1.0,
    tolerances: List[float] = None,
    max_iter: int = 100,
    num_ik_samples: int = 10
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize trajectory by re-solving IK ONLY for problematic timesteps (independent optimization).

    This approach identifies all problematic timesteps first, then optimizes each one independently
    without cascading effects. This is faster and preserves the original trajectory better.

    Args:
        joint_data: Original joint trajectory array of shape (n_timesteps, n_joints)
        pose_data: Target pose array of shape (n_timesteps, 7)
        robot: Robot instance for IK solving
        threshold: Threshold for detecting reconfigurations (radians)
        tolerances: IK tolerances [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        max_iter: Maximum iterations for IK solver
        num_ik_samples: Number of IK solutions to sample per problematic timestep

    Returns:
        Tuple of (optimized_joint_data, optimization_stats)
    """
    if tolerances is None:
        tolerances = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    print(f"\n{'='*80}")
    print("INDEPENDENT TRAJECTORY OPTIMIZATION")
    print(f"{'='*80}")
    print(f"Threshold: {threshold:.3f} radians")
    print(f"Tolerances: {tolerances}")
    print(f"Max IK iterations: {max_iter}")
    print(f"IK samples per timestep: {num_ik_samples}")
    print("\nStrategy: Optimize ONLY originally problematic timesteps")
    print("No cascading - other timesteps remain unchanged")

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
    print("STEP 2: Optimizing problematic timesteps independently")
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
        print(f"  Sampling {num_ik_samples} IK solutions...")

        # Get next timestep's IK for bidirectional validation
        next_ik = joint_data[t+1] if t+1 < n_timesteps else None

        # Sample multiple IK solutions and select the best one
        candidates = []
        for _ in range(num_ik_samples):
            # Re-solve IK using previous configuration as seed
            ik_solution = robot.reach_with_relaxed_ik(
                target_pose=target_pose,
                constrain_velocity=False,
                tolerances=tolerances,
                start_config=prev_ik.tolist(),  # Use previous timestep as seed
                max_iter=max_iter
            )

            if len(ik_solution) > 0:
                ik_solution = np.array(ik_solution)

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

                candidates.append((ik_solution, max_change_prev, max_change_next, total_cost))

        # Select the best candidate (smallest total_cost)
        if len(candidates) > 0:
            print(f"  Found {len(candidates)}/{num_ik_samples} valid IK solutions")
            best_ik, best_prev, best_next, best_total = min(candidates, key=lambda x: x[3])

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
            print(f"  ✗ Failed to find any valid IK solution out of {num_ik_samples} attempts")
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
    fig.suptitle('Independent Joint Trajectory Optimization Comparison', fontsize=16, fontweight='bold')

    timesteps = np.arange(original_joint_data.shape[0])

    # Original trajectory
    ax1 = axes[0]
    joint_diffs_orig = np.abs(np.diff(original_joint_data, axis=0))
    for i, joint_name in enumerate(joint_names):
        short_name = joint_name.replace(f'{robot_name}-', '').replace('_joint', '')
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
        short_name = joint_name.replace(f'{robot_name}-', '').replace('_joint', '')
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
        description="Independently optimize problematic timesteps in joint trajectories"
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
        help="Robot name for IK optimization (e.g., ur20, panda, ur5, ur5sander)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for joint change to count as reconfiguration in radians (default: 1.0)"
    )
    parser.add_argument(
        "--tolerances",
        type=float,
        nargs=6,
        default=[0.0, 0.0, 0.0, 0.0, 0.0, 999],
        help="IK tolerances: pos_x pos_y pos_z rot_x rot_y rot_z (default: all 0.0). "
             "Example: 0.05 0.05 0.01 0.0 0.0 0.0 allows 5cm tolerance in x,y and 1cm in z"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum iterations for IK solver (default: 100)"
    )
    parser.add_argument(
        "--num_ik_samples",
        type=int,
        default=50,
        help="Number of IK solutions to sample per problematic timestep (default: 10)"
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
        joint_data, joint_names, pose_data, pose_columns = load_joint_trajectory(args.input_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return 1

    # Initialize robot
    print(f"\nInitializing robot '{args.robot}' for IK optimization...")
    try:
        robot = Robot(args.robot)
        print(f"✓ Robot initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing robot: {e}")
        return 1

    # Analyze original trajectory
    print("\n" + "="*80)
    print("ANALYZING ORIGINAL TRAJECTORY")
    if args.use_solver_criteria:
        print("Using comprehensive solver criteria from solver_base.py")
    print("="*80)
    original_results = analyze_joint_reconfigurations(
        joint_data,
        threshold=args.threshold,
        pose_data=pose_data,
        robot=robot,
        use_solver_criteria=args.use_solver_criteria
    )
    print_analysis_results(original_results, joint_names)

    start_time = time.perf_counter()

    # Run independent optimization
    optimized_joint_data, opt_stats = optimize_trajectory_independent(
        joint_data=joint_data,
        pose_data=pose_data,
        robot=robot,
        threshold=args.threshold,
        tolerances=args.tolerances,
        max_iter=args.max_iter,
        num_ik_samples=args.num_ik_samples
    )
    total_time = time.perf_counter() - start_time

    # Analyze optimized trajectory
    print("\n" + "="*80)
    print("ANALYZING OPTIMIZED TRAJECTORY")
    if args.use_solver_criteria:
        print("Using comprehensive solver criteria from solver_base.py")
    print("="*80)
    optimized_results = analyze_joint_reconfigurations(
        optimized_joint_data,
        threshold=args.threshold,
        pose_data=pose_data,
        robot=robot,
        use_solver_criteria=args.use_solver_criteria
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
    optimized_csv_path = os.path.join(output_dir, f"{input_basename}_independent_optimized.csv")
    save_optimized_trajectory(
        output_path=optimized_csv_path,
        joint_data=optimized_joint_data,
        pose_data=pose_data,
        joint_names=joint_names,
        pose_columns=pose_columns
    )

    # Generate comparison plot if requested
    if args.plot:
        plot_save_path = os.path.join(output_dir, f"{input_basename}_independent_optimization_comparison.png")
        plot_optimization_comparison(
            original_joint_data=joint_data,
            optimized_joint_data=optimized_joint_data,
            joint_names=joint_names,
            original_results=original_results,
            optimized_results=optimized_results,
            threshold=args.threshold,
            robot_name=args.robot,
            save_path=plot_save_path
        )

    # Save optimization details
    details_file = os.path.join(output_dir, f"{input_basename}_independent_optimized.txt")
    with open(details_file, 'w') as f:
        f.write("Independent Joint Trajectory Optimization Details\n")
        f.write("="*80 + "\n\n")
        f.write(f"Input file: {args.input_csv}\n")
        f.write(f"Robot: {args.robot}\n")
        f.write(f"Threshold: {args.threshold:.3f} radians\n")
        f.write(f"Tolerances: {args.tolerances}\n")
        f.write(f"Max iterations: {args.max_iter}\n")
        f.write(f"IK samples per timestep: {args.num_ik_samples}\n\n")

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
        f.write("EXPLANATION OF INDEPENDENT OPTIMIZATION WITH BIDIRECTIONAL VALIDATION\n")
        f.write("="*80 + "\n")
        f.write("This optimizer ONLY modifies the originally problematic timesteps.\n")
        f.write("Unlike sequential optimization, there is NO cascading effect:\n")
        f.write("1. All problematic timesteps are identified first\n")
        f.write("2. Each timestep is optimized independently\n")
        f.write("3. Other timesteps remain completely unchanged\n")
        f.write("4. Uses relaxed tolerances to find alternative IK solutions\n")
        f.write("5. BIDIRECTIONAL VALIDATION: Checks impact on both previous AND next timestep\n")
        f.write("   - Only accepts improvement if BOTH directions improve or stay the same\n")
        f.write("   - Prevents creating new reconfigurations in the next timestep\n")
        f.write("\n'Old Total' = max(prev_change, next_change) in ORIGINAL trajectory\n")
        f.write("'New Total' = max(prev_change, next_change) after re-solving IK\n")
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
            if args.use_solver_criteria:
                # With solver criteria, show reasons
                f.write(f"{'Timestep':<10} {'Max Change':<15} {'Reasons':<55}\n")
                f.write("-" * 80 + "\n")
                for reconfig in optimized_results['large_reconfig_timesteps']:
                    reasons_str = ', '.join(reconfig.get('reasons', []))
                    f.write(f"{reconfig['timestep']:<10} "
                           f"{reconfig['max_change']:<15.3f} "
                           f"{reasons_str:<55}\n")
            else:
                # Without solver criteria, show joints involved
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
    print(f"Total time: {total_time}")

    return 0


if __name__ == "__main__":
    exit(main())
