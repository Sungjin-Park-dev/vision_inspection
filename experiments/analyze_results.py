#!/usr/bin/env python3
"""
Analyze Random Pose Experiment Results

This script loads the experiment CSV and provides statistical analysis
and visualization of the results.

Usage:
    python experiments/analyze_results.py experiments/results/experiment_dp_20250117_143022.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_results(csv_path: str) -> pd.DataFrame:
    """Load experiment results from CSV"""
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pose experiments from {csv_path}")
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY STATISTICS")
    print("=" * 70)

    total_poses = len(df)
    successful_poses = df['success'].sum()
    failed_poses = total_poses - successful_poses

    print(f"\nOverall:")
    print(f"  Total poses: {total_poses}")
    print(f"  Successful: {successful_poses} ({successful_poses/total_poses*100:.1f}%)")
    print(f"  Failed: {failed_poses} ({failed_poses/total_poses*100:.1f}%)")

    # Filter successful runs for detailed stats
    success_df = df[df['success'] == True]

    if len(success_df) == 0:
        print("\nNo successful runs to analyze.")
        return

    print(f"\n--- Timing Statistics (successful runs only) ---")
    timing_cols = ['ik_time_sec', 'plan_time_sec', 'collision_check_time_sec']
    timing_stats = success_df[timing_cols].describe()
    print(timing_stats)

    print(f"\nTotal pipeline time per pose:")
    success_df['total_time_sec'] = (
        success_df['ik_time_sec'] +
        success_df['plan_time_sec'] +
        success_df['collision_check_time_sec']
    )
    print(f"  Mean: {success_df['total_time_sec'].mean():.2f}s")
    print(f"  Std:  {success_df['total_time_sec'].std():.2f}s")
    print(f"  Min:  {success_df['total_time_sec'].min():.2f}s")
    print(f"  Max:  {success_df['total_time_sec'].max():.2f}s")

    print(f"\n--- IK Solution Statistics ---")
    ik_cols = ['ik_solutions_all', 'ik_solutions_safe']
    ik_stats = success_df[ik_cols].describe()
    print(ik_stats)

    print(f"\nSafe solution rate:")
    success_df['safe_rate'] = success_df['ik_solutions_safe'] / success_df['ik_solutions_all']
    print(f"  Mean: {success_df['safe_rate'].mean()*100:.1f}%")
    print(f"  Std:  {success_df['safe_rate'].std()*100:.1f}%")

    print(f"\n--- Collision Statistics ---")
    collision_cols = ['waypoint_collisions', 'segment_collisions', 'total_collisions']
    collision_stats = success_df[collision_cols].describe()
    print(collision_stats)

    print(f"\n--- Reconfiguration Statistics ---")
    reconfig_stats = success_df['reconfigurations'].describe()
    print(reconfig_stats)

    # Pose variation analysis
    print(f"\n--- Pose Variation ---")
    print(f"Glass position (X, Y, Z):")
    print(f"  X: {success_df['glass_pos_x'].min():.4f} to {success_df['glass_pos_x'].max():.4f} m")
    print(f"  Y: {success_df['glass_pos_y'].min():.4f} to {success_df['glass_pos_y'].max():.4f} m")
    print(f"  Z: {success_df['glass_pos_z'].min():.4f} to {success_df['glass_pos_z'].max():.4f} m")

    print(f"\nGlass rotation (quaternion w, x, y, z):")
    print(f"  W: {success_df['glass_quat_w'].min():.4f} to {success_df['glass_quat_w'].max():.4f}")
    print(f"  X: {success_df['glass_quat_x'].min():.4f} to {success_df['glass_quat_x'].max():.4f}")
    print(f"  Y: {success_df['glass_quat_y'].min():.4f} to {success_df['glass_quat_y'].max():.4f}")
    print(f"  Z: {success_df['glass_quat_z'].min():.4f} to {success_df['glass_quat_z'].max():.4f}")


def print_failure_analysis(df: pd.DataFrame):
    """Analyze and print failure cases"""
    failed_df = df[df['success'] == False]

    if len(failed_df) == 0:
        print("\n✓ No failures!")
        return

    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)

    print(f"\nFailed poses: {len(failed_df)}/{len(df)}")

    for idx, row in failed_df.iterrows():
        print(f"\nPose {row['pose_id']}:")
        print(f"  Position: [{row['glass_pos_x']:.4f}, {row['glass_pos_y']:.4f}, {row['glass_pos_z']:.4f}]")
        print(f"  Rotation: [{row['glass_quat_w']:.4f}, {row['glass_quat_x']:.4f}, "
              f"{row['glass_quat_y']:.4f}, {row['glass_quat_z']:.4f}]")
        print(f"  Error: {row['error_message']}")


def print_best_worst_cases(df: pd.DataFrame):
    """Print best and worst performing cases"""
    success_df = df[df['success'] == True]

    if len(success_df) == 0:
        return

    print("\n" + "=" * 70)
    print("BEST & WORST CASES")
    print("=" * 70)

    # Best: least collisions
    best_collision = success_df.loc[success_df['total_collisions'].idxmin()]
    print(f"\nBest (least collisions): Pose {best_collision['pose_id']}")
    print(f"  Total collisions: {best_collision['total_collisions']}")
    print(f"  Position: [{best_collision['glass_pos_x']:.4f}, "
          f"{best_collision['glass_pos_y']:.4f}, {best_collision['glass_pos_z']:.4f}]")

    # Worst: most collisions
    worst_collision = success_df.loc[success_df['total_collisions'].idxmax()]
    print(f"\nWorst (most collisions): Pose {worst_collision['pose_id']}")
    print(f"  Total collisions: {worst_collision['total_collisions']}")
    print(f"  Position: [{worst_collision['glass_pos_x']:.4f}, "
          f"{worst_collision['glass_pos_y']:.4f}, {worst_collision['glass_pos_z']:.4f}]")

    # Fastest
    success_df['total_time'] = (
        success_df['ik_time_sec'] +
        success_df['plan_time_sec'] +
        success_df['collision_check_time_sec']
    )
    fastest = success_df.loc[success_df['total_time'].idxmin()]
    print(f"\nFastest: Pose {fastest['pose_id']}")
    print(f"  Total time: {fastest['total_time']:.2f}s")

    # Slowest
    slowest = success_df.loc[success_df['total_time'].idxmax()]
    print(f"\nSlowest: Pose {slowest['pose_id']}")
    print(f"  Total time: {slowest['total_time']:.2f}s")


def export_summary_report(df: pd.DataFrame, output_path: str):
    """Export a text summary report"""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RANDOM POSE EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total poses: {len(df)}\n")
        f.write(f"Successful: {df['success'].sum()}\n")
        f.write(f"Failed: {(~df['success']).sum()}\n\n")

        success_df = df[df['success'] == True]

        if len(success_df) > 0:
            f.write("--- Timing Statistics ---\n")
            f.write(f"IK computation time: {success_df['ik_time_sec'].mean():.2f} ± "
                   f"{success_df['ik_time_sec'].std():.2f}s\n")
            f.write(f"Planning time: {success_df['plan_time_sec'].mean():.2f} ± "
                   f"{success_df['plan_time_sec'].std():.2f}s\n")
            f.write(f"Collision check time: {success_df['collision_check_time_sec'].mean():.2f} ± "
                   f"{success_df['collision_check_time_sec'].std():.2f}s\n\n")

            f.write("--- IK Statistics ---\n")
            f.write(f"All IK solutions: {success_df['ik_solutions_all'].mean():.1f} ± "
                   f"{success_df['ik_solutions_all'].std():.1f}\n")
            f.write(f"Safe IK solutions: {success_df['ik_solutions_safe'].mean():.1f} ± "
                   f"{success_df['ik_solutions_safe'].std():.1f}\n\n")

            f.write("--- Collision Statistics ---\n")
            f.write(f"Waypoint collisions: {success_df['waypoint_collisions'].mean():.1f} ± "
                   f"{success_df['waypoint_collisions'].std():.1f}\n")
            f.write(f"Segment collisions: {success_df['segment_collisions'].mean():.1f} ± "
                   f"{success_df['segment_collisions'].std():.1f}\n")
            f.write(f"Total collisions: {success_df['total_collisions'].mean():.1f} ± "
                   f"{success_df['total_collisions'].std():.1f}\n\n")

            f.write("--- Reconfiguration Statistics ---\n")
            f.write(f"Reconfigurations: {success_df['reconfigurations'].mean():.1f} ± "
                   f"{success_df['reconfigurations'].std():.1f}\n")

    print(f"\n✓ Summary report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze random pose experiment results"
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to experiment results CSV file"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export summary report to text file"
    )

    args = parser.parse_args()

    # Load results
    df = load_results(args.csv_file)

    # Print analyses
    print_summary_statistics(df)
    print_failure_analysis(df)
    print_best_worst_cases(df)

    # Export if requested
    if args.export:
        export_summary_report(df, args.export)

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
