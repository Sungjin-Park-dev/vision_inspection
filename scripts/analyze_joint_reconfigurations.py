#!/usr/bin/env python3

"""
Joint Reconfiguration Analysis Tool

This script analyzes joint trajectory CSV files to count and visualize joint reconfigurations.
A joint reconfiguration is defined as a significant change in joint position between consecutive time steps.

Usage:
    python analyze_joint_reconfigurations.py [--trajectory path] [--threshold value] [--plot]

Author: Analysis Script for Vision Inspection System
"""

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


def load_joint_trajectory(csv_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load joint trajectory data from CSV file.

    Args:
        csv_path: Path to the joint trajectory CSV file

    Returns:
        Tuple of (joint_data, joint_names) where:
        - joint_data: numpy array of shape (n_timesteps, n_joints)
        - joint_names: list of joint column names
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    joint_data = []
    joint_names = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Extract joint column names (exclude time and target pose columns)
        all_columns = reader.fieldnames
        joint_names = [col for col in all_columns if col.startswith('ur20-') and col.endswith('_joint')]

        print(f"Found {len(joint_names)} joints: {joint_names}")

        # Read joint data
        for row in reader:
            joint_values = [float(row[joint_name]) for joint_name in joint_names]
            joint_data.append(joint_values)

    joint_data = np.array(joint_data)
    print(f"Loaded trajectory with {joint_data.shape[0]} time steps and {joint_data.shape[1]} joints")

    return joint_data, joint_names


def analyze_joint_reconfigurations(joint_data: np.ndarray, threshold: float = 0.1, exclude_last_joint: bool = False) -> Dict:
    """
    Analyze joint reconfigurations in the trajectory.

    Args:
        joint_data: Array of shape (n_timesteps, n_joints)
        threshold: Minimum joint change (radians) to count as reconfiguration
        exclude_last_joint: If True, exclude the last joint from reconfiguration analysis

    Returns:
        Dictionary containing analysis results
    """
    n_timesteps, n_joints = joint_data.shape

    # Calculate joint differences between consecutive time steps
    joint_diffs = np.diff(joint_data, axis=0)  # Shape: (n_timesteps-1, n_joints)
    joint_changes = np.abs(joint_diffs)

    # Create mask to exclude last joint if requested
    if exclude_last_joint:
        joint_mask = np.ones(n_joints, dtype=bool)
        joint_mask[-1] = False  # 마지막 joint 제외
        joint_changes_for_reconfig = joint_changes[:, joint_mask]
    else:
        joint_mask = np.ones(n_joints, dtype=bool)
        joint_changes_for_reconfig = joint_changes

    # Count reconfigurations for each joint
    reconfigurations_per_joint = np.sum(joint_changes > threshold, axis=0)

    # Count total reconfigurations (any joint exceeding threshold, excluding last joint if requested)
    total_reconfigurations = np.sum(np.any(joint_changes_for_reconfig > threshold, axis=1))

    # Calculate movement statistics
    max_changes_per_joint = np.max(joint_changes, axis=0)
    mean_changes_per_joint = np.mean(joint_changes, axis=0)
    total_movement_per_joint = np.sum(joint_changes, axis=0)

    # Find timesteps with large reconfigurations (excluding last joint if requested)
    large_reconfig_timesteps = []
    for i in range(joint_changes.shape[0]):
        if np.any(joint_changes_for_reconfig[i] > threshold):
            max_change = np.max(joint_changes_for_reconfig[i])
            # joint_mask를 사용하여 제외되지 않은 joint만 표시
            joints_involved = np.where((joint_changes[i] > threshold) & joint_mask)[0]
            large_reconfig_timesteps.append({
                'timestep': i + 1,  # +1 because diff removes first timestep
                'max_change': max_change,
                'joints_involved': joints_involved.tolist(),
                'changes': joint_changes[i].tolist()
            })

    return {
        'total_timesteps': n_timesteps,
        'total_reconfigurations': int(total_reconfigurations),
        'reconfiguration_rate': float(total_reconfigurations) / (n_timesteps - 1),
        'reconfigurations_per_joint': reconfigurations_per_joint.tolist(),
        'max_changes_per_joint': max_changes_per_joint.tolist(),
        'mean_changes_per_joint': mean_changes_per_joint.tolist(),
        'total_movement_per_joint': total_movement_per_joint.tolist(),
        'large_reconfig_timesteps': large_reconfig_timesteps,
        'threshold_used': threshold,
        'excluded_last_joint': exclude_last_joint
    }


def print_analysis_results(results: Dict, joint_names: List[str]) -> None:
    """
    Print detailed analysis results.

    Args:
        results: Analysis results dictionary
        joint_names: List of joint names
    """
    print(f"\n{'='*80}")
    print("JOINT RECONFIGURATION ANALYSIS RESULTS")
    print(f"{'='*80}")

    print(f"Threshold used: {results['threshold_used']:.3f} radians")
    if results.get('excluded_last_joint', False):
        print(f"Last joint excluded from reconfiguration analysis: Yes")
    print(f"Total timesteps: {results['total_timesteps']}")
    print(f"Total reconfigurations: {results['total_reconfigurations']}")
    print(f"Reconfiguration rate: {results['reconfiguration_rate']:.1%}")

    print(f"\n{'Per-Joint Statistics:':<30}")
    print(f"{'Joint Name':<25} {'Reconfigs':<10} {'Max Change':<12} {'Mean Change':<12} {'Total Move':<12}")
    print("-" * 80)

    for i, joint_name in enumerate(joint_names):
        short_name = joint_name.replace('ur20-', '').replace('_joint', '')
        print(f"{short_name:<25} "
              f"{results['reconfigurations_per_joint'][i]:<10} "
              f"{results['max_changes_per_joint'][i]:<12.3f} "
              f"{results['mean_changes_per_joint'][i]:<12.3f} "
              f"{results['total_movement_per_joint'][i]:<12.3f}")

    print(f"\n{'Large Reconfigurations (first 10):'}")
    print(f"{'Timestep':<10} {'Max Change':<12} {'Joints Involved':<20}")
    print("-" * 50)

    for reconfig in results['large_reconfig_timesteps'][:10]:
        joint_indices = reconfig['joints_involved']
        involved_joints = [joint_names[i].replace('ur20-', '').replace('_joint', '') for i in joint_indices]
        print(f"{reconfig['timestep']:<10} "
              f"{reconfig['max_change']:<12.3f} "
              f"{', '.join(involved_joints):<20}")

    if len(results['large_reconfig_timesteps']) > 10:
        print(f"... and {len(results['large_reconfig_timesteps']) - 10} more")

    print(f"{'='*80}\n")


def plot_analysis_results(joint_data: np.ndarray, joint_names: List[str],
                         results: Dict, save_path: str = None) -> None:
    """
    Create visualization plots for joint trajectory analysis.

    Args:
        joint_data: Array of shape (n_timesteps, n_joints)
        joint_names: List of joint names
        results: Analysis results dictionary
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Joint Trajectory Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Joint trajectories over time
    ax1 = axes[0, 0]
    timesteps = np.arange(joint_data.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_names)))

    for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
        short_name = joint_name.replace('ur20-', '').replace('_joint', '')
        ax1.plot(timesteps, joint_data[:, i], label=short_name, color=color, alpha=0.8)

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Joint Position (radians)')
    ax1.set_title('Joint Trajectories Over Time')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Joint changes (differences) over time
    ax2 = axes[0, 1]
    joint_diffs = np.abs(np.diff(joint_data, axis=0))

    for i, (joint_name, color) in enumerate(zip(joint_names, colors)):
        short_name = joint_name.replace('ur20-', '').replace('_joint', '')
        ax2.plot(timesteps[1:], joint_diffs[:, i], label=short_name, color=color, alpha=0.8)

    ax2.axhline(y=results['threshold_used'], color='red', linestyle='--',
                label=f'Threshold ({results["threshold_used"]:.3f})')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Joint Change (radians)')
    ax2.set_title('Joint Changes Between Time Steps')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Plot 3: Reconfiguration count per joint (bar chart)
    ax3 = axes[1, 0]
    short_names = [name.replace('ur20-', '').replace('_joint', '') for name in joint_names]
    bars = ax3.bar(short_names, results['reconfigurations_per_joint'], color=colors)
    ax3.set_xlabel('Joint')
    ax3.set_ylabel('Number of Reconfigurations')
    ax3.set_title('Reconfigurations per Joint')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar, count in zip(bars, results['reconfigurations_per_joint']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')

    # Plot 4: Cumulative reconfigurations over time
    ax4 = axes[1, 1]
    joint_changes = np.abs(np.diff(joint_data, axis=0))
    reconfig_events = np.any(joint_changes > results['threshold_used'], axis=1)
    cumulative_reconfigs = np.cumsum(reconfig_events)

    ax4.plot(timesteps[1:], cumulative_reconfigs, 'b-', linewidth=2)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Cumulative Reconfigurations')
    ax4.set_title('Cumulative Reconfigurations Over Time')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze joint reconfigurations in robot trajectory CSV files"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default="data/output/joint_trajectory.csv",
        help="Path to input CSV file (default: data/output/joint_trajectory.csv)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for joint change to count as reconfiguration in radians (default: 0.1)"
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
        help="Directory to save analysis results (default: auto-generated from input filename)"
    )
    parser.add_argument(
        "--exclude-last-joint",
        action="store_true",
        help="Exclude the last joint from reconfiguration analysis"
    )

    args = parser.parse_args()

    # Generate output directory from input filename if not specified
    if args.output_dir is None:
        output_dir = os.path.dirname(args.trajectory)
    else:
        output_dir = args.output_dir

    # Load trajectory data
    try:
        joint_data, joint_names = load_joint_trajectory(args.trajectory)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return 1

    # Analyze reconfigurations
    results = analyze_joint_reconfigurations(joint_data, threshold=args.threshold, exclude_last_joint=args.exclude_last_joint)

    # Print results
    print_analysis_results(results, joint_names)

    # Generate output filenames from input filename
    input_basename = os.path.splitext(os.path.basename(args.trajectory))[0]

    # Generate plots if requested
    if args.plot:
        plot_save_path = os.path.join(output_dir, f"{input_basename}_reconfig.png")
        plot_analysis_results(joint_data, joint_names, results, save_path=plot_save_path)

    # Save detailed results to file
    results_file = os.path.join(output_dir, f"{input_basename}_reconfig.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(results_file, 'w') as f:
        f.write("Joint Reconfiguration Analysis Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"Input file: {args.trajectory}\n")
        f.write(f"Threshold: {args.threshold:.3f} radians\n")
        if results.get('excluded_last_joint', False):
            f.write(f"Last joint excluded from reconfiguration analysis: Yes\n")
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
            involved_joints = [joint_names[i].replace('ur20-', '').replace('_joint', '') for i in joint_indices]
            f.write(f"{reconfig['timestep']:<10} "
                   f"{reconfig['max_change']:<12.3f} "
                   f"{', '.join(involved_joints)}\n")

    print(f"Detailed results saved to: {results_file}")
    print(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())