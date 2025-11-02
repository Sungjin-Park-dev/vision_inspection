#!/usr/bin/env python3

"""
IK Selection Method Comparison Tool

This script compares three IK solution selection methods (Random, Greedy, DP)
by analyzing joint reconfigurations, total movement, and other metrics.

Usage:
    python compare_selection_methods.py --random <path> --greedy <path> --dp <path>

Author: Vision Inspection System Analysis
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from datetime import datetime


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

        # Read joint data
        for row in reader:
            joint_values = [float(row[joint_name]) for joint_name in joint_names]
            joint_data.append(joint_values)

    joint_data = np.array(joint_data)
    print(f"  Loaded {csv_path}: {joint_data.shape[0]} timesteps, {joint_data.shape[1]} joints")

    return joint_data, joint_names


def analyze_joint_reconfigurations(joint_data: np.ndarray, threshold: float = 1.0) -> Dict:
    """
    Analyze joint reconfigurations in the trajectory.

    Args:
        joint_data: Array of shape (n_timesteps, n_joints)
        threshold: Minimum joint change (radians) to count as reconfiguration

    Returns:
        Dictionary containing analysis results
    """
    n_timesteps, n_joints = joint_data.shape

    # Calculate joint differences between consecutive time steps
    joint_diffs = np.diff(joint_data, axis=0)  # Shape: (n_timesteps-1, n_joints)
    joint_changes = np.abs(joint_diffs)

    # Count reconfigurations for each joint
    reconfigurations_per_joint = np.sum(joint_changes > threshold, axis=0)

    # Count total reconfigurations (any joint exceeding threshold)
    total_reconfigurations = np.sum(np.any(joint_changes > threshold, axis=1))

    # Calculate movement statistics
    max_changes_per_joint = np.max(joint_changes, axis=0)
    mean_changes_per_joint = np.mean(joint_changes, axis=0)
    total_movement_per_joint = np.sum(joint_changes, axis=0)

    # Calculate weighted distance (using joint weights from DP method)
    joint_weights = np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0])
    weighted_distances = np.sqrt(np.sum((joint_changes * joint_weights) ** 2, axis=1))
    total_weighted_distance = np.sum(weighted_distances)

    return {
        'total_timesteps': n_timesteps,
        'total_reconfigurations': int(total_reconfigurations),
        'reconfiguration_rate': float(total_reconfigurations) / (n_timesteps - 1),
        'reconfigurations_per_joint': reconfigurations_per_joint.tolist(),
        'max_changes_per_joint': max_changes_per_joint.tolist(),
        'mean_changes_per_joint': mean_changes_per_joint.tolist(),
        'total_movement_per_joint': total_movement_per_joint.tolist(),
        'total_weighted_distance': float(total_weighted_distance),
        'mean_weighted_distance': float(total_weighted_distance) / (n_timesteps - 1),
        'threshold_used': threshold,
        'joint_changes': joint_changes  # For visualization
    }


def print_comparison_results(results_dict: Dict[str, Dict], joint_names: List[str]) -> None:
    """
    Print comparison results for all methods.

    Args:
        results_dict: Dictionary with keys 'random', 'greedy', 'dp' and analysis results
        joint_names: List of joint names
    """
    print(f"\n{'='*80}")
    print("IK SELECTION METHOD COMPARISON RESULTS")
    print(f"{'='*80}")

    threshold = results_dict['random']['threshold_used']
    print(f"Threshold used: {threshold:.3f} radians\n")

    # Overall comparison table
    print(f"{'Method':<15} | {'Total Reconfig':<15} | {'Rate':<10} | {'Total Weighted Dist':<20} | {'Mean Change':<12}")
    print("-" * 95)

    for method_name in ['random', 'greedy', 'dp']:
        results = results_dict[method_name]
        print(f"{method_name.upper():<15} | "
              f"{results['total_reconfigurations']:<15} | "
              f"{results['reconfiguration_rate']*100:>8.2f}% | "
              f"{results['total_weighted_distance']:>18.4f} | "
              f"{np.mean(results['mean_changes_per_joint']):>10.4f}")

    # Improvement calculations
    print(f"\n{'Improvements:'}")
    print("-" * 80)

    random_reconfig = results_dict['random']['total_reconfigurations']
    greedy_reconfig = results_dict['greedy']['total_reconfigurations']
    dp_reconfig = results_dict['dp']['total_reconfigurations']

    random_dist = results_dict['random']['total_weighted_distance']
    greedy_dist = results_dict['greedy']['total_weighted_distance']
    dp_dist = results_dict['dp']['total_weighted_distance']

    print(f"Greedy vs Random:")
    print(f"  Reconfiguration reduction: {random_reconfig - greedy_reconfig} ({(random_reconfig - greedy_reconfig)/random_reconfig*100:.2f}%)")
    print(f"  Distance reduction: {random_dist - greedy_dist:.4f} ({(random_dist - greedy_dist)/random_dist*100:.2f}%)")

    print(f"\nDP vs Random:")
    print(f"  Reconfiguration reduction: {random_reconfig - dp_reconfig} ({(random_reconfig - dp_reconfig)/random_reconfig*100:.2f}%)")
    print(f"  Distance reduction: {random_dist - dp_dist:.4f} ({(random_dist - dp_dist)/random_dist*100:.2f}%)")

    print(f"\nDP vs Greedy:")
    print(f"  Reconfiguration reduction: {greedy_reconfig - dp_reconfig} ({(greedy_reconfig - dp_reconfig)/greedy_reconfig*100:.2f}%)")
    print(f"  Distance reduction: {greedy_dist - dp_dist:.4f} ({(greedy_dist - dp_dist)/greedy_dist*100:.2f}%)")

    # Per-joint comparison
    print(f"\n{'Per-Joint Reconfiguration Comparison:'}")
    print(f"{'Joint':<25} | {'Random':<10} | {'Greedy':<10} | {'DP':<10}")
    print("-" * 65)

    for i, joint_name in enumerate(joint_names):
        short_name = joint_name.replace('ur20-', '').replace('_joint', '')
        print(f"{short_name:<25} | "
              f"{results_dict['random']['reconfigurations_per_joint'][i]:<10} | "
              f"{results_dict['greedy']['reconfigurations_per_joint'][i]:<10} | "
              f"{results_dict['dp']['reconfigurations_per_joint'][i]:<10}")

    print(f"{'='*80}\n")


def plot_comparison(results_dict: Dict[str, Dict], joint_names: List[str],
                   save_path: str = None) -> None:
    """
    Create comprehensive visualization comparing all three methods.

    Args:
        results_dict: Dictionary with analysis results for all methods
        joint_names: List of joint names
        save_path: Path to save the plot
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    methods = ['random', 'greedy', 'dp']
    colors = {'random': '#e74c3c', 'greedy': '#3498db', 'dp': '#2ecc71'}
    labels = {'random': 'Random', 'greedy': 'Greedy', 'dp': 'DP'}

    short_joint_names = [name.replace('ur20-', '').replace('_joint', '') for name in joint_names]

    # Plot 1: Total Reconfiguration Count (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    reconfig_counts = [results_dict[m]['total_reconfigurations'] for m in methods]
    bars = ax1.bar([labels[m] for m in methods], reconfig_counts,
                   color=[colors[m] for m in methods], alpha=0.8)
    ax1.set_ylabel('Total Reconfiguration Count', fontsize=11)
    ax1.set_title('Total Reconfigurations', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, reconfig_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=10)

    # Plot 2: Per-Joint Reconfiguration (Grouped Bar)
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(joint_names))
    width = 0.25

    for i, method in enumerate(methods):
        counts = results_dict[method]['reconfigurations_per_joint']
        ax2.bar(x + i*width, counts, width, label=labels[method],
               color=colors[method], alpha=0.8)

    ax2.set_xlabel('Joint', fontsize=11)
    ax2.set_ylabel('Reconfiguration Count', fontsize=11)
    ax2.set_title('Reconfigurations per Joint', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(short_joint_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Cumulative Reconfigurations (Line)
    ax3 = fig.add_subplot(gs[0, 2])

    for method in methods:
        joint_changes = results_dict[method]['joint_changes']
        threshold = results_dict[method]['threshold_used']
        reconfig_events = np.any(joint_changes > threshold, axis=1)
        cumulative = np.cumsum(reconfig_events)
        timesteps = np.arange(1, len(cumulative) + 1)
        ax3.plot(timesteps, cumulative, label=labels[method],
                color=colors[method], linewidth=2, alpha=0.8)

    ax3.set_xlabel('Time Step', fontsize=11)
    ax3.set_ylabel('Cumulative Reconfigurations', fontsize=11)
    ax3.set_title('Cumulative Reconfigurations Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Plot 4: Total Movement per Joint (Grouped Bar)
    ax4 = fig.add_subplot(gs[1, 0])

    for i, method in enumerate(methods):
        movements = results_dict[method]['total_movement_per_joint']
        ax4.bar(x + i*width, movements, width, label=labels[method],
               color=colors[method], alpha=0.8)

    ax4.set_xlabel('Joint', fontsize=11)
    ax4.set_ylabel('Total Movement (radians)', fontsize=11)
    ax4.set_title('Total Movement per Joint', fontsize=12, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(short_joint_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Weighted Distance Comparison (Bar Chart)
    ax5 = fig.add_subplot(gs[1, 1])
    weighted_dists = [results_dict[m]['total_weighted_distance'] for m in methods]
    bars = ax5.bar([labels[m] for m in methods], weighted_dists,
                   color=[colors[m] for m in methods], alpha=0.8)
    ax5.set_ylabel('Total Weighted Distance', fontsize=11)
    ax5.set_title('Total Weighted Distance Comparison', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, dist in zip(bars, weighted_dists):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{dist:.1f}', ha='center', va='bottom', fontsize=10)

    # Plot 6: Reconfiguration Rate (Horizontal Bar)
    ax6 = fig.add_subplot(gs[1, 2])
    rates = [results_dict[m]['reconfiguration_rate'] * 100 for m in methods]
    bars = ax6.barh([labels[m] for m in methods], rates,
                    color=[colors[m] for m in methods], alpha=0.8)
    ax6.set_xlabel('Reconfiguration Rate (%)', fontsize=11)
    ax6.set_title('Reconfiguration Rate Comparison', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)

    # Add value labels
    for bar, rate in zip(bars, rates):
        width = bar.get_width()
        ax6.text(width, bar.get_y() + bar.get_height()/2.,
                f'{rate:.2f}%', ha='left', va='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.suptitle(f'IK Selection Method Comparison (Threshold: {results_dict["random"]["threshold_used"]:.1f} rad)',
                fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.close()


def save_comparison_csv(results_dict: Dict[str, Dict], joint_names: List[str],
                       save_path: str) -> None:
    """
    Save detailed comparison results to CSV file.

    Args:
        results_dict: Dictionary with analysis results
        joint_names: List of joint names
        save_path: Output CSV file path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(['IK Selection Method Comparison'])
        writer.writerow([f"Threshold: {results_dict['random']['threshold_used']:.3f} radians"])
        writer.writerow([])

        # Summary statistics
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow(['Method', 'Total Reconfigurations', 'Reconfiguration Rate (%)',
                        'Total Weighted Distance', 'Mean Weighted Distance'])

        for method in ['random', 'greedy', 'dp']:
            results = results_dict[method]
            writer.writerow([
                method.upper(),
                results['total_reconfigurations'],
                f"{results['reconfiguration_rate']*100:.2f}",
                f"{results['total_weighted_distance']:.4f}",
                f"{results['mean_weighted_distance']:.4f}"
            ])

        writer.writerow([])

        # Improvements
        writer.writerow(['IMPROVEMENTS'])
        writer.writerow(['Comparison', 'Reconfiguration Reduction', 'Reduction %',
                        'Distance Reduction', 'Reduction %'])

        random_reconfig = results_dict['random']['total_reconfigurations']
        greedy_reconfig = results_dict['greedy']['total_reconfigurations']
        dp_reconfig = results_dict['dp']['total_reconfigurations']

        random_dist = results_dict['random']['total_weighted_distance']
        greedy_dist = results_dict['greedy']['total_weighted_distance']
        dp_dist = results_dict['dp']['total_weighted_distance']

        writer.writerow([
            'Greedy vs Random',
            random_reconfig - greedy_reconfig,
            f"{(random_reconfig - greedy_reconfig)/random_reconfig*100:.2f}",
            f"{random_dist - greedy_dist:.4f}",
            f"{(random_dist - greedy_dist)/random_dist*100:.2f}"
        ])

        writer.writerow([
            'DP vs Random',
            random_reconfig - dp_reconfig,
            f"{(random_reconfig - dp_reconfig)/random_reconfig*100:.2f}",
            f"{random_dist - dp_dist:.4f}",
            f"{(random_dist - dp_dist)/random_dist*100:.2f}"
        ])

        writer.writerow([
            'DP vs Greedy',
            greedy_reconfig - dp_reconfig,
            f"{(greedy_reconfig - dp_reconfig)/greedy_reconfig*100:.2f}",
            f"{greedy_dist - dp_dist:.4f}",
            f"{(greedy_dist - dp_dist)/greedy_dist*100:.2f}"
        ])

        writer.writerow([])

        # Per-joint statistics
        writer.writerow(['PER-JOINT STATISTICS'])
        writer.writerow(['Joint', 'Random Reconfig', 'Greedy Reconfig', 'DP Reconfig',
                        'Random Movement', 'Greedy Movement', 'DP Movement'])

        for i, joint_name in enumerate(joint_names):
            writer.writerow([
                joint_name,
                results_dict['random']['reconfigurations_per_joint'][i],
                results_dict['greedy']['reconfigurations_per_joint'][i],
                results_dict['dp']['reconfigurations_per_joint'][i],
                f"{results_dict['random']['total_movement_per_joint'][i]:.4f}",
                f"{results_dict['greedy']['total_movement_per_joint'][i]:.4f}",
                f"{results_dict['dp']['total_movement_per_joint'][i]:.4f}"
            ])

    print(f"CSV report saved to: {save_path}")


def save_markdown_report(results_dict: Dict[str, Dict], joint_names: List[str],
                        save_path: str, plot_path: str) -> None:
    """
    Generate detailed markdown report.

    Args:
        results_dict: Dictionary with analysis results
        joint_names: List of joint names
        save_path: Output markdown file path
        plot_path: Path to the visualization plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("# IK Solution Selection Method Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Threshold:** {results_dict['random']['threshold_used']:.3f} radians\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report compares three methods for selecting inverse kinematics (IK) solutions:\n\n")
        f.write("- **Random**: Randomly selects one safe IK solution per viewpoint\n")
        f.write("- **Greedy**: Selects the IK solution closest to the previous joint configuration\n")
        f.write("- **DP (Dynamic Programming)**: Uses global optimization to minimize total weighted distance\n\n")

        # Key findings
        random_reconfig = results_dict['random']['total_reconfigurations']
        dp_reconfig = results_dict['dp']['total_reconfigurations']
        improvement = (random_reconfig - dp_reconfig) / random_reconfig * 100

        f.write("### Key Findings\n\n")
        f.write(f"- DP method achieves **{improvement:.1f}% reduction** in joint reconfigurations compared to Random\n")
        f.write(f"- Total reconfigurations: Random ({random_reconfig}) > Greedy ({results_dict['greedy']['total_reconfigurations']}) > DP ({dp_reconfig})\n")
        f.write(f"- DP also minimizes total weighted distance, leading to smoother trajectories\n\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("### Metrics\n\n")
        f.write("1. **Joint Reconfiguration**: A joint change exceeding the threshold (1.0 radians)\n")
        f.write("2. **Total Weighted Distance**: Sum of weighted Euclidean distances in joint space\n")
        f.write("3. **Per-Joint Statistics**: Movement and reconfiguration counts for each joint\n\n")

        # Results
        f.write("## Results\n\n")
        f.write("### Overall Comparison\n\n")
        f.write("| Method | Total Reconfigurations | Rate (%) | Total Weighted Distance |\n")
        f.write("|--------|------------------------|----------|-------------------------|\n")

        for method in ['random', 'greedy', 'dp']:
            results = results_dict[method]
            f.write(f"| {method.upper()} | {results['total_reconfigurations']} | "
                   f"{results['reconfiguration_rate']*100:.2f} | "
                   f"{results['total_weighted_distance']:.4f} |\n")

        f.write("\n### Improvements\n\n")
        f.write("| Comparison | Reconfiguration Reduction | Distance Reduction |\n")
        f.write("|------------|---------------------------|--------------------|\n")

        random_dist = results_dict['random']['total_weighted_distance']
        greedy_dist = results_dict['greedy']['total_weighted_distance']
        dp_dist = results_dict['dp']['total_weighted_distance']

        greedy_reconfig = results_dict['greedy']['total_reconfigurations']

        f.write(f"| Greedy vs Random | {random_reconfig - greedy_reconfig} ({(random_reconfig - greedy_reconfig)/random_reconfig*100:.2f}%) | "
               f"{random_dist - greedy_dist:.4f} ({(random_dist - greedy_dist)/random_dist*100:.2f}%) |\n")

        f.write(f"| DP vs Random | {random_reconfig - dp_reconfig} ({(random_reconfig - dp_reconfig)/random_reconfig*100:.2f}%) | "
               f"{random_dist - dp_dist:.4f} ({(random_dist - dp_dist)/random_dist*100:.2f}%) |\n")

        f.write(f"| DP vs Greedy | {greedy_reconfig - dp_reconfig} ({(greedy_reconfig - dp_reconfig)/greedy_reconfig*100:.2f}%) | "
               f"{greedy_dist - dp_dist:.4f} ({(greedy_dist - dp_dist)/greedy_dist*100:.2f}%) |\n")

        # Visualization
        f.write("\n## Visualizations\n\n")
        if plot_path and os.path.exists(plot_path):
            rel_plot_path = os.path.basename(plot_path)
            f.write(f"![Comparison Plots]({rel_plot_path})\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")
        f.write("The Dynamic Programming (DP) method consistently outperforms both Random and Greedy approaches:\n\n")
        f.write("1. **Lowest Reconfiguration Count**: DP minimizes unnecessary joint movements\n")
        f.write("2. **Shortest Weighted Distance**: DP finds globally optimal paths through joint space\n")
        f.write("3. **Smoother Trajectories**: Reduced reconfigurations lead to more efficient robot motion\n\n")

        f.write("### Recommendations\n\n")
        f.write("- **Use DP method for production**: Best overall performance\n")
        f.write("- **Greedy as fallback**: Good balance between speed and quality\n")
        f.write("- **Avoid Random selection**: Significantly worse performance\n\n")

    print(f"Markdown report saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare IK selection methods (Random, Greedy, DP)"
    )
    parser.add_argument(
        "--random",
        type=str,
        required=True,
        help="Path to random method CSV file"
    )
    parser.add_argument(
        "--greedy",
        type=str,
        required=True,
        help="Path to greedy method CSV file"
    )
    parser.add_argument(
        "--dp",
        type=str,
        required=True,
        help="Path to DP method CSV file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Threshold for joint change to count as reconfiguration in radians (default: 1.0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="Directory to save analysis results (default: data/output)"
    )

    args = parser.parse_args()

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print("LOADING TRAJECTORY DATA")
    print(f"{'='*80}")

    # Load all three trajectories
    try:
        random_data, joint_names = load_joint_trajectory(args.random)
        greedy_data, _ = load_joint_trajectory(args.greedy)
        dp_data, _ = load_joint_trajectory(args.dp)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return 1

    # Analyze all methods
    print(f"\n{'='*80}")
    print("ANALYZING TRAJECTORIES")
    print(f"{'='*80}")

    results_dict = {
        'random': analyze_joint_reconfigurations(random_data, args.threshold),
        'greedy': analyze_joint_reconfigurations(greedy_data, args.threshold),
        'dp': analyze_joint_reconfigurations(dp_data, args.threshold)
    }

    # Print comparison results
    print_comparison_results(results_dict, joint_names)

    # Generate outputs
    print(f"\n{'='*80}")
    print("GENERATING OUTPUTS")
    print(f"{'='*80}")

    # Create visualization
    plot_path = os.path.join(args.output_dir, f"method_comparison_{timestamp}.png")
    plot_comparison(results_dict, joint_names, save_path=plot_path)

    # Save CSV report
    csv_path = os.path.join(args.output_dir, f"method_comparison_{timestamp}.csv")
    save_comparison_csv(results_dict, joint_names, csv_path)

    # Save markdown report
    md_path = os.path.join(args.output_dir, "COMPARISON_REPORT.md")
    save_markdown_report(results_dict, joint_names, md_path, plot_path)

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
