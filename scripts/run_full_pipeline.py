#!/usr/bin/env python3
"""
Run Full Vision Inspection Pipeline

This script runs the complete pipeline in sequence:
1. Compute IK solutions from TSP tour (CuRobo only, no Isaac Sim)
2. Plan trajectory using selected method (no Isaac Sim)
3. Simulate trajectory (Isaac Sim required, optional)

Usage:
    # Run full pipeline with simulation
    python scripts/run_full_pipeline.py \\
        --tsp_tour data/tour/tour_3000.h5 \\
        --method dp \\
        --simulate

    # Run without simulation (no Isaac Sim needed!)
    python scripts/run_full_pipeline.py \\
        --tsp_tour data/tour/tour_3000.h5 \\
        --method dp

    # Run only specific stages
    python scripts/run_full_pipeline.py \\
        --tsp_tour data/tour/tour_3000.h5 \\
        --method dp \\
        --skip_ik \\
        --ik_solutions data/ik/ik_solutions_3000.h5
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================================
# Color output for terminal
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    """Print a formatted header"""
    print(f"\n{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{text}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}→ {text}{Colors.ENDC}")


# ============================================================================
# Pipeline Execution
# ============================================================================
def run_command(cmd, description):
    """Run a shell command and handle errors

    Args:
        cmd: Command to run (list of strings)
        description: Human-readable description of the command

    Returns:
        True if successful, False otherwise
    """
    print_header(f"STEP: {description}")
    print_info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print_success(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"{description} failed: {e}")
        return False


def extract_num_points(tsp_tour_path):
    """Extract number of points from TSP tour filename or metadata"""
    import h5py

    try:
        with h5py.File(tsp_tour_path, 'r') as f:
            if 'metadata' in f:
                return f['metadata'].attrs.get('num_points', 'unknown')
    except:
        pass

    # Fallback: try to extract from filename
    filename = os.path.basename(tsp_tour_path)
    # Look for pattern like "tour_3000.h5"
    import re
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))

    return 'unknown'


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run full vision inspection pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output arguments
    parser.add_argument(
        "--tsp_tour",
        type=str,
        required=True,
        help="Path to TSP tour HDF5 file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="dp",
        choices=["random", "greedy", "dp"],
        help="Trajectory planning method (default: dp)"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="ur20.yml",
        help="Robot configuration file (default: ur20.yml)"
    )

    # Stage control arguments
    parser.add_argument(
        "--skip_ik",
        action="store_true",
        help="Skip IK computation (use existing IK solutions file)"
    )
    parser.add_argument(
        "--skip_planning",
        action="store_true",
        help="Skip trajectory planning (use existing trajectory file)"
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation after planning"
    )
    parser.add_argument(
        "--visualize_spheres",
        action="store_true",
        help="Visualize robot collision spheres in simulation"
    )

    # File path overrides
    parser.add_argument(
        "--ik_solutions",
        type=str,
        default=None,
        help="Path to IK solutions HDF5 (for --skip_ik or auto-generated)"
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help="Path to trajectory CSV (for --skip_planning or auto-generated)"
    )

    # Advanced arguments
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode"
    )
    parser.add_argument(
        "--interpolation_steps",
        type=int,
        default=60,
        help="Interpolation steps for simulation (default: 60)"
    )

    args = parser.parse_args()

    # ========================================================================
    # Pipeline Configuration
    # ========================================================================
    print_header("VISION INSPECTION PIPELINE")
    print(f"TSP Tour: {args.tsp_tour}")
    print(f"Method: {args.method}")
    print(f"Robot: {args.robot}")
    print(f"Skip IK: {args.skip_ik}")
    print(f"Skip Planning: {args.skip_planning}")
    print(f"Simulate: {args.simulate}")

    # Determine paths
    num_points = extract_num_points(args.tsp_tour)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.ik_solutions is None:
        ik_dir = "data/ik"
        os.makedirs(ik_dir, exist_ok=True)
        ik_solutions_path = f"{ik_dir}/ik_solutions_{num_points}_{timestamp}.h5"
    else:
        ik_solutions_path = args.ik_solutions

    if args.trajectory is None:
        traj_dir = f"data/trajectory/{num_points}"
        os.makedirs(traj_dir, exist_ok=True)
        trajectory_path = f"{traj_dir}/joint_trajectory_{args.method}.csv"
    else:
        trajectory_path = args.trajectory

    # ========================================================================
    # Stage 1: Compute IK Solutions
    # ========================================================================
    if not args.skip_ik:
        cmd = [
            "python",
            "scripts/compute_ik_solutions.py",
            "--tsp_tour", args.tsp_tour,
            "--output", ik_solutions_path,
            "--robot", args.robot,
        ]

        success = run_command(cmd, "Compute IK Solutions")
        if not success:
            print_error("Pipeline failed at IK computation stage")
            sys.exit(1)
    else:
        print_info(f"Skipping IK computation, using: {ik_solutions_path}")
        if not os.path.exists(ik_solutions_path):
            print_error(f"IK solutions file not found: {ik_solutions_path}")
            sys.exit(1)

    # ========================================================================
    # Stage 2: Plan Trajectory
    # ========================================================================
    if not args.skip_planning:
        cmd = [
            "python",
            "scripts/plan_trajectory.py",
            "--ik_solutions", ik_solutions_path,
            "--output", trajectory_path,
            "--method", args.method,
        ]

        success = run_command(cmd, "Plan Trajectory")
        if not success:
            print_error("Pipeline failed at trajectory planning stage")
            sys.exit(1)
    else:
        print_info(f"Skipping trajectory planning, using: {trajectory_path}")
        if not os.path.exists(trajectory_path):
            print_error(f"Trajectory file not found: {trajectory_path}")
            sys.exit(1)

    # ========================================================================
    # Stage 3: Simulate Trajectory (Optional)
    # ========================================================================
    if args.simulate:
        cmd = [
            "/isaac-sim/python.sh",
            "scripts/simulate_trajectory.py",
            "--trajectory", trajectory_path,
            "--robot", args.robot,
            "--interpolation_steps", str(args.interpolation_steps),
        ]
        if args.visualize_spheres:
            cmd.append("--visualize_spheres")
        if args.headless:
            cmd.extend(["--headless", "native"])

        success = run_command(cmd, "Simulate Trajectory")
        if not success:
            print_error("Pipeline failed at simulation stage")
            sys.exit(1)
    else:
        print_info("Skipping simulation (use --simulate to enable)")

    # ========================================================================
    # Pipeline Complete
    # ========================================================================
    print_header("PIPELINE COMPLETE")
    print_success("All stages completed successfully!")
    print(f"\nOutput files:")
    if not args.skip_ik:
        print(f"  IK Solutions: {ik_solutions_path}")
    if not args.skip_planning:
        print(f"  Trajectory: {trajectory_path}")
        print(f"  Analysis: {os.path.dirname(trajectory_path)}/joint_trajectory_{args.method}_reconfig.txt")

    print(f"\nNext steps:")
    if not args.simulate:
        print(f"  • Run simulation:")
        print(f"    omni_python scripts/simulate_trajectory.py --trajectory {trajectory_path}")
    print(f"  • Try different planning methods:")
    for method in ["random", "greedy", "dp"]:
        if method != args.method:
            print(f"    python scripts/plan_trajectory.py --ik_solutions {ik_solutions_path} --method {method}")


if __name__ == "__main__":
    main()
