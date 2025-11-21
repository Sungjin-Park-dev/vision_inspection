#!/usr/bin/env python3
"""
Full Pipeline Experiment Runner

Runs the complete 4-step vision inspection pipeline and measures execution time:
1. mesh_to_viewpoints.py - Generate viewpoints from mesh
2. compute_ik_solutions.py - Compute IK solution sets
3. fk_gtsp_gpu_claude2.py - TSP + IK solution selection
4. curobo_check.py - Collision checking and motion replanning

Experiments are performed with:
- Random glass positions
- Variable overlap ratios
- Timing measurements for each step
- Results saved to CSV
- Mesh file loaded from config.DEFAULT_MESH_FILE

Usage:
    # Basic experiment: 10 random positions, overlap ratio 0.5
    python experiments/run_full_pipeline_experiment.py \
        --num_experiments 10

    # Test multiple overlap ratios
    python experiments/run_full_pipeline_experiment.py \
        --num_experiments 5 \
        --overlap_ratios 0.25,0.5,0.75

    # Save to specific directory
    python experiments/run_full_pipeline_experiment.py \
        --num_experiments 3 \
        --output_dir experiments/results/my_experiment
"""

import os
import sys
import argparse
import subprocess
import time
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common import config
from experiments.config_manager import temporary_config
from experiments.random_pose import random_pose


@dataclass
class PipelineMetrics:
    """Container for pipeline execution metrics"""
    experiment_id: int
    timestamp: str
    glass_pos_x: float
    glass_pos_y: float
    glass_pos_z: float
    overlap_ratio: float

    # Timing (seconds)
    step1_time_sec: float = 0.0
    step2_time_sec: float = 0.0
    step3_time_sec: float = 0.0
    step4_time_sec: float = 0.0
    total_time_sec: float = 0.0

    # Metrics
    num_viewpoints: int = 0
    num_ik_all: int = 0
    num_ik_safe: int = 0
    num_collisions: int = 0

    # Status
    status: str = "pending"  # pending, success, failed

    # File paths (for debugging)
    viewpoints_h5: str = ""
    ik_solutions_h5: str = ""
    trajectory_csv: str = ""
    collision_report: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV writing"""
        return asdict(self)


class PipelineRunner:
    """Runs the 4-step vision inspection pipeline"""

    def __init__(self, verbose: bool = True):
        # Get mesh file from config
        self.mesh_file = Path(config.DEFAULT_MESH_FILE).resolve()
        self.verbose = verbose
        self.project_root = Path(__file__).parent.parent

        if not self.mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {self.mesh_file}")

        self.log(f"Using mesh file: {self.mesh_file}")

    def log(self, message: str):
        """Print message if verbose"""
        if self.verbose:
            print(message)

    def run_command(self, cmd: List[str], step_name: str) -> Tuple[float, bool]:
        """Run a command and measure execution time

        Args:
            cmd: Command and arguments to run
            step_name: Name of the step for logging

        Returns:
            Tuple of (execution_time_sec, success)
        """
        self.log(f"\n{'='*70}")
        self.log(f"{step_name}")
        self.log(f"{'='*70}")
        self.log(f"Command: {' '.join(cmd)}")

        start = time.perf_counter()
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=not self.verbose,
                text=True,
                cwd=self.project_root
            )
            elapsed = time.perf_counter() - start
            self.log(f"✓ {step_name} completed in {elapsed:.2f}s")
            return elapsed, True

        except subprocess.CalledProcessError as e:
            elapsed = time.perf_counter() - start
            self.log(f"✗ {step_name} failed after {elapsed:.2f}s")
            if not self.verbose:
                self.log(f"Error output:\n{e.stderr}")
            return elapsed, False

    def step1_viewpoints(self, overlap_ratio: float) -> Tuple[float, Optional[Path], bool]:
        """Step 1: Generate viewpoints from mesh

        Args:
            overlap_ratio: Camera overlap ratio (0-1)

        Returns:
            Tuple of (execution_time, viewpoints_h5_path, success)
        """
        cmd = [
            sys.executable,  # Use current Python interpreter
            "scripts/mesh_to_viewpoints.py",
            "--mesh_file", str(self.mesh_file),
            "--overlap", str(overlap_ratio),
            "--adaptive_sampling",
            "--use_poisson_disk",
            "--filter_downward",
            "--apply_tilt"
        ]

        elapsed, success = self.run_command(cmd, "STEP 1: VIEWPOINT GENERATION")

        if not success:
            return elapsed, None, False

        # Find generated viewpoints file (most recent)
        viewpoint_dirs = sorted(
            (self.project_root / "data" / "viewpoint").glob("*"),
            key=lambda p: p.stat().st_mtime
        )

        if viewpoint_dirs:
            viewpoints_h5 = viewpoint_dirs[-1] / "viewpoints.h5"
            if viewpoints_h5.exists():
                return elapsed, viewpoints_h5, True

        return elapsed, None, False

    def step2_ik(self, viewpoints_h5: Path) -> Tuple[float, Optional[Path], bool]:
        """Step 2: Compute IK solutions

        Args:
            viewpoints_h5: Path to viewpoints HDF5 file

        Returns:
            Tuple of (execution_time, ik_solutions_h5_path, success)
        """
        cmd = [
            sys.executable,
            "scripts/compute_ik_solutions.py",
            "--viewpoints", str(viewpoints_h5)
        ]

        elapsed, success = self.run_command(cmd, "STEP 2: IK COMPUTATION")

        if not success:
            return elapsed, None, False

        # Determine output path based on viewpoint directory name
        num_points = viewpoints_h5.parent.name
        ik_solutions_h5 = self.project_root / "data" / "ik" / num_points / "ik_solutions.h5"

        if ik_solutions_h5.exists():
            return elapsed, ik_solutions_h5, True

        return elapsed, None, False

    def step3_tsp(self, ik_solutions_h5: Path) -> Tuple[float, Optional[Path], bool]:
        """Step 3: TSP + IK solution selection

        Args:
            ik_solutions_h5: Path to IK solutions HDF5 file

        Returns:
            Tuple of (execution_time, trajectory_csv_path, success)
        """
        num_points = ik_solutions_h5.parent.name
        output_csv = self.project_root / "data" / "trajectory" / num_points / "joint_trajectory_dp.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "scripts/fk_gtsp_gpu_claude2.py",
            "--h5", str(ik_solutions_h5),
            "--csv_out", str(output_csv)
        ]

        elapsed, success = self.run_command(cmd, "STEP 3: TSP + IK SELECTION")

        if not success:
            return elapsed, None, False

        if output_csv.exists():
            return elapsed, output_csv, True

        return elapsed, None, False

    def step4_collision(self, trajectory_csv: Path) -> Tuple[float, Optional[Path], bool]:
        """Step 4: Collision checking and replanning

        Args:
            trajectory_csv: Path to trajectory CSV file

        Returns:
            Tuple of (execution_time, collision_report_path, success)
        """
        cmd = [
            sys.executable,
            "scripts/curobo_check.py",
            "--trajectory", str(trajectory_csv),
            "--mesh", str(self.mesh_file),
            "--interpolate",
            "--check_reconfig",
            "--verbose"
        ]

        elapsed, success = self.run_command(cmd, "STEP 4: COLLISION CHECKING")

        if not success:
            return elapsed, None, False

        # Find collision report
        num_points = trajectory_csv.parent.name
        collision_report = self.project_root / "data" / "collision" / num_points / "collision_curobo.txt"

        if collision_report.exists():
            return elapsed, collision_report, True

        return elapsed, None, False

    def extract_metrics(self, metrics: PipelineMetrics):
        """Extract metrics from output files

        Args:
            metrics: PipelineMetrics object to populate
        """
        # Extract number of viewpoints from HDF5
        if metrics.viewpoints_h5:
            try:
                import h5py
                with h5py.File(metrics.viewpoints_h5, 'r') as f:
                    metrics.num_viewpoints = int(f['metadata'].attrs['num_viewpoints'])
            except Exception as e:
                self.log(f"⚠️  Failed to extract viewpoint count: {e}")

        # Extract IK metrics from HDF5
        if metrics.ik_solutions_h5:
            try:
                import h5py
                with h5py.File(metrics.ik_solutions_h5, 'r') as f:
                    # Count total IK solutions across all viewpoints
                    total_all = 0
                    total_safe = 0
                    for vp_grp_name in f.keys():
                        if vp_grp_name.startswith('viewpoint_'):
                            vp_grp = f[vp_grp_name]
                            total_all += vp_grp.attrs.get('num_all_solutions', 0)
                            total_safe += vp_grp.attrs.get('num_safe_solutions', 0)
                    metrics.num_ik_all = total_all
                    metrics.num_ik_safe = total_safe
            except Exception as e:
                self.log(f"⚠️  Failed to extract IK metrics: {e}")

        # Extract collision count from report
        if metrics.collision_report:
            try:
                with open(metrics.collision_report, 'r') as f:
                    content = f.read()
                    # Parse collision count from report
                    # Format: "Total collisions: X"
                    for line in content.split('\n'):
                        if 'Total collisions:' in line:
                            parts = line.split(':')
                            if len(parts) >= 2:
                                metrics.num_collisions = int(parts[1].strip())
                                break
            except Exception as e:
                self.log(f"⚠️  Failed to extract collision count: {e}")

    def run_full_pipeline(
        self,
        glass_position: np.ndarray,
        overlap_ratio: float,
        experiment_id: int
    ) -> PipelineMetrics:
        """Run the complete 4-step pipeline

        Args:
            glass_position: Glass position [x, y, z]
            overlap_ratio: Camera overlap ratio
            experiment_id: Experiment number

        Returns:
            PipelineMetrics with results
        """
        metrics = PipelineMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            glass_pos_x=float(glass_position[0]),
            glass_pos_y=float(glass_position[1]),
            glass_pos_z=float(glass_position[2]),
            overlap_ratio=overlap_ratio
        )

        self.log(f"\n{'#'*70}")
        self.log(f"# EXPERIMENT {experiment_id}")
        self.log(f"# Glass position: [{glass_position[0]:.3f}, {glass_position[1]:.3f}, {glass_position[2]:.3f}]")
        self.log(f"# Overlap ratio: {overlap_ratio}")
        self.log(f"{'#'*70}")

        overall_start = time.perf_counter()

        try:
            # Step 1: Viewpoints
            t1, viewpoints_h5, success = self.step1_viewpoints(overlap_ratio)
            metrics.step1_time_sec = t1
            if not success or viewpoints_h5 is None:
                metrics.status = "failed_step1"
                return metrics
            metrics.viewpoints_h5 = str(viewpoints_h5)

            # Step 2: IK
            t2, ik_solutions_h5, success = self.step2_ik(viewpoints_h5)
            metrics.step2_time_sec = t2
            if not success or ik_solutions_h5 is None:
                metrics.status = "failed_step2"
                return metrics
            metrics.ik_solutions_h5 = str(ik_solutions_h5)

            # Step 3: TSP
            t3, trajectory_csv, success = self.step3_tsp(ik_solutions_h5)
            metrics.step3_time_sec = t3
            if not success or trajectory_csv is None:
                metrics.status = "failed_step3"
                return metrics
            metrics.trajectory_csv = str(trajectory_csv)

            # Step 4: Collision
            t4, collision_report, success = self.step4_collision(trajectory_csv)
            metrics.step4_time_sec = t4
            if not success or collision_report is None:
                metrics.status = "failed_step4"
                return metrics
            metrics.collision_report = str(collision_report)

            # Extract metrics from output files
            self.extract_metrics(metrics)

            metrics.status = "success"

        except Exception as e:
            self.log(f"\n✗ Experiment {experiment_id} failed with exception:")
            self.log(traceback.format_exc())
            metrics.status = f"failed_exception: {str(e)}"

        finally:
            metrics.total_time_sec = time.perf_counter() - overall_start

        # Print summary
        self.log(f"\n{'='*70}")
        self.log(f"EXPERIMENT {experiment_id} SUMMARY")
        self.log(f"{'='*70}")
        self.log(f"Status:                   {metrics.status}")
        self.log(f"Step 1 (Viewpoints):      {metrics.step1_time_sec:6.2f}s")
        self.log(f"Step 2 (IK):              {metrics.step2_time_sec:6.2f}s")
        self.log(f"Step 3 (TSP):             {metrics.step3_time_sec:6.2f}s")
        self.log(f"Step 4 (Collision):       {metrics.step4_time_sec:6.2f}s")
        self.log(f"Total:                    {metrics.total_time_sec:6.2f}s")
        self.log(f"Viewpoints:               {metrics.num_viewpoints}")
        self.log(f"IK solutions (all/safe):  {metrics.num_ik_all}/{metrics.num_ik_safe}")
        self.log(f"Collisions:               {metrics.num_collisions}")
        self.log(f"{'='*70}\n")

        return metrics


def save_results_to_csv(metrics_list: List[PipelineMetrics], output_file: Path):
    """Save experiment results to CSV

    Args:
        metrics_list: List of PipelineMetrics
        output_file: Path to output CSV file
    """
    if not metrics_list:
        print("⚠️  No results to save")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Get fieldnames from first metrics object
    fieldnames = list(metrics_list[0].to_dict().keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics.to_dict())

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")
    print(f"Total experiments: {len(metrics_list)}")
    print(f"Successful: {sum(1 for m in metrics_list if m.status == 'success')}")
    print(f"Failed: {sum(1 for m in metrics_list if m.status != 'success')}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    print(f"{'='*70}\n")


def print_summary_statistics(metrics_list: List[PipelineMetrics]):
    """Print summary statistics across all experiments

    Args:
        metrics_list: List of PipelineMetrics
    """
    if not metrics_list:
        return

    successful = [m for m in metrics_list if m.status == "success"]

    if not successful:
        print("\n⚠️  No successful experiments to analyze")
        return

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"Total experiments:        {len(metrics_list)}")
    print(f"Successful:               {len(successful)}")
    print(f"Success rate:             {len(successful) / len(metrics_list) * 100:.1f}%")

    # Timing statistics
    step1_times = [m.step1_time_sec for m in successful]
    step2_times = [m.step2_time_sec for m in successful]
    step3_times = [m.step3_time_sec for m in successful]
    step4_times = [m.step4_time_sec for m in successful]
    total_times = [m.total_time_sec for m in successful]

    print(f"\nTiming (mean ± std):")
    print(f"  Step 1 (Viewpoints):    {np.mean(step1_times):6.2f} ± {np.std(step1_times):5.2f}s")
    print(f"  Step 2 (IK):            {np.mean(step2_times):6.2f} ± {np.std(step2_times):5.2f}s")
    print(f"  Step 3 (TSP):           {np.mean(step3_times):6.2f} ± {np.std(step3_times):5.2f}s")
    print(f"  Step 4 (Collision):     {np.mean(step4_times):6.2f} ± {np.std(step4_times):5.2f}s")
    print(f"  Total:                  {np.mean(total_times):6.2f} ± {np.std(total_times):5.2f}s")

    # Metrics statistics
    num_viewpoints = [m.num_viewpoints for m in successful]
    num_ik_all = [m.num_ik_all for m in successful]
    num_ik_safe = [m.num_ik_safe for m in successful]
    num_collisions = [m.num_collisions for m in successful]

    print(f"\nMetrics (mean ± std):")
    print(f"  Viewpoints:             {np.mean(num_viewpoints):6.1f} ± {np.std(num_viewpoints):5.1f}")
    print(f"  IK solutions (all):     {np.mean(num_ik_all):6.1f} ± {np.std(num_ik_all):5.1f}")
    print(f"  IK solutions (safe):    {np.mean(num_ik_safe):6.1f} ± {np.std(num_ik_safe):5.1f}")
    print(f"  Collisions:             {np.mean(num_collisions):6.1f} ± {np.std(num_collisions):5.1f}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run full vision inspection pipeline experiments with timing measurements. "
                    "Mesh file is loaded from config.DEFAULT_MESH_FILE"
    )

    parser.add_argument(
        "--num_experiments",
        type=int,
        default=10,
        help="Number of experiments to run (default: 10)"
    )

    parser.add_argument(
        "--overlap_ratios",
        type=str,
        default="0.5",
        help="Comma-separated overlap ratios to test (default: 0.5). Example: 0.25,0.5,0.75"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: experiments/results/experiment_TIMESTAMP)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed output (default: True)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()

    # Parse overlap ratios
    overlap_ratios = [float(x.strip()) for x in args.overlap_ratios.split(',')]

    # Determine output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results" / f"experiment_{timestamp}"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    verbose = args.verbose and not args.quiet

    # Initialize pipeline runner (mesh file from config)
    runner = PipelineRunner(verbose=verbose)

    # Collect all metrics
    all_metrics: List[PipelineMetrics] = []

    experiment_id = 0

    # Run experiments for each overlap ratio
    for overlap_ratio in overlap_ratios:
        print(f"\n{'='*70}")
        print(f"TESTING OVERLAP RATIO: {overlap_ratio}")
        print(f"{'='*70}")

        for _ in range(args.num_experiments):
            experiment_id += 1

            # Generate random glass position
            glass_position, _ = random_pose()

            # Run pipeline with temporary config modifications
            with temporary_config(
                GLASS_POSITION=glass_position,
                CAMERA_OVERLAP_RATIO=overlap_ratio
            ):
                metrics = runner.run_full_pipeline(
                    glass_position=glass_position,
                    overlap_ratio=overlap_ratio,
                    experiment_id=experiment_id
                )
                all_metrics.append(metrics)

    # Save results
    output_csv = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    save_results_to_csv(all_metrics, output_csv)

    # Print summary statistics
    print_summary_statistics(all_metrics)

    print(f"\n✓ All experiments completed!")
    print(f"  Results saved to: {output_csv}")


if __name__ == "__main__":
    main()
