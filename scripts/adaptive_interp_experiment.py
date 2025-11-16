#!/usr/bin/env python3
"""Sweep adaptive interpolation thresholds and report collision statistics."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import config  # pylint: disable=wrong-import-position
from scripts.coal_check import (  # pylint: disable=wrong-import-position
    COALCollisionChecker,
    load_trajectory_csv,
)


def build_checker() -> COALCollisionChecker:
    """Construct a COALCollisionChecker using defaults from config.py."""
    return COALCollisionChecker(
        robot_urdf_path=config.DEFAULT_ROBOT_URDF,
        obstacle_mesh_paths=[config.DEFAULT_MESH_FILE],
        glass_position=config.GLASS_POSITION.copy(),
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
        use_curobo_interpolation=config.COLLISION_USE_CUROBO_INTERP,
    )


def format_table(rows: List[Dict[str, Any]]) -> str:
    """Return a simple ASCII table summarizing experiment runs."""
    headers = [
        ("step_deg", "Step(deg)", 10),
        ("interp_configs", "InterpCfg", 12),
        ("waypoint_collisions", "WpColl", 8),
        ("segment_pairs", "SegPairs", 10),
        ("runtime_sec", "Time(s)", 8),
    ]

    lines = []
    header_line = " ".join(f"{title:>{width}}" for _, title, width in headers)
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        line = " ".join(
            f"{row[key]:>{width}.2f}" if isinstance(row[key], float) else f"{row[key]:>{width}}"
            for key, _, width in headers
        )
        lines.append(line)

    return "\n" + "\n".join(lines)


def run_experiments(args: argparse.Namespace) -> List[Dict[str, Any]]:
    trajectory, joint_names = load_trajectory_csv(args.trajectory)
    print(f"Running adaptive interpolation sweep on {len(trajectory)} waypoints / {len(joint_names)} joints")

    checker = build_checker()

    base_kwargs = dict(
        verbose=args.verbose,
        show_link_collisions=False,
        interpolate=True,
        num_interp_steps=args.base_interp_steps,
        check_reconfig=args.reconfig_threshold_deg > 0.0,
        reconfig_threshold=np.deg2rad(args.reconfig_threshold_deg),
        parallel=args.parallel,
        num_workers=args.num_workers,
        adaptive_interp=True,
        adaptive_min_steps=args.min_steps,
        adaptive_max_steps=args.max_steps,
    )

    results: List[Dict[str, Any]] = []
    for step_deg in args.step_deg:
        print(f"\n=== Sweep value: max joint delta {step_deg} deg ===")
        start = time.perf_counter()
        trial = checker.check_trajectory(
            trajectory,
            adaptive_max_joint_step_deg=step_deg,
            **base_kwargs,
        )
        runtime = time.perf_counter() - start

        trial_summary = {
            'step_deg': float(step_deg),
            'total_configs': int(trial['total_configs_checked']),
            'interp_configs': int(trial.get('total_interpolated_configs', 0)),
            'seg_collisions': int(trial.get('num_segment_collisions', 0)),
            'waypoint_collisions': int(trial['num_collisions']),
            'segment_pairs': int(len({seg for seg, _ in trial.get('collision_segments', [])})),
            'reconfigs': int(trial.get('num_reconfigurations', 0)),
            'collision_time': float(trial.get('collision_check_time_sec', 0.0)),
            'reconfig_time': float(trial.get('reconfig_check_time_sec', 0.0)),
            'runtime_sec': float(runtime),
            'segment_interp_min': int(min(trial['segment_interp_counts'])) if trial['segment_interp_counts'] else 0,
            'segment_interp_max': int(max(trial['segment_interp_counts'])) if trial['segment_interp_counts'] else 0,
        }

        results.append(trial_summary)
        print(
            f"  Interpolated configs: {trial_summary['interp_configs']:,}"
            f" | Waypoint collisions: {trial_summary['waypoint_collisions']}"
            f" | Segment collisions (unique): {trial_summary['segment_pairs']}"
        )
        print(f"  Runtime: {runtime:.2f}s (collision {trial_summary['collision_time']:.2f}s)")

    return results


def plot_results(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot totals/collision metrics vs. adaptive step size."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - convenience path
        raise SystemExit("matplotlib is required for --plot-path") from exc

    steps = [row['step_deg'] for row in rows]
    unique_steps = sorted(set(steps))
    x_positions = [unique_steps.index(val) for val in steps]
    interp_configs = [row['interp_configs'] for row in rows]
    runtime_sec = [row['runtime_sec'] for row in rows]
    seg_pairs = [row['segment_pairs'] for row in rows]

    fig, (ax_cfg, ax_col) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax_cfg.plot(x_positions, interp_configs, marker='o', color='tab:blue', label='Interpolated configs')
    ax_cfg.set_ylabel('Interpolated configs (#)')
    ax_cfg.grid(True)

    ax_cfg_runtime = ax_cfg.twinx()
    ax_cfg_runtime.plot(x_positions, runtime_sec, marker='s', color='tab:orange', label='Runtime (s)')
    ax_cfg_runtime.set_ylabel('Runtime (s)')

    ax_col.plot(x_positions, seg_pairs, marker='x', color='tab:red')
    ax_col.set_ylabel('Segment collisions (#)')
    ax_col.grid(True)

    ax_col.set_xlabel('Adaptive max joint delta (deg)')
    ax_col.set_xticks(list(range(len(unique_steps))))
    ax_col.set_xticklabels([f"{val:g}" for val in unique_steps])

    # Combine legends for top subplot
    handles_cfg, labels_cfg = ax_cfg.get_legend_handles_labels()
    handles_cfg_runtime, labels_cfg_runtime = ax_cfg_runtime.get_legend_handles_labels()
    ax_cfg.legend(handles_cfg + handles_cfg_runtime, labels_cfg + labels_cfg_runtime, loc='upper right')

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run collision checks for multiple adaptive interpolation thresholds",
    )
    parser.add_argument(
        '--trajectory',
        type=str,
        required=True,
        help='Path to joint trajectory CSV file',
    )
    parser.add_argument(
        '--step-deg',
        type=float,
        nargs='+',
        required=True,
        help='List of adaptive max joint delta values (degrees) to sweep',
    )
    parser.add_argument(
        '--base-interp-steps',
        type=int,
        default=config.COLLISION_INTERP_STEPS,
        help='Baseline interpolation steps used when adaptive sampling requests more points',
    )
    parser.add_argument(
        '--min-steps',
        type=int,
        default=config.COLLISION_ADAPTIVE_MIN_STEPS,
        help='Minimum interpolation steps per segment in adaptive mode',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=config.COLLISION_ADAPTIVE_MAX_STEPS,
        help='Maximum interpolation steps per segment in adaptive mode (None = unlimited)',
    )
    parser.add_argument(
        '--reconfig-threshold-deg',
        type=float,
        default=np.rad2deg(config.RECONFIGURATION_THRESHOLD),
        help='Joint reconfiguration threshold in degrees (<=0 disables reconfig checks)',
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        default=config.COLLISION_PARALLEL,
        help='Enable multiprocessing during collision checking',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=config.COLLISION_NUM_WORKERS,
        help='Worker processes when --parallel is set (None = auto)',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose logs from the collision checker',
    )
    parser.add_argument(
        '--plot-path',
        type=str,
        help='Optional path to save a matplotlib PNG summarizing the sweep',
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.step_deg = sorted(set(args.step_deg))

    results = run_experiments(args)
    print(format_table(results))

    if args.plot_path:
        plot_results(results, Path(args.plot_path))


if __name__ == '__main__':
    main()
