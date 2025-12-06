#!/usr/bin/env python3
"""
Visualization Utilities for Vision Inspection Pipeline

Provides Isaac Sim visualization functions for debugging and analysis.
Consolidated from simulate_trajectory.py and other scripts.
"""

from typing import List

import numpy as np

# Import guard for Isaac Sim dependencies (optional)
try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        _debug_draw = None

try:
    from curobo.types.base import TensorDeviceType
    from curobo.wrap.reacher.ik_solver import IKSolver
except ImportError:
    TensorDeviceType = None
    IKSolver = None


# ============================================================================
# Isaac Sim Debug Visualization
# ============================================================================

def visualize_target_positions(
    target_positions: List[np.ndarray],
    draw,
    joint_targets: List[np.ndarray] = None,
    ik_solver = None
):
    """Visualize target waypoint positions as green points in Isaac Sim

    Args:
        target_positions: List of target positions (from CSV), or None to compute from FK
        draw: Debug draw interface (omni.isaac.debug_draw._debug_draw)
        joint_targets: List of joint configurations (used if target_positions is None)
        ik_solver: IK solver with kinematics (used if target_positions is None)

    Raises:
        ValueError: If target_positions is None but joint_targets or ik_solver is missing

    Example:
        >>> # In Isaac Sim context
        >>> from omni.isaac.debug_draw import _debug_draw
        >>> draw = _debug_draw.acquire_debug_draw_interface()
        >>> positions = [np.array([1.0, 0.0, 0.5]), np.array([1.0, 0.0, 0.6])]
        >>> visualize_target_positions(positions, draw)
        # Draws green points at the target positions
    """
    from common.cli_utils import print_section_header, print_key_value

    print_section_header("DEBUG: VISUALIZING TARGET POSITIONS", width=60)

    # If target_positions not provided, compute from FK
    if target_positions is None:
        if joint_targets is None or ik_solver is None:
            print("Error: Cannot visualize - no target positions or FK solver provided")
            return

        if TensorDeviceType is None:
            print("Error: CuRobo not available - cannot compute FK")
            return

        print("Computing target positions from forward kinematics...")
        tensor_args = TensorDeviceType()
        target_positions = []

        for joint_config in joint_targets:
            # Convert to tensor (batch size 1)
            q_tensor = tensor_args.to_device([joint_config])

            # Compute forward kinematics
            ee_pose = ik_solver.fk(q_tensor)
            ee_position = ee_pose.position[0].cpu().numpy()

            target_positions.append(ee_position)
    else:
        print("Using target positions from CSV file...")

    target_positions = np.array(target_positions)

    # Draw green points at target positions
    point_sizes = [10.0] * len(target_positions)
    colors = [(0.0, 1.0, 0.0, 1.0)] * len(target_positions)  # Green with alpha

    draw.draw_points(
        target_positions.tolist(),
        colors,
        point_sizes
    )

    print_key_value("Target waypoints visualized", len(target_positions))
    print_key_value("Color", "Green")
    print_key_value("Point size", 10.0)
    print()
