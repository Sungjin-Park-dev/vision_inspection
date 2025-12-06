#!/usr/bin/env python3
"""
Simulate Robot Trajectory in Isaac Sim

This script:
1. Loads joint trajectory from CSV file
2. Initializes Isaac Sim world with robot and glass object
3. Executes trajectory and visualizes robot motion

The trajectory is executed directly without additional interpolation.
Use collision-checked trajectory from curobo_check.py for collision-free execution.

Usage:
    omni_python scripts/simulate_trajectory.py \\
        --trajectory data/trajectory/3000/joint_trajectory_dp_curobo_interpolated.csv \\
        --robot ur20.yml \\
        --visualize_spheres
"""

# ============================================================================
# Standard Library Imports
# ============================================================================
import argparse
import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ============================================================================
# Third Party Imports
# ============================================================================
import numpy as np

# ============================================================================
# Isaac Sim Imports
# ============================================================================
try:
    import isaacsim
except ImportError:
    pass

from isaacsim.simulation_app import SimulationApp

# Parse arguments before SimulationApp initialization
parser = argparse.ArgumentParser(description="Simulate robot trajectory in Isaac Sim")
parser.add_argument(
    "--trajectory",
    type=str,
    required=True,
    help="Path to joint trajectory CSV file"
)
parser.add_argument(
    "--robot",
    type=str,
    default="ur20_safe.yml",
    help="Robot configuration file (default: ur20.yml)"
)
parser.add_argument(
    "--headless",
    type=str,
    default=None,
    help="Run headless: one of [native, websocket]"
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Visualize robot collision spheres",
    default=False
)
parser.add_argument(
    "--interpolation_steps",
    type=int,
    default=None,
    help="Fixed number of interpolation steps between waypoints (overrides adaptive mode)"
)
parser.add_argument(
    "--steps_per_radian",
    type=float,
    default=50.0,
    help="Number of interpolation steps per radian of joint movement (adaptive mode, default: 30.0)"
)
parser.add_argument(
    "--min_steps",
    type=int,
    default=5,
    help="Minimum interpolation steps for adaptive mode (default: 5)"
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=100,
    help="Maximum interpolation steps for adaptive mode (default: 100)"
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode: visualize target waypoint positions as green points"
)
args = parser.parse_args()

# Initialize SimulationApp
simulation_app = SimulationApp({
    "headless": args.headless is not None,
    "width": "1280",
    "height": "720",
})

# ============================================================================
# Isaac Sim Component Imports (after SimulationApp)
# ============================================================================
from omni.isaac.core import World
from omni.isaac.core.objects import sphere
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.utils.numpy.rotations import euler_angles_to_quats

try:
    from omni.isaac.debug_draw import _debug_draw
except ImportError:
    from isaacsim.util.debug_draw import _debug_draw

from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.materials import OmniGlass

# ============================================================================
# CuRobo Imports
# ============================================================================
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# ============================================================================
# Local Imports
# ============================================================================
from common import config
from common.cli_utils import print_section_header, print_key_value, print_success
from common.interpolation_utils import generate_interpolated_path
from common.world_setup import setup_collision_world
from common.data_io import load_trajectory_csv
from common.simulation_helper import add_extensions, add_robot_to_scene


# ============================================================================
# Configuration Classes
# ============================================================================
@dataclass
class SimulationConfig:
    """Configuration for simulation"""
    trajectory_path: str
    robot_config_file: str
    headless_mode: str
    visualize_spheres: bool
    debug: bool

    # Interpolation parameters
    interpolation_steps: int  # If set, overrides adaptive mode
    steps_per_radian: float
    min_steps: int
    max_steps: int

    # World configuration
    table_position: np.ndarray = field(default_factory=lambda: config.TABLE_POSITION.copy())
    table_dimensions: np.ndarray = field(default_factory=lambda: config.TABLE_DIMENSIONS.copy())
    glass_position: np.ndarray = field(default_factory=lambda: config.GLASS_POSITION.copy())
    glass_rotation: np.ndarray = field(default_factory=lambda: config.GLASS_ROTATION.copy())

    # Additional obstacles
    wall_position: np.ndarray = field(default_factory=lambda: config.WALL_POSITION.copy())
    wall_dimensions: np.ndarray = field(default_factory=lambda: config.WALL_DIMENSIONS.copy())
    workbench_position: np.ndarray = field(default_factory=lambda: config.WORKBENCH_POSITION.copy())
    workbench_dimensions: np.ndarray = field(default_factory=lambda: config.WORKBENCH_DIMENSIONS.copy())
    robot_mount_position: np.ndarray = field(default_factory=lambda: config.ROBOT_MOUNT_POSITION.copy())
    robot_mount_dimensions: np.ndarray = field(default_factory=lambda: config.ROBOT_MOUNT_DIMENSIONS.copy())

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SimulationConfig':
        """Create configuration from command line arguments"""
        return cls(
            trajectory_path=args.trajectory,
            robot_config_file=args.robot,
            headless_mode=args.headless,
            visualize_spheres=args.visualize_spheres,
            debug=args.debug,
            interpolation_steps=args.interpolation_steps,
            steps_per_radian=args.steps_per_radian,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )


@dataclass
class WorldState:
    """Encapsulates Isaac Sim world state"""
    world: World
    glass_prim: XFormPrim
    robot: any
    idx_list: List[int]
    ik_solver: IKSolver


# ============================================================================
# File I/O
# ============================================================================
def load_joint_trajectory_csv_for_sim(csv_path: str) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """Load joint trajectory and target poses from CSV file

    Returns:
        Tuple of (joint_targets, target_positions)
        - joint_targets: List of joint configurations (each is 6-element array)
        - target_positions: List of target positions (each is 3-element array [x, y, z])
    """
    print_section_header("LOADING JOINT TRAJECTORY", width=60)
    print_key_value("Input file", csv_path)

    # Use common utility to load trajectory
    trajectory, joint_names = load_trajectory_csv(csv_path, joint_prefix="ur20-")

    # Convert to list of arrays
    joint_targets = [np.array(config, dtype=np.float64) for config in trajectory]

    # Load target poses from CSV
    target_positions = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Extract target position (POS_X, POS_Y, POS_Z)
                pos_x = float(row['target-POS_X'])
                pos_y = float(row['target-POS_Y'])
                pos_z = float(row['target-POS_Z'])
                target_positions.append(np.array([pos_x, pos_y, pos_z]))
        print_key_value("Loaded target poses", len(target_positions))
    except (KeyError, ValueError) as e:
        print(f"  Warning: Could not load target poses from CSV: {e}")
        print("  → Will compute from FK instead")
        target_positions = None

    print_key_value("Loaded waypoints", len(joint_targets))
    print()

    return joint_targets, target_positions


def compute_adaptive_steps(
    start: np.ndarray,
    end: np.ndarray,
    steps_per_radian: float,
    min_steps: int,
    max_steps: int
) -> int:
    """
    Compute number of interpolation steps based on joint space distance

    Args:
        start: Starting joint configuration
        end: Ending joint configuration
        steps_per_radian: Number of steps per radian of movement
        min_steps: Minimum number of steps
        max_steps: Maximum number of steps

    Returns:
        Number of interpolation steps (clamped to [min_steps, max_steps])
    """
    # Exclude last joint (wrist_3) from distance calculation
    # Last joint only affects tool rotation, not collision
    start_excl_last = start[:-1] if len(start) > 1 else start
    end_excl_last = end[:-1] if len(end) > 1 else end

    # Compute Euclidean distance in joint space (excluding last joint)
    distance = np.linalg.norm(end_excl_last - start_excl_last)

    # Calculate steps proportional to distance
    steps = int(distance * steps_per_radian)

    # Clamp to [min_steps, max_steps]
    return max(min_steps, min(steps, max_steps))


# ============================================================================
# World Initialization
# ============================================================================
def create_world() -> World:
    """Create Isaac Sim world"""
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    return my_world


def setup_robot(my_world: World, cfg: SimulationConfig) -> dict:
    """Setup robot in the world

    Returns:
        dict with keys: robot, idx_list, robot_prim_path, robot_cfg
    """
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), cfg.robot_config_file))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(
        robot_config=robot_cfg,
        my_world=my_world,
        position=np.array([0.0, 0.0, 0.0]),
    )

    idx_list = [robot.get_dof_index(x) for x in j_names]
    robot.set_joint_positions(default_config, idx_list)

    return {
        'robot': robot,
        'idx_list': idx_list,
        'robot_prim_path': robot_prim_path,
        'robot_cfg': robot_cfg,
    }


def setup_glass_object_from_mesh(my_world: World, cfg: SimulationConfig, usd_helper: UsdHelper) -> XFormPrim:
    """Setup glass object using mesh file"""
    mesh_file_path = config.DEFAULT_MESH_FILE

    print_section_header("ADDING GLASS MESH TO STAGE", width=60)
    print_key_value("Mesh file", mesh_file_path)
    print_key_value("Position", cfg.glass_position)
    print()

    usd_helper.load_stage(my_world.stage)


    glass_mesh = Mesh(
        name="glass",
        file_path=mesh_file_path,
        pose=list(cfg.glass_position) + [1, 0, 0, 0],
        color=[1.0, 0.1, 0.1, 0.95]
    )

    glass_path = usd_helper.add_mesh_to_stage(
        obstacle=glass_mesh,
        base_frame="/World"
    )

    print_key_value("Glass prim path", glass_path)

    glass_prim = XFormPrim(glass_path)

    # Apply glass material (optional)
    # try:
    #     glass_material = OmniGlass(
    #         prim_path="/World/Looks/glass_mat",
    #         color=np.array([0.7, 0.85, 0.9]),
    #         ior=1.52,
    #         depth=0.01,
    #         thin_walled=False,
    #     )
    #     glass_prim.apply_visual_material(glass_material)
    #     print("Applied OmniGlass material")
    # except Exception as e:
    #     print(f"Warning: Could not apply glass material: {e}")

    return glass_prim


def setup_camera(robot_prim_path: str, my_world: World):
    """Setup camera mounted on robot end-effector"""
    tool_prim_path = robot_prim_path + "/tool0"
    camera_prim_path = tool_prim_path + "/mounted_camera"

    camera = Camera(
        prim_path=camera_prim_path,
        frequency=20,
        translation=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([1, 0, 0, 0]),
        resolution=(256, 256),
    )

    # Camera specifications
    camera.set_focal_length(38.0 / 1e3)
    camera.set_focus_distance(110.0 / 1e3)
    camera.set_horizontal_aperture(14.13 / 1e3)
    camera.set_vertical_aperture(10.35 / 1e3)
    camera.set_clipping_range(10/1e3, 100/1e3)
    camera.set_local_pose(
        np.array([0.0, 0.0, 0.0]),
        euler_angles_to_quats(np.array([0, 180, 0]), degrees=True),
        camera_axes="usd"
    )
    my_world.scene.add(camera)

    return camera


def setup_collision_checker(
    my_world: World,
    robot_state: dict,
    cfg: SimulationConfig
) -> IKSolver:
    """Setup collision checker and IK solver"""
    usd_helper = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = robot_state['robot_cfg']
    robot_prim_path = robot_state['robot_prim_path']

    # Setup world collision configuration using common utility
    world_cfg = setup_collision_world(
        table_position=cfg.table_position,
        table_dimensions=cfg.table_dimensions,
        wall_position=cfg.wall_position,
        wall_dimensions=cfg.wall_dimensions,
        workbench_position=cfg.workbench_position,
        workbench_dimensions=cfg.workbench_dimensions,
        robot_mount_position=cfg.robot_mount_position,
        robot_mount_dimensions=cfg.robot_mount_dimensions,
        mesh_files=[],  # No mesh obstacles for visualization
        verbose=False
    )

    # Add ground mesh (positioned below ground)
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    # Combine cuboids and mesh obstacles
    world_cfg = WorldConfig(cuboid=world_cfg.cuboid, mesh=world_cfg1.mesh)

    # Create IK solver (needed for sphere visualization)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=config.IK_ROTATION_THRESHOLD,
        position_threshold=config.IK_POSITION_THRESHOLD,
        num_seeds=config.IK_NUM_SEEDS,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=False,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": config.N_OBSTACLE_CUBOIDS, "mesh": config.N_OBSTACLE_MESH},
    )
    ik_solver = IKSolver(ik_config)

    # Setup world in USD
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(world_cfg, base_frame="/World")

    my_world.scene.add_default_ground_plane(z_position=-0.5)

    # Get obstacles from stage
    obstacles = usd_helper.get_obstacles_from_stage(
        only_paths=["/World"],
        reference_prim_path=robot_prim_path,
        ignore_substring=[
            robot_prim_path,
            "/World/defaultGroundPlane",
            "/curobo",
            "/World/mount",
        ],
    ).get_collision_check_world()

    ik_solver.update_world(obstacles)

    return ik_solver


def initialize_simulation(cfg: SimulationConfig) -> WorldState:
    """Initialize Isaac Sim world and all components"""
    print_section_header("INITIALIZING SIMULATION", width=60)
    print()

    my_world = create_world()
    robot_state = setup_robot(my_world, cfg)

    usd_helper = UsdHelper()
    glass_prim = setup_glass_object_from_mesh(my_world, cfg, usd_helper)

    camera = setup_camera(robot_state['robot_prim_path'], my_world)
    ik_solver = setup_collision_checker(my_world, robot_state, cfg)

    return WorldState(
        world=my_world,
        glass_prim=glass_prim,
        robot=robot_state['robot'],
        idx_list=robot_state['idx_list'],
        ik_solver=ik_solver,
    )


# ============================================================================
# Debug Visualization
# ============================================================================
def visualize_target_positions(
    target_positions: List[np.ndarray],
    draw: _debug_draw,
    joint_targets: List[np.ndarray] = None,
    ik_solver: IKSolver = None
):
    """Visualize target waypoint positions as green points

    Args:
        target_positions: List of target positions (from CSV), or None to compute from FK
        draw: Debug draw interface
        joint_targets: List of joint configurations (used if target_positions is None)
        ik_solver: IK solver with kinematics (used if target_positions is None)
    """
    print_section_header("DEBUG: VISUALIZING TARGET POSITIONS", width=60)

    # If target_positions not provided, compute from FK
    if target_positions is None:
        if joint_targets is None or ik_solver is None:
            print("Error: Cannot visualize - no target positions or FK solver provided")
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


# ============================================================================
# Simulation Loop
# ============================================================================
def get_active_joint_positions(robot, idx_list: List[int]) -> np.ndarray:
    """Get current joint positions for active joints"""
    all_positions = robot.get_joint_positions()
    return np.asarray([all_positions[i] for i in idx_list], dtype=np.float64)


def run_simulation(
    world_state: WorldState,
    joint_targets: List[np.ndarray],
    cfg: SimulationConfig
):
    """Run Isaac Sim simulation with planned trajectory"""
    print_section_header("STARTING SIMULATION", width=60)
    print_key_value("Total waypoints", len(joint_targets))
    print_key_value("Execution mode", "Direct (no interpolation)")
    print("  Trajectory contains pre-computed collision-free configurations")
    print("  Waypoints will be executed directly without additional interpolation")
    if joint_targets:
        print("  Last joint will stay fixed to its initial value")
    print()

    # Setup trajectory queue
    target_queue: Deque[np.ndarray] = deque(joint_targets)
    fixed_last_joint_value = joint_targets[0][-1] if joint_targets else None

    step_counter = 0
    idle_counter = 0
    waypoint_counter = 0

    # Time tracking
    start_time = None
    end_time = None

    # Sphere visualization
    spheres = None
    tensor_args = TensorDeviceType()

    # Main simulation loop
    while simulation_app.is_running():
        world_state.world.step(render=True)

        if not world_state.world.is_playing():
            if idle_counter % 100 == 0:
                print("**** Click Play to start simulation *****")
            idle_counter += 1
            continue

        # Start timer when simulation actually begins
        if start_time is None:
            start_time = time.time()
            print(f"Simulation started at {time.strftime('%H:%M:%S')}")

        idle_counter = 0
        step_counter += 1

        # Visualize robot spheres
        if cfg.visualize_spheres and step_counter % 2 == 0:
            # Get current joint state from simulator
            sim_js = world_state.robot.get_joints_state()
            sim_js_names = world_state.robot.dof_names

            # Convert to CuRobo joint state
            cu_js = JointState(
                position=tensor_args.to_device(sim_js.positions),
                velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            cu_js = cu_js.get_ordered_joint_state(world_state.ik_solver.kinematics.joint_names)

            # Get sphere representation
            sph_list = world_state.ik_solver.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # Create spheres
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                # Update sphere positions and radii
                for si, s in enumerate(sph_list[0]):
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))
                    
        # Execute trajectory waypoints directly
        # No interpolation - trajectory already contains all collision-checked configurations
        if target_queue:
            # Get next waypoint and execute directly
            next_waypoint = target_queue.popleft()

            # Keep last joint locked to avoid rotating the tool during playback
            if fixed_last_joint_value is not None and next_waypoint.size > 0:
                next_waypoint = np.copy(next_waypoint)
                next_waypoint[-1] = fixed_last_joint_value

            world_state.robot.set_joint_positions(next_waypoint.tolist(), world_state.idx_list)
            waypoint_counter += 1

        # Check if trajectory complete
        if not target_queue and not cfg.debug:
            end_time = time.time()
            elapsed_time = end_time - start_time

            print_section_header("SIMULATION COMPLETED", width=60)
            print_key_value("Waypoints executed", waypoint_counter)
            print_key_value("Total execution time", f"{elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
            print_key_value("Average time per waypoint", f"{elapsed_time/waypoint_counter:.4f}s")
            print_key_value("Waypoint execution rate", f"{waypoint_counter/elapsed_time:.2f} waypoints/s")
            print()
            break


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    cfg = SimulationConfig.from_args(args)

    print_section_header("SIMULATE TRAJECTORY", width=60)
    print_key_value("Trajectory", cfg.trajectory_path)
    print_key_value("Robot config", cfg.robot_config_file)
    print_key_value("Debug mode", "Enabled" if cfg.debug else "Disabled")
    if cfg.debug:
        print("  → Target waypoint positions will be visualized as green points")
    print()

    # Load joint trajectory and target poses
    joint_targets, target_positions = load_joint_trajectory_csv_for_sim(cfg.trajectory_path)

    # Initialize simulation
    world_state = initialize_simulation(cfg)

    # Debug visualization: draw target positions
    if cfg.debug:
        draw = _debug_draw.acquire_debug_draw_interface()
        visualize_target_positions(
            target_positions=target_positions,
            draw=draw,
            joint_targets=joint_targets,
            ik_solver=world_state.ik_solver
        )

    # Run simulation
    run_simulation(world_state, joint_targets, cfg)

    simulation_app.close()


if __name__ == "__main__":
    main()
