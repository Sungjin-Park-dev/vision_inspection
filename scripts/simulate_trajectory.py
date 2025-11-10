#!/usr/bin/env python3
"""
Simulate Robot Trajectory in Isaac Sim

This script:
1. Loads joint trajectory from CSV file
2. Initializes Isaac Sim world with robot and glass object
3. Executes trajectory and visualizes robot motion

Usage:
    omni_python scripts/simulate_trajectory.py \\
        --trajectory data/trajectory/3000/joint_trajectory_dp.csv \\
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
    default="ur20.yml",
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
    default=60,
    help="Number of interpolation steps between waypoints (default: 60)"
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
from common.interpolation_utils import generate_interpolated_path
from utilss.simulation_helper import add_extensions, add_robot_to_scene


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
    interpolation_steps: int

    # World configuration
    table_position: np.ndarray = field(default_factory=lambda: config.TABLE_POSITION.copy())
    table_dimensions: np.ndarray = field(default_factory=lambda: config.TABLE_DIMENSIONS.copy())
    glass_position: np.ndarray = field(default_factory=lambda: config.GLASS_POSITION.copy())

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
            interpolation_steps=args.interpolation_steps,
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
def load_joint_trajectory_csv(csv_path: str) -> List[np.ndarray]:
    """Load joint trajectory from CSV file

    Returns:
        List of joint configurations (each is 6-element array)
    """
    print(f"\n{'='*60}")
    print("LOADING JOINT TRAJECTORY")
    print(f"{'='*60}")
    print(f"Input file: {csv_path}")

    joint_targets = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract 6 joint values
            joints = np.array([
                float(row['ur20-shoulder_pan_joint']),
                float(row['ur20-shoulder_lift_joint']),
                float(row['ur20-elbow_joint']),
                float(row['ur20-wrist_1_joint']),
                float(row['ur20-wrist_2_joint']),
                float(row['ur20-wrist_3_joint']),
            ], dtype=np.float64)
            joint_targets.append(joints)

    print(f"Loaded {len(joint_targets)} waypoints")
    print(f"{'='*60}\n")

    return joint_targets


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

    print(f"\n{'='*60}")
    print("ADDING GLASS MESH TO STAGE")
    print(f"{'='*60}")
    print(f"Mesh file: {mesh_file_path}")
    print(f"Position: {cfg.glass_position}")
    print(f"{'='*60}\n")

    usd_helper.load_stage(my_world.stage)

    glass_mesh = Mesh(
        name="glass",
        file_path=mesh_file_path,
        pose=list(cfg.glass_position) + [1, 0, 0, 0],
        color=[0.7, 0.85, 0.9, 0.3]
    )

    glass_path = usd_helper.add_mesh_to_stage(
        obstacle=glass_mesh,
        base_frame="/World"
    )

    print(f"Glass mesh added at prim path: {glass_path}")

    glass_prim = XFormPrim(glass_path)

    # Apply glass material (optional)
    try:
        glass_material = OmniGlass(
            prim_path="/World/Looks/glass_mat",
            color=np.array([0.7, 0.85, 0.9]),
            ior=1.52,
            depth=0.01,
            thin_walled=False,
        )
        glass_prim.apply_visual_material(glass_material)
        print("Applied OmniGlass material")
    except Exception as e:
        print(f"Warning: Could not apply glass material: {e}")

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

    # Setup world collision configuration
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[:3] = cfg.table_position
    world_cfg_table.cuboid[0].dims[:3] = cfg.table_dimensions
    world_cfg_table.cuboid[0].name = "table"

    # Add wall cuboid
    wall_cuboid_dict = {
        "table": {
            "dims": cfg.wall_dimensions.tolist(),
            "pose": list(cfg.wall_position) + [1, 0, 0, 0]
        }
    }
    wall_cfg = WorldConfig.from_dict({"cuboid": wall_cuboid_dict})
    wall_cfg.cuboid[0].name = "wall"

    # Add workbench cuboid
    workbench_cuboid_dict = {
        "table": {
            "dims": cfg.workbench_dimensions.tolist(),
            "pose": list(cfg.workbench_position) + [1, 0, 0, 0]
        }
    }
    workbench_cfg = WorldConfig.from_dict({"cuboid": workbench_cuboid_dict})
    workbench_cfg.cuboid[0].name = "workbench"

    # Add robot mount cuboid
    robot_mount_cuboid_dict = {
        "table": {
            "dims": cfg.robot_mount_dimensions.tolist(),
            "pose": list(cfg.robot_mount_position) + [1, 0, 0, 0]
        }
    }
    robot_mount_cfg = WorldConfig.from_dict({"cuboid": robot_mount_cuboid_dict})
    robot_mount_cfg.cuboid[0].name = "robot_mount"

    # Combine all cuboids
    all_cuboids = (
        world_cfg_table.cuboid +
        wall_cfg.cuboid +
        workbench_cfg.cuboid +
        robot_mount_cfg.cuboid
    )

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=all_cuboids, mesh=world_cfg1.mesh)

    # Create IK solver (needed for sphere visualization)
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

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
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
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
    print(f"\n{'='*60}")
    print("INITIALIZING SIMULATION")
    print(f"{'='*60}\n")

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
    print(f"\n{'='*60}")
    print("STARTING SIMULATION")
    print(f"{'='*60}")
    print(f"Total waypoints: {len(joint_targets)}")
    print(f"Interpolation steps: {cfg.interpolation_steps}")
    print(f"{'='*60}\n")

    # Setup trajectory queue
    target_queue: Deque[np.ndarray] = deque(joint_targets)
    active_trajectory: List[np.ndarray] = []
    trajectory_step = 0

    step_counter = 0
    idle_counter = 0
    viewpoint_counter = 0

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

        # Execute trajectory
        if active_trajectory and trajectory_step < len(active_trajectory):
            joint_cmd = active_trajectory[trajectory_step]
            world_state.robot.set_joint_positions(joint_cmd.tolist(), world_state.idx_list)
            trajectory_step += 1

            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0

        elif target_queue:
            # Reached viewpoint
            viewpoint_counter += 1

            # Get next target
            next_target = target_queue.popleft()
            current_state = get_active_joint_positions(world_state.robot, world_state.idx_list)

            active_trajectory = generate_interpolated_path(
                current_state,
                next_target,
                cfg.interpolation_steps
            )
            trajectory_step = 0

            if not active_trajectory:
                active_trajectory = [next_target]

            joint_cmd = active_trajectory[trajectory_step]
            world_state.robot.set_joint_positions(joint_cmd.tolist(), world_state.idx_list)
            trajectory_step += 1

            if trajectory_step >= len(active_trajectory):
                active_trajectory.clear()
                trajectory_step = 0

        # Check if trajectory complete
        if not target_queue and not active_trajectory:
            print(f"\n✓ Simulation completed!")
            print(f"Viewpoints reached: {viewpoint_counter}")
            break


# ============================================================================
# Main Entry Point
# ============================================================================
def main():
    """Main entry point"""
    cfg = SimulationConfig.from_args(args)

    print(f"\n{'='*60}")
    print("SIMULATE TRAJECTORY")
    print(f"{'='*60}")
    print(f"Trajectory: {cfg.trajectory_path}")
    print(f"Robot config: {cfg.robot_config_file}")
    print(f"{'='*60}\n")

    # Load joint trajectory
    joint_targets = load_joint_trajectory_csv(cfg.trajectory_path)

    # Initialize simulation
    world_state = initialize_simulation(cfg)

    # Run simulation
    run_simulation(world_state, joint_targets, cfg)

    simulation_app.close()


if __name__ == "__main__":
    main()
