#!/usr/bin/env python3
"""
World Setup Utilities for CuRobo Collision World Configuration

Provides centralized functions for setting up collision worlds with obstacles
(table, walls, workbench, robot mount, and meshes).
"""

import numpy as np
from typing import List, Optional

from curobo.geom.types import WorldConfig, Mesh
from curobo.util_file import get_world_configs_path, join_path, load_yaml

from . import config


def setup_collision_world(
    table_position: Optional[np.ndarray] = None,
    table_dimensions: Optional[np.ndarray] = None,
    wall_position: Optional[np.ndarray] = None,
    wall_dimensions: Optional[np.ndarray] = None,
    workbench_position: Optional[np.ndarray] = None,
    workbench_dimensions: Optional[np.ndarray] = None,
    robot_mount_position: Optional[np.ndarray] = None,
    robot_mount_dimensions: Optional[np.ndarray] = None,
    mesh_files: Optional[List[str]] = None,
    mesh_position: Optional[np.ndarray] = None,
    mesh_rotation: Optional[np.ndarray] = None,
    verbose: bool = True
) -> WorldConfig:
    """
    Setup collision world configuration with all obstacles

    Creates a WorldConfig containing table, wall, workbench, robot mount cuboids
    and optional mesh obstacles. All parameters default to values from config.py.

    Args:
        table_position: Table position (x, y, z) in meters
        table_dimensions: Table dimensions (x, y, z) in meters
        wall_position: Wall position (x, y, z) in meters
        wall_dimensions: Wall dimensions (x, y, z) in meters
        workbench_position: Workbench position (x, y, z) in meters
        workbench_dimensions: Workbench dimensions (x, y, z) in meters
        robot_mount_position: Robot mount position (x, y, z) in meters
        robot_mount_dimensions: Robot mount dimensions (x, y, z) in meters
        mesh_files: List of paths to obstacle mesh files
        mesh_position: Position for mesh obstacles (x, y, z) in meters
        mesh_rotation: Rotation for mesh obstacles as quaternion (w, x, y, z)
        verbose: Print setup information (default: True)

    Returns:
        WorldConfig containing all configured obstacles

    Example:
        >>> world_cfg = setup_collision_world(
        ...     mesh_files=["data/object/glass.obj"],
        ...     verbose=True
        ... )
        Setting up collision world...
          Table: [0. 0. 0.] dims=[2. 2. 0.05]
          ...
    """
    # Apply config defaults
    if table_position is None:
        table_position = config.TABLE_POSITION.copy()
    if table_dimensions is None:
        table_dimensions = config.TABLE_DIMENSIONS.copy()
    if wall_position is None:
        wall_position = config.WALL_POSITION.copy()
    if wall_dimensions is None:
        wall_dimensions = config.WALL_DIMENSIONS.copy()
    if workbench_position is None:
        workbench_position = config.WORKBENCH_POSITION.copy()
    if workbench_dimensions is None:
        workbench_dimensions = config.WORKBENCH_DIMENSIONS.copy()
    if robot_mount_position is None:
        robot_mount_position = config.ROBOT_MOUNT_POSITION.copy()
    if robot_mount_dimensions is None:
        robot_mount_dimensions = config.ROBOT_MOUNT_DIMENSIONS.copy()
    if mesh_position is None:
        mesh_position = config.TARGET_OBJECT_POSITION.copy()
    if mesh_rotation is None:
        mesh_rotation = config.TARGET_OBJECT_ROTATION.copy()
    if mesh_files is None:
        mesh_files = []

    if verbose:
        print("\nSetting up collision world...")

    # Load base world config (table)
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[:3] = table_position
    world_cfg_table.cuboid[0].dims[:3] = table_dimensions
    world_cfg_table.cuboid[0].name = "table"

    if verbose:
        print(f"  Table: pos={table_position}, dims={table_dimensions}")

    # Add wall cuboid
    wall_cuboid_dict = {
        "table": {
            "dims": wall_dimensions.tolist(),
            "pose": list(wall_position) + [1, 0, 0, 0]
        }
    }
    wall_cfg = WorldConfig.from_dict({"cuboid": wall_cuboid_dict})
    wall_cfg.cuboid[0].name = "wall"

    if verbose:
        print(f"  Wall: pos={wall_position}, dims={wall_dimensions}")

    # Add workbench cuboid
    workbench_cuboid_dict = {
        "table": {
            "dims": workbench_dimensions.tolist(),
            "pose": list(workbench_position) + [1, 0, 0, 0]
        }
    }
    workbench_cfg = WorldConfig.from_dict({"cuboid": workbench_cuboid_dict})
    workbench_cfg.cuboid[0].name = "workbench"

    if verbose:
        print(f"  Workbench: pos={workbench_position}, dims={workbench_dimensions}")

    # Add robot mount cuboid
    robot_mount_cuboid_dict = {
        "table": {
            "dims": robot_mount_dimensions.tolist(),
            "pose": list(robot_mount_position) + [1, 0, 0, 0]
        }
    }
    robot_mount_cfg = WorldConfig.from_dict({"cuboid": robot_mount_cuboid_dict})
    robot_mount_cfg.cuboid[0].name = "robot_mount"

    if verbose:
        print(f"  Robot mount: pos={robot_mount_position}, dims={robot_mount_dimensions}")

    # Add mesh obstacles
    meshes = []
    for i, mesh_file in enumerate(mesh_files):
        mesh = Mesh(
            name=f"obstacle_mesh_{i}",
            file_path=mesh_file,
            pose=list(mesh_position) + list(mesh_rotation),  # position + quat (w,x,y,z)
        )
        meshes.append(mesh)
        if verbose:
            print(f"  Mesh {i}: {mesh_file} at pos={mesh_position}")

    # Combine all cuboids and meshes
    all_cuboids = (
        world_cfg_table.cuboid +
        wall_cfg.cuboid +
        workbench_cfg.cuboid +
        robot_mount_cfg.cuboid
    )

    world_cfg = WorldConfig(
        cuboid=all_cuboids,
        mesh=meshes
    )

    if verbose:
        print(f"  Total obstacles: {len(all_cuboids)} cuboids + {len(meshes)} meshes")

    return world_cfg


def setup_collision_world_from_config(
    obstacle_cfg: 'config.WorldObstacleConfig',
    verbose: bool = True
) -> WorldConfig:
    """
    Setup collision world from WorldObstacleConfig dataclass

    Convenience wrapper around setup_collision_world() that accepts
    a WorldObstacleConfig instance instead of individual parameters.

    Args:
        obstacle_cfg: WorldObstacleConfig instance with obstacle configuration
        verbose: Print setup information (default: True)

    Returns:
        WorldConfig containing all configured obstacles

    Example:
        >>> from common import config
        >>> obstacle_cfg = config.WorldObstacleConfig.from_object_name("glass")
        >>> world_cfg = setup_collision_world_from_config(obstacle_cfg)
        Setting up collision world...
          Table: [1. 0. -0.425] dims=[0.6 1.  0.5]
          ...
    """
    return setup_collision_world(**obstacle_cfg.to_world_setup_kwargs(), verbose=verbose)
