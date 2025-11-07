#!/usr/bin/env python3
"""
Debug FCL collision detection by visualizing specific waypoints
"""

import sys
sys.path.insert(0, '/isaac-sim/curobo/vision_inspection/scripts')

from fcl_check import FCLCollisionChecker, load_trajectory_csv
import numpy as np

# Load trajectory
trajectory, _ = load_trajectory_csv('data/trajectory/joint_trajectory_dp_5000_base.csv')

# Initialize checker with link meshes
checker = FCLCollisionChecker(
    robot_urdf_path='ur_description/ur20.urdf',
    obstacle_mesh_paths=['data/input/glass_o3d.obj'],
    glass_position=np.array([0.7, 0.0, 0.6]),
    use_link_meshes=True,
    mesh_base_path='ur_description'
)

print("\n" + "="*70)
print("DETAILED COLLISION ANALYSIS")
print("="*70)

# Check first few waypoints with details
print("\nChecking first 10 waypoints:")
for i in range(10):
    joint_config = trajectory[i]
    is_collision, distance = checker.check_collision_single_config(
        joint_config, 
        return_distance=True
    )
    
    print(f"\nWaypoint {i}:")
    print(f"  Joint config: {joint_config}")
    print(f"  Collision: {is_collision}")
    print(f"  Distance: {distance:.4f}m")
    
    # Get target position from trajectory CSV
    import csv
    with open('data/trajectory/joint_trajectory_dp_5000_base.csv', 'r') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if idx == i:
                target_pos = [float(row['target-POS_X']), 
                             float(row['target-POS_Y']), 
                             float(row['target-POS_Z'])]
                print(f"  Target position: {target_pos}")
                break

print("\n" + "="*70)
print("\nGlass object info:")
print(f"  Position: {checker.glass_position}")
print(f"  Bounds: Check the mesh size")

import trimesh
glass_mesh = trimesh.load('data/input/glass_o3d.obj')
print(f"  Glass mesh bounds (local):")
print(f"    Min: {glass_mesh.bounds[0]}")
print(f"    Max: {glass_mesh.bounds[1]}")
print(f"  Glass mesh bounds (world, translated by {checker.glass_position}):")
world_min = glass_mesh.bounds[0] + checker.glass_position
world_max = glass_mesh.bounds[1] + checker.glass_position
print(f"    Min: {world_min}")
print(f"    Max: {world_max}")

print("\n" + "="*70)
print("\nSuggestions:")
print("1. Check if glass_position [0.7, 0.0, 0.6] matches simulation")
print("2. Check if robot end-effector is too close to glass")
print("3. The target position should be ~10cm away from glass surface")
print("="*70)
