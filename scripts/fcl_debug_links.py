#!/usr/bin/env python3
"""
Debug which robot links are colliding with glass
"""

import sys
sys.path.insert(0, '/isaac-sim/curobo/vision_inspection/scripts')

from fcl_check import FCLCollisionChecker, load_trajectory_csv
import numpy as np
from collections import Counter

# Load trajectory
trajectory, _ = load_trajectory_csv('data/output/3000/joint_trajectory_dp_20251107_050415.csv')

# Initialize checker with link meshes
checker = FCLCollisionChecker(
    robot_urdf_path='ur_description/ur20.urdf',
    obstacle_mesh_paths=['data/input/glass_o3d.obj'],
    glass_position=np.array([0.7, 0.0, 0.6]),
    use_link_meshes=True,
    mesh_base_path='ur_description'
)

print("\n" + "="*70)
print("LINK-LEVEL COLLISION ANALYSIS")
print("="*70)

# Analyze first 20 waypoints
print("\nAnalyzing first 20 waypoints:")
print("-" * 70)

collision_by_link = Counter()
total_collisions = 0

for i in range(20):
    joint_config = trajectory[i]
    is_collision, distance, link_info = checker.check_collision_single_config(
        joint_config, 
        return_distance=True,
        return_link_info=True
    )
    
    if is_collision:
        total_collisions += 1
        print(f"\nWaypoint {i}: COLLISION")
        
        # Find which links are colliding
        colliding_links = [info for info in link_info if info['collision']]
        
        if colliding_links:
            print(f"  Colliding links:")
            for link in colliding_links:
                print(f"    - {link['link_name']}: distance = {link['distance']:.4f}m")
                collision_by_link[link['link_name']] += 1
        
        # Show closest non-colliding links
        non_colliding = [info for info in link_info if not info['collision']]
        non_colliding.sort(key=lambda x: x['distance'])
        if non_colliding[:3]:
            print(f"  Closest non-colliding links:")
            for link in non_colliding[:3]:
                print(f"    - {link['link_name']}: distance = {link['distance']:.4f}m")
    else:
        print(f"\nWaypoint {i}: OK (min distance: {distance:.4f}m)")

print("\n" + "="*70)
print("COLLISION SUMMARY")
print("="*70)
print(f"Total waypoints checked: 20")
print(f"Collisions detected: {total_collisions}")
print(f"\nCollisions by link (most common first):")
for link_name, count in collision_by_link.most_common():
    print(f"  {link_name}: {count} collisions")

print("\n" + "="*70)
print("\nRECOMMENDATIONS:")
if collision_by_link:
    most_common_link = collision_by_link.most_common(1)[0][0]
    print(f"- The '{most_common_link}' link has the most collisions")
    print(f"- This suggests the robot is too close to the glass")
    print(f"- Consider adjusting:")
    print(f"  1. Glass position/orientation")
    print(f"  2. Trajectory offset distance")
    print(f"  3. Collision mesh scale")
else:
    print("- No collisions detected in first 20 waypoints")
print("="*70)
