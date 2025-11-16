#!/usr/bin/env python3
"""
Simple FCL CCD test without trimesh
"""
import fcl
import numpy as np
import pinocchio as pin
import yaml
import csv

# Default obstacle positions (same as coal_check.py)
DEFAULT_GLASS_POSITION = np.array([1.0, 0.0, -0.13])
DEFAULT_GLASS_DIMENSIONS = np.array([0.2, 0.2, 0.08])  # Approximate glass as cuboid
DEFAULT_TABLE_POSITION = np.array([1.0, 0.0, -0.425])
DEFAULT_TABLE_DIMENSIONS = np.array([0.6, 1.0, 0.5])
DEFAULT_WALL_POSITION = np.array([-1.1, 0.0, 0.5])
DEFAULT_WALL_DIMENSIONS = np.array([0.1, 2.2, 1.0])
DEFAULT_WORKBENCH_POSITION = np.array([0.35, -1.1, 0.5])
DEFAULT_WORKBENCH_DIMENSIONS = np.array([3.0, 0.1, 1.0])
DEFAULT_ROBOT_MOUNT_POSITION = np.array([0.0, 0.0, -0.25])
DEFAULT_ROBOT_MOUNT_DIMENSIONS = np.array([0.3, 0.3, 0.5])


def load_obj_simple(filepath):
    """Simple OBJ file loader"""
    vertices = []
    faces = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                face_vertices = []
                for i in range(1, 4):
                    vertex_str = parts[i].split('/')[0]
                    face_vertices.append(int(vertex_str) - 1)
                faces.append(face_vertices)

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int32)

# Test without importing trimesh
print("Testing FCL CCD without trimesh...")

# Load robot
print("Loading robot...")
model = pin.buildModelFromUrdf('../ur_description/ur20.urdf')
data = model.createData()
print(f"Robot: {model.nq} DOF")

# Load collision spheres
print("Loading collision spheres...")
with open('../ur_description/ur20.yml', 'r') as f:
    config = yaml.safe_load(f)

collision_spheres = {}
robot_cfg = config.get('robot_cfg', {})
kinematics = robot_cfg.get('kinematics', {})
collision_spheres_cfg = kinematics.get('collision_spheres', {})

for link_name, spheres in collision_spheres_cfg.items():
    collision_spheres[link_name] = []
    for sphere in spheres:
        center = np.array(sphere['center'])
        radius = sphere['radius']
        if radius > 0:
            collision_spheres[link_name].append({
                'center': center,
                'radius': radius
            })

print(f"Loaded {len(collision_spheres)} links with spheres")

# Create obstacles (using cuboids only - mesh loading causes segfault in Isaac Sim)
print("\nCreating obstacles...")
obstacles = []

# Add all cuboid obstacles (including glass as cuboid)
for name, pos, dims in [
    ("Glass", DEFAULT_GLASS_POSITION, DEFAULT_GLASS_DIMENSIONS),
    ("Table", DEFAULT_TABLE_POSITION, DEFAULT_TABLE_DIMENSIONS),
    ("Wall", DEFAULT_WALL_POSITION, DEFAULT_WALL_DIMENSIONS),
    ("Workbench", DEFAULT_WORKBENCH_POSITION, DEFAULT_WORKBENCH_DIMENSIONS),
    ("Robot Mount", DEFAULT_ROBOT_MOUNT_POSITION, DEFAULT_ROBOT_MOUNT_DIMENSIONS),
]:
    box = fcl.Box(*dims)
    tf = fcl.Transform(pos)
    obj = fcl.CollisionObject(box, tf)
    obstacles.append((obj, tf))
    print(f"  {name} cuboid added at {pos}")

print(f"Total obstacles: {len(obstacles)}")

# Load trajectory
print("Loading trajectory...")
joint_angles = []
with open('../data/trajectory/675/joint_trajectory_dp.csv', 'r') as f:
    reader = csv.DictReader(f)
    headers = reader.fieldnames
    joint_names = [h for h in headers if 'joint' in h.lower()]

    for row in reader:
        config = [float(row[joint_name]) for joint_name in joint_names]
        joint_angles.append(config)

trajectory = np.array(joint_angles)
print(f"Loaded {len(trajectory)} waypoints")

# Test CCD on first segment
print("\nTesting CCD on first segment...")
q_start = trajectory[0]
q_end = trajectory[1]

# Compute FK for start config
pin.forwardKinematics(model, data, q_start)
pin.updateFramePlacements(model, data)

# Create robot spheres at start
robot_start = []
for link_name, spheres in collision_spheres.items():
    try:
        frame_id = model.getFrameId(link_name)
        transform_matrix = data.oMf[frame_id]
    except:
        try:
            joint_id = model.getJointId(link_name)
            transform_matrix = data.oMi[joint_id]
        except:
            continue

    for sphere_def in spheres:
        center_local = sphere_def['center']
        radius = sphere_def['radius']
        center_world = transform_matrix.translation + transform_matrix.rotation @ center_local

        sphere = fcl.Sphere(radius)
        tf = fcl.Transform(center_world)
        obj = fcl.CollisionObject(sphere, tf)
        robot_start.append(obj)

print(f"Created {len(robot_start)} robot spheres at start")

# Compute FK for end config
pin.forwardKinematics(model, data, q_end)
pin.updateFramePlacements(model, data)

# Create robot spheres at end
robot_end = []
for link_name, spheres in collision_spheres.items():
    try:
        frame_id = model.getFrameId(link_name)
        transform_matrix = data.oMf[frame_id]
    except:
        try:
            joint_id = model.getJointId(link_name)
            transform_matrix = data.oMi[joint_id]
        except:
            continue

    for sphere_def in spheres:
        center_local = sphere_def['center']
        radius = sphere_def['radius']
        center_world = transform_matrix.translation + transform_matrix.rotation @ center_local

        sphere_end_tf = fcl.Transform(center_world)
        robot_end.append(sphere_end_tf)

print(f"Created {len(robot_end)} end transforms")

# Perform CCD on all segments
print("\nPerforming CCD on all segments...")
import time
from pathlib import Path
from datetime import datetime

start_time = time.perf_counter()
collision_segments = []

for seg_idx in range(len(trajectory) - 1):
    if (seg_idx + 1) % 100 == 0:
        print(f"  Progress: {seg_idx+1}/{len(trajectory)-1} segments")

    q_seg_start = trajectory[seg_idx]
    q_seg_end = trajectory[seg_idx + 1]

    # Create robot at start
    pin.forwardKinematics(model, data, q_seg_start)
    pin.updateFramePlacements(model, data)

    robot_seg_start = []
    for link_name, spheres in collision_spheres.items():
        try:
            frame_id = model.getFrameId(link_name)
            transform_matrix = data.oMf[frame_id]
        except:
            try:
                joint_id = model.getJointId(link_name)
                transform_matrix = data.oMi[joint_id]
            except:
                continue

        for sphere_def in spheres:
            center_local = sphere_def['center']
            radius = sphere_def['radius']
            center_world = transform_matrix.translation + transform_matrix.rotation @ center_local

            sphere = fcl.Sphere(radius)
            tf = fcl.Transform(center_world)
            obj = fcl.CollisionObject(sphere, tf)
            robot_seg_start.append(obj)

    # Create robot at end
    pin.forwardKinematics(model, data, q_seg_end)
    pin.updateFramePlacements(model, data)

    robot_seg_end_tfs = []
    for link_name, spheres in collision_spheres.items():
        try:
            frame_id = model.getFrameId(link_name)
            transform_matrix = data.oMf[frame_id]
        except:
            try:
                joint_id = model.getJointId(link_name)
                transform_matrix = data.oMi[joint_id]
            except:
                continue

        for sphere_def in spheres:
            center_local = sphere_def['center']
            center_world = transform_matrix.translation + transform_matrix.rotation @ center_local
            tf_end_sphere = fcl.Transform(center_world)
            robot_seg_end_tfs.append(tf_end_sphere)

    # CCD check against all obstacles
    is_collision = False
    for i in range(len(robot_seg_start)):
        for obstacle_obj, obstacle_tf in obstacles:
            request = fcl.ContinuousCollisionRequest()
            result = fcl.ContinuousCollisionResult()

            toc = fcl.continuousCollide(robot_seg_start[i], robot_seg_end_tfs[i],
                                       obstacle_obj, obstacle_tf,
                                       request, result)

            if result.is_collide:
                is_collision = True
                collision_segments.append((seg_idx, toc))
                break
        if is_collision:
            break

check_time = time.perf_counter() - start_time

# Print results
num_segments = len(trajectory) - 1
num_collisions = len(collision_segments)
collision_rate = (num_collisions / num_segments * 100) if num_segments > 0 else 0

print(f"\n{'='*50}")
print("CCD Collision Check Results")
print(f"{'='*50}")
print(f"Total segments: {num_segments}")
print(f"Collision segments: {num_collisions}")
print(f"Collision-free segments: {num_segments - num_collisions}")
print(f"Collision rate: {collision_rate:.2f}%")
print(f"Check time: {check_time:.3f} seconds")
print(f"{'='*50}")

# Save report
report_dir = Path('../data/collision') / str(len(trajectory))
report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / 'collision_ccd.txt'

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
segment_indices = [seg_idx for seg_idx, _ in collision_segments]

lines = [
    f"=== CCD Collision Report @ {timestamp} ===",
    f"Trajectory: ../data/trajectory/675/joint_trajectory_dp.csv",
    f"Robot URDF: ../ur_description/ur20.urdf",
    f"Robot config: ../ur_description/ur20.yml",
    f"Obstacles: {len(obstacles)} cuboids (Glass, Table, Wall, Workbench, Robot Mount)",
    f"Note: Glass approximated as cuboid due to BVHModel issues in Isaac Sim",
    f"CCD enabled: true (FCL continuousCollide)",
    "",
    f"Total waypoints: {len(trajectory)}",
    f"Total segments: {num_segments}",
    f"Collision segments: {num_collisions}",
    f"Collision-free segments: {num_segments - num_collisions}",
    f"Collision rate (%): {collision_rate:.2f}",
    "",
    f"Collision segment indices: {', '.join(map(str, segment_indices[:50]))}",
    "",
    f"Timing: {check_time:.3f} s",
]

content = "\n".join(lines)
if report_path.exists() and report_path.stat().st_size > 0:
    with open(report_path, 'a') as f:
        f.write("\n\n" + content)
else:
    with open(report_path, 'w') as f:
        f.write(content)

print(f"\nReport saved to: {report_path}")
print("Test completed successfully!")
