# FCL Collision Checker Documentation

## Overview
This document describes the FCL-based collision detection system for robot trajectory validation. The system uses `python-fcl` (Flexible Collision Library) for collision checking and `pinocchio` for forward kinematics computation.

## Purpose
The FCL collision checker validates robot trajectories generated for vision inspection tasks by:
- Computing forward kinematics for each robot configuration
- Checking collisions between robot links and environment obstacles
- Identifying which specific links are colliding
- Providing collision statistics and analysis

## System Architecture

### Core Components

#### 1. **fcl_check.py** - Main Collision Checker
The primary collision detection system with three collision geometry modes.

**Key Class: `FCLCollisionChecker`**

**Initialization Parameters:**
- `robot_urdf_path`: Path to robot URDF file
- `obstacle_mesh_paths`: List of obstacle mesh file paths
- `glass_position`: 3D position of glass object in world frame (default: [0.7, 0.0, 0.6])
- `robot_config_path`: Path to CuRobo robot config YAML (for collision spheres)
- `use_capsules`: Use capsule approximations (default: False)
- `capsule_radius`: Radius for capsule geometry (default: 0.05m)
- `use_link_meshes`: Use actual URDF collision meshes (default: False, most accurate)
- `mesh_base_path`: Base path for robot mesh files (default: "ur_description")
- `collision_margin`: Safety margin in meters (positive = more conservative, negative = less conservative)

**Three Collision Geometry Modes:**

1. **Link Meshes Mode** (`use_link_meshes=True`)
   - Uses actual collision meshes from URDF
   - Most accurate representation
   - Loads STL meshes for each link:
     - base_link → base.stl
     - shoulder_link → shoulder.stl
     - upper_arm_link → upperarm.stl
     - forearm_link → forearm.stl
     - wrist_1_link → wrist1.stl
     - wrist_2_link → wrist2.stl
     - wrist_3_link → wrist3.stl
   - Higher computational cost

2. **Collision Spheres Mode** (`robot_config_path` provided)
   - Loads sphere approximations from CuRobo YAML config
   - Each link represented by multiple spheres
   - Balances accuracy and performance
   - Spheres defined with center position and radius

3. **Capsules Mode** (`use_capsules=True`)
   - Approximates links as capsules (cylinders with rounded ends)
   - Fastest computation
   - Default capsule definitions for UR20:
     - shoulder_link: 0.15m length
     - upper_arm_link: 0.80m length
     - forearm_link: 0.80m length
     - wrist_1_link: 0.15m length
     - wrist_2_link: 0.15m length
     - wrist_3_link: 0.10m length

#### 2. **fcl_debug.py** - Waypoint-Level Debugging
Detailed analysis of individual trajectory waypoints.

**Features:**
- Checks first 10 waypoints in detail
- Shows joint configuration for each waypoint
- Reports collision status and minimum distance
- Extracts target positions from CSV
- Displays glass mesh bounds in both local and world coordinates

**Use Cases:**
- Debugging why specific waypoints fail
- Verifying glass position matches simulation
- Checking if end-effector is too close to glass
- Validating that targets are ~10cm from glass surface

#### 3. **fcl_debug_links.py** - Link-Level Collision Analysis
Identifies which specific robot links are colliding.

**Features:**
- Analyzes first 20 waypoints
- Reports which links are colliding for each waypoint
- Counts collision frequency per link
- Shows distance for both colliding and non-colliding links
- Provides recommendations based on most common colliding link

**Output Statistics:**
- Total waypoints checked
- Number of collisions detected
- Collisions by link (sorted by frequency)
- Distance information for closest non-colliding links

## Implementation Details

### Forward Kinematics with Pinocchio

The system uses Pinocchio for FK computation:

```python
# Compute FK for all joints
pin.forwardKinematics(self.robot_model, self.robot_data, joint_positions)
pin.updateFramePlacements(self.robot_model, self.robot_data)

# Access link transforms
transform_matrix = self.robot_data.oMf[frame_id]  # For frames
transform_matrix = self.robot_data.oMi[joint_id]  # For joints

# Extract position and rotation
position = transform_matrix.translation
rotation = transform_matrix.rotation
```

### FCL Collision Detection

**Mesh-based Collision Objects:**
```python
# Create BVH (Bounding Volume Hierarchy) model
bvh = fcl.BVHModel()
bvh.beginModel(len(vertices), len(triangles))
bvh.addSubModel(vertices, triangles)
bvh.endModel()

# Create collision object with transform
transform = fcl.Transform(rotation, position)
collision_object = fcl.CollisionObject(bvh, transform)
```

**Collision Checking:**
```python
# Check collision
request = fcl.CollisionRequest()
result = fcl.CollisionResult()
ret = fcl.collide(robot_obj, obstacle_obj, request, result)

# Check distance
dist_request = fcl.DistanceRequest()
dist_result = fcl.DistanceResult()
dist = fcl.distance(robot_obj, obstacle_obj, dist_request, dist_result)
```

### Coordinate Transformations

**IMPORTANT - Coordinate System Consistency:**
- **Pinocchio** loads URDF which uses **Z-up** coordinate system
- **Robot FK calculations** are done in **Z-up** coordinates
- **Obstacle meshes MUST use Z-up** to match the robot coordinate system
- **Use `glass_zup.obj`**, NOT `glass_yup.obj`!

**Why Z-up for fcl_check.py:**
```
run_app_v3.py generates trajectory:
  Isaac Sim (Z-up) → Joint angles in Z-up coordinate system

fcl_check.py validates trajectory:
  Pinocchio loads URDF (Z-up) → Robot FK in Z-up
  Glass mesh must be Z-up → Collision check in consistent coordinate system
```

**Glass Object Positioning:**
- Glass mesh loaded in **Z-up coordinates** (glass_zup.obj)
- Transformed to world position using `glass_position`
- FCL transform applies translation to obstacle

**Robot Link Transforms:**
- Computed via Pinocchio FK in **Z-up**
- Converted to FCL transforms
- Applied to collision geometry (meshes/spheres/capsules)

**Sphere Transform (from YAML):**
```python
# Transform sphere center to world frame
center_world = transform_matrix.translation + transform_matrix.rotation @ center_local
```

## Trajectory Analysis

### Input Format

**CSV Trajectory File:**
- Header row with joint column names (containing 'joint' in name)
- Each row represents one waypoint
- Columns include joint angles and target positions
- Example: `joint_0`, `joint_1`, ..., `target-POS_X`, `target-POS_Y`, `target-POS_Z`

### Collision Check Process

```python
def check_trajectory(trajectory, verbose=True, show_link_collisions=False):
    """
    For each waypoint:
    1. Compute robot FK
    2. Create collision geometry
    3. Check against all obstacles
    4. Record collision indices
    5. Collect statistics
    """
```

### Output Statistics

**Returned Dictionary:**
- `total_waypoints`: Total number of waypoints checked
- `num_collisions`: Number of colliding waypoints
- `num_collision_free`: Number of safe waypoints
- `collision_rate`: Percentage of colliding waypoints
- `collision_indices`: List of indices where collision occurred
- `collision_free_indices`: List of safe waypoint indices
- `link_collisions`: Dictionary mapping link names to collision counts

## Usage Examples

### Basic Trajectory Check

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --glass_position 0.7 0.0 0.6

# IMPORTANT: Use glass_zup.obj (Z-up) to match Pinocchio/URDF coordinate system!
# Using glass_yup.obj will result in incorrect collision detection.
```

### Using Link Meshes (Most Accurate)

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf ur_description/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --use_link_meshes \
    --mesh_base_path ur_description
```

### Using Collision Spheres from YAML

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --robot_config ur20.yml
```

### Using Capsule Approximations

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --use_capsules \
    --capsule_radius 0.05
```

### With Collision Margin

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --collision_margin -0.05  # 5cm tolerance (less conservative)
```

### Link-Level Analysis

```bash
python scripts/fcl_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --show_link_collisions \
    --verbose
```

### Debug Specific Waypoints

```bash
python scripts/fcl_debug.py
```

### Analyze Link Collisions

```bash
python scripts/fcl_debug_links.py
```

## Configuration Files

### Robot Config YAML (ur20.yml)

Expected structure for collision spheres:
```yaml
robot_cfg:
  kinematics:
    collision_spheres:
      base_link:
        - center: [0.0, 0.0, 0.0]
          radius: 0.1
      shoulder_link:
        - center: [0.0, 0.0, 0.05]
          radius: 0.08
        - center: [0.0, 0.0, 0.10]
          radius: 0.08
      # ... more links and spheres
```

### URDF Structure

Expected collision mesh paths in URDF:
```
ur_description/
├── meshes/
│   └── ur20/
│       └── collision/
│           ├── base.stl
│           ├── shoulder.stl
│           ├── upperarm.stl
│           ├── forearm.stl
│           ├── wrist1.stl
│           ├── wrist2.stl
│           └── wrist3.stl
```

## Performance Considerations

### Accuracy vs Speed Trade-off

1. **Link Meshes** (Highest Accuracy, Slowest)
   - Use for final validation
   - Most accurate collision detection
   - Recommended for critical applications

2. **Collision Spheres** (Balanced)
   - Good accuracy with acceptable speed
   - Use for iterative development
   - Matches CuRobo collision model

3. **Capsules** (Fastest, Lowest Accuracy)
   - Quick preliminary checks
   - May miss collisions or report false positives
   - Requires manual tuning of capsule dimensions

### Optimization Tips

- Use `collision_margin` to adjust conservativeness
- Negative margin allows closer approach (less conservative)
- Positive margin adds safety buffer (more conservative)
- Set `verbose=False` for faster checking without progress updates
- Use `show_link_collisions` only when debugging specific issues

## Debugging Workflow

### Step 1: Run Full Trajectory Check
```bash
python scripts/fcl_check.py --trajectory <path> --use_link_meshes
```

### Step 2: If Collisions Found, Analyze Links
```bash
python scripts/fcl_debug_links.py
```
This identifies which links are problematic.

### Step 3: Debug Specific Waypoints
```bash
python scripts/fcl_debug.py
```
This shows detailed info for first 10 waypoints.

### Step 4: Adjust Parameters
Based on findings:
- Adjust `glass_position` if position mismatch
- Modify `collision_margin` to tune sensitivity
- Update trajectory generation parameters
- Check mesh scales and transforms

## Common Issues and Solutions

### Issue: High Collision Rate

**Possible Causes:**
- Glass position doesn't match simulation
- Target points too close to glass surface
- Collision margin too conservative

**Solutions:**
- Verify glass_position matches simulation world
- Increase NORMAL_SAMPLE_OFFSET in trajectory generation
- Use negative collision_margin for tolerance

### Issue: Specific Link Always Collides

**Possible Causes:**
- Link mesh/sphere/capsule too large
- Robot configuration brings link too close
- Trajectory planning issue

**Solutions:**
- Adjust collision geometry size
- Review IK solutions and joint limits
- Regenerate trajectory with different parameters

### Issue: Inconsistent Results

**Possible Causes:**
- Using different collision models than CuRobo
- Coordinate frame mismatch
- Mesh scale differences

**Solutions:**
- Use same collision spheres as CuRobo (load from YAML)
- Verify coordinate transforms
- Check mesh units and scaling

## Integration with Vision Inspection System

The FCL checker complements the main vision inspection system:

1. **CuRobo** (in run_app.py):
   - Generates trajectories
   - Uses GPU-accelerated collision checking
   - Integrated with motion planning

2. **FCL Checker** (fcl_check.py):
   - Independent validation
   - CPU-based checking
   - Detailed per-link analysis
   - Debugging and verification

**Recommended Workflow:**
1. Generate trajectories using CuRobo
2. Validate with FCL checker using same collision model
3. Debug any discrepancies with fcl_debug scripts
4. Adjust parameters and regenerate if needed

## Dependencies

```python
import fcl                    # python-fcl: Collision detection
import pinocchio as pin       # Forward kinematics
import trimesh               # Mesh loading and processing
import numpy as np           # Numerical operations
import yaml                  # Config file parsing
```

**Installation:**
```bash
pip install python-fcl
pip install pin  # or pinocchio
pip install trimesh
pip install pyyaml
```

## API Reference

### FCLCollisionChecker

**Methods:**

#### `__init__(...)`
Initialize collision checker with robot and obstacles.

#### `check_collision_single_config(joint_positions, return_distance=False, return_link_info=False)`
Check collision for one robot configuration.

**Returns:**
- `is_collision` (bool): Whether collision detected
- `distance` (float): Minimum distance to obstacles
- `link_info` (list): Per-link collision details (if requested)

#### `check_trajectory(trajectory, verbose=True, show_link_collisions=False, max_show=10)`
Check entire trajectory for collisions.

**Returns:**
- Dictionary with collision statistics

### Utility Functions

#### `load_trajectory_csv(csv_path)`
Load joint trajectory from CSV file.

**Returns:**
- `trajectory` (np.ndarray): (N, 6) array of joint angles
- `joint_names` (list): Joint column names

## Output Interpretation

### Console Output Example

```
======================================================================
FCL Collision Checker for Robot Trajectories
======================================================================

1. Loading trajectory from: data/trajectory/joint_trajectory_dp_5000_base.csv
Loaded trajectory: 5000 waypoints, 6 joints

2. Initializing collision checker
   Robot URDF: data/input/ur20.urdf
   Use link meshes: True

Loading obstacle meshes...
  Loaded: data/input/glass_o3d.obj (1234 vertices, 2468 faces)

Loading robot model with Pinocchio...
  Robot loaded: 6 DOF, 7 joints
  Using actual collision meshes from URDF
  Loaded 7 link meshes

3. Running collision detection
Checking 5000 waypoints for collisions...
  Progress: 100/5000 waypoints checked
  ...

======================================================================
COLLISION CHECK RESULTS
======================================================================
Total waypoints:        5000
Collision-free:         4523
Collisions detected:    477
Collision rate:         9.54%

Collision waypoint indices (first 50):
  [12, 45, 67, 89, ...]

Collisions by link:
  wrist_3_link: 245 (51.4% of collisions)
  forearm_link: 123 (25.8% of collisions)
  wrist_2_link: 109 (22.8% of collisions)
======================================================================
```

## Best Practices

1. **Always use link meshes for final validation**
2. **Match collision model with motion planner**
3. **Debug incrementally** (full check → link analysis → waypoint details)
4. **Verify coordinate frames** before large-scale checks
5. **Document collision margin settings** used
6. **Save collision analysis results** for comparison
7. **Cross-validate with simulation** visualization

## Future Enhancements

Potential improvements:
- Continuous collision detection (swept volumes)
- Parallel trajectory checking (multi-threading)
- Export collision visualization (meshes with highlights)
- Integration with trajectory optimization
- Real-time collision monitoring mode
- Support for multiple robot configurations

## References

- **FCL Documentation**: https://github.com/flexible-collision-library/fcl
- **Pinocchio Documentation**: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/
- **CuRobo**: https://curobo.org/
- **Isaac Sim**: https://developer.nvidia.com/isaac-sim

---

**File Location:** `/isaac-sim/curobo/vision_inspection/scripts/`

**Related Files:**
- `fcl_check.py` - Main collision checker (scripts/fcl_check.py:1)
- `fcl_debug.py` - Waypoint debugging (scripts/fcl_debug.py:1)
- `fcl_debug_links.py` - Link-level analysis (scripts/fcl_debug_links.py:1)
- `run_app_v3.py` - Vision inspection main application

**Last Updated:** 2025-11-07
