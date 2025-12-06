# COAL Collision Checker Documentation

## Overview
This document describes the COAL-based collision detection system for robot trajectory validation. The system uses COAL (Collision and Occupancy Algorithms Library) for collision checking and `pinocchio` for forward kinematics computation.

**COAL** is an improved version of FCL with 5-15x better performance, improved numerical stability, and native Pinocchio integration.

## Purpose
The COAL collision checker validates robot trajectories generated for vision inspection tasks by:
- Computing forward kinematics for each robot configuration
- Checking collisions between robot links and environment obstacles
- Identifying which specific links are colliding
- Providing collision statistics and analysis
- Delivering significantly faster performance compared to FCL

## System Architecture

### Core Components

#### 1. **coal_check.py** - Main Collision Checker
The primary collision detection system with three collision geometry modes, powered by COAL library.

**Key Class: `COALCollisionChecker`**

**Initialization Parameters:**
- `robot_urdf_path`: Path to robot URDF file
- `obstacle_mesh_paths`: List of obstacle mesh file paths
- `glass_position`: 3D position of glass object in world frame (default: [0.7, 0.0, 0.6])
- `table_position`: 3D position of table cuboid in world frame (default: [0.7, 0.0, 0.0])
- `table_dimensions`: Dimensions of table cuboid (x, y, z) in meters (default: [0.6, 1.0, 1.1])
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
   - Higher computational cost (but still faster than FCL)

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

## Implementation Details

### Forward Kinematics with Pinocchio

The system uses Pinocchio for FK computation (identical to FCL version):

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

### COAL Collision Detection

**Key Differences from FCL:**

COAL uses `Transform3s` with setter methods instead of constructor parameters:

```python
# Create transform (COAL way)
transform = coal.Transform3s()
transform.setRotation(rotation_matrix)  # 3x3 numpy array
transform.setTranslation(translation_vector)  # 3D numpy array
```

**Helper Method:**
```python
def _create_transform(self, rotation=None, translation=None) -> coal.Transform3s:
    """Helper to create COAL Transform3s"""
    transform = coal.Transform3s()
    if rotation is not None:
        transform.setRotation(rotation)
    if translation is not None:
        transform.setTranslation(translation)
    return transform
```

**Mesh-based Collision Objects:**
```python
# Create BVH (Bounding Volume Hierarchy) model
bvh = coal.BVHModel()
bvh.beginModel(len(vertices), len(triangles))
bvh.addSubModel(vertices, triangles)
bvh.endModel()

# Create collision object with transform
transform = coal.Transform3s()
transform.setRotation(rotation)
transform.setTranslation(position)
collision_object = coal.CollisionObject(bvh, transform)
```

**Collision Checking:**
```python
# Check collision
request = coal.CollisionRequest()
request.security_margin = collision_margin  # COAL feature
result = coal.CollisionResult()
ret = coal.collide(robot_obj, obstacle_obj, request, result)

# Check distance
dist_request = coal.DistanceRequest()
dist_request.enable_signed_distance = True  # COAL feature
dist_result = coal.DistanceResult()
dist = coal.distance(robot_obj, obstacle_obj, dist_request, dist_result)
```

**COAL-Specific Features:**
- `security_margin`: Built-in safety margin support
- `enable_signed_distance`: Get signed distance for penetrating objects
- Improved GJK/EPA algorithms for better numerical stability
- Accelerated collision detection (Nesterov-style optimization)

### Coordinate Transformations

**IMPORTANT - Coordinate System Consistency:**
- **Pinocchio** loads URDF which uses **Z-up** coordinate system
- **Robot FK calculations** are done in **Z-up** coordinates
- **Obstacle meshes MUST use Z-up** to match the robot coordinate system
- **Use `glass_zup.obj`**, NOT `glass_yup.obj`!

**Why Z-up for coal_check.py:**
```
run_app_v3.py generates trajectory:
  Isaac Sim (Z-up) → Joint angles in Z-up coordinate system

coal_check.py validates trajectory:
  Pinocchio loads URDF (Z-up) → Robot FK in Z-up
  Glass mesh must be Z-up → Collision check in consistent coordinate system
```

**Glass Object Positioning:**
- Glass mesh loaded in **Z-up coordinates** (glass_zup.obj)
- Transformed to world position using `glass_position`
- COAL transform applies translation to obstacle

**Table Cuboid:**
- Created using `coal.Box(x, y, z)` with full dimensions
- Positioned at `table_position` in world frame
- No mesh file needed - programmatically generated

**Robot Link Transforms:**
- Computed via Pinocchio FK in **Z-up**
- Converted to COAL transforms using `_create_transform()` helper
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
def check_trajectory(trajectory, verbose=True, show_link_collisions=False,
                     interpolate=True, num_interp_steps=30):
    """
    For each waypoint:
    1. Compute robot FK
    2. Create collision geometry
    3. Check against all obstacles
    4. Record collision indices
    5. Collect statistics

    If interpolation enabled:
    6. Generate intermediate configurations between waypoints
    7. Check each interpolated configuration
    8. Record segment collisions with alpha values
    """
```

### Interpolation Support

**New Feature:** COAL checker supports trajectory interpolation to detect collisions between waypoints.

**Parameters:**
- `interpolate`: Enable/disable interpolation (default: True)
- `num_interp_steps`: Number of steps between waypoints (default: 30)

**Benefits:**
- Detects collisions that occur during motion between waypoints
- More comprehensive collision coverage
- Matches run_app_v3.py behavior

**Example:**
```python
# Check with interpolation (default)
results = checker.check_trajectory(
    trajectory,
    interpolate=True,
    num_interp_steps=30
)

# Disable interpolation for faster checking
results = checker.check_trajectory(
    trajectory,
    interpolate=False
)
```

### Output Statistics

**Returned Dictionary:**
- `total_waypoints`: Total number of waypoints checked
- `total_configs_checked`: Total configurations checked (including interpolated)
- `interpolate`: Whether interpolation was used
- `num_interp_steps`: Number of interpolation steps (if used)
- `num_collisions`: Number of colliding waypoints
- `num_segment_collisions`: Number of collisions in interpolated segments
- `total_collisions`: Total collisions (waypoints + segments)
- `num_collision_free`: Number of safe configurations
- `collision_rate`: Percentage of colliding configurations
- `collision_indices`: List of waypoint indices where collision occurred
- `collision_segments`: List of (waypoint_idx, alpha) tuples for segment collisions
- `collision_free_indices`: List of safe waypoint indices
- `link_collisions`: Dictionary mapping link names to collision counts

## Usage Examples

**IMPORTANT:** All examples use `omni_python` instead of `python` for execution.

### Basic Trajectory Check

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --glass_position 0.7 0.0 0.6 \
    --table_position 0.7 0.0 0.0 \
    --table_dimensions 0.6 1.0 1.1

# IMPORTANT: Use glass_zup.obj (Z-up) to match Pinocchio/URDF coordinate system!
# Using glass_yup.obj will result in incorrect collision detection.
```

### Using Link Meshes (Most Accurate)

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf ur_description/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --use_link_meshes \
    --mesh_base_path ur_description
```

### Using Collision Spheres from YAML

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --robot_config ur20.yml
```

### Using Capsule Approximations

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --use_capsules \
    --capsule_radius 0.05
```

### With Collision Margin

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --collision_margin -0.05  # 5cm tolerance (less conservative)
```

### With Interpolation Control

```bash
# High-resolution interpolation (60 steps)
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --interp-steps 60

# Disable interpolation for faster checking
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --no-interpolate
```

### Link-Level Analysis

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/joint_trajectory_dp_5000_base.csv \
    --robot_urdf data/input/ur20.urdf \
    --mesh data/input/object/glass_zup.obj \
    --show_link_collisions \
    --verbose
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

### COAL Performance Benefits

**Speed Improvements:**
- 5-15x faster than original FCL
- Improved GJK/EPA algorithm implementation
- Accelerated collision detection with Nesterov-style optimization
- Distance lower bounds computed during collision checks

**Numerical Stability:**
- Better handling of edge cases
- Improved convergence in GJK/EPA
- More reliable penetration depth computation

### Accuracy vs Speed Trade-off

1. **Link Meshes** (Highest Accuracy, Still Fast with COAL)
   - Use for final validation
   - Most accurate collision detection
   - Recommended for critical applications
   - COAL makes this practical even for large trajectories

2. **Collision Spheres** (Balanced)
   - Good accuracy with excellent speed
   - Use for iterative development
   - Matches CuRobo collision model
   - Best balance for most use cases

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
- Adjust `num_interp_steps` based on trajectory smoothness:
  - Smooth trajectories: 10-30 steps
  - Complex trajectories: 30-60 steps
- Use `--no-interpolate` for quick validation of waypoints only

## Debugging Workflow

### Step 1: Run Full Trajectory Check
```bash
omni_python scripts/coal_check.py --trajectory <path> --use_link_meshes
```

### Step 2: If Collisions Found, Enable Link Analysis
```bash
omni_python scripts/coal_check.py \
    --trajectory <path> \
    --show_link_collisions \
    --verbose
```
This identifies which links are problematic.

### Step 3: Adjust Interpolation if Needed
```bash
# Increase interpolation resolution
omni_python scripts/coal_check.py \
    --trajectory <path> \
    --interp-steps 60

# Or disable to check waypoints only
omni_python scripts/coal_check.py \
    --trajectory <path> \
    --no-interpolate
```

### Step 4: Adjust Parameters
Based on findings:
- Adjust `glass_position` if position mismatch
- Adjust `table_position` and `table_dimensions` for table
- Modify `collision_margin` to tune sensitivity
- Update trajectory generation parameters
- Check mesh scales and transforms

## Common Issues and Solutions

### Issue: High Collision Rate

**Possible Causes:**
- Glass/table position doesn't match simulation
- Target points too close to obstacles
- Collision margin too conservative
- Interpolation detecting in-between collisions

**Solutions:**
- Verify glass_position and table_position match simulation world
- Increase NORMAL_SAMPLE_OFFSET in trajectory generation
- Use negative collision_margin for tolerance
- Check if collisions occur at waypoints or segments (interpolation)

### Issue: Specific Link Always Collides

**Possible Causes:**
- Link mesh/sphere/capsule too large
- Robot configuration brings link too close
- Trajectory planning issue

**Solutions:**
- Adjust collision geometry size
- Review IK solutions and joint limits
- Regenerate trajectory with different parameters
- Use `--show_link_collisions` to identify problematic links

### Issue: Segment Collisions but Waypoints are Clear

**Possible Causes:**
- Trajectory has sharp transitions between waypoints
- Linear interpolation doesn't match actual robot motion
- Joint space interpolation creates unexpected configurations

**Solutions:**
- Increase number of waypoints in trajectory generation
- Smooth trajectory before checking
- Adjust `num_interp_steps` to match motion planning resolution
- Consider using motion primitive interpolation instead of linear

### Issue: Inconsistent Results with FCL checker

**Possible Causes:**
- Different collision models
- Transform API differences
- Numerical precision differences

**Solutions:**
- Verify both use same collision geometry mode
- Check coordinate transforms
- COAL may be more accurate due to improved GJK/EPA

## Integration with Vision Inspection System

The COAL checker complements the main vision inspection system:

1. **CuRobo** (in run_app_v3.py):
   - Generates trajectories
   - Uses GPU-accelerated collision checking
   - Integrated with motion planning

2. **COAL Checker** (coal_check.py):
   - Independent validation
   - CPU-based checking (but 5-15x faster than FCL)
   - Detailed per-link analysis
   - Debugging and verification
   - Interpolation support for comprehensive checking

**Recommended Workflow:**
1. Generate trajectories using CuRobo
2. Validate with COAL checker using same collision model
3. Use `--show_link_collisions` to debug any discrepancies
4. Adjust parameters and regenerate if needed
5. Use interpolation to verify motion between waypoints

## Comparison: COAL vs FCL

| Feature | FCL (fcl_check.py) | COAL (coal_check.py) |
|---------|-------------------|----------------------|
| **Performance** | Baseline | 5-15x faster |
| **Transform API** | `fcl.Transform(R, t)` | `coal.Transform3s()` + setters |
| **Security Margin** | Basic support | Native, improved support |
| **Signed Distance** | No | Yes (`enable_signed_distance`) |
| **GJK/EPA** | Basic | Improved, more stable |
| **Pinocchio Integration** | Separate | Native |
| **Interpolation** | Yes | Yes |
| **Table Support** | Yes | Yes |
| **Execution** | `python` | `omni_python` |

**Migration from FCL to COAL:**
- Main change: Import and transform creation
- All collision checking logic remains similar
- Performance improvement without major code changes
- Same accuracy with better numerical stability

## Dependencies

```python
import coal                  # COAL: Collision detection (instead of fcl)
import pinocchio as pin      # Forward kinematics
import trimesh               # Mesh loading and processing
import numpy as np           # Numerical operations
import yaml                  # Config file parsing
```

**Installation:**

COAL is automatically installed with Pinocchio:
```bash
# Install Pinocchio (includes COAL)
pip install pin  # or pinocchio
conda install pinocchio -c conda-forge

# COAL is included as a dependency
# Can also install separately:
pip install coal
conda install coal -c conda-forge
```

**Backward Compatibility:**
```python
# Both work for backward compatibility:
import coal
import hppfcl  # Old name, still supported
```

## API Reference

### COALCollisionChecker

**Methods:**

#### `__init__(...)`
Initialize collision checker with robot and obstacles.

#### `_create_transform(rotation=None, translation=None)`
Helper method to create COAL Transform3s objects.

**Returns:**
- `coal.Transform3s`: Transform object

#### `check_collision_single_config(joint_positions, return_distance=False, return_link_info=False)`
Check collision for one robot configuration.

**Returns:**
- `is_collision` (bool): Whether collision detected
- `distance` (float): Minimum distance to obstacles
- `link_info` (list): Per-link collision details (if requested)

#### `check_trajectory(trajectory, verbose=True, show_link_collisions=False, max_show=10, interpolate=True, num_interp_steps=30)`
Check entire trajectory for collisions with interpolation support.

**Returns:**
- Dictionary with collision statistics (see Output Statistics section)

### Utility Functions

#### `load_trajectory_csv(csv_path)`
Load joint trajectory from CSV file.

**Returns:**
- `trajectory` (np.ndarray): (N, 6) array of joint angles
- `joint_names` (list): Joint column names

#### `generate_interpolated_path(start, end, num_steps)`
Generate linear interpolation between two joint configurations.

**Returns:**
- List of interpolated joint configurations (excludes start/end)

## Output Interpretation

### Console Output Example

```
======================================================================
COAL Collision Checker for Robot Trajectories
======================================================================
NOTE: Using COAL library (improved FCL) for better performance
Execute with: omni_python coal_check.py [options]
======================================================================

1. Loading trajectory from: data/trajectory/joint_trajectory_dp_5000_base.csv
Loaded trajectory: 5000 waypoints, 6 joints

2. Initializing COAL collision checker
   Robot URDF: data/input/ur20.urdf
   Robot config: ur20.yml
   Obstacle meshes: ['data/input/object/glass_zup.obj']
   Glass position: [0.7, 0.0, 0.6]
   Table position: [0.7, 0.0, 0.0]
   Table dimensions: [0.6, 1.0, 1.1]
   Use link meshes: True

Loading obstacle meshes...
  Loaded: data/input/glass_o3d.obj (1234 vertices, 2468 faces)

Adding table cuboid...
  Position: [0.7 0.0 0.0]
  Dimensions (x, y, z): [0.6 1.0 1.1]
  Table cuboid added successfully

Loading robot model with Pinocchio...
  Robot loaded: 6 DOF, 7 joints
  Using actual collision meshes from URDF
  Loaded 7 link meshes

3. Running collision detection with COAL
   Interpolation enabled: 30 steps between waypoints

Checking 5000 waypoints with interpolation (30 steps between waypoints)...
Total configurations to check: 154,970
  Progress: 500/154970 configurations checked
  Progress: 1000/154970 configurations checked
  ...

======================================================================
COLLISION CHECK RESULTS
======================================================================
Total waypoints:        5000
Total configurations:   154,970 (with 30 interpolation steps)
Collisions at waypoints:    477
Collisions in segments:     123
Total collisions:           600
Collision-free configs:     154,370
Collision rate:         0.39%

Collision waypoint indices (first 50):
  [12, 45, 67, 89, ...]

Collision segments (first 50):
  Waypoint 15→16 (α=0.33)
  Waypoint 23→24 (α=0.67)
  ...

Collisions by link:
  wrist_3_link: 245 (40.8% of collisions)
  forearm_link: 178 (29.7% of collisions)
  wrist_2_link: 177 (29.5% of collisions)
======================================================================

NOTE:
- Using COAL (Collision and Occupancy Algorithms Library)
  COAL is 5-15x faster than FCL with improved numerical stability
- Using Pinocchio for Forward Kinematics
- Interpolation ENABLED: Checking 30 intermediate configurations between each waypoint pair
  This detects collisions that occur during motion between waypoints
- Robot collision geometry:
  Using actual mesh geometries from URDF (most accurate)
- To improve accuracy:
  1. Use --use_link_meshes for most accurate collision checking
  2. Adjust --interp-steps to control interpolation density (default: 30)
  3. Use --collision_margin to add safety margins (e.g., 0.01 for 1cm)
======================================================================
```

## Best Practices

1. **Always use COAL for new development** (5-15x faster than FCL)
2. **Use link meshes for final validation** when accuracy is critical
3. **Match collision model with motion planner** for consistency
4. **Enable interpolation** to catch in-between collisions (default: on)
5. **Adjust interpolation resolution** based on trajectory smoothness
6. **Use `--show_link_collisions`** to identify problematic links
7. **Verify coordinate frames** before large-scale checks
8. **Document collision margin settings** used
9. **Save collision analysis results** for comparison
10. **Cross-validate with simulation** visualization
11. **Execute with `omni_python`** for proper environment

## Future Enhancements

Potential improvements:
- Continuous collision detection (swept volumes)
- Parallel trajectory checking (multi-threading)
- Export collision visualization (meshes with highlights)
- Integration with trajectory optimization
- Real-time collision monitoring mode
- Support for multiple robot configurations
- Self-collision detection
- Dynamic obstacle support

## References

- **COAL Documentation**: https://github.com/coal-library/coal
- **COAL (formerly HPP-FCL)**: Renamed in 2024, v3.0.0+
- **Pinocchio Documentation**: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/
- **CuRobo**: https://curobo.org/
- **Isaac Sim**: https://developer.nvidia.com/isaac-sim
- **Original FCL**: https://github.com/flexible-collision-library/fcl

---

**File Location:** `/isaac-sim/curobo/vision_inspection/scripts/`

**Related Files:**
- `coal_check.py` - Main COAL collision checker (scripts/coal_check.py:1)
- `fcl_check.py` - Legacy FCL collision checker (scripts/fcl_check.py:1)
- `run_app_v3.py` - Vision inspection main application

**Last Updated:** 2025-11-08

**Note:** This is the recommended collision checker for new projects. For legacy compatibility, `fcl_check.py` remains available.
