# Vision Inspection System Analysis

## Overview
This code implements a **robotic vision inspection system** using Isaac Sim (NVIDIA's robot simulation platform) and CuRobo (NVIDIA's CUDA-accelerated robotics library). The system automates the inspection of glass objects by computing optimal viewpoints from a 3D model and generating collision-free robot trajectories.

## Core Functionality

### 1. Point Cloud Processing
- **Input**: 3D mesh file (`glass_o3d.obj`) or pre-computed point cloud (`glass_pointcloud.pcd`)
- **Sampling**: Uses Poisson disk sampling to generate evenly distributed viewpoints
- **Normal Estimation**: Computes surface normals for each sampled point
- **Coordinate Transformation**: Converts between Open3D (Y-up) and Isaac Sim (Z-up) coordinate systems

### 2. Viewpoint Generation
The system generates inspection viewpoints by:
- Sampling points on the glass surface using Poisson disk sampling
- Offsetting each point along its surface normal by `NORMAL_SAMPLE_OFFSET` (0.1m)
- Computing a local coordinate frame aligned with the surface normal
- Creating a `Viewpoint` object storing:
  - Local pose (in object coordinates)
  - World pose (in simulation world coordinates)
  - IK solutions (all analytical solutions)
  - Safe IK solutions (collision-free solutions)

### 3. Inverse Kinematics (IK)
Two IK solvers are used:
- **EAIK** (Efficient Analytical IK): Fast analytical IK solver
  - Loads URDF robot model
  - Computes all possible joint configurations for each viewpoint
  - Applies coordinate transformation between CuRobo and EAIK conventions
- **CuRobo IK Solver**: GPU-accelerated numerical IK
  - Used for validation and collision checking
  - Configured with mesh-based collision detection

### 4. Collision Checking
- Uses CuRobo's collision checker with mesh-based collision detection
- World configuration includes:
  - Table cuboid obstacle
  - Ground plane
  - Glass object mesh
  - Robot self-collision checking (disabled in config)
- Filters IK solutions to retain only collision-free configurations
- Updates `safe_ik_solutions` for each viewpoint

### 5. Trajectory Planning
- **Path Sorting**: Sorts safe IK solutions by joint space norm (minimum movement heuristic)
- **Interpolation**: Generates smooth trajectories using linear interpolation with 60 steps (`INTERPOLATION_STEPS`)
- **Queue System**: Uses a deque to queue up inspection targets
- **Sequential Execution**: Moves through viewpoints one at a time

### 6. Simulation Loop
The main loop:
1. Waits for user to click "Play" in Isaac Sim
2. Executes interpolated trajectories step-by-step
3. Moves robot through collision-free inspection poses
4. Visualizes sampled points and normals using debug draw

## Key Classes and Data Structures

### Viewpoint
```python
@dataclass
class Viewpoint:
    index: int                              # Point cloud index
    local_pose: Optional[np.ndarray]        # 4x4 pose in object frame
    world_pose: Optional[np.ndarray]        # 4x4 pose in world frame
    all_ik_solutions: List[np.ndarray]      # All analytical IK solutions
    safe_ik_solutions: List[np.ndarray]     # Collision-free IK solutions
```

### Global State
- `SAMPLED_LOCAL_POINTS`: Sampled 3D points in object coordinates
- `SAMPLED_LOCAL_NORMALS`: Surface normals for sampled points
- `SAMPLED_VIEWPOINTS`: List of Viewpoint objects

## Coordinate System Transformations

### 1. Open3D → Isaac Sim
Rotation matrix that converts Y-up to Z-up:
```
[1,  0,  0]
[0,  0, -1]
[0,  1,  0]
```

### 2. CuRobo → EAIK Tool Frame
4x4 transformation matrix applied to align tool conventions:
```
[-1,  0,  0,  0]
[ 0,  0,  1,  0]
[ 0,  1,  0,  0]
[ 0,  0,  0,  1]
```

### 3. Local → World Pose
Applies: Scale → Rotation → Translation

## Command Line Arguments

- `--headless_mode`: Run without GUI (options: native, websocket)
- `--visualize_spheres`: Visualize robot collision spheres
- `--robot`: Robot configuration YAML (default: ur20.yml)
- `--number_of_points`: Number of sampled viewpoints (default: 10000)

## Pipeline Flow

1. **Load/Generate Point Cloud**
   - Check for existing PCD file
   - If not found, generate from mesh using Poisson sampling
   - Estimate normals and save PCD

2. **Initialize Simulation**
   - Create Isaac Sim world
   - Load robot (UR20 by default)
   - Add glass object with visual material
   - Setup collision environment

3. **Compute Viewpoints**
   - Sample points on glass surface
   - Offset along normals to create camera viewpoints
   - Convert to world coordinates

4. **Solve IK**
   - Use EAIK to compute all analytical solutions
   - Check each solution for collisions using CuRobo
   - Filter to safe solutions only

5. **Execute Inspection**
   - Sort viewpoints by joint space distance
   - Generate interpolated trajectories
   - Execute in simulation loop

## Performance Metrics

The code measures:
- IK solution assignment time
- Collision checking time
- Statistics: solved viewpoints vs. collision-free viewpoints

## Dependencies

- **Isaac Sim**: NVIDIA's robot simulation platform
- **CuRobo**: CUDA-accelerated motion planning
- **Open3D**: Point cloud processing
- **EAIK**: Analytical IK solver
- **PyTorch**: Tensor operations and GPU acceleration
- **NumPy**: Numerical computations

## File Paths

- Robot URDF: `/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf`
- Glass mesh: `/isaac-sim/curobo/vision_inspection/data/input/glass_o3d.obj`
- Glass USD: `/isaac-sim/curobo/vision_inspection/data/input/glass_isaac_ori.usdc`
- Point cloud: `/isaac-sim/curobo/vision_inspection/data/input/glass_pointcloud.pcd`
- Robot config: `robot_cfg` directory (ur20.yml)
- World config: `collision_table.yml`

## Notes

- The system is designed for glass inspection, with material properties set for realistic rendering
- Collision checking can be bypassed by uncommenting `return True` in `collision_checking()`
- The code includes extensive coordinate transformation utilities for proper alignment
- Debug visualization shows sampled points (red) and normal vectors
