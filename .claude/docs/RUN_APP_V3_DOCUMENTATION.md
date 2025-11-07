# run_app_v3.py - Vision Inspection Robot Trajectory Planner

## 📋 Overview

`run_app_v3.py` is a refactored version of `run_app_v2.py` that implements a robotic vision inspection system using Isaac Sim and CuRobo. This version improves code maintainability, testability, and readability by:

- **Eliminating global variables** - All state is encapsulated in classes
- **Breaking down large functions** - The 244-line `main()` is now 30 lines
- **Removing unused code** - ~200-300 lines of dead code removed
- **Making configuration explicit** - All magic numbers are now configurable parameters
- **Improving separation of concerns** - Clear boundaries between I/O, computation, and simulation

---

## 🏗️ Architecture

### High-Level Flow

```
main()
  ├─ 1. Load configuration from command line
  ├─ 2. Load TSP tour from HDF5 file
  ├─ 3. initialize_simulation() → WorldState
  │     ├─ Create Isaac Sim world
  │     ├─ Setup robot (UR20)
  │     ├─ Setup glass object
  │     ├─ Setup camera
  │     └─ Setup collision checker & IK solver
  ├─ 4. process_viewpoints() → ViewpointManager
  │     ├─ Create viewpoints from TSP tour
  │     ├─ Update world poses
  │     ├─ Compute EAIK IK solutions
  │     └─ Check collision constraints
  ├─ 5. plan_trajectory() → (targets, indices)
  │     ├─ Filter viewpoints with safe IK
  │     └─ Select method: random/greedy/dp
  ├─ 6. save_results()
  │     ├─ Save CSV trajectory
  │     ├─ Analyze reconfigurations
  │     └─ Save HDF5 with all IK solutions
  └─ 7. run_simulation() (optional)
        └─ Execute trajectory in Isaac Sim
```

---

## 🎯 Key Classes

### 1. SimulationConfig (dataclass)

**Purpose**: Central configuration for all simulation parameters

**Key Attributes**:
```python
# Command line arguments
robot_config_file: str          # Robot YAML config (default: ur20.yml)
tsp_tour_path: str              # Path to TSP HDF5 file
selection_method: str           # IK selection: random/greedy/dp
no_sim: bool                    # Skip simulation if True

# Robot parameters
normal_sample_offset: float     # Working distance (default: 0.11m = 110mm)
interpolation_steps: int        # Trajectory interpolation (default: 60)

# DP cost function parameters
joint_weights: np.ndarray       # Joint importance [2,2,2,1,1,1]
reconfig_threshold: float       # Reconfiguration threshold (1.0 rad)
reconfig_penalty: float         # Penalty per reconfigured joint (10.0)
max_move_weight: float          # Max movement penalty weight (5.0)

# Visualization
plot_interval: int              # Plot update frequency (default: 5)

# World configuration
table_position: np.ndarray      # Table pose [0.7, 0.0, 0.0]
table_dimensions: np.ndarray    # Table size [0.6, 1.0, 1.1]
glass_position: np.ndarray      # Glass pose [0.7, 0.0, 0.6]

# IK solver configuration
ik_rotation_threshold: float    # Rotation tolerance (0.05)
ik_position_threshold: float    # Position tolerance (0.005)
ik_num_seeds: int               # IK solver seeds (20)
```

**Factory Method**:
```python
config = SimulationConfig.from_args(args)
```

---

### 2. WorldState (dataclass)

**Purpose**: Encapsulates Isaac Sim world state

**Attributes**:
```python
world: World                    # Isaac Sim world
glass_prim: XFormPrim          # Glass object reference
robot: Any                      # Robot articulation
idx_list: List[int]            # Active joint indices
ik_solver: IKSolver            # CuRobo IK solver with collision checker
default_config: np.ndarray     # Retract joint configuration
```

**Creation**:
```python
world_state = initialize_simulation(config)
```

---

### 3. Viewpoint (dataclass)

**Purpose**: Represents a single camera viewpoint with IK solutions

**Attributes**:
```python
index: int                          # Viewpoint index in TSP order
local_pose: np.ndarray              # 4x4 pose in object frame (Open3D coords)
world_pose: np.ndarray              # 4x4 pose in world frame (Isaac coords)
all_ik_solutions: List[np.ndarray]  # All analytical IK solutions
safe_ik_solutions: List[np.ndarray] # Collision-free IK solutions
```

---

### 4. ViewpointManager (class)

**Purpose**: Manages viewpoint data and operations

**Attributes**:
```python
viewpoints: List[Viewpoint]     # All viewpoints in TSP order
local_points: np.ndarray        # Camera positions (Open3D coords)
local_normals: np.ndarray       # Camera directions (Open3D coords)
```

**Key Methods**:
```python
# Filter viewpoints
filter_with_safe_ik() -> List[Viewpoint]
    """Return only viewpoints with collision-free IK solutions"""

# Statistics
count_with_all_ik() -> int
count_with_safe_ik() -> int

# Coordinate transformations
update_world_poses(reference_prim: XFormPrim)
    """Transform local poses to world poses using glass object transform"""

collect_world_matrices() -> Tuple[np.ndarray, List[int]]
    """Collect 4x4 pose matrices for IK computation"""
```

**Creation**:
```python
viewpoint_mgr = create_viewpoints_from_tsp(tsp_result, config)
```

---

### 5. JointHistoryTracker (class)

**Purpose**: Tracks joint trajectory during simulation

**Attributes**:
```python
timestamps: List[int]              # Simulation timesteps
joint_values: List[np.ndarray]     # Joint values at each timestep
viewpoint_markers: List[int]       # Timesteps when viewpoints reached
```

**Methods**:
```python
record_step(timestamp: int, joints: np.ndarray)
    """Record joint configuration at timestep"""

mark_viewpoint(timestamp: int)
    """Mark that a viewpoint was reached"""

has_data() -> bool
get_joint_array() -> np.ndarray
    """Get all joint values as (N, 6) array"""
```

---

## 🔧 Core Functions

### Initialization

#### `initialize_simulation(config: SimulationConfig) -> WorldState`

Creates Isaac Sim world with all components:
1. `create_world()` - Isaac Sim world setup
2. `setup_robot()` - Load UR20 robot
3. `setup_glass_object()` - Add glass object with material
4. `setup_camera()` - Mount camera on end-effector
5. `setup_collision_checker()` - Create IK solver with collision checking

**Returns**: `WorldState` with all initialized components

---

### Viewpoint Processing

#### `create_viewpoints_from_tsp(tsp_result: dict, config: SimulationConfig) -> ViewpointManager`

Creates viewpoints from TSP tour data:

**Steps**:
1. Extract tour coordinates and normals (in TSP visit order)
2. Determine working distance from `camera_spec` or config default
3. **Important**: TSP file stores **surface positions**, not camera positions
4. Offset surface positions along normals to get camera positions
5. Create local pose matrices (position + orientation frame)
6. Return `ViewpointManager` with viewpoints in TSP order

**Coordinate Systems**:
- Input: Open3D coordinates (Y-up)
- Surface positions: Points on glass surface
- Camera positions: Surface + normal × working_distance

**Example**:
```python
# TSP stores surface positions
surface_pos = [0.1, 0.2, 0.3]
surface_normal = [0, 0, 1]  # Pointing outward
working_distance = 0.11  # 110mm

# Compute camera position
camera_pos = surface_pos + surface_normal * working_distance
# camera_pos = [0.1, 0.2, 0.41]

# Camera direction (toward surface)
camera_dir = -surface_normal  # [-0, -0, -1]
```

---

#### `process_viewpoints(config: SimulationConfig, world_state: WorldState) -> ViewpointManager`

Complete viewpoint processing pipeline:

**Steps**:
1. Load TSP tour from HDF5
2. Create viewpoints from TSP data
3. Transform local poses to world poses
4. Compute analytical IK solutions (EAIK)
5. Check collision constraints (CuRobo)
6. Update viewpoints with safe solutions

**Performance Metrics**:
- IK computation time
- Collision checking time
- Statistics: total / with IK / with safe IK

---

### Trajectory Planning

#### `plan_trajectory(viewpoint_mgr, config, world_state) -> Tuple[List[np.ndarray], List[int]]`

Plans joint trajectory using selected method:

**Methods**:

1. **Random/First** (`select_ik_random`):
   - Selects first safe IK solution for each viewpoint
   - Fastest but may result in large joint movements

2. **Greedy** (`select_ik_greedy`):
   - Selects IK solution closest to previous configuration
   - Good balance of speed and smoothness

3. **Dynamic Programming** (`select_ik_dp`):
   - Globally optimal solution minimizing total cost
   - Cost function: weighted joint distance
   - Slowest but smoothest trajectory

**Returns**:
- `joint_targets`: List of selected joint configurations
- `solution_indices`: Which IK solution was selected for each viewpoint

---

### File I/O

#### `save_results(trajectory, viewpoint_mgr, config, tsp_result)`

Saves all analysis results:

**Outputs**:
1. **CSV Trajectory** (`joint_trajectory_{method}_{timestamp}.csv`):
   - Columns: time, 6 joints, target pose (position + quaternion)
   - Compatible with reconfiguration analysis tools

2. **Reconfiguration Analysis** (`joint_trajectory_{method}_{timestamp}_reconfig.txt`):
   - Total reconfigurations
   - Per-joint statistics
   - Detailed timestep-by-timestep analysis

3. **HDF5 IK Solutions** (`ik_solutions_all_{timestamp}.h5`):
   - All IK solutions for each viewpoint
   - Collision-free mask
   - Selected solution index
   - Metadata (selection method, timestamp)

**Output Directory**: `data/output/{num_points}/`

---

### Simulation

#### `run_simulation(world_state, trajectory, viewpoint_mgr, config, tsp_result)`

Executes trajectory in Isaac Sim:

**Loop Structure**:
```python
while simulation_app.is_running():
    # Wait for user to click Play
    if not world.is_playing():
        continue

    # Record current state
    history.record_step(step_counter, current_joints)

    # Execute interpolated trajectory
    if active_trajectory:
        robot.set_joint_positions(trajectory[step])
        step++

    # Load next waypoint
    elif target_queue:
        active_trajectory = interpolate(current, next_target)
        history.mark_viewpoint(step_counter)
```

**Features**:
- Linear interpolation between waypoints
- Joint history tracking
- Viewpoint markers for analysis

---

## 📊 Coordinate Systems

### Open3D vs Isaac Sim

**Open3D** (Y-up):
```
Y
↑
|    Z (forward)
|   ↗
|  /
| /
└────→ X
```

**Isaac Sim** (Z-up):
```
Z
↑
|    Y (forward)
|   ↗
|  /
| /
└────→ X
```

**Transformation Matrix** (`OPEN3D_TO_ISAAC_ROT`):
```python
[1,  0,  0]
[0,  0, -1]
[0,  1,  0]
```

**Usage**:
```python
# Convert points
isaac_points = (OPEN3D_TO_ISAAC_ROT @ open3d_points.T).T

# Convert normals
isaac_normals = (OPEN3D_TO_ISAAC_ROT @ open3d_normals.T).T
```

---

### Local vs World Coordinates

**Local (Object) Frame**:
- Relative to glass object origin
- Open3D coordinate system (Y-up)
- Stored in TSP HDF5 file

**World Frame**:
- Isaac Sim global coordinates
- Z-up coordinate system
- Includes glass object's world transform

**Transformation Pipeline**:
```
Local Pose (Open3D)
  ↓ [OPEN3D_TO_ISAAC_ROT]
Isaac Local Pose
  ↓ [Scale]
Scaled Pose
  ↓ [Glass World Rotation]
Rotated Pose
  ↓ [Glass World Translation]
World Pose (Isaac Sim)
```

**Implementation**:
```python
def open3d_pose_to_world(pose_matrix, reference_prim):
    # 1. Convert coordinate system
    local_rot = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, :3]
    local_pos = OPEN3D_TO_ISAAC_ROT @ pose_matrix[:3, 3]

    # 2. Apply glass object's world transform
    scaled_pos = local_pos * glass_scale
    rotated_pos = glass_rotation @ scaled_pos
    world_pos = rotated_pos + glass_position

    # 3. Combine into 4x4 matrix
    world_pose = np.eye(4)
    world_pose[:3, :3] = glass_rotation @ local_rot
    world_pose[:3, 3] = world_pos
    return world_pose
```

---

## 🔄 IK Solution Pipeline

### 1. Analytical IK (EAIK)

**Purpose**: Fast analytical IK solver for UR robots

**Input**:
- `world_matrices`: (N, 4, 4) world poses
- `urdf_path`: Robot URDF file

**Process**:
1. Transform from CuRobo to EAIK tool frame
2. Compute all analytical solutions (up to 8 per pose)
3. Return solution objects with Q arrays

**Coordinate Transform** (`CUROBO_TO_EAIK_TOOL`):
```python
[-1,  0,  0,  0]
[ 0,  0,  1,  0]
[ 0,  1,  0,  0]
[ 0,  0,  0,  1]
```

**Function**:
```python
ik_results = compute_ik_eaik(world_matrices)
assign_ik_solutions_to_viewpoints(viewpoints, ik_results, indices)
```

---

### 2. Collision Checking (CuRobo)

**Purpose**: Filter IK solutions for collision-free configurations

**Process**:
1. Batch all IK solutions from all viewpoints
2. Create `JointState` tensor
3. Check constraints using CuRobo collision checker
4. Update `safe_ik_solutions` for each viewpoint

**Collision Objects**:
- Table cuboid
- Ground plane
- Glass mesh
- Robot self-collision (disabled in config)

**Function**:
```python
check_ik_solutions_collision(viewpoints, ik_solver)
```

**Result**:
- Each viewpoint has `all_ik_solutions` and `safe_ik_solutions`
- Only safe solutions are used for trajectory planning

---

## 📈 Cost Function (DP Method)

### Weighted Joint Distance

**Purpose**: Penalize movement in important joints more

**Formula**:
```python
weights = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]  # Base joints weighted higher
distance = sqrt(sum(weights[i] * (q1[i] - q2[i])^2))
```

**Rationale**:
- Shoulder joints (1-3): Heavy, slow, high inertia → weight = 2.0
- Wrist joints (4-6): Light, fast → weight = 1.0
- Joint 6 (tool rotation): Often ignored → weight = 0.0

---

### Reconfiguration Cost (Future Work)

**Components**:
```python
base_cost = weighted_joint_distance(q1, q2, weights)

reconfig_count = count(|q1[i] - q2[i]| > threshold)
reconfig_cost = reconfig_count * penalty

max_move = max(|q1[i] - q2[i]|)
max_move_cost = max_move * weight

total_cost = base_cost + reconfig_cost + max_move_cost
```

**Note**: Currently only `base_cost` is used (line 1032)

---

### Dynamic Programming Algorithm

**States**: DP table `dp[viewpoint_idx][solution_idx]`

**State Value**: `(min_cost_from_start, prev_solution_idx)`

**Initialization**:
```python
for sol_idx in first_viewpoint.safe_ik_solutions:
    cost = compute_cost(sol, initial_config)
    dp[0][sol_idx] = (cost, -1)
```

**Forward Pass**:
```python
for i in range(1, n_viewpoints):
    for curr_sol in viewpoint[i].safe_ik_solutions:
        min_cost = infinity
        for prev_sol_idx, (prev_cost, _) in dp[i-1].items():
            prev_sol = viewpoint[i-1].safe_ik_solutions[prev_sol_idx]
            transition_cost = compute_cost(curr_sol, prev_sol)
            total = prev_cost + transition_cost
            if total < min_cost:
                min_cost = total
                best_prev = prev_sol_idx
        dp[i][curr_sol_idx] = (min_cost, best_prev)
```

**Backward Pass**:
```python
best_final = argmin(dp[n-1][sol_idx][0])
path = reconstruct_path(dp, best_final)
```

**Complexity**:
- Time: O(n × k²) where n = viewpoints, k = avg solutions per viewpoint
- Space: O(n × k)

---

## 🗂️ File Formats

### Input: TSP Tour HDF5

**Structure**:
```
file.h5
├── metadata/
│   ├── num_points (int)
│   ├── mesh_file (str)
│   ├── timestamp (str)
│   ├── nn_cost (float)
│   ├── glop_cost (float)
│   ├── improvement (float)
│   └── camera_spec/           # Optional
│       ├── working_distance_mm (float)
│       ├── fov_width_mm (float)
│       └── fov_height_mm (float)
├── points/
│   ├── original (N, 3)        # Surface positions
│   ├── normalized (N, 3)
│   └── normalization_info/
│       ├── min (3,)
│       └── max (3,)
├── normals (N, 3)              # Surface normals
└── tour/
    ├── indices (N,)            # TSP visit order
    └── coordinates (N, 3)      # Points in TSP order
```

**Important**:
- `points/original` and `tour/coordinates` store **surface positions**, not camera positions
- `normals` stores **surface normals** (outward from glass), not camera directions

---

### Output: Joint Trajectory CSV

**Format**:
```csv
time,ur20-shoulder_pan_joint,ur20-shoulder_lift_joint,ur20-elbow_joint,ur20-wrist_1_joint,ur20-wrist_2_joint,ur20-wrist_3_joint,target-POS_X,target-POS_Y,target-POS_Z,target-ROT_X,target-ROT_Y,target-ROT_Z,target-ROT_W
0.0,1.234,-0.567,2.345,-1.234,0.456,0.789,0.7,0.0,0.6,0.0,0.0,0.0,1.0
1.0,1.245,-0.578,2.356,-1.245,0.467,0.790,0.71,0.01,0.61,0.0,0.0,0.0,1.0
...
```

**Columns**:
- `time`: Viewpoint index (0, 1, 2, ...)
- 6 joint columns: Joint angles in radians
- `target-POS_*`: Target end-effector position (meters)
- `target-ROT_*`: Target end-effector orientation (quaternion x, y, z, w)

---

### Output: IK Solutions HDF5

**Structure**:
```
ik_solutions.h5
├── metadata/
│   ├── num_viewpoints (int)
│   ├── num_viewpoints_with_solutions (int)
│   ├── num_viewpoints_with_safe_solutions (int)
│   ├── selection_method (str)
│   ├── timestamp (str)
│   └── tsp_tour_file (str)
└── viewpoint_0000/
    ├── @original_index (int)
    ├── @num_all_solutions (int)
    ├── @num_safe_solutions (int)
    ├── @selected_solution_index (int)
    ├── world_pose (4, 4)
    ├── all_ik_solutions (M, 6)
    └── collision_free_mask (M,) bool
```

**Usage**:
- Analyze IK solver performance
- Visualize solution distribution
- Debug collision detection issues
- Replay different solution selections

---

## 🚀 Usage Examples

### Basic Usage

```bash
# Run with DP method (recommended)
python scripts/run_app_v3.py \
    --tsp_tour_path data/input/viewpoint/glass_fov_500_tsp.h5 \
    --selection_method dp

# Run with greedy method
python scripts/run_app_v3.py \
    --tsp_tour_path data/input/viewpoint/glass_fov_500_tsp.h5 \
    --selection_method greedy

# Generate outputs without simulation
python scripts/run_app_v3.py \
    --tsp_tour_path data/input/viewpoint/glass_fov_500_tsp.h5 \
    --selection_method dp \
    --no_sim

# Run headless (for remote servers)
python scripts/run_app_v3.py \
    --tsp_tour_path data/input/viewpoint/glass_fov_500_tsp.h5 \
    --selection_method dp \
    --headless_mode native
```

---

### Advanced Configuration

To modify configuration parameters, edit `SimulationConfig` defaults in the code:

```python
@dataclass
class SimulationConfig:
    # Change working distance
    normal_sample_offset: float = 0.12  # 120mm instead of 110mm

    # More interpolation steps for smoother motion
    interpolation_steps: int = 100  # Instead of 60

    # Adjust DP cost function
    joint_weights: np.ndarray = field(
        default_factory=lambda: np.array([3.0, 3.0, 2.0, 1.0, 1.0, 0.5])
    )
    reconfig_threshold: float = 0.5  # More sensitive to reconfigurations
```

---

## 🐛 Troubleshooting

### Issue: "No viewpoints with safe IK solutions"

**Cause**: All IK solutions are in collision

**Solutions**:
1. Check glass object position and scale
2. Verify working distance is not too close/far
3. Adjust table position/dimensions in config
4. Check collision world configuration

**Debug**:
```python
# In process_viewpoints():
print(f"Viewpoints with any IK: {viewpoint_mgr.count_with_all_ik()}")
print(f"Viewpoints with safe IK: {viewpoint_mgr.count_with_safe_ik()}")

# If count_with_all_ik == 0: EAIK failed
# If count_with_safe_ik == 0: All solutions in collision
```

---

### Issue: "Points appear too far from glass"

**Cause**: Working distance mismatch or unit inconsistency

**Solutions**:
1. Verify `camera_spec.working_distance_mm` in HDF5
2. Check mesh units (should be meters)
3. Verify `normal_sample_offset` value

**Debug**:
```python
# In create_viewpoints_from_tsp():
print(f"Working distance: {working_distance_m*1000:.1f} mm")
print(f"Surface position range: {tour_coords.min(axis=0)} to {tour_coords.max(axis=0)}")
print(f"Camera position range: {offset_points.min(axis=0)} to {offset_points.max(axis=0)}")
```

---

### Issue: KeyError on HDF5 loading

**Cause**: File format mismatch (viewpoints vs TSP file)

**Current Limitation**: `load_tsp_result()` in `tsp_utils.py` expects TSP file format

**Workaround**:
- Ensure you're loading a TSP result file, not a viewpoints-only file
- TSP files are created by `mesh_to_tsp.py`
- Viewpoints files are created by `mesh_to_viewpoints.py`

**Check file type**:
```python
import h5py
with h5py.File('file.h5', 'r') as f:
    if 'format' in f['metadata'].attrs:
        print(f"Format: {f['metadata'].attrs['format']}")  # "viewpoints_only"
    elif 'num_points' in f['metadata'].attrs:
        print("Format: TSP result")
```

---

## 🔍 Key Differences from v2

| Aspect | v2 | v3 |
|--------|----|----|
| **Global Variables** | 5 globals | 0 globals |
| **main() Length** | 244 lines | 30 lines |
| **Configuration** | Hard-coded values | `SimulationConfig` class |
| **State Management** | Global dicts/lists | `WorldState`, `ViewpointManager` |
| **Unused Code** | ~200 lines | Removed |
| **Testability** | Low (globals, large functions) | High (pure functions, DI) |
| **Code Organization** | Mixed concerns | Clear separation |
| **Documentation** | Minimal | This document! |

---

## 📚 Related Files

### Dependencies
- `tsp_utils.py` - TSP file I/O functions
- `utilss/simulation_helper.py` - Robot setup helpers
- `analyze_joint_reconfigurations.py` - Reconfiguration analysis
- `eaik/IK_URDF.py` - EAIK analytical IK solver

### Input Files
- TSP tour HDF5 (from `mesh_to_tsp.py`)
- Robot config YAML (`robot_cfg/ur20.yml`)
- World config YAML (`collision_table.yml`)
- Robot URDF (`ur20.urdf`)
- Glass USD (`glass_isaac_ori.usdc`)

### Output Files
- `joint_trajectory_{method}_{timestamp}.csv`
- `joint_trajectory_{method}_{timestamp}_reconfig.txt`
- `ik_solutions_all_{timestamp}.h5`

---

## 🎓 Best Practices

### When to Use Each Method

**Random/First**:
- ✅ Quick prototyping
- ✅ When trajectory smoothness doesn't matter
- ❌ Production use (can cause large jumps)

**Greedy**:
- ✅ Good balance of speed and smoothness
- ✅ Real-time applications
- ✅ When DP is too slow
- ⚠️ Can get stuck in local minima

**Dynamic Programming**:
- ✅ Production quality trajectories
- ✅ When optimality is critical
- ✅ Offline planning
- ❌ Large viewpoint sets (>1000 points)

---

### Performance Optimization

**For Large Viewpoint Sets** (>500 points):
1. Use `greedy` method instead of `dp`
2. Reduce `ik_num_seeds` from 20 to 10
3. Consider subsampling viewpoints
4. Disable visualization (`--no_sim`)

**For Real-Time Applications**:
1. Pre-compute IK solutions offline
2. Cache collision checking results
3. Use `greedy` method for replanning
4. Reduce `interpolation_steps`

---

## 📝 Future Improvements

### Planned Features
- [ ] Support for multiple robots
- [ ] Parallel IK computation (GPU batching)
- [ ] Adaptive working distance
- [ ] Real-time trajectory replanning
- [ ] Visualization of collision spheres
- [ ] Web-based configuration UI

### Code Improvements
- [ ] Complete type hints coverage
- [ ] Unit tests for core functions
- [ ] Integration tests for full pipeline
- [ ] Logging instead of print statements
- [ ] Configuration file support (YAML/JSON)
- [ ] Plugin architecture for cost functions

---

## 🤝 Contributing

When modifying `run_app_v3.py`:

1. **Maintain backward compatibility** with v2 outputs
2. **Add type hints** to all new functions
3. **Document** configuration changes
4. **Test** with multiple TSP files
5. **Update** this documentation

---

## 📞 Support

For issues or questions:
1. Check this documentation first
2. Compare with `run_app_v2.py` behavior
3. Enable debug output in functions
4. Check Isaac Sim console for errors
5. Verify HDF5 file structure with `h5dump`

---

## 📄 License

Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

---

**Last Updated**: 2025-11-02
**Version**: 3.0
**Author**: Refactored from run_app_v2.py
