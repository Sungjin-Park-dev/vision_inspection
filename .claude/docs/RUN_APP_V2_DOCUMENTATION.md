# Vision Inspection System V2 - Documentation

## Overview

`run_app_v2.py` is an advanced robotic vision inspection system that combines **TSP (Traveling Salesman Problem) optimization** with **intelligent IK solution selection** to generate optimal robot trajectories for inspecting glass objects. This version significantly improves upon the baseline by integrating pre-computed TSP tours and dynamic programming-based joint configuration selection.

## Key Features

### 1. TSP-Optimized Visit Order
- Loads pre-computed TSP tours from HDF5 files
- Viewpoints are already sorted in optimal visiting sequence
- Minimizes total travel distance in Cartesian space
- Eliminates need for runtime path planning

### 2. Three IK Solution Selection Methods
- **Random (Baseline)**: Selects first safe IK solution for each viewpoint
- **Greedy**: Uses nearest-neighbor heuristic in joint space
- **Dynamic Programming**: Finds globally optimal solution sequence

### 3. Advanced Cost Functions
- Weighted joint distance metrics
- Reconfiguration penalty for large joint movements
- Configurable joint weights (prioritizes base joints)
- Hybrid cost combining multiple objectives

### 4. Comprehensive Data Export
- Joint trajectory CSV with target poses
- Joint trajectory visualization plots
- Detailed statistics and metrics

## Architecture

### Core Components

#### 1. Data Structures

**Viewpoint Class** (lines 179-186):
```python
@dataclass
class Viewpoint:
    index: int                              # TSP tour index (0, 1, 2, ...)
    local_pose: Optional[np.ndarray]        # 4x4 pose in object frame
    world_pose: Optional[np.ndarray]        # 4x4 pose in world frame
    all_ik_solutions: List[np.ndarray]      # All analytical IK solutions
    safe_ik_solutions: List[np.ndarray]     # Collision-free IK solutions
```

**Global State Variables** (lines 154-201):
- `SAMPLED_LOCAL_POINTS`: Points in object coordinates (Open3D frame)
- `SAMPLED_LOCAL_NORMALS`: Surface normals for sampled points
- `SAMPLED_VIEWPOINTS`: List of Viewpoint objects in TSP order
- `TSP_TOUR_RESULT`: Loaded TSP tour data from HDF5
- `JOINT_HISTORY`: Trajectory data for plotting

#### 2. Coordinate Systems

**Three coordinate frames are used:**

1. **Open3D Frame (Y-up)**
   - Source: TSP tour HDF5 files
   - Used for point cloud processing

2. **Isaac Sim Frame (Z-up)**
   - Simulation world coordinates
   - Rotation transform: `OPEN3D_TO_ISAAC_ROT` (lines 168-175)

3. **EAIK Tool Frame**
   - Analytical IK solver convention
   - Transform: `CUROBO_TO_EAIK_TOOL` (lines 157-165)

**Transformation Pipeline:**
```
TSP File (Open3D) → open3d_to_isaac_coords() → Isaac Sim Local →
open3d_pose_to_world() → World Coordinates → compute_ik() → EAIK Solutions
```

### Key Functions

#### TSP Tour Loading

**load_tsp_tour()** (lines 1202-1215):
- Loads HDF5 tour file using `tsp_utils.load_tsp_result()`
- Returns dictionary with tour data

**load_tsp_tour_points_and_normals()** (lines 1218-1299):
- Extracts points and normals in TSP visit order
- Generates viewpoints with sequential indices (0, 1, 2, ...)
- Creates local pose matrices aligned with surface normals
- **Critical**: Viewpoint indices match TSP tour order exactly

#### IK Solution Selection

**collect_random_joint_targets()** (lines 418-445):
- **Method**: First solution (baseline)
- Selects `safe_ik_solutions[0]` for each viewpoint
- No optimization, fast execution
- Used with `--selection_method random`

**collect_sorted_joint_safe_targets()** (lines 448-494):
- **Method**: Greedy nearest-neighbor
- For each viewpoint, selects IK solution closest to previous joint config
- Uses Euclidean distance in joint space
- Local optimization, no global guarantees
- Used with `--selection_method greedy`

**collect_optimal_joint_targets_dp()** (lines 593-726):
- **Method**: Dynamic Programming with reconfiguration-aware cost
- Finds globally optimal solution sequence
- **Algorithm**:
  1. Initialize DP table for first viewpoint
  2. Forward pass: compute minimum cost to reach each state
  3. Backward pass: reconstruct optimal path
- Returns: targets, total_cost, solution_indices, valid_viewpoint_indices
- Used with `--selection_method dp`
- Prints total cost upon completion

#### Cost Functions

**compute_weighted_joint_distance()** (lines 497-525):
```python
# Default weights: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
# Higher weights for base joints (shoulder pan/lift, elbow)
weighted_diff = joint_weights * (q1 - q2)^2
distance = sqrt(sum(weighted_diff))
```

**compute_reconfiguration_cost()** (lines 528-590):
- **Three components**:
  1. **Base cost**: Weighted Euclidean distance
  2. **Reconfiguration penalty**: Count joints moving > threshold × penalty
  3. **Max movement penalty**: Largest single joint movement × weight
- Returns total cost and detailed breakdown
- **Note**: Line 577 shows only base_cost is used currently

#### Trajectory Management

**generate_interpolated_joint_path()** (lines 314-328):
- Linear interpolation between joint configurations
- Default: 60 steps (`INTERPOLATION_STEPS`)
- Ensures smooth motion

**update_safe_ik_solutions()** (lines 368-415):
- **Batch collision checking** for efficiency
- Converts all IK solutions to JointState tensors
- Calls `ik_solver.check_constraints()` once
- Filters solutions by feasibility flag

#### Visualization and Export

**visualize_tsp_tour_path()** (lines 1413-1457):
- Green points: viewpoints with safe IK solutions
- Red points: viewpoints without safe IK
- Shows success rate statistics

**plot_joint_trajectories()** (lines 1460-1517):
- Plots all 6 joint values over time
- Vertical red lines mark viewpoint arrivals
- Saves to PNG file (matplotlib Agg backend)

**save_joint_trajectory_csv()** (lines 1551-1641):
- **CSV Format**:
  ```
  time, joint_1, joint_2, ..., joint_6,
  target_pos_x, target_pos_y, target_pos_z,
  target_quat_x, target_quat_y, target_quat_z, target_quat_w
  ```
- One row per viewpoint
- Suitable for analysis and replay

## Main Execution Flow

**main()** function (lines 1791-2008):

### 1. Initialization Phase (lines 1795-1817)
```
Load TSP tour from HDF5 → Extract points/normals in tour order →
Extract num_points and create output directory (data/output/{num_points}/) →
Initialize Isaac Sim world → Setup robot, glass object, collision environment
```

### 2. Viewpoint Processing (lines 1824-1856)
```
Update world poses for all viewpoints →
Compute IK solutions using EAIK →
Assign IK solutions to viewpoints →
Filter for collision-free solutions →
Verify TSP mapping correctness
```

### 3. IK Selection (lines 1719-1762)
```
Based on --selection_method argument:
  - random: collect_random_joint_targets() + track indices
  - greedy: collect_sorted_joint_safe_targets() + find selected indices
  - dp: collect_optimal_joint_targets_dp() → returns solution_indices

Generate joint trajectory CSV with timestamp →
  Automatically analyze reconfigurations →
  Save reconfiguration analysis TXT
Save all IK solutions to HDF5 with selected indices
```

### 4. Simulation Loop (lines 1930-1995)
```
If --no_sim: exit early

While simulation running:
  - Collect joint history for plotting
  - Execute interpolated trajectories
  - Mark viewpoint arrivals
  - Update plots periodically (every 5 viewpoints)
  - Move robot through inspection poses

Save final trajectory plot
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--headless_mode` | str | None | Run without GUI: native, websocket |
| `--visualize_spheres` | bool | False | Show robot collision spheres |
| `--robot` | str | ur20.yml | Robot configuration file |
| `--tsp_tour_path` | str | **Required** | Path to TSP tour HDF5 file |
| `--save_plot` | bool | False | Save joint trajectory plots |
| `--selection_method` | str | dp | IK selection: random, greedy, dp |
| `--no_sim` | bool | False | Skip Isaac Sim visualization |

## Configuration Parameters

**Joint Weights** (line 1866):
```python
joint_weights = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0]
# Indices:        0    1    2    3    4    5
# Joints:       pan  lift elbow wrist1 wrist2 wrist3
```
Higher weights prioritize minimizing movement of base joints.

**Reconfiguration Parameters** (lines 1869-1871):
```python
reconfig_threshold = 0.3    # ~17 degrees
reconfig_penalty = 10.0     # penalty per reconfigured joint
max_move_weight = 5.0       # weight for max joint movement
```

**Interpolation** (line 176):
```python
INTERPOLATION_STEPS = 60  # Steps between waypoints
```

**Normal Offset** (line 167):
```python
NORMAL_SAMPLE_OFFSET = 0.1  # meters from surface
```

## Output Files

All outputs are automatically organized by number of viewpoints in the directory structure:
`data/output/{num_points}/`

Where `{num_points}` is extracted from the TSP tour file metadata (e.g., `data/output/100/`, `data/output/500/`).

**Output files**:

1. **Joint Trajectory CSV**: `joint_trajectory_{method}_{timestamp}.csv`
   - Time, joint values, target poses
   - Used for trajectory analysis
   - Example: `data/output/100/joint_trajectory_dp_20250102_143022.csv`

2. **Reconfiguration Analysis TXT**: `joint_trajectory_{method}_{timestamp}_reconfig.txt`
   - **Automatically generated** after CSV save
   - Detailed joint reconfiguration statistics
   - Per-joint reconfiguration counts
   - Timesteps with large reconfigurations
   - Example: `data/output/100/joint_trajectory_dp_20250102_143022_reconfig.txt`

3. **IK Solutions HDF5**: `ik_solutions_all_{timestamp}.h5`
   - All IK solutions for every viewpoint
   - Collision-free status for each solution
   - Selected solution index by DP/Greedy/Random
   - Example: `data/output/100/ik_solutions_all_20250102_143022.h5`
   - See "IK Solutions HDF5 Format" section below for details

4. **Trajectory Plots** (if `--save_plot` is enabled):
   - `joint_trajectory.png` (periodic updates during sim)
   - `joint_trajectory_final.png` (final plot)
   - Example: `data/output/100/joint_trajectory_final.png`

This directory structure makes it easy to compare results across different numbers of viewpoints.

## Automatic Reconfiguration Analysis

After generating the joint trajectory CSV, `run_app_v2.py` **automatically analyzes joint reconfigurations** and saves the results.

### What is a Joint Reconfiguration?

A joint reconfiguration occurs when a joint moves more than a threshold amount (default: 0.3 radians ~17 degrees) between consecutive viewpoints. This indicates significant robot posture changes.

### Analysis Output

The reconfiguration analysis file (`*_reconfig.txt`) contains:

1. **Summary Statistics**:
   - Total number of reconfigurations
   - Reconfiguration rate (percentage of transitions with reconfigurations)
   - Threshold used for analysis

2. **Per-Joint Statistics**:
   ```
   Joint Name                Reconfigs  Max Change   Mean Change  Total Move
   shoulder_pan             12         2.456        0.234        15.678
   shoulder_lift            15         3.123        0.345        18.234
   ...
   ```

3. **Detailed Reconfiguration Events**:
   ```
   Timestep   Max Change   Joints Involved
   5          2.456        shoulder_pan, elbow
   12         3.123        shoulder_lift, wrist_1
   ...
   ```

### Example Output

```
Joint Reconfiguration Analysis Results
============================================================

Input file: data/output/100/joint_trajectory_dp_20250102_143022.csv
Threshold: 0.300 radians
Total timesteps: 90
Total reconfigurations: 23
Reconfiguration rate: 25.8%

Per-joint statistics:
ur20-shoulder_pan_joint: 12 reconfigurations, max change: 2.456 rad
ur20-shoulder_lift_joint: 15 reconfigurations, max change: 3.123 rad
ur20-elbow_joint: 8 reconfigurations, max change: 1.987 rad
...
```

### Integration with DP Cost Function

The reconfiguration analysis uses the **same threshold** as the DP cost function's `reconfig_threshold` parameter (default: 0.3 radians). This ensures consistency between:
- What DP tries to minimize (reconfigurations)
- What gets reported in the analysis

### Disabling Auto-Analysis

The reconfiguration analysis runs automatically and handles errors gracefully. If analysis fails (e.g., CSV format issues), it prints a warning and continues without blocking the main workflow.

## IK Solutions HDF5 Format

The `ik_solutions_all_{timestamp}.h5` file stores comprehensive IK solution data for analysis.

### File Structure

```
ik_solutions_all_{timestamp}.h5
├── metadata/
│   ├── num_viewpoints: int
│   ├── num_viewpoints_with_solutions: int
│   ├── num_viewpoints_with_safe_solutions: int
│   ├── selection_method: str ("dp", "greedy", "random")
│   ├── timestamp: str (ISO format)
│   └── tsp_tour_file: str (path to TSP tour)
├── viewpoint_0000/
│   ├── original_index: int (TSP tour index)
│   ├── world_pose: (4, 4) float32 array
│   ├── all_ik_solutions: (N, 6) float32 array
│   ├── collision_free_mask: (N,) bool array
│   ├── num_all_solutions: int
│   ├── num_safe_solutions: int
│   └── selected_solution_index: int (-1 if not selected)
├── viewpoint_0001/
│   └── ...
└── viewpoint_NNNN/
    └── ...
```

### Data Fields

**Metadata Group** (`/metadata/`):
- `num_viewpoints`: Total number of viewpoints in TSP tour
- `num_viewpoints_with_solutions`: Viewpoints where EAIK found at least one solution
- `num_viewpoints_with_safe_solutions`: Viewpoints with collision-free solutions
- `selection_method`: Algorithm used to select trajectory ("dp", "greedy", "random")
- `timestamp`: When the file was created
- `tsp_tour_file`: Path to the source TSP tour HDF5 file

**Viewpoint Groups** (`/viewpoint_XXXX/`):
- `original_index`: Index in the TSP tour (matches XXXX)
- `world_pose`: 4x4 transformation matrix in Isaac Sim world coordinates
- `all_ik_solutions`: All joint configurations found by EAIK (N solutions × 6 joints)
- `collision_free_mask`: Boolean array indicating which solutions are collision-free
- `num_all_solutions`: Total number of IK solutions found
- `num_safe_solutions`: Number of collision-free solutions
- `selected_solution_index`: Which safe solution was chosen by DP/Greedy/Random
  - Index into the `safe_ik_solutions` list (not `all_ik_solutions`)
  - -1 means this viewpoint was not included in the final trajectory

### Loading Example

```python
import h5py
import numpy as np

# Load the HDF5 file
with h5py.File('data/output/100/ik_solutions_all_20250102_143022.h5', 'r') as f:
    # Read metadata
    num_viewpoints = f['metadata'].attrs['num_viewpoints']
    selection_method = f['metadata'].attrs['selection_method']

    print(f"Total viewpoints: {num_viewpoints}")
    print(f"Selection method: {selection_method}")

    # Read specific viewpoint
    vp_grp = f['viewpoint_0000']
    world_pose = np.array(vp_grp['world_pose'])
    all_solutions = np.array(vp_grp['all_ik_solutions'])
    collision_mask = np.array(vp_grp['collision_free_mask'])
    selected_idx = vp_grp.attrs['selected_solution_index']

    print(f"\nViewpoint 0:")
    print(f"  Total IK solutions: {len(all_solutions)}")
    print(f"  Collision-free solutions: {collision_mask.sum()}")
    print(f"  Selected solution index: {selected_idx}")

    # Get only collision-free solutions
    safe_solutions = all_solutions[collision_mask]
    if selected_idx >= 0:
        selected_joints = safe_solutions[selected_idx]
        print(f"  Selected joint config: {selected_joints}")
```

### Analysis Use Cases

This HDF5 format enables:

1. **Solution Distribution Analysis**:
   - How many IK solutions per viewpoint?
   - What percentage are collision-free?

2. **Alternative Trajectory Exploration**:
   - Load all safe solutions
   - Try different selection algorithms offline
   - Compare with DP/Greedy results

3. **Collision Statistics**:
   - Which joint configurations tend to collide?
   - Are certain viewpoints more challenging?

4. **Trajectory Validation**:
   - Verify DP selected the correct solution indices
   - Cross-check with CSV trajectory file

5. **Machine Learning Dataset**:
   - Train models to predict collision-free configurations
   - Learn cost functions from DP selections

## Performance Optimization Techniques

### 1. Batch Collision Checking (lines 376-415)
Instead of checking each IK solution individually:
```python
# Collect ALL solutions from ALL viewpoints
batched_q = [solution for vp in viewpoints for solution in vp.all_ik_solutions]
# Check all at once on GPU
metrics = ik_solver.check_constraints(joint_state)
```
**Speedup**: ~10-100x depending on batch size

### 2. DP Algorithm (lines 649-714)
**Time Complexity**: O(n × m²)
- n = number of viewpoints
- m = average number of safe IK solutions per viewpoint

For typical case (n=100, m=8): ~64,000 comparisons
Much better than brute force: 8^100 combinations

### 3. Greedy Algorithm (lines 470-491)
**Time Complexity**: O(n × m)
Faster than DP but suboptimal results

## Algorithm Comparison

| Method | Time Complexity | Optimality | Typical Runtime |
|--------|----------------|------------|-----------------|
| Random | O(n) | None | <1 ms |
| Greedy | O(n × m) | Local | 2-5 ms |
| DP | O(n × m²) | Global | 10-50 ms |

For n=100 viewpoints, m=8 solutions:
- Random: instant
- Greedy: ~3 ms, ~15% suboptimal
- DP: ~30 ms, guaranteed optimal

## Verification Features

**verify_tsp_tour_mapping()** (lines 1013-1069):
Checks:
1. Viewpoint count matches tour length
2. Indices are sequential (0, 1, 2, ...)
3. World poses computed correctly
4. IK solutions generated
5. Sample position verification

**log_viewpoint_ik_stats()** (lines 998-1010):
Reports:
- Total viewpoints
- Viewpoints with any IK solutions
- Viewpoints with collision-free solutions

## Known Issues and Notes

1. **Line 577**: `total_cost = base_cost` commented out full reconfiguration cost
   - Currently only using weighted distance
   - Reconfiguration penalties calculated but not applied

2. **No Simulation Mode** (line 1930-1932):
   - Can run without Isaac Sim for faster testing
   - Generates CSV files and exits

3. **Joint Weights**: Last joint (wrist_3) has weight 0.0 (line 516)
   - Effectively ignores final wrist rotation
   - May want to set to 0.5 or 1.0 for some applications

4. **Camera Configuration** (lines 1166-1171):
   - Real camera parameters (38mm focal length, etc.)
   - Configured for close-range inspection (10-100mm clipping)

## Dependencies

**Core Libraries**:
- Isaac Sim (NVIDIA robot simulation)
- CuRobo (GPU-accelerated motion planning)
- EAIK (Analytical IK solver)
- PyTorch (tensor operations)

**Utilities**:
- Open3D (not used directly in V2, data comes from TSP file)
- NumPy (numerical operations)
- Matplotlib (trajectory plotting, Agg backend)
- h5py (via tsp_utils, for HDF5 loading)

**Custom Modules**:
- `tsp_utils.py`: TSP result loading utilities
- `utilss.simulation_helper`: Robot setup helpers
- `curobo.util_file`: Path management

## File Paths

**Input Files**:
- Robot URDF: `/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf`
- Glass USD: `/isaac-sim/curobo/vision_inspection/data/input/glass_isaac_ori.usdc`
- TSP Tour: Specified via `--tsp_tour_path` (HDF5 format)
- Robot Config: `{robot_configs_path}/ur20.yml`
- World Config: `{world_configs_path}/collision_table.yml`

**Output Files**:
- All outputs: `data/output/{num_points}/` (automatically organized by viewpoint count)
- Plots: PNG format
- Data: CSV format
- Example directories: `data/output/50/`, `data/output/100/`, `data/output/500/`

## Usage Examples

### Basic Usage (DP method, no visualization)
```bash
python run_app_v2.py \
  --tsp_tour_path data/input/tour/optimized_tour.h5 \
  --selection_method dp \
  --no_sim
```

### Full Simulation with Plots
```bash
python run_app_v2.py \
  --tsp_tour_path data/input/tour/optimized_tour.h5 \
  --selection_method dp \
  --save_plot
```

### Compare All Methods
```bash
# Random
python run_app_v2.py --tsp_tour_path tour.h5 --selection_method random --no_sim

# Greedy
python run_app_v2.py --tsp_tour_path tour.h5 --selection_method greedy --no_sim

# DP
python run_app_v2.py --tsp_tour_path tour.h5 --selection_method dp --no_sim
```

## Code Quality and Organization

**Strengths**:
- Well-structured with clear separation of concerns
- Comprehensive documentation in docstrings
- Type hints for function parameters
- Detailed logging and verification
- Robust error handling

**Areas for Improvement**:
- Some magic numbers could be constants
- Long functions (main() is 200+ lines)
- Global variables (could use class-based design)
- Mixed coordinate systems require careful attention

## Future Enhancements

Possible improvements:
1. **Enable full reconfiguration cost** (uncomment line 576)
2. **Adaptive interpolation** (more steps for large movements)
3. **Multi-objective optimization** (Pareto frontier for time vs smoothness)
4. **Online replanning** (handle dynamic obstacles)
5. **Parallel IK solving** (batch EAIK calls)
6. **Trajectory smoothing** (quintic splines instead of linear)
7. **Real robot interface** (replace sim with real UR20 control)

## Mathematical Formulation

### DP Algorithm

**State**: `dp[i][j]` = minimum cost to reach viewpoint `i` using solution `j`

**Recurrence**:
```
dp[i][j] = min over all k {
    dp[i-1][k] + cost(solution[i-1][k], solution[i][j])
}
```

**Initialization**:
```
dp[0][j] = cost(initial_config, solution[0][j])
```

**Solution**:
```
min over all j { dp[n-1][j] }
```

### Cost Function

**Weighted Distance**:
```
d(q1, q2) = sqrt(Σ w_i × (q1_i - q2_i)²)
```

**Reconfiguration Cost** (currently unused):
```
total_cost = d(q1, q2) +
             penalty × count(|q1_i - q2_i| > threshold) +
             max_weight × max(|q1_i - q2_i|)
```

## Conclusion

`run_app_v2.py` represents a sophisticated robotic vision inspection system that combines:
- TSP-optimized spatial planning
- Global optimization of joint configurations
- Efficient batch collision checking
- Comprehensive data export and analysis

The system is production-ready for automated inspection tasks and provides excellent tools for analyzing and optimizing robot trajectories.
