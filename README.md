# Vision Inspection Pipeline

Automated robot trajectory planning for comprehensive surface inspection using FOV-based viewpoint sampling, TSP optimization, and collision-aware motion planning.

---

## Overview

This pipeline generates collision-free robot trajectories for inspecting 3D objects using a mounted camera. The system:

1. **Samples viewpoints** from a 3D mesh based on camera field-of-view
2. **Optimizes visit order** using Traveling Salesman Problem (TSP) algorithms
3. **Computes IK solutions** with collision checking (CuRobo only, no simulation)
4. **Plans robot trajectories** using dynamic programming (no simulation)
5. **Simulates execution** in Isaac Sim (optional)
6. **Validates trajectories** for collisions using COAL library (optional)

### Key Features

- **FOV-based sampling**: Viewpoints determined by camera specifications (not arbitrary)
- **Modular pipeline**: Each stage runs independently with intermediate file outputs
- **Fast iteration**: Most stages run without Isaac Sim (only visualization needs it)
- **Collision-aware planning**: Validates entire trajectory including interpolated segments
- **Optimized visit order**: Multiple TSP algorithms (Nearest Neighbor, Random Insertion)
- **Multi-solution IK**: Dynamic programming selects optimal joint configurations
- **Z-up coordinate system**: Unified throughout (Isaac Sim, URDF, Pinocchio, COAL)
- **Centralized configuration**: All parameters in `common/config.py`

---

## Quick Start

### Prerequisites

- Python 3.8+
- Isaac Sim (only for visualization step, optional)
- Required packages: numpy, open3d, h5py, trimesh, torch, curobo

### Installation

The project is organized as follows:

```
vision_inspection/
├── common/                           # Shared utilities and configuration
│   ├── config.py                     # Central configuration
│   ├── coordinate_utils.py           # Geometric operations
│   ├── interpolation_utils.py        # Trajectory interpolation
│   ├── ik_utils.py                   # IK computation utilities
│   └── trajectory_planning.py        # Planning algorithms
│
├── scripts/                          # Main pipeline scripts
│   ├── mesh_to_viewpoints.py         # Step 1: Sample viewpoints
│   ├── viewpoints_to_tsp.py          # Step 2: Solve TSP
│   ├── compute_ik_solutions.py       # Step 3: Compute IK (CuRobo only)
│   ├── plan_trajectory.py            # Step 4: Plan trajectory
│   ├── simulate_trajectory.py        # Step 5: Simulate (Isaac Sim)
│   ├── run_full_pipeline.py          # Steps 3-5 integrated
│   ├── coal_check.py                 # Step 6: Validate collisions
│   ├── run_app_v3.py                 # DEPRECATED (kept for compatibility)
│   └── test_integration.py           # Integration tests
│
├── data/                             # Data directory
│   ├── object/                       # 3D mesh files (Z-up format)
│   ├── viewpoint/{num_points}/       # Sampled viewpoints (HDF5)
│   ├── tour/{num_points}/            # TSP-optimized tours (HDF5)
│   ├── ik/{num_points}/              # IK solutions (HDF5)
│   └── trajectory/{num_points}/      # Joint trajectories (CSV)
│
└── docs/                             # Documentation
    ├── REFACTORING_SUMMARY.md        # Modular pipeline details
    ├── FOV_VIEWPOINT_SAMPLING.md
    ├── VIEWPOITNS_TO_TSP_ANALYSIS.md
    ├── RUN_APP_V3_DOCUMENTATION.md
    └── COAL_COLLISION_CHECKER.md
```

---

## Pipeline Workflow

### Method 1: Step-by-Step (Recommended for Development)

This method allows you to inspect and modify intermediate results at each stage.

#### Step 1: Sample Viewpoints from Mesh

```bash
python scripts/mesh_to_viewpoints.py \
    --mesh_file data/object/glass_zup.obj \
    --save_path data/viewpoint/3000/viewpoints.h5 \
    --visualize
```

**Output**: `data/viewpoint/3000/viewpoints.h5` containing surface positions and normals

**Requirements**: Python only (no Isaac Sim)

**Notes**:
- The script now **automatically estimates** how many viewpoints are needed, so `--auto_num_points` is no longer required.
- Use `--adaptive_sampling` (optionally tune `--curvature_weight`) if you want curvature-aware density; otherwise sampling is uniform.
- Meshes must already be in meters. If your asset is authored in millimeters, scale it externally (e.g., in CAD or MeshLab) before running the script.
- Statistics plots are no longer generated; rely on the console summary or Open3D visualization.

---

#### Step 2: Optimize Visit Order with TSP

```bash
python scripts/viewpoints_to_tsp.py \
    --viewpoint_file data/viewpoint/3000/viewpoints.h5 \
    --algorithm both \
    --save_path data/tour/3000/tour.h5
```

**Output**: `data/tour/3000/tour.h5` containing optimized tour

**Requirements**: Python only (no Isaac Sim)

**Algorithms**:
- `nn`: Nearest Neighbor (fast, greedy)
- `ri`: Random Insertion (better quality)
- `both`: Try both and select best (default)

---

#### Step 3: Compute IK Solutions

```bash
python scripts/compute_ik_solutions.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --output data/ik/3000/ik_solutions.h5 \
    --robot ur20.yml
```

**Output**: `data/ik/3000/ik_solutions.h5` containing all IK solutions + collision-free flags

**Requirements**: CuRobo only (no Isaac Sim!)

**Time**: ~5-10 minutes for 3000 viewpoints

---

#### Step 4: Plan Robot Trajectory

```bash
python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method dp \
    --output data/trajectory/3000/joint_trajectory_dp.csv
```

**Output**:
- `data/trajectory/3000/joint_trajectory_dp.csv` - Joint trajectory
- `data/trajectory/3000/joint_trajectory_dp_reconfig.txt` - Analysis

**Requirements**: Python only (no Isaac Sim!)

**Time**: ~1-2 minutes

**Try different methods**:
```bash
# Dynamic programming (optimal, default)
python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method dp

# Greedy nearest neighbor (faster)
python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method greedy

# Random selection (baseline)
python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method random
```

---

#### Step 5: Simulate Trajectory (Optional, 범준이형)

```bash
omni_python scripts/simulate_trajectory.py \
    --trajectory data/trajectory/2369/joint_trajectory_dp.csv \
    --robot ur20.yml \
    --visualize_spheres \
    --interpolation_steps 60
```

**Output**: Visual confirmation in Isaac Sim

**Requirements**: Isaac Sim

**Time**: Real-time visualization

---

#### Step 6: Validate Collisions & Reconfigurations (Optional)

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --robot_urdf ur_description/ur20.urdf \
    --mesh data/object/glass_zup.obj \
    --interp-steps 30 \
    --check-reconfig \
    --reconfig-threshold 1.0 \
    --verbose
```

**Output**:
- Collision statistics and analysis
- Joint reconfiguration detection
- Collision report saved to `data/collision/{num_points}/collision.txt`

**Requirements**: Python with COAL library

**Time**: ~1-3 minutes

**New Features**:
- ✅ **Joint reconfiguration detection**: Identifies sudden large joint movements
- ✅ **Automated replanning**: Fixes both collisions and reconfigurations using CuRobo
- ✅ **Optimized batch planning**: Replan multiple segments efficiently
- ✅ **Last joint exclusion**: Ignores end-effector rotation in reconfiguration analysis

---

### Method 2: Integrated Pipeline (Recommended for Production)

Run steps 3-5 in one command:

#### Without Simulation (No Isaac Sim Needed!)

```bash
python scripts/run_full_pipeline.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --method dp
```

This will:
1. ✅ Compute IK solutions (CuRobo only)
2. ✅ Plan trajectory (pure Python)
3. ⏭️  Skip simulation

**Total time**: ~6-12 minutes (no Isaac Sim startup overhead)

---

#### With Simulation

```bash
python scripts/run_full_pipeline.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --method dp \
    --simulate
```

This will:
1. ✅ Compute IK solutions (CuRobo only)
2. ✅ Plan trajectory (pure Python)
3. ✅ Run simulation (Isaac Sim)

**Total time**: ~6-12 minutes + simulation time

---

#### Skip Completed Stages

```bash
# Skip IK computation if already done
python scripts/run_full_pipeline.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --method dp \
    --skip_ik \
    --ik_solutions data/ik/3000/ik_solutions.h5

# Skip both IK and planning
python scripts/run_full_pipeline.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --method dp \
    --skip_ik \
    --skip_planning \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --simulate
```

---

## Configuration

All system parameters are centralized in `common/config.py`.

### Camera Specifications

```python
CAMERA_SENSOR_WIDTH_PX = 4096
CAMERA_SENSOR_HEIGHT_PX = 3000
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0
CAMERA_WORKING_DISTANCE_MM = 110.0
CAMERA_DEPTH_OF_FIELD_MM = 0.5
CAMERA_OVERLAP_RATIO = 0.25
```

### World Configuration

```python
GLASS_POSITION = np.array([0.7, 0.0, 0.6])  # meters (x, y, z)
TABLE_POSITION = np.array([0.7, 0.0, 0.0])  # meters (x, y, z)
TABLE_DIMENSIONS = np.array([0.6, 1.0, 1.1])  # meters (x, y, z)
```

### Algorithm Parameters

```python
INTERPOLATION_STEPS = 60  # Steps between waypoints
IK_NUM_SEEDS = 20  # IK solver random seeds
COLLISION_INTERP_STEPS = 30  # Collision check interpolation
```

### File Paths

```python
DEFAULT_MESH_FILE = "data/object/glass_zup.obj"
DEFAULT_ROBOT_URDF = "ur_description/ur20.urdf"
DEFAULT_ROBOT_CONFIG_YAML = "ur_description/ur20.yml"
```

To modify configuration, edit `common/config.py` or override via command-line arguments.

---

## Coordinate System

**Important**: This pipeline uses **Z-up coordinate system** throughout.

- ✓ Isaac Sim: Z-up (native)
- ✓ URDF/Pinocchio: Z-up (native)
- ✓ COAL collision library: Z-up (native)
- ✓ Mesh files: Use Z-up format (e.g., `glass_zup.obj`)

**Migration from Y-up**: If you have Y-up meshes, convert using: `(x, y, z)_Yup → (z, -x, y)_Zup`

---

## Performance Notes

### Typical Runtime (3000 viewpoints)

| Stage | Runtime | Isaac Sim Required |
|-------|---------|-------------------|
| 1. Viewpoint Sampling | ~30 seconds | ❌ No |
| 2. TSP Optimization | ~2 minutes | ❌ No |
| 3. IK Computation | ~5-10 minutes | ❌ No (CuRobo only!) |
| 4. Trajectory Planning | ~1-2 minutes | ❌ No |
| 5. Simulation | Real-time | ✅ Yes |
| 6. Collision Validation | ~1-3 minutes | ❌ No |

**Total (without simulation)**: ~8-15 minutes

**Key Advantage**: Only the visualization step requires Isaac Sim!

### Optimization Tips

1. **Reduce viewpoint count**: Use `--num_points` or adjust camera overlap
2. **Use greedy IK selection**: `--method greedy` (faster than DP)
3. **Use faster TSP algorithm**: Set `--algorithm nn` for faster (but less optimal) tour
4. **Skip simulation**: Omit `--simulate` flag for faster testing

---

## Testing

### Integration Tests

Run the integration test suite:

```bash
/isaac-sim/python.sh scripts/test_integration.py
```

This verifies:
- ✓ Config module loads correctly
- ✓ Coordinate utilities work properly
- ✓ Interpolation utilities function correctly
- ✓ All scripts can import common modules

### Command-line Verification

Test each script with `--help`:

```bash
python scripts/mesh_to_viewpoints.py --help
python scripts/viewpoints_to_tsp.py --help
python scripts/compute_ik_solutions.py --help
python scripts/plan_trajectory.py --help
omni_python scripts/simulate_trajectory.py --help
python scripts/run_full_pipeline.py --help
```

---

## Advanced Usage

### Custom Camera Specifications

Override camera parameters:

```bash
python scripts/mesh_to_viewpoints.py \
    --mesh_file data/object/custom_object.obj \
    --fov_width 50.0 \
    --fov_height 40.0 \
    --working_distance 150.0 \
    --overlap 0.3
```

### Compare TSP Algorithms

```bash
# Try Nearest Neighbor only (fastest)
python scripts/viewpoints_to_tsp.py \
    --viewpoint_file data/viewpoint/3000/viewpoints.h5 \
    --algorithm nn \
    --num_starts 10

# Try Random Insertion only (better quality)
python scripts/viewpoints_to_tsp.py \
    --viewpoint_file data/viewpoint/3000/viewpoints.h5 \
    --algorithm ri \
    --num_starts 10

# Try both and select best (recommended)
python scripts/viewpoints_to_tsp.py \
    --viewpoint_file data/viewpoint/3000/viewpoints.h5 \
    --algorithm both \
    --num_starts 20
```

### Compare Trajectory Planning Methods

Since IK computation is separate, you can quickly compare different planning methods:

```bash
# Compute IK once
python scripts/compute_ik_solutions.py \
    --tsp_tour data/tour/3000/tour.h5

# Try all methods
python scripts/plan_trajectory.py --ik_solutions data/ik/3000/ik_solutions.h5 --method dp
python scripts/plan_trajectory.py --ik_solutions data/ik/3000/ik_solutions.h5 --method greedy
python scripts/plan_trajectory.py --ik_solutions data/ik/3000/ik_solutions.h5 --method random

# Compare results
cat data/trajectory/3000/joint_trajectory_*_reconfig.txt
```

### Collision & Reconfiguration Validation

#### Basic Collision Check with Safety Margin

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --collision_margin 0.01 \
    --verbose
```

#### Joint Reconfiguration Detection

Detect sudden large joint movements that may cause mechanical issues:

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --check-reconfig \
    --reconfig-threshold 1.0  # radians
```

**Parameters**:
- `--check-reconfig`: Enable reconfiguration detection (default: True)
- `--no-check-reconfig`: Disable reconfiguration detection
- `--reconfig-threshold`: Threshold in radians (default: 1.0)

**Output**:
```
Joint Reconfigurations:
  Threshold:            1.00 rad
  Excluded last joint:  True
  Total reconfigurations: 15
  Reconfiguration rate:   15.2%
  Reconfiguration segments: [12, 23, 34, 45, ...]
```

#### Automated Replanning for Problem Segments

Fix both collisions and reconfigurations using CuRobo:

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --attempt_replan \
    --replan_timeout 8.0 \
    --replan_max_attempts 20
```

**How it works**:
1. Detects collision and reconfiguration segments
2. Batches replanning requests for efficiency
3. Uses CuRobo MotionGen to generate collision-free paths
4. Rechecks only modified segments (optimized)
5. Saves collision-free trajectory if successful

**Output**:
```
4. Attempting CuRobo replanning for problematic segments...
  Segments requiring replanning:
    - Collision segments: 5
    - Reconfiguration segments: 8
    - Total (union): 12
  Planning 12 segments in batch...
  Batch planning completed in 14.52s
    Segment 3: SUCCESS (45 waypoints)
    Segment 7: SUCCESS (52 waypoints)
    ...
  Replanning summary: 11/12 segments successful
  Rechecking 11 replanned segments (optimized)...
  Checked 523 configurations (vs 1,090 for full trajectory)

✓ All collisions and reconfigurations resolved via replanning.
Saved to: data/trajectory/3000/collision_free_trajectory.csv
```

#### Use Actual Robot Meshes for Collision

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --use_link_meshes \
    --mesh_base_path ur_description
```

#### Complete Validation & Replanning Example

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/3000/joint_trajectory_dp.csv \
    --robot_urdf ur_description/ur20.urdf \
    --mesh data/object/glass_zup.obj \
    --interp-steps 30 \
    --check-reconfig \
    --reconfig-threshold 1.0 \
    --attempt_replan \
    --replan_timeout 8.0 \
    --collision_free_output data/trajectory/3000/optimized_trajectory.csv \
    --verbose
```

---

## Documentation

Detailed documentation for each component:

- 📖 [**Refactoring Summary**](docs/REFACTORING_SUMMARY.md) - Modular pipeline architecture
- 📖 [**FOV Viewpoint Sampling**](docs/FOV_VIEWPOINT_SAMPLING.md) - Viewpoint generation algorithm
- 📖 [**TSP Analysis**](docs/VIEWPOITNS_TO_TSP_ANALYSIS.md) - Tour optimization methods
- 📖 [**Run App V3**](docs/RUN_APP_V3_DOCUMENTATION.md) - Legacy trajectory planning (deprecated)
- 📖 [**COAL Collision Checker**](docs/COAL_COLLISION_CHECKER.md) - Collision validation

---

## Troubleshooting

### Common Issues

**Q: Import errors when running scripts**
```bash
ModuleNotFoundError: No module named 'common'
```
**A**: Make sure you're running from the project root (`vision_inspection/`).

---

**Q: Mesh file not found**
```bash
FileNotFoundError: data/object/glass_zup.obj
```
**A**: Check that you're in the project root directory or provide absolute path.

---

**Q: CUDA out of memory during IK computation**
```bash
RuntimeError: CUDA out of memory
```
**A**: Reduce batch size or split viewpoints into smaller chunks.

---

**Q: Few collision-free IK solutions**
```bash
With safe IK solutions: 150/3000
```
**A**: This may indicate:
1. Glass/table positions are too close to robot workspace
2. Viewpoints are in unreachable areas
3. Collision margins are too conservative
4. Check `common/config.py` world configuration

---

**Q: High reconfiguration count**
```bash
Total reconfigurations: 850
```
**A**: Try:
1. Use `--method dp` for optimal solution selection
2. Adjust `JOINT_WEIGHTS` in `common/config.py`
3. Increase `RECONFIGURATION_THRESHOLD`

---

## Migration from Old Workflow

### Old Workflow (Deprecated)

```bash
# Steps 1-2 (same)
python scripts/mesh_to_viewpoints.py ...
python scripts/viewpoints_to_tsp.py ...

# Step 3 (monolithic, requires Isaac Sim)
omni_python scripts/run_app_v3.py \
    --tsp_tour_path data/tour/3000/tour.h5 \
    --selection_method dp
```

### New Workflow (Recommended)

```bash
# Steps 1-2 (same)
python scripts/mesh_to_viewpoints.py ...
python scripts/viewpoints_to_tsp.py ...

# Steps 3-5 (modular, Isaac Sim optional)
python scripts/run_full_pipeline.py \
    --tsp_tour data/tour/3000/tour.h5 \
    --method dp \
    --simulate  # Optional
```

**Benefits**:
- ✅ Faster iteration (no Isaac Sim for most stages)
- ✅ Intermediate file outputs for debugging
- ✅ Can try multiple planning methods quickly
- ✅ Better separation of concerns

See [`docs/REFACTORING_SUMMARY.md`](docs/REFACTORING_SUMMARY.md) for detailed migration guide.

---

## Recent Changes (2025-11-08)

Major refactoring completed to improve modularity and performance:

### Modular Pipeline Architecture
- ✅ Split monolithic `run_app_v3.py` into 3 independent scripts
- ✅ `compute_ik_solutions.py` - CuRobo only (no Isaac Sim!)
- ✅ `plan_trajectory.py` - Pure Python trajectory planning
- ✅ `simulate_trajectory.py` - Isaac Sim visualization (optional)
- ✅ `run_full_pipeline.py` - Integrated workflow runner

### New Common Modules
- ✅ `common/ik_utils.py` - IK computation utilities
- ✅ `common/trajectory_planning.py` - Planning algorithms (DP, greedy, random)

### Key Improvements
- ✅ **Most pipeline runs without Isaac Sim** (only visualization needs it)
- ✅ **Faster iteration** - compute IK once, try multiple planning methods
- ✅ **Intermediate file outputs** - HDF5 for IK solutions, CSV for trajectories
- ✅ **Better testability** - each stage can be tested independently
- ✅ **Centralized configuration** in `common/config.py`
- ✅ **Z-up coordinate system** unified throughout

---

## Contributing

When adding new features:

1. Add configuration values to `common/config.py`
2. Use common utilities from `common/` modules
3. Maintain Z-up coordinate convention
4. Update relevant documentation
5. Add integration tests if needed

---

## License

[Specify license here]

---

## Contact

[Specify contact information here]

---

**Last Updated**: 2025-11-08
