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
omni_python scripts/mesh_to_viewpoints.py \
    --mesh_file data/object/glass_zup.obj \
    --overlap 0.5
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
omni_python scripts/viewpoints_to_tsp.py \
    --viewpoint_file data/viewpoint/3000/viewpoints.h5
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
omni_python scripts/compute_ik_solutions.py \
    --tsp_tour data/tour/3000/tour.h5
```

**Output**: `data/ik/3000/ik_solutions.h5` containing all IK solutions + collision-free flags

**Requirements**: CuRobo only (no Isaac Sim!)

**Time**: ~5-10 minutes for 3000 viewpoints

---

#### Step 4: Plan Robot Trajectory

```bash
omni_python scripts/plan_trajectory.py \
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
omni_python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method dp

# Greedy nearest neighbor (faster)
omni_python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method greedy

# Random selection (baseline)
omni_python scripts/plan_trajectory.py \
    --ik_solutions data/ik/3000/ik_solutions.h5 \
    --method random
```

---

#### Step 5: Simulate Trajectory (Optional, 범준이형)

```bash
omni_python scripts/simulate_trajectory.py \
    --trajectory data/trajectory/2369/joint_trajectory_dp.csv
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

**Time**: ~1-3 minutes (sequential) or ~10-30 seconds (parallel with 8 cores)

**New Features**:
- ✅ **Joint reconfiguration detection**: Identifies sudden large joint movements
- ✅ **Automated replanning**: Fixes both collisions and reconfigurations using CuRobo
- ✅ **Optimized batch planning**: Replan multiple segments efficiently
- ✅ **Last joint exclusion**: Ignores end-effector rotation in reconfiguration analysis
- ✅ **Batch interpolation**: Fast GPU-accelerated trajectory interpolation using CuRobo
- ✅ **Parallel collision checking**: Multi-core processing for 5-10x speedup on large trajectories

**Performance Optimization (NEW!)**:

For large trajectories (>1000 waypoints), enable parallel collision checking:

```bash
omni_python scripts/coal_check.py \
    --trajectory data/trajectory/5000/joint_trajectory_dp.csv \
    --parallel \
    --num-workers 8 \
    --interp-steps 5 \
    --verbose
```

**Performance Comparison (5,504 waypoints, 28,433 configs)**:
- Sequential: ~36 seconds
- Parallel (8 cores): ~7.7 seconds
- **Speedup: 4.7x faster!**
