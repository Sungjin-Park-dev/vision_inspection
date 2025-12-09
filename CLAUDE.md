# Vision Inspection Pipeline - Simplification Project

## Project Goal
Integrate the 4-stage inspection pipeline into an educational Script
- Maintain original functionality while maximizing simplification
- Remove complex logic, keep only core features

## Pipeline Stages
1. **Viewpoint Generation** - Mesh → Viewpoints (surface positions + normals)
2. **IK Computation** - Viewpoints → Joint configurations
3. **GTSP Optimization** - Joint configs → Optimized trajectory
4. **Collision Check** - Trajectory → Collision-free path

## Completed Work

### Step 1: Viewpoint Generation ✓
**Unified Script:** `new_scripts/1_create_viewpoint.py`

**Original:**
- `scripts/preprocess_mesh.py` - Multi-material preprocessing
- `scripts/mesh_to_viewpoints.py` - Viewpoint sampling

**Key Simplifications:**
- ❌ Removed curvature-adaptive sampling → ✅ Uniform Poisson disk only
- ❌ Removed stratified sampling
- ❌ Removed DOF checking
- ❌ Removed material name selection → ✅ RGB color matching only

**Usage:**
```bash
# Auto-path mode
/isaac-sim/python.sh new_scripts/1_create_viewpoint.py \
    --object sample \
    --material-rgb "0,255,0"

# Number of viewpoints is calculated automatically based on:
#   - Surface area of target mesh
#   - Camera FOV and working distance
#   - Overlap ratio

# Paths are auto-generated:
#   Input:  data/sample/mesh/source.obj (multi-material OBJ + MTL)
#   Output: data/sample/viewpoint/{calculated_num}/viewpoints.h5
```

**Input/Output:**
- Input: Multi-material OBJ + RGB color
- Output: HDF5 (surface positions + normals)
- Paths: Auto-generated from `--object` (num_viewpoints calculated automatically)

---

### Step 2: Trajectory Generation ✓
**Unified Script:** `new_scripts/2_generate_trajectory.py` (~1500 lines)

**Original:**
- `scripts/compute_ik_solutions.py` - IK computation
- `scripts/fk_gtsp_gpu_claude2.py` - GTSP optimization
- `scripts/check_collision.py` - Collision checking with replanning

**Key Simplifications:**
- ❌ Removed intermediate HDF5 saves → ✅ In-memory processing
- ❌ Removed CuPy GPU acceleration → ✅ NumPy only
- ❌ Removed complex 4-step pipeline → ✅ Simplified 3-step collision checking
- ❌ Removed adaptive MST neighbor graph → ✅ k-NN only
- ❌ Removed Numba JIT FK → ✅ Simple NumPy/Python FK using DH parameters
- ✅ Kept full MotionGen replanning capability
- ✅ Kept EAIK analytical IK solver
- ✅ Kept DP optimization for IK selection

**Usage:**
```bash
# Auto-path mode (recommended)
/isaac-sim/python.sh new_scripts/2_generate_trajectory.py \
    --object sample \
    --num_viewpoints 163 \
    --knn 5

# Paths are auto-generated:
#   Input:  data/sample/viewpoint/163/viewpoints.h5
#   Output: data/sample/trajectory/163/trajectory.csv
#   Mesh:   data/sample/mesh/source.obj (for collision checking)
```

**Input/Output:**
- Input: HDF5 (viewpoints from Step 1)
- Output: CSV (collision-free joint trajectory)
- Paths: Auto-generated from `--object` and `--num_viewpoints`

**Pipeline:**
1. Load viewpoints from HDF5
2. Setup collision world (auto-detect mesh)
3. Compute IK solutions using EAIK
4. Filter collision-free IK solutions
5. Optimize visit order using GTSP + DP
6. Interpolate trajectory and check collisions
7. Replan colliding segments with MotionGen
8. Save final trajectory CSV

**Results (163 viewpoints):**
- IK computation: 163/163 success
- Collision-free IK: 147/163 (90%)
- GTSP optimization: 147 clusters, cost 26.60
- Collision checking: 2/408 interpolated configs collided
- Replanning: 2/2 segments successfully replanned
- Execution time: ~2.4s

---

### Step 3: Simulation ✓
**Unified Script:** `new_scripts/3_simulation.py` (~1100 lines)

**Original:**
- `scripts/simulate_trajectory.py` - Isaac Sim trajectory execution

**Key Simplifications:**
- ❌ Removed adaptive interpolation → ✅ Direct waypoint execution
- ✅ Added auto-path generation (--object + --num_viewpoints)
- ❌ Removed all common/ dependencies → ✅ 100% self-contained
- ✅ Inlined ALL utilities: CLI utils, data I/O, world setup, simulation helper (~445 lines)
- ✅ Uses ONLY `new_common/config.py` for configuration
- ✅ Kept sphere visualization (optional)

**Usage:**
```bash
# Auto-path mode (recommended)
/isaac-sim/python.sh new_scripts/3_simulation.py \
    --object sample \
    --num_viewpoints 163

# With options
/isaac-sim/python.sh new_scripts/3_simulation.py \
    --object sample \
    --num_viewpoints 163 \
    --visualize_spheres

# Paths are auto-generated:
#   Trajectory: data/sample/trajectory/163/trajectory.csv
#   Mesh:       data/sample/mesh/source.obj
```

**Input/Output:**
- Input: CSV (collision-free trajectory from Step 2)
- Output: Visual simulation in Isaac Sim
- Paths: Auto-generated from `--object` and `--num_viewpoints`

**Pipeline:**
1. Load trajectory CSV (direct waypoint loading)
2. Initialize Isaac Sim world with robot
3. Setup collision world using config
4. Auto-detect and load object mesh
5. Execute waypoints directly (no interpolation)
6. Optional: Visualize robot collision spheres

**Features:**
- 100% self-contained (~1100 lines)
- NO common/ dependencies (only new_common/config)
- Direct waypoint execution (trajectory pre-interpolated)
- Simplified CLI with auto-path generation
- Auto-path resolution (--object + --num_viewpoints)

---

## Implementation Principles

### 1. File Structure
- `new_common/` - Common config only (camera specs)
- `new_scripts/` - Self-contained scripts per stage
- DO NOT use existing `common/` modules (minimize dependencies)

### 2. Code Style
- Each script is self-contained (includes necessary functions internally)
- Minimal CLI interface
- Clear input/output structure
- Easy to understand for educational purposes

### 3. Simplification Criteria
- Remove complex adaptive logic
- Minimize edge case handling
- Keep only core algorithms
- Visualization is optional (--visualize flag)

---

## Directory Structure

```
vision_inspection/
├── new_common/
│   ├── __init__.py
│   └── config.py                    # Camera specs + world config
├── new_scripts/
│   ├── 1_create_viewpoint.py       # ✅ Done (Step 1: Mesh → Viewpoints)
│   ├── 2_generate_trajectory.py    # ✅ Done (Step 2: IK + GTSP + Collision)
│   └── 3_simulation.py             # ✅ Done (Step 3: Isaac Sim simulation)
├── data/
│   └── sample/
│       ├── mesh/
│       │   └── source.obj
│       ├── viewpoint/
│       │   └── 163/viewpoints.h5
│       └── trajectory/
│           └── 163/trajectory.csv
└── CLAUDE.md                       # This file
```

---

## Reference

### Configuration (new_common/config.py)

**Camera Specs:**
```python
CAMERA_FOV_WIDTH_MM = 41.0
CAMERA_FOV_HEIGHT_MM = 30.0
CAMERA_WORKING_DISTANCE_MM = 110.0
CAMERA_OVERLAP_RATIO = 0.5
```

**World Configuration (meters, Z-up):**
```python
TARGET_OBJECT_POSITION = [1.00, 0.0, -0.172]
TABLE_POSITION = [1.0, 0.0, -0.425]
WALL_POSITION = [-1.1, 0.0, 0.5]
WORKBENCH_POSITION = [0.35, -1.1, 0.5]
ROBOT_MOUNT_POSITION = [0.0, 0.0, -0.25]
```

**IK & Collision Parameters:**
```python
IK_NUM_SEEDS = 32
IK_ROTATION_THRESHOLD = 0.05  # radians
IK_POSITION_THRESHOLD = 0.005  # meters
COLLISION_ADAPTIVE_MAX_JOINT_STEP_DEG = 5.0
```

### Test Data
- **Sample mesh:** `data/sample/mesh/source.obj`
- **Materials:**
  - RGB(0, 255, 0) - Green (inspection target)
  - RGB(170, 163, 158) - Gray (base)

### Python Execution
In Isaac Sim environment, always use `/isaac-sim/python.sh`

---

## Design Philosophy

1. **Simplicity > Completeness**
   - 80% functionality for 80% results
   - Focus on main workflow over edge cases

2. **Education > Optimization**
   - Code readability first
   - Remove complex optimizations
   - Clear step-by-step progression

3. **Independence > Reusability**
   - Each script can run independently
   - Minimize inter-module dependencies
   - Copy and include necessary functions
