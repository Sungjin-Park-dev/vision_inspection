# Vision Inspection Pipeline - Simplified Scripts

Educational implementation of a 3-stage robotic vision inspection pipeline.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Viewpoint Generation                                      │
│  Input:  Multi-material mesh (OBJ + MTL)                           │
│  Output: Surface viewpoints (positions + normals)                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Trajectory Planning                                       │
│  - IK computation (EAIK analytical solver)                          │
│  - Visit order optimization (GTSP + Dynamic Programming)            │
│  - Collision checking with adaptive interpolation                   │
│  - Replanning (MotionGen)                                           │
│  Output: Collision-free joint trajectory                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Simulation                                                │
│  - Execute trajectory in Isaac Sim                                  │
│  - Direct waypoint execution (no interpolation)                     │
│  - Optional collision sphere visualization                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Isaac Sim environment with CuRobo installed
- Input mesh in `data/{object}/mesh/source.obj` (multi-material OBJ + MTL)

### Complete Pipeline Execution

```bash
# Step 1: Generate viewpoints (auto-calculates required count)
/isaac-sim/python.sh new_scripts/1_create_viewpoint.py \
    --object sample \
    --material-rgb "0,255,0"

# Example output: data/sample/viewpoint/163/viewpoints.h5

# Step 2: Generate collision-free trajectory
/isaac-sim/python.sh new_scripts/2_generate_trajectory.py \
    --object sample \
    --num_viewpoints 163

# Example output: data/sample/trajectory/163/trajectory.csv

# Step 3: Simulate in Isaac Sim
/isaac-sim/python.sh new_scripts/3_simulation.py \
    --object sample \
    --num_viewpoints 163 \
    --visualize_spheres
```

## Script Details

### 1_create_viewpoint.py

**Purpose**: Extract inspection surface from multi-material mesh and generate optimal viewpoints.

**Key Features**:
- Material selection by RGB color matching
- Automatic viewpoint count calculation based on:
  - Surface area
  - Camera FOV (41mm × 30mm)
  - Working distance (110mm)
  - Overlap ratio (50%)
- Uniform Poisson disk sampling

**Arguments**:
```bash
--object         # Object name (required)
--material-rgb   # Target material RGB "R,G,B" (required)
--visualize      # Show Open3D visualization (optional)
```

**Auto-resolved Paths**:
- Input: `data/{object}/mesh/source.obj`
- Output: `data/{object}/viewpoint/{num}/viewpoints.h5`

---

### 2_generate_trajectory.py

**Purpose**: Compute collision-free robot trajectory visiting all viewpoints optimally.

**Pipeline**:
1. **IK Computation**: Analytical IK solver (EAIK) with 32 seeds
2. **Collision Filtering**: Remove infeasible IK solutions
3. **GTSP Optimization**: k-NN graph + Dynamic Programming
4. **Interpolation**: Adaptive interpolation (max 1° per step)
5. **Collision Checking**: Batch GPU collision detection
6. **Replanning**: MotionGen for colliding segments (max 5 attempts, 10s timeout)

**Key Features**:
- In-memory processing (no intermediate files)
- Replanning with success/failure tracking
- Detailed statistics output

**Arguments**:
```bash
--object          # Object name (required)
--num_viewpoints  # Number of viewpoints (required)
--knn             # k-NN neighbors (default: 5)
--lambda-rot      # Rotation cost weight (default: 1.0)
--visualize       # Show trajectory visualization (optional)
```

**Auto-resolved Paths**:
- Input: `data/{object}/viewpoint/{num}/viewpoints.h5`
- Output: `data/{object}/trajectory/{num}/trajectory.csv`
- Mesh: `data/{object}/mesh/source.obj` (for collision)

**Output Statistics**:
```
Waypoints: 146
Interpolated: 1484
Collisions found: 4
Replanning: 3/4 succeeded (75.0%)
⚠ Failed segments: [23]
  (Waypoint pairs: [(23, 24)])
```

---

### 3_simulation.py

**Purpose**: Execute trajectory in Isaac Sim with visual feedback.

**Key Features**:
- Direct waypoint execution (trajectory pre-interpolated in Stage 2)
- Auto-mesh loading for visual context
- Optional collision sphere visualization
- Optional debug mode (visualize target waypoints)

**Arguments**:
```bash
--object            # Object name (required)
--num_viewpoints    # Number of viewpoints (required)
--visualize_spheres # Show robot collision spheres (optional)
--debug             # Show target waypoint positions (optional)
--headless          # Run headless: native or websocket (optional)
```

**Auto-resolved Paths**:
- Trajectory: `data/{object}/trajectory/{num}/trajectory.csv`
- Mesh: `data/{object}/mesh/source.obj`

## Directory Structure

```
data/{object}/
├── mesh/
│   ├── source.obj        # Multi-material mesh (input for Stage 1 & 2)
│   ├── source.mtl        # Material definitions
│   └── target.ply        # Preprocessed inspection surface (optional)
├── viewpoint/
│   └── {num}/
│       └── viewpoints.h5 # Surface positions + normals (Stage 1 output)
└── trajectory/
    └── {num}/
        └── trajectory.csv # Collision-free trajectory (Stage 2 output)
```

## Configuration

All parameters are centralized in `new_common/config.py`:

**Camera Specifications**:
- FOV: 41mm × 30mm
- Working distance: 110mm
- Overlap ratio: 50%

**IK Parameters**:
- Seeds: 32
- Position threshold: 0.005m
- Rotation threshold: 0.05 rad

**Collision Parameters**:
- Max joint step: 1.0°
- Exclude last joint: True

**Replanning Parameters**:
- Enabled: True
- Max attempts: 5
- Timeout: 10.0s

## Design Principles

1. **Self-contained**: Each script includes all necessary functions
2. **Auto-path**: Consistent directory structure enforced
3. **Educational**: Clear code structure over optimization
4. **Minimal CLI**: Simple, consistent interface
5. **No common/ dependencies**: Only uses `new_common/config.py`

## Example: Complete Workflow

```bash
# 1. Generate viewpoints for "glass" object (green surface)
/isaac-sim/python.sh new_scripts/1_create_viewpoint.py \
    --object glass \
    --material-rgb "0,255,0"
# → Output: data/glass/viewpoint/245/viewpoints.h5 (245 viewpoints calculated)

# 2. Generate trajectory
/isaac-sim/python.sh new_scripts/2_generate_trajectory.py \
    --object glass \
    --num_viewpoints 245 \
    --knn 5
# → Output: data/glass/trajectory/245/trajectory.csv

# 3. Simulate
/isaac-sim/python.sh new_scripts/3_simulation.py \
    --object glass \
    --num_viewpoints 245 \
    --visualize_spheres
# → Isaac Sim visualization
```

## Troubleshooting

### "MTL file not found"
- Ensure `source.obj` uses multi-material format with `source.mtl`
- Stage 1 requires MTL for material selection

### "Replanning: 0/N succeeded"
- Check collision world configuration in `new_common/config.py`
- Increase `REPLAN_TIMEOUT` or `REPLAN_MAX_ATTEMPTS`
- Verify robot joint limits

### "Input mesh not found"
- Verify directory structure: `data/{object}/mesh/source.obj`
- Check file permissions

## Performance

Typical execution times (163 viewpoints, UR20 robot):
- Stage 1: ~5s (viewpoint sampling)
- Stage 2: ~2.4s (IK + GTSP + collision + replanning)
- Stage 3: Real-time simulation

## Reference

For detailed implementation notes, see `../CLAUDE.md`.
