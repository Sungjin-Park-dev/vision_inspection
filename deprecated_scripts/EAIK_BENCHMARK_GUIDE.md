# EAIK IK Benchmark Guide

## Overview

`benchmark_ik_eaik.py` provides benchmarking for EAIK (Efficient Analytical IK) solver, which computes all analytical IK solutions for a given pose.

## Key Features

✅ **Analytical Solutions**: Closed-form solution (no iteration)
✅ **All Solutions**: Returns all possible IK configurations
✅ **Deterministic**: Same input always produces same output
✅ **Batch Processing**: Process multiple poses efficiently
✅ **Optional Collision Checking**: Uses cuRobo for collision detection
✅ **Coordinate Transformation**: Handles CuRobo ↔ EAIK frame conversion

## Usage

### Basic Benchmark
```bash
python scripts/benchmark_ik_eaik.py --robot ur20.yml --num_poses 100 --num_samples 10
```

### With Collision Checking
```bash
python scripts/benchmark_ik_eaik.py --robot ur20.yml --collision_check
```

### Custom URDF
```bash
python scripts/benchmark_ik_eaik.py \
    --robot ur20.yml \
    --urdf /path/to/custom_robot.urdf \
    --num_poses 50
```

### Faster Benchmark
```bash
python scripts/benchmark_ik_eaik.py --num_poses 50 --num_samples 5
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--robot` | str | `ur20.yml` | Robot configuration YAML file |
| `--urdf` | str | `/isaac-sim/curobo/examples/lg_vision/simulation/helpers/ur20.urdf` | Path to URDF file for EAIK |
| `--num_poses` | int | `100` | Number of random poses to test |
| `--num_samples` | int | `10` | IK samples per pose |
| `--collision_check` | flag | `False` | Enable collision checking |
| `--batch_size` | int | `32` | Batch size for processing |
| `--output_dir` | str | `.` | Output directory for results |

## How It Works

### 1. EAIK Solver Initialization
```python
from eaik.IK_URDF import UrdfRobot

eaik_bot = UrdfRobot(urdf_path)
```

### 2. Pose Generation
- Uses cuRobo FK to generate reachable poses
- Ensures poses are within robot workspace

### 3. Coordinate Transformation
```python
# CuRobo → EAIK tool frame transformation
CUROBO_TO_EAIK_TOOL = np.array([
    [-1.0, 0.0, 0.0, 0.0],
    [ 0.0, 0.0, 1.0, 0.0],
    [ 0.0, 1.0, 0.0, 0.0],
    [ 0.0, 0.0, 0.0, 1.0],
])

pose_eaik = pose_curobo @ CUROBO_TO_EAIK_TOOL
```

### 4. Batch IK Solving
```python
# Solve IK for entire batch
eaik_results = eaik_bot.IK_batched(pose_matrices)

# Extract all analytical solutions
for result in eaik_results:
    solutions = result.Q  # All IK solutions for this pose
```

### 5. Collision Checking (Optional)
```python
# Use cuRobo collision checker
collision_flags = check_collision_batch(ik_solver, joint_configs)
```

## Output Format

```json
{
  "robot_name": "ur20",
  "urdf_path": "/path/to/ur20.urdf",
  "num_poses": 100,
  "num_samples": 10,
  "total_solves": 1000,
  "success_count": 950,
  "failure_count": 50,
  "collision_free_count": 920,
  "success_rate": 0.95,
  "collision_free_rate": 0.92,
  "total_time": 5.234,
  "init_time": 0.5,
  "gen_time": 1.2,
  "collision_time": 2.1,
  "mean_time": 0.00052,
  "median_time": 0.00048,
  "min_time": 0.00035,
  "max_time": 0.00089,
  "throughput": 191.06,
  "solver": "EAIK",
  "batch_size": 32,
  "collision_check": true
}
```

## Performance Comparison

### EAIK vs cuRobo vs Relaxed IK

| Metric | EAIK | cuRobo | Relaxed IK |
|--------|------|--------|------------|
| **Method** | Analytical | Numerical | Optimization |
| **Per-solve time** | 0.0001-0.001s | 0.005-0.015s | 0.05-0.15s |
| **Solutions** | All analytical | Single best | Single best |
| **Deterministic** | Yes | No (seeds) | No (initial) |
| **GPU Required** | No | Yes | No |
| **Collision Check** | Via cuRobo | Integrated | Not included |
| **Best Use Case** | Single pose | Batch poses | Relaxed constraints |

### Throughput Comparison (approximate)

**Single Pose:**
- EAIK: ~10,000 solves/sec (CPU)
- cuRobo: ~200 solves/sec (GPU, single)
- Relaxed IK: ~20 solves/sec (CPU)

**Batch Processing (100 poses):**
- EAIK: ~5,000 solves/sec (CPU batch)
- cuRobo: ~1,500 solves/sec (GPU batch, batch_size=32)
- Relaxed IK: ~20 solves/sec (sequential)

## When to Use EAIK

### ✅ Use EAIK When:
- Need all possible IK solutions
- Want deterministic results
- CPU-only environment
- Fast single-pose IK needed
- Exact analytical solutions required

### ❌ Avoid EAIK When:
- Robot doesn't have analytical IK (>6 DOF, complex kinematics)
- Need GPU-accelerated batch processing
- Require integrated collision checking
- Working with redundant manipulators

## EAIK Analytical Solutions

### UR20 Example
For a single target pose, EAIK typically returns **8 analytical solutions**:
- 2 shoulder configurations (left/right)
- 2 elbow configurations (up/down)
- 2 wrist configurations

```python
result = eaik_bot.IK(target_pose)
solutions = result.Q  # e.g., 8 solutions

# Example solutions:
# Solution 1: [0.1, -1.5, 1.6, -1.4, -1.6, 0.2]
# Solution 2: [0.1, -1.5, 1.6, -1.4, -1.6, 3.3]  # wrist flip
# Solution 3: [0.1, -0.8, 0.9, -1.4, -1.6, 0.2]  # elbow up
# ... (8 total)
```

## Collision Checking Integration

EAIK uses cuRobo for collision checking:

```python
# 1. EAIK computes all analytical solutions (fast, CPU)
eaik_results = eaik_bot.IK_batched(poses)

# 2. cuRobo checks collisions (GPU-accelerated)
collision_flags = collision_checker.check_constraints(joint_state)

# 3. Filter collision-free solutions
safe_solutions = [sol for sol, flag in zip(solutions, collision_flags) if flag]
```

## Coordinate System Notes

### CuRobo Frame
- Origin: Robot base
- Tool frame: Standard DH convention

### EAIK Frame
- Different tool frame orientation
- Requires transformation matrix

### Transformation
```python
# Apply before EAIK solve
pose_eaik = pose_curobo @ CUROBO_TO_EAIK_TOOL

# Joint solutions are in same joint space (no transformation needed)
```

## Example Workflow

```bash
# Step 1: Run EAIK benchmark
python scripts/benchmark_ik_eaik.py \
    --robot ur20.yml \
    --num_poses 100 \
    --num_samples 10 \
    --collision_check

# Step 2: Compare with cuRobo
python scripts/benchmark_ik_curobo.py \
    --robot ur20.yml \
    --num_poses 100 \
    --num_samples 10 \
    --collision_check

# Step 3: Analyze results
ls -lh benchmark_*.json

# Expected output:
# benchmark_eaik_ur20_20250124_120000.json
# benchmark_curobo_ur20_20250124_120100.json
```

## Troubleshooting

### Issue: No solutions found
**Cause:** Pose outside robot workspace
**Solution:** Check joint limits and workspace bounds

### Issue: Low success rate
**Cause:** Singular configurations
**Solution:** Use collision checking to filter invalid solutions

### Issue: Collision checking slow
**Cause:** Large number of solutions per pose
**Solution:** Reduce `--num_samples` or disable collision checking

### Issue: URDF not found
**Cause:** Incorrect URDF path
**Solution:** Specify correct path with `--urdf`

## Performance Tips

### 1. Maximize Speed
```bash
# Disable collision checking for fastest benchmark
python scripts/benchmark_ik_eaik.py --num_poses 1000 --num_samples 20
```

### 2. Quality Focus
```bash
# Enable collision checking for valid solutions
python scripts/benchmark_ik_eaik.py --collision_check --num_samples 50
```

### 3. Large Scale
```bash
# Increase batch size for better throughput
python scripts/benchmark_ik_eaik.py --batch_size 64 --num_poses 500
```

## Advantages of EAIK

1. **Speed**: Analytical solutions are instant (no iteration)
2. **Completeness**: Returns all possible configurations
3. **Determinism**: Reproducible results every time
4. **No Tuning**: No hyperparameters or convergence criteria
5. **CPU Efficiency**: Works well without GPU

## Limitations of EAIK

1. **6-DOF Only**: Limited to robots with analytical IK
2. **No Built-in Collision**: Requires external collision checker
3. **Frame Transformation**: Requires coordinate conversion
4. **Fixed Solutions**: Cannot optimize for specific criteria (e.g., minimum joint movement)

## Comparison Summary

| Feature | EAIK | cuRobo | Winner |
|---------|------|--------|--------|
| Speed (single) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | EAIK |
| Speed (batch) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | cuRobo |
| All solutions | ✅ Yes | ❌ No | EAIK |
| Collision check | ⚠️ External | ✅ Built-in | cuRobo |
| GPU required | ❌ No | ✅ Yes | EAIK |
| Deterministic | ✅ Yes | ❌ No | EAIK |
| Flexibility | ⭐⭐ | ⭐⭐⭐⭐⭐ | cuRobo |

## Conclusion

**Use EAIK for:**
- Fast analytical IK solutions
- Exploring all possible configurations
- CPU-only environments
- Deterministic results

**Use cuRobo for:**
- Large-scale batch processing
- GPU-accelerated solving
- Integrated collision checking
- Complex optimization criteria

Both solvers complement each other - use EAIK for fast analytical solutions, then cuRobo for collision checking and optimization!
