# Joint Reconfiguration Optimization: Relaxed IK vs cuRobo Comparison Guide

## Overview

This guide compares two IK solver approaches for optimizing joint trajectory reconfigurations:

1. **Relaxed IK** (`optimize_joint_reconfigurations_independent.py`) - Rust-based iterative optimization
2. **cuRobo** (`optimize_joint_reconfigurations_independent_curobo.py`) - GPU-accelerated batch IK solver

Both scripts implement the same **independent optimization strategy** with bidirectional validation, but use different IK solving engines.

---

## Key Differences

| Feature | Relaxed IK | cuRobo |
|---------|-----------|---------|
| **Engine** | Rust-based iterative solver | GPU-accelerated batch solver |
| **Constraints** | Tolerance-based (allows deviation) | Threshold-based (precision target) |
| **Sampling** | Sequential calls, randomized seeds | Batch processing, num_seeds parameter |
| **Speed** | CPU-bound, slower | GPU-accelerated, significantly faster |
| **Diversity** | max_iter controls convergence | num_seeds controls solution diversity |
| **Best for** | Flexible tolerance scenarios | Precision + speed requirements |

---

## Installation & Setup

### Relaxed IK Version
Requires the Rust-based relaxed_ik library:
```bash
# Already integrated in Manipulator-Coverage-Path-Planning/scripts/robot.py
# No additional setup needed
```

### cuRobo Version
Requires NVIDIA Isaac Sim environment:
```bash
# Use Isaac Sim's python environment
/isaac-sim/python.sh optimize_joint_reconfigurations_independent_curobo.py
```

---

## Usage Examples

### 1. Basic Optimization - Relaxed IK

```bash
/isaac-sim/python.sh optimize_joint_reconfigurations_independent.py \
    --input_csv motions/ur20_motion.csv \
    --robot ur20 \
    --threshold 1.0 \
    --num_ik_samples 50
```

**Parameters:**
- `--tolerances`: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z] - Allows deviation in meters/radians
- `--max_iter`: Maximum iterations for convergence (default: 100)
- `--num_ik_samples`: Number of sampling attempts per timestep (default: 50)

### 2. Basic Optimization - cuRobo

```bash
/isaac-sim/python.sh optimize_joint_reconfigurations_independent_curobo.py \
    --input_csv motions/ur20_motion.csv \
    --robot ur20.yml \
    --threshold 1.0 \
    --num_seeds 50
```

**Parameters:**
- `--position_threshold`: Position precision target in meters (default: 0.005)
- `--rotation_threshold`: Rotation precision target in radians (default: 0.05)
- `--num_seeds`: Number of diverse IK seeds to try (default: 20)
- `--num_ik_attempts`: Number of solve attempts per timestep (default: 10)

---

## Detailed Parameter Comparison

### Relaxed IK: Tolerance-Based Approach

```bash
--tolerances 0.05 0.05 0.01 0.0 0.0 0.0
# Allows:
# - ±5cm error in X, Y position
# - ±1cm error in Z position
# - Exact orientation match required
```

**When to use:**
- You need flexible constraints
- Surface coverage tasks where exact positioning is less critical
- Trading precision for smoother motion

### cuRobo: Threshold-Based Approach

```bash
--position_threshold 0.005  # 5mm precision target
--rotation_threshold 0.05   # ~2.86° precision target
```

**When to use:**
- You need guaranteed precision
- Fast computation is critical (GPU available)
- Batch processing multiple trajectories

---

## Optimization Strategy (Same for Both)

Both scripts follow this algorithm:

```
1. IDENTIFY problematic timesteps
   - Scan trajectory for joint changes > threshold
   - Record all problematic transitions

2. SAMPLE IK solutions independently
   - For each problematic timestep:
     * Use previous timestep as seed
     * Generate multiple candidate IK solutions
     * Use relaxed constraints (tolerances or thresholds)

3. BIDIRECTIONAL VALIDATION
   - For each candidate:
     * Calculate distance to PREVIOUS timestep
     * Calculate distance to NEXT timestep
     * Total cost = max(prev_distance, next_distance)

4. SELECT best solution
   - Choose candidate with minimum total cost
   - Only accept if better than original

5. NO CASCADING
   - Other timesteps remain completely unchanged
   - Independent optimization ensures stability
```

---

## Expected Performance

### Relaxed IK
- **Speed**: ~1-5 seconds per problematic timestep
- **Success Rate**: High with proper tolerances
- **Memory**: Low (CPU-based)
- **Scalability**: Linear with number of timesteps

### cuRobo
- **Speed**: ~0.1-0.5 seconds per problematic timestep (10-50x faster)
- **Success Rate**: High with appropriate thresholds
- **Memory**: GPU memory dependent
- **Scalability**: Excellent with batch processing

---

## Output Files

Both scripts generate identical output formats:

### 1. Optimized Trajectory CSV
```
{input_basename}_independent_optimized.csv          # Relaxed IK
{input_basename}_independent_optimized_curobo.csv   # cuRobo
```

Format: Same as input CSV with optimized joint values

### 2. Optimization Report
```
{input_basename}_independent_optimized.txt          # Relaxed IK
{input_basename}_independent_optimized_curobo.txt   # cuRobo
```

Contains:
- Summary statistics
- Problematic timesteps identified
- Optimization details (before/after for each timestep)
- Remaining reconfigurations

### 3. Visualization Plot (optional with `--plot`)
```
{input_basename}_independent_optimization_comparison.png          # Relaxed IK
{input_basename}_independent_optimization_comparison_curobo.png   # cuRobo
```

---

## Comparison Workflow

To compare both methods on the same trajectory:

```bash
# 1. Run Relaxed IK version
/isaac-sim/python.sh optimize_joint_reconfigurations_independent.py \
    --input_csv motions/ur20_motion.csv \
    --robot ur20 \
    --threshold 1.0 \
    --tolerances 0.05 0.05 0.01 0.0 0.0 0.0 \
    --num_ik_samples 50 \
    --plot

# 2. Run cuRobo version
/isaac-sim/python.sh optimize_joint_reconfigurations_independent_curobo.py \
    --input_csv motions/ur20_motion.csv \
    --robot ur20.yml \
    --threshold 1.0 \
    --position_threshold 0.05 \
    --rotation_threshold 0.05 \
    --num_seeds 50 \
    --num_ik_attempts 10 \
    --plot

# 3. Compare results
# - Check .txt files for statistics
# - Compare PNG plots visually
# - Compare execution time
# - Verify reconfiguration reduction
```

---

## Troubleshooting

### Relaxed IK: No improvements found
**Possible causes:**
- Tolerances too strict
- max_iter too low
- num_ik_samples too low

**Solutions:**
```bash
# Increase tolerances
--tolerances 0.1 0.1 0.05 0.1 0.1 0.1

# Increase iterations
--max_iter 200

# Increase sampling
--num_ik_samples 100
```

### cuRobo: No improvements found
**Possible causes:**
- Thresholds too strict
- num_seeds too low
- GPU memory issues

**Solutions:**
```bash
# Relax thresholds
--position_threshold 0.01
--rotation_threshold 0.1

# Increase seeds
--num_seeds 100

# Increase attempts
--num_ik_attempts 20
```

### cuRobo: CUDA out of memory
**Solutions:**
```bash
# Reduce batch size (modify source if needed)
# Or reduce num_seeds
--num_seeds 10
```

---

## Recommendations

### Use Relaxed IK when:
- You need flexible constraint specifications
- CPU-only environment
- Task allows tolerance in end-effector pose
- Researching different constraint combinations

### Use cuRobo when:
- GPU acceleration is available
- Speed is critical (batch processing, real-time)
- High precision required
- Processing multiple trajectories

### For Best Results:
1. Start with cuRobo for speed and precision
2. If not satisfactory, try Relaxed IK with custom tolerances
3. Compare both and choose based on:
   - Reconfiguration reduction
   - Joint movement smoothness
   - Execution time
   - Task requirements

---

## Parameter Tuning Guide

### Equivalence Mapping

To achieve similar behavior between methods:

| Relaxed IK | cuRobo | Notes |
|------------|--------|-------|
| `--tolerances 0.01 0.01 0.01 0.0 0.0 0.0` | `--position_threshold 0.01` | ±1cm position tolerance |
| `--tolerances 0.0 0.0 0.0 0.1 0.1 0.1` | `--rotation_threshold 0.1` | ±0.1 rad rotation |
| `--max_iter 100` | `--num_seeds 20` | Solution diversity |
| `--num_ik_samples 50` | `--num_ik_attempts 10` | Sampling attempts |

### Finding Optimal Parameters

```bash
# Start conservative (strict)
--position_threshold 0.005 / --tolerances 0.005 0.005 0.005 0.0 0.0 0.0

# If no improvements, gradually relax:
--position_threshold 0.01  / --tolerances 0.01 0.01 0.01 0.0 0.0 0.0
--position_threshold 0.02  / --tolerances 0.02 0.02 0.02 0.0 0.0 0.0

# Monitor reconfiguration reduction vs pose accuracy trade-off
```

---

## Example Results Format

### Console Output (both versions)

```
================================================================================
INDEPENDENT TRAJECTORY OPTIMIZATION (cuRobo/Relaxed IK)
================================================================================
Threshold: 1.000 radians
...

================================================================================
STEP 1: Identifying problematic timesteps in original trajectory
================================================================================
Found 15 problematic timesteps
Timesteps: [23, 45, 67, 89, 112, 134, 156, 178, 201, 223, 245, 267, 289, 312, 334]

================================================================================
STEP 2: Optimizing problematic timesteps independently
================================================================================

[1/15] Timestep 23: Original max change = 1.234 rad
  Sampling 10 IK solutions...
  Found 8 valid IK solutions
  ✓ Improved! Prev: 1.234→0.456, Next: 0.789→0.321, Total: 1.234→0.456 (Δ=0.778)

...

================================================================================
OPTIMIZATION SUMMARY
================================================================================
Originally problematic timesteps: 15
Reconfigurations fixed: 12
Reconfigurations unfixed: 3

================================================================================
BEFORE vs AFTER COMPARISON
================================================================================
Original reconfigurations: 15
Optimized reconfigurations: 3
Reduction: 12
Improvement: 80.0%

Total time: 45.234 seconds
```

---

## References

- **Relaxed IK**: [relaxed_ik_core repository](https://github.com/uwgraphics/relaxed_ik_core)
- **cuRobo**: [NVIDIA cuRobo documentation](https://curobo.org/)
- **Original paper**: ICRA'25 Hierarchical Coverage Path Planning

---

## Citation

If you use these optimization tools in your research, please cite:

```bibtex
@article{wang2025hierarchically,
  title={Hierarchically Accelerated Coverage Path Planning for Redundant Manipulators},
  author={Wang, Yeping and Gleicher, Michael},
  booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}
```
