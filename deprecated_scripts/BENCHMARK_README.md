# IK Benchmark Scripts

This directory contains two IK benchmark scripts for comparing different IK solving approaches:

## Scripts Overview

### 1. `benchmark_ik.py` - Relaxed IK Benchmark
Original benchmark script that uses the relaxed_ik approach from the Manipulator-Coverage-Path-Planning project.

**Key Features:**
- Uses `Robot` class with `reach_with_relaxed_ik()` method
- Rust-based relaxed IK solver
- Supports relaxed constraints via tolerances parameter
- Joint velocity constraints optional

**Usage:**
```bash
python scripts/benchmark_ik.py [robot_name] [num_poses] [num_samples]

# Examples:
python scripts/benchmark_ik.py panda 100 10
python scripts/benchmark_ik.py ur5 50 20
```

**Arguments:**
- `robot_name`: Robot configuration name (default: 'panda')
- `num_poses`: Number of random poses to test (default: 100)
- `num_samples`: IK samples per pose (default: 10)

### 2. `benchmark_ik_curobo.py` - cuRobo IK Benchmark
New benchmark script that uses NVIDIA cuRobo's GPU-accelerated IK solver.

**Key Features:**
- Uses cuRobo `IKSolver` with GPU acceleration
- **Batch processing for maximum GPU utilization**
- CUDA graph optimization for faster inference
- Configurable collision checking
- Multiple random seeds for robustness
- Precise position/rotation thresholds

**Usage:**
```bash
python scripts/benchmark_ik_curobo.py [options]

# Examples:
# Basic benchmark with UR20
python scripts/benchmark_ik_curobo.py --robot ur20.yml --num_poses 100 --num_samples 10

# With collision checking enabled
python scripts/benchmark_ik_curobo.py --robot ur20.yml --collision_check --num_seeds 30

# Faster benchmark with fewer poses
python scripts/benchmark_ik_curobo.py --num_poses 50 --num_samples 5

# High precision settings
python scripts/benchmark_ik_curobo.py --position_threshold 0.001 --rotation_threshold 0.01

# Custom batch size for GPU optimization
python scripts/benchmark_ik_curobo.py --batch_size 64  # Larger batch for bigger GPUs
python scripts/benchmark_ik_curobo.py --batch_size 16  # Smaller batch for limited memory
```

**Command Line Arguments:**
- `--robot`: Robot configuration YAML (default: 'ur20.yml')
- `--num_poses`: Number of random poses (default: 100)
- `--num_samples`: IK samples per pose (default: 10)
- `--collision_check`: Enable collision checking (default: False)
- `--num_seeds`: Number of random seeds (default: 20)
- `--position_threshold`: Position error threshold in meters (default: 0.005)
- `--rotation_threshold`: Rotation error threshold in radians (default: 0.05)
- `--batch_size`: Batch size for GPU processing (default: 32)
- `--output_dir`: Output directory for results (default: current directory)

## Output Format

Both scripts generate JSON output files with the following structure:

```json
{
  "robot_name": "ur20",
  "num_poses": 100,
  "num_samples_per_pose": 10,
  "total_solves": 1000,
  "success_count": 950,
  "failure_count": 50,
  "success_rate": 0.95,
  "total_time": 12.345,
  "init_time": 2.1,
  "gen_time": 1.5,
  "mean_time": 0.0123,
  "median_time": 0.0115,
  "min_time": 0.008,
  "max_time": 0.045,
  "std_time": 0.0056,
  "throughput": 81.23,
  "solve_times": [0.012, 0.011, ...],

  // cuRobo-specific fields:
  "batch_size": 32,
  "collision_check": false,
  "num_seeds": 20,
  "position_threshold": 0.005,
  "rotation_threshold": 0.05
}
```

**Output File Naming:**
- Relaxed IK: `benchmark_results_{robot_name}_{timestamp}.json`
- cuRobo: `benchmark_curobo_{robot_name}_{timestamp}.json`

## Performance Metrics Explained

| Metric | Description |
|--------|-------------|
| `total_solves` | Total number of IK solve attempts |
| `success_count` | Number of successful IK solutions |
| `success_rate` | Percentage of successful solves |
| `mean_time` | Average time per IK solve (seconds) |
| `median_time` | Median time per IK solve (seconds) |
| `min_time` | Fastest IK solve time |
| `max_time` | Slowest IK solve time |
| `throughput` | IK solves per second |
| `init_time` | Solver initialization time |
| `gen_time` | Time to generate random poses |

## Key Differences

### Algorithm Approach

| Feature | Relaxed IK | cuRobo IK |
|---------|-----------|-----------|
| **Engine** | Rust-based optimization | GPU-accelerated numerical solver |
| **Constraints** | Relaxed (via tolerances) | Strict (configurable thresholds) |
| **Parallelization** | CPU multi-threading | GPU parallelization |
| **Collision Check** | Not included | Optional mesh/primitive collision |
| **Random Seeds** | Single seed per solve | Multiple seeds (configurable) |
| **Convergence** | Gradient descent | Multi-seed optimization |

### Performance Characteristics

**Relaxed IK:**
- ✅ Flexible constraint handling
- ✅ Good for approximate solutions
- ✅ Lower memory usage
- ❌ Slower for batch solving
- ❌ Single-threaded per solve

**cuRobo IK:**
- ✅ Very fast (GPU acceleration with batch processing)
- ✅ CUDA graph optimization
- ✅ **Batch processing for 10-50x speedup**
- ✅ Collision checking integrated
- ✅ Configurable batch size for memory management
- ❌ Higher memory usage (GPU)
- ❌ Requires CUDA-capable GPU

## Benchmark Comparison Example

To compare both solvers on the same robot:

```bash
# Run relaxed IK benchmark
python scripts/benchmark_ik.py panda 100 10

# Run cuRobo benchmark
python scripts/benchmark_ik_curobo.py --robot franka.yml --num_poses 100 --num_samples 10

# Compare results
# Expected: cuRobo should be 5-10x faster per solve
```

## Requirements

### Relaxed IK (`benchmark_ik.py`)
```
- Robot class from Manipulator-Coverage-Path-Planning
- relaxed_ik_core (Rust library)
- RobotKinematics
```

### cuRobo (`benchmark_ik_curobo.py`)
```
- NVIDIA cuRobo library
- PyTorch with CUDA support
- CUDA-capable GPU (recommended: RTX series or better)
- cuRobo robot configuration files
```

## Troubleshooting

### Common Issues

**1. "Failed to generate any valid poses"**
- Solution: Check robot workspace limits
- Try: Reduce pose generation range or increase `num_attempts`

**2. Low success rate (< 50%)**
- Relaxed IK: Increase tolerance values
- cuRobo: Increase `num_seeds` or relax thresholds

**3. cuRobo out of memory**
- Solution: Reduce `num_seeds` or `num_poses`
- Try: Disable collision checking

**4. Slow performance**
- cuRobo: Enable CUDA graphs (default: enabled)
- Both: Reduce `num_poses` or `num_samples_per_pose`

## Notes

- **Coordinate Systems**: Both scripts handle coordinate transformations internally
- **Random Seed**: Set `np.random.seed()` for reproducible benchmarks
- **Warm-up**: First few solves may be slower due to GPU initialization (cuRobo)
- **Batch Size**: cuRobo now uses true batch processing for maximum GPU utilization
  - Default batch size: 32 (good for most GPUs)
  - Increase for larger GPUs (e.g., 64, 128)
  - Decrease for limited GPU memory (e.g., 16, 8)
  - Larger batches = better GPU utilization but more memory usage

## Citation

If you use these benchmarks in research, please cite:

**cuRobo:**
```
@article{curobo2023,
  title={cuRobo: Parallelized Collision-Free Robot Motion Generation},
  author={Sundaresan, Balakumar and others},
  journal={NVIDIA Technical Report},
  year={2023}
}
```

**Relaxed IK:**
```
@inproceedings{rakita2018relax,
  title={RelaxedIK: Real-time Synthesis of Accurate and Feasible Robot Arm Motion},
  author={Rakita, Daniel and others},
  booktitle={RSS},
  year={2018}
}
```
