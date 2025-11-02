#!/bin/bash
# Comparison script to run both IK benchmarks and compare results

echo "=========================================="
echo "IK Benchmark Comparison Script"
echo "=========================================="
echo ""

# Configuration
ROBOT="ur20.yml"
NUM_POSES=50
NUM_SAMPLES=10
OUTPUT_DIR="./benchmark_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Configuration:"
echo "  Robot: $ROBOT"
echo "  Number of poses: $NUM_POSES"
echo "  Samples per pose: $NUM_SAMPLES"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run cuRobo benchmark (without collision checking)
echo "=========================================="
echo "Running cuRobo IK Benchmark (no collision)"
echo "=========================================="
python scripts/benchmark_ik_curobo.py \
    --robot "$ROBOT" \
    --num_poses "$NUM_POSES" \
    --num_samples "$NUM_SAMPLES" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Running cuRobo IK Benchmark (with collision)"
echo "=========================================="
python scripts/benchmark_ik_curobo.py \
    --robot "$ROBOT" \
    --num_poses "$NUM_POSES" \
    --num_samples "$NUM_SAMPLES" \
    --collision_check \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To compare results, check the JSON files in $OUTPUT_DIR"
echo ""
echo "Key metrics to compare:"
echo "  - mean_time: Average solve time per IK"
echo "  - success_rate: Percentage of successful solves"
echo "  - throughput: IK solves per second"
echo ""
