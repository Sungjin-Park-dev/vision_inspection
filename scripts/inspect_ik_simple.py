#!/usr/bin/env python3
"""
Simple IK Solutions Inspector

Inspect IK solutions HDF5 files created by compute_ik_solutions.py
"""

import argparse
import h5py
import numpy as np
import os


def inspect_ik_solutions(h5_path: str):
    """Inspect IK solutions file"""

    if not os.path.exists(h5_path):
        print(f"Error: File not found: {h5_path}")
        return

    file_size_kb = os.path.getsize(h5_path) / 1024

    print(f"\n{'='*60}")
    print(f"IK SOLUTIONS INSPECTION")
    print(f"{'='*60}")
    print(f"File: {h5_path}")
    print(f"Size: {file_size_kb:.2f} KB")
    print(f"{'='*60}\n")

    with h5py.File(h5_path, 'r') as f:
        # Print metadata
        print("METADATA:")
        meta = f['metadata']
        for key, val in meta.attrs.items():
            print(f"  {key}: {val}")

        # Count viewpoints
        num_viewpoints = meta.attrs['num_viewpoints']
        num_with_solutions = meta.attrs['num_viewpoints_with_solutions']
        num_with_safe = meta.attrs['num_viewpoints_with_safe_solutions']

        print(f"\n{'='*60}")
        print("STATISTICS:")
        print(f"  Total viewpoints: {num_viewpoints}")
        print(f"  With IK solutions: {num_with_solutions} ({num_with_solutions/num_viewpoints*100:.1f}%)")
        print(f"  With safe solutions: {num_with_safe} ({num_with_safe/num_viewpoints*100:.1f}%)")

        # Analyze solution counts
        all_solution_counts = []
        safe_solution_counts = []

        for i in range(num_viewpoints):
            vp_name = f'viewpoint_{i:04d}'
            if vp_name in f:
                vp = f[vp_name]
                all_solution_counts.append(vp.attrs['num_all_solutions'])
                safe_solution_counts.append(vp.attrs['num_safe_solutions'])

        all_counts = np.array(all_solution_counts)
        safe_counts = np.array(safe_solution_counts)

        print(f"\nIK SOLUTIONS PER VIEWPOINT:")
        print(f"  All solutions:")
        print(f"    Mean: {all_counts.mean():.2f}")
        print(f"    Min: {all_counts.min()}")
        print(f"    Max: {all_counts.max()}")
        print(f"    Median: {np.median(all_counts):.0f}")

        print(f"  Safe solutions:")
        print(f"    Mean: {safe_counts.mean():.2f}")
        print(f"    Min: {safe_counts.min()}")
        print(f"    Max: {safe_counts.max()}")
        print(f"    Median: {np.median(safe_counts):.0f}")

        # Distribution
        print(f"\nSAFE SOLUTIONS DISTRIBUTION:")
        for count in range(0, min(11, safe_counts.max() + 1)):
            num_vp = np.sum(safe_counts == count)
            if num_vp > 0:
                bar = '#' * int(num_vp / num_viewpoints * 50)
                print(f"  {count:2d} solutions: {num_vp:4d} viewpoints {bar}")

        # Sample viewpoint
        print(f"\n{'='*60}")
        print("SAMPLE VIEWPOINT (viewpoint_0000):")
        vp0 = f['viewpoint_0000']
        print(f"  Original index: {vp0.attrs['original_index']}")
        print(f"  Total IK solutions: {vp0.attrs['num_all_solutions']}")
        print(f"  Safe IK solutions: {vp0.attrs['num_safe_solutions']}")
        print(f"  World pose shape: {vp0['world_pose'].shape}")
        print(f"  All IK solutions shape: {vp0['all_ik_solutions'].shape}")
        print(f"  Collision-free mask: {vp0['collision_free_mask'][...]}")

        if vp0.attrs['num_safe_solutions'] > 0:
            all_sols = vp0['all_ik_solutions'][...]
            mask = vp0['collision_free_mask'][...]
            safe_sols = all_sols[mask]
            print(f"\n  First safe solution (joints in radians):")
            print(f"    {safe_sols[0]}")

        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect IK solutions HDF5 file")
    parser.add_argument(
        "--h5_file",
        type=str,
        required=True,
        help="Path to IK solutions HDF5 file"
    )
    args = parser.parse_args()

    inspect_ik_solutions(args.h5_file)


if __name__ == "__main__":
    main()
