#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FK + GTSP/TSP 최적화 모듈

Forward Kinematics와 Generalized TSP 해결을 위한 GPU 가속 모듈
- GPU 가속 (CuPy optional)
- Numba 병렬화
- 배치 처리
- 메모리 최적화
- 캐싱

Used by vision_inspection_pipeline.ipynb Section 3.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

# Import common modules
from common import config
from common.data_io import build_clusters_from_h5
from common.kinematics_utils import rot_to_quat_batch, fk_batch
from common.graph_utils import (
    build_neighbors_knn,
    build_neighbors_auto,
    build_visit_order_robot_cost,
    choose_ik_given_order,
    pair_cost_matrix_joint_mid
)
from common.cli_utils import print_section_header, print_key_value, print_success, print_warning, print_error

# Optional visualization imports
try:
    import open3d as o3d
    from common.visualization_utils import (
        load_trajectory_poses_from_csv,
        load_mesh_for_visualization,
        visualize_trajectory_with_mesh,
        OPEN3D_AVAILABLE
    )
except ImportError:
    OPEN3D_AVAILABLE = False



# =============================
# CSV 출력
# =============================
def export_to_csv(
    csv_path: str,
    order: List[int],
    picked: Dict[int, int],
    clusters: List[Dict],
):
    """
    joint_trajectory_dp_5000_repaired.csv 스타일로 내보내기

    Columns:
      time,
      ur20-shoulder_pan_joint,
      ur20-shoulder_lift_joint,
      ur20-elbow_joint,
      ur20-wrist_1_joint,
      ur20-wrist_2_joint,
      ur20-wrist_3_joint,
      target-POS_X, target-POS_Y, target-POS_Z,
      target-ROT_X, target-ROT_Y, target-ROT_Z, target-ROT_W
    """
    import csv, sys

    if not order:
        print("[WARN] Cannot write CSV: order is empty.", file=sys.stderr)
        return

    headers = [
        "time",
        "ur20-shoulder_pan_joint",
        "ur20-shoulder_lift_joint",
        "ur20-elbow_joint",
        "ur20-wrist_1_joint",
        "ur20-wrist_2_joint",
        "ur20-wrist_3_joint",
        "target-POS_X", "target-POS_Y", "target-POS_Z",
        "target-ROT_X", "target-ROT_Y", "target-ROT_Z", "target-ROT_W",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)

        for step, cidx in enumerate(order):
            if cidx not in picked:
                print(f"[WARN] Skipping row: target_idx {cidx} not in picked.", file=sys.stderr)
                continue

            cl = clusters[cidx]
            ik = picked[cidx]

            # 조인트 6개만 기록(UR20 기준), 부족하면 0으로 패딩
            q = cl["q"][ik]
            if q.shape[0] >= 6:
                q6 = q[:6]
            else:
                q6 = list(q) + [0.0] * (6 - len(q))

            # 타깃 위치/자세: world_pose 기반이 있으면 사용, 없으면 FK 결과로 대체
            # 위치
            pos = cl["target"] if "target" in cl else cl["p"][ik]
            # 자세(쿼터니언): (w,x,y,z) → CSV(X,Y,Z,W)
            Qwxyz = cl.get("target_Q", cl["Q"][ik])
            rot_x, rot_y, rot_z, rot_w = Qwxyz[1], Qwxyz[2], Qwxyz[3], Qwxyz[0]

            row = [
                float(step),                  # time: 스텝 번호를 그대로 사용(1.0 간격)
                float(q6[0]), float(q6[1]), float(q6[2]),
                float(q6[3]), float(q6[4]), float(q6[5]),
                float(pos[0]), float(pos[1]), float(pos[2]),
                float(rot_x), float(rot_y), float(rot_z), float(rot_w),
            ]
            w.writerow(row)


# =============================
# 메인
# =============================
def main():
    """Main entry point - performs same tasks as notebook Section 3"""
    parser = argparse.ArgumentParser(
        description="Generate GTSP/TSP trajectory from IK solutions"
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Object name for automatic path generation (e.g., 'glass', 'phone'). "
             "If provided with --num_viewpoints, paths will be auto-generated."
    )
    parser.add_argument(
        "--num_viewpoints",
        type=int,
        default=None,
        help="Number of viewpoints (used with --object_name for path generation)"
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Path to IK solutions HDF5 file (e.g., data/glass/ik/500/ik_solutions.h5). "
             "Required if --object_name is not provided."
    )
    parser.add_argument(
        "--lam_rot",
        type=float,
        default=1.0,
        help="Rotation cost weight for trajectory optimization (default: 1.0)"
    )
    parser.add_argument(
        "--tool_len",
        type=float,
        default=0.0,
        help="Tool length offset in meters (default: 0.0)"
    )
    parser.add_argument(
        "--allow_unsafe",
        action="store_true",
        help="Use all IK solutions instead of only collision-free ones (default: False)"
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=5,
        help="Number of nearest neighbors for graph construction (default: 5)"
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="Path to save trajectory CSV file (default: auto-generate in data/{object_name}/trajectory/)"
    )

    # Visualization options
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize trajectory with mesh after generation (requires Open3D)"
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to mesh file for visualization (default: auto-detect from object_name)"
    )
    parser.add_argument(
        "--no-transform",
        action="store_true",
        help="Skip world transformation for visualization (show mesh at origin)"
    )
    parser.add_argument(
        "--show-frames",
        action="store_true",
        help="Show coordinate frames at each waypoint (default: show points)"
    )
    parser.add_argument(
        "--frame-size",
        type=float,
        default=0.02,
        help="Size of coordinate frames in meters (default: 0.02)"
    )
    parser.add_argument(
        "--no-path",
        action="store_true",
        help="Don't show trajectory path lines in visualization"
    )
    parser.add_argument(
        "--no-normals",
        action="store_true",
        help="Don't show viewing direction arrows in visualization"
    )
    parser.add_argument(
        "--normal-length",
        type=float,
        default=0.05,
        help="Length of viewing direction arrows in meters (default: 0.05)"
    )

    args = parser.parse_args()

    # Validate and resolve paths
    if args.object_name is None and args.h5 is None:
        parser.error("Either --object_name (with --num_viewpoints) or --h5 must be provided")

    if args.object_name and args.num_viewpoints is None:
        parser.error("--num_viewpoints is required when using --object_name")

    # Determine IK solutions path
    if args.object_name:
        if args.h5 is None:
            args.h5 = str(config.get_ik_path(args.object_name, args.num_viewpoints))
            print(f"Using auto-generated IK solutions path: {args.h5}")

    # Auto-generate output path if not provided
    csv_path = args.csv_out
    if csv_path is None:
        if args.object_name:
            # New structure: data/{object_name}/trajectory/{num_viewpoints}/gtsp.csv
            csv_path = str(config.get_trajectory_path(args.object_name, args.num_viewpoints, "gtsp.csv"))
        else:
            # Fallback to old structure for backward compatibility
            h5_dir = os.path.dirname(args.h5)
            dataset_name = os.path.basename(h5_dir)
            csv_path = f"data/trajectory/{dataset_name}/gtsp.csv"

    print(f"\n{'='*60}")
    print("GTSP TRAJECTORY GENERATION (Section 3)")
    print(f"{'='*60}")
    print(f"IK solutions: {args.h5}")
    print(f"Output CSV: {csv_path}")
    print(f"Lambda rotation: {args.lam_rot}")
    print(f"Tool length: {args.tool_len} m")
    print(f"Use safe only: {not args.allow_unsafe}")
    print(f"k-NN neighbors: {args.knn}")
    print(f"{'='*60}\n")

    # Step 1: Load IK solutions and build clusters
    print("Step 1: Loading IK solutions and building clusters...")
    clusters, target_coords, nonempty_map = build_clusters_from_h5(
        args.h5,
        use_safe_only=(not args.allow_unsafe),
        tool_z=args.tool_len
    )
    print(f"  ✓ Loaded {len(clusters)} viewpoints with IK solutions")

    if len(clusters) == 0:
        print("[ERROR] No valid clusters found. Cannot generate trajectory.")
        sys.exit(1)

    # Step 2: Build k-NN neighbor graph
    print(f"\nStep 2: Building k-NN neighbor graph (k={args.knn})...")
    nbrs = build_neighbors_knn(target_coords, k=args.knn)
    print(f"  ✓ Built neighbor graph")

    # Step 3: Determine visit order using robot cost
    print("\nStep 3: Determining visit order with robot cost...")
    order = build_visit_order_robot_cost(
        clusters, nbrs, lam_rot=args.lam_rot, tool_z=args.tool_len
    )
    print(f"  ✓ Computed visit order for {len(order)} viewpoints")

    # Step 4: Optimize IK selection using DP
    print("\nStep 4: Optimizing IK selection with Dynamic Programming...")
    picked, total_cost = choose_ik_given_order(
        clusters, order, lam_rot=args.lam_rot, tool_z=args.tool_len
    )
    print(f"  ✓ Optimized IK selection (total cost: {total_cost:.4f})")

    # Step 5: Export to CSV
    print(f"\nStep 5: Exporting trajectory to CSV...")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    export_to_csv(csv_path, order, picked, clusters)

    print(f"\n{'='*60}")
    print("TRAJECTORY SAVED")
    print(f"{'='*60}")
    print(f"Output path: {csv_path}")
    print(f"Total waypoints: {len(order)}")
    print(f"Total cost: {total_cost:.4f}")
    file_size_kb = os.path.getsize(csv_path) / 1024 if os.path.exists(csv_path) else 0
    print(f"File size: {file_size_kb:.2f} KB")
    print(f"{'='*60}\n")

    # Step 6: Visualization (optional)
    if args.visualize:
        if not OPEN3D_AVAILABLE:
            print_warning("Open3D not available - skipping visualization")
            print("  Install Open3D: pip install open3d\n")
        else:
            print(f"Step 6: Visualizing trajectory...")
            print_section_header("TRAJECTORY VISUALIZATION", width=60)

            # Resolve mesh path
            mesh_path = args.mesh
            if mesh_path is None:
                if args.object_name:
                    # Try target.obj first, then source.obj
                    for mesh_name in ["target.obj", "source.obj"]:
                        candidate = config.get_mesh_path(args.object_name, mesh_name)
                        if candidate.exists():
                            mesh_path = str(candidate)
                            break

                    if mesh_path is None:
                        print_error("Cannot auto-locate mesh file")
                        print(f"  Tried: {config.get_mesh_path(args.object_name, 'target.obj')}")
                        print(f"  Tried: {config.get_mesh_path(args.object_name, 'source.obj')}")
                        print("  Use --mesh to specify path explicitly")
                        sys.exit(1)
                else:
                    print_error("--mesh required for visualization when --object_name not provided")
                    sys.exit(1)

            print_key_value("Mesh file", mesh_path)
            print_key_value("CSV file", csv_path)
            print()

            # Load trajectory
            print("Loading trajectory poses...")
            positions, rotations = load_trajectory_poses_from_csv(csv_path)
            print_key_value("Loaded waypoints", len(positions))

            # Load mesh
            print("\nLoading mesh...")
            mesh = load_mesh_for_visualization(
                mesh_path,
                apply_world_transform=not args.no_transform,
                glass_position=config.GLASS_POSITION if not args.no_transform else None,
                glass_rotation=config.GLASS_ROTATION if not args.no_transform else None
            )
            print_key_value("Vertices", len(mesh.vertices))
            print_key_value("Triangles", len(mesh.triangles))

            # Visualize
            print("\nOpening visualization window...")
            print("  Legend:")
            print("    - Gray mesh: Object")
            print("    - Origin frame (large): World coordinate system")
            if args.show_frames:
                print(f"    - Small frames: Target poses (X=red, Y=green, Z=blue)")
            else:
                print("    - Green points: Target positions")
            if not args.no_normals:
                print("    - Red arrows: Camera viewing directions")
            if not args.no_path:
                print("    - Blue lines: Trajectory path")
            print("\n  Controls: Mouse to rotate/zoom, 'Q' to exit\n")

            visualize_trajectory_with_mesh(
                mesh=mesh,
                positions=positions,
                rotations=rotations,
                show_frames=args.show_frames,
                frame_size=args.frame_size,
                show_path=not args.no_path,
                show_normals=not args.no_normals,
                normal_length=args.normal_length,
            )

            print_success("Visualization complete")
            print()

    print("✓ Section 3 완료!")


if __name__ == "__main__":
    main()
