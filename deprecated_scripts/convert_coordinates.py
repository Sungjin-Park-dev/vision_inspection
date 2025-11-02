from typing import Optional, Tuple
import numpy as np

SAMPLED_LOCAL_POINTS: Optional[np.ndarray] = None
SAMPLED_LOCAL_NORMALS: Optional[np.ndarray] = None
PRESET_SAMPLE_INDICES = [0, 2500, 5000, 7500, 9999]

def load_opend3d():
    import open3d as o3d
    test_obj_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_o3d.obj"
    mesh = o3d.io.read_triangle_mesh(test_obj_path)
    mesh.compute_vertex_normals()

    print("Visualizing original mesh...")
    o3d.visualization.draw_geometries([mesh])

    print("Converting mesh to point cloud using Poisson sampling...")
    pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=20000)

    # Estimate normals for point cloud
    pcd_poisson.estimate_normals()
    # Set uniform color (gray) to prevent rainbow coloring
    pcd_poisson.paint_uniform_color([0.7, 0.7, 0.7])
    print(f"Point cloud has {len(pcd_poisson.points)} points and {len(pcd_poisson.normals)} normals")

    print("Visualizing Poisson sampled point cloud...")
    o3d.visualization.draw_geometries([pcd_poisson])

    points = np.asarray(pcd_poisson.points)
    normals = np.asarray(pcd_poisson.normals)

    valid_indices = [idx for idx in PRESET_SAMPLE_INDICES if idx < len(points)]

    if not valid_indices:
        valid_indices = list(range(min(len(points), 5)))

    sample_indices = np.asarray(valid_indices, dtype=np.int64)

    if sample_indices.size > 0:
        sampled_cloud = o3d.geometry.PointCloud()
        sampled_cloud.points = o3d.utility.Vector3dVector(points[sample_indices])
        sampled_cloud.paint_uniform_color([1.0, 0.0, 0.0])

        print("Visualizing preset point samples in Open3D alongside the full cloud...")
        o3d.visualization.draw_geometries([pcd_poisson, sampled_cloud])

        global SAMPLED_LOCAL_POINTS, SAMPLED_LOCAL_NORMALS
        SAMPLED_LOCAL_POINTS = points[sample_indices]
        SAMPLED_LOCAL_NORMALS = normals[sample_indices]

    return points, normals

if __name__ == "__main__":
    load_opend3d()