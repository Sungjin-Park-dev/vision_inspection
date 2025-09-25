import open3d as o3d
import numpy as np

test_obj_path = "/isaac-sim/curobo/vision_inspection/data/input/glass_o3d.obj"

bunny = o3d.data.BunnyMesh()
mesh = o3d.io.read_triangle_mesh(test_obj_path)
mesh.compute_vertex_normals()

print("Visualizing original mesh...")
o3d.visualization.draw_geometries([mesh])

# Uniform sampling
# print("Converting mesh to point cloud using uniform sampling...")
# pcd_uniform = mesh.sample_points_uniformly(number_of_points=20000)

# # Estimate normals for point cloud
# pcd_uniform.estimate_normals()
# # Set uniform color (gray) to prevent rainbow coloring
# pcd_uniform.paint_uniform_color([0.7, 0.7, 0.7])
# print(f"Point cloud has {len(pcd_uniform.points)} points and {len(pcd_uniform.normals)} normals")

# print("Visualizing uniform sampled point cloud...")
# o3d.visualization.draw_geometries([pcd_uniform])

# Poisson sampling
print("Converting mesh to point cloud using Poisson sampling...")
pcd_poisson = mesh.sample_points_poisson_disk(number_of_points=20000)

# Estimate normals for point cloud
pcd_poisson.estimate_normals()
# Set uniform color (gray) to prevent rainbow coloring
pcd_poisson.paint_uniform_color([0.7, 0.7, 0.7])
print(f"Point cloud has {len(pcd_poisson.points)} points and {len(pcd_poisson.normals)} normals")

print("Visualizing Poisson sampled point cloud...")
o3d.visualization.draw_geometries([pcd_poisson])

# Visualize normals as arrows
def visualize_normals_as_arrows(pcd, normal_length=0.1, subsample_ratio=0.1):
    """Visualize point cloud normals as arrows"""
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # Subsample for better visualization
    num_points = len(points)
    indices = np.random.choice(num_points, int(num_points * subsample_ratio), replace=False)

    # Create arrow geometries
    arrows = []
    for i in indices:
        # Create arrow from point to point + normal
        start = points[i]
        end = start + normals[i] * normal_length

        # Create line for arrow
        line_points = [start, end]
        line_indices = [[0, 1]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red arrows
        arrows.append(line_set)

    return arrows

# Visualize point cloud with normals as arrows
# print("Visualizing uniform sampled point cloud with normal arrows...")
# arrows = visualize_normals_as_arrows(pcd_uniform)
# o3d.visualization.draw_geometries([pcd_uniform] + arrows)

print("Visualizing Poisson sampled point cloud with normal arrows...")
arrows = visualize_normals_as_arrows(pcd_poisson)
o3d.visualization.draw_geometries([pcd_poisson] + arrows)