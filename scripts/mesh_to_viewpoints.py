#!/usr/bin/env python3
"""
FOV-based Viewpoint Sampling for Vision Inspection

Given camera specifications (FOV, working distance, depth of field),
this script samples optimal viewpoints that efficiently cover a 3D mesh object.

Key differences from mesh_to_tsp.py:
- Uses FOV-based sampling instead of Poisson disk sampling
- Considers camera specifications (FOV, WD, DOF)
- Generates viewpoints with proper spacing and overlap
- Validates depth of field constraints
- Outputs compatible HDF5 format for mesh_to_tsp.py
"""

import os
import sys
import argparse
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import open3d as o3d

# Import TSP utilities for saving results
from tsp_utils import save_viewpoints


@dataclass
class CameraSpec:
    """
    Camera and lens specifications

    Attributes:
        sensor_width_px: Sensor width in pixels (default: 4096)
        sensor_height_px: Sensor height in pixels (default: 3000)
        pixel_size_um: Pixel size in micrometers (default: 3.45)
        fov_width_mm: Field of view width in mm (default: 41.0)
        fov_height_mm: Field of view height in mm (default: 30.0)
        working_distance_mm: Working distance in mm (default: 110.0)
        depth_of_field_mm: Depth of field in mm (default: 0.5)
        overlap_ratio: Overlap ratio between adjacent views (default: 0.25 for 25%)
    """
    sensor_width_px: int = 4096
    sensor_height_px: int = 3000
    pixel_size_um: float = 3.45
    fov_width_mm: float = 41.0
    fov_height_mm: float = 30.0
    working_distance_mm: float = 110.0
    depth_of_field_mm: float = 0.5
    overlap_ratio: float = 0.25

    def get_effective_coverage_mm(self) -> Tuple[float, float]:
        """
        Calculate effective coverage per viewpoint considering overlap

        Returns:
            (width, height) in mm
        """
        effective_width = self.fov_width_mm * (1.0 - self.overlap_ratio)
        effective_height = self.fov_height_mm * (1.0 - self.overlap_ratio)
        return effective_width, effective_height

    def get_working_distance_m(self) -> float:
        """Get working distance in meters"""
        return self.working_distance_mm / 1000.0

    def get_dof_m(self) -> float:
        """Get depth of field in meters"""
        return self.depth_of_field_mm / 1000.0

    def get_fov_m(self) -> Tuple[float, float]:
        """Get FOV in meters (width, height)"""
        return self.fov_width_mm / 1000.0, self.fov_height_mm / 1000.0

    def to_dict(self) -> dict:
        """Convert camera spec to dictionary for HDF5 storage"""
        return {
            'sensor_width_px': self.sensor_width_px,
            'sensor_height_px': self.sensor_height_px,
            'pixel_size_um': self.pixel_size_um,
            'fov_width_mm': self.fov_width_mm,
            'fov_height_mm': self.fov_height_mm,
            'working_distance_mm': self.working_distance_mm,
            'depth_of_field_mm': self.depth_of_field_mm,
            'overlap_ratio': self.overlap_ratio,
        }

    def __str__(self) -> str:
        eff_w, eff_h = self.get_effective_coverage_mm()
        return (
            f"Camera Specifications:\n"
            f"  Sensor: {self.sensor_width_px} x {self.sensor_height_px} px\n"
            f"  Pixel size: {self.pixel_size_um} μm\n"
            f"  FOV: {self.fov_width_mm} x {self.fov_height_mm} mm\n"
            f"  Working Distance: {self.working_distance_mm} mm\n"
            f"  Depth of Field: {self.depth_of_field_mm} mm\n"
            f"  Overlap: {self.overlap_ratio * 100:.1f}%\n"
            f"  Effective coverage per view: {eff_w:.2f} x {eff_h:.2f} mm"
        )


@dataclass
class Viewpoint:
    """
    Represents a camera viewpoint

    Attributes:
        position: 3D position in meters (Open3D coordinate system, Y-up)
        normal: Surface normal direction (unit vector)
        coverage_area: Estimated coverage area in m²
        depth_variation: Max depth variation within FOV in meters
    """
    position: np.ndarray
    normal: np.ndarray
    coverage_area: float = 0.0
    depth_variation: float = 0.0


def load_mesh_file(file_path: str) -> Tuple[o3d.geometry.TriangleMesh, float]:
    """
    Load mesh file and compute its properties

    Args:
        file_path: Path to .obj file

    Returns:
        mesh: Open3D triangle mesh
        surface_area: Total surface area in m²
    """
    print(f"Loading mesh from: {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)

    if not mesh.has_vertex_normals():
        print("Computing vertex normals...")
        mesh.compute_vertex_normals()

    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    # Compute surface area
    surface_area = mesh.get_surface_area()

    num_vertices = len(mesh.vertices)
    num_triangles = len(mesh.triangles)

    # Get coordinate range to detect unit issues
    vertices = np.asarray(mesh.vertices)
    coord_min = vertices.min(axis=0)
    coord_max = vertices.max(axis=0)
    coord_range = coord_max - coord_min

    print(f"Loaded mesh: {num_vertices} vertices, {num_triangles} triangles")
    print(f"Surface area: {surface_area * 1e6:.2f} mm² (assuming mesh is in meters)")
    print(f"\nMesh coordinate range:")
    print(f"  X: [{coord_min[0]:.6f}, {coord_max[0]:.6f}] (range: {coord_range[0]:.6f})")
    print(f"  Y: [{coord_min[1]:.6f}, {coord_max[1]:.6f}] (range: {coord_range[1]:.6f})")
    print(f"  Z: [{coord_min[2]:.6f}, {coord_max[2]:.6f}] (range: {coord_range[2]:.6f})")

    # Detect likely unit issues
    max_range = coord_range.max()
    if max_range > 1.0:
        print(f"\n⚠️  WARNING: Mesh coordinates appear to be in MILLIMETERS (max range: {max_range:.2f})")
        print(f"  → Consider using --mesh_unit_scale 0.001 to convert to meters")
    elif max_range < 0.01:
        print(f"\n⚠️  WARNING: Mesh coordinates appear unusually small (max range: {max_range:.6f})")
    else:
        print(f"\n✓ Mesh coordinates appear to be in METERS (max range: {max_range:.4f}m)")

    return mesh, surface_area


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length

    Args:
        vectors: (N, 3) array of vectors

    Returns:
        normalized: (N, 3) array of unit vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    return vectors / norms


def compute_surface_curvature(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Compute approximate surface curvature at each vertex

    Args:
        mesh: Open3D triangle mesh

    Returns:
        curvatures: (N,) array of curvature values
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Compute vertex normals if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    normals = np.asarray(mesh.vertex_normals)

    # Estimate curvature by normal variation in neighborhood
    curvatures = np.zeros(len(vertices))

    for i, vertex in enumerate(vertices):
        # Find adjacent vertices
        adjacent_tris = triangles[np.any(triangles == i, axis=1)]
        adjacent_verts = np.unique(adjacent_tris.flatten())
        adjacent_verts = adjacent_verts[adjacent_verts != i]

        if len(adjacent_verts) > 0:
            # Compute normal variation
            normal_diffs = normals[adjacent_verts] - normals[i]
            curvatures[i] = np.mean(np.linalg.norm(normal_diffs, axis=1))

    return curvatures


def estimate_required_viewpoints(
    mesh: o3d.geometry.TriangleMesh,
    camera_spec: CameraSpec,
    target_coverage: float = 0.95,
    curvature_factor: float = 1.5
) -> int:
    """
    Estimate required number of viewpoints based on mesh properties

    Args:
        mesh: Open3D triangle mesh
        camera_spec: Camera specifications
        target_coverage: Target coverage ratio (default: 95%)
        curvature_factor: Multiplier for high-curvature regions (default: 1.5)

    Returns:
        num_viewpoints: Estimated number of viewpoints needed
    """
    # Get mesh surface area
    surface_area = mesh.get_surface_area()

    # Get effective coverage per viewpoint
    eff_w, eff_h = camera_spec.get_effective_coverage_mm()
    fov_area = (eff_w / 1000.0) * (eff_h / 1000.0)  # Convert to m²

    # Basic estimate (flat surface assumption)
    basic_estimate = int(np.ceil(surface_area * target_coverage / fov_area))

    # Adjust for surface curvature
    curvatures = compute_surface_curvature(mesh)
    avg_curvature = np.mean(curvatures)
    max_curvature = np.max(curvatures)

    # Higher curvature requires more viewpoints
    curvature_adjustment = 1.0 + (avg_curvature / max(max_curvature, 1e-6)) * (curvature_factor - 1.0)

    estimated_viewpoints = int(np.ceil(basic_estimate * curvature_adjustment))

    print(f"\nAutomatic viewpoint estimation:")
    print(f"  Surface area: {surface_area * 1e6:.2f} mm²")
    print(f"  FOV coverage per view: {fov_area * 1e6:.2f} mm²")
    print(f"  Basic estimate (flat): {basic_estimate} viewpoints")
    print(f"  Curvature adjustment: {curvature_adjustment:.2f}x")
    print(f"  Final estimate: {estimated_viewpoints} viewpoints")

    return estimated_viewpoints


def sample_points_uniform(mesh: o3d.geometry.TriangleMesh, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points uniformly on mesh surface using Poisson disk sampling

    Args:
        mesh: Open3D triangle mesh
        num_points: Number of points to sample

    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
    """
    print(f"Sampling {num_points} points using Poisson disk sampling...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals()

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    print(f"Sampled {len(points)} points")
    return points, normals


def sample_points_adaptive(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    curvature_weight: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points adaptively based on surface curvature

    Args:
        mesh: Open3D triangle mesh
        num_points: Number of points to sample
        curvature_weight: Weight for curvature-based sampling (0-1)

    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
    """
    print(f"Sampling {num_points} points using adaptive (curvature-based) sampling...")

    # Compute curvature
    curvatures = compute_surface_curvature(mesh)

    # Normalize curvatures to [0, 1]
    curvatures_norm = curvatures / (np.max(curvatures) + 1e-8)

    # Compute vertex weights (mix of uniform and curvature-based)
    uniform_weight = 1.0 - curvature_weight
    weights = uniform_weight + curvature_weight * curvatures_norm

    # Normalize to probabilities
    weights = weights / np.sum(weights)

    # Sample vertices according to weights
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Sample more from high-curvature regions
    # For each triangle, compute average weight
    tri_weights = np.mean(weights[triangles], axis=1)
    tri_weights = tri_weights / np.sum(tri_weights)

    # Sample points from triangles
    sampled_points = []
    sampled_normals = []

    # Compute triangle normals
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    tri_normals = np.asarray(mesh.triangle_normals)

    for _ in range(num_points):
        # Choose triangle based on weights
        tri_idx = np.random.choice(len(triangles), p=tri_weights)

        # Random barycentric coordinates
        r1, r2 = np.random.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2

        # Interpolate position
        tri = triangles[tri_idx]
        point = (1 - r1 - r2) * vertices[tri[0]] + r1 * vertices[tri[1]] + r2 * vertices[tri[2]]

        sampled_points.append(point)
        sampled_normals.append(tri_normals[tri_idx])

    points = np.array(sampled_points, dtype=np.float32)
    normals = np.array(sampled_normals, dtype=np.float32)

    print(f"Sampled {len(points)} points (adaptive)")
    return points, normals


def compute_viewpoints_from_surface(
    points: np.ndarray,
    normals: np.ndarray,
    camera_spec: CameraSpec
) -> List[Viewpoint]:
    """
    Compute viewpoints from surface points by offsetting along normals

    Args:
        points: (N, 3) surface points
        normals: (N, 3) surface normals
        camera_spec: Camera specifications

    Returns:
        viewpoints: List of Viewpoint objects
    """
    wd = camera_spec.get_working_distance_m()

    # Normalize normals
    normals_normalized = normalize_vectors(normals)

    # Offset points along normals by working distance
    viewpoint_positions = points + normals_normalized * wd

    # Camera looks toward surface (opposite of normal)
    camera_directions = -normals_normalized

    # Estimate coverage area (assuming flat surface perpendicular to camera)
    fov_w, fov_h = camera_spec.get_fov_m()
    coverage_area = fov_w * fov_h

    viewpoints = []
    for i in range(len(points)):
        vp = Viewpoint(
            position=viewpoint_positions[i],
            normal=camera_directions[i],
            coverage_area=coverage_area,
            depth_variation=0.0  # Will be computed later if needed
        )
        viewpoints.append(vp)

    return viewpoints


def filter_viewpoints_by_dof(
    viewpoints: List[Viewpoint],
    mesh: o3d.geometry.TriangleMesh,
    camera_spec: CameraSpec,
    remove_invalid: bool = False
) -> Tuple[List[Viewpoint], int]:
    """
    Check depth of field constraints for each viewpoint

    Args:
        viewpoints: List of viewpoints to check
        mesh: Original mesh for depth computation
        camera_spec: Camera specifications
        remove_invalid: If True, remove viewpoints that violate DOF constraints

    Returns:
        filtered_viewpoints: List of viewpoints (filtered if remove_invalid=True)
        num_violated: Number of viewpoints that violated DOF constraints
    """
    dof_limit = camera_spec.get_dof_m()
    fov_w, fov_h = camera_spec.get_fov_m()

    # Create mesh scene for raycasting
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_legacy)

    filtered_viewpoints = []
    num_violated = 0

    print(f"Checking DOF constraints (limit: {camera_spec.depth_of_field_mm:.2f} mm)...")

    for i, vp in enumerate(viewpoints):
        # Sample rays within FOV to check depth variation
        # Use a grid of rays (e.g., 5x5 grid)
        grid_size = 5
        rays_origin = []
        rays_direction = []

        # Build local coordinate frame
        z_axis = vp.normal  # Camera looks in this direction

        # Choose helper vector for cross product
        helper = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.9 else np.array([1, 0, 0])
        x_axis = np.cross(helper, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Sample grid within FOV
        for ix in range(grid_size):
            for iy in range(grid_size):
                u = (ix / (grid_size - 1) - 0.5) * fov_w
                v = (iy / (grid_size - 1) - 0.5) * fov_h

                # Ray origin at viewpoint position
                rays_origin.append(vp.position)

                # Ray direction: offset in local frame
                direction = z_axis + u * x_axis + v * y_axis
                direction = direction / np.linalg.norm(direction)
                rays_direction.append(direction)

        rays_origin = np.array(rays_origin, dtype=np.float32)
        rays_direction = np.array(rays_direction, dtype=np.float32)

        # Cast rays
        rays = np.concatenate([rays_origin, rays_direction], axis=1)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

        # Get hit distances
        t_hit = ans['t_hit'].numpy()

        # Filter valid hits (not inf)
        valid_hits = t_hit[np.isfinite(t_hit)]

        if len(valid_hits) > 0:
            depth_variation = valid_hits.max() - valid_hits.min()
            vp.depth_variation = depth_variation

            if depth_variation > dof_limit:
                num_violated += 1
                if not remove_invalid:
                    filtered_viewpoints.append(vp)
            else:
                filtered_viewpoints.append(vp)
        else:
            # No valid hits - keep viewpoint but mark as zero variation
            vp.depth_variation = 0.0
            filtered_viewpoints.append(vp)

    if remove_invalid:
        print(f"Removed {num_violated} viewpoints violating DOF constraints")
        print(f"Remaining viewpoints: {len(filtered_viewpoints)}")
    else:
        print(f"Found {num_violated} viewpoints violating DOF constraints (kept for analysis)")

    return filtered_viewpoints, num_violated


def compute_voxel_based_coverage(
    viewpoints: List[Viewpoint],
    mesh: o3d.geometry.TriangleMesh,
    camera_spec: CameraSpec,
    voxel_size: float = 0.002  # 2mm voxels
) -> Tuple[float, int, int]:
    """
    Compute accurate coverage using voxel grid (removes overlap)

    Args:
        viewpoints: List of viewpoints
        mesh: Original mesh
        camera_spec: Camera specifications
        voxel_size: Voxel size in meters (default: 2mm)

    Returns:
        coverage_ratio: Actual coverage ratio (0-1)
        covered_voxels: Number of covered voxels
        total_voxels: Total voxels on mesh surface
    """
    if len(viewpoints) == 0:
        return 0.0, 0, 0

    # Create voxel grid from mesh
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=voxel_size)

    # Get all voxel centers
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        return 0.0, 0, 0

    voxel_centers = np.array([voxel_grid.get_voxel_center_coordinate(v.grid_index) for v in voxels])
    total_voxels = len(voxel_centers)

    # Create mesh scene for raycasting
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_legacy)

    # Mark which voxels are covered by each viewpoint
    covered_mask = np.zeros(total_voxels, dtype=bool)

    fov_w, fov_h = camera_spec.get_fov_m()
    wd = camera_spec.get_working_distance_m()

    print(f"Computing voxel-based coverage ({total_voxels} voxels)...")

    for i, vp in enumerate(viewpoints):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing viewpoint {i+1}/{len(viewpoints)}...")

        # Build local coordinate frame
        z_axis = vp.normal  # Camera looks in this direction
        helper = np.array([0, 0, 1]) if abs(z_axis[2]) < 0.9 else np.array([1, 0, 0])
        x_axis = np.cross(helper, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        # Check which voxels are within FOV
        voxel_to_camera = voxel_centers - vp.position

        # Project onto camera axes
        proj_z = np.dot(voxel_to_camera, z_axis)  # Distance along viewing direction
        proj_x = np.dot(voxel_to_camera, x_axis)  # Horizontal offset
        proj_y = np.dot(voxel_to_camera, y_axis)  # Vertical offset

        # Check if within FOV cone
        # At distance d, FOV extends ±fov_w/2 horizontally and ±fov_h/2 vertically
        in_fov_mask = (
            (proj_z > 0) &  # In front of camera
            (proj_z < wd * 1.5) &  # Within reasonable distance
            (np.abs(proj_x) < fov_w / 2 * (proj_z / wd)) &  # Within horizontal FOV
            (np.abs(proj_y) < fov_h / 2 * (proj_z / wd))    # Within vertical FOV
        )

        if not np.any(in_fov_mask):
            continue

        # For voxels in FOV, check visibility via raycasting
        visible_indices = np.where(in_fov_mask)[0]

        for idx in visible_indices:
            voxel_center = voxel_centers[idx]

            # Ray from camera to voxel
            ray_dir = voxel_center - vp.position
            ray_dist = np.linalg.norm(ray_dir)
            ray_dir = ray_dir / ray_dist

            # Cast ray
            rays = np.array([[vp.position[0], vp.position[1], vp.position[2],
                            ray_dir[0], ray_dir[1], ray_dir[2]]], dtype=np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

            t_hit = ans['t_hit'].numpy()[0]

            # If hit distance is close to voxel distance, voxel is visible
            if np.isfinite(t_hit) and abs(t_hit - ray_dist) < voxel_size * 2:
                covered_mask[idx] = True

    covered_voxels = np.sum(covered_mask)
    coverage_ratio = covered_voxels / total_voxels if total_voxels > 0 else 0.0

    print(f"  Coverage: {covered_voxels}/{total_voxels} voxels ({coverage_ratio*100:.1f}%)")

    return coverage_ratio, covered_voxels, total_voxels


def compute_coverage_statistics(
    viewpoints: List[Viewpoint],
    mesh_surface_area: float,
    mesh: o3d.geometry.TriangleMesh = None,
    camera_spec: CameraSpec = None,
    use_voxel_coverage: bool = True
) -> dict:
    """
    Compute coverage statistics

    Args:
        viewpoints: List of viewpoints
        mesh_surface_area: Total mesh surface area in m²
        mesh: Mesh object (required for voxel-based coverage)
        camera_spec: Camera spec (required for voxel-based coverage)
        use_voxel_coverage: If True, use accurate voxel-based calculation

    Returns:
        stats: Dictionary with coverage statistics
    """
    # Simple coverage (with overlap)
    simple_coverage = sum(vp.coverage_area for vp in viewpoints)
    simple_ratio = simple_coverage / mesh_surface_area if mesh_surface_area > 0 else 0.0

    stats = {
        'num_viewpoints': len(viewpoints),
        'simple_coverage_m2': simple_coverage,
        'simple_coverage_ratio': simple_ratio,
        'mesh_area_m2': mesh_surface_area,
        'avg_depth_variation': np.mean([vp.depth_variation for vp in viewpoints]) if viewpoints else 0.0,
        'max_depth_variation': np.max([vp.depth_variation for vp in viewpoints]) if viewpoints else 0.0
    }

    # Voxel-based coverage (no overlap)
    if use_voxel_coverage and mesh is not None and camera_spec is not None:
        voxel_ratio, covered_voxels, total_voxels = compute_voxel_based_coverage(
            viewpoints, mesh, camera_spec
        )
        stats['voxel_coverage_ratio'] = voxel_ratio
        stats['covered_voxels'] = covered_voxels
        stats['total_voxels'] = total_voxels
    else:
        stats['voxel_coverage_ratio'] = None
        stats['covered_voxels'] = 0
        stats['total_voxels'] = 0

    return stats


def visualize_viewpoints(
    mesh: o3d.geometry.TriangleMesh,
    viewpoints: List[Viewpoint],
    camera_spec: CameraSpec,
    title: str = "Viewpoints Visualization"
):
    """
    Visualize viewpoints using Open3D

    Args:
        mesh: Original mesh
        viewpoints: List of viewpoints
        camera_spec: Camera specifications
        title: Window title
    """
    geometries = []

    # Add mesh (gray)
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(mesh_vis)

    # Add viewpoint positions as spheres (green)
    for vp in viewpoints:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
        sphere.translate(vp.position)
        sphere.paint_uniform_color([0.0, 1.0, 0.0])
        geometries.append(sphere)

        # Add normal direction as arrow
        arrow_length = camera_spec.get_working_distance_m() * 0.3
        arrow_end = vp.position + vp.normal * arrow_length

        points = [vp.position, arrow_end]
        lines = [[0, 1]]
        colors = [[1, 0, 0]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)

    # Visualize
    print(f"\nVisualizing {len(viewpoints)} viewpoints...")
    print("  Green spheres: viewpoint positions")
    print("  Red arrows: camera viewing directions")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720
    )


def plot_statistics(
    stats: dict,
    camera_spec: CameraSpec,
    output_path: str = "viewpoint_stats.png"
):
    """
    Plot statistics using matplotlib

    Args:
        stats: Statistics dictionary
        camera_spec: Camera specifications
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Coverage statistics
    ax = axes[0]

    # Use voxel coverage if available, otherwise use simple coverage
    coverage_ratio = stats.get('voxel_coverage_ratio', None)
    if coverage_ratio is None:
        coverage_ratio = stats['simple_coverage_ratio']
        title_suffix = "(Simple estimate)"
    else:
        title_suffix = "(Voxel-based)"

    # Handle cases where coverage > 100% (overlapping views - only in simple mode)
    if coverage_ratio <= 1.0:
        labels = ['Coverage', 'Uncovered']
        sizes = [
            coverage_ratio * 100,
            (1 - coverage_ratio) * 100
        ]
        colors = ['#66b3ff', '#ff9999']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"Coverage Statistics\n{stats['num_viewpoints']} viewpoints\n{title_suffix}")
    else:
        # Coverage > 100%, show as bar chart instead
        ax.barh(['Coverage'], [coverage_ratio * 100], color='#66b3ff')
        ax.axvline(x=100, color='red', linestyle='--', linewidth=2, label='100% coverage')
        ax.set_xlabel('Coverage (%)')
        ax.set_xlim(0, coverage_ratio * 110)
        ax.legend()
        ax.set_title(f"Coverage Statistics\n{stats['num_viewpoints']} viewpoints\n{title_suffix}")

    # Text summary
    ax = axes[1]
    ax.axis('off')

    eff_w, eff_h = camera_spec.get_effective_coverage_mm()

    # Build coverage text
    if stats.get('voxel_coverage_ratio') is not None:
        coverage_text = f"""  Voxel coverage: {stats['voxel_coverage_ratio'] * 100:.1f}% (no overlap)
  Voxels: {stats['covered_voxels']}/{stats['total_voxels']}
  Simple estimate: {stats['simple_coverage_ratio'] * 100:.1f}% (with overlap)"""
    else:
        coverage_text = f"""  Coverage: {stats['simple_coverage_ratio'] * 100:.1f}% (simple estimate)
  Total coverage: {stats['simple_coverage_m2'] * 1e6:.2f} mm²"""

    summary_text = f"""
Camera Specifications:
  FOV: {camera_spec.fov_width_mm} × {camera_spec.fov_height_mm} mm
  Working Distance: {camera_spec.working_distance_mm} mm
  Depth of Field: {camera_spec.depth_of_field_mm} mm
  Overlap: {camera_spec.overlap_ratio * 100:.1f}%
  Effective coverage: {eff_w:.2f} × {eff_h:.2f} mm

Results:
  Number of viewpoints: {stats['num_viewpoints']}
  Mesh surface area: {stats['mesh_area_m2'] * 1e6:.2f} mm²
{coverage_text}

Depth Variation:
  Average: {stats['avg_depth_variation'] * 1000:.3f} mm
  Maximum: {stats['max_depth_variation'] * 1000:.3f} mm
  DOF limit: {camera_spec.depth_of_field_mm:.2f} mm
"""

    ax.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved statistics plot to: {output_path}")
    plt.close()


def normalize_coordinates(points: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Normalize point coordinates to [0, 1] range

    Args:
        points: (N, 3) array

    Returns:
        normalized_points: (N, 3) array in [0, 1]
        normalization_info: dict with min/max for denormalization
    """
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Normalize to [0, 1]
    normalized = (points - min_coords) / (max_coords - min_coords + 1e-8)

    normalization_info = {
        'min': min_coords,
        'max': max_coords
    }

    return normalized, normalization_info


def main():
    parser = argparse.ArgumentParser(
        description='FOV-based Viewpoint Sampling for Vision Inspection'
    )

    # Input/Output
    parser.add_argument('--mesh_file', type=str, required=True,
                        help='Path to mesh file (.obj)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save viewpoints as HDF5 file (e.g., data/input/tour/glass_fov.h5)')
    parser.add_argument('--output', type=str, default='viewpoint_stats.png',
                        help='Output path for statistics plot')

    # Camera specifications
    parser.add_argument('--fov_width', type=float, default=41.0,
                        help='Field of view width in mm (default: 41.0)')
    parser.add_argument('--fov_height', type=float, default=30.0,
                        help='Field of view height in mm (default: 30.0)')
    parser.add_argument('--working_distance', type=float, default=110.0,
                        help='Working distance in mm (default: 110.0)')
    parser.add_argument('--depth_of_field', type=float, default=0.5,
                        help='Depth of field in mm (default: 0.5)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap ratio between views (default: 0.25 for 25%%)')

    # Sampling parameters
    parser.add_argument('--num_points', type=int, default=None,
                        help='Number of surface points to sample (default: auto-calculate)')
    parser.add_argument('--auto_num_points', action='store_true',
                        help='Automatically calculate optimal number of viewpoints based on mesh curvature')
    parser.add_argument('--target_coverage', type=float, default=0.95,
                        help='Target coverage ratio for auto calculation (default: 0.95)')
    parser.add_argument('--adaptive_sampling', action='store_true',
                        help='Use adaptive sampling based on surface curvature')
    parser.add_argument('--curvature_weight', type=float, default=0.5,
                        help='Weight for curvature in adaptive sampling (0-1, default: 0.5)')
    parser.add_argument('--check_dof', action='store_true',
                        help='Check depth of field constraints')
    parser.add_argument('--remove_invalid_dof', action='store_true',
                        help='Remove viewpoints that violate DOF constraints')
    parser.add_argument('--voxel_coverage', action='store_true',
                        help='Compute accurate voxel-based coverage (removes overlap)')
    parser.add_argument('--voxel_size', type=float, default=2.0,
                        help='Voxel size in mm for coverage calculation (default: 2.0)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize viewpoints with Open3D')
    parser.add_argument('--plot', action='store_true',
                        help='Save statistics plot')

    # Unit conversion
    parser.add_argument('--mesh_unit_scale', type=float, default=1.0,
                        help='Scale factor to convert mesh units to meters (e.g., 0.001 for mm→m, default: 1.0)')

    args = parser.parse_args()

    # Create camera spec
    camera_spec = CameraSpec(
        fov_width_mm=args.fov_width,
        fov_height_mm=args.fov_height,
        working_distance_mm=args.working_distance,
        depth_of_field_mm=args.depth_of_field,
        overlap_ratio=args.overlap
    )

    print("=" * 60)
    print("FOV-based Viewpoint Sampling")
    print("=" * 60)
    print(camera_spec)
    print("=" * 60)

    # Load mesh
    mesh, surface_area = load_mesh_file(args.mesh_file)

    # Apply unit scale if specified
    if args.mesh_unit_scale != 1.0:
        print(f"\n{'='*60}")
        print(f"APPLYING UNIT SCALE: {args.mesh_unit_scale}")
        print(f"{'='*60}")

        vertices = np.asarray(mesh.vertices)
        vertices_scaled = vertices * args.mesh_unit_scale
        mesh.vertices = o3d.utility.Vector3dVector(vertices_scaled)

        # Update surface area (scales by square of linear scale)
        surface_area = surface_area * (args.mesh_unit_scale ** 2)

        # Show new coordinate range
        coord_min = vertices_scaled.min(axis=0)
        coord_max = vertices_scaled.max(axis=0)
        coord_range = coord_max - coord_min

        print(f"Scaled mesh coordinates (meters):")
        print(f"  X: [{coord_min[0]:.6f}, {coord_max[0]:.6f}] (range: {coord_range[0]:.6f})")
        print(f"  Y: [{coord_min[1]:.6f}, {coord_max[1]:.6f}] (range: {coord_range[1]:.6f})")
        print(f"  Z: [{coord_min[2]:.6f}, {coord_max[2]:.6f}] (range: {coord_range[2]:.6f})")
        print(f"Scaled surface area: {surface_area * 1e6:.2f} mm²")
        print(f"{'='*60}\n")

    # Determine number of points
    if args.auto_num_points or args.num_points is None:
        num_points = estimate_required_viewpoints(
            mesh, camera_spec,
            target_coverage=args.target_coverage
        )
    else:
        num_points = args.num_points

    # Sample surface points
    if args.adaptive_sampling:
        surface_points, surface_normals = sample_points_adaptive(
            mesh, num_points, curvature_weight=args.curvature_weight
        )
    else:
        surface_points, surface_normals = sample_points_uniform(mesh, num_points)

    # Compute viewpoints
    wd_meters = camera_spec.get_working_distance_m()
    print(f"\nComputing viewpoints...")
    print(f"  Working distance: {args.working_distance} mm = {wd_meters} m")
    print(f"  Offsetting surface points by {wd_meters} m along normals")
    viewpoints = compute_viewpoints_from_surface(surface_points, surface_normals, camera_spec)
    print(f"  Generated {len(viewpoints)} viewpoints")

    # Check DOF constraints if requested
    num_violated = 0
    if args.check_dof:
        viewpoints, num_violated = filter_viewpoints_by_dof(
            viewpoints, mesh, camera_spec, remove_invalid=args.remove_invalid_dof
        )

    # Compute statistics
    stats = compute_coverage_statistics(
        viewpoints, surface_area,
        mesh=mesh if args.voxel_coverage else None,
        camera_spec=camera_spec if args.voxel_coverage else None,
        use_voxel_coverage=args.voxel_coverage
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of viewpoints: {stats['num_viewpoints']}")
    print(f"Mesh surface area: {stats['mesh_area_m2'] * 1e6:.2f} mm²")

    # Print both coverage metrics if voxel coverage was computed
    if stats['voxel_coverage_ratio'] is not None:
        print(f"\nCoverage (voxel-based, no overlap):")
        print(f"  Voxels: {stats['covered_voxels']}/{stats['total_voxels']}")
        print(f"  Coverage ratio: {stats['voxel_coverage_ratio'] * 100:.1f}%")
        print(f"\nCoverage (simple estimate, with overlap):")
        print(f"  Total coverage: {stats['simple_coverage_m2'] * 1e6:.2f} mm²")
        print(f"  Coverage ratio: {stats['simple_coverage_ratio'] * 100:.1f}%")
    else:
        print(f"Total coverage: {stats['simple_coverage_m2'] * 1e6:.2f} mm²")
        print(f"Coverage ratio (with overlap): {stats['simple_coverage_ratio'] * 100:.1f}%")

    if args.check_dof:
        print(f"\nDOF constraints:")
        print(f"  Violations: {num_violated}")
        print(f"  Avg depth variation: {stats['avg_depth_variation'] * 1000:.3f} mm")
        print(f"  Max depth variation: {stats['max_depth_variation'] * 1000:.3f} mm")
    print("=" * 60)

    # Save to HDF5 if requested
    if args.save_path:
        # IMPORTANT: Save surface positions and surface normals (NOT viewpoint positions)
        # This ensures compatibility with mesh_to_tsp.py and run_app_v2.py
        # which expect surface points and will apply NORMAL_SAMPLE_OFFSET themselves

        # Convert viewpoint positions back to surface positions
        # viewpoint.position = surface_point + normal * working_distance
        # Therefore: surface_point = viewpoint.position - normal * working_distance
        wd = camera_spec.get_working_distance_m()

        surface_positions = []
        surface_normals = []

        for vp in viewpoints:
            # Viewpoint stores: position (camera location), normal (camera direction = -surface_normal)
            # We need to recover: surface position, surface normal

            # Camera direction points toward surface, so surface normal = -camera_direction
            surface_normal = -vp.normal  # Flip back to surface normal

            # Forward: viewpoint_pos = surface_pos + surface_normal * wd
            # Inverse: surface_pos = viewpoint_pos - surface_normal * wd
            surface_pos = vp.position - surface_normal * wd

            surface_positions.append(surface_pos)
            surface_normals.append(surface_normal)

        surface_positions = np.array(surface_positions, dtype=np.float32)
        surface_normals = np.array(surface_normals, dtype=np.float32)

        # Verify conversion (compare first viewpoint)
        if len(viewpoints) > 0:
            first_vp = viewpoints[0]
            first_surface_pos = surface_positions[0]
            first_surface_normal = surface_normals[0]

            # Recompute viewpoint position to verify
            recomputed_vp_pos = first_surface_pos + first_surface_normal * wd
            position_error = np.linalg.norm(recomputed_vp_pos - first_vp.position)

        print(f"\n{'='*60}")
        print("COORDINATE CONVERSION FOR HDF5 SAVE")
        print(f"{'='*60}")
        print(f"Converting viewpoint positions → surface positions")
        print(f"  Working distance: {wd*1000:.1f} mm = {wd:.6f} m")
        print(f"  Forward:  viewpoint_pos = surface_pos + surface_normal × WD")
        print(f"  Inverse:  surface_pos = viewpoint_pos - surface_normal × WD")
        if len(viewpoints) > 0:
            print(f"\nVerification (first viewpoint):")
            print(f"  Original viewpoint pos:  {first_vp.position}")
            print(f"  Recovered surface pos:   {first_surface_pos}")
            print(f"  Recomputed viewpoint:    {recomputed_vp_pos}")
            print(f"  Position error:          {position_error*1000:.6f} mm")
            if position_error > 1e-6:
                print(f"  ⚠️  WARNING: Conversion error detected!")
            else:
                print(f"  ✓ Conversion verified (error < 1 μm)")
        print(f"\nSaving {len(surface_positions)} surface positions to HDF5")
        print(f"{'='*60}\n")

        # Save using simplified viewpoints format
        save_viewpoints(
            file_path=args.save_path,
            points=surface_positions,      # Surface positions (not camera positions)
            normals=surface_normals,       # Surface normals (not camera directions)
            mesh_file=args.mesh_file,
            camera_spec=camera_spec.to_dict(),
        )

    # Plot statistics if requested
    if args.plot:
        plot_statistics(stats, camera_spec, args.output)

    # Visualize if requested
    if args.visualize:
        visualize_viewpoints(mesh, viewpoints, camera_spec,
                           title=f"FOV-based Viewpoints ({len(viewpoints)} views)")

    print("\nDone!")


if __name__ == "__main__":
    main()
