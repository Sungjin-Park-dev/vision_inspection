#!/usr/bin/env python3
"""
FOV-based Viewpoint Sampling for Vision Inspection

Given camera specifications (FOV, working distance, depth of field),
this script samples optimal viewpoints that efficiently cover a 3D mesh object.

Key features:
- Uses FOV-based sampling with Poisson disk distribution
- Considers camera specifications (FOV, WD, DOF)
- Generates viewpoints with proper spacing and overlap
- Validates depth of field constraints
- Outputs compatible HDF5 format for viewpoints_to_tsp.py

Coordinate system:
- Input: Z-up mesh (compatible with Isaac Sim / URDF / Pinocchio)
- Output: Z-up surface positions and normals
- Camera positions computed as: surface_position + surface_normal * working_distance

Usage:
    omni_python scripts/mesh_to_viewpoints.py --mesh_file data/object/glass.obj
"""

import os
import sys
import argparse
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import open3d as o3d

# Import common utilities
from common import config
from common.cli_utils import print_section_header, print_key_value
from common.coordinate_utils import normalize_vectors, offset_points_along_normals

# Import TSP utilities for saving results
from common.data_io import save_viewpoints_hdf5


@dataclass
class CameraSpec:
    """
    Camera and lens specifications

    Default values are imported from common.config for consistency across the pipeline.

    Attributes:
        sensor_width_px: Sensor width in pixels
        sensor_height_px: Sensor height in pixels
        pixel_size_um: Pixel size in micrometers
        fov_width_mm: Field of view width in mm
        fov_height_mm: Field of view height in mm
        working_distance_mm: Working distance in mm
        depth_of_field_mm: Depth of field in mm
        overlap_ratio: Overlap ratio between adjacent views (0.25 = 25%)
    """
    sensor_width_px: int = config.CAMERA_SENSOR_WIDTH_PX
    sensor_height_px: int = config.CAMERA_SENSOR_HEIGHT_PX
    pixel_size_um: float = config.CAMERA_PIXEL_SIZE_UM
    fov_width_mm: float = config.CAMERA_FOV_WIDTH_MM
    fov_height_mm: float = config.CAMERA_FOV_HEIGHT_MM
    working_distance_mm: float = config.CAMERA_WORKING_DISTANCE_MM
    depth_of_field_mm: float = config.CAMERA_DEPTH_OF_FIELD_MM
    overlap_ratio: float = config.CAMERA_OVERLAP_RATIO

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
    print(f"\nMesh coordinate range (Z-up coordinate system):")
    print(f"  X: [{coord_min[0]:.6f}, {coord_max[0]:.6f}] (range: {coord_range[0]:.6f})")
    print(f"  Y: [{coord_min[1]:.6f}, {coord_max[1]:.6f}] (range: {coord_range[1]:.6f})")
    print(f"  Z: [{coord_min[2]:.6f}, {coord_max[2]:.6f}] (range: {coord_range[2]:.6f}) ← up direction")

    # Detect likely unit issues
    max_range = coord_range.max()
    if max_range > 1.0:
        print(f"\n⚠️  WARNING: Mesh coordinates appear to be in MILLIMETERS (max range: {max_range:.2f})")
        print(f"  → Consider scaling the mesh to meters before running this script (e.g., ×0.001)")
    elif max_range < 0.01:
        print(f"\n⚠️  WARNING: Mesh coordinates appear unusually small (max range: {max_range:.6f})")
    else:
        print(f"\n✓ Mesh coordinates appear to be in METERS (max range: {max_range:.4f}m)")
        print(f"✓ Using Z-up coordinate system (compatible with Isaac Sim / URDF / Pinocchio)")

    return mesh, surface_area


# normalize_vectors() is now imported from common.coordinate_utils


def compute_local_overlap(
    curvature_norm: float,
    curvature_weight: float,
    base_overlap_ratio: float,
    max_extra_overlap: float = 0.30
) -> float:
    """
    Calculate local overlap ratio based on surface curvature

    Args:
        curvature_norm: Normalized curvature value [0, 1]
            - 0.0: Flat surface
            - 1.0: Maximum curvature (edges, corners)
        curvature_weight: Weight for curvature influence [0, 1]
            - 0.0: Ignore curvature (use default overlap)
            - 1.0: Maximum curvature influence
        base_overlap_ratio: Baseline overlap ratio (e.g., camera_spec.overlap_ratio)
        max_extra_overlap: Maximum additional overlap that curvature can add (default 0.30)

    Returns:
        overlap: Local overlap ratio
            - Low curvature → baseline overlap (satisfies user-specified overlap everywhere)
            - High curvature → baseline + extra overlap scaled by curvature_weight (capped ≤ 1.0)

    Examples:
        >>> compute_local_overlap(0.0, 1.0, 0.25)  # Flat surface
        0.25
        >>> compute_local_overlap(1.0, 1.0, 0.25)  # Maximum curvature
        0.55
        >>> compute_local_overlap(0.5, 0.0, 0.25)  # Curvature ignored
        0.25
        >>> compute_local_overlap(1.0, 1.0, 0.50)  # High-overlap baseline
        0.80
    """
    if curvature_weight < 1e-6:
        # No curvature influence - use default overlap
        return base_overlap_ratio

    # Additional overlap grows with curvature but never exceeds remaining headroom or max_extra_overlap
    available_headroom = max(0.0, 1.0 - base_overlap_ratio)
    extra_overlap_cap = min(max_extra_overlap, available_headroom)
    extra_overlap = curvature_norm * curvature_weight * extra_overlap_cap

    overlap = base_overlap_ratio + extra_overlap

    return overlap


def sample_points_uniform_with_normals(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample a mesh and guarantee non-empty points/normals.

    Falls back to vertex-based sampling if Open3D returns an empty point cloud,
    which avoids downstream AABB warnings from zero-length geometries.
    """
    if num_points <= 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32)
        )

    target = max(1, num_points)
    pcd = mesh.sample_points_uniformly(number_of_points=target)
    points = np.asarray(pcd.points, dtype=np.float32)

    if len(points) == 0:
        # Fall back to sampling vertices directly
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if len(vertices) == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32)
            )

        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        vertex_normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        indices = np.random.choice(
            len(vertices),
            size=min(num_points, len(vertices)),
            replace=len(vertices) < num_points
        )
        return vertices[indices], vertex_normals[indices]

    if not pcd.has_normals():
        pcd.estimate_normals()
    normals = np.asarray(pcd.normals, dtype=np.float32)

    # Match requested count (Open3D may over/under sample slightly)
    if len(points) != num_points:
        indices = np.random.choice(
            len(points),
            size=num_points,
            replace=len(points) < num_points
        )
        points = points[indices]
        normals = normals[indices]

    return points, normals


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
    target_coverage: float = 1.0,
    curvature_weight: float = 0.5
) -> int:
    """
    Estimate required number of viewpoints based on mesh properties and curvature

    Uses adaptive overlap based on surface curvature:
    - Flat regions: 10% overlap (sparse sampling)
    - High-curvature regions: up to 40% overlap (dense sampling)

    Args:
        mesh: Open3D triangle mesh
        camera_spec: Camera specifications
        target_coverage: Target coverage ratio (default: 1.0)
        curvature_weight: Weight for curvature influence (0-1, default: 0.5)
            - 0.0: Uniform overlap (25% everywhere, ignores curvature)
            - 0.5: Moderate adaptive overlap (10-25%)
            - 1.0: Aggressive adaptive overlap (10-40%)

    Returns:
        num_viewpoints: Estimated number of viewpoints needed
    """
    # Get mesh surface area
    surface_area = mesh.get_surface_area()

    # Get base FOV dimensions
    fov_w_mm, fov_h_mm = camera_spec.fov_width_mm, camera_spec.fov_height_mm
    fov_w_m, fov_h_m = fov_w_mm / 1000.0, fov_h_mm / 1000.0
    fov_area = fov_w_m * fov_h_m
    base_overlap_ratio = camera_spec.overlap_ratio

    # Basic estimate using default overlap
    eff_w, eff_h = camera_spec.get_effective_coverage_mm()
    basic_fov_area = (eff_w / 1000.0) * (eff_h / 1000.0)
    basic_estimate = int(np.ceil(surface_area * target_coverage / basic_fov_area))

    # If curvature_weight is zero, use basic estimate
    if curvature_weight < 1e-6:
        print(f"\nAutomatic viewpoint estimation:")
        print(f"  Surface area: {surface_area * 1e6:.2f} mm²")
        print(f"  FOV: {fov_w_mm:.1f} × {fov_h_mm:.1f} mm")
        print(f"  Uniform overlap: {camera_spec.overlap_ratio * 100:.0f}%")
        print(f"  Effective coverage: {eff_w:.2f} × {eff_h:.2f} mm")
        print(f"  Estimated viewpoints: {basic_estimate}")
        return basic_estimate

    # Compute curvature for adaptive estimation
    print(f"\nComputing surface curvature for adaptive estimation...")
    curvatures = compute_surface_curvature(mesh)
    max_curvature = np.max(curvatures)

    # Normalize curvatures to [0, 1]
    curvatures_norm = curvatures / (max_curvature + 1e-8)

    # Compute weighted surface area based on local overlap
    # Higher curvature → higher overlap → smaller effective coverage → more viewpoints
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    weighted_viewpoint_count = 0.0

    for tri_idx, tri in enumerate(triangles):
        # Get triangle vertices
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]

        # Compute triangle area
        tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

        # Average curvature of triangle vertices
        tri_curvature_norm = np.mean(curvatures_norm[tri])

        # Compute local overlap based on curvature
        local_overlap = compute_local_overlap(
            tri_curvature_norm, curvature_weight, base_overlap_ratio
        )

        # Compute effective coverage for this triangle
        local_eff_w = fov_w_m * (1.0 - local_overlap)
        local_eff_h = fov_h_m * (1.0 - local_overlap)
        local_fov_area = local_eff_w * local_eff_h

        # Add weighted contribution
        weighted_viewpoint_count += tri_area / local_fov_area

    estimated_viewpoints = int(np.ceil(weighted_viewpoint_count * target_coverage))

    # Compute statistics for reporting
    avg_curvature = np.mean(curvatures)
    avg_overlap = np.mean([
        compute_local_overlap(c, curvature_weight, base_overlap_ratio)
        for c in curvatures_norm
    ])
    min_overlap = compute_local_overlap(0.0, curvature_weight, base_overlap_ratio)
    max_overlap = compute_local_overlap(1.0, curvature_weight, base_overlap_ratio)

    print(f"\nAutomatic viewpoint estimation (adaptive overlap):")
    print(f"  Surface area: {surface_area * 1e6:.2f} mm²")
    print(f"  FOV: {fov_w_mm:.1f} × {fov_h_mm:.1f} mm")
    print(f"  Curvature weight: {curvature_weight:.2f}")
    print(f"  Adaptive overlap range: {min_overlap*100:.0f}% (flat) → {max_overlap*100:.0f}% (curved)")
    print(f"  Average overlap: {avg_overlap*100:.1f}%")
    print(f"  Average curvature: {avg_curvature:.4f}")
    print(f"  Max curvature: {max_curvature:.4f}")
    print(f"  Basic estimate (uniform): {basic_estimate} viewpoints")
    print(f"  Adaptive estimate: {estimated_viewpoints} viewpoints")
    print(f"  Increase: {((estimated_viewpoints / basic_estimate - 1) * 100):.1f}%")

    return estimated_viewpoints


def sample_points_adaptive_poisson(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int,
    curvature_weight: float = 0.5,
    base_overlap_ratio: float = config.CAMERA_OVERLAP_RATIO,
    num_strata: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points adaptively using Poisson disk sampling stratified by surface curvature

    This approach combines the benefits of:
    1. Poisson disk sampling: Maintains minimum distance between samples (blue noise)
    2. Curvature-adaptive density: More samples in high-curvature regions

    Method:
    - Partition mesh into curvature strata (low/medium/high)
    - Apply Poisson disk sampling independently to each stratum
    - Allocate more samples to high-curvature strata
    - Merge samples from all strata

    Args:
        mesh: Open3D triangle mesh
        num_points: Target total number of points to sample
        curvature_weight: Weight for curvature influence on sample allocation (0-1)
            - 0.0: Uniform allocation across strata
            - 1.0: Maximum bias toward high-curvature strata
        base_overlap_ratio: Baseline overlap ratio (affects sample allocation)
        num_strata: Number of curvature strata (default: 3 for low/medium/high)

    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
    """
    print(f"Sampling {num_points} points using curvature-stratified Poisson disk sampling...")
    print(f"  Curvature weight: {curvature_weight:.2f}")
    print(f"  Number of strata: {num_strata}")

    # Compute curvature
    curvatures = compute_surface_curvature(mesh)
    max_curvature = np.max(curvatures)
    curvatures_norm = curvatures / (max_curvature + 1e-8)

    # Get mesh data
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # Define curvature strata
    if num_strata == 3:
        strata_bounds = [(0.0, 0.33, 'low'), (0.33, 0.67, 'medium'), (0.67, 1.0, 'high')]
    elif num_strata == 5:
        strata_bounds = [
            (0.0, 0.2, 'very_low'),
            (0.2, 0.4, 'low'),
            (0.4, 0.6, 'medium'),
            (0.6, 0.8, 'high'),
            (0.8, 1.0, 'very_high')
        ]
    else:
        # Generic strata
        strata_bounds = [(i/num_strata, (i+1)/num_strata, f'stratum_{i}')
                         for i in range(num_strata)]

    # Compute triangle areas and average curvature
    tri_areas = []
    tri_curvatures_norm = []

    for tri in triangles:
        v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
        tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        tri_areas.append(tri_area)

        # Average curvature of triangle vertices
        tri_curv_norm = np.mean(curvatures_norm[tri])
        tri_curvatures_norm.append(tri_curv_norm)

    tri_areas = np.array(tri_areas)
    tri_curvatures_norm = np.array(tri_curvatures_norm)
    total_area = np.sum(tri_areas)

    # Allocate samples to each stratum
    all_points = []
    all_normals = []

    print(f"\nStratum allocation:")

    for low, high, label in strata_bounds:
        # Find triangles in this stratum
        mask = (tri_curvatures_norm >= low) & (tri_curvatures_norm < high)

        if not np.any(mask):
            print(f"  {label} [{low:.2f}-{high:.2f}]: No triangles, skipping")
            continue

        stratum_area = np.sum(tri_areas[mask])
        area_ratio = stratum_area / total_area

        # Compute sample allocation for this stratum
        # Base allocation: proportional to area
        # Curvature bonus: proportional to curvature level
        mid_curvature = (low + high) / 2
        curvature_factor = 1.0 + curvature_weight * mid_curvature

        # Number of samples for this stratum
        stratum_samples = int(num_points * area_ratio * curvature_factor)
        stratum_samples = max(1, stratum_samples)  # At least 1 sample

        print(f"  {label} [{low:.2f}-{high:.2f}]: {stratum_samples} samples "
              f"(area: {area_ratio*100:.1f}%, factor: {curvature_factor:.2f})")

        # Extract triangles for this stratum
        stratum_tri_indices = np.where(mask)[0]

        # Create submesh for this stratum
        # Need to create new mesh with selected triangles
        stratum_triangles = triangles[mask]

        # Find unique vertices used by these triangles
        unique_vertex_indices = np.unique(stratum_triangles.flatten())

        # Create mapping from old to new vertex indices
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertex_indices)}

        # Create new mesh
        stratum_mesh = o3d.geometry.TriangleMesh()
        stratum_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_vertex_indices])

        # Remap triangle indices
        new_triangles = np.array([[old_to_new[v] for v in tri] for tri in stratum_triangles])
        stratum_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)

        # Compute normals for submesh
        stratum_mesh.compute_vertex_normals()
        stratum_mesh.compute_triangle_normals()

        # Sample points from this stratum using Poisson disk
        stratum_points = np.empty((0, 3), dtype=np.float32)
        stratum_normals = np.empty((0, 3), dtype=np.float32)
        try:
            stratum_pcd = stratum_mesh.sample_points_poisson_disk(
                number_of_points=stratum_samples,
                init_factor=5
            )

            # Get points and normals
            stratum_points = np.asarray(stratum_pcd.points, dtype=np.float32)
            if len(stratum_points) == 0:
                raise RuntimeError("Poisson disk sampling returned 0 points")

            # Estimate normals if not present
            if not stratum_pcd.has_normals():
                stratum_pcd.estimate_normals()

            stratum_normals = np.asarray(stratum_pcd.normals, dtype=np.float32)

        except Exception as e:
            print(f"    ⚠ Warning: Poisson disk sampling failed for {label}: {e}")
            print(f"    → Falling back to uniform sampling")

            # Fallback: uniform sampling with extra safeguard against empty point clouds
            stratum_points, stratum_normals = sample_points_uniform_with_normals(
                stratum_mesh, stratum_samples
            )

        if len(stratum_points) == 0:
            print(f"    ⚠ Warning: No samples generated for {label} stratum, skipping")
            continue

        all_points.append(stratum_points)
        all_normals.append(stratum_normals)

        print(f"    → Sampled {len(stratum_points)} points")


    # Merge all samples
    if len(all_points) == 0:
        print("  ⚠ Warning: No samples generated, falling back to uniform Poisson disk")
        try:
            fallback_pcd = mesh.sample_points_poisson_disk(
                number_of_points=num_points,
                init_factor=5
            )
            fallback_points = np.asarray(fallback_pcd.points, dtype=np.float32)
            if len(fallback_points) == 0:
                raise RuntimeError("Poisson disk fallback returned 0 points")

            if not fallback_pcd.has_normals():
                fallback_pcd.estimate_normals()

            fallback_normals = np.asarray(fallback_pcd.normals, dtype=np.float32)
        except Exception as e:
            print(f"    ⚠ Poisson disk fallback failed: {e}")
            print("    → Using uniform sampling instead")
            fallback_points, fallback_normals = sample_points_uniform_with_normals(
                mesh, num_points
            )
        else:
            return (
                fallback_points,
                fallback_normals
            )

        if len(fallback_points) == 0:
            print("    ⚠ Unable to sample any points from mesh")
        return fallback_points, fallback_normals

    points = np.vstack(all_points)
    normals = np.vstack(all_normals)

    print(f"\nTotal sampled: {len(points)} points from {len(all_points)} strata")

    # If we got too many or too few samples, adjust
    if len(points) != num_points:
        print(f"  Adjusting from {len(points)} to {num_points} points...")

        if len(points) > num_points:
            # Randomly downsample
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
            normals = normals[indices]
        else:
            # Need more samples - add uniform samples to fill gap
            gap = num_points - len(points)
            print(f"  Adding {gap} uniform samples to reach target")
            extra_points, extra_normals = sample_points_uniform_with_normals(mesh, gap)

            points = np.vstack([points, extra_points])
            normals = np.vstack([normals, extra_normals])

    print(f"Final count: {len(points)} points")
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

    # Add coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5,  # 50mm frame size
        origin=[0, 0, 0]
    )
    geometries.append(coord_frame)

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
    print("  Coordinate frame: X (red), Y (green), Z (blue)")
    print("  Green spheres: viewpoint positions")
    print("  Red arrows: camera viewing directions")

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=720
    )


def main():
    parser = argparse.ArgumentParser(
        description='FOV-based Viewpoint Sampling for Vision Inspection'
    )

    # Input/Output
    parser.add_argument('--object_name', type=str, default=None,
                        help='Object name for automatic path generation (e.g., "glass", "phone"). '
                             'If provided, mesh_file and save_path will be auto-generated.')
    parser.add_argument('--mesh_file', type=str, default=None,
                        help='Path to mesh file (.obj) - Z-up coordinate system. '
                             'Required if --object_name is not provided.')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save viewpoints as HDF5 file (default: data/{object_name}/viewpoint/{num_points}/viewpoints.h5)')
    
    # Camera specifications (defaults from common.config)
    parser.add_argument('--fov_width', type=float, default=config.CAMERA_FOV_WIDTH_MM,
                        help=f'Field of view width in mm (default: {config.CAMERA_FOV_WIDTH_MM})')
    parser.add_argument('--fov_height', type=float, default=config.CAMERA_FOV_HEIGHT_MM,
                        help=f'Field of view height in mm (default: {config.CAMERA_FOV_HEIGHT_MM})')
    parser.add_argument('--working_distance', type=float, default=config.CAMERA_WORKING_DISTANCE_MM,
                        help=f'Working distance in mm (default: {config.CAMERA_WORKING_DISTANCE_MM})')
    parser.add_argument('--depth_of_field', type=float, default=config.CAMERA_DEPTH_OF_FIELD_MM,
                        help=f'Depth of field in mm (default: {config.CAMERA_DEPTH_OF_FIELD_MM})')
    parser.add_argument('--overlap', type=float, default=config.CAMERA_OVERLAP_RATIO,
                        help=f'Overlap ratio between views (default: {config.CAMERA_OVERLAP_RATIO} for {config.CAMERA_OVERLAP_RATIO*100:.0f}%%)')

    # Sampling parameters
    parser.add_argument('--num_points', type=int, default=None,
                        help='Number of viewpoints to sample (default: auto-calculate based on mesh and camera FOV)')
    parser.add_argument('--adaptive_sampling', action='store_true',
                        help='Use adaptive sampling based on surface curvature (for point distribution)')
    parser.add_argument('--curvature_weight', type=float, default=0.5,
                        help='Curvature influence on viewpoint count and sampling density (0-1, default: 0.5).\n'
                             '  0.0: Uniform overlap (25%% everywhere), ignores curvature\n'
                             '  0.5: Moderate adaptive overlap (10-25%%), balanced approach\n'
                             '  1.0: Aggressive adaptive overlap (10-40%%), maximum differentiation\n'
                             '  Higher values → more viewpoints in high-curvature regions (edges, corners)')
    parser.add_argument('--use_poisson_disk', action='store_true',
                        help='Use Poisson disk sampling for adaptive mode (maintains minimum distance between samples, blue noise distribution)')
    parser.add_argument('--check_dof', action='store_true',
                        help='Check depth of field constraints')
    parser.add_argument('--remove_invalid_dof', action='store_true',
                        help='Remove viewpoints that violate DOF constraints')
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize viewpoints with Open3D')


    args = parser.parse_args()

    # Validate and resolve paths
    if args.object_name is None and args.mesh_file is None:
        parser.error("Either --object_name or --mesh_file must be provided")

    # Determine mesh file path
    if args.object_name:
        if args.mesh_file is None:
            # Use target mesh (inspection surface) for viewpoint sampling
            args.mesh_file = str(config.get_mesh_path(args.object_name, mesh_type="target"))
            print(f"Using auto-generated mesh path: {args.mesh_file}")
            print(f"  → Using 'target' mesh (inspection surface only)")

    # Create camera spec
    camera_spec = CameraSpec(
        fov_width_mm=args.fov_width,
        fov_height_mm=args.fov_height,
        working_distance_mm=args.working_distance,
        depth_of_field_mm=args.depth_of_field,
        overlap_ratio=args.overlap
    )

    print_section_header("FOV-BASED VIEWPOINT SAMPLING", width=60)
    print(camera_spec)
    print("=" * 60)

    # Load mesh
    mesh, surface_area = load_mesh_file(args.mesh_file)

    # Determine number of viewpoints
    if args.num_points is not None:
        # User specified number of points
        num_points = args.num_points
        print(f"\n✓ Using user-specified number of viewpoints: {num_points}")
    else:
        # Automatically estimate required number of viewpoints
        num_points = estimate_required_viewpoints(
            mesh, camera_spec,
            target_coverage=1.0,
            curvature_weight=args.curvature_weight
        )

    # Sample surface points
    # Use curvature-stratified Poisson disk sampling
    surface_points, surface_normals = sample_points_adaptive_poisson(
        mesh,
        num_points,
        curvature_weight=args.curvature_weight,
        base_overlap_ratio=camera_spec.overlap_ratio,
        num_strata=3
    )
    
    # Compute viewpoints
    wd_meters = camera_spec.get_working_distance_m()
    print(f"\nComputing viewpoints...")
    print(f"  Working distance: {args.working_distance} mm = {wd_meters} m")
    print(f"  Offsetting surface points by {wd_meters} m along normals")
    viewpoints = compute_viewpoints_from_surface(surface_points, surface_normals, camera_spec)
    print(f"  Generated {len(viewpoints)} viewpoints")

    # Determine save path
    if args.save_path is None:
        # Auto-generate save path with num_points subdirectory
        num_points = len(viewpoints)
        if args.object_name:
            # New structure: data/{object_name}/viewpoint/{num_points}/viewpoints.h5
            args.save_path = str(config.get_viewpoint_path(args.object_name, num_points))
        else:
            # Fallback to old structure for backward compatibility
            output_dir = f'data/viewpoint/{num_points}'
            os.makedirs(output_dir, exist_ok=True)
            args.save_path = f'{output_dir}/viewpoints.h5'
        print(f"\nAuto-generated save path: {args.save_path}")

    # Save to HDF5
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

        print_section_header("COORDINATE CONVERSION FOR HDF5 SAVE", width=60)
        print("Converting viewpoint positions → surface positions")
        print_key_value("Working distance", f"{wd*1000:.1f} mm = {wd:.6f} m")
        print("  Forward:  viewpoint_pos = surface_pos + surface_normal × WD")
        print("  Inverse:  surface_pos = viewpoint_pos - surface_normal × WD")
        if len(viewpoints) > 0:
            print(f"\nVerification (first viewpoint):")
            print(f"  Original viewpoint pos:  {first_vp.position}")
            print(f"  Recovered surface pos:   {first_surface_pos}")
            print(f"  Recomputed viewpoint:    {recomputed_vp_pos}")
            print_key_value("Position error", f"{position_error*1000:.6f} mm")
            if position_error > 1e-6:
                print("  ⚠️  WARNING: Conversion error detected!")
            else:
                print("  ✓ Conversion verified (error < 1 μm)")
        print(f"\nSaving {len(surface_positions)} surface positions to HDF5")
        print()

        # Save using simplified viewpoints format
        save_viewpoints_hdf5(
            positions=surface_positions,      # Surface positions (not camera positions)
            normals=surface_normals,       # Surface normals (not camera directions)
            output_path=args.save_path,
            camera_spec=camera_spec.to_dict(),
        )

    # Visualize if requested
    if args.visualize:
        visualize_viewpoints(mesh, viewpoints, camera_spec,
                           title=f"FOV-based Viewpoints ({len(viewpoints)} views)")

    print("\nDone!")


if __name__ == "__main__":
    main()
