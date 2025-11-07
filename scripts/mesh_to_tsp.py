#!/usr/bin/env python3
"""
3D Mesh to TSP Solver (Nearest Neighbor)
Load a 3D mesh/point cloud, sample points, and solve TSP using Nearest Neighbor algorithm
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple, Optional

# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Try to import plotly for interactive plots
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go  # noqa: F401
    PLOTLY_AVAILABLE = True
except ImportError:
    go = None

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import open3d as o3d

# Import TSP utilities for saving/loading results
from tsp_utils import save_tsp_result, load_viewpoints


def read_pcd_file_simple(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple PCD file reader without Open3D dependency
    Supports ASCII format PCD files only (not binary)

    Returns:
        points: (N, 3) array
        normals: (N, 3) array (zeros if not available)
    """
    points = []
    normals = []
    has_normals = False

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read header
            header_ended = False
            data_format = 'ascii'

            for line in f:
                line = line.strip()

                if line.startswith('FIELDS'):
                    # Check if normals are present
                    fields = line.split()[1:]
                    has_normals = 'normal_x' in fields

                elif line.startswith('DATA'):
                    data_format = line.split()[1]
                    header_ended = True

                    # Check if binary format
                    if data_format.lower() == 'binary':
                        raise ValueError("Binary PCD format not supported. Please use Open3D or convert to ASCII format.")
                    continue

                elif header_ended and line:
                    # Parse data
                    values = [float(x) for x in line.split()]
                    if len(values) >= 3:
                        points.append(values[:3])
                        if has_normals and len(values) >= 6:
                            normals.append(values[3:6])
    except UnicodeDecodeError:
        raise ValueError("Binary PCD format detected. This file requires Open3D to read. Please use --random flag for testing or provide ASCII PCD file.")

    if len(points) == 0:
        raise ValueError("No points could be read from PCD file")

    points = np.array(points, dtype=np.float32)

    if has_normals and len(normals) > 0:
        normals = np.array(normals, dtype=np.float32)
    else:
        # Generate dummy normals pointing up
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0

    return points, normals


def load_mesh_file(file_path: str, num_points: int = 50) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Load mesh file (.obj) and sample points using Poisson disk sampling

    Args:
        file_path: Path to .obj file
        num_points: Number of points to sample

    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
        pcd: point cloud object (or None if Open3D not available)
    """
    print(f"Loading mesh from: {file_path}")
    mesh = o3d.io.read_triangle_mesh(file_path)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print(f"Sampling {num_points} points using Poisson disk sampling...")
    pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)

    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals()

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    print(f"Sampled {len(points)} points")
    return points, normals, pcd


def load_pcd_file(file_path: str, num_points: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Load point cloud file (.pcd) and optionally downsample

    Args:
        file_path: Path to .pcd file
        num_points: Number of points to downsample to (None = use all)

    Returns:
        points: (N, 3) array of point coordinates
        normals: (N, 3) array of surface normals
        pcd: point cloud object (or None if using fallback)
    """
    print(f"Loading point cloud from: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)

    if len(pcd.points) == 0:
        raise ValueError("No points loaded from PCD file")

    # Estimate normals if not present
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals()

    # Downsample if requested
    if num_points is not None and len(pcd.points) > num_points:
        print(f"Downsampling from {len(pcd.points)} to {num_points} points...")
        # Use random sampling for downsampling
        indices = np.random.choice(len(pcd.points), num_points, replace=False)
        pcd = pcd.select_by_index(indices)

    points = np.asarray(pcd.points, dtype=np.float32)
    normals = np.asarray(pcd.normals, dtype=np.float32)

    print(f"Loaded {len(points)} points")
    return points, normals, pcd


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


def denormalize_coordinates(points: np.ndarray, normalization_info: dict) -> np.ndarray:
    """Denormalize coordinates back to original scale"""
    min_coords = normalization_info['min']
    max_coords = normalization_info['max']
    return points * (max_coords - min_coords) + min_coords


def compute_tour_length(points: np.ndarray, tour: np.ndarray) -> float:
    """
    Compute total tour length

    Args:
        points: (N, 3) array
        tour: (N,) array or tensor of node indices

    Returns:
        length: total Euclidean distance
    """
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()

    # Reorder points according to tour
    ordered_points = points[tour]

    # Compute distances between consecutive points
    distances = np.linalg.norm(ordered_points[1:] - ordered_points[:-1], axis=1)

    # Add distance from last to first (close the loop)
    closing_distance = np.linalg.norm(ordered_points[-1] - ordered_points[0])

    total_length = distances.sum() + closing_distance
    return float(total_length)


def calc_pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    """
    Calculate pairwise Euclidean distances between all points

    Args:
        points: (N, 3) or (batch, N, 3) tensor

    Returns:
        dist: (N, N) or (batch, N, N) tensor of pairwise distances
    """
    if points.dim() == 2:
        # (N, 3) -> (N, N)
        diff = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (N, N)
    else:
        # (batch, N, 3) -> (batch, N, N)
        diff = points.unsqueeze(2) - points.unsqueeze(1)  # (batch, N, N, 3)
        dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (batch, N, N)

    return dist


def nearest_neighbor_torch(points: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """
    Vectorized Nearest Neighbor algorithm using PyTorch
    Based on the original GLOP implementation but simplified

    Args:
        points: (N, 3) tensor of point coordinates
        start_idx: Starting point index

    Returns:
        tour: (N,) tensor of node indices representing the tour
    """
    device = points.device
    n = len(points)

    # Precompute all pairwise distances
    dist = calc_pairwise_distances(points)  # (N, N)

    # Initialize
    current = torch.tensor([start_idx], dtype=torch.long, device=device)
    dist_to_start = dist[start_idx].clone()
    tour = [current]

    # Build tour greedily
    for i in range(n - 1):
        # Mark current node as visited (set distance to infinity)
        dist[:, current] = float('inf')

        # Get distances from current node to all others
        nn_dist = dist[current].squeeze(0)  # (N,)

        # Find nearest unvisited node
        current = nn_dist.argmin().unsqueeze(0)
        tour.append(current)

    # Stack tour into single tensor
    tour = torch.cat(tour)  # (N,)

    return tour


def generate_multiple_nn_tours_torch(points: torch.Tensor, num_starts: int = 10) -> list:
    """
    Generate multiple NN tours with different random starting points using PyTorch

    Args:
        points: (N, 3) tensor of point coordinates
        num_starts: Number of different starting points to try

    Returns:
        tours: List of (tour, cost) tuples, where tour is a tensor
    """
    n = len(points)
    tours = []

    # Use different starting points
    if num_starts >= n:
        start_indices = list(range(n))
    else:
        # Random selection of starting points
        torch.manual_seed(42)
        start_indices = torch.randperm(n)[:num_starts].tolist()

    for start_idx in start_indices:
        tour = nearest_neighbor_torch(points, start_idx)
        cost = compute_tour_length(points.cpu().numpy(), tour.cpu().numpy())
        tours.append((tour, cost))

    return tours


def random_insertion_torch(points: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    Vectorized Random Insertion heuristic for TSP using PyTorch (GPU-accelerated)

    Algorithm:
    1. Start with a partial tour of 3 random points (forming a triangle)
    2. Randomly select a remaining point
    3. Insert it at the position that minimizes tour length increase (vectorized calculation)
    4. Repeat until all points are in the tour

    Key optimization: All insertion positions are evaluated in parallel using GPU

    Args:
        points: (N, 3) tensor of point coordinates
        seed: Random seed for reproducibility

    Returns:
        tour: (N,) tensor of node indices representing the tour
    """
    device = points.device
    n = len(points)

    # Precompute distance matrix once (stays on GPU)
    dist_matrix = calc_pairwise_distances(points)  # (N, N)

    # Set random seed
    torch.manual_seed(seed)

    # Start with 3 random points forming initial tour
    initial_indices = torch.randperm(n, device=device)[:3]
    tour = initial_indices.clone()  # Keep as tensor for GPU operations

    # Create mask for remaining points
    remaining_mask = torch.ones(n, dtype=torch.bool, device=device)
    remaining_mask[initial_indices] = False

    # Get remaining indices and shuffle them
    remaining_indices = torch.nonzero(remaining_mask).squeeze(1)
    torch.manual_seed(seed + 1)
    perm = torch.randperm(len(remaining_indices), device=device)
    remaining_indices = remaining_indices[perm]

    # Insert remaining points one by one
    for point_idx in remaining_indices:
        tour_len = len(tour)

        # Vectorized calculation of insertion cost for ALL positions
        # Current edges: tour[i] -> tour[i+1]
        indices_before = tour  # (tour_len,)
        indices_after = torch.cat([tour[1:], tour[0:1]])  # Circular shift: (tour_len,)

        # Cost of current edges
        current_edges_cost = dist_matrix[indices_before, indices_after]  # (tour_len,)

        # Cost of new edges if we insert point_idx between tour[i] and tour[i+1]
        # New edges: tour[i] -> point_idx -> tour[i+1]
        new_edge1_cost = dist_matrix[indices_before, point_idx]  # (tour_len,)
        new_edge2_cost = dist_matrix[point_idx, indices_after]   # (tour_len,)

        # Calculate cost increase for each insertion position (vectorized!)
        cost_increases = new_edge1_cost + new_edge2_cost - current_edges_cost  # (tour_len,)

        # Find position with minimum cost increase
        best_pos = cost_increases.argmin().item()

        # Insert point at best position (tensor concatenation)
        # Insert after position best_pos
        tour = torch.cat([
            tour[:best_pos + 1],
            point_idx.unsqueeze(0),
            tour[best_pos + 1:]
        ])

    return tour


def generate_multiple_random_insertion_tours(points: torch.Tensor, num_starts: int = 10) -> list:
    """
    Generate multiple Random Insertion tours with different random seeds

    Args:
        points: (N, 3) tensor of point coordinates
        num_starts: Number of different random seeds to try

    Returns:
        tours: List of (tour, cost) tuples, where tour is a tensor
    """
    tours = []

    for seed in range(num_starts):
        tour = random_insertion_torch(points, seed=seed)
        cost = compute_tour_length(points.cpu().numpy(), tour.cpu().numpy())
        tours.append((tour, cost))

    return tours


def compute_tour_cost_torch(points: torch.Tensor, tour: torch.Tensor, dist_matrix: torch.Tensor = None) -> torch.Tensor:
    """
    Compute tour cost using PyTorch (GPU-accelerated)

    Args:
        points: (N, 3) tensor of coordinates
        tour: (N,) tensor of node indices
        dist_matrix: (N, N) optional precomputed distance matrix

    Returns:
        cost: scalar tensor
    """
    if dist_matrix is None:
        dist_matrix = calc_pairwise_distances(points)

    # Distance from each node to next node in tour
    edge_dists = dist_matrix[tour[:-1], tour[1:]]
    # Distance from last to first (close the loop)
    closing_dist = dist_matrix[tour[-1], tour[0]]

    return edge_dists.sum() + closing_dist


def two_opt_improve_torch_vectorized(points: torch.Tensor, tour: torch.Tensor, max_iterations: int = 100) -> tuple:
    """
    Fully vectorized 2-opt local search with GPU parallelism

    This implementation minimizes CPU-GPU data transfer and maximizes GPU parallelism
    by computing all possible edge swaps in parallel for each iteration.

    Args:
        points: (N, 3) tensor of point coordinates
        tour: (N,) tensor of node indices (initial tour)
        max_iterations: Maximum number of improvement iterations

    Returns:
        improved_tour: (N,) tensor of improved tour
        improved_cost: Cost of the improved tour (Python float)
    """
    device = points.device
    n = len(tour)
    best_tour = tour.clone()

    # Precompute distance matrix once (stays on GPU)
    dist_matrix = calc_pairwise_distances(points)  # (N, N)

    # Compute initial cost
    best_cost = compute_tour_cost_torch(points, best_tour, dist_matrix)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        best_improvement = 0
        best_i, best_j = -1, -1

        # Try all possible 2-opt swaps using vectorization
        for i in range(1, n - 1):
            # Create all valid j indices for this i (j must be > i+1)
            j_start = i + 2
            j_end = n

            if j_start >= j_end:
                continue

            # Vectorized computation for all j values at once
            j_indices = torch.arange(j_start, j_end, device=device)

            # Get tour indices for the 4 points involved in each swap
            idx_i_minus_1 = best_tour[i - 1]
            idx_i = best_tour[i]
            idx_j = best_tour[j_indices]  # Vector of all j positions
            idx_j_plus_1 = best_tour[(j_indices + 1) % n]  # Handle wrap-around

            # Current edges: (i-1, i) and (j, j+1)
            old_edge1 = dist_matrix[idx_i_minus_1, idx_i]  # Scalar
            old_edge2 = dist_matrix[idx_j, idx_j_plus_1]   # Vector (len = num_j)
            old_edges_cost = old_edge1 + old_edge2

            # New edges after swap: (i-1, j) and (i, j+1)
            new_edge1 = dist_matrix[idx_i_minus_1, idx_j]   # Vector
            new_edge2 = dist_matrix[idx_i, idx_j_plus_1]    # Vector
            new_edges_cost = new_edge1 + new_edge2

            # Calculate improvement for all j values (positive = improvement)
            improvements = old_edges_cost - new_edges_cost  # Vector

            # Find the best improvement in this batch
            max_imp_idx = improvements.argmax()
            max_imp_value = improvements[max_imp_idx]

            # Update best improvement found so far
            if max_imp_value > best_improvement:
                best_improvement = max_imp_value.item()
                best_i = i
                best_j = j_indices[max_imp_idx].item()

        # Apply the best swap found in this iteration
        if best_improvement > 1e-10:
            # Reverse the segment from best_i to best_j
            best_tour[best_i:best_j + 1] = best_tour[best_i:best_j + 1].flip(0)
            best_cost -= best_improvement
            improved = True
            print(f"    2-opt iteration {iteration}: cost = {best_cost:.6f} (improved by {best_improvement:.6f})")

    # Final cost computation
    final_cost = compute_tour_cost_torch(points, best_tour, dist_matrix).item()

    return best_tour, final_cost


def solve_tsp_with_heuristics_and_2opt(
    points: np.ndarray,
    algorithm: str = 'both',
    num_starts: int = 10,
    max_2opt_iterations: int = 100,
    device: str = 'cuda'
) -> Tuple[np.ndarray, float, float, str]:
    """
    Solve TSP using heuristic algorithms (NN and/or Random Insertion) + optional 2-opt

    Algorithm:
    1. Convert points to PyTorch tensor and move to GPU if available
    2. Run selected algorithm(s): nn, ri, or both
    3. Select the best initial tour
    4. Optionally apply 2-opt local search if max_2opt_iterations > 0

    Args:
        points: (N, 3) normalized coordinates (NumPy array)
        algorithm: Which algorithm to use ('nn', 'ri', or 'both')
        num_starts: Number of initial solutions to generate
        max_2opt_iterations: Maximum iterations for 2-opt (0 = skip 2-opt)
        device: 'cuda' or 'cpu'

    Returns:
        final_tour: (N,) array of node indices
        initial_cost: Best initial tour cost (before 2-opt)
        final_cost: Final tour cost (after 2-opt, or same as initial if skipped)
        algorithm_used: Name of algorithm that produced best initial tour
    """
    # Convert to PyTorch tensor and move to device
    if device == 'cuda' and torch.cuda.is_available():
        points_tensor = torch.from_numpy(points).float().cuda()
        print(f"Using GPU acceleration (CUDA)")
    else:
        points_tensor = torch.from_numpy(points).float()
        print(f"Using CPU")

    print(f"\n{'=' * 60}")
    print(f"Algorithm: {algorithm}")
    print(f"{'=' * 60}")

    all_tours = []
    algorithm_labels = []

    # Run Nearest Neighbor if requested
    if algorithm in ['nn', 'both']:
        print(f"\nGenerating {num_starts} Nearest Neighbor solutions...")
        nn_tours = generate_multiple_nn_tours_torch(points_tensor, num_starts)

        nn_costs = [cost for _, cost in nn_tours]
        print(f"  Best NN cost: {min(nn_costs):.6f}")
        print(f"  Worst NN cost: {max(nn_costs):.6f}")
        print(f"  Average NN cost: {np.mean(nn_costs):.6f}")

        all_tours.extend(nn_tours)
        algorithm_labels.extend(['NN'] * len(nn_tours))

    # Run Random Insertion if requested
    if algorithm in ['ri', 'both']:
        print(f"\nGenerating {num_starts} Random Insertion solutions...")
        ri_tours = generate_multiple_random_insertion_tours(points_tensor, num_starts)

        ri_costs = [cost for _, cost in ri_tours]
        print(f"  Best RI cost: {min(ri_costs):.6f}")
        print(f"  Worst RI cost: {max(ri_costs):.6f}")
        print(f"  Average RI cost: {np.mean(ri_costs):.6f}")

        all_tours.extend(ri_tours)
        algorithm_labels.extend(['Random Insertion'] * len(ri_tours))

    # Select the best initial tour
    all_costs = [cost for _, cost in all_tours]
    best_idx = np.argmin(all_costs)
    best_initial_tour, best_initial_cost = all_tours[best_idx]
    best_algorithm = algorithm_labels[best_idx]

    print(f"\nSelected best initial tour: {best_algorithm} (cost: {best_initial_cost:.6f})")

    # Apply 2-opt if requested
    if max_2opt_iterations > 0:
        print(f"\nApplying vectorized 2-opt local search (max {max_2opt_iterations} iterations)...")
        final_tour, final_cost = two_opt_improve_torch_vectorized(
            points_tensor, best_initial_tour, max_2opt_iterations
        )

        improvement_pct = (best_initial_cost - final_cost) / best_initial_cost * 100
        print(f"\n2-opt improvement: {best_initial_cost:.6f} -> {final_cost:.6f} ({improvement_pct:.2f}% better)")
    else:
        print(f"\nSkipping 2-opt optimization (max_2opt_iterations=0)")
        final_tour = best_initial_tour
        final_cost = best_initial_cost

    # Convert back to NumPy array for compatibility
    final_tour_np = final_tour.cpu().numpy()

    return final_tour_np, best_initial_cost, final_cost, best_algorithm




def visualize_viewpoints_only(
    points: np.ndarray,
    normals: np.ndarray,
    use_open3d: bool = True,
    use_matplotlib: bool = False,
    output_path: str = "viewpoints_visualization.png"
):
    """
    Visualize viewpoints and their normals without TSP tour

    Args:
        points: (N, 3) array of viewpoint positions
        normals: (N, 3) array of camera direction vectors
        use_open3d: If True, show interactive Open3D viewer
        use_matplotlib: If True, save matplotlib visualization
        output_path: Path for matplotlib output
    """
    print(f"\n{'='*60}")
    print(f"Visualizing {len(points)} viewpoints")
    print(f"{'='*60}")

    # Open3D visualization (interactive)
    if use_open3d:
        # Create point cloud for viewpoint positions
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.0, 1.0, 0.0])  # Green for viewpoints

        # Create line set for normals (arrows)
        arrow_length = 0.01  # 10mm arrows
        line_points = []
        line_indices = []
        line_colors = []

        for i, (pt, normal) in enumerate(zip(points, normals)):
            start_idx = len(line_points)
            line_points.append(pt)
            line_points.append(pt + normal * arrow_length)
            line_indices.append([start_idx, start_idx + 1])
            line_colors.append([1.0, 0.0, 0.0])  # Red for normals

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_points)
        line_set.lines = o3d.utility.Vector2iVector(line_indices)
        line_set.colors = o3d.utility.Vector3dVector(line_colors)

        print("\nOpen3D Visualization:")
        print("  Green points: Viewpoint positions")
        print("  Red arrows: Camera viewing directions (normals)")
        print("  Use mouse to rotate, zoom, and pan")

        o3d.visualization.draw_geometries(
            [pcd, line_set],
            window_name=f"Viewpoints ({len(points)} points)",
            width=1280,
            height=720
        )

    # Matplotlib visualization (static image)
    if use_matplotlib:
        fig = plt.figure(figsize=(16, 12))

        # Create 4 different views
        views = [
            (1, 'XY View (Top)', 0, 90),      # Top view
            (2, 'XZ View (Front)', 0, 0),     # Front view
            (3, 'YZ View (Side)', 90, 0),     # Side view
            (4, '3D View', 45, 45)             # 3D perspective
        ]

        arrow_length = 0.01  # 10mm arrows

        for subplot_idx, view_title, azim, elev in views:
            ax = fig.add_subplot(2, 2, subplot_idx, projection='3d')

            # Plot viewpoint positions
            ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                      c='green', s=50, alpha=0.8, label='Viewpoints')

            # Plot normal arrows (subsample for clarity)
            show_every = max(1, len(points) // 50)  # Show at most 50 arrows
            for i in range(0, len(points), show_every):
                pt = points[i]
                normal = normals[i]
                end_pt = pt + normal * arrow_length

                ax.plot([pt[0], end_pt[0]],
                       [pt[1], end_pt[1]],
                       [pt[2], end_pt[2]],
                       'r-', linewidth=1, alpha=0.7)

            # Set labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(view_title)

            # Set view angle
            ax.view_init(elev=elev, azim=azim)

            # Equal aspect ratio
            max_range = np.array([
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min()
            ]).max() / 2.0

            mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
            mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
            mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            if subplot_idx == 1:
                ax.legend()

        # Add overall title
        fig.suptitle(f'Viewpoints Visualization\n{len(points)} viewpoints',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved viewpoints visualization to: {output_path}")
        plt.close()


def visualize_tour(
    pcd,
    tour: torch.Tensor,
    title: str = "TSP Tour"
):
    """
    Visualize point cloud with TSP tour using Open3D

    Args:
        pcd: Open3D point cloud (or None if not available)
        tour: (N,) tensor of node indices
        title: Window title
    """
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()

    # Get points
    points = np.asarray(pcd.points)

    # Create line set for tour
    lines = []
    for i in range(len(tour)):
        lines.append([tour[i], tour[(i + 1) % len(tour)]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # Color the tour path (red)
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Color point cloud (gray)
    pcd_colored = o3d.geometry.PointCloud(pcd)
    pcd_colored.paint_uniform_color([0.7, 0.7, 0.7])

    # Visualize
    print(f"\nVisualizing {title}...")
    o3d.visualization.draw_geometries(
        [pcd_colored, line_set],
        window_name=title,
        width=1280,
        height=720
    )


def plot_tour_interactive(
    points: np.ndarray,
    tour: torch.Tensor,
    output_path: str = "tsp_tour_3d.html",
    title: str = "3D TSP Tour",
    nn_cost: Optional[float] = None,
    glop_cost: Optional[float] = None
):
    """
    Create interactive 3D TSP tour visualization using plotly

    Args:
        points: (N, 3) array of point coordinates
        tour: (N,) tensor of node indices
        output_path: Path to save the HTML file
        title: Plot title
        nn_cost: Nearest neighbor baseline cost (optional)
        glop_cost: GLOP optimized cost (optional)
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return

    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()

    # Reorder points according to tour
    tour_points = points[tour]

    # Create traces
    traces = []

    # Add all points
    traces.append(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=6, color='gray', opacity=0.6),
        name='Points',
        hovertemplate='Point %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        text=[f'{i}' for i in range(len(points))]
    ))

    # Add tour path with visit order
    tour_x = []
    tour_y = []
    tour_z = []
    for i in range(len(tour)):
        tour_x.extend([tour_points[i, 0], tour_points[(i + 1) % len(tour), 0], None])
        tour_y.extend([tour_points[i, 1], tour_points[(i + 1) % len(tour), 1], None])
        tour_z.extend([tour_points[i, 2], tour_points[(i + 1) % len(tour), 2], None])

    traces.append(go.Scatter3d(
        x=tour_x,
        y=tour_y,
        z=tour_z,
        mode='lines',
        line=dict(color='red', width=4),
        name='Tour Path',
        hoverinfo='skip'
    ))

    # Add visit order markers (show sequence numbers)
    traces.append(go.Scatter3d(
        x=tour_points[:, 0],
        y=tour_points[:, 1],
        z=tour_points[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='orange', opacity=0.8),
        text=[f'{i}' for i in range(len(tour))],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        name='Visit Order',
        hovertemplate='Visit #%{text}<br>Point %{customdata}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        customdata=tour
    ))

    # Highlight start point
    traces.append(go.Scatter3d(
        x=[tour_points[0, 0]],
        y=[tour_points[0, 1]],
        z=[tour_points[0, 2]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='diamond'),
        name='Start Point',
        hovertemplate='Start Point<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))

    # Create title text
    if nn_cost is not None and glop_cost is not None:
        improvement = (nn_cost - glop_cost) / nn_cost * 100
        title_text = f'{title}<br>{len(points)} points | NN: {nn_cost:.4f} | GLOP: {glop_cost:.4f} | Improvement: {improvement:.2f}%'
    else:
        title_text = f'{title}<br>{len(points)} points'

    # Create layout
    layout = go.Layout(
        title=dict(text=title_text, font=dict(size=18)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        showlegend=True,
        hovermode='closest',
        width=1200,
        height=900
    )

    # Create figure
    fig = go.Figure(data=traces, layout=layout)

    # Save to HTML
    fig.write_html(output_path)
    print(f"\nSaved interactive visualization to: {output_path}")
    print(f"  Open this file in a web browser to interact with the 3D plot")


def plot_tour_matplotlib(
    points: np.ndarray,
    tour: torch.Tensor,
    output_path: str = "tsp_tour_3d.png",
    title: str = "3D TSP Tour",
    nn_cost: Optional[float] = None,
    glop_cost: Optional[float] = None
):
    """
    Visualize 3D TSP tour using matplotlib and save to file

    Args:
        points: (N, 3) array of point coordinates
        tour: (N,) tensor of node indices
        output_path: Path to save the image
        title: Plot title
        nn_cost: Nearest neighbor baseline cost (optional)
        glop_cost: GLOP optimized cost (optional)
    """
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Create 4 different views
    views = [
        (1, 'XY View (Top)', 0, 90),      # Top view
        (2, 'XZ View (Front)', 0, 0),     # Front view
        (3, 'YZ View (Side)', 90, 0),     # Side view
        (4, '3D View', 45, 45)             # 3D perspective
    ]

    for subplot_idx, view_title, azim, elev in views:
        ax = fig.add_subplot(2, 2, subplot_idx, projection='3d')

        # Plot all points
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='gray', s=50, alpha=0.6, label='Points')

        # Plot tour path
        tour_points = points[tour]
        for i in range(len(tour)):
            start = tour_points[i]
            end = tour_points[(i + 1) % len(tour)]
            ax.plot([start[0], end[0]],
                   [start[1], end[1]],
                   [start[2], end[2]],
                   'r-', linewidth=2, alpha=0.7)

        # Add visit order numbers (only show every Nth to avoid clutter)
        show_every = max(1, len(tour) // 20)  # Show at most 20 labels
        for i in range(0, len(tour), show_every):
            ax.text(tour_points[i, 0], tour_points[i, 1], tour_points[i, 2],
                   f'{i}', fontsize=8, color='orange', weight='bold',
                   ha='center', va='bottom')

        # Highlight start point
        start_point = tour_points[0]
        ax.scatter([start_point[0]], [start_point[1]], [start_point[2]],
                  c='green', s=200, marker='*', label='Start', zorder=5)

        # Set labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(view_title)

        # Set view angle
        ax.view_init(elev=elev, azim=azim)

        # Equal aspect ratio
        max_range = np.array([
            points[:, 0].max() - points[:, 0].min(),
            points[:, 1].max() - points[:, 1].min(),
            points[:, 2].max() - points[:, 2].min()
        ]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if subplot_idx == 1:
            ax.legend()

    # Add overall title with costs
    if nn_cost is not None and glop_cost is not None:
        improvement = (nn_cost - glop_cost) / nn_cost * 100
        fig.suptitle(f'{title}\n{len(points)} points | NN: {nn_cost:.4f} | GLOP: {glop_cost:.4f} | Improvement: {improvement:.2f}%',
                    fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f'{title}\n{len(points)} points', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='3D Mesh to TSP Solver using NN/Random Insertion + 2-opt')
    parser.add_argument('--mesh_file', type=str,
                        default="data/object/glass_yup.obj",
                        help='Path to mesh file (.obj or .pcd). If not provided, uses random points.')
    parser.add_argument('--num_points', type=int, default=50,
                        help='Number of points to sample')
    parser.add_argument('--algorithm', type=str, default='both',
                        choices=['nn', 'ri', 'both'],
                        help='Algorithm to use: nn (Nearest Neighbor), ri(random_insertion), or both (default: both)')
    parser.add_argument('--num_starts', type=int, default=10,
                        help='Number of initial solutions to generate (default: 10)')
    parser.add_argument('--max_2opt_iterations', type=int, default=0,
                        help='Maximum number of 2-opt iterations (0 = skip 2-opt, default: 100)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on: cuda or cpu (default: cuda)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the result with Open3D')
    parser.add_argument('--plot', action='store_true',
                        help='Save 3D plot using matplotlib (works in headless mode)')
    parser.add_argument('--interactive', action='store_true',
                        help='Save interactive 3D HTML plot using plotly (rotatable in browser)')
    parser.add_argument('--output', type=str, default='tsp_tour_3d.png',
                        help='Output path for plot (default: tsp_tour_3d.png for matplotlib, .html for interactive)')
    parser.add_argument('--random', action='store_true',
                        help='Use random 3D points instead of loading from file')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save TSP result as HDF5 file (e.g., data/output/glass_50_tour.h5)')
    parser.add_argument('--use_viewpoints', action='store_true',
                        help='Load pre-computed viewpoints from HDF5 file instead of sampling from mesh')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("3D Mesh TSP Solver")
    print("=" * 60)

    # Load mesh or point cloud
    pcd = None
    normals = None
    mesh_file_path = args.mesh_file if args.mesh_file else "random_points"

    if args.use_viewpoints:
        # Load pre-computed viewpoints from HDF5
        if not args.mesh_file:
            raise ValueError("--mesh_file is required when using --use_viewpoints")

        file_ext = os.path.splitext(args.mesh_file)[1].lower()
        if file_ext != '.h5':
            raise ValueError(f"--use_viewpoints requires .h5 file, got: {file_ext}")

        print(f"Loading pre-computed viewpoints from: {args.mesh_file}")
        points, normals, metadata = load_viewpoints(args.mesh_file)
        mesh_file_path = metadata['mesh_file']

        # Note: num_points argument is ignored when using viewpoints
        if args.num_points != 50:  # 50 is the default
            print(f"Note: --num_points is ignored when using --use_viewpoints (loaded {len(points)} viewpoints)")

    elif args.random or args.mesh_file is None:
        # Generate random 3D points
        print(f"Generating {args.num_points} random 3D points in unit cube...")
        np.random.seed(42)
        points = np.random.rand(args.num_points, 3).astype(np.float32)
        normals = np.zeros_like(points)
        normals[:, 2] = 1.0  # Point up
        print(f"Generated {len(points)} random points")
    else:
        file_ext = os.path.splitext(args.mesh_file)[1].lower()

        if file_ext == '.obj':
            points, normals, pcd = load_mesh_file(args.mesh_file, args.num_points)
        elif file_ext == '.pcd':
            points, normals, pcd = load_pcd_file(args.mesh_file, args.num_points)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

    # Normalize coordinates
    print("\nNormalizing coordinates to [0, 1]...")
    normalized_points, norm_info = normalize_coordinates(points)

    print(f"Original coordinate range:")
    print(f"  X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # Solve TSP using selected heuristic algorithm(s) + optional 2-opt
    print("\n" + "=" * 60)
    print("Solving TSP")
    print("=" * 60)
    tour, initial_cost, final_cost, best_algorithm = solve_tsp_with_heuristics_and_2opt(
        normalized_points,
        args.algorithm,
        args.num_starts,
        args.max_2opt_iterations,
        args.device
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of points: {len(points)}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Number of starts: {args.num_starts}")
    print(f"Best initial algorithm: {best_algorithm}")
    print(f"Initial cost: {initial_cost:.6f}")
    print(f"Final cost: {final_cost:.6f}")
    improvement_pct = (initial_cost - final_cost) / initial_cost * 100
    print(f"Improvement: {improvement_pct:.2f}%")
    print("=" * 60)

    # Save TSP result if requested
    if args.save_path:
        if normals is None:
            # Create dummy normals if not available
            normals = np.zeros_like(points)
            normals[:, 2] = 1.0

        # Extract camera_spec from metadata if loaded from viewpoints file
        camera_spec_dict = None
        if args.use_viewpoints and 'camera_spec' in metadata:
            camera_spec_dict = metadata['camera_spec']
            print(f"\n{'='*60}")
            print("PROPAGATING CAMERA SPEC TO TSP FILE")
            print(f"{'='*60}")
            print(f"Camera spec found in viewpoints file:")
            if 'working_distance_mm' in camera_spec_dict:
                print(f"  Working distance: {camera_spec_dict['working_distance_mm']} mm")
            if 'fov_width_mm' in camera_spec_dict:
                print(f"  FOV: {camera_spec_dict['fov_width_mm']} x {camera_spec_dict.get('fov_height_mm', 'N/A')} mm")
            print(f"This will be saved to TSP file for use in run_app_v2.py")
            print(f"{'='*60}\n")

        save_tsp_result(
            file_path=args.save_path,
            points_original=points,
            points_normalized=normalized_points,
            normalization_info=norm_info,
            normals=normals,
            tour_indices=tour,
            mesh_file=mesh_file_path,
            nn_cost=initial_cost,
            glop_cost=final_cost,  # Store final cost for compatibility
            revision_lens=[],
            revision_iters=[],
            camera_spec=camera_spec_dict,  # Pass camera_spec if available
        )

    # Visualize viewpoints if loaded from HDF5 (before TSP)
    if args.use_viewpoints and (args.visualize or args.plot):
        visualize_viewpoints_only(points, normals, args.visualize, args.plot, args.output)

    # Visualize if requested
    if args.visualize and pcd is not None:
        title = f"{best_algorithm} TSP Tour ({len(points)} points)"
        visualize_tour(pcd, tour, title=title)

    # Plot with matplotlib if requested
    if args.plot:
        plot_tour_matplotlib(
            points,
            tour,
            output_path=args.output,
            title=f"3D TSP Tour ({best_algorithm}) - {len(points)} Points",
            nn_cost=initial_cost,
            glop_cost=final_cost
        )

    # Interactive plot with plotly if requested
    if args.interactive:
        # Auto-generate HTML filename if output is PNG
        if args.output.endswith('.png'):
            html_output = args.output.replace('.png', '.html')
        else:
            html_output = args.output if args.output.endswith('.html') else args.output + '.html'

        plot_tour_interactive(
            points,
            tour,
            output_path=html_output,
            title=f"3D TSP Tour ({best_algorithm}) - {len(points)} Points",
            nn_cost=initial_cost,
            glop_cost=final_cost
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
