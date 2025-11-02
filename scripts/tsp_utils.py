#!/usr/bin/env python3
"""
TSP Result I/O Utilities
Functions for saving and loading TSP tour results
"""

import os
import h5py
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import torch


def save_tsp_result(
    file_path: str,
    points_original: np.ndarray,
    points_normalized: np.ndarray,
    normalization_info: dict,
    normals: np.ndarray,
    tour_indices: torch.Tensor,
    mesh_file: str,
    nn_cost: float,
    glop_cost: float,
    revision_lens: list,
    revision_iters: list,
) -> None:
    """
    Save TSP result to an HDF5 file

    HDF5 format provides excellent compatibility across NumPy versions:
    - NumPy 1.x (tested with 1.26)
    - NumPy 2.x (tested with 2.2.6)

    This ensures the TSP tour can be loaded in IsaacSim environments
    that typically use NumPy 1.26.

    Args:
        file_path: Path to save the HDF5 file (use .h5 extension)
        points_original: (N, 3) array of original point coordinates (Open3D coordinate system)
        points_normalized: (N, 3) array of normalized [0, 1] coordinates
        normalization_info: dict with 'min' and 'max' arrays for denormalization
        normals: (N, 3) array of surface normals (Open3D coordinate system)
        tour_indices: (N,) tensor of TSP tour node indices
        mesh_file: Path to the original mesh file
        nn_cost: Nearest neighbor baseline cost
        glop_cost: GLOP optimized cost
        revision_lens: List of reviser sizes used
        revision_iters: List of revision iterations used
    """
    # Convert tour indices to numpy if needed
    if isinstance(tour_indices, torch.Tensor):
        tour_indices = tour_indices.cpu().numpy()

    # Reorder coordinates according to tour
    tour_coordinates = points_original[tour_indices]

    # Calculate improvement
    improvement = (nn_cost - glop_cost) / nn_cost * 100 if nn_cost > 0 else 0.0

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save to HDF5 file
    with h5py.File(file_path, 'w') as f:
        # Create groups
        metadata_grp = f.create_group('metadata')
        points_grp = f.create_group('points')
        norm_info_grp = points_grp.create_group('normalization_info')
        tour_grp = f.create_group('tour')

        # Save metadata as attributes
        metadata_grp.attrs['num_points'] = len(points_original)
        metadata_grp.attrs['mesh_file'] = mesh_file
        metadata_grp.attrs['nn_cost'] = float(nn_cost)
        metadata_grp.attrs['glop_cost'] = float(glop_cost)
        metadata_grp.attrs['improvement'] = float(improvement)
        metadata_grp.attrs['timestamp'] = datetime.now().isoformat()
        metadata_grp.attrs['revision_lens'] = np.array(revision_lens, dtype=np.int32)
        metadata_grp.attrs['revision_iters'] = np.array(revision_iters, dtype=np.int32)

        # Save points datasets
        points_grp.create_dataset('original', data=points_original.astype(np.float32))
        points_grp.create_dataset('normalized', data=points_normalized.astype(np.float32))
        norm_info_grp.create_dataset('min', data=normalization_info["min"].astype(np.float32))
        norm_info_grp.create_dataset('max', data=normalization_info["max"].astype(np.float32))

        # Save normals
        f.create_dataset('normals', data=normals.astype(np.float32))

        # Save tour
        tour_grp.create_dataset('indices', data=tour_indices.astype(np.int32))
        tour_grp.create_dataset('coordinates', data=tour_coordinates.astype(np.float32))

    print(f"\n{'='*60}")
    print("TSP Result Saved Successfully (HDF5 format)")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Size: {os.path.getsize(file_path) / 1024:.2f} KB")
    print(f"Points: {len(points_original)}")
    print(f"NN Cost: {nn_cost:.6f}")
    print(f"GLOP Cost: {glop_cost:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"{'='*60}\n")


def load_tsp_result(file_path: str) -> Dict[str, Any]:
    """
    Load TSP result from an HDF5 file

    Args:
        file_path: Path to the HDF5 file

    Returns:
        Dictionary containing TSP result data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TSP result file not found: {file_path}")

    # Load from HDF5 file
    with h5py.File(file_path, 'r') as f:
        # Load metadata
        metadata_grp = f['metadata']
        metadata = {
            'num_points': int(metadata_grp.attrs['num_points']),
            'mesh_file': str(metadata_grp.attrs['mesh_file']),
            'nn_cost': float(metadata_grp.attrs['nn_cost']),
            'glop_cost': float(metadata_grp.attrs['glop_cost']),
            'improvement': float(metadata_grp.attrs['improvement']),
            'timestamp': str(metadata_grp.attrs['timestamp']),
            'revision_lens': metadata_grp.attrs['revision_lens'].tolist(),
            'revision_iters': metadata_grp.attrs['revision_iters'].tolist(),
        }

        # Load points
        points_grp = f['points']
        points = {
            'original': np.array(points_grp['original']),
            'normalized': np.array(points_grp['normalized']),
            'normalization_info': {
                'min': np.array(points_grp['normalization_info']['min']),
                'max': np.array(points_grp['normalization_info']['max']),
            }
        }

        # Load normals
        normals = np.array(f['normals'])

        # Load tour
        tour_grp = f['tour']
        tour = {
            'indices': np.array(tour_grp['indices']),
            'coordinates': np.array(tour_grp['coordinates']),
        }

        # Construct result dictionary
        tsp_result = {
            'metadata': metadata,
            'points': points,
            'normals': normals,
            'tour': tour,
        }

    # Validate data structure
    validate_tsp_result(tsp_result)

    print(f"\n{'='*60}")
    print("TSP Result Loaded Successfully (HDF5 format)")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Points: {tsp_result['metadata']['num_points']}")
    print(f"Mesh: {tsp_result['metadata']['mesh_file']}")
    print(f"NN Cost: {tsp_result['metadata']['nn_cost']:.6f}")
    print(f"GLOP Cost: {tsp_result['metadata']['glop_cost']:.6f}")
    print(f"Improvement: {tsp_result['metadata']['improvement']:.2f}%")
    print(f"Timestamp: {tsp_result['metadata']['timestamp']}")
    print(f"{'='*60}\n")

    return tsp_result


def validate_tsp_result(tsp_result: Dict[str, Any]) -> None:
    """
    Validate TSP result data structure

    Args:
        tsp_result: Dictionary containing TSP result data

    Raises:
        ValueError: If data structure is invalid
    """
    # Check required top-level keys
    required_keys = ["metadata", "points", "normals", "tour"]
    for key in required_keys:
        if key not in tsp_result:
            raise ValueError(f"Missing required key: {key}")

    # Check metadata
    metadata_keys = ["num_points", "mesh_file", "nn_cost", "glop_cost",
                     "improvement", "timestamp", "revision_lens", "revision_iters"]
    for key in metadata_keys:
        if key not in tsp_result["metadata"]:
            raise ValueError(f"Missing metadata key: {key}")

    # Check points
    if "original" not in tsp_result["points"]:
        raise ValueError("Missing original points")
    if "normalized" not in tsp_result["points"]:
        raise ValueError("Missing normalized points")
    if "normalization_info" not in tsp_result["points"]:
        raise ValueError("Missing normalization info")

    # Check tour
    if "indices" not in tsp_result["tour"]:
        raise ValueError("Missing tour indices")
    if "coordinates" not in tsp_result["tour"]:
        raise ValueError("Missing tour coordinates")

    # Check dimensions
    num_points = tsp_result["metadata"]["num_points"]

    points_original = tsp_result["points"]["original"]
    points_normalized = tsp_result["points"]["normalized"]
    normals = tsp_result["normals"]
    tour_indices = tsp_result["tour"]["indices"]
    tour_coords = tsp_result["tour"]["coordinates"]

    if points_original.shape != (num_points, 3):
        raise ValueError(f"Invalid original points shape: {points_original.shape}, expected ({num_points}, 3)")

    if points_normalized.shape != (num_points, 3):
        raise ValueError(f"Invalid normalized points shape: {points_normalized.shape}, expected ({num_points}, 3)")

    if normals.shape != (num_points, 3):
        raise ValueError(f"Invalid normals shape: {normals.shape}, expected ({num_points}, 3)")

    if tour_indices.shape != (num_points,):
        raise ValueError(f"Invalid tour indices shape: {tour_indices.shape}, expected ({num_points},)")

    if tour_coords.shape != (num_points, 3):
        raise ValueError(f"Invalid tour coordinates shape: {tour_coords.shape}, expected ({num_points}, 3)")

    # Check tour indices are valid permutation
    if not np.array_equal(np.sort(tour_indices), np.arange(num_points)):
        raise ValueError("Tour indices are not a valid permutation of 0 to num_points-1")

    print("TSP result data validation passed!")


def get_tour_coordinates_in_order(tsp_result: Dict[str, Any]) -> np.ndarray:
    """
    Get tour coordinates in visit order (convenience function)

    Args:
        tsp_result: Dictionary containing TSP result data

    Returns:
        (N, 3) array of coordinates in tour visit order
    """
    return tsp_result["tour"]["coordinates"]


def get_tour_normals_in_order(tsp_result: Dict[str, Any]) -> np.ndarray:
    """
    Get surface normals in tour visit order (convenience function)

    Args:
        tsp_result: Dictionary containing TSP result data

    Returns:
        (N, 3) array of normals in tour visit order
    """
    tour_indices = tsp_result["tour"]["indices"]
    normals = tsp_result["normals"]
    return normals[tour_indices]


def denormalize_coordinates(
    normalized_coords: np.ndarray,
    normalization_info: dict
) -> np.ndarray:
    """
    Denormalize coordinates from [0, 1] back to original scale

    Args:
        normalized_coords: (N, 3) array of normalized coordinates
        normalization_info: dict with 'min' and 'max' arrays

    Returns:
        (N, 3) array of denormalized coordinates
    """
    min_coords = normalization_info["min"]
    max_coords = normalization_info["max"]
    return normalized_coords * (max_coords - min_coords) + min_coords
