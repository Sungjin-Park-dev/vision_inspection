#!/usr/bin/env python3
"""
Mesh Preprocessing for Vision Inspection

Separates multi-material OBJ files by material for vision inspection pipeline.
Target materials (e.g., green inspection surfaces) are extracted for viewpoint
sampling, while the full mesh remains available for collision checking.

Usage:
    # By material name
    omni_python preprocess_mesh.py \\
        --input data/object/sample_step_scaled.obj \\
        --material-name "Opaque(0,255,0).001" \\
        --visualize

    # By RGB color (recommended)
    omni_python preprocess_mesh.py \\
        --input data/object/sample_step_scaled.obj \\
        --material-rgb "0,255,0" \\
        --color-tolerance 5.0 \\
        --output data/object/target_surface.ply

Coordinate system: Z-up (Isaac Sim / URDF / Pinocchio convention)
"""

import os
import sys
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import copy

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Third-party imports
import trimesh
import open3d as o3d

# Common utilities
from common import config
from common.cli_utils import (
    print_section_header,
    print_key_value,
    print_success,
    print_warning,
    print_error
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for mesh preprocessing

    Attributes:
        input_path: Input OBJ file path
        object_name: Object name for auto-path generation
        output_path: Output PLY file path (auto-generated if None)
        material_name: Exact material name to extract
        material_rgb: RGB color string "R,G,B" to match
        color_tolerance: RGB distance tolerance for color matching
        visualize: Show meshes in Open3D viewer
        no_save: Skip saving output file
    """
    # Input/Output
    input_path: Optional[str] = None
    object_name: Optional[str] = None
    output_path: Optional[str] = None

    # Material selection (exactly one must be set)
    material_name: Optional[str] = None
    material_rgb: Optional[str] = None

    # Options
    color_tolerance: float = 5.0
    visualize: bool = False
    no_save: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'PreprocessConfig':
        """Create configuration from command line arguments"""
        return cls(
            input_path=args.input,
            object_name=args.object_name,
            output_path=args.output,
            material_name=args.material_name,
            material_rgb=args.material_rgb,
            color_tolerance=args.color_tolerance,
            visualize=args.visualize,
            no_save=args.no_save,
        )

    def resolve_paths(self) -> None:
        """Resolve input/output paths from object_name if needed"""
        # Generate input path from object_name
        if self.input_path is None and self.object_name:
            # Try multiple possible source files
            possible_sources = ["target.obj", "source.obj", f"{self.object_name}.obj"]
            for source_name in possible_sources:
                candidate = config.get_mesh_path(self.object_name, source_name)
                if candidate.exists():
                    self.input_path = str(candidate)
                    break

            if self.input_path is None:
                # Default to source.obj even if it doesn't exist (will error later)
                self.input_path = str(config.get_mesh_path(self.object_name, "source.obj"))

        # Generate output path
        if self.output_path is None:
            if self.object_name:
                # New structure: data/{object_name}/mesh/target.ply
                self.output_path = str(config.get_mesh_path(self.object_name, "target.ply"))
            elif self.input_path:
                # Fallback: same directory as input
                self.output_path = auto_generate_output_path(self.input_path)

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Must have input source
        if self.input_path is None and self.object_name is None:
            errors.append("Either --input or --object_name must be provided")

        # Must have exactly one material selection method
        if self.material_name is None and self.material_rgb is None:
            errors.append("Either --material-name or --material-rgb must be provided")

        if self.material_name is not None and self.material_rgb is not None:
            errors.append("Cannot specify both --material-name and --material-rgb")

        return errors


# ============================================================================
# Material Parsing Functions
# ============================================================================

def parse_mtl_file(mtl_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse MTL file and extract material properties

    Args:
        mtl_path: Path to MTL file

    Returns:
        Dictionary mapping material name to properties:
        {
            "MaterialName": {
                "Kd": np.array([r, g, b]),  # diffuse color (0.0-1.0)
                "Ka": np.array([r, g, b]),  # ambient color
                "Ks": np.array([r, g, b]),  # specular color
            }
        }
    """
    materials = {}
    current_material = None

    with open(mtl_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            # New material definition
            if parts[0] == 'newmtl':
                current_material = ' '.join(parts[1:])  # Material name may have spaces
                materials[current_material] = {}

            # Diffuse color (Kd)
            elif parts[0] == 'Kd' and current_material:
                try:
                    kd = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    materials[current_material]['Kd'] = kd
                except (IndexError, ValueError):
                    print_warning(f"Invalid Kd values in material '{current_material}'")

            # Ambient color (Ka)
            elif parts[0] == 'Ka' and current_material:
                try:
                    ka = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    materials[current_material]['Ka'] = ka
                except (IndexError, ValueError):
                    pass

            # Specular color (Ks)
            elif parts[0] == 'Ks' and current_material:
                try:
                    ks = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    materials[current_material]['Ks'] = ks
                except (IndexError, ValueError):
                    pass

    return materials


def parse_obj_material_usage(obj_path: str) -> Tuple[Dict[int, str], str]:
    """
    Parse OBJ file and map triangle indices to materials

    Args:
        obj_path: Path to OBJ file

    Returns:
        Tuple of:
        - triangle_materials: {triangle_idx: material_name}
        - mtl_file: MTL file path from mtllib directive
    """
    triangle_materials = {}
    mtl_file = None
    current_material = None
    face_idx = 0

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            # MTL library reference
            if parts[0] == 'mtllib':
                mtl_file = ' '.join(parts[1:])

            # Material assignment
            elif parts[0] == 'usemtl':
                current_material = ' '.join(parts[1:])

            # Face definition
            elif parts[0] == 'f':
                # Extract vertex indices (handle v, v/vt, v/vt/vn, v//vn formats)
                vertices = []
                for vertex_str in parts[1:]:
                    vertex_idx = vertex_str.split('/')[0]
                    vertices.append(int(vertex_idx))

                # Triangulate if quad (split into 2 triangles)
                if len(vertices) == 3:
                    triangle_materials[face_idx] = current_material
                    face_idx += 1
                elif len(vertices) == 4:
                    # Triangle 1: v0, v1, v2
                    triangle_materials[face_idx] = current_material
                    face_idx += 1
                    # Triangle 2: v0, v2, v3
                    triangle_materials[face_idx] = current_material
                    face_idx += 1
                elif len(vertices) > 4:
                    # Fan triangulation for n-gons
                    for i in range(1, len(vertices) - 1):
                        triangle_materials[face_idx] = current_material
                        face_idx += 1

    if mtl_file is None:
        raise ValueError(f"No MTL file found in {obj_path}")

    return triangle_materials, mtl_file


def rgb_to_kd(rgb: Tuple[int, int, int]) -> np.ndarray:
    """
    Convert RGB (0-255) to Kd (0.0-1.0)

    Args:
        rgb: RGB tuple (0-255 range)

    Returns:
        Kd array (0.0-1.0 range)
    """
    return np.array([rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0])


def kd_to_rgb(kd: np.ndarray) -> Tuple[int, int, int]:
    """
    Convert Kd (0.0-1.0) to RGB (0-255)

    Args:
        kd: Kd array (0.0-1.0 range)

    Returns:
        RGB tuple (0-255 range)
    """
    return (
        int(np.round(kd[0] * 255)),
        int(np.round(kd[1] * 255)),
        int(np.round(kd[2] * 255))
    )


def match_material_by_color(
    materials: Dict[str, Dict[str, Any]],
    target_rgb: Tuple[int, int, int],
    tolerance: float = 5.0
) -> List[str]:
    """
    Find materials matching RGB color within tolerance

    Args:
        materials: Material dictionary from parse_mtl_file()
        target_rgb: Target RGB color (0-255 range)
        tolerance: RGB distance tolerance (default: 5.0)

    Returns:
        List of matching material names
    """
    target_array = np.array(target_rgb, dtype=float)
    matching_materials = []

    for name, props in materials.items():
        if 'Kd' not in props:
            continue

        # Convert Kd to RGB and compute distance
        material_rgb = np.array(kd_to_rgb(props['Kd']), dtype=float)
        distance = np.linalg.norm(material_rgb - target_array)

        if distance <= tolerance:
            matching_materials.append(name)

    return matching_materials


# ============================================================================
# Mesh Operations
# ============================================================================

def load_obj_with_materials(obj_path: str) -> Tuple[trimesh.Trimesh, Dict]:
    """
    Load OBJ using trimesh while preserving material information

    Args:
        obj_path: Path to OBJ file

    Returns:
        Tuple of:
        - mesh: Trimesh object
        - material_info: Dictionary with 'triangle_materials' and 'mtl_file'
    """
    print("Loading mesh with trimesh...")

    # Load mesh (force='mesh' combines multi-part meshes)
    mesh = trimesh.load(obj_path, force='mesh', process=False)

    # Parse material usage manually (Trimesh may not preserve properly)
    tri_materials, mtl_file = parse_obj_material_usage(obj_path)

    print(f"  Loaded {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles")
    print(f"  MTL file: {mtl_file}")

    return mesh, {'triangle_materials': tri_materials, 'mtl_file': mtl_file}


def separate_mesh_by_material(
    mesh: trimesh.Trimesh,
    material_map: Dict[int, str],
    target_materials: List[str]
) -> Tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """
    Split mesh into target and full meshes

    Args:
        mesh: Trimesh object
        material_map: {triangle_idx: material_name}
        target_materials: List of materials to extract

    Returns:
        Tuple of (target_mesh, full_mesh)
    """
    print(f"\nSeparating mesh by material...")
    print(f"  Target materials: {target_materials}")

    # Create boolean mask for target triangles
    target_mask = np.array([
        material_map.get(i, "") in target_materials
        for i in range(len(mesh.faces))
    ])

    num_target = np.sum(target_mask)
    num_total = len(mesh.faces)

    print(f"  Target triangles: {num_target} / {num_total} ({num_target / num_total * 100:.1f}%)")

    # Extract target faces
    target_faces = mesh.faces[target_mask]

    # Create target mesh (trimesh auto-removes unused vertices)
    target_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=target_faces,
        process=True  # Clean up and remove unused vertices
    )

    # Full mesh is just a copy
    full_mesh = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=True
    )

    print(f"  Target mesh: {len(target_mesh.vertices)} vertices, {len(target_mesh.faces)} faces")
    print(f"  Full mesh: {len(full_mesh.vertices)} vertices, {len(full_mesh.faces)} faces")

    return target_mesh, full_mesh


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Convert Trimesh to Open3D TriangleMesh

    Args:
        mesh: Trimesh object

    Returns:
        Open3D TriangleMesh
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    o3d_mesh.compute_triangle_normals()
    return o3d_mesh


# ============================================================================
# Visualization & I/O
# ============================================================================

def visualize_separated_meshes(
    target_mesh: o3d.geometry.TriangleMesh,
    full_mesh: o3d.geometry.TriangleMesh
):
    """
    Display both meshes in Open3D viewer

    Args:
        target_mesh: Target (green) mesh
        full_mesh: Full mesh (all materials)
    """
    print("\nVisualizing meshes...")

    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05,  # 50mm
        origin=[0, 0, 0]
    )

    # Show target mesh (green)
    print("  Showing target mesh (green)...")
    target_vis = copy.deepcopy(target_mesh)
    target_vis.paint_uniform_color([0.0, 1.0, 0.0])  # Green

    o3d.visualization.draw_geometries(
        [target_vis, coord_frame],
        window_name="Target Mesh (Green Material Only)",
        width=1280,
        height=720
    )

    # Show full mesh (gray)
    print("  Showing full mesh (all materials)...")
    full_vis = copy.deepcopy(full_mesh)
    full_vis.paint_uniform_color([0.7, 0.7, 0.7])  # Gray

    o3d.visualization.draw_geometries(
        [full_vis, coord_frame],
        window_name="Full Mesh (All Materials)",
        width=1280,
        height=720
    )

    print("  Visualization complete")


def print_mesh_statistics(mesh: o3d.geometry.TriangleMesh, name: str):
    """
    Print mesh statistics using cli_utils formatting

    Args:
        mesh: Open3D TriangleMesh
        name: Mesh name for header
    """
    print_section_header(f"MESH STATISTICS: {name}", width=70)

    vertices = np.asarray(mesh.vertices)
    surface_area = mesh.get_surface_area()

    print_key_value("Vertices", len(mesh.vertices))
    print_key_value("Triangles", len(mesh.triangles))
    print_key_value("Surface area", f"{surface_area * 1e6:.2f} mm²")

    # Bounding box
    coord_min = vertices.min(axis=0)
    coord_max = vertices.max(axis=0)
    coord_range = coord_max - coord_min

    print(f"\nBounding box (Z-up coordinate system):")
    print(f"  X: [{coord_min[0]:.6f}, {coord_max[0]:.6f}] m (range: {coord_range[0]:.6f} m)")
    print(f"  Y: [{coord_min[1]:.6f}, {coord_max[1]:.6f}] m (range: {coord_range[1]:.6f} m)")
    print(f"  Z: [{coord_min[2]:.6f}, {coord_max[2]:.6f}] m (range: {coord_range[2]:.6f} m) ← up")

    print("=" * 70)


def save_mesh_ply(mesh: o3d.geometry.TriangleMesh, output_path: str):
    """
    Save mesh as binary PLY format

    Args:
        mesh: Open3D TriangleMesh
        output_path: Output file path
    """
    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as binary PLY
    success = o3d.io.write_triangle_mesh(
        output_path,
        mesh,
        write_ascii=False,
        compressed=True
    )

    if not success:
        raise IOError(f"Failed to save mesh to {output_path}")


def auto_generate_output_path(input_path: str) -> str:
    """
    Generate output path from input path

    Args:
        input_path: Input OBJ file path

    Returns:
        Output PLY file path (e.g., "mesh.obj" -> "mesh_target.ply")
    """
    base_dir = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_name = f"{base_name}_target.ply"

    if base_dir:
        return os.path.join(base_dir, output_name)
    else:
        return output_name


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess multi-material OBJ for vision inspection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using object name (recommended - NEW)
  omni_python preprocess_mesh.py \\
      --object_name glass \\
      --material_rgb "0,255,0" \\
      --visualize

  # Using explicit paths
  omni_python preprocess_mesh.py \\
      --input data/object/sample_step_scaled.obj \\
      --material-name "Opaque(0,255,0).001" \\
      --output data/object/target_surface.ply
        """
    )

    # Input/Output (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--object_name',
                             help='Object name for auto-path generation (e.g., "glass", "phone")')
    input_group.add_argument('--input',
                             help='Input OBJ file path')

    parser.add_argument('--output', default=None,
                        help='Output PLY file path (auto-generated if not specified)')

    # Material selection (mutually exclusive)
    material_group = parser.add_mutually_exclusive_group(required=True)
    material_group.add_argument('--material-name',
                                help='Material name (e.g., "Opaque(0,255,0).001")')
    material_group.add_argument('--material-rgb',
                                help='RGB color (format: "R,G,B", e.g., "0,255,0")')

    parser.add_argument('--color-tolerance', type=float, default=5.0,
                        help='RGB distance tolerance for color matching (default: 5.0)')

    # Options
    parser.add_argument('--visualize', action='store_true',
                        help='Show meshes in Open3D viewer')
    parser.add_argument('--no-save', action='store_true',
                        help='Skip saving (inspection only)')

    args = parser.parse_args()

    # Create and validate config
    cfg = PreprocessConfig.from_args(args)
    errors = cfg.validate()
    if errors:
        for error in errors:
            parser.error(error)

    # Resolve paths
    cfg.resolve_paths()

    # ========================================================================
    # 1. Configuration
    # ========================================================================
    print_section_header("MESH PREPROCESSING", width=70)
    print(f"Coordinate system: Z-up (Isaac Sim / URDF / Pinocchio convention)\n")

    if cfg.object_name:
        print_key_value("Object name", cfg.object_name, width=35)
    print_key_value("Input", cfg.input_path, width=35)
    print_key_value("Output", cfg.output_path if not cfg.no_save else "(dry run)", width=35)
    print()

    if not os.path.exists(cfg.input_path):
        print_error(f"Input file not found: {cfg.input_path}")
        if cfg.object_name:
            print("\nSuggestion:")
            print(f"  Place your mesh file at: {config.get_mesh_path(cfg.object_name, 'source.obj')}")
        sys.exit(1)

    # ========================================================================
    # 2. Load mesh with trimesh
    # ========================================================================
    print_section_header("LOADING MESH", width=70)
    try:
        mesh_trimesh, material_info = load_obj_with_materials(cfg.input_path)
    except Exception as e:
        print_error(f"Failed to load mesh: {e}")
        sys.exit(1)

    # ========================================================================
    # 3. Parse MTL file
    # ========================================================================
    print_section_header("PARSING MATERIALS", width=70)
    mtl_path = os.path.join(os.path.dirname(cfg.input_path), material_info['mtl_file'])

    if not os.path.exists(mtl_path):
        print_error(f"MTL file not found: {mtl_path}")
        sys.exit(1)

    try:
        materials = parse_mtl_file(mtl_path)
    except Exception as e:
        print_error(f"Failed to parse MTL file: {e}")
        sys.exit(1)

    print(f"Found {len(materials)} materials:")
    for name, props in materials.items():
        if 'Kd' in props:
            rgb = kd_to_rgb(props['Kd'])
            print(f"  - {name}: RGB{rgb}")
        else:
            print(f"  - {name}: (no diffuse color)")

    # ========================================================================
    # 4. Determine target materials
    # ========================================================================
    print_section_header("SELECTING TARGET MATERIAL", width=70)

    if cfg.material_name:
        # By material name
        if cfg.material_name not in materials:
            print_error(f"Material '{cfg.material_name}' not found in MTL file")
            print("\nAvailable materials:")
            for name, props in materials.items():
                if 'Kd' in props:
                    rgb = kd_to_rgb(props['Kd'])
                    print(f"  - {name}: RGB{rgb}")
            sys.exit(1)

        target_materials = [cfg.material_name]
        print(f"Selected material: {cfg.material_name}")

    else:
        # By RGB color
        try:
            r, g, b = map(int, cfg.material_rgb.split(','))
            target_rgb = (r, g, b)
        except ValueError:
            print_error(f"Invalid RGB format: '{cfg.material_rgb}'")
            print("Expected format: R,G,B (e.g., '0,255,0')")
            sys.exit(1)

        target_materials = match_material_by_color(materials, target_rgb, cfg.color_tolerance)

        if not target_materials:
            print_error(f"No materials matched RGB{target_rgb}")
            print(f"Tolerance: ±{cfg.color_tolerance}")
            print("\nAll materials with distances:")
            for name, props in materials.items():
                if 'Kd' in props:
                    rgb = kd_to_rgb(props['Kd'])
                    dist = np.linalg.norm(np.array(target_rgb) - np.array(rgb))
                    print(f"  - {name}: RGB{rgb} (distance: {dist:.1f})")
            sys.exit(1)

        print(f"Target RGB: {target_rgb}")
        print(f"Tolerance: ±{cfg.color_tolerance}")
        print(f"Matched materials: {target_materials}")

    # ========================================================================
    # 5. Separate mesh
    # ========================================================================
    print_section_header("SEPARATING MESH", width=70)
    try:
        target_trimesh, full_trimesh = separate_mesh_by_material(
            mesh_trimesh,
            material_info['triangle_materials'],
            target_materials
        )
    except Exception as e:
        print_error(f"Failed to separate mesh: {e}")
        sys.exit(1)

    # Check if target mesh is empty
    if len(target_trimesh.faces) == 0:
        print_error("Target mesh is empty (no triangles matched material)")
        sys.exit(1)

    # ========================================================================
    # 6. Convert to Open3D
    # ========================================================================
    print("\nConverting to Open3D format...")
    target_mesh = trimesh_to_open3d(target_trimesh)
    full_mesh = trimesh_to_open3d(full_trimesh)
    print_success("Conversion complete")

    # ========================================================================
    # 7. Print statistics
    # ========================================================================
    print()
    print_mesh_statistics(target_mesh, "TARGET MESH (Selected Material)")
    print()
    print_mesh_statistics(full_mesh, "FULL MESH (All Materials)")

    # ========================================================================
    # 8. Visualize if requested
    # ========================================================================
    if cfg.visualize:
        print_section_header("VISUALIZATION", width=70)
        visualize_separated_meshes(target_mesh, full_mesh)

    # ========================================================================
    # 9. Save target mesh
    # ========================================================================
    if not cfg.no_save:
        print_section_header("SAVING TARGET MESH", width=70)

        output_path = cfg.output_path
        print(f"Output: {output_path}")
        print(f"Format: Binary PLY (compressed)")

        try:
            save_mesh_ply(target_mesh, output_path)
            print_success(f"Saved to {output_path}")
        except Exception as e:
            print_error(f"Failed to save mesh: {e}")
            sys.exit(1)

    # ========================================================================
    # Done
    # ========================================================================
    print_section_header("COMPLETE", width=70)
    print("Next steps:")
    if not cfg.no_save:
        print(f"  1. Verify target mesh: {cfg.output_path}")
        print(f"  2. Generate viewpoints:")
        print(f"     omni_python scripts/mesh_to_viewpoints.py \\")
        print(f"         --mesh_file {cfg.output_path} \\")
        print(f"         --num_points 100 \\")
        print(f"         --visualize")
    print("\n✓ Mesh preprocessing complete!")


if __name__ == "__main__":
    main()
