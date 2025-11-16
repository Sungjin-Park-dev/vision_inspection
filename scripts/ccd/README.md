# FCL CCD Collision Checker

`fcl_ccd_check.cpp` is a standalone C++ implementation of `scripts/coal_check.py` that uses FCL's continuous collision detection (CCD) for robot trajectory validation.

## Overview

### Project Structure

```
scripts/ccd/
├── fcl_ccd_check.cpp      # Main C++ implementation (769 lines)
├── CMakeLists.txt         # CMake build configuration
├── compile_fcl.sh         # Build script (auto-builds FCL if needed)
├── build_fcl/             # Build output directory
│   └── fcl_ccd_check      # Compiled executable
└── README.md              # This file
```

### Key Features

- **True Continuous Collision Detection**: Uses FCL's `continuousCollide()` with `InterpMotion` for swept-volume checks between trajectory waypoints
- **Pinocchio Forward Kinematics**: Efficient FK computation for robot configurations
- **CuRobo Collision Spheres**: Loads collision geometry from YAML config
- **Mesh Support**: Handles arbitrary obstacle meshes via Assimp
- **Primitive Obstacles**: Built-in support for box obstacles (table, wall, workbench, robot mount)
- **Detailed Reports**: Generates collision reports compatible with `coal_check.py` output format

### Comparison with coal_check.py

| Feature | coal_check.py | fcl_ccd_check.cpp |
|---------|---------------|-------------------|
| Collision Detection | COAL (discrete) | FCL (continuous CCD) |
| Language | Python | C++ |
| Forward Kinematics | CuRobo | Pinocchio C++ |
| Collision Geometry | COAL primitives | FCL BVH + primitives |
| Swept Volume | ❌ (point-wise) | ✅ (continuous) |
| Performance | Moderate | High (C++, no GIL) |

## Environment Setup

### For Isaac Sim Environment

**Prerequisites:**
- Isaac Sim installed at `/isaac-sim/`
- Python 3.11 with Pinocchio already available via pip in Isaac Sim's Python

**Dependencies:**

The following are required and can be installed via apt:

```bash
sudo apt-get install libeigen3-dev libccd-dev libassimp-dev libyaml-cpp-dev
```

**Important:** `robotpkg-pinocchio` is **NOT** required in Isaac Sim environment. The build script automatically uses Pinocchio from Isaac Sim's Python packages (`/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix/`).

### For Standard Ubuntu Environment

If you're **not** using Isaac Sim, install all dependencies:

```bash
sudo apt-get install libeigen3-dev libccd-dev libassimp-dev libyaml-cpp-dev robotpkg-pinocchio
```

You may need to add the RobotPkg repository first:

```bash
sudo sh -c "echo 'deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -sc) robotpkg' >> /etc/apt/sources.list.d/robotpkg.list"
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key | sudo apt-key add -
sudo apt-get update
```

## Build Instructions

### Compile

```bash
cd /isaac-sim/curobo/vision_inspection/scripts/ccd
bash compile_fcl.sh   # Builds FCL (../../fcl) if needed, then fcl_ccd_check
```

**What the build script does:**
1. Checks if FCL is built at `../../fcl/build/`, builds it if missing
2. Configures CMake with proper paths:
   - Local FCL build directory
   - `/opt/openrobots` (if exists)
   - Isaac Sim's cmeel.prefix (if exists)
3. Compiles `fcl_ccd_check.cpp` with optimizations (`-O3`)
4. Outputs executable to `build_fcl/fcl_ccd_check`

**Build output:**

```
Build successful!
Executable: /isaac-sim/curobo/vision_inspection/scripts/ccd/build_fcl/fcl_ccd_check
```

## Usage

### Basic Usage

Run from the repository root:

```bash
cd /isaac-sim/curobo/vision_inspection
./scripts/ccd/build_fcl/fcl_ccd_check \
    --trajectory data/trajectory/675/joint_trajectory_dp.csv \
    --robot_urdf ur_description/ur20.urdf \
    --robot_config ur_description/ur20_safe.yml \
    --mesh data/object/glass_zup.obj \
    --verbose
```

### Command-Line Arguments

All arguments are optional and default to match `coal_check.py` configuration:

| Argument | Description | Default |
|----------|-------------|---------|
| `--trajectory <path>` | Joint trajectory CSV file | `data/trajectory/joint_trajectory_dp_5000_base.csv` |
| `--robot_urdf <path>` | Robot URDF file | `ur_description/ur20.urdf` |
| `--robot_config <path>` | CuRobo robot YAML config | `ur_description/ur20_safe.yml` |
| `--mesh <path>` | Obstacle mesh file (repeatable) | `data/object/glass_zup.obj` |
| `--glass_position x y z` | Glass mesh origin (meters) | `1.0 0.0 -0.13` |
| `--table_position x y z` | Table cuboid center | `1.0 0.0 -0.425` |
| `--table_dimensions x y z` | Table size | `0.6 1.0 0.5` |
| `--wall_position x y z` | Wall cuboid center | `-1.1 0.0 0.5` |
| `--wall_dimensions x y z` | Wall size | `0.1 2.2 1.0` |
| `--workbench_position x y z` | Workbench cuboid center | `0.35 -1.1 0.5` |
| `--workbench_dimensions x y z` | Workbench size | `3.0 0.1 1.0` |
| `--robot_mount_position x y z` | Robot mount center | `0.0 0.0 -0.25` |
| `--robot_mount_dimensions x y z` | Robot mount size | `0.3 0.3 0.5` |
| `--collision_margin <val>` | CCD collision margin (meters) | `0.0` |
| `--verbose` | Print progress every 100 segments | (off) |
| `-h, --help` | Show help message | - |

**Notes:**
- Paths may be absolute or relative to the repository root
- `--mesh` can be specified multiple times to add multiple obstacle meshes
- Use `-h` to see the full list of options

### Execution Pipeline

The tool automatically performs these steps:

1. **Parse Trajectory**: Reads CSV file with `time, joint_*` columns
2. **Load Robot Model**: Uses Pinocchio to load URDF and compute FK
3. **Load Collision Spheres**: Parses CuRobo YAML to extract collision geometry
4. **Load Obstacles**:
   - Mesh obstacles via Assimp (triangulated BVH)
   - Primitive box obstacles (table, wall, workbench, robot mount)
5. **CCD Checks**: For each trajectory segment (waypoint i → i+1):
   - Compute FK at both waypoints
   - Create `InterpMotion` for each collision sphere
   - Call `continuousCollide()` with Conservative Advancement solver
   - Record collision segments
6. **Generate Report**: Write results to `data/collision/<num_points>/collision_ccd.txt`

## Sample Output

### Console Output (Primitive Obstacles Only - Working Configuration)

```bash
# Command: ./build_fcl/fcl_ccd_check --trajectory data/trajectory/675/joint_trajectory_dp.csv --mesh "" --verbose
```

```
FCL CCD Collision Checker
=========================
Trajectory CSV: "/isaac-sim/curobo/vision_inspection/data/trajectory/675/joint_trajectory_dp.csv"
Robot URDF:    "/isaac-sim/curobo/vision_inspection/ur_description/ur20.urdf"
Robot config:  "/isaac-sim/curobo/vision_inspection/ur_description/ur20_safe.yml"
Meshes:        0
Collision margin: 0 m

Environment:
  Glass position:      [    1     0 -0.13]
  Table position:      [     1      0 -0.425]
  Table dimensions:    [0.6   1 0.5]
  Wall position:       [-1.1    0  0.5]
  Wall dimensions:     [0.1 2.2   1]
  Workbench position:  [0.35 -1.1  0.5]
  Workbench dimensions:[  3 0.1   1]
  Robot mount position:[    0     0 -0.25]
  Robot mount dims:    [0.3 0.3 0.5]

Loaded trajectory: 675 waypoints, 6 joints
Loading robot model with Pinocchio...
  Robot loaded: 6 DOF, 7 joints

Loading collision spheres from YAML...
  Loaded 24 collision spheres

Loading obstacle meshes...
  Glass position: [    1     0 -0.13]
  (no mesh obstacles configured)

Adding cuboid obstacles...
  Table added: pos=[     1      0 -0.425], dims=[0.6   1 0.5]
  Wall added: pos=[-1.1    0  0.5], dims=[0.1 2.2   1]
  Workbench added: pos=[0.35 -1.1  0.5], dims=[  3 0.1   1]
  Robot mount added: pos=[    0     0 -0.25], dims=[0.3 0.3 0.5]

========================================
Checking trajectory with CCD...
Total waypoints: 675
Total segments: 674
========================================
  Progress: 100/674 segments checked
  Progress: 200/674 segments checked
  Progress: 300/674 segments checked
  Progress: 400/674 segments checked
  Progress: 500/674 segments checked
  Progress: 600/674 segments checked
========================================
CCD Collision Check Results
========================================
Total segments checked: 674
Total CCD queries: 64704
Collision segments: 0
Collision-free segments: 674
Collision rate: 0.00%
Check time: 0.05 seconds
========================================
Report saved to: /isaac-sim/curobo/vision_inspection/data/collision/675/collision_ccd.txt
```

**Note:** With mesh obstacles (e.g., `glass_zup.obj`), the program crashes with segmentation fault due to FCL's Sphere-BVH CCD bug. See Troubleshooting section for details.

### Report File Format

The collision report (`data/collision/<num_points>/collision_ccd.txt`) contains:

```
=== CCD Collision Report @ 2025-11-16 12:34:56 ===
Trajectory: data/trajectory/675/joint_trajectory_dp.csv
Robot URDF: ur_description/ur20.urdf
Robot config: ur_description/ur20_safe.yml
Obstacle meshes:
Collision margin: 0
CCD enabled: true (InterpMotion with Conservative Advancement)

Total waypoints: 675
Total segments: 674
Total CCD queries: 64704
Collision segments: 0
Collision-free segments: 674
Collision rate (%): 0.00

Collision segment indices:
```

The collision segment list (first 50 entries) is printed to console and appended to the report for downstream analysis.

**Note:** Example above shows output with primitive obstacles only (no mesh). Mesh-based CCD currently unavailable due to FCL bug.

## Code Architecture

### Main Components

**`fcl_ccd_check.cpp`** (769 lines) contains:

#### 1. Configuration Structures

```cpp
struct Config {
    Vec3 glass_position;
    Vec3 table_position;
    Vec3 table_dimensions;
    Vec3 wall_position;
    Vec3 wall_dimensions;
    Vec3 workbench_position;
    Vec3 workbench_dimensions;
    Vec3 robot_mount_position;
    Vec3 robot_mount_dimensions;
    double collision_margin;
};

struct CollisionSphere {
    Vec3 center;
    double radius;
};
```

#### 2. FCLCCDCollisionChecker Class

**Key Members:**
- `pinocchio::Model robot_model_` - Robot kinematic model
- `pinocchio::Data robot_data_` - FK computation cache
- `std::map<std::string, std::vector<CollisionSphere>> collision_spheres_` - Per-link collision geometry
- `std::vector<std::shared_ptr<CollisionObject<Scalar>>> obstacle_objects_` - Static obstacles
- `Config config_` - Environment configuration

**Key Methods:**

| Method | Line | Description |
|--------|------|-------------|
| `loadCollisionSpheresFromYAML()` | 192 | Parse CuRobo YAML to extract collision spheres |
| `loadObstacleMeshes()` | 238 | Load and triangulate mesh obstacles with Assimp |
| `addCuboidObstacles()` | 302 | Create FCL box primitives for table/wall/etc |
| `createRobotCollisionObjects()` | 326 | Compute FK and place spheres in world frame |
| `checkCollisionSingleConfig()` | 374 | Discrete collision check at single configuration |
| `checkSegmentCCD()` | 397 | **Core CCD logic** for trajectory segment |
| `checkTrajectory()` | 462 | Main loop over all trajectory segments |
| `saveCollisionReport()` | 506 | Write results to text file |

#### 3. CCD Algorithm Implementation (Line 397-457)

```cpp
bool checkSegmentCCD(const VecX& q_start, const VecX& q_end, int segment_idx) {
    // 1. Get robot collision objects at start and end
    auto robot_start = createRobotCollisionObjects(q_start);
    auto robot_end = createRobotCollisionObjects(q_end);

    // 2. For each robot sphere
    for (size_t i = 0; i < robot_start.size(); ++i) {
        Transform3<Scalar> tf_start = robot_start[i]->getTransform();
        Transform3<Scalar> tf_end = robot_end[i]->getTransform();

        // 3. Create linear interpolation motion
        auto motion_robot = std::make_shared<InterpMotion<Scalar>>(tf_start, tf_end);

        // 4. For each obstacle
        for (const auto& obstacle_obj : obstacle_objects_) {
            // 5. Configure CCD request
            ContinuousCollisionRequest<Scalar> request;
            request.ccd_motion_type = CCDM_LINEAR;
            request.ccd_solver_type = CCDC_CONSERVATIVE_ADVANCEMENT;
            request.num_max_iterations = 10;
            request.toc_err = 0.0001;

            // 6. Perform continuous collision detection
            continuousCollide(
                robot_start[i]->collisionGeometry().get(), motion_robot.get(),
                obstacle_obj->collisionGeometry().get(), motion_obstacle.get(),
                request, result
            );

            // 7. Record collision if detected
            if (result.is_collide) {
                collision_segments_.push_back(segment_idx);
                return true;  // Early exit on first collision
            }
        }
    }
    return false;
}
```

#### 4. Utility Functions

| Function | Line | Description |
|----------|------|-------------|
| `resolvePath()` | 97 | Convert relative paths to absolute |
| `parseVec3Arg()` | 112 | Parse CLI vector arguments |
| `printUsage()` | 124 | Display help message |
| `loadTrajectoryCSV()` | 583 | Read joint trajectory from CSV |
| `main()` | 657 | CLI parsing and program entry point |

### Dependencies

**External Libraries:**
- **FCL** (0.6+): Collision detection primitives and CCD
- **Pinocchio** (3.x): Robot kinematics and URDF parsing
- **Eigen3** (3.0.5+): Linear algebra
- **Assimp**: Mesh loading and triangulation
- **yaml-cpp**: YAML configuration parsing

**Standard Library:**
- `<filesystem>`: Path manipulation
- `<chrono>`: Timing measurements
- `<fstream>`, `<sstream>`: File I/O and string parsing

## Technical Details

### Continuous Collision Detection (CCD)

**Algorithm:** Conservative Advancement

- **Motion Model**: `InterpMotion<Scalar>` - Linear interpolation between start/end transforms
- **Solver**: `CCDC_CONSERVATIVE_ADVANCEMENT` - Iterative refinement to find time-of-contact
- **Convergence**: Maximum 10 iterations, TOC error tolerance 0.0001

**Advantages over Discrete Checking:**
- **No tunneling**: Detects fast-moving objects passing through thin obstacles
- **Swept volume**: Checks entire motion path, not just endpoints
- **Exact TOC**: Returns time-of-contact for collision response

### FCL vs COAL API Differences

This implementation uses **FCL** headers but may link against **COAL** (FCL fork) via Pinocchio:

| Feature | FCL | COAL/hpp-fcl |
|---------|-----|--------------|
| CCD Support | ✅ `continuousCollide()` | ✅ Same API |
| `security_margin` field | ❌ Not supported | ✅ Supported |
| Namespace | `fcl::` | `coal::` (or `hpp::fcl::`) |

**Resolution in this code:**
- Lines 381, 429: `security_margin` assignment is commented out for FCL compatibility
- The `collision_margin` parameter is accepted but not used in FCL mode
- If linking against COAL, uncommenting these lines would enable margin-based collision

### Collision Sphere Geometry

Robot collision geometry is loaded from CuRobo YAML format:

```yaml
robot_cfg:
  kinematics:
    collision_spheres:
      link_1:
        - center: [0.0, 0.0, 0.1]
          radius: 0.05
        - center: [0.0, 0.0, 0.2]
          radius: 0.05
```

**Processing:**
1. Parse YAML to extract per-link spheres
2. At each waypoint, compute FK to get link transforms
3. Transform sphere centers from link frame to world frame
4. Create FCL `Sphere<Scalar>` collision objects
5. Attach to `InterpMotion` for swept-volume checking

### BVH Triangle Meshes

Obstacle meshes are converted to BVH (Bounding Volume Hierarchy) for efficient collision queries:

```cpp
auto bvh_model = std::make_shared<BVHModel<OBBRSSd>>();
bvh_model->beginModel();
// ... add triangles ...
bvh_model->endModel();  // Builds BVH tree
```

**BVH Type:** `OBBRSS` (Oriented Bounding Box + Rectangle Swept Sphere)
- Tight bounding volumes for complex geometry
- Efficient for both discrete and continuous queries

## Troubleshooting

### CMake Error: "Could not find pinocchio"

**Symptom:**
```
CMake Error at CMakeLists.txt:18 (find_package):
  Could not find a package configuration file provided by "pinocchio"
```

**Solution for Isaac Sim environment:**

The build script (`compile_fcl.sh`) automatically detects and uses Pinocchio from Isaac Sim's Python packages. If you still see this error:

1. **Verify Pinocchio installation:**
   ```bash
   /isaac-sim/python.sh -c "import pinocchio; print(pinocchio.__file__)"
   ```
   Should print: `/isaac-sim/kit/python/lib/python3.11/site-packages/...`

2. **Check compile_fcl.sh includes Isaac Sim path:**
   ```bash
   grep "ISAAC_CMEEL_PREFIX" scripts/ccd/compile_fcl.sh
   ```
   Should contain lines 40-43 that add the cmeel.prefix path.

3. **Manual CMake configuration** (if needed):
   ```bash
   export CMAKE_PREFIX_PATH=/isaac-sim/kit/python/lib/python3.11/site-packages/cmeel.prefix:$CMAKE_PREFIX_PATH
   bash compile_fcl.sh
   ```

**Solution for standard Ubuntu:**

Install via RobotPkg (see Environment Setup section).

### Compilation Error: "'security_margin' is not a member"

**Symptom:**
```
error: 'struct fcl::CollisionRequest<double>' has no member named 'security_margin'
```

**Root Cause:**
- FCL does not support `security_margin` field
- COAL (FCL fork) does support it
- Mixing FCL headers with COAL libraries causes API mismatch

**Solution:**
This was fixed in commit [hash] by commenting out `security_margin` assignments (lines 381, 429). If you encounter this:

1. **Update fcl_ccd_check.cpp:**
   ```cpp
   // Line 381 and 429 should be:
   // request.security_margin = config_.collision_margin;  // Commented out
   ```

2. **Alternative:** Set `collision_margin` to 0 (default) and ignore the parameter

### Warning: "Please update your includes from 'hpp/fcl' to 'coal'"

**Symptom:**
```
note: '#pragma message: Please update your includes from 'hpp/fcl' to 'coal' or define COAL_DISABLE_HPP_FCL_WARNINGS'
```

**Cause:**
Pinocchio includes COAL headers which emit deprecation warnings about the `hpp-fcl` → `coal` naming transition.

**Impact:** **Informational only** - does not affect compilation or runtime.

**Suppress (optional):**
Add to `CMakeLists.txt` line 47:
```cmake
target_compile_options(fcl_ccd_check PRIVATE -O3 -Wall -Wextra -DCOAL_DISABLE_HPP_FCL_WARNINGS)
```

### Warning: Member initialization order

**Symptom:**
```
warning: 'FCLCCDCollisionChecker::config_' will be initialized after [-Wreorder]
```

**Cause:** C++ class members are initialized in declaration order, not constructor initializer list order.

**Impact:** **None** - all members are properly initialized, just not in the expected order.

**Fix (optional):** Reorder member declarations in line 151-158 to match constructor initializer list.

### Runtime Error: "Could not open trajectory file"

**Symptom:**
Program exits with error message about missing trajectory CSV.

**Solutions:**
1. **Check file path:** Ensure trajectory file exists and path is correct
2. **Use absolute paths:**
   ```bash
   ./build_fcl/fcl_ccd_check --trajectory /full/path/to/trajectory.csv
   ```
3. **Run from repo root:**
   ```bash
   cd /isaac-sim/curobo/vision_inspection
   ./scripts/ccd/build_fcl/fcl_ccd_check --trajectory data/trajectory/675/joint_trajectory_dp.csv
   ```

### Runtime Error: "Error loading mesh"

**Symptom:**
Assimp fails to load obstacle mesh file.

**Solutions:**
1. **Check mesh format:** Assimp supports OBJ, STL, DAE, etc. Verify file is valid
2. **Check file permissions:** Ensure mesh file is readable
3. **Test with default mesh:**
   ```bash
   ls -la data/object/glass_zup.obj  # Should exist
   ```

### Segmentation Fault during CCD with Mesh Obstacles

**Symptom:**
```
========================================
Checking trajectory with CCD...
Total waypoints: 675
Total segments: 674
========================================

Segmentation fault (core dumped)
```

**Root Cause:**

FCL's Continuous Collision Detection between **Sphere and BVH mesh** has a critical bug that causes infinite recursion in `distanceRecurse()`, leading to stack overflow.

**GDB Backtrace Analysis:**
```
#104542-104607 0x00007c209aa126fb in fcl::detail::distanceRecurse<double>()
    ... (infinite recursion - 66+ levels deep)
#104608 conservativeAdvancement<Sphere, OBBRSS, GJKSolver>()
#104609 ShapeBVHConservativeAdvancement<Sphere, OBBRSS>()
#104610 continuousCollideConservativeAdvancement()
#104611-104613 fcl::continuousCollide<double>()
#104614 FCLCCDCollisionChecker::checkSegmentCCD()
```

The Conservative Advancement algorithm for Sphere-BVH CCD gets stuck in an infinite loop when computing distances during swept-volume queries.

**Verification:**

CCD works perfectly with **primitive obstacles only** (no mesh):

```bash
# Test without mesh - works fine
./build_fcl/fcl_ccd_check --trajectory data/trajectory/675/joint_trajectory_dp.csv --mesh "" --verbose

# Output:
# Total segments checked: 674
# Total CCD queries: 64704
# Check time: 0.05 seconds  ✓ SUCCESS
```

```bash
# Test with mesh (glass_zup.obj) - segfault
./build_fcl/fcl_ccd_check --trajectory data/trajectory/675/joint_trajectory_dp.csv --verbose

# Output:
# Segmentation fault (core dumped)  ✗ CRASH
```

**Current Status:**

- ✅ **Sphere-Box CCD**: Works perfectly
- ✅ **Sphere-Primitive CCD**: Works perfectly
- ❌ **Sphere-BVH Mesh CCD**: Causes segfault due to FCL bug

**Workaround Options:**

1. **Replace mesh with primitive approximations** (recommended for now):
   ```bash
   # Use box obstacles instead of mesh
   ./build_fcl/fcl_ccd_check --trajectory <path> --mesh ""
   ```
   - Modify `addCuboidObstacles()` to add glass as a box primitive
   - Fully functional CCD with ~64k queries in 0.05 seconds

2. **Hybrid approach** (CCD for primitives, discrete for mesh):
   - Use CCD for box/sphere obstacles
   - Use interpolated discrete collision for mesh obstacles
   - Requires code modification in `checkSegmentCCD()`

3. **Full interpolated discrete collision**:
   - Replace CCD with multiple discrete checks along interpolated path
   - Sample 5-10 points per segment for swept-volume coverage
   - Slower but stable for all geometry types

4. **Different FCL version** (experimental):
   - Try newer/older FCL releases
   - May have different bugs or fixes
   - Requires rebuilding FCL

**Related Issues:**

- FCL GitHub Issue: Similar Sphere-BVH CCD crashes reported
- Conservative Advancement algorithm known to have edge cases with complex BVH geometry
- Workaround commonly used in robotics: discretize motion or simplify geometry

**Recommendation:**

Until FCL fixes the Sphere-BVH CCD bug, **use primitive obstacle approximations** or implement **hybrid collision checking** (CCD for primitives, discrete for meshes).

## Performance Notes

### Benchmarks

**Environment:** Ubuntu 22.04, Intel Core i7, 32GB RAM, Isaac Sim environment

**Configuration:** Primitive obstacles only (4 boxes: table, wall, workbench, robot mount) - no mesh

| Trajectory Waypoints | Collision Spheres | Obstacles | CCD Queries | Time | Queries/sec |
|---------------------|-------------------|-----------|-------------|------|-------------|
| 675 | 24 | 4 boxes | 64,704 | 0.05 sec | 1,294,080 |

**Complexity:** O(n × s × o)
- n = number of trajectory segments
- s = number of collision spheres (24 per configuration)
- o = number of obstacles (4 primitives)

**Note:** Performance with BVH mesh obstacles cannot be measured due to FCL segfault bug (see Troubleshooting section above).

### Optimization Techniques

1. **Early Exit:** `checkSegmentCCD()` returns on first collision in a segment
2. **BVH Acceleration:** OBBRSS tree reduces triangle intersection tests
3. **Conservative Advancement:** Adaptive step size for fast convergence
4. **Compiled C++:** ~10x faster than equivalent Python code

### Scaling Recommendations

For large-scale collision checking:

1. **Parallel Processing:** Modify to use OpenMP for segment-level parallelism:
   ```cpp
   #pragma omp parallel for
   for (size_t i = 0; i < trajectory.size() - 1; ++i) {
       checkSegmentCCD(trajectory[i], trajectory[i + 1], i);
   }
   ```

2. **Batch Processing:** Split large trajectories into chunks and process separately

3. **Collision Sphere Reduction:** Use fewer, larger spheres for initial screening

4. **Mesh Simplification:** Reduce triangle count for complex obstacles

## Related Files

- **`scripts/coal_check.py`**: Python implementation using COAL (discrete collision)
- **`common/config.py`**: Default environment configuration
- **`ur_description/ur20.urdf`**: Robot kinematic description
- **`ur_description/ur20_safe.yml`**: CuRobo collision sphere configuration
- **`data/object/glass_zup.obj`**: Example obstacle mesh
- **`data/trajectory/*/joint_trajectory_dp.csv`**: Example trajectories

## References

- **FCL Documentation**: https://github.com/flexible-collision-library/fcl
- **Pinocchio Documentation**: https://stack-of-tasks.github.io/pinocchio/
- **CuRobo**: https://github.com/NVlabs/curobo
- **Conservative Advancement**: Mirtich, B. (1996). "Impulse-based Dynamic Simulation of Rigid Body Systems"

## License

This code follows the same license as the parent repository.
