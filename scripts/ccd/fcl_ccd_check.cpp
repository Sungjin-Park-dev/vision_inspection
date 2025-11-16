/**
 * FCL CCD-based Collision Checker for Robot Trajectories
 *
 * This program checks for collisions along a robot trajectory using FCL's
 * Continuous Collision Detection (CCD) capabilities.
 *
 * Features:
 * - True CCD with InterpMotion for swept-volume collision detection
 * - Pinocchio C++ for forward kinematics
 * - Collision spheres from CuRobo YAML config
 * - Text report output (similar to coal_check.py)
 *
 * Usage:
 *   ./fcl_ccd_check --trajectory data/trajectory/joint_trajectory_dp.csv \
 *                   --robot_urdf ur_description/ur20.urdf \
 *                   --robot_config ur_description/ur20.yml \
 *                   --mesh data/object/glass_zup.obj
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <ctime>

// FCL headers
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/collision_object.h>
#include <fcl/narrowphase/continuous_collision.h>
#include <fcl/narrowphase/continuous_collision_request.h>
#include <fcl/narrowphase/continuous_collision_result.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/sphere.h>
// Note: Motion headers not needed when using Transform-based continuousCollide overload
// #include <fcl/math/motion/interp_motion.h>
// #include <fcl/math/motion/translation_motion.h>

// Pinocchio headers
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>

// Other dependencies
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// Type aliases
using namespace fcl;
using Scalar = double;
using Vec3 = Eigen::Vector3d;
using VecX = Eigen::VectorXd;
using Mat3 = Eigen::Matrix3d;
namespace fs = std::filesystem;

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "."
#endif

// Configuration struct
struct Config {
    Vec3 glass_position = Vec3(1.0, 0.0, -0.13);
    Vec3 table_position = Vec3(1.0, 0.0, -0.425);
    Vec3 table_dimensions = Vec3(0.6, 1.0, 0.5);
    Vec3 wall_position = Vec3(-1.1, 0.0, 0.5);
    Vec3 wall_dimensions = Vec3(0.1, 2.2, 1.0);
    Vec3 workbench_position = Vec3(0.35, -1.1, 0.5);
    Vec3 workbench_dimensions = Vec3(3.0, 0.1, 1.0);
    Vec3 robot_mount_position = Vec3(0.0, 0.0, -0.25);
    Vec3 robot_mount_dimensions = Vec3(0.3, 0.3, 0.5);
    double collision_margin = 0.0;
};

// Collision sphere definition
struct CollisionSphere {
    Vec3 center;
    double radius;
};

constexpr const char* kDefaultTrajectory = "data/trajectory/joint_trajectory_dp_5000_base.csv";
constexpr const char* kDefaultRobotUrdf = "ur_description/ur20.urdf";
constexpr const char* kDefaultRobotConfig = "ur_description/ur20_safe.yml";
constexpr const char* kDefaultMesh = "data/object/glass_zup.obj";

fs::path resolvePath(const std::string& input) {
    fs::path path(input);
    if (path.is_absolute()) {
        return path;
    }

    fs::path base(PROJECT_ROOT_DIR);
    fs::path candidate = base / path;
    if (fs::exists(candidate.parent_path()) || fs::exists(candidate)) {
        return candidate;
    }

    return fs::absolute(path);
}

bool parseVec3Arg(const std::string& name, int& idx, int argc, char** argv, Vec3& target) {
    if (idx + 3 >= argc) {
        std::cerr << "Error: " << name << " requires 3 float values" << std::endl;
        return false;
    }

    for (int j = 0; j < 3; ++j) {
        target[j] = std::stod(argv[++idx]);
    }
    return true;
}

void printUsage(const char* exec) {
    std::cout << "Usage: " << exec << " [options]\n"
              << "  --trajectory <path>        Joint trajectory CSV (default: " << kDefaultTrajectory << ")\n"
              << "  --robot_urdf <path>        Robot URDF file (default: " << kDefaultRobotUrdf << ")\n"
              << "  --robot_config <path>      CuRobo robot YAML (default: " << kDefaultRobotConfig << ")\n"
              << "  --mesh <path>              Obstacle mesh (repeatable, default: " << kDefaultMesh << ")\n"
              << "  --glass_position x y z     Glass mesh origin (meters)\n"
              << "  --table_position x y z     Table cuboid center\n"
              << "  --table_dimensions x y z   Table dimensions\n"
              << "  --wall_position x y z      Wall cuboid center\n"
              << "  --wall_dimensions x y z    Wall dimensions\n"
              << "  --workbench_position x y z Workbench cuboid center\n"
              << "  --workbench_dimensions x y z Workbench dimensions\n"
              << "  --robot_mount_position x y z Robot mount center\n"
              << "  --robot_mount_dimensions x y z Robot mount dimensions\n"
              << "  --collision_margin <val>   CCD collision margin (meters)\n"
              << "  --verbose                  Print CCD progress every 100 segments\n"
              << "  -h, --help                 Show this message\n" << std::endl;
}

/**
 * FCL CCD Collision Checker Class
 */
class FCLCCDCollisionChecker {
private:
    // Pinocchio robot model
    pinocchio::Model robot_model_;
    pinocchio::Data robot_data_;

    // Collision geometry
    std::map<std::string, std::vector<CollisionSphere>> collision_spheres_;
    std::vector<std::shared_ptr<CollisionObject<Scalar>>> obstacle_objects_;

    // Configuration
    Config config_;

    // Statistics
    int total_ccd_checks_ = 0;
    int total_collisions_ = 0;
    std::vector<int> collision_segments_;

public:
    FCLCCDCollisionChecker(const std::string& robot_urdf_path,
                           const std::string& robot_config_path,
                           const std::vector<std::string>& obstacle_mesh_paths,
                           const Config& config = Config())
        : config_(config), robot_data_(robot_model_) {

        // Load robot model with Pinocchio
        std::cout << "Loading robot model with Pinocchio..." << std::endl;
        pinocchio::urdf::buildModel(robot_urdf_path, robot_model_);
        robot_data_ = pinocchio::Data(robot_model_);
        std::cout << "  Robot loaded: " << robot_model_.nq << " DOF, "
                  << robot_model_.njoints << " joints" << std::endl;

        // Load collision spheres from YAML
        loadCollisionSpheresFromYAML(robot_config_path);

        // Load obstacle meshes
        loadObstacleMeshes(obstacle_mesh_paths);

        // Add cuboid obstacles
        addCuboidObstacles();
    }

    /**
     * Load collision spheres from CuRobo YAML config
     */
    void loadCollisionSpheresFromYAML(const std::string& yaml_path) {
        collision_spheres_.clear();
        std::cout << "\nLoading collision spheres from YAML..." << std::endl;

        try {
            auto yaml_abs = resolvePath(yaml_path);
            YAML::Node config = YAML::LoadFile(yaml_abs.string());
            YAML::Node robot_cfg = config["robot_cfg"].IsDefined() ? config["robot_cfg"] : config;
            YAML::Node kinematics = robot_cfg["kinematics"];
            YAML::Node collision_spheres = kinematics ? kinematics["collision_spheres"] : YAML::Node();

            if (!collision_spheres || !collision_spheres.IsMap()) {
                std::cerr << "  Warning: collision_spheres missing in " << yaml_path << std::endl;
                return;
            }

            for (auto it = collision_spheres.begin(); it != collision_spheres.end(); ++it) {
                const std::string link_name = it->first.as<std::string>();
                const YAML::Node spheres = it->second;

                for (const auto& sphere : spheres) {
                    Vec3 center;
                    center << sphere["center"][0].as<double>(),
                              sphere["center"][1].as<double>(),
                              sphere["center"][2].as<double>();
                    double radius = sphere["radius"].as<double>();

                    if (radius > 0) {
                        collision_spheres_[link_name].push_back({center, radius});
                    }
                }
            }

            int total_spheres = 0;
            for (const auto& pair : collision_spheres_) {
                total_spheres += static_cast<int>(pair.second.size());
            }
            std::cout << "  Loaded " << total_spheres << " collision spheres" << std::endl;
        } catch (const std::exception& exc) {
            std::cerr << "  Error loading YAML: " << exc.what() << std::endl;
        }
    }

    /**
     * Load obstacle meshes using Assimp
     */
    void loadObstacleMeshes(const std::vector<std::string>& mesh_paths) {
        std::cout << "\nLoading obstacle meshes..." << std::endl;
        std::cout << "  Glass position: [" << config_.glass_position.transpose() << "]" << std::endl;

        if (mesh_paths.empty()) {
            std::cout << "  (no mesh obstacles configured)" << std::endl;
            return;
        }

        for (const auto& mesh_path : mesh_paths) {
            const auto mesh_abs = resolvePath(mesh_path);
            Assimp::Importer importer;
            const aiScene* scene = importer.ReadFile(
                mesh_abs.string(),
                aiProcess_Triangulate | aiProcess_JoinIdenticalVertices
            );

            if (!scene || !scene->HasMeshes()) {
                std::cerr << "  Error loading mesh: " << mesh_abs << std::endl;
                continue;
            }

            auto bvh_model = std::make_shared<BVHModel<OBBRSSd>>();
            bvh_model->beginModel();

            for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
                const aiMesh* mesh = scene->mMeshes[m];

                for (unsigned int f = 0; f < mesh->mNumFaces; ++f) {
                    const aiFace& face = mesh->mFaces[f];
                    if (face.mNumIndices != 3) continue;

                    Vec3 v0, v1, v2;
                    v0 << mesh->mVertices[face.mIndices[0]].x,
                          mesh->mVertices[face.mIndices[0]].y,
                          mesh->mVertices[face.mIndices[0]].z;
                    v1 << mesh->mVertices[face.mIndices[1]].x,
                          mesh->mVertices[face.mIndices[1]].y,
                          mesh->mVertices[face.mIndices[1]].z;
                    v2 << mesh->mVertices[face.mIndices[2]].x,
                          mesh->mVertices[face.mIndices[2]].y,
                          mesh->mVertices[face.mIndices[2]].z;

                    bvh_model->addTriangle(v0, v1, v2);
                }
            }

            bvh_model->endModel();

            Transform3<Scalar> tf = Transform3<Scalar>::Identity();
            tf.translation() = config_.glass_position;

            auto col_obj = std::make_shared<CollisionObject<Scalar>>(bvh_model, tf);
            obstacle_objects_.push_back(col_obj);

            std::cout << "  Loaded: " << mesh_abs << " ("
                      << bvh_model->num_vertices << " vertices, "
                      << bvh_model->num_tris << " triangles)" << std::endl;
        }
    }

    /**
     * Add cuboid obstacles (table, wall, workbench, robot mount)
     */
    void addCuboidObstacles() {
        std::cout << "\nAdding cuboid obstacles..." << std::endl;

        auto add_box = [this](const std::string& name, const Vec3& pos, const Vec3& dims) {
            auto box = std::make_shared<Box<Scalar>>(dims[0], dims[1], dims[2]);
            Transform3<Scalar> tf = Transform3<Scalar>::Identity();
            tf.translation() = pos;

            auto col_obj = std::make_shared<CollisionObject<Scalar>>(box, tf);
            obstacle_objects_.push_back(col_obj);

            std::cout << "  " << name << " added: pos=[" << pos.transpose()
                      << "], dims=[" << dims.transpose() << "]" << std::endl;
        };

        add_box("Table", config_.table_position, config_.table_dimensions);
        add_box("Wall", config_.wall_position, config_.wall_dimensions);
        add_box("Workbench", config_.workbench_position, config_.workbench_dimensions);
        add_box("Robot mount", config_.robot_mount_position, config_.robot_mount_dimensions);
    }

    /**
     * Create robot collision objects for a given joint configuration
     */
    std::vector<std::shared_ptr<CollisionObject<Scalar>>>
    createRobotCollisionObjects(const VecX& q) {
        // Compute forward kinematics
        pinocchio::forwardKinematics(robot_model_, robot_data_, q);
        pinocchio::updateFramePlacements(robot_model_, robot_data_);

        std::vector<std::shared_ptr<CollisionObject<Scalar>>> robot_objects;

        // Create sphere collision objects for each link
        for (const auto& link_pair : collision_spheres_) {
            const std::string& link_name = link_pair.first;
            const auto& spheres = link_pair.second;

            // Get link transform
            pinocchio::SE3 link_transform;
            try {
                auto frame_id = robot_model_.getFrameId(link_name);
                link_transform = robot_data_.oMf[frame_id];
            } catch (...) {
                try {
                    auto joint_id = robot_model_.getJointId(link_name);
                    link_transform = robot_data_.oMi[joint_id];
                } catch (...) {
                    continue;
                }
            }

            // Create sphere for each collision sphere
            for (const auto& sphere_def : spheres) {
                // Transform sphere center to world frame
                Vec3 center_world = link_transform.translation() +
                                   link_transform.rotation() * sphere_def.center;

                auto sphere = std::make_shared<Sphere<Scalar>>(sphere_def.radius);
                Transform3<Scalar> tf = Transform3<Scalar>::Identity();
                tf.translation() = center_world;

                auto col_obj = std::make_shared<CollisionObject<Scalar>>(sphere, tf);
                robot_objects.push_back(col_obj);
            }
        }

        return robot_objects;
    }

    /**
     * Check collision for a single configuration (discrete)
     */
    bool checkCollisionSingleConfig(const VecX& q) {
        auto robot_objects = createRobotCollisionObjects(q);

        for (const auto& robot_obj : robot_objects) {
            for (const auto& obstacle_obj : obstacle_objects_) {
                CollisionRequest<Scalar> request;
                // Note: FCL does not support security_margin (COAL-specific feature)
                // request.security_margin = config_.collision_margin;
                CollisionResult<Scalar> result;

                collide(robot_obj.get(), obstacle_obj.get(), request, result);

                if (result.isCollision()) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Check collision along trajectory segment using CCD
     */
    bool checkSegmentCCD(const VecX& q_start, const VecX& q_end, int segment_idx) {
        // Get robot collision objects at start and end configurations
        auto robot_start = createRobotCollisionObjects(q_start);
        auto robot_end = createRobotCollisionObjects(q_end);

        if (robot_start.size() != robot_end.size()) {
            std::cerr << "Warning: Robot object count mismatch" << std::endl;
            return false;
        }

        bool collision_detected = false;

        // Check CCD for each robot sphere against each obstacle
        for (size_t i = 0; i < robot_start.size(); ++i) {
            // Get the goal transform for this robot sphere
            Transform3<Scalar> tf_goal = robot_end[i]->getTransform();

            for (const auto& obstacle_obj : obstacle_objects_) {
                // Obstacle is static, so goal transform = current transform
                Transform3<Scalar> obstacle_tf = obstacle_obj->getTransform();

                // Configure CCD request
                ContinuousCollisionRequest<Scalar> request;
                request.ccd_motion_type = CCDM_LINEAR;
                request.ccd_solver_type = CCDC_CONSERVATIVE_ADVANCEMENT;
                request.num_max_iterations = 10;
                request.toc_err = 0.0001;

                ContinuousCollisionResult<Scalar> result;

                // Perform continuous collision detection
                // robot_start[i] already has the start transform set
                // We pass the goal transform for the robot sphere
                // Obstacle is static, so its goal = current position
                continuousCollide(
                    robot_start[i].get(), tf_goal,
                    obstacle_obj.get(), obstacle_tf,
                    request, result
                );

                total_ccd_checks_++;

                if (result.is_collide) {
                    collision_detected = true;
                    total_collisions_++;

                    // Record collision details
                    if (std::find(collision_segments_.begin(), collision_segments_.end(), segment_idx)
                        == collision_segments_.end()) {
                        collision_segments_.push_back(segment_idx);
                    }

                    // Early exit on first collision
                    return true;
                }
            }
        }

        return collision_detected;
    }

    /**
     * Check entire trajectory using CCD
     */
    void checkTrajectory(const std::vector<VecX>& trajectory, bool verbose = true) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Checking trajectory with CCD..." << std::endl;
        std::cout << "Total waypoints: " << trajectory.size() << std::endl;
        std::cout << "Total segments: " << (trajectory.size() - 1) << std::endl;
        std::cout << "========================================\n" << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Reset statistics
        total_ccd_checks_ = 0;
        total_collisions_ = 0;
        collision_segments_.clear();

        // Check each segment
        for (size_t i = 0; i < trajectory.size() - 1; ++i) {
            if (verbose && (i + 1) % 100 == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << (trajectory.size() - 1)
                          << " segments checked" << std::endl;
            }

            checkSegmentCCD(trajectory[i], trajectory[i + 1], i);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Print results
        std::cout << "\n========================================" << std::endl;
        std::cout << "CCD Collision Check Results" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Total segments checked: " << (trajectory.size() - 1) << std::endl;
        std::cout << "Total CCD queries: " << total_ccd_checks_ << std::endl;
        std::cout << "Collision segments: " << collision_segments_.size() << std::endl;
        std::cout << "Collision-free segments: " << (trajectory.size() - 1 - collision_segments_.size()) << std::endl;
        std::cout << "Collision rate: " << std::fixed << std::setprecision(2)
                  << (100.0 * collision_segments_.size() / (trajectory.size() - 1)) << "%" << std::endl;
        std::cout << "Check time: " << (duration.count() / 1000.0) << " seconds" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }

    /**
     * Save collision report to text file
     */
    void saveCollisionReport(const std::string& trajectory_path,
                            const std::string& robot_urdf,
                            const std::string& robot_config,
                            const std::vector<std::string>& meshes,
                            const std::vector<VecX>& trajectory) {
        // Create output directory
        const std::string num_points = std::to_string(trajectory.size());
        const fs::path report_dir = fs::path(PROJECT_ROOT_DIR) / "data" / "collision" / num_points;
        fs::create_directories(report_dir);

        const fs::path report_path = report_dir / "collision_ccd.txt";

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream timestamp;
        timestamp << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");

        // Open file in append mode
        std::ofstream report(report_path, std::ios::app);

        if (!report.is_open()) {
            std::cerr << "Error: Could not open report file: " << report_path << std::endl;
            return;
        }

        // Check if file already has content
        report.seekp(0, std::ios::end);
        if (report.tellp() > 0) {
            report << "\n\n";
        }

        // Write report
        report << "=== CCD Collision Report @ " << timestamp.str() << " ===" << std::endl;
        report << "Trajectory: " << trajectory_path << std::endl;
        report << "Robot URDF: " << robot_urdf << std::endl;
        report << "Robot config: " << robot_config << std::endl;
        report << "Obstacle meshes: ";
        for (size_t i = 0; i < meshes.size(); ++i) {
            report << meshes[i];
            if (i < meshes.size() - 1) report << ", ";
        }
        report << std::endl;
        report << "Collision margin: " << config_.collision_margin << std::endl;
        report << "CCD enabled: true (InterpMotion with Conservative Advancement)" << std::endl;
        report << std::endl;

        report << "Total waypoints: " << trajectory.size() << std::endl;
        report << "Total segments: " << (trajectory.size() - 1) << std::endl;
        report << "Total CCD queries: " << total_ccd_checks_ << std::endl;
        report << "Collision segments: " << collision_segments_.size() << std::endl;
        report << "Collision-free segments: " << (trajectory.size() - 1 - collision_segments_.size()) << std::endl;
        report << "Collision rate (%): " << std::fixed << std::setprecision(2)
               << (100.0 * collision_segments_.size() / (trajectory.size() - 1)) << std::endl;
        report << std::endl;

        // List collision segments
        report << "Collision segment indices: ";
        for (size_t i = 0; i < collision_segments_.size(); ++i) {
            report << collision_segments_[i];
            if (i < collision_segments_.size() - 1) report << ", ";
            if (i >= 50) {
                report << " ... (+" << (collision_segments_.size() - 50) << " more)";
                break;
            }
        }
        report << std::endl;

        report.close();

        std::cout << "Report saved to: " << report_path << std::endl;
    }
};

/**
 * Load trajectory from CSV file
 */
std::vector<VecX> loadTrajectoryCSV(const fs::path& csv_path, int& dof) {
    std::vector<VecX> trajectory;
    std::ifstream file(csv_path);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open trajectory file: " << csv_path << std::endl;
        return trajectory;
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: Empty trajectory file: " << csv_path << std::endl;
        return trajectory;
    }

    std::vector<std::string> header_fields;
    std::stringstream header_ss(line);
    std::string field;
    while (std::getline(header_ss, field, ',')) {
        header_fields.push_back(field);
    }

    std::vector<int> joint_columns;
    for (size_t idx = 0; idx < header_fields.size(); ++idx) {
        if (header_fields[idx].find("joint") != std::string::npos) {
            joint_columns.push_back(static_cast<int>(idx));
        }
    }

    if (joint_columns.empty() && header_fields.size() > 1) {
        for (size_t idx = 1; idx < header_fields.size(); ++idx) {
            joint_columns.push_back(static_cast<int>(idx));
        }
    }

    dof = static_cast<int>(joint_columns.size());
    if (dof == 0) {
        std::cerr << "Error: No joint columns detected in " << csv_path << std::endl;
        return trajectory;
    }

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        VecX config = VecX::Zero(dof);
        std::string value;
        int column = 0;
        size_t joint_idx = 0;

        while (std::getline(ss, value, ',')) {
            if (joint_idx < joint_columns.size() && column == joint_columns[joint_idx]) {
                config[static_cast<int>(joint_idx)] = std::stod(value);
                joint_idx++;
            }
            column++;
        }

        if (joint_idx == joint_columns.size()) {
            trajectory.push_back(config);
        }
    }

    std::cout << "Loaded trajectory: " << trajectory.size() << " waypoints, "
              << dof << " joints" << std::endl;

    return trajectory;
}

/**
 * Main function
 */
int main(int argc, char** argv) {
    std::string trajectory_path = kDefaultTrajectory;
    std::string robot_urdf = kDefaultRobotUrdf;
    std::string robot_config = kDefaultRobotConfig;
    std::vector<std::string> mesh_paths = {std::string(kDefaultMesh)};
    bool meshes_overridden = false;
    Config config;
    bool verbose = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        auto requireValue = [&](const std::string& name) -> bool {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires a value" << std::endl;
                return false;
            }
            return true;
        };

        if (arg == "--trajectory") {
            if (!requireValue(arg)) return 1;
            trajectory_path = argv[++i];
        } else if (arg == "--robot_urdf") {
            if (!requireValue(arg)) return 1;
            robot_urdf = argv[++i];
        } else if (arg == "--robot_config") {
            if (!requireValue(arg)) return 1;
            robot_config = argv[++i];
        } else if (arg == "--mesh") {
            if (!requireValue(arg)) return 1;
            if (!meshes_overridden) {
                mesh_paths.clear();
                meshes_overridden = true;
            }
            mesh_paths.push_back(argv[++i]);
        } else if (arg == "--glass_position") {
            if (!parseVec3Arg(arg, i, argc, argv, config.glass_position)) return 1;
        } else if (arg == "--table_position") {
            if (!parseVec3Arg(arg, i, argc, argv, config.table_position)) return 1;
        } else if (arg == "--table_dimensions") {
            if (!parseVec3Arg(arg, i, argc, argv, config.table_dimensions)) return 1;
        } else if (arg == "--wall_position") {
            if (!parseVec3Arg(arg, i, argc, argv, config.wall_position)) return 1;
        } else if (arg == "--wall_dimensions") {
            if (!parseVec3Arg(arg, i, argc, argv, config.wall_dimensions)) return 1;
        } else if (arg == "--workbench_position") {
            if (!parseVec3Arg(arg, i, argc, argv, config.workbench_position)) return 1;
        } else if (arg == "--workbench_dimensions") {
            if (!parseVec3Arg(arg, i, argc, argv, config.workbench_dimensions)) return 1;
        } else if (arg == "--robot_mount_position") {
            if (!parseVec3Arg(arg, i, argc, argv, config.robot_mount_position)) return 1;
        } else if (arg == "--robot_mount_dimensions") {
            if (!parseVec3Arg(arg, i, argc, argv, config.robot_mount_dimensions)) return 1;
        } else if (arg == "--collision_margin") {
            if (!requireValue(arg)) return 1;
            config.collision_margin = std::stod(argv[++i]);
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    const fs::path traj_abs = resolvePath(trajectory_path);
    const fs::path urdf_abs = resolvePath(robot_urdf);
    const fs::path robot_config_abs = resolvePath(robot_config);
    std::vector<std::string> mesh_abs;
    mesh_abs.reserve(mesh_paths.size());
    for (const auto& mesh : mesh_paths) {
        mesh_abs.push_back(resolvePath(mesh).string());
    }

    std::cout << "FCL CCD Collision Checker" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Trajectory CSV: " << traj_abs << std::endl;
    std::cout << "Robot URDF:    " << urdf_abs << std::endl;
    std::cout << "Robot config:  " << robot_config_abs << std::endl;
    std::cout << "Meshes:        " << mesh_abs.size() << std::endl;
    std::cout << "Collision margin: " << config.collision_margin << " m" << std::endl;
    std::cout << std::endl;

    std::cout << "Environment:" << std::endl;
    std::cout << "  Glass position:      [" << config.glass_position.transpose() << "]" << std::endl;
    std::cout << "  Table position:      [" << config.table_position.transpose() << "]" << std::endl;
    std::cout << "  Table dimensions:    [" << config.table_dimensions.transpose() << "]" << std::endl;
    std::cout << "  Wall position:       [" << config.wall_position.transpose() << "]" << std::endl;
    std::cout << "  Wall dimensions:     [" << config.wall_dimensions.transpose() << "]" << std::endl;
    std::cout << "  Workbench position:  [" << config.workbench_position.transpose() << "]" << std::endl;
    std::cout << "  Workbench dimensions:[" << config.workbench_dimensions.transpose() << "]" << std::endl;
    std::cout << "  Robot mount position:[" << config.robot_mount_position.transpose() << "]" << std::endl;
    std::cout << "  Robot mount dims:    [" << config.robot_mount_dimensions.transpose() << "]" << std::endl;
    std::cout << std::endl;

    int dof = 0;
    auto trajectory = loadTrajectoryCSV(traj_abs, dof);
    if (trajectory.size() < 2) {
        std::cerr << "Error: Trajectory must contain at least two waypoints" << std::endl;
        return 1;
    }

    FCLCCDCollisionChecker checker(urdf_abs.string(), robot_config_abs.string(), mesh_abs, config);
    checker.checkTrajectory(trajectory, verbose);
    checker.saveCollisionReport(traj_abs.string(), urdf_abs.string(), robot_config_abs.string(), mesh_abs, trajectory);

    return 0;
}
