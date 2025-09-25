#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, List

# Third Party
import torch

import numpy as np
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.rotations import quat_to_rot_matrix
from pxr import UsdPhysics

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform
from curobo.types.math import quat_multiply


ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

try:
    # import for older isaacsim installations
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    # import for isaac sim 4.5+
    from isaacsim.core.api.materials import OmniPBR


# Standard Library
from typing import Optional

# Third Party
from omni.isaac.core.utils.extensions import enable_extension

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,
    my_world: World,
    load_from_usd: bool = False,
    subroot: str = "",
    robot_name: str = "robot",
    position: np.array = np.array([0, 0, 0]),
    initialize_world: bool = True,
):

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            root_path, get_filename(file_name, remove_extension=True) + "_temp.usd"
        )

        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:

        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )

    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path)
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    return robot, robot_path

def calculate_world_goal_poses(
    target_position,
    target_orientation,
    local_position_offset,
    local_orientation_offset,
    tensor_args,
):
    """
    타겟의 포즈와 로컬 오프셋을 기반으로 월드 좌표계의 목표 포즈들을 계산합니다.

    Args:
        target_position (np.ndarray): 타겟의 월드 위치 (x,y,z).
        target_orientation (np.ndarray): 타겟의 월드 방향 쿼터니언 (w,x,y,z).
        local_position_offset (torch.Tensor): 로컬 좌표계의 위치 오프셋 그리드.
        local_orientation_offset (torch.Tensor): 로컬 좌표계의 방향 오프셋 그리드.
        tensor_args (TensorDeviceType): 디바이스 정보 (CPU/GPU).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - world_positions (torch.Tensor): 계산된 월드 좌표계 위치들.
            - world_orientations (torch.Tensor): 계산된 월드 좌표계 방향들.
    """
    # 1. 위치 변환 (Transformation Matrix 사용)
    # 1-1. 타겟의 4x4 변환 행렬 생성
    target_transform_matrix = np.eye(4)
    target_transform_matrix[:3, :3] = quat_to_rot_matrix(target_orientation)
    target_transform_matrix[:3, 3] = target_position

    # 1-2. 로컬 오프셋 포인트를 월드 좌표계로 변환
    local_points_cpu = local_position_offset.cpu().numpy()
    num_points = local_points_cpu.shape[0]
    local_points_homogeneous = np.hstack([local_points_cpu, np.ones((num_points, 1))])
    world_points_homogeneous = target_transform_matrix @ local_points_homogeneous.T
    world_points_cpu = world_points_homogeneous.T[:, :3]

    # 1-3. 계산된 월드 포인트를 최종 텐서로 변환
    world_positions = tensor_args.to_device(world_points_cpu)

    # 2. 방향 변환 (쿼터니언 곱셈 사용)
    # 2-1. 타겟 방향을 배치 크기에 맞게 복제
    target_orientation_tensor = tensor_args.to_device(target_orientation)
    target_orientation_expanded = target_orientation_tensor.unsqueeze(0).expand(num_points, -1)

    # 2-2. 결과를 저장할 빈 텐서 생성 및 쿼터니언 곱셈 호출
    world_orientations = torch.empty_like(local_orientation_offset)
    quat_multiply(target_orientation_expanded, local_orientation_offset, world_orientations)

    return world_positions, world_orientations

class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))
