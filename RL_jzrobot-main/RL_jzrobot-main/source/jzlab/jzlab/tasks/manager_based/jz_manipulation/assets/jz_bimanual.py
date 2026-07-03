"""JZ dual-arm articulation config for Isaac Lab."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .. import JZ_MANIPULATION_ROOT_DIR
from ..constants import BODY_JOINTS, LEFT_ARM_JOINTS, LEFT_GRIPPER_OPEN, RIGHT_ARM_JOINTS, RIGHT_GRIPPER_OPEN
from .urdf_utils import SOURCE_URDF_PATH, get_resolved_urdf_path


USD_DIR = JZ_MANIPULATION_ROOT_DIR / "usds" / "jz_bimanual"
USD_PATH = USD_DIR / "jz_bimanual.usd"
USD_BASE_PATH = USD_DIR / "configuration" / "jz_bimanual_base.usd"
FORCE_USD_REBUILD = False


def _is_fresh_cache(cache_paths: tuple[Path, ...], dependency_paths: tuple[Path, ...]) -> bool:
    if not all(path.is_file() for path in cache_paths):
        return False

    if USD_BASE_PATH.stat().st_size < 1_000_000:
        return False

    oldest_cache_mtime = min(path.stat().st_mtime for path in cache_paths)
    newest_dependency_mtime = max(path.stat().st_mtime for path in dependency_paths)
    return oldest_cache_mtime >= newest_dependency_mtime


def _make_spawn_cfg():
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=0,
    )

    resolved_urdf_path = get_resolved_urdf_path()

    if not FORCE_USD_REBUILD and _is_fresh_cache((USD_PATH, USD_BASE_PATH), (SOURCE_URDF_PATH, resolved_urdf_path)):
        return sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            rigid_props=rigid_props,
            articulation_props=articulation_props,
        )

    return sim_utils.UrdfFileCfg(
        asset_path=str(resolved_urdf_path),
        usd_dir=str(USD_DIR),
        usd_file_name=USD_PATH.name,
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=True,
        force_usd_conversion=True,
        collision_from_visuals=False,
        self_collision=False,
        collider_type="convex_decomposition",
        rigid_props=rigid_props,
        articulation_props=articulation_props,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=None,
                damping=None,
            ),
        ),
    )


_INITIAL_JOINT_POS = {name: 0.0 for name in BODY_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS}
_INITIAL_JOINT_POS.update(
    {
        "left_arm_joint4": -0.8,
        "left_arm_joint6": 0.3,
        "right_arm_joint4": 0.8,
        "right_arm_joint6": 0.3,
    }
)
_INITIAL_JOINT_POS.update(LEFT_GRIPPER_OPEN)
_INITIAL_JOINT_POS.update(RIGHT_GRIPPER_OPEN)

NEUTRAL_GRASP_JOINT_POS = {
    "body_joint1": 0.11,
    "body_joint2": 0.15,
    "body_joint5": 0.21,
    "left_arm_joint1": -0.65,
    "left_arm_joint2": -1.16,
    "left_arm_joint3": 0.45,
    "left_arm_joint4": -0.90,
    "left_arm_joint5": 1.01,
    "left_arm_joint6": -0.05,
    "left_arm_joint7": 0.01,
    **LEFT_GRIPPER_OPEN,
    "right_arm_joint1": 0.65,
    "right_arm_joint2": 1.16,
    "right_arm_joint3": -0.45,
    "right_arm_joint4": 0.90,
    "right_arm_joint5": -1.01,
    "right_arm_joint6": 0.05,
    "right_arm_joint7": 0.01,
    **RIGHT_GRIPPER_OPEN,
}

JZ_BIMANUAL_CFG = ArticulationCfg(
    spawn=_make_spawn_cfg(),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_INITIAL_JOINT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=["body_joint[1-5]"],
            velocity_limit_sim={
                "body_joint1": 1.5,
                "body_joint2": 1.5,
                "body_joint3": 2.0,
                "body_joint4": 2.5,
                "body_joint5": 2.5,
            },
            effort_limit_sim={
                "body_joint1": 2000.0,
                "body_joint2": 200.0,
                "body_joint3": 180.0,
                "body_joint4": 100.0,
                "body_joint5": 100.0,
            },
            stiffness=600.0,
            damping=80.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_arm_joint[1-7]",
                "right_arm_joint[1-7]",
            ],
            velocity_limit_sim={
                "left_arm_joint[1-2]": 3.0,
                "right_arm_joint[1-2]": 3.0,
                "left_arm_joint3": 3.5,
                "right_arm_joint3": 3.5,
                "left_arm_joint4": 4.0,
                "right_arm_joint4": 4.0,
                "left_arm_joint5": 5.0,
                "right_arm_joint5": 5.0,
                "left_arm_joint[6-7]": 6.0,
                "right_arm_joint[6-7]": 6.0,
            },
            effort_limit_sim={
                "left_arm_joint[1-2]": 80.0,
                "right_arm_joint[1-2]": 80.0,
                "left_arm_joint3": 60.0,
                "right_arm_joint3": 60.0,
                "left_arm_joint4": 40.0,
                "right_arm_joint4": 40.0,
                "left_arm_joint5": 30.0,
                "right_arm_joint5": 30.0,
                "left_arm_joint6": 20.0,
                "right_arm_joint6": 20.0,
                "left_arm_joint7": 15.0,
                "right_arm_joint7": 15.0,
            },
            stiffness=180.0,
            damping=24.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_gripper_(narrow|wide)_joint",
                "right_gripper_(narrow|wide)_joint",
            ],
            velocity_limit_sim=2.0,
            effort_limit_sim=20.0,
            stiffness=200.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


JZ_BIMANUAL_HIGH_PD_CFG = JZ_BIMANUAL_CFG.copy()
JZ_BIMANUAL_HIGH_PD_CFG.actuators["body"].stiffness = 900.0
JZ_BIMANUAL_HIGH_PD_CFG.actuators["body"].damping = 120.0
JZ_BIMANUAL_HIGH_PD_CFG.actuators["arm"].stiffness = 260.0
JZ_BIMANUAL_HIGH_PD_CFG.actuators["arm"].damping = 36.0
JZ_BIMANUAL_HIGH_PD_CFG.actuators["gripper"].stiffness = 400.0
JZ_BIMANUAL_HIGH_PD_CFG.actuators["gripper"].damping = 30.0
