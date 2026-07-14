"""Joint-position-control specialization for JZ bimanual grasp."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from .. import mdp
from ..grasp_env_cfg import GraspEnvCfg, GraspSceneCfg, OBJECT_INITIAL_CENTER_Z, TABLE_TOP_Z
from ....assets.jz_bimanual import JZ_BIMANUAL_HIGH_PD_CFG
from ....constants import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_TCP_ORIENTATION_LINK,
    LEFT_TCP_ORIENTATION_OFFSET_QUAT,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_TCP_ORIENTATION_LINK,
    RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
    RIGHT_TCP_POSITION_LINKS,
)


_ACTION_SCALE = 1.0
_FIXED_ARM_ACTION_SCALE = 0.20
_GRIPPER_ACTION_SCALE = 1.15
_APPROACH_SIDE_TARGET_Y = 0.45
_APPROACH_TARGET_OFFSET_Z = 0.0
_APPROACH_3D_SIDE_TARGET_Y = 0.60
_APPROACH_3D_TARGET_OFFSET_Z = 0.12
# Easy target is the fixed object's root, i.e. the middle bottle target, not a side offset.
_APPROACH_3D_EASY_SIDE_TARGET_Y = 0.0
_APPROACH_3D_EASY_TARGET_OFFSET_Z = 0.0
_APPROACH_3D_EASY_ARM_ACTION_SCALE = 0.25
_TWO_TARGET_LEFT_OBJECT_POS = (0.62, 0.38, OBJECT_INITIAL_CENTER_Z)
_TWO_TARGET_RIGHT_OBJECT_POS = (0.62, -0.38, OBJECT_INITIAL_CENTER_Z)
_TWO_TARGET_TCP_TARGET_OFFSET = (0.0, 0.0, 0.0)
_TWO_TARGET_TCP_TABLE_CLEARANCE_HEIGHT = TABLE_TOP_Z + 0.055


@configclass
class JZGraspTwoTargetSceneCfg(GraspSceneCfg):
    """Grasp scene with one fixed visual target per arm."""

    right_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TWO_TARGET_RIGHT_OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.30),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.95)),
        ),
    )


@configclass
class JZGraspTwoTargetCloseSceneCfg(JZGraspTwoTargetSceneCfg):
    """Dynamic cylindrical targets with filtered fingertip contact sensors."""

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TWO_TARGET_LEFT_OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.12,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.35, 1.0)),
        ),
    )
    right_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RightObject",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_TWO_TARGET_RIGHT_OBJECT_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.12,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=2.0,
                linear_damping=2.0,
                angular_damping=2.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.20),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05)),
        ),
    )

    left_narrow_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_narrow3_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_wide_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_wide3_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    right_narrow_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_narrow3_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_wide_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_wide3_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )


@configclass
class JZGraspTwoTargetFullContactSceneCfg(JZGraspTwoTargetCloseSceneCfg):
    """Two-target scene with per-link full-finger and palm contact coverage."""

    left_narrow1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_narrow1_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_narrow2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_narrow2_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_wide1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_wide1_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_wide2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper_wide2_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_palm9_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_link9",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    left_palm10_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_arm_link10",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    right_narrow1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_narrow1_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_narrow2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_narrow2_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_wide1_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_wide1_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_wide2_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper_wide2_link",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_palm9_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_link9",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )
    right_palm10_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_arm_link10",
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/RightObject"],
    )


@configclass
class JZGraspEnvCfg(GraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = JZ_BIMANUAL_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=JZ_BIMANUAL_HIGH_PD_CFG.init_state.joint_pos,
            ),
        )
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        self.scene.robot.actuators["arm"].stiffness = 260.0
        self.scene.robot.actuators["arm"].damping = 36.0

        self.actions.left_arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINTS,
            scale=_ACTION_SCALE,
            rescale_to_limits=True,
        )
        self.actions.right_arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=_ACTION_SCALE,
            rescale_to_limits=True,
        )
        self.actions.left_gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=LEFT_GRIPPER_JOINTS,
            scale=_ACTION_SCALE,
            rescale_to_limits=True,
        )
        self.actions.right_gripper_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINTS,
            scale=_ACTION_SCALE,
            rescale_to_limits=True,
        )


@configclass
class JZGraspEnvCfg_PLAY(JZGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspFixedEnvCfg(JZGraspEnvCfg):
    """Fixed-start Grasp variant for learning a stable, natural object approach first."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.rigid_props.disable_gravity = True

        self.events.reset_robot_arm_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_arm_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_gripper_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_gripper_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_object_pose.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.reset_object_pose.params["velocity_range"] = {}

        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINTS,
            scale=_FIXED_ARM_ACTION_SCALE,
            use_default_offset=True,
        )
        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=_FIXED_ARM_ACTION_SCALE,
            use_default_offset=True,
        )
        self.actions.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_GRIPPER_JOINTS,
            scale=_GRIPPER_ACTION_SCALE,
            use_default_offset=True,
        )
        self.actions.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINTS,
            scale=_GRIPPER_ACTION_SCALE,
            use_default_offset=True,
        )

        self.rewards.action_rate.weight = -1.0e-3
        self.rewards.action_max_abs_penalty.weight = -1.0e-3
        self.rewards.left_joint_vel.weight = -1.0e-3
        self.rewards.right_joint_vel.weight = -1.0e-3
        self.rewards.left_joint_posture = RewTerm(
            func=mdp.joint_deviation_from_default_l2,
            weight=-2.0e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
        )
        self.rewards.right_joint_posture = RewTerm(
            func=mdp.joint_deviation_from_default_l2,
            weight=-2.0e-3,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
        )
        self.rewards.left_joint_limit_margin = RewTerm(
            func=mdp.joint_limit_margin_penalty,
            weight=-5.0e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS), "margin": 0.15},
        )
        self.rewards.right_joint_limit_margin = RewTerm(
            func=mdp.joint_limit_margin_penalty,
            weight=-5.0e-2,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS), "margin": 0.15},
        )


@configclass
class JZGraspFixedEnvCfg_PLAY(JZGraspFixedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproachEnvCfg(JZGraspFixedEnvCfg):
    """Approach-only curriculum: move both grippers naturally toward reachable side targets."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.object.init_state.pos = (0.62, 0.0, OBJECT_INITIAL_CENTER_Z)
        self.scene.object.spawn.rigid_props.disable_gravity = True

        left_target_offset = (0.0, _APPROACH_SIDE_TARGET_Y, _APPROACH_TARGET_OFFSET_Z)
        right_target_offset = (0.0, -_APPROACH_SIDE_TARGET_Y, _APPROACH_TARGET_OFFSET_Z)

        self.observations.policy.left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
            },
        )
        self.observations.policy.right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
            },
        )

        self.rewards.left_tcp_object_progress.func = mdp.tcp_to_object_side_target_progress_reward
        self.rewards.left_tcp_object_progress.params = {
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_progress.func = mdp.tcp_to_object_side_target_progress_reward
        self.rewards.right_tcp_object_progress.params = {
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }
        self.rewards.left_tcp_object_tracking.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.left_tcp_object_tracking.weight = 0.8
        self.rewards.left_tcp_object_tracking.params = {
            "std": 0.12,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.right_tcp_object_tracking.weight = 0.8
        self.rewards.right_tcp_object_tracking.params = {
            "std": 0.12,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }
        self.rewards.left_tcp_object_tracking_fine.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.left_tcp_object_tracking_fine.weight = 0.6
        self.rewards.left_tcp_object_tracking_fine.params = {
            "std": 0.04,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking_fine.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.right_tcp_object_tracking_fine.weight = 0.6
        self.rewards.right_tcp_object_tracking_fine.params = {
            "std": 0.04,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }

        self.rewards.left_tcp_approach_speed = None
        self.rewards.right_tcp_approach_speed = None
        self.rewards.left_tcp_closing_speed_penalty = None
        self.rewards.right_tcp_closing_speed_penalty = None
        self.rewards.bimanual_grasp_ready_bonus = None
        self.rewards.left_gripper_close_near_object = None
        self.rewards.right_gripper_close_near_object = None
        self.rewards.left_gripper_contact_reward = None
        self.rewards.right_gripper_contact_reward = None
        self.rewards.object_lifted = None
        self.rewards.bimanual_stable_grasp_dwell = None
        self.grasp_success = None
        self.object_lin_vel_penalty = None
        self.object_ang_vel_penalty = None
        self.table_penetration = None
        self.left_tcp_approach_orientation = None
        self.right_tcp_approach_orientation = None
        self.arm_asymmetry_penalty = None


@configclass
class JZGraspApproachEnvCfg_PLAY(JZGraspApproachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DEnvCfg(JZGraspFixedEnvCfg):
    """Fixed 3D TCP approach task without IK, orientation, contact, or grasp rewards."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.object.init_state.pos = (0.62, 0.0, OBJECT_INITIAL_CENTER_Z)
        self.scene.object.spawn.rigid_props.disable_gravity = True

        left_target_offset = (0.0, _APPROACH_3D_SIDE_TARGET_Y, _APPROACH_3D_TARGET_OFFSET_Z)
        right_target_offset = (0.0, -_APPROACH_3D_SIDE_TARGET_Y, _APPROACH_3D_TARGET_OFFSET_Z)

        self.observations.policy.left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
            },
        )
        self.observations.policy.right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
            },
        )

        self.rewards.left_tcp_object_progress = None
        self.rewards.right_tcp_object_progress = None
        self.rewards.left_tcp_object_tracking.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.left_tcp_object_tracking.weight = 1.0
        self.rewards.left_tcp_object_tracking.params = {
            "std": 0.20,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.right_tcp_object_tracking.weight = 1.0
        self.rewards.right_tcp_object_tracking.params = {
            "std": 0.20,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }
        self.rewards.left_tcp_object_tracking_fine.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.left_tcp_object_tracking_fine.weight = 0.5
        self.rewards.left_tcp_object_tracking_fine.params = {
            "std": 0.06,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking_fine.func = mdp.tcp_to_object_side_target_distance_tanh
        self.rewards.right_tcp_object_tracking_fine.weight = 0.5
        self.rewards.right_tcp_object_tracking_fine.params = {
            "std": 0.06,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }

        self.rewards.action_rate.weight = -2.0e-3
        self.rewards.action_max_abs_penalty.weight = -1.0e-3
        self.rewards.left_joint_vel.func = mdp.joint_vel_l2
        self.rewards.left_joint_vel.weight = -1.0e-3
        self.rewards.left_joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
        }
        self.rewards.right_joint_vel.func = mdp.joint_vel_l2
        self.rewards.right_joint_vel.weight = -1.0e-3
        self.rewards.right_joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
        }
        self.rewards.left_joint_posture.weight = -3.0e-3
        self.rewards.right_joint_posture.weight = -3.0e-3
        self.rewards.left_joint_limit_margin.weight = -1.0e-1
        self.rewards.right_joint_limit_margin.weight = -1.0e-1

        self.rewards.left_tcp_approach_speed = None
        self.rewards.right_tcp_approach_speed = None
        self.rewards.left_tcp_closing_speed_penalty = None
        self.rewards.right_tcp_closing_speed_penalty = None
        self.rewards.bimanual_grasp_ready_bonus = None
        self.rewards.left_gripper_close_near_object = None
        self.rewards.right_gripper_close_near_object = None
        self.rewards.left_gripper_contact_reward = None
        self.rewards.right_gripper_contact_reward = None
        self.rewards.object_lifted = None
        self.rewards.bimanual_stable_grasp_dwell = None
        self.grasp_success = None
        self.object_lin_vel_penalty = None
        self.object_ang_vel_penalty = None
        self.table_penetration = None
        self.left_tcp_approach_orientation = None
        self.right_tcp_approach_orientation = None
        self.arm_asymmetry_penalty = None

        self.curriculum.left_joint_vel_near_object = None
        self.curriculum.right_joint_vel_near_object = None
        self.curriculum.left_tcp_relative_speed_near_object = None
        self.curriculum.right_tcp_relative_speed_near_object = None
        self.curriculum.left_action_rate_near_object = None
        self.curriculum.right_action_rate_near_object = None


@configclass
class JZGraspApproach3DEnvCfg_PLAY(JZGraspApproach3DEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DEasyEnvCfg(JZGraspApproach3DEnvCfg):
    """Fixed 3D approach target at the middle bottle object root."""

    def __post_init__(self):
        super().__post_init__()

        left_target_offset = (0.0, _APPROACH_3D_EASY_SIDE_TARGET_Y, _APPROACH_3D_EASY_TARGET_OFFSET_Z)
        right_target_offset = (0.0, -_APPROACH_3D_EASY_SIDE_TARGET_Y, _APPROACH_3D_EASY_TARGET_OFFSET_Z)

        self.actions.left_arm_action.scale = _APPROACH_3D_EASY_ARM_ACTION_SCALE
        self.actions.right_arm_action.scale = _APPROACH_3D_EASY_ARM_ACTION_SCALE

        self.observations.policy.left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
            },
        )
        self.observations.policy.right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
            },
        )

        # Direct distance terms make the first PPO attempt less likely to prefer staying still.
        self.rewards.left_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
            },
        )
        self.rewards.right_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
            },
        )
        self.rewards.left_tcp_object_tracking.weight = 1.5
        self.rewards.left_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking.weight = 1.5
        self.rewards.right_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }
        self.rewards.left_tcp_object_tracking_fine.weight = 0.5
        self.rewards.left_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
        }
        self.rewards.right_tcp_object_tracking_fine.weight = 0.5
        self.rewards.right_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
        }

        self.rewards.action_rate.weight = -5.0e-4
        self.rewards.action_max_abs_penalty.weight = -5.0e-4
        self.rewards.left_joint_vel.weight = -5.0e-4
        self.rewards.right_joint_vel.weight = -5.0e-4
        self.rewards.left_joint_posture.weight = -1.0e-3
        self.rewards.right_joint_posture.weight = -1.0e-3
        self.rewards.left_joint_limit_margin.weight = -5.0e-2
        self.rewards.right_joint_limit_margin.weight = -5.0e-2


@configclass
class JZGraspApproach3DEasyEnvCfg_PLAY(JZGraspApproach3DEasyEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DEasySmoothEnvCfg(JZGraspApproach3DEasyEnvCfg):
    """Easy bottle-root target with stronger smoothness/posture penalties."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.action_rate.weight = -1.0e-3
        self.rewards.action_max_abs_penalty.weight = -5.0e-4
        self.rewards.left_joint_vel.weight = -8.0e-4
        self.rewards.right_joint_vel.weight = -8.0e-4
        self.rewards.left_joint_posture.weight = -1.5e-3
        self.rewards.right_joint_posture.weight = -1.5e-3
        self.rewards.left_joint_limit_margin.weight = -5.0e-2
        self.rewards.right_joint_limit_margin.weight = -5.0e-2


@configclass
class JZGraspApproach3DEasySmoothEnvCfg_PLAY(JZGraspApproach3DEasySmoothEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DEasyWeightedEnvCfg(JZGraspApproach3DEasySmoothEnvCfg):
    """Easy Smooth variant with joint-index weighted velocity penalties."""

    def __post_init__(self):
        super().__post_init__()

        joint_weights = [1.5, 1.5, 1.2, 1.2, 0.8, 0.6, 0.5]
        self.rewards.left_joint_vel.func = mdp.weighted_joint_vel_l2
        self.rewards.left_joint_vel.weight = -8.0e-4
        self.rewards.left_joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
            "weights": joint_weights,
        }
        self.rewards.right_joint_vel.func = mdp.weighted_joint_vel_l2
        self.rewards.right_joint_vel.weight = -8.0e-4
        self.rewards.right_joint_vel.params = {
            "asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
            "weights": joint_weights,
        }


@configclass
class JZGraspApproach3DEasyWeightedEnvCfg_PLAY(JZGraspApproach3DEasyWeightedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetEnvCfg(JZGraspApproach3DEasyWeightedEnvCfg):
    """Fixed two-target curriculum: each arm approaches its own visible target."""

    scene: JZGraspTwoTargetSceneCfg = JZGraspTwoTargetSceneCfg(num_envs=512, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        self.scene.object.init_state.pos = _TWO_TARGET_LEFT_OBJECT_POS
        self.scene.object.spawn.rigid_props.kinematic_enabled = True
        self.scene.object.spawn.rigid_props.disable_gravity = True
        self.scene.object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.35, 1.0))
        self.scene.right_object.init_state.pos = _TWO_TARGET_RIGHT_OBJECT_POS
        self.scene.right_object.spawn.rigid_props.kinematic_enabled = True
        self.scene.right_object.spawn.rigid_props.disable_gravity = True
        self.scene.right_object.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.82, 0.05))

        self.events.reset_right_object_pose = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("right_object"),
            },
        )

        left_target_offset = _TWO_TARGET_TCP_TARGET_OFFSET
        right_target_offset = _TWO_TARGET_TCP_TARGET_OFFSET
        left_tcp_links = LEFT_TCP_POSITION_LINKS
        right_tcp_links = RIGHT_TCP_POSITION_LINKS

        # Target vectors are enough for this fixed 3D curriculum and avoid a misleading single object-position term.
        self.observations.policy.object_position = None
        self.observations.policy.left_tcp_position = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links)},
        )
        self.observations.policy.right_tcp_position = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links)},
        )
        self.observations.policy.left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.observations.policy.right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        self.rewards.left_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_tcp_table_clearance = RewTerm(
            func=mdp.tcp_table_clearance_penalty,
            weight=-30.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "minimum_height": _TWO_TARGET_TCP_TABLE_CLEARANCE_HEIGHT,
            },
        )
        self.rewards.right_tcp_table_clearance = RewTerm(
            func=mdp.tcp_table_clearance_penalty,
            weight=-30.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "minimum_height": _TWO_TARGET_TCP_TABLE_CLEARANCE_HEIGHT,
            },
        )
        self.rewards.left_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
            "target_offset": left_target_offset,
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.right_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
            "target_offset": right_target_offset,
            "object_cfg": SceneEntityCfg("right_object"),
        }
        self.rewards.left_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
            "target_offset": left_target_offset,
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.right_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
            "target_offset": right_target_offset,
            "object_cfg": SceneEntityCfg("right_object"),
        }


@configclass
class JZGraspApproach3DTwoTargetEnvCfg_PLAY(JZGraspApproach3DTwoTargetEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetStableEnvCfg(JZGraspApproach3DTwoTargetEnvCfg):
    """TwoTarget approach with stabilization rewards gated by the actual target points."""

    def __post_init__(self):
        super().__post_init__()

        left_target_offset = _TWO_TARGET_TCP_TARGET_OFFSET
        right_target_offset = _TWO_TARGET_TCP_TARGET_OFFSET
        left_tcp_links = LEFT_TCP_POSITION_LINKS
        right_tcp_links = RIGHT_TCP_POSITION_LINKS

        self.rewards.left_joint_vel_near_object = RewTerm(
            func=mdp.joint_vel_l2_when_close_to_object_side_target,
            weight=-2.0e-3,
            params={
                "threshold": 0.08,
                "joint_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_joint_vel_near_object = RewTerm(
            func=mdp.joint_vel_l2_when_close_to_object_side_target,
            weight=-2.0e-3,
            params={
                "threshold": 0.08,
                "joint_asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_tcp_relative_speed_near_object = RewTerm(
            func=mdp.tcp_speed_l2_when_close_to_object_side_target,
            weight=-5.0e-2,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_tcp_relative_speed_near_object = RewTerm(
            func=mdp.tcp_speed_l2_when_close_to_object_side_target,
            weight=-5.0e-2,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_action_rate_near_object = RewTerm(
            func=mdp.action_rate_l2_when_close_to_object_side_target,
            weight=-5.0e-3,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "target_offset": left_target_offset,
                "action_name": "left_arm_action",
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_action_rate_near_object = RewTerm(
            func=mdp.action_rate_l2_when_close_to_object_side_target,
            weight=-5.0e-3,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "target_offset": right_target_offset,
                "action_name": "right_arm_action",
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.bimanual_stable_grasp_dwell = RewTerm(
            func=mdp.bimanual_tcp_stable_near_object_side_target_dwell_reward,
            weight=1.0,
            params={
                "threshold": 0.05,
                "speed_threshold": 0.04,
                "hold_steps": 20,
                "left_tcp_asset_cfg": SceneEntityCfg("robot", body_names=left_tcp_links),
                "right_tcp_asset_cfg": SceneEntityCfg("robot", body_names=right_tcp_links),
                "left_target_offset": left_target_offset,
                "right_target_offset": right_target_offset,
                "left_object_cfg": SceneEntityCfg("object"),
                "right_object_cfg": SceneEntityCfg("right_object"),
            },
        )


@configclass
class JZGraspApproach3DTwoTargetStableEnvCfg_PLAY(JZGraspApproach3DTwoTargetStableEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetTrackEnvCfg(JZGraspApproach3DTwoTargetEnvCfg):
    """TwoTarget approach with stronger distance tracking and no extra stabilization penalties."""

    def __post_init__(self):
        super().__post_init__()

        self.rewards.left_tcp_object_progress.weight = -3.0
        self.rewards.right_tcp_object_progress.weight = -3.0
        self.rewards.left_tcp_object_tracking.weight = 1.5
        self.rewards.right_tcp_object_tracking.weight = 1.5
        self.rewards.left_tcp_object_tracking_fine.weight = 1.0
        self.rewards.right_tcp_object_tracking_fine.weight = 1.0
        self.rewards.left_tcp_object_tracking_fine.params["std"] = 0.05
        self.rewards.right_tcp_object_tracking_fine.params["std"] = 0.05


@configclass
class JZGraspApproach3DTwoTargetTrackEnvCfg_PLAY(JZGraspApproach3DTwoTargetTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetDynamicEnvCfg(JZGraspApproach3DTwoTargetTrackEnvCfg):
    """TwoTarget contact-validation variant with dynamic bottles."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.object.spawn.rigid_props.kinematic_enabled = False
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.object.spawn.rigid_props.linear_damping = 2.0
        self.scene.object.spawn.rigid_props.angular_damping = 2.0
        self.scene.right_object.spawn.rigid_props.kinematic_enabled = False
        self.scene.right_object.spawn.rigid_props.disable_gravity = False
        self.scene.right_object.spawn.rigid_props.linear_damping = 2.0
        self.scene.right_object.spawn.rigid_props.angular_damping = 2.0


@configclass
class JZGraspApproach3DTwoTargetDynamicEnvCfg_PLAY(JZGraspApproach3DTwoTargetDynamicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetOpen6DEnvCfg(JZGraspApproach3DTwoTargetTrackEnvCfg):
    """Open-gripper approach task with near-object bottle-axis alignment."""

    scene: JZGraspTwoTargetCloseSceneCfg = JZGraspTwoTargetCloseSceneCfg(num_envs=256, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object.spawn.rigid_props.kinematic_enabled = True
        self.scene.object.spawn.rigid_props.disable_gravity = True
        self.scene.right_object.spawn.rigid_props.kinematic_enabled = True
        self.scene.right_object.spawn.rigid_props.disable_gravity = True
        self.actions.left_gripper_action.scale = 0.0
        self.actions.right_gripper_action.scale = 0.0

        self.rewards.left_tcp_axes_alignment = RewTerm(
            func=mdp.tcp_axes_align_with_object_nearby,
            weight=0.5,
            params={
                "distance_scale": 0.20,
                "fallback_distance": 0.03,
                "tcp_position_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "tcp_orientation_cfg": SceneEntityCfg("robot", body_names=[LEFT_TCP_ORIENTATION_LINK]),
                "orientation_offset": LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                "nominal_inward_direction": (0.0, -1.0, 0.0),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_tcp_axes_alignment = RewTerm(
            func=mdp.tcp_axes_align_with_object_nearby,
            weight=0.5,
            params={
                "distance_scale": 0.20,
                "fallback_distance": 0.03,
                "tcp_position_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "tcp_orientation_cfg": SceneEntityCfg("robot", body_names=[RIGHT_TCP_ORIENTATION_LINK]),
                "orientation_offset": RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                "nominal_inward_direction": (0.0, 1.0, 0.0),
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        self.rewards.left_gripper_close_near_object = None
        self.rewards.right_gripper_close_near_object = None
        self.rewards.left_gripper_contact_reward = None
        self.rewards.right_gripper_contact_reward = None
        self.rewards.left_gripper_early_close = None
        self.rewards.right_gripper_early_close = None
        self.rewards.left_stable_finger_contact = None
        self.rewards.right_stable_finger_contact = None
        self.rewards.object_lifted = None
        self.rewards.bimanual_stable_grasp_dwell = None


@configclass
class JZGraspApproach3DTwoTargetOpen6DEnvCfg_PLAY(JZGraspApproach3DTwoTargetOpen6DEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg(JZGraspApproach3DTwoTargetOpen6DEnvCfg):
    """Open-finger pre-grasp using real finger/cylinder collision-surface geometry."""

    def __post_init__(self):
        super().__post_init__()

        left_narrow_cfg = SceneEntityCfg("robot", body_names=["left_gripper_narrow3_link"])
        left_wide_cfg = SceneEntityCfg("robot", body_names=["left_gripper_wide3_link"])
        right_narrow_cfg = SceneEntityCfg("robot", body_names=["right_gripper_narrow3_link"])
        right_wide_cfg = SceneEntityCfg("robot", body_names=["right_gripper_wide3_link"])
        cylinder_geometry = {"cylinder_radius": 0.03, "cylinder_half_height": 0.06}

        # The synthetic TCP is near the gripper root, so center-distance rewards
        # would pull the physical fingers through the cylinder.
        self.rewards.left_tcp_object_progress = None
        self.rewards.right_tcp_object_progress = None
        self.rewards.left_tcp_object_tracking = None
        self.rewards.right_tcp_object_tracking = None
        self.rewards.left_tcp_object_tracking_fine = None
        self.rewards.right_tcp_object_tracking_fine = None
        self.rewards.left_tcp_axes_alignment = None
        self.rewards.right_tcp_axes_alignment = None

        self.rewards.left_fingertip_surface_clearance = RewTerm(
            func=mdp.fingertip_surface_clearance_reward,
            weight=1.0,
            params={
                "desired_gap": 0.015,
                "gap_std": 0.03,
                **cylinder_geometry,
                "narrow_cfg": left_narrow_cfg,
                "wide_cfg": left_wide_cfg,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_fingertip_surface_clearance = RewTerm(
            func=mdp.fingertip_surface_clearance_reward,
            weight=1.0,
            params={
                "desired_gap": 0.015,
                "gap_std": 0.03,
                **cylinder_geometry,
                "narrow_cfg": right_narrow_cfg,
                "wide_cfg": right_wide_cfg,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        center_params = {
            "near_scale": 0.12,
            "horizontal_std": 0.03,
            "vertical_std": 0.025,
            "level_std": 0.015,
            "between_scale": 0.1,
            **cylinder_geometry,
        }
        self.rewards.left_fingertip_grasp_center = RewTerm(
            func=mdp.fingertip_grasp_center_reward,
            weight=1.0,
            params={
                **center_params,
                "narrow_cfg": left_narrow_cfg,
                "wide_cfg": left_wide_cfg,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_fingertip_grasp_center = RewTerm(
            func=mdp.fingertip_grasp_center_reward,
            weight=1.0,
            params={
                **center_params,
                "narrow_cfg": right_narrow_cfg,
                "wide_cfg": right_wide_cfg,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        orientation_params = {"near_scale": 0.12, **cylinder_geometry}
        self.rewards.left_fingertip_surface_orientation = RewTerm(
            func=mdp.fingertip_surface_orientation_reward,
            weight=0.5,
            params={
                **orientation_params,
                "grasp_axis_sign": -1.0,
                "narrow_cfg": left_narrow_cfg,
                "wide_cfg": left_wide_cfg,
                "tcp_orientation_cfg": SceneEntityCfg("robot", body_names=[LEFT_TCP_ORIENTATION_LINK]),
                "orientation_offset": LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_fingertip_surface_orientation = RewTerm(
            func=mdp.fingertip_surface_orientation_reward,
            weight=0.5,
            params={
                **orientation_params,
                "grasp_axis_sign": 1.0,
                "narrow_cfg": right_narrow_cfg,
                "wide_cfg": right_wide_cfg,
                "tcp_orientation_cfg": SceneEntityCfg("robot", body_names=[RIGHT_TCP_ORIENTATION_LINK]),
                "orientation_offset": RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_early_fingertip_contact = RewTerm(
            func=mdp.early_fingertip_contact_penalty,
            weight=-0.5,
            params={
                "narrow_sensor_name": "left_narrow_contact",
                "wide_sensor_name": "left_wide_contact",
                "force_threshold": 1.0,
            },
        )
        self.rewards.right_early_fingertip_contact = RewTerm(
            func=mdp.early_fingertip_contact_penalty,
            weight=-0.5,
            params={
                "narrow_sensor_name": "right_narrow_contact",
                "wide_sensor_name": "right_wide_contact",
                "force_threshold": 1.0,
            },
        )


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg_PLAY(JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg(JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg):
    """Directional-observation, additive-reward surface pre-grasp curriculum."""

    def __post_init__(self):
        super().__post_init__()

        left_narrow = SceneEntityCfg("robot", body_names=["left_gripper_narrow3_link"])
        left_wide = SceneEntityCfg("robot", body_names=["left_gripper_wide3_link"])
        right_narrow = SceneEntityCfg("robot", body_names=["right_gripper_narrow3_link"])
        right_wide = SceneEntityCfg("robot", body_names=["right_gripper_wide3_link"])
        left_orientation = SceneEntityCfg("robot", body_names=[LEFT_TCP_ORIENTATION_LINK])
        right_orientation = SceneEntityCfg("robot", body_names=[RIGHT_TCP_ORIENTATION_LINK])
        cylinder = {"cylinder_radius": 0.03, "cylinder_half_height": 0.06}

        self.observations.policy.left_pregrasp_geometry = ObsTerm(
            func=mdp.fingertip_pregrasp_observation,
            params={
                **cylinder,
                "grasp_axis_sign": -1.0,
                "narrow_cfg": left_narrow,
                "wide_cfg": left_wide,
                "tcp_orientation_cfg": left_orientation,
                "orientation_offset": LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.observations.policy.right_pregrasp_geometry = ObsTerm(
            func=mdp.fingertip_pregrasp_observation,
            params={
                **cylinder,
                "grasp_axis_sign": 1.0,
                "narrow_cfg": right_narrow,
                "wide_cfg": right_wide,
                "tcp_orientation_cfg": right_orientation,
                "orientation_offset": RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        for name in (
            "left_fingertip_surface_clearance",
            "right_fingertip_surface_clearance",
            "left_fingertip_grasp_center",
            "right_fingertip_grasp_center",
            "left_fingertip_surface_orientation",
            "right_fingertip_surface_orientation",
            "left_early_fingertip_contact",
            "right_early_fingertip_contact",
        ):
            setattr(self.rewards, name, None)

        def geometry(narrow, wide, object_name):
            return {**cylinder, "narrow_cfg": narrow, "wide_cfg": wide, "object_cfg": SceneEntityCfg(object_name)}

        left_geometry = geometry(left_narrow, left_wide, "object")
        right_geometry = geometry(right_narrow, right_wide, "right_object")
        for side, geom in (("left", left_geometry), ("right", right_geometry)):
            setattr(
                self.rewards,
                f"{side}_v2_clearance",
                RewTerm(
                    func=mdp.fingertip_individual_clearance_reward,
                    weight=1.0,
                    params={
                        **geom,
                        "desired_gaps": (0.05, 0.03, 0.015),
                        "stage_steps": (2400, 4800),
                        "gap_std": 0.03,
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v2_gap_balance",
                RewTerm(func=mdp.fingertip_gap_balance_penalty, weight=-0.5, params={**geom, "scale": 0.05}),
            )
            setattr(
                self.rewards,
                f"{side}_v2_between",
                RewTerm(func=mdp.fingertip_between_reward, weight=0.5, params={**geom, "between_scale": 0.1}),
            )
            setattr(
                self.rewards,
                f"{side}_v2_horizontal",
                RewTerm(func=mdp.fingertip_horizontal_center_reward, weight=0.5, params={**geom, "std": 0.03}),
            )
            setattr(
                self.rewards,
                f"{side}_v2_vertical",
                RewTerm(func=mdp.fingertip_vertical_center_reward, weight=1.0, params={**geom, "std": 0.06}),
            )
            setattr(
                self.rewards,
                f"{side}_v2_level",
                RewTerm(func=mdp.fingertip_level_reward, weight=1.0, params={**geom, "std": 0.10}),
            )
            setattr(
                self.rewards,
                f"{side}_v2_too_close",
                RewTerm(func=mdp.fingertip_too_close_penalty, weight=-3.0, params={**geom, "minimum_gap": 0.01}),
            )

        self.rewards.left_v2_z_axis = RewTerm(
            func=mdp.fingertip_signed_axis_reward,
            weight=1.5,
            params={
                "axis": "z",
                "grasp_axis_sign": -1.0,
                "narrow_cfg": left_narrow,
                "wide_cfg": left_wide,
                "tcp_orientation_cfg": left_orientation,
                "orientation_offset": LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_v2_z_axis = RewTerm(
            func=mdp.fingertip_signed_axis_reward,
            weight=1.5,
            params={
                "axis": "z",
                "grasp_axis_sign": 1.0,
                "narrow_cfg": right_narrow,
                "wide_cfg": right_wide,
                "tcp_orientation_cfg": right_orientation,
                "orientation_offset": RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_v2_approach_axis = RewTerm(
            func=mdp.fingertip_signed_axis_reward,
            weight=1.0,
            params={
                "axis": "approach",
                "grasp_axis_sign": -1.0,
                "narrow_cfg": left_narrow,
                "wide_cfg": left_wide,
                "tcp_orientation_cfg": left_orientation,
                "orientation_offset": LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_v2_approach_axis = RewTerm(
            func=mdp.fingertip_signed_axis_reward,
            weight=1.0,
            params={
                "axis": "approach",
                "grasp_axis_sign": 1.0,
                "narrow_cfg": right_narrow,
                "wide_cfg": right_wide,
                "tcp_orientation_cfg": right_orientation,
                "orientation_offset": RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        contact_terms = (
            ("left", "left_narrow_contact", "left_wide_contact"),
            ("right", "right_narrow_contact", "right_wide_contact"),
        )
        for side, narrow_sensor, wide_sensor in contact_terms:
            contact_params = {
                "narrow_sensor_name": narrow_sensor,
                "wide_sensor_name": wide_sensor,
                "force_threshold": 1.0,
            }
            setattr(
                self.rewards,
                f"{side}_v2_contact_terminal",
                RewTerm(func=mdp.early_fingertip_contact_penalty, weight=-5.0, params=contact_params),
            )
            setattr(
                self.terminations,
                f"{side}_early_fingertip_contact",
                DoneTerm(func=mdp.early_fingertip_contact_termination, params=contact_params),
            )


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg
):
    """Distance-first pre-grasp curriculum with proximity-gated pose shaping."""

    def __post_init__(self):
        super().__post_init__()

        sides = (
            (
                "left",
                SceneEntityCfg("robot", body_names=["left_gripper_narrow3_link"]),
                SceneEntityCfg("robot", body_names=["left_gripper_wide3_link"]),
                SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "object",
            ),
            (
                "right",
                SceneEntityCfg("robot", body_names=["right_gripper_narrow3_link"]),
                SceneEntityCfg("robot", body_names=["right_gripper_wide3_link"]),
                SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "right_object",
            ),
        )
        for side, narrow, wide, tcp, object_name in sides:
            geometry = {
                "narrow_cfg": narrow,
                "wide_cfg": wide,
                "object_cfg": SceneEntityCfg(object_name),
            }
            approach = {"tcp_asset_cfg": tcp, "object_cfg": SceneEntityCfg(object_name)}
            setattr(
                self.rewards,
                f"{side}_v3_distance",
                RewTerm(func=mdp.tcp_to_object_distance, weight=-2.0, params=approach),
            )
            setattr(
                self.rewards,
                f"{side}_v3_tracking",
                RewTerm(
                    func=mdp.tcp_to_object_distance_tracking,
                    weight=1.0,
                    params={**approach, "std": 0.25},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v3_progress",
                RewTerm(
                    func=mdp.tcp_to_object_signed_progress_reward,
                    weight=2.0,
                    params={**approach, "reward_key": f"{side}_v3_tcp_progress", "max_progress": 0.02},
                ),
            )

            for term_name in (
                "clearance",
                "gap_balance",
                "between",
                "horizontal",
                "vertical",
                "level",
            ):
                getattr(self.rewards, f"{side}_v2_{term_name}").params["distance_gate_scale"] = 0.08
            getattr(self.rewards, f"{side}_v2_z_axis").params["distance_gate_scale"] = 0.15
            getattr(self.rewards, f"{side}_v2_approach_axis").params["distance_gate_scale"] = 0.15
            getattr(self.rewards, f"{side}_v2_contact_terminal").weight = -1.0
            setattr(self.terminations, f"{side}_early_fingertip_contact", None)

        self.rewards.left_v2_clearance.weight = 0.75
        self.rewards.right_v2_clearance.weight = 0.75
        self.rewards.left_v2_between.weight = 0.5
        self.rewards.right_v2_between.weight = 0.5
        self.rewards.left_v2_horizontal.weight = 0.5
        self.rewards.right_v2_horizontal.weight = 0.5
        self.rewards.left_v2_vertical.weight = 0.75
        self.rewards.right_v2_vertical.weight = 0.75
        self.rewards.left_v2_level.weight = 0.5
        self.rewards.right_v2_level.weight = 0.5
        self.rewards.left_v2_z_axis.weight = 0.75
        self.rewards.right_v2_z_axis.weight = 0.75
        self.rewards.left_v2_approach_axis.weight = 0.5
        self.rewards.right_v2_approach_axis.weight = 0.5


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg
):
    """Cylinder-axis approach with independent height control and near-target braking."""

    def __post_init__(self):
        super().__post_init__()

        for name in (
            "left_v3_distance",
            "right_v3_distance",
            "left_v3_tracking",
            "right_v3_tracking",
            "left_v3_progress",
            "right_v3_progress",
        ):
            setattr(self.rewards, name, None)

        sides = (
            (
                "left",
                SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "object",
                "left_narrow_contact",
                "left_wide_contact",
            ),
            (
                "right",
                SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "right_object",
                "right_narrow_contact",
                "right_wide_contact",
            ),
        )
        for side, tcp, object_name, narrow_sensor, wide_sensor in sides:
            axis = {
                "desired_radial_distance": 0.03,
                "tcp_asset_cfg": tcp,
                "object_cfg": SceneEntityCfg(object_name),
            }
            contact = {
                "narrow_sensor_name": narrow_sensor,
                "wide_sensor_name": wide_sensor,
                "force_threshold": 1.0,
            }
            setattr(
                self.rewards,
                f"{side}_v4_radial_tracking",
                RewTerm(func=mdp.tcp_cylinder_axis_tracking, weight=1.0, params={**axis, "radial_std": 0.20}),
            )
            setattr(
                self.rewards,
                f"{side}_v4_radial_tracking_fine",
                RewTerm(func=mdp.tcp_cylinder_axis_tracking, weight=1.0, params={**axis, "radial_std": 0.04}),
            )
            setattr(
                self.rewards,
                f"{side}_v4_height_tracking",
                RewTerm(
                    func=mdp.tcp_cylinder_height_tracking,
                    weight=0.75,
                    params={
                        "desired_height_offset": 0.0,
                        "height_std": 0.05,
                        "tcp_asset_cfg": tcp,
                        "object_cfg": SceneEntityCfg(object_name),
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_height_tracking_fine",
                RewTerm(
                    func=mdp.tcp_cylinder_height_tracking,
                    weight=0.5,
                    params={
                        "desired_height_offset": 0.0,
                        "height_std": 0.02,
                        "tcp_asset_cfg": tcp,
                        "object_cfg": SceneEntityCfg(object_name),
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_radial_progress",
                RewTerm(
                    func=mdp.tcp_cylinder_radial_progress_reward,
                    weight=2.0,
                    params={
                        **axis,
                        **contact,
                        "reward_key": f"{side}_v4_radial_progress",
                        "max_progress": 0.02,
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_radial_speed",
                RewTerm(
                    func=mdp.tcp_cylinder_radial_speed_near_target_l2,
                    weight=-0.5,
                    params={**axis, "gate_std": 0.06},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_relative_speed",
                RewTerm(
                    func=mdp.tcp_cylinder_relative_speed_near_target_l2,
                    weight=-0.2,
                    params={**axis, "gate_std": 0.06},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_persistent_contact",
                RewTerm(
                    func=mdp.persistent_fingertip_contact_penalty,
                    weight=-2.0,
                    params={**contact, "reward_key": f"{side}_v4_contact_streak", "hold_steps": 15},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v4_stable_pregrasp",
                RewTerm(
                    func=mdp.stable_cylinder_axis_pregrasp_reward,
                    weight=2.0,
                    params={
                        **axis,
                        **contact,
                        "reward_key": f"{side}_v4_stable_pregrasp",
                        "radial_tolerance": 0.02,
                        "desired_height_offset": 0.0,
                        "height_tolerance": 0.025,
                        "speed_threshold": 0.05,
                        "hold_steps": 15,
                    },
                ),
            )

            getattr(self.rewards, f"{side}_v2_contact_terminal").weight = -3.0
            for term_name in ("clearance", "gap_balance", "between", "horizontal", "vertical", "level"):
                getattr(self.rewards, f"{side}_v2_{term_name}").params["distance_gate_scale"] = 0.15
            getattr(self.rewards, f"{side}_v2_z_axis").params["distance_gate_scale"] = 0.30
            getattr(self.rewards, f"{side}_v2_approach_axis").params["distance_gate_scale"] = 0.30


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg
):
    """V4 refinement with stronger height, braking, and pose shaping but no table terms."""

    def __post_init__(self):
        super().__post_init__()

        sides = (
            ("left", SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS), "object"),
            ("right", SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS), "right_object"),
        )
        for side, tcp, object_name in sides:
            object_cfg = SceneEntityCfg(object_name)
            axis = {
                "desired_radial_distance": 0.03,
                "tcp_asset_cfg": tcp,
                "object_cfg": object_cfg,
            }

            getattr(self.rewards, f"{side}_v4_height_tracking").params.update(
                desired_height_offset=0.03,
                height_std=0.08,
            )
            getattr(self.rewards, f"{side}_v4_height_tracking_fine").params.update(
                desired_height_offset=0.03,
                height_std=0.025,
            )
            setattr(
                self.rewards,
                f"{side}_v41_radial_error",
                RewTerm(func=mdp.tcp_cylinder_radial_abs_error, weight=-1.0, params=axis),
            )
            setattr(
                self.rewards,
                f"{side}_v41_height_error",
                RewTerm(
                    func=mdp.tcp_cylinder_height_abs_error,
                    weight=-2.0,
                    params={
                        "desired_height_offset": 0.03,
                        "tcp_asset_cfg": tcp,
                        "object_cfg": object_cfg,
                    },
                ),
            )

            getattr(self.rewards, f"{side}_v4_radial_speed").weight = -1.0
            getattr(self.rewards, f"{side}_v4_relative_speed").weight = -0.5
            getattr(self.rewards, f"{side}_v4_stable_pregrasp").params.update(
                desired_height_offset=0.03,
                speed_threshold=0.04,
                hold_steps=20,
            )
            getattr(self.rewards, f"{side}_v2_z_axis").weight = 1.5
            getattr(self.rewards, f"{side}_v2_approach_axis").weight = 1.0
            getattr(self.rewards, f"{side}_action_rate_near_object").weight = -3.0e-3


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV42EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg
):
    """V4.1 with an external stop radius and progressive approach braking."""

    def __post_init__(self):
        super().__post_init__()

        desired_radial_distance = 0.05
        sides = (
            ("left", SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS), "object"),
            ("right", SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS), "right_object"),
        )
        for side, tcp, object_name in sides:
            object_cfg = SceneEntityCfg(object_name)
            axis = {
                "desired_radial_distance": desired_radial_distance,
                "tcp_asset_cfg": tcp,
                "object_cfg": object_cfg,
            }
            for term_name in (
                "radial_tracking",
                "radial_tracking_fine",
                "radial_progress",
                "radial_speed",
                "relative_speed",
                "stable_pregrasp",
            ):
                getattr(self.rewards, f"{side}_v4_{term_name}").params[
                    "desired_radial_distance"
                ] = desired_radial_distance
            getattr(self.rewards, f"{side}_v41_radial_error").params[
                "desired_radial_distance"
            ] = desired_radial_distance
            getattr(self.rewards, f"{side}_v4_stable_pregrasp").params.update(
                radial_tolerance=0.015,
                speed_threshold=0.03,
            )
            setattr(
                self.rewards,
                f"{side}_v42_scheduled_speed",
                RewTerm(
                    func=mdp.tcp_cylinder_scheduled_speed_excess_l2,
                    weight=-5.0,
                    params={
                        **axis,
                        "far_error": 0.10,
                        "medium_error": 0.05,
                        "near_error": 0.02,
                        "far_speed_limit": 0.15,
                        "medium_speed_limit": 0.08,
                        "near_speed_limit": 0.03,
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v42_inward_overshoot",
                RewTerm(func=mdp.tcp_cylinder_inward_speed_after_overshoot, weight=-10.0, params=axis),
            )


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV42EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV42EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg
):
    """V4.1 with signed linear radial-velocity tracking and a 5cm stop radius."""

    def __post_init__(self):
        super().__post_init__()

        desired_radial_distance = 0.05
        sides = (
            ("left", SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS), "object"),
            ("right", SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS), "right_object"),
        )
        for side, tcp, object_name in sides:
            object_cfg = SceneEntityCfg(object_name)
            axis = {
                "desired_radial_distance": desired_radial_distance,
                "tcp_asset_cfg": tcp,
                "object_cfg": object_cfg,
            }
            for term_name in (
                "radial_tracking",
                "radial_tracking_fine",
                "radial_progress",
                "stable_pregrasp",
            ):
                getattr(self.rewards, f"{side}_v4_{term_name}").params[
                    "desired_radial_distance"
                ] = desired_radial_distance
            getattr(self.rewards, f"{side}_v41_radial_error").params[
                "desired_radial_distance"
            ] = desired_radial_distance

            setattr(self.rewards, f"{side}_v4_radial_speed", None)
            setattr(self.rewards, f"{side}_v4_relative_speed", None)
            getattr(self.rewards, f"{side}_v4_stable_pregrasp").weight = 3.0
            getattr(self.rewards, f"{side}_v4_stable_pregrasp").params.update(
                radial_tolerance=0.015,
                speed_threshold=0.03,
            )
            setattr(
                self.rewards,
                f"{side}_v43_radial_velocity",
                RewTerm(
                    func=mdp.tcp_cylinder_linear_radial_velocity_error,
                    weight=-1.0,
                    params={
                        **axis,
                        "error_deadband": 0.005,
                        "velocity_gain": 1.5,
                        "max_inward_speed": 0.15,
                        "max_outward_speed": 0.08,
                    },
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v43_tangential_speed",
                RewTerm(
                    func=mdp.tcp_cylinder_near_target_tangential_speed,
                    weight=-0.5,
                    params={**axis, "gate_distance": 0.05},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v43_inward_overshoot",
                RewTerm(func=mdp.tcp_cylinder_inward_speed_after_overshoot, weight=-5.0, params=axis),
            )


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV44EnvCfg(
    JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg
):
    """V4.3 with full-gripper collision coverage and a static radial safety barrier."""

    scene: JZGraspTwoTargetFullContactSceneCfg = JZGraspTwoTargetFullContactSceneCfg(
        num_envs=256,
        env_spacing=2.5,
    )

    def __post_init__(self):
        super().__post_init__()

        sides = (
            (
                "left",
                SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "object",
                (
                    "left_narrow_contact",
                    "left_wide_contact",
                    "left_narrow1_contact",
                    "left_narrow2_contact",
                    "left_wide1_contact",
                    "left_wide2_contact",
                ),
                ("left_palm9_contact", "left_palm10_contact"),
            ),
            (
                "right",
                SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "right_object",
                (
                    "right_narrow_contact",
                    "right_wide_contact",
                    "right_narrow1_contact",
                    "right_narrow2_contact",
                    "right_wide1_contact",
                    "right_wide2_contact",
                ),
                ("right_palm9_contact", "right_palm10_contact"),
            ),
        )
        for side, tcp, object_name, finger_sensors, palm_sensors in sides:
            object_cfg = SceneEntityCfg(object_name)
            axis = {
                "desired_radial_distance": 0.05,
                "tcp_asset_cfg": tcp,
                "object_cfg": object_cfg,
            }
            all_contact = {
                "sensor_names": finger_sensors + palm_sensors,
                "force_threshold": 0.5,
            }

            getattr(self.rewards, f"{side}_v43_radial_velocity").params["inside_weight_scale"] = 3.0
            getattr(self.rewards, f"{side}_v4_height_tracking_fine").weight = 1.0
            getattr(self.rewards, f"{side}_v41_height_error").weight = -3.0
            getattr(self.rewards, f"{side}_v4_stable_pregrasp").params.update(
                radial_tolerance=0.005,
                height_tolerance=0.02,
                speed_threshold=0.03,
                force_threshold=0.5,
                additional_sensor_names=finger_sensors[2:] + palm_sensors,
            )
            getattr(self.rewards, f"{side}_v4_radial_progress").params.update(
                force_threshold=0.5,
                additional_sensor_names=finger_sensors[2:] + palm_sensors,
            )

            setattr(self.rewards, f"{side}_v2_contact_terminal", None)
            setattr(self.rewards, f"{side}_v4_persistent_contact", None)
            setattr(
                self.rewards,
                f"{side}_v44_safety_barrier",
                RewTerm(
                    func=mdp.tcp_cylinder_static_safety_penetration,
                    weight=-3.0,
                    params={**axis, "barrier_width": 0.02},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v44_full_contact",
                RewTerm(func=mdp.any_filtered_contact_penalty, weight=-3.0, params=all_contact),
            )
            setattr(
                self.rewards,
                f"{side}_v44_palm_contact",
                RewTerm(
                    func=mdp.any_filtered_contact_penalty,
                    weight=-2.0,
                    params={"sensor_names": palm_sensors, "force_threshold": 0.5},
                ),
            )
            setattr(
                self.rewards,
                f"{side}_v44_persistent_contact",
                RewTerm(
                    func=mdp.persistent_multi_sensor_contact_penalty,
                    weight=-2.0,
                    params={
                        **all_contact,
                        "reward_key": f"{side}_v44_full_contact_streak",
                        "hold_steps": 15,
                    },
                ),
            )


@configclass
class JZGraspApproach3DTwoTargetSurfacePregraspV44EnvCfg_PLAY(
    JZGraspApproach3DTwoTargetSurfacePregraspV44EnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class JZGraspApproach3DTwoTargetGraspCloseEnvCfg(JZGraspApproach3DTwoTargetTrackEnvCfg):
    """First closure curriculum: approach, close near the cylinder, and hold bilateral fingertip contact."""

    scene: JZGraspTwoTargetCloseSceneCfg = JZGraspTwoTargetCloseSceneCfg(num_envs=256, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.spawn.activate_contact_sensors = True
        self.scene.object.spawn.rigid_props.kinematic_enabled = False
        self.scene.object.spawn.rigid_props.disable_gravity = False
        self.scene.right_object.spawn.rigid_props.kinematic_enabled = False
        self.scene.right_object.spawn.rigid_props.disable_gravity = False

        left_tcp_cfg = SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)
        right_tcp_cfg = SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)

        self.rewards.left_gripper_early_close = RewTerm(
            func=mdp.gripper_early_close_penalty,
            weight=-0.3,
            params={
                "far_threshold": 0.10,
                "tcp_asset_cfg": left_tcp_cfg,
                "narrow_joint_name": "left_gripper_narrow_joint",
                "wide_joint_name": "left_gripper_wide_joint",
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_gripper_early_close = RewTerm(
            func=mdp.gripper_early_close_penalty,
            weight=-0.3,
            params={
                "far_threshold": 0.10,
                "tcp_asset_cfg": right_tcp_cfg,
                "narrow_joint_name": "right_gripper_narrow_joint",
                "wide_joint_name": "right_gripper_wide_joint",
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_gripper_close_near_object = RewTerm(
            func=mdp.gripper_close_when_pregrasp_ready,
            weight=0.8,
            params={
                "distance_threshold": 0.07,
                "speed_threshold": 0.04,
                "tcp_asset_cfg": left_tcp_cfg,
                "narrow_joint_name": "left_gripper_narrow_joint",
                "wide_joint_name": "left_gripper_wide_joint",
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_gripper_close_near_object = RewTerm(
            func=mdp.gripper_close_when_pregrasp_ready,
            weight=0.8,
            params={
                "distance_threshold": 0.07,
                "speed_threshold": 0.04,
                "tcp_asset_cfg": right_tcp_cfg,
                "narrow_joint_name": "right_gripper_narrow_joint",
                "wide_joint_name": "right_gripper_wide_joint",
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_gripper_contact_reward = RewTerm(
            func=mdp.dual_finger_contact_reward,
            weight=2.0,
            params={
                "narrow_sensor_name": "left_narrow_contact",
                "wide_sensor_name": "left_wide_contact",
                "force_threshold": 0.5,
                "narrow_joint_name": "left_gripper_narrow_joint",
                "wide_joint_name": "left_gripper_wide_joint",
            },
        )
        self.rewards.right_gripper_contact_reward = RewTerm(
            func=mdp.dual_finger_contact_reward,
            weight=2.0,
            params={
                "narrow_sensor_name": "right_narrow_contact",
                "wide_sensor_name": "right_wide_contact",
                "force_threshold": 0.5,
                "narrow_joint_name": "right_gripper_narrow_joint",
                "wide_joint_name": "right_gripper_wide_joint",
            },
        )
        self.rewards.left_stable_finger_contact = RewTerm(
            func=mdp.dual_finger_contact_dwell_reward,
            weight=1.0,
            params={
                "reward_key": "left_stable_finger_contact_streak",
                "narrow_sensor_name": "left_narrow_contact",
                "wide_sensor_name": "left_wide_contact",
                "force_threshold": 0.5,
                "tcp_asset_cfg": left_tcp_cfg,
                "narrow_joint_name": "left_gripper_narrow_joint",
                "wide_joint_name": "left_gripper_wide_joint",
                "speed_threshold": 0.05,
                "hold_steps": 15,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_stable_finger_contact = RewTerm(
            func=mdp.dual_finger_contact_dwell_reward,
            weight=1.0,
            params={
                "reward_key": "right_stable_finger_contact_streak",
                "narrow_sensor_name": "right_narrow_contact",
                "wide_sensor_name": "right_wide_contact",
                "force_threshold": 0.5,
                "tcp_asset_cfg": right_tcp_cfg,
                "narrow_joint_name": "right_gripper_narrow_joint",
                "wide_joint_name": "right_gripper_wide_joint",
                "speed_threshold": 0.05,
                "hold_steps": 15,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        # This first closure curriculum deliberately has no lift reward.
        self.rewards.object_lifted = None
        self.rewards.bimanual_stable_grasp_dwell = None
        self.terminations.right_object_dropped = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.78, "asset_cfg": SceneEntityCfg("right_object")},
        )


@configclass
class JZGraspApproach3DTwoTargetGraspCloseEnvCfg_PLAY(JZGraspApproach3DTwoTargetGraspCloseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
