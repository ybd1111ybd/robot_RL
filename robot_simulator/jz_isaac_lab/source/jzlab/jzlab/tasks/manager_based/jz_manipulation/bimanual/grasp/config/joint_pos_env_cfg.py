"""Joint-position-control specialization for JZ bimanual grasp."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .. import mdp
from ..grasp_env_cfg import GraspEnvCfg, GraspSceneCfg, OBJECT_INITIAL_CENTER_Z
from ....assets.jz_bimanual import JZ_BIMANUAL_HIGH_PD_CFG
from ....constants import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_TCP_POSITION_LINKS,
)


_ACTION_SCALE = 1.0
_FIXED_ARM_ACTION_SCALE = 0.20
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

        # Target vectors are enough for this fixed 3D curriculum and avoid a misleading single object-position term.
        self.observations.policy.object_position = None
        self.observations.policy.left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.observations.policy.right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_side_target_vector_b,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )

        self.rewards.left_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_tcp_object_progress = RewTerm(
            func=mdp.tcp_to_object_side_target_distance,
            weight=-2.0,
            params={
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.right_tcp_object_tracking.params = {
            "std": 0.25,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "target_offset": right_target_offset,
            "object_cfg": SceneEntityCfg("right_object"),
        }
        self.rewards.left_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "target_offset": left_target_offset,
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.right_tcp_object_tracking_fine.params = {
            "std": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
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

        self.actions.left_gripper_action.scale = 0.0
        self.actions.right_gripper_action.scale = 0.0

        self.rewards.left_joint_vel_near_object = RewTerm(
            func=mdp.joint_vel_l2_when_close_to_object_side_target,
            weight=-2.0e-3,
            params={
                "threshold": 0.08,
                "joint_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
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
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_tcp_relative_speed_near_object = RewTerm(
            func=mdp.tcp_speed_l2_when_close_to_object_side_target,
            weight=-5.0e-2,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "target_offset": left_target_offset,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.rewards.right_tcp_relative_speed_near_object = RewTerm(
            func=mdp.tcp_speed_l2_when_close_to_object_side_target,
            weight=-5.0e-2,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
                "target_offset": right_target_offset,
                "object_cfg": SceneEntityCfg("right_object"),
            },
        )
        self.rewards.left_action_rate_near_object = RewTerm(
            func=mdp.action_rate_l2_when_close_to_object_side_target,
            weight=-5.0e-3,
            params={
                "threshold": 0.08,
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
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
                "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
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
                "left_tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
                "right_tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
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
