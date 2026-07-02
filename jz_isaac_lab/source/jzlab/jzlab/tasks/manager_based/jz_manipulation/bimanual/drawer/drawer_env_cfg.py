"""Base environment config for JZ bimanual drawer opening."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from ...assets.jz_bimanual import JZ_BIMANUAL_HIGH_PD_CFG
from ...constants import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_TCP_ORIENTATION_LINK,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_TCP_ORIENTATION_LINK,
    BODY_JOINTS,
    RIGHT_TCP_POSITION_LINKS,
)

_WIDE_JOINT_SAFE_MIN = -0.049
_LEFT_GRIPPER_CLOSED = {"left_gripper_narrow_joint": -0.05, "left_gripper_wide_joint": _WIDE_JOINT_SAFE_MIN}
_RIGHT_GRIPPER_CLOSED = {"right_gripper_narrow_joint": -0.05, "right_gripper_wide_joint": _WIDE_JOINT_SAFE_MIN}

NEUTRAL_INIT_JOINTS = {
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
    "left_gripper_narrow_joint": -0.05,
    "left_gripper_wide_joint": _WIDE_JOINT_SAFE_MIN,
    "right_arm_joint1": 0.65,
    "right_arm_joint2": 1.16,
    "right_arm_joint3": -0.45,
    "right_arm_joint4": 0.90,
    "right_arm_joint5": -1.01,
    "right_arm_joint6": 0.05,
    "right_arm_joint7": 0.01,
    "right_gripper_narrow_joint": -0.05,
    "right_gripper_wide_joint": _WIDE_JOINT_SAFE_MIN,
}

_CABINET_HANDLE_OFFSET_POS = (0.222, 0.0, 0.005)
_CABINET_HANDLE_OFFSET_ROT = (0.5, 0.5, -0.5, -0.5)
_TABLE_TOP_THICKNESS = 0.025
_TABLE_TOP_CENTER_X = 1.35
_TABLE_TOP_CENTER_Y = 0.0
_TABLE_TOP_CENTER_Z = 1.0
_TABLE_SIZE_X = 0.8
_TABLE_SIZE_Y = 1.0
_TABLE_LEG_MARGIN_X = 0.06
_TABLE_LEG_MARGIN_Y = 0.08
_TABLE_LEG_SIZE_X = 0.05
_TABLE_LEG_SIZE_Y = 0.05
_TABLE_LEG_HEIGHT = _TABLE_TOP_CENTER_Z - _TABLE_TOP_THICKNESS / 2.0
_TABLE_LEG_CENTER_Z = _TABLE_LEG_HEIGHT / 2.0
_CABINET_CENTER_X = 1.35
_CABINET_CENTER_Y = 0.0
_CABINET_CENTER_Z = _TABLE_TOP_CENTER_Z + 0.3125


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


_LEFT_TCP_FRAME_ROT = _quat_from_rpy(0.0, math.pi / 2.0, 0.0)
_RIGHT_TCP_FRAME_ROT = _quat_from_rpy(-math.pi, -math.pi / 2.0, 0.0)


@configclass
class JZDrawerSceneCfg(InteractiveSceneCfg):
    """Scene with the JZ robot, support table, and a Sektion cabinet."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    robot: ArticulationCfg = MISSING

    table_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableTop",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(_TABLE_TOP_CENTER_X, _TABLE_TOP_CENTER_Y, _TABLE_TOP_CENTER_Z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_SIZE_X, _TABLE_SIZE_Y, _TABLE_TOP_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
        ),
    )

    table_leg_front_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                _TABLE_TOP_CENTER_X + _TABLE_SIZE_X / 2.0 - _TABLE_LEG_MARGIN_X,
                _TABLE_TOP_CENTER_Y + _TABLE_SIZE_Y / 2.0 - _TABLE_LEG_MARGIN_Y,
                _TABLE_LEG_CENTER_Z,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIZE_X, _TABLE_LEG_SIZE_Y, _TABLE_LEG_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.33, 0.22)),
        ),
    )
    table_leg_front_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                _TABLE_TOP_CENTER_X + _TABLE_SIZE_X / 2.0 - _TABLE_LEG_MARGIN_X,
                _TABLE_TOP_CENTER_Y - _TABLE_SIZE_Y / 2.0 + _TABLE_LEG_MARGIN_Y,
                _TABLE_LEG_CENTER_Z,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIZE_X, _TABLE_LEG_SIZE_Y, _TABLE_LEG_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.33, 0.22)),
        ),
    )
    table_leg_back_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBackLeft",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                _TABLE_TOP_CENTER_X - _TABLE_SIZE_X / 2.0 + _TABLE_LEG_MARGIN_X,
                _TABLE_TOP_CENTER_Y + _TABLE_SIZE_Y / 2.0 - _TABLE_LEG_MARGIN_Y,
                _TABLE_LEG_CENTER_Z,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIZE_X, _TABLE_LEG_SIZE_Y, _TABLE_LEG_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.33, 0.22)),
        ),
    )
    table_leg_back_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBackRight",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                _TABLE_TOP_CENTER_X - _TABLE_SIZE_X / 2.0 + _TABLE_LEG_MARGIN_X,
                _TABLE_TOP_CENTER_Y - _TABLE_SIZE_Y / 2.0 + _TABLE_LEG_MARGIN_Y,
                _TABLE_LEG_CENTER_Z,
            ),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=(_TABLE_LEG_SIZE_X, _TABLE_LEG_SIZE_Y, _TABLE_LEG_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.33, 0.22)),
        ),
    )

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
            scale=(0.75, 0.75, 0.75),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(_CABINET_CENTER_X, _CABINET_CENTER_Y, _CABINET_CENTER_Z),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "bottom_drawer": ImplicitActuatorCfg(
                joint_names_expr=["drawer_bottom_joint"],
                effort_limit=200.0,
                velocity_limit=5.0,
                stiffness=0.0,
                damping=30.0,
            ),
            "top_drawer_lock": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint"],
                effort_limit=400.0,
                velocity_limit=0.1,
                stiffness=5.0e3,
                damping=500.0,
            ),
            "door_lock": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=400.0,
                velocity_limit=0.1,
                stiffness=5.0e3,
                damping=500.0,
            ),
        },
    )

    left_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_TCP_POSITION_LINKS[0]}",
                name="left_finger_narrow",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_TCP_POSITION_LINKS[1]}",
                name="left_finger_wide",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{LEFT_TCP_ORIENTATION_LINK}",
                name="left_tcp_orientation",
                offset=OffsetCfg(rot=_LEFT_TCP_FRAME_ROT),
            ),
        ],
    )

    right_ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_TCP_POSITION_LINKS[0]}",
                name="right_finger_narrow",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_TCP_POSITION_LINKS[1]}",
                name="right_finger_wide",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{RIGHT_TCP_ORIENTATION_LINK}",
                name="right_tcp_orientation",
                offset=OffsetCfg(rot=_RIGHT_TCP_FRAME_ROT),
            ),
        ],
    )

    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_bottom",
                name="drawer_handle_bottom",
                offset=OffsetCfg(pos=_CABINET_HANDLE_OFFSET_POS, rot=_CABINET_HANDLE_OFFSET_ROT),
            ),
        ],
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING
    left_gripper_action: ActionTerm = MISSING
    right_gripper_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS)},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        drawer_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        drawer_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"])},
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        tcp_to_handle = ObsTerm(func=mdp.tcp_to_handle_distances_b, noise=Unoise(n_min=-0.005, n_max=0.005))
        actions = ObsTerm(func=mdp.last_action_padded, params={"dim": 32}, noise=Unoise(n_min=-0.002, n_max=0.002))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
            "position_range": (-0.12, 0.12),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_body_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINTS),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_gripper_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    approach_handle = RewTerm(func=mdp.approach_handle, weight=2.0, params={'std': 0.15})
    align_handle = RewTerm(func=mdp.align_handle, weight=1.0)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.8,
        params={
            'distance_threshold': 0.06,
            'closed_joint_target': 0.08,
        },
    )
    drawer_opening_progress = RewTerm(
        func=mdp.drawer_opening_progress_gated,
        weight=6.0,
        params={},
    )
    drawer_open_success = RewTerm(
        func=mdp.drawer_open_success,
        weight=10.0,
        params={'open_threshold': 0.25},
    )
    right_arm_neutral = RewTerm(
        func=mdp.right_arm_stay_neutral,
        weight=-0.1,
        params={
            'joint_names': RIGHT_ARM_JOINTS,
            'joint_values': [
                NEUTRAL_INIT_JOINTS['right_arm_joint1'],
                NEUTRAL_INIT_JOINTS['right_arm_joint2'],
                NEUTRAL_INIT_JOINTS['right_arm_joint3'],
                NEUTRAL_INIT_JOINTS['right_arm_joint4'],
                NEUTRAL_INIT_JOINTS['right_arm_joint5'],
                NEUTRAL_INIT_JOINTS['right_arm_joint6'],
                NEUTRAL_INIT_JOINTS['right_arm_joint7'],
            ],
        },
    )
    table_penetration = RewTerm(
        func=mdp.table_penetration_penalty,
        weight=-5.0,
        params={
            'top_z': _TABLE_TOP_CENTER_Z,
            'x_min': _TABLE_TOP_CENTER_X - _TABLE_SIZE_X / 2.0,
            'x_max': _TABLE_TOP_CENTER_X + _TABLE_SIZE_X / 2.0,
            'y_min': _TABLE_TOP_CENTER_Y - _TABLE_SIZE_Y / 2.0,
            'y_max': _TABLE_TOP_CENTER_Y + _TABLE_SIZE_Y / 2.0,
            'margin': 0.02,
        },
    )

    # --- Additional drawer rewards (inspired by arms_ros2_control + IsaacLab cabinet) ---
    approach_speed = RewTerm(
        func=mdp.tcp_approach_speed_reward,
        weight=1.0,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    closing_speed_penalty = RewTerm(
        func=mdp.tcp_closing_speed_penalty,
        weight=-0.3,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    gripper_contact_effort = RewTerm(
        func=mdp.gripper_contact_effort,
        weight=0.5,
        params={"gripper_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS)},
    )
    cabinet_displacement_penalty = RewTerm(
        func=mdp.cabinet_displacement_penalty,
        weight=-1.0,
        params={},
    )
    drawer_stable_pull = RewTerm(
        func=mdp.drawer_stable_pull_reward,
        weight=0.5,
        params={},
    )

    action_rate_penalty = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    action_max_abs_penalty = RewTerm(func=mdp.action_max_abs, weight=-1.0e-3)
    joint_vel_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={
            'asset_cfg': SceneEntityCfg(
                'robot', joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS
            )
        },
    )


class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    action_rate_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_penalty", "weight": -0.001, "num_steps": 5000},
    )
    action_max_abs_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_max_abs_penalty", "weight": -0.002, "num_steps": 5000},
    )
    joint_vel_penalty = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_vel_penalty", "weight": -0.0005, "num_steps": 5000},
    )


@configclass
class JZDrawerEnvCfg(ManagerBasedRLEnvCfg):
    scene: JZDrawerSceneCfg = JZDrawerSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    commands = None

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (2.8, 2.2, 1.8)
        self.viewer.lookat = (_CABINET_CENTER_X, _CABINET_CENTER_Y, _TABLE_TOP_CENTER_Z)
        self.sim.dt = 1.0 / 60.0
        self.sim.physx.enable_external_forces_every_iteration = False

        self.scene.robot = JZ_BIMANUAL_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    **JZ_BIMANUAL_HIGH_PD_CFG.init_state.joint_pos,
                    **NEUTRAL_INIT_JOINTS,
                },
            ),
        )
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
        self.scene.robot.actuators["arm"].stiffness = 260.0
        self.scene.robot.actuators["arm"].damping = 36.0
        self.scene.robot.actuators["gripper"].stiffness = 400.0
        self.scene.robot.actuators["gripper"].damping = 30.0

        self.actions.left_arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=LEFT_ARM_JOINTS,
            scale=1.0,
            rescale_to_limits=True,
        )
        self.actions.right_arm_action = mdp.JointPositionToLimitsActionCfg(
            asset_name="robot",
            joint_names=RIGHT_ARM_JOINTS,
            scale=1.0,
            rescale_to_limits=True,
        )
        self.actions.left_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=LEFT_GRIPPER_JOINTS,
            open_command_expr={"left_gripper_narrow_joint": -1.2, "left_gripper_wide_joint": 1.2},
            close_command_expr=_LEFT_GRIPPER_CLOSED,
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=RIGHT_GRIPPER_JOINTS,
            open_command_expr={"right_gripper_narrow_joint": -1.2, "right_gripper_wide_joint": 1.2},
            close_command_expr=_RIGHT_GRIPPER_CLOSED,
        )


@configclass
class JZDrawerEnvCfg_PLAY(JZDrawerEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
