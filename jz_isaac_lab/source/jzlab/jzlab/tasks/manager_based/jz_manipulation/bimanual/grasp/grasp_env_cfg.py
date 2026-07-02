"""Base environment config for JZ bimanual grasp."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils import configclass

from . import mdp
from ...constants import (
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    BODY_JOINTS,
    RIGHT_TCP_POSITION_LINKS,
)

TABLE_CENTER_Z = 0.83
TABLE_THICKNESS = 0.06
TABLE_TOP_Z = TABLE_CENTER_Z + TABLE_THICKNESS / 2.0
TABLE_BOTTOM_Z = TABLE_CENTER_Z - TABLE_THICKNESS / 2.0
TABLE_SIZE = (1.0, 0.90, TABLE_THICKNESS)
TABLE_LEG_HEIGHT = TABLE_BOTTOM_Z
TABLE_LEG_CENTER_Z = TABLE_LEG_HEIGHT / 2.0
TABLE_LEG_SIZE = (0.06, 0.06, TABLE_LEG_HEIGHT)
OBJECT_HEIGHT = 0.12
OBJECT_INITIAL_CENTER_Z = TABLE_TOP_Z + OBJECT_HEIGHT / 2.0
OBJECT_LIFT_SUCCESS_Z = OBJECT_INITIAL_CENTER_Z + 0.08


@configclass
class GraspSceneCfg(InteractiveSceneCfg):
    """Scene with the JZ robot and one graspable rigid cuboid."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: ArticulationCfg = MISSING
    table_leg_front_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80 - 0.34, 0.0 - 0.37, TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_LEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
        ),
    )
    table_leg_front_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegFrontRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80 - 0.34, 0.0 + 0.37, TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_LEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
        ),
    )
    table_leg_back_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBackLeft",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80 + 0.34, 0.0 - 0.37, TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_LEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
        ),
    )
    table_leg_back_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TableLegBackRight",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80 + 0.34, 0.0 + 0.37, TABLE_LEG_CENTER_Z)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_LEG_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.42, 0.38)),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.80, 0.0, TABLE_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.42)),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.80, 0.0, OBJECT_INITIAL_CENTER_Z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.12),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.35, 0.15)),
        ),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
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
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
        )
        left_gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS)},
        )
        right_gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS)},
        )
        left_gripper_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS)},
        )
        right_gripper_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS)},
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        left_tcp_position = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
        )
        right_tcp_position = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
        )
        left_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_vector_b,
            params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
        )
        right_tcp_to_object = ObsTerm(
            func=mdp.fingertip_midpoint_to_object_vector_b,
            params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_arm_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
            "position_range": (-0.15, 0.15),
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
    reset_object_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.12, 0.12),
                "y": (-0.16, 0.16),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.6, 0.6),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    # --- Continuous progress reward (new, most important) ---
    left_tcp_object_progress = RewTerm(
        func=mdp.tcp_to_object_progress_reward,
        weight=1.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_object_progress = RewTerm(
        func=mdp.tcp_to_object_progress_reward,
        weight=1.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    # --- Tanh shaping (kept as auxiliary signal, reduced weight) ---
    left_tcp_object_tracking = RewTerm(
        func=mdp.tcp_to_object_distance_tanh,
        weight=0.10,
        params={"std": 0.10, "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_object_tracking = RewTerm(
        func=mdp.tcp_to_object_distance_tanh,
        weight=0.10,
        params={"std": 0.10, "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    left_tcp_object_tracking_fine = RewTerm(
        func=mdp.tcp_to_object_distance_tanh,
        weight=0.10,
        params={"std": 0.03, "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_object_tracking_fine = RewTerm(
        func=mdp.tcp_to_object_distance_tanh,
        weight=0.10,
        params={"std": 0.03, "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    # --- NEW: TCP approach speed reward - moderate speed near object ---
    left_tcp_approach_speed = RewTerm(
        func=mdp.tcp_approach_speed_reward,
        weight=1.0,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_approach_speed = RewTerm(
        func=mdp.tcp_approach_speed_reward,
        weight=1.0,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    # --- TCP closing speed penalty - prevents overshoot ---
    left_tcp_closing_speed_penalty = RewTerm(
        func=mdp.tcp_closing_speed_penalty,
        weight=-0.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_closing_speed_penalty = RewTerm(
        func=mdp.tcp_closing_speed_penalty,
        weight=-0.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    bimanual_grasp_ready_bonus = RewTerm(
        func=mdp.bimanual_tcp_close_to_object_bonus,
        weight=1.0,
        params={
            "threshold": 0.06,
            "left_tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "right_tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )
    left_gripper_close_near_object = RewTerm(
        func=mdp.gripper_closed_when_near_object,
        weight=0.4,
        params={
            "threshold": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "narrow_joint_name": "left_gripper_narrow_joint",
            "wide_joint_name": "left_gripper_wide_joint",
        },
    )
    left_gripper_contact_reward = RewTerm(
        func=mdp.gripper_contact_reward_when_near_object,
        weight=0.35,
        params={
            "distance_threshold": 0.07,
            "effort_threshold": 0.75,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS),
        },
    )
    right_gripper_close_near_object = RewTerm(
        func=mdp.gripper_closed_when_near_object,
        weight=0.4,
        params={
            "threshold": 0.08,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "narrow_joint_name": "right_gripper_narrow_joint",
            "wide_joint_name": "right_gripper_wide_joint",
        },
    )
    right_gripper_contact_reward = RewTerm(
        func=mdp.gripper_contact_reward_when_near_object,
        weight=0.35,
        params={
            "distance_threshold": 0.07,
            "effort_threshold": 0.75,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS),
        },
    )
    object_lifted = RewTerm(
        func=mdp.object_is_lifted,
        weight=5.0,
        params={"minimal_height": OBJECT_LIFT_SUCCESS_Z},
    )
    bimanual_stable_grasp_dwell = RewTerm(
        func=mdp.bimanual_tcp_stable_near_object_dwell_reward,
        weight=0.6,
        params={
            "threshold": 0.055,
            "speed_threshold": 0.06,
            "hold_steps": 10,
            "left_tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "right_tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )
    # Penalize only aggressive motion when close to object (not globally from step 0)
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2_when_close_to_object,
        weight=-5.0e-4,
        params={
            "threshold": 0.10,
            "joint_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
        },
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2_when_close_to_object,
        weight=-5.0e-4,
        params={
            "threshold": 0.10,
            "joint_asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-5)
    action_max_abs_penalty = RewTerm(func=mdp.action_max_abs, weight=-1.0e-4)
    left_joint_vel_near_object = RewTerm(
        func=mdp.joint_vel_l2_when_close_to_object,
        weight=-1.0e-4,
        params={
            "threshold": 0.055,
            "joint_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS),
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
        },
    )
    right_joint_vel_near_object = RewTerm(
        func=mdp.joint_vel_l2_when_close_to_object,
        weight=-1.0e-4,
        params={
            "threshold": 0.055,
            "joint_asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS),
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )
    left_tcp_relative_speed_near_object = RewTerm(
        func=mdp.tcp_relative_speed_l2_when_close_to_object,
        weight=-2.0e-4,
        params={
            "threshold": 0.055,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
        },
    )
    right_tcp_relative_speed_near_object = RewTerm(
        func=mdp.tcp_relative_speed_l2_when_close_to_object,
        weight=-2.0e-4,
        params={
            "threshold": 0.055,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )
    left_action_rate_near_object = RewTerm(
        func=mdp.action_rate_l2_when_close_to_object,
        weight=-1.0e-4,
        params={
            "threshold": 0.055,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "action_name": "left_arm_action",
        },
    )
    right_action_rate_near_object = RewTerm(
        func=mdp.action_rate_l2_when_close_to_object,
        weight=-1.0e-4,
        params={
            "threshold": 0.055,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "action_name": "right_arm_action",
        },
    )
    left_gripper_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS)},
    )
    left_gripper_joint_vel_after_contact = RewTerm(
        func=mdp.gripper_joint_speed_penalty_after_contact,
        weight=-7.5e-4,
        params={
            "distance_threshold": 0.07,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS),
        },
    )
    right_gripper_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS)},
    )
    right_gripper_joint_vel_after_contact = RewTerm(
        func=mdp.gripper_joint_speed_penalty_after_contact,
        weight=-7.5e-4,
        params={
            "distance_threshold": 0.07,
            "tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_GRIPPER_JOINTS),
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.78, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    # No global penalty curriculum - penalties are gated by proximity instead
    pass
    left_joint_vel_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel_near_object", "weight": -0.0002, "num_steps": 3000},
    )
    right_joint_vel_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel_near_object", "weight": -0.0002, "num_steps": 3000},
    )
    left_tcp_relative_speed_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_tcp_relative_speed_near_object", "weight": -0.0004, "num_steps": 3000},
    )
    right_tcp_relative_speed_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_tcp_relative_speed_near_object", "weight": -0.0004, "num_steps": 3000},
    )
    left_action_rate_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_action_rate_near_object", "weight": -0.0002, "num_steps": 3000},
    )
    right_action_rate_near_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_action_rate_near_object", "weight": -0.0002, "num_steps": 3000},
    )


@configclass
class GraspEnvCfg(ManagerBasedRLEnvCfg):
    scene: GraspSceneCfg = GraspSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # --- Critical new rewards ---
    grasp_success = RewTerm(func=mdp.grasp_success_bonus, weight=15.0, params={})
    object_lin_vel_penalty = RewTerm(
        func=mdp.object_lin_vel_penalty,
        weight=-0.5,
        params={"gripper_asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_GRIPPER_JOINTS)},
    )
    object_ang_vel_penalty = RewTerm(func=mdp.object_ang_vel_penalty, weight=-0.5, params={})
    table_penetration = RewTerm(func=mdp.table_penetration_penalty, weight=-1.0, params={})
    left_tcp_approach_orientation = RewTerm(
        func=mdp.tcp_approach_orientation_reward,
        weight=0.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
    )
    right_tcp_approach_orientation = RewTerm(
        func=mdp.tcp_approach_orientation_reward,
        weight=0.5,
        params={"tcp_asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
    )
    arm_asymmetry_penalty = RewTerm(
        func=mdp.arm_asymmetry_penalty,
        weight=-0.5,
        params={
            "left_tcp_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "right_tcp_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
        },
    )

    commands = None

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 8.0
        self.viewer.eye = (2.8, 2.2, 2.0)
        self.sim.dt = 1.0 / 60.0
        self.sim.physx.enable_external_forces_every_iteration = False
        # Soften object physics to prevent GPU crash
        if hasattr(self.scene, 'object') and hasattr(self.scene.object, 'spawn'):
            self.scene.object.spawn.rigid_props.linear_damping = 5.0
            self.scene.object.spawn.rigid_props.angular_damping = 5.0
            self.scene.object.spawn.rigid_props.max_depenetration_velocity = 1.0
