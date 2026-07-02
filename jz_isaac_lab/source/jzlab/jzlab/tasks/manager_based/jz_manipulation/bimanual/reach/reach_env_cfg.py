"""Base environment config for JZ bimanual reach."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
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
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from .orientation_presets import get_active_orientation_preset
from .workspace_data import WORKSPACE_DATA_PATH
from ...constants import (
    GRIPPER_MOUNT_QUAT,
    LEFT_ARM_JOINTS,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_TCP_POSITION_LINKS,
)


_, _LEFT_COMMAND_QUAT, _RIGHT_COMMAND_QUAT = get_active_orientation_preset()


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Scene with a single JZ robot replicated across environments."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )
    robot: ArticulationCfg = MISSING
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


@configclass
class CommandsCfg:
    """Dual end-effector pose command generators sampled in the robot base frame."""

    left_ee_pose = mdp.ReachableWorkspacePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        dataset_file=str(WORKSPACE_DATA_PATH),
        dataset_key="left_train_positions",
        use_fixed_quaternion=True,
        fixed_quaternion=_LEFT_COMMAND_QUAT,
        current_position_body_names=tuple(LEFT_TCP_POSITION_LINKS),
        current_quaternion_offset=GRIPPER_MOUNT_QUAT,
        curriculum_stage_fractions=(1.0, 1.0, 1.0),
        curriculum_stage_steps=(0, 0),
    )
    right_ee_pose = mdp.ReachableWorkspacePoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        dataset_file=str(WORKSPACE_DATA_PATH),
        dataset_key="right_train_positions",
        use_fixed_quaternion=True,
        fixed_quaternion=_RIGHT_COMMAND_QUAT,
        current_position_body_names=tuple(RIGHT_TCP_POSITION_LINKS),
        current_quaternion_offset=GRIPPER_MOUNT_QUAT,
        curriculum_stage_fractions=(1.0, 1.0, 1.0),
        curriculum_stage_steps=(0, 0),
    )


@configclass
class ActionsCfg:
    left_arm_action: ActionTerm = MISSING
    right_arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        left_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        right_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        left_tcp_pos = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)},
        )
        right_tcp_pos = ObsTerm(
            func=mdp.fingertip_midpoint_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)},
        )
        left_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        left_tcp_error = ObsTerm(
            func=mdp.fingertip_midpoint_position_command_error_vector_b,
            params={
                "command_name": "left_ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            },
        )
        right_tcp_error = ObsTerm(
            func=mdp.fingertip_midpoint_position_command_error_vector_b,
            params={
                "command_name": "right_ee_pose",
                "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            },
        )
        left_actions = ObsTerm(func=mdp.last_action, params={"action_name": "left_arm_action"})
        right_actions = ObsTerm(func=mdp.last_action, params={"action_name": "right_arm_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS),
            "position_range": (-0.25, 0.25),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    left_end_effector_position_tracking = RewTerm(
        func=mdp.fingertip_midpoint_position_command_error,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS), "command_name": "left_ee_pose"},
    )
    right_end_effector_position_tracking = RewTerm(
        func=mdp.fingertip_midpoint_position_command_error,
        weight=-0.20,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS), "command_name": "right_ee_pose"},
    )
    left_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.fingertip_midpoint_position_command_error_tanh,
        weight=0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS),
            "std": 0.10,
            "command_name": "left_ee_pose",
        },
    )
    right_end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.fingertip_midpoint_position_command_error_tanh,
        weight=0.20,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS),
            "std": 0.10,
            "command_name": "right_ee_pose",
        },
    )
    left_end_effector_position_progress = None
    right_end_effector_position_progress = None
    left_end_effector_goal_bonus = None
    right_end_effector_goal_bonus = None
    bimanual_end_effector_goal_bonus = None
    left_end_effector_stable_goal_bonus = None
    right_end_effector_stable_goal_bonus = None
    bimanual_end_effector_stable_goal_bonus = None
    left_end_effector_orientation_tracking = None
    right_end_effector_orientation_tracking = None
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    action_max_abs_penalty = None
    left_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)},
    )
    right_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)},
    )
    left_joint_deviation = None
    right_joint_deviation = None
    left_joint_vel_near_goal = None
    right_joint_vel_near_goal = None
    left_tcp_speed_near_goal = None
    right_tcp_speed_near_goal = None
    left_action_rate_near_goal = None
    right_action_rate_near_goal = None


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500},
    )
    left_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "left_joint_vel", "weight": -0.001, "num_steps": 4500},
    )
    right_joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "right_joint_vel", "weight": -0.001, "num_steps": 4500},
    )


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=512, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 24.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 60.0
        self.sim.physx.enable_external_forces_every_iteration = False
