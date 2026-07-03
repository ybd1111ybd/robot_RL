"""Joint-position-control specialization for JZ bimanual reach."""

from __future__ import annotations

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from .. import mdp
from ..orientation_presets import get_active_orientation_preset
from ..reach_env_cfg import ReachEnvCfg
from ....assets.jz_bimanual import JZ_BIMANUAL_HIGH_PD_CFG
from ....constants import LEFT_ARM_JOINTS, LEFT_TCP_ORIENTATION_LINK, RIGHT_ARM_JOINTS, RIGHT_TCP_ORIENTATION_LINK


_, _LEFT_COMMAND_QUAT, _RIGHT_COMMAND_QUAT = get_active_orientation_preset()

_ACTION_SCALE = 1.0


@configclass
class JZReachEnvCfg(ReachEnvCfg):
    """JZ dual-arm reach task with 14-DoF joint-position actions."""

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

        self.commands.left_ee_pose.body_name = LEFT_TCP_ORIENTATION_LINK
        self.commands.right_ee_pose.body_name = RIGHT_TCP_ORIENTATION_LINK
        self.commands.left_ee_pose.use_fixed_quaternion = True
        self.commands.right_ee_pose.use_fixed_quaternion = True
        self.commands.left_ee_pose.fixed_quaternion = _LEFT_COMMAND_QUAT
        self.commands.right_ee_pose.fixed_quaternion = _RIGHT_COMMAND_QUAT
        self.commands.left_ee_pose.resampling_time_range = (4.0, 4.0)
        self.commands.right_ee_pose.resampling_time_range = (4.0, 4.0)


@configclass
class JZReachEnvCfg_PLAY(JZReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.left_ee_pose.dataset_key = "left_eval_positions"
        self.commands.right_ee_pose.dataset_key = "right_eval_positions"
        self.commands.left_ee_pose.curriculum_stage_fractions = (1.0, 1.0, 1.0)
        self.commands.right_ee_pose.curriculum_stage_fractions = (1.0, 1.0, 1.0)
        self.commands.left_ee_pose.curriculum_stage_steps = (0, 0)
        self.commands.right_ee_pose.curriculum_stage_steps = (0, 0)
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
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
