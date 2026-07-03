"""Joint-position-control specialization for JZ bimanual grasp."""

from __future__ import annotations

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from .. import mdp
from ..grasp_env_cfg import GraspEnvCfg
from ....assets.jz_bimanual import JZ_BIMANUAL_HIGH_PD_CFG
from ....constants import LEFT_ARM_JOINTS, LEFT_GRIPPER_JOINTS, RIGHT_ARM_JOINTS, RIGHT_GRIPPER_JOINTS


_ACTION_SCALE = 1.0


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
