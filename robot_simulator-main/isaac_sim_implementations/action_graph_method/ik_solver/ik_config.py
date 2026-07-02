"""
IK Configuration Module

Manages all IK-related parameters and settings.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class IKConfig:
    """Configuration for IK solver."""

    # Control mode
    control_mode: str = "joint"  # joint | ee_pose | auto

    # Topic names
    left_ee_topic: str = "/arm_left/ee_target_pose"
    right_ee_topic: str = "/arm_right/ee_target_pose"
    left_ee_topic_compat: str = "/arm_left/ee_target"
    right_ee_topic_compat: str = "/arm_right/ee_target"

    # End-effector links
    left_ee_link: str = "left_gripper_center_tcp"
    right_ee_link: str = "right_gripper_center_tcp"

    # Joint groups
    left_arm_joints: List[str] = None
    right_arm_joints: List[str] = None

    # IK algorithm parameters
    ik_rate_hz: float = 50.0
    ik_damping: float = 0.15
    ik_pos_gain: float = 0.8
    ik_ori_gain: float = 0.6
    ik_max_dq: float = 0.15
    ik_max_joint_vel: float = 1.5
    ik_pos_tol: float = 0.02
    ik_ori_tol: float = 0.05
    ik_timeout_sec: float = 1.0
    ik_hold_last_target: bool = True
    ik_enable_orientation: bool = False
    ik_direct_apply: bool = False
    ik_perf_log_interval_sec: float = 5.0
    ik_perf_warn_threshold_ms: float = 5.0

    # Command topic remapping (for output)
    left_cmd_topic: str = "/arm_left/joint_commands"
    right_cmd_topic: str = "/arm_right/joint_commands"

    def __post_init__(self):
        """Initialize default joint groups if not provided."""
        if self.left_arm_joints is None:
            self.left_arm_joints = [
                "left_arm_joint1",
                "left_arm_joint2",
                "left_arm_joint3",
                "left_arm_joint4",
                "left_arm_joint5",
                "left_arm_joint6",
                "left_arm_joint7",
            ]

        if self.right_arm_joints is None:
            self.right_arm_joints = [
                "right_arm_joint1",
                "right_arm_joint2",
                "right_arm_joint3",
                "right_arm_joint4",
                "right_arm_joint5",
                "right_arm_joint6",
                "right_arm_joint7",
            ]


    @property
    def ik_dt(self) -> float:
        """Time step for IK updates."""
        return 1.0 / self.ik_rate_hz

    def is_ik_enabled(self) -> bool:
        """Check if IK mode is enabled."""
        return self.control_mode in ["ee_pose", "auto"]
