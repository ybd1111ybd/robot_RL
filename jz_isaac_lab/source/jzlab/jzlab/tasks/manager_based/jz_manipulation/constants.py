"""Shared constants for JZ dual-arm manipulation tasks."""

from __future__ import annotations

import math


BODY_JOINTS = [f"body_joint{i}" for i in range(1, 6)]
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ["left_gripper_narrow_joint", "left_gripper_wide_joint"]
RIGHT_GRIPPER_JOINTS = ["right_gripper_narrow_joint", "right_gripper_wide_joint"]

LEFT_TCP_POSITION_LINKS = ["left_gripper_narrow3_link", "left_gripper_wide3_link"]
RIGHT_TCP_POSITION_LINKS = ["right_gripper_narrow3_link", "right_gripper_wide3_link"]
LEFT_TCP_ORIENTATION_LINK = "left_arm_link9"
RIGHT_TCP_ORIENTATION_LINK = "right_arm_link9"


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


GRIPPER_MOUNT_QUAT = _quat_from_rpy(-2.9951, -1.5708, -0.15964)

_NARROW_JOINT_SAFE_CLOSED = -0.049
_WIDE_JOINT_SAFE_CLOSED = -0.049

# URDF limits: narrow [-1.2, -0.05], wide [-0.05, 1.2]
# OPEN: narrow = -1.2, wide = 1.2
# CLOSED: narrow = -0.05, wide = -0.05
LEFT_GRIPPER_OPEN = {"left_gripper_narrow_joint": -1.2, "left_gripper_wide_joint": 1.2}
RIGHT_GRIPPER_OPEN = {"right_gripper_narrow_joint": -1.2, "right_gripper_wide_joint": 1.2}

LEFT_GRIPPER_CLOSED = {
    "left_gripper_narrow_joint": _NARROW_JOINT_SAFE_CLOSED,
    "left_gripper_wide_joint": _WIDE_JOINT_SAFE_CLOSED,
}
RIGHT_GRIPPER_CLOSED = {
    "right_gripper_narrow_joint": _NARROW_JOINT_SAFE_CLOSED,
    "right_gripper_wide_joint": _WIDE_JOINT_SAFE_CLOSED,
}
