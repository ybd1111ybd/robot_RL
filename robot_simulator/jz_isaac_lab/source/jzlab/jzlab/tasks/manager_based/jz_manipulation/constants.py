"""Shared constants for JZ dual-arm manipulation tasks."""

from __future__ import annotations

import math


BODY_JOINTS = [f"body_joint{i}" for i in range(1, 6)]
LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
LEFT_GRIPPER_JOINTS = ["left_gripper_narrow_joint", "left_gripper_wide_joint"]
RIGHT_GRIPPER_JOINTS = ["right_gripper_narrow_joint", "right_gripper_wide_joint"]

LEFT_TCP_POSITION_LINKS = ["left_arm_link7"]
RIGHT_TCP_POSITION_LINKS = ["right_arm_link7"]
LEFT_TCP_ORIENTATION_LINK = "left_arm_link9"
RIGHT_TCP_ORIENTATION_LINK = "right_arm_link9"

# Fixed right-handed grasp target frames in world coordinates. The right frame
# has X outward (-world Y), Y forward (+world X), Z up. The left is its 180-deg
# rotation around world Z.
LEFT_GRASP_TARGET_QUAT_W = (0.70710678, 0.0, 0.0, 0.70710678)
RIGHT_GRASP_TARGET_QUAT_W = (0.70710678, 0.0, 0.0, -0.70710678)

# Local corrections from the asymmetric URDF link9 frames to symmetric semantic
# TCP frames. They are calibrated at the fixed default arm pose above.
LEFT_TCP_ORIENTATION_OFFSET_QUAT = (0.68512633, -0.17494122, -0.68512338, 0.17493835)
RIGHT_TCP_ORIENTATION_OFFSET_QUAT = (-0.36959607, -0.60282643, -0.36959748, -0.60282397)

# Local points used by the synthetic TCP helper.
TCP_CONTACT_LOCAL_OFFSETS = {
    "left_arm_link7": (0.0385, 0.23772, 0.0),
    "left_gripper_narrow1_link": (0.0, 0.0, 0.0),
    "left_gripper_wide1_link": (0.0, 0.0, 0.0),
    "left_gripper_narrow3_link": (0.0214, 0.0054, 0.0),
    "left_gripper_wide3_link": (0.0214, -0.0053, 0.0),
    "right_arm_link7": (0.0385, -0.23772, 0.0),
    "right_gripper_narrow1_link": (0.0, 0.0, 0.0),
    "right_gripper_wide1_link": (0.0, 0.0, 0.0),
    "right_gripper_narrow3_link": (0.0214, 0.0054, 0.0),
    "right_gripper_wide3_link": (0.0214, -0.0053, 0.0),
}

# Collision-mesh inner surfaces found by ray casting from the legacy outer
# narrow3/wide3 points toward the finger gap. These are the pre-grasp reward
# contact representatives, distinct from the synthetic TCP offsets above.
FINGERTIP_INNER_CONTACT_LOCAL_OFFSETS = {
    "left_gripper_narrow3_link": (0.0214, -0.01704202, 0.0),
    "left_gripper_wide3_link": (0.0214, 0.01789719, 0.0),
    "right_gripper_narrow3_link": (0.0214, -0.01704202, 0.0),
    "right_gripper_wide3_link": (0.0214, 0.01789719, 0.0),
}


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

_NARROW_JOINT_SAFE_CLOSED = -0.0501
_WIDE_JOINT_SAFE_CLOSED = -0.0499

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
