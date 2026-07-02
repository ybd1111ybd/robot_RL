"""Shared constants and math helpers for the Action Graph runtime."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


LEFT_ARM_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
BODY_JOINTS = [f"body_joint{i}" for i in range(1, 6)]
LEFT_GRIPPER_JOINTS = ["left_gripper_narrow_joint", "left_gripper_wide_joint"]
RIGHT_GRIPPER_JOINTS = ["right_gripper_narrow_joint", "right_gripper_wide_joint"]
LEFT_GRIPPER_STATE_JOINTS = list(LEFT_GRIPPER_JOINTS)
RIGHT_GRIPPER_STATE_JOINTS = list(RIGHT_GRIPPER_JOINTS)
GRIPPER_CLOSED: Dict[str, float] = {"narrow": -0.05, "wide": -0.05}
GRIPPER_OPEN: Dict[str, float] = {"narrow": -1.2, "wide": 1.2}


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    arr = np.asarray(quat, dtype=float).reshape(4)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return arr / norm


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = _normalize_quaternion(lhs)
    rw, rx, ry, rz = _normalize_quaternion(rhs)
    return _normalize_quaternion(
        np.array(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=float,
        )
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(quat)
    v = np.asarray(vec, dtype=float).reshape(3)
    q_vec = q[1:4]
    q_w = float(q[0])
    return (
        2.0 * np.dot(q_vec, v) * q_vec
        + (q_w * q_w - np.dot(q_vec, q_vec)) * v
        + 2.0 * q_w * np.cross(q_vec, v)
    )


def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value
