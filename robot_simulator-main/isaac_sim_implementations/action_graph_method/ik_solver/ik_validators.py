"""
IK Input Validation Module

Provides validation functions for IK solver inputs.
"""

import numpy as np
from typing import Tuple, Optional
from .ik_exceptions import IKInputValidationError, IKConfigurationError


def validate_position(pos: np.ndarray, name: str = "position") -> None:
    """
    Validate position vector.

    Args:
        pos: Position array [x, y, z]
        name: Name for error messages

    Raises:
        IKInputValidationError: If position is invalid
    """
    if pos is None:
        raise IKInputValidationError(f"{name} is None")

    if not isinstance(pos, np.ndarray):
        raise IKInputValidationError(f"{name} must be numpy array, got {type(pos)}")

    if pos.shape != (3,):
        raise IKInputValidationError(f"{name} must have shape (3,), got {pos.shape}")

    if not np.all(np.isfinite(pos)):
        raise IKInputValidationError(f"{name} contains NaN or inf: {pos}")


def validate_quaternion(quat: np.ndarray, name: str = "quaternion") -> None:
    """
    Validate quaternion.

    Args:
        quat: Quaternion [w, x, y, z]
        name: Name for error messages

    Raises:
        IKInputValidationError: If quaternion is invalid
    """
    if quat is None:
        raise IKInputValidationError(f"{name} is None")

    if not isinstance(quat, np.ndarray):
        raise IKInputValidationError(f"{name} must be numpy array, got {type(quat)}")

    if quat.shape != (4,):
        raise IKInputValidationError(f"{name} must have shape (4,), got {quat.shape}")

    if not np.all(np.isfinite(quat)):
        raise IKInputValidationError(f"{name} contains NaN or inf: {quat}")

    # Check normalization (with tolerance)
    norm = np.linalg.norm(quat)
    if not (0.9 < norm < 1.1):
        raise IKInputValidationError(f"{name} is not normalized (norm={norm:.4f}): {quat}")


def validate_jacobian(jac: np.ndarray, expected_dofs: Optional[int] = None) -> None:
    """
    Validate Jacobian matrix.

    Args:
        jac: Jacobian matrix (3 x n or 6 x n)
        expected_dofs: Expected number of DOFs (optional)

    Raises:
        IKInputValidationError: If Jacobian is invalid
    """
    if jac is None:
        raise IKInputValidationError("Jacobian is None")

    if not isinstance(jac, np.ndarray):
        raise IKInputValidationError(f"Jacobian must be numpy array, got {type(jac)}")

    if jac.ndim != 2:
        raise IKInputValidationError(f"Jacobian must be 2D, got shape {jac.shape}")

    if jac.shape[0] not in [3, 6]:
        raise IKInputValidationError(f"Jacobian must have 3 or 6 rows, got {jac.shape[0]}")

    if expected_dofs is not None and jac.shape[1] != expected_dofs:
        raise IKInputValidationError(
            f"Jacobian has {jac.shape[1]} columns, expected {expected_dofs}"
        )

    if not np.all(np.isfinite(jac)):
        raise IKInputValidationError("Jacobian contains NaN or inf")


def validate_joint_positions(
    joint_pos: np.ndarray,
    joint_limits: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    name: str = "joint_positions"
) -> None:
    """
    Validate joint positions.

    Args:
        joint_pos: Joint position array
        joint_limits: Optional tuple of (lower_limits, upper_limits)
        name: Name for error messages

    Raises:
        IKInputValidationError: If joint positions are invalid
    """
    if joint_pos is None:
        raise IKInputValidationError(f"{name} is None")

    if not isinstance(joint_pos, np.ndarray):
        raise IKInputValidationError(f"{name} must be numpy array, got {type(joint_pos)}")

    if joint_pos.ndim != 1:
        raise IKInputValidationError(f"{name} must be 1D, got shape {joint_pos.shape}")

    if len(joint_pos) == 0:
        raise IKInputValidationError(f"{name} is empty")

    if not np.all(np.isfinite(joint_pos)):
        raise IKInputValidationError(f"{name} contains NaN or inf: {joint_pos}")

    # Validate against limits if provided
    if joint_limits is not None:
        lower_limits, upper_limits = joint_limits

        if lower_limits.shape != joint_pos.shape:
            raise IKInputValidationError(
                f"Lower limits shape {lower_limits.shape} != {name} shape {joint_pos.shape}"
            )

        if upper_limits.shape != joint_pos.shape:
            raise IKInputValidationError(
                f"Upper limits shape {upper_limits.shape} != {name} shape {joint_pos.shape}"
            )

        # Check if any joints are outside limits (with small tolerance)
        tolerance = 1e-4
        violations = []
        for i, (pos, lower, upper) in enumerate(zip(joint_pos, lower_limits, upper_limits)):
            if pos < lower - tolerance or pos > upper + tolerance:
                violations.append(f"joint[{i}]={pos:.4f} outside [{lower:.4f}, {upper:.4f}]")

        if violations:
            raise IKInputValidationError(f"{name} limit violations: {', '.join(violations)}")


def validate_joint_limits(limits: Tuple[np.ndarray, np.ndarray], n_joints: int) -> None:
    """
    Validate joint limits.

    Args:
        limits: Tuple of (lower_limits, upper_limits)
        n_joints: Expected number of joints

    Raises:
        IKInputValidationError: If limits are invalid
    """
    if limits is None:
        raise IKInputValidationError("Joint limits are None")

    if not isinstance(limits, tuple) or len(limits) != 2:
        raise IKInputValidationError("Joint limits must be tuple of (lower, upper)")

    lower_limits, upper_limits = limits

    if not isinstance(lower_limits, np.ndarray) or not isinstance(upper_limits, np.ndarray):
        raise IKInputValidationError("Joint limits must be numpy arrays")

    if lower_limits.shape != (n_joints,):
        raise IKInputValidationError(
            f"Lower limits shape {lower_limits.shape} != expected ({n_joints},)"
        )

    if upper_limits.shape != (n_joints,):
        raise IKInputValidationError(
            f"Upper limits shape {upper_limits.shape} != expected ({n_joints},)"
        )

    if not np.all(np.isfinite(lower_limits)):
        raise IKInputValidationError("Lower limits contain NaN or inf")

    if not np.all(np.isfinite(upper_limits)):
        raise IKInputValidationError("Upper limits contain NaN or inf")

    # Check that lower < upper
    if not np.all(lower_limits < upper_limits):
        violations = []
        for i, (lower, upper) in enumerate(zip(lower_limits, upper_limits)):
            if lower >= upper:
                violations.append(f"joint[{i}]: lower={lower:.4f} >= upper={upper:.4f}")
        raise IKInputValidationError(f"Invalid joint limits: {', '.join(violations)}")


def validate_config(config) -> None:
    """
    Validate IK configuration.

    Args:
        config: IKConfig instance

    Raises:
        IKConfigurationError: If configuration is invalid
    """
    if config.control_mode not in ["joint", "ee_pose", "auto"]:
        raise IKConfigurationError(
            f"Invalid control_mode '{config.control_mode}', must be 'joint', 'ee_pose', or 'auto'"
        )

    if config.ik_rate_hz <= 0:
        raise IKConfigurationError(f"ik_rate_hz must be positive, got {config.ik_rate_hz}")

    if config.ik_damping <= 0:
        raise IKConfigurationError(f"ik_damping must be positive, got {config.ik_damping}")

    if config.ik_pos_gain <= 0:
        raise IKConfigurationError(f"ik_pos_gain must be positive, got {config.ik_pos_gain}")

    if config.ik_ori_gain <= 0:
        raise IKConfigurationError(f"ik_ori_gain must be positive, got {config.ik_ori_gain}")

    if config.ik_max_dq <= 0:
        raise IKConfigurationError(f"ik_max_dq must be positive, got {config.ik_max_dq}")

    if config.ik_max_joint_vel <= 0:
        raise IKConfigurationError(f"ik_max_joint_vel must be positive, got {config.ik_max_joint_vel}")

    if config.ik_pos_tol <= 0:
        raise IKConfigurationError(f"ik_pos_tol must be positive, got {config.ik_pos_tol}")

    if config.ik_ori_tol <= 0:
        raise IKConfigurationError(f"ik_ori_tol must be positive, got {config.ik_ori_tol}")

    if config.ik_timeout_sec <= 0:
        raise IKConfigurationError(f"ik_timeout_sec must be positive, got {config.ik_timeout_sec}")

    if config.ik_perf_log_interval_sec < 0:
        raise IKConfigurationError(
            f"ik_perf_log_interval_sec must be non-negative, got {config.ik_perf_log_interval_sec}"
        )

    if config.ik_perf_warn_threshold_ms <= 0:
        raise IKConfigurationError(
            f"ik_perf_warn_threshold_ms must be positive, got {config.ik_perf_warn_threshold_ms}"
        )

    if not config.left_arm_joints or len(config.left_arm_joints) == 0:
        raise IKConfigurationError("left_arm_joints is empty")

    if not config.right_arm_joints or len(config.right_arm_joints) == 0:
        raise IKConfigurationError("right_arm_joints is empty")
