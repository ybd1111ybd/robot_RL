"""
IK Solver Core Module

Implements Jacobian-based DLS (Damped Least Squares) inverse kinematics.
"""

import numpy as np
from typing import Tuple, Optional
from .ik_exceptions import (
    IKSingularityError,
    IKJointLimitError,
    IKInputValidationError,
)
from .ik_validators import (
    validate_position,
    validate_quaternion,
    validate_jacobian,
    validate_joint_positions,
    validate_joint_limits,
)

# Try to import Isaac Sim rotation utils, fallback to numpy if not available
try:
    import omni.isaac.core.utils.rotations as rotation_utils
    HAS_ISAAC_SIM = True
except ImportError:
    HAS_ISAAC_SIM = False
    # Fallback rotation utilities for testing without Isaac Sim
    class rotation_utils:
        @staticmethod
        def quat_conjugate(q):
            """Quaternion conjugate [w, x, y, z]."""
            return np.array([q[0], -q[1], -q[2], -q[3]])

        @staticmethod
        def quat_mul(q1, q2):
            """Quaternion multiplication [w, x, y, z]."""
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
            ])

        @staticmethod
        def quat_to_axis_angle(q):
            """Convert quaternion to axis-angle [w, x, y, z]."""
            w, x, y, z = q
            angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))
            s = np.sqrt(1 - w*w)
            if s < 1e-8:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([x, y, z]) / s
            return axis, angle


class IKSolverCore:
    """Core IK solver using Jacobian DLS method."""

    def __init__(
        self,
        damping: float = 0.03,
        pos_gain: float = 1.2,
        ori_gain: float = 0.6,
        max_dq: float = 0.08,
        max_joint_vel: float = 1.5,
        pos_tol: float = 0.01,
        ori_tol: float = 0.05,
        enable_orientation: bool = False,
        singularity_threshold: float = 1e-4,
        max_condition_number: float = 1e6,
    ):
        """
        Initialize IK solver.

        Args:
            damping: DLS damping factor (lambda)
            pos_gain: Position error gain
            ori_gain: Orientation error gain
            max_dq: Maximum joint angle change per step
            max_joint_vel: Maximum joint velocity (rad/s)
            pos_tol: Position convergence tolerance (m)
            ori_tol: Orientation convergence tolerance (rad)
            enable_orientation: Whether to include orientation in IK
            singularity_threshold: Threshold for detecting singularities
            max_condition_number: Maximum allowed condition number
        """
        self.damping = damping
        self.pos_gain = pos_gain
        self.ori_gain = ori_gain
        self.max_dq = max_dq
        self.max_joint_vel = max_joint_vel
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.enable_orientation = enable_orientation
        self.singularity_threshold = singularity_threshold
        self.max_condition_number = max_condition_number

        # Error tracking
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10

    def compute_position_error(
        self, target_pos: np.ndarray, current_pos: np.ndarray
    ) -> np.ndarray:
        """
        Compute position error.

        Args:
            target_pos: Target position [x, y, z]
            current_pos: Current position [x, y, z]

        Returns:
            Position error vector

        Raises:
            IKInputValidationError: If inputs are invalid
        """
        try:
            validate_position(target_pos, "target_pos")
            validate_position(current_pos, "current_pos")
        except IKInputValidationError as e:
            print(f"[IK Solver] Position validation error: {e}")
            raise

        return target_pos - current_pos

    def compute_orientation_error(
        self, target_quat: np.ndarray, current_quat: np.ndarray
    ) -> np.ndarray:
        """
        Compute orientation error in axis-angle form.

        Args:
            target_quat: Target quaternion [w, x, y, z]
            current_quat: Current quaternion [w, x, y, z]

        Returns:
            Orientation error vector (axis-angle)

        Raises:
            IKInputValidationError: If inputs are invalid
        """
        try:
            validate_quaternion(target_quat, "target_quat")
            validate_quaternion(current_quat, "current_quat")
        except IKInputValidationError as e:
            print(f"[IK Solver] Quaternion validation error: {e}")
            raise

        # Compute relative rotation: q_error = q_target * q_current^(-1)
        current_quat_inv = rotation_utils.quat_conjugate(current_quat)
        q_error = rotation_utils.quat_mul(target_quat, current_quat_inv)

        # Convert to axis-angle
        axis, angle = rotation_utils.quat_to_axis_angle(q_error)

        # Return scaled axis (axis * angle)
        return axis * angle

    def solve_dls(
        self,
        jacobian: np.ndarray,
        task_error: np.ndarray,
        current_joint_pos: np.ndarray,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]],
        dt: float,
        debug_label: Optional[str] = None,
        joint_names: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Solve IK using Damped Least Squares method.

        Args:
            jacobian: Jacobian matrix (6 x n_joints or 3 x n_joints)
            task_error: Task space error vector
            current_joint_pos: Current joint positions
            joint_limits: Tuple of (lower_limits, upper_limits)
            dt: Time step
            debug_label: Optional label for diagnostics
            joint_names: Optional joint names aligned with local indices

        Returns:
            Tuple of (joint_command, status_dict)

        Raises:
            IKInputValidationError: If inputs are invalid
            IKSingularityError: If Jacobian is singular
            IKJointLimitError: If joint limits are violated
        """
        joint_pos = np.asarray(current_joint_pos, dtype=float).reshape(-1)
        if joint_limits is not None:
            try:
                lower_limits = np.asarray(joint_limits[0], dtype=float).reshape(-1)
                upper_limits = np.asarray(joint_limits[1], dtype=float).reshape(-1)
                if lower_limits.shape == joint_pos.shape and upper_limits.shape == joint_pos.shape:
                    joint_pos = np.minimum(np.maximum(joint_pos, lower_limits), upper_limits)
            except Exception:
                pass

        # Validate inputs
        try:
            validate_jacobian(jacobian, expected_dofs=len(current_joint_pos))
            validate_joint_positions(joint_pos, joint_limits, "current_joint_pos")

            if joint_limits is not None:
                validate_joint_limits(joint_limits, len(current_joint_pos))

            if not np.all(np.isfinite(task_error)):
                raise IKInputValidationError(f"task_error contains NaN or inf: {task_error}")

            if dt <= 0 or not np.isfinite(dt):
                raise IKInputValidationError(f"Invalid dt: {dt}")

        except IKInputValidationError as e:
            print(f"[IK Solver] Input validation failed: {e}")
            return joint_pos, {
                "status": "validation_error",
                "error_norm": np.linalg.norm(task_error) if np.all(np.isfinite(task_error)) else float('inf'),
                "error_message": str(e),
                "limit_hits": [],
            }

        current_joint_pos = joint_pos
        n_joints = jacobian.shape[1]

        # DLS solution: dq = J^T (J J^T + lambda^2 I)^(-1) e
        JJT = jacobian @ jacobian.T
        damping_matrix = (self.damping ** 2) * np.eye(JJT.shape[0])
        JJT_damped = JJT + damping_matrix

        # Check condition number for singularity detection
        try:
            condition_number = np.linalg.cond(JJT_damped)
            if condition_number > self.max_condition_number:
                print(f"[IK Solver] Warning: High condition number {condition_number:.2e}, near singularity")
                self._consecutive_failures += 1

                if self._consecutive_failures > self._max_consecutive_failures:
                    raise IKSingularityError(
                        f"Persistent singularity detected (condition number: {condition_number:.2e})"
                    )

                return current_joint_pos, {
                    "status": "near_singular",
                    "error_norm": np.linalg.norm(task_error),
                    "condition_number": condition_number,
                    "limit_hits": [],
                }
        except np.linalg.LinAlgError:
            self._consecutive_failures += 1
            return current_joint_pos, {
                "status": "singular",
                "error_norm": np.linalg.norm(task_error),
                "limit_hits": [],
            }

        # Invert matrix
        try:
            JJT_inv = np.linalg.inv(JJT_damped)
        except np.linalg.LinAlgError as e:
            print(f"[IK Solver] Matrix inversion failed: {e}")
            self._consecutive_failures += 1
            return current_joint_pos, {
                "status": "singular",
                "error_norm": np.linalg.norm(task_error),
                "error_message": str(e),
                "limit_hits": [],
            }

        dq = jacobian.T @ JJT_inv @ task_error

        # Check for NaN/inf in solution
        if not np.all(np.isfinite(dq)):
            print(f"[IK Solver] Solution contains NaN or inf: {dq}")
            self._consecutive_failures += 1
            return current_joint_pos, {
                "status": "numerical_error",
                "error_norm": np.linalg.norm(task_error),
                "limit_hits": [],
            }

        # Limit joint velocity
        max_dq_vel = self.max_joint_vel * dt
        dq_clipped_vel = np.clip(dq, -max_dq_vel, max_dq_vel)

        # Limit per-step change
        dq_clipped = np.clip(dq_clipped_vel, -self.max_dq, self.max_dq)

        # Track if clipping occurred
        was_clipped = not np.allclose(dq, dq_clipped)

        # Integrate
        joint_cmd = current_joint_pos + dq_clipped

        # Apply joint limits
        joint_cmd_limited = joint_cmd.copy()
        limit_hits = []
        if joint_limits is not None:
            lower_limits, upper_limits = joint_limits
            joint_cmd_limited = np.clip(joint_cmd, lower_limits, upper_limits)

            # Check if any joints hit limits
            at_lower = np.isclose(joint_cmd_limited, lower_limits, atol=1e-4)
            at_upper = np.isclose(joint_cmd_limited, upper_limits, atol=1e-4)

            if np.any(at_lower) or np.any(at_upper):
                for i in range(n_joints):
                    if at_lower[i] or at_upper[i]:
                        joint_name = None
                        if joint_names is not None and i < len(joint_names):
                            joint_name = joint_names[i]
                        limit_hits.append({
                            "index": i,
                            "name": joint_name,
                            "side": "lower" if at_lower[i] else "upper",
                            "label": debug_label,
                        })

        # Compute status
        error_norm = np.linalg.norm(task_error)
        dq_norm = np.linalg.norm(dq_clipped)

        # Reset failure counter on success
        if error_norm < self.pos_tol:
            self._consecutive_failures = 0
            status_str = "converged"
        elif was_clipped:
            status_str = "active_clipped"
        else:
            status_str = "active"

        status = {
            "status": status_str,
            "error_norm": error_norm,
            "dq_norm": dq_norm,
            "was_clipped": was_clipped,
            "consecutive_failures": self._consecutive_failures,
            "limit_hits": limit_hits,
        }

        return joint_cmd_limited, status

    def compute_ik_step(
        self,
        target_pos: np.ndarray,
        target_quat: Optional[np.ndarray],
        current_pos: np.ndarray,
        current_quat: np.ndarray,
        jacobian: np.ndarray,
        current_joint_pos: np.ndarray,
        joint_limits: Optional[Tuple[np.ndarray, np.ndarray]],
        dt: float,
        debug_label: Optional[str] = None,
        joint_names: Optional[list[str]] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Compute one IK step.

        Args:
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (optional)
            current_pos: Current position [x, y, z]
            current_quat: Current quaternion [w, x, y, z]
            jacobian: Full Jacobian matrix (6 x n_joints)
            current_joint_pos: Current joint positions
            joint_limits: Tuple of (lower_limits, upper_limits)
            dt: Time step

        Returns:
            Tuple of (joint_command, status_dict)

        Raises:
            IKInputValidationError: If inputs are invalid
        """
        try:
            # Compute position error
            pos_error = self.compute_position_error(target_pos, current_pos)
            pos_error_weighted = self.pos_gain * pos_error

            if self.enable_orientation and target_quat is not None:
                # Compute orientation error
                ori_error = self.compute_orientation_error(target_quat, current_quat)
                ori_error_weighted = self.ori_gain * ori_error

                # Combined task error
                task_error = np.concatenate([pos_error_weighted, ori_error_weighted])

                # Use full 6D Jacobian
                jac = jacobian
            else:
                # Position-only IK
                task_error = pos_error_weighted

                # Use only position part of Jacobian (first 3 rows)
                jac = jacobian[:3, :]

            # Solve DLS
            joint_cmd, status = self.solve_dls(
                jac,
                task_error,
                current_joint_pos,
                joint_limits,
                dt,
                debug_label=debug_label,
                joint_names=joint_names,
            )

            # Add error information to status
            status["pos_error"] = np.linalg.norm(pos_error)
            if self.enable_orientation and target_quat is not None:
                status["ori_error"] = np.linalg.norm(ori_error)

            return joint_cmd, status

        except IKInputValidationError as e:
            # Return current position on validation error
            print(f"[IK Solver] Validation error in compute_ik_step: {e}")
            return current_joint_pos, {
                "status": "validation_error",
                "error_norm": float('inf'),
                "error_message": str(e),
                "limit_hits": [],
            }
        except Exception as e:
            # Catch any unexpected errors
            print(f"[IK Solver] Unexpected error in compute_ik_step: {e}")
            return current_joint_pos, {
                "status": "error",
                "error_norm": float('inf'),
                "error_message": str(e),
                "limit_hits": [],
            }

    def reset_failure_counter(self):
        """Reset the consecutive failure counter."""
        self._consecutive_failures = 0

    def get_failure_count(self) -> int:
        """Get the current consecutive failure count."""
        return self._consecutive_failures
