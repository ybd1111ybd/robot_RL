#!/usr/bin/env python3
"""
Test error handling and edge cases for IK solver.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ik_solver.ik_solver_core import IKSolverCore
from ik_solver.ik_exceptions import (
    IKInputValidationError,
    IKSingularityError,
    IKJointLimitError,
)


class TestErrorHandling:
    """Test error handling in IK solver."""

    def setup_method(self):
        """Set up test fixtures."""
        self.solver = IKSolverCore(
            pos_gain=1.0,
            ori_gain=1.0,
            damping=0.05,
            enable_orientation=True,
        )

    def test_invalid_position_nan(self):
        """Test handling of NaN in position."""
        target_pos = np.array([np.nan, 0.5, 0.8])
        current_pos = np.array([0.5, 0.3, 0.8])

        with pytest.raises(IKInputValidationError):
            self.solver.compute_position_error(target_pos, current_pos)

    def test_invalid_position_inf(self):
        """Test handling of inf in position."""
        target_pos = np.array([0.5, np.inf, 0.8])
        current_pos = np.array([0.5, 0.3, 0.8])

        with pytest.raises(IKInputValidationError):
            self.solver.compute_position_error(target_pos, current_pos)

    def test_invalid_quaternion_not_normalized(self):
        """Test handling of non-normalized quaternion."""
        target_quat = np.array([2.0, 0.0, 0.0, 0.0])  # Not normalized
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])

        with pytest.raises(IKInputValidationError):
            self.solver.compute_orientation_error(target_quat, current_quat)

    def test_invalid_quaternion_nan(self):
        """Test handling of NaN in quaternion."""
        target_quat = np.array([np.nan, 0.0, 0.0, 0.0])
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])

        with pytest.raises(IKInputValidationError):
            self.solver.compute_orientation_error(target_quat, current_quat)

    def test_singular_jacobian(self):
        """Test handling of singular Jacobian."""
        # Create a singular Jacobian (all zeros)
        jacobian = np.zeros((6, 7))
        task_error = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        joint_cmd, status = self.solver.solve_dls(
            jacobian, task_error, current_joint_pos, joint_limits, dt=0.01
        )

        # With damping, zero Jacobian still produces a solution (damped)
        # Status should be "active" since the solver can still compute
        assert status["status"] in ["active", "active_clipped", "singular"]
        assert "error_norm" in status

    def test_joint_limits_violation(self):
        """Test handling of joint limit violations."""
        # Create a simple Jacobian
        jacobian = np.eye(3, 7)
        task_error = np.array([1.0, 0.0, 0.0])  # Large error
        current_joint_pos = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        joint_limits = (
            np.full(7, -np.pi),
            np.full(7, np.pi),
        )

        joint_cmd, status = self.solver.solve_dls(
            jacobian, task_error, current_joint_pos, joint_limits, dt=0.01
        )

        # Joint command should be within limits
        assert np.all(joint_cmd >= joint_limits[0])
        assert np.all(joint_cmd <= joint_limits[1])

    def test_invalid_jacobian_shape(self):
        """Test handling of invalid Jacobian shape."""
        jacobian = np.eye(3, 5)  # Wrong number of joints
        task_error = np.array([0.1, 0.0, 0.0])
        current_joint_pos = np.zeros(7)  # 7 joints expected
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        # The implementation returns early with validation_error status
        joint_cmd, status = self.solver.solve_dls(
            jacobian, task_error, current_joint_pos, joint_limits, dt=0.01
        )

        assert status["status"] == "validation_error"
        assert "error_message" in status
        assert np.allclose(joint_cmd, current_joint_pos)

    def test_compute_ik_step_with_invalid_input(self):
        """Test compute_ik_step with invalid input."""
        target_pos = np.array([np.nan, 0.5, 0.8])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        current_pos = np.array([0.5, 0.3, 0.8])
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        jacobian = np.eye(6, 7)
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        # Should not raise, but return error status
        joint_cmd, status = self.solver.compute_ik_step(
            target_pos,
            target_quat,
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            joint_limits,
            dt=0.01,
        )

        # Should return current position
        assert np.allclose(joint_cmd, current_joint_pos)
        assert status["status"] == "validation_error"

    def test_consecutive_failures_tracking(self):
        """Test consecutive failure tracking."""
        # Create a Jacobian with very high condition number to trigger near_singular
        # Use a nearly rank-deficient matrix
        jacobian = np.eye(6, 7) * 1e-10  # Very small values
        task_error = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        # Trigger multiple failures
        for i in range(3):
            joint_cmd, status = self.solver.solve_dls(
                jacobian, task_error, current_joint_pos, joint_limits, dt=0.01
            )
            # Consecutive failures only increment on singularity/numerical errors
            if status["status"] in ["near_singular", "singular", "numerical_error"]:
                assert status["consecutive_failures"] == i + 1
            else:
                # If solver handles it gracefully, consecutive_failures stays 0
                assert status["consecutive_failures"] == 0

        # Reset counter
        self.solver.reset_failure_counter()
        assert self.solver.get_failure_count() == 0

    def test_large_error_clamping(self):
        """Test that large errors are properly clamped."""
        jacobian = np.eye(3, 7)
        task_error = np.array([100.0, 0.0, 0.0])  # Very large error
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        joint_cmd, status = self.solver.solve_dls(
            jacobian, task_error, current_joint_pos, joint_limits, dt=0.01
        )

        # Joint velocity should be clamped
        dq = joint_cmd - current_joint_pos
        max_dq = self.solver.max_joint_vel * 0.01
        assert np.all(np.abs(dq) <= max_dq + 1e-6)  # Small tolerance for numerical errors

    def test_position_only_ik(self):
        """Test position-only IK (no orientation)."""
        solver = IKSolverCore(enable_orientation=False)

        target_pos = np.array([0.6, 0.3, 0.8])
        current_pos = np.array([0.5, 0.3, 0.8])
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        jacobian = np.eye(6, 7)
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        joint_cmd, status = solver.compute_ik_step(
            target_pos,
            None,  # No target quaternion
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            joint_limits,
            dt=0.01,
        )

        # Should succeed
        assert status["status"] in ["active", "converged", "active_clipped"]
        assert "pos_error" in status
        assert "ori_error" not in status  # No orientation error

    def test_limit_hits_include_joint_names(self):
        """Test that limit hits carry local indices and joint names."""
        solver = IKSolverCore(enable_orientation=False)

        target_pos = np.array([1.0, 0.0, 0.0])
        current_pos = np.array([0.0, 0.0, 0.0])
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        jacobian = np.zeros((6, 7))
        jacobian[0, 3] = 1.0
        current_joint_pos = np.array([0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.0])
        lower_limits = np.full(7, -np.pi)
        upper_limits = np.full(7, np.pi)
        upper_limits[3] = 0.17

        _, status = solver.compute_ik_step(
            target_pos,
            None,
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            (lower_limits, upper_limits),
            dt=0.1,
            debug_label="left",
            joint_names=[f"joint{i}" for i in range(7)],
        )

        assert status["limit_hits"]
        assert status["limit_hits"][0]["index"] == 3
        assert status["limit_hits"][0]["name"] == "joint3"
        assert status["limit_hits"][0]["side"] == "upper"

    def test_zero_dt(self):
        """Test handling of zero time step."""
        jacobian = np.eye(3, 7)
        task_error = np.array([0.1, 0.0, 0.0])
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        # Should handle zero dt gracefully
        joint_cmd, status = self.solver.solve_dls(
            jacobian, task_error, current_joint_pos, joint_limits, dt=0.0
        )

        # With zero dt, velocity limit becomes zero, so no movement
        assert np.allclose(joint_cmd, current_joint_pos)


class TestEdgeCases:
    """Test edge cases in IK solver."""

    def test_target_equals_current(self):
        """Test when target equals current pose."""
        solver = IKSolverCore()

        target_pos = np.array([0.5, 0.3, 0.8])
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        current_pos = target_pos.copy()
        current_quat = target_quat.copy()
        jacobian = np.eye(6, 7)
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        joint_cmd, status = solver.compute_ik_step(
            target_pos,
            target_quat,
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            joint_limits,
            dt=0.01,
        )

        # Should converge immediately
        assert status["status"] == "converged"
        assert status["pos_error"] < solver.pos_tol

    def test_very_small_error(self):
        """Test with very small error (near convergence)."""
        solver = IKSolverCore(pos_tol=0.01)

        target_pos = np.array([0.5, 0.3, 0.8])
        current_pos = np.array([0.5, 0.3, 0.8001])  # Very small error
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        jacobian = np.eye(6, 7)
        current_joint_pos = np.zeros(7)
        joint_limits = (np.full(7, -np.pi), np.full(7, np.pi))

        joint_cmd, status = solver.compute_ik_step(
            target_pos,
            None,
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            joint_limits,
            dt=0.01,
        )

        # Should converge
        assert status["status"] == "converged"

    def test_at_joint_limits(self):
        """Test when already at joint limits."""
        solver = IKSolverCore()

        target_pos = np.array([0.6, 0.3, 0.8])
        current_pos = np.array([0.5, 0.3, 0.8])
        current_quat = np.array([1.0, 0.0, 0.0, 0.0])
        jacobian = np.eye(3, 7)
        current_joint_pos = np.array([3.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # At upper limit
        joint_limits = (
            np.full(7, -np.pi),
            np.full(7, np.pi),
        )

        joint_cmd, status = solver.compute_ik_step(
            target_pos,
            None,
            current_pos,
            current_quat,
            jacobian,
            current_joint_pos,
            joint_limits,
            dt=0.01,
        )

        # Should respect limits
        assert np.all(joint_cmd >= joint_limits[0])
        assert np.all(joint_cmd <= joint_limits[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
