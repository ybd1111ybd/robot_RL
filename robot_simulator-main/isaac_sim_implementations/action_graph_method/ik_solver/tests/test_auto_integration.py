#!/usr/bin/env python3
"""
Integration tests for auto mode IK functionality.

Tests the complete pipeline from target pose to joint commands.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ik_solver.ik_config import IKConfig
from ik_solver.ik_solver_core import IKSolverCore
from ik_solver.ik_bridge import ArmIKController


def test_ik_solver_core_basic():
    """Test IKSolverCore basic functionality."""
    # Create solver with default parameters
    solver = IKSolverCore(
        damping=0.03,
        pos_gain=1.2,
        max_dq=0.08,
        pos_tol=0.01,
        enable_orientation=False
    )

    # Test position error computation
    target_pos = np.array([0.5, 0.3, 0.8])
    current_pos = np.array([0.4, 0.3, 0.8])
    pos_error = solver.compute_position_error(target_pos, current_pos)

    assert np.allclose(pos_error, [0.1, 0.0, 0.0])
    print("✓ Position error computation works")


def test_ik_solver_dls():
    """Test DLS solver with simple Jacobian."""
    solver = IKSolverCore(damping=0.03, pos_tol=0.01)

    # Simple 3-DOF example
    jacobian = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 1.0, 0.3],
        [0.0, 0.0, 1.0]
    ])

    task_error = np.array([0.1, 0.05, 0.02])
    current_joints = np.array([0.0, 0.0, 0.0])
    joint_limits = (np.array([-3.14, -3.14, -3.14]), np.array([3.14, 3.14, 3.14]))
    dt = 0.01

    joint_cmd, status = solver.solve_dls(
        jacobian, task_error, current_joints, joint_limits, dt
    )

    # Verify output structure
    assert joint_cmd.shape == (3,)
    assert 'status' in status
    assert 'error_norm' in status
    assert 'dq_norm' in status
    print(f"✓ DLS solver works: status={status['status']}, error={status['error_norm']:.4f}")


def test_ik_solver_compute_step():
    """Test compute_ik_step with position-only IK."""
    solver = IKSolverCore(
        damping=0.03,
        pos_gain=1.0,
        enable_orientation=False
    )

    # Target and current poses
    target_pos = np.array([0.5, 0.3, 0.8])
    current_pos = np.array([0.45, 0.3, 0.8])
    current_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    # Simple Jacobian (3x7 for position-only)
    jacobian = np.random.randn(6, 7) * 0.1  # Full 6D jacobian
    current_joints = np.zeros(7)
    joint_limits = (np.full(7, -3.14), np.full(7, 3.14))
    dt = 0.01

    joint_cmd, status = solver.compute_ik_step(
        target_pos, None, current_pos, current_quat,
        jacobian, current_joints, joint_limits, dt
    )

    # Verify output
    assert joint_cmd.shape == (7,)
    assert 'pos_error' in status
    assert status['pos_error'] > 0  # Should have some error
    print(f"✓ IK step computation works: pos_error={status['pos_error']:.4f}")


def test_arm_controller_target_management():
    """Test ArmIKController target management."""
    config = IKConfig(control_mode='auto')
    solver = IKSolverCore()

    controller = ArmIKController(
        arm_name="left",
        ee_link="left_ee_link",
        joint_names=["joint1", "joint2", "joint3"],
        ik_solver=solver,
        config=config
    )

    # Initially no target
    assert not controller.has_active_target()

    # Set a target
    pos = np.array([0.5, 0.3, 0.8])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    controller.set_target_pose(pos, quat, "world")

    # Should have active target now
    assert controller.has_active_target()

    # Get target back
    ret_pos, ret_quat, ret_frame, timestamp = controller.get_target_pose()
    assert np.allclose(ret_pos, pos)
    assert np.allclose(ret_quat, quat)
    assert ret_frame == "world"
    assert timestamp > 0

    print("✓ Arm controller target management works")


def test_auto_mode_config():
    """Test that auto mode configuration is correct."""
    config = IKConfig(control_mode='auto')

    assert config.control_mode == 'auto'
    assert config.is_ik_enabled()

    # In auto mode, IK should only run when targets are active
    print("✓ Auto mode configuration is correct")


if __name__ == '__main__':
    print("Running auto mode integration tests...\n")

    test_ik_solver_core_basic()
    test_ik_solver_dls()
    test_ik_solver_compute_step()
    test_arm_controller_target_management()
    test_auto_mode_config()

    print("\n✅ All integration tests passed!")
