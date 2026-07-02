#!/usr/bin/env python3
"""
IK Solver Module Test

Basic tests to verify module structure and imports.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from ik_solver import IKConfig, EndEffectorIKBridge
        print("✓ Main module imports successful")
    except ImportError as e:
        print(f"✗ Main module import failed: {e}")
        return False

    try:
        from ik_solver.ik_config import IKConfig
        print("✓ IKConfig import successful")
    except ImportError as e:
        print(f"✗ IKConfig import failed: {e}")
        return False

    try:
        from ik_solver.ik_solver_core import IKSolverCore
        print("✓ IKSolverCore import successful")
    except ImportError as e:
        print(f"✗ IKSolverCore import failed: {e}")
        return False

    try:
        from ik_solver.ik_bridge import EndEffectorIKBridge, ArmIKController
        print("✓ IK Bridge imports successful")
    except ImportError as e:
        print(f"✗ IK Bridge import failed: {e}")
        return False

    try:
        from ik_solver.ik_utils import get_end_effector_pose_from_articulation
        print("✓ IK Utils import successful")
    except ImportError as e:
        print(f"✗ IK Utils import failed: {e}")
        return False

    return True


def test_config():
    """Test IKConfig creation and properties."""
    print("\nTesting IKConfig...")

    from ik_solver import IKConfig

    # Test default config
    config = IKConfig()
    assert config.control_mode == "joint"
    assert config.ik_rate_hz == 50.0
    assert config.ik_dt == 0.02
    assert len(config.left_arm_joints) == 7
    assert len(config.right_arm_joints) == 7
    print("✓ Default config created successfully")

    # Test custom config
    config = IKConfig(
        control_mode="ee_pose",
        ik_rate_hz=100.0,
        ik_damping=0.05,
    )
    assert config.control_mode == "ee_pose"
    assert config.ik_rate_hz == 100.0
    assert config.ik_dt == 0.01
    assert config.ik_damping == 0.05
    assert config.is_ik_enabled() == True
    print("✓ Custom config created successfully")

    # Test joint mode
    config = IKConfig(control_mode="joint")
    assert config.is_ik_enabled() == False
    print("✓ Joint mode config correct")

    return True


def test_ik_solver_core():
    """Test IKSolverCore basic functionality."""
    print("\nTesting IKSolverCore...")

    import numpy as np
    from ik_solver.ik_solver_core import IKSolverCore

    solver = IKSolverCore(
        damping=0.03,
        pos_gain=1.2,
        max_dq=0.08,
    )
    print("✓ IKSolverCore created successfully")

    # Test position error computation
    target_pos = np.array([0.5, 0.3, 0.8])
    current_pos = np.array([0.4, 0.3, 0.8])
    error = solver.compute_position_error(target_pos, current_pos)
    assert np.allclose(error, [0.1, 0.0, 0.0])
    print("✓ Position error computation correct")

    # Test DLS with simple case
    jacobian = np.eye(3, 7)  # Simple 3x7 Jacobian
    task_error = np.array([0.1, 0.0, 0.0])
    current_joint_pos = np.zeros(7)
    joint_limits = (np.full(7, -3.14), np.full(7, 3.14))

    joint_cmd, status = solver.solve_dls(
        jacobian=jacobian,
        task_error=task_error,
        current_joint_pos=current_joint_pos,
        joint_limits=joint_limits,
        dt=0.02,
    )

    assert joint_cmd.shape == (7,)
    assert "status" in status
    assert "error_norm" in status
    print("✓ DLS solver executed successfully")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("IK Solver Module Tests")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("IK Solver Core", test_ik_solver_core),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {name} tests passed")
            else:
                failed += 1
                print(f"\n✗ {name} tests failed")
        except Exception as e:
            failed += 1
            print(f"\n✗ {name} tests failed with exception: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
