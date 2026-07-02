#!/usr/bin/env python3
"""
Test script for auto mode IK functionality.

This script verifies that:
1. Arm target activation and timeout behavior work correctly
2. `auto` mode stays idle without active targets
3. `ee_pose` mode continues publishing current EE poses while idle
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ik_solver.ik_config import IKConfig
from ik_solver.ik_bridge import EndEffectorIKBridge, ArmIKController
from ik_solver.ik_solver_core import IKSolverCore


def rotate_x_axis(quaternion: np.ndarray) -> np.ndarray:
    """Return the world direction of the local +X axis for [w, x, y, z]."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        dtype=float,
    )


class DummyPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class DummyClock:
    def now(self):
        return self

    def to_msg(self):
        return {"stamp": "dummy"}


class DummyNode:
    def __init__(self):
        self._clock = DummyClock()

    def get_clock(self):
        return self._clock


class DummyHeader:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class DummyPosition:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class DummyOrientation:
    def __init__(self):
        self.w = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class DummyPose:
    def __init__(self):
        self.position = DummyPosition()
        self.orientation = DummyOrientation()


class DummyPoseStamped:
    def __init__(self):
        self.header = DummyHeader()
        self.pose = DummyPose()


class DummyJointState:
    def __init__(self):
        self.header = DummyHeader()
        self.name = []
        self.position = []


class DummyString:
    def __init__(self):
        self.data = ""


class DummyArticulationView:
    def __init__(self):
        self.dof_names = ["left_joint", "right_joint"]
        self.body_names = [
            "base_link",
            "left_arm_link9",
            "left_gripper_narrow3_link",
            "left_gripper_wide3_link",
            "right_arm_link9",
            "right_gripper_narrow3_link",
            "right_gripper_wide3_link",
        ]
        self.prim_paths = ["/robot/base_link"]

    def get_jacobians(self):
        jac = np.zeros((1, 6, 6, 2), dtype=float)
        jac[0, 0, 3, 0] = 1.0
        jac[0, 1, 0, 0] = 0.2
        jac[0, 2, 0, 0] = 0.4
        jac[0, 3, 3, 1] = 1.0
        jac[0, 4, 0, 1] = 0.3
        jac[0, 5, 0, 1] = 0.5
        return jac

    def get_joint_positions(self):
        return np.zeros((1, 2), dtype=float)

    def get_dof_limits(self):
        lower = np.array([-np.pi, -np.pi], dtype=float)
        upper = np.array([np.pi, np.pi], dtype=float)
        return lower, upper

    def get_world_poses(self, indices=None):
        if indices == [0]:
            pos = np.array([[0.45, 0.25, 0.85]], dtype=float)
        else:
            pos = np.array([[0.45, -0.25, 0.85]], dtype=float)
        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
        return pos, quat


class DummyRigidPrim:
    def __init__(self, prim_paths_expr, name="rigid_prim_view"):
        self.prim_paths_expr = prim_paths_expr
        self.name = name

    def get_world_poses(self):
        pose_map = {
            "/robot/left_arm_link9": np.array([[0.12, 0.22, 0.31]], dtype=float),
            "/robot/left_gripper_narrow3_link": np.array([[0.10, 0.20, 0.32]], dtype=float),
            "/robot/left_gripper_wide3_link": np.array([[0.14, 0.24, 0.32]], dtype=float),
            "/robot/right_arm_link9": np.array([[0.42, 0.52, 0.61]], dtype=float),
            "/robot/right_gripper_narrow3_link": np.array([[0.40, 0.50, 0.62]], dtype=float),
            "/robot/right_gripper_wide3_link": np.array([[0.44, 0.54, 0.62]], dtype=float),
        }
        pos = pose_map.get(self.prim_paths_expr, np.array([[0.0, 0.0, 0.0]], dtype=float))
        quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float)
        return pos, quat


def create_test_bridge(control_mode: str) -> EndEffectorIKBridge:
    config = IKConfig(
        control_mode=control_mode,
        ik_timeout_sec=0.1,
        left_arm_joints=["left_joint"],
        right_arm_joints=["right_joint"],
    )
    bridge = EndEffectorIKBridge(config, DummyArticulationView())
    bridge._running = True
    bridge._node = DummyNode()
    bridge._PoseStamped = DummyPoseStamped
    bridge._JointState = DummyJointState
    bridge._String = DummyString
    bridge._left_cmd_pub = DummyPublisher()
    bridge._right_cmd_pub = DummyPublisher()
    bridge._left_pose_pub = DummyPublisher()
    bridge._right_pose_pub = DummyPublisher()
    bridge._left_status_pub = DummyPublisher()
    bridge._right_status_pub = DummyPublisher()
    bridge._rigid_prim_cls = DummyRigidPrim
    bridge._rigid_prim_import_attempted = True
    return bridge


def test_arm_controller_target_management():
    """Test that ArmIKController properly manages active targets."""
    print("\n=== Test: Arm Controller Target Management ===")

    # Create config and solver
    config = IKConfig(control_mode="auto")
    ik_solver = IKSolverCore()

    controller = ArmIKController(
        arm_name="left",
        joint_names=["joint1", "joint2", "joint3"],
        ee_link="left_ee",
        ik_solver=ik_solver,
        config=config,
    )

    # Initially no target
    assert not controller.has_active_target(), "Should have no active target initially"

    # Set a target
    target_pos = np.array([0.5, 0.3, 0.8])
    target_quat = np.array([1.0, 0.0, 0.0, 0.0])
    controller.set_target_pose(target_pos, target_quat, "world")

    assert controller.has_active_target(), "Should have active target after setting"

    # Get target
    pos, quat, frame, timestamp = controller.get_target_pose()
    assert np.allclose(pos, target_pos), "Target position should match"
    assert np.allclose(quat, target_quat), "Target quaternion should match"
    assert frame == "world", "Frame should match"

    print("✓ Arm controller target management works correctly")


def test_auto_mode_config():
    """Test that auto mode is properly configured."""
    print("\n=== Test: Auto Mode Configuration ===")

    # Create config with auto mode
    config = IKConfig(
        control_mode="auto",
        left_ee_link="left_ee",
        right_ee_link="right_ee",
        left_ee_topic="/left_ee_target",
        right_ee_topic="/right_ee_target",
    )

    assert config.control_mode == "auto", "Control mode should be 'auto'"
    assert config.is_ik_enabled(), "IK should be enabled in auto mode"

    print(f"✓ Auto mode config: control_mode={config.control_mode}")
    print(f"  Left EE: {config.left_ee_link}, topic: {config.left_ee_topic}")
    print(f"  Right EE: {config.right_ee_link}, topic: {config.right_ee_topic}")


def test_auto_mode_behavior():
    """Test auto mode behavior (without full ROS2/Isaac Sim setup)."""
    print("\n=== Test: Auto Mode Behavior ===")

    # Create mock config and solver
    config = IKConfig(
        control_mode="auto",
        ik_timeout_sec=0.1,
        left_ee_link="left_ee",
        right_ee_link="right_ee",
    )
    ik_solver = IKSolverCore()

    # Create arm controllers
    left_arm = ArmIKController(
        arm_name="left",
        joint_names=["l_j1", "l_j2", "l_j3"],
        ee_link="left_ee",
        ik_solver=ik_solver,
        config=config,
    )

    right_arm = ArmIKController(
        arm_name="right",
        joint_names=["r_j1", "r_j2", "r_j3"],
        ee_link="right_ee",
        ik_solver=ik_solver,
        config=config,
    )

    # Test 1: No active targets
    print("\nTest 1: No active targets")
    assert not left_arm.has_active_target(), "Left arm should have no target"
    assert not right_arm.has_active_target(), "Right arm should have no target"
    print("✓ Both arms have no active targets")

    # Test 2: Left arm has target
    print("\nTest 2: Left arm receives target")
    left_arm.set_target_pose(np.array([0.5, 0.3, 0.8]), np.array([1.0, 0.0, 0.0, 0.0]), "world")
    assert left_arm.has_active_target(), "Left arm should have active target"
    assert not right_arm.has_active_target(), "Right arm should have no target"
    print("✓ Only left arm has active target")

    # Test 3: Both arms have targets
    print("\nTest 3: Both arms receive targets")
    right_arm.set_target_pose(np.array([0.5, -0.3, 0.8]), np.array([1.0, 0.0, 0.0, 0.0]), "world")
    assert left_arm.has_active_target(), "Left arm should have active target"
    assert right_arm.has_active_target(), "Right arm should have active target"
    print("✓ Both arms have active targets")

    # Test 4: Targets timeout after configured duration
    print("\nTest 4: Target timeout behavior")
    time.sleep(config.ik_timeout_sec + 0.05)
    assert not left_arm.has_active_target(), "Left arm target should time out"
    assert not right_arm.has_active_target(), "Right arm target should time out"
    print("✓ Targets correctly become inactive after timeout")


def test_control_mode_update_behavior():
    """Verify bridge update behavior for auto vs ee_pose idle cases."""
    print("\n=== Test: Bridge Update Behavior ===")

    auto_bridge = create_test_bridge("auto")
    auto_bridge.update()
    auto_status = auto_bridge.get_control_status()

    assert len(auto_bridge._left_pose_pub.messages) == 0
    assert len(auto_bridge._right_pose_pub.messages) == 0
    assert auto_status["performance"]["update_calls"] == 1
    assert auto_status["performance"]["active_update_calls"] == 0
    print("✓ auto mode stays idle without active targets")

    ee_pose_bridge = create_test_bridge("ee_pose")
    ee_pose_bridge.update()
    ee_status = ee_pose_bridge.get_control_status()
    assert len(ee_pose_bridge._left_pose_pub.messages) == 1
    assert len(ee_pose_bridge._right_pose_pub.messages) == 1
    assert len(ee_pose_bridge._left_cmd_pub.messages) == 0
    assert len(ee_pose_bridge._right_cmd_pub.messages) == 0
    assert ee_pose_bridge._left_status_pub.messages[0].data.startswith("idle")
    assert ee_pose_bridge._right_status_pub.messages[0].data.startswith("idle")
    assert np.isclose(ee_pose_bridge._left_pose_pub.messages[0].pose.position.x, 0.12)
    assert np.isclose(ee_pose_bridge._left_pose_pub.messages[0].pose.position.y, 0.22)
    assert np.isclose(ee_pose_bridge._left_pose_pub.messages[0].pose.position.z, 0.32)
    assert np.isclose(ee_pose_bridge._right_pose_pub.messages[0].pose.position.x, 0.42)
    assert np.isclose(ee_pose_bridge._right_pose_pub.messages[0].pose.position.y, 0.52)
    assert np.isclose(ee_pose_bridge._right_pose_pub.messages[0].pose.position.z, 0.62)
    assert ee_status["performance"]["update_calls"] == 1
    assert ee_status["performance"]["active_update_calls"] == 1
    assert ee_status["performance"]["avg_active_update_ms"] is not None
    assert ee_status["left_arm"]["last_status"] == "idle"
    assert ee_status["right_arm"]["last_status"] == "idle"
    print("✓ ee_pose mode publishes current EE pose and performance stats while idle")


def test_gripper_center_tcp_pose_uses_midpoint_and_component_links():
    """Synthetic TCP pose should use fingertip midpoint and resolve component link prim paths."""
    bridge = create_test_bridge("ee_pose")

    pos, quat = bridge._get_ee_pose(None, "left_gripper_center_tcp")

    assert np.allclose(pos, [0.12, 0.22, 0.32])
    assert np.allclose(rotate_x_axis(quat), [0.0, 0.0, 1.0], atol=1e-6)
    assert bridge._ee_prim_paths["left_gripper_narrow3_link"] == "/robot/left_gripper_narrow3_link"
    assert bridge._ee_prim_paths["left_gripper_wide3_link"] == "/robot/left_gripper_wide3_link"
    assert bridge._ee_prim_paths["left_arm_link9"] == "/robot/left_arm_link9"
    print("✓ gripper center TCP pose uses midpoint and component link paths")


def test_gripper_center_tcp_jacobian_averages_fingertips():
    """Synthetic TCP translational Jacobian should be the fingertip midpoint Jacobian."""
    bridge = create_test_bridge("ee_pose")

    jacobian = bridge._extract_arm_jacobian(
        bridge.articulation_view.get_jacobians()[0],
        None,
        "left_gripper_center_tcp",
        np.array([0], dtype=np.int32),
        "left",
    )

    assert jacobian.shape == (3, 1)
    assert np.allclose(jacobian[:, 0], [0.3, 0.0, 0.0])
    print("✓ gripper center TCP Jacobian averages fingertip Jacobians")


def test_ee_pose_mode_vs_auto_mode():
    """Compare ee_pose mode and auto mode behavior."""
    print("\n=== Test: EE Pose Mode vs Auto Mode ===")

    # ee_pose mode: IK enabled and publishes current EE pose while idle
    config_ee_pose = IKConfig(control_mode="ee_pose")
    print(f"✓ ee_pose mode: is_ik_enabled={config_ee_pose.is_ik_enabled()}")
    print("  Behavior: IK stays enabled and publishes current EE poses")

    # auto mode: IK only runs when targets are active
    config_auto = IKConfig(control_mode="auto")
    print(f"✓ auto mode: is_ik_enabled={config_auto.is_ik_enabled()}")
    print("  Behavior: IK runs only when EE pose targets are received")

    # joint mode: IK disabled
    config_joint = IKConfig(control_mode="joint")
    print(f"✓ joint mode: is_ik_enabled={config_joint.is_ik_enabled()}")
    print("  Behavior: Direct joint control, no IK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Auto Mode IK Tests")
    print("=" * 60)

    try:
        test_arm_controller_target_management()
        test_auto_mode_config()
        test_auto_mode_behavior()
        test_control_mode_update_behavior()
        test_gripper_center_tcp_pose_uses_midpoint_and_component_links()
        test_gripper_center_tcp_jacobian_averages_fingertips()
        test_ee_pose_mode_vs_auto_mode()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
