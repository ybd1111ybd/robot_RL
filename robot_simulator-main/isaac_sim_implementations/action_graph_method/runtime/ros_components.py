"""ROS2-side runtime helpers for the Action Graph mainline."""

from __future__ import annotations

import threading
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from ros2_singleton import ProcessSingletonLock
from runtime.mainline_shared import (
    BODY_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRIPPER_STATE_JOINTS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRIPPER_STATE_JOINTS,
    _clamp01,
)


class GripperSwitchRemapper:
    """Map a single gripper switch command into two physical gripper joints."""

    def __init__(
        self,
        left_in_topic: str,
        right_in_topic: str,
        left_out_topic: str,
        right_out_topic: str,
    ) -> None:
        self.left_in_topic = left_in_topic
        self.right_in_topic = right_in_topic
        self.left_out_topic = left_out_topic
        self.right_out_topic = right_out_topic
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._executor = None
        self._node = None
        self._JointState = None
        self._rclpy = None
        self._owns_rclpy_context = False
        self._left_pub = None
        self._right_pub = None
        self._left_debug_count = 0
        self._right_debug_count = 0

    @staticmethod
    def _arm_names(side: str):
        return LEFT_ARM_JOINTS if side == "left" else RIGHT_ARM_JOINTS

    @staticmethod
    def _gripper_names(side: str):
        return LEFT_GRIPPER_JOINTS if side == "left" else RIGHT_GRIPPER_JOINTS

    @staticmethod
    def _switch_aliases(side: str):
        return (
            f"{side}_gripper",
            f"{side}_gripper_switch",
            f"{side}_gripper_cmd",
            f"{side}_gripper_open",
            "gripper",
            "gripper_switch",
            "gripper_cmd",
            "gripper_open",
        )

    def _extract_switch(self, side: str, msg, name_to_pos: dict[str, float]) -> float | None:
        for alias in self._switch_aliases(side):
            if alias in name_to_pos:
                return float(name_to_pos[alias])

        if len(msg.position) >= 8:
            if len(msg.name) <= 8:
                return float(msg.position[7])
        return None

    def _build_output(self, side: str, msg):
        out = self._JointState()
        out.header = msg.header

        max_pairs = min(len(msg.name), len(msg.position))
        name_to_pos = {msg.name[i]: float(msg.position[i]) for i in range(max_pairs)}

        out_names: list[str] = []
        out_positions: list[float] = []

        arm_names = self._arm_names(side)
        compact_mode = len(msg.position) >= 7 and len(msg.name) in (0, 7)
        if compact_mode:
            out_names.extend(arm_names)
            out_positions.extend(float(v) for v in msg.position[:7])
        else:
            for arm_joint in arm_names:
                if arm_joint in name_to_pos:
                    out_names.append(arm_joint)
                    out_positions.append(name_to_pos[arm_joint])

        switch = self._extract_switch(side, msg, name_to_pos)
        narrow_name, wide_name = self._gripper_names(side)
        if switch is not None:
            s = _clamp01(switch)
            narrow_pos = GRIPPER_CLOSED["narrow"] + (
                GRIPPER_OPEN["narrow"] - GRIPPER_CLOSED["narrow"]
            ) * s
            wide_pos = GRIPPER_CLOSED["wide"] + (
                GRIPPER_OPEN["wide"] - GRIPPER_CLOSED["wide"]
            ) * s
            out_names.extend([narrow_name, wide_name])
            out_positions.extend([narrow_pos, wide_pos])
        else:
            if narrow_name in name_to_pos:
                out_names.append(narrow_name)
                out_positions.append(name_to_pos[narrow_name])
            if wide_name in name_to_pos:
                out_names.append(wide_name)
                out_positions.append(name_to_pos[wide_name])

        out.name = out_names
        out.position = out_positions
        return out

    def _on_left(self, msg) -> None:
        out = self._build_output("left", msg)
        if self._left_debug_count < 3:
            print(
                "[gripper-remap:left]"
                f" in(name={len(msg.name)}, pos={len(msg.position)})"
                f" -> out(name={len(out.name)}, pos={len(out.position)})"
                f" names={list(out.name)}"
                f" pos={list(out.position)}"
            )
            self._left_debug_count += 1
        self._left_pub.publish(out)

    def _on_right(self, msg) -> None:
        out = self._build_output("right", msg)
        if self._right_debug_count < 3:
            print(
                "[gripper-remap:right]"
                f" in(name={len(msg.name)}, pos={len(msg.position)})"
                f" -> out(name={len(out.name)}, pos={len(out.position)})"
                f" names={list(out.name)}"
                f" pos={list(out.position)}"
            )
            self._right_debug_count += 1
        self._right_pub.publish(out)

    def _spin(self) -> None:
        while self._running:
            self._executor.spin_once(timeout_sec=0.05)

    def start(self) -> None:
        if self._running:
            return
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy
        self._JointState = JointState

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True
        self._node = rclpy.create_node("gripper_switch_remapper")
        self._left_pub = self._node.create_publisher(JointState, self.left_out_topic, 10)
        self._right_pub = self._node.create_publisher(JointState, self.right_out_topic, 10)
        self._node.create_subscription(JointState, self.left_in_topic, self._on_left, 10)
        self._node.create_subscription(JointState, self.right_in_topic, self._on_right, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()


class ManualJointStatePublisher:
    def __init__(
        self,
        articulation: Any,
        left_topic: str,
        right_topic: str,
        body_topic: str,
        publish_rate_hz: float,
    ) -> None:
        self._articulation = articulation
        self._left_topic = left_topic
        self._right_topic = right_topic
        self._body_topic = body_topic
        self._publish_period = 1.0 / max(publish_rate_hz, 1e-3)
        self._last_publish = 0.0
        self._warned: set[str] = set()
        self._owns_rclpy_context = False
        self._rclpy = None
        self._node = None
        self._JointState = None
        self._left_pub = None
        self._right_pub = None
        self._body_pub = None
        self._left_names: List[str] = []
        self._right_names: List[str] = []
        self._body_names: List[str] = []
        self._left_indices = np.array([], dtype=np.int32)
        self._right_indices = np.array([], dtype=np.int32)
        self._body_indices = np.array([], dtype=np.int32)
        self._timeline = None
        self._singleton_lock: Optional[ProcessSingletonLock] = None
        self._disabled = False

        self._singleton_lock = ProcessSingletonLock("manual_joint_state_publisher")
        if not self._singleton_lock.acquire():
            self._disabled = True
            print(
                "[JointStatePublisher] Duplicate publisher suppressed:"
                f" existing owner {self._singleton_lock.owner_description()}"
            )
            return

        try:
            import omni.timeline
            import rclpy
            from sensor_msgs.msg import JointState

            self._rclpy = rclpy
            self._JointState = JointState
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy_context = True
            self._node = rclpy.create_node("manual_joint_state_publisher")
            self._left_pub = self._node.create_publisher(JointState, left_topic, 10)
            self._right_pub = self._node.create_publisher(JointState, right_topic, 10)
            self._body_pub = self._node.create_publisher(JointState, body_topic, 10)
            self._timeline = omni.timeline.get_timeline_interface()
        except Exception:
            if self._singleton_lock is not None:
                self._singleton_lock.release()
            self._singleton_lock = None
            raise

        self._refresh_indices()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        print(message)

    def _refresh_indices(self) -> None:
        if self._articulation is None:
            return
        try:
            self._articulation.initialize()
        except Exception:
            pass

        dof_names = list(getattr(self._articulation, "dof_names", []) or [])
        if not dof_names:
            self._warn_once(
                "joint-state-no-dofs",
                "[JointStatePublisher] Warning: articulation has no dof names yet.",
            )
            return

        name_to_idx = {name: idx for idx, name in enumerate(dof_names)}

        def resolve(names: List[str], label: str) -> Tuple[List[str], np.ndarray]:
            resolved: List[str] = []
            indices: List[int] = []
            missing: List[str] = []
            for name in names:
                idx = name_to_idx.get(name)
                if idx is None:
                    missing.append(name)
                else:
                    resolved.append(name)
                    indices.append(idx)
            if missing:
                self._warn_once(
                    f"joint-state-missing-{label}",
                    f"[JointStatePublisher] Warning: missing joints for {label}: {missing}",
                )
            return resolved, np.array(indices, dtype=np.int32)

        self._left_names, self._left_indices = resolve(
            LEFT_ARM_JOINTS + LEFT_GRIPPER_STATE_JOINTS, "left"
        )
        self._right_names, self._right_indices = resolve(
            RIGHT_ARM_JOINTS + RIGHT_GRIPPER_STATE_JOINTS, "right"
        )
        self._body_names, self._body_indices = resolve(BODY_JOINTS, "body")

    def _ensure_ready(self) -> bool:
        if self._articulation is None:
            self._warn_once(
                "joint-state-no-articulation",
                "[JointStatePublisher] Error: articulation view is missing.",
            )
            return False
        if self._timeline is not None:
            try:
                if not self._timeline.is_playing():
                    return False
            except Exception:
                pass
        if hasattr(self._articulation, "is_physics_handle_valid"):
            try:
                if not self._articulation.is_physics_handle_valid():
                    self._refresh_indices()
                    return False
            except Exception:
                pass
        if (
            len(self._left_indices) == 0
            and len(self._right_indices) == 0
            and len(self._body_indices) == 0
        ):
            self._refresh_indices()
            return False
        return True

    @staticmethod
    def _safe_array(raw) -> Optional[np.ndarray]:
        if raw is None:
            return None
        try:
            arr = np.asarray(raw, dtype=np.float32)
        except Exception:
            return None
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        if arr.ndim == 1:
            return arr
        return None

    def _publish_group(
        self,
        publisher,
        names: List[str],
        indices: np.ndarray,
        positions: np.ndarray,
        velocities: Optional[np.ndarray],
        efforts: Optional[np.ndarray],
    ) -> None:
        if not names or len(indices) == 0:
            return
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = names
        msg.position = [float(positions[i]) for i in indices]
        if velocities is not None:
            msg.velocity = [float(velocities[i]) for i in indices]
        if efforts is not None:
            msg.effort = [float(efforts[i]) for i in indices]
        publisher.publish(msg)

    def publish_if_due(self) -> None:
        if self._disabled or self._node is None:
            return
        now = time.time()
        if now - self._last_publish < self._publish_period:
            return
        if not self._ensure_ready():
            return

        try:
            positions_raw = self._articulation.get_joint_positions()
            velocities_raw = self._articulation.get_joint_velocities()
            efforts_raw = None
            if hasattr(self._articulation, "get_joint_efforts"):
                try:
                    efforts_raw = self._articulation.get_joint_efforts()
                except Exception:
                    efforts_raw = None
        except Exception as exc:
            self._warn_once(
                "joint-state-read-fail",
                f"[JointStatePublisher] Warning: failed to read articulation state: {exc}",
            )
            self._refresh_indices()
            return

        positions = self._safe_array(positions_raw)
        velocities = self._safe_array(velocities_raw)
        efforts = self._safe_array(efforts_raw)
        if positions is None:
            self._warn_once(
                "joint-state-no-positions",
                "[JointStatePublisher] Warning: joint positions not available.",
            )
            return

        self._publish_group(
            self._left_pub,
            self._left_names,
            self._left_indices,
            positions,
            velocities,
            efforts,
        )
        self._publish_group(
            self._right_pub,
            self._right_names,
            self._right_indices,
            positions,
            velocities,
            efforts,
        )
        self._publish_group(
            self._body_pub,
            self._body_names,
            self._body_indices,
            positions,
            velocities,
            efforts,
        )
        self._last_publish = now

    def stop(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        if self._singleton_lock is not None:
            self._singleton_lock.release()
            self._singleton_lock = None


class BodyPostureKeeper:
    def __init__(
        self,
        articulation: Any,
        joint_names: List[str],
    ) -> None:
        self._articulation = articulation
        self._joint_names = list(joint_names)
        self._target_positions: Optional[np.ndarray] = None
        self._enabled = False
        self._capture_targets()

    def _capture_targets(self) -> None:
        try:
            self._articulation.initialize()
        except Exception:
            pass
        dof_names = list(getattr(self._articulation, "dof_names", []) or [])
        if not dof_names:
            return
        name_to_index = {name: idx for idx, name in enumerate(dof_names)}
        indices = [name_to_index[name] for name in self._joint_names if name in name_to_index]
        if len(indices) != len(self._joint_names):
            missing = [name for name in self._joint_names if name not in name_to_index]
            print(f"[BodyHold] Warning: missing body joints for posture hold: {missing}")
            return
        try:
            joint_positions = self._articulation.get_joint_positions()
        except Exception:
            return
        if joint_positions is None:
            return
        arr = np.asarray(joint_positions, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        elif arr.ndim != 1:
            return
        self._target_positions = np.asarray([arr[idx] for idx in indices], dtype=np.float32)
        self._enabled = True
        print(
            "[BodyHold] Captured startup posture:"
            f" joints={self._joint_names}"
            f" targets={self._target_positions.tolist()}"
        )

    def apply(self) -> None:
        if not self._enabled or self._target_positions is None:
            return
        try:
            self._articulation.set_joint_position_targets(
                self._target_positions.reshape(1, -1),
                joint_names=self._joint_names,
            )
        except Exception as exc:
            print(f"[BodyHold] Warning: failed to apply body posture hold: {exc}")
            self._enabled = False
