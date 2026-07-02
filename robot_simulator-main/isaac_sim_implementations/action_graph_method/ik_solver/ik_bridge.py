"""
IK Bridge Module

Handles ROS2 communication and integrates IK solver with Isaac Sim.
Runs in a separate thread similar to GripperSwitchRemapper.
"""

import threading
import time
import numpy as np
from typing import Optional, Dict, Tuple

from .ik_config import IKConfig
from .ik_solver_core import IKSolverCore
from .ik_exceptions import (
    IKException,
    IKArticulationError,
    IKInputValidationError,
)
from .ik_validators import validate_config


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """Return a normalized [w, x, y, z] quaternion."""
    quat = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat / norm


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert URDF-style fixed-axis RPY to [w, x, y, z] quaternion."""
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    return _normalize_quaternion(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=float,
        )
    )


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Hamilton product for [w, x, y, z] quaternions."""
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


def _quat_rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a [w, x, y, z] quaternion."""
    quat = _normalize_quaternion(quaternion)
    vec = np.asarray(vector, dtype=float).reshape(3)
    vec_quat = np.array([0.0, vec[0], vec[1], vec[2]], dtype=float)
    quat_conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)
    rotated = _quat_multiply(_quat_multiply(quat, vec_quat), quat_conj)
    return rotated[1:4]


def _quat_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized [w, x, y, z] quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=float,
        )
    else:
        diagonal = np.diag(matrix)
        axis_index = int(np.argmax(diagonal))
        if axis_index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ],
                dtype=float,
            )
        elif axis_index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ],
                dtype=float,
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
    return _normalize_quaternion(quat)


GRIPPER_MOUNT_QUAT = _quat_from_rpy(-2.9951, -1.5708, -0.15964)


SYNTHETIC_TCP_SPECS = {
    "left_gripper_center_tcp": {
        "position_links": ("left_gripper_narrow3_link", "left_gripper_wide3_link"),
        "orientation_link": "left_arm_link9",
        "orientation_offset_quat": GRIPPER_MOUNT_QUAT,
    },
    "right_gripper_center_tcp": {
        "position_links": ("right_gripper_narrow3_link", "right_gripper_wide3_link"),
        "orientation_link": "right_arm_link9",
        "orientation_offset_quat": GRIPPER_MOUNT_QUAT,
    },
}


class ArmIKController:
    """IK controller for a single arm."""

    def __init__(
        self,
        arm_name: str,
        ee_link: str,
        joint_names: list,
        ik_solver: IKSolverCore,
        config: IKConfig,
    ):
        """
        Initialize arm IK controller.

        Args:
            arm_name: Name of the arm (e.g., "left", "right")
            ee_link: End-effector link name
            joint_names: List of joint names for this arm
            ik_solver: IK solver instance
            config: IK configuration
        """
        self.arm_name = arm_name
        self.ee_link = ee_link
        self.joint_names = joint_names
        self.ik_solver = ik_solver
        self.config = config

        # Target pose (thread-safe)
        self._target_lock = threading.Lock()
        self._target_pos: Optional[np.ndarray] = None
        self._target_quat: Optional[np.ndarray] = None
        self._target_frame: Optional[str] = None
        self._last_target_time: float = 0.0

        # Status
        self.last_status: Dict = {}
        self._last_limit_signature: Optional[str] = None
        self._last_limit_log_time: float = 0.0

    def set_target_pose(
        self, position: np.ndarray, orientation: np.ndarray, frame_id: str
    ):
        """
        Set target pose (thread-safe).

        Args:
            position: Target position [x, y, z]
            orientation: Target quaternion [w, x, y, z]
            frame_id: Frame ID
        """
        with self._target_lock:
            self._target_pos = position.copy()
            self._target_quat = orientation.copy()
            self._target_frame = frame_id
            self._last_target_time = time.time()

    def get_target_pose(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str], float]:
        """
        Get target pose (thread-safe).

        Returns:
            Tuple of (position, quaternion, frame_id, timestamp)
        """
        with self._target_lock:
            return (
                self._target_pos.copy() if self._target_pos is not None else None,
                self._target_quat.copy() if self._target_quat is not None else None,
                self._target_frame,
                self._last_target_time,
            )

    def has_active_target(self) -> bool:
        """Check if there is an active target within timeout."""
        with self._target_lock:
            if self._target_pos is None:
                return False
            elapsed = time.time() - self._last_target_time
            return elapsed < self.config.ik_timeout_sec


class EndEffectorIKBridge:
    """
    ROS2 bridge for end-effector IK control.

    Subscribes to end-effector pose targets in a separate thread.
    Main simulation loop calls update() to compute and publish IK commands.
    """

    def __init__(
        self,
        config: IKConfig,
        articulation_view,
    ):
        """
        Initialize IK bridge.

        Args:
            config: IK configuration
            articulation_view: Isaac Sim ArticulationView
        """
        validate_config(config)
        self.config = config
        self.articulation_view = articulation_view

        # Create IK solver
        self.ik_solver = IKSolverCore(
            damping=config.ik_damping,
            pos_gain=config.ik_pos_gain,
            ori_gain=config.ik_ori_gain,
            max_dq=config.ik_max_dq,
            max_joint_vel=config.ik_max_joint_vel,
            pos_tol=config.ik_pos_tol,
            ori_tol=config.ik_ori_tol,
            enable_orientation=config.ik_enable_orientation,
        )

        # Create arm controllers
        self.left_arm = ArmIKController(
            arm_name="left",
            ee_link=config.left_ee_link,
            joint_names=config.left_arm_joints,
            ik_solver=self.ik_solver,
            config=config,
        )

        self.right_arm = ArmIKController(
            arm_name="right",
            ee_link=config.right_ee_link,
            joint_names=config.right_arm_joints,
            ik_solver=self.ik_solver,
            config=config,
        )

        # Joint indices cache
        self._left_joint_indices: Optional[np.ndarray] = None
        self._right_joint_indices: Optional[np.ndarray] = None

        # End-effector body indices cache
        self._left_ee_body_index: Optional[int] = None
        self._right_ee_body_index: Optional[int] = None
        self._body_index_cache: Dict[str, Optional[int]] = {}
        self._ee_prim_paths: Dict[str, Optional[str]] = {}
        self._ee_pose_prims: Dict[str, object] = {}
        self._rigid_prim_cls = None
        self._rigid_prim_import_attempted = False

        # Runtime warning guards
        self._warned_missing_joint_limits = False
        self._warned_jacobian_shape = False
        self._warned_messages = set()
        self._articulation_root_hint = None
        self._articulation_rewrap_attempts = 0
        self._articulation_rewrap_last_time = 0.0
        self._articulation_fail_count = 0
        self._articulation_fail_last_time = 0.0
        self._articulation_last_ok_time = 0.0
        self._base_link_name = "base_link"
        self._timeline = None
        self._stage_utils = None
        # Avoid creating new PhysX views during runtime (can invalidate tensor views).
        # RigidPrim pose lookups can trigger view creation; keep disabled by default.
        self._allow_rigid_prim_pose = False

        # ROS2 thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._executor = None
        self._node = None
        self._rclpy = None
        self._owns_rclpy_context = False

        # Publishers (created in ROS2 thread)
        self._left_cmd_pub = None
        self._right_cmd_pub = None
        self._left_pose_pub = None
        self._right_pose_pub = None
        self._left_status_pub = None
        self._right_status_pub = None

        # Message types (imported in ROS2 thread)
        self._PoseStamped = None
        self._JointState = None
        self._String = None

        # Statistics
        self._stats = {
            "left_ik_updates": 0,
            "right_ik_updates": 0,
            "last_left_target_time": 0.0,
            "last_right_target_time": 0.0,
            "last_left_ik_time": 0.0,
            "last_right_ik_time": 0.0,
            "update_calls": 0,
            "active_update_calls": 0,
            "update_total_time_sec": 0.0,
            "active_update_total_time_sec": 0.0,
            "update_max_time_sec": 0.0,
            "active_update_max_time_sec": 0.0,
            "last_update_duration_sec": 0.0,
            "perf_warn_count": 0,
            "last_perf_log_time": 0.0,
        }

        print(f"[IK Bridge] Initialized with control_mode={config.control_mode}")
        print(f"[IK Bridge] Left EE: {config.left_ee_link}, topic: {config.left_ee_topic}")
        print(f"[IK Bridge] Right EE: {config.right_ee_link}, topic: {config.right_ee_topic}")
        if config.control_mode == "auto":
            print("[IK Bridge] Auto mode: IK activates only when EE pose targets are received")

        self._articulation_root_hint = self._get_articulation_root_path()
        try:
            import omni.timeline

            self._timeline = omni.timeline.get_timeline_interface()
        except Exception:
            self._timeline = None

        try:
            import isaacsim.core.utils.stage as stage_utils

            self._stage_utils = stage_utils
        except Exception:
            self._stage_utils = None

    def _on_left_ee_target(self, msg):
        """Callback for left arm end-effector target."""
        try:
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ])

            # Validate inputs
            if not np.all(np.isfinite(pos)):
                print(f"[IK Bridge] Invalid left EE position: {pos}")
                return

            if not np.all(np.isfinite(quat)):
                print(f"[IK Bridge] Invalid left EE quaternion: {quat}")
                return

            # Normalize quaternion
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-6:
                print(f"[IK Bridge] Left EE quaternion has zero norm")
                return
            quat = quat / quat_norm

            self.left_arm.set_target_pose(pos, quat, msg.header.frame_id)
            self._stats["last_left_target_time"] = time.time()

        except Exception as e:
            print(f"[IK Bridge] Error in left EE target callback: {e}")

    def _on_right_ee_target(self, msg):
        """Callback for right arm end-effector target."""
        try:
            pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ])

            # Validate inputs
            if not np.all(np.isfinite(pos)):
                print(f"[IK Bridge] Invalid right EE position: {pos}")
                return

            if not np.all(np.isfinite(quat)):
                print(f"[IK Bridge] Invalid right EE quaternion: {quat}")
                return

            # Normalize quaternion
            quat_norm = np.linalg.norm(quat)
            if quat_norm < 1e-6:
                print(f"[IK Bridge] Right EE quaternion has zero norm")
                return
            quat = quat / quat_norm

            self.right_arm.set_target_pose(pos, quat, msg.header.frame_id)
            self._stats["last_right_target_time"] = time.time()

        except Exception as e:
            print(f"[IK Bridge] Error in right EE target callback: {e}")

    def _spin(self):
        """ROS2 spin loop in separate thread."""
        while self._running:
            self._executor.spin_once(timeout_sec=0.05)

    def start(self):
        """Start ROS2 subscriber thread."""
        if self._running:
            return

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rcl_interfaces.msg import SetParametersResult
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        self._rclpy = rclpy
        self._PoseStamped = PoseStamped
        self._JointState = JointState
        self._String = String

        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy_context = True

        self._node = rclpy.create_node("ik_bridge")

        # Declare tunable parameters for runtime adjustment.
        self._node.declare_parameter("ik_damping", float(self.config.ik_damping))
        self._node.declare_parameter("ik_pos_gain", float(self.config.ik_pos_gain))
        self._node.declare_parameter("ik_ori_gain", float(self.config.ik_ori_gain))
        self._node.declare_parameter("ik_max_dq", float(self.config.ik_max_dq))
        self._node.declare_parameter("ik_max_joint_vel", float(self.config.ik_max_joint_vel))
        self._node.declare_parameter("ik_pos_tol", float(self.config.ik_pos_tol))
        self._node.declare_parameter("ik_ori_tol", float(self.config.ik_ori_tol))
        self._node.declare_parameter("ik_enable_orientation", bool(self.config.ik_enable_orientation))
        self._node.declare_parameter("ik_timeout_sec", float(self.config.ik_timeout_sec))

        def _on_param_update(params):
            for param in params:
                name = param.name
                try:
                    if name == "ik_damping":
                        value = float(param.value)
                        if value < 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_damping = value
                        self.ik_solver.damping = value
                    elif name == "ik_pos_gain":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_pos_gain = value
                        self.ik_solver.pos_gain = value
                    elif name == "ik_ori_gain":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_ori_gain = value
                        self.ik_solver.ori_gain = value
                    elif name == "ik_max_dq":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_max_dq = value
                        self.ik_solver.max_dq = value
                    elif name == "ik_max_joint_vel":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_max_joint_vel = value
                        self.ik_solver.max_joint_vel = value
                    elif name == "ik_pos_tol":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_pos_tol = value
                        self.ik_solver.pos_tol = value
                    elif name == "ik_ori_tol":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_ori_tol = value
                        self.ik_solver.ori_tol = value
                    elif name == "ik_enable_orientation":
                        value = bool(param.value)
                        self.config.ik_enable_orientation = value
                        self.ik_solver.enable_orientation = value
                    elif name == "ik_timeout_sec":
                        value = float(param.value)
                        if value <= 0:
                            return SetParametersResult(successful=False)
                        self.config.ik_timeout_sec = value
                except Exception:
                    return SetParametersResult(successful=False)
            return SetParametersResult(successful=True)

        self._node.add_on_set_parameters_callback(_on_param_update)

        # Create subscribers
        self._node.create_subscription(
            PoseStamped,
            self.config.left_ee_topic,
            self._on_left_ee_target,
            10,
        )
        self._node.create_subscription(
            PoseStamped,
            self.config.left_ee_topic_compat,
            self._on_left_ee_target,
            10,
        )
        self._node.create_subscription(
            PoseStamped,
            self.config.right_ee_topic,
            self._on_right_ee_target,
            10,
        )
        self._node.create_subscription(
            PoseStamped,
            self.config.right_ee_topic_compat,
            self._on_right_ee_target,
            10,
        )

        # Create publishers
        self._left_cmd_pub = self._node.create_publisher(
            JointState, self.config.left_cmd_topic, 10
        )
        self._right_cmd_pub = self._node.create_publisher(
            JointState, self.config.right_cmd_topic, 10
        )
        self._left_pose_pub = self._node.create_publisher(
            PoseStamped, "/arm_left/ee_current_pose", 10
        )
        self._right_pose_pub = self._node.create_publisher(
            PoseStamped, "/arm_right/ee_current_pose", 10
        )
        self._left_status_pub = self._node.create_publisher(
            String, "/arm_left/ee_ik_status", 10
        )
        self._right_status_pub = self._node.create_publisher(
            String, "/arm_right/ee_ik_status", 10
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

        print("[IK Bridge] ROS2 thread started")

    def stop(self):
        """Stop ROS2 subscriber thread."""
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

        print("[IK Bridge] Stopped")

    def _get_joint_indices(self, joint_names: list) -> np.ndarray:
        """Get joint indices from articulation view."""
        all_joint_names = self.articulation_view.dof_names
        indices = []
        for name in joint_names:
            try:
                idx = all_joint_names.index(name)
                indices.append(idx)
            except ValueError:
                print(f"[IK Bridge] Warning: Joint {name} not found in articulation")
        return np.array(indices, dtype=np.int32)

    def _get_tcp_spec(self, ee_link_name: Optional[str]):
        """Return synthetic TCP definition when ee_link_name is a virtual TCP alias."""
        if ee_link_name is None:
            return None
        return SYNTHETIC_TCP_SPECS.get(ee_link_name)

    def _get_link_body_index(self, link_name: str) -> Optional[int]:
        """Get body index for any physical link name from the articulation view."""
        if link_name in self._body_index_cache:
            return self._body_index_cache[link_name]

        try:
            body_names = list(getattr(self.articulation_view, "body_names", []) or [])
            for index, name in enumerate(body_names):
                if str(name).lower().endswith(link_name.lower()):
                    self._body_index_cache[link_name] = index
                    return index
            self._warn_once(
                f"body-index-missing:{link_name}",
                f"[IK Bridge] Warning: Link {link_name} not found in body_names: {body_names}",
            )
        except Exception as exc:
            self._warn_once(
                f"body-index-error:{link_name}",
                f"[IK Bridge] Warning: Failed to get body index for {link_name}: {exc}",
            )

        self._body_index_cache[link_name] = None
        return None

    def _get_ee_body_index(self, ee_link_name: str) -> Optional[int]:
        """Get end-effector body index from articulation view."""
        return self._get_link_body_index(ee_link_name)

    def _warn_once(self, key: str, message: str) -> None:
        """Print a warning only once per bridge lifecycle."""
        if key in self._warned_messages:
            return
        self._warned_messages.add(key)
        print(message)

    def _note_articulation_success(self) -> None:
        """Record a successful articulation query to reset failure backoff."""
        self._articulation_fail_count = 0
        self._articulation_fail_last_time = 0.0
        self._articulation_last_ok_time = time.time()

    def _note_articulation_failure(
        self,
        reason: str,
        exc: Exception | None = None,
        force_rewrap: bool = False,
    ) -> None:
        """Track articulation failures and attempt rewrap when needed."""
        now = time.time()
        if now - self._articulation_fail_last_time > 2.0:
            self._articulation_fail_count = 0

        self._articulation_fail_count += 1
        self._articulation_fail_last_time = now

        key = f"articulation-fail:{reason}"
        message = f"[IK Bridge] Warning: Articulation view failure ({reason})"
        if exc is not None:
            message = f"{message}: {exc}"
        self._warn_once(key, message)

        if force_rewrap or self._articulation_fail_count >= 2:
            if self._rewrap_articulation_view(reason):
                self._articulation_fail_count = 0

    def _reset_cached_articulation_state(self) -> None:
        """Clear caches that depend on articulation view layout."""
        self._left_joint_indices = None
        self._right_joint_indices = None
        self._left_ee_body_index = None
        self._right_ee_body_index = None
        self._body_index_cache.clear()
        self._ee_prim_paths.clear()
        self._ee_pose_prims.clear()

    def _rewrap_articulation_view(self, reason: str) -> bool:
        """Attempt to create a fresh Articulation view when the existing one is invalid."""
        now = time.time()
        if self._articulation_rewrap_attempts >= 3 and now - self._articulation_rewrap_last_time < 5.0:
            return False
        if now - self._articulation_rewrap_last_time < 1.0:
            return False

        self._articulation_rewrap_attempts += 1
        self._articulation_rewrap_last_time = now

        root_path = self._articulation_root_hint or self._get_articulation_root_path()
        if not root_path:
            self._warn_once(
                "articulation-rewrap-root",
                "[IK Bridge] Warning: Unable to rewrap articulation (missing root path).",
            )
            return False

        try:
            from isaacsim.core.prims import Articulation
        except Exception as exc:
            self._warn_once(
                "articulation-rewrap-import",
                f"[IK Bridge] Warning: Failed to import Articulation for rewrap: {exc}",
            )
            return False

        try:
            new_view = Articulation(prim_paths_expr=root_path, name="ik_bridge_view")
            if hasattr(new_view, "initialize"):
                try:
                    new_view.initialize()
                except Exception as exc:
                    self._warn_once(
                        "articulation-rewrap-init",
                        f"[IK Bridge] Warning: Articulation initialize failed after rewrap: {exc}",
                    )
            self.articulation_view = new_view
            self._articulation_root_hint = root_path
            self._reset_cached_articulation_state()
            self._warn_once(
                "articulation-rewrap-ok",
                f"[IK Bridge] Rewrapped articulation view ({reason}).",
            )
            return True
        except Exception as exc:
            self._warn_once(
                "articulation-rewrap-fail",
                f"[IK Bridge] Warning: Failed to rewrap articulation view: {exc}",
            )
            return False

    def _ensure_articulation_ready(self) -> bool:
        """Ensure articulation view can provide jacobians and joint states."""
        if self._stage_utils is not None:
            try:
                if self._stage_utils.is_stage_loading():
                    self._warn_once(
                        "stage-loading",
                        "[IK Bridge] Stage still loading; skipping IK update.",
                    )
                    return False
            except Exception:
                pass

        if self._timeline is not None:
            try:
                if not self._timeline.is_playing():
                    return False
            except Exception:
                pass

        if self.articulation_view is None:
            self._warn_once(
                "articulation-none",
                "[IK Bridge] Error: Articulation view is not available.",
            )
            return False

        if not hasattr(self.articulation_view, "_physics_view"):
            self._warn_once(
                "articulation-missing-physics-view",
                "[IK Bridge] Warning: Articulation view missing _physics_view; attempting rewrap.",
            )
            self._rewrap_articulation_view("missing _physics_view")
            return True

        if hasattr(self.articulation_view, "initialize"):
            try:
                self.articulation_view.initialize()
            except Exception as exc:
                self._warn_once(
                    "articulation-init",
                    f"[IK Bridge] Warning: Articulation initialize failed: {exc}",
                )

        if hasattr(self.articulation_view, "is_physics_handle_valid"):
            try:
                if not self.articulation_view.is_physics_handle_valid():
                    self._warn_once(
                        "articulation-handle",
                        "[IK Bridge] Warning: Articulation physics handle not valid yet.",
                    )
                    self._note_articulation_failure(
                        "invalid physics handle",
                        force_rewrap=True,
                    )
                    return False
            except Exception as exc:
                self._warn_once(
                    "articulation-handle-error",
                    f"[IK Bridge] Warning: Articulation handle check failed: {exc}",
                )

        return True

    def _get_articulation_root_path(self) -> Optional[str]:
        """Get the articulation root prim path from the Isaac wrapper."""
        candidates = []

        for attr_name in ("prim_paths", "_prim_paths", "prim_paths_expr"):
            value = getattr(self.articulation_view, attr_name, None)
            if value is None:
                continue
            if isinstance(value, str):
                candidates.append(value)
                continue
            try:
                if len(value) > 0:
                    candidates.append(str(value[0]))
            except Exception:
                continue

        for candidate in candidates:
            if candidate:
                return candidate
        return None

    def _resolve_ee_prim_path(self, ee_link_name: str) -> Optional[str]:
        """Resolve the USD prim path for an end-effector link."""
        if ee_link_name in self._ee_prim_paths:
            return self._ee_prim_paths[ee_link_name]

        articulation_root = self._get_articulation_root_path()
        if not articulation_root:
            self._warn_once(
                f"ee-path-root:{ee_link_name}",
                f"[IK Bridge] Warning: Unable to resolve articulation root while locating {ee_link_name}",
            )
            self._ee_prim_paths[ee_link_name] = None
            return None

        body_names = list(getattr(self.articulation_view, "body_names", []) or [])
        candidates = []

        if articulation_root.endswith(f"/{ee_link_name}"):
            candidates.append(articulation_root)

        if body_names:
            root_body_name = str(body_names[0])
            if articulation_root.endswith(f"/{root_body_name}"):
                parent = articulation_root.rsplit("/", 1)[0]
                candidates.append(f"{parent}/{ee_link_name}")

        candidates.append(f"{articulation_root.rstrip('/')}/{ee_link_name}")

        deduped_candidates = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                deduped_candidates.append(candidate)

        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                for candidate in deduped_candidates:
                    prim = stage.GetPrimAtPath(candidate)
                    if prim and prim.IsValid():
                        self._ee_prim_paths[ee_link_name] = candidate
                        return candidate
        except Exception:
            pass

        resolved = deduped_candidates[0] if deduped_candidates else None
        if resolved is None:
            self._warn_once(
                f"ee-path-empty:{ee_link_name}",
                f"[IK Bridge] Warning: Failed to derive EE prim path for {ee_link_name}",
            )
        self._ee_prim_paths[ee_link_name] = resolved
        return resolved

    def _get_rigid_prim_class(self):
        """Lazily import RigidPrim so unit tests can run without Isaac Sim."""
        if self._rigid_prim_cls is not None:
            return self._rigid_prim_cls
        if self._rigid_prim_import_attempted:
            return None

        self._rigid_prim_import_attempted = True
        try:
            from isaacsim.core.prims import RigidPrim
        except Exception as exc:
            self._warn_once(
                "rigid-prim-import",
                f"[IK Bridge] Warning: Failed to import RigidPrim for EE pose lookup: {exc}",
            )
            return None

        self._rigid_prim_cls = RigidPrim
        return self._rigid_prim_cls

    def _get_ee_rigid_prim(self, ee_link_name: str):
        """Create or fetch a cached RigidPrim wrapper for the EE link."""
        if ee_link_name in self._ee_pose_prims:
            return self._ee_pose_prims[ee_link_name]

        rigid_prim_cls = self._get_rigid_prim_class()
        if rigid_prim_cls is None:
            return None

        ee_prim_path = self._resolve_ee_prim_path(ee_link_name)
        if not ee_prim_path:
            return None

        try:
            rigid_prim = rigid_prim_cls(
                prim_paths_expr=ee_prim_path,
                name=f"ik_{ee_link_name}_pose_view",
            )
        except Exception as exc:
            self._warn_once(
                f"rigid-prim-create:{ee_link_name}",
                f"[IK Bridge] Warning: Failed to create RigidPrim for {ee_link_name} at {ee_prim_path}: {exc}",
            )
            return None

        self._ee_pose_prims[ee_link_name] = rigid_prim
        return rigid_prim

    def _ensure_joint_indices(self):
        """Ensure joint indices are cached."""
        if self._left_joint_indices is None:
            self._left_joint_indices = self._get_joint_indices(self.config.left_arm_joints)
            print(f"[IK Bridge] Left arm joint indices: {self._left_joint_indices}")

        if self._right_joint_indices is None:
            self._right_joint_indices = self._get_joint_indices(self.config.right_arm_joints)
            print(f"[IK Bridge] Right arm joint indices: {self._right_joint_indices}")

    def _ensure_ee_body_indices(self):
        """Ensure end-effector body indices are cached."""
        left_tcp_spec = self._get_tcp_spec(self.config.left_ee_link)
        if left_tcp_spec is not None:
            self._left_ee_body_index = None
            position_links = ", ".join(left_tcp_spec["position_links"])
            self._warn_once(
                "left-ee-synthetic-tcp",
                f"[IK Bridge] Left EE synthetic TCP: {self.config.left_ee_link} -> midpoint({position_links}), +X points wrist->tcp midpoint",
            )
        elif self._left_ee_body_index is None:
            self._left_ee_body_index = self._get_ee_body_index(self.config.left_ee_link)
            if self._left_ee_body_index is not None:
                print(f"[IK Bridge] Left EE body index: {self._left_ee_body_index}")

        right_tcp_spec = self._get_tcp_spec(self.config.right_ee_link)
        if right_tcp_spec is not None:
            self._right_ee_body_index = None
            position_links = ", ".join(right_tcp_spec["position_links"])
            self._warn_once(
                "right-ee-synthetic-tcp",
                f"[IK Bridge] Right EE synthetic TCP: {self.config.right_ee_link} -> midpoint({position_links}), +X points wrist->tcp midpoint",
            )
        elif self._right_ee_body_index is None:
            self._right_ee_body_index = self._get_ee_body_index(self.config.right_ee_link)
            if self._right_ee_body_index is not None:
                print(f"[IK Bridge] Right EE body index: {self._right_ee_body_index}")

    def _extract_joint_limits(self, joint_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize Isaac dof limit output to two 1-D arrays."""
        joint_limits = self.articulation_view.get_dof_limits()
        dof_count = int(joint_positions.shape[-1])

        if joint_limits is None:
            if not self._warned_missing_joint_limits:
                print("[IK Bridge] Warning: No joint limits available, using +/-inf fallback")
                self._warned_missing_joint_limits = True
            return (
                np.full(dof_count, -np.inf, dtype=float),
                np.full(dof_count, np.inf, dtype=float),
            )

        if isinstance(joint_limits, tuple) and len(joint_limits) == 2:
            lower = np.asarray(joint_limits[0], dtype=float)
            upper = np.asarray(joint_limits[1], dtype=float)
        else:
            limits = np.asarray(joint_limits, dtype=float)
            if limits.ndim == 3 and limits.shape[0] == 1 and limits.shape[2] == 2:
                lower = limits[0, :, 0]
                upper = limits[0, :, 1]
            elif limits.ndim == 2 and limits.shape[1] == 2:
                lower = limits[:, 0]
                upper = limits[:, 1]
            elif limits.ndim == 2 and limits.shape[0] == 2:
                lower = limits[0]
                upper = limits[1]
            else:
                if not self._warned_missing_joint_limits:
                    print(
                        f"[IK Bridge] Warning: Unsupported joint limit shape {limits.shape}, using +/-inf fallback"
                    )
                    self._warned_missing_joint_limits = True
                return (
                    np.full(dof_count, -np.inf, dtype=float),
                    np.full(dof_count, np.inf, dtype=float),
                )

        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.size != dof_count or upper.size != dof_count:
            if not self._warned_missing_joint_limits:
                print(
                    f"[IK Bridge] Warning: Joint limit size mismatch lower={lower.size} upper={upper.size} dofs={dof_count}, using +/-inf fallback"
                )
                self._warned_missing_joint_limits = True
            return (
                np.full(dof_count, -np.inf, dtype=float),
                np.full(dof_count, np.inf, dtype=float),
            )

        return lower, upper

    def _extract_raw_link_jacobian(
        self,
        jacobians: np.ndarray,
        body_index: Optional[int],
        arm_name: str,
        link_name: str,
    ) -> Optional[np.ndarray]:
        """Normalize Isaac Jacobian output to a single link Jacobian matrix."""
        if body_index is None:
            self._warn_once(
                f"jacobian-body-missing:{arm_name}:{link_name}",
                f"[IK Bridge] Warning: Missing body index for {arm_name} jacobian link {link_name}",
            )
            return None

        jac = np.asarray(jacobians)
        body_names = list(getattr(self.articulation_view, "body_names", []) or [])
        jacobian_index = body_index
        if jac.ndim == 4 and body_names and jac.shape[1] == len(body_names) - 1 and body_index > 0:
            jacobian_index = body_index - 1
        elif jac.ndim == 3 and body_names and jac.shape[0] == len(body_names) - 1 and body_index > 0:
            jacobian_index = body_index - 1

        try:
            if jac.ndim == 4:
                link_jacobian = jac[0, jacobian_index]
            elif jac.ndim == 3:
                if jac.shape[0] == 1 and jac.shape[1] in (3, 6):
                    link_jacobian = jac[0]
                elif body_names and jac.shape[0] == len(body_names) - 1:
                    link_jacobian = jac[jacobian_index]
                elif jac.shape[1] in (3, 6):
                    link_jacobian = jac[jacobian_index]
                elif jac.shape[0] in (3, 6):
                    link_jacobian = jac
                else:
                    raise ValueError(f"unsupported 3-D jacobian shape {jac.shape}")
            elif jac.ndim == 2:
                link_jacobian = jac
            else:
                raise ValueError(f"unsupported jacobian shape {jac.shape}")
        except Exception as exc:
            if not self._warned_jacobian_shape:
                print(f"[IK Bridge] Warning: Failed to normalize Jacobian shape {jac.shape}: {exc}")
                self._warned_jacobian_shape = True
            return None

        link_jacobian = np.asarray(link_jacobian, dtype=float)
        if link_jacobian.ndim != 2:
            if not self._warned_jacobian_shape:
                print(
                    f"[IK Bridge] Warning: Normalized Jacobian for {arm_name}/{link_name} is not 2-D: {link_jacobian.shape}"
                )
                self._warned_jacobian_shape = True
            return None

        return link_jacobian

    def _extract_arm_jacobian(
        self,
        jacobians: np.ndarray,
        ee_body_index: Optional[int],
        ee_link_name: Optional[str],
        joint_indices: np.ndarray,
        arm_name: str,
    ) -> Optional[np.ndarray]:
        """Extract a 3xN or 6xN Jacobian for a specific arm."""
        tcp_spec = self._get_tcp_spec(ee_link_name)
        if tcp_spec is not None:
            position_blocks = []
            for link_name in tcp_spec["position_links"]:
                body_index = self._get_link_body_index(link_name)
                link_jacobian = self._extract_raw_link_jacobian(jacobians, body_index, arm_name, link_name)
                if link_jacobian is None or link_jacobian.shape[0] < 3:
                    return None
                position_blocks.append(link_jacobian[:3, joint_indices])

            position_jacobian = np.mean(position_blocks, axis=0)
            if not self.config.ik_enable_orientation:
                return position_jacobian

            orientation_link = tcp_spec["orientation_link"]
            orientation_body_index = self._get_link_body_index(orientation_link)
            orientation_jacobian = self._extract_raw_link_jacobian(
                jacobians,
                orientation_body_index,
                arm_name,
                orientation_link,
            )
            if orientation_jacobian is None or orientation_jacobian.shape[0] < 6:
                self._warn_once(
                    f"tcp-orientation-jacobian:{arm_name}:{orientation_link}",
                    f"[IK Bridge] Warning: Orientation Jacobian unavailable for TCP {ee_link_name} via {orientation_link}",
                )
                return None
            return np.vstack([position_jacobian, orientation_jacobian[3:6, joint_indices]])

        if ee_body_index is None:
            print(f"[IK Bridge] Warning: EE body index unavailable for {arm_name}")
            return None

        ee_jacobian = self._extract_raw_link_jacobian(
            jacobians,
            ee_body_index,
            arm_name,
            ee_link_name or arm_name,
        )
        if ee_jacobian is None:
            return None

        task_rows = 6 if self.config.ik_enable_orientation and ee_jacobian.shape[0] >= 6 else 3
        if ee_jacobian.shape[0] < task_rows:
            if not self._warned_jacobian_shape:
                print(
                    f"[IK Bridge] Warning: Jacobian rows insufficient for {arm_name}: {ee_jacobian.shape}"
                )
                self._warned_jacobian_shape = True
            return None

        return ee_jacobian[:task_rows, joint_indices]

    def _get_link_pose(self, link_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get world pose for any physical link, preferring USD Xform to avoid PhysX views."""
        position, orientation = self._get_link_pose_usd(link_name)
        if position is not None and orientation is not None:
            return position, orientation

        if self._allow_rigid_prim_pose:
            ee_rigid_prim = self._get_ee_rigid_prim(link_name)
            if ee_rigid_prim is not None:
                try:
                    positions, orientations = ee_rigid_prim.get_world_poses()
                    if positions is not None and len(positions) > 0:
                        return np.asarray(positions[0], dtype=float), np.asarray(orientations[0], dtype=float)
                except Exception as exc:
                    self._warn_once(
                        f"link-pose-rigid:{link_name}",
                        f"[IK Bridge] Warning: Failed to read world pose via RigidPrim for {link_name}: {exc}",
                    )

        body_index = self._get_link_body_index(link_name)
        if body_index is None:
            return None, None

        try:
            if hasattr(self.articulation_view, "get_body_coms"):
                body_positions, body_orientations = self.articulation_view.get_body_coms(body_indices=[body_index])
                if body_positions is not None and len(body_positions) > 0:
                    body_positions = np.asarray(body_positions, dtype=float)
                    body_orientations = np.asarray(body_orientations, dtype=float)
                    if body_positions.ndim == 3:
                        return body_positions[0, 0], body_orientations[0, 0]
                    return body_positions[0], body_orientations[0]
        except Exception as exc:
            self._warn_once(
                f"link-pose-body-com:{link_name}",
                f"[IK Bridge] Warning: Failed to read pose via body COM API for {link_name}: {exc}",
            )

        try:
            body_positions, body_orientations = self.articulation_view.get_world_poses(indices=[body_index])
            if body_positions is not None and len(body_positions) > 0:
                return np.asarray(body_positions[0], dtype=float), np.asarray(body_orientations[0], dtype=float)
        except Exception as exc:
            self._warn_once(
                f"link-pose-legacy:{link_name}",
                f"[IK Bridge] Warning: Failed legacy pose lookup for {link_name}: {exc}",
            )

        return None, None

    def _get_link_pose_usd(self, link_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get link pose from USD Xform to avoid PhysX tensor views."""
        prim_path = self._resolve_ee_prim_path(link_name)
        if not prim_path:
            return None, None

        try:
            import omni.usd
            from pxr import UsdGeom, Gf
        except Exception as exc:
            self._warn_once(
                "usd-pose-import",
                f"[IK Bridge] Warning: Failed to import USD pose helpers: {exc}",
            )
            return None, None

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return None, None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None, None

        try:
            cache = UsdGeom.XformCache()
            matrix = cache.GetLocalToWorldTransform(prim)
            transform = Gf.Transform(matrix)
            translation = transform.GetTranslation()
            rotation = transform.GetRotation()
            quat = rotation.GetQuat()
            pos = np.array([translation[0], translation[1], translation[2]], dtype=float)
            imag = quat.GetImaginary()
            quat_arr = np.array([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=float)
            return pos, quat_arr
        except Exception as exc:
            self._warn_once(
                f"usd-pose-fail:{link_name}",
                f"[IK Bridge] Warning: Failed USD pose lookup for {link_name}: {exc}",
            )
            return None, None

    def _is_base_link_frame(self, frame_id: Optional[str]) -> bool:
        if frame_id is None:
            return True
        frame = str(frame_id).strip()
        if not frame:
            return True
        if frame in {"base_link", "/base_link"}:
            return True
        return frame.endswith("/base_link")

    def _is_world_frame(self, frame_id: Optional[str]) -> bool:
        if frame_id is None:
            return False
        frame = str(frame_id).strip()
        if not frame:
            return False
        if frame in {"world", "/world"}:
            return True
        return frame.endswith("/world")

    def _resolve_target_in_world(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        frame_id: Optional[str],
        arm_label: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert target pose to world frame if provided in base_link."""
        if self._is_world_frame(frame_id):
            return target_pos, target_quat

        if not self._is_base_link_frame(frame_id):
            self._warn_once(
                f"target-frame-unknown:{frame_id}",
                f"[IK Bridge] Warning: Unrecognized target frame '{frame_id}' for {arm_label}; "
                "assuming world frame.",
            )
            return target_pos, target_quat

        base_pos, base_quat = self._get_link_pose_usd(self._base_link_name)
        if base_pos is None or base_quat is None:
            self._warn_once(
                "base-link-pose-missing",
                "[IK Bridge] Warning: base_link USD pose unavailable; cannot transform target to world.",
            )
            return target_pos, target_quat

        world_pos = np.asarray(base_pos, dtype=float) + _quat_rotate_vector(base_quat, target_pos)
        world_quat = _quat_multiply(np.asarray(base_quat, dtype=float), np.asarray(target_quat, dtype=float))
        return world_pos, world_quat

    def _get_synthetic_tcp_pose(self, ee_link_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Construct a synthetic TCP pose from gripper fingertip links."""
        tcp_spec = self._get_tcp_spec(ee_link_name)
        if tcp_spec is None:
            return None, None

        positions = []
        for link_name in tcp_spec["position_links"]:
            position, _ = self._get_link_pose(link_name)
            if position is None:
                return None, None
            positions.append(position)

        midpoint = np.mean(np.asarray(positions, dtype=float), axis=0)
        wrist_position, orientation = self._get_link_pose(tcp_spec["orientation_link"])
        if wrist_position is not None:
            tcp_forward = midpoint - wrist_position
            tcp_forward_norm = np.linalg.norm(tcp_forward)
            jaw_span = positions[1] - positions[0]
            if tcp_forward_norm > 1e-6:
                jaw_span = jaw_span - (
                    np.dot(jaw_span, tcp_forward) / (tcp_forward_norm * tcp_forward_norm)
                ) * tcp_forward
            jaw_span_norm = np.linalg.norm(jaw_span)

            if tcp_forward_norm > 1e-6 and jaw_span_norm > 1e-6:
                x_axis = tcp_forward / tcp_forward_norm
                y_axis = jaw_span / jaw_span_norm
                z_axis = np.cross(x_axis, y_axis)
                z_axis_norm = np.linalg.norm(z_axis)
                if z_axis_norm > 1e-6:
                    z_axis = z_axis / z_axis_norm
                    y_axis = np.cross(z_axis, x_axis)
                    y_axis_norm = np.linalg.norm(y_axis)
                    if y_axis_norm > 1e-6:
                        y_axis = y_axis / y_axis_norm
                        rotation = np.column_stack((x_axis, y_axis, z_axis))
                        return midpoint, _quat_from_rotation_matrix(rotation)

        if orientation is None:
            _, orientation = self._get_link_pose(tcp_spec["position_links"][0])
        if orientation is None:
            return None, None

        orientation = _normalize_quaternion(orientation)
        orientation_offset_quat = tcp_spec.get("orientation_offset_quat")
        if orientation_offset_quat is not None:
            orientation = _quat_multiply(orientation, np.asarray(orientation_offset_quat, dtype=float))

        return midpoint, orientation

    def _get_ee_pose(
        self,
        ee_body_index: Optional[int],
        ee_link_name: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get end-effector pose from articulation view.

        Args:
            ee_body_index: Body index of the end-effector

        Returns:
            Tuple of (position, quaternion) or (None, None) if failed
        """
        tcp_spec = self._get_tcp_spec(ee_link_name)
        if tcp_spec is not None:
            return self._get_synthetic_tcp_pose(ee_link_name)

        if ee_link_name is not None:
            position, orientation = self._get_link_pose(ee_link_name)
            if position is not None and orientation is not None:
                return position, orientation

        if ee_body_index is not None:
            try:
                body_positions, body_orientations = self.articulation_view.get_world_poses(indices=[ee_body_index])
                if body_positions is not None and len(body_positions) > 0:
                    return np.asarray(body_positions[0], dtype=float), np.asarray(body_orientations[0], dtype=float)
            except Exception as exc:
                self._warn_once(
                    f"ee-pose-legacy:{ee_link_name or ee_body_index}",
                    f"[IK Bridge] Warning: Failed legacy EE pose lookup for {ee_link_name or ee_body_index}: {exc}",
                )

        return None, None

    def _publish_current_pose(self, pose_pub, position: np.ndarray, orientation: np.ndarray):
        """Publish current end-effector pose if ROS2 publishers are ready."""
        if pose_pub is None or self._PoseStamped is None or self._node is None:
            return

        try:
            pose_msg = self._PoseStamped()
            pose_msg.header.stamp = self._node.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"
            pose_msg.pose.position.x = float(position[0])
            pose_msg.pose.position.y = float(position[1])
            pose_msg.pose.position.z = float(position[2])
            pose_msg.pose.orientation.w = float(orientation[0])
            pose_msg.pose.orientation.x = float(orientation[1])
            pose_msg.pose.orientation.y = float(orientation[2])
            pose_msg.pose.orientation.z = float(orientation[3])
            pose_pub.publish(pose_msg)
        except Exception as e:
            print(f"[IK Bridge] Error publishing EE pose: {e}")

    def _publish_status(self, status_pub, status_text: str):
        """Publish IK status if ROS2 publishers are ready."""
        if status_pub is None or self._String is None:
            return

        try:
            status_msg = self._String()
            status_msg.data = status_text
            status_pub.publish(status_msg)
        except Exception as e:
            print(f"[IK Bridge] Error publishing status: {e}")

    def _maybe_log_limit_hits(
        self,
        arm_controller: ArmIKController,
        status: dict,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
    ):
        """Emit rate-limited per-arm joint-limit diagnostics."""
        limit_hits = status.get("limit_hits") or []
        if not limit_hits:
            arm_controller._last_limit_signature = None
            return

        hit_labels = []
        for hit in limit_hits:
            joint_index = hit.get("index")
            joint_name = hit.get("name") or f"joint[{joint_index}]"
            hit_labels.append(f"{joint_name}(local[{joint_index}]) at {hit.get('side', 'unknown')} limit")

        signature = "|".join(hit_labels)
        now = time.time()
        if (
            signature == arm_controller._last_limit_signature
            and now - arm_controller._last_limit_log_time < 2.0
        ):
            return

        print(
            f"[IK Solver][{arm_controller.arm_name}] Joint limits reached: {', '.join(hit_labels)}; "
            f"current=({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}); "
            f"target=({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}); "
            f"pos_error={status.get('pos_error', status.get('error_norm', 0.0)):.4f}"
        )
        arm_controller._last_limit_signature = signature
        arm_controller._last_limit_log_time = now


    def _record_update_timing(self, duration_sec: float, active: bool):
        """Track IK update timing and emit periodic performance logs."""
        self._stats["update_calls"] += 1
        self._stats["last_update_duration_sec"] = duration_sec
        self._stats["update_total_time_sec"] += duration_sec
        self._stats["update_max_time_sec"] = max(
            self._stats["update_max_time_sec"], duration_sec
        )

        if active:
            self._stats["active_update_calls"] += 1
            self._stats["active_update_total_time_sec"] += duration_sec
            self._stats["active_update_max_time_sec"] = max(
                self._stats["active_update_max_time_sec"], duration_sec
            )

        duration_ms = duration_sec * 1000.0
        if active and duration_ms > self.config.ik_perf_warn_threshold_ms:
            self._stats["perf_warn_count"] += 1

        log_interval = self.config.ik_perf_log_interval_sec
        now = time.time()
        if not active or log_interval <= 0:
            return

        if now - self._stats["last_perf_log_time"] < log_interval:
            return

        active_calls = self._stats["active_update_calls"]
        active_avg_ms = (
            self._stats["active_update_total_time_sec"] / active_calls * 1000.0
            if active_calls > 0
            else 0.0
        )
        print(
            "[IK Bridge] Perf:"
            f" avg_active={active_avg_ms:.3f}ms"
            f" last={duration_ms:.3f}ms"
            f" max={self._stats['active_update_max_time_sec'] * 1000.0:.3f}ms"
            f" warn_threshold={self.config.ik_perf_warn_threshold_ms:.3f}ms"
            f" warn_count={self._stats['perf_warn_count']}"
        )
        self._stats["last_perf_log_time"] = now

    def update(self):
        """
        Update IK and publish commands.

        Should be called from the main simulation loop at ik_rate_hz.

        In 'auto' mode, IK only runs when there are active EE pose targets.
        In 'ee_pose' mode, IK always runs (publishes current EE poses even without targets).
        """
        if not self.config.is_ik_enabled():
            return

        if not self._running:
            return

        update_start = time.perf_counter()
        active_update = False

        try:
            if not self._ensure_articulation_ready():
                return
            self._ensure_joint_indices()
            self._ensure_ee_body_indices()

            has_left_target = self.left_arm.has_active_target()
            has_right_target = self.right_arm.has_active_target()

            if self.config.control_mode == "auto" and not has_left_target and not has_right_target:
                return

            try:
                try:
                    jacobians = self.articulation_view.get_jacobians()
                except AttributeError as exc:
                    if "_physics_view" in str(exc):
                        self._warn_once(
                            "articulation-jacobian-physics-view",
                            "[IK Bridge] Warning: Jacobian fetch failed due to missing _physics_view; attempting rewrap.",
                        )
                        self._rewrap_articulation_view("jacobian fetch")
                        return
                    raise
                except Exception as exc:
                    message = str(exc)
                    if "invalidated" in message or "Failed to get Jacobians" in message:
                        self._note_articulation_failure("jacobians invalidated", exc=exc, force_rewrap=True)
                        return
                    self._note_articulation_failure("jacobians exception", exc=exc)
                    return
                if jacobians is None or len(jacobians) == 0:
                    self._note_articulation_failure("jacobians empty")
                    return

                try:
                    joint_positions = self.articulation_view.get_joint_positions()
                except AttributeError as exc:
                    if "_physics_view" in str(exc):
                        self._warn_once(
                            "articulation-joint-physics-view",
                            "[IK Bridge] Warning: Joint state fetch failed due to missing _physics_view; attempting rewrap.",
                        )
                        self._rewrap_articulation_view("joint state fetch")
                        return
                    raise
                except Exception as exc:
                    message = str(exc)
                    if "invalidated" in message or "Failed to get Dof" in message:
                        self._note_articulation_failure("joint positions invalidated", exc=exc, force_rewrap=True)
                        return
                    self._note_articulation_failure("joint positions exception", exc=exc)
                    return
                if joint_positions is None or len(joint_positions) == 0:
                    self._note_articulation_failure("joint positions empty")
                    return

                self._note_articulation_success()

                joint_limits_lower, joint_limits_upper = self._extract_joint_limits(
                    joint_positions
                )

                if not np.all(np.isfinite(jacobians)):
                    self._note_articulation_failure("jacobians non-finite")
                    return

                if not np.all(np.isfinite(joint_positions)):
                    self._note_articulation_failure("joint positions non-finite")
                    return

                jacobian_full = jacobians[0]
                joint_pos_full = joint_positions[0]

                should_update_left = has_left_target or self.config.control_mode == "ee_pose"
                should_update_right = has_right_target or self.config.control_mode == "ee_pose"
                active_update = should_update_left or should_update_right

                direct_target_full = None
                if self.config.ik_direct_apply and self.articulation_view is not None:
                    direct_target_full = np.array(joint_pos_full, dtype=np.float32)
                    if direct_target_full.ndim > 1:
                        direct_target_full = direct_target_full.reshape(-1)

                if should_update_left:
                    self._update_arm(
                        self.left_arm,
                        has_left_target,
                        self._left_joint_indices,
                        self._left_ee_body_index,
                        jacobian_full,
                        joint_pos_full,
                        joint_limits_lower,
                        joint_limits_upper,
                        self._left_cmd_pub,
                        self._left_pose_pub,
                        self._left_status_pub,
                        direct_target_full,
                    )

                if should_update_right:
                    self._update_arm(
                        self.right_arm,
                        has_right_target,
                        self._right_joint_indices,
                        self._right_ee_body_index,
                        jacobian_full,
                        joint_pos_full,
                        joint_limits_lower,
                        joint_limits_upper,
                        self._right_cmd_pub,
                        self._right_pose_pub,
                        self._right_status_pub,
                        direct_target_full,
                    )

                if direct_target_full is not None and active_update:
                    try:
                        self.articulation_view.set_joint_position_targets(
                            direct_target_full.reshape(1, -1)
                        )
                    except Exception as exc:
                        print(f"[IK Bridge] Warning: Direct apply failed: {exc}")

            except IKArticulationError as e:
                print(f"[IK Bridge] Articulation error: {e}")
            except Exception as e:
                message = str(e)
                if "Failed to get Jacobians" in message or "invalidated" in message:
                    self._note_articulation_failure("backend invalidated", exc=e, force_rewrap=True)
                    return
                self._note_articulation_failure("articulation error", exc=e)
                print(f"[IK Bridge] Error getting articulation state: {e}")

        except Exception as e:
            print(f"[IK Bridge] Unexpected error in update: {e}")
        finally:
            self._record_update_timing(time.perf_counter() - update_start, active_update)

    def _update_arm(
        self,
        arm_controller: ArmIKController,
        has_active_target: bool,
        joint_indices: np.ndarray,
        ee_body_index: Optional[int],
        jacobian_full: np.ndarray,
        joint_pos_full: np.ndarray,
        joint_limits_lower: np.ndarray,
        joint_limits_upper: np.ndarray,
        cmd_pub,
        pose_pub,
        status_pub,
        direct_target_full: np.ndarray | None,
    ):
        """Update single arm IK."""
        try:
            if len(joint_indices) == 0:
                print(f"[IK Bridge] Warning: No joint indices for {arm_controller.arm_name}")
                return

            # Validate joint indices
            if np.any(joint_indices >= len(joint_pos_full)):
                print(f"[IK Bridge] Error: Invalid joint indices for {arm_controller.arm_name}")
                return

            # Extract arm-specific data
            jacobian_arm = self._extract_arm_jacobian(
                jacobian_full,
                ee_body_index,
                arm_controller.ee_link,
                joint_indices,
                arm_controller.arm_name,
            )
            if jacobian_arm is None:
                return
            joint_pos_arm = joint_pos_full[joint_indices]
            limits_lower_arm = joint_limits_lower[joint_indices]
            limits_upper_arm = joint_limits_upper[joint_indices]

            # Validate extracted data
            if not np.all(np.isfinite(jacobian_arm)):
                print(f"[IK Bridge] Warning: Jacobian contains NaN/inf for {arm_controller.arm_name}")
                return

            if not np.all(np.isfinite(joint_pos_arm)):
                print(f"[IK Bridge] Warning: Joint positions contain NaN/inf for {arm_controller.arm_name}")
                return

            # Get current end-effector pose from articulation view
            current_pos = None
            current_quat = None
            if ee_body_index is not None or self._get_tcp_spec(arm_controller.ee_link) is not None:
                current_pos, current_quat = self._get_ee_pose(ee_body_index, arm_controller.ee_link)

            # Fallback to placeholder if FK failed
            if current_pos is None or current_quat is None:
                print(f"[IK Bridge] Warning: Failed to get EE pose for {arm_controller.arm_name}, using placeholder")
                current_pos = np.array([0.5, 0.3, 0.8])
                current_quat = np.array([1.0, 0.0, 0.0, 0.0])

            self._publish_current_pose(pose_pub, current_pos, current_quat)

            # Get target pose after publishing FK feedback so ee_pose mode can
            # continuously expose the current arm state even when idle.
            target_pos, target_quat, target_frame, target_timestamp = arm_controller.get_target_pose()
            if not has_active_target:
                target_age = time.time() - target_timestamp if target_timestamp > 0 else None
                status_label = "target_timeout" if target_pos is not None else "idle"
                status_text = f"{status_label}, error=0.0000"
                if target_age is not None:
                    status_text += f", target_age={target_age:.3f}s"
                self._publish_status(status_pub, status_text)
                arm_controller.last_status = {
                    "status": status_label,
                    "pos_error": 0.0,
                    "target_age": target_age,
                }
                return

            if target_pos is None:
                self._publish_status(status_pub, "idle, error=0.0000")
                arm_controller.last_status = {"status": "idle", "pos_error": 0.0}
                return

            target_pos, target_quat = self._resolve_target_in_world(
                target_pos,
                target_quat if target_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]),
                target_frame,
                arm_controller.arm_name,
            )

            # Compute IK step
            try:
                joint_cmd, status = arm_controller.ik_solver.compute_ik_step(
                    target_pos=target_pos,
                    target_quat=target_quat if self.config.ik_enable_orientation else None,
                    current_pos=current_pos,
                    current_quat=current_quat,
                    jacobian=jacobian_arm,
                    current_joint_pos=joint_pos_arm,
                    joint_limits=(limits_lower_arm, limits_upper_arm),
                    dt=self.config.ik_dt,
                    debug_label=arm_controller.arm_name,
                    joint_names=arm_controller.joint_names,
                )

                # Validate IK output
                if not np.all(np.isfinite(joint_cmd)):
                    print(f"[IK Bridge] Warning: IK output contains NaN/inf for {arm_controller.arm_name}")
                    return

            except IKException as e:
                print(f"[IK Bridge] IK computation failed for {arm_controller.arm_name}: {e}")
                return
            except Exception as e:
                print(f"[IK Bridge] Unexpected error in IK computation for {arm_controller.arm_name}: {e}")
                return

            if direct_target_full is not None:
                try:
                    direct_target_full[joint_indices] = joint_cmd
                except Exception as exc:
                    print(
                        f"[IK Bridge] Warning: Failed to update direct target buffer for "
                        f"{arm_controller.arm_name}: {exc}"
                    )

            # Publish joint command
            if cmd_pub is not None and self._JointState is not None:
                try:
                    msg = self._JointState()
                    msg.header.stamp = self._node.get_clock().now().to_msg()
                    msg.name = arm_controller.joint_names
                    msg.position = joint_cmd.tolist()
                    cmd_pub.publish(msg)
                except Exception as e:
                    print(f"[IK Bridge] Error publishing joint command for {arm_controller.arm_name}: {e}")

            self._maybe_log_limit_hits(arm_controller, status, current_pos, target_pos)
            status_text = (
                f"{status.get('status', 'unknown')}, "
                f"error={status.get('pos_error', status.get('error_norm', 0.0)):.4f}"
            )
            error_message = status.get("error_message")
            if error_message:
                status_text += f", msg={error_message}"
            limit_hits = status.get("limit_hits") or []
            if limit_hits:
                hit_labels = []
                for hit in limit_hits:
                    joint_index = hit.get("index")
                    joint_name = hit.get("name")
                    side = hit.get("side", "unknown")
                    if not joint_name and isinstance(joint_index, int) and 0 <= joint_index < len(arm_controller.joint_names):
                        joint_name = arm_controller.joint_names[joint_index]
                    if not joint_name:
                        joint_name = f"joint{joint_index}"
                    hit_labels.append(f"{joint_name}:{side}")
                status_text += f", limits={len(limit_hits)}[{';'.join(hit_labels)}]"
            self._publish_status(status_pub, status_text)
            arm_controller.last_status = status

            # Update statistics
            if arm_controller == self.left_arm:
                self._stats["left_ik_updates"] += 1
                self._stats["last_left_ik_time"] = time.time()
            else:
                self._stats["right_ik_updates"] += 1
                self._stats["last_right_ik_time"] = time.time()

        except Exception as e:
            print(f"[IK Bridge] Unexpected error in _update_arm for {arm_controller.arm_name}: {e}")

    def get_control_status(self):
        """
        Get current control status for monitoring.

        Returns:
            dict: Status information including active targets, per-arm state,
            and aggregated IK performance metrics.
        """
        current_time = time.time()

        update_calls = self._stats["update_calls"]
        active_update_calls = self._stats["active_update_calls"]
        avg_update_ms = (
            self._stats["update_total_time_sec"] / update_calls * 1000.0
            if update_calls > 0
            else None
        )
        avg_active_update_ms = (
            self._stats["active_update_total_time_sec"] / active_update_calls * 1000.0
            if active_update_calls > 0
            else None
        )

        return {
            "control_mode": self.config.control_mode,
            "left_arm": {
                "has_active_target": self.left_arm.has_active_target(),
                "ik_updates": self._stats["left_ik_updates"],
                "last_target_age": current_time - self._stats["last_left_target_time"]
                if self._stats["last_left_target_time"] > 0
                else None,
                "last_ik_age": current_time - self._stats["last_left_ik_time"]
                if self._stats["last_left_ik_time"] > 0
                else None,
                "last_status": self.left_arm.last_status.get("status")
                if self.left_arm.last_status
                else None,
            },
            "right_arm": {
                "has_active_target": self.right_arm.has_active_target(),
                "ik_updates": self._stats["right_ik_updates"],
                "last_target_age": current_time - self._stats["last_right_target_time"]
                if self._stats["last_right_target_time"] > 0
                else None,
                "last_ik_age": current_time - self._stats["last_right_ik_time"]
                if self._stats["last_right_ik_time"] > 0
                else None,
                "last_status": self.right_arm.last_status.get("status")
                if self.right_arm.last_status
                else None,
            },
            "performance": {
                "update_calls": update_calls,
                "active_update_calls": active_update_calls,
                "last_update_ms": self._stats["last_update_duration_sec"] * 1000.0,
                "avg_update_ms": avg_update_ms,
                "max_update_ms": self._stats["update_max_time_sec"] * 1000.0,
                "avg_active_update_ms": avg_active_update_ms,
                "max_active_update_ms": self._stats["active_update_max_time_sec"] * 1000.0,
                "warn_threshold_ms": self.config.ik_perf_warn_threshold_ms,
                "warn_count": self._stats["perf_warn_count"],
            },
        }
