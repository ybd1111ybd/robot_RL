"""Builtin IK bridge using Isaac Sim motion_generation (Lula) solvers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from ros2_singleton import ProcessSingletonLock

try:
    from isaacsim.robot_motion.motion_generation import (
        ArticulationKinematicsSolver,
        LulaKinematicsSolver,
    )

    _HAS_MOTION_GENERATION = True
except Exception:
    try:
        from omni.isaac.motion_generation import (
            ArticulationKinematicsSolver,
            LulaKinematicsSolver,
        )

        _HAS_MOTION_GENERATION = True
    except Exception:
        ArticulationKinematicsSolver = None  # type: ignore[assignment]
        LulaKinematicsSolver = None  # type: ignore[assignment]
        _HAS_MOTION_GENERATION = False


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize a quaternion in [w, x, y, z] order."""
    q = np.asarray(quat, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


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


def _quat_rotate_vector(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a 3D vector by a quaternion (w, x, y, z)."""
    q = _normalize_quaternion(quat)
    v = np.asarray(vec, dtype=float).reshape(3)
    q_vec = q[1:4]
    q_w = float(q[0])
    return (
        2.0 * np.dot(q_vec, v) * q_vec
        + (q_w * q_w - np.dot(q_vec, q_vec)) * v
        + 2.0 * q_w * np.cross(q_vec, v)
    )


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    q = _normalize_quaternion(quat)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def _quat_angle(lhs: np.ndarray, rhs: np.ndarray) -> float:
    """Return absolute angle between two [w, x, y, z] quaternions."""
    l = _normalize_quaternion(lhs)
    r = _normalize_quaternion(rhs)
    dot = float(np.dot(l, r))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * float(np.arccos(dot))


def _quat_slerp(lhs: np.ndarray, rhs: np.ndarray, t: float) -> np.ndarray:
    """Slerp for [w, x, y, z] quaternions with shortest-path handling."""
    q0 = _normalize_quaternion(lhs)
    q1 = _normalize_quaternion(rhs)
    tt = float(max(0.0, min(1.0, t)))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))

    if dot > 0.9995:
        out = q0 + tt * (q1 - q0)
        return _normalize_quaternion(out)

    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 < 1e-8:
        return q0
    theta = theta_0 * tt
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    out = s0 * q0 + s1 * q1
    return _normalize_quaternion(out)


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


def _quat_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
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


def _detect_repo_root() -> Optional[Path]:
    env_root = os.environ.get("JZ_REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "jz_descripetion").exists():
            return candidate

    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "jz_descripetion").exists():
            return parent
    return None


def _default_urdf_path() -> Optional[Path]:
    repo_root = _detect_repo_root()
    if repo_root is None:
        return None
    candidate = repo_root / "jz_descripetion/robot_urdf/urdf/robot_model_isaac_v2.urdf"
    if candidate.exists():
        return candidate
    return None


def _apply_suffix(topic: str, suffix: str) -> str:
    if not suffix:
        return topic
    if topic.endswith(suffix):
        return topic
    return f"{topic}{suffix}"


GRIPPER_MOUNT_QUAT = _quat_from_rpy(-2.9951, -1.5708, -0.15964)
GRIPPER_TCP_SOLVER_OFFSET_LOCAL = np.array([0.0, 0.0, 0.10547], dtype=float)

SYNTHETIC_TCP_SPECS = {
    "left_gripper_center_tcp": {
        "position_links": ("left_gripper_narrow3_link", "left_gripper_wide3_link"),
        "orientation_link": "left_arm_link9",
        "orientation_offset_quat": GRIPPER_MOUNT_QUAT,
        "solver_offset_local": GRIPPER_TCP_SOLVER_OFFSET_LOCAL,
    },
    "right_gripper_center_tcp": {
        "position_links": ("right_gripper_narrow3_link", "right_gripper_wide3_link"),
        "orientation_link": "right_arm_link9",
        "orientation_offset_quat": GRIPPER_MOUNT_QUAT,
        "solver_offset_local": GRIPPER_TCP_SOLVER_OFFSET_LOCAL,
    },
}


@dataclass
class BuiltinIKConfig:
    """Configuration for builtin Lula IK bridge."""

    left_ee_topic: str = "/arm_left/ee_target_pose"
    right_ee_topic: str = "/arm_right/ee_target_pose"

    left_cmd_topic: str = "/arm_left/joint_commands"
    right_cmd_topic: str = "/arm_right/joint_commands"
    left_pose_topic: str = "/arm_left/ee_current_pose"
    right_pose_topic: str = "/arm_right/ee_current_pose"
    left_status_topic: str = "/arm_left/ee_ik_status"
    right_status_topic: str = "/arm_right/ee_ik_status"

    left_ee_link: str = "left_gripper_center_tcp"
    right_ee_link: str = "right_gripper_center_tcp"
    left_solver_base_link: str = "body_link3"
    right_solver_base_link: str = "body_link3"

    left_arm_joints: Optional[List[str]] = None
    right_arm_joints: Optional[List[str]] = None

    urdf_path: str = ""
    robot_description_path: str = ""
    left_robot_description_path: str = ""
    right_robot_description_path: str = ""

    cmd_topic_suffix: str = ""
    ik_timeout_sec: float = 1.0
    enable_orientation: bool = False
    control_mode: str = "ee_pose"
    feedback_frame_id: str = "base_link"
    enable_target_smoothing: bool = True
    target_smoothing_alpha: float = 0.35
    max_target_position_step_m: float = 0.02
    max_target_orientation_step_rad: float = 0.20
    direct_apply_joint_targets: bool = True
    max_joint_delta_rad: float = 0.15
    max_joint_position_abs_rad: float = 6.0
    orientation_position_gate_m: float = 0.06
    approx_orientation_accept_position_error_m: float = 0.03
    close_range_orientation_distance_m: float = 0.03
    close_range_max_orientation_step_rad: float = 0.08

    def __post_init__(self) -> None:
        if self.left_arm_joints is None:
            self.left_arm_joints = [f"left_arm_joint{i}" for i in range(1, 8)]
        if self.right_arm_joints is None:
            self.right_arm_joints = [f"right_arm_joint{i}" for i in range(1, 8)]

        if not self.urdf_path:
            default_urdf = _default_urdf_path()
            if default_urdf is not None:
                self.urdf_path = str(default_urdf)

        if not self.left_robot_description_path:
            self.left_robot_description_path = self.robot_description_path
        if not self.right_robot_description_path:
            self.right_robot_description_path = self.robot_description_path

        if self.cmd_topic_suffix:
            self.left_cmd_topic = _apply_suffix(self.left_cmd_topic, self.cmd_topic_suffix)
            self.right_cmd_topic = _apply_suffix(self.right_cmd_topic, self.cmd_topic_suffix)
            self.left_pose_topic = _apply_suffix(self.left_pose_topic, self.cmd_topic_suffix)
            self.right_pose_topic = _apply_suffix(self.right_pose_topic, self.cmd_topic_suffix)
            self.left_status_topic = _apply_suffix(self.left_status_topic, self.cmd_topic_suffix)
            self.right_status_topic = _apply_suffix(self.right_status_topic, self.cmd_topic_suffix)

        mode = (self.control_mode or "").strip().lower()
        if mode not in {"joint", "ee_pose", "auto"}:
            self.control_mode = "ee_pose"
        else:
            self.control_mode = mode

        self.target_smoothing_alpha = float(max(0.0, min(1.0, self.target_smoothing_alpha)))
        self.max_target_position_step_m = float(max(0.0, self.max_target_position_step_m))
        self.max_target_orientation_step_rad = float(max(0.0, self.max_target_orientation_step_rad))
        self.max_joint_delta_rad = float(max(0.0, self.max_joint_delta_rad))
        self.max_joint_position_abs_rad = float(max(0.0, self.max_joint_position_abs_rad))
        self.orientation_position_gate_m = float(max(0.0, self.orientation_position_gate_m))
        self.approx_orientation_accept_position_error_m = float(
            max(0.0, self.approx_orientation_accept_position_error_m)
        )
        self.close_range_orientation_distance_m = float(
            max(0.0, self.close_range_orientation_distance_m)
        )
        self.close_range_max_orientation_step_rad = float(
            max(0.0, self.close_range_max_orientation_step_rad)
        )


class _TargetState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._position: Optional[np.ndarray] = None
        self._orientation: Optional[np.ndarray] = None
        self._frame_id: str = ""
        self._stamp: float = 0.0

    def set(self, position: np.ndarray, orientation: np.ndarray, frame_id: str) -> None:
        with self._lock:
            self._position = np.array(position, dtype=float)
            self._orientation = np.array(orientation, dtype=float)
            self._frame_id = frame_id or ""
            self._stamp = time.time()

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, float]:
        with self._lock:
            position = None if self._position is None else self._position.copy()
            orientation = None if self._orientation is None else self._orientation.copy()
            return position, orientation, self._frame_id, self._stamp

    def is_active(self, timeout_sec: float) -> bool:
        with self._lock:
            if self._position is None:
                return False
            if timeout_sec <= 0:
                return True
            return (time.time() - self._stamp) <= timeout_sec


class _ArticulationAdapter:
    """Adapter to provide SingleArticulation-like API on top of Articulation view."""

    def __init__(self, articulation) -> None:
        self._articulation = articulation

    def __getattr__(self, name: str):
        return getattr(self._articulation, name)

    @property
    def handles_initialized(self) -> bool:
        if hasattr(self._articulation, "handles_initialized"):
            return bool(self._articulation.handles_initialized)
        if hasattr(self._articulation, "is_physics_handle_valid"):
            try:
                return bool(self._articulation.is_physics_handle_valid())
            except Exception:
                return False
        return True

    @property
    def num_dof(self) -> int:
        return int(getattr(self._articulation, "num_dof", 0) or 0)

    def get_joint_positions(self):
        positions = self._articulation.get_joint_positions()
        if positions is None:
            return None
        arr = np.asarray(positions)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def get_joint_velocities(self):
        velocities = self._articulation.get_joint_velocities()
        if velocities is None:
            return None
        arr = np.asarray(velocities)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr

    def get_joint_efforts(self):
        efforts = self._articulation.get_joint_efforts()
        if efforts is None:
            return None
        arr = np.asarray(efforts)
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0]
        return arr


class BuiltinIKBridge:
    """ROS2 bridge for Lula-based IK using ArticulationKinematicsSolver."""

    def __init__(self, config: BuiltinIKConfig, articulation) -> None:
        self.config = config
        self.articulation = articulation
        self._repo_root = _detect_repo_root()

        self._left_target = _TargetState()
        self._right_target = _TargetState()

        self._left_lula_solver = None
        self._right_lula_solver = None
        self._left_solver = None
        self._right_solver = None
        self._lula_articulation = _ArticulationAdapter(articulation)

        self._dof_name_to_index: Dict[str, int] = {}

        self._initialized = False
        self._init_failed = False

        self._thread: Optional[threading.Thread] = None
        self._executor = None
        self._node = None
        self._rclpy = None
        self._running = False
        self._owns_rclpy_context = False
        self._singleton_lock: Optional[ProcessSingletonLock] = None
        self._duplicate_suppressed = False

        self._PoseStamped = None
        self._JointState = None
        self._String = None

        self._left_cmd_pub = None
        self._right_cmd_pub = None
        self._left_pose_pub = None
        self._right_pose_pub = None
        self._left_status_pub = None
        self._right_status_pub = None

        self._warned_missing_joints = False
        self._warned_solver = False
        self._warned_unknown_frame = False
        self._warned_base_pose = False
        self._warned_feedback_base_pose = False
        self._warned_cached_base_pose = False
        self._warned_direct_apply = False
        self._warned_solver_pose_fallback = False
        self._warned_joint_delta_clip = False
        self._warned_joint_abs_clip = False
        self._logged_ik_fail_debug: Dict[str, bool] = {}
        self._logged_ik_approx_debug: Dict[str, bool] = {}
        self._logged_orientation_position_gate: Dict[str, bool] = {}
        self._logged_orientation_position_fallback: Dict[str, bool] = {}
        self._left_tcp_spec = SYNTHETIC_TCP_SPECS.get(self.config.left_ee_link)
        self._right_tcp_spec = SYNTHETIC_TCP_SPECS.get(self.config.right_ee_link)
        self._left_solver_frame = (
            self._left_tcp_spec["orientation_link"]
            if self._left_tcp_spec
            else self.config.left_ee_link
        )
        self._right_solver_frame = (
            self._right_tcp_spec["orientation_link"]
            if self._right_tcp_spec
            else self.config.right_ee_link
        )
        self._prim_path_cache: Dict[str, Optional[str]] = {}
        self._smoothed_targets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._tcp_offset_local_cache: Dict[str, Optional[np.ndarray]] = {
            "left": None,
            "right": None,
        }
        self._logged_tcp_offset_cache: Dict[str, bool] = {}
        self._last_current_pose_world: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_robot_base_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._logged_link_resolution_failures: Dict[str, bool] = {}
        self._cached_base_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_base_pose_stamp: float = 0.0
        self._orientation_mode_hysteresis: Dict[str, str] = {"left": "full", "right": "full"}

    def set_target_pose(
        self,
        side: str,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        frame_id: str = "base_link",
    ) -> None:
        side_key = (side or "").strip().lower()
        if side_key not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side '{side}'")
        target = self._left_target if side_key == "left" else self._right_target
        quat = (
            np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            if orientation is None
            else _normalize_quaternion(np.asarray(orientation, dtype=float))
        )
        target.set(np.asarray(position, dtype=float), quat, frame_id or "")

    def get_robot_base_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pose = self._get_robot_base_pose()
        if pose is None:
            return None
        position, orientation = pose
        return np.asarray(position, dtype=float), _normalize_quaternion(orientation)

    def get_current_ee_pose(
        self, side: str, frame_id: str = "base_link"
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        side_key = (side or "").strip().lower()
        if side_key not in {"left", "right"}:
            raise ValueError(f"Unsupported arm side '{side}'")
        world_pos, world_quat = self._get_current_ee_pose_world(side_key)
        if world_pos is None or world_quat is None:
            cached = self._last_current_pose_world.get(side_key)
            if cached is None:
                return None
            world_pos, world_quat = cached
        else:
            self._last_current_pose_world[side_key] = (
                np.asarray(world_pos, dtype=float).copy(),
                _normalize_quaternion(np.asarray(world_quat, dtype=float)),
            )
        if world_pos is None or world_quat is None:
            return None
        if self._is_world_frame(frame_id):
            return np.asarray(world_pos, dtype=float), _normalize_quaternion(world_quat)
        local_pos, local_quat, _ = self._world_to_feedback_frame(world_pos, world_quat)
        return np.asarray(local_pos, dtype=float), _normalize_quaternion(local_quat)

    def _resolve_path(self, raw_path: str) -> Optional[Path]:
        if not raw_path:
            return None
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        if self._repo_root is not None:
            repo_candidate = (self._repo_root / candidate).resolve()
            if repo_candidate.exists():
                return repo_candidate
        return (Path.cwd() / candidate).resolve()

    def _get_articulation_root_path(self) -> Optional[str]:
        candidates = []
        for attr_name in ("prim_paths", "_prim_paths", "prim_paths_expr"):
            value = getattr(self.articulation, attr_name, None)
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

    @staticmethod
    def _stage_prim_is_valid(stage, prim_path: Optional[str]) -> bool:
        if stage is None or not prim_path:
            return False
        try:
            prim = stage.GetPrimAtPath(prim_path)
            return bool(prim and prim.IsValid())
        except Exception:
            return False

    def _find_link_prim_in_stage(
        self, stage, link_name: str, articulation_root: str
    ) -> Optional[str]:
        if stage is None or not link_name:
            return None

        articulation_root = articulation_root.rstrip("/")
        preferred_prefixes = [articulation_root]
        if "/" in articulation_root:
            preferred_prefixes.append(articulation_root.rsplit("/", 1)[0])

        best_match = None
        fallback_match = None
        try:
            for prim in stage.Traverse():
                if not prim or not prim.IsValid():
                    continue
                prim_name = str(prim.GetName())
                if prim_name != link_name:
                    continue
                prim_path = str(prim.GetPath())
                if "/visuals/" in prim_path or "/colliders/" in prim_path:
                    continue
                if prim_path.endswith(f"/{link_name}"):
                    fallback_match = prim_path
                    if any(
                        prim_path == prefix or prim_path.startswith(f"{prefix}/")
                        for prefix in preferred_prefixes
                        if prefix
                    ):
                        best_match = prim_path
                        break
            return best_match or fallback_match
        except Exception:
            return None

    def _resolve_link_prim_path(self, link_name: str) -> Optional[str]:
        cached = self._prim_path_cache.get(link_name)
        if cached is not None:
            try:
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            except Exception:
                stage = None
            if self._stage_prim_is_valid(stage, cached):
                return cached
            self._prim_path_cache.pop(link_name, None)

        articulation_root = self._get_articulation_root_path()
        if not articulation_root:
            return None

        body_names = list(getattr(self.articulation, "body_names", []) or [])
        candidates = []

        if articulation_root.endswith(f"/{link_name}"):
            candidates.append(articulation_root)

        if body_names:
            root_body_name = str(body_names[0])
            if articulation_root.endswith(f"/{root_body_name}"):
                parent = articulation_root.rsplit("/", 1)[0]
                candidates.append(f"{parent}/{link_name}")

        candidates.append(f"{articulation_root.rstrip('/')}/{link_name}")

        deduped = []
        seen = set()
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                deduped.append(candidate)

        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                for candidate in deduped:
                    prim = stage.GetPrimAtPath(candidate)
                    if prim and prim.IsValid():
                        self._prim_path_cache[link_name] = candidate
                        return candidate
                searched = self._find_link_prim_in_stage(stage, link_name, articulation_root)
                if searched is not None:
                    self._prim_path_cache[link_name] = searched
                    return searched
        except Exception:
            pass

        if not self._logged_link_resolution_failures.get(link_name):
            print(
                "[builtin-ik] Failed to resolve link prim path:"
                f" link={link_name}"
                f" articulation_root={articulation_root}"
                f" candidates={deduped}"
            )
            self._logged_link_resolution_failures[link_name] = True
        return None

    def _get_link_pose_usd(self, link_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        prim_path = self._resolve_link_prim_path(link_name)
        if not prim_path:
            return None, None
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
        except Exception:
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
            return pos, _normalize_quaternion(quat_arr)
        except Exception:
            return None, None

    def _get_tcp_offset(
        self, tcp_spec: Dict[str, object]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        position_links = tcp_spec.get("position_links")
        orientation_link = tcp_spec.get("orientation_link")
        if not position_links or not orientation_link:
            return None, None, None

        positions = []
        for link_name in position_links:
            pos, _ = self._get_link_pose_usd(str(link_name))
            if pos is None:
                return None, None, None
            positions.append(pos)
        midpoint = np.mean(np.asarray(positions, dtype=float), axis=0)

        ref_pos, ref_quat = self._get_link_pose_usd(str(orientation_link))
        if ref_pos is None:
            return None, None, None

        offset_world = midpoint - ref_pos
        ref_quat = _normalize_quaternion(ref_quat) if ref_quat is not None else None
        offset_local = None
        if ref_quat is not None:
            offset_local = _quat_rotate_vector(_quat_conjugate(ref_quat), offset_world)
        return offset_world, offset_local, ref_quat

    def _get_tcp_midpoint_world(self, tcp_spec: Dict[str, object]) -> Optional[np.ndarray]:
        position_links = tcp_spec.get("position_links")
        if not position_links:
            return None

        positions = []
        for link_name in position_links:
            pos, _ = self._get_link_pose_usd(str(link_name))
            if pos is None:
                return None
            positions.append(np.asarray(pos, dtype=float))
        if not positions:
            return None
        return np.mean(np.asarray(positions, dtype=float), axis=0)

    def _get_cached_tcp_offset_local(self, side: str) -> Optional[np.ndarray]:
        side_key = (side or "").strip().lower()
        if side_key not in {"left", "right"}:
            return None
        tcp_spec = self._left_tcp_spec if side_key == "left" else self._right_tcp_spec
        if tcp_spec is None:
            return None
        _offset_world, offset_local, _ref_quat = self._get_tcp_offset(tcp_spec)
        if offset_local is not None:
            offset = np.asarray(offset_local, dtype=float).reshape(3)
            self._tcp_offset_local_cache[side_key] = offset.copy()
            if not self._logged_tcp_offset_cache.get(side_key):
                print(
                    "[builtin-ik] Using live TCP local offset:"
                    f" side={side_key}"
                    f" offset={np.array2string(offset, precision=4)}"
                )
                self._logged_tcp_offset_cache[side_key] = True
            return offset

        cached = self._tcp_offset_local_cache.get(side_key)
        if cached is not None:
            return np.asarray(cached, dtype=float)

        solver_offset = tcp_spec.get("solver_offset_local")
        if solver_offset is not None:
            offset = np.asarray(solver_offset, dtype=float).reshape(3)
            self._tcp_offset_local_cache[side_key] = offset.copy()
            if not self._logged_tcp_offset_cache.get(side_key):
                print(
                    "[builtin-ik] Using fallback solver TCP local offset:"
                    f" side={side_key}"
                    f" offset={np.array2string(offset, precision=4)}"
                )
                self._logged_tcp_offset_cache[side_key] = True
            return offset
        return None

    def _initialize_solvers(self) -> bool:
        if self._initialized:
            return True
        if self._init_failed:
            return False
        if not _HAS_MOTION_GENERATION:
            if not self._warned_solver:
                print("[builtin-ik] motion_generation (Lula) is not available.")
                self._warned_solver = True
            self._init_failed = True
            return False

        urdf_path = self._resolve_path(self.config.urdf_path)
        if urdf_path is None or not urdf_path.exists():
            print(f"[builtin-ik] URDF path not found: {self.config.urdf_path}")
            self._init_failed = True
            return False

        left_desc = self._resolve_path(self.config.left_robot_description_path)
        right_desc = self._resolve_path(self.config.right_robot_description_path)
        if left_desc is None or not left_desc.exists():
            print(
                "[builtin-ik] Left robot description not found:"
                f" {self.config.left_robot_description_path}"
            )
            self._init_failed = True
            return False
        if right_desc is None or not right_desc.exists():
            print(
                "[builtin-ik] Right robot description not found:"
                f" {self.config.right_robot_description_path}"
            )
            self._init_failed = True
            return False

        try:
            self._left_lula_solver = LulaKinematicsSolver(
                robot_description_path=str(left_desc),
                urdf_path=str(urdf_path),
            )
            self._right_lula_solver = LulaKinematicsSolver(
                robot_description_path=str(right_desc),
                urdf_path=str(urdf_path),
            )
            left_frames = set(self._left_lula_solver.get_all_frame_names())
            right_frames = set(self._right_lula_solver.get_all_frame_names())
            if self._left_solver_frame not in left_frames:
                print(
                    f"[builtin-ik] Left solver frame '{self._left_solver_frame}' not in URDF frames."
                )
            if self._right_solver_frame not in right_frames:
                print(
                    f"[builtin-ik] Right solver frame '{self._right_solver_frame}' not in URDF frames."
                )

            self._left_solver = ArticulationKinematicsSolver(
                self._lula_articulation, self._left_lula_solver, self._left_solver_frame
            )
            self._right_solver = ArticulationKinematicsSolver(
                self._lula_articulation, self._right_lula_solver, self._right_solver_frame
            )
        except Exception as exc:
            # Solver construction can race articulation/USD readiness at startup.
            # Keep startup warm-up, but allow later update cycles to retry.
            print(f"[builtin-ik] Failed to initialize Lula IK solvers (will retry): {exc}")
            return False

        self._cache_joint_indices()
        self._initialized = True
        return True

    def _cache_joint_indices(self) -> None:
        dof_names = list(getattr(self.articulation, "dof_names", []) or [])
        if not dof_names:
            return
        self._dof_name_to_index = {name: idx for idx, name in enumerate(dof_names)}
        missing = []
        for name in (self.config.left_arm_joints or []) + (self.config.right_arm_joints or []):
            if name not in self._dof_name_to_index:
                missing.append(name)
        if missing and not self._warned_missing_joints:
            print(f"[builtin-ik] Missing DOF names in articulation: {missing}")
            self._warned_missing_joints = True

    def _get_link_pose_world(self, link_name: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pos, quat = self._get_link_pose_usd(link_name)
        if pos is not None and quat is not None:
            return np.asarray(pos, dtype=float), _normalize_quaternion(quat)
        return None

    def _get_robot_base_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        now = time.time()
        if self._cached_base_pose is not None and (now - self._cached_base_pose_stamp) < 0.05:
            return self._cached_base_pose

        pose = self._get_link_pose_world("base_link")
        if pose is not None:
            pos, quat = pose
            result = (
                np.asarray(pos, dtype=float).copy(),
                _normalize_quaternion(np.asarray(quat, dtype=float)),
            )
            self._last_robot_base_pose = result
            self._cached_base_pose = result
            self._cached_base_pose_stamp = now
            return result

        if self._last_robot_base_pose is not None:
            if not self._warned_cached_base_pose:
                print(
                    "[builtin-ik] base_link pose temporarily unavailable;"
                    " reusing last resolved USD pose."
                )
                self._warned_cached_base_pose = True
            pos, quat = self._last_robot_base_pose
            return np.asarray(pos, dtype=float), _normalize_quaternion(quat)

        return None

    def _get_solver_base_pose(self, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        side_key = (side or "").strip().lower()
        if side_key == "left":
            solver_base_link = (self.config.left_solver_base_link or "").strip()
        elif side_key == "right":
            solver_base_link = (self.config.right_solver_base_link or "").strip()
        else:
            solver_base_link = ""
        if solver_base_link:
            pose = self._get_link_pose_world(solver_base_link)
            if pose is not None:
                return pose
        return self._get_robot_base_pose()

    def _set_solver_base_pose(self, solver, side: str) -> None:
        # Lula targets are expressed in the USD world frame, but the solver still
        # expects its internal robot base pose to match the root_link declared
        # in the robot description. Our arm-only IK models root at body_link3.
        pose = self._get_solver_base_pose(side)
        if pose is None:
            return
        position, orientation = pose
        try:
            solver.set_robot_base_pose(position, orientation)
        except Exception:
            return

    @staticmethod
    def _is_world_frame(frame_id: str) -> bool:
        frame = (frame_id or "").strip()
        if not frame:
            return False
        if frame in {"world", "/world"}:
            return True
        return frame.endswith("/world")

    @staticmethod
    def _is_base_frame(frame_id: str) -> bool:
        frame = (frame_id or "").strip()
        if not frame:
            return True
        if frame in {"base_link", "/base_link"}:
            return True
        return frame.endswith("/base_link")

    def _resolve_target_in_world(
        self, position: np.ndarray, orientation: Optional[np.ndarray], frame_id: str
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        if self._is_world_frame(frame_id):
            return position, orientation
        if not self._is_base_frame(frame_id):
            if not self._warned_unknown_frame:
                print(
                    f"[builtin-ik] Unknown target frame '{frame_id}', assuming world frame."
                )
                self._warned_unknown_frame = True
            return position, orientation

        base_pose = self._get_robot_base_pose()
        if base_pose is None:
            if not self._warned_base_pose:
                print(
                    "[builtin-ik] base_link pose unavailable;"
                    " deferring base_link target until the USD pose resolves."
                )
                self._warned_base_pose = True
            return None
        base_pos, base_quat = base_pose
        world_pos = np.asarray(base_pos, dtype=float) + _quat_rotate_vector(
            np.asarray(base_quat, dtype=float), position
        )
        if orientation is None:
            return world_pos, None
        world_quat = _quat_multiply(
            np.asarray(base_quat, dtype=float), np.asarray(orientation, dtype=float)
        )
        return world_pos, world_quat

    def _world_to_feedback_frame(
        self, world_pos: np.ndarray, world_quat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        frame = (self.config.feedback_frame_id or "").strip()
        if not frame or self._is_world_frame(frame):
            return np.asarray(world_pos, dtype=float), _normalize_quaternion(world_quat), "world"
        if not self._is_base_frame(frame):
            return np.asarray(world_pos, dtype=float), _normalize_quaternion(world_quat), "world"

        base_pose = self._get_robot_base_pose()
        if base_pose is None:
            if not self._warned_feedback_base_pose:
                print(
                    "[builtin-ik] base_link pose unavailable; publishing ee_current_pose in world frame."
                )
                self._warned_feedback_base_pose = True
            return np.asarray(world_pos, dtype=float), _normalize_quaternion(world_quat), "world"

        base_pos, base_quat = base_pose
        base_conj = _quat_conjugate(np.asarray(base_quat, dtype=float))
        local_pos = _quat_rotate_vector(base_conj, np.asarray(world_pos, dtype=float) - np.asarray(base_pos, dtype=float))
        local_quat = _quat_multiply(base_conj, np.asarray(world_quat, dtype=float))
        return local_pos, local_quat, "base_link"

    def _get_current_ee_pose_world(self, side: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if side == "left":
            tcp_spec = self._left_tcp_spec
            fallback_link = self._left_solver_frame
            solver = self._left_solver
        else:
            tcp_spec = self._right_tcp_spec
            fallback_link = self._right_solver_frame
            solver = self._right_solver

        if tcp_spec is not None:
            orientation_link = tcp_spec.get("orientation_link")
            ref_pos = None
            ref_quat = None
            if orientation_link:
                ref_pos, ref_quat = self._get_link_pose_usd(str(orientation_link))
            midpoint = self._get_tcp_midpoint_world(tcp_spec)
            if midpoint is not None and ref_quat is not None:
                ref_q = _normalize_quaternion(ref_quat)
                ee_pos = np.asarray(midpoint, dtype=float)
                offset_quat = tcp_spec.get("orientation_offset_quat")
                if offset_quat is not None:
                    ee_quat = _quat_multiply(
                        ref_q, np.asarray(offset_quat, dtype=float)
                    )
                else:
                    ee_quat = ref_q
                return ee_pos, ee_quat
            return None, None

        pos, quat = self._get_link_pose_usd(fallback_link)
        if pos is None or quat is None:
            if solver is None:
                return None, None
            try:
                kin_solver = solver.get_kinematics_solver() if hasattr(solver, "get_kinematics_solver") else solver
                self._set_solver_base_pose(kin_solver, side)
                solver_pos, solver_rot = solver.compute_end_effector_pose(position_only=False)
                if not self._warned_solver_pose_fallback:
                    print(
                        "[builtin-ik] Falling back to solver FK for ee_current_pose;"
                        f" side={side} frame={fallback_link}"
                    )
                    self._warned_solver_pose_fallback = True
                return (
                    np.asarray(solver_pos, dtype=float),
                    _quat_from_rotation_matrix(np.asarray(solver_rot, dtype=float)),
                )
            except Exception:
                return None, None
        return np.asarray(pos, dtype=float), _normalize_quaternion(quat)

    def _publish_current_pose(self, pose_pub, world_pos: np.ndarray, world_quat: np.ndarray) -> None:
        if pose_pub is None or self._PoseStamped is None or self._node is None:
            return
        try:
            local_pos, local_quat, frame_id = self._world_to_feedback_frame(world_pos, world_quat)
            msg = self._PoseStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = frame_id
            msg.pose.position.x = float(local_pos[0])
            msg.pose.position.y = float(local_pos[1])
            msg.pose.position.z = float(local_pos[2])
            msg.pose.orientation.w = float(local_quat[0])
            msg.pose.orientation.x = float(local_quat[1])
            msg.pose.orientation.y = float(local_quat[2])
            msg.pose.orientation.z = float(local_quat[3])
            pose_pub.publish(msg)
        except Exception as exc:
            print(f"[builtin-ik] Failed to publish current EE pose: {exc}")

    def _publish_status(self, status_pub, status_text: str) -> None:
        if status_pub is None or self._String is None:
            return
        try:
            msg = self._String()
            msg.data = status_text
            status_pub.publish(msg)
        except Exception as exc:
            print(f"[builtin-ik] Failed to publish IK status: {exc}")

    def _smooth_target(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        *,
        current_pos: Optional[np.ndarray] = None,
        current_quat: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config.enable_target_smoothing:
            return np.asarray(target_pos, dtype=float), _normalize_quaternion(target_quat)

        prev = self._smoothed_targets.get(side)
        cur_pos = np.asarray(target_pos, dtype=float)
        cur_quat = _normalize_quaternion(target_quat)
        if prev is None:
            if current_pos is None or current_quat is None:
                self._smoothed_targets[side] = (cur_pos.copy(), cur_quat.copy())
                return cur_pos, cur_quat
            prev = (
                np.asarray(current_pos, dtype=float).copy(),
                _normalize_quaternion(np.asarray(current_quat, dtype=float)),
            )

        prev_pos, prev_quat = prev
        alpha = float(max(0.0, min(1.0, self.config.target_smoothing_alpha)))
        pos = prev_pos + alpha * (cur_pos - prev_pos)
        delta = pos - prev_pos
        max_step = float(max(0.0, self.config.max_target_position_step_m))
        norm = float(np.linalg.norm(delta))
        if max_step > 0.0 and norm > max_step and norm > 1e-12:
            pos = prev_pos + delta / norm * max_step

        quat = _quat_slerp(prev_quat, cur_quat, alpha)
        max_ang = float(max(0.0, self.config.max_target_orientation_step_rad))
        if current_pos is not None:
            close_range_distance = float(self.config.close_range_orientation_distance_m)
            if close_range_distance > 0.0:
                distance_to_target = float(
                    np.linalg.norm(
                        np.asarray(cur_pos, dtype=float)
                        - np.asarray(current_pos, dtype=float)
                    )
                )
                if distance_to_target <= close_range_distance:
                    close_range_max_ang = float(
                        max(0.0, self.config.close_range_max_orientation_step_rad)
                    )
                    if close_range_max_ang > 0.0:
                        max_ang = (
                            close_range_max_ang
                            if max_ang <= 0.0
                            else min(max_ang, close_range_max_ang)
                        )
        ang = _quat_angle(prev_quat, quat)
        if max_ang > 0.0 and ang > max_ang and ang > 1e-12:
            quat = _quat_slerp(prev_quat, quat, max_ang / ang)

        pos = np.asarray(pos, dtype=float)
        quat = _normalize_quaternion(quat)
        self._smoothed_targets[side] = (pos.copy(), quat.copy())
        return pos, quat

    def _extract_positions(
        self, action, joint_names: List[str]
    ) -> Optional[List[float]]:
        if action is None:
            return None
        joint_positions = getattr(action, "joint_positions", None)
        if joint_positions is None:
            return None
        joint_positions = np.asarray(joint_positions, dtype=float).reshape(-1)
        joint_indices = getattr(action, "joint_indices", None)

        positions: List[float] = []
        if joint_indices is None or len(joint_indices) == 0:
            for name in joint_names:
                idx = self._dof_name_to_index.get(name)
                if idx is None or idx >= len(joint_positions):
                    positions.append(float("nan"))
                else:
                    positions.append(float(joint_positions[idx]))
        else:
            idx_to_pos = {
                int(joint_indices[i]): float(joint_positions[i])
                for i in range(min(len(joint_indices), len(joint_positions)))
            }
            for name in joint_names:
                idx = self._dof_name_to_index.get(name)
                if idx is None:
                    positions.append(float("nan"))
                else:
                    positions.append(float(idx_to_pos.get(idx, float("nan"))))
        return positions

    def _fill_missing_from_current(
        self, positions: List[float], joint_names: List[str]
    ) -> List[float]:
        if not positions or not joint_names:
            return positions
        if all(np.isfinite(positions)):
            return positions
        current = None
        try:
            current = self.articulation.get_joint_positions()
        except Exception:
            current = None
        if current is None:
            return [float(p) if np.isfinite(p) else 0.0 for p in positions]
        current = np.asarray(current).reshape(-1)
        out: List[float] = []
        for name, pos in zip(joint_names, positions):
            if np.isfinite(pos):
                out.append(float(pos))
                continue
            idx = self._dof_name_to_index.get(name)
            if idx is None or idx >= len(current):
                out.append(0.0)
            else:
                out.append(float(current[idx]))
        return out

    def _sanitize_joint_targets(
        self, positions: List[float], joint_names: List[str]
    ) -> List[float]:
        if not positions or not joint_names:
            return positions

        try:
            current = self.articulation.get_joint_positions()
        except Exception:
            current = None
        if current is None:
            return positions

        current_arr = np.asarray(current, dtype=float).reshape(-1)
        max_delta = float(self.config.max_joint_delta_rad)
        max_abs = float(self.config.max_joint_position_abs_rad)
        out: List[float] = []
        clipped_delta = False
        clipped_abs = False

        for joint_name, target in zip(joint_names, positions):
            joint_index = self._dof_name_to_index.get(joint_name)
            if joint_index is None or joint_index >= current_arr.size:
                out.append(float(target))
                continue

            current_value = float(current_arr[joint_index])
            value = float(target)
            if max_abs > 0.0 and abs(value) > max_abs:
                value = max(-max_abs, min(max_abs, value))
                clipped_abs = True

            if max_delta > 0.0:
                delta = value - current_value
                if abs(delta) > max_delta:
                    value = current_value + np.sign(delta) * max_delta
                    clipped_delta = True

            out.append(value)

        if clipped_abs and not self._warned_joint_abs_clip:
            print(
                "[builtin-ik] Clipped oversized IK joint target:"
                f" abs_limit={max_abs:.3f}rad"
            )
            self._warned_joint_abs_clip = True
        if clipped_delta and not self._warned_joint_delta_clip:
            print(
                "[builtin-ik] Clipped aggressive IK joint step:"
                f" step_limit={max_delta:.3f}rad"
            )
            self._warned_joint_delta_clip = True

        return out

    def _compute_action(
        self, solver, position: np.ndarray, orientation: Optional[np.ndarray]
    ) -> Tuple[Optional[object], bool]:
        try:
            if self.config.enable_orientation and orientation is not None:
                result = solver.compute_inverse_kinematics(
                    target_position=position, target_orientation=orientation
                )
            else:
                result = solver.compute_inverse_kinematics(target_position=position)
        except Exception as exc:
            print(f"[builtin-ik] compute_inverse_kinematics failed: {exc}")
            return None, False

        if isinstance(result, tuple) and len(result) == 2:
            first, second = result
            if isinstance(first, bool) and not isinstance(second, bool):
                return second, bool(first)
            return first, bool(second)
        return result, True

    def _compute_action_with_orientation_policy(
        self,
        *,
        side: str,
        solver,
        position: np.ndarray,
        orientation: Optional[np.ndarray],
        current_pos: np.ndarray,
        desired_ee_pos: np.ndarray,
    ) -> Tuple[Optional[object], bool, str]:
        use_orientation = bool(self.config.enable_orientation and orientation is not None)
        if not use_orientation:
            action, success = self._compute_action(solver, position, None)
            return action, success, "disabled"

        position_error = float(
            np.linalg.norm(
                np.asarray(current_pos, dtype=float) - np.asarray(desired_ee_pos, dtype=float)
            )
        )
        gate_distance = float(self.config.orientation_position_gate_m)
        fallback_distance = float(self.config.approx_orientation_accept_position_error_m)
        close_range_distance = float(self.config.close_range_orientation_distance_m)
        current_mode = self._orientation_mode_hysteresis.get(side, "full")
        solve_orientation: Optional[np.ndarray] = orientation

        enter_full_thresh = close_range_distance * 2.0 if close_range_distance > 0.0 else gate_distance
        exit_full_thresh = gate_distance * 1.5 if gate_distance > 0.0 else enter_full_thresh

        if current_mode == "position_first":
            if position_error <= enter_full_thresh:
                current_mode = "full"
                self._orientation_mode_hysteresis[side] = "full"
        elif current_mode == "full":
            if use_orientation and position_error >= exit_full_thresh:
                current_mode = "position_first"
                self._orientation_mode_hysteresis[side] = "position_first"
                if not self._logged_orientation_position_gate.get(side):
                    print(
                        "[builtin-ik] Temporarily gating orientation tracking until position is closer:"
                        f" side={side}"
                        f" position_error={position_error:.4f}"
                        f" gate={exit_full_thresh:.4f}"
                    )
                    self._logged_orientation_position_gate[side] = True

        orientation_mode = current_mode
        if current_mode == "position_first":
            solve_orientation = None

        action, success = self._compute_action(solver, position, solve_orientation)
        if orientation_mode != "full":
            return action, success, orientation_mode

        needs_position_fallback = action is None
        if not needs_position_fallback and not success and fallback_distance > 0.0:
            needs_position_fallback = position_error > fallback_distance
        if not needs_position_fallback:
            return action, success, orientation_mode

        fallback_action, fallback_success = self._compute_action(solver, position, None)
        if fallback_action is None:
            return action, success, orientation_mode

        if not self._logged_orientation_position_fallback.get(side):
            print(
                "[builtin-ik] Falling back to position-first IK for this target update:"
                f" side={side}"
                f" position_error={position_error:.4f}"
                f" approx_accept={fallback_distance:.4f}"
            )
            self._logged_orientation_position_fallback[side] = True
        return fallback_action, fallback_success, "position_fallback"

    def _publish(self, publisher, joint_names: List[str], positions: List[float]) -> None:
        if publisher is None or self._JointState is None:
            return
        msg = self._JointState()
        msg.name = list(joint_names)
        msg.position = [float(p) for p in positions]
        publisher.publish(msg)

    def _build_direct_target_full(self) -> Optional[np.ndarray]:
        if not self.config.direct_apply_joint_targets:
            return None
        try:
            joint_positions = self.articulation.get_joint_positions()
        except Exception:
            return None
        if joint_positions is None:
            return None
        arr = np.asarray(joint_positions, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = arr.reshape(-1)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            return None
        return arr.astype(np.float32, copy=True)

    def _merge_direct_joint_targets(
        self,
        direct_target_full: Optional[np.ndarray],
        joint_names: List[str],
        positions: List[float],
    ) -> None:
        if direct_target_full is None:
            return
        for joint_name, joint_pos in zip(joint_names, positions):
            joint_index = self._dof_name_to_index.get(joint_name)
            if joint_index is None or joint_index >= direct_target_full.size:
                continue
            direct_target_full[joint_index] = float(joint_pos)

    def _apply_direct_target_full(self, direct_target_full: Optional[np.ndarray]) -> None:
        if direct_target_full is None or not self.config.direct_apply_joint_targets:
            return
        try:
            self.articulation.set_joint_position_targets(
                np.asarray(direct_target_full, dtype=np.float32).reshape(1, -1)
            )
        except Exception as exc:
            if not self._warned_direct_apply:
                print(f"[builtin-ik] Direct joint target apply failed: {exc}")
                self._warned_direct_apply = True

    def _log_ik_fail_debug(
        self,
        side: str,
        *,
        frame_id: str,
        current_pos: np.ndarray,
        desired_ee_pos: np.ndarray,
        solver_target_pos: Optional[np.ndarray] = None,
        solver_frame_pos: Optional[np.ndarray] = None,
        usd_frame_pos: Optional[np.ndarray] = None,
    ) -> None:
        if self._logged_ik_fail_debug.get(side):
            return
        self._logged_ik_fail_debug[side] = True
        base_pose = self._get_robot_base_pose()
        if base_pose is None:
            base_pos_text = "None"
            base_quat_text = "None"
        else:
            base_pos, base_quat = base_pose
            base_pos_text = np.array2string(np.asarray(base_pos, dtype=float), precision=4)
            base_quat_text = np.array2string(np.asarray(base_quat, dtype=float), precision=4)
        solver_base_pose = self._get_solver_base_pose(side)
        if solver_base_pose is None:
            solver_base_pos_text = "None"
            solver_base_quat_text = "None"
        else:
            solver_base_pos, solver_base_quat = solver_base_pose
            solver_base_pos_text = np.array2string(
                np.asarray(solver_base_pos, dtype=float), precision=4
            )
            solver_base_quat_text = np.array2string(
                np.asarray(solver_base_quat, dtype=float), precision=4
            )
        print(
            "[builtin-ik] First ik_failed debug:"
            f" side={side}"
            f" frame={frame_id!r}"
            f" current_world={np.array2string(np.asarray(current_pos, dtype=float), precision=4)}"
            f" desired_world={np.array2string(np.asarray(desired_ee_pos, dtype=float), precision=4)}"
            f" solver_target_world={('None' if solver_target_pos is None else np.array2string(np.asarray(solver_target_pos, dtype=float), precision=4))}"
            f" solver_frame_world={('None' if solver_frame_pos is None else np.array2string(np.asarray(solver_frame_pos, dtype=float), precision=4))}"
            f" usd_frame_world={('None' if usd_frame_pos is None else np.array2string(np.asarray(usd_frame_pos, dtype=float), precision=4))}"
            f" base_world={base_pos_text}"
            f" base_quat={base_quat_text}"
            f" solver_base_world={solver_base_pos_text}"
            f" solver_base_quat={solver_base_quat_text}"
        )

    def _update_arm(
        self,
        side: str,
        has_active_target: bool,
        direct_target_full: Optional[np.ndarray] = None,
    ) -> bool:
        if side == "left":
            target = self._left_target
            solver = self._left_solver
            joint_names = self.config.left_arm_joints or []
            publisher = self._left_cmd_pub
            tcp_spec = self._left_tcp_spec
            pose_pub = self._left_pose_pub
            status_pub = self._left_status_pub
        else:
            target = self._right_target
            solver = self._right_solver
            joint_names = self.config.right_arm_joints or []
            publisher = self._right_cmd_pub
            tcp_spec = self._right_tcp_spec
            pose_pub = self._right_pose_pub
            status_pub = self._right_status_pub

        if solver is None:
            return False
        if not self._dof_name_to_index:
            self._cache_joint_indices()
        current_pos, current_quat = self._get_current_ee_pose_world(side)
        if current_pos is not None and current_quat is not None:
            self._last_current_pose_world[side] = (
                np.asarray(current_pos, dtype=float).copy(),
                _normalize_quaternion(np.asarray(current_quat, dtype=float)),
            )
            self._publish_current_pose(pose_pub, current_pos, current_quat)

        position, orientation, frame_id, target_stamp = target.get()
        target_age = (time.time() - target_stamp) if target_stamp > 0 else 0.0
        if not has_active_target:
            self._smoothed_targets.pop(side, None)
            label = "target_timeout" if position is not None else "idle"
            self._publish_status(status_pub, f"{label}, error=0.0000, target_age={target_age:.3f}s")
            return False
        if position is None:
            self._smoothed_targets.pop(side, None)
            self._publish_status(status_pub, "idle, error=0.0000")
            return False
        if orientation is None:
            orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        else:
            orientation = _normalize_quaternion(orientation)

        if current_pos is None or current_quat is None:
            current_pos = np.asarray(position, dtype=float)
            current_quat = _normalize_quaternion(orientation)

        resolved_target = self._resolve_target_in_world(position, orientation, frame_id)
        if resolved_target is None:
            self._smoothed_targets.pop(side, None)
            self._publish_status(
                status_pub,
                f"base_unavailable, error=0.0000, target_age={target_age:.3f}s",
            )
            return False
        position, orientation = resolved_target
        position, orientation = self._smooth_target(
            side,
            position,
            orientation,
            current_pos=current_pos,
            current_quat=current_quat,
        )
        desired_ee_pos = np.asarray(position, dtype=float)
        desired_ee_quat = _normalize_quaternion(orientation)

        if hasattr(solver, "get_kinematics_solver"):
            kin_solver = solver.get_kinematics_solver()
        else:
            kin_solver = solver
        self._set_solver_base_pose(kin_solver, side)

        solver_frame_pos = None
        solver_frame_quat = None
        if tcp_spec is not None:
            try:
                solver_frame_pos, solver_frame_rot = solver.compute_end_effector_pose(
                    position_only=False
                )
                solver_frame_pos = np.asarray(solver_frame_pos, dtype=float)
                solver_frame_quat = _quat_from_rotation_matrix(
                    np.asarray(solver_frame_rot, dtype=float)
                )
            except Exception:
                solver_frame_pos = None
                solver_frame_quat = None

        if tcp_spec is not None:
            orientation_link = tcp_spec.get("orientation_link")
            ref_pos = None
            offset_local = self._get_cached_tcp_offset_local(side)
            ref_quat = None
            if orientation_link:
                ref_pos, ref_quat = self._get_link_pose_usd(str(orientation_link))
            desired_link_quat = None
            if self.config.enable_orientation and orientation is not None:
                offset_quat = tcp_spec.get("orientation_offset_quat")
                if offset_quat is not None:
                    desired_link_quat = _quat_multiply(
                        np.asarray(orientation, dtype=float),
                        _quat_conjugate(np.asarray(offset_quat, dtype=float)),
                    )
                else:
                    desired_link_quat = np.asarray(orientation, dtype=float)
                orientation = desired_link_quat

            if offset_local is not None:
                if desired_link_quat is not None:
                    offset_world_desired = _quat_rotate_vector(
                        desired_link_quat, offset_local
                    )
                    position = np.asarray(position, dtype=float) - offset_world_desired
                elif solver_frame_quat is not None:
                    offset_world_current = _quat_rotate_vector(
                        solver_frame_quat, offset_local
                    )
                    position = np.asarray(position, dtype=float) - offset_world_current
                elif ref_quat is not None:
                    offset_world_current = _quat_rotate_vector(
                        _normalize_quaternion(ref_quat), offset_local
                    )
                    position = np.asarray(position, dtype=float) - offset_world_current

        action, success, orientation_mode = self._compute_action_with_orientation_policy(
            side=side,
            solver=solver,
            position=np.asarray(position, dtype=float),
            orientation=orientation,
            current_pos=np.asarray(current_pos, dtype=float),
            desired_ee_pos=np.asarray(desired_ee_pos, dtype=float),
        )
        if action is None:
            err = float(np.linalg.norm(np.asarray(current_pos, dtype=float) - desired_ee_pos))
            self._log_ik_fail_debug(
                side,
                frame_id=frame_id,
                current_pos=np.asarray(current_pos, dtype=float),
                desired_ee_pos=np.asarray(desired_ee_pos, dtype=float),
                solver_target_pos=np.asarray(position, dtype=float),
                solver_frame_pos=solver_frame_pos,
                usd_frame_pos=ref_pos,
            )
            self._publish_status(status_pub, f"ik_failed, error={err:.4f}, target_age={target_age:.3f}s")
            return False

        positions = self._extract_positions(action, joint_names)
        if positions is None:
            err = float(np.linalg.norm(np.asarray(current_pos, dtype=float) - desired_ee_pos))
            self._log_ik_fail_debug(
                side,
                frame_id=frame_id,
                current_pos=np.asarray(current_pos, dtype=float),
                desired_ee_pos=np.asarray(desired_ee_pos, dtype=float),
                solver_target_pos=np.asarray(position, dtype=float),
                solver_frame_pos=solver_frame_pos,
                usd_frame_pos=ref_pos,
            )
            self._publish_status(status_pub, f"ik_failed, error={err:.4f}, target_age={target_age:.3f}s")
            return False
        positions = self._fill_missing_from_current(positions, joint_names)
        positions = self._sanitize_joint_targets(positions, joint_names)
        if not all(np.isfinite(positions)):
            if not self._warned_missing_joints:
                print("[builtin-ik] Joint positions contain non-finite values.")
                self._warned_missing_joints = True
            err = float(np.linalg.norm(np.asarray(current_pos, dtype=float) - desired_ee_pos))
            self._log_ik_fail_debug(
                side,
                frame_id=frame_id,
                current_pos=np.asarray(current_pos, dtype=float),
                desired_ee_pos=np.asarray(desired_ee_pos, dtype=float),
                solver_target_pos=np.asarray(position, dtype=float),
                solver_frame_pos=solver_frame_pos,
                usd_frame_pos=ref_pos,
            )
            self._publish_status(status_pub, f"ik_failed, error={err:.4f}, target_age={target_age:.3f}s")
            return False
        self._merge_direct_joint_targets(direct_target_full, joint_names, positions)
        self._publish(publisher, joint_names, positions)
        err = float(np.linalg.norm(np.asarray(current_pos, dtype=float) - desired_ee_pos))
        approximate = not success
        if approximate and not self._logged_ik_approx_debug.get(side):
            print(
                "[builtin-ik] Using approximate IK action:"
                f" side={side}"
                f" frame={frame_id!r}"
                f" desired_world={np.array2string(np.asarray(desired_ee_pos, dtype=float), precision=4)}"
                f" solver_target_world={np.array2string(np.asarray(position, dtype=float), precision=4)}"
            )
            self._logged_ik_approx_debug[side] = True
        if orientation_mode == "position_fallback":
            status_label = (
                "tracking_pos_fallback_approx" if approximate else "tracking_pos_fallback"
            )
        elif orientation_mode == "position_first":
            status_label = (
                "tracking_pos_first_approx" if approximate else "tracking_pos_first"
            )
        elif approximate:
            status_label = "tracking_approx"
        else:
            status_label = "converged" if err <= 0.01 else "tracking"
        self._publish_status(status_pub, f"{status_label}, error={err:.4f}, target_age={target_age:.3f}s")
        return True

    def _on_left_target(self, msg) -> None:
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
            return
        quat = _normalize_quaternion(quat)
        self._left_target.set(pos, quat, msg.header.frame_id)

    def _on_right_target(self, msg) -> None:
        pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float
        )
        quat = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
            return
        quat = _normalize_quaternion(quat)
        self._right_target.set(pos, quat, msg.header.frame_id)

    def _spin(self) -> None:
        try:
            from rclpy.executors import ExternalShutdownException
        except Exception:
            ExternalShutdownException = Exception
        while self._running:
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except ExternalShutdownException:
                break
            except Exception:
                if not self._running:
                    break
                raise

    def start(self) -> None:
        if self._running:
            return
        self._singleton_lock = ProcessSingletonLock("builtin_ik_bridge")
        if not self._singleton_lock.acquire():
            self._duplicate_suppressed = True
            print(
                "[builtin-ik] Duplicate bridge suppressed:"
                f" existing owner {self._singleton_lock.owner_description()}"
            )
            self._singleton_lock = None
            return
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from geometry_msgs.msg import PoseStamped
            from sensor_msgs.msg import JointState
            from std_msgs.msg import String
        except Exception as exc:
            if self._singleton_lock is not None:
                self._singleton_lock.release()
                self._singleton_lock = None
            print(f"[builtin-ik] ROS2 not available: {exc}")
            return

        try:
            self._rclpy = rclpy
            self._PoseStamped = PoseStamped
            self._JointState = JointState
            self._String = String

            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy_context = True

            self._node = rclpy.create_node("builtin_ik_bridge")
            self._left_cmd_pub = self._node.create_publisher(
                JointState, self.config.left_cmd_topic, 10
            )
            self._right_cmd_pub = self._node.create_publisher(
                JointState, self.config.right_cmd_topic, 10
            )
            self._left_pose_pub = self._node.create_publisher(
                PoseStamped, self.config.left_pose_topic, 10
            )
            self._right_pose_pub = self._node.create_publisher(
                PoseStamped, self.config.right_pose_topic, 10
            )
            self._left_status_pub = self._node.create_publisher(
                String, self.config.left_status_topic, 10
            )
            self._right_status_pub = self._node.create_publisher(
                String, self.config.right_status_topic, 10
            )
            self._node.create_subscription(
                PoseStamped, self.config.left_ee_topic, self._on_left_target, 10
            )
            self._node.create_subscription(
                PoseStamped, self.config.right_ee_topic, self._on_right_target, 10
            )

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._running = True
            self._initialize_solvers()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        except Exception:
            if self._singleton_lock is not None:
                self._singleton_lock.release()
                self._singleton_lock = None
            raise

        print(
            "[builtin-ik] Started."
            f" left_target={self.config.left_ee_topic}"
            f" right_target={self.config.right_ee_topic}"
            f" left_cmd={self.config.left_cmd_topic}"
            f" right_cmd={self.config.right_cmd_topic}"
            f" control_mode={self.config.control_mode}"
            f" left_solver_base={self.config.left_solver_base_link}"
            f" right_solver_base={self.config.right_solver_base_link}"
            f" feedback_frame={self.config.feedback_frame_id}"
            f" direct_apply={self.config.direct_apply_joint_targets}"
            f" orientation_gate={self.config.orientation_position_gate_m:.3f}"
            f" approx_orientation_accept={self.config.approx_orientation_accept_position_error_m:.3f}"
            f" close_range_orientation_distance={self.config.close_range_orientation_distance_m:.3f}"
            f" close_range_orientation_step={self.config.close_range_max_orientation_step_rad:.3f}"
        )

    def stop(self) -> None:
        if not self._running:
            if self._singleton_lock is not None:
                self._singleton_lock.release()
                self._singleton_lock = None
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._owns_rclpy_context and self._rclpy is not None and self._rclpy.ok():
            self._rclpy.shutdown()
        if self._singleton_lock is not None:
            self._singleton_lock.release()
            self._singleton_lock = None

    def update(self) -> None:
        if not self._running:
            return
        self._cached_base_pose = None
        self._cached_base_pose_stamp = 0.0
        if not self._initialize_solvers():
            return
        has_left_target = self._left_target.is_active(self.config.ik_timeout_sec)
        has_right_target = self._right_target.is_active(self.config.ik_timeout_sec)
        direct_target_full = self._build_direct_target_full()
        active_update = False

        if self.config.control_mode == "auto":
            # Keep EE feedback alive in auto mode so RViz tools can seed from the
            # current pose, but only apply commands on arms with active targets.
            active_update = self._update_arm("left", has_left_target, direct_target_full) or active_update
            active_update = self._update_arm("right", has_right_target, direct_target_full) or active_update
            if active_update:
                self._apply_direct_target_full(direct_target_full)
            return

        active_update = self._update_arm("left", has_left_target, direct_target_full) or active_update
        active_update = self._update_arm("right", has_right_target, direct_target_full) or active_update
        if active_update:
            self._apply_direct_target_full(direct_target_full)
