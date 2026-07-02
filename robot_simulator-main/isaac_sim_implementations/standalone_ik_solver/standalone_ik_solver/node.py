"""ROS 2 node wrapper for the standalone IK solver."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .config import SolverConfig
from .kinematics import (
    ArmTCPModel,
    GRIPPER_MOUNT_QUAT,
    HybridIKSolver,
    KinematicsChain,
    RealtimeDLSIKSolver,
    _normalize_quaternion,
    _quat_conjugate,
    _quat_diff_angle,
    _quat_multiply,
    _quat_rotate_vector,
    _quat_slerp,
)


@dataclass
class _TargetSnapshot:
    position: Optional[np.ndarray]
    orientation: Optional[np.ndarray]
    frame_id: str
    stamp_sec: float


class _TargetState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._position: Optional[np.ndarray] = None
        self._orientation: Optional[np.ndarray] = None
        self._frame_id: str = ""
        self._stamp_sec: float = 0.0

    def set(self, position: np.ndarray, orientation: np.ndarray, frame_id: str) -> None:
        with self._lock:
            self._position = np.asarray(position, dtype=float).copy()
            self._orientation = _normalize_quaternion(orientation)
            self._frame_id = frame_id or ""
            self._stamp_sec = time.time()

    def get(self) -> _TargetSnapshot:
        with self._lock:
            return _TargetSnapshot(
                None if self._position is None else self._position.copy(),
                None if self._orientation is None else self._orientation.copy(),
                self._frame_id,
                self._stamp_sec,
            )

    def is_active(self, timeout_sec: float) -> bool:
        with self._lock:
            if self._position is None:
                return False
            if timeout_sec <= 0.0:
                return True
            return (time.time() - self._stamp_sec) <= timeout_sec


class StandaloneIKSolverNode(Node):
    def __init__(self, config: SolverConfig) -> None:
        super().__init__("standalone_ik_solver")
        self._config = config
        self._left_target = _TargetState()
        self._right_target = _TargetState()
        self._left_base_chain: Optional[KinematicsChain] = None
        self._right_base_chain: Optional[KinematicsChain] = None
        if config.urdf_path and Path(config.urdf_path).expanduser().exists():
            left_orientation_chain = KinematicsChain.from_urdf(
                config.urdf_path,
                config.left_arm_joints,
                base_link=config.left_solver_base_link,
                tip_link="left_arm_link9",
                tcp_offset_m=0.0,
                tcp_orientation_offset_quat=GRIPPER_MOUNT_QUAT,
            )
            right_orientation_chain = KinematicsChain.from_urdf(
                config.urdf_path,
                config.right_arm_joints,
                base_link=config.right_solver_base_link,
                tip_link="right_arm_link9",
                tcp_offset_m=0.0,
                tcp_orientation_offset_quat=GRIPPER_MOUNT_QUAT,
            )
            self._left_chain = ArmTCPModel(
                left_orientation_chain,
                fingertip_chains=[
                    KinematicsChain.from_urdf(
                        config.urdf_path,
                        [*config.left_arm_joints, config.left_gripper_joints[0]],
                        base_link=config.left_solver_base_link,
                        tip_link="left_gripper_narrow3_link",
                        tcp_offset_m=0.0,
                        tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    ),
                    KinematicsChain.from_urdf(
                        config.urdf_path,
                        [*config.left_arm_joints, config.left_gripper_joints[1]],
                        base_link=config.left_solver_base_link,
                        tip_link="left_gripper_wide3_link",
                        tcp_offset_m=0.0,
                        tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    ),
                ],
            )
            self._right_chain = ArmTCPModel(
                right_orientation_chain,
                fingertip_chains=[
                    KinematicsChain.from_urdf(
                        config.urdf_path,
                        [*config.right_arm_joints, config.right_gripper_joints[0]],
                        base_link=config.right_solver_base_link,
                        tip_link="right_gripper_narrow3_link",
                        tcp_offset_m=0.0,
                        tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    ),
                    KinematicsChain.from_urdf(
                        config.urdf_path,
                        [*config.right_arm_joints, config.right_gripper_joints[1]],
                        base_link=config.right_solver_base_link,
                        tip_link="right_gripper_wide3_link",
                        tcp_offset_m=0.0,
                        tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                    ),
                ],
            )
            if config.left_solver_base_link and config.left_solver_base_link != "base_link":
                self._left_base_chain = KinematicsChain.from_urdf(
                    config.urdf_path,
                    config.body_joints,
                    base_link="base_link",
                    tip_link=config.left_solver_base_link,
                    tcp_offset_m=0.0,
                    tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                )
            if config.right_solver_base_link and config.right_solver_base_link != "base_link":
                self._right_base_chain = KinematicsChain.from_urdf(
                    config.urdf_path,
                    config.body_joints,
                    base_link="base_link",
                    tip_link=config.right_solver_base_link,
                    tcp_offset_m=0.0,
                    tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                )
        else:
            self._left_chain = KinematicsChain(config.link_lengths, config.joint_limits)
            self._right_chain = KinematicsChain(config.link_lengths, config.joint_limits)
        self._left_solver = HybridIKSolver(
            self._left_chain,
            fabrik_iterations=config.fabrik_iterations,
            lbfgsb_maxiter=config.lbfgsb_maxiter,
            damping_lambda=config.damping_lambda,
            position_tolerance_m=config.position_tolerance_m,
            orientation_tolerance_rad=config.orientation_tolerance_rad,
        )
        self._right_solver = HybridIKSolver(
            self._right_chain,
            fabrik_iterations=config.fabrik_iterations,
            lbfgsb_maxiter=config.lbfgsb_maxiter,
            damping_lambda=config.damping_lambda,
            position_tolerance_m=config.position_tolerance_m,
            orientation_tolerance_rad=config.orientation_tolerance_rad,
        )
        self._left_fast_solver = RealtimeDLSIKSolver(
            self._left_chain,
            damping=config.realtime_dls_damping,
            position_gain=config.realtime_dls_position_gain,
            orientation_gain=config.realtime_dls_orientation_gain,
            max_step_rad=min(config.realtime_dls_max_step_rad, config.max_joint_delta_rad),
            position_tolerance_m=config.position_tolerance_m,
            orientation_tolerance_rad=config.orientation_tolerance_rad,
        )
        self._right_fast_solver = RealtimeDLSIKSolver(
            self._right_chain,
            damping=config.realtime_dls_damping,
            position_gain=config.realtime_dls_position_gain,
            orientation_gain=config.realtime_dls_orientation_gain,
            max_step_rad=min(config.realtime_dls_max_step_rad, config.max_joint_delta_rad),
            position_tolerance_m=config.position_tolerance_m,
            orientation_tolerance_rad=config.orientation_tolerance_rad,
        )

        self._left_last_joints = np.zeros(7, dtype=float)
        self._right_last_joints = np.zeros(7, dtype=float)
        self._body_joint_state_map: Dict[str, float] = {}
        self._left_joint_state_map: Dict[str, float] = {}
        self._right_joint_state_map: Dict[str, float] = {}
        self._left_smoothed_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._right_smoothed_pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._orientation_mode_hysteresis: Dict[str, str] = {"left": "position", "right": "position"}
        self._feedbackless_prediction_active: Dict[str, bool] = {"left": False, "right": False}
        self._warned_target_frames: set[str] = set()
        self._warned_body_state_default = False
        self._rate_hz = float(max(1e-3, config.ik_rate_hz))
        self._sleep_dt = 1.0 / self._rate_hz

        self._left_cmd_pub = self.create_publisher(JointState, config.left_cmd_topic, 10)
        self._right_cmd_pub = self.create_publisher(JointState, config.right_cmd_topic, 10)
        self._left_pose_pub = self.create_publisher(PoseStamped, config.left_pose_topic, 10)
        self._right_pose_pub = self.create_publisher(PoseStamped, config.right_pose_topic, 10)
        self._left_status_pub = self.create_publisher(String, config.left_status_topic, 10)
        self._right_status_pub = self.create_publisher(String, config.right_status_topic, 10)

        self.create_subscription(PoseStamped, config.left_ee_topic, self._on_left_target, 10)
        self.create_subscription(PoseStamped, config.right_ee_topic, self._on_right_target, 10)
        self.create_subscription(
            JointState,
            config.aggregate_state_topic,
            self._on_aggregate_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            config.body_state_topic,
            self._on_body_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            config.left_state_topic,
            self._on_left_joint_state,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            config.right_state_topic,
            self._on_right_joint_state,
            qos_profile_sensor_data,
        )

    def start(self) -> None:
        self.get_logger().info(
            "standalone_ik_solver started:"
            f" rate={self._rate_hz:.1f}Hz"
            f" enable_orientation={self._config.enable_orientation}"
        )

    def stop(self) -> None:
        pass

    def _sleep(self) -> None:
        time.sleep(self._sleep_dt)

    def _on_left_target(self, msg: PoseStamped) -> None:
        self._set_target(self._left_target, msg)

    def _on_right_target(self, msg: PoseStamped) -> None:
        self._set_target(self._right_target, msg)

    def _on_left_joint_state(self, msg: JointState) -> None:
        self._update_joint_state_map(self._left_joint_state_map, msg)

    def _on_right_joint_state(self, msg: JointState) -> None:
        self._update_joint_state_map(self._right_joint_state_map, msg)

    def _on_body_joint_state(self, msg: JointState) -> None:
        self._update_joint_state_map(self._body_joint_state_map, msg)

    def _on_aggregate_joint_state(self, msg: JointState) -> None:
        self._update_joint_state_map(self._left_joint_state_map, msg)
        self._update_joint_state_map(self._right_joint_state_map, msg)
        self._update_joint_state_map(self._body_joint_state_map, msg)
        if self._warned_body_state_default and all(
            name in self._body_joint_state_map for name in self._config.body_joints[:3]
        ):
            self._warned_body_state_default = False

    def _set_target(self, target: _TargetState, msg: PoseStamped) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=float,
        )
        orientation = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
            return
        target.set(position, orientation, msg.header.frame_id)

    def _update_joint_state_map(self, state_map: Dict[str, float], msg: JointState) -> None:
        pair_count = min(len(msg.name), len(msg.position))
        for index in range(pair_count):
            state_map[str(msg.name[index])] = float(msg.position[index])

    def _measured_joints(self, side: str, joint_names) -> Optional[np.ndarray]:
        state_map = self._left_joint_state_map if side == "left" else self._right_joint_state_map
        if all(name in state_map for name in joint_names):
            return np.asarray([state_map[name] for name in joint_names], dtype=float)
        return None

    def _current_joints(self, side: str, joint_names) -> np.ndarray:
        measured = self._measured_joints(side, joint_names)
        if measured is not None:
            return measured
        return (self._left_last_joints if side == "left" else self._right_last_joints).copy()

    def _sync_aux_joint_state(self, side: str) -> None:
        chain = self._left_chain if side == "left" else self._right_chain
        if not isinstance(chain, ArmTCPModel):
            return
        state_map = self._left_joint_state_map if side == "left" else self._right_joint_state_map
        chain.set_aux_joint_positions(state_map)

    @staticmethod
    def _normalize_frame_id(frame_id: str) -> str:
        return (frame_id or "").strip().lstrip("/")

    def _solver_base_link(self, side: str) -> str:
        if side == "left":
            return self._normalize_frame_id(self._config.left_solver_base_link)
        return self._normalize_frame_id(self._config.right_solver_base_link)

    def _feedback_frame(self) -> str:
        return self._normalize_frame_id(self._config.feedback_frame_id) or "base_link"

    def _current_body_joints(self, chain: Optional[KinematicsChain]) -> Optional[np.ndarray]:
        if chain is None:
            return None
        joint_names = chain.joint_names
        if not joint_names:
            return None
        if not all(name in self._body_joint_state_map for name in joint_names):
            if not self._warned_body_state_default:
                self.get_logger().warning(
                    "body joint states not available yet; assuming zero body pose for standalone IK"
                )
                self._warned_body_state_default = True
            return np.zeros(chain.dof, dtype=float)
        return np.asarray([self._body_joint_state_map[name] for name in joint_names], dtype=float)

    def _current_solver_base_pose(self, side: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        solver_base_link = self._solver_base_link(side)
        if not solver_base_link or solver_base_link == "base_link":
            return np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        chain = self._left_base_chain if side == "left" else self._right_base_chain
        body_joints = self._current_body_joints(chain)
        if chain is None or body_joints is None:
            return None
        fk = chain.forward_kinematics(body_joints)
        return (
            np.asarray(fk["ee_position"], dtype=float),
            _normalize_quaternion(np.asarray(fk["ee_quaternion"], dtype=float)),
        )

    def _target_to_solver_frame(
        self,
        side: str,
        position: np.ndarray,
        orientation: np.ndarray,
        frame_id: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        frame = self._normalize_frame_id(frame_id)
        solver_base_link = self._solver_base_link(side)
        if not frame or frame == "base_link":
            solver_base_pose = self._current_solver_base_pose(side)
            if solver_base_pose is None:
                return None
            base_pos, base_quat = solver_base_pose
            base_conj = _quat_conjugate(base_quat)
            return (
                _quat_rotate_vector(base_conj, np.asarray(position, dtype=float) - np.asarray(base_pos, dtype=float)),
                _quat_multiply(base_conj, np.asarray(orientation, dtype=float)),
            )
        if frame == solver_base_link:
            return np.asarray(position, dtype=float), _normalize_quaternion(orientation)
        if frame not in self._warned_target_frames:
            self.get_logger().warning(
                f"unsupported target frame '{frame_id}', assuming base_link for standalone IK"
            )
            self._warned_target_frames.add(frame)
        return self._target_to_solver_frame(side, position, orientation, "base_link")

    def _solver_to_feedback_frame(
        self,
        side: str,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        frame = self._feedback_frame()
        if frame == self._solver_base_link(side):
            return np.asarray(position, dtype=float), _normalize_quaternion(orientation), frame
        if frame == "base_link":
            solver_base_pose = self._current_solver_base_pose(side)
            if solver_base_pose is not None:
                base_pos, base_quat = solver_base_pose
                return (
                    np.asarray(base_pos, dtype=float)
                    + _quat_rotate_vector(np.asarray(base_quat, dtype=float), np.asarray(position, dtype=float)),
                    _quat_multiply(np.asarray(base_quat, dtype=float), np.asarray(orientation, dtype=float)),
                    "base_link",
                )
        return np.asarray(position, dtype=float), _normalize_quaternion(orientation), self._solver_base_link(side)

    def _smooth_target(
        self,
        side: str,
        position: np.ndarray,
        orientation: np.ndarray,
        *,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        prev = self._left_smoothed_pose if side == "left" else self._right_smoothed_pose
        if prev is None:
            prev_pos = np.asarray(current_pos, dtype=float)
            prev_quat = _normalize_quaternion(current_quat)
        else:
            prev_pos = prev[0]
            prev_quat = prev[1]

        if not self._config.enable_target_smoothing:
            return np.asarray(position, dtype=float), _normalize_quaternion(orientation)

        alpha = float(max(0.0, min(1.0, self._config.target_smoothing_alpha)))
        target_pos = np.asarray(position, dtype=float)
        smoothed_pos = prev_pos + alpha * (target_pos - prev_pos)
        delta = smoothed_pos - prev_pos
        delta_norm = float(np.linalg.norm(delta))
        max_step = float(self._config.max_target_position_step_m)
        if max_step > 0.0 and delta_norm > max_step and delta_norm > 1e-12:
            smoothed_pos = prev_pos + (delta / delta_norm) * max_step

        target_quat = _normalize_quaternion(orientation)
        smoothed_quat = _quat_slerp(prev_quat, target_quat, alpha)
        max_angle = float(self._config.max_target_orientation_step_rad)
        close_range_distance = float(self._config.close_range_orientation_distance_m)
        if close_range_distance > 0.0:
            distance_to_target = float(np.linalg.norm(target_pos - np.asarray(current_pos, dtype=float)))
            if distance_to_target <= close_range_distance:
                close_max = float(self._config.close_range_max_orientation_step_rad)
                if close_max > 0.0:
                    max_angle = min(max_angle, close_max) if max_angle > 0.0 else close_max
        quat_delta = _quat_diff_angle(prev_quat, smoothed_quat)
        if max_angle > 0.0 and quat_delta > max_angle and quat_delta > 1e-12:
            smoothed_quat = _quat_slerp(prev_quat, smoothed_quat, max_angle / quat_delta)

        smoothed = (smoothed_pos.copy(), smoothed_quat.copy())
        if side == "left":
            self._left_smoothed_pose = smoothed
        else:
            self._right_smoothed_pose = smoothed
        return smoothed

    def _apply_joint_limits(self, side: str, positions: np.ndarray) -> np.ndarray:
        chain = self._left_chain if side == "left" else self._right_chain
        limited = chain.clip_to_limits(np.asarray(positions, dtype=float))
        abs_limit = float(self._config.max_joint_position_abs_rad)
        if abs_limit > 0.0:
            limited = np.clip(limited, -abs_limit, abs_limit)
        return limited

    def _command_has_subscriber(self, side: str) -> bool:
        publisher = self._left_cmd_pub if side == "left" else self._right_cmd_pub
        try:
            return publisher.get_subscription_count() > 0
        except Exception:
            return True

    def _clamp_joint_delta(self, proposed: np.ndarray, current: np.ndarray) -> np.ndarray:
        current_values = np.asarray(current, dtype=float).reshape(-1)
        values = np.asarray(proposed, dtype=float).reshape(current_values.shape).copy()
        max_delta = float(self._config.max_joint_delta_rad)
        if max_delta <= 0.0:
            return values
        delta = np.clip(values - current_values, -max_delta, max_delta)
        return current_values + delta

    def _compute_ik(
        self,
        side: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        current_joints: np.ndarray,
        *,
        current_pos: np.ndarray,
        current_quat: np.ndarray,
    ):
        solver = self._left_solver if side == "left" else self._right_solver
        fast_solver = self._left_fast_solver if side == "left" else self._right_fast_solver
        use_orientation = bool(self._config.enable_orientation)
        position_error = float(
            np.linalg.norm(np.asarray(current_pos, dtype=float) - np.asarray(target_pos, dtype=float))
        )
        gate_distance = float(self._config.orientation_position_gate_m)
        close_range_distance = float(self._config.close_range_orientation_distance_m)
        fallback_distance = float(self._config.approx_orientation_accept_position_error_m)
        current_mode = self._orientation_mode_hysteresis.get(side, "full")
        enter_full_thresh = close_range_distance * 2.0 if close_range_distance > 0.0 else gate_distance
        exit_full_thresh = gate_distance * 1.5 if gate_distance > 0.0 else enter_full_thresh

        if use_orientation:
            if current_mode == "position":
                if position_error <= enter_full_thresh:
                    current_mode = "full"
                    self._orientation_mode_hysteresis[side] = "full"
            elif current_mode == "full":
                if position_error >= exit_full_thresh:
                    current_mode = "position"
                    self._orientation_mode_hysteresis[side] = "position"

        solve_orientation = target_quat if use_orientation and current_mode == "full" else None
        fast_iterations = 1
        near_target_distance = float(self._config.realtime_dls_near_target_distance_m)
        if near_target_distance > 0.0 and position_error <= near_target_distance:
            fast_iterations = max(1, int(self._config.realtime_dls_near_target_substeps))
        fast_result = fast_solver.solve_step(
            target_pos,
            solve_orientation,
            current_joints,
            enable_orientation=solve_orientation is not None,
            iterations=fast_iterations,
        )
        proposed_fast = self._apply_joint_limits(
            side,
            self._clamp_joint_delta(fast_result.positions, current_joints),
        )
        current_orientation_error = 0.0
        if solve_orientation is not None:
            current_orientation_error = _quat_diff_angle(
                np.asarray(current_quat, dtype=float),
                np.asarray(solve_orientation, dtype=float),
            )

        if solve_orientation is None:
            return proposed_fast, fast_result
        if fast_result.success:
            return proposed_fast, fast_result
        if (
            fast_result.position_error <= position_error
            and fast_result.orientation_error <= current_orientation_error
        ):
            return proposed_fast, fast_result

        result = solver.solve(
            target_pos,
            solve_orientation,
            current_joints,
            enable_orientation=solve_orientation is not None,
        )
        if solve_orientation is not None and not result.success and position_error > fallback_distance:
            fallback = solver.solve(
                target_pos,
                None,
                current_joints,
                enable_orientation=False,
            )
            if fallback.position_error <= result.position_error or not result.success:
                result = fallback
                result.mode = "position_fallback"
        elif solve_orientation is None:
            result.mode = "position"
        proposed = self._clamp_joint_delta(result.positions, current_joints)
        proposed = self._apply_joint_limits(side, proposed)
        return proposed, result

    def update(self) -> None:
        if not rclpy.ok():
            return
        for side in ("left", "right"):
            target = self._left_target if side == "left" else self._right_target
            chain = self._left_chain if side == "left" else self._right_chain
            self._sync_aux_joint_state(side)
            joint_names = (
                self._config.left_arm_joints if side == "left" else self._config.right_arm_joints
            )
            cmd_pub = self._left_cmd_pub if side == "left" else self._right_cmd_pub
            pose_pub = self._left_pose_pub if side == "left" else self._right_pose_pub
            status_pub = self._left_status_pub if side == "left" else self._right_status_pub
            last_joints = self._left_last_joints if side == "left" else self._right_last_joints
            snapshot = target.get()
            target_age = (time.time() - snapshot.stamp_sec) if snapshot.stamp_sec > 0.0 else 0.0
            measured_joints = self._measured_joints(side, joint_names)
            has_cmd_subscriber = self._command_has_subscriber(side)
            if has_cmd_subscriber:
                self._feedbackless_prediction_active[side] = False
                if measured_joints is not None:
                    last_joints[:] = measured_joints
                current_joints = measured_joints.copy() if measured_joints is not None else last_joints.copy()
            else:
                if snapshot.position is not None and target.is_active(self._config.ik_timeout_sec):
                    if not self._feedbackless_prediction_active[side] and measured_joints is not None:
                        last_joints[:] = measured_joints
                    self._feedbackless_prediction_active[side] = True
                    current_joints = last_joints.copy()
                else:
                    self._feedbackless_prediction_active[side] = False
                    if measured_joints is not None:
                        last_joints[:] = measured_joints
                    current_joints = measured_joints.copy() if measured_joints is not None else last_joints.copy()

            current_fk = chain.forward_kinematics(current_joints)
            current_pos = np.asarray(current_fk["ee_position"], dtype=float)
            current_quat = np.asarray(current_fk["ee_quaternion"], dtype=float)
            current_feedback_pos, current_feedback_quat, current_feedback_frame = self._solver_to_feedback_frame(
                side,
                current_pos,
                current_quat,
            )
            self._publish_pose(
                pose_pub,
                current_feedback_pos,
                current_feedback_quat,
                current_feedback_frame,
            )

            if snapshot.position is None:
                if side == "left":
                    self._left_smoothed_pose = None
                else:
                    self._right_smoothed_pose = None
                self._orientation_mode_hysteresis[side] = "position"
                self._feedbackless_prediction_active[side] = False
                self._publish_status(status_pub, "idle, error=0.0000")
                continue
            if not target.is_active(self._config.ik_timeout_sec):
                if side == "left":
                    self._left_smoothed_pose = None
                else:
                    self._right_smoothed_pose = None
                self._orientation_mode_hysteresis[side] = "position"
                self._feedbackless_prediction_active[side] = False
                self._publish_status(status_pub, f"target_timeout, error=0.0000, target_age={target_age:.3f}s")
                continue

            target_pos = np.asarray(snapshot.position, dtype=float)
            target_quat = (
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
                if snapshot.orientation is None
                else _normalize_quaternion(snapshot.orientation)
            )
            resolved_target = self._target_to_solver_frame(side, target_pos, target_quat, snapshot.frame_id)
            if resolved_target is None:
                if side == "left":
                    self._left_smoothed_pose = None
                else:
                    self._right_smoothed_pose = None
                self._orientation_mode_hysteresis[side] = "position"
                self._feedbackless_prediction_active[side] = False
                self._publish_status(
                    status_pub,
                    f"solver_base_unavailable, error=0.0000, target_age={target_age:.3f}s",
                )
                continue
            target_pos, target_quat = resolved_target
            target_pos, target_quat = self._smooth_target(
                side,
                target_pos,
                target_quat,
                current_pos=current_pos,
                current_quat=current_quat,
            )
            proposed, result = self._compute_ik(
                side,
                target_pos,
                target_quat,
                current_joints,
                current_pos=current_pos,
                current_quat=current_quat,
            )
            last_joints[:] = proposed
            solved_fk = chain.forward_kinematics(last_joints)
            solved_pos = np.asarray(solved_fk["ee_position"], dtype=float)
            solved_quat = np.asarray(solved_fk["ee_quaternion"], dtype=float)
            error = float(np.linalg.norm(solved_pos - target_pos))
            solved_feedback_pos, solved_feedback_quat, solved_feedback_frame = self._solver_to_feedback_frame(
                side,
                solved_pos,
                solved_quat,
            )

            self._publish_command(cmd_pub, joint_names, last_joints)
            self._publish_pose(
                pose_pub,
                solved_feedback_pos,
                solved_feedback_quat,
                solved_feedback_frame,
            )
            if result.mode == "position_fallback":
                label = "tracking_pos_fallback_approx" if result.approximate else "tracking_pos_fallback"
            elif result.mode == "position":
                label = "tracking_pos_first_approx" if result.approximate else "tracking_pos_first"
            elif result.approximate:
                label = "tracking_approx"
            else:
                label = "converged" if error <= 0.01 else "tracking"
            self._publish_status(status_pub, f"{label}, error={error:.4f}, target_age={target_age:.3f}s")

    def _publish_command(self, pub, joint_names, positions: np.ndarray) -> None:
        if not rclpy.ok():
            return
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(value) for value in positions]
        try:
            pub.publish(msg)
        except Exception:
            if rclpy.ok():
                raise

    def _publish_pose(
        self,
        pub,
        position: np.ndarray,
        quaternion: np.ndarray,
        frame_id: str,
    ) -> None:
        if not rclpy.ok():
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = float(quaternion[0])
        msg.pose.orientation.x = float(quaternion[1])
        msg.pose.orientation.y = float(quaternion[2])
        msg.pose.orientation.z = float(quaternion[3])
        try:
            pub.publish(msg)
        except Exception:
            if rclpy.ok():
                raise

    def _publish_status(self, pub, status: str) -> None:
        if not rclpy.ok():
            return
        msg = String()
        msg.data = status
        try:
            pub.publish(msg)
        except Exception:
            if rclpy.ok():
                raise
