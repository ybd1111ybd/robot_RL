"""Pure Python kinematics and IK solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

try:
    from scipy.optimize import minimize

    _HAS_SCIPY = True
except Exception:
    minimize = None
    _HAS_SCIPY = False


TCP_OFFSET_M = 0.10547
JOINT_AXES = (
    np.array([0.0, 0.0, 1.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
    np.array([1.0, 0.0, 0.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
    np.array([1.0, 0.0, 0.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
)


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _normalize_quat(quaternion: np.ndarray) -> np.ndarray:
    return _normalize_quaternion(quaternion)


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion(quaternion)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)


def _quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
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
    quat = _normalize_quaternion(quaternion)
    vec = np.asarray(vector, dtype=float).reshape(3)
    q_vec = quat[1:4]
    q_w = float(quat[0])
    return (
        2.0 * np.dot(q_vec, vec) * q_vec
        + (q_w * q_w - np.dot(q_vec, q_vec)) * vec
        + 2.0 * q_w * np.cross(q_vec, vec)
    )


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
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


GRIPPER_MOUNT_QUAT = _quat_from_rpy(-2.9951, -1.5708, -0.15964)


def _quat_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
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
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12)) * 2.0
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
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12)) * 2.0
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
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12)) * 2.0
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


def _quat_diff_angle(lhs: np.ndarray, rhs: np.ndarray) -> float:
    left = _normalize_quaternion(lhs)
    right = _normalize_quaternion(rhs)
    dot = float(np.dot(left, right))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * float(np.arccos(dot))


def _quat_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion(quaternion)
    imag_norm = float(np.linalg.norm(quat[1:]))
    if imag_norm < 1e-9:
        return np.zeros(3, dtype=float)
    angle = 2.0 * math.atan2(imag_norm, quat[0])
    axis = quat[1:] / imag_norm
    return axis * angle


def _orientation_error_vector(target_quaternion: np.ndarray, current_quaternion: np.ndarray) -> np.ndarray:
    delta = _quat_multiply(target_quaternion, _quat_conjugate(current_quaternion))
    return _quat_to_axis_angle(delta)


def _quat_slerp(lhs: np.ndarray, rhs: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(lhs)
    q1 = _normalize_quaternion(rhs)
    t = float(max(0.0, min(1.0, alpha)))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + t * (q1 - q0))
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    if sin_theta_0 < 1e-8:
        return q0
    theta = theta_0 * t
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    return _normalize_quaternion(s0 * q0 + s1 * q1)


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_vec = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis_vec))
    if norm < 1e-12:
        return np.eye(3, dtype=float)
    x, y, z = axis_vec / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return (
        _rotation_matrix(np.array([0.0, 0.0, 1.0], dtype=float), yaw)
        @ _rotation_matrix(np.array([0.0, 1.0, 0.0], dtype=float), pitch)
        @ _rotation_matrix(np.array([1.0, 0.0, 0.0], dtype=float), roll)
    )


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_matrix_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    transform[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return transform


@dataclass
class IKSolveResult:
    positions: np.ndarray
    success: bool
    approximate: bool
    position_error: float
    orientation_error: float
    mode: str


@dataclass
class _ChainSegment:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    controlled_index: Optional[int]


class KinematicsChain:
    """Simple 7-DOF chain with alternating rotary axes and TCP offset."""

    def __init__(
        self,
        link_lengths: Sequence[float],
        joint_limits: Sequence[Tuple[float, float]],
        tcp_offset_m: float = TCP_OFFSET_M,
        *,
        joint_names: Optional[Sequence[str]] = None,
        segments: Optional[Sequence[_ChainSegment]] = None,
        tcp_orientation_offset_quat: Optional[np.ndarray] = None,
    ) -> None:
        if segments is None and len(link_lengths) != len(joint_limits):
            raise ValueError("link_lengths and joint_limits must have the same size")
        if segments is None and len(joint_limits) != len(JOINT_AXES):
            raise ValueError("Fallback chain expects 7 joint limits")
        self.link_lengths = [float(value) for value in link_lengths]
        self.joint_limits = [(float(low), float(high)) for low, high in joint_limits]
        self.joint_names = [str(name) for name in joint_names] if joint_names is not None else []
        self.dof = len(self.joint_limits)
        self.tcp_offset_m = float(tcp_offset_m)
        self._segments = list(segments) if segments is not None else None
        self._tcp_orientation_offset_quat = _normalize_quaternion(
            GRIPPER_MOUNT_QUAT if tcp_orientation_offset_quat is None else tcp_orientation_offset_quat
        )

    @classmethod
    def from_urdf(
        cls,
        urdf_path: str,
        joint_names: Sequence[str],
        *,
        base_link: str,
        tip_link: str,
        tcp_offset_m: float = TCP_OFFSET_M,
        tcp_orientation_offset_quat: Optional[np.ndarray] = None,
    ) -> "KinematicsChain":
        path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(path).getroot()
        joints_by_child: Dict[str, ET.Element] = {}
        named_joint_limits: Dict[str, Tuple[float, float]] = {}
        requested_joint_names = {str(name) for name in joint_names}

        for joint_node in root.findall("joint"):
            child_node = joint_node.find("child")
            if child_node is not None and child_node.attrib.get("link"):
                joints_by_child[str(child_node.attrib["link"])] = joint_node
            name = joint_node.attrib.get("name", "")
            if name in requested_joint_names:
                limit_node = joint_node.find("limit")
                if limit_node is not None:
                    lower = float(limit_node.attrib.get("lower", str(-math.pi)))
                    upper = float(limit_node.attrib.get("upper", str(math.pi)))
                else:
                    lower = -math.pi
                    upper = math.pi
                named_joint_limits[name] = (lower, upper)

        segments_reversed: List[_ChainSegment] = []
        active_joint_names_reversed: List[str] = []
        current_link = tip_link
        while current_link != base_link:
            joint_node = joints_by_child.get(current_link)
            if joint_node is None:
                raise ValueError(f"Unable to resolve URDF chain from {base_link} to {tip_link}")
            origin_node = joint_node.find("origin")
            xyz_text = "0 0 0" if origin_node is None else origin_node.attrib.get("xyz", "0 0 0")
            rpy_text = "0 0 0" if origin_node is None else origin_node.attrib.get("rpy", "0 0 0")
            axis_node = joint_node.find("axis")
            axis_text = "0 0 1" if axis_node is None else axis_node.attrib.get("xyz", "0 0 1")
            parent_node = joint_node.find("parent")
            child_node = joint_node.find("child")
            parent_link = "" if parent_node is None else str(parent_node.attrib.get("link", ""))
            child_link = "" if child_node is None else str(child_node.attrib.get("link", ""))
            name = str(joint_node.attrib.get("name", ""))
            if name in requested_joint_names:
                active_joint_names_reversed.append(name)
            segments_reversed.append(
                _ChainSegment(
                    name=name,
                    joint_type=str(joint_node.attrib.get("type", "fixed")),
                    parent_link=parent_link,
                    child_link=child_link,
                    origin_xyz=np.asarray([float(value) for value in xyz_text.split()], dtype=float),
                    origin_rpy=np.asarray([float(value) for value in rpy_text.split()], dtype=float),
                    axis=np.asarray([float(value) for value in axis_text.split()], dtype=float),
                    controlled_index=None,
                )
            )
            current_link = parent_link
        segments = list(reversed(segments_reversed))
        active_joint_names = list(reversed(active_joint_names_reversed))
        controlled_index_map = {name: index for index, name in enumerate(active_joint_names)}
        for segment in segments:
            if segment.name in controlled_index_map:
                segment.controlled_index = controlled_index_map[segment.name]

        limits = [named_joint_limits.get(name, (-math.pi, math.pi)) for name in active_joint_names]
        return cls(
            [0.0] * len(active_joint_names),
            limits,
            tcp_offset_m=tcp_offset_m,
            joint_names=active_joint_names,
            segments=segments,
            tcp_orientation_offset_quat=tcp_orientation_offset_quat,
        )

    def clip_to_limits(self, joints: np.ndarray) -> np.ndarray:
        values = np.asarray(joints, dtype=float).reshape(self.dof).copy()
        for index, (low, high) in enumerate(self.joint_limits):
            values[index] = float(np.clip(values[index], low, high))
        return values

    def joint_frames(
        self, joints: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        values = self.clip_to_limits(joints)
        if self._segments is not None:
            transform = np.eye(4, dtype=float)
            joint_positions: List[np.ndarray] = []
            axes_world: List[np.ndarray] = []
            endpoints: List[np.ndarray] = []
            for segment in self._segments:
                transform = transform @ _transform_from_xyz_rpy(segment.origin_xyz, segment.origin_rpy)
                if segment.controlled_index is not None:
                    joint_positions.append(transform[:3, 3].copy())
                    axis_world = transform[:3, :3] @ segment.axis
                    axes_world.append(axis_world.copy())
                    transform[:3, :3] = transform[:3, :3] @ _rotation_matrix(
                        segment.axis, float(values[segment.controlled_index])
                    )
                endpoints.append(transform[:3, 3].copy())
            return joint_positions, axes_world, endpoints, transform[:3, :3].copy()
        position = np.zeros(3, dtype=float)
        rotation = np.eye(3, dtype=float)
        joint_positions: List[np.ndarray] = []
        axes_world: List[np.ndarray] = []
        endpoints: List[np.ndarray] = []

        for index, angle in enumerate(values):
            joint_positions.append(position.copy())
            axis_local = JOINT_AXES[index]
            axes_world.append(rotation @ axis_local)
            rotation = rotation @ _rotation_matrix(axis_local, float(angle))
            position = position + rotation @ np.array([0.0, 0.0, self.link_lengths[index]], dtype=float)
            endpoints.append(position.copy())
        return joint_positions, axes_world, endpoints, rotation

    def forward_kinematics(self, joints: np.ndarray) -> Dict[str, object]:
        joint_positions, _, endpoints, rotation = self.joint_frames(joints)
        base = np.zeros(3, dtype=float)
        tcp_position = endpoints[-1] + rotation @ np.array([0.0, 0.0, self.tcp_offset_m], dtype=float)
        link_positions = [base.copy()]
        link_positions.extend(point.copy() for point in endpoints[:-1])
        link_positions.append(tcp_position.copy())
        solver_quaternion = _quat_from_rotation_matrix(rotation)
        return {
            "link_positions": link_positions,
            "ee_position": tcp_position,
            "ee_quaternion": _quat_multiply(solver_quaternion, self._tcp_orientation_offset_quat),
            "solver_quaternion": solver_quaternion,
        }

    def compute_jacobian(self, joints: np.ndarray) -> np.ndarray:
        joint_positions, axes_world, endpoints, rotation = self.joint_frames(joints)
        tcp_position = endpoints[-1] + rotation @ np.array([0.0, 0.0, self.tcp_offset_m], dtype=float)
        jacobian = np.zeros((6, self.dof), dtype=float)
        for index, (joint_position, axis_world) in enumerate(zip(joint_positions, axes_world)):
            axis = np.asarray(axis_world, dtype=float)
            jacobian[:3, index] = np.cross(axis, tcp_position - joint_position)
            jacobian[3:, index] = axis
        return jacobian


class ArmTCPModel:
    """Composite arm model that matches builtin synthetic TCP behavior."""

    def __init__(
        self,
        orientation_chain: KinematicsChain,
        fingertip_chains: Optional[Sequence[KinematicsChain]] = None,
    ) -> None:
        self._orientation_chain = orientation_chain
        self._fingertip_chains = list(fingertip_chains or [])
        self._aux_joint_positions: Dict[str, float] = {}
        self.joint_limits = list(orientation_chain.joint_limits)
        self.joint_names = list(orientation_chain.joint_names)
        self.dof = int(orientation_chain.dof)

    def set_aux_joint_positions(self, joint_positions: Optional[Dict[str, float]]) -> None:
        self._aux_joint_positions = {
            str(name): float(value) for name, value in (joint_positions or {}).items()
        }

    def clip_to_limits(self, joints: np.ndarray) -> np.ndarray:
        return self._orientation_chain.clip_to_limits(joints)

    def joint_frames(
        self, joints: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray]:
        return self._orientation_chain.joint_frames(joints)

    def _compose_chain_values(self, chain: KinematicsChain, arm_joints: np.ndarray) -> np.ndarray:
        clipped_arm = self.clip_to_limits(arm_joints)
        arm_joint_map = {
            str(name): float(clipped_arm[index]) for index, name in enumerate(self.joint_names)
        }
        values = np.zeros(chain.dof, dtype=float)
        for index, name in enumerate(chain.joint_names):
            if name in arm_joint_map:
                values[index] = arm_joint_map[name]
            else:
                values[index] = float(self._aux_joint_positions.get(name, 0.0))
        return chain.clip_to_limits(values)

    def forward_kinematics(self, joints: np.ndarray) -> Dict[str, object]:
        clipped = self.clip_to_limits(joints)
        orientation_fk = self._orientation_chain.forward_kinematics(clipped)
        ee_position = np.asarray(orientation_fk["ee_position"], dtype=float)

        if self._fingertip_chains:
            fingertip_positions = []
            for chain in self._fingertip_chains:
                chain_fk = chain.forward_kinematics(self._compose_chain_values(chain, clipped))
                fingertip_positions.append(np.asarray(chain_fk["ee_position"], dtype=float))
            if fingertip_positions:
                ee_position = np.mean(np.asarray(fingertip_positions, dtype=float), axis=0)

        return {
            "link_positions": orientation_fk.get("link_positions"),
            "ee_position": ee_position,
            "ee_quaternion": np.asarray(orientation_fk["ee_quaternion"], dtype=float),
            "solver_quaternion": np.asarray(orientation_fk["solver_quaternion"], dtype=float),
        }

    def compute_jacobian(self, joints: np.ndarray) -> np.ndarray:
        clipped = self.clip_to_limits(joints)
        orientation_jacobian = self._orientation_chain.compute_jacobian(clipped)

        if not self._fingertip_chains:
            return orientation_jacobian

        position_blocks = []
        for chain in self._fingertip_chains:
            fingertip_jacobian = chain.compute_jacobian(self._compose_chain_values(chain, clipped))
            position_blocks.append(np.asarray(fingertip_jacobian[:3, : self.dof], dtype=float))
        if not position_blocks:
            return orientation_jacobian

        position_jacobian = np.mean(np.asarray(position_blocks, dtype=float), axis=0)
        return np.vstack([position_jacobian, orientation_jacobian[3:6, :]])


class RealtimeDLSIKSolver:
    """Incremental Jacobian DLS solver for realtime tracking."""

    def __init__(
        self,
        chain,
        *,
        damping: float = 0.15,
        position_gain: float = 1.2,
        orientation_gain: float = 0.6,
        max_step_rad: float = 0.08,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
    ) -> None:
        self._chain = chain
        self._damping = float(max(1e-6, damping))
        self._position_gain = float(max(1e-6, position_gain))
        self._orientation_gain = float(max(1e-6, orientation_gain))
        self._max_step_rad = float(max(1e-6, max_step_rad))
        self._position_tolerance_m = float(max(1e-9, position_tolerance_m))
        self._orientation_tolerance_rad = float(max(1e-9, orientation_tolerance_rad))

    @staticmethod
    def _objective_score(result: IKSolveResult) -> float:
        return float(result.position_error + 0.10 * result.orientation_error)

    def _evaluate(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        *,
        enable_orientation: bool,
        mode: str,
    ) -> IKSolveResult:
        clipped = self._chain.clip_to_limits(joints)
        fk = self._chain.forward_kinematics(clipped)
        position_error = float(
            np.linalg.norm(np.asarray(target_position, dtype=float) - np.asarray(fk["ee_position"], dtype=float))
        )
        orientation_error = 0.0
        if enable_orientation and target_quaternion is not None:
            orientation_error = _quat_diff_angle(
                np.asarray(target_quaternion, dtype=float),
                np.asarray(fk["ee_quaternion"], dtype=float),
            )
        success = position_error <= self._position_tolerance_m
        if enable_orientation and target_quaternion is not None:
            success = success and orientation_error <= self._orientation_tolerance_rad
        return IKSolveResult(
            positions=clipped,
            success=success,
            approximate=not success,
            position_error=position_error,
            orientation_error=orientation_error,
            mode=mode,
        )

    def solve_step(
        self,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        current_joints: np.ndarray,
        *,
        enable_orientation: bool = True,
        iterations: int = 1,
    ) -> IKSolveResult:
        clipped = self._chain.clip_to_limits(current_joints)
        mode = "full" if enable_orientation and target_quaternion is not None else "position"
        best = self._evaluate(
            clipped,
            target_position,
            target_quaternion,
            enable_orientation=bool(enable_orientation and target_quaternion is not None),
            mode=mode,
        )
        iterations = max(1, int(iterations))
        working = clipped.copy()

        for _ in range(iterations):
            fk = self._chain.forward_kinematics(working)
            position_error = np.asarray(target_position, dtype=float) - np.asarray(
                fk["ee_position"], dtype=float
            )
            if enable_orientation and target_quaternion is not None:
                orientation_error = _orientation_error_vector(
                    np.asarray(target_quaternion, dtype=float),
                    np.asarray(fk["ee_quaternion"], dtype=float),
                )
                task_error = np.concatenate(
                    [self._position_gain * position_error, self._orientation_gain * orientation_error],
                    axis=0,
                )
                jacobian = self._chain.compute_jacobian(working)
            else:
                task_error = self._position_gain * position_error
                jacobian = self._chain.compute_jacobian(working)[:3, :]

            jjt = jacobian @ jacobian.T
            damp = self._damping
            try:
                delta = jacobian.T @ np.linalg.solve(
                    jjt + (damp * damp) * np.eye(jjt.shape[0], dtype=float),
                    task_error,
                )
            except np.linalg.LinAlgError:
                break

            delta = np.clip(np.asarray(delta, dtype=float), -self._max_step_rad, self._max_step_rad)
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.1):
                candidate = self._evaluate(
                    working + scale * delta,
                    target_position,
                    target_quaternion,
                    enable_orientation=bool(enable_orientation and target_quaternion is not None),
                    mode=mode,
                )
                if self._objective_score(candidate) < self._objective_score(best):
                    best = candidate
                    working = candidate.positions.copy()
                    improved = True
                    break
            if not improved:
                break
        return best


class FABRIKSolver:
    """Position-first iterative solver using CCD-style updates."""

    def __init__(
        self,
        chain: KinematicsChain,
        *,
        max_iterations: int = 40,
        tolerance_m: float = 1e-4,
    ) -> None:
        self._chain = chain
        self._max_iterations = int(max_iterations)
        self._tolerance_m = float(tolerance_m)

    def solve(self, target_position: np.ndarray, current_joints: np.ndarray) -> IKSolveResult:
        target = np.asarray(target_position, dtype=float).reshape(3)
        joints = self._chain.clip_to_limits(current_joints)
        best = joints.copy()
        best_error = float("inf")

        for _ in range(self._max_iterations):
            fk = self._chain.forward_kinematics(joints)
            ee_position = np.asarray(fk["ee_position"], dtype=float)
            error = float(np.linalg.norm(target - ee_position))
            if error < best_error:
                best_error = error
                best = joints.copy()
            if error <= self._tolerance_m:
                return IKSolveResult(
                    positions=joints,
                    success=True,
                    approximate=False,
                    position_error=error,
                    orientation_error=0.0,
                    mode="position",
                )

            joint_positions, axes_world, _, _ = self._chain.joint_frames(joints)
            for index in range(len(joint_positions) - 1, -1, -1):
                joint_position = joint_positions[index]
                axis_world = axes_world[index]
                current_vec = ee_position - joint_position
                target_vec = target - joint_position
                axis_norm = float(np.linalg.norm(axis_world))
                current_norm = float(np.linalg.norm(current_vec))
                target_norm = float(np.linalg.norm(target_vec))
                if axis_norm < 1e-9 or current_norm < 1e-9 or target_norm < 1e-9:
                    continue
                axis_unit = axis_world / axis_norm
                current_proj = current_vec - axis_unit * float(np.dot(current_vec, axis_unit))
                target_proj = target_vec - axis_unit * float(np.dot(target_vec, axis_unit))
                current_proj_norm = float(np.linalg.norm(current_proj))
                target_proj_norm = float(np.linalg.norm(target_proj))
                if current_proj_norm < 1e-9 or target_proj_norm < 1e-9:
                    continue
                current_unit = current_proj / current_proj_norm
                target_unit = target_proj / target_proj_norm
                dot = float(np.clip(np.dot(current_unit, target_unit), -1.0, 1.0))
                cross = np.cross(current_unit, target_unit)
                signed = float(np.dot(axis_unit, cross))
                angle = float(math.atan2(signed, dot))
                angle = float(np.clip(angle, -0.35, 0.35))
                joints[index] += angle
                joints = self._chain.clip_to_limits(joints)
                ee_position = np.asarray(self._chain.forward_kinematics(joints)["ee_position"], dtype=float)

        return IKSolveResult(
            positions=best,
            success=best_error <= self._tolerance_m,
            approximate=True,
            position_error=best_error,
            orientation_error=0.0,
            mode="position",
        )


class LBFGSBSolver:
    """Joint-space refinement with bounds."""

    def __init__(
        self,
        chain: KinematicsChain,
        *,
        max_iterations: int = 80,
        damping_lambda: float = 1e-3,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
    ) -> None:
        self._chain = chain
        self._max_iterations = int(max_iterations)
        self._damping_lambda = float(damping_lambda)
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        initial_joints: np.ndarray,
        *,
        reference_joints: Optional[np.ndarray] = None,
        enable_orientation: bool = True,
    ) -> IKSolveResult:
        target_pos = np.asarray(target_position, dtype=float).reshape(3)
        target_quat = None if target_quaternion is None else _normalize_quaternion(target_quaternion)
        initial = self._chain.clip_to_limits(initial_joints)
        reference = initial if reference_joints is None else self._chain.clip_to_limits(reference_joints)

        def objective(values: np.ndarray) -> float:
            joints = self._chain.clip_to_limits(values)
            fk = self._chain.forward_kinematics(joints)
            position_error = float(
                np.linalg.norm(np.asarray(fk["ee_position"], dtype=float) - target_pos)
            )
            orientation_error = 0.0
            if enable_orientation and target_quat is not None:
                orientation_error = _quat_diff_angle(
                    np.asarray(fk["ee_quaternion"], dtype=float),
                    target_quat,
                )
            damping = float(np.linalg.norm(joints - reference))
            return (
                position_error * position_error
                + 0.10 * orientation_error * orientation_error
                + self._damping_lambda * damping * damping
            )

        if _HAS_SCIPY:
            result = minimize(
                objective,
                initial,
                method="L-BFGS-B",
                bounds=self._chain.joint_limits,
                options={"maxiter": self._max_iterations},
            )
            joints = self._chain.clip_to_limits(result.x)
        else:
            joints = initial.copy()

        fk = self._chain.forward_kinematics(joints)
        position_error = float(np.linalg.norm(np.asarray(fk["ee_position"], dtype=float) - target_pos))
        orientation_error = 0.0
        if enable_orientation and target_quat is not None:
            orientation_error = _quat_diff_angle(
                np.asarray(fk["ee_quaternion"], dtype=float),
                target_quat,
            )

        success = position_error <= self._position_tolerance_m
        if enable_orientation and target_quat is not None:
            success = success and orientation_error <= self._orientation_tolerance_rad
        return IKSolveResult(
            positions=joints,
            success=success,
            approximate=not success,
            position_error=position_error,
            orientation_error=orientation_error,
            mode="full" if enable_orientation and target_quat is not None else "position",
        )


class HybridIKSolver:
    """Position-first seed plus bounded refinement."""

    def __init__(
        self,
        chain: KinematicsChain,
        *,
        fabrik_iterations: int = 40,
        lbfgsb_maxiter: int = 80,
        damping_lambda: float = 1e-3,
        position_tolerance_m: float = 1e-4,
        orientation_tolerance_rad: float = 1e-3,
    ) -> None:
        self._chain = chain
        self._damping_lambda = float(damping_lambda)
        self._position_tolerance_m = float(position_tolerance_m)
        self._orientation_tolerance_rad = float(orientation_tolerance_rad)
        self._fabrik = FABRIKSolver(
            chain,
            max_iterations=fabrik_iterations,
            tolerance_m=position_tolerance_m,
        )
        self._lbfgsb = LBFGSBSolver(
            chain,
            max_iterations=lbfgsb_maxiter,
            damping_lambda=damping_lambda,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

    def _seed_candidates(self, current_joints: np.ndarray) -> List[np.ndarray]:
        current = self._chain.clip_to_limits(current_joints)
        midpoint = self._chain.clip_to_limits(
            np.array([(low + high) * 0.5 for low, high in self._chain.joint_limits], dtype=float)
        )
        zero = self._chain.clip_to_limits(np.zeros(self._chain.dof, dtype=float))
        candidates: List[np.ndarray] = []
        for seed in (current, midpoint, zero):
            if any(np.allclose(seed, existing, atol=1e-6) for existing in candidates):
                continue
            candidates.append(seed.copy())
        return candidates

    def _evaluate_result(
        self,
        joints: np.ndarray,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        *,
        enable_orientation: bool,
        mode: str,
    ) -> IKSolveResult:
        clipped = self._chain.clip_to_limits(joints)
        fk = self._chain.forward_kinematics(clipped)
        position_error = float(
            np.linalg.norm(np.asarray(fk["ee_position"], dtype=float) - np.asarray(target_position, dtype=float))
        )
        orientation_error = 0.0
        if enable_orientation and target_quaternion is not None:
            orientation_error = _quat_diff_angle(
                np.asarray(fk["ee_quaternion"], dtype=float),
                np.asarray(target_quaternion, dtype=float),
            )
        success = position_error <= self._position_tolerance_m
        if enable_orientation and target_quaternion is not None:
            success = success and orientation_error <= self._orientation_tolerance_rad
        return IKSolveResult(
            positions=clipped,
            success=success,
            approximate=not success,
            position_error=position_error,
            orientation_error=orientation_error,
            mode=mode,
        )

    def _objective_score(self, result: IKSolveResult) -> float:
        return float(result.position_error + 0.10 * result.orientation_error)

    def _dls_polish(
        self,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        joints: np.ndarray,
        *,
        enable_orientation: bool,
        iterations: int = 20,
    ) -> np.ndarray:
        values = self._chain.clip_to_limits(joints)
        target_pos = np.asarray(target_position, dtype=float).reshape(3)
        target_quat = (
            None if target_quaternion is None else _normalize_quaternion(np.asarray(target_quaternion, dtype=float))
        )
        for _ in range(max(0, int(iterations))):
            fk = self._chain.forward_kinematics(values)
            current_pos = np.asarray(fk["ee_position"], dtype=float)
            current_quat = np.asarray(fk["ee_quaternion"], dtype=float)
            pos_error = target_pos - current_pos
            if enable_orientation and target_quat is not None:
                ori_error = _orientation_error_vector(target_quat, current_quat)
                task_error = np.concatenate((pos_error, ori_error), axis=0)
                jacobian = self._chain.compute_jacobian(values)
            else:
                task_error = pos_error
                jacobian = self._chain.compute_jacobian(values)[:3, :]
            if float(np.linalg.norm(task_error)) < 1e-6:
                break
            jjt = jacobian @ jacobian.T
            damp = float(max(1e-6, self._damping_lambda))
            delta = jacobian.T @ np.linalg.solve(
                jjt + (damp * damp) * np.eye(jjt.shape[0], dtype=float),
                task_error,
            )
            delta = np.clip(delta, -0.08, 0.08)
            values = self._chain.clip_to_limits(values + delta)
        return values

    def solve(
        self,
        target_position: np.ndarray,
        target_quaternion: Optional[np.ndarray],
        current_joints: np.ndarray,
        *,
        enable_orientation: bool = True,
    ) -> IKSolveResult:
        seeds = self._seed_candidates(current_joints)
        best_result: Optional[IKSolveResult] = None
        reference = self._chain.clip_to_limits(current_joints)

        for base_seed in seeds:
            seed_result = self._fabrik.solve(target_position, base_seed)
            seed = seed_result.positions
            seed_position_result = self._evaluate_result(
                seed,
                target_position,
                None,
                enable_orientation=False,
                mode="position_fallback" if target_quaternion is not None else "position",
            )

            if not enable_orientation or target_quaternion is None:
                if best_result is None or (
                    seed_position_result.success and not best_result.success
                ) or (
                    seed_position_result.success == best_result.success
                    and self._objective_score(seed_position_result) < self._objective_score(best_result)
                ):
                    best_result = seed_position_result
                if seed_position_result.success:
                    return seed_position_result

                polished_seed = self._dls_polish(
                    target_position,
                    None,
                    seed,
                    enable_orientation=False,
                    iterations=8,
                )
                polished_seed_result = self._evaluate_result(
                    polished_seed,
                    target_position,
                    None,
                    enable_orientation=False,
                    mode="position",
                )
                if best_result is None or (
                    polished_seed_result.success and not best_result.success
                ) or (
                    polished_seed_result.success == best_result.success
                    and self._objective_score(polished_seed_result) < self._objective_score(best_result)
                ):
                    best_result = polished_seed_result
                if polished_seed_result.success:
                    return polished_seed_result

            if enable_orientation and target_quaternion is not None:
                fast_full = self._dls_polish(
                    target_position,
                    target_quaternion,
                    seed,
                    enable_orientation=True,
                    iterations=12,
                )
                fast_full_result = self._evaluate_result(
                    fast_full,
                    target_position,
                    target_quaternion,
                    enable_orientation=True,
                    mode="full",
                )
                if best_result is None or (
                    fast_full_result.success and not best_result.success
                ) or (
                    fast_full_result.success == best_result.success
                    and self._objective_score(fast_full_result) < self._objective_score(best_result)
                ):
                    best_result = fast_full_result
                if fast_full_result.success:
                    return fast_full_result

                full_result = self._lbfgsb.solve(
                    target_position,
                    target_quaternion,
                    seed,
                    reference_joints=reference,
                    enable_orientation=True,
                )
                polished_full = self._dls_polish(
                    target_position,
                    target_quaternion,
                    full_result.positions,
                    enable_orientation=True,
                )
                full_result = self._evaluate_result(
                    polished_full,
                    target_position,
                    target_quaternion,
                    enable_orientation=True,
                    mode="full",
                )
                if best_result is None or (
                    full_result.success and not best_result.success
                ) or (
                    full_result.success == best_result.success
                    and self._objective_score(full_result) < self._objective_score(best_result)
                ):
                    best_result = full_result
                if full_result.success:
                    return full_result

            position_result = self._lbfgsb.solve(
                target_position,
                None,
                seed,
                reference_joints=reference,
                enable_orientation=False,
            )
            polished_position = self._dls_polish(
                target_position,
                None,
                position_result.positions,
                enable_orientation=False,
            )
            position_result = self._evaluate_result(
                polished_position,
                target_position,
                None,
                enable_orientation=False,
                mode="position_fallback" if target_quaternion is not None else "position",
            )
            if best_result is None or (
                position_result.success and not best_result.success
            ) or (
                position_result.success == best_result.success
                and self._objective_score(position_result) < self._objective_score(best_result)
            ):
                best_result = position_result

        if best_result is not None:
            return best_result

        fallback = self._evaluate_result(
            reference,
            target_position,
            target_quaternion if enable_orientation else None,
            enable_orientation=bool(enable_orientation and target_quaternion is not None),
            mode="position_fallback" if target_quaternion is not None else "position",
        )
        fallback.approximate = True
        fallback.success = False
        return fallback
