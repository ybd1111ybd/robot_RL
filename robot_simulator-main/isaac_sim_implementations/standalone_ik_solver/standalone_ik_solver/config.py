"""Configuration for the standalone IK solver."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple


DEFAULT_LINK_LENGTHS: List[float] = [0.34, 0.32, 0.33, 0.33, 0.32, 0.22, 0.14]
DEFAULT_JOINT_LIMITS: List[Tuple[float, float]] = [
    (-2.00, 2.00),
    (-1.57, 1.57),
    (-2.09, 2.09),
    (-2.09, 2.09),
    (-2.09, 2.09),
    (-2.09, 2.09),
    (-1.05, 1.05),
]


def _load_link_lengths() -> List[float]:
    raw = os.environ.get("STANDALONE_IK_LINK_LENGTHS", "").strip()
    if not raw:
        return list(DEFAULT_LINK_LENGTHS)
    try:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError:
        return list(DEFAULT_LINK_LENGTHS)
    return values if len(values) == 7 else list(DEFAULT_LINK_LENGTHS)


def _load_joint_limits() -> List[Tuple[float, float]]:
    raw = os.environ.get("STANDALONE_IK_JOINT_LIMITS", "").strip()
    if not raw:
        return list(DEFAULT_JOINT_LIMITS)
    pairs: List[Tuple[float, float]] = []
    try:
        for item in raw.split(","):
            low_text, high_text = item.split(":", 1)
            pairs.append((float(low_text.strip()), float(high_text.strip())))
    except ValueError:
        return list(DEFAULT_JOINT_LIMITS)
    return pairs if len(pairs) == 7 else list(DEFAULT_JOINT_LIMITS)


def _detect_repo_root() -> Path | None:
    env_root = os.environ.get("JZ_REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "jz_descripetion").exists():
            return candidate
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "jz_descripetion").exists() and (parent / "robot_simulator").exists():
            return parent
    return None


def _default_urdf_path() -> str:
    override = os.environ.get("STANDALONE_IK_URDF_PATH", "").strip()
    if override:
        return str(Path(override).expanduser().resolve())
    repo_root = _detect_repo_root()
    if repo_root is None:
        return ""
    candidate = repo_root / "jz_descripetion/robot_urdf/urdf/robot_model_isaac_v2.urdf"
    return str(candidate) if candidate.exists() else ""


@dataclass
class SolverConfig:
    ik_rate_hz: float = 50.0
    target_smoothing_alpha: float = 0.2
    max_target_position_step_m: float = 0.015
    max_target_orientation_step_rad: float = 0.10
    close_range_orientation_distance_m: float = 0.025
    close_range_max_orientation_step_rad: float = 0.07
    max_joint_delta_rad: float = 0.12
    max_joint_position_abs_rad: float = 6.0
    enable_orientation: bool = True
    enable_target_smoothing: bool = True
    ik_timeout_sec: float = 0.5
    orientation_position_gate_m: float = 0.05
    approx_orientation_accept_position_error_m: float = 0.02
    fabrik_iterations: int = 40
    lbfgsb_maxiter: int = 80
    damping_lambda: float = 1e-3
    realtime_dls_damping: float = 0.15
    realtime_dls_position_gain: float = 1.2
    realtime_dls_orientation_gain: float = 0.6
    realtime_dls_max_step_rad: float = 0.12
    realtime_dls_near_target_distance_m: float = 0.06
    realtime_dls_near_target_substeps: int = 2
    position_tolerance_m: float = 1e-4
    orientation_tolerance_rad: float = 1e-3
    feedback_frame_id: str = "base_link"
    left_solver_base_link: str = "body_link3"
    right_solver_base_link: str = "body_link3"
    urdf_path: str = field(default_factory=_default_urdf_path)
    left_arm_joints: List[str] = field(
        default_factory=lambda: [f"left_arm_joint{i}" for i in range(1, 8)]
    )
    right_arm_joints: List[str] = field(
        default_factory=lambda: [f"right_arm_joint{i}" for i in range(1, 8)]
    )
    left_gripper_joints: List[str] = field(
        default_factory=lambda: ["left_gripper_narrow_joint", "left_gripper_wide_joint"]
    )
    right_gripper_joints: List[str] = field(
        default_factory=lambda: ["right_gripper_narrow_joint", "right_gripper_wide_joint"]
    )
    link_lengths: List[float] = field(default_factory=_load_link_lengths)
    joint_limits: List[Tuple[float, float]] = field(default_factory=_load_joint_limits)

    left_ee_topic: str = "/arm_left/ee_target_pose"
    right_ee_topic: str = "/arm_right/ee_target_pose"
    aggregate_state_topic: str = "/joint_states"
    body_state_topic: str = "/body/joint_states"
    left_state_topic: str = "/arm_left/joint_states"
    right_state_topic: str = "/arm_right/joint_states"
    left_cmd_topic: str = "/arm_left/joint_commands"
    right_cmd_topic: str = "/arm_right/joint_commands"
    left_pose_topic: str = "/arm_left/ee_current_pose"
    right_pose_topic: str = "/arm_right/ee_current_pose"
    left_status_topic: str = "/arm_left/ee_ik_status"
    right_status_topic: str = "/arm_right/ee_ik_status"
    body_joints: List[str] = field(default_factory=lambda: [f"body_joint{i}" for i in range(1, 6)])
