from __future__ import annotations

from pathlib import Path
import argparse
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from isaacsim import SimulationApp
from config_loader import load_toml_config, load_yaml_config
from runtime.demo_grasp import MainlineGraspDemo
from runtime.mainline_shared import BODY_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from runtime.ros_components import (
    BodyPostureKeeper,
    GripperSwitchRemapper,
    ManualJointStatePublisher,
)


SCRIPT_DIR = Path(__file__).resolve().parent
ACTION_GRAPH_DIR = SCRIPT_DIR.parent
DEFAULT_RUNTIME_CONFIG = ACTION_GRAPH_DIR / "config" / "runtime_defaults.toml"
DEFAULT_BUILTIN_IK_CONFIG = ACTION_GRAPH_DIR / "config" / "builtin_ik.yaml"


def detect_repo_root() -> Path:
    env_root = os.environ.get("JZ_REPO_ROOT", "").strip()
    if env_root:
        p = Path(env_root).expanduser().resolve()
        if (p / "jz_descripetion").exists():
            return p
    for parent in [ACTION_GRAPH_DIR] + list(ACTION_GRAPH_DIR.parents):
        if (parent / "jz_descripetion").exists() and (parent / "robot_simulator").exists():
            return parent
    return ACTION_GRAPH_DIR.parents[2]


REPO_ROOT = detect_repo_root()
DEFAULT_URDF = REPO_ROOT / "jz_descripetion/robot_urdf/urdf/robot_model_isaac_v2.urdf"
DEFAULT_LULA_LEFT_DESC = ACTION_GRAPH_DIR / "builtin_ik_solver/config/jz_left_arm_robot_description.yaml"
DEFAULT_LULA_RIGHT_DESC = ACTION_GRAPH_DIR / "builtin_ik_solver/config/jz_right_arm_robot_description.yaml"


def _as_usd_fs_path(path: Path) -> str:
    return path.resolve().as_posix()


def resolve_input_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return (REPO_ROOT / path).resolve()


def _resolve_config_path(raw_value: str, default_path: Path) -> Path:
    text = (raw_value or "").strip()
    if not text:
        return default_path
    return resolve_input_path(text)


def _nested_get(mapping: Dict[str, Any], path: Tuple[str, ...], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_string_list(value: Any, default: List[str]) -> List[str]:
    if not isinstance(value, list):
        return list(default)
    items: List[str] = []
    for raw in value:
        text = str(raw).strip()
        if text:
            items.append(text)
    return items or list(default)


def _parse_xyz_arg(raw_value: str, default: Tuple[float, float, float]) -> np.ndarray:
    text = (raw_value or "").strip()
    if not text:
        return np.array(default, dtype=float)
    parts = [item.strip() for item in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected x,y,z but got '{raw_value}'")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def _read_package_name(package_xml: Path, fallback: str) -> str:
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(package_xml).getroot()
        node = root.find("name")
        if node is not None and node.text:
            name = node.text.strip()
            if name:
                return name
    except Exception:
        pass
    return fallback


def _prepend_env_paths(var_name: str, candidates: List[Path]) -> None:
    sep = ";" if os.name == "nt" else ":"
    cur = os.environ.get(var_name, "").strip()
    existing = [p for p in cur.split(sep) if p] if cur else []
    ordered: List[str] = []
    seen = set()
    for path in candidates:
        if not path.exists():
            continue
        text = str(path.resolve())
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    for raw in existing:
        raw_path = Path(raw)
        text = str(raw_path.resolve()) if raw_path.exists() else raw
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    if ordered:
        os.environ[var_name] = sep.join(ordered)


def ensure_ros_package_path_for_urdf(urdf_path: Path) -> None:
    package_root = urdf_path.parent.parent
    package_xml = package_root / "package.xml"
    if not package_xml.exists():
        return
    package_name = _read_package_name(package_xml, fallback=package_root.name)
    package_parent = package_root.parent
    candidates: List[Path] = []
    if package_root.name == package_name or (package_parent / package_name).exists():
        candidates.append(package_parent)
    candidates.append(package_root)
    _prepend_env_paths("ROS_PACKAGE_PATH", candidates)
    _prepend_env_paths("AMENT_PREFIX_PATH", candidates)
    _prepend_env_paths("CMAKE_PREFIX_PATH", candidates)
    _prepend_env_paths("COLCON_PREFIX_PATH", candidates)


def _discover_package_root(package_name: str) -> Optional[Path]:
    sep = ";" if os.name == "nt" else ":"
    candidates: List[Path] = []
    ros_package_path = os.environ.get("ROS_PACKAGE_PATH", "").strip()
    if ros_package_path:
        for raw in [p for p in ros_package_path.split(sep) if p]:
            base = Path(raw)
            candidates.append(base / package_name)
            candidates.append(base)
    ament_prefix_path = os.environ.get("AMENT_PREFIX_PATH", "").strip()
    if ament_prefix_path:
        for raw in [p for p in ament_prefix_path.split(sep) if p]:
            base = Path(raw)
            candidates.append(base / "share" / package_name)

    for candidate in candidates:
        if not candidate.exists():
            continue
        package_xml = candidate / "package.xml"
        if package_xml.exists():
            if _read_package_name(package_xml, fallback=candidate.name) == package_name:
                return candidate
        if candidate.name == package_name and (candidate / "meshes").exists():
            return candidate
    return None


def _make_mesh_resolved_urdf(urdf_path: Path) -> Path:
    content = urdf_path.read_text(encoding="utf-8")
    package_names = set()
    for token in content.split("package://")[1:]:
        package_names.add(token.split("/", 1)[0])

    resolved: Dict[str, Path] = {}
    for name in package_names:
        root = _discover_package_root(name)
        if root is not None:
            resolved[name] = root

    for name, root in resolved.items():
        prefix = root.resolve().as_posix().rstrip("/") + "/"
        content = content.replace(f"package://{name}/", prefix)

    temp_dir = Path(tempfile.gettempdir()) / "jz_robot_urdf_cache"
    temp_dir.mkdir(parents=True, exist_ok=True)
    out_path = temp_dir / f"{urdf_path.stem}.isaac_resolved.urdf"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def configure_ros2_bridge_env() -> None:
    if not os.environ.get("ROS_DISTRO"):
        os.environ["ROS_DISTRO"] = "humble"
    if not os.environ.get("RMW_IMPLEMENTATION"):
        os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

    isaac_root = os.environ.get("ISAAC_SIM_PATH", "").strip()
    if not isaac_root:
        exe_path = Path(sys.executable).resolve()
        for parent in exe_path.parents:
            if (parent / "exts" / "isaacsim.ros2.bridge").exists():
                isaac_root = str(parent)
                break
    if not isaac_root:
        return
    ros2_lib_dir = Path(isaac_root) / "exts" / "isaacsim.ros2.bridge" / "humble" / "lib"
    if not ros2_lib_dir.exists():
        return
    sep = ";" if os.name == "nt" else ":"
    cur = os.environ.get("PATH", "")
    ros2_lib_str = str(ros2_lib_dir)
    if ros2_lib_str not in cur:
        os.environ["PATH"] = f"{cur}{sep}{ros2_lib_str}" if cur else ros2_lib_str


def _load_mainline_configs() -> Tuple[Path, Dict[str, Any], Path, Dict[str, Any]]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--runtime-config", type=str, default=os.environ.get("JZ_RUNTIME_CONFIG", ""))
    pre_parser.add_argument("--ik-config", type=str, default=os.environ.get("JZ_IK_CONFIG", ""))
    pre_args, _ = pre_parser.parse_known_args()

    runtime_config_path = _resolve_config_path(pre_args.runtime_config, DEFAULT_RUNTIME_CONFIG)
    ik_config_path = _resolve_config_path(pre_args.ik_config, DEFAULT_BUILTIN_IK_CONFIG)
    runtime_config = load_toml_config(runtime_config_path)
    ik_config = load_yaml_config(ik_config_path)
    return runtime_config_path, runtime_config, ik_config_path, ik_config


RUNTIME_CONFIG_PATH, _RUNTIME_CONFIG_DATA, IK_CONFIG_PATH, _IK_CONFIG_DATA = _load_mainline_configs()

RUNTIME_DEFAULTS = {
    "fix_base": bool(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "fix_base"), True)),
    "domain_id": int(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "domain_id"), -1)),
    "graph_path": str(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "graph_path"), "/ActionGraph")),
    "node_namespace": str(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "node_namespace"), "")),
    "manual_joint_state_rate": float(
        _nested_get(_RUNTIME_CONFIG_DATA, ("app", "manual_joint_state_rate"), 30.0)
    ),
    "control_mode": str(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "control_mode"), "ee_pose")),
    "ik_rate_hz": float(_nested_get(_RUNTIME_CONFIG_DATA, ("app", "ik_rate_hz"), 50.0)),
    "arm_left_cmd_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "arm_left_cmd_topic"), "/arm_left/joint_commands")
    ),
    "arm_right_cmd_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "arm_right_cmd_topic"), "/arm_right/joint_commands")
    ),
    "body_cmd_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "body_cmd_topic"), "/body/joint_commands")
    ),
    "arm_left_state_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "arm_left_state_topic"), "/arm_left/joint_states")
    ),
    "arm_right_state_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "arm_right_state_topic"), "/arm_right/joint_states")
    ),
    "body_state_topic": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("topics", "body_state_topic"), "/body/joint_states")
    ),
    "hold_enabled": bool(_nested_get(_RUNTIME_CONFIG_DATA, ("hold", "arms", "enabled"), True)),
    "hold_stiffness": float(_nested_get(_RUNTIME_CONFIG_DATA, ("hold", "arms", "stiffness"), 20000.0)),
    "hold_damping": float(_nested_get(_RUNTIME_CONFIG_DATA, ("hold", "arms", "damping"), 1500.0)),
    "hold_max_force": float(_nested_get(_RUNTIME_CONFIG_DATA, ("hold", "arms", "max_force"), 10000.0)),
    "body_hold_stiffness": float(
        _nested_get(_RUNTIME_CONFIG_DATA, ("hold", "body", "stiffness"), 60000.0)
    ),
    "body_hold_damping": float(
        _nested_get(_RUNTIME_CONFIG_DATA, ("hold", "body", "damping"), 4000.0)
    ),
    "body_hold_max_force": float(
        _nested_get(_RUNTIME_CONFIG_DATA, ("hold", "body", "max_force"), 40000.0)
    ),
    "body_posture_hold": bool(
        _nested_get(_RUNTIME_CONFIG_DATA, ("hold", "body", "posture_hold"), True)
    ),
    "demo_grasp_arm": str(_nested_get(_RUNTIME_CONFIG_DATA, ("demo_grasp", "arm"), "right")),
    "demo_grasp_assist_attach": bool(
        _nested_get(_RUNTIME_CONFIG_DATA, ("demo_grasp", "assist_attach"), True)
    ),
    "demo_grasp_trigger_key": str(
        _nested_get(_RUNTIME_CONFIG_DATA, ("demo_grasp", "trigger_key"), "Q")
    ),
}

BUILTIN_IK_DEFAULTS = {
    "left_ee_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "left_ee_topic"), "/arm_left/ee_target_pose")),
    "right_ee_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "right_ee_topic"), "/arm_right/ee_target_pose")),
    "left_pose_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "left_pose_topic"), "/arm_left/ee_current_pose")),
    "right_pose_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "right_pose_topic"), "/arm_right/ee_current_pose")),
    "left_status_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "left_status_topic"), "/arm_left/ee_ik_status")),
    "right_status_topic": str(_nested_get(_IK_CONFIG_DATA, ("topics", "right_status_topic"), "/arm_right/ee_ik_status")),
    "left_ee_link": str(_nested_get(_IK_CONFIG_DATA, ("links", "left_ee_link"), "left_gripper_center_tcp")),
    "right_ee_link": str(_nested_get(_IK_CONFIG_DATA, ("links", "right_ee_link"), "right_gripper_center_tcp")),
    "left_solver_base_link": str(
        _nested_get(_IK_CONFIG_DATA, ("links", "left_solver_base_link"), "body_link3")
    ),
    "right_solver_base_link": str(
        _nested_get(_IK_CONFIG_DATA, ("links", "right_solver_base_link"), "body_link3")
    ),
    "left_arm_joints": _as_string_list(
        _nested_get(
            _IK_CONFIG_DATA,
            ("arms", "left_arm_joints"),
            [f"left_arm_joint{i}" for i in range(1, 8)],
        ),
        [f"left_arm_joint{i}" for i in range(1, 8)],
    ),
    "right_arm_joints": _as_string_list(
        _nested_get(
            _IK_CONFIG_DATA,
            ("arms", "right_arm_joints"),
            [f"right_arm_joint{i}" for i in range(1, 8)],
        ),
        [f"right_arm_joint{i}" for i in range(1, 8)],
    ),
    "ik_timeout_sec": float(_nested_get(_IK_CONFIG_DATA, ("solver", "timeout_sec"), 1.0)),
    "enable_orientation": _nested_get(_IK_CONFIG_DATA, ("solver", "enable_orientation"), None),
    "enable_target_smoothing": bool(
        _nested_get(_IK_CONFIG_DATA, ("solver", "enable_target_smoothing"), True)
    ),
    "target_smoothing_alpha": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "target_smoothing_alpha"), 0.35)
    ),
    "max_target_position_step_m": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "max_target_position_step_m"), 0.02)
    ),
    "max_target_orientation_step_rad": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "max_target_orientation_step_rad"), 0.20)
    ),
    "direct_apply_joint_targets": bool(
        _nested_get(_IK_CONFIG_DATA, ("solver", "direct_apply_joint_targets"), True)
    ),
    "max_joint_delta_rad": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "max_joint_delta_rad"), 0.15)
    ),
    "max_joint_position_abs_rad": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "max_joint_position_abs_rad"), 6.0)
    ),
    "orientation_position_gate_m": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "orientation_position_gate_m"), 0.06)
    ),
    "approx_orientation_accept_position_error_m": float(
        _nested_get(
            _IK_CONFIG_DATA,
            ("solver", "approx_orientation_accept_position_error_m"),
            0.03,
        )
    ),
    "close_range_orientation_distance_m": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "close_range_orientation_distance_m"), 0.03)
    ),
    "close_range_max_orientation_step_rad": float(
        _nested_get(_IK_CONFIG_DATA, ("solver", "close_range_max_orientation_step_rad"), 0.08)
    ),
    "feedback_frame_id": str(_nested_get(_IK_CONFIG_DATA, ("feedback", "frame_id"), "base_link")),
}


parser = argparse.ArgumentParser(
    description="Minimal Lula Action Graph runner (target pose + joint commands)."
)
parser.add_argument("--runtime-config", type=str, default=str(RUNTIME_CONFIG_PATH))
parser.add_argument("--ik-config", type=str, default=str(IK_CONFIG_PATH))
parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--steps", type=int, default=0)
parser.add_argument("--hold", action="store_true")
parser.add_argument(
    "--fix-base",
    dest="fix_base",
    action="store_true",
    help="Fix the robot base to the world frame (default).",
)
parser.add_argument(
    "--free-base",
    dest="fix_base",
    action="store_false",
    help="Allow the robot base to move under physics.",
)
parser.add_argument("--domain-id", type=int, default=RUNTIME_DEFAULTS["domain_id"])
parser.add_argument("--graph-path", type=str, default=RUNTIME_DEFAULTS["graph_path"])
parser.add_argument("--node-namespace", type=str, default=RUNTIME_DEFAULTS["node_namespace"])
parser.set_defaults(fix_base=RUNTIME_DEFAULTS["fix_base"])

parser.add_argument("--arm-left-cmd-topic", type=str, default=RUNTIME_DEFAULTS["arm_left_cmd_topic"])
parser.add_argument("--arm-right-cmd-topic", type=str, default=RUNTIME_DEFAULTS["arm_right_cmd_topic"])
parser.add_argument("--body-cmd-topic", type=str, default=RUNTIME_DEFAULTS["body_cmd_topic"])
parser.add_argument("--arm-left-state-topic", type=str, default=RUNTIME_DEFAULTS["arm_left_state_topic"])
parser.add_argument("--arm-right-state-topic", type=str, default=RUNTIME_DEFAULTS["arm_right_state_topic"])
parser.add_argument("--body-state-topic", type=str, default=RUNTIME_DEFAULTS["body_state_topic"])
parser.add_argument("--manual-joint-state-rate", type=float, default=RUNTIME_DEFAULTS["manual_joint_state_rate"])

parser.add_argument(
    "--control-mode",
    type=str,
    default=RUNTIME_DEFAULTS["control_mode"],
    choices=["joint", "ee_pose", "auto"],
)
parser.add_argument("--left-ee-topic", type=str, default=BUILTIN_IK_DEFAULTS["left_ee_topic"])
parser.add_argument("--right-ee-topic", type=str, default=BUILTIN_IK_DEFAULTS["right_ee_topic"])
parser.add_argument("--left-ee-link", type=str, default=BUILTIN_IK_DEFAULTS["left_ee_link"])
parser.add_argument("--right-ee-link", type=str, default=BUILTIN_IK_DEFAULTS["right_ee_link"])
parser.add_argument(
    "--ik-enable-orientation",
    dest="ik_enable_orientation",
    action="store_true",
    help="Force full 6DoF IK orientation tracking on.",
)
parser.add_argument(
    "--ik-disable-orientation",
    dest="ik_enable_orientation",
    action="store_false",
    help="Disable orientation tracking and use position-only IK.",
)
parser.add_argument("--ik-rate-hz", type=float, default=RUNTIME_DEFAULTS["ik_rate_hz"])
parser.add_argument("--ik-timeout-sec", type=float, default=BUILTIN_IK_DEFAULTS["ik_timeout_sec"])
parser.add_argument(
    "--feedback-frame-id",
    type=str,
    default=BUILTIN_IK_DEFAULTS["feedback_frame_id"],
    help="Frame used for /arm_*/ee_current_pose feedback topics.",
)
parser.add_argument(
    "--target-smoothing-alpha",
    type=float,
    default=BUILTIN_IK_DEFAULTS["target_smoothing_alpha"],
    help="Low-pass coefficient for target smoothing in builtin IK bridge.",
)
parser.add_argument(
    "--max-target-position-step-m",
    type=float,
    default=BUILTIN_IK_DEFAULTS["max_target_position_step_m"],
    help="Maximum Cartesian target step per IK update.",
)
parser.add_argument(
    "--max-target-orientation-step-rad",
    type=float,
    default=BUILTIN_IK_DEFAULTS["max_target_orientation_step_rad"],
    help="Maximum orientation target step per IK update.",
)
parser.add_argument(
    "--disable-target-smoothing",
    action="store_true",
    help="Disable target smoothing and per-step servo limiting in builtin IK bridge.",
)
parser.set_defaults(ik_enable_orientation=BUILTIN_IK_DEFAULTS["enable_orientation"])
parser.add_argument("--lula-left-desc", type=str, default=str(DEFAULT_LULA_LEFT_DESC))
parser.add_argument("--lula-right-desc", type=str, default=str(DEFAULT_LULA_RIGHT_DESC))
parser.add_argument("--lula-desc", type=str, default="")
parser.add_argument(
    "--hold-stiffness",
    type=float,
    default=RUNTIME_DEFAULTS["hold_stiffness"],
    help="Drive stiffness used to hold joint pose against gravity.",
)
parser.add_argument(
    "--hold-damping",
    type=float,
    default=RUNTIME_DEFAULTS["hold_damping"],
    help="Drive damping used to hold joint pose against gravity.",
)
parser.add_argument(
    "--hold-max-force",
    type=float,
    default=RUNTIME_DEFAULTS["hold_max_force"],
    help="Drive max effort/torque used to hold joint pose against gravity.",
)
parser.add_argument(
    "--body-hold-stiffness",
    type=float,
    default=RUNTIME_DEFAULTS["body_hold_stiffness"],
    help="Drive stiffness used to hold body joints against sagging.",
)
parser.add_argument(
    "--body-hold-damping",
    type=float,
    default=RUNTIME_DEFAULTS["body_hold_damping"],
    help="Drive damping used to hold body joints against sagging.",
)
parser.add_argument(
    "--body-hold-max-force",
    type=float,
    default=RUNTIME_DEFAULTS["body_hold_max_force"],
    help="Drive max effort/torque used to hold body joints against sagging.",
)
parser.add_argument(
    "--disable-hold-drives",
    action="store_true",
    help="Disable applying hold drive gains.",
)
parser.set_defaults(disable_hold_drives=not RUNTIME_DEFAULTS["hold_enabled"])
parser.add_argument(
    "--disable-gripper-switch-remap",
    action="store_true",
    help=(
        "Disable remapping from single gripper switch command to two gripper joints. "
        "When enabled (default), the 8th position on arm command controls gripper open/close."
    ),
)
parser.add_argument(
    "--disable-body-posture-hold",
    action="store_true",
    help="Disable continuously holding body joints at their startup posture.",
)
parser.add_argument(
    "--demo-grasp",
    action="store_true",
    help="Enable a demo-like block grasp validation overlay driven by the main IK line.",
)
parser.add_argument(
    "--demo-grasp-arm",
    choices=["left", "right"],
    default=RUNTIME_DEFAULTS["demo_grasp_arm"],
    help="Arm used by the visual grasp validation overlay.",
)
parser.add_argument(
    "--demo-grasp-auto-start",
    action="store_true",
    help="Start the visual grasp validation cycle automatically.",
)
parser.add_argument(
    "--demo-grasp-block-base",
    type=str,
    default="",
    help="Initial grasp block position in base_link frame as x,y,z.",
)
parser.add_argument(
    "--demo-grasp-place-base",
    type=str,
    default="",
    help="Place marker position in base_link frame as x,y,z.",
)
parser.add_argument(
    "--demo-grasp-trigger-key",
    type=str,
    default=RUNTIME_DEFAULTS["demo_grasp_trigger_key"],
    help="Viewport keyboard key used to start the demo grasp cycle.",
)
parser.add_argument(
    "--demo-grasp-assist-attach",
    dest="demo_grasp_assist_attach",
    action="store_true",
    help="Kinematically attach the block to the gripper once it is close enough after closing.",
)
parser.add_argument(
    "--demo-grasp-no-assist-attach",
    dest="demo_grasp_assist_attach",
    action="store_false",
    help="Disable kinematic attach assistance and rely on pure contact for the demo block.",
)
parser.set_defaults(demo_grasp_assist_attach=RUNTIME_DEFAULTS["demo_grasp_assist_attach"])

parser.add_argument("--keep-base-collision", action="store_true", default=False)
parser.add_argument("--disable-all-collisions", action="store_true")
parser.add_argument("--no-init", action="store_true", help="Skip initial joint pose.")
args, _ = parser.parse_known_args()


configure_ros2_bridge_env()
simulation_app = SimulationApp({"headless": args.headless})

import carb
import omni.graph.core as og
import omni.kit.commands
import omni.timeline
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.prims import Articulation
from isaacsim.core.utils import extensions
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.utils.stage import add_reference_to_stage


NEUTRAL_INIT_JOINTS = {
    "body_joint1": 0.11,
    "body_joint5": 0.21,
    "left_arm_joint1": -0.65,
    "left_arm_joint2": -1.16,
    "left_arm_joint3": 0.45,
    "left_arm_joint4": -0.90,
    "left_arm_joint5": 1.01,
    "left_arm_joint6": -0.05,
    "left_arm_joint7": 0.01,
    "left_gripper_narrow_joint": -0.05,
    "left_gripper_wide_joint": -0.05,
    "right_arm_joint1": 0.65,
    "right_arm_joint2": 1.16,
    "right_arm_joint3": -0.45,
    "right_arm_joint4": 0.90,
    "right_arm_joint5": -1.01,
    "right_arm_joint6": 0.05,
    "right_arm_joint7": 0.01,
    "right_gripper_narrow_joint": -0.05,
    "right_gripper_wide_joint": -0.05,
}


def build_ros2_action_graph(
    articulation_root: str,
    left_cmd_topic: str,
    right_cmd_topic: str,
    body_cmd_topic: str,
) -> None:
    keys = og.Controller.Keys
    create_nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("Context", "isaacsim.ros2.bridge.ROS2Context"),
    ]
    set_values = [("ReadSimTime.inputs:resetOnStop", False)]
    connect = []

    if args.domain_id >= 0:
        set_values.extend(
            [
                ("Context.inputs:useDomainIDEnvVar", False),
                ("Context.inputs:domain_id", args.domain_id),
            ]
        )
    else:
        set_values.append(("Context.inputs:useDomainIDEnvVar", True))

    groups = [
        ("Left", left_cmd_topic),
        ("Right", right_cmd_topic),
        ("Body", body_cmd_topic),
    ]

    for group, cmd_topic in groups:
        sub_name = f"SubscribeJointState_{group}"
        ctrl_name = f"ArticulationController_{group}"
        create_nodes.extend(
            [
                (sub_name, "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                (ctrl_name, "isaacsim.core.nodes.IsaacArticulationController"),
            ]
        )
        connect.extend(
            [
                ("OnPlaybackTick.outputs:tick", f"{sub_name}.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", f"{ctrl_name}.inputs:execIn"),
                ("Context.outputs:context", f"{sub_name}.inputs:context"),
                (f"{sub_name}.outputs:jointNames", f"{ctrl_name}.inputs:jointNames"),
                (f"{sub_name}.outputs:positionCommand", f"{ctrl_name}.inputs:positionCommand"),
                (f"{sub_name}.outputs:velocityCommand", f"{ctrl_name}.inputs:velocityCommand"),
                (f"{sub_name}.outputs:effortCommand", f"{ctrl_name}.inputs:effortCommand"),
            ]
        )
        set_values.extend(
            [
                (f"{sub_name}.inputs:topicName", cmd_topic),
                (f"{sub_name}.inputs:nodeNamespace", args.node_namespace),
                (f"{ctrl_name}.inputs:robotPath", articulation_root),
            ]
        )

    og.Controller.edit(
        {"graph_path": args.graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: create_nodes,
            keys.CONNECT: connect,
            keys.SET_VALUES: set_values,
        },
    )


def import_robot(urdf_path: Path) -> str:
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    if not status:
        raise RuntimeError("URDFCreateImportConfig failed")
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = bool(args.fix_base)
    import_config.distance_scale = 1.0
    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
        get_articulation_root=True,
    )
    if not status:
        raise RuntimeError(f"Failed to import URDF: {urdf_path}")
    return prim_path


def disable_collision_prims(articulation_root: str, mode: str) -> None:
    if mode == "none":
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return
    to_remove: List[str] = []
    if mode in ("base", "all"):
        base_collisions = f"{articulation_root}/collisions"
        prim = stage.GetPrimAtPath(base_collisions)
        if prim and prim.IsValid():
            to_remove.append(base_collisions)
    if mode == "all":
        prefix = f"{articulation_root}/"
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim_path.startswith(prefix) and prim.GetName() == "collisions":
                to_remove.append(prim_path)
    for path in sorted(set(to_remove), key=len, reverse=True):
        stage.RemovePrim(path)


def apply_hold_gains_articulation(
    articulation: Articulation,
    joint_names: List[str],
    stiffness: float,
    damping: float,
    max_force: float,
) -> Tuple[int, List[str]]:
    try:
        articulation.initialize()
    except Exception:
        # The articulation may already be initialized; continue and try writes.
        pass

    dof_names = list(getattr(articulation, "dof_names", []) or [])
    if not dof_names:
        carb.log_warn("Articulation has no dof names; cannot apply hold gains")
        return 0, sorted(set(joint_names))

    dof_set = set(dof_names)
    valid_names = [name for name in joint_names if name in dof_set]
    missing = sorted(set(joint_names) - set(valid_names))
    if not valid_names:
        return 0, missing

    kps = np.full((1, len(valid_names)), float(stiffness), dtype=np.float32)
    kds = np.full((1, len(valid_names)), float(damping), dtype=np.float32)
    articulation.set_gains(kps=kps, kds=kds, joint_names=valid_names)

    max_efforts = np.full((1, len(valid_names)), float(max_force), dtype=np.float32)
    articulation.set_max_efforts(max_efforts, joint_names=valid_names)
    return len(valid_names), missing


def apply_initial_joint_positions_articulation(
    articulation: Articulation,
    joint_targets: Dict[str, float],
) -> Tuple[int, List[str]]:
    try:
        articulation.initialize()
    except Exception:
        pass

    dof_names = list(getattr(articulation, "dof_names", []) or [])
    if not dof_names:
        carb.log_warn("Articulation has no dof names; cannot apply initial joint pose")
        return 0, sorted(joint_targets.keys())

    dof_name_to_index = {name: idx for idx, name in enumerate(dof_names)}
    valid_names = [name for name in joint_targets.keys() if name in dof_name_to_index]
    missing = sorted(set(joint_targets.keys()) - set(valid_names))
    if not valid_names:
        return 0, missing

    current_positions = articulation.get_joint_positions()
    if current_positions is None:
        full_positions = np.zeros(len(dof_names), dtype=np.float32)
    else:
        full_positions = np.asarray(current_positions, dtype=np.float32)
        if full_positions.ndim == 2 and full_positions.shape[0] == 1:
            full_positions = full_positions[0]
        elif full_positions.ndim != 1:
            raise RuntimeError(
                f"Unexpected joint position shape while applying init pose: {full_positions.shape}"
            )
        full_positions = full_positions.copy()

    for name in valid_names:
        full_positions[dof_name_to_index[name]] = float(joint_targets[name])

    articulation.set_joint_positions(full_positions)
    return len(valid_names), missing


def apply_joint_drive_gains(
    articulation_root: str,
    joint_names: List[str],
    stiffness: float,
    damping: float,
    max_force: float,
) -> Tuple[int, List[str]]:
    import omni.usd
    from pxr import UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        carb.log_warn("Cannot set drive gains: USD stage is not available")
        return 0, []

    joint_name_set = set(joint_names)
    if not joint_name_set:
        return 0, []

    seen: set[str] = set()
    applied = 0
    for prim in stage.Traverse():
        joint_name = prim.GetName()
        if joint_name not in joint_name_set:
            continue

        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_type = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_type = "linear"
        else:
            continue

        drive = UsdPhysics.DriveAPI.Get(prim, drive_type)
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(prim, drive_type)
        drive.GetStiffnessAttr().Set(float(stiffness))
        drive.GetDampingAttr().Set(float(damping))
        drive.GetMaxForceAttr().Set(float(max_force))
        seen.add(joint_name)
        applied += 1

    missing = sorted(joint_name_set - seen)
    if missing:
        carb.log_warn(f"Drive gains not applied (joint not found): {missing}")
    return applied, missing


def main() -> int:
    urdf_path = resolve_input_path(args.urdf)
    if not urdf_path.exists():
        carb.log_error(f"URDF not found: {urdf_path}")
        return 1

    ensure_ros_package_path_for_urdf(urdf_path)
    import_urdf_path = _make_mesh_resolved_urdf(urdf_path)

    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    world = World(stage_units_in_meters=1.0)
    world.scene.add(
        GroundPlane(prim_path="/World/defaultGroundPlane", name="default_ground_plane", z_position=0.0)
    )

    articulation_root = import_robot(import_urdf_path)
    wait_start = time.time()
    while stage_utils.is_stage_loading() and time.time() - wait_start < 10.0:
        simulation_app.update()

    disable_mode = "none"
    if args.disable_all_collisions:
        disable_mode = "all"
    elif not args.keep_base_collision:
        disable_mode = "base"
    disable_collision_prims(articulation_root, disable_mode)

    world.reset()
    articulation = Articulation(prim_paths_expr=articulation_root, name="jz_robot_view")
    world.scene.add(articulation)
    articulation.initialize()

    if not args.no_init:
        init_applied, init_missing = apply_initial_joint_positions_articulation(
            articulation=articulation,
            joint_targets=NEUTRAL_INIT_JOINTS,
        )
        print(
            "Initial joint pose applied:"
            f" joints={init_applied}"
            f" missing={init_missing}"
        )

    arm_hold_joint_names = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
    if not args.disable_hold_drives:
        stage_applied, stage_missing = apply_joint_drive_gains(
            articulation_root=articulation_root,
            joint_names=arm_hold_joint_names,
            stiffness=args.hold_stiffness,
            damping=args.hold_damping,
            max_force=args.hold_max_force,
        )
        print(
            "Hold drive gains applied (stage, arms):"
            f" joints={stage_applied}"
            f" stiffness={args.hold_stiffness}"
            f" damping={args.hold_damping}"
            f" max_force={args.hold_max_force}"
            f" missing={len(stage_missing)}"
        )

        art_applied, art_missing = apply_hold_gains_articulation(
            articulation=articulation,
            joint_names=arm_hold_joint_names,
            stiffness=args.hold_stiffness,
            damping=args.hold_damping,
            max_force=args.hold_max_force,
        )
        print(
            "Hold drive gains applied (articulation, arms):"
            f" joints={art_applied}"
            f" stiffness={args.hold_stiffness}"
            f" damping={args.hold_damping}"
            f" max_force={args.hold_max_force}"
            f" missing={len(art_missing)}"
        )

        body_stage_applied, body_stage_missing = apply_joint_drive_gains(
            articulation_root=articulation_root,
            joint_names=BODY_JOINTS,
            stiffness=args.body_hold_stiffness,
            damping=args.body_hold_damping,
            max_force=args.body_hold_max_force,
        )
        print(
            "Hold drive gains applied (stage, body):"
            f" joints={body_stage_applied}"
            f" stiffness={args.body_hold_stiffness}"
            f" damping={args.body_hold_damping}"
            f" max_force={args.body_hold_max_force}"
            f" missing={len(body_stage_missing)}"
        )

        body_art_applied, body_art_missing = apply_hold_gains_articulation(
            articulation=articulation,
            joint_names=BODY_JOINTS,
            stiffness=args.body_hold_stiffness,
            damping=args.body_hold_damping,
            max_force=args.body_hold_max_force,
        )
        print(
            "Hold drive gains applied (articulation, body):"
            f" joints={body_art_applied}"
            f" stiffness={args.body_hold_stiffness}"
            f" damping={args.body_hold_damping}"
            f" max_force={args.body_hold_max_force}"
            f" missing={len(body_art_missing)}"
        )

    left_graph_cmd_topic = args.arm_left_cmd_topic
    right_graph_cmd_topic = args.arm_right_cmd_topic
    remapper: Optional[GripperSwitchRemapper] = None
    if not args.disable_gripper_switch_remap:
        remapper = GripperSwitchRemapper(
            left_in_topic=args.arm_left_cmd_topic,
            right_in_topic=args.arm_right_cmd_topic,
            left_out_topic=f"{args.arm_left_cmd_topic}_mapped",
            right_out_topic=f"{args.arm_right_cmd_topic}_mapped",
        )
        remapper.start()
        left_graph_cmd_topic = remapper.left_out_topic
        right_graph_cmd_topic = remapper.right_out_topic
        print(
            "Gripper switch remap enabled:"
            f" {args.arm_left_cmd_topic} -> {left_graph_cmd_topic},"
            f" {args.arm_right_cmd_topic} -> {right_graph_cmd_topic}"
        )

    build_ros2_action_graph(
        articulation_root,
        left_cmd_topic=left_graph_cmd_topic,
        right_cmd_topic=right_graph_cmd_topic,
        body_cmd_topic=args.body_cmd_topic,
    )

    ik_bridge = None
    if args.control_mode in ["ee_pose", "auto"]:
        from builtin_ik_solver import BuiltinIKBridge, BuiltinIKConfig

        if args.ik_enable_orientation is None:
            effective_enable_orientation = args.control_mode in ["ee_pose", "auto"]
            config_orientation = BUILTIN_IK_DEFAULTS["enable_orientation"]
            if config_orientation is not None:
                effective_enable_orientation = bool(config_orientation)
            orientation_source = "auto-default"
        else:
            effective_enable_orientation = bool(args.ik_enable_orientation)
            orientation_source = "cli"
        effective_target_smoothing = bool(BUILTIN_IK_DEFAULTS["enable_target_smoothing"])
        if args.disable_target_smoothing:
            effective_target_smoothing = False
        if args.demo_grasp and effective_target_smoothing:
            effective_target_smoothing = False
            print("[IK] Demo grasp enabled: target smoothing disabled for responsive overlay.")
        print(
            "[IK] Orientation tracking:"
            f" enabled={effective_enable_orientation}"
            f" source={orientation_source}"
        )
        ik_config = BuiltinIKConfig(
            left_ee_topic=args.left_ee_topic,
            right_ee_topic=args.right_ee_topic,
            left_cmd_topic=args.arm_left_cmd_topic,
            right_cmd_topic=args.arm_right_cmd_topic,
            left_pose_topic=BUILTIN_IK_DEFAULTS["left_pose_topic"],
            right_pose_topic=BUILTIN_IK_DEFAULTS["right_pose_topic"],
            left_status_topic=BUILTIN_IK_DEFAULTS["left_status_topic"],
            right_status_topic=BUILTIN_IK_DEFAULTS["right_status_topic"],
            left_ee_link=args.left_ee_link,
            right_ee_link=args.right_ee_link,
            left_solver_base_link=BUILTIN_IK_DEFAULTS["left_solver_base_link"],
            right_solver_base_link=BUILTIN_IK_DEFAULTS["right_solver_base_link"],
            left_arm_joints=BUILTIN_IK_DEFAULTS["left_arm_joints"],
            right_arm_joints=BUILTIN_IK_DEFAULTS["right_arm_joints"],
            urdf_path=str(import_urdf_path),
            robot_description_path=args.lula_desc,
            left_robot_description_path=args.lula_left_desc,
            right_robot_description_path=args.lula_right_desc,
            ik_timeout_sec=args.ik_timeout_sec,
            enable_orientation=effective_enable_orientation,
            control_mode=args.control_mode,
            feedback_frame_id=args.feedback_frame_id,
            enable_target_smoothing=effective_target_smoothing,
            target_smoothing_alpha=args.target_smoothing_alpha,
            max_target_position_step_m=args.max_target_position_step_m,
            max_target_orientation_step_rad=args.max_target_orientation_step_rad,
            direct_apply_joint_targets=BUILTIN_IK_DEFAULTS["direct_apply_joint_targets"],
            max_joint_delta_rad=BUILTIN_IK_DEFAULTS["max_joint_delta_rad"],
            max_joint_position_abs_rad=BUILTIN_IK_DEFAULTS["max_joint_position_abs_rad"],
            orientation_position_gate_m=BUILTIN_IK_DEFAULTS["orientation_position_gate_m"],
            approx_orientation_accept_position_error_m=BUILTIN_IK_DEFAULTS[
                "approx_orientation_accept_position_error_m"
            ],
            close_range_orientation_distance_m=BUILTIN_IK_DEFAULTS[
                "close_range_orientation_distance_m"
            ],
            close_range_max_orientation_step_rad=BUILTIN_IK_DEFAULTS[
                "close_range_max_orientation_step_rad"
            ],
        )
        ik_bridge = BuiltinIKBridge(ik_config, articulation)
        ik_bridge.start()
        print("[IK] Builtin Lula enabled.")

    joint_state_pub = ManualJointStatePublisher(
        articulation=articulation,
        left_topic=args.arm_left_state_topic,
        right_topic=args.arm_right_state_topic,
        body_topic=args.body_state_topic,
        publish_rate_hz=args.manual_joint_state_rate,
    )
    body_posture_hold_enabled = RUNTIME_DEFAULTS["body_posture_hold"] and not args.disable_body_posture_hold
    body_posture_keeper = None if not body_posture_hold_enabled else BodyPostureKeeper(
        articulation=articulation,
        joint_names=BODY_JOINTS,
    )

    demo_controller = None
    if args.demo_grasp:
        demo_block_base = (
            _parse_xyz_arg(args.demo_grasp_block_base, (0.0, 0.0, 0.0))
            if (args.demo_grasp_block_base or "").strip()
            else None
        )
        demo_place_base = (
            _parse_xyz_arg(args.demo_grasp_place_base, (0.0, 0.0, 0.0))
            if (args.demo_grasp_place_base or "").strip()
            else None
        )
        demo_controller = MainlineGraspDemo(
            world=world,
            articulation=articulation,
            ik_bridge=ik_bridge,
            arm=args.demo_grasp_arm,
            auto_start=args.demo_grasp_auto_start,
            assist_attach=args.demo_grasp_assist_attach,
            trigger_key=args.demo_grasp_trigger_key,
            block_base=demo_block_base,
            place_base=demo_place_base,
            track_orientation=effective_enable_orientation,
            headless=args.headless,
        )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    step_limit = args.steps if args.steps > 0 else None
    step_count = 0
    ik_last_update = 0.0
    ik_update_interval = 1.0 / args.ik_rate_hz if args.control_mode in ["ee_pose", "auto"] else 0.0

    while simulation_app.app.is_running() and not simulation_app.is_exiting():
        if step_limit is not None and step_count >= step_limit and not args.hold:
            break
        world.step(render=not args.headless)
        step_count += 1

        if demo_controller is not None:
            demo_controller.update()

        if ik_bridge is not None:
            current_time = time.time()
            if current_time - ik_last_update >= ik_update_interval:
                ik_bridge.update()
                ik_last_update = current_time

        if body_posture_keeper is not None:
            body_posture_keeper.apply()

        if joint_state_pub is not None:
            joint_state_pub.publish_if_due()

    if ik_bridge is not None:
        ik_bridge.stop()
    if remapper is not None:
        remapper.stop()
    if joint_state_pub is not None:
        joint_state_pub.stop()
    return 0


exit_code = 1
try:
    exit_code = main()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    simulation_app.close()
    if exit_code != 0:
        sys.exit(exit_code)
