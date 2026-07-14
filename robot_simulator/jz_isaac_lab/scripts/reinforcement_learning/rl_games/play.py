"""Play back RL-Games checkpoints for JZ Isaac Lab tasks."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PACKAGE_DIR = PROJECT_ROOT / "source" / "jzlab"
if str(SOURCE_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_DIR))

ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "")).resolve() if os.environ.get("ISAACLAB_PATH") else None
if ISAACLAB_ROOT:
    for rel_path in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_rl", "source/isaaclab_assets"):
        candidate = ISAACLAB_ROOT / rel_path
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record a playback video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--camera_eye", type=str, default="3.2,3.4,3.0", help="Viewer camera eye xyz for recorded videos.")
parser.add_argument("--camera_lookat", type=str, default="0.65,0.0,1.35", help="Viewer camera look-at xyz for recorded videos.")
parser.add_argument("--colorize_arms", action="store_true", default=False, help="Color left/right robot arms for video debugging.")
parser.add_argument("--left_arm_color", type=str, default="0.05,0.35,1.0", help="RGB color for the left arm when colorizing.")
parser.add_argument("--right_arm_color", type=str, default="1.0,0.82,0.05", help="RGB color for the right arm when colorizing.")
parser.add_argument("--debug_axes", action="store_true", default=False, help="Draw coordinate axes at both TCPs and targets.")
parser.add_argument("--debug_axis_scale", type=float, default=0.06, help="Scale for playback debug coordinate axes.")
parser.add_argument(
    "--debug_fingertip_contact_points",
    action="store_true",
    default=False,
    help="Draw the four inner collision-surface points used by pre-grasp rewards.",
)
parser.add_argument("--debug_colliders", action="store_true", default=False, help="Show all PhysX collision shapes.")
parser.add_argument(
    "--print_gripper_diagnostics",
    action="store_true",
    default=False,
    help="Print gripper joints, filtered fingertip forces, and object positions.",
)
parser.add_argument("--diagnostic_every", type=int, default=30, help="Step interval for gripper diagnostics.")
parser.add_argument(
    "--debug_grasp_target_offset",
    type=str,
    default="0.0,0.0,0.0",
    help="Base-frame xyz offset from grasp object centers to the visualized target point.",
)
parser.add_argument(
    "--force_gripper_close",
    action="store_true",
    default=False,
    help="Playback-only override: keep checkpoint arm actions but force both gripper action terms to close.",
)
parser.add_argument(
    "--start_gripper_closed",
    action="store_true",
    default=False,
    help="Playback-only override: set both grippers closed in the robot reset/default joint pose before creating the env.",
)
parser.add_argument(
    "--gripper_close_value",
    type=float,
    default=1.0,
    help="Absolute close command used with --force_gripper_close.",
)
parser.add_argument(
    "--gripper_close_ramp_steps",
    type=int,
    default=90,
    help="Ramp length in playback steps for --force_gripper_close.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use a published pre-trained checkpoint.")
parser.add_argument(
    "--use_last_checkpoint", action="store_true", help="Use the last saved model if no checkpoint is provided."
)
parser.add_argument(
    "--disable_play_task_switch",
    action="store_true",
    default=False,
    help="Keep the requested task unchanged instead of auto-switching to the dedicated *-Play-vN variant.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import math
import random
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import jzlab.tasks  # noqa: F401
from jzlab.tasks.manager_based.jz_manipulation.bimanual.reach import mdp
from jzlab.tasks.manager_based.jz_manipulation.constants import (
    FINGERTIP_INNER_CONTACT_LOCAL_OFFSETS,
    LEFT_ARM_JOINTS,
    LEFT_GRIPPER_CLOSED,
    LEFT_GRIPPER_JOINTS,
    LEFT_GRASP_TARGET_QUAT_W,
    LEFT_TCP_ORIENTATION_LINK,
    LEFT_TCP_ORIENTATION_OFFSET_QUAT,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_GRIPPER_CLOSED,
    RIGHT_GRIPPER_JOINTS,
    RIGHT_GRASP_TARGET_QUAT_W,
    RIGHT_TCP_ORIENTATION_LINK,
    RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
    RIGHT_TCP_POSITION_LINKS,
    TCP_CONTACT_LOCAL_OFFSETS,
)
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _resolve_play_task_name(task_name: str | None) -> str | None:
    """Prefer the dedicated *-Play-vN task variant for playback when available."""

    if task_name is None or "-Play-" in task_name:
        return task_name

    if "-v" not in task_name:
        return task_name

    stem, version = task_name.rsplit("-v", 1)
    candidate = f"{stem}-Play-v{version}"
    try:
        gym.spec(candidate)
    except Exception:
        return task_name

    print(f"[INFO] Switching playback task from '{task_name}' to dedicated play task '{candidate}'.")
    return candidate


if not args_cli.disable_play_task_switch:
    args_cli.task = _resolve_play_task_name(args_cli.task)


def _load_checkpoint_file(checkpoint_path: str) -> dict:
    """Load a trusted local checkpoint across PyTorch versions."""

    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_checkpoint_env_params(checkpoint_path: str) -> dict | None:
    """Load the saved env params that were written alongside the checkpoint, if present."""

    checkpoint_file = Path(checkpoint_path).resolve()
    run_dir = checkpoint_file.parent.parent
    params_path = run_dir / "params" / "env.yaml"
    if not params_path.is_file():
        return None

    with params_path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _load_checkpoint_agent_params(checkpoint_path: str) -> dict | None:
    """Load the saved agent params that were written alongside the checkpoint, if present."""

    checkpoint_file = Path(checkpoint_path).resolve()
    run_dir = checkpoint_file.parent.parent
    params_path = run_dir / "params" / "agent.yaml"
    if not params_path.is_file():
        return None

    with params_path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _infer_checkpoint_obs_dim(checkpoint_path: str) -> int | None:
    """Infer the policy observation dimension stored in an RL-Games checkpoint."""

    checkpoint = _load_checkpoint_file(checkpoint_path)
    model_state = checkpoint.get("model", {})

    running_mean = model_state.get("running_mean_std.running_mean")
    if running_mean is not None and len(running_mean.shape) == 1:
        return int(running_mean.shape[0])

    actor_weight = model_state.get("a2c_network.actor_mlp.0.weight")
    if actor_weight is not None and len(actor_weight.shape) == 2:
        return int(actor_weight.shape[1])

    return None


def _parse_vec3(text: str, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        values = [float(item.strip()) for item in str(text).split(",")]
    except ValueError:
        return fallback
    if len(values) != 3:
        return fallback
    return (values[0], values[1], values[2])


def _override_gripper_close_actions(actions: torch.Tensor, step: int) -> torch.Tensor:
    if not args_cli.force_gripper_close or actions.shape[-1] < 18:
        return actions
    ramp_steps = max(1, int(args_cli.gripper_close_ramp_steps))
    value = min(1.0, float(step) / float(ramp_steps)) * float(args_cli.gripper_close_value)
    actions = actions.clone()
    actions[:, 14] = value
    actions[:, 15] = -value
    actions[:, 16] = value
    actions[:, 17] = -value
    return actions


def _set_robot_gripper_default_closed(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg) -> None:
    if not args_cli.start_gripper_closed or not hasattr(env_cfg, "scene") or not hasattr(env_cfg.scene, "robot"):
        return
    joint_pos = dict(getattr(env_cfg.scene.robot.init_state, "joint_pos", {}) or {})
    joint_pos.update(LEFT_GRIPPER_CLOSED)
    joint_pos.update(RIGHT_GRIPPER_CLOSED)
    env_cfg.scene.robot.init_state.joint_pos = joint_pos
    print("[INFO] Playback reset override: starting both grippers closed.")


def _enable_collider_visualization() -> None:
    if not args_cli.debug_colliders:
        return
    import carb

    carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)
    print("[INFO] PhysX collider visualization enabled: Show on All.")


class _GripperDiagnostics:
    """Playback-only numeric check for closure and filtered fingertip contact."""

    _SENSOR_NAMES = (
        "left_narrow_contact",
        "left_wide_contact",
        "right_narrow_contact",
        "right_wide_contact",
    )

    def __init__(self, env):
        self._env = getattr(env, "unwrapped", env)
        self._robot = self._env.scene["robot"]
        self._left_joint_ids = list(self._robot.find_joints(LEFT_GRIPPER_JOINTS)[0])
        self._right_joint_ids = list(self._robot.find_joints(RIGHT_GRIPPER_JOINTS)[0])

    def print(self, step: int) -> None:
        left_pos = self._robot.data.joint_pos[0, self._left_joint_ids].detach().cpu().tolist()
        right_pos = self._robot.data.joint_pos[0, self._right_joint_ids].detach().cpu().tolist()
        print(f"[step {step}] gripper diagnostics")
        print(f"  left_gripper_joint_pos:  {left_pos}")
        print(f"  right_gripper_joint_pos: {right_pos}")
        for sensor_name in self._SENSOR_NAMES:
            try:
                force_matrix_w = self._env.scene[sensor_name].data.force_matrix_w
            except Exception as exc:
                print(f"  {sensor_name}_filtered_force_n: unavailable ({exc})")
                continue
            if force_matrix_w is None:
                print(f"  {sensor_name}_filtered_force_n: unavailable (no force matrix)")
                continue
            force_n = torch.linalg.vector_norm(force_matrix_w[0], dim=-1).max().item()
            print(f"  {sensor_name}_filtered_force_n: {force_n:.6f}")
        for object_name in ("object", "right_object"):
            try:
                pos = self._env.scene[object_name].data.root_pos_w[0].detach().cpu().tolist()
            except Exception:
                continue
            print(f"  {object_name}_root_pos_w: {pos}")


def _infer_checkpoint_mlp_units(checkpoint_path: str) -> list[int] | None:
    """Infer hidden-layer widths from an RL-Games actor MLP checkpoint."""

    checkpoint = _load_checkpoint_file(checkpoint_path)
    model_state = checkpoint.get("model", {})

    units: list[int] = []
    layer_index = 0
    while True:
        weight = model_state.get(f"a2c_network.actor_mlp.{layer_index}.weight")
        if weight is None:
            break
        if len(weight.shape) != 2:
            break
        units.append(int(weight.shape[0]))
        layer_index += 2

    return units or None


def _apply_network_compat_for_checkpoint(agent_cfg: dict, checkpoint_path: str) -> bool:
    """Align the agent MLP layout with the checkpoint before building the policy."""

    checkpoint_units = _infer_checkpoint_mlp_units(checkpoint_path)
    if checkpoint_units is None:
        return False

    network_cfg = agent_cfg.get("params", {}).get("network", {})
    mlp_cfg = network_cfg.get("mlp", {})
    current_units = list(mlp_cfg.get("units", []))
    if current_units == checkpoint_units:
        return False

    mlp_cfg["units"] = checkpoint_units
    print(
        "[INFO] Applying checkpoint network compatibility for playback: "
        f"using hidden units {checkpoint_units} instead of configured {current_units}."
    )
    return True


def _apply_obs_compat_for_checkpoint(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, obs_dim: int) -> bool:
    """Adjust known observation-layout changes so older checkpoints can still be played back."""

    policy_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_cfg is None:
        return False

    changed = False
    if obs_dim in (56, 62, 68):
        for term_name in ("left_pose_command", "right_pose_command"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                getattr(policy_cfg, term_name).func = mdp.generated_commands
                changed = True

    if obs_dim == 56:
        for term_name in (
            "left_tcp_error",
            "right_tcp_error",
            "left_tcp_pos",
            "right_tcp_pos",
            "left_tcp_lin_vel",
            "right_tcp_lin_vel",
            "left_wrist_gravity",
            "right_wrist_gravity",
        ):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                setattr(policy_cfg, term_name, None)
                changed = True
        # Match the pre-smoothing training configuration for older checkpoints.
        if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
            env_cfg.sim.physx.enable_external_forces_every_iteration = False
        robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
        if robot_cfg is not None and getattr(robot_cfg, "spawn", None) is not None:
            articulation_props = getattr(robot_cfg.spawn, "articulation_props", None)
            if articulation_props is not None:
                articulation_props.solver_velocity_iteration_count = 1
        if robot_cfg is not None and getattr(robot_cfg, "actuators", None) is not None:
            arm_actuator = robot_cfg.actuators.get("arm")
            if arm_actuator is not None:
                arm_actuator.damping = 80.0

    if obs_dim == 62:
        for term_name in (
            "left_tcp_pos",
            "right_tcp_pos",
            "left_tcp_lin_vel",
            "right_tcp_lin_vel",
            "left_wrist_gravity",
            "right_wrist_gravity",
        ):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                setattr(policy_cfg, term_name, None)
                changed = True

    if obs_dim == 66:
        for term_name in ("left_pose_command", "right_pose_command"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                getattr(policy_cfg, term_name).func = mdp.generated_command_positions
                changed = True
        for term_name in ("left_tcp_pos", "right_tcp_pos"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                setattr(policy_cfg, term_name, None)
                changed = True
        if getattr(policy_cfg, "left_tcp_lin_vel", None) is None:
            policy_cfg.left_tcp_lin_vel = _make_tcp_lin_vel_obs_term(LEFT_TCP_POSITION_LINKS)
            changed = True
        if getattr(policy_cfg, "right_tcp_lin_vel", None) is None:
            policy_cfg.right_tcp_lin_vel = _make_tcp_lin_vel_obs_term(RIGHT_TCP_POSITION_LINKS)
            changed = True
        if getattr(policy_cfg, "left_wrist_gravity", None) is None:
            policy_cfg.left_wrist_gravity = _make_projected_gravity_obs_term(LEFT_TCP_ORIENTATION_LINK)
            changed = True
        if getattr(policy_cfg, "right_wrist_gravity", None) is None:
            policy_cfg.right_wrist_gravity = _make_projected_gravity_obs_term(RIGHT_TCP_ORIENTATION_LINK)
            changed = True

    if obs_dim == 68:
        if getattr(policy_cfg, "left_tcp_pos", None) is None:
            policy_cfg.left_tcp_pos = _make_tcp_position_obs_term(LEFT_TCP_POSITION_LINKS)
            changed = True
        if getattr(policy_cfg, "right_tcp_pos", None) is None:
            policy_cfg.right_tcp_pos = _make_tcp_position_obs_term(RIGHT_TCP_POSITION_LINKS)
            changed = True
        for term_name in ("left_tcp_lin_vel", "right_tcp_lin_vel", "left_wrist_gravity", "right_wrist_gravity"):
            if hasattr(policy_cfg, term_name) and getattr(policy_cfg, term_name) is not None:
                setattr(policy_cfg, term_name, None)
                changed = True

    if changed:
        print(f"[INFO] Applying legacy {obs_dim}-dim checkpoint compatibility for playback.")

    return changed


_OBS_GROUP_CFG_KEYS = (
    "concatenate_terms",
    "concatenate_dim",
    "enable_corruption",
    "history_length",
    "flatten_history_dim",
)


def _make_tcp_position_obs_term(body_names: list[str]) -> ObsTerm:
    return ObsTerm(func=mdp.fingertip_midpoint_position_b, params={"asset_cfg": SceneEntityCfg("robot", body_names=body_names)})


def _make_tcp_lin_vel_obs_term(body_names: list[str]) -> ObsTerm:
    return ObsTerm(
        func=mdp.fingertip_midpoint_linear_velocity_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=body_names)},
    )


def _make_projected_gravity_obs_term(body_name: str) -> ObsTerm:
    return ObsTerm(
        func=mdp.body_projected_gravity_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[body_name])},
    )


def _make_tcp_error_obs_term(command_name: str, body_names: list[str]) -> ObsTerm:
    return ObsTerm(
        func=mdp.fingertip_midpoint_position_command_error_vector_b,
        params={"command_name": command_name, "asset_cfg": SceneEntityCfg("robot", body_names=body_names)},
    )


_POLICY_TERM_NAMES = (
    "left_joint_pos",
    "right_joint_pos",
    "left_joint_vel",
    "right_joint_vel",
    "left_tcp_pos",
    "right_tcp_pos",
    "left_pose_command",
    "right_pose_command",
    "left_tcp_error",
    "right_tcp_error",
    "left_tcp_lin_vel",
    "right_tcp_lin_vel",
    "left_wrist_gravity",
    "right_wrist_gravity",
    "left_actions",
    "right_actions",
)


def _build_policy_term(term_name: str, saved_term_cfg: dict, current_terms: dict[str, ObsTerm | None]) -> ObsTerm | None:
    term_cfg = current_terms.get(term_name)

    if term_name == "left_tcp_pos":
        return _make_tcp_position_obs_term(LEFT_TCP_POSITION_LINKS)
    if term_name == "right_tcp_pos":
        return _make_tcp_position_obs_term(RIGHT_TCP_POSITION_LINKS)
    if term_name == "left_tcp_lin_vel":
        return _make_tcp_lin_vel_obs_term(LEFT_TCP_POSITION_LINKS)
    if term_name == "right_tcp_lin_vel":
        return _make_tcp_lin_vel_obs_term(RIGHT_TCP_POSITION_LINKS)
    if term_name == "left_wrist_gravity":
        return _make_projected_gravity_obs_term(LEFT_TCP_ORIENTATION_LINK)
    if term_name == "right_wrist_gravity":
        return _make_projected_gravity_obs_term(RIGHT_TCP_ORIENTATION_LINK)
    if term_name == "left_tcp_error" and term_cfg is None:
        return _make_tcp_error_obs_term("left_ee_pose", LEFT_TCP_POSITION_LINKS)
    if term_name == "right_tcp_error" and term_cfg is None:
        return _make_tcp_error_obs_term("right_ee_pose", RIGHT_TCP_POSITION_LINKS)
    if term_name in ("left_pose_command", "right_pose_command"):
        if term_cfg is None:
            return None
        func_name = str(saved_term_cfg.get("func", ""))
        term_cfg.func = mdp.generated_commands if "generated_commands" in func_name else mdp.generated_command_positions
        return term_cfg

    return term_cfg


def _build_action_cfg(saved_action_cfg: dict, joint_names: list[str]):
    """Rebuild a joint action config from a saved env.yaml action block."""

    class_type = str(saved_action_cfg.get("class_type", ""))
    asset_name = saved_action_cfg.get("asset_name", "robot")
    resolved_joint_names = saved_action_cfg.get("joint_names", joint_names)
    scale = saved_action_cfg.get("scale", 1.0)
    offset = saved_action_cfg.get("offset", 0.0)
    preserve_order = bool(saved_action_cfg.get("preserve_order", False))

    if "RelativeJointPositionAction" in class_type:
        return mdp.RelativeJointPositionActionCfg(
            asset_name=asset_name,
            joint_names=resolved_joint_names,
            scale=scale,
            offset=offset,
            preserve_order=preserve_order,
            use_zero_offset=bool(saved_action_cfg.get("use_zero_offset", True)),
        )
    if "JointPositionAction" in class_type and "ToLimits" not in class_type:
        return mdp.JointPositionActionCfg(
            asset_name=asset_name,
            joint_names=resolved_joint_names,
            scale=scale,
            offset=offset,
            preserve_order=preserve_order,
            use_default_offset=bool(saved_action_cfg.get("use_default_offset", True)),
        )

    common_kwargs = {
        "asset_name": asset_name,
        "joint_names": resolved_joint_names,
        "scale": scale,
        "rescale_to_limits": bool(saved_action_cfg.get("rescale_to_limits", True)),
        "preserve_order": preserve_order,
    }
    if "EMAJointPositionToLimitsAction" in class_type:
        return mdp.EMAJointPositionToLimitsActionCfg(alpha=saved_action_cfg.get("alpha", {}), **common_kwargs)
    if "JointPositionToLimitsAction" in class_type:
        return mdp.JointPositionToLimitsActionCfg(**common_kwargs)
    return None


def _apply_saved_agent_compat_for_checkpoint(agent_cfg: dict, checkpoint_path: str) -> bool:
    """Restore saved RL-Games wrapper settings for older checkpoints."""

    saved_agent_cfg = _load_checkpoint_agent_params(checkpoint_path)
    if not saved_agent_cfg:
        return False

    changed = False
    live_env_cfg = agent_cfg.setdefault("params", {}).setdefault("env", {})
    saved_env_cfg = saved_agent_cfg.get("params", {}).get("env", {})
    for key in ("clip_observations", "clip_actions"):
        saved_value = saved_env_cfg.get(key)
        if saved_value is None or live_env_cfg.get(key) == saved_value:
            continue
        live_env_cfg[key] = saved_value
        changed = True

    if changed:
        print("[INFO] Applying saved RL-Games wrapper settings from checkpoint params.")
    return changed


def _apply_saved_env_compat_for_checkpoint(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, checkpoint_path: str
) -> bool:
    """Restore historical observation/action/control settings from the checkpoint's saved env params."""

    saved_env_cfg = _load_checkpoint_env_params(checkpoint_path)
    if not saved_env_cfg:
        return False

    changed = False

    policy_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    saved_policy_cfg = (saved_env_cfg.get("observations") or {}).get("policy") or {}
    if policy_cfg is not None and saved_policy_cfg:
        current_terms = {term_name: getattr(policy_cfg, term_name, None) for term_name in _POLICY_TERM_NAMES}
        for term_name in _POLICY_TERM_NAMES:
            if hasattr(policy_cfg, term_name) or current_terms.get(term_name) is not None:
                setattr(policy_cfg, term_name, None)
        for term_name, saved_term_cfg in saved_policy_cfg.items():
            if term_name not in _POLICY_TERM_NAMES or saved_term_cfg is None:
                continue
            term_cfg = _build_policy_term(term_name, saved_term_cfg, current_terms)
            if term_cfg is not None:
                setattr(policy_cfg, term_name, term_cfg)
                changed = True
        policy_cfg.enable_corruption = False

    commands_cfg = saved_env_cfg.get("commands") or {}
    for command_name in ("left_ee_pose", "right_ee_pose"):
        saved_command_cfg = commands_cfg.get(command_name, {})
        live_command_cfg = getattr(getattr(env_cfg, "commands", None), command_name, None)
        if live_command_cfg is None:
            continue
        for attr_name in (
            "dataset_key",
            "body_name",
            "debug_vis",
            "use_fixed_quaternion",
            "fixed_quaternion",
            "resampling_time_range",
            "current_position_body_names",
            "current_quaternion_offset",
        ):
            if attr_name not in saved_command_cfg:
                continue
            saved_value = saved_command_cfg.get(attr_name)
            if getattr(live_command_cfg, attr_name, None) != saved_value:
                setattr(live_command_cfg, attr_name, saved_value)
                changed = True

    sim_cfg = saved_env_cfg.get("sim") or {}
    saved_physx_cfg = sim_cfg.get("physx") or {}
    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
        ext_forces = saved_physx_cfg.get("enable_external_forces_every_iteration")
        if ext_forces is not None:
            env_cfg.sim.physx.enable_external_forces_every_iteration = ext_forces
            changed = True

    robot_cfg = ((saved_env_cfg.get("scene") or {}).get("robot")) or {}
    live_robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if live_robot_cfg is not None:
        solver_vel_iters = robot_cfg.get("spawn", {}).get("articulation_props", {}).get("solver_velocity_iteration_count")
        if solver_vel_iters is not None and getattr(live_robot_cfg, "spawn", None) is not None:
            articulation_props = getattr(live_robot_cfg.spawn, "articulation_props", None)
            if articulation_props is not None:
                articulation_props.solver_velocity_iteration_count = solver_vel_iters
                changed = True

        arm_damping = robot_cfg.get("actuators", {}).get("arm", {}).get("damping")
        arm_stiffness = robot_cfg.get("actuators", {}).get("arm", {}).get("stiffness")
        if arm_damping is not None and getattr(live_robot_cfg, "actuators", None) is not None:
            arm_actuator = live_robot_cfg.actuators.get("arm")
            if arm_actuator is not None:
                arm_actuator.damping = arm_damping
                changed = True
                if arm_stiffness is not None:
                    arm_actuator.stiffness = arm_stiffness
                    changed = True

    saved_actions_cfg = saved_env_cfg.get("actions") or {}
    left_action_cfg = _build_action_cfg(saved_actions_cfg.get("left_arm_action", {}), LEFT_ARM_JOINTS)
    right_action_cfg = _build_action_cfg(saved_actions_cfg.get("right_arm_action", {}), RIGHT_ARM_JOINTS)
    if left_action_cfg is not None:
        env_cfg.actions.left_arm_action = left_action_cfg
        changed = True
    if right_action_cfg is not None:
        env_cfg.actions.right_arm_action = right_action_cfg
        changed = True

    if changed:
        print("[INFO] Applying playback compatibility from saved run params next to checkpoint.")
    return changed


def _infer_env_policy_obs_dim(env: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | gym.Env) -> int | None:
    """Infer the unbatched policy observation dimension from an Isaac Lab env instance."""

    base_env = getattr(env, "unwrapped", env)
    single_obs_space = getattr(base_env, "single_observation_space", None)
    if single_obs_space is not None:
        policy_space = single_obs_space.get("policy")
        if policy_space is not None and getattr(policy_space, "shape", None) is not None:
            return int(policy_space.shape[0])

    obs_space = getattr(env, "observation_space", None)
    if getattr(obs_space, "shape", None) is not None:
        return int(obs_space.shape[0])

    return None


def _make_debug_material(stage, material_path: str, color: tuple[float, float, float]):
    """Create a simple preview-surface material for playback-only color debugging."""

    from pxr import Gf, Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader_surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material_surface = material.CreateSurfaceOutput()
    connected = False
    for connect_fn in (
        lambda: material_surface.ConnectToSource(shader.ConnectableAPI(), "surface"),
        lambda: material_surface.ConnectToSource(shader, "surface"),
        lambda: material_surface.ConnectToSource(shader_surface),
    ):
        try:
            connect_fn()
            connected = True
            break
        except TypeError:
            continue
    if not connected:
        print(f"[WARN] Created debug material without a connected surface output: {material_path}")
    return material


def _colorize_robot_arms_for_playback(
    left_color: tuple[float, float, float],
    right_color: tuple[float, float, float],
) -> None:
    """Apply strong USD material bindings to left/right arm and gripper visuals."""

    import omni.usd
    from pxr import Gf, UsdGeom, UsdShade

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[WARN] Unable to colorize arms: USD stage is unavailable.")
        return

    left_material = _make_debug_material(stage, "/World/Looks/JZ_Debug_Left_Arm_Blue", left_color)
    right_material = _make_debug_material(stage, "/World/Looks/JZ_Debug_Right_Arm_Yellow", right_color)
    left_markers = ("left_arm_", "left_gripper_")
    right_markers = ("right_arm_", "right_gripper_")
    left_count = 0
    right_count = 0

    for prim in stage.Traverse():
        path_text = str(prim.GetPath()).lower()
        material = None
        color = None
        if any(marker in path_text for marker in left_markers):
            material = left_material
            color = left_color
        elif any(marker in path_text for marker in right_markers):
            material = right_material
            color = right_color
        if material is None:
            continue

        try:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
        except Exception:
            continue
        if prim.IsA(UsdGeom.Gprim):
            try:
                UsdGeom.Gprim(prim).CreateDisplayColorPrimvar("constant").Set([Gf.Vec3f(*color)])
            except Exception:
                pass
        if material == left_material:
            left_count += 1
        else:
            right_count += 1

    print(f"[INFO] Colorized robot arms for playback: left_prims={left_count} right_prims={right_count}")


class _PlaybackDebugAxes:
    """Playback-only frame markers for checking whether TCPs and targets coincide."""

    def __init__(
        self,
        env,
        task_name: str,
        axis_scale: float,
        grasp_target_offset_b: tuple[float, float, float],
    ):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        self._env = getattr(env, "unwrapped", env)
        self._task_name = task_name
        self._grasp_target_offset_b = torch.tensor(grasp_target_offset_b, device=self._env.device, dtype=torch.float32)
        self._resolved = False
        self._warned = False

        def make_marker(name: str):
            cfg = FRAME_MARKER_CFG.copy()
            cfg.prim_path = f"/Visuals/PlaybackDebugAxes/{name}"
            for marker_cfg in cfg.markers.values():
                if hasattr(marker_cfg, "scale"):
                    marker_cfg.scale = (axis_scale, axis_scale, axis_scale)
            return VisualizationMarkers(cfg)

        self._markers = {
            "left_tcp": make_marker("left_tcp"),
            "right_tcp": make_marker("right_tcp"),
            "left_target": make_marker("left_target"),
            "right_target": make_marker("right_target"),
        }

    def _resolve_once(self) -> bool:
        if self._resolved:
            return True
        try:
            robot = self._env.scene["robot"]
            self._robot = robot
            self._left_tcp_body_ids = [int(body_id) for body_id in robot.find_bodies(LEFT_TCP_POSITION_LINKS)[0]]
            self._right_tcp_body_ids = [int(body_id) for body_id in robot.find_bodies(RIGHT_TCP_POSITION_LINKS)[0]]
            self._left_tcp_orientation_body_id = int(robot.find_bodies([LEFT_TCP_ORIENTATION_LINK])[0][0])
            self._right_tcp_orientation_body_id = int(robot.find_bodies([RIGHT_TCP_ORIENTATION_LINK])[0][0])
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Unable to initialize playback debug TCP axes: {exc}")
                self._warned = True
            return False

        self._resolved = True
        return True

    def _tcp_pose_w(
        self, body_ids: list[int], orientation_body_id: int, orientation_offset: tuple[float, float, float, float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos_w = self._robot.data.body_pos_w[:, body_ids, :]
        body_quat_w = self._robot.data.body_quat_w[:, body_ids, :]
        local_offsets = torch.tensor(
            [TCP_CONTACT_LOCAL_OFFSETS.get(self._robot.body_names[body_id], (0.0, 0.0, 0.0)) for body_id in body_ids],
            device=body_pos_w.device,
            dtype=body_pos_w.dtype,
        )
        offsets_w = quat_apply(
            body_quat_w.reshape(-1, 4),
            local_offsets.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1).reshape(-1, 3),
        ).reshape_as(body_pos_w)
        pos_w = (body_pos_w + offsets_w).mean(dim=1)
        link_quat_w = self._robot.data.body_quat_w[:, orientation_body_id, :]
        offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype)
        quat_w = quat_mul(link_quat_w, offset.view(1, 4).expand_as(link_quat_w))
        return pos_w, quat_w

    def _command_target_pose_w(self, command_name: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        command_manager = getattr(self._env, "command_manager", None)
        terms = getattr(command_manager, "_terms", None)
        term = terms.get(command_name) if isinstance(terms, dict) else None
        if term is None or not hasattr(term, "pose_command_w"):
            return None
        pose_command_w = term.pose_command_w
        return pose_command_w[:, :3], pose_command_w[:, 3:]

    def _scene_object_pose_w(self, object_name: str, side: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        try:
            obj = self._env.scene[object_name]
        except Exception:
            return None
        pos_w = obj.data.root_pos_w + self._grasp_target_offset_b.view(1, 3)
        target_quat = LEFT_GRASP_TARGET_QUAT_W if side == "left" else RIGHT_GRASP_TARGET_QUAT_W
        quat_w = torch.tensor(target_quat, device=pos_w.device, dtype=pos_w.dtype).view(1, 4).expand(pos_w.shape[0], -1)
        return pos_w, quat_w

    def _target_pose_w(self, side: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        command_name = f"{side}_ee_pose"
        command_pose = self._command_target_pose_w(command_name)
        if command_pose is not None:
            return command_pose

        if side == "right":
            right_object_pose = self._scene_object_pose_w("right_object", side)
            if right_object_pose is not None:
                return right_object_pose
        return self._scene_object_pose_w("object", side)

    def update(self) -> None:
        if not self._resolve_once():
            return

        try:
            left_tcp_pos, left_tcp_quat = self._tcp_pose_w(
                self._left_tcp_body_ids, self._left_tcp_orientation_body_id, LEFT_TCP_ORIENTATION_OFFSET_QUAT
            )
            right_tcp_pos, right_tcp_quat = self._tcp_pose_w(
                self._right_tcp_body_ids, self._right_tcp_orientation_body_id, RIGHT_TCP_ORIENTATION_OFFSET_QUAT
            )
            left_target = self._target_pose_w("left")
            right_target = self._target_pose_w("right")

            self._markers["left_tcp"].visualize(left_tcp_pos, left_tcp_quat)
            self._markers["right_tcp"].visualize(right_tcp_pos, right_tcp_quat)
            if left_target is not None:
                self._markers["left_target"].visualize(left_target[0], left_target[1])
            if right_target is not None:
                self._markers["right_target"].visualize(right_target[0], right_target[1])
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Failed to update playback debug axes for {self._task_name}: {exc}")
                self._warned = True


class _PlaybackFingertipContactPoints:
    """Playback markers for the inner narrow3/wide3 collision-surface points."""

    _BODY_NAMES = (
        "left_gripper_narrow3_link",
        "left_gripper_wide3_link",
        "right_gripper_narrow3_link",
        "right_gripper_wide3_link",
    )

    def __init__(self, env, radius: float = 0.003):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.visualization_markers import VisualizationMarkersCfg

        self._env = getattr(env, "unwrapped", env)
        self._resolved = False
        self._warned = False
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/PlaybackFingertipContactPoints",
            markers={
                "narrow": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.05)),
                ),
                "wide": sim_utils.SphereCfg(
                    radius=radius,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 1.0, 0.15)),
                ),
            },
        )
        self._marker = VisualizationMarkers(cfg)

    def _resolve_once(self) -> bool:
        if self._resolved:
            return True
        try:
            self._robot = self._env.scene["robot"]
            self._body_ids = [int(self._robot.find_bodies([name])[0][0]) for name in self._BODY_NAMES]
            self._local_offsets = torch.tensor(
                [FINGERTIP_INNER_CONTACT_LOCAL_OFFSETS[name] for name in self._BODY_NAMES],
                device=self._env.device,
                dtype=torch.float32,
            )
            self._marker_indices = torch.tensor(
                [0, 1, 0, 1], device=self._env.device, dtype=torch.int32
            ).view(1, 4).expand(self._env.num_envs, -1).reshape(-1)
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Unable to initialize playback fingertip points: {exc}")
                self._warned = True
            return False
        self._resolved = True
        return True

    def update(self) -> None:
        if not self._resolve_once():
            return
        try:
            body_pos_w = self._robot.data.body_pos_w[:, self._body_ids, :]
            body_quat_w = self._robot.data.body_quat_w[:, self._body_ids, :]
            offsets_w = quat_apply(
                body_quat_w.reshape(-1, 4),
                self._local_offsets.to(dtype=body_pos_w.dtype)
                .unsqueeze(0)
                .expand(body_pos_w.shape[0], -1, -1)
                .reshape(-1, 3),
            ).reshape_as(body_pos_w)
            self._marker.visualize(
                translations=(body_pos_w + offsets_w).reshape(-1, 3),
                marker_indices=self._marker_indices,
            )
        except Exception as exc:
            if not self._warned:
                print(f"[WARN] Failed to update playback fingertip points: {exc}")
                self._warned = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] No pre-trained checkpoint is currently available for this task.")
            return
    elif args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    env_cfg.log_dir = log_dir
    _apply_saved_agent_compat_for_checkpoint(agent_cfg, resume_path)
    checkpoint_obs_dim = _infer_checkpoint_obs_dim(resume_path)
    applied_saved_env_compat = _apply_saved_env_compat_for_checkpoint(env_cfg, resume_path)
    if checkpoint_obs_dim is not None and not applied_saved_env_compat:
        _apply_obs_compat_for_checkpoint(env_cfg, checkpoint_obs_dim)
    _apply_network_compat_for_checkpoint(agent_cfg, resume_path)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg) and hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = _parse_vec3(args_cli.camera_eye, (3.2, 3.4, 3.0))
        env_cfg.viewer.lookat = _parse_vec3(args_cli.camera_lookat, (0.65, 0.0, 1.35))

    _set_robot_gripper_default_closed(env_cfg)
    _enable_collider_visualization()

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    except Exception:
        print("[ERROR] Failed while constructing playback environment.")
        traceback.print_exc()
        raise

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if checkpoint_obs_dim is not None:
        env_obs_dim = _infer_env_policy_obs_dim(env)
        if env_obs_dim is None:
            raise RuntimeError("Unable to infer policy observation dimension from the constructed environment.")
        if env_obs_dim != checkpoint_obs_dim:
            raise RuntimeError(
                "Checkpoint observation dimension mismatch after compatibility handling: "
                f"checkpoint expects {checkpoint_obs_dim}, environment provides {env_obs_dim}. "
                "Use a checkpoint trained with the current observation layout or add another compatibility mapping."
            )

    if args_cli.colorize_arms:
        _colorize_robot_arms_for_playback(
            _parse_vec3(args_cli.left_arm_color, (0.05, 0.35, 1.0)),
            _parse_vec3(args_cli.right_arm_color, (1.0, 0.82, 0.05)),
        )

    debug_axes = None
    if args_cli.debug_axes:
        debug_axes = _PlaybackDebugAxes(
            env,
            args_cli.task,
            max(float(args_cli.debug_axis_scale), 0.01),
            _parse_vec3(args_cli.debug_grasp_target_offset, (0.0, 0.0, 0.0)),
        )
    fingertip_markers = _PlaybackFingertipContactPoints(env) if args_cli.debug_fingertip_contact_points else None

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    try:
        env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)
    except Exception:
        print("[ERROR] Failed while wrapping playback environment for RL-Games.")
        traceback.print_exc()
        raise

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    try:
        runner = Runner()
        runner.load(agent_cfg)
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)
        agent.reset()
    except Exception:
        print("[ERROR] Failed while loading playback checkpoint/player.")
        traceback.print_exc()
        raise

    dt = env.unwrapped.step_dt
    try:
        obs = env.reset()
    except Exception:
        print("[ERROR] Failed while resetting playback environment.")
        traceback.print_exc()
        raise
    if isinstance(obs, dict):
        obs = obs["obs"]
    if debug_axes is not None:
        debug_axes.update()
    if fingertip_markers is not None:
        fingertip_markers.update()
    diagnostics = _GripperDiagnostics(env) if args_cli.print_gripper_diagnostics else None
    if diagnostics is not None:
        diagnostics.print(0)
    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            actions = _override_gripper_close_actions(actions, timestep)
            obs, _, dones, _ = env.step(actions)
            if debug_axes is not None:
                debug_axes.update()
            if fingertip_markers is not None:
                fingertip_markers.update()
            if diagnostics is not None and (timestep + 1) % max(1, int(args_cli.diagnostic_every)) == 0:
                diagnostics.print(timestep + 1)

            if len(dones) > 0 and agent.is_rnn and agent.states is not None:
                for state in agent.states:
                    state[:, dones, :] = 0.0

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
