"""Play back RL-Games checkpoints for JZ Isaac Lab tasks."""

from __future__ import annotations

import argparse
import os
import sys
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
    LEFT_ARM_JOINTS,
    LEFT_TCP_ORIENTATION_LINK,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_TCP_ORIENTATION_LINK,
    RIGHT_TCP_POSITION_LINKS,
)
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

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            obs = agent.obs_to_torch(obs)
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)

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
