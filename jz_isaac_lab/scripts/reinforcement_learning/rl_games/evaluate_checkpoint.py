"""Evaluate a JZ RL-Games checkpoint for reach accuracy and smoothness."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

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


parser = argparse.ArgumentParser(description="Evaluate a checkpoint for reach accuracy and stability.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=600, help="Number of evaluation steps to run.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument(
    "--stochastic",
    action="store_true",
    default=False,
    help="Sample actions from the policy instead of using deterministic inference.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import yaml

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

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
from isaaclab_tasks.utils.hydra import hydra_task_config


def _load_checkpoint_file(checkpoint_path: str) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_checkpoint_yaml(checkpoint_path: str, filename: str) -> dict | None:
    checkpoint_file = Path(checkpoint_path).resolve()
    run_dir = checkpoint_file.parent.parent
    yaml_path = run_dir / "params" / filename
    if not yaml_path.is_file():
        return None
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.unsafe_load(f)


def _infer_checkpoint_obs_dim(checkpoint_path: str) -> int | None:
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
    checkpoint = _load_checkpoint_file(checkpoint_path)
    model_state = checkpoint.get("model", {})
    units: list[int] = []
    layer_index = 0
    while True:
        weight = model_state.get(f"a2c_network.actor_mlp.{layer_index}.weight")
        if weight is None or len(weight.shape) != 2:
            break
        units.append(int(weight.shape[0]))
        layer_index += 2
    return units or None


def _apply_network_compat_for_checkpoint(agent_cfg: dict, checkpoint_path: str) -> bool:
    checkpoint_units = _infer_checkpoint_mlp_units(checkpoint_path)
    if checkpoint_units is None:
        return False
    network_cfg = agent_cfg.get("params", {}).get("network", {})
    mlp_cfg = network_cfg.get("mlp", {})
    current_units = list(mlp_cfg.get("units", []))
    if current_units == checkpoint_units:
        return False
    mlp_cfg["units"] = checkpoint_units
    return True


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


def _apply_obs_compat_for_checkpoint(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, obs_dim: int) -> bool:
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

    return changed


def _build_action_cfg(saved_action_cfg: dict, joint_names: list[str]):
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


def _apply_saved_agent_compat_for_checkpoint(agent_cfg: dict, checkpoint_path: str) -> bool:
    saved_agent_cfg = _load_checkpoint_yaml(checkpoint_path, "agent.yaml")
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
    return changed


def _apply_saved_env_compat_for_checkpoint(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, checkpoint_path: str
) -> bool:
    saved_env_cfg = _load_checkpoint_yaml(checkpoint_path, "env.yaml")
    if not saved_env_cfg:
        return False

    changed = False
    # Skip restoring the ground spawn if the saved config used a cloud-hosted USD
    # (the network is not accessible so this would raise FileNotFoundError)
    saved_ground_usd = (
        saved_env_cfg.get("scene", {}).get("ground", {}).get("spawn", {}).get("usd_path", "")
    )
    if saved_ground_usd and (
        saved_ground_usd.startswith("http://")
        or saved_ground_usd.startswith("https://")
        or saved_ground_usd.startswith("s3://")
    ):
        saved_env_cfg["scene"].pop("ground", None)

    policy_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    saved_policy_cfg = saved_env_cfg.get("observations", {}).get("policy", {})
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

    commands_cfg = saved_env_cfg.get("commands", {})
    for command_name in ("left_ee_pose", "right_ee_pose"):
        saved_command_cfg = commands_cfg.get(command_name, {})
        live_command_cfg = getattr(getattr(env_cfg, "commands", None), command_name, None)
        if live_command_cfg is None:
            continue
        for attr_name in ("dataset_key", "body_name", "debug_vis", "use_fixed_quaternion", "fixed_quaternion", "resampling_time_range", "current_position_body_names", "current_quaternion_offset"):
            if attr_name in saved_command_cfg:
                setattr(live_command_cfg, attr_name, saved_command_cfg[attr_name])
                changed = True

    if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "physx"):
        saved_physx_cfg = saved_env_cfg.get("sim", {}).get("physx", {})
        ext_forces = saved_physx_cfg.get("enable_external_forces_every_iteration")
        if ext_forces is not None:
            env_cfg.sim.physx.enable_external_forces_every_iteration = ext_forces
            changed = True

    live_robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    robot_cfg = saved_env_cfg.get("scene", {}).get("robot", {})
    if live_robot_cfg is not None and getattr(live_robot_cfg, "spawn", None) is not None:
        articulation_props = getattr(live_robot_cfg.spawn, "articulation_props", None)
        solver_vel_iters = robot_cfg.get("spawn", {}).get("articulation_props", {}).get("solver_velocity_iteration_count")
        if articulation_props is not None and solver_vel_iters is not None:
            articulation_props.solver_velocity_iteration_count = solver_vel_iters
            changed = True
    if live_robot_cfg is not None and getattr(live_robot_cfg, "actuators", None) is not None:
        arm_actuator = live_robot_cfg.actuators.get("arm")
        if arm_actuator is not None:
            arm_cfg = robot_cfg.get("actuators", {}).get("arm", {})
            if "stiffness" in arm_cfg:
                arm_actuator.stiffness = arm_cfg["stiffness"]
                changed = True
            if "damping" in arm_cfg:
                arm_actuator.damping = arm_cfg["damping"]
                changed = True

    saved_actions_cfg = saved_env_cfg.get("actions", {})
    left_action_cfg = _build_action_cfg(saved_actions_cfg.get("left_arm_action", {}), LEFT_ARM_JOINTS)
    right_action_cfg = _build_action_cfg(saved_actions_cfg.get("right_arm_action", {}), RIGHT_ARM_JOINTS)
    if left_action_cfg is not None:
        env_cfg.actions.left_arm_action = left_action_cfg
        changed = True
    if right_action_cfg is not None:
        env_cfg.actions.right_arm_action = right_action_cfg
        changed = True

    return changed


def _infer_env_policy_obs_dim(env: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg | gym.Env) -> int | None:
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
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["seed"] = env_cfg.seed

    resume_path = retrieve_file_path(args_cli.checkpoint)

    _apply_saved_agent_compat_for_checkpoint(agent_cfg, resume_path)
    checkpoint_obs_dim = _infer_checkpoint_obs_dim(resume_path)
    applied_saved_env_compat = _apply_saved_env_compat_for_checkpoint(env_cfg, resume_path)
    if checkpoint_obs_dim is not None and not applied_saved_env_compat:
        _apply_obs_compat_for_checkpoint(env_cfg, checkpoint_obs_dim)
    _apply_network_compat_for_checkpoint(agent_cfg, resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if checkpoint_obs_dim is not None:
        env_obs_dim = _infer_env_policy_obs_dim(env)
        if env_obs_dim != checkpoint_obs_dim:
            raise RuntimeError(f"Observation mismatch: env={env_obs_dim}, checkpoint={checkpoint_obs_dim}")

    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    dt = env.unwrapped.step_dt
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    prev_actions: torch.Tensor | None = None
    prev_joint_vel: torch.Tensor | None = None
    consecutive_settle = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.int32)
    left_tcp_cfg = SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)
    right_tcp_cfg = SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)
    left_joint_cfg = SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)
    right_joint_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)

    left_error_samples = []
    right_error_samples = []
    left_tcp_speed_samples = []
    right_tcp_speed_samples = []
    action_rate_samples = []
    max_action_samples = []
    joint_vel_max_abs_samples = []
    joint_accel_max_abs_samples = []
    settle_ratio_samples = []
    settle_streak_samples = []
    near_goal_ratio_samples = []

    for _ in range(args_cli.steps):
        start_time = time.time()
        with torch.inference_mode():
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=not args_cli.stochastic)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

        left_error = torch.norm(
            mdp.fingertip_midpoint_position_command_error_vector_b(
                env.unwrapped, command_name="left_ee_pose", asset_cfg=left_tcp_cfg
            ),
            dim=1,
        )
        right_error = torch.norm(
            mdp.fingertip_midpoint_position_command_error_vector_b(
                env.unwrapped, command_name="right_ee_pose", asset_cfg=right_tcp_cfg
            ),
            dim=1,
        )
        left_tcp_speed = torch.norm(mdp.fingertip_midpoint_linear_velocity_b(env.unwrapped, asset_cfg=left_tcp_cfg), dim=1)
        right_tcp_speed = torch.norm(
            mdp.fingertip_midpoint_linear_velocity_b(env.unwrapped, asset_cfg=right_tcp_cfg), dim=1
        )
        joint_vel = torch.cat(
            (
                mdp.joint_vel_rel(env.unwrapped, asset_cfg=left_joint_cfg),
                mdp.joint_vel_rel(env.unwrapped, asset_cfg=right_joint_cfg),
            ),
            dim=1,
        )
        joint_vel_max_abs = torch.max(torch.abs(joint_vel), dim=1).values

        executed_actions = torch.cat(
            (
                env.unwrapped.action_manager.get_term("left_arm_action").raw_actions,
                env.unwrapped.action_manager.get_term("right_arm_action").raw_actions,
            ),
            dim=1,
        )
        if prev_actions is None:
            action_rate = torch.zeros(executed_actions.shape[0], device=executed_actions.device)
        else:
            action_rate = torch.norm(executed_actions - prev_actions, dim=1)
        prev_actions = executed_actions.clone()
        if prev_joint_vel is None:
            joint_accel_max_abs = torch.zeros(joint_vel.shape[0], device=joint_vel.device)
        else:
            joint_accel_max_abs = torch.max(torch.abs((joint_vel - prev_joint_vel) / dt), dim=1).values
        prev_joint_vel = joint_vel.clone()

        max_action = torch.max(torch.abs(executed_actions), dim=1).values
        is_near_goal = (left_error <= 0.04) & (right_error <= 0.04)
        is_settled = (
            is_near_goal
            & (left_tcp_speed <= 0.03)
            & (right_tcp_speed <= 0.03)
            & (joint_vel_max_abs <= 0.20)
            & (action_rate <= 0.03)
        )
        consecutive_settle = torch.where(is_settled, consecutive_settle + 1, torch.zeros_like(consecutive_settle))

        left_error_samples.append(left_error.mean().item())
        right_error_samples.append(right_error.mean().item())
        left_tcp_speed_samples.append(left_tcp_speed.mean().item())
        right_tcp_speed_samples.append(right_tcp_speed.mean().item())
        action_rate_samples.append(action_rate.mean().item())
        max_action_samples.append(max_action.mean().item())
        joint_vel_max_abs_samples.append(joint_vel_max_abs.mean().item())
        joint_accel_max_abs_samples.append(joint_accel_max_abs.mean().item())
        near_goal_ratio_samples.append(is_near_goal.float().mean().item())
        settle_ratio_samples.append(is_settled.float().mean().item())
        settle_streak_samples.append((consecutive_settle >= 15).float().mean().item())

        if len(dones) > 0 and agent.is_rnn and agent.states is not None:
            for state in agent.states:
                state[:, dones, :] = 0.0

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def summarize(values: list[float]) -> tuple[float, float]:
        return float(np.mean(values)), float(np.mean(values[-50:])) if len(values) >= 50 else float(np.mean(values))

    left_error_mean, left_error_tail = summarize(left_error_samples)
    right_error_mean, right_error_tail = summarize(right_error_samples)
    left_tcp_speed_mean, left_tcp_speed_tail = summarize(left_tcp_speed_samples)
    right_tcp_speed_mean, right_tcp_speed_tail = summarize(right_tcp_speed_samples)
    action_rate_mean, action_rate_tail = summarize(action_rate_samples)
    max_action_mean, max_action_tail = summarize(max_action_samples)
    joint_vel_max_abs_mean, joint_vel_max_abs_tail = summarize(joint_vel_max_abs_samples)
    joint_accel_max_abs_mean, joint_accel_max_abs_tail = summarize(joint_accel_max_abs_samples)
    near_goal_mean, near_goal_tail = summarize(near_goal_ratio_samples)
    settle_mean, settle_tail = summarize(settle_ratio_samples)
    settle_streak_mean, settle_streak_tail = summarize(settle_streak_samples)

    print(f"checkpoint: {resume_path}")
    print(f"steps: {args_cli.steps}")
    print(f"stochastic: {args_cli.stochastic}")
    print(f"left_error_mean: {left_error_mean:.6f}")
    print(f"left_error_last50: {left_error_tail:.6f}")
    print(f"right_error_mean: {right_error_mean:.6f}")
    print(f"right_error_last50: {right_error_tail:.6f}")
    print(f"left_tcp_speed_mean: {left_tcp_speed_mean:.6f}")
    print(f"left_tcp_speed_last50: {left_tcp_speed_tail:.6f}")
    print(f"right_tcp_speed_mean: {right_tcp_speed_mean:.6f}")
    print(f"right_tcp_speed_last50: {right_tcp_speed_tail:.6f}")
    print(f"action_rate_mean: {action_rate_mean:.6f}")
    print(f"action_rate_last50: {action_rate_tail:.6f}")
    print(f"max_action_mean: {max_action_mean:.6f}")
    print(f"max_action_last50: {max_action_tail:.6f}")
    print(f"joint_vel_max_abs_mean: {joint_vel_max_abs_mean:.6f}")
    print(f"joint_vel_max_abs_last50: {joint_vel_max_abs_tail:.6f}")
    print(f"joint_accel_max_abs_mean: {joint_accel_max_abs_mean:.6f}")
    print(f"joint_accel_max_abs_last50: {joint_accel_max_abs_tail:.6f}")
    print(f"near_goal_ratio_mean: {near_goal_mean:.6f}")
    print(f"near_goal_ratio_last50: {near_goal_tail:.6f}")
    print(f"settle_ratio_mean: {settle_mean:.6f}")
    print(f"settle_ratio_last50: {settle_tail:.6f}")
    print(f"settle_streak_ratio_mean: {settle_streak_mean:.6f}")
    print(f"settle_streak_ratio_last50: {settle_streak_tail:.6f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
