"""Evaluate a JZ Grasp RL-Games checkpoint against object/TCP distances."""

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


parser = argparse.ArgumentParser(description="Evaluate a Grasp checkpoint for TCP-to-object approach accuracy.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rl_games_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model checkpoint.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=300, help="Number of evaluation steps to run.")
parser.add_argument("--near_threshold", type=float, default=0.05, help="Distance threshold for near-object ratio.")
parser.add_argument("--left_object", type=str, default="object", help="Scene object used as the left target root.")
parser.add_argument("--right_object", type=str, default="right_object", help="Scene object used as the right target root.")
parser.add_argument("--left_target_offset", type=str, default="0,0,0", help="Left target offset from object root, xyz.")
parser.add_argument("--right_target_offset", type=str, default="0,0,0", help="Right target offset from object root, xyz.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time if possible.")
parser.add_argument("--stochastic", action="store_true", default=False, help="Sample actions instead of deterministic policy.")
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
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import jzlab.tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from jzlab.tasks.manager_based.jz_manipulation.bimanual.grasp import mdp as grasp_mdp
from jzlab.tasks.manager_based.jz_manipulation.constants import (
    LEFT_ARM_JOINTS,
    LEFT_TCP_ORIENTATION_LINK,
    LEFT_TCP_ORIENTATION_OFFSET_QUAT,
    LEFT_TCP_POSITION_LINKS,
    RIGHT_ARM_JOINTS,
    RIGHT_TCP_ORIENTATION_LINK,
    RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
    RIGHT_TCP_POSITION_LINKS,
    TCP_CONTACT_LOCAL_OFFSETS,
)


def _parse_vec3(text: str) -> tuple[float, float, float]:
    values = [float(item.strip()) for item in str(text).split(",")]
    if len(values) != 3:
        raise ValueError(f"Expected xyz vector, got: {text!r}")
    return (values[0], values[1], values[2])


def _load_checkpoint_file(checkpoint_path: str) -> dict:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _load_checkpoint_yaml(checkpoint_path: str, filename: str) -> dict | None:
    checkpoint_file = Path(checkpoint_path).resolve()
    yaml_path = checkpoint_file.parent.parent / "params" / filename
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


def _apply_checkpoint_network_compat(agent_cfg: dict, checkpoint_path: str) -> None:
    checkpoint_units = _infer_checkpoint_mlp_units(checkpoint_path)
    if checkpoint_units is None:
        return
    mlp_cfg = agent_cfg.get("params", {}).get("network", {}).get("mlp", {})
    if list(mlp_cfg.get("units", [])) != checkpoint_units:
        mlp_cfg["units"] = checkpoint_units
        print(f"[INFO] Using checkpoint MLP hidden units: {checkpoint_units}")


def _apply_saved_agent_compat(agent_cfg: dict, checkpoint_path: str) -> None:
    saved_agent_cfg = _load_checkpoint_yaml(checkpoint_path, "agent.yaml")
    if not saved_agent_cfg:
        return
    live_env_cfg = agent_cfg.setdefault("params", {}).setdefault("env", {})
    saved_env_cfg = saved_agent_cfg.get("params", {}).get("env", {})
    for key in ("clip_observations", "clip_actions"):
        if key in saved_env_cfg:
            live_env_cfg[key] = saved_env_cfg[key]


def _infer_env_policy_obs_dim(env) -> int | None:
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


def _action_manager_raw_actions(env) -> torch.Tensor:
    action_manager = getattr(env.unwrapped, "action_manager", None)
    if action_manager is None:
        return torch.zeros((env.unwrapped.num_envs, 0), device=env.unwrapped.device)
    chunks = []
    for term_name in ("left_arm_action", "right_arm_action", "left_gripper_action", "right_gripper_action"):
        try:
            term = action_manager.get_term(term_name)
        except Exception:
            continue
        raw = getattr(term, "raw_actions", None)
        if raw is not None:
            chunks.append(raw)
    if not chunks:
        return torch.zeros((env.unwrapped.num_envs, 0), device=env.unwrapped.device)
    return torch.cat(chunks, dim=1)


def _tcp_orientation_metrics(
    env,
    tcp_pos_w: torch.Tensor,
    orientation_body_id: int,
    orientation_offset: tuple[float, float, float, float],
    object_cfg: SceneEntityCfg,
    nominal_inward_direction: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene[object_cfg.name]
    link_quat_w = robot.data.body_quat_w[:, orientation_body_id, :]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))

    unit_y = torch.tensor((0.0, 1.0, 0.0), device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).expand_as(tcp_pos_w)
    unit_z = torch.tensor((0.0, 0.0, 1.0), device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).expand_as(tcp_pos_w)
    tcp_y_w = quat_apply(tcp_quat_w, unit_y)
    tcp_z_w = quat_apply(tcp_quat_w, unit_z)
    object_z_w = quat_apply(obj.data.root_quat_w, unit_z)

    to_object = obj.data.root_pos_w - tcp_pos_w
    planar = to_object - torch.sum(to_object * object_z_w, dim=-1, keepdim=True) * object_z_w
    planar_norm = torch.linalg.vector_norm(planar, dim=-1, keepdim=True)
    nominal = torch.tensor(nominal_inward_direction, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype).expand_as(planar)
    target_y_w = torch.where(planar_norm > 0.03, planar / planar_norm.clamp_min(1.0e-6), nominal)

    z_dot = torch.sum(tcp_z_w * object_z_w, dim=-1).clamp(-1.0, 1.0)
    y_dot = torch.sum(tcp_y_w * target_y_w, dim=-1).clamp(-1.0, 1.0)
    distance = torch.linalg.vector_norm(to_object, dim=-1)
    gate = torch.exp(-torch.square(distance / 0.20))
    score = gate * 0.5 * (z_dot.clamp(0.0, 1.0) + y_dot.clamp(0.0, 1.0))
    return z_dot, y_dot, score


def _tcp_midpoint_position_w(env, tcp_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    body_ids = list(tcp_cfg.body_ids)
    body_pos_w = robot.data.body_pos_w[:, body_ids, :]
    body_quat_w = robot.data.body_quat_w[:, body_ids, :]
    local_offsets = torch.tensor(
        [TCP_CONTACT_LOCAL_OFFSETS.get(robot.body_names[body_id], (0.0, 0.0, 0.0)) for body_id in body_ids],
        device=body_pos_w.device,
        dtype=body_pos_w.dtype,
    )
    offsets_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        local_offsets.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1).reshape(-1, 3),
    ).reshape_as(body_pos_w)
    return (body_pos_w + offsets_w).mean(dim=1)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict) -> None:
    left_target_offset = _parse_vec3(args_cli.left_target_offset)
    right_target_offset = _parse_vec3(args_cli.right_target_offset)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["seed"] = env_cfg.seed

    resume_path = retrieve_file_path(args_cli.checkpoint)
    _apply_saved_agent_compat(agent_cfg, resume_path)
    _apply_checkpoint_network_compat(agent_cfg, resume_path)
    checkpoint_obs_dim = _infer_checkpoint_obs_dim(resume_path)

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
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    left_tcp_cfg = SceneEntityCfg("robot", body_names=LEFT_TCP_POSITION_LINKS)
    right_tcp_cfg = SceneEntityCfg("robot", body_names=RIGHT_TCP_POSITION_LINKS)
    left_object_cfg = SceneEntityCfg(args_cli.left_object)
    right_object_cfg = SceneEntityCfg(args_cli.right_object)
    left_joint_cfg = SceneEntityCfg("robot", joint_names=LEFT_ARM_JOINTS)
    right_joint_cfg = SceneEntityCfg("robot", joint_names=RIGHT_ARM_JOINTS)
    left_orientation_cfg = SceneEntityCfg("robot", body_names=[LEFT_TCP_ORIENTATION_LINK])
    right_orientation_cfg = SceneEntityCfg("robot", body_names=[RIGHT_TCP_ORIENTATION_LINK])
    configs_to_resolve = [
        left_tcp_cfg,
        right_tcp_cfg,
        left_object_cfg,
        right_object_cfg,
        left_joint_cfg,
        right_joint_cfg,
        left_orientation_cfg,
        right_orientation_cfg,
    ]
    surface_pregrasp = "SurfacePregrasp" in args_cli.task
    if surface_pregrasp:
        left_narrow_cfg = SceneEntityCfg("robot", body_names=["left_gripper_narrow3_link"])
        left_wide_cfg = SceneEntityCfg("robot", body_names=["left_gripper_wide3_link"])
        right_narrow_cfg = SceneEntityCfg("robot", body_names=["right_gripper_narrow3_link"])
        right_wide_cfg = SceneEntityCfg("robot", body_names=["right_gripper_wide3_link"])
        configs_to_resolve.extend((left_narrow_cfg, left_wide_cfg, right_narrow_cfg, right_wide_cfg))
    for cfg in configs_to_resolve:
        cfg.resolve(env.unwrapped.scene)

    left_target_samples = []
    right_target_samples = []
    left_center_samples = []
    right_center_samples = []
    near_target_samples = []
    near_center_samples = []
    max_action_samples = []
    action_rate_samples = []
    joint_vel_max_samples = []
    left_z_dot_samples = []
    right_z_dot_samples = []
    left_y_dot_samples = []
    right_y_dot_samples = []
    left_orientation_score_samples = []
    right_orientation_score_samples = []
    surface_metric_samples: dict[str, list[float]] = {}
    prev_actions: torch.Tensor | None = None

    final_left_object_pos_b = None
    final_right_object_pos_b = None
    final_left_tcp_b = None
    final_right_tcp_b = None

    for _ in range(args_cli.steps):
        start_time = time.time()
        with torch.inference_mode():
            obs_torch = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_torch, is_deterministic=not args_cli.stochastic)
            obs, _, dones, _ = env.step(actions)
            if isinstance(obs, dict):
                obs = obs["obs"]

        left_target_dist = grasp_mdp.tcp_to_object_side_target_distance(
            env.unwrapped, tcp_asset_cfg=left_tcp_cfg, target_offset=left_target_offset, object_cfg=left_object_cfg
        )
        right_target_dist = grasp_mdp.tcp_to_object_side_target_distance(
            env.unwrapped, tcp_asset_cfg=right_tcp_cfg, target_offset=right_target_offset, object_cfg=right_object_cfg
        )
        left_center_dist = grasp_mdp.tcp_to_object_distance(
            env.unwrapped, tcp_asset_cfg=left_tcp_cfg, object_cfg=left_object_cfg
        )
        right_center_dist = grasp_mdp.tcp_to_object_distance(
            env.unwrapped, tcp_asset_cfg=right_tcp_cfg, object_cfg=right_object_cfg
        )
        left_tcp_w = _tcp_midpoint_position_w(env, left_tcp_cfg)
        right_tcp_w = _tcp_midpoint_position_w(env, right_tcp_cfg)
        left_z_dot, left_y_dot, left_orientation_score = _tcp_orientation_metrics(
            env,
            left_tcp_w,
            int(left_orientation_cfg.body_ids[0]),
            LEFT_TCP_ORIENTATION_OFFSET_QUAT,
            left_object_cfg,
            (0.0, -1.0, 0.0),
        )
        right_z_dot, right_y_dot, right_orientation_score = _tcp_orientation_metrics(
            env,
            right_tcp_w,
            int(right_orientation_cfg.body_ids[0]),
            RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
            right_object_cfg,
            (0.0, 1.0, 0.0),
        )
        if surface_pregrasp:
            side_metrics = {
                "left": grasp_mdp.fingertip_pregrasp_diagnostics(
                    env.unwrapped,
                    grasp_axis_sign=-1.0,
                    cylinder_radius=0.03,
                    cylinder_half_height=0.06,
                    narrow_cfg=left_narrow_cfg,
                    wide_cfg=left_wide_cfg,
                    tcp_orientation_cfg=left_orientation_cfg,
                    orientation_offset=LEFT_TCP_ORIENTATION_OFFSET_QUAT,
                    narrow_sensor_name="left_narrow_contact",
                    wide_sensor_name="left_wide_contact",
                    object_cfg=left_object_cfg,
                ),
                "right": grasp_mdp.fingertip_pregrasp_diagnostics(
                    env.unwrapped,
                    grasp_axis_sign=1.0,
                    cylinder_radius=0.03,
                    cylinder_half_height=0.06,
                    narrow_cfg=right_narrow_cfg,
                    wide_cfg=right_wide_cfg,
                    tcp_orientation_cfg=right_orientation_cfg,
                    orientation_offset=RIGHT_TCP_ORIENTATION_OFFSET_QUAT,
                    narrow_sensor_name="right_narrow_contact",
                    wide_sensor_name="right_wide_contact",
                    object_cfg=right_object_cfg,
                ),
            }
            desired_radial_distance = (
                0.05
                if "SurfacePregraspV42" in args_cli.task
                or "SurfacePregraspV43" in args_cli.task
                or "SurfacePregraspV44" in args_cli.task
                else 0.03
            )
            desired_height_offset = (
                0.03
                if "SurfacePregraspV41" in args_cli.task
                or "SurfacePregraspV42" in args_cli.task
                or "SurfacePregraspV43" in args_cli.task
                or "SurfacePregraspV44" in args_cli.task
                else 0.0
            )
            side_metrics["left"].update(
                grasp_mdp.tcp_cylinder_axis_diagnostics(
                    env.unwrapped,
                    desired_radial_distance=desired_radial_distance,
                    desired_height_offset=desired_height_offset,
                    tcp_asset_cfg=left_tcp_cfg,
                    object_cfg=left_object_cfg,
                )
            )
            side_metrics["right"].update(
                grasp_mdp.tcp_cylinder_axis_diagnostics(
                    env.unwrapped,
                    desired_radial_distance=desired_radial_distance,
                    desired_height_offset=desired_height_offset,
                    tcp_asset_cfg=right_tcp_cfg,
                    object_cfg=right_object_cfg,
                )
            )
            if "SurfacePregraspV44" in args_cli.task:
                for side, finger_sensors, palm_sensors in (
                    (
                        "left",
                        (
                            "left_narrow_contact",
                            "left_wide_contact",
                            "left_narrow1_contact",
                            "left_narrow2_contact",
                            "left_wide1_contact",
                            "left_wide2_contact",
                        ),
                        ("left_palm9_contact", "left_palm10_contact"),
                    ),
                    (
                        "right",
                        (
                            "right_narrow_contact",
                            "right_wide_contact",
                            "right_narrow1_contact",
                            "right_narrow2_contact",
                            "right_wide1_contact",
                            "right_wide2_contact",
                        ),
                        ("right_palm9_contact", "right_palm10_contact"),
                    ),
                ):
                    full_force = torch.stack(
                        [
                            grasp_mdp.filtered_contact_force_magnitude(env.unwrapped, sensor_name)
                            for sensor_name in finger_sensors
                        ],
                        dim=1,
                    ).amax(dim=1)
                    palm_force = torch.stack(
                        [
                            grasp_mdp.filtered_contact_force_magnitude(env.unwrapped, sensor_name)
                            for sensor_name in palm_sensors
                        ],
                        dim=1,
                    ).amax(dim=1)
                    side_metrics[side]["full_finger_force"] = full_force
                    side_metrics[side]["palm_force"] = palm_force
                    side_metrics[side]["full_gripper_contact_ratio"] = (
                        (full_force > 0.5) | (palm_force > 0.5)
                    ).float()
            for side, metrics in side_metrics.items():
                for metric_name, values in metrics.items():
                    surface_metric_samples.setdefault(f"{side}_{metric_name}", []).append(values.mean().item())
                contact = (metrics["narrow_force"] > 1.0) | (metrics["wide_force"] > 1.0)
                surface_metric_samples.setdefault(f"{side}_early_contact_ratio", []).append(
                    contact.float().mean().item()
                )
        final_left_object_pos_b = grasp_mdp.object_position_in_robot_root_frame(
            env.unwrapped, object_cfg=left_object_cfg
        )[0].detach().cpu()
        final_right_object_pos_b = grasp_mdp.object_position_in_robot_root_frame(
            env.unwrapped, object_cfg=right_object_cfg
        )[0].detach().cpu()
        final_left_tcp_b = grasp_mdp.fingertip_midpoint_position_b(env.unwrapped, asset_cfg=left_tcp_cfg)[0].detach().cpu()
        final_right_tcp_b = grasp_mdp.fingertip_midpoint_position_b(env.unwrapped, asset_cfg=right_tcp_cfg)[0].detach().cpu()

        joint_vel = torch.cat(
            (
                env.unwrapped.scene["robot"].data.joint_vel[:, left_joint_cfg.joint_ids],
                env.unwrapped.scene["robot"].data.joint_vel[:, right_joint_cfg.joint_ids],
            ),
            dim=1,
        )
        joint_vel_max = torch.max(torch.abs(joint_vel), dim=1).values
        raw_actions = _action_manager_raw_actions(env)
        if raw_actions.shape[1] == 0:
            action_rate = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
            max_action = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
        else:
            max_action = torch.max(torch.abs(raw_actions), dim=1).values
            if prev_actions is None:
                action_rate = torch.zeros(raw_actions.shape[0], device=raw_actions.device)
            else:
                action_rate = torch.norm(raw_actions - prev_actions, dim=1)
            prev_actions = raw_actions.clone()

        near_target = (left_target_dist <= args_cli.near_threshold) & (right_target_dist <= args_cli.near_threshold)
        near_center = (left_center_dist <= args_cli.near_threshold) & (right_center_dist <= args_cli.near_threshold)

        left_target_samples.append(left_target_dist.mean().item())
        right_target_samples.append(right_target_dist.mean().item())
        left_center_samples.append(left_center_dist.mean().item())
        right_center_samples.append(right_center_dist.mean().item())
        near_target_samples.append(near_target.float().mean().item())
        near_center_samples.append(near_center.float().mean().item())
        max_action_samples.append(max_action.mean().item())
        action_rate_samples.append(action_rate.mean().item())
        joint_vel_max_samples.append(joint_vel_max.mean().item())
        left_z_dot_samples.append(left_z_dot.mean().item())
        right_z_dot_samples.append(right_z_dot.mean().item())
        left_y_dot_samples.append(left_y_dot.mean().item())
        right_y_dot_samples.append(right_y_dot.mean().item())
        left_orientation_score_samples.append(left_orientation_score.mean().item())
        right_orientation_score_samples.append(right_orientation_score.mean().item())

        if len(dones) > 0 and agent.is_rnn and agent.states is not None:
            for state in agent.states:
                state[:, dones, :] = 0.0

        if args_cli.real_time:
            sleep_time = env.unwrapped.step_dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def summarize(values: list[float]) -> tuple[float, float]:
        mean = float(np.mean(values))
        tail = float(np.mean(values[-50:])) if len(values) >= 50 else mean
        return mean, tail

    print(f"checkpoint: {resume_path}")
    print(f"task: {args_cli.task}")
    print(f"steps: {args_cli.steps}")
    print(f"stochastic: {bool(args_cli.stochastic)}")
    print(f"near_threshold: {float(args_cli.near_threshold):.6f}")
    print(f"left_object: {args_cli.left_object}")
    print(f"right_object: {args_cli.right_object}")
    print(f"left_target_offset: {left_target_offset}")
    print(f"right_target_offset: {right_target_offset}")
    print(f"left_object_pos_b_last: {None if final_left_object_pos_b is None else final_left_object_pos_b.tolist()}")
    print(f"right_object_pos_b_last: {None if final_right_object_pos_b is None else final_right_object_pos_b.tolist()}")
    print(f"left_tcp_b_last: {None if final_left_tcp_b is None else final_left_tcp_b.tolist()}")
    print(f"right_tcp_b_last: {None if final_right_tcp_b is None else final_right_tcp_b.tolist()}")
    for name, values in (
        ("left_target_dist", left_target_samples),
        ("right_target_dist", right_target_samples),
        ("left_object_center_dist", left_center_samples),
        ("right_object_center_dist", right_center_samples),
        ("near_target_ratio", near_target_samples),
        ("near_object_center_ratio", near_center_samples),
        ("max_action", max_action_samples),
        ("action_rate", action_rate_samples),
        ("joint_vel_max_abs", joint_vel_max_samples),
        ("left_tcp_z_object_z_dot", left_z_dot_samples),
        ("right_tcp_z_object_z_dot", right_z_dot_samples),
        ("left_tcp_y_to_object_dot", left_y_dot_samples),
        ("right_tcp_y_to_object_dot", right_y_dot_samples),
        ("left_orientation_reward_raw", left_orientation_score_samples),
        ("right_orientation_reward_raw", right_orientation_score_samples),
    ):
        mean, tail = summarize(values)
        print(f"{name}_mean: {mean:.6f}")
        print(f"{name}_last50: {tail:.6f}")
    for name, values in surface_metric_samples.items():
        mean, tail = summarize(values)
        print(f"{name}_mean: {mean:.6f}")
        print(f"{name}_last50: {tail:.6f}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
