"""Custom reward helpers for JZ synthetic TCP tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _as_batch_quat(env: ManagerBasedRLEnv, quat: tuple[float, float, float, float]) -> torch.Tensor:
    return torch.tensor(quat, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)


def _base_link_index(asset: RigidObject) -> int:
    return asset.body_names.index("base_link")


def _base_link_pose_w(asset: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    base_idx = _base_link_index(asset)
    return asset.data.body_pos_w[:, base_idx], asset.data.body_quat_w[:, base_idx]


def _fingertip_midpoint_pos_w(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)  # type: ignore[index]


def _fingertip_midpoint_lin_vel_w(asset: RigidObject, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].mean(dim=1)  # type: ignore[index]


def _reward_state_tensor(
    env: ManagerBasedRLEnv,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not hasattr(env, "_jz_reward_state"):
        env._jz_reward_state = {}

    state = env._jz_reward_state
    tensor = state.get(name)
    if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
        tensor = torch.zeros(shape, device=env.device, dtype=dtype)
        state[name] = tensor
    return tensor


def fingertip_midpoint_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Synthetic TCP position of the gripper center in the robot base_link frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    base_pos_w, base_quat_w = _base_link_pose_w(asset)
    curr_pos_w = _fingertip_midpoint_pos_w(asset, asset_cfg)
    curr_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, curr_pos_w)
    return curr_pos_b


def fingertip_midpoint_linear_velocity_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Synthetic TCP linear velocity expressed in the robot base_link frame."""

    asset: RigidObject = env.scene[asset_cfg.name]
    _, base_quat_w = _base_link_pose_w(asset)
    curr_lin_vel_w = _fingertip_midpoint_lin_vel_w(asset, asset_cfg)
    return quat_apply_inverse(base_quat_w, curr_lin_vel_w)


def fingertip_midpoint_position_command_error_vector_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Desired minus current gripper-center position in the base_link frame."""

    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    curr_pos_b = fingertip_midpoint_position_b(env, asset_cfg)
    return des_pos_b - curr_pos_b


def generated_command_positions(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Desired command position in the base_link frame."""

    return env.command_manager.get_command(command_name)[:, :3]


def fingertip_midpoint_position_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 position error between the desired TCP command and fingertip midpoint."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    base_pos_w, base_quat_w = _base_link_pose_w(asset)
    des_pos_w, _ = combine_frame_transforms(base_pos_w, base_quat_w, des_pos_b)
    curr_pos_w = _fingertip_midpoint_pos_w(asset, asset_cfg)
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def fingertip_midpoint_position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Tanh-shaped position reward using the fingertip midpoint as synthetic TCP."""

    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    return 1.0 - torch.tanh(distance / std)


def fingertip_midpoint_position_command_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Positive reward for reducing TCP position error while a command stays unchanged."""

    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    command = env.command_manager.get_command(command_name)[:, :3]

    prev_distance = _reward_state_tensor(
        env, f"{command_name}_prev_distance", tuple(distance.shape), distance.dtype
    )
    prev_command = _reward_state_tensor(
        env, f"{command_name}_prev_command", tuple(command.shape), command.dtype
    )

    is_reset = env.episode_length_buf == 0
    command_changed = torch.norm(command - prev_command, dim=1) > 1.0e-6

    reward = torch.clamp(prev_distance - distance, min=0.0)
    reward = torch.where(is_reset | command_changed, torch.zeros_like(reward), reward)

    prev_distance.copy_(distance)
    prev_command.copy_(command)
    return reward


def fingertip_midpoint_position_command_success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Binary bonus when the synthetic TCP is inside the goal radius."""

    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    return (distance <= threshold).to(distance.dtype)


def bimanual_fingertip_midpoint_position_command_success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    left_command_name: str,
    left_asset_cfg: SceneEntityCfg,
    right_command_name: str,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Binary bonus when both synthetic TCPs are inside the goal radius together."""

    left_distance = fingertip_midpoint_position_command_error(env, left_command_name, left_asset_cfg)
    right_distance = fingertip_midpoint_position_command_error(env, right_command_name, right_asset_cfg)
    return ((left_distance <= threshold) & (right_distance <= threshold)).to(left_distance.dtype)


def fingertip_midpoint_speed_l2_when_close_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """TCP linear-speed penalty activated only when the gripper center is close to the target."""

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    is_close = (distance <= threshold).to(asset.data.joint_vel.dtype)
    tcp_speed_sq = torch.sum(torch.square(_fingertip_midpoint_lin_vel_w(asset, asset_cfg)), dim=1)
    return is_close * tcp_speed_sq


def action_rate_l2_when_close_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    action_name: str,
) -> torch.Tensor:
    """Action-rate penalty activated only near the goal to suppress target-point dithering."""

    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    action_term = env.action_manager.get_term(action_name)
    current_action = action_term.raw_actions
    prev_action = _reward_state_tensor(
        env, f"{action_name}_prev_raw_action", tuple(current_action.shape), current_action.dtype
    )

    is_close = (distance <= threshold).to(current_action.dtype)
    is_reset = env.episode_length_buf == 0
    action_rate_sq = torch.sum(torch.square(current_action - prev_action), dim=1)
    action_rate_sq = torch.where(is_reset, torch.zeros_like(action_rate_sq), action_rate_sq)
    prev_action.copy_(current_action)
    return is_close * action_rate_sq


def fingertip_midpoint_stable_goal_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    speed_threshold: float,
) -> torch.Tensor:
    """Bonus for being near the goal while keeping the TCP almost stationary."""

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    tcp_speed = torch.norm(_fingertip_midpoint_lin_vel_w(asset, asset_cfg), dim=1)
    return ((distance <= threshold) & (tcp_speed <= speed_threshold)).to(distance.dtype)


def fingertip_midpoint_stable_goal_dwell_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
    speed_threshold: float,
    hold_steps: int,
) -> torch.Tensor:
    """Reward consecutive stable steps near the goal to encourage settling instead of dithering."""

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = fingertip_midpoint_position_command_error(env, command_name, asset_cfg)
    command = env.command_manager.get_command(command_name)[:, :3]
    tcp_speed = torch.norm(_fingertip_midpoint_lin_vel_w(asset, asset_cfg), dim=1)

    prev_command = _reward_state_tensor(
        env, f"{command_name}_stable_prev_command", tuple(command.shape), command.dtype
    )
    streak = _reward_state_tensor(
        env, f"{command_name}_stable_streak", tuple(distance.shape), torch.int32
    )

    is_reset = env.episode_length_buf == 0
    command_changed = torch.norm(command - prev_command, dim=1) > 1.0e-6
    stable = (distance <= threshold) & (tcp_speed <= speed_threshold)

    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(is_reset | command_changed, torch.zeros_like(streak), streak)
    env._jz_reward_state[f"{command_name}_stable_streak"] = streak
    prev_command.copy_(command)

    dwell = torch.clamp(streak.to(distance.dtype) / float(hold_steps), max=1.0)
    return dwell


def bimanual_fingertip_midpoint_stable_goal_dwell_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    speed_threshold: float,
    hold_steps: int,
    left_command_name: str,
    left_asset_cfg: SceneEntityCfg,
    right_command_name: str,
    right_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward consecutive stable steps only when both end-effectors settle together."""

    left_distance = fingertip_midpoint_position_command_error(env, left_command_name, left_asset_cfg)
    right_distance = fingertip_midpoint_position_command_error(env, right_command_name, right_asset_cfg)
    left_speed = torch.norm(_fingertip_midpoint_lin_vel_w(env.scene[left_asset_cfg.name], left_asset_cfg), dim=1)
    right_speed = torch.norm(_fingertip_midpoint_lin_vel_w(env.scene[right_asset_cfg.name], right_asset_cfg), dim=1)
    left_command = env.command_manager.get_command(left_command_name)[:, :3]
    right_command = env.command_manager.get_command(right_command_name)[:, :3]
    command = torch.cat((left_command, right_command), dim=1)

    prev_command = _reward_state_tensor(env, "bimanual_stable_prev_command", tuple(command.shape), command.dtype)
    streak = _reward_state_tensor(env, "bimanual_stable_streak", tuple(left_distance.shape), torch.int32)

    is_reset = env.episode_length_buf == 0
    command_changed = torch.norm(command - prev_command, dim=1) > 1.0e-6
    stable = (
        (left_distance <= threshold)
        & (right_distance <= threshold)
        & (left_speed <= speed_threshold)
        & (right_speed <= speed_threshold)
    )

    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(is_reset | command_changed, torch.zeros_like(streak), streak)
    env._jz_reward_state["bimanual_stable_streak"] = streak
    prev_command.copy_(command)

    dwell = torch.clamp(streak.to(left_distance.dtype) / float(hold_steps), max=1.0)
    return dwell


def joint_vel_l2_when_close_to_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    tcp_asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Joint-velocity penalty activated only when the gripper center is already close to the target."""

    asset: RigidObject = env.scene[asset_cfg.name]
    distance = fingertip_midpoint_position_command_error(env, command_name, tcp_asset_cfg)
    is_close = (distance <= threshold).to(asset.data.joint_vel.dtype)
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return is_close * joint_vel_sq


def action_max_abs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the largest absolute action component to suppress single-joint spikes."""

    return torch.max(torch.abs(env.action_manager.action), dim=1).values


def orientation_command_error_with_offset(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    offset_quat: tuple[float, float, float, float],
) -> torch.Tensor:
    """Orientation error against a body frame composed with a fixed local quaternion offset."""

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    _, base_quat_w = _base_link_pose_w(asset)
    des_quat_w = quat_mul(base_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore[index]
    curr_quat_w = quat_mul(curr_quat_w, _as_batch_quat(env, offset_quat))
    return quat_error_magnitude(curr_quat_w, des_quat_w)
