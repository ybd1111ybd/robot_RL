"""Reward helpers for the JZ bimanual drawer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

from ....constants import LEFT_GRIPPER_JOINTS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _left_tcp_midpoint_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["left_ee_frame"].data.target_pos_w[..., :2, :].mean(dim=1)


def _right_tcp_midpoint_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["right_ee_frame"].data.target_pos_w[..., :2, :].mean(dim=1)


def _left_tcp_orientation_quat_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["left_ee_frame"].data.target_quat_w[..., 2, :]


def _handle_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]


def _handle_quat_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]


def _drawer_joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    cabinet: Articulation = env.scene["cabinet"]
    joint_id = cabinet.joint_names.index("drawer_bottom_joint")
    return cabinet.data.joint_pos[:, joint_id]


def _left_handle_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.norm(_handle_pos_w(env) - _left_tcp_midpoint_pos_w(env), dim=1)


def _tcp_midpoint_pos_w(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    return robot.data.body_pos_w[:, tcp_asset_cfg.body_ids, :].mean(dim=1)


def _tcp_midpoint_lin_vel_w(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    return robot.data.body_lin_vel_w[:, tcp_asset_cfg.body_ids, :].mean(dim=1)


def approach_handle(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    """Reward moving the left working hand towards the drawer handle."""

    distance = _left_handle_distance(env)
    return 1.0 - torch.tanh(distance / std)


def align_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward orienting the left gripper to pull the handle cleanly."""

    ee_tcp_rot_mat = matrix_from_quat(_left_tcp_orientation_quat_w(env))
    handle_rot_mat = matrix_from_quat(_handle_quat_w(env))

    handle_x = handle_rot_mat[..., 0]
    handle_y = handle_rot_mat[..., 1]
    ee_x = ee_tcp_rot_mat[..., 0]
    ee_z = ee_tcp_rot_mat[..., 2]

    align_z = torch.bmm(ee_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_x.unsqueeze(1), handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)

    return 0.5 * (torch.clamp(align_z, min=0.0) ** 2 + torch.clamp(align_x, min=0.0) ** 2)


def grasp_handle(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    closed_joint_target: float,
) -> torch.Tensor:
    """Reward closing the left gripper once it is close enough to the handle."""

    robot: Articulation = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in LEFT_GRIPPER_JOINTS]
    mean_abs_joint = torch.mean(torch.abs(robot.data.joint_pos[:, joint_ids]), dim=1)
    closed_score = torch.clamp(1.0 - mean_abs_joint / closed_joint_target, min=0.0, max=1.0)
    return (_left_handle_distance(env) <= distance_threshold).to(closed_score.dtype) * closed_score


def drawer_opening_progress_gated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward drawer opening more strongly when the left hand is already in a pull-ready pose."""

    drawer_pos = torch.clamp(_drawer_joint_pos(env), min=0.0)
    approach_score = approach_handle(env, std=0.10)
    align_score = align_handle(env)
    grasp_score = grasp_handle(env, distance_threshold=0.07, closed_joint_target=0.08)
    gate = torch.clamp(0.35 * approach_score + 0.35 * align_score + 0.30 * grasp_score, max=1.0)
    return drawer_pos * (0.2 + 0.8 * gate)


def drawer_open_success(env: ManagerBasedRLEnv, open_threshold: float) -> torch.Tensor:
    """Binary success bonus once the bottom drawer reaches the desired opening."""

    return (_drawer_joint_pos(env) >= open_threshold).to(torch.float32)


def right_arm_stay_neutral(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    joint_values: list[float],
) -> torch.Tensor:
    """Penalize the support arm for drifting away from its parked configuration."""

    robot: Articulation = env.scene["robot"]
    joint_ids = [robot.joint_names.index(name) for name in joint_names]
    target = torch.tensor(joint_values, device=env.device, dtype=robot.data.joint_pos.dtype).unsqueeze(0)
    return torch.sum(torch.square(robot.data.joint_pos[:, joint_ids] - target), dim=1)


def table_penetration_penalty(
    env: ManagerBasedRLEnv,
    top_z: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    margin: float,
) -> torch.Tensor:
    """Penalize fingertips moving through the tabletop region."""

    fingertip_pos_w = torch.cat(
        (
            env.scene["left_ee_frame"].data.target_pos_w[..., :2, :],
            env.scene["right_ee_frame"].data.target_pos_w[..., :2, :],
        ),
        dim=1,
    )

    over_table = (
        (fingertip_pos_w[..., 0] >= x_min - margin)
        & (fingertip_pos_w[..., 0] <= x_max + margin)
        & (fingertip_pos_w[..., 1] >= y_min - margin)
        & (fingertip_pos_w[..., 1] <= y_max + margin)
    )
    penetration = torch.clamp((top_z + margin) - fingertip_pos_w[..., 2], min=0.0)
    return torch.sum(over_table.to(penetration.dtype) * penetration, dim=1)


def tcp_approach_speed_reward(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    tcp_pos_w = _tcp_midpoint_pos_w(env, tcp_asset_cfg)
    handle_pos_w = _handle_pos_w(env)
    to_obj = handle_pos_w - tcp_pos_w
    dist = torch.norm(to_obj, dim=1).clamp(min=1e-6)
    direction = to_obj / dist.unsqueeze(-1)
    vel = _tcp_midpoint_lin_vel_w(env, tcp_asset_cfg)
    approach_speed = torch.sum(vel * direction, dim=1)
    speed_reward = torch.exp(-torch.square((approach_speed - 0.05) / 0.03))
    active = (dist <= 0.25).to(speed_reward.dtype)
    return active * speed_reward

def tcp_closing_speed_penalty(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    tcp_pos_w = _tcp_midpoint_pos_w(env, tcp_asset_cfg)
    handle_pos_w = _handle_pos_w(env)
    dist = torch.norm(handle_pos_w - tcp_pos_w, dim=1)
    speed = torch.norm(_tcp_midpoint_lin_vel_w(env, tcp_asset_cfg), dim=1)
    approach_mode = (dist <= 0.20).to(speed.dtype)
    return approach_mode * torch.square(speed)

def gripper_contact_effort(env: ManagerBasedRLEnv, gripper_asset_cfg: SceneEntityCfg):
    from isaaclab.assets import Articulation
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    effort = torch.mean(torch.abs(robot.data.applied_torque[:, gripper_asset_cfg.joint_ids]), dim=1)
    return torch.clamp(effort / 0.5, max=1.0)

def cabinet_displacement_penalty(env: ManagerBasedRLEnv):
    from isaaclab.assets import Articulation
    cabinet: Articulation = env.scene["cabinet"]
    lin_vel_sq = torch.sum(torch.square(cabinet.data.root_lin_vel_w), dim=1)
    ang_vel_sq = torch.sum(torch.square(cabinet.data.root_ang_vel_w), dim=1)
    return lin_vel_sq + 0.5 * ang_vel_sq

def drawer_stable_pull_reward(env: ManagerBasedRLEnv):
    from isaaclab.assets import Articulation
    cabinet: Articulation = env.scene["cabinet"]
    joint_id = cabinet.joint_names.index("drawer_bottom_joint")
    drawer_vel = torch.abs(cabinet.data.joint_vel[:, joint_id])
    steady = (drawer_vel > 0.001).to(drawer_vel.dtype)
    return steady * drawer_vel


def action_max_abs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.max(torch.abs(env.action_manager.action), dim=1).values
