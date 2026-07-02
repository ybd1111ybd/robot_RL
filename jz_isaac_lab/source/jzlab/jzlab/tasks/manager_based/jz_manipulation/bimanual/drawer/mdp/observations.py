"""Observation helpers for the JZ bimanual drawer task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _left_tcp_midpoint_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["left_ee_frame"].data.target_pos_w[..., :2, :].mean(dim=1)


def _right_tcp_midpoint_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["right_ee_frame"].data.target_pos_w[..., :2, :].mean(dim=1)


def _handle_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    return env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]


def tcp_to_handle_distances_b(env: ManagerBasedEnv) -> torch.Tensor:
    """Left/right TCP-to-handle vectors expressed in the robot base frame."""

    robot: Articulation = env.scene["robot"]
    handle_pos_w = _handle_pos_w(env)
    left_tcp_pos_w = _left_tcp_midpoint_pos_w(env)
    right_tcp_pos_w = _right_tcp_midpoint_pos_w(env)

    handle_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, handle_pos_w)
    left_tcp_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, left_tcp_pos_w)
    right_tcp_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, right_tcp_pos_w)

    left_delta_b = handle_pos_b - left_tcp_pos_b
    right_delta_b = handle_pos_b - right_tcp_pos_b
    return torch.cat((left_delta_b, right_delta_b), dim=1)


def last_action_padded(env: ManagerBasedEnv, dim: int) -> torch.Tensor:
    """Return the last action tensor padded to a fixed width."""

    actions = env.action_manager.action
    if actions.shape[1] >= dim:
        return actions[:, :dim]

    padding = torch.zeros((env.num_envs, dim - actions.shape[1]), device=actions.device, dtype=actions.dtype)
    return torch.cat((actions, padding), dim=1)
