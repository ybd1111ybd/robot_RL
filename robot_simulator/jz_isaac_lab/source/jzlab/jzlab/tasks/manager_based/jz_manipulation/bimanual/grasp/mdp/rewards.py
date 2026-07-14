"""Observation and reward helpers for the JZ bimanual grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul, subtract_frame_transforms

from ....constants import FINGERTIP_INNER_CONTACT_LOCAL_OFFSETS, TCP_CONTACT_LOCAL_OFFSETS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _base_link_pose_w(asset: Articulation):
    base_idx = asset.body_names.index("base_link")
    return asset.data.body_pos_w[:, base_idx], asset.data.body_quat_w[:, base_idx]


def _fingertip_midpoint_pos_w(asset: Articulation, asset_cfg: SceneEntityCfg):
    body_ids = list(asset_cfg.body_ids)
    body_pos_w = asset.data.body_pos_w[:, body_ids, :]
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    local_offsets = torch.tensor(
        [TCP_CONTACT_LOCAL_OFFSETS.get(asset.body_names[body_id], (0.0, 0.0, 0.0)) for body_id in body_ids],
        device=body_pos_w.device,
        dtype=body_pos_w.dtype,
    )
    offsets_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        local_offsets.unsqueeze(0).expand(body_pos_w.shape[0], -1, -1).reshape(-1, 3),
    ).reshape_as(body_pos_w)
    return (body_pos_w + offsets_w).mean(dim=1)


def _fingertip_midpoint_lin_vel_w(asset: Articulation, asset_cfg: SceneEntityCfg):
    body_ids = list(asset_cfg.body_ids)
    body_quat_w = asset.data.body_quat_w[:, body_ids, :]
    body_lin_vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_ids, :]
    local_offsets = torch.tensor(
        [TCP_CONTACT_LOCAL_OFFSETS.get(asset.body_names[body_id], (0.0, 0.0, 0.0)) for body_id in body_ids],
        device=body_lin_vel_w.device,
        dtype=body_lin_vel_w.dtype,
    )
    offsets_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        local_offsets.unsqueeze(0).expand(body_lin_vel_w.shape[0], -1, -1).reshape(-1, 3),
    ).reshape_as(body_lin_vel_w)
    return (body_lin_vel_w + torch.cross(body_ang_vel_w, offsets_w, dim=-1)).mean(dim=1)


def _reward_state_tensor(env: ManagerBasedRLEnv, name: str, shape, dtype):
    if not hasattr(env, "_jz_reward_state"):
        env._jz_reward_state = {}
    state = env._jz_reward_state
    tensor = state.get(name)
    if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
        tensor = torch.zeros(shape, device=env.device, dtype=dtype)
        state[name] = tensor
    return tensor


def fingertip_midpoint_position_b(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    base_pos_w, base_quat_w = _base_link_pose_w(asset)
    curr_pos_w = _fingertip_midpoint_pos_w(asset, asset_cfg)
    curr_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, curr_pos_w)
    return curr_pos_b

def object_position_in_robot_root_frame(env: ManagerBasedEnv, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    base_pos_w, base_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
    obj_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, obj_pos_w)
    return obj_pos_b


def fingertip_midpoint_to_object_vector_b(
    env: ManagerBasedEnv,
    tcp_asset_cfg: SceneEntityCfg,
    robot_cfg=SceneEntityCfg("robot"),
    object_cfg=SceneEntityCfg("object"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    base_pos_w, base_quat_w = _base_link_pose_w(robot)
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    obj_pos_w = obj.data.root_pos_w
    tcp_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, tcp_pos_w)
    obj_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, obj_pos_w)
    return obj_pos_b - tcp_pos_b


def _object_side_target_pos_w(env: ManagerBasedEnv, object_cfg: SceneEntityCfg, target_offset: tuple[float, float, float]):
    obj: RigidObject = env.scene[object_cfg.name]
    offset = torch.tensor(target_offset, device=env.device, dtype=obj.data.root_pos_w.dtype).unsqueeze(0)
    return obj.data.root_pos_w + offset


def fingertip_midpoint_to_object_side_target_vector_b(
    env: ManagerBasedEnv,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    robot_cfg=SceneEntityCfg("robot"),
    object_cfg=SceneEntityCfg("object"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_pos_w, base_quat_w = _base_link_pose_w(robot)
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    target_pos_w = _object_side_target_pos_w(env, object_cfg=object_cfg, target_offset=target_offset)
    tcp_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, tcp_pos_w)
    target_pos_b, _ = subtract_frame_transforms(base_pos_w, base_quat_w, target_pos_w)
    return target_pos_b - tcp_pos_b


def tcp_to_object_side_target_distance(
    env: ManagerBasedRLEnv,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    object_cfg=SceneEntityCfg("object"),
):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    target_pos_w = _object_side_target_pos_w(env, object_cfg=object_cfg, target_offset=target_offset)
    return torch.norm(tcp_pos_w - target_pos_w, dim=1)


def tcp_to_object_side_target_distance_tanh(
    env: ManagerBasedRLEnv,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    object_cfg=SceneEntityCfg("object"),
    std=0.10,
):
    distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=tcp_asset_cfg, target_offset=target_offset, object_cfg=object_cfg
    )
    return 1.0 - torch.tanh(distance / std)


def tcp_to_object_side_target_progress_reward(
    env: ManagerBasedRLEnv,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    object_cfg=SceneEntityCfg("object"),
):
    distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=tcp_asset_cfg, target_offset=target_offset, object_cfg=object_cfg
    )
    cache_key = f"tcp_side_target_progress_{tcp_asset_cfg.name}_{target_offset}"
    prev_distance = _reward_state_tensor(env, cache_key, tuple(distance.shape), distance.dtype)
    reward = torch.clamp(prev_distance - distance, min=0.0)
    is_reset = env.episode_length_buf == 0
    reward = torch.where(is_reset, torch.zeros_like(reward), reward)
    prev_distance.copy_(distance)
    return reward

def tcp_to_object_distance(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    return torch.norm(tcp_pos_w - obj_pos_w, dim=1)

def tcp_to_object_distance_tanh(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object"), std=0.10):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    return 1.0 - torch.tanh(distance / std)


def tcp_to_object_distance_tracking(
    env: ManagerBasedRLEnv,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    std=0.25,
):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    return torch.exp(-torch.square(distance / std))


def tcp_to_object_signed_progress_reward(
    env: ManagerBasedRLEnv,
    reward_key: str,
    max_progress: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    previous = _reward_state_tensor(env, reward_key, tuple(distance.shape), distance.dtype)
    progress = torch.clamp(previous - distance, min=-max_progress, max=max_progress)
    progress = torch.where(env.episode_length_buf == 0, torch.zeros_like(progress), progress)
    previous.copy_(distance)
    return progress


def _tcp_cylinder_axis_state(
    env: ManagerBasedEnv,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return radial distance, signed height, radial speed, and relative TCP speed."""

    robot: Articulation = env.scene[tcp_asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    tcp_vel_w = _fingertip_midpoint_lin_vel_w(robot, tcp_asset_cfg)
    rel_pos_w = tcp_pos_w - obj.data.root_pos_w
    rel_vel_w = tcp_vel_w - obj.data.root_lin_vel_w

    z_basis = torch.zeros_like(rel_pos_w)
    z_basis[:, 2] = 1.0
    cylinder_z_w = quat_apply(obj.data.root_quat_w, z_basis)
    height = torch.sum(rel_pos_w * cylinder_z_w, dim=1)
    radial_vec_w = rel_pos_w - height.unsqueeze(1) * cylinder_z_w
    radial_distance = torch.linalg.vector_norm(radial_vec_w, dim=1)
    radial_direction_w = radial_vec_w / radial_distance.unsqueeze(1).clamp_min(1.0e-6)
    radial_speed = torch.sum(rel_vel_w * radial_direction_w, dim=1)
    relative_speed = torch.linalg.vector_norm(rel_vel_w, dim=1)
    return radial_distance, height, radial_speed, relative_speed


def tcp_cylinder_axis_diagnostics(
    env: ManagerBasedEnv,
    desired_radial_distance: float,
    tcp_asset_cfg: SceneEntityCfg,
    desired_height_offset: float = 0.0,
    object_cfg=SceneEntityCfg("object"),
) -> dict[str, torch.Tensor]:
    radial_distance, height, radial_speed, relative_speed = _tcp_cylinder_axis_state(
        env, tcp_asset_cfg, object_cfg
    )
    return {
        "tcp_axis_radial_distance": radial_distance,
        "tcp_axis_radial_error": radial_distance - desired_radial_distance,
        "tcp_axis_height_error": height - desired_height_offset,
        "tcp_axis_radial_speed": radial_speed,
        "tcp_axis_relative_speed": relative_speed,
    }


def tcp_cylinder_axis_tracking(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    radial_std: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    radial_distance, _, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    return torch.exp(-torch.square((radial_distance - desired_radial_distance) / radial_std))


def tcp_cylinder_height_tracking(
    env: ManagerBasedRLEnv,
    desired_height_offset: float,
    height_std: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    _, height, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    return torch.exp(-torch.square((height - desired_height_offset) / height_std))


def tcp_cylinder_radial_abs_error(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return absolute TCP radial stand-off error from the cylinder axis."""

    radial_distance, _, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    return torch.abs(radial_distance - desired_radial_distance)


def tcp_cylinder_height_abs_error(
    env: ManagerBasedRLEnv,
    desired_height_offset: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return absolute TCP height error in the cylinder frame."""

    _, height, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    return torch.abs(height - desired_height_offset)


def tcp_cylinder_radial_progress_reward(
    env: ManagerBasedRLEnv,
    reward_key: str,
    desired_radial_distance: float,
    max_progress: float,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    additional_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    radial_distance, _, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    error = torch.abs(radial_distance - desired_radial_distance)
    previous = _reward_state_tensor(env, reward_key, tuple(error.shape), error.dtype)
    progress = torch.clamp(previous - error, min=-max_progress, max=max_progress)
    contact = (filtered_contact_force_magnitude(env, narrow_sensor_name) > force_threshold) | (
        filtered_contact_force_magnitude(env, wide_sensor_name) > force_threshold
    )
    for sensor_name in additional_sensor_names:
        contact |= filtered_contact_force_magnitude(env, sensor_name) > force_threshold
    progress = torch.where(contact, torch.minimum(progress, torch.zeros_like(progress)), progress)
    progress = torch.where(env.episode_length_buf == 0, torch.zeros_like(progress), progress)
    previous.copy_(error)
    return progress


def tcp_cylinder_radial_speed_near_target_l2(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    gate_std: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    radial_distance, _, radial_speed, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    gate = torch.exp(-torch.square((radial_distance - desired_radial_distance) / gate_std))
    return gate * torch.square(radial_speed)


def tcp_cylinder_relative_speed_near_target_l2(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    gate_std: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    radial_distance, _, _, relative_speed = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    gate = torch.exp(-torch.square((radial_distance - desired_radial_distance) / gate_std))
    return gate * torch.square(relative_speed)


def tcp_cylinder_scheduled_speed_excess_l2(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    far_error: float,
    medium_error: float,
    near_error: float,
    far_speed_limit: float,
    medium_speed_limit: float,
    near_speed_limit: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize only speed above a distance-dependent approach limit."""

    radial_distance, _, _, relative_speed = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    error = torch.abs(radial_distance - desired_radial_distance)
    speed_limit = torch.where(
        error > medium_error,
        torch.full_like(error, far_speed_limit),
        torch.where(
            error > near_error,
            torch.full_like(error, medium_speed_limit),
            torch.full_like(error, near_speed_limit),
        ),
    )
    excess = torch.relu(relative_speed - speed_limit)
    return torch.where(error > far_error, torch.zeros_like(excess), torch.square(excess))


def tcp_cylinder_inward_speed_after_overshoot(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize continued inward motion after crossing the radial stop target."""

    radial_distance, _, radial_speed, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    overshot = radial_distance < desired_radial_distance
    inward_speed = torch.relu(-radial_speed)
    return torch.where(overshot, inward_speed, torch.zeros_like(inward_speed))


def tcp_cylinder_linear_radial_velocity_error(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    error_deadband: float,
    velocity_gain: float,
    max_inward_speed: float,
    max_outward_speed: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    inside_weight_scale: float = 1.0,
) -> torch.Tensor:
    """Track a signed radial velocity target that decreases linearly near the stop radius."""

    radial_distance, _, radial_speed, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    error = radial_distance - desired_radial_distance
    effective_error = torch.sign(error) * torch.relu(torch.abs(error) - error_deadband)
    target_radial_speed = torch.clamp(
        -velocity_gain * effective_error,
        min=-max_inward_speed,
        max=max_outward_speed,
    )
    error_magnitude = torch.abs(radial_speed - target_radial_speed)
    return torch.where(error < 0.0, inside_weight_scale * error_magnitude, error_magnitude)


def tcp_cylinder_near_target_tangential_speed(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    gate_distance: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize non-radial relative speed only near the radial stop target."""

    radial_distance, _, radial_speed, relative_speed = _tcp_cylinder_axis_state(
        env, tcp_asset_cfg, object_cfg
    )
    radial_error = torch.abs(radial_distance - desired_radial_distance)
    gate = torch.clamp(1.0 - radial_error / gate_distance, min=0.0, max=1.0)
    tangential_speed = torch.sqrt(torch.clamp(torch.square(relative_speed) - torch.square(radial_speed), min=0.0))
    return gate * tangential_speed


def tcp_cylinder_static_safety_penetration(
    env: ManagerBasedRLEnv,
    desired_radial_distance: float,
    barrier_width: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return normalized penetration inside the radial safety boundary."""

    radial_distance, _, _, _ = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    return torch.clamp(
        (desired_radial_distance - radial_distance) / barrier_width,
        min=0.0,
        max=1.0,
    )


def persistent_fingertip_contact_penalty(
    env: ManagerBasedRLEnv,
    reward_key: str,
    hold_steps: int,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    narrow_force = filtered_contact_force_magnitude(env, narrow_sensor_name)
    wide_force = filtered_contact_force_magnitude(env, wide_sensor_name)
    contact = (narrow_force > force_threshold) | (wide_force > force_threshold)
    streak = _reward_state_tensor(env, reward_key, tuple(narrow_force.shape), torch.int32)
    streak = torch.where(contact, streak + 1, torch.zeros_like(streak))
    streak = torch.where(env.episode_length_buf == 0, torch.zeros_like(streak), streak)
    env._jz_reward_state[reward_key] = streak
    return torch.clamp(streak.to(narrow_force.dtype) / float(hold_steps), max=1.0)


def any_filtered_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Return one when any filtered contact sensor exceeds the threshold."""

    force = filtered_contact_force_magnitude(env, sensor_names[0])
    contact = force > force_threshold
    for sensor_name in sensor_names[1:]:
        contact |= filtered_contact_force_magnitude(env, sensor_name) > force_threshold
    return contact.to(force.dtype)


def persistent_multi_sensor_contact_penalty(
    env: ManagerBasedRLEnv,
    reward_key: str,
    hold_steps: int,
    sensor_names: tuple[str, ...],
    force_threshold: float,
) -> torch.Tensor:
    """Ramp a persistent-contact penalty across multiple filtered sensors."""

    force = filtered_contact_force_magnitude(env, sensor_names[0])
    contact = force > force_threshold
    for sensor_name in sensor_names[1:]:
        contact |= filtered_contact_force_magnitude(env, sensor_name) > force_threshold
    streak = _reward_state_tensor(env, reward_key, tuple(force.shape), torch.int32)
    streak = torch.where(contact, streak + 1, torch.zeros_like(streak))
    streak = torch.where(env.episode_length_buf == 0, torch.zeros_like(streak), streak)
    env._jz_reward_state[reward_key] = streak
    return torch.clamp(streak.to(force.dtype) / float(hold_steps), max=1.0)


def stable_cylinder_axis_pregrasp_reward(
    env: ManagerBasedRLEnv,
    reward_key: str,
    desired_radial_distance: float,
    radial_tolerance: float,
    desired_height_offset: float,
    height_tolerance: float,
    speed_threshold: float,
    hold_steps: int,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    additional_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
    radial_distance, height, _, relative_speed = _tcp_cylinder_axis_state(env, tcp_asset_cfg, object_cfg)
    contact = (filtered_contact_force_magnitude(env, narrow_sensor_name) > force_threshold) | (
        filtered_contact_force_magnitude(env, wide_sensor_name) > force_threshold
    )
    for sensor_name in additional_sensor_names:
        contact |= filtered_contact_force_magnitude(env, sensor_name) > force_threshold
    stable = (
        (torch.abs(radial_distance - desired_radial_distance) <= radial_tolerance)
        & (torch.abs(height - desired_height_offset) <= height_tolerance)
        & (relative_speed <= speed_threshold)
        & ~contact
    )
    streak = _reward_state_tensor(env, reward_key, tuple(radial_distance.shape), torch.int32)
    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(env.episode_length_buf == 0, torch.zeros_like(streak), streak)
    env._jz_reward_state[reward_key] = streak
    return torch.clamp(streak.to(radial_distance.dtype) / float(hold_steps), max=1.0)


def tcp_to_object_progress_reward(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    cache_key = f"tcp_object_progress_{tcp_asset_cfg.name}"
    prev_distance = _reward_state_tensor(env, cache_key, tuple(distance.shape), distance.dtype)
    reward = torch.clamp(prev_distance - distance, min=0.0)
    is_reset = env.episode_length_buf == 0
    reward = torch.where(is_reset, torch.zeros_like(reward), reward)
    prev_distance.copy_(distance)
    return reward

def tcp_approach_speed_reward(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    obj_pos_w = env.scene[object_cfg.name].data.root_pos_w
    to_object = obj_pos_w - tcp_pos_w
    dist_safe = distance.clamp(min=1e-6)
    direction = to_object / dist_safe.unsqueeze(-1)
    tcp_vel_w = _fingertip_midpoint_lin_vel_w(robot, tcp_asset_cfg)
    approach_speed = torch.sum(tcp_vel_w * direction, dim=1)
    speed_reward = torch.exp(-torch.square((approach_speed - 0.05) / 0.03))
    active = (distance <= 0.25).to(speed_reward.dtype)
    return active * speed_reward

def tcp_closing_speed_penalty(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_lin_vel_w = _fingertip_midpoint_lin_vel_w(robot, tcp_asset_cfg)
    speed = torch.norm(tcp_lin_vel_w, dim=1)
    approach_mode = (distance <= 0.20).to(speed.dtype)
    return approach_mode * torch.square(speed)

def tcp_relative_to_object_speed(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_lin_vel_w = _fingertip_midpoint_lin_vel_w(robot, tcp_asset_cfg)
    rel_lin_vel_w = tcp_lin_vel_w - obj.data.root_lin_vel_w
    return torch.norm(rel_lin_vel_w, dim=1)

def gripper_closed_fraction(env: ManagerBasedEnv, narrow_joint_name: str, wide_joint_name: str, robot_cfg=SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    narrow = robot.data.joint_pos[:, robot.joint_names.index(narrow_joint_name)]
    wide = robot.data.joint_pos[:, robot.joint_names.index(wide_joint_name)]
    narrow_open, narrow_closed_target = -1.2, -0.05
    wide_open, wide_closed_target = 1.2, -0.05
    narrow_closed = ((narrow - narrow_open) / (narrow_closed_target - narrow_open)).clamp(0.0, 1.0)
    wide_closed = ((wide_open - wide) / (wide_open - wide_closed_target)).clamp(0.0, 1.0)
    return 0.5 * (narrow_closed + wide_closed)

def gripper_mean_abs_effort(env: ManagerBasedEnv, gripper_asset_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    return torch.mean(torch.abs(robot.data.applied_torque[:, gripper_asset_cfg.joint_ids]), dim=1)

def gripper_closed_when_near_object(env: ManagerBasedRLEnv, threshold: float, tcp_asset_cfg: SceneEntityCfg, narrow_joint_name: str, wide_joint_name: str, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    is_near = (distance <= threshold).to(dtype=torch.float32)
    return is_near * gripper_closed_fraction(env, narrow_joint_name=narrow_joint_name, wide_joint_name=wide_joint_name, robot_cfg=robot_cfg)

def gripper_contact_reward_when_near_object(env: ManagerBasedRLEnv, distance_threshold: float, effort_threshold: float, tcp_asset_cfg: SceneEntityCfg, gripper_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    effort = gripper_mean_abs_effort(env, gripper_asset_cfg=gripper_asset_cfg)
    is_near = (distance <= distance_threshold).to(effort.dtype)
    return is_near * torch.clamp(effort / effort_threshold, max=1.0)

def gripper_joint_speed_penalty_after_contact(env: ManagerBasedRLEnv, distance_threshold: float, tcp_asset_cfg: SceneEntityCfg, gripper_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    effort = gripper_mean_abs_effort(env, gripper_asset_cfg=gripper_asset_cfg)
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, gripper_asset_cfg.joint_ids]
    is_contact = (effort > 0.5).to(joint_vel.dtype)
    return is_contact * torch.sum(torch.square(joint_vel), dim=1)


def filtered_contact_force_magnitude(env: ManagerBasedEnv, sensor_name: str) -> torch.Tensor:
    """Return the largest filtered normal-force magnitude for one fingertip sensor."""

    sensor = env.scene[sensor_name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(f"Contact sensor '{sensor_name}' needs filter_prim_paths_expr configured.")
    force_magnitude = torch.linalg.vector_norm(force_matrix_w, dim=-1)
    return force_magnitude.reshape(force_magnitude.shape[0], -1).amax(dim=1)


def _inner_fingertip_points_w(
    robot: Articulation,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ray-cast inner collision-surface point on each distal finger."""

    def point_w(cfg: SceneEntityCfg) -> torch.Tensor:
        body_id = cfg.body_ids[0]
        body_name = robot.body_names[body_id]
        body_pos_w = robot.data.body_pos_w[:, body_id]
        body_quat_w = robot.data.body_quat_w[:, body_id]
        local_point = torch.tensor(
            FINGERTIP_INNER_CONTACT_LOCAL_OFFSETS[body_name],
            device=body_pos_w.device,
            dtype=body_pos_w.dtype,
        ).view(1, 3)
        return body_pos_w + quat_apply(body_quat_w, local_point.expand_as(body_pos_w))

    return point_w(narrow_cfg), point_w(wide_cfg)


def _fingertip_cylinder_geometry(
    env: ManagerBasedEnv,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    cylinder_radius: float,
    cylinder_half_height: float,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed cylinder gaps and both inner finger points in object coordinates."""

    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    narrow_o, _ = subtract_frame_transforms(obj.data.root_pos_w, obj.data.root_quat_w, narrow_w)
    wide_o, _ = subtract_frame_transforms(obj.data.root_pos_w, obj.data.root_quat_w, wide_w)

    def finite_cylinder_sdf(point_o: torch.Tensor) -> torch.Tensor:
        radial_gap = torch.linalg.vector_norm(point_o[:, :2], dim=1) - cylinder_radius
        height_gap = torch.abs(point_o[:, 2]) - cylinder_half_height
        distance_components = torch.stack((radial_gap, height_gap), dim=1)
        outside = torch.linalg.vector_norm(torch.clamp(distance_components, min=0.0), dim=1)
        inside = torch.minimum(torch.maximum(radial_gap, height_gap), torch.zeros_like(radial_gap))
        return outside + inside

    return finite_cylinder_sdf(narrow_o), finite_cylinder_sdf(wide_o), narrow_o, wide_o


def fingertip_surface_clearance_reward(
    env: ManagerBasedRLEnv,
    desired_gap: float,
    gap_std: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the mean inner-finger collision-surface gap to a finite cylinder."""

    narrow_gap, wide_gap, _, _ = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    pair_gap = 0.5 * (narrow_gap + wide_gap)
    return torch.exp(-torch.square((pair_gap - desired_gap) / gap_std))


def fingertip_grasp_center_reward(
    env: ManagerBasedRLEnv,
    near_scale: float,
    horizontal_std: float,
    vertical_std: float,
    level_std: float,
    between_scale: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a cylinder centered horizontally and vertically between open fingers."""

    narrow_gap, wide_gap, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    delta_xy = wide_o[:, :2] - narrow_o[:, :2]
    span_sq = torch.sum(torch.square(delta_xy), dim=1).clamp_min(1.0e-8)
    t = torch.sum(-narrow_o[:, :2] * delta_xy, dim=1) / span_sq
    between = torch.sigmoid(t / between_scale) * torch.sigmoid((1.0 - t) / between_scale)
    closest_xy = narrow_o[:, :2] + t.clamp(0.0, 1.0).unsqueeze(1) * delta_xy
    horizontal = torch.exp(-torch.square(torch.linalg.vector_norm(closest_xy, dim=1) / horizontal_std))

    mid_height = 0.5 * (narrow_o[:, 2] + wide_o[:, 2])
    height_delta = torch.abs(narrow_o[:, 2] - wide_o[:, 2])
    vertical = torch.exp(-torch.square(mid_height / vertical_std))
    level = torch.exp(-torch.square(height_delta / level_std))
    pair_gap = 0.5 * (narrow_gap + wide_gap)
    near_gate = torch.exp(-torch.square(torch.clamp(pair_gap, min=0.0) / near_scale))
    return near_gate * between * horizontal * vertical * level


def fingertip_surface_orientation_reward(
    env: ManagerBasedRLEnv,
    grasp_axis_sign: float,
    near_scale: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    tcp_orientation_cfg: SceneEntityCfg,
    orientation_offset: tuple[float, float, float, float],
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Align bottle Z and the side-corrected semantic grasp axis near the cylinder."""

    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_gap, wide_gap, _, _ = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    finger_midpoint_w = 0.5 * (narrow_w + wide_w)

    link_quat_w = robot.data.body_quat_w[:, tcp_orientation_cfg.body_ids[0]]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))
    basis_y = torch.zeros_like(finger_midpoint_w)
    basis_y[:, 1] = grasp_axis_sign
    basis_z = torch.zeros_like(finger_midpoint_w)
    basis_z[:, 2] = 1.0
    grasp_axis_w = quat_apply(tcp_quat_w, basis_y)
    tcp_z_w = quat_apply(tcp_quat_w, basis_z)
    bottle_z_w = quat_apply(obj.data.root_quat_w, basis_z)

    to_object_w = obj.data.root_pos_w - finger_midpoint_w
    planar_to_object_w = to_object_w - torch.sum(to_object_w * bottle_z_w, dim=1, keepdim=True) * bottle_z_w
    planar_norm = torch.linalg.vector_norm(planar_to_object_w, dim=1, keepdim=True)
    fallback = torch.zeros_like(planar_to_object_w)
    fallback[:, 0] = 1.0
    target_axis_w = torch.where(
        planar_norm > 1.0e-4,
        planar_to_object_w / planar_norm.clamp_min(1.0e-6),
        fallback,
    )

    z_score = 0.5 * (torch.sum(tcp_z_w * bottle_z_w, dim=1).clamp(-1.0, 1.0) + 1.0)
    approach_score = 0.5 * (torch.sum(grasp_axis_w * target_axis_w, dim=1).clamp(-1.0, 1.0) + 1.0)
    pair_gap = 0.5 * (narrow_gap + wide_gap)
    near_gate = torch.exp(-torch.square(torch.clamp(pair_gap, min=0.0) / near_scale))
    return near_gate * 0.5 * (z_score + approach_score)


def early_fingertip_contact_penalty(
    env: ManagerBasedRLEnv,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    """Penalize any real finger contact during the open pre-grasp stage."""

    narrow_force = filtered_contact_force_magnitude(env, narrow_sensor_name)
    wide_force = filtered_contact_force_magnitude(env, wide_sensor_name)
    return ((narrow_force > force_threshold) | (wide_force > force_threshold)).to(narrow_force.dtype)


def fingertip_pregrasp_diagnostics(
    env: ManagerBasedEnv,
    grasp_axis_sign: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    tcp_orientation_cfg: SceneEntityCfg,
    orientation_offset: tuple[float, float, float, float],
    narrow_sensor_name: str,
    wide_sensor_name: str,
    object_cfg=SceneEntityCfg("object"),
) -> dict[str, torch.Tensor]:
    """Return raw geometry, orientation, and contact metrics for checkpoint evaluation."""

    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_gap, wide_gap, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    delta_xy = wide_o[:, :2] - narrow_o[:, :2]
    span_sq = torch.sum(torch.square(delta_xy), dim=1).clamp_min(1.0e-8)
    t = torch.sum(-narrow_o[:, :2] * delta_xy, dim=1) / span_sq
    closest_xy = narrow_o[:, :2] + t.clamp(0.0, 1.0).unsqueeze(1) * delta_xy

    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    midpoint_w = 0.5 * (narrow_w + wide_w)
    link_quat_w = robot.data.body_quat_w[:, tcp_orientation_cfg.body_ids[0]]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))
    basis_y = torch.zeros_like(midpoint_w)
    basis_y[:, 1] = grasp_axis_sign
    basis_z = torch.zeros_like(midpoint_w)
    basis_z[:, 2] = 1.0
    grasp_axis_w = quat_apply(tcp_quat_w, basis_y)
    tcp_z_w = quat_apply(tcp_quat_w, basis_z)
    bottle_z_w = quat_apply(obj.data.root_quat_w, basis_z)
    to_object_w = obj.data.root_pos_w - midpoint_w
    planar_to_object_w = to_object_w - torch.sum(to_object_w * bottle_z_w, dim=1, keepdim=True) * bottle_z_w
    target_axis_w = planar_to_object_w / torch.linalg.vector_norm(planar_to_object_w, dim=1, keepdim=True).clamp_min(1.0e-6)

    narrow_force = filtered_contact_force_magnitude(env, narrow_sensor_name)
    wide_force = filtered_contact_force_magnitude(env, wide_sensor_name)
    return {
        "inner_midpoint_distance": torch.linalg.vector_norm(midpoint_w - obj.data.root_pos_w, dim=1),
        "narrow_gap": narrow_gap,
        "wide_gap": wide_gap,
        "pair_gap": 0.5 * (narrow_gap + wide_gap),
        "projection_t": t,
        "centerline_error": torch.linalg.vector_norm(closest_xy, dim=1),
        "mid_height": 0.5 * (narrow_o[:, 2] + wide_o[:, 2]),
        "height_difference": torch.abs(narrow_o[:, 2] - wide_o[:, 2]),
        "z_dot": torch.sum(tcp_z_w * bottle_z_w, dim=1).clamp(-1.0, 1.0),
        "approach_dot": torch.sum(grasp_axis_w * target_axis_w, dim=1).clamp(-1.0, 1.0),
        "narrow_force": narrow_force,
        "wide_force": wide_force,
    }


def fingertip_pregrasp_observation(
    env: ManagerBasedEnv,
    grasp_axis_sign: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    tcp_orientation_cfg: SceneEntityCfg,
    orientation_offset: tuple[float, float, float, float],
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return 15 directional surface-pregrasp features for one hand."""

    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_gap, wide_gap, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    delta_xy = wide_o[:, :2] - narrow_o[:, :2]
    span_sq = torch.sum(torch.square(delta_xy), dim=1).clamp_min(1.0e-8)
    t = torch.sum(-narrow_o[:, :2] * delta_xy, dim=1) / span_sq
    closest_xy = narrow_o[:, :2] + t.clamp(0.0, 1.0).unsqueeze(1) * delta_xy
    mid_height = 0.5 * (narrow_o[:, 2] + wide_o[:, 2])
    signed_height_difference = narrow_o[:, 2] - wide_o[:, 2]

    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    midpoint_w = 0.5 * (narrow_w + wide_w)
    link_quat_w = robot.data.body_quat_w[:, tcp_orientation_cfg.body_ids[0]]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))
    basis_y = torch.zeros_like(midpoint_w)
    basis_y[:, 1] = grasp_axis_sign
    basis_z = torch.zeros_like(midpoint_w)
    basis_z[:, 2] = 1.0
    grasp_axis_w = quat_apply(tcp_quat_w, basis_y)
    tcp_z_w = quat_apply(tcp_quat_w, basis_z)
    bottle_z_w = quat_apply(obj.data.root_quat_w, basis_z)
    to_object_w = obj.data.root_pos_w - midpoint_w
    planar_to_object_w = to_object_w - torch.sum(to_object_w * bottle_z_w, dim=1, keepdim=True) * bottle_z_w
    target_axis_w = planar_to_object_w / torch.linalg.vector_norm(planar_to_object_w, dim=1, keepdim=True).clamp_min(1.0e-6)
    z_dot = torch.sum(tcp_z_w * bottle_z_w, dim=1).clamp(-1.0, 1.0)
    z_cross = torch.cross(tcp_z_w, bottle_z_w, dim=1)
    approach_dot = torch.sum(grasp_axis_w * target_axis_w, dim=1).clamp(-1.0, 1.0)
    approach_cross = torch.cross(grasp_axis_w, target_axis_w, dim=1)

    return torch.cat(
        (
            (narrow_gap / 0.10).unsqueeze(1),
            (wide_gap / 0.10).unsqueeze(1),
            (t - 0.5).unsqueeze(1),
            closest_xy / 0.10,
            (mid_height / 0.10).unsqueeze(1),
            (signed_height_difference / 0.20).unsqueeze(1),
            z_dot.unsqueeze(1),
            z_cross,
            approach_dot.unsqueeze(1),
            approach_cross,
        ),
        dim=1,
    )


def _curriculum_gap(env: ManagerBasedEnv, gaps: tuple[float, float, float], stage_steps: tuple[int, int]) -> float:
    step = int(getattr(env, "common_step_counter", 0))
    if step < stage_steps[0]:
        return gaps[0]
    if step < stage_steps[1]:
        return gaps[1]
    return gaps[2]


def fingertip_inner_midpoint_object_distance(
    env: ManagerBasedEnv,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Measure object-center distance from the calibrated inner fingertip midpoint."""

    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    return torch.linalg.vector_norm(0.5 * (narrow_w + wide_w) - obj.data.root_pos_w, dim=1)


def fingertip_inner_midpoint_distance_tracking(
    env: ManagerBasedRLEnv,
    std: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    distance = fingertip_inner_midpoint_object_distance(env, narrow_cfg, wide_cfg, object_cfg)
    return torch.exp(-torch.square(distance / std))


def fingertip_inner_midpoint_progress_reward(
    env: ManagerBasedRLEnv,
    reward_key: str,
    max_progress: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    distance = fingertip_inner_midpoint_object_distance(env, narrow_cfg, wide_cfg, object_cfg)
    previous = _reward_state_tensor(env, reward_key, tuple(distance.shape), distance.dtype)
    progress = torch.clamp(previous - distance, min=-max_progress, max=max_progress)
    progress = torch.where(env.episode_length_buf == 0, torch.zeros_like(progress), progress)
    previous.copy_(distance)
    return progress


def _fingertip_distance_gate(
    env: ManagerBasedEnv,
    scale: float | None,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor | float:
    if scale is None:
        return 1.0
    distance = fingertip_inner_midpoint_object_distance(env, narrow_cfg, wide_cfg, object_cfg)
    return torch.exp(-torch.square(distance / scale))


def fingertip_individual_clearance_reward(
    env: ManagerBasedRLEnv,
    desired_gaps: tuple[float, float, float],
    stage_steps: tuple[int, int],
    gap_std: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    narrow_gap, wide_gap, _, _ = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    desired = _curriculum_gap(env, desired_gaps, stage_steps)
    narrow_reward = torch.exp(-torch.square((narrow_gap - desired) / gap_std))
    wide_reward = torch.exp(-torch.square((wide_gap - desired) / gap_std))
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * 0.5 * (narrow_reward + wide_reward)


def fingertip_gap_balance_penalty(
    env: ManagerBasedRLEnv,
    scale: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    narrow_gap, wide_gap, _, _ = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * torch.abs(narrow_gap - wide_gap) / scale


def fingertip_too_close_penalty(
    env: ManagerBasedRLEnv,
    minimum_gap: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    narrow_gap, wide_gap, _, _ = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    violation = torch.relu((minimum_gap - torch.minimum(narrow_gap, wide_gap)) / minimum_gap)
    return torch.square(violation)


def fingertip_between_reward(
    env: ManagerBasedRLEnv,
    between_scale: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    _, _, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    delta = wide_o[:, :2] - narrow_o[:, :2]
    t = torch.sum(-narrow_o[:, :2] * delta, dim=1) / torch.sum(torch.square(delta), dim=1).clamp_min(1.0e-8)
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * torch.sigmoid(t / between_scale) * torch.sigmoid((1.0 - t) / between_scale)


def fingertip_horizontal_center_reward(
    env: ManagerBasedRLEnv,
    std: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    _, _, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    delta = wide_o[:, :2] - narrow_o[:, :2]
    t = torch.sum(-narrow_o[:, :2] * delta, dim=1) / torch.sum(torch.square(delta), dim=1).clamp_min(1.0e-8)
    closest = narrow_o[:, :2] + t.clamp(0.0, 1.0).unsqueeze(1) * delta
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * torch.exp(-torch.square(torch.linalg.vector_norm(closest, dim=1) / std))


def fingertip_vertical_center_reward(
    env: ManagerBasedRLEnv,
    std: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    _, _, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    mid_height = 0.5 * (narrow_o[:, 2] + wide_o[:, 2])
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * torch.exp(-torch.square(mid_height / std))


def fingertip_level_reward(
    env: ManagerBasedRLEnv,
    std: float,
    cylinder_radius: float,
    cylinder_half_height: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    _, _, narrow_o, wide_o = _fingertip_cylinder_geometry(
        env, narrow_cfg, wide_cfg, cylinder_radius, cylinder_half_height, object_cfg
    )
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * torch.exp(-torch.square(torch.abs(narrow_o[:, 2] - wide_o[:, 2]) / std))


def fingertip_signed_axis_reward(
    env: ManagerBasedRLEnv,
    axis: str,
    grasp_axis_sign: float,
    narrow_cfg: SceneEntityCfg,
    wide_cfg: SceneEntityCfg,
    tcp_orientation_cfg: SceneEntityCfg,
    orientation_offset: tuple[float, float, float, float],
    object_cfg=SceneEntityCfg("object"),
    distance_gate_scale: float | None = None,
) -> torch.Tensor:
    robot: Articulation = env.scene[narrow_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    narrow_w, wide_w = _inner_fingertip_points_w(robot, narrow_cfg, wide_cfg)
    midpoint_w = 0.5 * (narrow_w + wide_w)
    link_quat_w = robot.data.body_quat_w[:, tcp_orientation_cfg.body_ids[0]]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))
    basis = torch.zeros_like(midpoint_w)
    if axis == "z":
        basis[:, 2] = 1.0
        score = torch.sum(quat_apply(tcp_quat_w, basis) * quat_apply(obj.data.root_quat_w, basis), dim=1)
        gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
        return gate * score
    basis[:, 1] = grasp_axis_sign
    grasp_axis_w = quat_apply(tcp_quat_w, basis)
    bottle_z_basis = torch.zeros_like(midpoint_w)
    bottle_z_basis[:, 2] = 1.0
    bottle_z_w = quat_apply(obj.data.root_quat_w, bottle_z_basis)
    to_object = obj.data.root_pos_w - midpoint_w
    planar = to_object - torch.sum(to_object * bottle_z_w, dim=1, keepdim=True) * bottle_z_w
    target = planar / torch.linalg.vector_norm(planar, dim=1, keepdim=True).clamp_min(1.0e-6)
    score = torch.sum(grasp_axis_w * target, dim=1).clamp(-1.0, 1.0)
    gate = _fingertip_distance_gate(env, distance_gate_scale, narrow_cfg, wide_cfg, object_cfg)
    return gate * score


def early_fingertip_contact_termination(
    env: ManagerBasedEnv, narrow_sensor_name: str, wide_sensor_name: str, force_threshold: float
) -> torch.Tensor:
    return (filtered_contact_force_magnitude(env, narrow_sensor_name) > force_threshold) | (
        filtered_contact_force_magnitude(env, wide_sensor_name) > force_threshold
    )


def tcp_axes_align_with_object_nearby(
    env: ManagerBasedRLEnv,
    distance_scale: float,
    tcp_position_cfg: SceneEntityCfg,
    tcp_orientation_cfg: SceneEntityCfg,
    orientation_offset: tuple[float, float, float, float],
    nominal_inward_direction: tuple[float, float, float],
    fallback_distance: float = 0.03,
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Align semantic TCP Z with bottle Z and TCP Y toward the bottle when nearby."""

    robot: Articulation = env.scene[tcp_position_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_position_cfg)
    link_quat_w = robot.data.body_quat_w[:, tcp_orientation_cfg.body_ids[0]]
    offset = torch.tensor(orientation_offset, device=link_quat_w.device, dtype=link_quat_w.dtype).view(1, 4)
    tcp_quat_w = quat_mul(link_quat_w, offset.expand_as(link_quat_w))

    basis_y = torch.zeros_like(tcp_pos_w)
    basis_y[:, 1] = 1.0
    basis_z = torch.zeros_like(tcp_pos_w)
    basis_z[:, 2] = 1.0
    tcp_y_w = quat_apply(tcp_quat_w, basis_y)
    tcp_z_w = quat_apply(tcp_quat_w, basis_z)
    bottle_z_w = quat_apply(obj.data.root_quat_w, basis_z)

    to_object_w = obj.data.root_pos_w - tcp_pos_w
    planar_to_object_w = to_object_w - torch.sum(to_object_w * bottle_z_w, dim=1, keepdim=True) * bottle_z_w
    planar_distance = torch.linalg.vector_norm(planar_to_object_w, dim=1, keepdim=True)
    nominal_inward = torch.tensor(
        nominal_inward_direction, device=tcp_pos_w.device, dtype=tcp_pos_w.dtype
    ).view(1, 3)
    nominal_inward = nominal_inward.expand_as(tcp_pos_w)
    target_y_w = torch.where(
        planar_distance > fallback_distance,
        planar_to_object_w / torch.clamp(planar_distance, min=1.0e-6),
        nominal_inward,
    )

    z_score = torch.clamp(torch.sum(tcp_z_w * bottle_z_w, dim=1), min=0.0, max=1.0)
    y_score = torch.clamp(torch.sum(tcp_y_w * target_y_w, dim=1), min=0.0, max=1.0)
    distance = torch.linalg.vector_norm(to_object_w, dim=1)
    near_gate = torch.exp(-torch.square(distance / distance_scale))
    return near_gate * 0.5 * (z_score + y_score)


def gripper_early_close_penalty(
    env: ManagerBasedRLEnv,
    far_threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    narrow_joint_name: str,
    wide_joint_name: str,
    robot_cfg=SceneEntityCfg("robot"),
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize closing while the TCP is still too far away to grasp safely."""

    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    closed = gripper_closed_fraction(
        env, narrow_joint_name=narrow_joint_name, wide_joint_name=wide_joint_name, robot_cfg=robot_cfg
    )
    return (distance > far_threshold).to(closed.dtype) * closed


def gripper_close_when_pregrasp_ready(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    speed_threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    narrow_joint_name: str,
    wide_joint_name: str,
    robot_cfg=SceneEntityCfg("robot"),
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward closure only near a stationary grasp target."""

    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    relative_speed = tcp_relative_to_object_speed(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    closed = gripper_closed_fraction(
        env, narrow_joint_name=narrow_joint_name, wide_joint_name=wide_joint_name, robot_cfg=robot_cfg
    )
    ready = (distance <= distance_threshold) & (relative_speed <= speed_threshold)
    return ready.to(closed.dtype) * closed


def dual_finger_contact_reward(
    env: ManagerBasedRLEnv,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
    narrow_joint_name: str,
    wide_joint_name: str,
    robot_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward simultaneous contacts from both movable fingertips, gated by closure."""

    narrow_force = filtered_contact_force_magnitude(env, narrow_sensor_name)
    wide_force = filtered_contact_force_magnitude(env, wide_sensor_name)
    bilateral_contact = torch.minimum(
        torch.clamp(narrow_force / force_threshold, max=1.0),
        torch.clamp(wide_force / force_threshold, max=1.0),
    )
    closed = gripper_closed_fraction(
        env, narrow_joint_name=narrow_joint_name, wide_joint_name=wide_joint_name, robot_cfg=robot_cfg
    )
    return bilateral_contact * closed


def dual_finger_contact_dwell_reward(
    env: ManagerBasedRLEnv,
    reward_key: str,
    narrow_sensor_name: str,
    wide_sensor_name: str,
    force_threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    narrow_joint_name: str,
    wide_joint_name: str,
    speed_threshold: float,
    hold_steps: int,
    robot_cfg=SceneEntityCfg("robot"),
    object_cfg=SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward sustained, slow, bilateral fingertip contact around a closed gripper."""

    narrow_force = filtered_contact_force_magnitude(env, narrow_sensor_name)
    wide_force = filtered_contact_force_magnitude(env, wide_sensor_name)
    closed = gripper_closed_fraction(
        env, narrow_joint_name=narrow_joint_name, wide_joint_name=wide_joint_name, robot_cfg=robot_cfg
    )
    relative_speed = tcp_relative_to_object_speed(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    stable = (
        (narrow_force >= force_threshold)
        & (wide_force >= force_threshold)
        & (closed >= 0.65)
        & (relative_speed <= speed_threshold)
    )
    streak = _reward_state_tensor(env, reward_key, tuple(closed.shape), torch.int32)
    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(env.episode_length_buf == 0, torch.zeros_like(streak), streak)
    env._jz_reward_state[reward_key] = streak
    return torch.clamp(streak.to(closed.dtype) / float(hold_steps), max=1.0)


def tcp_relative_speed_l2_when_close_to_object(env: ManagerBasedRLEnv, threshold: float, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    is_close = (distance <= threshold).to(dtype=torch.float32)
    rel_speed_sq = torch.square(tcp_relative_to_object_speed(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg))
    return is_close * rel_speed_sq


def tcp_speed_l2_when_close_to_object_side_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    object_cfg=SceneEntityCfg("object"),
):
    distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=tcp_asset_cfg, target_offset=target_offset, object_cfg=object_cfg
    )
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    is_close = (distance <= threshold).to(dtype=torch.float32)
    tcp_speed_sq = torch.sum(torch.square(_fingertip_midpoint_lin_vel_w(robot, tcp_asset_cfg)), dim=1)
    return is_close * tcp_speed_sq

def joint_vel_l2_when_close_to_object(env: ManagerBasedRLEnv, threshold: float, joint_asset_cfg: SceneEntityCfg, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[joint_asset_cfg.name]
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    is_close = (distance <= threshold).to(robot.data.joint_vel.dtype)
    joint_vel_sq = torch.sum(torch.square(robot.data.joint_vel[:, joint_asset_cfg.joint_ids]), dim=1)
    return is_close * joint_vel_sq


def joint_vel_l2_when_close_to_object_side_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    joint_asset_cfg: SceneEntityCfg,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    object_cfg=SceneEntityCfg("object"),
):
    robot: Articulation = env.scene[joint_asset_cfg.name]
    distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=tcp_asset_cfg, target_offset=target_offset, object_cfg=object_cfg
    )
    is_close = (distance <= threshold).to(robot.data.joint_vel.dtype)
    joint_vel_sq = torch.sum(torch.square(robot.data.joint_vel[:, joint_asset_cfg.joint_ids]), dim=1)
    return is_close * joint_vel_sq


def weighted_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, weights: list[float]):
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    weight_tensor = torch.tensor(weights, device=env.device, dtype=joint_vel.dtype).unsqueeze(0)
    return torch.sum(weight_tensor * torch.square(joint_vel), dim=1)

def action_rate_l2_when_close_to_object(env: ManagerBasedRLEnv, threshold: float, tcp_asset_cfg: SceneEntityCfg, action_name: str, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    action_term = env.action_manager.get_term(action_name)
    current_action = action_term.raw_actions
    prev_action = _reward_state_tensor(env, f"{action_name}_prev_raw_action", tuple(current_action.shape), current_action.dtype)
    is_close = (distance <= threshold).to(current_action.dtype)
    is_reset = env.episode_length_buf == 0
    action_rate_sq = torch.sum(torch.square(current_action - prev_action), dim=1)
    action_rate_sq = torch.where(is_reset, torch.zeros_like(action_rate_sq), action_rate_sq)
    prev_action.copy_(current_action)
    return is_close * action_rate_sq


def action_rate_l2_when_close_to_object_side_target(
    env: ManagerBasedRLEnv,
    threshold: float,
    tcp_asset_cfg: SceneEntityCfg,
    target_offset: tuple[float, float, float],
    action_name: str,
    object_cfg=SceneEntityCfg("object"),
):
    distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=tcp_asset_cfg, target_offset=target_offset, object_cfg=object_cfg
    )
    action_term = env.action_manager.get_term(action_name)
    current_action = action_term.raw_actions
    prev_action = _reward_state_tensor(
        env, f"{action_name}_side_target_prev_raw_action", tuple(current_action.shape), current_action.dtype
    )
    is_close = (distance <= threshold).to(current_action.dtype)
    is_reset = env.episode_length_buf == 0
    action_rate_sq = torch.sum(torch.square(current_action - prev_action), dim=1)
    action_rate_sq = torch.where(is_reset, torch.zeros_like(action_rate_sq), action_rate_sq)
    prev_action.copy_(current_action)
    return is_close * action_rate_sq

def bimanual_tcp_close_to_object_bonus(env: ManagerBasedRLEnv, threshold: float, left_tcp_asset_cfg: SceneEntityCfg, right_tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    left_distance = tcp_to_object_distance(env, tcp_asset_cfg=left_tcp_asset_cfg, object_cfg=object_cfg)
    right_distance = tcp_to_object_distance(env, tcp_asset_cfg=right_tcp_asset_cfg, object_cfg=object_cfg)
    return ((left_distance <= threshold) & (right_distance <= threshold)).to(left_distance.dtype)

def bimanual_tcp_stable_near_object_dwell_reward(env: ManagerBasedRLEnv, threshold: float, speed_threshold: float, hold_steps: int, left_tcp_asset_cfg: SceneEntityCfg, right_tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    left_distance = tcp_to_object_distance(env, tcp_asset_cfg=left_tcp_asset_cfg, object_cfg=object_cfg)
    right_distance = tcp_to_object_distance(env, tcp_asset_cfg=right_tcp_asset_cfg, object_cfg=object_cfg)
    left_rel_speed = tcp_relative_to_object_speed(env, tcp_asset_cfg=left_tcp_asset_cfg, object_cfg=object_cfg)
    right_rel_speed = tcp_relative_to_object_speed(env, tcp_asset_cfg=right_tcp_asset_cfg, object_cfg=object_cfg)
    streak = _reward_state_tensor(env, "bimanual_stable_near_object_streak", tuple(left_distance.shape), torch.int32)
    is_reset = env.episode_length_buf == 0
    stable = (left_distance <= threshold) & (right_distance <= threshold) & (left_rel_speed <= speed_threshold) & (right_rel_speed <= speed_threshold)
    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(is_reset, torch.zeros_like(streak), streak)
    env._jz_reward_state["bimanual_stable_near_object_streak"] = streak
    return torch.clamp(streak.to(left_distance.dtype) / float(hold_steps), max=1.0)


def bimanual_tcp_stable_near_object_side_target_dwell_reward(
    env: ManagerBasedRLEnv,
    threshold: float,
    speed_threshold: float,
    hold_steps: int,
    left_tcp_asset_cfg: SceneEntityCfg,
    right_tcp_asset_cfg: SceneEntityCfg,
    left_target_offset: tuple[float, float, float],
    right_target_offset: tuple[float, float, float],
    left_object_cfg=SceneEntityCfg("object"),
    right_object_cfg=SceneEntityCfg("object"),
):
    left_distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=left_tcp_asset_cfg, target_offset=left_target_offset, object_cfg=left_object_cfg
    )
    right_distance = tcp_to_object_side_target_distance(
        env, tcp_asset_cfg=right_tcp_asset_cfg, target_offset=right_target_offset, object_cfg=right_object_cfg
    )
    robot: Articulation = env.scene[left_tcp_asset_cfg.name]
    left_speed = torch.norm(_fingertip_midpoint_lin_vel_w(robot, left_tcp_asset_cfg), dim=1)
    right_speed = torch.norm(_fingertip_midpoint_lin_vel_w(robot, right_tcp_asset_cfg), dim=1)
    streak = _reward_state_tensor(env, "bimanual_stable_near_side_target_streak", tuple(left_distance.shape), torch.int32)
    is_reset = env.episode_length_buf == 0
    stable = (
        (left_distance <= threshold)
        & (right_distance <= threshold)
        & (left_speed <= speed_threshold)
        & (right_speed <= speed_threshold)
    )
    streak = torch.where(stable, streak + 1, torch.zeros_like(streak))
    streak = torch.where(is_reset, torch.zeros_like(streak), streak)
    env._jz_reward_state["bimanual_stable_near_side_target_streak"] = streak
    return torch.clamp(streak.to(left_distance.dtype) / float(hold_steps), max=1.0)

def object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    object_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(object_height > minimal_height, 1.0, 0.0)

def action_max_abs(env: ManagerBasedRLEnv):
    return torch.max(torch.abs(env.action_manager.action), dim=1).values


def joint_deviation_from_default_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)


def joint_limit_margin_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, margin: float = 0.15):
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    lower_margin = torch.clamp(joint_limits[..., 0] + margin - joint_pos, min=0.0)
    upper_margin = torch.clamp(joint_pos - (joint_limits[..., 1] - margin), min=0.0)
    return torch.sum(torch.square(lower_margin) + torch.square(upper_margin), dim=1)


def grasp_success_bonus(env: ManagerBasedRLEnv, object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = obj.data.root_pos_w[:, 2] > OBJECT_LIFT_SUCCESS_Z
    return lifted.to(torch.float32) * 15.0


def object_lin_vel_penalty(env: ManagerBasedRLEnv, gripper_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[gripper_asset_cfg.name]
    lin_vel_sq = torch.sum(torch.square(obj.data.root_lin_vel_w), dim=1)
    effort = torch.mean(torch.abs(robot.data.applied_torque[:, gripper_asset_cfg.joint_ids]), dim=1)
    not_grasper = (effort < 0.5).to(lin_vel_sq.dtype)
    return not_grasper * lin_vel_sq


def object_ang_vel_penalty(env: ManagerBasedRLEnv):
    obj: RigidObject = env.scene["object"]
    return torch.sum(torch.square(obj.data.root_ang_vel_w), dim=1)


def table_penetration_penalty(env: ManagerBasedRLEnv):
    obj: RigidObject = env.scene["object"]
    table_z = TABLE_TOP_Z
    penetration = torch.clamp(table_z + 0.01 - obj.data.root_pos_w[:, 2], min=0.0)
    return penetration * 50.0


def tcp_table_clearance_penalty(
    env: ManagerBasedRLEnv,
    tcp_asset_cfg: SceneEntityCfg,
    minimum_height: float,
):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    env_origins = getattr(env.scene, "env_origins", None)
    origin_z = env_origins[:, 2] if env_origins is not None else torch.zeros_like(tcp_pos_w[:, 2])
    minimum_height_w = origin_z + float(minimum_height)
    return torch.clamp(minimum_height_w - tcp_pos_w[:, 2], min=0.0)


def tcp_approach_orientation_reward(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    to_obj = obj.data.root_pos_w - tcp_pos_w
    dist = torch.norm(to_obj, dim=1).clamp(min=1e-6)
    approach_z = -to_obj[:, 2] / dist
    active = (dist <= 0.20).to(approach_z.dtype)
    return active * approach_z.clamp(min=0.0)


def arm_asymmetry_penalty(env: ManagerBasedRLEnv, left_tcp_cfg: SceneEntityCfg, right_tcp_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene["robot"]
    right_vel = _fingertip_midpoint_lin_vel_w(robot, right_tcp_cfg)
    left_dist = tcp_to_object_distance(env, tcp_asset_cfg=left_tcp_cfg, object_cfg=object_cfg)
    right_dist = tcp_to_object_distance(env, tcp_asset_cfg=right_tcp_cfg, object_cfg=object_cfg)
    left_near = (left_dist < 0.10).to(right_vel.dtype)
    right_far = (right_dist > 0.15).to(right_vel.dtype)
    right_speed = torch.norm(right_vel, dim=1)
    return left_near * right_far * right_speed
