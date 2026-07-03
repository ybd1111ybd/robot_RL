"""Observation and reward helpers for the JZ bimanual grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _base_link_pose_w(asset: Articulation):
    base_idx = asset.body_names.index("base_link")
    return asset.data.body_pos_w[:, base_idx], asset.data.body_quat_w[:, base_idx]

def _fingertip_midpoint_pos_w(asset: Articulation, asset_cfg: SceneEntityCfg):
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)

def _fingertip_midpoint_lin_vel_w(asset: Articulation, asset_cfg: SceneEntityCfg):
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :].mean(dim=1)

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

def tcp_to_object_distance(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[tcp_asset_cfg.name]
    tcp_pos_w = _fingertip_midpoint_pos_w(robot, tcp_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    obj_pos_w = obj.data.root_pos_w
    return torch.norm(tcp_pos_w - obj_pos_w, dim=1)

def tcp_to_object_distance_tanh(env: ManagerBasedRLEnv, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object"), std=0.10):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    return 1.0 - torch.tanh(distance / std)

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

def tcp_relative_speed_l2_when_close_to_object(env: ManagerBasedRLEnv, threshold: float, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    is_close = (distance <= threshold).to(dtype=torch.float32)
    rel_speed_sq = torch.square(tcp_relative_to_object_speed(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg))
    return is_close * rel_speed_sq

def joint_vel_l2_when_close_to_object(env: ManagerBasedRLEnv, threshold: float, joint_asset_cfg: SceneEntityCfg, tcp_asset_cfg: SceneEntityCfg, object_cfg=SceneEntityCfg("object")):
    robot: Articulation = env.scene[joint_asset_cfg.name]
    distance = tcp_to_object_distance(env, tcp_asset_cfg=tcp_asset_cfg, object_cfg=object_cfg)
    is_close = (distance <= threshold).to(robot.data.joint_vel.dtype)
    joint_vel_sq = torch.sum(torch.square(robot.data.joint_vel[:, joint_asset_cfg.joint_ids]), dim=1)
    return is_close * joint_vel_sq

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

def object_is_lifted(env: ManagerBasedRLEnv, minimal_height: float, object_cfg=SceneEntityCfg("object")):
    obj: RigidObject = env.scene[object_cfg.name]
    object_height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.where(object_height > minimal_height, 1.0, 0.0)

def action_max_abs(env: ManagerBasedRLEnv):
    return torch.max(torch.abs(env.action_manager.action), dim=1).values


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
