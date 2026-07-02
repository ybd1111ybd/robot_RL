"""Generate reachable base-frame target datasets for JZ dual-arm reach training.

This generator defines the reachable region in the robot base frame (`base_link`)
using forward kinematics only:

1. sample arm joints inside joint limits with a configurable safety margin
2. compute dual-fingertip midpoint TCP positions
3. keep only points inside the desired base-frame workspace box
4. deduplicate with a voxel grid so targets are spatially spread out

The resulting dataset is position-focused and is meant to keep RL targets inside
the actually reachable workspace of the local robot model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STANDALONE_IK_DIR = (
    PROJECT_ROOT.parent
    / "robot_simulator"
    / "isaac_sim_implementations"
    / "standalone_ik_solver"
)

if str(STANDALONE_IK_DIR) not in sys.path:
    sys.path.insert(0, str(STANDALONE_IK_DIR))


from standalone_ik_solver.config import SolverConfig
from standalone_ik_solver.kinematics import ArmTCPModel, GRIPPER_MOUNT_QUAT, KinematicsChain


WORKSPACE_DATA_PATH = (
    PROJECT_ROOT
    / "source"
    / "jzlab"
    / "jzlab"
    / "tasks"
    / "manager_based"
    / "jz_manipulation"
    / "bimanual"
    / "reach"
    / "workspace"
    / "reachable_workspace.json"
)
LEFT_GRIPPER_OPEN = {"left_gripper_narrow_joint": -1.0, "left_gripper_wide_joint": 1.0}
RIGHT_GRIPPER_OPEN = {"right_gripper_narrow_joint": -1.0, "right_gripper_wide_joint": 1.0}


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _quat_rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion(quaternion)
    vec = np.asarray(vector, dtype=float).reshape(3)
    q_vec = quat[1:4]
    q_w = float(quat[0])
    return (
        2.0 * np.dot(q_vec, vec) * q_vec
        + (q_w * q_w - np.dot(q_vec, q_vec)) * vec
        + 2.0 * q_w * np.cross(q_vec, vec)
    )


def _solver_to_base_frame(
    position_solver: np.ndarray,
    solver_base_pos: np.ndarray,
    solver_base_quat: np.ndarray,
) -> np.ndarray:
    return np.asarray(solver_base_pos, dtype=float) + _quat_rotate_vector(
        np.asarray(solver_base_quat, dtype=float),
        np.asarray(position_solver, dtype=float),
    )


def _joint_margin_ratio(joints: np.ndarray, joint_limits: list[tuple[float, float]]) -> float:
    margins = []
    for value, (low, high) in zip(np.asarray(joints, dtype=float), joint_limits):
        span = float(high - low)
        if span <= 1e-9:
            continue
        lower = float((value - low) / span)
        upper = float((high - value) / span)
        margins.append(min(lower, upper))
    return min(margins) if margins else 0.0


def _build_arm_model(config: SolverConfig, side: str) -> tuple[ArmTCPModel, np.ndarray, np.ndarray]:
    if side == "left":
        arm_joints = config.left_arm_joints
        gripper_joints = config.left_gripper_joints
        solver_base_link = config.left_solver_base_link
        tip_link = "left_arm_link9"
        fingertip_links = ("left_gripper_narrow3_link", "left_gripper_wide3_link")
        aux_joints = LEFT_GRIPPER_OPEN
    else:
        arm_joints = config.right_arm_joints
        gripper_joints = config.right_gripper_joints
        solver_base_link = config.right_solver_base_link
        tip_link = "right_arm_link9"
        fingertip_links = ("right_gripper_narrow3_link", "right_gripper_wide3_link")
        aux_joints = RIGHT_GRIPPER_OPEN

    orientation_chain = KinematicsChain.from_urdf(
        config.urdf_path,
        arm_joints,
        base_link=solver_base_link,
        tip_link=tip_link,
        tcp_offset_m=0.0,
        tcp_orientation_offset_quat=GRIPPER_MOUNT_QUAT,
    )
    fingertip_chains = [
        KinematicsChain.from_urdf(
            config.urdf_path,
            [*arm_joints, gripper_joint],
            base_link=solver_base_link,
            tip_link=fingertip_link,
            tcp_offset_m=0.0,
            tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )
        for gripper_joint, fingertip_link in zip(gripper_joints, fingertip_links)
    ]
    arm_model = ArmTCPModel(orientation_chain, fingertip_chains=fingertip_chains)
    arm_model.set_aux_joint_positions(aux_joints)

    if solver_base_link == "base_link":
        return arm_model, np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    base_chain = KinematicsChain.from_urdf(
        config.urdf_path,
        config.body_joints,
        base_link="base_link",
        tip_link=solver_base_link,
        tcp_offset_m=0.0,
        tcp_orientation_offset_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    base_fk = base_chain.forward_kinematics(np.zeros(base_chain.dof, dtype=float))
    return (
        arm_model,
        np.asarray(base_fk["ee_position"], dtype=float),
        np.asarray(base_fk["ee_quaternion"], dtype=float),
    )


def _sample_joint_positions(
    joint_limits: list[tuple[float, float]],
    joint_margin_threshold: float,
    rng: random.Random,
) -> np.ndarray:
    values = np.zeros(len(joint_limits), dtype=float)
    margin_ratio = min(max(float(joint_margin_threshold), 0.0), 0.49)
    for index, (low, high) in enumerate(joint_limits):
        span = float(high - low)
        if span <= 1e-9:
            values[index] = float(low)
            continue
        sample_low = float(low + margin_ratio * span)
        sample_high = float(high - margin_ratio * span)
        if sample_high <= sample_low:
            values[index] = 0.5 * float(low + high)
        else:
            values[index] = rng.uniform(sample_low, sample_high)
    return values


def _voxel_key(position: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    return tuple(int(math.floor(float(value) / voxel_size)) for value in np.asarray(position, dtype=float))


def _generate_arm_positions(
    *,
    side: str,
    solver_config: SolverConfig,
    count: int,
    max_samples: int,
    voxel_size: float,
    joint_margin_threshold: float,
    x_range: tuple[float, float],
    y_abs_range: tuple[float, float],
    z_range: tuple[float, float],
    rng: random.Random,
) -> list[list[float]]:
    arm_model, solver_base_pos, solver_base_quat = _build_arm_model(solver_config, side)
    accepted: dict[tuple[int, int, int], tuple[float, np.ndarray]] = {}
    expected_y_sign = 1.0 if side == "left" else -1.0

    for sample_index in range(int(max_samples)):
        joint_positions = _sample_joint_positions(arm_model.joint_limits, joint_margin_threshold, rng)
        current_score = _joint_margin_ratio(joint_positions, arm_model.joint_limits)
        if current_score < joint_margin_threshold:
            continue

        fk = arm_model.forward_kinematics(joint_positions)
        solved_base = _solver_to_base_frame(
            np.asarray(fk["ee_position"], dtype=float),
            solver_base_pos,
            solver_base_quat,
        )

        if not (x_range[0] <= float(solved_base[0]) <= x_range[1]):
            continue
        if not (z_range[0] <= float(solved_base[2]) <= z_range[1]):
            continue
        if expected_y_sign * float(solved_base[1]) <= 0.0:
            continue

        solved_y_abs = abs(float(solved_base[1]))
        if not (y_abs_range[0] <= solved_y_abs <= y_abs_range[1]):
            continue

        key = _voxel_key(solved_base, voxel_size)
        previous = accepted.get(key)
        if previous is None or current_score > previous[0]:
            accepted[key] = (current_score, solved_base)

        if (sample_index + 1) % 10000 == 0:
            print(
                f"[{side}] sampled {sample_index + 1}/{int(max_samples)}, accepted {len(accepted)}/{count}",
                flush=True,
            )

        if len(accepted) >= count:
            break

    positions = [item[1].tolist() for item in accepted.values()]
    rng.shuffle(positions)
    return positions


def _split_positions(
    positions: list[list[float]],
    train_count: int,
    eval_count: int,
    rng: random.Random,
) -> tuple[list[list[float]], list[list[float]]]:
    samples = list(positions)
    rng.shuffle(samples)
    if len(samples) < train_count + eval_count:
        raise RuntimeError(
            f"Not enough reachable points collected: got {len(samples)}, need at least {train_count + eval_count}"
        )
    train_positions = samples[:train_count]
    eval_positions = samples[train_count : train_count + eval_count]
    return train_positions, eval_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reachable workspace datasets for JZ dual-arm reach.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--eval-count", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=120000)
    parser.add_argument("--voxel-size", type=float, default=0.04)
    parser.add_argument("--joint-margin-threshold", type=float, default=0.05)
    parser.add_argument("--x-min", type=float, default=0.30)
    parser.add_argument("--x-max", type=float, default=1.00)
    parser.add_argument("--y-abs-min", type=float, default=0.30)
    parser.add_argument("--y-abs-max", type=float, default=0.90)
    parser.add_argument("--z-min", type=float, default=0.85)
    parser.add_argument("--z-max", type=float, default=1.55)
    parser.add_argument("--output", type=str, default=str(WORKSPACE_DATA_PATH))
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    np.random.seed(int(args.seed))

    solver_config = SolverConfig()
    per_arm_count = int(args.train_count + args.eval_count)

    left_positions = _generate_arm_positions(
        side="left",
        solver_config=solver_config,
        count=per_arm_count,
        max_samples=int(args.max_samples),
        voxel_size=float(args.voxel_size),
        joint_margin_threshold=float(args.joint_margin_threshold),
        x_range=(float(args.x_min), float(args.x_max)),
        y_abs_range=(float(args.y_abs_min), float(args.y_abs_max)),
        z_range=(float(args.z_min), float(args.z_max)),
        rng=rng,
    )
    right_positions = _generate_arm_positions(
        side="right",
        solver_config=solver_config,
        count=per_arm_count,
        max_samples=int(args.max_samples),
        voxel_size=float(args.voxel_size),
        joint_margin_threshold=float(args.joint_margin_threshold),
        x_range=(float(args.x_min), float(args.x_max)),
        y_abs_range=(float(args.y_abs_min), float(args.y_abs_max)),
        z_range=(float(args.z_min), float(args.z_max)),
        rng=rng,
    )

    left_train, left_eval = _split_positions(left_positions, int(args.train_count), int(args.eval_count), rng)
    right_train, right_eval = _split_positions(right_positions, int(args.train_count), int(args.eval_count), rng)

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "frame_id": "base_link",
        "sampling_mode": "forward_kinematics_random_joint_sampling",
        "source_urdf": str(Path(solver_config.urdf_path).expanduser().resolve()),
        "seed": int(args.seed),
        "voxel_size_m": float(args.voxel_size),
        "joint_margin_threshold": float(args.joint_margin_threshold),
        "candidate_box": {
            "x": [float(args.x_min), float(args.x_max)],
            "y_abs": [float(args.y_abs_min), float(args.y_abs_max)],
            "z": [float(args.z_min), float(args.z_max)],
        },
        "left_train_positions": left_train,
        "left_eval_positions": left_eval,
        "right_train_positions": right_train,
        "right_eval_positions": right_eval,
    }
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")

    print(f"Saved workspace dataset to: {output_path}")
    print(f"Left  train/eval points : {len(left_train)}/{len(left_eval)}")
    print(f"Right train/eval points : {len(right_train)}/{len(right_eval)}")


if __name__ == "__main__":
    main()
