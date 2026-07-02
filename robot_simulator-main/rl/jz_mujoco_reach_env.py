"""Minimal MuJoCo reach environment for the JZ dual-arm robot.

This module intentionally avoids ROS 2 and Isaac Sim. It is the smallest
project-local RL boundary around the available MJCF model, so training code can
iterate on observations, actions, and rewards before moving to Isaac Lab.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent.
    gym = None
    spaces = None

import mujoco


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "Jz_descripetion-main"
    / "Jz_descripetion-main"
    / "robot_urdf"
    / "urdf"
    / "robot_model.mjcf.xml"
)

ARM_JOINTS = (
    "left_arm1",
    "left_arm2",
    "left_arm3",
    "left_arm4",
    "left_arm5",
    "left_arm6",
    "left_arm7",
    "right_arm1",
    "right_arm2",
    "right_arm3",
    "right_arm4",
    "right_arm5",
    "right_arm6",
    "right_arm7",
)

BODY_JOINTS = ("body3", "body2", "body1", "hand2", "hand1")


@dataclass(frozen=True)
class ReachTargets:
    left: np.ndarray
    right: np.ndarray


class _FallbackBox:
    """Small replacement used by smoke tests when Gymnasium is not installed."""

    def __init__(self, low: np.ndarray, high: np.ndarray, dtype: np.dtype = np.float32) -> None:
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class _FallbackEnv:
    pass


BaseEnv = gym.Env if gym is not None else _FallbackEnv


class JzMujocoReachEnv(BaseEnv):
    """Gymnasium-style reach task for the loaded MJCF robot.

    Action:
        14 normalized values in [-1, 1], applied as target-position deltas for
        left/right arm position actuators.

    Observation:
        arm qpos, arm qvel, left/right end-effector positions, left/right target
        positions, and end-effector position errors.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        episode_steps: int = 250,
        frame_skip: int = 10,
        action_scale: float = 0.04,
        success_tolerance_m: float = 0.04,
        target_noise_m: float = 0.12,
        seed: int | None = None,
    ) -> None:
        if gym is not None:
            super().__init__()

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.episode_steps = int(episode_steps)
        self.frame_skip = int(frame_skip)
        self.action_scale = float(action_scale)
        self.success_tolerance_m = float(success_tolerance_m)
        self.target_noise_m = float(target_noise_m)
        self.np_random = np.random.default_rng(seed)

        self._joint_qpos_addr = self._resolve_joint_qpos(ARM_JOINTS)
        self._joint_dof_addr = self._resolve_joint_dof(ARM_JOINTS)
        self._actuator_addr = self._resolve_actuators(ARM_JOINTS)
        self._body_joint_qpos_addr = self._resolve_joint_qpos(BODY_JOINTS)
        self._body_actuator_addr = self._resolve_actuators(BODY_JOINTS)
        self._left_site_id = self._resolve_site("left_ee")
        self._right_site_id = self._resolve_site("right_ee")
        self._home_key_id = self._resolve_key("teleop_home", fallback="home")

        self._ctrl_low = self.model.actuator_ctrlrange[self._actuator_addr, 0].astype(float)
        self._ctrl_high = self.model.actuator_ctrlrange[self._actuator_addr, 1].astype(float)
        self._targets = ReachTargets(np.zeros(3, dtype=float), np.zeros(3, dtype=float))
        self._steps = 0

        action_low = -np.ones(len(ARM_JOINTS), dtype=np.float32)
        action_high = np.ones(len(ARM_JOINTS), dtype=np.float32)
        observation_size = len(ARM_JOINTS) * 2 + 3 * 6
        obs_low = -np.inf * np.ones(observation_size, dtype=np.float32)
        obs_high = np.inf * np.ones(observation_size, dtype=np.float32)

        if spaces is not None:
            self.action_space = spaces.Box(action_low, action_high, dtype=np.float32)
            self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        else:
            self.action_space = _FallbackBox(action_low, action_high)
            self.observation_space = _FallbackBox(obs_low, obs_high)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, object] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, object]]:
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)
        mujoco.mj_forward(self.model, self.data)

        self.data.ctrl[self._actuator_addr] = self.data.qpos[self._joint_qpos_addr]
        self.data.ctrl[self._body_actuator_addr] = self.data.qpos[self._body_joint_qpos_addr]

        self._steps = 0
        self._targets = self._sample_targets()
        return self._observation(), self._info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        action = np.asarray(action, dtype=float)
        if action.shape != (len(ARM_JOINTS),):
            raise ValueError(f"action must have shape {(len(ARM_JOINTS),)}, got {action.shape}")

        delta = np.clip(action, -1.0, 1.0) * self.action_scale
        current_ctrl = self.data.ctrl[self._actuator_addr].astype(float)
        next_ctrl = np.clip(current_ctrl + delta, self._ctrl_low, self._ctrl_high)
        self.data.ctrl[self._actuator_addr] = next_ctrl

        # Hold body and wrist-base joints at their reset positions while the
        # first RL task learns arm reaching.
        self.data.ctrl[self._body_actuator_addr] = self.data.qpos[self._body_joint_qpos_addr]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._steps += 1
        obs = self._observation()
        info = self._info()
        reward = self._reward(info)
        terminated = bool(info["success"])
        truncated = self._steps >= self.episode_steps
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        return None

    def _sample_targets(self) -> ReachTargets:
        left_ee, right_ee = self._ee_positions()
        left_target = left_ee + self.np_random.uniform(
            -self.target_noise_m, self.target_noise_m, size=3
        )
        right_target = right_ee + self.np_random.uniform(
            -self.target_noise_m, self.target_noise_m, size=3
        )
        left_target[2] = max(left_target[2], 0.25)
        right_target[2] = max(right_target[2], 0.25)
        return ReachTargets(left_target.astype(float), right_target.astype(float))

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self._joint_qpos_addr]
        qvel = self.data.qvel[self._joint_dof_addr]
        left_ee, right_ee = self._ee_positions()
        obs = np.concatenate(
            [
                qpos,
                qvel,
                left_ee,
                right_ee,
                self._targets.left,
                self._targets.right,
                self._targets.left - left_ee,
                self._targets.right - right_ee,
            ]
        )
        return obs.astype(np.float32)

    def _reward(self, info: Dict[str, object]) -> float:
        left_dist = float(info["left_distance_m"])
        right_dist = float(info["right_distance_m"])
        action_penalty = 1e-4 * float(np.sum(np.square(self.data.ctrl[self._actuator_addr])))
        success_bonus = 2.0 if bool(info["success"]) else 0.0
        return -(left_dist + right_dist) - action_penalty + success_bonus

    def _info(self) -> Dict[str, object]:
        left_ee, right_ee = self._ee_positions()
        left_dist = float(np.linalg.norm(self._targets.left - left_ee))
        right_dist = float(np.linalg.norm(self._targets.right - right_ee))
        return {
            "left_ee": left_ee.copy(),
            "right_ee": right_ee.copy(),
            "left_target": self._targets.left.copy(),
            "right_target": self._targets.right.copy(),
            "left_distance_m": left_dist,
            "right_distance_m": right_dist,
            "success": left_dist < self.success_tolerance_m
            and right_dist < self.success_tolerance_m,
        }

    def _ee_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.data.site_xpos[self._left_site_id].copy(),
            self.data.site_xpos[self._right_site_id].copy(),
        )

    def _resolve_joint_qpos(self, names: Iterable[str]) -> np.ndarray:
        addresses = []
        for name in names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint not found in model: {name}")
            addresses.append(int(self.model.jnt_qposadr[joint_id]))
        return np.asarray(addresses, dtype=int)

    def _resolve_joint_dof(self, names: Iterable[str]) -> np.ndarray:
        addresses = []
        for name in names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint not found in model: {name}")
            addresses.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(addresses, dtype=int)

    def _resolve_actuators(self, names: Iterable[str]) -> np.ndarray:
        addresses = []
        for name in names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"Actuator not found in model: {name}")
            addresses.append(actuator_id)
        return np.asarray(addresses, dtype=int)

    def _resolve_site(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site not found in model: {name}")
        return site_id

    def _resolve_key(self, preferred: str, fallback: str) -> int:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, preferred)
        if key_id >= 0:
            return key_id
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, fallback)
        if key_id >= 0:
            return key_id
        raise ValueError(f"Neither keyframe '{preferred}' nor '{fallback}' exists")

