#!/usr/bin/env python3
"""Smoke-test the project-local MuJoCo reach environment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rl.jz_mujoco_reach_env import DEFAULT_MODEL_PATH, JzMujocoReachEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = JzMujocoReachEnv(
        model_path=Path(args.model_path),
        episode_steps=args.steps,
        seed=args.seed,
    )

    rng = np.random.default_rng(args.seed)
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        final_info = info
        for _ in range(args.steps):
            action = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
            obs, reward, terminated, truncated, final_info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(
            "episode={episode} obs_shape={obs_shape} total_reward={reward:.3f} "
            "left_dist={left:.4f} right_dist={right:.4f} success={success}".format(
                episode=episode,
                obs_shape=obs.shape,
                reward=total_reward,
                left=float(final_info["left_distance_m"]),
                right=float(final_info["right_distance_m"]),
                success=bool(final_info["success"]),
            )
        )


if __name__ == "__main__":
    main()

