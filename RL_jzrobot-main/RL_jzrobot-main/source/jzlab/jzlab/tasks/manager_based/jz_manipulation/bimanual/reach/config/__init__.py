"""Gym registrations for JZ bimanual reach."""

import gymnasium as gym


gym.register(
    id="Isaac-Reach-JZ-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZReachEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-JZ-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZReachEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)
