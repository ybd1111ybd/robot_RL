"""Gym registrations for JZ bimanual drawer opening."""

import gymnasium as gym


gym.register(
    id="Isaac-Open-Drawer-JZ-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "jzlab.tasks.manager_based.jz_manipulation.bimanual.drawer.drawer_env_cfg:JZDrawerEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Open-Drawer-JZ-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "jzlab.tasks.manager_based.jz_manipulation.bimanual.drawer.drawer_env_cfg:JZDrawerEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)
