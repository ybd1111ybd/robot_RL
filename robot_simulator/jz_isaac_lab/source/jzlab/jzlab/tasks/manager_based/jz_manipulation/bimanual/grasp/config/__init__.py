"""Gym registrations for JZ bimanual grasp."""

import gymnasium as gym


gym.register(
    id="Isaac-Grasp-JZ-Bi-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspFixedEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Fixed-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspFixedEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproachEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproachEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasyEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasyEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasySmoothEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasySmoothEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasyWeightedEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DEasyWeightedEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetStableEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetStableEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetTrackEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetTrackEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Dynamic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetDynamicEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Dynamic-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetDynamicEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-GraspClose-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetGraspCloseEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Open6D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetOpen6DEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Open6D-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetOpen6DEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregrasp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV2EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV3EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV4-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV4EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV41-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV41-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV41EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV42-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV42EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV42-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV42EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV43-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV43-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV43EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV44-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV44EnvCfg",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV44-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetSurfacePregraspV44EnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_v41_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-GraspClose-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:JZGraspApproach3DTwoTargetGraspCloseEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{__name__}.agents:rl_games_ppo_cfg.yaml",
    },
)
