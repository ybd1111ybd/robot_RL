# Project State

## Phase
Harness initialization and Isaac Lab reinforcement learning bring-up.

## Repository Layout
- `robot_simulator-main/`: main simulator, Docker, ROS, MuJoCo, and Isaac runtime configuration area.
- `Jz_descripetion-main/`: JZ robot model assets, including URDF, MJCF, meshes, and related description files.
- `jz_isaac_lab/`: Isaac Lab reinforcement learning project mounted into Docker as `/workspace/jz_isaac_lab`.
- `harness/`: AI collaboration harness for task protocols, contracts, runtime context, and decisions.

## Current Technical State
- Isaac Sim / Isaac Lab Docker workflow is being prepared for JZ robot reinforcement learning.
- `jz_isaac_lab` contains JZ Isaac Lab task code, RL-Games scripts, documentation, and task registrations.
- The initial validated training target is `Isaac-Reach-JZ-Bi-v0`.
- A short smoke training run for the Reach task has completed in Docker with RL-Games.
- Smoke checkpoint path:
  `/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/docker_smoke_32x5_no_compile/nn/last_jz_bi_reach_ep_5_rew_-inf.pth`
- Harness documentation now records Docker/Isaac Lab runtime context, RL task contracts, and the Reach training procedure.
- Reach task anatomy has been documented, including task registration, action space, observation space, command sampling, rewards, termination, curriculum, and PPO settings.
- Server migration guidance has been documented for moving the project, Docker image, paths, logs, and validation workflow to another machine.
- Baseline `reach_128env_2000it_v1` completed. Reward improved to about `2.76`, but headless evaluation still shows large TCP errors and zero near-goal ratio, so the policy is not yet a successful reaching policy.
- User video observation showed apparent joint-limit hits and erratic motion. Experimental Easy Reach task `Isaac-Reach-JZ-Bi-Easy-v0` was added to reduce action scale, add command curriculum, and enable progress/goal/action-bound rewards.
- Easy run `reach_easy_128env_2000it_v1` completed. Reward rose to about `36.35`, but headless evaluation worsened actual reaching error and action aggressiveness, so this exact Easy v1 configuration should not be continued.
- Current Reach tuning rationale and parameter plan are recorded in `harness/tasks/reach_tuning_log.md`.

## Known Working Baseline
- Docker mount target for the RL project: `/workspace/jz_isaac_lab`.
- Host RL project path: `/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab`.
- Initial task: `Isaac-Reach-JZ-Bi-v0`.
- Experimental task: `Isaac-Reach-JZ-Bi-Easy-v0`.
- Training can run headless with `+agent.params.config.torch_compile=False`.
- Runtime context: `harness/context/docker_isaac_lab.md`.
- Server migration guide: `harness/context/server_migration.md`.
- Reach task anatomy: `harness/context/reach_task_anatomy.md`.
- RL task contract: `harness/contracts/rl_tasks.md`.
- Reach training task guide: `harness/tasks/reach_training.md`.
- Reach tuning log: `harness/tasks/reach_tuning_log.md`.
- Latest baseline result: `harness/tasks/reach_training.md`.

## Long-Term Memory Rule
This file is long-term memory. Chat history is not long-term memory. Update this file whenever repository structure, configuration, interfaces, runtime behavior, or project phase changes.
