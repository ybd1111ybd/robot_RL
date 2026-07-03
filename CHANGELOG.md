# Changelog

## 2026-07-02
- Created AI Harness entry file `agent.md`.
- Created long-term memory files `PROJECT_STATE.md`, `ROADMAP.md`, and `CHANGELOG.md`.
- Created harness directory structure:
  - `harness/tasks/`
  - `harness/contracts/`
  - `harness/context/`
  - `harness/decisions/`
- Recorded initial project phase and Isaac Lab RL bring-up baseline.
- Added `harness/context/docker_isaac_lab.md` with host/container paths, Docker service, environment variables, and runtime notes.
- Added `harness/contracts/rl_tasks.md` with RL task IDs, script entry points, CLI contracts, and checkpoint path rules.
- Added `harness/tasks/reach_training.md` with smoke training, longer training, monitoring, playback, and study order.
- Updated `PROJECT_STATE.md` to reference the new harness documentation.
- Added `harness/context/server_migration.md` with target server requirements, project copy list, Docker image migration options, `.env` path updates, verification commands, and common failure points.
- Updated `PROJECT_STATE.md` to reference the server migration guide.
- Added `harness/context/reach_task_anatomy.md` documenting the Reach task registration, scene, actions, observations, commands, rewards, termination, curriculum, and PPO configuration.
- Updated `PROJECT_STATE.md` to reference the Reach task anatomy document.
- Recorded the completed `reach_128env_2000it_v1` baseline result and headless checkpoint evaluation in `harness/tasks/reach_training.md`.
- Updated `PROJECT_STATE.md` with the baseline conclusion: reward improved, but reaching behavior is not yet successful.
- Added experimental Easy Reach task registrations `Isaac-Reach-JZ-Bi-Easy-v0` and `Isaac-Reach-JZ-Bi-Easy-Play-v0`.
- Added `JZReachEasyEnvCfg` and `JZReachEasyEnvCfg_PLAY` with reduced action scale, workspace curriculum, progress rewards, goal bonuses, and action max-absolute penalty.
- Updated RL task contracts, Reach training notes, and project state with the Easy Reach experiment.
- Recorded completed `reach_easy_128env_2000it_v1` results. Despite higher reward, numerical evaluation showed worse TCP errors and more aggressive actions than baseline.
- Updated `PROJECT_STATE.md` with the Easy v1 conclusion and marked the exact configuration as unsuitable for continuation.
- Added `harness/tasks/reach_tuning_log.md` recording the current Reach tuning rationale, baseline symptoms, planned parameter changes, evaluation criteria, and command templates.
- Updated `PROJECT_STATE.md` to reference the Reach tuning log.
- Rewrote `harness/tasks/reach_tuning_log.md` in Chinese and shortened it into a practical working log for current tuning.
