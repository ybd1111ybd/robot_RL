# Reach Training Task

## Purpose
This document records the working procedure for learning and training the JZ bimanual Reach task.

## Current Goal
Use `Isaac-Reach-JZ-Bi-v0` as the first stable reinforcement learning target before moving to Grasp or Drawer tasks.

## Smoke Training Baseline
The Docker training path has been validated with:
- Task: `Isaac-Reach-JZ-Bi-v0`
- Environment count: `32`
- Iterations: `5`
- Mode: headless
- Torch compile: disabled
- Result: training completed and saved a checkpoint.

Smoke checkpoint on host:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/docker_smoke_32x5_no_compile/nn/last_jz_bi_reach_ep_5_rew_-inf.pth
```

The smoke reward may show `-inf` because the run is intentionally too short for meaningful learning and may finish before any environment terminates.

## Smoke Training Command
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Reach-JZ-Bi-v0 \
    --num_envs 32 \
    --max_iterations 5 \
    --headless \
    +agent.params.config.full_experiment_name=docker_smoke_32x5_no_compile \
    agent.params.config.save_frequency=1 \
    agent.params.config.save_best_after=0 \
    +agent.params.config.torch_compile=False
'
```

## Recommended First Longer Training Run
Use this if GPU memory allows:
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Reach-JZ-Bi-v0 \
    --num_envs 512 \
    --max_iterations 2000 \
    --headless \
    +agent.params.config.full_experiment_name=reach_512env_2000it_v1 \
    +agent.params.config.torch_compile=False
'
```

If memory is limited, use `--num_envs 128` or `--num_envs 256`.

## Baseline Result: reach_128env_2000it_v1
Run status:
- Completed.
- Task: `Isaac-Reach-JZ-Bi-v0`.
- Environment count: `128`.
- Iterations: `2000`.
- Latest checkpoint:
  `/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/reach_128env_2000it_v1/nn/last_jz_bi_reach_ep_2000_rew_2.7608962.pth`

Checkpoint reward trend from saved filenames:
- Early run included negative rewards, including epoch 300 at about `-2.77`.
- Later run improved to about `2.76` at epoch 2000.
- This indicates PPO optimization is active and reward improved during training.

Headless evaluation of the epoch-2000 checkpoint with `64` envs and `600` steps:
```text
left_error_mean: 0.680429
left_error_last50: 0.683472
right_error_mean: 0.725496
right_error_last50: 0.743378
near_goal_ratio_mean: 0.000000
near_goal_ratio_last50: 0.000000
settle_ratio_mean: 0.000000
settle_ratio_last50: 0.000000
max_action_mean: 4.980184
action_rate_mean: 1.451408
joint_accel_max_abs_mean: 147.527506
```

Interpretation:
- The training loop works and reward improved.
- The learned policy is not yet a successful reaching policy.
- TCP errors remain large, near-goal ratio is zero, and the policy appears to use large/aggressive actions.
- Next work should inspect playback visually if possible, then tune reward shaping, action scale, workspace difficulty, or success/settling rewards.

User video observation:
- A 9-second headless MP4 showed four parallel robots.
- The robots appeared to hit joint limits and move erratically.
- This matches the high `max_action_mean`, `action_rate_mean`, and `joint_accel_max_abs_mean` from numerical evaluation.

## Experimental Easy Reach Variant
Task IDs:
- `Isaac-Reach-JZ-Bi-Easy-v0`
- `Isaac-Reach-JZ-Bi-Easy-Play-v0`

Purpose:
- Reduce joint-limit spikes.
- Make early reaching easier.
- Add clearer progress and goal-reaching learning signals.
- Preserve the original `Isaac-Reach-JZ-Bi-v0` baseline unchanged.

Changes from baseline:
- Action scale reduced from `1.0` to `0.25`.
- Workspace curriculum enabled with fractions `(0.2, 0.5, 1.0)` and stage steps `(5000, 20000)`.
- Left/right progress rewards enabled.
- Left/right 10 cm goal bonuses enabled.
- Action max-absolute penalty enabled with weight `-1.0e-3`.

Smoke command:
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Reach-JZ-Bi-Easy-v0 \
    --num_envs 32 \
    --max_iterations 5 \
    --headless \
    +agent.params.config.full_experiment_name=reach_easy_smoke_32x5_v1 \
    agent.params.config.save_frequency=1 \
    agent.params.config.save_best_after=0 \
    +agent.params.config.torch_compile=False
'
```

First comparable training run:
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Reach-JZ-Bi-Easy-v0 \
    --num_envs 128 \
    --max_iterations 2000 \
    --headless \
    +agent.params.config.full_experiment_name=reach_easy_128env_2000it_v1 \
    +agent.params.config.torch_compile=False
'
```

Comparison target against baseline:
```text
baseline left_error_mean: 0.680429
baseline right_error_mean: 0.725496
baseline near_goal_ratio_mean: 0.000000
baseline action_rate_mean: 1.451408
baseline joint_accel_max_abs_mean: 147.527506
```

## Monitoring
TensorBoard log root:
```bash
tensorboard --logdir /home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games
```

Expected Reach run directory:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/<run-name>
```

## Playback
Replace `<checkpoint>` with a real `.pth` file:
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Reach-JZ-Bi-v0 \
    --num_envs 8 \
    --checkpoint <checkpoint> \
    --headless
'
```

## Study Order
Study Reach in this order:
1. Task registration.
2. Environment configuration.
3. Actions.
4. Observations.
5. Commands and target sampling.
6. Rewards.
7. Terminations and curriculum.
8. PPO / RL-Games configuration.

## Learning Questions
- What does the policy observe?
- What can the policy control?
- What behavior is rewarded?
- When does an episode terminate?
- Which PPO settings affect stability and speed?
