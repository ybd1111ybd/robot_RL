# Reach Task Anatomy

## Purpose
This document explains the current `Isaac-Reach-JZ-Bi-v0` task so future training, debugging, and reward changes start from the same mental model.

## Scope
This document covers only the JZ bimanual Reach task. It does not cover Grasp or Drawer.

## Main Files
- Task registration:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/config/__init__.py`
- Base environment config:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/reach_env_cfg.py`
- Joint-position specialization:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/config/joint_pos_env_cfg.py`
- PPO config:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/config/agents/rl_games_ppo_cfg.yaml`
- Custom MDP helpers:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/mdp/`
- Shared robot constants:
  `jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/constants.py`

## Task Registration
Two Gymnasium task IDs are registered:
- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

Both use:
```text
entry_point="isaaclab.envs:ManagerBasedRLEnv"
```

Training task config:
```text
env_cfg_entry_point = joint_pos_env_cfg:JZReachEnvCfg
rl_games_cfg_entry_point = agents:rl_games_ppo_cfg.yaml
```

Playback task config:
```text
env_cfg_entry_point = joint_pos_env_cfg:JZReachEnvCfg_PLAY
rl_games_cfg_entry_point = agents:rl_games_ppo_cfg.yaml
```

## Environment Type
The task is an Isaac Lab manager-based RL environment:
```text
ManagerBasedRLEnvCfg
```

The environment is composed from managers:
- scene
- actions
- observations
- commands
- rewards
- terminations
- events
- curriculum

## Scene
The base scene contains:
- ground plane at `/World/ground`
- dome light at `/World/light`
- one JZ bimanual robot replicated across environments

Default training scene:
```text
num_envs = 512
env_spacing = 2.5
```

Playback scene:
```text
num_envs = 50
env_spacing = 2.5
```

The active robot asset is assigned in `JZReachEnvCfg.__post_init__()`:
```text
JZ_BIMANUAL_HIGH_PD_CFG
```

The robot is spawned under:
```text
{ENV_REGEX_NS}/Robot
```

## Simulation Step Timing
The base config sets:
```text
decimation = 2
render_interval = decimation
```

This means one policy action is applied across two lower-level simulation steps.

## Robot Control
This task controls only the two 7-DoF arms.

Left arm joints:
```text
left_arm_joint1 ... left_arm_joint7
```

Right arm joints:
```text
right_arm_joint1 ... right_arm_joint7
```

Gripper joints exist in shared constants, but they are not part of the Reach action space.

Actuator tuning for this task:
```text
arm stiffness = 260.0
arm damping = 36.0
solver_velocity_iteration_count = 1
```

## Action Space
The policy action shape is:
```text
14
```

It is split as:
- `left_arm_action`: 7 dimensions
- `right_arm_action`: 7 dimensions

Both action terms use:
```text
mdp.JointPositionToLimitsActionCfg
scale = 1.0
rescale_to_limits = True
```

Interpretation:
- The policy outputs normalized continuous values.
- The action term maps them into valid joint-position targets.
- The controller follows joint-position targets using the high-PD robot config.

## TCP Definition
The task uses a synthetic TCP position per gripper.

Left TCP position links:
```text
left_gripper_narrow3_link
left_gripper_wide3_link
```

Right TCP position links:
```text
right_gripper_narrow3_link
right_gripper_wide3_link
```

The synthetic TCP position is the midpoint of the two fingertip link positions.

Orientation body links:
```text
left_arm_link9
right_arm_link9
```

The command/current orientation uses `GRIPPER_MOUNT_QUAT` as an offset.

## Command Generation
The task has two active command terms:
- `left_ee_pose`
- `right_ee_pose`

Both use:
```text
ReachableWorkspacePoseCommandCfg
```

Target positions are sampled from:
```text
reach/workspace/reachable_workspace.json
```

Training dataset keys:
```text
left_train_positions
right_train_positions
```

Playback dataset keys:
```text
left_eval_positions
right_eval_positions
```

Command positions are expressed in the robot `base_link` frame.

The task currently uses fixed target quaternions from the active orientation preset:
```text
use_fixed_quaternion = True
```

Training command resampling time:
```text
4.0 seconds
```

Base config originally defines `6.0 seconds`, but `JZReachEnvCfg` overrides both arms to `4.0 seconds`.

Curriculum for command sampling is currently neutral:
```text
curriculum_stage_fractions = (1.0, 1.0, 1.0)
curriculum_stage_steps = (0, 0)
```

This means the full workspace dataset is available from the start.

## Observation Space
The observed policy vector is concatenated into:
```text
shape = 68
```

Observation corruption is enabled for training:
```text
enable_corruption = True
```

Playback disables observation corruption:
```text
enable_corruption = False
```

Observation terms:

| Term | Shape | Meaning |
| --- | ---: | --- |
| `left_joint_pos` | 7 | relative left arm joint positions |
| `right_joint_pos` | 7 | relative right arm joint positions |
| `left_joint_vel` | 7 | relative left arm joint velocities |
| `right_joint_vel` | 7 | relative right arm joint velocities |
| `left_tcp_pos` | 3 | left synthetic TCP position in base frame |
| `right_tcp_pos` | 3 | right synthetic TCP position in base frame |
| `left_pose_command` | 7 | left desired pose command |
| `right_pose_command` | 7 | right desired pose command |
| `left_tcp_error` | 3 | left desired minus current TCP position |
| `right_tcp_error` | 3 | right desired minus current TCP position |
| `left_actions` | 7 | previous left arm action |
| `right_actions` | 7 | previous right arm action |

Shape check:
```text
7 + 7 + 7 + 7 + 3 + 3 + 7 + 7 + 3 + 3 + 7 + 7 = 68
```

Noise:
- joint positions have uniform noise in `[-0.01, 0.01]`
- joint velocities have uniform noise in `[-0.01, 0.01]`
- TCP positions, commands, errors, and previous actions do not explicitly define noise in the current config

## Reset Event
On reset, both arms are reset with:
```text
position_range = (-0.25, 0.25)
velocity_range = (0.0, 0.0)
```

The reset applies to:
```text
LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
```

## Termination
The only active termination is:
```text
time_out
```

There is no success termination, collision termination, or distance-failure termination in the current Reach config.

## Active Rewards
There are 7 active reward terms.

| Term | Weight | Meaning |
| --- | ---: | --- |
| `left_end_effector_position_tracking` | -0.20 | penalizes left TCP distance to target |
| `right_end_effector_position_tracking` | -0.20 | penalizes right TCP distance to target |
| `left_end_effector_position_tracking_fine_grained` | 0.20 | tanh-shaped left proximity reward |
| `right_end_effector_position_tracking_fine_grained` | 0.20 | tanh-shaped right proximity reward |
| `action_rate` | -0.0001 | penalizes action changes |
| `left_joint_vel` | -0.0001 | penalizes left arm joint velocity |
| `right_joint_vel` | -0.0001 | penalizes right arm joint velocity |

The coarse tracking term is distance-based:
```text
reward = weight * ||current_tcp - target_tcp||
```

The fine-grained term uses:
```text
1.0 - tanh(distance / std)
std = 0.10
```

Interpretation:
- Far from the target, the negative distance penalty dominates.
- Near the target, the tanh term gives a positive shaping reward.
- Smoothness is encouraged through action-rate and joint-velocity penalties.

## Disabled Reward Hooks
Several reward hooks exist but are currently set to `None`, including:
- progress reward
- success bonus
- bimanual success bonus
- stable goal bonus
- orientation tracking
- max action penalty
- joint deviation penalty
- near-goal velocity and action-rate penalties

These hooks are useful candidates for later reward iteration, but should not be enabled until baseline training curves are understood.

## Reward Curriculum
Three curriculum terms gradually increase smoothness penalties over `4500` steps:

| Term | Initial Weight | Target Weight | Steps |
| --- | ---: | ---: | ---: |
| `action_rate` | -0.0001 | -0.005 | 4500 |
| `left_joint_vel` | -0.0001 | -0.001 | 4500 |
| `right_joint_vel` | -0.0001 | -0.001 | 4500 |

Interpretation:
- Early learning is allowed to move more freely.
- Smoothness pressure becomes stronger after the policy starts learning target tracking.

## PPO / RL-Games Configuration
Config name:
```text
jz_bi_reach
```

Main settings:
```text
seed = 42
max_epochs = 2000
horizon_length = 24
minibatch_size = 96
mini_epochs = 5
learning_rate = 1e-3
gamma = 0.99
tau = 0.95
entropy_coef = 0.01
network units = [128, 128]
activation = elu
normalize_input = True
normalize_value = True
```

Important training implication:
- `horizon_length * num_envs` should be large enough for `minibatch_size`.
- With `horizon_length = 24` and `minibatch_size = 96`, avoid very small `num_envs`.
- The validated smoke run used `num_envs = 32`.

## Current Mental Model
At each policy step:
1. The command manager samples left and right target poses from reachable-workspace datasets.
2. The observation manager builds a 68-dimensional vector.
3. The policy outputs 14 normalized continuous actions.
4. The action manager maps actions to left/right arm joint-position targets.
5. The simulator advances with decimation `2`.
6. Reward terms score target tracking and smoothness.
7. Episodes end only on timeout.

## Safe Next Experiments
Before changing rewards, run a longer baseline:
```text
Isaac-Reach-JZ-Bi-v0
num_envs = 128 or 512
max_iterations = 2000
torch_compile = False
```

Then inspect:
- episode reward trend
- left/right position error metrics
- action-rate penalty
- joint-velocity penalty
- checkpoint playback behavior

## First Reward Changes To Consider Later
Only after baseline training is understood, consider:
- enabling progress reward to encourage each command step to reduce error
- adding success bonus once the policy can reach targets
- adding stable goal dwell reward if the arms reach but oscillate
- adjusting command resampling time if targets change too quickly
- narrowing workspace dataset exposure if early exploration is too hard
