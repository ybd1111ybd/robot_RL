# MuJoCo RL Entry

This directory is the project-local reinforcement learning entry point for the
available JZ MuJoCo robot model. It does not require Isaac Sim, Isaac Lab, or
ROS 2.

## Current Model

Default model path:

```bash
/home/ybd/robot_simulator-main/Jz_descripetion-main/Jz_descripetion-main/robot_urdf/urdf/robot_model.mjcf.xml
```

The model provides:

- 19 joints and 19 position actuators.
- `left_ee` and `right_ee` MuJoCo sites.
- `home` and `teleop_home` keyframes.

## Smoke Test

Run from the repository root:

```bash
cd /home/ybd/robot_simulator-main/robot_simulator-main
python3 -m rl.smoke_reach_env
```

The smoke test creates `JzMujocoReachEnv`, resets it, applies random arm
actions, and prints end-effector distances to sampled reach targets.

## Environment Contract

`JzMujocoReachEnv` is Gymnasium-style, but Gymnasium is optional for the smoke
test.

- Action: 14 normalized values in `[-1, 1]` for left/right arm joint target
  deltas.
- Observation: arm joint positions, arm joint velocities, end-effector
  positions, target positions, and position errors.
- Reward: negative left/right reach distance, small control penalty, success
  bonus.

This is a minimal reach task. It is meant to establish the RL loop before
adding PPO/rl_games, orientation rewards, grasping, drawers, or Isaac Lab task
registration.

