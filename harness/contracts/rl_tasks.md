# RL Task Contracts

## Purpose
This document records stable task names, script entry points, and CLI contracts for JZ Isaac Lab reinforcement learning.

## Project Entry Points
- Training script:
  `/workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py`
- Playback script:
  `/workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/play.py`
- Task listing script:
  `/workspace/jz_isaac_lab/scripts/tools/list_envs.py`

## Known Task IDs
- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`
- `Isaac-Reach-JZ-Bi-Easy-v0`
- `Isaac-Reach-JZ-Bi-Easy-Play-v0`
- `Isaac-Grasp-JZ-Bi-v0`
- `Isaac-Grasp-JZ-Bi-Play-v0`
- `Isaac-Open-Drawer-JZ-Bi-v0`
- `Isaac-Open-Drawer-JZ-Bi-Play-v0`

## Initial Validated Task
- Training task: `Isaac-Reach-JZ-Bi-v0`
- Preferred playback task: `Isaac-Reach-JZ-Bi-Play-v0`
- RL-Games config name: `jz_bi_reach`
- Known observation shape: `68`
- Known action shape: `14`
- Action split:
  - left arm: `7`
  - right arm: `7`
- Known active reward terms: `7`

## Experimental Easy Reach Task
- Training task: `Isaac-Reach-JZ-Bi-Easy-v0`
- Preferred playback task: `Isaac-Reach-JZ-Bi-Easy-Play-v0`
- Purpose: reduce joint-limit spikes and bootstrap target tracking after the baseline policy showed large/aggressive actions.
- Main differences from `Isaac-Reach-JZ-Bi-v0`:
  - action scale reduced from `1.0` to `0.25`
  - workspace command curriculum enabled with fractions `(0.2, 0.5, 1.0)`
  - left/right progress rewards enabled
  - left/right 10 cm goal bonuses enabled
  - action max-absolute penalty enabled

## Training CLI Contract
Minimal training command shape:
```bash
/isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
  --task <task-id> \
  --num_envs <env-count> \
  --max_iterations <iterations> \
  --headless
```

Recommended JZ Reach override:
```bash
+agent.params.config.torch_compile=False
```

Stable experiment naming override:
```bash
+agent.params.config.full_experiment_name=<run-name>
```

Existing RL-Games config fields should be overridden without `+`, for example:
```bash
agent.params.config.save_frequency=50
agent.params.config.save_best_after=0
```

New Hydra keys should be added with `+`.

## Playback CLI Contract
Minimal playback command shape:
```bash
/isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/play.py \
  --task <task-id> \
  --num_envs <env-count> \
  --checkpoint <checkpoint-path>
```

For early validation, use `--headless`. For visual playback, omit `--headless` only after display forwarding is known to work.

## Checkpoint Contract
RL-Games checkpoints are expected under:
```text
/workspace/jz_isaac_lab/logs/rl_games/<config-name>/<run-name>/nn/
```
For Reach, the host path is:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/<run-name>/nn/
```
