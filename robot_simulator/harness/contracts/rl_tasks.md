# RL Task Contracts

## Purpose
This document records stable task names, script entry points, and CLI contracts for JZ Isaac Lab reinforcement learning.

## Project Entry Points
- Training script:
  `/workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py`
- Playback script:
  `/workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/play.py`
  - Grasp video clarity options: `--colorize_arms`, `--left_arm_color r,g,b`, `--right_arm_color r,g,b`
  - Playback debug coordinate axes: `--debug_axes`, `--debug_axis_scale <float>`, `--debug_grasp_target_offset x,y,z`
  - The debug TCP axes use the midpoint of the configured fingertip links. For Grasp object targets, debug target axes use the object root plus `--debug_grasp_target_offset`.
- Grasp checkpoint evaluator:
  `/workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/evaluate_grasp_checkpoint.py`
- Task listing script:
  `/workspace/jz_isaac_lab/scripts/tools/list_envs.py`
- Grasp IK sanity script:
  `/workspace/jz_isaac_lab/scripts/tools/debug_grasp_ik.py`

## Known Task IDs
- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`
- `Isaac-Reach-JZ-Bi-Easy-v0`
- `Isaac-Reach-JZ-Bi-Easy-Play-v0`
- `Isaac-Reach-JZ-Bi-Control-v0`
- `Isaac-Reach-JZ-Bi-Control-Play-v0`
- `Isaac-Reach-JZ-Bi-PosOnly-v0`
- `Isaac-Reach-JZ-Bi-PosOnly-Play-v0`
- `Isaac-Reach-JZ-Bi-PosOnly-Smooth-v0`
- `Isaac-Reach-JZ-Bi-PosOnly-Smooth-Play-v0`
- `Isaac-Grasp-JZ-Bi-v0`
- `Isaac-Grasp-JZ-Bi-Play-v0`
- `Isaac-Grasp-JZ-Bi-Fixed-v0`
- `Isaac-Grasp-JZ-Bi-Fixed-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-v0`
- `Isaac-Grasp-JZ-Bi-Approach-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-Play-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-v0`
- `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-Play-v0`
- `Isaac-Open-Drawer-JZ-Bi-v0`
- `Isaac-Open-Drawer-JZ-Bi-Play-v0`

## Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Play-v0`
- Host source path:
  `/mnt/data2/ybd/robot_simulator/jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/grasp/`
- Smoke-validated on GPU 4 in headless Docker.
- Known observation shape: `69`
- Known action shape: `18`
- Action split:
  - left arm: `7`
  - right arm: `7`
  - left gripper: `2`
  - right gripper: `2`
- Known active reward terms: `31`
- Current caveat: arm actions use `JointPositionToLimitsActionCfg`, so zero action maps to joint-limit midpoint targets rather than holding the reset pose.

## Drawer Task
- Training task: `Isaac-Open-Drawer-JZ-Bi-v0`
- Preferred playback task: `Isaac-Open-Drawer-JZ-Bi-Play-v0`
- Host source path:
  `/mnt/data2/ybd/robot_simulator/jz_isaac_lab/source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/drawer/`
- Smoke-validated on GPU 4 in headless Docker.
- Known observation shape: `72`
- Known action shape: `16`
- Action split:
  - left arm: `7`
  - right arm: `7`
  - left gripper: `1`
  - right gripper: `1`
- Known active reward terms: `15`
- Current caveat: arm actions use `JointPositionToLimitsActionCfg`, so zero action maps to joint-limit midpoint targets rather than holding the reset pose.

## Fixed Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Fixed-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Fixed-Play-v0`
- Purpose: learn a stable, natural approach to one fixed object before adding robot/object reset randomization.
- Known observation shape: `69`
- Known action shape: `18`
- Action split:
  - left arm: `7`
  - right arm: `7`
  - left gripper: `2`
  - right gripper: `2`
- Main differences from `Isaac-Grasp-JZ-Bi-v0`:
  - disables robot gravity for the short-term training target
  - fixes robot arm and gripper reset randomization to zero ranges
  - fixes object reset pose to zero ranges
  - replaces arm `JointPositionToLimitsActionCfg` with `JointPositionActionCfg(use_default_offset=True)`
  - sets arm action scale to `0.20`
  - strengthens action-rate, action-max, and joint-velocity penalties
  - adds joint default-posture and joint-limit-margin penalties
- Zero-action validation: arm drift over 120 steps is near zero; body joint 0 still drifts about `0.025 rad`.

## Approach-Only Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-Play-v0`
- Purpose: first train natural left/right approach motion before enabling real grasp/contact/lift rewards.
- Known observation shape: `69`
- Known action shape: `18`
- Main differences from `Isaac-Grasp-JZ-Bi-Fixed-v0`:
  - fixes a gravity-disabled object at a more reachable training pose
  - replaces left/right `tcp_to_object` observation vectors with vectors to side-specific virtual approach targets
  - uses side-target progress/tracking/fine-tracking rewards instead of direct object-center tracking
  - disables object lift, gripper-close/contact, stable-grasp dwell, object motion, table penetration, approach-orientation, and arm-asymmetry rewards for this first curriculum stage
- Rationale: fixed Grasp starts with the object center about `1.11m` from the left TCP and `1.03m` from the right TCP, which induced odd postures when both hands were rewarded to reach the same object center.
- Before PPO training, validate the target with `debug_grasp_ik.py` and inspect the IK interpolation video. If IK position error is above roughly `0.03m` to `0.05m` or the interpolation video is unnatural, adjust target geometry before training.
- Important: `side_target_y` is a virtual side target offset from the object center. The initial `0.45m` target is a far transitional target, not a grasp-ready target near the object. For grasp approach, scan smaller offsets such as `0.25`, `0.18`, and `0.12` before changing the training task.
- The IK sanity gate should also check `joint_margin_ratio` and joint-space path length. A target with position error below `0.05m` but `joint_margin_ratio=0.0000` is still not training-ready because it places at least one joint on a limit. A target that reaches numerically but has large `joint_delta_max_abs` is also not training-ready because interpolation can produce wrist or arm loops.

## 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-Play-v0`
- Purpose: first PPO-only fixed-scene 3D TCP approach demo without IK, 6D orientation, contact, gripper close, object lift, or reset randomization.
- Main differences from `Isaac-Grasp-JZ-Bi-Approach-v0`:
  - target is hand-written near the reset TCP path, not IK-generated
  - left side target offset is `(0.0, 0.60, 0.12)` from the fixed object
  - right side target offset is `(0.0, -0.60, 0.12)` from the fixed object
  - progress reward is disabled
  - 3D tanh tracking weights are `1.0` with `std=0.20`
  - fine 3D tanh tracking weights are `0.5` with `std=0.06`
  - global arm joint velocity penalty is enabled at `-1.0e-3`
  - action rate is `-2.0e-3`, posture is `-3.0e-3`, joint-limit margin is `-1.0e-1`
  - proximity reward curriculums are disabled for this first fixed-scene run
- Qualitative result: `grasp_approach3d_256env_2000it_v1` did not reach the visible targets reliably. Do not continue this exact configuration by only increasing iterations.

## Easy 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Play-v0`
- Purpose: easier PPO sanity run after the first 3D approach target did not reach.
- Main differences from `Isaac-Grasp-JZ-Bi-Approach-3D-v0`:
  - target is the middle bottle object root: left offset `(0.0, 0.0, 0.0)`, right offset `(0.0, 0.0, 0.0)`
  - arm action scale is raised to `0.25`
  - direct side-target distance penalty is enabled with weight `-2.0` per arm
  - broad tanh tracking is strengthened to `1.5` with `std=0.25`
  - fine tanh tracking remains `0.5` with `std=0.08`
  - smoothness/posture/limit penalties are relaxed to avoid a stay-still local optimum
- Current qualitative result: playback still did not show the TCPs reaching the middle bottle. Run `evaluate_grasp_checkpoint.py` and use numeric distances before continuing or changing rewards.

## Easy Smooth 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Smooth-Play-v0`
- Purpose: side-by-side comparison against `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-v0` with only first-layer anti-shake penalties changed.
- Main differences from Easy:
  - `action_rate=-1.0e-3`
  - `action_max_abs_penalty=-5.0e-4`
  - left/right `joint_vel=-8.0e-4`
  - left/right `joint_posture=-1.5e-3`
  - left/right `joint_limit_margin=-5.0e-2`
- Target point, distance rewards, action scale, PPO network, gripper/contact/lift/orientation rewards are unchanged from Easy.

## Easy Weighted 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-Easy-Weighted-Play-v0`
- Purpose: second-layer anti-swing comparison after Easy/Smooth still failed to reach with large action changes.
- Main differences from Easy Smooth:
  - left/right `joint_vel` reward function is `mdp.weighted_joint_vel_l2`
  - joint weights are `[1.5, 1.5, 1.2, 1.2, 0.8, 0.6, 0.5]`
  - left/right `joint_vel` reward weight remains `-8.0e-4`
- Target point, distance rewards, action scale, PPO network, gripper/contact/lift/orientation rewards are unchanged from Easy Smooth.

## TwoTarget 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Play-v0`
- Purpose: easier curriculum after single middle-bottle tracking failed. Each arm tracks its own visible fixed target instead of both arms competing for one center point.
- Scene targets:
  - left visible target object: `(0.62, 0.38, OBJECT_INITIAL_CENTER_Z)`
  - right visible target object: `(0.62, -0.38, OBJECT_INITIAL_CENTER_Z)`
  - TCP target offset for both: `(0.0, 0.0, 0.0)` from the corresponding object root, so the current target is the bottle/target object center
- Visual colors:
  - left arm/gripper and left target: blue
  - right arm/gripper and right target: yellow
- Inherits Easy-Weighted smoothness/velocity penalties.
- Keeps gripper/contact/lift/orientation rewards disabled.
- The current TwoTarget objects have collision, mass, friction, and damping configured, but the approach curriculum keeps them kinematic and gravity-disabled fixed targets.
- Evaluate with `--left_object object --right_object right_object --left_target_offset 0,0,0 --right_target_offset 0,0,0`.

## TwoTarget Stable 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Stable-Play-v0`
- Purpose: fine-tune the TwoTarget checkpoint for steadier target holding before adding gripper closure.
- Main differences from TwoTarget:
  - near-goal joint velocity, TCP speed, and action-rate penalties are gated by the actual side target points, not by the visual object centers
  - bimanual stable dwell reward is enabled around the actual target points
  - gripper action scales are set to zero because this stage is still approach-only
- It keeps observation and action dimensions compatible with the existing TwoTarget checkpoint, so it can resume from `grasp_approach3d_twotarget_256env_2000it_v1/nn/jz_bi_grasp.pth`.

## TwoTarget Track 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-Play-v0`
- Purpose: improve target-reaching accuracy before adding stabilization or gripper closure.
- Main differences from TwoTarget:
  - left/right direct target-distance penalty is strengthened from `-2.0` to `-3.0`
  - left/right fine tanh tracking weight is strengthened from `0.5` to `1.0`
  - fine tracking `std` is tightened from `0.08` to `0.05`
- It intentionally does not add the Stable variant's extra near-target speed/action-rate penalties.
- Current good bottle-center checkpoint preserved outside the RL-Games run directory:
  `/workspace/saved_weights/grasp_twotarget_bottlecenter_good_20260707/jz_bi_grasp.pth`
  Host path:
  `/mnt/data2/ybd/robot_simulator/saved_weights/grasp_twotarget_bottlecenter_good_20260707/jz_bi_grasp.pth`

## Grasp Checkpoint Evaluation CLI Contract
Minimal command shape:
```bash
/isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/evaluate_grasp_checkpoint.py \
  --task Isaac-Grasp-JZ-Bi-Approach-3D-Easy-v0 \
  --num_envs 1 \
  --steps 300 \
  --checkpoint logs/rl_games/jz_bi_grasp/grasp_approach3d_easy_bottle_256env_2000it_v1/nn/jz_bi_grasp.pth \
  --headless
```

Defaults evaluate both TCPs against the bottle root with target offsets `(0, 0, 0)`. For side targets, pass `--left_target_offset x,y,z` and `--right_target_offset x,y,z`.
For separate target objects, pass `--left_object <scene_name>` and `--right_object <scene_name>`.

## Grasp IK Sanity CLI Contract
Minimal command shape:
```bash
/isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/tools/debug_grasp_ik.py \
  --task Isaac-Grasp-JZ-Bi-Approach-v0 \
  --num_envs 1 \
  --headless \
  --ik_random_starts 64 \
  --min_joint_margin 0.03 \
  --max_joint_delta 1.20 \
  --max_joint_delta_norm 2.50
```

For a headless MP4:
```bash
/isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/tools/debug_grasp_ik.py \
  --task Isaac-Grasp-JZ-Bi-Approach-v0 \
  --num_envs 1 \
  --steps 180 \
  --hold_steps 420 \
  --headless \
  --video \
  --video_length 600 \
  --ik_random_starts 64 \
  --min_joint_margin 0.03 \
  --max_joint_delta 1.20 \
  --max_joint_delta_norm 2.50 \
  --camera_eye 3.0,3.2,2.8 \
  --camera_lookat 0.65,0.0,1.25 \
  --run_name grasp_approach_ik_sanity_v1
```

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

## Experimental Control Reach Task
- Training task: `Isaac-Reach-JZ-Bi-Control-v0`
- Preferred playback task: `Isaac-Reach-JZ-Bi-Control-Play-v0`
- Purpose: align JZ Reach control semantics with the OpenArm-style behavior interface before adding more reward shaping.
- Main differences from `Isaac-Reach-JZ-Bi-v0`:
  - replaces `JointPositionToLimitsActionCfg(rescale_to_limits=True)` with `JointPositionActionCfg`
  - uses relative joint-position offsets around the default pose with `use_default_offset=True`
  - action scale set to `0.5`
  - near-target command curriculum set to `(0.1, 0.2, 0.5)`
  - keeps baseline position tracking and tanh fine rewards
  - does not enable Easy v1 progress rewards or goal bonuses
  - strengthens action-rate, action-max-absolute, and joint-velocity penalties

## Short-Term Position-Only Reach Task
- Training task: `Isaac-Reach-JZ-Bi-PosOnly-v0`
- Preferred playback task: `Isaac-Reach-JZ-Bi-PosOnly-Play-v0`
- Purpose: produce a few-day usable 3D Reach result after environment validation showed the current robot default pose drifts under gravity.
- Main differences from `Isaac-Reach-JZ-Bi-Control-v0`:
  - disables robot gravity for the short-term training target
  - keeps OpenArm-style `JointPositionActionCfg(use_default_offset=True)`
  - action scale set to `0.35`
  - reset joint randomization disabled with zero position and velocity ranges
  - exposes only the nearest workspace target pool fractions `(0.05, 0.10, 0.20)`
  - keeps position tracking and tanh fine rewards only, without 6D orientation reward

## Short-Term Smooth Position-Only Reach Task
- Training task: `Isaac-Reach-JZ-Bi-PosOnly-Smooth-v0`
- Preferred playback task: `Isaac-Reach-JZ-Bi-PosOnly-Smooth-Play-v0`
- Purpose: reduce the looping/excessive joint motion seen in `reach_posonly_64env_300it_v1`.
- Main differences from `Isaac-Reach-JZ-Bi-PosOnly-v0`:
  - action scale reduced from `0.35` to `0.20`
  - action-rate, action-max, and joint-velocity penalties strengthened to `-2.0e-3`

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

Headless playback videos can set camera position:
```bash
--camera_eye 3.2,3.4,3.0
--camera_lookat 0.65,0.0,1.35
```

## Checkpoint Contract
RL-Games checkpoints are expected under:
```text
/workspace/jz_isaac_lab/logs/rl_games/<config-name>/<run-name>/nn/
```
For Reach, the host path is:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/jz_bi_reach/<run-name>/nn/
```
