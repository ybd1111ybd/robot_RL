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
  - Playback-only gripper close override: `--force_gripper_close`, `--gripper_close_value <float>`, `--gripper_close_ramp_steps <int>`. This keeps checkpoint arm actions and overwrites Grasp action channels 14-17 for visual validation only.
  - The debug TCP axes use the configured TCP helper on `left_arm_link7/right_arm_link7`. The current link7-local offsets are left `(0.0385, 0.23772, 0.0)` and right `(0.0385, -0.23772, 0.0)`, moved `0.03m` farther toward the corresponding bottle on 2026-07-13. For Grasp object targets, debug target axes use the object root plus `--debug_grasp_target_offset`.
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
  - gripper rewards remain disabled for approach-only learning, but gripper actions keep a nonzero default-pose-offset scale for numeric joint validation and later closure experiments
- It keeps observation and action dimensions compatible with the existing TwoTarget checkpoint, so it can resume from `grasp_approach3d_twotarget_256env_2000it_v1/nn/jz_bi_grasp.pth`.
- Gripper actions are no longer zero-scaled in the current config. They use default-pose-offset joint-position actions, so zero action holds the open reset pose and nonzero gripper actions can move the gripper joints and the visible moving finger meshes while preserving the 18-D action layout.
- For visual checkpoint validation only, `play.py --start_gripper_closed` changes the robot's gripper reset/default pose to narrow `-0.0501` and wide `-0.0499` before creating the environment. This does not alter the training task's open reset pose, observation layout, or 18-D action contract.
- Playback diagnostics can use `--debug_colliders` to show all PhysX collision shapes and `--print_gripper_diagnostics --diagnostic_every <steps>` to print both gripper joint pairs, four filtered fingertip forces, and both object positions.

## TwoTarget Track 3D Approach Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Track-Play-v0`
- Purpose: improve target-reaching accuracy before adding stabilization or gripper closure.
- Main differences from TwoTarget:
  - left/right direct target-distance penalty is strengthened from `-2.0` to `-3.0`
  - left/right fine tanh tracking weight is strengthened from `0.5` to `1.0`
  - fine tracking `std` is tightened from `0.08` to `0.05`
- It intentionally does not add the Stable variant's extra near-target speed/action-rate penalties.

## TwoTarget Dynamic 3D Contact Validation Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Dynamic-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Dynamic-Play-v0`
- Purpose: preserve the Track checkpoint interface while switching both target cuboids from fixed visual targets to dynamic rigid bodies for contact validation.
- Main differences from Track:
  - `object` and `right_object` use `kinematic_enabled=False`
  - `object` and `right_object` use `disable_gravity=False`
  - object damping is kept at `linear_damping=2.0`, `angular_damping=2.0`
- It keeps the 18-D action layout and 66-D policy observation layout, so the existing bottle-center TwoTarget checkpoint can be played against dynamic objects before adding true closure/lift rewards.
- Current good bottle-center checkpoint preserved outside the RL-Games run directory:
  `/workspace/saved_weights/grasp_twotarget_bottlecenter_good_20260707/jz_bi_grasp.pth`

## TwoTarget Open-Gripper 6DOF Approach Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Open6D-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-Open6D-Play-v0`
- Contract: 66-D policy observation, 18-D action, fixed/kinematic cylinders, and zero-scaled gripper actions holding the open reset pose.
- Per-side orientation reward: `exp(-(distance/0.20)^2) * 0.5 * (clamp(dot(TCP_Z, object_Z), 0, 1) + clamp(dot(TCP_Y, planar_direction_to_object), 0, 1))`, weighted by `+0.5` for each arm.
- The planar direction is the TCP-to-cylinder-center vector projected perpendicular to object Z; within `0.03m`, it falls back to inward world directions `(0,-1,0)` left and `(0,+1,0)` right.
- Run `grasp_open6d_axes_256env_300it_v1` completed 300 additional epochs from the bottle-center checkpoint on physical GPU 0. It preserved approach accuracy but failed bilateral posture learning: deterministic 64-env evaluation of the training-best checkpoint produced last-50 left/right Z dots `-0.750/0.941` and Y dots `-0.003/0.799`.
- This run is diagnostic only. Do not promote it to closure/contact training; first correct and validate the left-side orientation learning signal.

## TwoTarget Surface Pre-Grasp Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregrasp-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregrasp-Play-v0`
- Contract: 66-D policy observation, 18-D action, fixed `0.03m` radius by `0.12m` cylinders, and zero-scaled gripper actions holding the open pose.
- Active per-side task rewards are inner collision-surface clearance `+1.0` with desired gap `0.015m`, multiplicative 3D grasp centering `+1.0`, side-corrected semantic orientation `+0.5` (`-TCP_Y` left, `+TCP_Y` right), and any filtered contact above `1N` at `-0.5`.
- It uses ray-cast narrow3/wide3 inner-surface local points `(0.0214, -0.01704202, 0)` and `(0.0214, 0.01789719, 0)`. TCP-to-cylinder-center progress/tracking rewards are disabled.
- Run `grasp_surface_pregrasp_256env_300it_v1` completed through epoch 600 on GPU 0 but failed geometric acceptance. Epoch 550 is the safer diagnostic checkpoint; epoch 600 learned sustained right-finger collision and must not be used.
- Playback can add `--debug_fingertip_contact_points` to show the exact reward points as 3mm red narrow/green wide markers. Current epoch-550 video: `jz_isaac_lab/logs/rl_games/jz_bi_grasp/grasp_surface_pregrasp_256env_300it_v1/videos/play/rl-video-step-0.mp4`.
  Host path:
  `/mnt/data2/ybd/robot_simulator/saved_weights/grasp_twotarget_bottlecenter_good_20260707/jz_bi_grasp.pth`

## TwoTarget Surface Pre-Grasp V3 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV3-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV3-Play-v0`
- Contract: 96-D policy observation, 18-D action, fixed cylinders, calibrated inner-fingertip contact points, and zero-scaled open-gripper actions.
- V3 is independent of V1/V2 and trains from scratch; do not load a 66-D checkpoint or continue the failed V2 checkpoint.
- Dense per-side approach shaping uses the generic position TCP (current link7-local offsets left `(0.0385, 0.23772, 0)` and right `(0.0385, -0.23772, 0)`) at distance weight `-2.0`, Gaussian tracking with `std=0.25m` at `+1.0`, and signed/clamped step progress at `+2.0`. Fine geometry and contact terms still use calibrated narrow3/wide3 inner-surface points.
- Z/approach-axis pose terms are gated by `exp(-(distance/0.15)^2)`. Clearance, gap balance, between-finger, horizontal, vertical, and level terms are gated by `exp(-(distance/0.08)^2)`.
- Contact above `1N` is penalized at `-1.0` but does not terminate V3 episodes. The 50/30/15mm desired-gap curriculum remains active.
- Required progression is `64 env x 20 epochs` smoke, then `256 x 100` probe, then `256 x 300` only after numeric acceptance. Total return alone is not acceptance evidence; evaluate midpoint distance, 8cm hit rate, surface gaps, center/height error, axis dots, and early-contact ratio.

## TwoTarget Surface Pre-Grasp V4 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV4-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV4-Play-v0`
- Contract: 96-D policy observation, 18-D action, fixed cylinders, and zero-scaled open grippers. Train from scratch; earlier V3 checkpoints use a different approach objective.
- The updated generic TCP tracks a `0.03m` radial stand-off from the cylinder Z axis with broad/fine widths `0.20m/0.04m`. Bottle-center height is tracked independently with widths `0.05m/0.02m`.
- Radial progress is signed and clipped, and cannot be positive during real fingertip contact. Within a `0.06m` radial-error gate, radial speed and total TCP/object relative speed are penalized at `-0.5/-0.2`.
- Pose and fine-geometry proximity gates are `0.30m/0.15m`. Contact above `1N` is penalized at `-3.0`; a 15-step persistent-contact term ramps to an additional `-2.0`. Contact does not terminate the episode.
- A no-contact hold within `0.02m` radial error, `0.025m` height error, and `0.05m/s` relative speed ramps to `+2.0` over 15 steps.
- Evaluator acceptance fields include `left/right_tcp_axis_radial_distance`, `radial_error`, `height_error`, `radial_speed`, `relative_speed`, plus existing surface geometry and contact metrics.

## TwoTarget Surface Pre-Grasp V4.1 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV41-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV41-Play-v0`
- V4 remains unchanged. V4.1 retains its `96-D/18-D` interface and uses V4's unclipped policy-action behavior (`clip_actions: 100.0` in the wrapper and RL-Games policy clipping disabled).
- The TCP height target is `0.03m` above bottle center with broad/fine widths `0.08m/0.025m`; direct radial and height absolute-error penalties have weights `-1.0/-2.0`.
- Near-target radial and relative-speed penalties are strengthened to `-1.0/-0.5`. Stable hold requires `0.04m/s` or less for 20 steps, and near-object action-rate weight is `-0.003`.
- Pose shaping remains gated at `0.30m`, with per-side corrected Z-axis and approach-axis weights raised from V4's `0.75/0.5` to `1.5/1.0`.
- V4.1 adds no table ContactSensors, table-contact rewards, table-clearance rewards, or table-dependent stable-hold conditions. It still inherits V4's pre-existing generic-TCP clearance term unchanged.

## TwoTarget Surface Pre-Grasp V4.2 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV42-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV42-Play-v0`
- V4.2 inherits the current unclipped V4.1 contract but moves the TCP radial stop target from `0.03m` to `0.05m`, leaving `0.02m` outside the cylinder surface before control overshoot.
- A scheduled relative-speed excess penalty at `-5.0` applies only inside `0.10m` radial error. Allowed speeds are `0.15m/s` at `0.05-0.10m`, `0.08m/s` at `0.02-0.05m`, and `0.03m/s` inside `0.02m`.
- Crossing inside the `0.05m` target while continuing inward is penalized linearly at `-10.0`. Stable hold requires radial error within `0.015m` and relative speed at most `0.03m/s`.
- V4.2 adds no table-specific sensor or reward behavior. V4 and V4.1 remain unchanged for comparison.

## TwoTarget Surface Pre-Grasp V4.3 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV43-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV43-Play-v0`
- V4.3 branches from unclipped V4.1 rather than V4.2 and uses a `0.05m` radial stop target. V4/V4.1/V4.2 remain unchanged.
- Target radial velocity is `clamp(-1.5 * effective_error, -0.15, 0.08)m/s`, where effective error has a `0.005m` deadband. Absolute signed radial-velocity error has initial weight `-1.0`, so stopping or moving outward while far from the target remains penalized.
- Non-radial relative speed is penalized linearly at `-0.5` through a linear gate over the final `0.05m`. Continued inward motion after crossing the stop radius is penalized at `-5.0`.
- V4/V4.1 squared radial and relative-speed terms are disabled in V4.3. Stable no-contact hold rises to `+3.0` and requires radial error within `0.015m` and total relative speed at most `0.03m/s`.
- V4.3 adds no table-specific sensor or reward behavior and retains the 96-D observation and 18-D action contract.

## TwoTarget Surface Pre-Grasp V4.4 Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV44-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-SurfacePregraspV44-Play-v0`
- V4.4 inherits V4.3 linear velocity control and the `0.05m` radial target. It adds no table-specific shaping and does not alter the 96-D observation or 18-D action interface.
- Per side, individual filtered ContactSensors cover all six `gripper_(narrow|wide)[1-3]_link` bodies and both `arm_link9/10` palm/wrist bodies. Forces are aggregated in rewards/evaluation and filtered only against the assigned cylinder at a `0.5N` threshold; distal tip sensors remain individually visible for diagnostics.
- A normalized static safety barrier is zero at or outside `0.05m`, reaches `-1.5` at `0.04m`, and saturates at `-3.0` at or inside the `0.03m` bottle radius. Inside-target linear radial velocity error is multiplied by `3.0`.
- Any full-gripper contact is `-3.0`, palm contact adds `-2.0`, and persistent full-gripper contact ramps to `-2.0` over 15 steps. Contact does not terminate the episode.
- Stable hold is `+3.0` only within `0.005m` radial error, `0.02m` height error, `0.03m/s` relative speed, and no tip/full-finger/palm contact. Height fine tracking rises to `+1.0` and direct height error to `-3.0`.
- Evaluator fields add per-side `full_finger_force`, `palm_force`, and `full_gripper_contact_ratio`.

## TwoTarget Minimal Gripper Closure Task
- Training task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-GraspClose-v0`
- Preferred playback task: `Isaac-Grasp-JZ-Bi-Approach-3D-TwoTarget-GraspClose-Play-v0`
- Purpose: learn approach, near-object closure, real bilateral fingertip contact, and short stable-contact dwell before any lift curriculum.
- Interface remains compatible with the bottle-center approach checkpoint:
  - policy observation: `66`
  - action: `18` (`7 + 7` arm and `2 + 2` gripper)
  - contact forces are reward-only and are not added to policy observations
- Both targets are dynamic cylinders with radius `0.03m`, height `0.12m`, mass `0.20kg`, gravity enabled, and filtered ContactSensors on each distal `narrow3/wide3` finger link.
- Added reward weights per arm:
  - close while farther than `0.10m`: `-0.3`
  - close within `0.07m` while relative speed is at most `0.04m/s`: `+0.8`
  - real simultaneous narrow/wide fingertip contact: `+2.0`
  - bilateral contact dwell for 15 steps while relative speed is at most `0.05m/s`: `+1.0`
- Lift rewards are disabled in this task. Do not treat object motion or rising total reward alone as grasp success; require bilateral contact and playback validation.
- Symmetric right-handed semantic TCP/target frames use `wxyz` quaternions left `(0.70710678, 0, 0, 0.70710678)` and right `(0.70710678, 0, 0, -0.70710678)`. On the right, blue Z is world-up, red X points outward (`-world Y`), and green Y points forward (`+world X`); the left frame is rotated 180 degrees around world Z. Side-specific fixed offsets convert the asymmetric URDF `link9` orientations into these semantic TCP frames. Cylinder geometry remains upright.

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
