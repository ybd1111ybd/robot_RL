# Roadmap

## Phase 1: Harness and Environment Stabilization
- Establish AI Harness files and directory structure.
- Keep project state, roadmap, changelog, runtime context, contracts, and decisions in repository files.
- Verify Docker, Isaac Lab, RL-Games, JZ assets, and task registration are reproducible.

## Phase 2: Reach Task Learning Path
- Study `Isaac-Reach-JZ-Bi-v0` task registration, environment configuration, observations, actions, commands, rewards, and PPO configuration.
- Run longer Reach training sessions with stable experiment names.
- Inspect TensorBoard summaries and saved checkpoints.
- Validate checkpoint playback with `play.py`.

## Phase 3: Reward and Curriculum Iteration
- Tune Reach reward terms, command sampling ranges, action scaling, and curriculum behavior.
- Record each meaningful experiment and its outcome in project memory.
- Promote reliable training commands into harness runtime context.

## Phase 4: Grasp and Drawer Tasks
- Move from Reach to Grasp only after Reach training and playback are understood.
- Move from Grasp to Drawer after object interaction and contact behavior are stable.
- Update contracts and context when task interfaces or runtime assumptions change.

## Phase 5: Reproducible Training Workflow
- Document repeatable train, monitor, evaluate, and play commands.
- Define checkpoint naming and experiment logging conventions.
- Preserve validated configurations and major decisions in the harness.
