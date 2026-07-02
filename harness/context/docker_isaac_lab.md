# Docker Isaac Lab Context

## Purpose
This document records the reproducible Docker runtime context for JZ Isaac Lab reinforcement learning.

## Host Paths
- Project root: `/home/cqy/workspace/middle_platform/robot_simulator`
- Main simulator repo: `/home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main`
- Docker compose directory: `/home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac`
- JZ robot assets: `/home/cqy/workspace/middle_platform/robot_simulator/Jz_descripetion-main/Jz_descripetion-main`
- Isaac Lab RL project: `/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab`
- Isaac cache root: `/home/cqy/docker/isaac-sim-5.1`

## Container Paths
- RL project: `/workspace/jz_isaac_lab`
- Main simulator repo: `/workspace/robot_simulator`
- JZ robot assets are mounted for compatibility at:
  - `/workspace/Jz_descripetion`
  - `/workspace/jz_descripetion`

## Compose Service
- Service name: `isaac-lab`
- Docker image: `jz-isaaclab:5.1.0`
- Compose command root:
  ```bash
  cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac
  ```

## Required Runtime Environment Variables
Use these inside the container before running JZ Isaac Lab scripts:
```bash
export JZLAB_WORKSPACE_ROOT=/workspace
export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
export HYDRA_FULL_ERROR=1
```

## Known Working Checks
List registered JZ tasks:
```bash
docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/tools/list_envs.py
'
```

## Important Runtime Notes
- Run training through `/isaac-sim/python.sh`, not system `python3`.
- Use headless training first to avoid display/X11 issues.
- Use `+agent.params.config.torch_compile=False` unless the image has a working C/C++ compiler toolchain for PyTorch Inductor.
- The JZ bimanual asset path currently relies on a cached/generated USD path to avoid runtime URDF importer extension download issues.
- If forcing USD rebuild later, check the JZ asset conversion path and URDF importer availability first.

## Logs and Outputs
RL-Games logs are written under:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/rl_games/
```
Inside the container this is:
```text
/workspace/jz_isaac_lab/logs/rl_games/
```
