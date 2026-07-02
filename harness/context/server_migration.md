# Server Migration Guide

## Purpose
This document describes how to migrate the JZ robot simulator and Isaac Lab reinforcement learning environment to another server.

## What Moves With the Project
Copying the project directory moves source code, Docker configuration, robot assets, harness memory, training scripts, task definitions, and any logs/checkpoints stored under the project.

Recommended project directory shape on the target server:
```text
robot_simulator/
├── robot_simulator-main/
├── Jz_descripetion-main/
├── jz_isaac_lab/
├── harness/
├── agent.md
├── PROJECT_STATE.md
├── ROADMAP.md
└── CHANGELOG.md
```

## What Does Not Move Automatically
Docker images are not stored inside the project directory. They are stored by the Docker engine, usually under `/var/lib/docker/`.

The project contains the Docker build and compose configuration, but not the built image itself.

Current training image:
```text
jz-isaaclab:5.1.0
```

## Target Server Requirements
The target server should have:
- NVIDIA GPU.
- Compatible NVIDIA driver.
- Docker.
- Docker Compose plugin.
- NVIDIA Container Toolkit.
- Enough disk space for Isaac Sim, Isaac Lab, Docker images, caches, logs, and checkpoints.

Basic GPU check:
```bash
nvidia-smi
```

Basic Docker GPU check:
```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

If Docker cannot access the GPU, Isaac Lab training will not run correctly.

## Files and Directories to Copy
At minimum, copy:
```text
/home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main
/home/cqy/workspace/middle_platform/robot_simulator/Jz_descripetion-main
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab
/home/cqy/workspace/middle_platform/robot_simulator/harness
/home/cqy/workspace/middle_platform/robot_simulator/agent.md
/home/cqy/workspace/middle_platform/robot_simulator/PROJECT_STATE.md
/home/cqy/workspace/middle_platform/robot_simulator/ROADMAP.md
/home/cqy/workspace/middle_platform/robot_simulator/CHANGELOG.md
```

To preserve trained models and experiment history, keep:
```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab/logs/
```

The most important training outputs are under:
```text
jz_isaac_lab/logs/rl_games/
```

## Docker Image Migration Options

### Option A: Rebuild on the Target Server
Use this when the target server has reliable network access to the required base image and Python packages.

On the target server:
```bash
cd /path/to/robot_simulator/robot_simulator-main/docker/isaac
bash build_isaaclab_image.sh
```

Or, if using Compose build directly:
```bash
docker compose --env-file .env build isaac-lab
```

This is usually the cleanest method.

### Option B: Export and Import the Existing Image
Use this when the target server has poor network access or you need the exact current image.

On the source server:
```bash
docker save jz-isaaclab:5.1.0 -o jz-isaaclab-5.1.0.tar
```

Copy the tar file to the target server, then load it:
```bash
docker load -i jz-isaaclab-5.1.0.tar
```

Verify:
```bash
docker images | grep jz-isaaclab
```

The image tar can be large. Plan for transfer time and disk space.

## Update Target Server Paths
After copying the project, edit:
```text
robot_simulator-main/docker/isaac/.env
```

Update host-specific paths such as:
```text
ROBOT_SIMULATOR_ROOT
JZ_DESCRIPTION_ROOT
JZ_ISAAC_LAB_ROOT
ISAAC_CACHE_ROOT
ISAAC_LAB_IMAGE
```

Keep container paths stable when possible:
```text
/workspace/jz_isaac_lab
/workspace/jz_descripetion
/workspace/Jz_descripetion
```

Stable container paths reduce the chance of breaking training scripts and task code.

## Optional Cache Migration
Isaac cache can be regenerated, but migrating it can reduce first-start delay.

Current cache root:
```text
/home/cqy/docker/isaac-sim-5.1
```

If the cache is not copied, expect the first run on the target server to take longer.

## Post-Migration Verification

### 1. Check Mounted Project Paths
```bash
cd /path/to/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  ls -la /workspace/jz_isaac_lab
  ls -la /workspace/jz_descripetion
  ls -la /workspace/Jz_descripetion
'
```

### 2. Check Registered Tasks
```bash
cd /path/to/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/tools/list_envs.py
'
```

Expected task IDs include:
```text
Isaac-Reach-JZ-Bi-v0
Isaac-Reach-JZ-Bi-Play-v0
Isaac-Grasp-JZ-Bi-v0
Isaac-Grasp-JZ-Bi-Play-v0
Isaac-Open-Drawer-JZ-Bi-v0
Isaac-Open-Drawer-JZ-Bi-Play-v0
```

### 3. Run Smoke Training
```bash
cd /path/to/robot_simulator/robot_simulator-main/docker/isaac

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
    +agent.params.config.full_experiment_name=migration_smoke_32x5 \
    agent.params.config.save_frequency=1 \
    agent.params.config.save_best_after=0 \
    +agent.params.config.torch_compile=False
'
```

The migration is considered basically successful if task listing works and the smoke training completes.

## Common Failure Points
- `nvidia-smi` fails on the host: fix NVIDIA driver first.
- Docker GPU test fails: install or repair NVIDIA Container Toolkit.
- Container cannot see `/workspace/jz_isaac_lab`: fix `.env` host paths and Compose mounts.
- JZ robot assets are missing: verify both description mounts and mesh/URDF files.
- Hydra override errors: existing config keys use no `+`; new keys use `+`.
- PyTorch Inductor compiler error: keep `+agent.params.config.torch_compile=False` or install a compiler toolchain in the image.
- First run is slow: Isaac cache may be missing and regenerated.

## Recommended Migration Strategy
Use Git or `rsync` for project files and logs. Rebuild the Docker image on the target server when network access is reliable. Use `docker save` and `docker load` only when exact image reproduction or offline migration is needed.
