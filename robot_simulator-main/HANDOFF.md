# Handoff: JZ Robot Simulation / RL Setup

## Current Goal

The user wants to run reinforcement learning for the JZ dual-arm robot with
Isaac Sim in Docker, so the environment can be migrated and packaged later.

The selected Isaac Sim version is **5.1.0**.

## Repository Layout

Current working tree:

```text
/home/ybd/robot_simulator-main/
├── robot_simulator-main/          # main simulator/control repo
├── Jz_descripetion-main/          # robot description/assets repo
└── jz_isaac_lab/                  # placeholder for Isaac Lab/RL task project
```

Important paths:

```text
robot simulator:
/home/ybd/robot_simulator-main/robot_simulator-main

robot MJCF:
/home/ybd/robot_simulator-main/Jz_descripetion-main/Jz_descripetion-main/robot_urdf/urdf/robot_model.mjcf.xml

robot assets root:
/home/ybd/robot_simulator-main/Jz_descripetion-main/Jz_descripetion-main

Docker Isaac setup:
/home/ybd/robot_simulator-main/robot_simulator-main/docker/isaac
```

## What Was Found

The original `robot_simulator-main` repo did not include robot assets. The user
later added:

```text
/home/ybd/robot_simulator-main/Jz_descripetion-main
```

That asset package contains:

- `robot_model.mjcf.xml`
- one URDF file
- visual/collision STL meshes

The MJCF was tested with MuJoCo and loads successfully.

MuJoCo model summary:

```text
nq=19
nv=19
nu=19
njnt=19
nbody=31
ngeom=61
nkey=2
keyframes: home, teleop_home
sites: left_ee, right_ee
```

Joint and actuator names match the existing `mujoco_sim_node` expectations:

```text
body3 body2 body1 hand2 hand1
left_arm1 ... left_arm7
right_arm1 ... right_arm7
```

## Changes Already Made

### 1. Minimal MuJoCo RL Environment

Added:

```text
rl/__init__.py
rl/jz_mujoco_reach_env.py
rl/smoke_reach_env.py
rl/README.md
```

This is a lightweight Gymnasium-style reach environment that does not require
ROS 2, Isaac Sim, or Isaac Lab.

Contract:

```text
action: 14 values in [-1, 1]
observation: 46 values
task: left/right end-effector reach to sampled targets
reward: negative reach distance + small control penalty + success bonus
```

Verified command:

```bash
cd /home/ybd/robot_simulator-main/robot_simulator-main
python3 -m rl.smoke_reach_env --episodes 2 --steps 50
```

Observed result:

```text
episode=0 obs_shape=(46,) ...
episode=1 obs_shape=(46,) ...
```

This is for fast local reward/action debugging before porting the final task to
Isaac Lab.

### 2. Isaac Sim 5.1 Docker Setup

Added/updated:

```text
docker/isaac/.env
docker/isaac/.env.example
docker/isaac/compose.yaml
docker/isaac/README.md
docker/isaac/setup_host.sh
docker/isaac/preflight_host.sh
docker/isaac/pull_image.sh
docker/isaac/run_shell.sh
docker/isaac/run_headless.sh
docker/isaac/run_python_headless.sh
docker/isaac/compatibility_check.sh
docker/isaac/check_container.sh
```

Docker image is pinned to:

```text
nvcr.io/nvidia/isaac-sim:5.1.0
```

Container paths:

```text
/workspace/robot_simulator
/workspace/Jz_descripetion
/workspace/jz_isaac_lab
```

The compose file runs the official Isaac Sim 5.1 container as user:

```text
1234:1234
```

and uses persistent host cache root:

```text
/home/ybd/docker/isaac-sim-5.1
```

## Docker Status

The user successfully pulled the Isaac Sim 5.1 image:

```text
nvcr.io/nvidia/isaac-sim:5.1.0
Digest: sha256:f3563cb2ba0c18af0b2fb321360dcb73a917b899f879e3213623d6bee484fa54
```

The first compatibility script failed because it ran from `/workspace` while
the Isaac Sim scripts are in `/isaac-sim`. This was fixed by changing:

```text
compatibility_check.sh
run_headless.sh
```

to `cd /isaac-sim` before executing Isaac scripts.

The user then ran the compatibility check successfully enough to get a real
hardware report.

Hardware reported by Isaac Sim:

```text
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
driver: 610.62
VRAM: 8.59 GB
RAM: 16.63 GB
CPU: 12th Gen Intel Core i7-12800HX, 24 cores
OS: Ubuntu 24.04.2 LTS
```

Compatibility result:

```text
System checking result: FAILED
```

Reason:

```text
VRAM minimum: 10 GB, available: 8.59 GB
RAM minimum: 32 GB, available: 16.63 GB
```

GPU and driver are supported. The problem is resource capacity, not a missing
driver.

Implication:

- Do not run full UI.
- Avoid livestream unless required.
- Use headless mode.
- Start Isaac Lab with `--num_envs 1`.
- Increase gradually only if memory allows.
- The previous `--num_envs 64` command from the user's RL document is too heavy
  for this laptop.

## Useful Commands

Host setup:

```bash
cd /home/ybd/robot_simulator-main/robot_simulator-main/docker/isaac
bash setup_host.sh
bash preflight_host.sh
```

Pull Isaac Sim image:

```bash
bash pull_image.sh
```

Run compatibility check:

```bash
bash compatibility_check.sh
```

Enter container:

```bash
bash run_shell.sh
```

Inside container, check mounts and MJCF:

```bash
bash /workspace/robot_simulator/docker/isaac/check_container.sh
```

Run headless Isaac Sim:

```bash
bash run_headless.sh
```

Run a Python script in Isaac Sim container:

```bash
bash run_python_headless.sh /workspace/robot_simulator/isaac_sim_implementations/action_graph_method/jinzhi_ros2_action_graph.py --headless
```

## Migration To Another Machine

Recommended copy command:

```bash
rsync -avh --progress /home/ybd/robot_simulator-main/ USER@TARGET_IP:/home/USER/robot_simulator-main/
```

or:

```bash
scp -r /home/ybd/robot_simulator-main USER@TARGET_IP:/home/USER/
```

Do not copy Docker image layers by hand unless needed. On the target machine,
pull the image again:

```bash
docker pull nvcr.io/nvidia/isaac-sim:5.1.0
```

On the target machine, edit:

```text
robot_simulator-main/robot_simulator-main/docker/isaac/.env
```

so host paths match the target machine. Then run:

```bash
cd /home/USER/robot_simulator-main/robot_simulator-main/docker/isaac
bash setup_host.sh
bash preflight_host.sh
bash compatibility_check.sh
```

## External RL Project Still Missing

The user's RL document references a separate Isaac Lab project:

```text
RL_jzrobot
jz_isaac_lab
task: Isaac-Reach-JZ-Bi-v0
framework: rl_games
```

Current local directory:

```text
/home/ybd/robot_simulator-main/jz_isaac_lab
```

exists but is currently just a placeholder/empty mount.

The current repo does not contain:

```text
Isaac-Reach-JZ-Bi-v0
jz_bi_reach
rl_games training config
Isaac Lab task registration
USD articulation asset config
trained checkpoint
```

Next AI should not search for those inside `robot_simulator-main`; they need to
be added from the separate RL project or recreated.

## Recommended Next Steps

1. Put the actual Isaac Lab RL task project into:

   ```text
   /home/ybd/robot_simulator-main/jz_isaac_lab
   ```

2. Verify inside container:

   ```bash
   bash /workspace/robot_simulator/docker/isaac/check_container.sh
   ls /workspace/jz_isaac_lab
   ```

3. Find task registration for:

   ```text
   Isaac-Reach-JZ-Bi-v0
   ```

4. Fix any asset paths to use container paths:

   ```text
   /workspace/Jz_descripetion
   /workspace/robot_simulator
   /workspace/jz_isaac_lab
   ```

5. Start with low-resource Isaac Lab command:

   ```bash
   --headless --num_envs 1
   ```

6. Only after it runs, try:

   ```bash
   --num_envs 2
   --num_envs 4
   --num_envs 8
   ```

Avoid `--num_envs 64` on the current 8.6 GB VRAM / 16.6 GB RAM laptop.

## Important Caveats

- The current directory `/home/ybd/robot_simulator-main/robot_simulator-main`
  is not a git repository in this environment; `git status` failed there.
- ROS 2 exists on the host, but `rclpy` only imports correctly after:

  ```bash
  source /opt/ros/humble/setup.bash
  ```

- Earlier ROS node launch needed `ROS_LOG_DIR`/`ROS_HOME` moved to `/tmp` in
  the sandbox because `/home/ybd/.ros` was read-only there.
- Docker daemon access and GPU access may differ between Codex sandbox and the
  user's actual terminal. The user's actual terminal already pulled the Isaac
  Sim image and ran compatibility check.

