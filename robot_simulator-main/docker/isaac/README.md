# Docker Isaac Sim Deployment

This directory keeps the Isaac Sim 5.1.0 container setup portable. It mounts
the current `robot_simulator` code, the JZ robot description package, and an
optional Isaac Lab task workspace into stable container paths.

## Host Requirements

- Linux host with an NVIDIA GPU.
- Recent NVIDIA driver.
- Docker Engine and Docker Compose.
- NVIDIA Container Toolkit.
- NGC access for `nvcr.io/nvidia/isaac-sim:5.1.0`.

Official references:

- Isaac Sim container installation:
  <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html>
- Isaac Lab Docker deployment:
  <https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html>

## Configure Host Directories

From this directory:

```bash
bash setup_host.sh
```

This creates `.env` if missing, creates persistent cache directories, and fixes
permissions for the Isaac Sim 5.1 rootless container user `1234:1234`.

Edit `.env` if the host machine uses different paths. The current local model
path is:

```text
/home/cqy/workspace/middle_platform/robot_simulator/Jz_descripetion-main/Jz_descripetion-main/robot_urdf/urdf/robot_model.mjcf.xml
```

## Pull Image

Login to NGC first if required, then:

```bash
bash pull_image.sh
```

## Start A Shell

```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac
bash run_shell.sh
```

Inside the container, expected paths are:

```text
/workspace/robot_simulator
/workspace/Jz_descripetion
/workspace/jz_isaac_lab
```

## Container Check

Inside the container:

```bash
bash /workspace/robot_simulator/docker/isaac/check_container.sh
```

This verifies that the robot model mount exists and that the image has the
expected Isaac Sim Python entrypoint.

## Build Isaac Lab Training Image

The official Isaac Sim image does not include Isaac Lab or `rl_games`. Build the
local training image before running RL commands:

```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac
bash build_isaaclab_image.sh
```

This creates `${ISAAC_LAB_IMAGE}` from `.env`, defaulting to:

```text
jz-isaaclab:5.1.0
```

The Dockerfile installs `isaaclab==2.3.2.post1` and `rl_games==1.6.5` while
constraining PyTorch/Torchvision to the versions bundled with Isaac Sim 5.1.0.

Check the training container:

```bash
docker compose --env-file .env run --rm isaac-lab \
  -lc 'bash /workspace/robot_simulator/docker/isaac/check_isaaclab_container.sh'
```

Start an Isaac Lab shell:

```bash
bash run_isaaclab_shell.sh
```

Inside the training container, expected paths are still:

```text
/workspace/robot_simulator
/workspace/Jz_descripetion
/workspace/jz_isaac_lab
```

Put or clone the actual JZ Isaac Lab task project into the host directory:

```text
/home/cqy/workspace/middle_platform/robot_simulator/jz_isaac_lab
```

Then run that project's training command from inside the `isaac-lab` container.
For this laptop, start with:

```bash
--headless --num_envs 1
```

## Compatibility Check

On the host:

```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac
bash compatibility_check.sh
```

On an 8 GB laptop GPU, this may fail the recommended VRAM/RAM threshold even
when the driver and GPU are usable. That means keep runs headless and small:
start with `--num_envs 1`, then increase gradually.

## Headless Isaac Sim

On the host:

```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac
bash run_headless.sh
```

For script execution:

```bash
bash run_python_headless.sh /workspace/robot_simulator/isaac_sim_implementations/action_graph_method/jinzhi_ros2_action_graph.py --headless
```

## Headless Isaac Sim Script Pattern

Use the Isaac Sim Python wrapper inside the container:

```bash
/isaac-sim/python.sh /workspace/robot_simulator/isaac_sim_implementations/action_graph_method/jinzhi_ros2_action_graph.py --headless
```

For Isaac Lab training, mount or clone the task project into
`/workspace/jz_isaac_lab`, then run its training command from the container
using the Isaac Lab launcher used by that project.

## Notes

- The compose file uses `runtime: nvidia`, `network_mode: host`, `ipc: host`,
  and user `1234:1234`, matching Isaac Sim 5.1.0 container deployment.
- Keep host paths out of task code. Use the container paths above so the setup
  can move between machines.
- The current project-local MuJoCo RL environment under `rl/` remains useful
  for fast reward/action debugging before porting the final task to Isaac Lab.
- This environment pins Isaac Sim to `5.1.0` and the local training image uses
  Isaac Lab `2.3.2.post1`.
- The current checked machine has an RTX 4070 Laptop GPU with about 8.6 GB VRAM
  and about 16.6 GB RAM, below Isaac Sim 5.1's recommended 10 GB VRAM / 32 GB
  RAM. Use headless mode, avoid livestream/UI, and keep Isaac Lab `num_envs`
  small.
