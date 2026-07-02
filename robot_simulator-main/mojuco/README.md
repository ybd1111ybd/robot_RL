# MuJoCo Package

`mojuco/` 是当前仓库中的 MuJoCo ROS2 包。

## 目录结构

- `mujoco_simulator/`
  - Python 包主实现。
- `launch/`
  - MuJoCo 启动文件。
- `config/`
  - 仿真参数配置。
- `resource/`
  - ROS2 包资源。
- `package.xml`
- `setup.py`
- `setup.cfg`

## 常用命令

构建：

```bash
cd /mnt/e/jz_robot/robot_simulator
colcon build --packages-select mujoco_simulator
source install/setup.bash
```

带界面启动：

```bash
ros2 launch mujoco_simulator mujoco_sim.launch.py
```

无头启动：

```bash
ros2 launch mujoco_simulator mujoco_sim_headless.launch.py
```

## 维护说明

- 这个目录名当前保持 `mojuco/`，是为了兼容现有路径。
- 如果后续要正式更名为 `mujoco/`，需要同步改 package 路径、文档和外部脚本。
