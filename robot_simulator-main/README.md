# Robot Simulator

这个仓库是仿真代码仓，不是工作区总入口。

当前维护两条路径：

- `mojuco/`
  - MuJoCo ROS2 包。
- `isaac_sim_implementations/action_graph_method/`
  - 当前 Isaac Sim Action Graph 主线。

说明：

- 目录名 `mojuco/` 保持现状以兼容已有路径，不在这次重构中改名。
- 工作区统一入口仍然是 `/mnt/e/jz_robot/env.sh`。
- 当前 WSL 工作区路径为 `/home/czy/workspace/robot_ws` 时，优先使用本文的
  “本机 WSL 快速启动”命令。

## 仓库结构

```text
robot_simulator/
├── README.md
├── mojuco/
│   ├── README.md
│   ├── mujoco_simulator/
│   ├── launch/
│   ├── config/
│   ├── resource/
│   ├── package.xml
│   └── setup.py
└── isaac_sim_implementations/
    ├── README.md
    └── action_graph_method/
```

## 当前主线

### Isaac Sim

- 只维护 `isaac_sim_implementations/action_graph_method`
- UI + ROS2 topic I/O 是当前主验证方式
- RViz2 bridge 入口在描述仓：
  - `/mnt/e/jz_robot/jz_descripetion/robot_urdf/launch/isaac_rviz2_bridge.launch.py`

### MuJoCo

- `mojuco/` 仍作为独立 ROS2 包保留
- 适合做 MuJoCo 对照验证和独立仿真

## 本机 WSL 快速启动

当前机器的工作区路径：

```bash
/home/czy/workspace/robot_ws
```

先构建：

```bash
cd /home/czy/workspace/robot_ws
colcon build --packages-select mujoco_simulator standalone_ik_solver
source install/setup.bash
```

### 推荐：无界面启动

WSL 下优先使用无界面模式，避免 MuJoCo viewer 的 OpenGL 渲染把系统卡住：

```bash
cd /home/czy/workspace/robot_ws
source install/setup.bash

ros2 launch mujoco_simulator mujoco_sim_headless.launch.py \
  model_path:=/home/czy/workspace/robot_ws/src/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml
```

说明：

- `mujoco_sim_headless.launch.py` 不会弹可视化窗口。
- 仿真仍然会运行，ROS2 topic 仍然可用。
- 这是 WSL 下最稳的启动方式。

### 带 MuJoCo 可视化启动

只有确认 WSLg / OpenGL / 显卡加速正常时，再使用带 viewer 的启动：

```bash
cd /home/czy/workspace/robot_ws
source install/setup.bash

ros2 launch mujoco_simulator mujoco_sim.launch.py \
  model_path:=/home/czy/workspace/robot_ws/src/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml
```

如果笔记本有 NVIDIA 独显，WSL 默认可能会选择 Intel 核显。可以临时强制
Mesa D3D12 后端选择 NVIDIA：

```bash
cd /home/czy/workspace/robot_ws
source install/setup.bash

MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
ros2 launch mujoco_simulator mujoco_sim.launch.py \
  viewer_rate:=60.0 \
  model_path:=/home/czy/workspace/robot_ws/src/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml
```

确认是否已经走 NVIDIA：

```bash
MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
glxinfo | grep -E "OpenGL vendor|OpenGL renderer|OpenGL version"
```

期望看到类似：

```text
OpenGL renderer string: D3D12 (NVIDIA RTX 1000 Ada Generation Laptop GPU)
```

如果确认有效，可以写入 shell 配置，后续终端默认使用 NVIDIA：

```bash
echo 'export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA' >> ~/.bashrc
source ~/.bashrc
```

注意：

- `mujoco_sim.launch.py` 会尝试打开 MuJoCo viewer。
- `viewer_rate:=60.0` 会把 MuJoCo viewer 同步限制到 60Hz，物理仿真仍然按
  `sim_rate` 运行，避免在 WSLg 中每个仿真步都刷新窗口。
- 在 WSL 中，如果 OpenGL 没有使用 GPU 加速，可能退化成 CPU 软件渲染，
  机器人模型网格较多时会导致系统明显卡顿，甚至看起来像卡死。
- 如果默认选择了 Intel 核显，`OpenGL renderer` 会显示
  `D3D12 (Intel(R) Graphics)`；使用上面的环境变量后应切到 NVIDIA。
- 如果已经出现卡死，优先关闭该终端或杀掉 `ros2` / `mujoco_sim_node`
  进程，然后改用 headless 模式。

检查 WSL 图形环境：

```bash
echo $DISPLAY
echo $WAYLAND_DISPLAY
glxinfo | grep -E "OpenGL vendor|OpenGL renderer|OpenGL version"
```

如果 `OpenGL renderer` 里看到 `llvmpipe`、`software rasterizer` 等字样，说明很可能在用
CPU 软件渲染，不建议直接开 MuJoCo viewer。

### 检查仿真话题

另开一个终端：

```bash
cd /home/czy/workspace/robot_ws
source install/setup.bash
ros2 topic list
```

常见输出应包含：

```text
/arm_left/joint_states
/arm_right/joint_states
/body/joint_states
/arm_left/joint_commands
/arm_right/joint_commands
/body/joint_commands
```

查看左臂状态：

```bash
ros2 topic echo /arm_left/joint_states
```

### 发送关节命令

节点启动后前 10 秒会忽略关节命令，用于等待 MuJoCo 初始化完成。看到
`Initialization complete - accepting joint commands` 后再发命令。

左臂示例：

```bash
ros2 topic pub --once /arm_left/joint_commands sensor_msgs/msg/JointState "
name: ['left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6', 'left_joint7']
position: [0.0, -1.2, 0.0, -1.3, 1.2, 0.0, 0.0]
"
```

右臂示例：

```bash
ros2 topic pub --once /arm_right/joint_commands sensor_msgs/msg/JointState "
name: ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_joint7']
position: [0.0, 1.2, 0.0, 1.3, -1.2, 0.0, 0.0]
"
```

身体关节示例：

```bash
ros2 topic pub --once /body/joint_commands sensor_msgs/msg/JointState "
name: ['body1', 'body2', 'body3', 'hand1', 'hand2']
position: [0.0, 0.0, 0.0, 0.0, 0.0]
"
```

### 模型路径说明

不要直接使用：

```bash
/home/czy/workspace/robot_ws/src/robot_simulator/mojuco/robot_model.mjcf.xml
```

这份 MJCF 会引用 `../meshes/visual/...` 和 `../meshes/collision/...`，但
`robot_simulator/mojuco/` 下没有完整 meshes 目录。

当前工作区里推荐使用：

```bash
/home/czy/workspace/robot_ws/src/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml
```

它配套的 mesh 目录在：

```bash
/home/czy/workspace/robot_ws/src/jz_descripetion/robot_urdf/meshes
```

## 常用命令

构建 MuJoCo 包：

```bash
cd /mnt/e/jz_robot/robot_simulator
colcon build --packages-select mujoco_simulator
source install/setup.bash
```

启动 MuJoCo：

```bash
ros2 launch mujoco_simulator mujoco_sim_headless.launch.py
```

启动 Isaac Sim Action Graph：

```powershell
cd E:\jz_robot\robot_simulator\isaac_sim_implementations\action_graph_method
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --hold --control-mode auto --domain-id 77
```

## 文档入口

- `mojuco/README.md`
- `isaac_sim_implementations/README.md`
- `isaac_sim_implementations/action_graph_method/README.md`

- **降低仿真频率**：如果 CPU 使用率过高，可以降低 `sim_rate`
- **降低发布频率**：降低 `pub_rate` 可以减少网络带宽占用
- **禁用可视化**：生产环境中使用 `enable_viewer: false`

### 许可证

Apache License 2.0

---

<a name="english"></a>
## English Documentation

### Overview

A high-fidelity dual-arm robot simulation system based on MuJoCo physics engine, providing ROS2 interface for VR teleoperation. The system offers real-time physics simulation, state feedback, and multiple control modes.

### Key Features

- **High-Fidelity Physics Simulation**: Powered by MuJoCo engine for realistic robot dynamics
- **Native ROS2 Support**: Complete ROS2 interface with standard message types
- **Dual-Arm Control**: Independent control of left arm, right arm, and body joints
- **Multiple Control Modes**:
  - Position control mode
  - Passthrough direct control mode
  - Global speed adjustment
- **Real-time State Feedback**:
  - Joint positions, velocities, and efforts
  - End-effector poses
  - Movement completion status
- **Visualization**: Optional MuJoCo viewer window for real-time simulation display
- **Service Interface**: Robot enable/disable, mode switching services

### System Architecture

See the directory structure in the Chinese section above.

### Installation

#### Requirements

- Ubuntu 22.04 or later
- ROS2 Humble or later
- Python 3.8+

#### Installation Steps

1. **Install ROS2 dependencies**
```bash
sudo apt update
sudo apt install ros-humble-sensor-msgs ros-humble-geometry-msgs \
                 ros-humble-std-msgs ros-humble-std-srvs \
                 ros-humble-tf2 ros-humble-tf2-ros
```

2. **Install Python dependencies**
```bash
pip install mujoco numpy
```

3. **Build workspace**
```bash
cd /mnt/e/jz_robot/robot_simulator
colcon build --packages-select mujoco_simulator
source install/setup.bash
```

### Usage

#### Launch Simulator

**With visualization:**
```bash
ros2 launch mujoco_simulator mujoco_sim.launch.py
```

**Headless mode:**
```bash
ros2 launch mujoco_simulator mujoco_sim_headless.launch.py
```

#### Custom Parameters

```bash
ros2 launch mujoco_simulator mujoco_sim.launch.py \
    sim_rate:=100.0 \
    pub_rate:=50.0 \
    enable_viewer:=true \
    position_control_kp:=100.0 \
    position_control_kd:=10.0
```

### ROS2 Interface

#### Subscribed Topics

| Topic | Message Type | Description |
|-------|-------------|-------------|
| `/arm_left/joint_commands` | `sensor_msgs/JointState` | Left arm joint position commands |
| `/arm_right/joint_commands` | `sensor_msgs/JointState` | Right arm joint position commands |
| `/body/joint_commands` | `sensor_msgs/JointState` | Body joint position commands |
| `/arm_left/joint_passthrough` | `sensor_msgs/JointState` | Left arm direct control commands |
| `/arm_right/joint_passthrough` | `sensor_msgs/JointState` | Right arm direct control commands |
| `/body/joint_passthrough` | `sensor_msgs/JointState` | Body direct control commands |
| `/arm_controller/global_speed` | `std_msgs/Int32` | Global speed control (0-100) |

#### Published Topics

| Topic | Message Type | Rate | Description |
|-------|-------------|------|-------------|
| `/arm_left/joint_states` | `sensor_msgs/JointState` | 50Hz | Left arm joint states |
| `/arm_right/joint_states` | `sensor_msgs/JointState` | 50Hz | Right arm joint states |
| `/body/joint_states` | `sensor_msgs/JointState` | 50Hz | Body joint states |
| `/arm_left/world_pose` | `geometry_msgs/PoseStamped` | 50Hz | Left arm end-effector pose |
| `/arm_right/world_pose` | `geometry_msgs/PoseStamped` | 50Hz | Right arm end-effector pose |
| `robot/movement_complete` | `std_msgs/Int32` | Event | Movement completion notification |

#### Services

| Service | Type | Description |
|---------|------|-------------|
| `enable_robot` | `std_srvs/Trigger` | Enable robot control |
| `disable_robot` | `std_srvs/Trigger` | Disable robot control |
| `set_passthrough_mode` | `std_srvs/SetBool` | Set passthrough control mode |

### Joint Definitions

#### Body Joints (5)
- `joint1`, `joint2`, `joint3`, `joint4`, `joint5`

#### Left Arm Joints (7)
- `joint6l`, `joint7l`, `joint8l`, `joint9l`, `joint10l`, `joint11l`, `joint12l`

#### Right Arm Joints (7)
- `joint6r`, `joint7r`, `joint8r`, `joint9r`, `joint10r`, `joint11r`, `joint12r`

### Configuration Parameters

Configuration file: `mojuco/config/sim_params.yaml`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sim_rate` | float | 100.0 | Simulation rate (Hz) |
| `pub_rate` | float | 50.0 | State publishing rate (Hz) |
| `enable_viewer` | bool | true | Enable visualization window |
| `viewer_rate` | float | 60.0 | MuJoCo viewer sync rate (Hz) |
| `position_control_kp` | float | 100.0 | Position control proportional gain |
| `position_control_kd` | float | 10.0 | Position control derivative gain |
| `model_path` | string | "" | Model file path (auto-detect if empty) |

### Examples

#### 1. Control Left Arm

```bash
ros2 topic pub /arm_left/joint_commands sensor_msgs/JointState \
"{name: ['joint6l', 'joint7l', 'joint8l', 'joint9l', 'joint10l', 'joint11l', 'joint12l'], \
  position: [0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0]}"
```

#### 2. Monitor Joint States

```bash
ros2 topic echo /arm_left/joint_states
```

#### 3. Disable Robot

```bash
ros2 service call /disable_robot std_srvs/srv/Trigger
```

#### 4. Enable Passthrough Mode

```bash
ros2 service call /set_passthrough_mode std_srvs/srv/SetBool "{data: true}"
```

### Control Modes

#### Position Control Mode (Default)
- Uses PD controller to track target positions
- Receives target positions via `joint_commands` topics
- Suitable for trajectory tracking and precise positioning

#### Passthrough Mode
- Direct joint position control, bypasses standard control flow
- Must be enabled via service first
- Receives commands via `joint_passthrough` topics
- Suitable for low-latency direct control

### Troubleshooting

#### 1. Model File Not Found
```
Error: Could not find robot_model.mjcf.xml
Solution: Check model_path parameter or place model file in default location
```

#### 2. MuJoCo Not Installed
```
Error: Warning: mujoco not installed
Solution: pip install mujoco
```

#### 3. Viewer Fails to Launch
```
Warning: Failed to launch viewer
Solutions:
- Check if graphical interface is available
- Use headless mode: mujoco_sim_headless.launch.py
- Verify OpenGL drivers are properly installed
```

#### 4. Mesh Files Not Found
```
Error: Error opening file '.../meshes/link0.STL': No such file or directory
Solution:
1. Ensure mesh files are included in setup.py installation
2. Rebuild the package:
   cd /mnt/e/jz_robot/robot_simulator
   colcon build --packages-select mujoco_simulator
3. Source the environment: source install/setup.bash
4. Verify mesh files are installed:
   ls install/mujoco_simulator/share/mujoco_simulator/models/meshes/
```

### Development & Extension

#### Modify Robot Model
1. Edit `robot_model.mjcf.xml` file
2. Update mesh files in `meshes/` directory
3. Rebuild: `colcon build --packages-select mujoco_simulator`

#### Add New Sensors
1. Add sensor definition in MJCF model
2. Add publishers in `mujoco_sim_node.py`
3. Publish sensor data in `_publish_states()` method

### Performance Tuning

- **Lower simulation rate**: Reduce `sim_rate` if CPU usage is too high
- **Lower publishing rate**: Reduce `pub_rate` to save network bandwidth
- **Disable viewer**: Use `enable_viewer: false` in production

### License

Apache License 2.0

### Maintainer

For issues and questions, please contact the maintainer or open an issue in the repository.
