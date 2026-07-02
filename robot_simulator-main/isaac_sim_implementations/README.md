# Isaac Sim Implementations

这个目录是 Isaac Sim 相关实现的统一入口层。

当前只维护并默认使用：

- `action_graph_method/`

## 当前主线

- `ros2_wrapper_method` 已下线
- 只维护 Action Graph + ROS2 topic I/O + builtin IK 主线
- UI 模式仍然是当前优先验证方式

## 目录结构

```text
isaac_sim_implementations/
├── README.md
└── action_graph_method/
    ├── README.md
    ├── config/
    ├── runtime/
    ├── builtin_ik_solver/
    ├── platform/
    ├── diagnostics/
    ├── docs/
    └── scripts/
```

## 入口关系

- Windows 启动入口：
  - `action_graph_method/run_with_isaac_fixed.bat`
- Python 稳定入口：
  - `action_graph_method/jinzhi_ros2_action_graph.py`
- 默认运行时：
  - `action_graph_method/runtime/lula_action_graph_min.py`
- RViz2 bridge：
  - `/mnt/e/jz_robot/jz_descripetion/robot_urdf/launch/isaac_rviz2_bridge.launch.py`

## 快速开始

WSL：

```bash
source /mnt/e/jz_robot/env.sh
```

Windows：

```powershell
cd E:\jz_robot\robot_simulator\isaac_sim_implementations\action_graph_method
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --hold --control-mode auto --domain-id 77
```

最小链路烟测：

```bash
python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/action_graph_topic_io_check.py \
  --groups left,right,body \
  --duration 5 \
  --publish-rate 20
```

## 文档入口

- `action_graph_method/README.md`
- `action_graph_method/docs/README.md`
- `action_graph_method/diagnostics/README.md`
