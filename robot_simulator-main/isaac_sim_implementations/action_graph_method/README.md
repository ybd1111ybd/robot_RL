# Action Graph 目录导航

当前目录不要再按“只有一个脚本有用”来理解。更准确的划分是：

- 默认主线
- 运行支撑
- 诊断验收
- 参考保留
- 文档入口

## 日常优先看的文件

- `runtime/lula_action_graph_min.py`
  - 当前默认运行时。
- `builtin_ik_solver/builtin_ik_bridge.py`
  - 当前默认 builtin IK 主线。
- `config/runtime_defaults.toml`
  - 运行时默认参数。
- `config/builtin_ik.yaml`
  - builtin IK 结构化参数。
- `scripts/reset_ik_stack.sh`
  - WSL/Windows 现场清场。
- `diagnostics/action_graph_topic_io_check.py`
  - 最小话题烟测。
- `diagnostics/run_phase4_acceptance.sh`
  - 主线验收快捷入口。
- `docs/指令文件.md`
  - 当前常用操作命令总入口。

## 当前有用文件分类

### 1. 入口与兼容包装

- `run_with_isaac_fixed.bat`
  - Windows 友好入口，实际转发到 `platform/windows/run_with_isaac_fixed.bat`。
- `setup_windows_ros2_fixed.bat`
  - Windows 环境包装入口，实际转发到 `platform/windows/setup_windows_ros2_fixed.bat`。
- `setup_wsl_ros2_fixed.sh`
  - WSL 环境包装入口，实际 source/exec `platform/wsl/setup_wsl_ros2_fixed.sh`。
- `ros2_singleton.py`
  - Windows/ROS2 单实例锁。
- `config_loader.py`
  - TOML/YAML 配置加载。
- `fastdds_profile.xml`
- `fastdds_profile_windows.xml`
  - WSL/Windows FastDDS 配置。

### 2. 主线运行时与 IK

- `runtime/lula_action_graph_min.py`
  - 当前默认运行时。
- `builtin_ik_solver/builtin_ik_bridge.py`
  - 当前默认 builtin IK 桥。
- `builtin_ik_solver/config/jz_left_arm_robot_description.yaml`
- `builtin_ik_solver/config/jz_right_arm_robot_description.yaml`
  - Lula 机器人描述。

### 3. 参数与可视化配置

- `config/runtime_defaults.toml`
  - Action Graph、hold、body hold、demo 等默认参数。
- `config/builtin_ik.yaml`
  - builtin IK 目标、反馈、平滑、保护参数。
- `config/ik_control.rviz`
  - RViz2 联调配置。

### 4. 平台实现

- `platform/windows/`
  - Windows 端真实 bat 实现。
- `platform/wsl/`
  - WSL 端真实 shell 实现。

### 5. 联调脚本

- `scripts/reset_ik_stack.sh`
  - 清残留进程、锁文件与 daemon。
- `scripts/interactive_ik_marker.py`
  - 交互式 6DoF 目标发布。

### 6. 诊断与专项分析

- `diagnostics/action_graph_topic_io_check.py`
  - 最小 I/O 烟测。
- `diagnostics/ik_phase4_acceptance_check.py`
  - 主线 IK 验收。
- `diagnostics/run_phase4_acceptance.sh`
  - 验收包装脚本。
- `diagnostics/compare_mjcf_isaac_drives.py`
- `diagnostics/extract_mjcf_position_gains.py`
  - MJCF/Isaac drive 参数对比分析。
- `diagnostics/startup_jitter_probe.py`
- `diagnostics/run_startup_jitter_matrix.sh`
  - 启动抖动专项诊断。

### 7. 参考保留

- `ik_solver/`
  - 早期模块化 IK 实现与测试，适合做算法对照和回查，不是当前主线运行时。
- 部分更早期的运行时代码当前已不再单独保留为独立文件。
  - 需要看当前运行主链时，直接以 `runtime/lula_action_graph_min.py` 为准。

### 8. 文档

- `docs/README.md`
  - 文档索引。
- `docs/PROJECT_STRUCTURE.md`
  - 当前文件分类和边界说明。
- `docs/ACTION_GRAPH_MAINLINE_TECHNICAL_GUIDE.md`
  - 架构与问题解决手册。
- `docs/指令文件.md`
  - 当前主线命令手册。
- `指令文件.md`
  - 兼容入口，指向 `docs/指令文件.md`。
- `diagnostics/README.md`
  - 诊断脚本索引。

## 参数文件规则

- CLI 参数优先于配置文件。
- 当前环境不强依赖 `PyYAML`，所以 `config/builtin_ik.yaml` 使用 JSON-compatible YAML 写法。

## 最常用命令

Windows 启动：

```powershell
cd E:\jz_robot\robot_simulator\isaac_sim_implementations\action_graph_method
Remove-Item "$env:LOCALAPPDATA\Temp\jz_robot_isaac_singletons\*_domain_77.json" -Force -ErrorAction SilentlyContinue
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --hold --control-mode auto --domain-id 77 --disable-target-smoothing
```

WSL 清场：

```bash
source /mnt/e/jz_robot/env.sh
bash /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/scripts/reset_ik_stack.sh 77
```

话题烟测：

```bash
/usr/bin/python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/action_graph_topic_io_check.py \
  --groups left,right,body \
  --duration 5 \
  --publish-rate 20
```
