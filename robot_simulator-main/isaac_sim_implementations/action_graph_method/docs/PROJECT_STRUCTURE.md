# 项目结构与文件分类

这份文档描述的是“当前仍然有用的文件如何分类”

日常联调时，先抓住下面这几个核心文件即可：

- `jinzhi_ros2_action_graph.py`
- `runtime/lula_action_graph_min.py`
- `builtin_ik_solver/builtin_ik_bridge.py`
- `config/runtime_defaults.toml`
- `config/builtin_ik.yaml`
- `scripts/reset_ik_stack.sh`
- `diagnostics/action_graph_topic_io_check.py`
- `diagnostics/run_phase4_acceptance.sh`

## 1. 入口与兼容包装

- `jinzhi_ros2_action_graph.py`
  - 当前推荐的 Python 入口。
  - 只负责把入口稳定在根目录，实际运行时在 `runtime/`。
- `run_with_isaac_fixed.bat`
  - Windows 启动包装脚本。
  - 实际调用 `platform/windows/run_with_isaac_fixed.bat`。
- `setup_windows_ros2_fixed.bat`
  - Windows 环境包装脚本。
  - 实际调用 `platform/windows/setup_windows_ros2_fixed.bat`。
- `setup_wsl_ros2_fixed.sh`
  - WSL 环境包装脚本。
  - 实际 source/exec `platform/wsl/setup_wsl_ros2_fixed.sh`。
- `ros2_singleton.py`
  - 按 `service_name + domain_id` 维护单实例锁。
- `config_loader.py`
  - 统一加载 `config/` 下的 TOML/YAML 参数。
- `fastdds_profile.xml`
- `fastdds_profile_windows.xml`
  - WSL 与 Windows 的 DDS profile。

这一级文件的作用是“提供稳定入口”，不是重复垃圾。根目录脚本保留，是为了让操作路径固定，不必每次直接进 `platform/`。

## 2. 主线运行时

- `runtime/lula_action_graph_min.py`
  - 当前默认运行时。
  - 负责 Isaac Sim、Action Graph、hold drives、body posture keeper、builtin IK 集成
判断标准：
- 想改“现在到底怎么跑”：看 `runtime/lula_action_graph_min.py`

## 3. 主线 IK 与机器人描述

- `builtin_ik_solver/builtin_ik_bridge.py`
  - 当前默认的 builtin IK 主线实现。
- `builtin_ik_solver/config/jz_left_arm_robot_description.yaml`
- `builtin_ik_solver/config/jz_right_arm_robot_description.yaml`
  - Lula 侧机器人描述。

这部分属于“当前主线”，修改后会直接影响现网联调行为。

## 4. 参数与可视化配置

- `config/runtime_defaults.toml`
  - 放运行时默认值。
  - 包括 topic、domain、hold、body hold、demo-grasp 等默认项。
- `config/builtin_ik.yaml`
  - 放 builtin IK 结构化参数。
  - 包括目标输入、反馈输出、solver base、平滑与保护参数。
- `config/ik_control.rviz`
  - RViz2 调试与演示配置。

参数管理规则：

- 运行时参数优先放 `config/runtime_defaults.toml`
- IK 结构化参数优先放 `config/builtin_ik.yaml`
- CLI 只做覆盖，不再把大量默认值散回脚本里

## 5. 平台实现

- `platform/windows/run_with_isaac_fixed.bat`
  - Windows 端 Isaac 启动与 ROS2/FastDDS 环境装配的真实实现。
- `platform/windows/setup_windows_ros2_fixed.bat`
  - Windows 环境变量配置实现。
- `platform/wsl/setup_wsl_ros2_fixed.sh`
  - WSL 侧 ROS2、DDS 与 IP 探测配置实现。

`platform/` 是“真实实现层”，根目录同名脚本是“稳定入口层”。

## 6. 联调脚本

- `scripts/reset_ik_stack.sh`
  - 清理锁文件、残留进程、daemon，适合现场一键清场。
- `scripts/interactive_ik_marker.py`
  - 发布交互式 6DoF 目标，适合 RViz 拖动联调。

这部分是“常驻实用工具”，不是测试垃圾。

## 7. 诊断与专项分析

### 日常验收

- `diagnostics/action_graph_topic_io_check.py`
  - 最小话题 I/O 烟测。
- `diagnostics/ik_phase4_acceptance_check.py`
  - 主线 builtin IK 验收。
- `diagnostics/run_phase4_acceptance.sh`
  - 验收快捷包装。

### 专项分析

- `diagnostics/compare_mjcf_isaac_drives.py`
- `diagnostics/extract_mjcf_position_gains.py`
  - drive 参数对照与抽取。
- `diagnostics/startup_jitter_probe.py`
- `diagnostics/run_startup_jitter_matrix.sh`
  - 启动抖动专项诊断。

这类脚本不是每天都跑，但在定位问题时仍然是有用文件，不应在结构说明里被误写成“已无价值”。

## 8. 参考保留

- `ik_solver/`
  - 较早期的模块化 IK 实现、辅助模块与测试。
  - 当前不是默认主线，但适合做算法对照、接口回查和单元测试样例参考。

这类文件的标签应该是“参考保留”，而不是“当前默认入口”。

## 9. 文档

- `README.md`
  - 目录导航。
- `docs/README.md`
  - 文档索引。
- `docs/指令文件.md`
  - 当前常用命令。
- `docs/ACTION_GRAPH_MAINLINE_TECHNICAL_GUIDE.md`
  - 架构、问题、修复策略。
- `diagnostics/README.md`
  - 诊断脚本分类说明。
- `指令文件.md`
  - 根目录兼容入口，指向 `docs/指令文件.md`。

## 10. 生成物

以下内容视为生成物或运行结果，不纳入手工维护结构：

- `__pycache__/`
- `.pytest_cache/`
- `*.tmp.*`
- `diagnostics/results/`
