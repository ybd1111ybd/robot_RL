# Platform

这里放平台相关的真实实现脚本。

## 目录说明

- `windows/`
  - Windows 侧 bat 实现。
- `wsl/`
  - WSL 侧 shell 实现。

## 入口关系

根目录下这些脚本是稳定入口包装：

- `run_with_isaac_fixed.bat`
- `setup_windows_ros2_fixed.bat`
- `setup_wsl_ros2_fixed.sh`

`platform/` 下才是对应平台的真实实现。
