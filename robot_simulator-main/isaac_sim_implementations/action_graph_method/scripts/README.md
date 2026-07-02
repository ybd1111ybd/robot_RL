# Scripts

这里放主线联调脚本，不放运行时核心逻辑。

## 当前脚本

- `reset_ik_stack.sh`
  - 清理现场残留进程、锁文件、daemon。
- `interactive_ik_marker.py`
  - 发布交互式 6DoF 目标。

## 维护规则

- 主线联调脚本继续放这里。
- 诊断验收脚本放 `diagnostics/`，不要和这里混放。
