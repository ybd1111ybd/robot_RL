# Config

这里放 `action_graph_method` 当前主线配置文件。

## 配置分类

- `runtime_defaults.toml`
  - 运行时默认参数。
- `builtin_ik.yaml`
  - builtin IK 结构化参数。
- `ik_control.rviz`
  - RViz2 交互和可视化配置。

## 维护规则

- 默认值优先外置到这里，不再把大量默认参数散在脚本中。
- CLI 参数只做覆盖，不做主配置落点。
