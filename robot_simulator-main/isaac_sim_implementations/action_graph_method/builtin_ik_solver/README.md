# Builtin IK Solver

这里是当前 Action Graph 主线的 builtin IK 实现层。

## 当前主线文件

- `builtin_ik_bridge.py`
  - 当前默认 builtin IK bridge。
- `config/jz_left_arm_robot_description.yaml`
- `config/jz_right_arm_robot_description.yaml`
  - Lula 机器人描述。

## 维护规则

- 末端目标求解、IK 状态反馈、solver 侧保护逻辑在这里维护。
- 运行时默认参数仍优先放仓库根层 `config/builtin_ik.yaml`。
