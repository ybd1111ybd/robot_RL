# Standalone IK Solver 技术文档

## 1. 目标

`standalone_ik_solver` 是一个不依赖 Isaac API 的 ROS2 末端位姿求解器。它复用当前主线话题契约：

- 输入：`/arm_*/ee_target_pose`
- 输出：`/arm_*/joint_commands`
- 反馈：`/arm_*/ee_current_pose`、`/arm_*/ee_ik_status`

设计目标不是替换 Action Graph，而是在保持主链不冲突的前提下，提供一套可迁移到实机的 IK 路径。

## 2. 当前架构

```text
/arm_*/ee_target_pose
        |
        v
standalone_ik_solver
  |- 读取 /joint_states（优先）或 /arm_*/joint_states + /body/joint_states
  |- base_link <-> body_link3 坐标变换
  |- URDF 链式 FK / IK
  |- position-first / full-orientation 策略切换
        |
        +--> /arm_*/joint_commands
        +--> /arm_*/ee_current_pose
        +--> /arm_*/ee_ik_status
```

当前环境里分组状态话题可能没有 publisher，因此节点已兼容汇总 `/joint_states`。

## 3. 求解流程

### 3.1 状态输入

- 左右臂当前关节优先取实时 joint state。
- `body_link3` 基座位姿由 URDF 中 `body_joint1-3` 正向计算得到。
- 当分组 joint state 不存在时，使用汇总 `/joint_states` 中的同名关节回填。

### 3.2 坐标系处理

- ROS 目标仍以 `base_link` 表达。
- 求解器内部工作在 `body_link3`。
- 进入 IK 前执行 `base_link -> body_link3` 变换。
- 发布 `ee_current_pose` 前再变回 `base_link`。

这一步是手写求解器和 Isaac 内建解算器对齐的关键，否则两者解的不是同一个几何问题。

### 3.3 数值策略

1. 先用 URDF 链 FK 计算当前末端位姿。
2. 依据位置误差做 orientation gating：
   - 远距离：`position-first`
   - 近距离：`full`
3. 主回路先走 `RealtimeDLSIKSolver` 单步 Jacobian DLS。
4. 远距离目标以 position-only 方式按周期渐进逼近，行为接近旧 API 规划器。
5. 仅在近场姿态阶段或快速步进不够时，才进入 `HybridIKSolver` 补救。

### 3.4 TCP 建模

- 机械臂方向参考链取 `left_arm_link9` / `right_arm_link9`。
- TCP 方向使用 `GRIPPER_MOUNT_QUAT`，与 builtin 定义保持一致。
- TCP 位置不再使用固定偏移，而是取：
  - 左臂：`mean(left_gripper_narrow3_link, left_gripper_wide3_link)`
  - 右臂：`mean(right_gripper_narrow3_link, right_gripper_wide3_link)`
- 夹爪角度直接来自现有 `/joint_states`，不新增任何夹爪控制接口。

### 3.5 近期关键修复

- `KinematicsChain.forward_kinematics()` 不再对所有链硬编码 `GRIPPER_MOUNT_QUAT`。
  - `body_link3` 基座链现在使用单位四元数，不再错误旋转 `base_link <-> body_link3` 变换。
- URDF `origin rpy` 旋转顺序修正为 `Rz @ Ry @ Rx`。
  - 这一步修复后，夹爪链 `left_gripper_narrow3_link` / `left_gripper_wide3_link` 已与 `/tf` 严格对齐。
- 近目标区域为 realtime DLS 增加 1 次细化子步，减少 `1cm ~ 4cm` 残差。

## 4. 关键实现点

### 已完成

- URDF 驱动的臂链和基座链解析
- `base_link <-> body_link3` 实时参考系变换
- `/joint_states` 聚合回退
- 真实 joint state 作为求解 seed
- builtin 等价 synthetic TCP 中点建模
- `RealtimeDLSIKSolver` 实时步进层
- side-channel 验证模式（`*_standalone` remap）

### 仍在优化

- 近场 full-orientation 路径仍比 builtin 慢
- benchmark 脚本对重复状态消息仍会采到旧样本

## 5. 运行方式

```bash
source /mnt/e/jz_robot/env.sh
source /mnt/e/jz_robot/robot_simulator/install/setup.bash

ros2 run standalone_ik_solver standalone_ik_solver --ros-args \
  -r /arm_left/joint_commands:=/arm_left/joint_commands_standalone \
  -r /arm_right/joint_commands:=/arm_right/joint_commands_standalone \
  -r /arm_left/ee_current_pose:=/arm_left/ee_current_pose_standalone \
  -r /arm_right/ee_current_pose:=/arm_right/ee_current_pose_standalone \
  -r /arm_left/ee_ik_status:=/arm_left/ee_ik_status_standalone \
  -r /arm_right/ee_ik_status:=/arm_right/ee_ik_status_standalone
```

## 6. 验证命令

```bash
python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/standalone_ik_solver/tools/benchmark_left_arm_ik_topics.py --settle-sec 0.8 --warmup-sec 0.5
ros2 topic echo --once /joint_states
ros2 node info /standalone_ik_solver
```

## 7. 最近结论

- 修复前：由于拿不到 body / arm 状态，`body_link3` 基座错误，position solve 会退化到数百毫秒。
- 修复后：节点已从汇总 `/joint_states` 恢复实时状态，并切换到 DLS 主回路；live 首包延迟已回到几十毫秒量级。
- 当前模型 FK 已与 `/tf` 对齐，synthetic TCP 中点也已与 builtin `ee_current_pose` 对齐。
- 同场景小范围 live benchmark 中，standalone 的 `best_pose_err_m` 已进入 builtin 同一量级，当前主要剩余差距是少量 joint-space 分支差异。
