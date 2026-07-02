# IK Solver 优化说明

## 1. 范围

本轮优化同时覆盖两条链路：

- 主线 `builtin_ik_solver`：继续保留 Isaac Sim 内建求解器
- `standalone_ik_solver`：提供不依赖 Isaac API 的等效 IK 路径

目标不是改接口，而是提升响应、减少坐标系偏差，并为 sim-to-real 做准备。

## 2. 已落地的 builtin 优化

### `builtin_ik_bridge.py`

- 启动阶段预热 solver，避免第一帧现初始化抖动
- `update()` 中增加 solver 延迟初始化重试
- base pose 引入缓存和单周期复位
- 修正 orientation hysteresis 方向：
  - 远距离：`position_first`
  - 近距离：`full`

### `builtin_ik.yaml`

- `target_smoothing_alpha: 0.35 -> 0.20`
- `max_target_position_step_m: 0.02 -> 0.015`
- `max_target_orientation_step_rad: 0.20 -> 0.12`
- `max_joint_delta_rad: 0.15 -> 0.12`
- `orientation_position_gate_m: 0.06 -> 0.05`
- `approx_orientation_accept_position_error_m: 0.03 -> 0.02`
- `close_range_orientation_distance_m: 0.03 -> 0.025`
- `close_range_max_orientation_step_rad: 0.08 -> 0.07`

## 3. 已落地的 standalone 优化

### 建模层

- 机械臂链改为直接从 URDF 构建
- `body_link3` 基座链也由 URDF 解析得到
- 末端方向加入 `GRIPPER_MOUNT_QUAT`
- TCP 位置改为双指尖中点，和 builtin synthetic TCP 对齐
- 新增 Jacobian 计算与 DLS polish
- 修复 `KinematicsChain.forward_kinematics()` 中错误写死 `GRIPPER_MOUNT_QUAT` 的问题：
  - 现在所有链都使用各自的 `tcp_orientation_offset_quat`
  - `body_link3` 基座链不再被错误叠加夹爪姿态
- 修复 URDF `origin rpy` 旋转顺序：
  - `Rx @ Ry @ Rz` 更正为 URDF 兼容的 `Rz @ Ry @ Rx`
  - 该修复直接消除了夹爪链和 `/tf` 不一致的问题

### 状态层

- 左右臂 seed 改为优先读取实时 joint state
- 新增汇总 `/joint_states` 回退，兼容当前 Isaac 图中“只有 merger 输出”的情况
- `base_link -> body_link3` 变换基于 body 关节实时求解

### 求解层

- hybrid solver 增加多 seed
- position-only 路径支持 FABRIK 快速短路
- full solve 前先试 DLS，重优化只在必要时进入
- orientation policy 与 builtin 对齐为 position-first / near-full
- 新增 `RealtimeDLSIKSolver` 作为主回路单步解算器，行为贴近旧 API 规划器

## 4. 关键发现

### 4.1 最大误差源不是数值器，而是参考系

早期 standalone 直接把 `base_link` 目标丢给 `body_link3` 根链，等价于“解错问题”。修复 `base_link <-> body_link3` 后，行为才开始接近 builtin。

### 4.2 当前环境中分组 joint state 话题没有 publisher

运行时观察到：

- `/arm_left/joint_states` publisher 数：`0`
- `/arm_right/joint_states` publisher 数：`0`
- `/body/joint_states` publisher 数：`0`
- `/joint_states` 由 `joint_state_merger` 发布

因此 standalone 必须支持汇总 `/joint_states`，不能只依赖分组状态话题。

### 4.3 position-first 已经显著提速

离线 / live profile 显示：

- 旧 arm-only fixed TCP：position-only 约 `14ms ~ 36ms`
- 新 synthetic TCP + realtime DLS：单步约 `12ms ~ 13ms`
- full solve fallback：仍可到 `500ms+`

这说明主回路应该依赖实时 DLS 步进，近场姿态精修再继续优化。

## 5. 当前 live 验证结论

已在 `*_standalone` side-channel 上验证：

- `ros2 node info /standalone_ik_solver` 正常
- `/arm_left/joint_commands_standalone`、`/arm_right/joint_commands_standalone` 正常发布
- 修复前首包常见在 `500ms ~ 800ms`
- 接入汇总 `/joint_states` 后，position-first 首包已回到几十毫秒量级
- 修复 `tcp_orientation_offset` / URDF `rpy` 后：
  - `left_arm_link9` 链与 `/tf` 对齐
  - `left_gripper_narrow3_link` / `left_gripper_wide3_link` 与 `/tf` 对齐
  - synthetic TCP 中点已与 builtin `ee_current_pose` 对齐
- 当前同场景小范围 benchmark 结果：
  - builtin `best_pose_err_m` 约 `0.006 ~ 0.020`
  - standalone `best_pose_err_m` 约 `0.007 ~ 0.019`
  - 剩余差距主要在首样本发现抖动与少量 joint-space 分支差异

当前残留问题：

- benchmark 首样本仍可能受 DDS 发现时延影响，已增加 warmup 缓冲
- 个别目标点上 standalone 与 builtin 的 joint-space 解仍有 `0.2rad` 级差异，需要继续压缩

## 6. 下一步

1. 继续压缩 near-target full solve 的耗时。
2. 校准 synthetic TCP 位置模型，缩小与 builtin 的末端定义差异。
3. 改善 benchmark，按目标窗口过滤旧状态样本。
4. 在 builtin 可用的同场景下继续做 joint diff 对比。
