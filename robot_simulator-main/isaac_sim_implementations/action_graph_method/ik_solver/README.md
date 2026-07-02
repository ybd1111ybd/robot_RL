# IK Solver Module

模块化的逆运动学求解器，用于Isaac Sim Action Graph末端位姿控制。

## 模块结构

```
ik_solver/
├── __init__.py           # 模块入口
├── ik_config.py          # 配置管理
├── ik_solver_core.py     # IK核心算法（Jacobian DLS）
├── ik_bridge.py          # ROS2桥接和集成
├── ik_utils.py           # 辅助函数（FK等）
└── README.md             # 本文档
```

## 模块说明

### ik_config.py
- `IKConfig`: 配置数据类，管理所有IK相关参数
- 包含控制模式、话题名称、末端执行器链接、算法参数等

### ik_solver_core.py
- `IKSolverCore`: 核心IK算法实现
- 使用Jacobian DLS（阻尼最小二乘）方法
- 支持位置和姿态控制
- 包含速度限制、关节限位等安全机制

### ik_bridge.py
- `EndEffectorIKBridge`: ROS2通信桥接
- `ArmIKController`: 单臂IK控制器
- 在独立线程中运行ROS2订阅
- 主循环调用`update()`执行IK计算和发布命令

### ik_utils.py
- 辅助函数集合
- 前向运动学（FK）计算
- Jacobian提取和处理

## 使用方法

### 1. 启用IK模式

```bash
# Windows (Isaac Sim)
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --control-mode ee_pose --domain-id 77

# 或使用 auto 模式（仅在存在活动末端目标时执行 IK）
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --control-mode auto --domain-id 77
```

### 2. 发布末端目标位姿

```bash
# WSL (ROS2)
source /mnt/e/jz_robot/env.sh
# 默认目标超时为 0.5 秒，建议以 10 Hz 或更高频率持续发布

# 左臂目标
ros2 topic pub -r 10 /arm_left/ee_target_pose geometry_msgs/msg/PoseStamped \
  "{header:{frame_id:'world'},pose:{position:{x:0.45,y:0.25,z:0.85},orientation:{w:1.0}}}"

# 右臂目标
ros2 topic pub -r 10 /arm_right/ee_target_pose geometry_msgs/msg/PoseStamped \
  "{header:{frame_id:'world'},pose:{position:{x:0.45,y:-0.25,z:0.85},orientation:{w:1.0}}}"
```

### 3. 监控状态

```bash
# 查看IK状态
ros2 topic echo /arm_left/ee_ik_status
ros2 topic echo /arm_right/ee_ik_status

# 查看当前末端位姿
ros2 topic echo /arm_left/ee_current_pose
ros2 topic echo /arm_right/ee_current_pose
```

## 参数说明

### 控制模式
- `--control-mode joint`: 仅关节控制（默认，向后兼容）
- `--control-mode ee_pose`: IK 持续启用；无活动目标时持续发布当前末端位姿和状态
- `--control-mode auto`: 仅在活动末端目标存在时执行 IK

### 话题配置
- `--left-ee-topic`: 左臂末端目标话题（默认：`/arm_left/ee_target_pose`）
- `--right-ee-topic`: 右臂末端目标话题（默认：`/arm_right/ee_target_pose`）
- `--left-ee-link`: 左臂末端链接名（默认：`left_gripper_center_tcp`）
- `--right-ee-link`: 右臂末端链接名（默认：`right_gripper_center_tcp`）

### IK算法参数
- `--ik-rate-hz`: IK更新频率（默认：50 Hz）
- `--ik-damping`: DLS阻尼因子（默认：0.15）
- `--ik-pos-gain`: 位置误差增益（默认：0.8）
- `--ik-ori-gain`: 姿态误差增益（默认：0.6）
- `--ik-max-dq`: 单步最大关节变化（默认：0.15 rad）
- `--ik-max-joint-vel`: 最大关节速度（默认：1.5 rad/s）
- `--ik-pos-tol`: 位置收敛容差（默认：0.02 m）
- `--ik-ori-tol`: 姿态收敛容差（默认：0.05 rad）
- `--ik-timeout-sec`: 目标超时时间（默认：1.0 s）
- `--ik-perf-log-interval-sec`: IK 性能摘要日志周期（默认：5.0 s，设为 0 可关闭）
- `--ik-perf-warn-threshold-ms`: 单次 IK 更新耗时告警阈值（默认：5.0 ms）
- `--ik-enable-orientation`: 启用姿态控制（默认：仅位置）

## 架构设计

### 数据流
1. ROS2订阅线程接收`PoseStamped`目标
2. 目标缓存在`ArmIKController`（线程安全）
3. 主仿真循环按`ik_rate_hz`调用`update()`
4. IK计算输出`JointState`命令
5. 发布到Action Graph命令输入话题

### 线程模型
- **ROS2线程**: 独立线程运行订阅/发布
- **主线程**: Isaac Sim仿真循环，调用IK更新

### 与现有系统集成
- 不修改Action Graph结构
- 通过现有关节命令话题发布IK输出
- 完全兼容`--control-mode joint`模式

## 待完成功能（TODO）

### Phase 1 - 核心功能（已完成）
- [x] 模块化结构
- [x] 配置管理
- [x] DLS IK 算法
- [x] ROS2 桥接
- [x] 真实 FK 反馈（通过 articulation world pose）
- [x] 末端链接索引查找
- [x] Jacobian 提取与关节限位

### Phase 2 - 稳定性（部分完成）
- [x] 数值稳定性保护
- [x] 奇异位姿检测
- [x] 超时与异常保护
- [ ] 端到端性能优化与统计

### Phase 3 - 高级功能（部分完成）
- [x] auto 模式按需 IK
- [x] ee_pose 模式 idle 状态发布
- [ ] 姿态控制系统级验收
- [ ] 碰撞检测集成
- [ ] 轨迹平滑

## 调试技巧

### 1. 检查话题连接
```bash
ros2 topic list | grep -E "arm_left|arm_right|ee_"
```

### 2. 查看IK日志
启动时会打印：
- IK配置参数
- 关节索引映射
- 末端链接名称

### 3. 测试单臂
先测试单臂（如左臂），确认工作后再测试双臂。

### 4. 调整参数
如果出现抖动或发散：
- 增大`--ik-damping`（如0.2）
- 减小`--ik-pos-gain`（如0.6）
- 减小`--ik-max-dq`（如0.1）

## 回滚方案

如果IK出现问题，立即回滚到关节控制模式：

```bash
# 使用默认joint模式启动
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --domain-id 77

# 或显式指定
.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --control-mode joint --domain-id 77
```

## 参考文档

- 技术方案: `/mnt/e/jz_robot/docs/action_planer/ik_solver_final_technical_solution.md`
- 执行计划: `/mnt/e/jz_robot/docs/action_planer/ik_solver_final_execution_plan.md`
- 主项目文档: `/mnt/e/jz_robot/CLAUDE.md`
