# Action Graph 主线技术手册

## 1. 当前默认主线

当前默认运行链路是：

- Windows Isaac Sim
- Action Graph
- builtin IK / Lula
- WSL ROS2
- RViz2 bridge

真实入口：

- `jinzhi_ros2_action_graph.py`
- `runtime/lula_action_graph_min.py`

主线 IK：

- `builtin_ik_solver/builtin_ik_bridge.py`

补充说明：

- 当前目录里不只有主线文件，还保留了启动包装、诊断脚本和参考实现。
- 如果你想区分“默认入口”和“参考保留”，看 `docs/PROJECT_STRUCTURE.md`。

## 2. 当前架构

```text
Windows
  run_with_isaac_fixed.bat
    -> jinzhi_ros2_action_graph.py
      -> runtime/lula_action_graph_min.py
        -> Action Graph
        -> manual_joint_state_publisher
        -> builtin_ik_bridge
        -> hold drives / body posture keeper
        -> FastDDS

WSL
  isaac_rviz2_bridge.launch.py
    -> joint_state_merger
    -> robot_state_publisher
    -> world_to_base_link_tf
    -> rviz2
```

可以把它分成三层：

1. 关节命令执行层
   - Action Graph 收关节命令并驱动 articulation。

2. 末端目标求解层
   - builtin IK 收 `ee_target_pose`，输出关节目标。

3. 可视化与验收层
   - WSL 侧合并 `/joint_states`、发布 `/tf`、做 RViz 与验收。

## 3. 参数文件

### `config/runtime_defaults.toml`

用于运行时默认值：

- Action Graph 话题
- Domain / namespace
- manual joint state rate
- control mode
- hold/body hold
- demo-grasp 默认项

### `config/builtin_ik.yaml`

用于 builtin IK 结构化参数：

- 末端输入话题
- 反馈与状态话题
- ee link / solver base
- smoothing 与步长限制
- 近场姿态策略
- 关节保护参数

说明：

- 当前环境不假设 `PyYAML` 一定存在；
- 因此 `builtin_ik.yaml` 使用 JSON-compatible YAML 写法；
- 有 `yaml` 包时按 YAML 解析，没有时按 JSON 解析。

## 4. 关键技术

### Action Graph

- 职责是关节链，不负责 IK 数学。
- 重点看 `joint_commands` 是否进入控制器。

### builtin IK / Lula

- 职责是末端链。
- 输入 `PoseStamped`，输出关节目标。
- 同时发布 `ee_current_pose` 与 `ee_ik_status`。

### ROS2 + FastDDS

- Windows 与 WSL 通过同一 DDS 域联通。
- `ROS_DOMAIN_ID` 不一致时是“看不到话题”。
- 域内重复实例时是“能看到话题但状态会抖”。

### Hold Drives 与 BodyPostureKeeper

- hold drives 负责给足关节保持力。
- `BodyPostureKeeper` 负责把身体持续拉回启动姿态。
- 前伸时整机下塌，通常要同时看这两层。

### 单实例锁

- `ros2_singleton.py` 按 `service_name + ROS_DOMAIN_ID` 建锁文件。
- `builtin_ik_bridge` 和 `manual_joint_state_publisher` 都已接入。
- Windows 下已经改为 Win32 进程探活，不再用 `os.kill(pid, 0)`。

## 5. 这次主要解决过的问题

### 问题 1：RViz2 / Isaac Sim 抽搐

根因：

- 同一 Domain 中曾同时存在两套 `builtin_ik_bridge` 和 `manual_joint_state_publisher`。

修复：

- 两个节点都加了单实例锁；
- `reset_ik_stack.sh` 同步清 Windows 锁文件和残留进程。

### 问题 2：Windows 锁文件残留后启动崩溃

根因：

- 用了不适合 Windows 的 `os.kill(pid, 0)` 探活逻辑。

修复：

- 改为 `OpenProcess + GetExitCodeProcess`。

### 问题 3：手前伸时身体下塌

根因：

- body 与 arms 共用一套较软的 hold 参数。

修复：

- arms / body hold 参数拆开；
- 新增 `BodyPostureKeeper`；
- 参数已经外置到 `runtime_defaults.toml`。

### 问题 4：拖动 6DoF 目标时不够跟手

根因：

- 调试阶段难分辨是 IK 真的慢，还是目标平滑在拖后腿。

修复：

- 保留 smoothing，但加了 `--disable-target-smoothing`；
- 调试时先关平滑确认主链是否正确。

## 6. 推荐调试顺序

1. 先看节点是不是唯一

```bash
source /mnt/e/jz_robot/env.sh
export ROS_DOMAIN_ID=77
ros2 node list
```

2. 再看话题发布者是不是唯一

```bash
ros2 topic info -v /arm_left/joint_states
ros2 topic info -v /joint_states
```

3. 再跑最小烟测

```bash
/usr/bin/python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/action_graph_topic_io_check.py \
  --groups left,right,body \
  --duration 5 \
  --publish-rate 20
```

4. 最后跑主线验收

```bash
bash /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/run_phase4_acceptance.sh --label ui_validation
```

## 7. 维护原则

- 先保证单实例、单发布者，再谈 IK 参数优化。
- 先保证 `/joint_states`、`/tf`、`/ee_current_pose` 三条观测链，再谈抓取效果。
- 新参数优先落 `config/`，不要再散落到脚本默认值里。
