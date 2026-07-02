# Diagnostics

`diagnostics/` 里的文件分成两类：日常验收工具，以及问题出现时才会用到的专项分析工具。

## 日常验收

### `action_graph_topic_io_check.py`

用途：

- 做最小 ROS2 读写烟测
- 只验证命令话题能进去、状态话题能出来

命令：

```bash
source /mnt/e/jz_robot/env.sh
/usr/bin/python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/action_graph_topic_io_check.py \
  --groups left,right,body \
  --duration 5 \
  --publish-rate 20
```

### `ik_phase4_acceptance_check.py`

用途：

- 对主线 builtin IK 做持续目标驱动验收
- 检查 `/joint_states`、`/tf`、`ee_current_pose`、`ee_ik_status`

命令：

```bash
source /mnt/e/jz_robot/env.sh
/usr/bin/python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/ik_phase4_acceptance_check.py \
  --arms left,right \
  --duration 6 \
  --publish-rate 10 \
  --label ui_validation
```

### `run_phase4_acceptance.sh`

用途：

- WSL 侧主线验收快捷入口

命令：

```bash
source /mnt/e/jz_robot/env.sh
bash /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/run_phase4_acceptance.sh --label ui_validation
```

## 专项分析

### `compare_mjcf_isaac_drives.py`

- 对比 MJCF 与 Isaac drive 参数。

### `extract_mjcf_position_gains.py`

- 抽取 MJCF position gains，便于对照调参。

### `startup_jitter_probe.py`

- 采样启动初期的时序抖动与状态误差。

### `run_startup_jitter_matrix.sh`

- 批量运行启动抖动矩阵诊断。

这些脚本不是每天都跑，但在排查“启动抖动、驱动不一致、参数迁移”时依然有用。

## 输出目录

运行结果通常落到：

```text
diagnostics/results/
├── analysis/
├── reports/
└── runs/
```
