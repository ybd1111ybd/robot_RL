# RL 训练代码交接清单（中文）

本文用于把 `RL_jzrobot` 交接给新同事时，确保其能在首日独立完成：

1. 环境配置
2. 任务注册校验
3. 训练冒烟
4. 模型回放（checkpoint）
5. 训练监控与问题定位

## 1. 交接前提

请确认以下目录与依赖都已到位：

- `IsaacLab`（可运行）
- `RL_jzrobot`（本仓库）
- `jz_descripetion`（URDF + meshes，必需）
- `robot_simulator`（仅生成 reachable workspace 工具必需）

建议目录布局：

```text
<workspace_root>/
  IsaacLab/
  RL_jzrobot/
  jz_descripetion/
  robot_simulator/
```

## 2. 首次环境配置（PowerShell）

```powershell
$env:ISAACLAB_PATH = "E:\isaac-lab\IsaacLab"
$env:JZLAB_WORKSPACE_ROOT = "E:\jz_robot"
$env:JZLAB_PROJECT_PATH = "$env:JZLAB_WORKSPACE_ROOT\RL_jzrobot"

cd $env:JZLAB_PROJECT_PATH
python -m pip install -e .\source\jzlab
```

关键说明：

- `JZLAB_WORKSPACE_ROOT` 决定了代码如何定位 `jz_descripetion` 与 `robot_simulator`。
- 如果不设该变量，默认会把本仓库的上一级目录当作 workspace root。

## 3. 交接验收（必须通过）

### A. 任务注册验收

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\list_envs.py"
```

应至少看到：

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`

### B. USD 生成验收

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\convert_jz_bimanual.py" --headless
```

### C. 训练冒烟验收（仅验证链路，建议 5 轮）

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 32 --max_iterations 5 --headless
```

说明：该步骤主要验证“能否正常开训”，不保证产出 checkpoint。

### D. 产出 checkpoint 验收（用于回放）

由于默认配置是每 `100` 轮保存一次模型，建议至少跑到 `120` 轮：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

检查模型是否生成：

```powershell
Get-ChildItem "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\*.pth"
```

若希望更快看到模型文件，可临时覆盖保存频率：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 64 --max_iterations 20 --headless +agent.params.config.save_frequency=5 +agent.params.config.save_best_after=1
```

### E. 回放验收

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\jz_bi_reach.pth"
```

### F. 已实测命令（2026-04-29）

以下命令在你的机器上已成功启动训练并稳定推进到 epoch 9：

```powershell
cd E:\isaac-lab\IsaacLab
conda activate env_isaacsim
.\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --headless --num_envs 64 --max_iterations 6000
```

实测关键信息（可作为交接佐证）：

- 任务配置与场景创建成功，`Observation shape = (68,)`，`Action shape = 14`。
- 训练正常推进，`fps step` 约 `3.7k ~ 3.9k`。
- 该次运行的实验名为 `2026-04-29_22-39-06`。
- 由于在 `epoch 9` 手动 `Ctrl+C` 停止，且默认保存频率为 100 轮，因此 `nn` 目录可能为空。

## 4. 推荐交接方式（省心）

优先用一键脚本启动训练与监控脚本：

```powershell
cd $env:JZLAB_PROJECT_PATH
powershell -ExecutionPolicy Bypass -File ".\scripts\reinforcement_learning\rl_games\launch_training_windows.ps1" -NumEnvs 1024 -MaxIterations 2000 -StartTensorBoard
```

常用参数：

- `-RunName`：固定实验名，便于排查与复现实验。
- `-IsaacLabRoot`：不依赖环境变量时可直接传目录。
- `-CondaEnvName`：默认 `env_isaacsim`。
- `-Checkpoint`：从已有模型恢复训练。

## 5. 日志与产物位置

- 训练目录：`$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>`
- 模型目录：`...\<run_name>\nn\`
- 配置快照：`...\<run_name>\params\env.yaml` 与 `agent.yaml`
- 监控日志：`...\<run_name>\watch_status.log` 与 `watch_eval.log`

## 6. 常见问题排查

1) 报错找不到 `jz_descripetion`

- 原因：`JZLAB_WORKSPACE_ROOT` 不正确或目录缺失。
- 处理：确认 `<workspace_root>\jz_descripetion\robot_urdf\urdf\robot_model_isaac_v2.urdf` 存在。

2) `ISAACLAB_PATH` 相关报错

- 原因：环境变量未设置。
- 处理：设置环境变量，或在启动脚本传 `-IsaacLabRoot`。

3) `conda activate` 失败

- 原因：环境名与脚本默认不一致。
- 处理：使用 `-CondaEnvName <你的环境名>`。

4) 首次运行很慢

- 原因：首轮 USD 转换和 shader 缓存。
- 处理：属于正常现象，第二轮通常明显更快。

## 7. 代码责任区（便于同事接手）

- 任务配置：`source/jzlab/jzlab/tasks/manager_based/jz_manipulation/bimanual/reach/`
- 训练入口：`scripts/reinforcement_learning/rl_games/train.py`
- 回放入口：`scripts/reinforcement_learning/rl_games/play.py`
- 监控评估：`scripts/reinforcement_learning/rl_games/monitor_training.py`、`watch_training.py`、`evaluate_checkpoint.py`
- USD 转换：`scripts/tools/convert_jz_bimanual.py`

## 8. 交接完成标准

以下 4 条全部满足，才算“同事可独立上手”：

1. 能在新机器列出 JZ 任务。
2. 能完成 5 轮训练冒烟且生成日志。
3. 能回放一次“已产出 checkpoint”的训练结果。
4. 能说清楚日志目录、模型目录、配置快照目录。
