# JZ Isaac Lab

JZ 双臂机器人在 Isaac Lab 上的强化学习任务集合，包含任务定义、训练入口、回放脚本以及若干辅助工具。

当前包含三类任务：

- `Isaac-Reach-JZ-Bi-v0`
- `Isaac-Reach-JZ-Bi-Play-v0`
- `Isaac-Grasp-JZ-Bi-v0`
- `Isaac-Grasp-JZ-Bi-Play-v0`
- `Isaac-Open-Drawer-JZ-Bi-v0`
- `Isaac-Open-Drawer-JZ-Bi-Play-v0`

## 项目结构（仓库内）

```text
RL_jzrobot/
  README.md
  docs/
    handoff_quickstart_zh.md               # 补充说明（中文）
  scripts/
    reinforcement_learning/rl_games/
      train.py                             # 训练入口
      play.py                              # 回放入口
      evaluate_checkpoint.py               # 离线评估
      monitor_training.py                  # 标量摘要
      watch_training.py                    # 训练期周期评估
      launch_training_windows.ps1          # Windows 一键训练
    tools/
      convert_jz_bimanual.py               # URDF -> USD
      list_envs.py                         # 列出 JZ 任务
      generate_reachable_workspace.py      # 采样可达空间数据
  source/jzlab/
    jzlab/tasks/manager_based/jz_manipulation/
      assets/                              # 机器人资产配置
      bimanual/reach/                      # Reach 任务配置与 MDP
      usds/                                # 生成的 USD 与配置
```

## 工作区结构

推荐目录布局如下：

```text
<workspace_root>/
  IsaacLab/                                # Isaac Lab 主仓库
  RL_jzrobot/                              # 当前仓库
  jz_descripetion/                         # 必需：URDF + meshes
  robot_simulator/                         # 可选：workspace 采样工具依赖
```

说明：

- `jz_descripetion` 是训练和 USD 生成的必需依赖。
- `robot_simulator` 仅在生成 reachable workspace 数据时需要。

## 环境准备

### 1. 环境变量

```powershell
$env:ISAACLAB_PATH = "<isaaclab_root>"
$env:JZLAB_WORKSPACE_ROOT = "<workspace_root>"
$env:JZLAB_PROJECT_PATH = "$env:JZLAB_WORKSPACE_ROOT\RL_jzrobot"
```

说明：

- `ISAACLAB_PATH` 必填，用于调用 `isaaclab.bat`。
- `JZLAB_WORKSPACE_ROOT` 建议设置；代码会据此寻找 `jz_descripetion` 与 `robot_simulator`。

### 2. Isaac Lab 侧的要求

通常不需要修改 `IsaacLab` 源码，只需要完成以下运行环境设置：

1. 使用 Isaac Lab 对应的 conda 环境，例如 `env_isaacsim`。
2. 设置 `ISAACLAB_PATH` 指向 Isaac Lab 根目录。
3. 在该环境中安装本仓库包：`python -m pip install -e .\source\jzlab`。

可选项（为了每次终端自动生效）：

```powershell
setx ISAACLAB_PATH "<isaaclab_root>"
setx JZLAB_WORKSPACE_ROOT "<workspace_root>"
setx JZLAB_PROJECT_PATH "<workspace_root>\\RL_jzrobot"
```

不建议：

- 不建议直接改 `IsaacLab` 仓库里的任务源码来适配本项目。
- 不建议把 `jzlab` 代码复制进 `IsaacLab/source`（后续升级难维护）。
- 不建议在脚本中硬编码机器路径，优先使用环境变量。

### 3. 安装 Python 包

```powershell
cd $env:JZLAB_PROJECT_PATH
python -m pip install -e .\source\jzlab
```

## 基础验证

### 1. 验证任务注册

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\list_envs.py"
```

看到以下任务即通过：

- `Isaac-Reach-JZ-Bi-v0` / `Isaac-Reach-JZ-Bi-Play-v0`
- `Isaac-Grasp-JZ-Bi-v0` / `Isaac-Grasp-JZ-Bi-Play-v0`
- `Isaac-Open-Drawer-JZ-Bi-v0` / `Isaac-Open-Drawer-JZ-Bi-Play-v0`

### 2. 生成 USD

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\tools\convert_jz_bimanual.py" --headless
```

### 3. 训练冒烟

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 32 --max_iterations 5 --headless
```

日志目录：`$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>`

若要保证产出可回放模型，建议至少执行：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

然后检查：

```powershell
Get-ChildItem "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\*.pth"
```

### 4. 回放 checkpoint

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\jz_bi_reach.pth"
```

## 任务命令

以下命令均已完成最小训练验证。示例统一使用 `num_envs=64`，以避免 `rl_games` 的 `minibatch_size` 约束在极小批量下触发断言。

### Reach

训练：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

回放：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_reach\<run_name>\nn\jz_bi_reach.pth"
```

### Grasp

训练：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Grasp-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

回放：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Grasp-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_grasp\<run_name>\nn\jz_bi_grasp.pth"
```

### Open Drawer

训练：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Open-Drawer-JZ-Bi-v0 --num_envs 64 --max_iterations 120 --headless
```

回放：

```powershell
& "$env:ISAACLAB_PATH\isaaclab.bat" -p "$env:JZLAB_PROJECT_PATH\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Open-Drawer-JZ-Bi-Play-v0 --num_envs 1 --checkpoint "$env:ISAACLAB_PATH\logs\rl_games\jz_bi_open_drawer\<run_name>\nn\jz_bi_open_drawer.pth"
```

如果不确定 checkpoint 文件名，可先找最新 `.pth`：

```powershell
$ckpt = Get-ChildItem "$env:ISAACLAB_PATH\logs\rl_games\<task_log_dir>\<run_name>\nn\*.pth" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$ckpt.FullName
```

## 一键训练（含自动监控）

```powershell
cd $env:JZLAB_PROJECT_PATH
powershell -ExecutionPolicy Bypass -File ".\scripts\reinforcement_learning\rl_games\launch_training_windows.ps1" -NumEnvs 1024 -MaxIterations 2000 -StartTensorBoard
```

说明：当前一键脚本内置的 watcher 主要按 Reach 任务配置（`Isaac-Reach-JZ-Bi-Play-v0`）运行。`Grasp` 和 `Open Drawer` 建议先使用上面的手动训练/回放命令。

可选参数：

- `-IsaacLabRoot`：显式指定 Isaac Lab 目录。
- `-CondaEnvName`：默认 `env_isaacsim`。
- `-RunName`：指定实验名称，便于区分不同实验。

## 常见问题

- 找不到 `jz_descripetion`：检查 `JZLAB_WORKSPACE_ROOT` 是否正确。
- `ISAACLAB_PATH` 为空：先设置环境变量，或在启动脚本传 `-IsaacLabRoot`。
- 首次训练慢：USD 转换与 shader 缓存会拉长首轮耗时。
- `num_envs=1` 时某些任务可能在 `rl_games` 侧触发 `batch_size % minibatch_size` 断言，建议使用 `32` 或 `64` 以上环境做验证。
