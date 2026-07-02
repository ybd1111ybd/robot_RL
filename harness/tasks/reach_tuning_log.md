# Reach 调参记录

## 当前问题
第一轮 baseline 训练已经跑通，但策略还没有真正学会 Reach。

baseline：
```text
任务: Isaac-Reach-JZ-Bi-v0
运行名: reach_128env_2000it_v1
并行环境: 128
训练轮数: 2000
最新权重: logs/rl_games/jz_bi_reach/reach_128env_2000it_v1/nn/last_jz_bi_reach_ep_2000_rew_2.7608962.pth
```

训练 reward 有上升：
```text
epoch 300:  约 -2.77
epoch 600:  约  0.13
epoch 1000: 约  1.25
epoch 1500: 约  2.58
epoch 2000: 约  2.76
```

但是 headless 评估结果不好：
```text
left_error_mean: 0.680429
right_error_mean: 0.725496
near_goal_ratio_mean: 0.000000
settle_ratio_mean: 0.000000
max_action_mean: 4.980184
action_rate_mean: 1.451408
joint_accel_max_abs_mean: 147.527506
```

视频观察：
```text
--num_envs 4 所以视频里有 4 个机器人。
四个机器人看起来有关节打到限位、动作乱甩的现象。
```

结论：
```text
训练链路是通的，PPO 也确实在优化 reward。
但当前策略不是一个成功的 Reach 策略。
问题更像是动作太激进、目标太难、奖励缺少“朝目标前进/到点”的明确引导。
```

## 当前实验方向
不覆盖原始任务，新增 Easy 版本做 A/B 对比：
```text
Isaac-Reach-JZ-Bi-v0        原 baseline，保留
Isaac-Reach-JZ-Bi-Easy-v0   新实验，降低难度和动作激进程度
```

Easy 版本目标：
```text
减少关节打限位
减少乱甩
让 TCP 先学会朝目标移动
让 near_goal_ratio 从 0 变成非 0
```

## 当前在调哪些参数

| 参数 | baseline | Easy 目标 | 为什么调 |
| --- | --- | --- | --- |
| action scale | `1.0` | `0.25` | 降低每步关节目标幅度，减少打限位和乱甩 |
| action max penalty | 关闭 | 打开，约 `-1e-4` | 惩罚单个关节动作尖峰 |
| action rate penalty | 已有 | 先保持 | 已经有动作变化惩罚，先看降 action scale 是否足够 |
| joint velocity penalty | 已有 | 先保持 | 已经有速度惩罚，先避免把动作压得完全不动 |
| workspace curriculum | `(1.0,1.0,1.0)` | `(0.2,0.5,1.0)` | 先学近目标，再逐渐扩大目标范围 |
| progress reward | 关闭 | 打开 | 只要 TCP 比上一步更接近目标就奖励 |
| goal bonus | 关闭 | 打开，阈值约 `0.10m` | 给“进入目标附近”一个明确奖励 |
| stable/settle reward | 关闭 | 暂不优先 | 当前还到不了目标，先不调稳定停留 |
| PPO 参数 | 默认 | 暂不优先 | reward 在上升，PPO 没明显数值崩溃 |
| PD gains | `260/36` | 暂不优先 | 先不动控制器物理响应，避免干扰对比 |

## 重点参数说明

### 1. action scale
当前动作太猛，视频里疑似打限位，评估里 `max_action_mean` 和 `joint_accel` 都偏大。

所以 Easy 版本先把：
```text
_ACTION_SCALE = 1.0
```
降到：
```text
_EASY_ACTION_SCALE = 0.25
```

预期：
```text
动作更小
关节更少撞限位
视频里不再明显乱甩
joint_accel 降低
```

风险：
```text
如果太小，机器人可能动得太慢，到不了远目标。
```

### 2. workspace curriculum
baseline 一开始就从完整 reachable workspace 采样目标：
```text
(1.0, 1.0, 1.0)
```

Easy 版本改成：
```text
(0.2, 0.5, 1.0)
```

意思是：
```text
前期只用较容易的一部分目标
中期扩大
后期再覆盖完整目标空间
```

预期：
```text
TCP error 更快下降
near_goal_ratio 更早变成非 0
```

### 3. progress reward
baseline 没有打开：
```text
left_end_effector_position_progress = None
right_end_effector_position_progress = None
```

Easy 版本打开后，含义是：
```text
如果这一帧比上一帧更接近目标，就给奖励。
```

预期：
```text
策略更容易学到“朝目标方向移动”。
```

风险：
```text
只靠 progress reward 可能会鼓励短期靠近，但不一定稳定到点，所以要配合 goal bonus。
```

### 4. goal bonus
baseline 的 `near_goal_ratio` 是 0，说明策略没有进入目标附近。

Easy 版本先打开左右手单独 goal bonus：
```text
距离目标小于约 0.10m 时给奖励。
```

预期：
```text
near_goal_ratio 从 0 变成非 0
策略知道“到目标附近”是明确好行为
```

## 暂时不优先调什么

暂时不优先调 PPO：
```text
learning_rate
horizon_length
minibatch_size
entropy_coef
network size
```

原因：
```text
当前 reward 能上升，训练没有 NaN 或明显数值崩溃。
首要问题更像环境奖励和动作尺度，而不是 PPO 本身。
```

暂时不优先调 PD gains：
```text
stiffness = 260.0
damping = 36.0
```

原因：
```text
改 PD 会改变机器人整体物理响应，先不要把控制器和奖励问题混在一起。
```

## 下一轮实验
推荐运行：
```text
任务: Isaac-Reach-JZ-Bi-Easy-v0
运行名: reach_easy_128env_2000it_v1
并行环境: 128
训练轮数: 2000
```

训练命令：
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Reach-JZ-Bi-Easy-v0 \
    --num_envs 128 \
    --max_iterations 2000 \
    --headless \
    +agent.params.config.full_experiment_name=reach_easy_128env_2000it_v1 \
    +agent.params.config.torch_compile=False
'
```

录视频命令：
```bash
cd /home/cqy/workspace/middle_platform/robot_simulator/robot_simulator-main/docker/isaac

docker compose --env-file .env run --rm isaac-lab -lc '
  export JZLAB_WORKSPACE_ROOT=/workspace
  export JZLAB_PROJECT_PATH=/workspace/jz_isaac_lab
  export HYDRA_FULL_ERROR=1
  export JZLAB_ALLOW_STALE_USD_CACHE=1
  export JZLAB_FORCE_USD_REBUILD=0

  cd /workspace/jz_isaac_lab

  /isaac-sim/python.sh /workspace/jz_isaac_lab/scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Reach-JZ-Bi-Easy-v0 \
    --num_envs 4 \
    --checkpoint /workspace/jz_isaac_lab/logs/rl_games/jz_bi_reach/reach_easy_128env_2000it_v1/nn/<checkpoint>.pth \
    --headless \
    --video \
    --video_length 300
'
```

## 对比指标
baseline：
```text
left_error_mean: 0.680429
right_error_mean: 0.725496
near_goal_ratio_mean: 0.000000
action_rate_mean: 1.451408
max_action_mean: 4.980184
joint_accel_max_abs_mean: 147.527506
```

Easy 版本希望看到：
```text
left_error_mean 明显下降
right_error_mean 明显下降
near_goal_ratio_mean > 0
max_action_mean 下降
joint_accel_max_abs_mean 下降
视频里明显少打限位、少乱甩
```

## 视频判断标准
每次 MP4 回放后，把现象归类：
```text
A. 基本不动
B. 动作乱甩 / 关节大幅摆动
C. 有往目标方向走，但距离还远
D. 能靠近目标，但抖动停不住
E. 一只手好，另一只手差
F. 目标点看起来太远或太散
```

当前 baseline 归类：
```text
B. 动作乱甩 / 关节大幅摆动
疑似关节打到限位
```

对应调参方向：
```text
优先降低 action scale
打开 action max penalty
再看是否需要加强 action_rate 或 joint velocity 惩罚
```

## 下一步决策
Easy 版本训练后按结果决定：

如果误差下降、动作更平滑：
```text
继续 Easy 方向，逐步扩大目标范围或训练更久。
```

如果还是乱甩：
```text
继续降低 action scale，例如 0.25 -> 0.1。
或增强 action_max_abs / action_rate 惩罚。
```

如果几乎不动：
```text
action scale 可能太小，改为 0.5。
或降低动作惩罚。
```

如果误差下降但 near_goal_ratio 仍为 0：
```text
保留 progress reward，适当放宽 goal threshold 或训练更久。
```

如果一只手明显差：
```text
检查左右臂目标分布、关节映射、奖励权重是否对称。
```
