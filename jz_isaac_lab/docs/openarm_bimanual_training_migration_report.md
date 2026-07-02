# OpenArm 双臂训练移植报告

日期：2026-04-12  
工作区：`E:\openarm\ros2_ws`

## 1. 目标

本文档说明如何把 OpenArm 当前的双臂训练能力迁移到其他双臂训练项目中，并尽量复现当前仓库里的训练效果，而不是只完成模型导入。

这里的“达到一样的效果”，按当前仓库代码，指的是复现 `Isaac-Reach-OpenArm-Bi-v0` 的行为特征，而不是直接得到最优真机策略。

## 2. 结论

如果目标是复现当前 OpenArm 双臂 reach 任务的训练表现，最关键的不是 URDF/USD 本身，而是下面这组组合条件：

- 双臂任务实际使用的是 `OPEN_ARM_HIGH_PD_CFG`，不是默认机器人配置。
- 该配置关闭了重力：`disable_gravity=True`。
- 动作接口是双臂 14 维关节位置动作，不是力矩动作，也不是任务空间 IK 动作。
- 底层执行依赖较硬的 PD：臂部 `stiffness=400.0`，`damping=80.0`。
- 奖励、命令采样范围、观测拼接顺序、初始姿态、步长和求解器参数都对训练结果有直接影响。

因此，迁移时如果只复制 OpenArm 几何模型和关节限制，但把训练项目改成“有重力 + 另一套控制器 + 另一套观测动作定义”，训练效果通常不会和当前仓库一致。

## 3. 当前基线实现

### 3.1 任务入口

当前双臂 reach 任务注册在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/__init__.py`

任务名：

- `Isaac-Reach-OpenArm-Bi-v0`

环境入口：

- `joint_pos_env_cfg:OpenArmReachEnvCfg`

### 3.2 机器人资产与动力学假设

当前双臂资产定义在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/assets/openarm_bimanual.py`

关键参数：

- 资产文件：`usds/openarm_bimanual/openarm_bimanual.usd`
- 默认配置：`disable_gravity=False`
- 任务实际使用的高 PD 配置：`OPEN_ARM_HIGH_PD_CFG`
- 高 PD 配置中显式改为：`disable_gravity=True`
- 刚体参数：`max_depenetration_velocity=5.0`
- articulation 求解器：`solver_position_iteration_count=8`
- articulation 求解器：`solver_velocity_iteration_count=0`
- 自碰撞：`enabled_self_collisions=False`

执行器参数：

- 默认臂部：`stiffness=80.0`，`damping=4.0`
- 任务实际臂部：`stiffness=400.0`，`damping=80.0`
- 夹爪：`stiffness=2e3`，`damping=1e2`

这说明当前任务本质上是在“无重力、高刚度位置伺服”的仿真假设下训练。

### 3.3 动作接口

当前动作定义在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/joint_pos_env_cfg.py`

动作不是力矩控制，而是两组关节位置动作：

- 左臂 7 关节 `JointPositionActionCfg`
- 右臂 7 关节 `JointPositionActionCfg`
- `scale=0.5`
- `use_default_offset=True`

这意味着策略学习的是“关节目标偏移到期望位置”的映射，而不是直接学习动力学补偿。

### 3.4 观测接口

当前观测定义在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/reach_env_cfg.py`

策略观测由以下项拼接：

- 左臂关节相对位置
- 右臂关节相对位置
- 左臂关节相对速度
- 右臂关节相对速度
- 左臂目标末端位姿命令
- 右臂目标末端位姿命令
- 左臂上一步动作
- 右臂上一步动作

观测噪声：

- 关节位置和速度项均加入 `[-0.01, 0.01]` 均匀噪声
- `enable_corruption=True`
- `concatenate_terms=True`

迁移时如果观测顺序、维度、归一化方式、是否拼接上一步动作发生变化，训练结果通常会直接偏移。

### 3.5 命令采样范围

当前末端目标采样范围定义在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/reach_env_cfg.py`

左臂：

- `pos_x=(0.15, 0.3)`
- `pos_y=(0.15, 0.25)`
- `pos_z=(0.3, 0.5)`
- `roll=(-pi/6, pi/6)`
- `pitch=(3pi/2, 3pi/2)`
- `yaw=(8pi/9, 10pi/9)`

右臂：

- `pos_x=(0.15, 0.3)`
- `pos_y=(-0.25, -0.15)`
- `pos_z=(0.3, 0.5)`
- `roll=(-pi/6, pi/6)`
- `pitch=(3pi/2, 3pi/2)`
- `yaw=(8pi/9, 10pi/9)`

这组范围决定了训练分布。换项目时如果工作空间范围、末端坐标系定义或手爪朝向约定改变，效果不会等价。

### 3.6 奖励设计

当前奖励定义在：

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/reach_env_cfg.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/mdp/rewards.py`

主要奖励项：

- 左末端位置误差：`weight=-0.2`
- 右末端位置误差：`weight=-0.25`
- 左末端位置 tanh 精细奖励：`weight=0.1`，`std=0.1`
- 右末端位置 tanh 精细奖励：`weight=0.2`，`std=0.1`
- 左末端姿态误差：`weight=-0.1`
- 右末端姿态误差：`weight=-0.1`
- 动作变化惩罚：`weight=-1e-4`
- 左右关节速度惩罚：`weight=-1e-4`

curriculum：

- `action_rate` 在 `4500` 步内调到 `-0.005`
- `left_joint_vel` 在 `4500` 步内调到 `-0.001`
- `right_joint_vel` 在 `4500` 步内调到 `-0.001`

如果目标项目已有自己的奖励模板，建议不要直接混用；要么完整复刻当前奖励，要么明确接受训练行为会变化。

### 3.7 仿真与并行度

当前环境基础参数：

- `num_envs=4096`
- `env_spacing=2.5`
- `decimation=2`
- `sim.dt=1/60`
- 控制周期约为 `1/30 s`
- `episode_length_s=24.0`

如果目标项目默认控制频率更高、更低，或者解算器稳定性不同，位置环的表现会明显变化。

### 3.8 真机侧补偿链路

仓库里的真机控制相关逻辑在：

- `ros2_ws/src/openarm_teleop/src/controller/control.cpp`

可以确认当前真机控制代码中存在：

- 重力补偿 `GetGravity(...)`
- 科氏项计算 `GetCoriolis(...)`
- 摩擦补偿 `ComputeFriction(...)`
- MIT 模式下的 `Kp/Kd/q/dq/tau` 指令构造

这意味着 OpenArm 当前生态不是完全依赖策略自己学会补偿；真机控制侧本身就具备模型补偿能力。

## 4. 要迁移的不是“模型”，而是“行为接口”

如果目标是在其他双臂训练项目中得到和当前仓库尽量一致的训练效果，建议按下面四层迁移。

### 4.1 第一层：机器人结构层

必须一致：

- 双臂 14 关节的关节顺序
- 左右手末端 link 名称与末端坐标系定义
- 关节正方向
- 初始姿态
- 速度/力矩限制
- 夹爪开合语义

建议迁移来源：

- `openarm_bimanual.usd`
- 或由 `openarm_description` 中的双臂 URDF/xacro 重新导出到目标项目的模型格式

### 4.2 第二层：执行器与控制语义层

这是最容易被忽略、但最影响复现结果的一层。

必须尽量一致：

- 关节位置伺服，而不是力矩策略
- 臂部高 PD：`400 / 80`
- 动作缩放：`0.5`
- 以默认关节偏置为基础的相对动作解释
- 无重力假设，或者功能等效的重力补偿

如果目标项目不支持 `disable_gravity=True` 这种按机器人关闭重力的方式，可以使用下面的等效方案：

- 保持全局重力开启
- 在每个控制步为 OpenArm 双臂注入解析式或数值式重力补偿力矩
- 让策略仍然只面对“接近无重力的位置控制对象”

这比直接开重力训练更接近当前仓库的行为。

### 4.3 第三层：任务定义层

必须尽量一致：

- 左右臂目标位姿采样范围
- 末端目标 body 名称
- 观测组成和拼接顺序
- 奖励项和权重
- reset 时关节初始化

尤其要避免下面几种常见偏差：

- 把末端命令从机器人根坐标改成世界坐标但奖励没同步改
- 把左右手的末端 frame 选错
- 把当前的相对关节位置观测改成绝对关节位置观测
- 去掉上一时刻动作观测

### 4.4 第四层：训练系统层

建议尽量一致：

- 控制频率
- 仿真步长
- 求解器迭代次数
- 并行环境规模
- 观测噪声注入

当前双臂 reach 任务里没有看到明显的动力学域随机化配置；因此如果目标项目默认开了质量、摩擦、时延、外力扰动随机化，训练效果会和当前基线不同。

## 5. 两种迁移策略

### 5.1 策略 A：效果复现优先

适用于你想在别的双臂训练框架里，先跑出和 OpenArm 当前仓库尽量接近的学习曲线和策略行为。

做法：

- 复刻双臂资产和关节顺序
- 复刻高 PD 位置伺服
- 复刻无重力或等效重力补偿
- 复刻观测、动作、奖励、命令采样范围
- 复刻步长和控制频率

优点：

- 最容易对齐当前仓库表现
- 更容易判断迁移是否成功

缺点：

- 与真机动力学仍有差距
- 不能直接证明策略真机鲁棒性

### 5.2 策略 B：真机部署优先

适用于目标项目不仅要训练，还要更稳地上真机。

做法：

- 第一阶段先复刻当前无重力高 PD 基线，确认任务可学
- 第二阶段逐步引入重力或重力补偿误差
- 第三阶段加入轻量域随机化
- 最终在真机低层使用重力/摩擦补偿和受限速度、受限加速度执行

优点：

- 更接近真实部署条件

缺点：

- 训练调参成本更高
- 不再是当前仓库的严格等价复刻

## 6. 建议的最小迁移清单

如果只做最小可用迁移，建议至少完成下面这些项。

### 6.1 训练侧必做

- 导入 OpenArm 双臂模型，并校正关节顺序与末端 frame
- 实现 14 维双臂关节位置动作
- 实现高 PD 位置伺服
- 关闭重力，或为 OpenArm 单独添加重力补偿
- 复刻双臂 reach 的命令范围
- 复刻观测拼接顺序与噪声
- 复刻奖励权重
- 复刻初始姿态和 reset 策略

### 6.2 若要对接真机，额外必做

- 提供低层位置控制器
- 在低层添加重力补偿
- 在低层添加摩擦补偿
- 做动作限幅、速度限幅、急停
- 做真机零位和关节方向校验

## 7. 不建议的移植方式

下面这些做法看上去工作量小，但通常复现不出当前效果。

- 只导入 URDF/USD，不复制动作与奖励定义
- 直接改成力矩控制训练
- 保持开重力，但不做任何补偿，也不重调奖励和增益
- 用 IK 动作替代当前 14 维 joint position 动作
- 训练时加入大量域随机化，却拿它和当前仓库结果做一一对比

## 8. 验收标准

建议用下面的标准判断迁移是否成功。

### 8.1 仿真内对齐

- 双臂末端都能稳定到达各自采样目标
- 左右臂不存在明显一侧更容易抖动或下垂
- 成功率、收敛速度和轨迹风格与当前仓库大体接近
- 相同随机种子下，动作分布和关节速度量级接近

### 8.2 真机前检查

- 单臂悬空时，不因重力导致大幅稳态误差
- 双臂同步动作时，无持续振荡
- 末端低速靠近目标时，无高频抖动
- 夹爪与腕部在极限位附近不会突然冲击

## 9. 推荐实施顺序

1. 先在目标项目中复刻 `Isaac-Reach-OpenArm-Bi-v0` 的训练设定，不要一开始就加真机优化。
2. 确认观测、动作、奖励完全对齐后，再比较收敛曲线。
3. 若目标是上真机，再把重力补偿和摩擦补偿放到低层控制器里。
4. 最后再决定是否引入开重力训练或域随机化。

## 10. 最终建议

如果你的目标是“迁移后先得到和 OpenArm 当前双臂项目同样的训练效果”，建议把迁移目标定义为：

- 复刻当前的无重力高 PD 双臂位置控制任务

而不是：

- 直接构建一个真机友好的、完全不同动力学假设的新任务

这两件事的优先级应该分开。先做等价复现，再做真机增强，风险最低。

## 11. 参考代码位置

- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/assets/openarm_bimanual.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/__init__.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/config/joint_pos_env_cfg.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/reach_env_cfg.py`
- `openarm_isaac_lab/source/openarm/openarm/tasks/manager_based/openarm_manipulation/bimanual/reach/mdp/rewards.py`
- `ros2_ws/src/openarm_teleop/src/controller/control.cpp`
