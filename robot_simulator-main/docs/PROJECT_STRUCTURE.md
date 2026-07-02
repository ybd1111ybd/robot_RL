# Robot Simulator 结构说明

这份文档描述的是 `robot_simulator/` 仓库内部的工程化分层。

## 1. 仓库分层

当前仓库分成三层：

- 并行运行面
- 参考/兼容资产
- 仓库级文档

## 2. 当前结构

```text
robot_simulator/
├── README.md
├── docs/
│   ├── README.md
│   └── PROJECT_STRUCTURE.md
│
├── mojuco/
│   ├── README.md
│   ├── mujoco_simulator/
│   ├── launch/
│   ├── config/
│   ├── resource/
│   ├── package.xml
│   └── setup.py
│
└── isaac_sim_implementations/
    ├── README.md
    └── action_graph_method/
        ├── README.md
        ├── runtime/
        ├── builtin_ik_solver/
        ├── config/
        ├── platform/
        ├── scripts/
        ├── diagnostics/
        ├── docs/
        └── ik_solver/
```

## 3. 各层职责

### MuJoCo 运行面

- `mojuco/`
  - 独立 ROS2 包。
  - 适合 MuJoCo 仿真和对照验证。

### Isaac 运行面

- `isaac_sim_implementations/action_graph_method/`
  - 当前 Isaac Sim 主线。
  - 当前默认链路是 Action Graph + builtin IK + Windows Isaac Sim + WSL ROS2。

### 参考/兼容层

在 `action_graph_method/` 内：

- `ik_solver/`
  - 早期模块化 IK 实现，当前按参考保留理解。
- 部分专项诊断脚本
  - 启动抖动、MJCF/Isaac drive 对照等，不是日常入口。

## 4. 维护规则

- 改当前 Isaac 主线，优先在 `action_graph_method/` 下改
- 改 MuJoCo 包，优先在 `mojuco/` 下改
- 改仓库级导航和结构说明，才在 `docs/` 和仓库根 README 改
- `__pycache__/`、诊断结果目录、构建产物都按生成物处理
