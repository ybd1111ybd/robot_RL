# Runtime

这里放 `action_graph_method` 的运行时实现。

## 当前主线

- `lula_action_graph_min.py`
  - 当前默认运行时。
  - 保留对外主入口和主调度逻辑。
  - 负责 Isaac Sim 启动、Action Graph 接线、builtin IK 主流程集成。
- `mainline_shared.py`
  - 放共享常量和基础数学工具。
  - 避免关节名、夹爪开合映射、四元数辅助函数散落在多个文件里。
- `ros_components.py`
  - 放 ROS2 侧运行时组件。
  - 当前包含 gripper remap、手动 `joint_states` 发布、body posture keeper。
- `demo_grasp.py`
  - 放抓取演示状态机和交互控制。
  - 单独拆出后，避免主运行时同时维护 UI、演示流程和桥接逻辑。

## 维护规则

- 对外稳定入口仍然使用根目录 `jinzhi_ros2_action_graph.py`。
- 新增运行时功能时，优先放到对应职责模块，不要继续向 `lula_action_graph_min.py` 回填大段实现。
- `lula_action_graph_min.py` 应尽量保留为装配层；通用常量、ROS2 组件、演示状态机分别放入独立模块。
