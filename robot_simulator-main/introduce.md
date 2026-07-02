强化学习对接文档
1. Isaac Sim 仿真环境搭建与资产配置
本模块主要包含基于 Isaac Sim 的基础仿真环境搭建，以及自研机器人的导入与调试。
isaacsim有很多版本，我使用的是5.1.0windows版本，网上教程大多是4.5.0Linux
Isaac Sim官方学习网址
What Is Isaac Sim? — Isaac Sim Documentation
isaacsim页面常用功能B站视频
【06.IsaacSim初体验-IsaacSim页面常用功能】https://www.bilibili.com/video/BV1cgnnzfEaV?vd_source=f0f665afefd1f9899ac346acd7be12f3
isaacsim仿真设置
【07.IsaacSim初体验-IsaacSim物理仿真初识】https://www.bilibili.com/video/BV1XAn6zNEKm?vd_source=f0f665afefd1f9899ac346acd7be12f3
Isaac Lab 官方文档 (强化学习核心)
https://isaac-sim.github.io/IsaacLab/main/index.html
Isaac Sim 自带了海量的 Python 示例代码，涵盖了从基础 API 调用到前沿强化学习的完整生态。它们主要分布在三大目录下，请根据开发需求“对号入座”寻找参考代码：
1. standalone_examples (独立运行脚本 )
这是最常用、最核心的代码库。里面的脚本都在最开头实例化了 SimulationApp，意味着它们可以完全脱离 UI 界面，直接在终端通过 ./python.bat xxx.py 运行。
这个目录下的子文件夹按功能划分，极具参考价值：
- api/ (底层核心 API 示例)
  - 看点： 包含了如何用代码控制物理引擎、生成刚体、修改材质等基础操作。
  - 亮点代码： 我们之前分析的批量生成和控制多个方块的 rigid_prim_view.py 就在这里。如果未来需要做多机器人并行仿真，一定要参考这里的 View 系列 API。
- robotics/ (机器人应用示例)
  - 看点： 这里是针对机械臂（Franka, UR10 等）和移动底盘（Carter 等）的专属控制代码。
  - 亮点代码： 包含了底层控制器使用、ROS/ROS2 桥接（ROS2 Bridge）配置、以及如何给机械臂发送关节控制指令的示例。如果你想升级现有的 IK（逆运动学）算法或接入 MoveIt2，这里的代码是最佳模板。
- replicator/ (合成数据生成 SDG)
  - 看点： 极度契合具身需求！ Replicator 是 NVIDIA 强大的域随机化（Domain Randomization）和数据采集引擎。
  - 亮点代码： 包含了如何自动随机变换光照、切换物块颜色、采集相机 RGB-D 图像以及自动生成 2D/3D 边界框和语义分割掩码的代码。如果后续要为 ACT、DP 或 VLA 大模型采集海量多样化的仿真训练数据，直接来这里抄代码。
- tutorials/ (新手入门向)
  - 看点： 基础的环境搭建指南（比如如何加地板、加灯光、导入简易资产）。
2. extension_examples (自定义 UI 插件模板)
- 应用场景： 如果你不想每次都在终端敲代码，而是想在 Isaac Sim 的软件界面里，做一个带按钮的控制面板（比如加一个“一键重置抓取物”的按钮）。
- 看点： 这里全都是带 UI 框架的插件（Extension）开发模板。它们继承了特定的基类，需要挂载到软件主循环中运行。
3. exts/omni.isaac.examples/ (软件内置示例的源码)
- 应用场景： 当你在 Isaac Sim 软件顶部的 Isaac Examples 菜单栏里点开了一个官方演示（比如 Franka 机械臂开抽屉、或者四足机器人走路），就可以来这个文件夹翻源码。
- 看点： 它们展示了 Isaac Sim 官方团队是如何组织一个复杂的、带有完整状态机的交互式仿真项目的。
动作图方式实现，sim发布ros2话题
isaacsim的action graph图形化编程
【08.IsaacSim初体验-事件流程图编程初识】https://www.bilibili.com/video/BV1j742zrEEu?vd_source=f0f665afefd1f9899ac346acd7be12f3
isaacsim利用action graph发布ros2 话题
【09.IsaacSim初体验-发布RosTopic】https://www.bilibili.com/video/BV1oW4tz9E7R?vd_source=f0f665afefd1f9899ac346acd7be12f3
URDF浏览网址
https://viewer.robotsfan.com/
机器人及其夹爪URDF文件见package包
脚本操作
使用isaacsim自带的启动器启动
CMake
cd E:\isaac\isaac-sim 
.\python.bat standalone_examples\api\isaacsim.core.api\hello_world.py
2. 强化学习项目对接
2.1 相关资料
项目地址：https://github.com/xiepi/RL_jzrobot
1.环境搭建这个项目的readme已经很清楚了。
2.搭建完了可以试着启动一下这个命令
3.env_isaacsim这个conda环境官网有完整依赖包，可能会有一些不兼容的问题，尤其是下一些强化学习框架的时候，我使用的是rl_games这个，这个在windows能跑
SQL
(base) PS C:\Users\LEGION\Desktop> cd E:\isaac-lab\IsaacLab
(base) PS E:\isaac-lab\IsaacLab>   conda activate env_isaacsim
(env_isaacsim) PS E:\isaac-lab\IsaacLab> .\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\train.py" --task Isaac-Reach-JZ-Bi-v0 --headless --num_envs 64 --max_iterations 6000
[INFO] Using python from: D:\Anaconda3\envs\env_isaacsim\python.exe
[INFO][AppLauncher]: Using device: cuda:0
[INFO][AppLauncher]: Loading experience file: E:\isaac-lab\IsaacLab\apps\isaaclab.python.headless.kit
Loading user config located at: 'd:/anaconda3/envs/env_isaacsim/lib/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json'
[Info] [carb] Logging to file: d:/anaconda3/envs/env_isaacsim/lib/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/kit_20260429_223834.log
1.Reach任务
这是我训练好的权重
'logs\rl_games\jz_bi_reach\2026-04-16_23-37-28\nn\jz_bi_reach.pth'（这个是我的路径，我的这个是在isaaclab安装的那个目录下面：完整的目录是
E:\isaac-lab\IsaacLab\logs\rl_games\jz_bi_reach\2026-04-16_23-37-28\nn\jz_bi_reach.pth）根据自己的放的位置，改一下就行
[jz_bi_reach.pth]
终端输入
CMake
 .\isaaclab.bat -p "E:\jz_robot\jz_isaac_lab\scripts\reinforcement_learning\rl_games\play.py" --task Isaac-Reach-JZ-Bi-v0  --num_envs 1  --checkpoint "logs\rl_games\jz_bi_reach\2026-04-16_23-37-28\nn\jz_bi_reach.pth"
[QQ20260429-232526.mp4]
做了一版本3dof的，现在这版本还没有加入6dof，主要有两个问题
1.机器人的urdf尤其是夹爪的部分，是镜像的tf坐标。我看网上openarm的urdf是统一右手的tf坐标系
2.强化学习训练的，生成的目标点位，有的时候那个关节和末端的方向，根本转不到那去。在机器人前面一大块是没问题的，但是涉及到比较远的地方，再变个夹爪方向，就不好变了。
后续加入6dof功能可以针对上面的问题进行修改
2.Grasp任务
已经搭建好了平台以及奖励函数，可能得多训练几轮然后再调整奖励函数

3.Drawer任务
已经搭建好了桌子和抽屉以及奖励函数，可能得多训练几轮然后再调整奖励函数


3.1 具身模型github
Openarm(这个里面有开箱子的强化学习demo，单臂和双臂的运动控制强化学习)
https://github.com/enactic/openarm
arms_ros2_control（b站上有这个博主的讲解）
https://github.com/fiveages-sim/arms_ros2_control#




ACT
https://github.com/tonyzhaozh/act
DP
https://diffusion-policy.cs.columbia.edu/
https://github.com/real-stanford/diffusion_policy
Groot
https://huggingface.co/docs/lerobot/installation
π系列
https://github.com/Physical-Intelligence/openpi
https://www.pi.website/blog/pi05
Lingbot
https://github.com/Robbyant/lingbot-depth
https://github.com/Robbyant/lingbot-vla
https://github.com/Robbyant/lingbot-world
https://github.com/Robbyant/lingbot-va