## 完整启动步骤 (Windows + WSL)

适用于 Action Graph + RViz 交互 IK 的完整启动流程。

### 终端 1 (Windows PowerShell): 启动 Isaac Sim

   ```bat 
   cd E:\jz_robot\robot_simulator\isaac_sim_implementations\action_graph_method
   ./run_with_isaac_fixed.bat --control-mode ee_pose --ik-backend lula
   ```

   注意：
      启动后保持 **Play** 状态，不要点击 Stop。+- Stop 会让 physics tensor view 失效，IK 与关节状态会崩。

### 终端 2 (WSL): 话题烟测 (可选)

   ```bash
    conda deactivate 2>/dev/null || true                                 
    source /mnt/e/jz_robot/env.sh                                        
    export ROS_DOMAIN_ID=77                                              
    15 +python3 /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/action_graph_topic_io_check.py --groups left,right,body --duration 5 --publish-rate 20            
    ```                                                                                                                                       
   ### 终端 3 (WSL): RViz2 Bridge                                       
   ```bash                                                              
   conda deactivate 2>/dev/null || true                                 
   source /mnt/e/jz_robot/env.sh                                        
   export ROS_DOMAIN_ID=77                                              
   ros2 launch /mnt/e/jz_robot/jz_descripetion/robot_urdf/launch/isaac_rviz2_bridge.launch.py                                                
   ```

   ### 终端 4 (WSL): Phase4 验收脚本 (可选)

   ```bash
   conda deactivate 2>/dev/null || true                                 
   source /mnt/e/jz_robot/env.sh                                        
   export ROS_DOMAIN_ID=77                                              
   bash /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method/diagnostics/run_phase4_acceptance.sh --label debug_20260309                                             
   ```

   ### 终端 5 (WSL): 交互式 IK Marker (推荐)
   
   ```bash
   conda deactivate 2>/dev/null || true                                 
   source /mnt/e/jz_robot/env.sh                                        
   cd /mnt/e/jz_robot/robot_simulator/isaac_sim_implementations/action_graph_method                                                          
   bash scripts/launch_rviz_ik_control.sh --arm left --mode marker --frame base_link                                                         
   ```

   ``` bash
   ros2 topic list | rg -n "arm_left|arm_right|ik_target|joint"
   ros2 topic echo /arm_left/ee_target_pose --once
   ros2 topic echo /arm_left/joint_commands_mapped --once
   ```

   ### RViz 里看不到 IK Marker 时

    在 RViz 中添加 `InteractiveMarkers` 显示项，Namespace 设置为 `ik_target`