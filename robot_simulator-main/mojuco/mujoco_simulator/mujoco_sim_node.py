#!/usr/bin/env python3
"""
MuJoCo Simulation Node for Dual-Arm Robot

This node provides MuJoCo physics simulation with ROS2 interfaces
compatible with the armcontrol package.

ROS2 Interface:
Subscriptions:
    - /arm_left/joint_commands (sensor_msgs/JointState)
    - /arm_right/joint_commands (sensor_msgs/JointState)
    - /body/joint_commands (sensor_msgs/JointState)
Publishers:
    - /arm_left/joint_states (sensor_msgs/JointState)
    - /arm_right/joint_states (sensor_msgs/JointState)
    - /body/joint_states (sensor_msgs/JointState)
Services:
    - enable_robot (std_srvs/Trigger)
    - disable_robot (std_srvs/Trigger)

"""

import os
import threading
import time
import numpy as np
from typing import Optional, Dict, List

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: mujoco not installed. Install with: pip install mujoco")


class MujocoSimNode(Node):
    """MuJoCo Simulation Node with ROS2 interface compatible with armcontrol."""

    # Joint name mappings
    BODY_JOINTS = ['body1', 'body2', 'body3', 'hand1', 'hand2']
    LEFT_ARM_JOINTS = ['left_arm1', 'left_arm2', 'left_arm3', 'left_arm4', 'left_arm5', 'left_arm6', 'left_arm7']
    RIGHT_ARM_JOINTS = ['right_arm1', 'right_arm2', 'right_arm3', 'right_arm4', 'right_arm5', 'right_arm6', 'right_arm7']
    LEFT_ARM_ROS_JOINTS = ['left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6', 'left_joint7']
    RIGHT_ARM_ROS_JOINTS = ['right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6', 'right_joint7']
    LEFT_ARM_LEGACY_ROS_JOINTS = ['left_joint_1', 'left_joint_2', 'left_joint_3', 'left_joint_4', 'left_joint_5', 'left_joint_6', 'left_joint_7']
    RIGHT_ARM_LEGACY_ROS_JOINTS = ['right_joint_1', 'right_joint_2', 'right_joint_3', 'right_joint_4', 'right_joint_5', 'right_joint_6', 'right_joint_7']

    def __init__(self):
        super().__init__('mujoco_sim_node')

        # Declare parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('sim_rate', 100.0)  # Hz
        self.declare_parameter('pub_rate', 50.0)   # Hz
        self.declare_parameter('enable_viewer', True)
        self.declare_parameter('viewer_rate', 60.0)  # Hz
        self.declare_parameter('base_body_name', 'link0')
        # Legacy parameters - kept for compatibility but not used with position servos
        self.declare_parameter('position_control_kp', 100.0)
        self.declare_parameter('position_control_kd', 10.0)
        # Gravity compensation
        self.declare_parameter('gravity_compensation', True)
        self.declare_parameter('gravity_compensation_scale', 1.0)
        self.declare_parameter('gravity_compensation_body_only', True)
        self.declare_parameter('gravity_compensation_body_scale', 1.0)
        self.declare_parameter('gravity_compensation_body1_scale', 1.2)
        self.declare_parameter('gravity_compensation_body2_scale', 1.0)
        self.declare_parameter('gravity_compensation_body3_scale', 1.0)
        # Gravity compensation low-pass filter (0..1). Higher = more responsive, lower = smoother.
        self.declare_parameter('gravity_compensation_filter_alpha', 1.0)
        # Velocity-based attenuation to avoid oscillation when joints move fast
        self.declare_parameter('gravity_compensation_velocity_threshold', 0.0)
        # Gravity compensation ramp (seconds) to avoid startup jitter
        self.declare_parameter('gravity_compensation_start_delay', 0.0)
        self.declare_parameter('gravity_compensation_ramp_time', 0.0)
        # Body joint deadband to reduce small jitter (radians)
        self.declare_parameter('body_position_deadband', 0.0)
        # Contact simulation toggle (disabling contacts can avoid keyframe self-collision drift)
        self.declare_parameter('enable_contact', True)

        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.sim_rate = self.get_parameter('sim_rate').value
        self.pub_rate = self.get_parameter('pub_rate').value
        self.enable_viewer = self.get_parameter('enable_viewer').value
        self.viewer_rate = float(self.get_parameter('viewer_rate').value)
        self.base_body_name = str(self.get_parameter('base_body_name').value)
        self.gravity_compensation = bool(self.get_parameter('gravity_compensation').value)
        self.gravity_compensation_scale = float(self.get_parameter('gravity_compensation_scale').value)
        self.gravity_compensation_body_only = bool(self.get_parameter('gravity_compensation_body_only').value)
        self.gravity_compensation_body_scale = float(self.get_parameter('gravity_compensation_body_scale').value)
        self.gravity_compensation_body1_scale = float(self.get_parameter('gravity_compensation_body1_scale').value)
        self.gravity_compensation_body2_scale = float(self.get_parameter('gravity_compensation_body2_scale').value)
        self.gravity_compensation_body3_scale = float(self.get_parameter('gravity_compensation_body3_scale').value)
        self.gravity_compensation_filter_alpha = float(self.get_parameter('gravity_compensation_filter_alpha').value)
        self.gravity_compensation_velocity_threshold = float(
            self.get_parameter('gravity_compensation_velocity_threshold').value
        )
        self.gravity_compensation_start_delay = float(
            self.get_parameter('gravity_compensation_start_delay').value
        )
        self.gravity_compensation_ramp_time = float(
            self.get_parameter('gravity_compensation_ramp_time').value
        )
        self.body_position_deadband = float(self.get_parameter('body_position_deadband').value)
        self.enable_contact = bool(self.get_parameter('enable_contact').value)
        if self.gravity_compensation_scale < 0.0:
            self.gravity_compensation_scale = 0.0
        if self.gravity_compensation_scale > 2.0:
            self.gravity_compensation_scale = 2.0
        if self.gravity_compensation_body_scale < 0.0:
            self.gravity_compensation_body_scale = 0.0
        if self.gravity_compensation_body_scale > 3.0:
            self.gravity_compensation_body_scale = 3.0
        if self.gravity_compensation_body1_scale < 0.0:
            self.gravity_compensation_body1_scale = 0.0
        if self.gravity_compensation_body1_scale > 3.0:
            self.gravity_compensation_body1_scale = 3.0
        if self.gravity_compensation_body2_scale < 0.0:
            self.gravity_compensation_body2_scale = 0.0
        if self.gravity_compensation_body2_scale > 3.0:
            self.gravity_compensation_body2_scale = 3.0
        if self.gravity_compensation_body3_scale < 0.0:
            self.gravity_compensation_body3_scale = 0.0
        if self.gravity_compensation_body3_scale > 3.0:
            self.gravity_compensation_body3_scale = 3.0
        if self.gravity_compensation_filter_alpha < 0.0:
            self.gravity_compensation_filter_alpha = 0.0
        if self.gravity_compensation_filter_alpha > 1.0:
            self.gravity_compensation_filter_alpha = 1.0
        if self.gravity_compensation_velocity_threshold < 0.0:
            self.gravity_compensation_velocity_threshold = 0.0
        if self.gravity_compensation_start_delay < 0.0:
            self.gravity_compensation_start_delay = 0.0
        if self.gravity_compensation_ramp_time < 0.0:
            self.gravity_compensation_ramp_time = 0.0
        if self.body_position_deadband < 0.0:
            self.body_position_deadband = 0.0
        if self.viewer_rate < 0.0:
            self.viewer_rate = 0.0
        self.viewer_period = 1.0 / self.viewer_rate if self.viewer_rate > 0.0 else 0.0
        self.last_viewer_sync_time = 0.0
        # Note: kp and kd are now defined in the MJCF file for position servos

        # Find model path if not specified
        if not model_path:
            model_path = self._find_model_path()

        self.get_logger().info(f"Loading MuJoCo model from: {model_path}")

        # Initialize MuJoCo
        if not MUJOCO_AVAILABLE:
            self.get_logger().error("MuJoCo not available. Please install mujoco package.")
            return

        try:
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
        except Exception as e:
            self.get_logger().error(f"Failed to load MuJoCo model: {e}")
            return

        if not self.enable_contact:
            self.model.opt.disableflags = int(self.model.opt.disableflags) | int(
                mujoco.mjtDisableBit.mjDSBL_CONTACT
            )
            self.get_logger().warn("Contact simulation is disabled (enable_contact=false)")

        # Get joint indices
        self._setup_joint_mappings()
        self._setup_body_dof_indices()
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.base_body_name)
        if self.base_body_id < 0:
            self.get_logger().warn(
                f"Base body '{self.base_body_name}' not found, target markers will use world frame directly"
            )
        else:
            self.get_logger().info(
                f"Base body for target pose transform: {self.base_body_name} (id={self.base_body_id})"
            )

        # 仅用于可视化：将 workspace_center 小球轻微外移/上移，避免被机身遮挡
        # 注意：不会修改 topic 数据本身，仅影响 MuJoCo UI 中 marker 的显示位置
        self.left_workspace_center_marker_offset_base = np.array([0.0, 0.08, 0.05], dtype=float)
        self.right_workspace_center_marker_offset_base = np.array([0.0, -0.08, 0.05], dtype=float)

        # List all available keyframes
        self.get_logger().info(f"Available keyframes: {self.model.nkey}")
        keyframe_names: List[str] = []
        for i in range(self.model.nkey):
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i)
            keyframe_names.append(key_name or '')
            self.get_logger().info(f"  Keyframe {i}: '{key_name}'")

        # Apply keyframe if specified (or select a reasonable default)
        self.declare_parameter('keyframe_name', '')
        self.declare_parameter('keyframe_zero_velocity', True)
        keyframe_name = self.get_parameter('keyframe_name').value
        zero_keyframe_velocity = self.get_parameter('keyframe_zero_velocity').value

        if not keyframe_name and keyframe_names:
            # Prefer a non-home keyframe if available
            for candidate in keyframe_names:
                if candidate and candidate.lower() != 'home':
                    keyframe_name = candidate
                    break
            if not keyframe_name:
                keyframe_name = keyframe_names[0]
            self.get_logger().info(f"No keyframe_name specified, defaulting to '{keyframe_name}'")

        self.get_logger().info(f"keyframe_name parameter: '{keyframe_name}'")
        if keyframe_name:
            keyframe_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
            self.get_logger().info(f"keyframe_id: {keyframe_id}")
            if keyframe_id >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe_id)
                if zero_keyframe_velocity:
                    self.data.qvel[:] = 0.0
                    if hasattr(self.data, 'qacc'):
                        self.data.qacc[:] = 0.0
                self.get_logger().info(f"Applied keyframe: {keyframe_name}")
                self.get_logger().info(f"qpos after keyframe: {self.data.qpos[:10]}...")
                self.get_logger().info(f"ctrl BEFORE sync: {self.data.ctrl[:10]}...")
                # For position servos, sync ctrl to qpos to prevent movement
                mujoco.mj_forward(self.model, self.data)
                for joint_name, act_idx in self.actuator_indices.items():
                    qpos_idx = self.joint_indices.get(joint_name)
                    if qpos_idx is not None:
                        self.data.ctrl[act_idx] = self.data.qpos[qpos_idx]
                self.get_logger().info(f"ctrl AFTER sync: {self.data.ctrl[:12]}...")
                self.get_logger().info("Synchronized ctrl to qpos for position servos")
            else:
                self.get_logger().warn(f"Keyframe '{keyframe_name}' not found, using default initial state")
        else:
            self.get_logger().info("No keyframe specified, using default initial state")

        # Control state
        self.enabled = True
        self.target_positions: Dict[str, float] = {}
        self.lock = threading.Lock()
        self.initialization_complete = False  # Flag to ignore commands during initialization
        self._init_command_ignored_logged = False

        # Initialize targets to current positions
        mujoco.mj_forward(self.model, self.data)
        for joint_name in self.BODY_JOINTS + self.LEFT_ARM_JOINTS + self.RIGHT_ARM_JOINTS:
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                self.target_positions[joint_name] = self.data.qpos[idx]
            else:
                self.get_logger().warn(f"Joint '{joint_name}' not found in model!")

        # Debug: Print initial target positions
        self.get_logger().info("Initial target positions:")
        for joint_name in self.BODY_JOINTS:
            if joint_name in self.target_positions:
                self.get_logger().info(f"  {joint_name}: {self.target_positions[joint_name]}")

        # Debug: Print left arm target positions
        self.get_logger().info("Left arm target positions:")
        for joint_name in self.LEFT_ARM_JOINTS:
            if joint_name in self.target_positions:
                self.get_logger().info(f"  {joint_name}: {self.target_positions[joint_name]}")

        # Debug: Print right arm target positions
        self.get_logger().info("Right arm target positions:")
        for joint_name in self.RIGHT_ARM_JOINTS:
            if joint_name in self.target_positions:
                self.get_logger().info(f"  {joint_name}: {self.target_positions[joint_name]}")

        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscribers
        self.sub_left_cmd = self.create_subscription(
            JointState, '/arm_left/joint_commands',
            self._left_arm_cmd_callback, qos)
        self.sub_right_cmd = self.create_subscription(
            JointState, '/arm_right/joint_commands',
            self._right_arm_cmd_callback, qos)
        self.sub_body_cmd = self.create_subscription(
            JointState, '/body/joint_commands',
            self._body_cmd_callback, qos)
        self.sub_left_target_pose = self.create_subscription(
            PoseStamped, '/arm_left/target_pose',
            self._left_target_pose_callback, qos)
        self.sub_right_target_pose = self.create_subscription(
            PoseStamped, '/arm_right/target_pose',
            self._right_target_pose_callback, qos)
        self.sub_left_workspace_center = self.create_subscription(
            PoseStamped, '/arm_left/workspace_center',
            self._left_workspace_center_callback, qos)
        self.sub_right_workspace_center = self.create_subscription(
            PoseStamped, '/arm_right/workspace_center',
            self._right_workspace_center_callback, qos)

        # Create publishers
        self.pub_left_state = self.create_publisher(JointState, '/arm_left/joint_states', qos)
        self.pub_right_state = self.create_publisher(JointState, '/arm_right/joint_states', qos)
        self.pub_body_state = self.create_publisher(JointState, '/body/joint_states', qos)

        # Resolve optional mocap marker indices for IK target visualization
        self.left_target_mocap_id = self._get_mocap_id_by_body_name('left_target_marker')
        self.right_target_mocap_id = self._get_mocap_id_by_body_name('right_target_marker')
        self.left_workspace_center_mocap_id = self._get_mocap_id_by_body_name('left_workspace_center_marker')
        self.right_workspace_center_mocap_id = self._get_mocap_id_by_body_name('right_workspace_center_marker')
        if self.left_target_mocap_id >= 0:
            self.get_logger().info("IK marker enabled: left_target_marker")
        else:
            self.get_logger().warn("IK marker missing: left_target_marker")
        if self.right_target_mocap_id >= 0:
            self.get_logger().info("IK marker enabled: right_target_marker")
        else:
            self.get_logger().warn("IK marker missing: right_target_marker")
        if self.left_workspace_center_mocap_id >= 0:
            self.get_logger().info("Workspace center marker enabled: left_workspace_center_marker")
        else:
            self.get_logger().warn("Workspace center marker missing: left_workspace_center_marker")
        if self.right_workspace_center_mocap_id >= 0:
            self.get_logger().info("Workspace center marker enabled: right_workspace_center_marker")
        else:
            self.get_logger().warn("Workspace center marker missing: right_workspace_center_marker")

        # Create services
        self.srv_enable = self.create_service(Trigger, 'enable_robot', self._enable_robot_callback)
        self.srv_disable = self.create_service(Trigger, 'disable_robot', self._disable_robot_callback)

        # Create timers
        self.sim_timer = self.create_timer(1.0 / self.sim_rate, self._simulation_step)
        self.pub_timer = self.create_timer(1.0 / self.pub_rate, self._publish_states)

        # Timer to mark initialization complete after 10 seconds
        self.init_timer = self.create_timer(10.0, self._complete_initialization)

        # Viewer (runs in main thread with simulation)
        self.viewer = None
        if self.enable_viewer:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                if self.viewer_rate > 0.0:
                    self.get_logger().info(f"MuJoCo viewer sync limited to {self.viewer_rate:.1f} Hz")
                else:
                    self.get_logger().info("MuJoCo viewer sync rate limit disabled")
            except Exception as e:
                self.get_logger().warn(f"Failed to launch viewer: {e}")
                self.viewer = None

        self.get_logger().info("MuJoCo simulation node initialized")
        if self.gravity_compensation:
            scope = 'body-only' if self.gravity_compensation_body_only else 'all joints'
            self.get_logger().info(
                f"Gravity compensation enabled ({scope}, scale={self.gravity_compensation_scale:.2f}, "
                f"body_scale={self.gravity_compensation_body_scale:.2f}, "
                f"body1_scale={self.gravity_compensation_body1_scale:.2f}, "
                f"body2_scale={self.gravity_compensation_body2_scale:.2f}, "
                f"body3_scale={self.gravity_compensation_body3_scale:.2f}, "
                f"filter_alpha={self.gravity_compensation_filter_alpha:.2f}, "
                f"vel_thresh={self.gravity_compensation_velocity_threshold:.2f}, "
                f"start_delay={self.gravity_compensation_start_delay:.2f}s, "
                f"ramp_time={self.gravity_compensation_ramp_time:.2f}s)"
            )
        else:
            self.get_logger().info("Gravity compensation disabled")

    def _find_model_path(self) -> str:
        """Find the robot model XML file."""
        # Priority: Use jz_descripetion as the primary source
        possible_paths = [
            '/home/lwtcnecz0100684/workspace/teleop_ws/src/jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml',
            # Fallback paths
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'robot_model.mjcf.xml'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mojuco', 'robot_model.mjcf.xml'),
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path):
                return abs_path

        raise FileNotFoundError("Could not find robot_model.mjcf.xml")

    def _setup_joint_mappings(self):
        """Setup mappings between joint names and MuJoCo indices."""
        self.joint_indices: Dict[str, int] = {}
        self.actuator_indices: Dict[str, int] = {}

        # Map joint names to qpos indices
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                self.joint_indices[joint_name] = self.model.jnt_qposadr[i]

        # Map actuator names to control indices
        # Note: Actuator names now match joint names (body1, left_arm1, etc.)
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name:
                # Actuator name matches joint name directly
                self.actuator_indices[actuator_name] = i

        self.get_logger().info(f"Found {len(self.joint_indices)} joints, {len(self.actuator_indices)} actuators")
        self.get_logger().info(f"Joint names: {list(self.joint_indices.keys())}")
        self.get_logger().info(f"Actuator names: {list(self.actuator_indices.keys())}")

    def _setup_body_dof_indices(self):
        """Cache DOF indices for body joints (for gravity compensation)."""
        self.body_dof_indices: List[int] = []
        self.body_dof_index_map: Dict[str, List[int]] = {}
        for joint_name in self.BODY_JOINTS:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                self.get_logger().warn(f"Body joint '{joint_name}' not found for gravity compensation")
                continue
            dof_start = int(self.model.jnt_dofadr[joint_id])
            # MuJoCo Python bindings may not expose jnt_dofnum; derive from joint type
            joint_type = int(self.model.jnt_type[joint_id])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                dof_count = 6
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                dof_count = 3
            else:
                # HINGE or SLIDE
                dof_count = 1
            dof_indices = list(range(dof_start, dof_start + dof_count))
            self.body_dof_indices.extend(dof_indices)
            self.body_dof_index_map[joint_name] = dof_indices
        # Remove duplicates and keep order stable
        self.body_dof_indices = sorted(set(self.body_dof_indices))
        if self.body_dof_indices:
            self.get_logger().info(f"Body DOF indices for gravity compensation: {self.body_dof_indices}")
        else:
            self.get_logger().warn("No body DOF indices found for gravity compensation")

    def _get_gravity_forces(self) -> np.ndarray:
        """Return gravity-only generalized forces for current qpos."""
        if hasattr(self.data, 'qfrc_grav'):
            return self.data.qfrc_grav

        # Fallback: compute bias forces at zero velocity/acceleration
        if not hasattr(self, '_grav_data'):
            self._grav_data = mujoco.MjData(self.model)

        self._grav_data.qpos[:] = self.data.qpos
        self._grav_data.qvel[:] = 0.0
        if hasattr(self._grav_data, 'qacc'):
            self._grav_data.qacc[:] = 0.0
        mujoco.mj_forward(self.model, self._grav_data)
        return self._grav_data.qfrc_bias

    def _simulation_step(self):
        """Execute one simulation step."""
        if not self.enabled:
            return

        with self.lock:
            dt = 1.0 / self.sim_rate

            # Position servos for all joints: directly set target positions
            # MuJoCo handles the position control internally with kp/kv from MJCF
            for joint_name, target_pos in self.target_positions.items():
                if joint_name in self.actuator_indices:
                    act_idx = self.actuator_indices[joint_name]
                    # Deadband for body joints to reduce micro-oscillation
                    if self.body_position_deadband > 0.0 and joint_name in self.BODY_JOINTS:
                        qpos_idx = self.joint_indices.get(joint_name)
                        if qpos_idx is not None:
                            if abs(target_pos - self.data.qpos[qpos_idx]) < self.body_position_deadband:
                                self.data.ctrl[act_idx] = self.data.qpos[qpos_idx]
                                continue
                    self.data.ctrl[act_idx] = target_pos

            # Gravity compensation: apply bias forces to counteract gravity
            if self.gravity_compensation:
                grav = self._get_gravity_forces()
                if self.gravity_compensation_filter_alpha > 0.0:
                    if not hasattr(self, '_grav_filtered'):
                        self._grav_filtered = grav.copy()
                    else:
                        alpha = self.gravity_compensation_filter_alpha
                        self._grav_filtered *= (1.0 - alpha)
                        self._grav_filtered += alpha * grav
                    grav = self._grav_filtered
                if self.gravity_compensation_velocity_threshold > 0.0:
                    vth = self.gravity_compensation_velocity_threshold
                    vel = np.abs(self.data.qvel)
                    atten = 1.0 / (1.0 + (vel / vth) ** 2)
                    grav = grav * atten
                ramp = 1.0
                if (self.gravity_compensation_start_delay > 0.0 or
                        self.gravity_compensation_ramp_time > 0.0):
                    sim_time = float(self.data.time)
                    if sim_time < self.gravity_compensation_start_delay:
                        ramp = 0.0
                    elif self.gravity_compensation_ramp_time > 0.0:
                        ramp = min(
                            (sim_time - self.gravity_compensation_start_delay) /
                            self.gravity_compensation_ramp_time,
                            1.0
                        )
                if self.gravity_compensation_body_only:
                    self.data.qfrc_applied[:] = 0.0
                    if self.body_dof_indices:
                        scale = self.gravity_compensation_scale * ramp
                        self.data.qfrc_applied[self.body_dof_indices] = (
                            scale * grav[self.body_dof_indices]
                        )
                        # Extra compensation for body joints if requested
                        extra = max(0.0, self.gravity_compensation_body_scale - 1.0) * ramp
                        if extra > 0.0:
                            for joint_name in ('body1', 'body2', 'body3'):
                                dof_ids = self.body_dof_index_map.get(joint_name, [])
                                if dof_ids:
                                    self.data.qfrc_applied[dof_ids] += (
                                        extra * grav[dof_ids]
                                    )
                        # Additional extra just for body1
                        body1_extra = max(
                            0.0,
                            self.gravity_compensation_body1_scale - self.gravity_compensation_body_scale
                        ) * ramp
                        if body1_extra > 0.0:
                            dof_ids = self.body_dof_index_map.get('body1', [])
                            if dof_ids:
                                self.data.qfrc_applied[dof_ids] += (
                                    body1_extra * grav[dof_ids]
                                )
                        # Additional extra just for body2
                        body2_extra = max(
                            0.0,
                            self.gravity_compensation_body2_scale - self.gravity_compensation_body_scale
                        ) * ramp
                        if body2_extra > 0.0:
                            dof_ids = self.body_dof_index_map.get('body2', [])
                            if dof_ids:
                                self.data.qfrc_applied[dof_ids] += (
                                    body2_extra * grav[dof_ids]
                                )
                        # Additional extra just for body3
                        body3_extra = max(
                            0.0,
                            self.gravity_compensation_body3_scale - self.gravity_compensation_body_scale
                        ) * ramp
                        if body3_extra > 0.0:
                            dof_ids = self.body_dof_index_map.get('body3', [])
                            if dof_ids:
                                self.data.qfrc_applied[dof_ids] += (
                                    body3_extra * grav[dof_ids]
                                )
                else:
                    self.data.qfrc_applied[:] = self.gravity_compensation_scale * ramp * grav
            else:
                self.data.qfrc_applied[:] = 0.0

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Sync viewer if available
            if self.viewer is not None and self.viewer.is_running():
                now = time.monotonic()
                if self.viewer_period <= 0.0 or now - self.last_viewer_sync_time >= self.viewer_period:
                    self.viewer.sync()
                    self.last_viewer_sync_time = now

    def _publish_states(self):
        """Publish current joint states."""
        now = self.get_clock().now().to_msg()

        # Publish body joint states
        body_msg = self._create_joint_state_msg(self.BODY_JOINTS, now)
        self.pub_body_state.publish(body_msg)

        # Publish left arm joint states
        left_msg = self._create_joint_state_msg(
            self.LEFT_ARM_JOINTS, now, ros_joint_names=self.LEFT_ARM_ROS_JOINTS)
        self.pub_left_state.publish(left_msg)

        # Publish right arm joint states
        right_msg = self._create_joint_state_msg(
            self.RIGHT_ARM_JOINTS, now, ros_joint_names=self.RIGHT_ARM_ROS_JOINTS)
        self.pub_right_state.publish(right_msg)

    def _create_joint_state_msg(
            self,
            joint_names: List[str],
            stamp,
            ros_joint_names: Optional[List[str]] = None) -> JointState:
        """Create a JointState message for the specified joints."""
        msg = JointState()
        msg.header.stamp = stamp
        msg.header.frame_id = 'base_link'

        positions = []
        velocities = []
        efforts = []

        for joint_name in joint_names:
            if joint_name in self.joint_indices:
                idx = self.joint_indices[joint_name]
                positions.append(float(self.data.qpos[idx]))
                velocities.append(float(self.data.qvel[idx]))
                # Effort from actuator force if available
                if joint_name in self.actuator_indices:
                    act_idx = self.actuator_indices[joint_name]
                    efforts.append(float(self.data.actuator_force[act_idx]))
                else:
                    efforts.append(0.0)
            else:
                positions.append(0.0)
                velocities.append(0.0)
                efforts.append(0.0)

        msg.name = list(ros_joint_names if ros_joint_names is not None else joint_names)
        msg.position = positions
        msg.velocity = velocities
        msg.effort = efforts

        return msg

    def _get_mocap_id_by_body_name(self, body_name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return -1
        mocap_id = int(self.model.body_mocapid[body_id])
        return mocap_id if mocap_id >= 0 else -1

    def _set_target_marker(self, mocap_id: int, msg: PoseStamped,
                           position_offset_base: Optional[np.ndarray] = None):
        if mocap_id < 0:
            return
        pos_base = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z)
        ], dtype=float)
        if position_offset_base is not None:
            pos_base = pos_base + position_offset_base
        quat_local_wxyz = np.array([
            float(msg.pose.orientation.w),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z)
        ], dtype=float)

        with self.lock:
            if self.base_body_id >= 0:
                base_pos_world = self.data.xpos[self.base_body_id]
                base_rot_world = self.data.xmat[self.base_body_id].reshape(3, 3)
                pos_world = base_pos_world + base_rot_world.dot(pos_base)

                base_quat_wxyz = self.data.xquat[self.base_body_id]
                quat_world_wxyz = np.array([
                    base_quat_wxyz[0] * quat_local_wxyz[0] - base_quat_wxyz[1] * quat_local_wxyz[1] - base_quat_wxyz[2] * quat_local_wxyz[2] - base_quat_wxyz[3] * quat_local_wxyz[3],
                    base_quat_wxyz[0] * quat_local_wxyz[1] + base_quat_wxyz[1] * quat_local_wxyz[0] + base_quat_wxyz[2] * quat_local_wxyz[3] - base_quat_wxyz[3] * quat_local_wxyz[2],
                    base_quat_wxyz[0] * quat_local_wxyz[2] - base_quat_wxyz[1] * quat_local_wxyz[3] + base_quat_wxyz[2] * quat_local_wxyz[0] + base_quat_wxyz[3] * quat_local_wxyz[1],
                    base_quat_wxyz[0] * quat_local_wxyz[3] + base_quat_wxyz[1] * quat_local_wxyz[2] - base_quat_wxyz[2] * quat_local_wxyz[1] + base_quat_wxyz[3] * quat_local_wxyz[0],
                ], dtype=float)
            else:
                pos_world = pos_base
                quat_world_wxyz = quat_local_wxyz

            self.data.mocap_pos[mocap_id][0] = float(pos_world[0])
            self.data.mocap_pos[mocap_id][1] = float(pos_world[1])
            self.data.mocap_pos[mocap_id][2] = float(pos_world[2])
            self.data.mocap_quat[mocap_id][0] = float(quat_world_wxyz[0])
            self.data.mocap_quat[mocap_id][1] = float(quat_world_wxyz[1])
            self.data.mocap_quat[mocap_id][2] = float(quat_world_wxyz[2])
            self.data.mocap_quat[mocap_id][3] = float(quat_world_wxyz[3])

    def _left_target_pose_callback(self, msg: PoseStamped):
        self._set_target_marker(self.left_target_mocap_id, msg)

    def _right_target_pose_callback(self, msg: PoseStamped):
        self._set_target_marker(self.right_target_mocap_id, msg)

    def _left_workspace_center_callback(self, msg: PoseStamped):
        self._set_target_marker(
            self.left_workspace_center_mocap_id,
            msg,
            position_offset_base=self.left_workspace_center_marker_offset_base
        )

    def _right_workspace_center_callback(self, msg: PoseStamped):
        self._set_target_marker(
            self.right_workspace_center_mocap_id,
            msg,
            position_offset_base=self.right_workspace_center_marker_offset_base
        )

    def _left_arm_cmd_callback(self, msg: JointState):
        """Handle left arm joint commands."""
        self._apply_joint_commands(
            msg,
            self.LEFT_ARM_JOINTS,
            ros_joint_names=self.LEFT_ARM_ROS_JOINTS,
            legacy_ros_joint_names=self.LEFT_ARM_LEGACY_ROS_JOINTS)

    def _right_arm_cmd_callback(self, msg: JointState):
        """Handle right arm joint commands."""
        self._apply_joint_commands(
            msg,
            self.RIGHT_ARM_JOINTS,
            ros_joint_names=self.RIGHT_ARM_ROS_JOINTS,
            legacy_ros_joint_names=self.RIGHT_ARM_LEGACY_ROS_JOINTS)

    def _body_cmd_callback(self, msg: JointState):
        """Handle body joint commands."""
        self._apply_joint_commands(msg, self.BODY_JOINTS)

    def _complete_initialization(self):
        """Mark initialization as complete and allow joint commands."""
        self.initialization_complete = True
        self.init_timer.cancel()
        self.get_logger().info("Initialization complete - accepting joint commands")

    def _apply_joint_commands(
            self,
            msg: JointState,
            expected_joints: List[str],
            ros_joint_names: Optional[List[str]] = None,
            legacy_ros_joint_names: Optional[List[str]] = None):
        """Apply joint commands from a JointState message."""
        if not self.enabled:
            return

        # Ignore commands during initialization to preserve keyframe position
        if not self.initialization_complete:
            if not self._init_command_ignored_logged:
                self.get_logger().info(
                    "Joint commands are ignored until MuJoCo initialization completes"
                )
                self._init_command_ignored_logged = True
            return

        with self.lock:
            joint_name_map = {name: name for name in expected_joints}
            if ros_joint_names is not None:
                joint_name_map.update({
                    ros_name: internal_name
                    for ros_name, internal_name in zip(ros_joint_names, expected_joints)
                })
            if legacy_ros_joint_names is not None:
                joint_name_map.update({
                    ros_name: internal_name
                    for ros_name, internal_name in zip(legacy_ros_joint_names, expected_joints)
                })

            if len(msg.name) > 0:
                # Use joint names from message
                for i, name in enumerate(msg.name):
                    internal_name = joint_name_map.get(name)
                    if internal_name is not None and i < len(msg.position):
                        self.target_positions[internal_name] = msg.position[i]
            elif len(msg.position) == len(expected_joints):
                # No names, use positions directly
                for i, joint_name in enumerate(expected_joints):
                    self.target_positions[joint_name] = msg.position[i]

    def _enable_robot_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Enable robot service callback."""
        self.enabled = True
        response.success = True
        response.message = "Robot enabled"
        self.get_logger().info("Robot enabled")
        return response

    def _disable_robot_callback(self, request: Trigger.Request, response: Trigger.Response):
        """Disable robot service callback."""
        self.enabled = False
        # Zero all control signals
        self.data.ctrl[:] = 0
        response.success = True
        response.message = "Robot disabled"
        self.get_logger().info("Robot disabled")
        return response

    def destroy_node(self):
        """Cleanup on node destruction."""
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MujocoSimNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
