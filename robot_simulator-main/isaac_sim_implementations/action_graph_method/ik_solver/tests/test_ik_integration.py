#!/usr/bin/env python3
"""
Integration test for IK solver with ROS2 topics.

This script tests the complete IK pipeline:
1. Publish target EE pose
2. Verify IK Bridge computes joint angles
3. Verify joint commands are published
4. Monitor robot motion
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class IKIntegrationTester(Node):
    def __init__(self):
        super().__init__('ik_integration_tester')

        # Publishers for target poses
        self.left_target_pub = self.create_publisher(
            PoseStamped, '/arm_left/ee_target_pose', 10
        )
        self.right_target_pub = self.create_publisher(
            PoseStamped, '/arm_right/ee_target_pose', 10
        )

        # Subscribers for monitoring
        self.left_cmd_sub = self.create_subscription(
            JointState, '/arm_left/joint_commands',
            self.on_left_cmd, 10
        )
        self.right_cmd_sub = self.create_subscription(
            JointState, '/arm_right/joint_commands',
            self.on_right_cmd, 10
        )
        self.left_state_sub = self.create_subscription(
            JointState, '/arm_left/joint_states',
            self.on_left_state, 10
        )
        self.right_state_sub = self.create_subscription(
            JointState, '/arm_right/joint_states',
            self.on_right_state, 10
        )
        self.left_pose_sub = self.create_subscription(
            PoseStamped, '/arm_left/ee_current_pose',
            self.on_left_pose, 10
        )
        self.right_pose_sub = self.create_subscription(
            PoseStamped, '/arm_right/ee_current_pose',
            self.on_right_pose, 10
        )
        self.left_status_sub = self.create_subscription(
            String, '/arm_left/ee_ik_status',
            self.on_left_status, 10
        )
        self.right_status_sub = self.create_subscription(
            String, '/arm_right/ee_ik_status',
            self.on_right_status, 10
        )

        # Data storage
        self.left_cmd_count = 0
        self.right_cmd_count = 0
        self.left_state_count = 0
        self.right_state_count = 0
        self.left_pose_count = 0
        self.right_pose_count = 0

        self.last_left_cmd = None
        self.last_right_cmd = None
        self.last_left_state = None
        self.last_right_state = None
        self.last_left_pose = None
        self.last_right_pose = None
        self.last_left_status = None
        self.last_right_status = None

        self.get_logger().info('IK Integration Tester initialized')

    def on_left_cmd(self, msg):
        self.left_cmd_count += 1
        self.last_left_cmd = msg
        if self.left_cmd_count % 10 == 0:
            self.get_logger().info(f'Left cmd #{self.left_cmd_count}: {len(msg.name)} joints')

    def on_right_cmd(self, msg):
        self.right_cmd_count += 1
        self.last_right_cmd = msg
        if self.right_cmd_count % 10 == 0:
            self.get_logger().info(f'Right cmd #{self.right_cmd_count}: {len(msg.name)} joints')

    def on_left_state(self, msg):
        self.left_state_count += 1
        self.last_left_state = msg

    def on_right_state(self, msg):
        self.right_state_count += 1
        self.last_right_state = msg

    def on_left_pose(self, msg):
        self.left_pose_count += 1
        self.last_left_pose = msg
        if self.left_pose_count % 10 == 0:
            pos = msg.pose.position
            self.get_logger().info(
                f'Left EE pose #{self.left_pose_count}: '
                f'[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]'
            )

    def on_right_pose(self, msg):
        self.right_pose_count += 1
        self.last_right_pose = msg
        if self.right_pose_count % 10 == 0:
            pos = msg.pose.position
            self.get_logger().info(
                f'Right EE pose #{self.right_pose_count}: '
                f'[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}]'
            )

    def on_left_status(self, msg):
        if msg.data != self.last_left_status:
            self.get_logger().info(f'Left status: {msg.data}')
            self.last_left_status = msg.data

    def on_right_status(self, msg):
        if msg.data != self.last_right_status:
            self.get_logger().info(f'Right status: {msg.data}')
            self.last_right_status = msg.data

    def publish_target_pose(self, arm='left', x=0.5, y=0.3, z=0.8):
        """Publish a target end-effector pose."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0

        if arm == 'left':
            self.left_target_pub.publish(msg)
            self.get_logger().info(f'Published left target: [{x:.3f}, {y:.3f}, {z:.3f}]')
        else:
            self.right_target_pub.publish(msg)
            self.get_logger().info(f'Published right target: [{x:.3f}, {y:.3f}, {z:.3f}]')

    def print_status(self):
        """Print current status of all topics."""
        self.get_logger().info('=' * 60)
        self.get_logger().info('IK Integration Test Status:')
        self.get_logger().info(f'  Left arm:')
        self.get_logger().info(f'    Commands received: {self.left_cmd_count}')
        self.get_logger().info(f'    States received: {self.left_state_count}')
        self.get_logger().info(f'    Poses received: {self.left_pose_count}')
        self.get_logger().info(f'    Status: {self.last_left_status}')
        self.get_logger().info(f'  Right arm:')
        self.get_logger().info(f'    Commands received: {self.right_cmd_count}')
        self.get_logger().info(f'    States received: {self.right_state_count}')
        self.get_logger().info(f'    Poses received: {self.right_pose_count}')
        self.get_logger().info(f'    Status: {self.last_right_status}')
        self.get_logger().info('=' * 60)


def main():
    rclpy.init()
    tester = IKIntegrationTester()

    # Wait for connections
    tester.get_logger().info('Waiting 2 seconds for connections...')
    time.sleep(2.0)

    # Test sequence
    try:
        # Test 1: Publish left arm target
        tester.get_logger().info('\n=== Test 1: Left arm target ===')
        tester.publish_target_pose('left', x=0.5, y=0.3, z=0.8)

        # Spin for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3.0:
            rclpy.spin_once(tester, timeout_sec=0.1)

        tester.print_status()

        # Test 2: Publish right arm target
        tester.get_logger().info('\n=== Test 2: Right arm target ===')
        tester.publish_target_pose('right', x=0.5, y=-0.3, z=0.8)

        # Spin for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3.0:
            rclpy.spin_once(tester, timeout_sec=0.1)

        tester.print_status()

        # Test 3: Both arms
        tester.get_logger().info('\n=== Test 3: Both arms ===')
        tester.publish_target_pose('left', x=0.6, y=0.4, z=0.7)
        tester.publish_target_pose('right', x=0.6, y=-0.4, z=0.7)

        # Spin for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5.0:
            rclpy.spin_once(tester, timeout_sec=0.1)

        tester.print_status()

        # Final status
        tester.get_logger().info('\n=== Final Summary ===')
        if tester.left_cmd_count > 0:
            tester.get_logger().info('✓ Left arm IK commands are being published')
        else:
            tester.get_logger().warn('✗ No left arm IK commands received')

        if tester.right_cmd_count > 0:
            tester.get_logger().info('✓ Right arm IK commands are being published')
        else:
            tester.get_logger().warn('✗ No right arm IK commands received')

        if tester.left_state_count > 0:
            tester.get_logger().info('✓ Left arm joint states are being published')
        else:
            tester.get_logger().warn('✗ No left arm joint states received')

        if tester.right_state_count > 0:
            tester.get_logger().info('✓ Right arm joint states are being published')
        else:
            tester.get_logger().warn('✗ No right arm joint states received')

    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
