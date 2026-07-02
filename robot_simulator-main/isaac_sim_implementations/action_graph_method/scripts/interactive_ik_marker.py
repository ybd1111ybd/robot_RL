#!/usr/bin/env python3
"""6-DoF interactive marker publisher for IK targets in RViz2."""

from __future__ import annotations

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

try:
    from interactive_markers.interactive_marker_server import InteractiveMarkerServer
    from visualization_msgs.msg import InteractiveMarker, InteractiveMarkerControl, Marker
except ModuleNotFoundError as exc:
    raise SystemExit(
        "interactive_markers not available. Install ROS2 package: ros-humble-interactive-markers"
    ) from exc


VALID_ARMS = {"left", "right"}


class InteractiveIKMarker(Node):
    def __init__(self) -> None:
        super().__init__("interactive_ik_marker")

        self.declare_parameter("arm", "left")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("default_x", 0.45)
        self.declare_parameter("default_y", 0.25)
        self.declare_parameter("default_z", 0.85)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("marker_scale", 0.25)

        arm = str(self.get_parameter("arm").value).strip().lower()
        if arm not in VALID_ARMS:
            raise ValueError(f"Unsupported arm '{arm}', expected one of {sorted(VALID_ARMS)}")

        self.frame_id = str(self.get_parameter("frame_id").value).strip() or "world"
        self.default_x = float(self.get_parameter("default_x").value)
        self.default_y = float(self.get_parameter("default_y").value)
        self.default_z = float(self.get_parameter("default_z").value)
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        if publish_rate <= 0.0:
            raise ValueError("publish_rate must be > 0")

        self.target_pub = self.create_publisher(
            PoseStamped,
            f"/arm_{arm}/ee_target_pose",
            10,
        )
        self.current_pose_sub = self.create_subscription(
            PoseStamped,
            f"/arm_{arm}/ee_current_pose",
            self._on_current_pose,
            20,
        )
        self._marker_name = f"ik_target_{arm}"
        self._seeded_from_current_pose = False

        self.current_target = PoseStamped()
        self.current_target.header.frame_id = self.frame_id
        self.current_target.pose.position.x = self.default_x
        self.current_target.pose.position.y = self.default_y
        self.current_target.pose.position.z = self.default_z
        self.current_target.pose.orientation.w = 1.0

        self.publishing = False
        self.timer = self.create_timer(1.0 / publish_rate, self.publish_target)

        self.server = InteractiveMarkerServer(self, "ik_target")
        self._init_marker(arm)

        self.get_logger().info(f"Interactive IK Marker started for {arm} arm")
        self.get_logger().info("Use the RViz2 Interact tool to move/rotate the marker.")
        self.get_logger().info(
            f"Default target: ({self.default_x}, {self.default_y}, {self.default_z})"
        )
        self.get_logger().info("Press Ctrl+C to stop")

    def _init_marker(self, arm: str) -> None:
        marker = InteractiveMarker()
        marker.header.frame_id = self.frame_id
        marker.name = self._marker_name
        marker.description = f"{arm} ee target"
        marker.scale = max(self.marker_scale, 0.05)
        marker.pose = self.current_target.pose

        box = Marker()
        box.type = Marker.CUBE
        box.scale.x = 0.07
        box.scale.y = 0.07
        box.scale.z = 0.07
        box.color.r = 0.1
        box.color.g = 0.9
        box.color.b = 0.1
        box.color.a = 0.9

        control = InteractiveMarkerControl()
        control.always_visible = True
        control.interaction_mode = InteractiveMarkerControl.MOVE_ROTATE_3D
        control.markers.append(box)
        marker.controls.append(control)

        for axis, orientation in (
            ("x", (1.0, 1.0, 0.0, 0.0)),
            ("y", (1.0, 0.0, 1.0, 0.0)),
            ("z", (1.0, 0.0, 0.0, 1.0)),
        ):
            move_ctrl = InteractiveMarkerControl()
            move_ctrl.name = f"move_{axis}"
            move_ctrl.orientation.w = orientation[0]
            move_ctrl.orientation.x = orientation[1]
            move_ctrl.orientation.y = orientation[2]
            move_ctrl.orientation.z = orientation[3]
            move_ctrl.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            marker.controls.append(move_ctrl)

            rotate_ctrl = InteractiveMarkerControl()
            rotate_ctrl.name = f"rotate_{axis}"
            rotate_ctrl.orientation.w = orientation[0]
            rotate_ctrl.orientation.x = orientation[1]
            rotate_ctrl.orientation.y = orientation[2]
            rotate_ctrl.orientation.z = orientation[3]
            rotate_ctrl.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            marker.controls.append(rotate_ctrl)

        self.server.insert(marker, feedback_callback=self._on_feedback)
        self.server.applyChanges()

    def _on_current_pose(self, msg: PoseStamped) -> None:
        if self._seeded_from_current_pose or self.publishing:
            return
        self._seeded_from_current_pose = True
        self.frame_id = msg.header.frame_id or self.frame_id
        self.current_target.header.frame_id = self.frame_id
        self.current_target.pose = msg.pose
        try:
            self.server.setPose(self._marker_name, msg.pose)
            self.server.applyChanges()
        except Exception as exc:
            self.get_logger().warning(f"Failed to seed interactive marker pose: {exc}")
            return
        self.get_logger().info(
            "Seeded interactive marker from current EE pose: "
            f"frame={self.frame_id}"
        )

    def _on_feedback(self, feedback) -> None:
        self.current_target.header.frame_id = self.frame_id
        self.current_target.pose = feedback.pose
        self.publishing = True
        self.publish_target()

    def publish_target(self) -> None:
        if self.publishing:
            self.current_target.header.stamp = self.get_clock().now().to_msg()
            self.target_pub.publish(self.current_target)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = InteractiveIKMarker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
