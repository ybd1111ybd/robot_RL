#!/usr/bin/env python3
"""Smoke-check Action Graph ROS2 topic input/output."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
except ModuleNotFoundError:
    rclpy = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass

    class JointState:  # type: ignore[no-redef]
        pass

    qos_profile_sensor_data = 50  # type: ignore[assignment]


LEFT_JOINTS = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_JOINTS = [f"right_arm_joint{i}" for i in range(1, 8)]
BODY_JOINTS = [f"body_joint{i}" for i in range(1, 6)]

GROUP_CONFIG = {
    "left": {
        "cmd": "/arm_left/joint_commands",
        "state": "/arm_left/joint_states",
        "joints": LEFT_JOINTS,
    },
    "right": {
        "cmd": "/arm_right/joint_commands",
        "state": "/arm_right/joint_states",
        "joints": RIGHT_JOINTS,
    },
    "body": {
        "cmd": "/body/joint_commands",
        "state": "/body/joint_states",
        "joints": BODY_JOINTS,
    },
}


def _parse_groups(raw: str) -> List[str]:
    groups = [g.strip().lower() for g in raw.split(",") if g.strip()]
    unknown = [g for g in groups if g not in GROUP_CONFIG]
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    if not groups:
        raise ValueError("groups cannot be empty")
    return groups


class TopicIOSmoke(Node):
    def __init__(self, groups: List[str], rate_hz: float) -> None:
        super().__init__("action_graph_topic_io_check")
        self._groups = groups
        self._rate_hz = max(rate_hz, 1.0)

        self._cmd_publishers: Dict[str, object] = {}
        self._state_count: Dict[str, int] = {g: 0 for g in groups}
        self._last_state_time: Dict[str, float] = {g: -1.0 for g in groups}

        self._state_subscriptions = []
        for group in groups:
            cfg = GROUP_CONFIG[group]
            self._cmd_publishers[group] = self.create_publisher(JointState, cfg["cmd"], 10)
            self._state_subscriptions.append(
                self.create_subscription(
                    JointState,
                    cfg["state"],
                    self._make_state_cb(group),
                    qos_profile_sensor_data,
                )
            )

        self._timer = self.create_timer(1.0 / self._rate_hz, self._publish_once)

    def _make_state_cb(self, group: str):
        def _cb(msg: JointState) -> None:  # noqa: ARG001
            self._state_count[group] += 1
            self._last_state_time[group] = time.monotonic()

        return _cb

    def _publish_once(self) -> None:
        for group in self._groups:
            cfg = GROUP_CONFIG[group]
            joints = cfg["joints"]
            msg = JointState()
            msg.name = list(joints)
            msg.position = [0.0 for _ in joints]
            self._cmd_publishers[group].publish(msg)

    @property
    def state_count(self) -> Dict[str, int]:
        return self._state_count



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="left,right,body", help="CSV: left,right,body")
    parser.add_argument("--duration", type=float, default=5.0, help="check duration seconds")
    parser.add_argument("--publish-rate", type=float, default=20.0, help="command publish rate")
    args = parser.parse_args()

    if rclpy is None:
        print(
            "rclpy is not available. Please source ROS2 first:\n"
            "  source /opt/ros/humble/setup.bash",
            file=sys.stderr,
        )
        return 1

    try:
        groups = _parse_groups(args.groups)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = TopicIOSmoke(groups, args.publish_rate)

    deadline = time.monotonic() + max(args.duration, 0.5)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        summary = node.state_count
        node.destroy_node()
        rclpy.shutdown()

    print("Action Graph topic IO summary:")
    ok = True
    for group in groups:
        count = summary[group]
        print(f"  {group}: state_samples={count}")
        if count <= 0:
            ok = False

    if not ok:
        print("FAIL: at least one state topic has no samples", file=sys.stderr)
        return 1

    print("PASS: all requested state topics received samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
