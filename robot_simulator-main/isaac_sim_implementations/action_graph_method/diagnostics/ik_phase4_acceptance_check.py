#!/usr/bin/env python3
"""Phase 4 acceptance check for IK Action Graph + RViz2 topic visibility."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
except ModuleNotFoundError:
    rclpy = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass

    class PoseStamped:  # type: ignore[no-redef]
        pass

    class JointState:  # type: ignore[no-redef]
        pass

    class String:  # type: ignore[no-redef]
        pass

    class TFMessage:  # type: ignore[no-redef]
        pass

    qos_profile_sensor_data = 50  # type: ignore[assignment]


DEFAULT_TARGETS = {
    "left": (0.45, 0.25, 0.85, 1.0, 0.0, 0.0, 0.0),
    "right": (0.45, -0.25, 0.85, 1.0, 0.0, 0.0, 0.0),
}


def _parse_arms(raw: str) -> List[str]:
    arms = [item.strip().lower() for item in raw.split(",") if item.strip()]
    if not arms:
        raise ValueError("arms cannot be empty")
    unknown = [arm for arm in arms if arm not in DEFAULT_TARGETS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    return arms


def _parse_pose(raw: str) -> Tuple[float, float, float, float, float, float, float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != 7:
        raise ValueError("pose must contain 7 comma-separated values: x,y,z,qw,qx,qy,qz")
    return tuple(values)  # type: ignore[return-value]


class IKPhase4AcceptanceNode(Node):
    def __init__(
        self,
        *,
        arms: List[str],
        publish_rate: float,
        frame_id: str,
        joint_states_topic: str,
        tf_topic: str,
        left_target_topic: str,
        right_target_topic: str,
        left_pose_topic: str,
        right_pose_topic: str,
        left_status_topic: str,
        right_status_topic: str,
        left_target: Tuple[float, float, float, float, float, float, float],
        right_target: Tuple[float, float, float, float, float, float, float],
    ) -> None:
        super().__init__("ik_phase4_acceptance_check")
        self._arms = arms
        self._frame_id = frame_id
        self._targets = {"left": left_target, "right": right_target}

        self._target_publishers = {}
        if "left" in arms:
            self._target_publishers["left"] = self.create_publisher(PoseStamped, left_target_topic, 10)
        if "right" in arms:
            self._target_publishers["right"] = self.create_publisher(PoseStamped, right_target_topic, 10)

        self._target_publish_count: Dict[str, int] = {arm: 0 for arm in arms}
        self._current_pose_count: Dict[str, int] = {arm: 0 for arm in arms}
        self._status_count: Dict[str, int] = {arm: 0 for arm in arms}
        self._active_status_count: Dict[str, int] = {arm: 0 for arm in arms}
        self._last_status: Dict[str, Optional[str]] = {arm: None for arm in arms}
        self._first_pose: Dict[str, Optional[Tuple[float, float, float]]] = {arm: None for arm in arms}
        self._last_pose: Dict[str, Optional[Tuple[float, float, float]]] = {arm: None for arm in arms}

        self._joint_states_count = 0
        self._tf_count = 0

        self._topic_subscriptions = [
            self.create_subscription(
                JointState,
                joint_states_topic,
                self._on_joint_states,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                TFMessage,
                tf_topic,
                self._on_tf,
                qos_profile_sensor_data,
            ),
        ]

        if "left" in arms:
            self._topic_subscriptions.append(
                self.create_subscription(PoseStamped, left_pose_topic, self._make_pose_cb("left"), 20)
            )
            self._topic_subscriptions.append(
                self.create_subscription(String, left_status_topic, self._make_status_cb("left"), 20)
            )
        if "right" in arms:
            self._topic_subscriptions.append(
                self.create_subscription(PoseStamped, right_pose_topic, self._make_pose_cb("right"), 20)
            )
            self._topic_subscriptions.append(
                self.create_subscription(String, right_status_topic, self._make_status_cb("right"), 20)
            )

        self._timer = self.create_timer(1.0 / max(publish_rate, 1.0), self._publish_targets)

    def _on_joint_states(self, msg: JointState) -> None:  # noqa: ARG002
        self._joint_states_count += 1

    def _on_tf(self, msg: TFMessage) -> None:  # noqa: ARG002
        self._tf_count += 1

    def _make_pose_cb(self, arm: str):
        def _cb(msg: PoseStamped) -> None:
            position = (
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            )
            self._current_pose_count[arm] += 1
            if self._first_pose[arm] is None:
                self._first_pose[arm] = position
            self._last_pose[arm] = position

        return _cb

    def _make_status_cb(self, arm: str):
        def _cb(msg: String) -> None:
            status = msg.data.strip()
            self._status_count[arm] += 1
            self._last_status[arm] = status
            status_prefix = status.split(",", 1)[0].strip().lower()
            if status_prefix not in {"idle", "target_timeout", ""}:
                self._active_status_count[arm] += 1

        return _cb

    def _publish_targets(self) -> None:
        for arm in self._arms:
            publisher = self._target_publishers[arm]
            x, y, z, qw, qx, qy, qz = self._targets[arm]
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self._frame_id
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = z
            msg.pose.orientation.w = qw
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            publisher.publish(msg)
            self._target_publish_count[arm] += 1

    def build_summary(self) -> Dict[str, object]:
        arms_summary = {}
        for arm in self._arms:
            pose_delta = None
            if self._first_pose[arm] is not None and self._last_pose[arm] is not None:
                fx, fy, fz = self._first_pose[arm]  # type: ignore[misc]
                lx, ly, lz = self._last_pose[arm]  # type: ignore[misc]
                pose_delta = ((lx - fx) ** 2 + (ly - fy) ** 2 + (lz - fz) ** 2) ** 0.5
            arms_summary[arm] = {
                "target_publish_count": self._target_publish_count[arm],
                "ee_current_pose_count": self._current_pose_count[arm],
                "ee_status_count": self._status_count[arm],
                "active_status_count": self._active_status_count[arm],
                "last_status": self._last_status[arm],
                "pose_displacement_m": pose_delta,
            }

        return {
            "global": {
                "joint_states_count": self._joint_states_count,
                "tf_count": self._tf_count,
            },
            "arms": arms_summary,
        }


def _evaluate(summary: Dict[str, object], arms: List[str], require_tf: bool, require_joint_states: bool) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    global_summary = summary["global"]  # type: ignore[index]
    if require_joint_states and global_summary["joint_states_count"] <= 0:  # type: ignore[index]
        failures.append("missing /joint_states samples")
    if require_tf and global_summary["tf_count"] <= 0:  # type: ignore[index]
        failures.append("missing /tf samples")

    arms_summary = summary["arms"]  # type: ignore[index]
    for arm in arms:
        arm_summary = arms_summary[arm]  # type: ignore[index]
        if arm_summary["target_publish_count"] <= 0:  # type: ignore[index]
            failures.append(f"{arm}: no target messages published")
        if arm_summary["ee_current_pose_count"] <= 0:  # type: ignore[index]
            failures.append(f"{arm}: missing ee_current_pose samples")
        if arm_summary["ee_status_count"] <= 0:  # type: ignore[index]
            failures.append(f"{arm}: missing ee_ik_status samples")
        if arm_summary["active_status_count"] <= 0:  # type: ignore[index]
            failures.append(f"{arm}: status never became active/converged")

    return (len(failures) == 0, failures)


def _write_reports(output_dir: Path, label: str, payload: Dict[str, object]) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"ik_phase4_acceptance_{label}_{timestamp}.json"
    md_path = output_dir / f"ik_phase4_acceptance_{label}_{timestamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# IK Phase 4 Acceptance Report",
        "",
        f"- Label: `{label}`",
        f"- Passed: `{payload['passed']}`",
        f"- Duration: `{payload['duration_sec']}` s",
        f"- Arms: `{','.join(payload['arms_checked'])}`",
        "",
        "## Global",
        f"- `/joint_states`: `{payload['summary']['global']['joint_states_count']}` samples",
        f"- `/tf`: `{payload['summary']['global']['tf_count']}` samples",
        "",
        "## Arms",
    ]
    for arm in payload["arms_checked"]:
        arm_summary = payload["summary"]["arms"][arm]
        lines.extend(
            [
                f"### {arm}",
                f"- target publishes: `{arm_summary['target_publish_count']}`",
                f"- ee_current_pose samples: `{arm_summary['ee_current_pose_count']}`",
                f"- ee_ik_status samples: `{arm_summary['ee_status_count']}`",
                f"- active statuses: `{arm_summary['active_status_count']}`",
                f"- last status: `{arm_summary['last_status']}`",
                f"- pose displacement: `{arm_summary['pose_displacement_m']}`",
                "",
            ]
        )
    if payload["failures"]:
        lines.append("## Failures")
        for failure in payload["failures"]:
            lines.append(f"- {failure}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", default="left,right", help="CSV arms to command: left,right")
    parser.add_argument("--duration", type=float, default=6.0, help="Run duration in seconds")
    parser.add_argument("--publish-rate", type=float, default=10.0, help="Target publish rate in Hz")
    parser.add_argument("--frame-id", default="base_link")
    parser.add_argument("--left-target", default=",".join(str(v) for v in DEFAULT_TARGETS["left"]))
    parser.add_argument("--right-target", default=",".join(str(v) for v in DEFAULT_TARGETS["right"]))
    parser.add_argument("--joint-states-topic", default="/joint_states")
    parser.add_argument("--tf-topic", default="/tf")
    parser.add_argument("--left-target-topic", default="/arm_left/ee_target_pose")
    parser.add_argument("--right-target-topic", default="/arm_right/ee_target_pose")
    parser.add_argument("--left-pose-topic", default="/arm_left/ee_current_pose")
    parser.add_argument("--right-pose-topic", default="/arm_right/ee_current_pose")
    parser.add_argument("--left-status-topic", default="/arm_left/ee_ik_status")
    parser.add_argument("--right-status-topic", default="/arm_right/ee_ik_status")
    parser.add_argument("--label", default="phase4")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "results" / "reports"),
    )
    parser.add_argument("--skip-tf", action="store_true", help="Do not require /tf samples for PASS")
    parser.add_argument(
        "--skip-joint-states",
        action="store_true",
        help="Do not require /joint_states samples for PASS",
    )
    args = parser.parse_args()

    if rclpy is None:
        print(
            "rclpy is not available. Please source ROS2 first:\n"
            "  source /opt/ros/humble/setup.bash",
            file=sys.stderr,
        )
        return 1

    try:
        arms = _parse_arms(args.arms)
        left_target = _parse_pose(args.left_target)
        right_target = _parse_pose(args.right_target)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    rclpy.init()
    node = IKPhase4AcceptanceNode(
        arms=arms,
        publish_rate=args.publish_rate,
        frame_id=args.frame_id,
        joint_states_topic=args.joint_states_topic,
        tf_topic=args.tf_topic,
        left_target_topic=args.left_target_topic,
        right_target_topic=args.right_target_topic,
        left_pose_topic=args.left_pose_topic,
        right_pose_topic=args.right_pose_topic,
        left_status_topic=args.left_status_topic,
        right_status_topic=args.right_status_topic,
        left_target=left_target,
        right_target=right_target,
    )

    deadline = time.monotonic() + max(args.duration, 0.5)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        summary = node.build_summary()
        node.destroy_node()
        rclpy.shutdown()

    passed, failures = _evaluate(
        summary,
        arms,
        require_tf=not args.skip_tf,
        require_joint_states=not args.skip_joint_states,
    )
    payload = {
        "label": args.label,
        "passed": passed,
        "failures": failures,
        "duration_sec": args.duration,
        "publish_rate_hz": args.publish_rate,
        "arms_checked": arms,
        "summary": summary,
    }
    json_path, md_path = _write_reports(Path(args.output_dir), args.label, payload)

    print("IK Phase 4 acceptance summary:")
    print(f"  report json: {json_path}")
    print(f"  report md:   {md_path}")
    print(f"  passed:      {passed}")
    print(f"  joint_states samples: {summary['global']['joint_states_count']}")
    print(f"  tf samples:           {summary['global']['tf_count']}")
    for arm in arms:
        arm_summary = summary["arms"][arm]
        print(
            f"  {arm}: pose_samples={arm_summary['ee_current_pose_count']}"
            f" status_samples={arm_summary['ee_status_count']}"
            f" active_statuses={arm_summary['active_status_count']}"
            f" last_status={arm_summary['last_status']}"
        )

    if not passed:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print("PASS: IK Phase 4 acceptance requirements satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
