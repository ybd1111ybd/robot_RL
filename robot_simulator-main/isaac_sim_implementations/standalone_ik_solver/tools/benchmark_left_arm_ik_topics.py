#!/usr/bin/env python3
"""Benchmark builtin IK vs standalone IK topic outputs for left-arm EE targets."""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String


LEFT_TARGET_TOPIC = "/arm_left/ee_target_pose"
BUILTIN_CMD_TOPIC = "/arm_left/joint_commands"
STANDALONE_CMD_TOPIC = "/arm_left/joint_commands_standalone"
BUILTIN_STATUS_TOPIC = "/arm_left/ee_ik_status"
STANDALONE_STATUS_TOPIC = "/arm_left/ee_ik_status_standalone"
BUILTIN_POSE_TOPIC = "/arm_left/ee_current_pose"
STANDALONE_POSE_TOPIC = "/arm_left/ee_current_pose_standalone"

DEFAULT_RELATIVE_OFFSETS: Tuple[Tuple[float, float, float], ...] = (
    (0.04, 0.00, 0.00),
    (0.03, 0.03, -0.02),
    (0.02, -0.03, 0.02),
    (-0.02, 0.02, -0.03),
    (0.00, -0.04, 0.00),
)


@dataclass
class CommandSnapshot:
    stamp_sec: float
    positions: List[float]


@dataclass
class PoseSnapshot:
    stamp_sec: float
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]


@dataclass
class SampleRecord:
    index: int
    target: PoseStamped
    publish_wall_sec: float
    builtin_first_cmd: Optional[CommandSnapshot] = None
    standalone_first_cmd: Optional[CommandSnapshot] = None
    builtin_last_cmd: Optional[CommandSnapshot] = None
    standalone_last_cmd: Optional[CommandSnapshot] = None
    builtin_first_status: Optional[str] = None
    standalone_first_status: Optional[str] = None
    builtin_last_status: Optional[str] = None
    standalone_last_status: Optional[str] = None
    builtin_last_pose: Optional[PoseSnapshot] = None
    standalone_last_pose: Optional[PoseSnapshot] = None
    builtin_best_pose_error_m: Optional[float] = None
    standalone_best_pose_error_m: Optional[float] = None


@dataclass
class BenchmarkState:
    active_sample: Optional[SampleRecord] = None
    completed: List[SampleRecord] = field(default_factory=list)


class LeftArmIkBenchmark(Node):
    def __init__(
        self,
        targets: List[PoseStamped],
        settle_sec: float,
        *,
        center_topic: Optional[str] = None,
        relative_offsets: Sequence[Tuple[float, float, float]] = DEFAULT_RELATIVE_OFFSETS,
        warmup_sec: float = 0.5,
    ) -> None:
        super().__init__("left_arm_ik_benchmark")
        self._targets = list(targets)
        self._settle_sec = float(settle_sec)
        self._publish_period_sec = 1.0 / 20.0
        self._warmup_sec = float(max(0.0, warmup_sec))
        self._center_topic = (center_topic or "").strip()
        self._relative_offsets = list(relative_offsets)
        self._state = BenchmarkState()
        self._center_pose: Optional[PoseStamped] = None

        self._target_pub = self.create_publisher(PoseStamped, LEFT_TARGET_TOPIC, 10)
        self.create_subscription(JointState, BUILTIN_CMD_TOPIC, self._on_builtin_cmd, 10)
        self.create_subscription(JointState, STANDALONE_CMD_TOPIC, self._on_standalone_cmd, 10)
        self.create_subscription(String, BUILTIN_STATUS_TOPIC, self._on_builtin_status, 10)
        self.create_subscription(String, STANDALONE_STATUS_TOPIC, self._on_standalone_status, 10)
        self.create_subscription(PoseStamped, BUILTIN_POSE_TOPIC, self._on_builtin_pose, 10)
        self.create_subscription(PoseStamped, STANDALONE_POSE_TOPIC, self._on_standalone_pose, 10)
        if self._center_topic:
            self.create_subscription(PoseStamped, self._center_topic, self._on_center_pose, 10)
        else:
            self.create_subscription(PoseStamped, BUILTIN_POSE_TOPIC, self._on_center_pose, 10)
            self.create_subscription(PoseStamped, STANDALONE_POSE_TOPIC, self._on_center_pose, 10)

    def run(self) -> None:
        if self._warmup_sec > 0.0:
            warmup_deadline = time.time() + self._warmup_sec
            while rclpy.ok() and time.time() < warmup_deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
        if not self._targets:
            self._targets = self._build_relative_targets()
        self.get_logger().info(
            f"starting benchmark: targets={len(self._targets)} settle_sec={self._settle_sec:.2f}"
        )
        for index, target in enumerate(self._targets, start=1):
            record = SampleRecord(index=index, target=target, publish_wall_sec=time.time())
            self._state.active_sample = record
            self._target_pub.publish(target)
            self.get_logger().info(
                "published target "
                f"#{index}: pos=({target.pose.position.x:.3f}, {target.pose.position.y:.3f}, {target.pose.position.z:.3f}) "
                f"quat=({target.pose.orientation.w:.3f}, {target.pose.orientation.x:.3f}, "
                f"{target.pose.orientation.y:.3f}, {target.pose.orientation.z:.3f})"
            )
            deadline = time.time() + self._settle_sec
            next_publish_sec = time.time() + self._publish_period_sec
            while rclpy.ok() and time.time() < deadline:
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.01)
                now = time.time()
                if now >= next_publish_sec:
                    self._target_pub.publish(target)
                    next_publish_sec = now + self._publish_period_sec
            self._state.completed.append(record)
            self._state.active_sample = None
            self._print_summary(record)
            time.sleep(0.2)

    def _on_center_pose(self, msg: PoseStamped) -> None:
        if self._center_pose is None:
            self._center_pose = PoseStamped()
        self._center_pose = msg

    def _build_relative_targets(self) -> List[PoseStamped]:
        deadline = time.time() + 5.0
        while rclpy.ok() and self._center_pose is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._center_pose is None:
            raise RuntimeError(
                "failed to receive current EE pose for benchmark center; "
                "pass --center-topic explicitly if needed"
            )
        center = self._center_pose
        base_position = (
            float(center.pose.position.x),
            float(center.pose.position.y),
            float(center.pose.position.z),
        )
        base_orientation = (
            float(center.pose.orientation.w),
            float(center.pose.orientation.x),
            float(center.pose.orientation.y),
            float(center.pose.orientation.z),
        )
        targets: List[PoseStamped] = []
        for dx, dy, dz in self._relative_offsets:
            targets.append(
                _target(
                    base_position[0] + dx,
                    base_position[1] + dy,
                    base_position[2] + dz,
                    qw=base_orientation[0],
                    qx=base_orientation[1],
                    qy=base_orientation[2],
                    qz=base_orientation[3],
                )
            )
        self.get_logger().info(
            "benchmark center pose: "
            f"topic={self._center_topic or 'auto'} "
            f"pos=({base_position[0]:.3f}, {base_position[1]:.3f}, {base_position[2]:.3f})"
        )
        return targets

    def _on_builtin_cmd(self, msg: JointState) -> None:
        self._capture_command("builtin", msg)

    def _on_standalone_cmd(self, msg: JointState) -> None:
        self._capture_command("standalone", msg)

    def _on_builtin_status(self, msg: String) -> None:
        self._capture_status("builtin", msg.data)

    def _on_standalone_status(self, msg: String) -> None:
        self._capture_status("standalone", msg.data)

    def _on_builtin_pose(self, msg: PoseStamped) -> None:
        self._capture_pose("builtin", msg)

    def _on_standalone_pose(self, msg: PoseStamped) -> None:
        self._capture_pose("standalone", msg)

    def _capture_command(self, source: str, msg: JointState) -> None:
        record = self._state.active_sample
        if record is None:
            return
        now = time.time()
        if now < record.publish_wall_sec:
            return
        snapshot = CommandSnapshot(stamp_sec=now, positions=[float(value) for value in msg.position])
        if source == "builtin":
            if record.builtin_first_cmd is None:
                record.builtin_first_cmd = snapshot
            record.builtin_last_cmd = snapshot
        else:
            if record.standalone_first_cmd is None:
                record.standalone_first_cmd = snapshot
            record.standalone_last_cmd = snapshot

    def _capture_status(self, source: str, status: str) -> None:
        record = self._state.active_sample
        if record is None:
            return
        now = time.time()
        if now < record.publish_wall_sec:
            return
        target_age = _status_target_age(status)
        if target_age is not None and target_age > (self._settle_sec + 0.1):
            return
        if source == "builtin":
            if record.builtin_first_status is None:
                record.builtin_first_status = status
            if not (_is_idle_like_status(status) and record.builtin_last_status is not None):
                record.builtin_last_status = status
        else:
            if record.standalone_first_status is None:
                record.standalone_first_status = status
            if not (_is_idle_like_status(status) and record.standalone_last_status is not None):
                record.standalone_last_status = status

    def _capture_pose(self, source: str, msg: PoseStamped) -> None:
        record = self._state.active_sample
        if record is None:
            return
        now = time.time()
        if now < record.publish_wall_sec:
            return
        snapshot = PoseSnapshot(
            stamp_sec=now,
            position=(
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ),
            orientation=(
                float(msg.pose.orientation.w),
                float(msg.pose.orientation.x),
                float(msg.pose.orientation.y),
                float(msg.pose.orientation.z),
            ),
        )
        target_position = (
            float(record.target.pose.position.x),
            float(record.target.pose.position.y),
            float(record.target.pose.position.z),
        )
        position_error = _position_error_m(snapshot, target_position)
        if source == "builtin":
            record.builtin_last_pose = snapshot
            if record.builtin_best_pose_error_m is None or position_error < record.builtin_best_pose_error_m:
                record.builtin_best_pose_error_m = position_error
        else:
            record.standalone_last_pose = snapshot
            if (
                record.standalone_best_pose_error_m is None
                or position_error < record.standalone_best_pose_error_m
            ):
                record.standalone_best_pose_error_m = position_error

    def _print_summary(self, record: SampleRecord) -> None:
        builtin_latency_ms = _latency_ms(record.publish_wall_sec, record.builtin_first_cmd)
        standalone_latency_ms = _latency_ms(record.publish_wall_sec, record.standalone_first_cmd)
        joint_l2 = _joint_l2(record.builtin_last_cmd, record.standalone_last_cmd)
        joint_linf = _joint_linf(record.builtin_last_cmd, record.standalone_last_cmd)
        builtin_final_pose_error_m = _final_pose_error_m(record.builtin_last_pose, record.target)
        standalone_final_pose_error_m = _final_pose_error_m(record.standalone_last_pose, record.target)
        print(
            f"[sample {record.index:02d}] "
            f"builtin_latency_ms={_fmt_float(builtin_latency_ms)} "
            f"standalone_latency_ms={_fmt_float(standalone_latency_ms)} "
            f"builtin_status={record.builtin_last_status or 'NONE'} "
            f"standalone_status={record.standalone_last_status or 'NONE'} "
            f"builtin_best_pose_err_m={_fmt_float(record.builtin_best_pose_error_m)} "
            f"standalone_best_pose_err_m={_fmt_float(record.standalone_best_pose_error_m)} "
            f"builtin_final_pose_err_m={_fmt_float(builtin_final_pose_error_m)} "
            f"standalone_final_pose_err_m={_fmt_float(standalone_final_pose_error_m)} "
            f"joint_l2={_fmt_float(joint_l2)} "
            f"joint_linf={_fmt_float(joint_linf)}"
        )


def _latency_ms(publish_wall_sec: float, snapshot: Optional[CommandSnapshot]) -> Optional[float]:
    if snapshot is None:
        return None
    return (snapshot.stamp_sec - publish_wall_sec) * 1000.0


def _joint_l2(
    builtin_snapshot: Optional[CommandSnapshot],
    standalone_snapshot: Optional[CommandSnapshot],
) -> Optional[float]:
    if builtin_snapshot is None or standalone_snapshot is None:
        return None
    if len(builtin_snapshot.positions) != len(standalone_snapshot.positions):
        return None
    total = 0.0
    for left, right in zip(builtin_snapshot.positions, standalone_snapshot.positions):
        diff = left - right
        total += diff * diff
    return math.sqrt(total)


def _joint_linf(
    builtin_snapshot: Optional[CommandSnapshot],
    standalone_snapshot: Optional[CommandSnapshot],
) -> Optional[float]:
    if builtin_snapshot is None or standalone_snapshot is None:
        return None
    if len(builtin_snapshot.positions) != len(standalone_snapshot.positions):
        return None
    return max(abs(left - right) for left, right in zip(builtin_snapshot.positions, standalone_snapshot.positions))


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NONE"
    return f"{value:.4f}"


def _is_idle_like_status(status: str) -> bool:
    lowered = status.strip().lower()
    return lowered.startswith("idle") or lowered.startswith("target_timeout")


def _status_target_age(status: str) -> Optional[float]:
    marker = "target_age="
    if marker not in status:
        return None
    suffix = status.split(marker, 1)[1]
    numeric = suffix.split("s", 1)[0].strip()
    try:
        return float(numeric)
    except ValueError:
        return None


def _position_error_m(snapshot: PoseSnapshot, target_position: Tuple[float, float, float]) -> float:
    return math.sqrt(
        (snapshot.position[0] - target_position[0]) ** 2
        + (snapshot.position[1] - target_position[1]) ** 2
        + (snapshot.position[2] - target_position[2]) ** 2
    )


def _final_pose_error_m(snapshot: Optional[PoseSnapshot], target: PoseStamped) -> Optional[float]:
    if snapshot is None:
        return None
    return _position_error_m(
        snapshot,
        (
            float(target.pose.position.x),
            float(target.pose.position.y),
            float(target.pose.position.z),
        ),
    )


def _target(
    x: float,
    y: float,
    z: float,
    qw: float = 1.0,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = "base_link"
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    msg.pose.orientation.w = float(qw)
    msg.pose.orientation.x = float(qx)
    msg.pose.orientation.y = float(qy)
    msg.pose.orientation.z = float(qz)
    return msg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settle-sec",
        type=float,
        default=1.5,
        help="Time to wait after each target publish before summarizing.",
    )
    parser.add_argument(
        "--center-topic",
        default="",
        help=(
            "Current pose topic used as the center for relative benchmark targets. "
            "Default auto-detects from builtin then standalone pose topics."
        ),
    )
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=0.5,
        help="Initial subscriber discovery / warmup time before the first sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = LeftArmIkBenchmark(
        [],
        settle_sec=args.settle_sec,
        center_topic=args.center_topic,
        warmup_sec=args.warmup_sec,
    )
    try:
        node.run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
