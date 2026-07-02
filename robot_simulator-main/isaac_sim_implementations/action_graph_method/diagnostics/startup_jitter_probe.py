#!/usr/bin/env python3
"""Collect startup jitter metrics from ROS2 joint command/state topics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ModuleNotFoundError:
    rclpy = None  # type: ignore[assignment]

    class Node:  # type: ignore[no-redef]
        pass

    class JointState:  # type: ignore[no-redef]
        pass


def _parse_csv_text(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_csv_float(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


@dataclass
class Sample:
    t: float
    positions: Dict[str, Optional[float]]
    errors: Dict[str, Optional[float]]


class StartupJitterProbe(Node):
    def __init__(
        self,
        *,
        cmd_topic: str,
        state_topic: str,
        joint_names: List[str],
        target_positions: List[float],
        start_delay: float,
        publish_rate: float,
        publish_duration: float,
        record_duration: float,
    ) -> None:
        super().__init__("startup_jitter_probe")
        self._cmd_topic = cmd_topic
        self._state_topic = state_topic
        self._joint_names = joint_names
        self._target_map = dict(zip(joint_names, target_positions))
        self._start_delay = start_delay
        self._publish_rate = publish_rate
        self._publish_duration = publish_duration
        self._record_duration = record_duration

        self._pub = self.create_publisher(JointState, self._cmd_topic, 10)
        self._sub = self.create_subscription(
            JointState, self._state_topic, self._on_state, 50
        )
        self._tick_timer = self.create_timer(0.01, self._tick)
        self._pub_timer = None

        self._boot_time = time.monotonic()
        self._command_start_time: Optional[float] = None
        self._publish_end_time: Optional[float] = None
        self._record_end_time: Optional[float] = None
        self._done = False
        self._publish_count = 0
        self.samples: List[Sample] = []

    @property
    def done(self) -> bool:
        return self._done

    @property
    def publish_count(self) -> int:
        return self._publish_count

    def _tick(self) -> None:
        now = time.monotonic()

        if self._command_start_time is None:
            if (now - self._boot_time) >= self._start_delay:
                self._command_start_time = now
                self._publish_end_time = now + self._publish_duration
                self._record_end_time = now + self._record_duration
                self.get_logger().info(
                    (
                        f"Start publishing to {self._cmd_topic} after delay "
                        f"{self._start_delay:.3f}s"
                    )
                )
                period = 1.0 / max(self._publish_rate, 1e-6)
                self._pub_timer = self.create_timer(period, self._publish_once)
                self._publish_once()
            return

        if self._record_end_time is not None and now >= self._record_end_time:
            self._done = True

    def _publish_once(self) -> None:
        now = time.monotonic()
        if self._publish_end_time is not None and now > self._publish_end_time:
            if self._pub_timer is not None:
                self._pub_timer.cancel()
                self._pub_timer = None
            return

        msg = JointState()
        msg.name = list(self._joint_names)
        msg.position = [self._target_map[name] for name in self._joint_names]
        self._pub.publish(msg)
        self._publish_count += 1

    def _on_state(self, msg: JointState) -> None:
        if self._command_start_time is None:
            return
        now = time.monotonic()
        t = now - self._command_start_time
        lookup = dict(zip(msg.name, msg.position))
        positions: Dict[str, Optional[float]] = {}
        errors: Dict[str, Optional[float]] = {}
        for joint in self._joint_names:
            q = lookup.get(joint)
            positions[joint] = q
            if q is None:
                errors[joint] = None
            else:
                errors[joint] = abs(self._target_map[joint] - q)
        self.samples.append(Sample(t=t, positions=positions, errors=errors))


def _compute_settling_time(
    samples: List[Sample], joints: List[str], threshold: float
) -> Optional[float]:
    if not samples:
        return None

    within = []
    for sample in samples:
        ok = True
        for joint in joints:
            e = sample.errors.get(joint)
            if e is None or e > threshold:
                ok = False
                break
        within.append(ok)

    earliest: Optional[float] = None
    all_ok_from_here = True
    for idx in range(len(samples) - 1, -1, -1):
        all_ok_from_here = within[idx] and all_ok_from_here
        if all_ok_from_here:
            earliest = samples[idx].t
    return earliest


def _compute_metrics(
    samples: List[Sample],
    joints: List[str],
    window_sec: float,
    settle_threshold: float,
) -> Dict[str, object]:
    window_samples = [s for s in samples if 0.0 <= s.t <= window_sec]
    per_joint: Dict[str, Dict[str, Optional[float]]] = {}
    peaks = []
    for joint in joints:
        errs = [s.errors[joint] for s in window_samples if s.errors[joint] is not None]
        peak = max(errs) if errs else None
        per_joint[joint] = {"peak_error_0_5s": peak}
        if peak is not None:
            peaks.append(peak)

    settling = _compute_settling_time(samples, joints, settle_threshold)
    avg_peak = sum(peaks) / len(peaks) if peaks else None
    max_peak = max(peaks) if peaks else None

    return {
        "sample_count": len(samples),
        "window_sec": window_sec,
        "settle_threshold": settle_threshold,
        "average_peak_error_0_5s": avg_peak,
        "max_peak_error_0_5s": max_peak,
        "settling_time_sec": settling,
        "per_joint": per_joint,
    }


def _write_samples_csv(path: Path, samples: List[Sample], joints: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t"]
    for joint in joints:
        fields.append(f"q_{joint}")
    for joint in joints:
        fields.append(f"err_{joint}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {"t": f"{sample.t:.6f}"}
            for joint in joints:
                q = sample.positions[joint]
                e = sample.errors[joint]
                row[f"q_{joint}"] = "" if q is None else f"{q:.9f}"
                row[f"err_{joint}"] = "" if e is None else f"{e:.9f}"
            writer.writerow(row)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cmd-topic", default="/arm_left/joint_commands")
    parser.add_argument("--state-topic", default="/arm_left/joint_states")
    parser.add_argument(
        "--joint-names",
        default=(
            "left_arm_joint1,left_arm_joint2,left_arm_joint3,left_arm_joint4,"
            "left_arm_joint5,left_arm_joint6,left_arm_joint7"
        ),
    )
    parser.add_argument(
        "--target-positions",
        default="0.8,-0.4,0.3,-0.8,0.2,0.4,0.0",
    )
    parser.add_argument("--start-delay", type=float, default=1.0)
    parser.add_argument("--publish-rate", type=float, default=20.0)
    parser.add_argument("--publish-duration", type=float, default=5.0)
    parser.add_argument("--record-duration", type=float, default=8.0)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--settle-threshold", type=float, default=0.03)
    parser.add_argument("--label", default="trial")
    parser.add_argument(
        "--output-dir",
        default=str(
            Path(__file__).resolve().parent / "results" / "runs"
        ),
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if rclpy is None:
        print(
            "rclpy is not available. Please source ROS2 first:\n"
            "  source /opt/ros/humble/setup.bash",
            file=sys.stderr,
        )
        return 1
    joints = _parse_csv_text(args.joint_names)
    targets = _parse_csv_float(args.target_positions)
    if len(joints) == 0:
        print("No joint names specified", file=sys.stderr)
        return 2
    if len(joints) != len(targets):
        print("joint-names and target-positions length mismatch", file=sys.stderr)
        return 2

    rclpy.init()
    probe = StartupJitterProbe(
        cmd_topic=args.cmd_topic,
        state_topic=args.state_topic,
        joint_names=joints,
        target_positions=targets,
        start_delay=args.start_delay,
        publish_rate=args.publish_rate,
        publish_duration=args.publish_duration,
        record_duration=args.record_duration,
    )

    try:
        deadline = time.monotonic() + args.start_delay + args.record_duration + 10.0
        while rclpy.ok() and not probe.done:
            rclpy.spin_once(probe, timeout_sec=0.05)
            if time.monotonic() > deadline:
                print("Timeout waiting for samples", file=sys.stderr)
                break
    except KeyboardInterrupt:
        pass
    finally:
        probe.destroy_node()
        rclpy.shutdown()

    if not probe.samples:
        print("No joint_state samples captured", file=sys.stderr)
        return 1

    metrics = _compute_metrics(
        probe.samples,
        joints,
        window_sec=args.window_sec,
        settle_threshold=args.settle_threshold,
    )
    out_base = Path(args.output_dir).resolve()
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_base / f"{args.label}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "samples.csv"
    json_path = run_dir / "metrics.json"
    _write_samples_csv(csv_path, probe.samples, joints)

    payload = {
        "metadata": {
            "cmd_topic": args.cmd_topic,
            "state_topic": args.state_topic,
            "joint_names": joints,
            "target_positions": targets,
            "start_delay": args.start_delay,
            "publish_rate": args.publish_rate,
            "publish_duration": args.publish_duration,
            "record_duration": args.record_duration,
            "published_messages": probe.publish_count,
        },
        "metrics": metrics,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved samples: {csv_path}")
    print(f"Saved metrics: {json_path}")
    print(
        "Summary: "
        f"avg_peak_0_5s={metrics['average_peak_error_0_5s']}, "
        f"max_peak_0_5s={metrics['max_peak_error_0_5s']}, "
        f"settling_time={metrics['settling_time_sec']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
