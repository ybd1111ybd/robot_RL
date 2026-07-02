#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)
ACTION_GRAPH_DIR="$REPO_ROOT/robot_simulator/isaac_sim_implementations/action_graph_method"
ACCEPT_SCRIPT="$ACTION_GRAPH_DIR/diagnostics/ik_phase4_acceptance_check.py"
PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}
ARMS=${PHASE4_ARMS:-left,right}
DURATION=${PHASE4_DURATION:-6}
PUBLISH_RATE=${PHASE4_PUBLISH_RATE:-10}
LABEL=${PHASE4_LABEL:-ui_validation}
TOPIC_WAIT_SEC=${PHASE4_TOPIC_WAIT_SEC:-10}
EXTRA_ARGS=()

show_usage() {
  cat <<'EOF'
Usage: run_phase4_acceptance.sh [options] [-- extra_args_for_python_checker]

Options:
  --arms <csv>             Arms to validate, default: left,right
  --duration <sec>         Acceptance duration, default: 6
  --publish-rate <hz>      EE target publish rate, default: 10
  --label <text>           Report label, default: ui_validation
  --topic-wait-sec <sec>   Topic wait timeout, default: 10
  -h, --help               Show this help

Examples:
  bash diagnostics/run_phase4_acceptance.sh
  bash diagnostics/run_phase4_acceptance.sh --arms left --label left_only
  bash diagnostics/run_phase4_acceptance.sh -- --skip-tf
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      show_usage
      exit 0
      ;;
    --arms)
      ARMS="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --publish-rate)
      PUBLISH_RATE="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --topic-wait-sec)
      TOPIC_WAIT_SEC="$2"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -f /mnt/e/jz_robot/env.sh ]]; then
  # shellcheck source=/dev/null
  if [[ $- == *u* ]]; then
    __jz_had_nounset=1
    set +u
  else
    __jz_had_nounset=0
  fi
  source /mnt/e/jz_robot/env.sh
  if [[ ${__jz_had_nounset} -eq 1 ]]; then
    set -u
  fi
  unset __jz_had_nounset
else
  echo "ERROR: missing /mnt/e/jz_robot/env.sh" >&2
  exit 1
fi

if ! command -v ros2 >/dev/null 2>&1; then
  echo "ERROR: ros2 command not found after sourcing /mnt/e/jz_robot/env.sh" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: Python binary not executable: $PYTHON_BIN" >&2
  exit 1
fi

WINDOWS_DIR='E:\jz_robot\robot_simulator\isaac_sim_implementations\action_graph_method'
WINDOWS_CMD='\.\run_with_isaac_fixed.bat jinzhi_ros2_action_graph.py --control-mode auto --domain-id 77'

required_topics=(/joint_states /tf)
if [[ ",$ARMS," == *,left,* ]]; then
  required_topics+=(/arm_left/ee_current_pose /arm_left/ee_ik_status)
fi
if [[ ",$ARMS," == *,right,* ]]; then
  required_topics+=(/arm_right/ee_current_pose /arm_right/ee_ik_status)
fi

echo "=========================================="
echo "Phase 4 Acceptance Helper (WSL)"
echo "=========================================="
echo "Windows step (start first):"
echo "  cd /d $WINDOWS_DIR"
echo "  $WINDOWS_CMD"
echo ""
echo "WSL settings:"
echo "  PYTHON_BIN=$PYTHON_BIN"
echo "  ARMS=$ARMS"
echo "  DURATION=$DURATION"
echo "  PUBLISH_RATE=$PUBLISH_RATE"
echo "  LABEL=$LABEL"
echo ""
echo "Waiting up to ${TOPIC_WAIT_SEC}s for required topic samples..."

if ! "$PYTHON_BIN" - "$TOPIC_WAIT_SEC" "${required_topics[@]}" <<'PY'
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

TOPIC_TYPES = {
    "/joint_states": JointState,
    "/tf": TFMessage,
    "/arm_left/ee_current_pose": PoseStamped,
    "/arm_right/ee_current_pose": PoseStamped,
    "/arm_left/ee_ik_status": String,
    "/arm_right/ee_ik_status": String,
}


class TopicWaitNode(Node):
    def __init__(self, topics: list[str]) -> None:
        super().__init__("phase4_topic_waiter")
        self.counts: dict[str, int] = {topic: 0 for topic in topics}
        self._topic_subscriptions = []
        for topic in topics:
            msg_type = TOPIC_TYPES.get(topic)
            if msg_type is None:
                continue
            self._topic_subscriptions.append(
                self.create_subscription(msg_type, topic, self._make_cb(topic), 20)
            )

    def _make_cb(self, topic: str):
        def _cb(_msg) -> None:
            self.counts[topic] += 1

        return _cb

    def all_ready(self) -> bool:
        return all(count > 0 for count in self.counts.values())


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: wait_topics.py <timeout_sec> <topic1> [topic2...]", file=sys.stderr)
        return 2
    timeout_sec = float(sys.argv[1])
    topics = sys.argv[2:]
    rclpy.init(args=None)
    node = TopicWaitNode(topics)
    deadline = time.monotonic() + max(timeout_sec, 0.5)
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.all_ready():
                print("Topic samples ready:")
                for topic, count in node.counts.items():
                    print(f"  {topic}: {count}")
                return 0
    finally:
        counts = dict(node.counts)
        node.destroy_node()
        rclpy.shutdown()
    print("ERROR: required topic samples not received before timeout", file=sys.stderr)
    for topic, count in counts.items():
        print(f"  {topic}: {count}", file=sys.stderr)
    return 2


raise SystemExit(main())
PY
then
  exit 2
fi

echo "Topics ready. Running acceptance check..."
exec "$PYTHON_BIN" "$ACCEPT_SCRIPT" \
  --arms "$ARMS" \
  --duration "$DURATION" \
  --publish-rate "$PUBLISH_RATE" \
  --label "$LABEL" \
  "${EXTRA_ARGS[@]}"
