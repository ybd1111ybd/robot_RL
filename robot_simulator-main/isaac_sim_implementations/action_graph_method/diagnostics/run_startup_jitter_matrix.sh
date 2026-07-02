#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ "${PYTHON_BIN}" == */* ]]; then
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
else
  resolved_py="$(command -v "${PYTHON_BIN}" || true)"
  if [[ -z "${resolved_py}" ]]; then
    echo "Python executable not found in PATH: ${PYTHON_BIN}" >&2
    exit 1
  fi
  PYTHON_BIN="${resolved_py}"
fi

if ! "${PYTHON_BIN}" -c "import rclpy" >/dev/null 2>&1; then
  echo "rclpy is not available in ${PYTHON_BIN}. Please source ROS2 env first." >&2
  exit 1
fi

DELAYS=("1" "5" "15")
for d in "${DELAYS[@]}"; do
  echo "=== running delay ${d}s ==="
  "${PYTHON_BIN}" "${SCRIPT_DIR}/startup_jitter_probe.py" \
    --start-delay "${d}" \
    --label "delay_${d}s" \
    "$@"
done

echo "All startup jitter trials finished."
