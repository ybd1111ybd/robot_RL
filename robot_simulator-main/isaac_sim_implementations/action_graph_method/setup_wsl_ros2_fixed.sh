#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${SCRIPT_DIR}/platform/wsl/setup_wsl_ros2_fixed.sh"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  exec bash "${TARGET}" "$@"
else
  source "${TARGET}" "$@"
fi
