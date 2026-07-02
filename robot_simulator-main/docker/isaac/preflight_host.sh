#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "missing .env; run: bash ${SCRIPT_DIR}/setup_host.sh"
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

failures=0

check() {
  local label="$1"
  shift
  if "$@" >/tmp/jz_isaac_preflight.out 2>&1; then
    echo "[OK] ${label}"
  else
    echo "[FAIL] ${label}"
    sed 's/^/  /' /tmp/jz_isaac_preflight.out
    failures=$((failures + 1))
  fi
}

check "docker client" docker --version
check "docker compose" docker compose version
check "docker daemon access" docker info
check "nvidia-smi" nvidia-smi
check "robot simulator path" test -d "${ROBOT_SIMULATOR_ROOT}"
check "JZ description path" test -d "${JZ_DESCRIPTION_ROOT}"
check "MJCF model" test -f "${JZ_DESCRIPTION_ROOT}/robot_urdf/urdf/robot_model.mjcf.xml"
check "Isaac Lab workspace path" test -d "${JZ_ISAAC_LAB_ROOT}"

echo "image=${ISAAC_SIM_IMAGE}"
echo "cache_root=${ISAAC_CACHE_ROOT}"

if [[ "${failures}" -ne 0 ]]; then
  echo "Preflight finished with ${failures} failure(s)."
  exit 1
fi

echo "Preflight passed."

