#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  cp "${SCRIPT_DIR}/.env.example" "${ENV_FILE}"
  echo "Created ${ENV_FILE}; edit it if this machine uses different paths."
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required_paths=(
  "${ROBOT_SIMULATOR_ROOT}"
  "${JZ_DESCRIPTION_ROOT}"
  "${JZ_ISAAC_LAB_ROOT}"
)

for path in "${required_paths[@]}"; do
  mkdir -p "${path}"
done

cache_paths=(
  "${ISAAC_CACHE_ROOT}/cache/main/ov"
  "${ISAAC_CACHE_ROOT}/cache/main/warp"
  "${ISAAC_CACHE_ROOT}/cache/computecache"
  "${ISAAC_CACHE_ROOT}/config"
  "${ISAAC_CACHE_ROOT}/data/documents"
  "${ISAAC_CACHE_ROOT}/data/Kit"
  "${ISAAC_CACHE_ROOT}/logs"
  "${ISAAC_CACHE_ROOT}/pkg"
)

for path in "${cache_paths[@]}"; do
  mkdir -p "${path}"
done

if [[ "$(id -u)" -eq 0 ]]; then
  chown -R "${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_GID}" "${ISAAC_CACHE_ROOT}" "${JZ_ISAAC_LAB_ROOT}"
elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo chown -R "${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_GID}" "${ISAAC_CACHE_ROOT}" "${JZ_ISAAC_LAB_ROOT}"
else
  chmod -R a+rwX "${ISAAC_CACHE_ROOT}" "${JZ_ISAAC_LAB_ROOT}"
  cat <<EOF
Could not run passwordless sudo to chown directories to ${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_GID}.
Applied a+rwX permissions instead so the Isaac container user can write caches
and the task workspace. For stricter permissions, run:

  sudo chown -R ${ISAAC_CONTAINER_UID}:${ISAAC_CONTAINER_GID} ${ISAAC_CACHE_ROOT} ${JZ_ISAAC_LAB_ROOT}

EOF
fi

echo "Isaac Sim 5.1 host directories are ready."
echo "Cache root: ${ISAAC_CACHE_ROOT}"
echo "Task workspace: ${JZ_ISAAC_LAB_ROOT}"
