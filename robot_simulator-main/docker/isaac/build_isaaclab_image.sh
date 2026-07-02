#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run: bash ${SCRIPT_DIR}/setup_host.sh" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

cd "${SCRIPT_DIR}"
docker build \
  --build-arg "ISAAC_SIM_IMAGE=${ISAAC_SIM_IMAGE}" \
  --build-arg "ISAAC_LAB_VERSION=${ISAAC_LAB_VERSION:-2.3.2.post1}" \
  -f Dockerfile.isaaclab \
  -t "${ISAAC_LAB_IMAGE:-jz-isaaclab:5.1.0}" \
  .
