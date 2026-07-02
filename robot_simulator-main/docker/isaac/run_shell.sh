#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Run: bash ${SCRIPT_DIR}/setup_host.sh" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
docker compose --env-file "${ENV_FILE}" run --rm isaac-sim

