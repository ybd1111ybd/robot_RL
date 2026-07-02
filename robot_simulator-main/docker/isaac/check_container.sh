#!/usr/bin/env bash
set -euo pipefail

echo "ROBOT_SIMULATOR_ROOT=${ROBOT_SIMULATOR_ROOT:-}"
echo "JZ_DESCRIPTION_ROOT=${JZ_DESCRIPTION_ROOT:-}"
echo "JZ_ISAAC_LAB_ROOT=${JZ_ISAAC_LAB_ROOT:-}"

if [[ ! -d "${ROBOT_SIMULATOR_ROOT:-}" ]]; then
  echo "Missing robot simulator mount: ${ROBOT_SIMULATOR_ROOT:-}" >&2
  exit 1
fi

if [[ ! -f "${JZ_DESCRIPTION_ROOT:-}/robot_urdf/urdf/robot_model.mjcf.xml" ]]; then
  echo "Missing MJCF model under JZ_DESCRIPTION_ROOT" >&2
  exit 1
fi

if [[ -x /isaac-sim/python.sh ]]; then
  PYTHON_BIN=(/isaac-sim/python.sh)
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=(python3)
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=(python)
else
  echo "No Python interpreter found in container." >&2
  exit 1
fi

"${PYTHON_BIN[@]}" - <<'PY'
import os
from pathlib import Path
import xml.etree.ElementTree as ET

model_path = Path(os.environ["JZ_DESCRIPTION_ROOT"]) / "robot_urdf/urdf/robot_model.mjcf.xml"
print(f"model_path={model_path}")
root = ET.parse(model_path).getroot()
print(f"mjcf xml ok: root={root.tag}")

try:
    import mujoco
except Exception as exc:
    print(f"mujoco import skipped: {exc}")
else:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print(f"mujoco model ok: nq={model.nq} nv={model.nv} nu={model.nu}")
PY

if [[ -x /isaac-sim/python.sh ]]; then
  /isaac-sim/python.sh - <<'PY'
try:
    import isaacsim
    print("isaacsim import ok")
except Exception as exc:
    print(f"isaacsim import check failed: {exc}")
PY
else
  echo "/isaac-sim/python.sh not found; image layout may differ from expected Isaac Sim container."
fi
