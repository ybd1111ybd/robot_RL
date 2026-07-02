#!/usr/bin/env bash
set -euo pipefail

echo "ROBOT_SIMULATOR_ROOT=${ROBOT_SIMULATOR_ROOT:-}"
echo "JZ_DESCRIPTION_ROOT=${JZ_DESCRIPTION_ROOT:-}"
echo "JZ_ISAAC_LAB_ROOT=${JZ_ISAAC_LAB_ROOT:-}"

if [[ ! -x /isaac-sim/python.sh ]]; then
  echo "Missing /isaac-sim/python.sh; this does not look like an Isaac Sim container." >&2
  exit 1
fi

if [[ ! -f "${JZ_DESCRIPTION_ROOT:-}/robot_urdf/urdf/robot_model.mjcf.xml" ]]; then
  echo "Missing MJCF model under JZ_DESCRIPTION_ROOT" >&2
  exit 1
fi

if [[ ! -d "${JZ_ISAAC_LAB_ROOT:-}" ]]; then
  echo "Missing Isaac Lab task workspace: ${JZ_ISAAC_LAB_ROOT:-}" >&2
  exit 1
fi

/isaac-sim/python.sh - <<'PY'
import importlib
import os
from pathlib import Path

model_path = Path(os.environ["JZ_DESCRIPTION_ROOT"]) / "robot_urdf/urdf/robot_model.mjcf.xml"
print(f"model_path={model_path}")

for name in ("isaacsim", "isaaclab", "gymnasium", "rl_games", "torch"):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: ok ({version})")

import torch
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

if find "${JZ_ISAAC_LAB_ROOT}" -mindepth 1 -maxdepth 2 -print -quit | grep -q .; then
  echo "task_workspace=nonempty"
else
  echo "task_workspace=empty; add the JZ Isaac Lab task project before training."
fi
