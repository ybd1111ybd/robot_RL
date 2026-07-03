"""List JZ Isaac Lab environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PACKAGE_DIR = PROJECT_ROOT / "source" / "jzlab"
if str(SOURCE_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_PACKAGE_DIR))

ISAACLAB_ROOT = Path(os.environ.get("ISAACLAB_PATH", "")).resolve() if os.environ.get("ISAACLAB_PATH") else None
if ISAACLAB_ROOT:
    for rel_path in ("source/isaaclab", "source/isaaclab_tasks", "source/isaaclab_rl", "source/isaaclab_assets"):
        candidate = ISAACLAB_ROOT / rel_path
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


from isaaclab.app import AppLauncher


app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


import gymnasium as gym
from prettytable import PrettyTable

import jzlab.tasks  # noqa: F401


def main() -> None:
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available JZ Isaac Lab Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "JZ" in task_spec.id:
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
