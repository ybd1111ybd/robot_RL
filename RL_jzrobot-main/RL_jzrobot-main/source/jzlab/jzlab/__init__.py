"""JZ Isaac Lab package."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
SOURCE_PACKAGE_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SOURCE_PACKAGE_DIR.parent.parent
DEFAULT_WORKSPACE_DIR = PROJECT_DIR.parent
WORKSPACE_DIR = Path(os.environ.get("JZLAB_WORKSPACE_ROOT", str(DEFAULT_WORKSPACE_DIR))).resolve()

JZ_DESCRIPTION_DIR = WORKSPACE_DIR / "jz_descripetion"
ROBOT_SIMULATOR_DIR = WORKSPACE_DIR / "robot_simulator"
