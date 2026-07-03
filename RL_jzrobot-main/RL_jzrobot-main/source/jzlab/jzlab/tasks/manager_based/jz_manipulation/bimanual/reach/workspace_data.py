"""Workspace dataset paths and helpers for JZ bimanual reach tasks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REACH_TASK_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = REACH_TASK_DIR / "workspace"
WORKSPACE_DATA_PATH = WORKSPACE_DIR / "reachable_workspace.json"


def _find_project_root() -> Path:
    for parent in REACH_TASK_DIR.parents:
        if (parent / "scripts" / "tools" / "generate_reachable_workspace.py").is_file():
            return parent
    raise FileNotFoundError("Unable to locate jz_isaac_lab project root from reach task directory.")


PROJECT_ROOT = _find_project_root()
WORKSPACE_GENERATOR_PATH = PROJECT_ROOT / "scripts" / "tools" / "generate_reachable_workspace.py"


def ensure_workspace_dataset(path: str | Path | None = None) -> Path:
    """Create the workspace dataset on demand so play/train is self-contained."""

    dataset_path = WORKSPACE_DATA_PATH if path is None else Path(path).expanduser().resolve()
    if dataset_path.is_file():
        return dataset_path

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(WORKSPACE_GENERATOR_PATH), "--output", str(dataset_path)]
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Workspace dataset was not created: {dataset_path}")
    return dataset_path
