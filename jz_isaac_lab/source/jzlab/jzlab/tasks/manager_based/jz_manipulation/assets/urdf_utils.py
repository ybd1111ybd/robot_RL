"""Helpers for preparing Isaac-compatible JZ URDF assets."""

from __future__ import annotations

from pathlib import Path

from jzlab import JZ_DESCRIPTION_DIR

from .. import JZ_MANIPULATION_ROOT_DIR


SOURCE_URDF_PATH = JZ_DESCRIPTION_DIR / "robot_urdf" / "urdf" / "robot_model_isaac_v2.urdf"
RESOLVED_URDF_DIR = JZ_MANIPULATION_ROOT_DIR / "generated"
RESOLVED_URDF_PATH = RESOLVED_URDF_DIR / "robot_model_isaac_v2.resolved.urdf"
_PACKAGE_PREFIX = "package://jz_robot_description/"


def get_resolved_urdf_path(force: bool = False) -> Path:
    """Return a URDF with package:// mesh paths rewritten to local absolute paths."""

    source_text = SOURCE_URDF_PATH.read_text(encoding="utf-8")
    replacement_prefix = f"{(JZ_DESCRIPTION_DIR / 'robot_urdf').resolve().as_posix()}/"
    resolved_text = source_text.replace(_PACKAGE_PREFIX, replacement_prefix)

    if force or not RESOLVED_URDF_PATH.is_file() or RESOLVED_URDF_PATH.read_text(encoding="utf-8") != resolved_text:
        RESOLVED_URDF_DIR.mkdir(parents=True, exist_ok=True)
        RESOLVED_URDF_PATH.write_text(resolved_text, encoding="utf-8")

    return RESOLVED_URDF_PATH
