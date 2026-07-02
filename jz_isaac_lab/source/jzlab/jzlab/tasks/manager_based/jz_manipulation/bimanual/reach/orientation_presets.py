"""Named left/right end-effector orientation presets for JZ bimanual reach."""

from __future__ import annotations

import math
import os


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


DEFAULT_REACH_ORIENTATION_PRESET = "tf_tree_nominal"

# Quaternions are defined in base_link frame.
# This preset matches the user-provided TF semantics:
# - left gripper: blue forward, red down, green outward
# - right gripper: blue forward, red up, green outward
REACH_ORIENTATION_PRESETS: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "tf_tree_nominal": {
        "left": _quat_from_rpy(0.0, math.pi / 2.0, 0.0),
        "right": _quat_from_rpy(-math.pi, -math.pi / 2.0, 0.0),
    },
    "legacy_shared_down_180": {
        "left": _quat_from_rpy(0.0, math.pi, math.pi),
        "right": _quat_from_rpy(0.0, math.pi, math.pi),
    },
}


def get_active_orientation_preset() -> tuple[str, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Resolve the active left/right orientation preset from the environment."""

    preset_name = os.environ.get("JZ_REACH_ORIENTATION_PRESET", DEFAULT_REACH_ORIENTATION_PRESET).strip()
    if not preset_name:
        preset_name = DEFAULT_REACH_ORIENTATION_PRESET
    if preset_name not in REACH_ORIENTATION_PRESETS:
        valid = ", ".join(sorted(REACH_ORIENTATION_PRESETS))
        raise ValueError(
            f"Unsupported JZ_REACH_ORIENTATION_PRESET='{preset_name}'. "
            f"Expected one of: {valid}"
        )
    preset = REACH_ORIENTATION_PRESETS[preset_name]
    return preset_name, preset["left"], preset["right"]
