"""Convert the authoritative JZ URDF into a cached USD for Isaac Lab."""

from __future__ import annotations

import argparse
import contextlib
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


parser = argparse.ArgumentParser(description="Convert the JZ robot URDF into USD.")
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="Force USD regeneration even if the cached file exists.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import carb
import omni.kit.app

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.dict import print_dict

from jzlab.tasks.manager_based.jz_manipulation.assets.urdf_utils import SOURCE_URDF_PATH, get_resolved_urdf_path


WORKSPACE_ROOT = PROJECT_ROOT.parent
USD_DIR = PROJECT_ROOT / "source" / "jzlab" / "jzlab" / "tasks" / "manager_based" / "jz_manipulation" / "usds" / "jz_bimanual"
USD_PATH = USD_DIR / "jz_bimanual.usd"


def main() -> None:
    USD_DIR.mkdir(parents=True, exist_ok=True)
    resolved_urdf_path = get_resolved_urdf_path(force=args_cli.force)

    cfg = UrdfConverterCfg(
        asset_path=str(resolved_urdf_path),
        usd_dir=str(USD_DIR),
        usd_file_name=USD_PATH.name,
        fix_base=True,
        merge_fixed_joints=False,
        force_usd_conversion=args_cli.force,
        make_instanceable=True,
        collision_from_visuals=False,
        self_collision=False,
        collider_type="convex_hull",
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
        ),
    )

    print("-" * 80)
    print(f"Source URDF file: {SOURCE_URDF_PATH}")
    print(f"Resolved URDF file: {resolved_urdf_path}")
    print(f"Output USD file: {USD_PATH}")
    print_dict(cfg.to_dict(), nesting=0)
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"Generated USD file: {converter.usd_path}")

    carb_settings_iface = carb.settings.get_settings()
    local_gui = carb_settings_iface.get("/app/window/enabled")
    livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

    if local_gui or livestream_gui:
        sim_utils.open_stage(converter.usd_path)
        app = omni.kit.app.get_app_interface()
        with contextlib.suppress(KeyboardInterrupt):
            while app.is_running():
                app.update()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
