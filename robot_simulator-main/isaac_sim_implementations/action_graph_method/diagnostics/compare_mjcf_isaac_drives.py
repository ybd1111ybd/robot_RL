#!/usr/bin/env python3
"""Compare MJCF actuator kp/kv against Isaac imported drive stiffness/damping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    text = str(v).strip()
    if text == "" or text.lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _map_isaac_joint_to_mjcf(isaac_joint: str) -> Optional[str]:
    if isaac_joint.startswith("left_arm_joint"):
        suffix = isaac_joint[len("left_arm_joint") :]
        return f"left_arm{suffix}"
    if isaac_joint.startswith("right_arm_joint"):
        suffix = isaac_joint[len("right_arm_joint") :]
        return f"right_arm{suffix}"
    body_map = {
        "body_joint1": "body3",
        "body_joint2": "body2",
        "body_joint3": "body1",
        "body_joint4": "hand2",
        "body_joint5": "hand1",
    }
    return body_map.get(isaac_joint)


def _load_mjcf_rows(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        joint = str(row.get("joint", "")).strip()
        if not joint:
            continue
        out[joint] = {
            "kp": _to_float(row.get("kp")),
            "kv": _to_float(row.get("kv")),
        }
    return out


def _load_isaac_rows(path: Path) -> Dict[str, Dict[str, Optional[float]]]:
    # Prefer angular drive row for revolute joints.
    out: Dict[str, Dict[str, Optional[float]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            joint = str(row.get("joint_name", "")).strip()
            if not joint:
                continue
            drive_type = str(row.get("drive_type", "")).strip().lower()
            candidate = {
                "drive_type": drive_type,
                "stiffness": _to_float(row.get("stiffness")),
                "damping": _to_float(row.get("damping")),
                "max_force": _to_float(row.get("max_force")),
            }
            current = out.get(joint)
            if current is None:
                out[joint] = candidate
                continue
            # Replace existing row when new row is angular and old isn't.
            if current.get("drive_type") != "angular" and drive_type == "angular":
                out[joint] = candidate
    return out


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    default_out_dir = script_dir / "results" / "analysis"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mjcf-json",
        default=str(default_out_dir / "mjcf_position_gains.json"),
    )
    parser.add_argument(
        "--isaac-drive-csv",
        required=True,
        help="CSV produced by jinzhi_ros2_action_graph.py --dump-drive-csv",
    )
    parser.add_argument("--out-dir", default=str(default_out_dir))
    args = parser.parse_args()

    mjcf_map = _load_mjcf_rows(Path(args.mjcf_json).resolve())
    isaac_map = _load_isaac_rows(Path(args.isaac_drive_csv).resolve())
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for isaac_joint in sorted(isaac_map.keys()):
        mjcf_joint = _map_isaac_joint_to_mjcf(isaac_joint)
        if mjcf_joint is None:
            continue
        mj = mjcf_map.get(mjcf_joint, {})
        isa = isaac_map[isaac_joint]
        kp = _to_float(mj.get("kp"))
        kv = _to_float(mj.get("kv"))
        stiffness = _to_float(isa.get("stiffness"))
        damping = _to_float(isa.get("damping"))
        rows.append(
            {
                "isaac_joint": isaac_joint,
                "mjcf_joint": mjcf_joint,
                "mjcf_kp": kp,
                "mjcf_kv": kv,
                "isaac_drive_type": isa.get("drive_type"),
                "isaac_stiffness": stiffness,
                "isaac_damping": damping,
                "isaac_max_force": _to_float(isa.get("max_force")),
                "stiffness_over_kp": (
                    None if kp in (None, 0.0) or stiffness is None else stiffness / kp
                ),
                "damping_over_kv": (
                    None if kv in (None, 0.0) or damping is None else damping / kv
                ),
            }
        )

    csv_path = out_dir / "mjcf_vs_isaac_drive_compare.csv"
    json_path = out_dir / "mjcf_vs_isaac_drive_compare.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "isaac_joint",
                "mjcf_joint",
                "mjcf_kp",
                "mjcf_kv",
                "isaac_drive_type",
                "isaac_stiffness",
                "isaac_damping",
                "isaac_max_force",
                "stiffness_over_kp",
                "damping_over_kv",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "count": len(rows),
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved comparison rows: {len(rows)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
