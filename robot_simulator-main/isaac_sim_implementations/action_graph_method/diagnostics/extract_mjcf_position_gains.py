#!/usr/bin/env python3
"""Extract MuJoCo position actuator gains (kp/kv) from MJCF."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


def detect_repo_root(script_dir: Path) -> Path:
    candidates = [script_dir] + list(script_dir.parents)
    for c in candidates:
        if (c / "robot_simulator").exists() and (c / "jz_descripetion").exists():
            return c
    return script_dir.parents[4]


def parse_actuators(mjcf_path: Path) -> List[Dict[str, object]]:
    root = ET.parse(mjcf_path).getroot()
    rows: List[Dict[str, object]] = []
    for elem in root.findall(".//actuator/position"):
        joint = elem.get("joint")
        if not joint:
            continue
        name = elem.get("name", "")
        kp = elem.get("kp")
        kv = elem.get("kv")
        ctrlrange = elem.get("ctrlrange", "")
        rows.append(
            {
                "actuator_name": name,
                "joint": joint,
                "kp": float(kp) if kp is not None else None,
                "kv": float(kv) if kv is not None else None,
                "ctrlrange": ctrlrange,
            }
        )
    return rows


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    repo_root = detect_repo_root(script_dir)
    default_mjcf = repo_root / "jz_descripetion/robot_urdf/urdf/robot_model.mjcf.xml"
    default_out_dir = script_dir / "results" / "analysis"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjcf", default=str(default_mjcf))
    parser.add_argument("--out-dir", default=str(default_out_dir))
    args = parser.parse_args()

    mjcf_path = Path(args.mjcf).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_actuators(mjcf_path)
    rows_sorted = sorted(rows, key=lambda x: str(x["joint"]))

    csv_path = out_dir / "mjcf_position_gains.csv"
    json_path = out_dir / "mjcf_position_gains.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["actuator_name", "joint", "kp", "kv", "ctrlrange"],
        )
        writer.writeheader()
        writer.writerows(rows_sorted)

    json_path.write_text(
        json.dumps(
            {
                "mjcf_path": str(mjcf_path),
                "count": len(rows_sorted),
                "rows": rows_sorted,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Saved {len(rows_sorted)} actuator rows")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
