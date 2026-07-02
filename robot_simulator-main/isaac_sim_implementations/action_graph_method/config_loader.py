from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_toml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import tomllib  # type: ignore[attr-defined]
    except Exception:
        import tomli as tomllib  # type: ignore[import-not-found]

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
