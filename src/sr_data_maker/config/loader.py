from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = _read_yaml(config_path)
    bases = data.pop("base", []) or []
    if isinstance(bases, (str, Path)):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = merge_dicts(merged, load_config(base_path))

    return merge_dicts(merged, data)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data
