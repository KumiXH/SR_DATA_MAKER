from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config cannot start a valid run."""


def validate_config(config: dict[str, Any]) -> None:
    for key in ("name", "runtime", "paths", "source", "tasks"):
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")

    paths = config["paths"]
    input_root = Path(paths.get("input_root", ""))
    if not input_root.exists():
        raise ConfigError(f"input_root does not exist: {input_root}")
    if not input_root.is_dir():
        raise ConfigError(f"input_root is not a directory: {input_root}")

    output_root = Path(paths.get("output_root", ""))
    if not output_root:
        raise ConfigError("paths.output_root is required")
    output_root.mkdir(parents=True, exist_ok=True)

    source_type = config["source"].get("type")
    if not source_type:
        raise ConfigError("source.type is required")

    tasks = config["tasks"]
    if not isinstance(tasks, list):
        raise ConfigError("tasks must be a list")
    for task in tasks:
        _validate_task(task)


def _validate_task(task: dict[str, Any]) -> None:
    if task.get("enabled", True) is False:
        return
    for key in ("name", "type", "runner"):
        if key not in task:
            raise ConfigError(f"enabled task missing {key}")
    runner_type = task["runner"].get("type") if isinstance(task.get("runner"), dict) else None
    if not runner_type:
        raise ConfigError(f"task {task.get('name', '<unnamed>')} missing runner.type")
    target_type = (task.get("output") or {}).get("target_type")
    if target_type and target_type not in {"real_gt", "pseudo_gt", "restored_output"}:
        raise ConfigError(f"invalid target_type: {target_type}")
