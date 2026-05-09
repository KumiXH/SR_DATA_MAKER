from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_data_maker.config.loader import load_config
from sr_data_maker.config.validator import validate_config
from sr_data_maker.orchestration.executor import PipelineExecutor


def main() -> int:
    parser = argparse.ArgumentParser(prog="sr-data-maker")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "validate", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=(name in {"run", "validate"}))
        if name == "inspect":
            cmd.add_argument("--dataset", required=True)

    args = parser.parse_args()
    if args.command == "run":
        return _run(args.config)
    if args.command == "validate":
        return _validate(args.config)
    return _inspect(args.dataset)


def _run(config_path: str) -> int:
    config = load_config(config_path)
    PipelineExecutor().run(config)
    return 0


def _validate(config_path: str) -> int:
    config = load_config(config_path)
    validate_config(config)
    return 0


def _inspect(dataset_root: str) -> int:
    summary = Path(dataset_root) / "manifests" / "run_summary.json"
    if summary.exists():
        print(json.dumps(json.loads(summary.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
