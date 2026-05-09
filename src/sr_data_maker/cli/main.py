from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_data_maker.config.loader import load_config
from sr_data_maker.config.validator import validate_config
from sr_data_maker.orchestration.executor import PipelineExecutor
from sr_data_maker.setup.realesrgan import print_setup_summary, setup_realesrgan_from_config


def main() -> int:
    parser = argparse.ArgumentParser(prog="sr-data-maker")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("run", "validate", "inspect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=(name in {"run", "validate"}))
        if name == "inspect":
            cmd.add_argument("--dataset", required=True)
    setup = sub.add_parser("setup")
    setup_sub = setup.add_subparsers(dest="setup_target", required=True)
    realesrgan = setup_sub.add_parser("realesrgan")
    realesrgan.add_argument("--config", required=True)
    realesrgan.add_argument("--project-root", default=".")

    args = parser.parse_args()
    if args.command == "run":
        return _run(args.config)
    if args.command == "validate":
        return _validate(args.config)
    if args.command == "setup" and args.setup_target == "realesrgan":
        return _setup_realesrgan(args.config, args.project_root)
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


def _setup_realesrgan(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_realesrgan_from_config(config, project_root=project_root)
    print_setup_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
