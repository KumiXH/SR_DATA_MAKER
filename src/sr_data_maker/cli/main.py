from __future__ import annotations

import argparse
import json
from pathlib import Path

from sr_data_maker.config.loader import load_config
from sr_data_maker.config.validator import validate_config
from sr_data_maker.orchestration.executor import PipelineExecutor
from sr_data_maker.setup.face_teacher import (
    print_face_teacher_setup_summary,
    setup_codeformer_from_config,
    setup_gfpgan_from_config,
    setup_vqfr_from_config,
)
from sr_data_maker.setup.pytorch_teacher import print_teacher_setup_summary, setup_hat_from_config, setup_swinir_from_config
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
    swinir = setup_sub.add_parser("swinir")
    swinir.add_argument("--config", required=True)
    swinir.add_argument("--project-root", default=".")
    hat = setup_sub.add_parser("hat")
    hat.add_argument("--config", required=True)
    hat.add_argument("--project-root", default=".")
    gfpgan = setup_sub.add_parser("gfpgan")
    gfpgan.add_argument("--config", required=True)
    gfpgan.add_argument("--project-root", default=".")
    codeformer = setup_sub.add_parser("codeformer")
    codeformer.add_argument("--config", required=True)
    codeformer.add_argument("--project-root", default=".")
    vqfr = setup_sub.add_parser("vqfr")
    vqfr.add_argument("--config", required=True)
    vqfr.add_argument("--project-root", default=".")

    args = parser.parse_args()
    if args.command == "run":
        return _run(args.config)
    if args.command == "validate":
        return _validate(args.config)
    if args.command == "setup" and args.setup_target == "realesrgan":
        return _setup_realesrgan(args.config, args.project_root)
    if args.command == "setup" and args.setup_target == "swinir":
        return _setup_swinir(args.config, args.project_root)
    if args.command == "setup" and args.setup_target == "hat":
        return _setup_hat(args.config, args.project_root)
    if args.command == "setup" and args.setup_target == "gfpgan":
        return _setup_gfpgan(args.config, args.project_root)
    if args.command == "setup" and args.setup_target == "codeformer":
        return _setup_codeformer(args.config, args.project_root)
    if args.command == "setup" and args.setup_target == "vqfr":
        return _setup_vqfr(args.config, args.project_root)
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


def _setup_swinir(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_swinir_from_config(config, project_root=project_root)
    print_teacher_setup_summary("swinir", results)
    return 0


def _setup_hat(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_hat_from_config(config, project_root=project_root)
    print_teacher_setup_summary("hat", results)
    return 0


def _setup_gfpgan(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_gfpgan_from_config(config, project_root=project_root)
    print_face_teacher_setup_summary("gfpgan", results)
    return 0


def _setup_codeformer(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_codeformer_from_config(config, project_root=project_root)
    print_face_teacher_setup_summary("codeformer", results)
    return 0


def _setup_vqfr(config_path: str, project_root: str) -> int:
    config = load_config(config_path)
    results = setup_vqfr_from_config(config, project_root=project_root)
    print_face_teacher_setup_summary("vqfr", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
