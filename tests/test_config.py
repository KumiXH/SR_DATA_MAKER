from pathlib import Path

import pytest

from sr_data_maker.config.loader import load_config
from sr_data_maker.config.validator import ConfigError, validate_config


def test_load_config_merges_base_and_child(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("runtime:\n  device: cpu\n  num_workers: 1\n", encoding="utf-8")
    child.write_text("base:\n  - base.yaml\nruntime:\n  num_workers: 3\nname: demo\n", encoding="utf-8")

    config = load_config(child)

    assert config["name"] == "demo"
    assert config["runtime"]["device"] == "cpu"
    assert config["runtime"]["num_workers"] == 3


def test_validate_config_rejects_missing_input_root(tmp_path):
    config = {
        "name": "demo",
        "runtime": {"device": "cpu"},
        "paths": {"input_root": str(tmp_path / "missing"), "output_root": str(tmp_path / "out")},
        "source": {"type": "ImageFolderSourceReader"},
        "tasks": [],
    }

    with pytest.raises(ConfigError, match="input_root"):
        validate_config(config)
