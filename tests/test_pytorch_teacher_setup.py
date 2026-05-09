from __future__ import annotations

from pathlib import Path

import pytest

from sr_data_maker.cli.main import main
from sr_data_maker.setup.pytorch_teacher import (
    HAT_REPO_URL,
    SWINIR_REPO_URL,
    PyTorchTeacherSetup,
    find_teacher_models,
)


def test_find_teacher_models_reads_enabled_adapter_tasks():
    config = {
        "tasks": [
            {
                "enabled": True,
                "runner": {"type": "SwinIRAdapter"},
                "model": {"name": "SwinIR_x2_classical", "weights": "./weights/swinir.pth"},
            },
            {
                "enabled": True,
                "runner": {"type": "HATAdapter"},
                "model": {"name": "HAT_SRx2", "weights": "./weights/hat.pth"},
            },
            {
                "enabled": False,
                "runner": {"type": "SwinIRAdapter"},
                "model": {"name": "disabled"},
            },
        ]
    }

    assert [item["adapter_type"] for item in find_teacher_models(config, {"SwinIRAdapter", "HATAdapter"})] == [
        "SwinIRAdapter",
        "HATAdapter",
    ]


def test_swinir_setup_clones_repo_and_downloads_weight(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        (destination / "models").mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = PyTorchTeacherSetup(
        project_root=tmp_path,
        adapter_type="SwinIRAdapter",
        default_repo_url=SWINIR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "SwinIR_x2_classical",
            "weights": "./weights/SwinIR_x2_classical.pth",
            "repo_root": "./third_party/SwinIR",
            "download_url": "https://example.test/swinir.pth",
        }
    )

    assert result["model_name"] == "SwinIR_x2_classical"
    assert result["repos"]["main"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert calls == [
        ("clone", SWINIR_REPO_URL, tmp_path / "third_party" / "SwinIR"),
        ("download", "https://example.test/swinir.pth", tmp_path / "weights" / "SwinIR_x2_classical.pth"),
    ]


def test_hat_setup_clones_hat_and_basicsr_then_downloads_weight(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = PyTorchTeacherSetup(
        project_root=tmp_path,
        adapter_type="HATAdapter",
        default_repo_url=HAT_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "HAT_SRx2",
            "weights": "./weights/HAT_SRx2.pth",
            "repo_root": "./third_party/HAT",
            "basicsr_root": "./third_party/BasicSR",
            "download_url": "https://example.test/hat.pth",
        }
    )

    assert result["repos"]["main"]["status"] == "cloned"
    assert result["repos"]["basicsr"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert calls == [
        ("clone", HAT_REPO_URL, tmp_path / "third_party" / "HAT"),
        ("clone", "https://github.com/XPixelGroup/BasicSR.git", tmp_path / "third_party" / "BasicSR"),
        ("download", "https://example.test/hat.pth", tmp_path / "weights" / "HAT_SRx2.pth"),
    ]


def test_setup_requires_download_url_when_weight_is_missing(tmp_path):
    setup = PyTorchTeacherSetup(
        project_root=tmp_path,
        adapter_type="SwinIRAdapter",
        default_repo_url=SWINIR_REPO_URL,
        clone=lambda _repo_url, destination: destination.mkdir(parents=True),
        download=lambda _url, _destination: None,
    )

    with pytest.raises(ValueError, match="download_url"):
        setup.prepare_model(
            {
                "name": "SwinIR_x2_classical",
                "weights": "./weights/SwinIR_x2_classical.pth",
                "repo_root": "./third_party/SwinIR",
            }
        )


def test_cli_setup_swinir_uses_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
tasks:
  - enabled: true
    runner:
      type: SwinIRAdapter
    model:
      name: SwinIR_x2_classical
      weights: ./weights/SwinIR_x2_classical.pth
      repo_root: ./third_party/SwinIR
      download_url: https://example.test/swinir.pth
""",
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_setup(config, project_root):
        calls.append((config, Path(project_root)))
        return [{"model_name": "SwinIR_x2_classical"}]

    monkeypatch.setattr("sr_data_maker.cli.main.setup_swinir_from_config", fake_setup)
    monkeypatch.setattr(
        "sys.argv",
        ["sr-data-maker", "setup", "swinir", "--config", str(config_path), "--project-root", str(tmp_path)],
    )

    assert main() == 0
    assert len(calls) == 1
    assert calls[0][1] == tmp_path
