from __future__ import annotations

from pathlib import Path

from sr_data_maker.cli.main import main
from sr_data_maker.setup.realesrgan import RealESRGANSetup, find_realesrgan_models


def test_find_realesrgan_models_reads_enabled_runner_tasks():
    config = {
        "tasks": [
            {
                "name": "teacher_sr_realesrgan",
                "enabled": True,
                "runner": {"type": "RealESRGANRunner"},
                "model": {
                    "name": "RealESRGAN_x2plus",
                    "weights": "./weights/RealESRGAN_x2plus.pth",
                    "repo_root": "./third_party/Real-ESRGAN",
                    "basicsr_root": "./third_party/BasicSR",
                },
            },
            {
                "name": "disabled_teacher",
                "enabled": False,
                "runner": {"type": "RealESRGANRunner"},
                "model": {"name": "RealESRGAN_x4plus"},
            },
        ],
    }

    models = find_realesrgan_models(config)

    assert len(models) == 1
    assert models[0]["name"] == "RealESRGAN_x2plus"
    assert models[0]["weights"] == "./weights/RealESRGAN_x2plus.pth"


def test_realesrgan_setup_clones_missing_repos_downloads_weight_and_writes_versions(tmp_path):
    calls: list[tuple[str, object]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = RealESRGANSetup(project_root=tmp_path, clone=clone, download=download)
    model = {
        "name": "RealESRGAN_x2plus",
        "weights": "./weights/RealESRGAN_x2plus.pth",
        "repo_root": "./third_party/Real-ESRGAN",
        "basicsr_root": "./third_party/BasicSR",
    }

    result = setup.prepare_model(model)

    assert result["model_name"] == "RealESRGAN_x2plus"
    assert result["repos"]["real_esrgan"]["status"] == "cloned"
    assert result["repos"]["basicsr"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert (tmp_path / "weights" / "RealESRGAN_x2plus.pth").read_bytes() == b"weights"
    assert (tmp_path / "third_party" / "Real-ESRGAN" / "realesrgan" / "version.py").exists()
    assert (tmp_path / "third_party" / "BasicSR" / "basicsr" / "version.py").exists()
    assert calls == [
        (
            "clone",
            "https://github.com/xinntao/Real-ESRGAN.git",
            tmp_path / "third_party" / "Real-ESRGAN",
        ),
        (
            "clone",
            "https://github.com/XPixelGroup/BasicSR.git",
            tmp_path / "third_party" / "BasicSR",
        ),
        (
            "download",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            tmp_path / "weights" / "RealESRGAN_x2plus.pth",
        ),
    ]


def test_realesrgan_setup_is_idempotent_when_resources_exist(tmp_path):
    real_repo = tmp_path / "third_party" / "Real-ESRGAN"
    basic_repo = tmp_path / "third_party" / "BasicSR"
    weights = tmp_path / "weights" / "RealESRGAN_x2plus.pth"
    (real_repo / "realesrgan").mkdir(parents=True)
    (basic_repo / "basicsr").mkdir(parents=True)
    weights.parent.mkdir()
    weights.write_bytes(b"existing")

    setup = RealESRGANSetup(
        project_root=tmp_path,
        clone=lambda _repo_url, _destination: (_ for _ in ()).throw(AssertionError("clone should not run")),
        download=lambda _url, _destination: (_ for _ in ()).throw(AssertionError("download should not run")),
    )

    result = setup.prepare_model(
        {
            "name": "RealESRGAN_x2plus",
            "weights": "./weights/RealESRGAN_x2plus.pth",
            "repo_root": "./third_party/Real-ESRGAN",
            "basicsr_root": "./third_party/BasicSR",
        }
    )

    assert result["repos"]["real_esrgan"]["status"] == "exists"
    assert result["repos"]["basicsr"]["status"] == "exists"
    assert result["weights"]["status"] == "exists"


def test_cli_setup_realesrgan_uses_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    input_root = tmp_path / "raw"
    input_root.mkdir()
    config_path.write_text(
        f"""
name: demo
runtime:
  device: cpu
paths:
  input_root: {input_root.as_posix()}
  output_root: {(tmp_path / "out").as_posix()}
source:
  type: ImageFolderSourceReader
tasks:
  - name: teacher_sr_realesrgan
    enabled: true
    type: TeacherSRGenerator
    runner:
      type: RealESRGANRunner
    model:
      name: RealESRGAN_x2plus
      weights: ./weights/RealESRGAN_x2plus.pth
      repo_root: ./third_party/Real-ESRGAN
      basicsr_root: ./third_party/BasicSR
""",
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_setup(config, project_root):
        calls.append((config, Path(project_root)))
        return [{"model_name": "RealESRGAN_x2plus"}]

    monkeypatch.setattr("sr_data_maker.cli.main.setup_realesrgan_from_config", fake_setup)
    monkeypatch.setattr(
        "sys.argv",
        ["sr-data-maker", "setup", "realesrgan", "--config", str(config_path), "--project-root", str(tmp_path)],
    )

    assert main() == 0
    assert len(calls) == 1
    assert calls[0][1] == tmp_path
    assert calls[0][0]["tasks"][0]["model"]["name"] == "RealESRGAN_x2plus"
