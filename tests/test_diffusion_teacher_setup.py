from __future__ import annotations

from pathlib import Path

import pytest

from sr_data_maker.setup.diffusion_teacher import (
    RESSHIFT_REPO_URL,
    STABLESR_REPO_URL,
    SUPIR_REPO_URL,
    DiffusionTeacherSetup,
    find_diffusion_teacher_models,
)


def test_find_diffusion_teacher_models_reads_enabled_matching_tasks():
    config = {
        "tasks": [
            {
                "enabled": True,
                "runner": {"type": "StableSRRunner"},
                "model": {"name": "StableSR_x4", "weights": "./weights/stablesr.safetensors"},
            },
            {
                "enabled": True,
                "runner": {"type": "ResShiftRunner"},
                "model": {"name": "ResShift_x4", "weights": "./weights/resshift.safetensors"},
            },
            {
                "enabled": False,
                "runner": {"type": "SUPIRRunner"},
                "model": {"name": "SUPIR_x4"},
            },
        ]
    }

    models = find_diffusion_teacher_models(config, {"StableSRRunner", "ResShiftRunner", "SUPIRRunner"})

    assert [item["runner_type"] for item in models] == ["StableSRRunner", "ResShiftRunner"]


def test_stablesr_setup_clones_expected_repo_and_downloads_weights(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = DiffusionTeacherSetup(
        project_root=tmp_path,
        runner_type="StableSRRunner",
        default_repo_url=STABLESR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "StableSR_x4",
            "weights": "./weights/stablesr/stablesr_x4.safetensors",
            "vqgan_weights": "./weights/stablesr/vqgan_cfw_00011.ckpt",
            "repo_root": "./third_party/StableSR",
            "download_url": "https://example.test/stablesr.safetensors",
            "vqgan_download_url": "https://example.test/vqgan.ckpt",
        }
    )

    assert result["repos"]["main"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert result["artifacts"]["vqgan"]["status"] == "downloaded"
    assert calls == [
        ("clone", STABLESR_REPO_URL, tmp_path / "third_party" / "StableSR"),
        ("download", "https://example.test/stablesr.safetensors", tmp_path / "weights" / "stablesr" / "stablesr_x4.safetensors"),
        ("download", "https://example.test/vqgan.ckpt", tmp_path / "weights" / "stablesr" / "vqgan_cfw_00011.ckpt"),
    ]


def test_resshift_setup_requires_download_url_when_weight_missing(tmp_path):
    setup = DiffusionTeacherSetup(
        project_root=tmp_path,
        runner_type="ResShiftRunner",
        default_repo_url=RESSHIFT_REPO_URL,
        clone=lambda _repo_url, destination: destination.mkdir(parents=True),
        download=lambda _url, _destination: None,
    )

    with pytest.raises(ValueError, match="download_url"):
        setup.prepare_model(
            {
                "name": "ResShift_x4",
                "weights": "./weights/resshift.safetensors",
                "repo_root": "./third_party/ResShift",
            }
        )


def test_supir_setup_uses_default_repo_url(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = DiffusionTeacherSetup(
        project_root=tmp_path,
        runner_type="SUPIRRunner",
        default_repo_url=SUPIR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "SUPIR_x4",
            "weights": "./weights/SUPIR/supir_x4.safetensors",
            "repo_root": "./third_party/SUPIR",
            "download_url": "https://example.test/supir.safetensors",
        }
    )

    assert result["repos"]["main"]["path"].endswith("third_party\\SUPIR")
    assert result["weights"]["path"].endswith("weights\\SUPIR\\supir_x4.safetensors")


def test_resshift_setup_downloads_autoencoder_artifact_when_configured(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = DiffusionTeacherSetup(
        project_root=tmp_path,
        runner_type="ResShiftRunner",
        default_repo_url=RESSHIFT_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "ResShift_x4",
            "weights": "./weights/resshift/resshift_realsrx4_s4_v3.pth",
            "repo_root": "./third_party/ResShift",
            "download_url": "https://example.test/resshift.pth",
            "autoencoder_weights": "./weights/resshift/autoencoder_vq_f4.pth",
            "autoencoder_download_url": "https://example.test/autoencoder_vq_f4.pth",
        }
    )

    assert result["artifacts"]["autoencoder"]["status"] == "downloaded"


def test_stablesr_setup_clones_extra_repo_roots_when_declared(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = DiffusionTeacherSetup(
        project_root=tmp_path,
        runner_type="StableSRRunner",
        default_repo_url=STABLESR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "StableSR_x4",
            "weights": "./weights/stablesr/stablesr_x4.safetensors",
            "vqgan_weights": "./weights/stablesr/vqgan_cfw_00011.ckpt",
            "repo_root": "./third_party/StableSR",
            "download_url": "https://example.test/stablesr.safetensors",
            "vqgan_download_url": "https://example.test/vqgan.ckpt",
            "extra_repo_roots": [
                {"path": "./third_party/taming-transformers", "repo_url": "https://github.com/CompVis/taming-transformers.git"},
                {"path": "./third_party/CLIP", "repo_url": "https://github.com/openai/CLIP.git"},
            ],
        }
    )

    assert result["repos"]["extra_1"]["path"].endswith("third_party\\taming-transformers")
    assert result["repos"]["extra_2"]["path"].endswith("third_party\\CLIP")
    assert calls == [
        ("clone", STABLESR_REPO_URL, tmp_path / "third_party" / "StableSR"),
        ("clone", "https://github.com/CompVis/taming-transformers.git", tmp_path / "third_party" / "taming-transformers"),
        ("clone", "https://github.com/openai/CLIP.git", tmp_path / "third_party" / "CLIP"),
        ("download", "https://example.test/stablesr.safetensors", tmp_path / "weights" / "stablesr" / "stablesr_x4.safetensors"),
        ("download", "https://example.test/vqgan.ckpt", tmp_path / "weights" / "stablesr" / "vqgan_cfw_00011.ckpt"),
    ]
