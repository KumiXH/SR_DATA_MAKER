from __future__ import annotations

from pathlib import Path

import pytest

from sr_data_maker.setup.face_teacher import (
    CODEFORMER_REPO_URL,
    FACEXLIB_REPO_URL,
    GFPGAN_REPO_URL,
    VQFR_REPO_URL,
    FaceTeacherSetup,
    find_face_teacher_models,
)


def test_find_face_teacher_models_reads_enabled_matching_tasks():
    config = {
        "tasks": [
            {
                "enabled": True,
                "runner": {"type": "GFPGANRunner"},
                "model": {"name": "GFPGAN_x2", "weights": "./weights/gfpgan.pth"},
            },
            {
                "enabled": True,
                "runner": {"type": "CodeFormerRunner"},
                "model": {"name": "CodeFormer_x2", "weights": "./weights/codeformer.pth"},
            },
            {
                "enabled": False,
                "runner": {"type": "VQFRRunner"},
                "model": {"name": "disabled"},
            },
        ]
    }

    models = find_face_teacher_models(config, {"GFPGANRunner", "CodeFormerRunner", "VQFRRunner"})

    assert [item["runner_type"] for item in models] == ["GFPGANRunner", "CodeFormerRunner"]


def test_gfpgan_setup_clones_expected_repos_and_downloads_weights(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="GFPGANRunner",
        default_repo_url=GFPGAN_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "GFPGAN_x2",
            "weights": "./weights/GFPGAN_x2.pth",
            "repo_root": "./third_party/GFPGAN",
            "facelib_root": "./third_party/facexlib",
            "basicsr_root": "./third_party/BasicSR",
            "download_url": "https://example.test/gfpgan.pth",
        }
    )

    assert result["repos"]["main"]["status"] == "cloned"
    assert result["repos"]["facexlib"]["status"] == "cloned"
    assert result["repos"]["basicsr"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert calls == [
        ("clone", GFPGAN_REPO_URL, tmp_path / "third_party" / "GFPGAN"),
        ("clone", FACEXLIB_REPO_URL, tmp_path / "third_party" / "facexlib"),
        ("clone", "https://github.com/XPixelGroup/BasicSR.git", tmp_path / "third_party" / "BasicSR"),
        ("download", "https://example.test/gfpgan.pth", tmp_path / "weights" / "GFPGAN_x2.pth"),
    ]


def test_codeformer_setup_requires_download_url_when_weight_missing(tmp_path):
    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="CodeFormerRunner",
        default_repo_url=CODEFORMER_REPO_URL,
        clone=lambda _repo_url, destination: destination.mkdir(parents=True),
        download=lambda _url, _destination: None,
    )

    with pytest.raises(ValueError, match="download_url"):
        setup.prepare_model(
            {
                "name": "CodeFormer_x2",
                "weights": "./weights/codeformer.pth",
                "repo_root": "./third_party/CodeFormer",
                "facelib_root": "./third_party/facexlib",
                "basicsr_root": "./third_party/BasicSR",
            }
        )


def test_vqfr_setup_uses_default_repo_url(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="VQFRRunner",
        default_repo_url=VQFR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "VQFR_x2",
            "weights": "./weights/VQFR_x2.pth",
            "repo_root": "./third_party/VQFR",
            "facelib_root": "./third_party/facexlib",
            "basicsr_root": "./third_party/BasicSR",
            "download_url": "https://example.test/vqfr.pth",
        }
    )

    assert result["repos"]["main"]["path"].endswith("VQFR")
    assert result["weights"]["path"].endswith("VQFR_x2.pth")
