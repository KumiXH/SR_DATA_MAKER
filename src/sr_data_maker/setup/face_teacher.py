from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from sr_data_maker.setup.realesrgan import BASICSR_REPO_URL, clone_git_repo, download_file

GFPGAN_REPO_URL = "https://github.com/TencentARC/GFPGAN.git"
CODEFORMER_REPO_URL = "https://github.com/sczhou/CodeFormer.git"
VQFR_REPO_URL = "https://github.com/TencentARC/VQFR.git"
FACEXLIB_REPO_URL = "https://github.com/xinntao/facexlib.git"

CloneFunc = Callable[[str, Path], None]
DownloadFunc = Callable[[str, Path], None]


def find_face_teacher_models(config: dict[str, Any], runner_types: set[str]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for task in config.get("tasks", []):
        if task.get("enabled", True) is False:
            continue
        runner = task.get("runner") or {}
        runner_type = runner.get("type")
        if runner_type not in runner_types:
            continue
        model = dict(task.get("model") or {})
        if not model:
            continue
        model["runner_type"] = runner_type
        models.append(model)
    return models


class FaceTeacherSetup:
    def __init__(
        self,
        project_root: str | Path,
        runner_type: str,
        default_repo_url: str,
        clone: CloneFunc | None = None,
        download: DownloadFunc | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runner_type = runner_type
        self.default_repo_url = default_repo_url
        self.clone = clone or clone_git_repo
        self.download = download or download_file

    def prepare_from_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        models = find_face_teacher_models(config, {self.runner_type})
        if not models:
            raise ValueError(f"No enabled {self.runner_type} task found in config.")
        return [self.prepare_model(model) for model in models]

    def prepare_model(self, model: dict[str, Any]) -> dict[str, Any]:
        model_name = str(model.get("name", self.runner_type))
        repo_root = self._resolve_path(model.get("repo_root", f"./third_party/{model_name}"))
        weights = self._resolve_path(model.get("weights", f"./weights/{model_name}.pth"))
        repos = {
            "main": {
                "path": str(repo_root),
                "status": self._ensure_repo(str(model.get("repo_url", self.default_repo_url)), repo_root),
            }
        }
        self._ensure_repo_local_versions(repo_root)

        facelib_root = model.get("facelib_root")
        if facelib_root:
            facelib_path = self._resolve_path(facelib_root)
            repos["facexlib"] = {
                "path": str(facelib_path),
                "status": self._ensure_repo(str(model.get("facelib_repo_url", FACEXLIB_REPO_URL)), facelib_path),
            }
            self._ensure_version_file(facelib_path / "facexlib" / "version.py")

        basicsr_root = model.get("basicsr_root")
        if basicsr_root:
            basicsr_path = self._resolve_path(basicsr_root)
            repos["basicsr"] = {
                "path": str(basicsr_path),
                "status": self._ensure_repo(str(model.get("basicsr_repo_url", BASICSR_REPO_URL)), basicsr_path),
            }
            self._ensure_version_file(basicsr_path / "basicsr" / "version.py")

        weight_status = self._ensure_weights(model, weights)
        return {
            "runner_type": self.runner_type,
            "model_name": model_name,
            "repos": repos,
            "weights": {"path": str(weights), "status": weight_status},
        }

    def _resolve_path(self, value: Any) -> Path:
        path = Path(str(value))
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _ensure_repo(self, repo_url: str, destination: Path) -> str:
        if _looks_like_existing_repo(destination):
            return "exists"
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Directory exists but is not a usable repo: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.clone(repo_url, destination)
        return "cloned"

    def _ensure_weights(self, model: dict[str, Any], destination: Path) -> str:
        if destination.exists():
            return "exists"
        download_url = model.get("download_url")
        if not download_url:
            raise ValueError(f"{self.runner_type} model {model.get('name', '<unnamed>')} is missing model.download_url")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.download(str(download_url), destination)
        return "downloaded"

    @staticmethod
    def _ensure_version_file(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('__version__ = "local"\n__gitsha__ = "local"\n', encoding="utf-8")

    def _ensure_repo_local_versions(self, repo_root: Path) -> None:
        basicsr_version = repo_root / "basicsr" / "version.py"
        if basicsr_version.parent.exists():
            self._ensure_version_file(basicsr_version)
        facexlib_version = repo_root / "facexlib" / "version.py"
        if facexlib_version.parent.exists():
            self._ensure_version_file(facexlib_version)


def setup_gfpgan_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return FaceTeacherSetup(
        project_root=project_root,
        runner_type="GFPGANRunner",
        default_repo_url=GFPGAN_REPO_URL,
    ).prepare_from_config(config)


def setup_codeformer_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return FaceTeacherSetup(
        project_root=project_root,
        runner_type="CodeFormerRunner",
        default_repo_url=CODEFORMER_REPO_URL,
    ).prepare_from_config(config)


def setup_vqfr_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return FaceTeacherSetup(
        project_root=project_root,
        runner_type="VQFRRunner",
        default_repo_url=VQFR_REPO_URL,
    ).prepare_from_config(config)


def print_face_teacher_setup_summary(target: str, results: list[dict[str, Any]]) -> None:
    print(json.dumps({target: results}, indent=2))


def _looks_like_existing_repo(path: Path) -> bool:
    return path.is_dir() and ((path / ".git").exists() or any(path.iterdir()))
