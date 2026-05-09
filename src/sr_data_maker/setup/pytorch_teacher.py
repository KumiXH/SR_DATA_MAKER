from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from sr_data_maker.setup.realesrgan import BASICSR_REPO_URL, clone_git_repo, download_file

SWINIR_REPO_URL = "https://github.com/JingyunLiang/SwinIR.git"
HAT_REPO_URL = "https://github.com/XPixelGroup/HAT.git"

CloneFunc = Callable[[str, Path], None]
DownloadFunc = Callable[[str, Path], None]


def find_teacher_models(config: dict[str, Any], adapter_types: set[str]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for task in config.get("tasks", []):
        if task.get("enabled", True) is False:
            continue
        runner = task.get("runner") or {}
        adapter_type = runner.get("type")
        if adapter_type not in adapter_types:
            continue
        model = dict(task.get("model") or {})
        if not model:
            continue
        model["adapter_type"] = adapter_type
        models.append(model)
    return models


class PyTorchTeacherSetup:
    def __init__(
        self,
        project_root: str | Path,
        adapter_type: str,
        default_repo_url: str,
        clone: CloneFunc | None = None,
        download: DownloadFunc | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.adapter_type = adapter_type
        self.default_repo_url = default_repo_url
        self.clone = clone or clone_git_repo
        self.download = download or download_file

    def prepare_from_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        models = find_teacher_models(config, {self.adapter_type})
        if not models:
            raise ValueError(f"No enabled {self.adapter_type} task found in config.")
        return [self.prepare_model(model) for model in models]

    def prepare_model(self, model: dict[str, Any]) -> dict[str, Any]:
        model_name = str(model.get("name", self.adapter_type))
        repo_root = self._resolve_path(model.get("repo_root", f"./third_party/{model_name}"))
        weights = self._resolve_path(model.get("weights", f"./weights/{model_name}.pth"))
        repo_status = self._ensure_repo(str(model.get("repo_url", self.default_repo_url)), repo_root)
        repos: dict[str, dict[str, str]] = {"main": {"path": str(repo_root), "status": repo_status}}

        basicsr_root = model.get("basicsr_root")
        if basicsr_root:
            basicsr_path = self._resolve_path(basicsr_root)
            repos["basicsr"] = {
                "path": str(basicsr_path),
                "status": self._ensure_repo(str(model.get("basicsr_repo_url", BASICSR_REPO_URL)), basicsr_path),
            }
            self._ensure_version_file(basicsr_path / "basicsr" / "version.py")

        for index, extra in enumerate(self._extra_repo_roots(model), start=1):
            extra_path = self._resolve_path(extra["path"])
            repos[f"extra_{index}"] = {
                "path": str(extra_path),
                "status": self._ensure_repo(str(extra["repo_url"]), extra_path),
            }

        weight_status = self._ensure_weights(model, weights)
        return {
            "adapter_type": self.adapter_type,
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
            raise ValueError(f"{self.adapter_type} model {model.get('name', '<unnamed>')} is missing model.download_url")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.download(str(download_url), destination)
        return "downloaded"

    @staticmethod
    def _ensure_version_file(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text('__version__ = "local"\n__gitsha__ = "local"\n', encoding="utf-8")

    @staticmethod
    def _extra_repo_roots(model: dict[str, Any]) -> list[dict[str, Any]]:
        roots = model.get("extra_repo_roots") or []
        if isinstance(roots, (str, Path)):
            raise ValueError("extra_repo_roots entries must include both path and repo_url")
        return [dict(root) for root in roots]


def setup_swinir_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return PyTorchTeacherSetup(
        project_root=project_root,
        adapter_type="SwinIRAdapter",
        default_repo_url=SWINIR_REPO_URL,
    ).prepare_from_config(config)


def setup_hat_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return PyTorchTeacherSetup(
        project_root=project_root,
        adapter_type="HATAdapter",
        default_repo_url=HAT_REPO_URL,
    ).prepare_from_config(config)


def print_teacher_setup_summary(target: str, results: list[dict[str, Any]]) -> None:
    print(json.dumps({target: results}, indent=2))


def _looks_like_existing_repo(path: Path) -> bool:
    return path.is_dir() and ((path / ".git").exists() or any(path.iterdir()))
