from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from sr_data_maker.setup.realesrgan import clone_git_repo, download_file

STABLESR_REPO_URL = "https://github.com/IceClear/StableSR.git"
RESSHIFT_REPO_URL = "https://github.com/zsyOAOA/ResShift.git"
SUPIR_REPO_URL = "https://github.com/Fanghua-Yu/SUPIR.git"

CloneFunc = Callable[[str, Path], None]
DownloadFunc = Callable[[str, Path], None]


def find_diffusion_teacher_models(config: dict[str, Any], runner_types: set[str]) -> list[dict[str, Any]]:
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


class DiffusionTeacherSetup:
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
        models = find_diffusion_teacher_models(config, {self.runner_type})
        if not models:
            raise ValueError(f"No enabled {self.runner_type} task found in config.")
        return [self.prepare_model(model) for model in models]

    def prepare_model(self, model: dict[str, Any]) -> dict[str, Any]:
        model_name = str(model.get("name", self.runner_type))
        repo_root = self._resolve_path(model.get("repo_root", f"./third_party/{model_name}"))
        weights = self._resolve_path(model.get("weights", f"./weights/{model_name}.safetensors"))
        repo_status = self._ensure_repo(str(model.get("repo_url", self.default_repo_url)), repo_root)
        repos: dict[str, dict[str, str]] = {"main": {"path": str(repo_root), "status": repo_status}}
        for index, extra in enumerate(self._extra_repo_roots(model), start=1):
            extra_path = self._resolve_path(extra["path"])
            repos[f"extra_{index}"] = {
                "path": str(extra_path),
                "status": self._ensure_repo(str(extra["repo_url"]), extra_path),
            }
        weight_status = self._ensure_weights(model, weights)
        artifacts = self._prepare_artifacts(model)
        return {
            "runner_type": self.runner_type,
            "model_name": model_name,
            "repos": repos,
            "weights": {"path": str(weights), "status": weight_status},
            "artifacts": artifacts,
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

    def _prepare_artifacts(self, model: dict[str, Any]) -> dict[str, dict[str, str]]:
        artifacts: dict[str, dict[str, str]] = {}
        artifact_specs = (
            ("vqgan", "vqgan_weights", "vqgan_download_url"),
            ("autoencoder", "autoencoder_weights", "autoencoder_download_url"),
        )
        for name, path_key, url_key in artifact_specs:
            artifact_path = model.get(path_key)
            if not artifact_path:
                continue
            destination = self._resolve_path(artifact_path)
            if destination.exists():
                status = "exists"
            else:
                download_url = model.get(url_key)
                if not download_url:
                    raise ValueError(
                        f"{self.runner_type} model {model.get('name', '<unnamed>')} is missing model.{url_key}"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.download(str(download_url), destination)
                status = "downloaded"
            artifacts[name] = {"path": str(destination), "status": status}
        return artifacts

    @staticmethod
    def _extra_repo_roots(model: dict[str, Any]) -> list[dict[str, Any]]:
        roots = model.get("extra_repo_roots") or []
        if isinstance(roots, (str, Path)):
            raise ValueError("extra_repo_roots entries must include both path and repo_url")
        return [dict(root) for root in roots]


def setup_stablesr_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return DiffusionTeacherSetup(
        project_root=project_root,
        runner_type="StableSRRunner",
        default_repo_url=STABLESR_REPO_URL,
    ).prepare_from_config(config)


def setup_resshift_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return DiffusionTeacherSetup(
        project_root=project_root,
        runner_type="ResShiftRunner",
        default_repo_url=RESSHIFT_REPO_URL,
    ).prepare_from_config(config)


def setup_supir_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return DiffusionTeacherSetup(
        project_root=project_root,
        runner_type="SUPIRRunner",
        default_repo_url=SUPIR_REPO_URL,
    ).prepare_from_config(config)


def print_diffusion_teacher_setup_summary(target: str, results: list[dict[str, Any]]) -> None:
    print(json.dumps({target: results}, indent=2))


def _looks_like_existing_repo(path: Path) -> bool:
    return path.is_dir() and ((path / ".git").exists() or any(path.iterdir()))
