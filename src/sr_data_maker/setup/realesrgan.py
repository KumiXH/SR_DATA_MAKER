from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

REAL_ESRGAN_REPO_URL = "https://github.com/xinntao/Real-ESRGAN.git"
BASICSR_REPO_URL = "https://github.com/XPixelGroup/BasicSR.git"

MODEL_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

CloneFunc = Callable[[str, Path], None]
DownloadFunc = Callable[[str, Path], None]


def find_realesrgan_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for task in config.get("tasks", []):
        if task.get("enabled", True) is False:
            continue
        runner = task.get("runner") or {}
        if runner.get("type") != "RealESRGANRunner":
            continue
        model = dict(task.get("model") or {})
        if not model:
            continue
        models.append(model)
    return models


class RealESRGANSetup:
    def __init__(
        self,
        project_root: str | Path,
        clone: CloneFunc | None = None,
        download: DownloadFunc | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.clone = clone or clone_git_repo
        self.download = download or download_file

    def prepare_from_config(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        models = find_realesrgan_models(config)
        if not models:
            raise ValueError("No enabled RealESRGANRunner task found in config.")
        return [self.prepare_model(model) for model in models]

    def prepare_model(self, model: dict[str, Any]) -> dict[str, Any]:
        model_name = str(model.get("name", "RealESRGAN_x2plus"))
        real_repo = self._resolve_path(model.get("repo_root", "./third_party/Real-ESRGAN"))
        basic_repo = self._resolve_path(model.get("basicsr_root", "./third_party/BasicSR"))
        weights = self._resolve_path(model.get("weights", f"./weights/{model_name}.pth"))

        real_status = self._ensure_repo(REAL_ESRGAN_REPO_URL, real_repo)
        basic_status = self._ensure_repo(BASICSR_REPO_URL, basic_repo)
        self._ensure_version_file(real_repo / "realesrgan" / "version.py")
        self._ensure_version_file(basic_repo / "basicsr" / "version.py")
        weight_status = self._ensure_weights(model_name, weights)

        return {
            "model_name": model_name,
            "repos": {
                "real_esrgan": {"path": str(real_repo), "status": real_status},
                "basicsr": {"path": str(basic_repo), "status": basic_status},
            },
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

    def _ensure_weights(self, model_name: str, destination: Path) -> str:
        if destination.exists():
            return "exists"
        url = MODEL_URLS.get(model_name)
        if url is None:
            raise ValueError(f"No default weight URL for Real-ESRGAN model: {model_name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.download(url, destination)
        return "downloaded"

    @staticmethod
    def _ensure_version_file(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        path.write_text('__version__ = "local"\n__gitsha__ = "local"\n', encoding="utf-8")


def clone_git_repo(repo_url: str, destination: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("git is required to clone Real-ESRGAN dependencies.")
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(destination)], check=True)


def download_file(url: str, destination: Path) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def setup_realesrgan_from_config(config: dict[str, Any], project_root: str | Path) -> list[dict[str, Any]]:
    return RealESRGANSetup(project_root=project_root).prepare_from_config(config)


def print_setup_summary(results: list[dict[str, Any]]) -> None:
    print(json.dumps({"realesrgan": results}, indent=2))


def _looks_like_existing_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / ".git").exists() or (path / "realesrgan").is_dir() or (path / "basicsr").is_dir()
