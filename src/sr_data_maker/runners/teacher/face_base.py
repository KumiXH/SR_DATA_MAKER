from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sr_data_maker.core.types import RunnerOutput


class FaceTeacherRunnerBase:
    family = "FaceTeacher"
    display_name = "Face Teacher"

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        weights = self._weights_path()
        if not weights.exists():
            raise FileNotFoundError(f"{self.display_name} weights not found: {weights}")

        self._add_repo_to_path()
        torch = self._import_torch()
        output = self._restore_image(image, torch)
        return RunnerOutput(outputs={"image": output}, meta=self._provenance())

    def _weights_path(self) -> Path:
        weights = self.model.get("weights")
        return Path(str(weights))

    def _repo_roots(self) -> list[Path]:
        roots: list[Path] = []
        repo_root = self.model.get("repo_root")
        if repo_root:
            roots.append(Path(str(repo_root)))
        facelib_root = self.model.get("facelib_root")
        if facelib_root:
            roots.append(Path(str(facelib_root)))
        basicsr_root = self.model.get("basicsr_root")
        if basicsr_root:
            roots.append(Path(str(basicsr_root)))
        return roots

    def _add_repo_to_path(self) -> None:
        for repo_path in reversed(self._repo_roots()):
            if not repo_path.exists():
                raise FileNotFoundError(f"{self.display_name} repo root not found: {repo_path}")
            repo_str = str(repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

    @staticmethod
    def _import_torch():
        import torch

        return torch

    def _run_inference(self, image: Any, torch: Any):
        raise NotImplementedError

    def _restore_image(self, image: Any, torch: Any):
        return self._run_inference(image, torch)

    def _provenance(self) -> dict[str, Any]:
        return {
            "face_model": True,
            "face_model_family": self.family,
            "scale": int(self.model.get("scale", 1)),
        }
