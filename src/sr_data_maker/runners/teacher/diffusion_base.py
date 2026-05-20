from __future__ import annotations

import sys
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Any

from sr_data_maker.core.types import RunnerOutput


class DiffusionTeacherRunnerBase:
    family = "DiffusionTeacher"
    display_name = "Diffusion Teacher"

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        weights = self._weights_path()
        if not weights.exists():
            raise FileNotFoundError(f"{self.display_name} weights not found: {weights}")

        self._add_repo_to_path()
        torch = self._import_torch()
        output = self._run_inference(image, torch)
        return RunnerOutput(outputs={"image": output}, meta=self._provenance())

    def _weights_path(self) -> Path:
        weights = self.model.get("weights")
        return self._resolve_model_path(weights)

    def _repo_roots(self) -> list[Path]:
        roots: list[Path] = []
        repo_root = self.model.get("repo_root")
        if repo_root:
            roots.append(self._resolve_model_path(repo_root))
        extra_roots = self.model.get("extra_repo_roots") or []
        if isinstance(extra_roots, (str, Path)):
            extra_roots = [extra_roots]
        roots.extend(self._resolve_model_path(root) for root in extra_roots)
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

    def _run_subprocess_inference(self, image: Any, command: list[str], output_resolver) -> Any:
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix=f"{self.family.lower()}_") as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            input_dir = tmp_dir / "inputs"
            input_dir.mkdir(parents=True, exist_ok=True)
            input_path = input_dir / "input.png"
            output_dir = tmp_dir / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            image.convert("RGB").save(input_path)

            final_command = command_builder_replace(command, input_path, output_dir, input_dir)
            env = os.environ.copy()
            env.update(self._subprocess_env())
            completed = subprocess.run(
                final_command,
                cwd=str(self._repo_root()),
                capture_output=True,
                text=True,
                env=env,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{self.display_name} inference failed with exit code {completed.returncode}\n"
                    f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )

            output_path = output_resolver(output_dir, input_path)
            if not output_path.exists():
                raise FileNotFoundError(f"{self.display_name} output not found: {output_path}")
            return Image.open(output_path).convert("RGB")

    def _repo_root(self) -> Path:
        repo_root = self.model.get("repo_root")
        if not repo_root:
            raise FileNotFoundError(f"{self.display_name} repo root not configured")
        return self._resolve_model_path(repo_root)

    def _python_executable(self) -> str:
        python_executable = self.model.get("python_executable")
        if python_executable:
            return str(self._resolve_model_path(python_executable))
        return sys.executable

    def _subprocess_env(self) -> dict[str, str]:
        return {}

    @staticmethod
    def _resolve_model_path(value: Any) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else path.resolve()

    def _provenance(self) -> dict[str, Any]:
        meta = {
            "diffusion_model": True,
            "diffusion_model_family": self.family,
            "real_world_sr": True,
            "scale": int(self.model.get("scale", 1)),
        }
        for key in (
            "steps",
            "precision",
            "python_executable",
            "tile_size",
            "tile_stride",
            "chop_size",
            "chop_stride",
            "prompt",
            "negative_prompt",
            "guidance_scale",
            "color_fix_type",
            "sampler",
        ):
            if key in self.model:
                meta[key] = self.model[key]
        return meta


def command_builder_replace(command: list[str], input_path: Path, output_dir: Path, input_dir: Path | None = None) -> list[str]:
    resolved_input_dir = input_dir or input_path.parent
    replaced: list[str] = []
    for item in command:
        replaced.append(
            item.replace("{input_path}", str(input_path))
            .replace("{input_dir}", str(resolved_input_dir))
            .replace("{output_dir}", str(output_dir))
        )
    return replaced
