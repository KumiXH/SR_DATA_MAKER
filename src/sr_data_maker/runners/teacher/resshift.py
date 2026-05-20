from __future__ import annotations

import os
import shutil
from pathlib import Path

from sr_data_maker.runners.teacher.diffusion_base import DiffusionTeacherRunnerBase


class ResShiftRunner(DiffusionTeacherRunnerBase):
    family = "ResShift"
    display_name = "ResShift"

    def _run_inference(self, image, torch):
        self._prepare_runtime_artifacts()
        command = self._build_command(Path("{input_path}"), Path("{output_dir}"))
        return self._run_subprocess_inference(image, command, self._resolve_output_path)

    def _prepare_runtime_artifacts(self) -> None:
        weights_dir = self._repo_root() / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        self._stage_artifact(self._weights_path(), weights_dir / self._weights_path().name)

        autoencoder = self.model.get("autoencoder_weights")
        if autoencoder:
            autoencoder_path = Path(str(autoencoder))
            if not autoencoder_path.exists():
                raise FileNotFoundError(f"{self.display_name} autoencoder weights not found: {autoencoder_path}")
            self._stage_artifact(autoencoder_path, weights_dir / autoencoder_path.name)

    @staticmethod
    def _stage_artifact(source: Path, destination: Path) -> None:
        if destination.exists() and destination.stat().st_size == source.stat().st_size:
            return
        shutil.copy2(source, destination)

    def _build_command(self, input_path: Path, output_dir: Path) -> list[str]:
        script = self._repo_root() / "inference_resshift.py"
        return [
            self._python_executable(),
            str(script),
            "-i",
            str(input_path),
            "-o",
            str(output_dir),
            "--task",
            str(self.model.get("task", "realsr")),
            "--scale",
            str(int(self.model.get("scale", 4))),
            "--version",
            str(self.model.get("version", "v3")),
            "--chop_size",
            str(int(self.model.get("chop_size", 256))),
            "--chop_stride",
            str(int(self.model.get("chop_stride", 128))),
            "--bs",
            str(int(self.model.get("bs", 1))),
            "--seed",
            str(int(self.model.get("seed", 12345))),
        ]

    def _subprocess_env(self) -> dict[str, str]:
        hf_cache = self.model.get("hf_cache_dir") or str((Path.cwd() / "weights" / "supir" / "hf_cache").resolve())
        openclip_root = self.model.get("openclip_root") or str((Path.home() / ".cache" / "huggingface" / "hub").resolve())
        repo_root = str(self._repo_root())
        pythonpath = os.pathsep.join(filter(None, [repo_root, os.environ.get("PYTHONPATH", "")]))
        return {
            "PYTHONPATH": pythonpath,
            "HF_HOME": hf_cache,
            "HUGGINGFACE_HUB_CACHE": hf_cache,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "SR_DATA_MAKER_HF_CACHE": hf_cache,
            "SR_DATA_MAKER_OPENCLIP_ROOT": openclip_root,
        }

    @staticmethod
    def _resolve_output_path(output_dir: Path, input_path: Path) -> Path:
        expected = output_dir / f"{input_path.stem}.png"
        if expected.exists():
            return expected
        matches = sorted(output_dir.rglob(f"{input_path.stem}*.png"))
        if not matches:
            return expected
        return matches[0]
