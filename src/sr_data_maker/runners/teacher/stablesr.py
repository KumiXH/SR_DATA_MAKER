from __future__ import annotations

import os
from pathlib import Path

from sr_data_maker.runners.teacher.diffusion_base import DiffusionTeacherRunnerBase


class StableSRRunner(DiffusionTeacherRunnerBase):
    family = "StableSR"
    display_name = "StableSR"

    def _run_inference(self, image, torch):
        command = self._build_command(Path("{input_path}"), Path("{output_dir}"))
        return self._run_subprocess_inference(image, command, self._resolve_output_path)

    def _build_command(self, input_path: Path, output_dir: Path) -> list[str]:
        script = self._repo_root() / "scripts" / "sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py"
        config_value = self.model.get("config_path", self._repo_root() / "configs" / "stableSRNew" / "v2-finetune_text_T_512.yaml")
        config = str(self._resolve_model_path(config_value))
        vqgan = self.model.get("vqgan_weights")
        if not vqgan:
            raise FileNotFoundError("StableSR requires model.vqgan_weights")
        vqgan = str(self._resolve_model_path(vqgan))
        return [
            self._python_executable(),
            str(script),
            "--config",
            config,
            "--ckpt",
            str(self._weights_path()),
            "--vqgan_ckpt",
            vqgan,
            "--init-img",
            str(input_path),
            "--outdir",
            str(output_dir),
            "--ddpm_steps",
            str(int(self.model.get("steps", 50))),
            "--dec_w",
            str(float(self.model.get("dec_w", 0.5))),
            "--colorfix_type",
            str(self.model.get("color_fix_type", "wavelet")),
            "--upscale",
            str(int(self.model.get("scale", 4))),
            "--seed",
            str(int(self.model.get("seed", 42))),
            "--n_samples",
            "1",
            "--input_size",
            str(int(self.model.get("input_size", 512))),
            "--tile_overlap",
            str(int(self.model.get("tile_overlap", 48))),
            "--vqgantile_size",
            str(int(self.model.get("vqgan_tile_size", 1280))),
            "--vqgantile_stride",
            str(int(self.model.get("vqgan_tile_stride", 1000))),
        ]

    def _subprocess_env(self) -> dict[str, str]:
        repo_root = str(self._repo_root())
        extra_paths = [repo_root]
        for candidate in ("src/taming-transformers", "src/clip"):
            path = (Path.cwd() / candidate).resolve()
            if path.exists():
                extra_paths.append(str(path))
        pythonpath = os.pathsep.join(filter(None, [*extra_paths, os.environ.get("PYTHONPATH", "")]))
        hf_cache = self.model.get("hf_cache_dir") or str((Path.cwd() / "weights" / "supir" / "hf_cache").resolve())
        openclip_root = self.model.get("openclip_root") or str((Path.home() / ".cache" / "huggingface" / "hub").resolve())
        return {
            "PYTHONPATH": pythonpath,
            "KMP_DUPLICATE_LIB_OK": str(self.model.get("kmp_duplicate_lib_ok", "TRUE")),
            "HF_HOME": hf_cache,
            "HUGGINGFACE_HUB_CACHE": hf_cache,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "SR_DATA_MAKER_HF_CACHE": hf_cache,
            "SR_DATA_MAKER_OPENCLIP_ROOT": openclip_root,
        }

    @staticmethod
    def _resolve_output_path(output_dir: Path, input_path: Path) -> Path:
        stem = input_path.stem
        matches = sorted(output_dir.rglob(f"{stem}*.png"))
        if not matches:
            return output_dir / f"{stem}.png"
        return matches[0]
