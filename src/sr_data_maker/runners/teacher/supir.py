from __future__ import annotations

from pathlib import Path

from sr_data_maker.runners.teacher.diffusion_base import DiffusionTeacherRunnerBase


class SUPIRRunner(DiffusionTeacherRunnerBase):
    family = "SUPIR"
    display_name = "SUPIR"

    def _run_inference(self, image, torch):
        command = self._build_command(Path("{input_dir}"), Path("{output_dir}"))
        return self._run_subprocess_inference(image, command, self._resolve_output_path)

    def _build_command(self, input_path: Path, output_dir: Path) -> list[str]:
        input_dir = input_path if input_path.name in {"{input_path}", "{input_dir}"} or not input_path.suffix else input_path.parent
        command = [
            self._python_executable(),
            str(self._repo_root() / "test.py"),
            "--opt",
            str(self._resolve_opt_path()),
            "--img_dir",
            str(input_dir),
            "--save_dir",
            str(output_dir),
            "--upscale",
            str(int(self.model.get("scale", 4))),
            "--SUPIR_sign",
            str(self.model.get("supir_sign", "Q")),
            "--seed",
            str(int(self.model.get("seed", 1234))),
            "--edm_steps",
            str(int(self.model.get("steps", self.model.get("edm_steps", 50)))),
            "--s_stage2",
            str(float(self.model.get("s_stage2", 1.0))),
            "--a_prompt",
            str(self.model.get("prompt", "")),
            "--n_prompt",
            str(self.model.get("negative_prompt", "")),
            "--color_fix_type",
            str(self.model.get("color_fix_type", "Wavelet")),
            "--min_size",
            str(int(self.model.get("min_size", 512))),
            "--ae_dtype",
            str(self.model.get("ae_dtype", "bf16")),
            "--diff_dtype",
            str(self.model.get("diff_dtype", "fp16")),
        ]
        if bool(self.model.get("no_llava", True)):
            command.append("--no_llava")
        if bool(self.model.get("loading_half_params", False)):
            command.append("--loading_half_params")
        if bool(self.model.get("use_tile_vae", False)):
            command.extend(
                [
                    "--use_tile_vae",
                    "--encoder_tile_size",
                    str(int(self.model.get("encoder_tile_size", 512))),
                    "--decoder_tile_size",
                    str(int(self.model.get("decoder_tile_size", 64))),
                ]
            )
        return command

    def _resolve_opt_path(self) -> Path:
        opt_path = self.model.get("opt_path", "options/SUPIR_v0.yaml")
        return self._resolve_model_path(opt_path)

    @staticmethod
    def _resolve_output_path(output_dir: Path, input_path: Path) -> Path:
        stem = input_path.stem
        matches = sorted(output_dir.rglob(f"{stem}_*.png"))
        if not matches:
            return output_dir / f"{stem}_0.png"
        return matches[0]
