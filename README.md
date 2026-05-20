# SR Data Maker

[中文说明 / Chinese README](README.zh-CN.md)

YAML-driven training data generation toolkit for super-resolution datasets.

The current project supports:

- Classical degradation output under `degraded/<task_name>/<source_rel_path>`.
- Teacher-model output under `teacher/<model_name>/<source_rel_path>`.
- Generic SR teachers and face-focused SR/restoration teachers in one pipeline.
- Nested source image folders with mirrored output paths.
- JSONL manifests for provenance.
- Windows and Linux execution with non-interactive commands.

## Model Registry

The table below summarizes the SR and face-teacher models currently integrated in this repository.

| Model | Runner / Adapter | Category | Nominal Scale | Actual Output Behavior | Source | Example YAML |
| --- | --- | --- | --- | --- | --- | --- |
| Real-ESRGAN x2plus | `RealESRGANRunner` | General SR | x2 | Whole-image x2 SR | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `configs/examples/local_realesrgan_x2.yaml` |
| SwinIR x2 classical | `SwinIRAdapter` | General SR | x2 | Whole-image x2 SR | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | `configs/examples/local_swinir_x2.yaml` |
| HAT SRx2 | `HATAdapter` | General SR | x2 | Whole-image x2 SR | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | `configs/examples/local_hat_x2.yaml` |
| GFPGAN v1.4 | `GFPGANRunner` | Face SR / restoration | x2 | Whole-image face restoration with x2 output | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | `configs/examples/local_gfpgan_x2.yaml` |
| CodeFormer | `CodeFormerRunner` | Face SR / restoration | x2 config | Whole-image face restoration plus face upsample; output size follows the official face-upsample path and may exceed strict x2 | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | `configs/examples/local_codeformer_x2.yaml` |
| VQFR v2 | `VQFRRunner` | Face SR / restoration | x2 | Whole-image face restoration with x2 output | [TencentARC/VQFR](https://github.com/TencentARC/VQFR) | `configs/examples/local_vqfr_x2.yaml` |
| StableSR | `StableSRRunner` | Diffusion real-world SR | x4 config | Whole-image diffusion SR with conservative tile-friendly config | [IceClear/StableSR](https://github.com/IceClear/StableSR) | `configs/examples/local_stablesr_x4.yaml` |
| ResShift | `ResShiftRunner` | Diffusion real-world SR | x4 config | Whole-image diffusion SR with low-step real-world restoration flow | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) | `configs/examples/local_resshift_x4.yaml` |
| SUPIR | `SUPIRRunner` | Diffusion restoration / SR | x4 config | Whole-image diffusion restoration used as SR teacher with prompt controls | [Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR) | `configs/examples/local_supir_x4.yaml` |

## Datasets

For China-focused face-dataset research and selection notes, see:

- [China-focused face datasets](docs/face-datasets-china-focused.md)
- [中国人脸数据集整理](docs/face-datasets-china-focused.zh-CN.md)

## Reference YAMLs

Local example configs:

- `configs/examples/local_realesrgan_x2.yaml`
- `configs/examples/local_swinir_x2.yaml`
- `configs/examples/local_hat_x2.yaml`
- `configs/examples/local_gfpgan_x2.yaml`
- `configs/examples/local_codeformer_x2.yaml`
- `configs/examples/local_vqfr_x2.yaml`
- `configs/examples/local_stablesr_x4.yaml`
- `configs/examples/local_resshift_x4.yaml`
- `configs/examples/local_resshift_x4_realtest.yaml`
- `configs/examples/local_resshift_x4_realtest_lowmem.yaml`
- `configs/examples/local_supir_x4.yaml`

Real face-dataset reference configs:

- `configs/examples/scut_face_gfpgan_x2.yaml`
- `configs/examples/scut_face_codeformer_x2.yaml`
- `configs/examples/scut_face_vqfr_x2.yaml`

Five-image smoke-test configs:

- `configs/examples/scut_face_5_gfpgan_x2.yaml`
- `configs/examples/scut_face_5_codeformer_x2.yaml`
- `configs/examples/scut_face_5_vqfr_x2.yaml`

## Quick Start

Install the project dependencies in your Python environment:

```powershell
python -m pip install -e .[dev]
```

Prepare the local Real-ESRGAN repositories and x2 weights from the YAML config:

```powershell
python -m sr_data_maker.cli.main setup realesrgan --config configs/examples/local_realesrgan_x2.yaml --project-root .
```

Prepare SwinIR or HAT in the same style:

```powershell
python -m sr_data_maker.cli.main setup swinir --config configs/examples/local_swinir_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup hat --config configs/examples/local_hat_x2.yaml --project-root .
```

Prepare the face-focused teacher models in the same style:

```powershell
python -m sr_data_maker.cli.main setup gfpgan --config configs/examples/local_gfpgan_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup codeformer --config configs/examples/local_codeformer_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup vqfr --config configs/examples/local_vqfr_x2.yaml --project-root .
```

Prepare the diffusion-based real-world SR teachers in the same style:

```powershell
python -m sr_data_maker.cli.main setup stablesr --config configs/examples/local_stablesr_x4.yaml --project-root .
python -m sr_data_maker.cli.main setup resshift --config configs/examples/local_resshift_x4.yaml --project-root .
python -m sr_data_maker.cli.main setup supir --config configs/examples/local_supir_x4.yaml --project-root .
```

Run a teacher pipeline:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONPATH='src'
python -m sr_data_maker.cli.main run --config configs/examples/local_realesrgan_x2.yaml
```

On Linux, the same commands work with shell syntax for environment variables:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=src
python -m sr_data_maker.cli.main setup realesrgan --config configs/examples/local_realesrgan_x2.yaml --project-root .
python -m sr_data_maker.cli.main run --config configs/examples/local_realesrgan_x2.yaml
```

## Setup Commands

`setup realesrgan` reads enabled `RealESRGANRunner` tasks from the YAML config and prepares:

- `third_party/Real-ESRGAN`
- `third_party/BasicSR`
- `weights/RealESRGAN_x2plus.pth`
- local `version.py` compatibility files needed by the source checkouts

Supported default model downloads:

- `RealESRGAN_x2plus`
- `RealESRGAN_x4plus`

`setup swinir` and `setup hat` read enabled `SwinIRAdapter` / `HATAdapter` tasks from YAML and prepare:

- the configured `repo_root`
- the configured `basicsr_root` for HAT when present
- `model.weights` by downloading `model.download_url`

`setup gfpgan`, `setup codeformer`, and `setup vqfr` read enabled face-teacher tasks from YAML and prepare:

- the configured model repo root
- `facexlib` when configured
- `BasicSR` when configured
- `model.weights` by downloading `model.download_url`

`setup stablesr`, `setup resshift`, and `setup supir` read enabled diffusion-teacher tasks from YAML and prepare:

- the configured model repo root
- the configured `model.weights`
- any explicitly configured extra repo roots when added later

The example diffusion YAMLs are intentionally conservative for first-pass local testing on a single GPU. For `5070 Ti`-class cards, start with `ResShift`, then try `StableSR` with tiling, and treat `SUPIR` as the heaviest option in this first batch.

Real local validation note:

- `configs/examples/local_resshift_x4_realtest_lowmem.yaml` was verified against `C:/Users/kumi/Desktop/SR_HR_IMG_5/LR` on a `5070 Ti`-class card with `chop_size: 64` and `chop_stride: 32`.
- The matching output folder layout for side-by-side inspection is available under `data/outputs/local_resshift_x4_realtest_lowmem_compare/` with `input/` and `teacher/ResShift_x4/`.

Weight preparation notes:

- `ResShift` example YAML uses the official GitHub release weights and autoencoder checkpoint.
- `StableSR` example YAML uses the official Hugging Face `stablesr_turbo.ckpt` and `vqgan_cfw_00011.ckpt`.
- `SUPIR` currently ships best with manual checkpoint preparation because the official release is mainly distributed as Google Drive / Baidu folders rather than one stable direct file URL.

All setup commands are designed to be idempotent. Existing repositories and weights are skipped. `third_party/` and `weights/` stay ignored by Git so external code and model files remain local.

## Output Layout

Source images are not copied into the output dataset by default. Generated files preserve the relative path from `paths.input_root`:

```text
data/outputs/local_realesrgan_x2/
  manifests/
    samples.jsonl
    failures.jsonl
    run_summary.json
  teacher/
    RealESRGAN_x2plus/
      city/day/img001.png
```

For manual visual comparison, you can additionally place source images beside model outputs in an ad hoc `compare/` folder.

## User Workflow

Project commands are designed to be non-interactive. Development, setup, validation, test, and run commands should execute directly and report results through terminal output, manifests, and summaries.
