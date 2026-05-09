# SR Data Maker

YAML-driven training data generation toolkit for super-resolution datasets.

The MVP supports:

- Classical degradation output under `degraded/<task_name>/<source_rel_path>`.
- Real-ESRGAN, SwinIR, and HAT teacher output under `teacher/<model_name>/<source_rel_path>`.
- Nested source image folders with mirrored output paths.
- JSONL manifests for provenance.
- Windows and Linux execution with non-interactive commands.

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

Run the Real-ESRGAN x2 teacher pipeline:

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

The command is idempotent. Existing repositories and weights are skipped. `third_party/` and `weights/` are intentionally ignored by Git, so generated external code and model files stay local.

Supported default model downloads:

- `RealESRGAN_x2plus`
- `RealESRGAN_x4plus`

`setup swinir` and `setup hat` read enabled `SwinIRAdapter` / `HATAdapter` tasks from YAML and prepare:

- the configured `repo_root`
- the configured `basicsr_root` for HAT when present
- `model.weights` by downloading `model.download_url`

SwinIR uses an official GitHub release link in `configs/examples/local_swinir_x2.yaml`. HAT's official project distributes weights mostly through Google Drive, so `configs/examples/local_hat_x2.yaml` uses a HuggingFace mirror link. Replace `model.download_url` with your preferred internal or official mirror when needed.

## SwinIR And HAT Adapters

SwinIR and HAT use the same teacher task shape as Real-ESRGAN:

```yaml
runner:
  type: SwinIRAdapter
model:
  name: SwinIR_x2_classical
  weights: ./weights/SwinIR_x2_classical.pth
  download_url: https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth
  repo_root: ./third_party/SwinIR
  scale: 2
  tile: 512
  tile_pad: 16
  half: false
```

```yaml
runner:
  type: HATAdapter
model:
  name: HAT_SRx2
  weights: ./weights/HAT_SRx2.pth
  download_url: https://huggingface.co/jaideepsingh/upscale_models/resolve/main/HAT/HAT_SRx2.pth?download=true
  repo_root: ./third_party/HAT
  basicsr_root: ./third_party/BasicSR
  scale: 2
  tile: 512
  tile_pad: 16
  half: false
```

Example configs are available at `configs/examples/local_swinir_x2.yaml` and `configs/examples/local_hat_x2.yaml`.

## Output Layout

Source images are not copied into the output dataset. Generated files preserve the relative path from `paths.input_root`:

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

## User Workflow

Project commands are designed to be non-interactive. Development, setup, validation, test, and run commands should execute directly and report results through terminal output, manifests, and summaries.
