# SR Data Maker

[English README](README.md)

一个基于 YAML 的超分训练数据生成工具，用来批量产出退化数据和 teacher 数据。

当前项目支持：

- 在 `degraded/<task_name>/<source_rel_path>` 下输出退化结果。
- 在 `teacher/<model_name>/<source_rel_path>` 下输出 teacher 模型结果。
- 在同一套流水线中同时支持通用超分模型和人脸超分/修复模型。
- 保留输入目录的相对路径结构。
- 输出 JSONL manifest 记录溯源信息。
- 支持 Windows 和 Linux 的非交互式命令执行。

## 模型总表

下面这张表汇总了仓库里当前已经接入的超分与人脸 teacher 模型，方便快速查看。

| 模型 | Runner / Adapter | 类别 | 标称倍率 | 实际输出行为 | 来源 | 示例 YAML |
| --- | --- | --- | --- | --- | --- | --- |
| Real-ESRGAN x2plus | `RealESRGANRunner` | 通用超分 | x2 | 整图 x2 超分 | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | `configs/examples/local_realesrgan_x2.yaml` |
| SwinIR x2 classical | `SwinIRAdapter` | 通用超分 | x2 | 整图 x2 超分 | [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | `configs/examples/local_swinir_x2.yaml` |
| HAT SRx2 | `HATAdapter` | 通用超分 | x2 | 整图 x2 超分 | [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | `configs/examples/local_hat_x2.yaml` |
| GFPGAN v1.4 | `GFPGANRunner` | 人脸超分 / 修复 | x2 | 整图人脸修复并输出 x2 结果 | [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | `configs/examples/local_gfpgan_x2.yaml` |
| CodeFormer | `CodeFormerRunner` | 人脸超分 / 修复 | 配置为 x2 | 走官方整图人脸修复与 face upsample 路线，最终输出尺寸可能大于严格 x2 | [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | `configs/examples/local_codeformer_x2.yaml` |
| VQFR v2 | `VQFRRunner` | 人脸超分 / 修复 | x2 | 整图人脸修复并输出 x2 结果 | [TencentARC/VQFR](https://github.com/TencentARC/VQFR) | `configs/examples/local_vqfr_x2.yaml` |
| StableSR | `StableSRRunner` | Diffusion 真实世界超分 | x4 配置 | 支持保守 tile 配置的整图 diffusion 超分 | [IceClear/StableSR](https://github.com/IceClear/StableSR) | `configs/examples/local_stablesr_x4.yaml` |
| ResShift | `ResShiftRunner` | Diffusion 真实世界超分 | x4 配置 | 低步数 real-world SR 路线的整图 diffusion 超分 | [zsyOAOA/ResShift](https://github.com/zsyOAOA/ResShift) | `configs/examples/local_resshift_x4.yaml` |
| SUPIR | `SUPIRRunner` | Diffusion 修复 / 超分 | x4 配置 | 作为 teacher 使用的整图 diffusion 修复与超分 | [Fanghua-Yu/SUPIR](https://github.com/Fanghua-Yu/SUPIR) | `configs/examples/local_supir_x4.yaml` |

## 数据集

如果要看偏中国/东亚人脸的数据集调研和选型说明，可以直接跳转到：

- [China-focused face datasets](docs/face-datasets-china-focused.md)
- [中国人脸数据集整理](docs/face-datasets-china-focused.zh-CN.md)

## 参考 YAML

本地示例配置：

- `configs/examples/local_realesrgan_x2.yaml`
- `configs/examples/local_swinir_x2.yaml`
- `configs/examples/local_hat_x2.yaml`
- `configs/examples/local_gfpgan_x2.yaml`
- `configs/examples/local_codeformer_x2.yaml`
- `configs/examples/local_vqfr_x2.yaml`
- `configs/examples/local_stablesr_x4.yaml`
- `configs/examples/local_resshift_x4.yaml`
- `configs/examples/local_supir_x4.yaml`

真实人脸数据集参考配置：

- `configs/examples/scut_face_gfpgan_x2.yaml`
- `configs/examples/scut_face_codeformer_x2.yaml`
- `configs/examples/scut_face_vqfr_x2.yaml`

五张样本 smoke test 配置：

- `configs/examples/scut_face_5_gfpgan_x2.yaml`
- `configs/examples/scut_face_5_codeformer_x2.yaml`
- `configs/examples/scut_face_5_vqfr_x2.yaml`

## 快速开始

先在当前 Python 环境中安装项目依赖：

```powershell
python -m pip install -e .[dev]
```

按 YAML 配置准备 Real-ESRGAN 本地仓库和权重：

```powershell
python -m sr_data_maker.cli.main setup realesrgan --config configs/examples/local_realesrgan_x2.yaml --project-root .
```

SwinIR 和 HAT 也是同样的准备方式：

```powershell
python -m sr_data_maker.cli.main setup swinir --config configs/examples/local_swinir_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup hat --config configs/examples/local_hat_x2.yaml --project-root .
```

人脸 teacher 模型也是同样的准备方式：

```powershell
python -m sr_data_maker.cli.main setup gfpgan --config configs/examples/local_gfpgan_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup codeformer --config configs/examples/local_codeformer_x2.yaml --project-root .
python -m sr_data_maker.cli.main setup vqfr --config configs/examples/local_vqfr_x2.yaml --project-root .
```

Diffusion 真实世界超分 teacher 也是同样的准备方式：

```powershell
python -m sr_data_maker.cli.main setup stablesr --config configs/examples/local_stablesr_x4.yaml --project-root .
python -m sr_data_maker.cli.main setup resshift --config configs/examples/local_resshift_x4.yaml --project-root .
python -m sr_data_maker.cli.main setup supir --config configs/examples/local_supir_x4.yaml --project-root .
```

权重准备说明：

- `ResShift` 示例 YAML 直接使用官方 GitHub Release 权重和 autoencoder 权重。
- `StableSR` 示例 YAML 直接使用官方 Hugging Face 的 `stablesr_turbo.ckpt` 和 `vqgan_cfw_00011.ckpt`。
- `SUPIR` 官方发布更偏 Google Drive / 百度网盘目录分发，所以当前示例更适合“手动准备 checkpoint 后再跑”，不建议把它当成和前两者完全一样的自动下载流程。

运行一个 teacher 流水线：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:PYTHONPATH='src'
python -m sr_data_maker.cli.main run --config configs/examples/local_realesrgan_x2.yaml
```

Linux 下可以用下面这种环境变量写法：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=src
python -m sr_data_maker.cli.main setup realesrgan --config configs/examples/local_realesrgan_x2.yaml --project-root .
python -m sr_data_maker.cli.main run --config configs/examples/local_realesrgan_x2.yaml
```

## Setup 命令说明

`setup realesrgan` 会从 YAML 中读取启用的 `RealESRGANRunner` 任务，并准备：

- `third_party/Real-ESRGAN`
- `third_party/BasicSR`
- `weights/RealESRGAN_x2plus.pth`
- 源码仓库需要的本地 `version.py` 兼容文件

默认支持的 Real-ESRGAN 权重：

- `RealESRGAN_x2plus`
- `RealESRGAN_x4plus`

`setup swinir` 和 `setup hat` 会从 YAML 中读取启用的 `SwinIRAdapter` / `HATAdapter` 任务，并准备：

- 配置里的 `repo_root`
- HAT 需要时的 `basicsr_root`
- 通过 `model.download_url` 下载 `model.weights`

`setup gfpgan`、`setup codeformer`、`setup vqfr` 会从 YAML 中读取启用的人脸 teacher 任务，并准备：

- 配置里的模型仓库目录
- 配置里的人脸依赖 `facexlib`
- 配置里的 `BasicSR`
- 通过 `model.download_url` 下载 `model.weights`

这些 setup 命令都按幂等方式设计：已有仓库和权重会被跳过。`third_party/` 和 `weights/` 默认不纳入 Git，因此外部源码和模型文件保持本地状态。

## 输出目录结构

默认情况下，源图不会被复制进输出数据集。生成结果会保留 `paths.input_root` 之下的相对路径：

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

如果需要人工比图，可以额外在输出目录下整理 `compare/` 文件夹，把原图和输出图放在一起。

## 使用约定

项目命令都按非交互方式设计。开发、setup、验证、测试和运行命令应该直接执行，并通过终端输出、manifest 和 summary 汇报结果。
