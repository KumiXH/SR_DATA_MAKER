# SR Data Maker 设计文档

## 项目目标

SR Data Maker 是一个面向超分辨率和图像修复训练数据生产的 Python/PyTorch 框架。
项目从一棵源图片目录树出发，同时支持两种生成模式：

- 退化模式：把源图片视为真实目标图，生成一张退化图。
- 教师超分模式：把源图片视为低质量输入图，调用超分教师模型，生成一张教师输出图。
- 图像修复扩展模式：把源图片视为输入图，调用降噪、去紫边、去模糊、去压缩伪影等 image-to-image 修复模型，生成一张修复输出图。

这个项目的核心目标不是写一组一次性脚本，而是建立一套后续可以长期扩展的训练数据生成框架。
新的退化算法、学习型退化网络、PyTorch 超分模型、图像修复模型和外部开源仓库，都应该通过注册的 runner 或 adapter 接入，而不是修改核心流水线。

## 设计原则

- YAML 是主要用户配置入口。
- 配置风格参考 BasicSR、Real-ESRGAN 这类常见开源深度学习项目。
- 内部架构保持模块化，分为 source reader、generator、runner、adapter、dataset writer 和 orchestration。
- 输出目录要简单，直接服务图片比较和实验。
- 默认不把源图片复制到输出数据集里，避免额外占用空间。
- 生成图片目录必须保持源图片的相对路径结构。
- 教师输出目录必须包含教师模型名，方便多个模型做对比实验。
- 图像修复输出目录也必须包含模型名，方便多个修复模型做对比实验。
- manifest 保留，用于记录语义、参数、seed、模型名和失败信息。
- 代码必须同时兼容 Windows 和 Linux。
- GPU 卡号、精度、tile 大小、显存回退策略都通过 YAML 配置。
- 第一版实现以单机稳定 MVP 为目标，不做分布式平台。

## 开发协作偏好

这个项目的用户偏好是直接执行：后续任何命令都不需要先请求确认。为了完成当前任务所需的开发、检查、校验、测试、格式化和实现命令，都应该以非交互方式直接运行。

这个偏好需要落实到工具和脚本选择上：

- 优先使用非交互 CLI 命令。
- 正常项目命令执行前不暂停请求确认。
- 项目脚本默认不要设计成交互式提示，除非用户明确要求。
- 命令执行结果通过日志、summary 和生成报告体现。

## 核心语义

源图片本身没有固定语义。它在样本里的角色由生成任务决定。

退化模式中：

- `source.role` 是 `hr`。
- `input` 是生成出来的退化图。
- `target` 引用原始源图片。
- `target_type` 是 `real_gt`。
- MVP 中不把源图片复制到输出数据集。

教师超分模式中：

- `source.role` 是 `lq`。
- `input` 引用原始源图片。
- `target` 是教师模型输出。
- `target_type` 是 `pseudo_gt`。

项目里不应该把教师模型输出直接叫作普通 `GT`。
它是模型生成的伪目标图，语义上必须和真实 `real_gt` 区分开。

未来图像修复模式中：

- `source.role` 是 `input`。
- `input` 引用原始源图片。
- `target` 是修复模型输出。
- `target_type` 是 `pseudo_gt` 或 `restored_output`。
- 输出目录按修复模型名分组。

## 简化后的数据集目录结构

默认目录结构刻意保持简单：

```text
data/
  sources/
    raw/
      city/day/img001.png
      portrait/a/img002.jpg

  outputs/
    sr_mixed_v1/
      manifests/
        samples.jsonl
        failures.jsonl
        run_summary.json

      degraded/
        degradation_x2/
          city/day/img001.png
          portrait/a/img002.jpg

      teacher/
        RealESRGAN_x2plus/
          city/day/img001.png
          portrait/a/img002.jpg
        SwinIR_x2/
          city/day/img001.png
          portrait/a/img002.jpg

      restored/
        SwinIR_Denoise/
          city/day/img001.png
        DefringeNet_A/
          city/day/img001.png

      logs/
      tmp/
```

规则：

- `data/sources/raw` 是输入图片目录树。
- `outputs/<dataset_name>/degraded/<task_name>/...` 保存退化图。
- `outputs/<dataset_name>/teacher/<model_name>/...` 保存教师模型输出图。
- `outputs/<dataset_name>/restored/<model_name>/...` 预留给降噪、去紫边等图像修复输出。
- `<task_name>` 或 `<model_name>` 后面的路径保持源图片相对路径。
- 源图片默认不复制到 `outputs`。
- `manifests` 保留，因为它记录 source 路径、任务名、target type、模型名、参数、seed 和失败信息。
- `logs` 和 `tmp` 是运行辅助目录，生成完成后可以清理。

例如源图片路径是：

```text
data/sources/raw/city/day/img001.png
```

那么输出路径是：

```text
data/outputs/sr_mixed_v1/degraded/degradation_x2/city/day/img001.png
data/outputs/sr_mixed_v1/teacher/RealESRGAN_x2plus/city/day/img001.png
data/outputs/sr_mixed_v1/teacher/SwinIR_x2/city/day/img001.png
data/outputs/sr_mixed_v1/restored/SwinIR_Denoise/city/day/img001.png
```

这样同一个相对路径可以在源图、退化图、不同教师模型输出和不同修复模型输出之间一一对应，方便 compare。

## Manifest 记录

每个生成样本对应一条 manifest record。
manifest 不再用于支撑复杂目录结构，只保留文件夹看不出来的语义和可复现信息。

退化样本示例：

```json
{
  "sample_id": "degradation_x2::city/day/img001.png",
  "task_type": "superres",
  "generation_mode": "degradation",
  "source": {
    "root": "data/sources/raw",
    "rel_path": "city/day/img001.png",
    "role": "hr"
  },
  "input": {
    "path": "degraded/degradation_x2/city/day/img001.png",
    "role": "lq"
  },
  "target": {
    "source_ref": "city/day/img001.png",
    "role": "target",
    "target_type": "real_gt",
    "storage": "source_reference"
  },
  "outputs": [
    {
      "name": "degraded",
      "path": "degraded/degradation_x2/city/day/img001.png",
      "type": "image"
    }
  ],
  "provenance": {
    "generator": "DegradationGenerator",
    "runner": "ClassicalDegradationRunner",
    "task_name": "degradation_x2",
    "config_hash": "sha256:...",
    "seed": 1234,
    "params": {
      "scale": 2,
      "blur": {"type": "gaussian", "sigma": 1.2},
      "jpeg": {"quality": 70}
    }
  }
}
```

教师样本示例：

```json
{
  "sample_id": "RealESRGAN_x2plus::city/day/img001.png",
  "task_type": "superres",
  "generation_mode": "teacher_superres",
  "source": {
    "root": "data/sources/raw",
    "rel_path": "city/day/img001.png",
    "role": "lq"
  },
  "input": {
    "source_ref": "city/day/img001.png",
    "role": "lq",
    "storage": "source_reference"
  },
  "target": {
    "path": "teacher/RealESRGAN_x2plus/city/day/img001.png",
    "role": "target",
    "target_type": "pseudo_gt"
  },
  "outputs": [
    {
      "name": "teacher",
      "path": "teacher/RealESRGAN_x2plus/city/day/img001.png",
      "type": "image",
      "model_name": "RealESRGAN_x2plus"
    }
  ],
  "provenance": {
    "generator": "TeacherSRGenerator",
    "runner": "PyTorchTeacherRunner",
    "task_name": "teacher_sr_realesrgan",
    "model_name": "RealESRGAN_x2plus",
    "config_hash": "sha256:...",
    "seed": 1234
  }
}
```

未来图像修复样本示例：

```json
{
  "sample_id": "SwinIR_Denoise::city/day/img001.png",
  "task_type": "restoration",
  "generation_mode": "restoration",
  "source": {
    "root": "data/sources/raw",
    "rel_path": "city/day/img001.png",
    "role": "input"
  },
  "input": {
    "source_ref": "city/day/img001.png",
    "role": "input",
    "storage": "source_reference"
  },
  "target": {
    "path": "restored/SwinIR_Denoise/city/day/img001.png",
    "role": "target",
    "target_type": "restored_output"
  },
  "outputs": [
    {
      "name": "restored",
      "path": "restored/SwinIR_Denoise/city/day/img001.png",
      "type": "image",
      "model_name": "SwinIR_Denoise"
    }
  ],
  "provenance": {
    "generator": "RestorationGenerator",
    "runner": "SwinIRDenoiseRunner",
    "task_name": "denoise_swinir",
    "model_name": "SwinIR_Denoise",
    "config_hash": "sha256:...",
    "seed": 1234
  }
}
```

必须遵守的语义规则：

- `source.rel_path` 必须是相对于 `paths.input_root` 的路径。
- 生成文件必须在各自输出分支下保持 `source.rel_path`。
- MVP 阶段 `target.target_type` 只能是 `real_gt` 或 `pseudo_gt`。未来图像修复任务可以使用 `restored_output`。
- 退化参数默认写入 manifest，不再为每张图额外生成 JSON 文件。
- 教师输出文件夹名默认使用 `model.name`。
- 图像修复输出文件夹名默认使用 `model.name`。
- 退化输出文件夹名默认使用 `task.name`。

## YAML 配置设计

配置系统采用一个主 YAML 文件，并支持简单的 `base` 继承。
后面的配置覆盖前面的 base 配置。
第一版不做复杂表达式和动态插值。

示例配置：

```yaml
base:
  - ./base/runtime_single_gpu.yaml

name: sr_mixed_v1
model_type: SRDataMaker

runtime:
  device: cuda
  gpu_ids: [0]
  num_workers: 4
  batch_size: 1
  use_fp16: true
  cudnn_benchmark: true
  seed: 1234
  resume: true
  oom_fallback:
    enabled: true
    reduce_tile: true

paths:
  input_root: ./data/sources/raw
  output_root: ./data/outputs/sr_mixed_v1
  weights_root: ./weights
  log_root: ./data/outputs/sr_mixed_v1/logs
  temp_root: ./data/outputs/sr_mixed_v1/tmp

source:
  type: ImageFolderSourceReader
  recursive: true
  exts: [png, jpg, jpeg, webp]

layout:
  preserve_relative_path: true
  copy_source: false
  degraded_root: degraded
  teacher_root: teacher
  restored_root: restored
  filename_policy: keep_source_name

tasks:
  - name: degradation_x2
    enabled: true
    type: DegradationGenerator
    runner:
      type: ClassicalDegradationRunner
    policy:
      mode: repeat_n
      count: 2
    degradation:
      scale: 2
      blur:
        enabled: true
        type: gaussian
        kernel_size: 9
        sigma: [0.2, 2.0]
      resize:
        enabled: true
        mode: [bilinear, bicubic]
      noise:
        enabled: true
        type: gaussian
        std: [1, 10]
      jpeg:
        enabled: true
        quality: [30, 95]
    output:
      root: degraded
      folder_name: degradation_x2
      target_type: real_gt
      save_params_to_manifest: true

  - name: teacher_sr_realesrgan
    enabled: true
    type: TeacherSRGenerator
    runner:
      type: PyTorchTeacherRunner
    policy:
      mode: once
    model:
      type: RealESRGAN
      name: RealESRGAN_x2plus
      weights: ./weights/realesrgan_x2plus.pth
      scale: 2
      tile: 512
      tile_pad: 16
      pre_pad: 0
      half: true
    output:
      root: teacher
      folder_name: RealESRGAN_x2plus
      target_type: pseudo_gt

  # 未来扩展示例。MVP 不要求实现。
  - name: denoise_swinir
    enabled: false
    type: RestorationGenerator
    runner:
      type: SwinIRDenoiseRunner
    policy:
      mode: once
    model:
      type: SwinIRDenoise
      name: SwinIR_Denoise
      weights: ./weights/swinir_denoise.pth
      tile: 512
      half: true
    output:
      root: restored
      folder_name: SwinIR_Denoise
      target_type: restored_output

manifest:
  save_jsonl: true
  save_summary: true
```

配置分区说明：

- `runtime` 控制设备、GPU 卡号、精度、worker、随机种子、resume 和显存回退。
- `paths` 控制输入、输出、权重、日志和临时目录。
- `source` 控制输入数据读取方式。
- `layout` 控制简化后的镜像输出目录。
- `tasks` 定义生成任务。每个任务声明 generator type、runner type、任务参数、执行策略和输出位置。
- `degradation` 保存退化参数。
- `model` 保存教师超分模型或图像修复模型参数。
- `manifest` 控制 manifest 写入。

任务和参数开关规则：

- 每个 task 都支持 `enabled: true | false`。
- 被关闭的 task 会被跳过，也不要求对应 runner 或权重存在。
- 退化内部每个算子都有自己的 `enabled` 开关，所以 blur、resize、noise、JPEG 可以独立开启或关闭。
- Real-ESRGAN teacher 的 `weights`、`scale`、`tile`、`tile_pad`、`pre_pad`、`half` 等参数都从 YAML 读取。
- MVP 默认 teacher 使用 Real-ESRGAN 2x：`model.name: RealESRGAN_x2plus`，`scale: 2`。

## GPU 与跨平台要求

项目必须同时支持 Windows 和 Linux。

实现要求：

- 所有文件路径使用 `pathlib.Path`。
- 核心执行路径不依赖 Linux-only shell 脚本。
- 外部开源仓库调用统一通过 Python `subprocess` 封装。
- 临时目录、日志、缓存和输出目录都必须从 YAML 配置读取。
- 不假设系统一定支持 symlink，因为简化后的 MVP 不需要链接。
- 代码里不写死 CUDA 卡号。

GPU 行为：

- `runtime.gpu_ids` 选择当前运行使用的 CUDA 卡。
- MVP 阶段 `[0]` 在 PyTorch 内部映射到 `cuda:0`。
- V100 和 RTX 5070 Ti 通过不同 YAML 配置适配，不在代码里写型号分支。
- tile size、batch size、half precision 和 OOM fallback 都可以配置。
- 如果配置要求 `cuda` 但 CUDA 不可用，应该在 validate 阶段失败，而不是生成到一半才失败。

## 架构设计

Python 包采用 `src` 结构：

```text
src/
  sr_data_maker/
    config/
      loader.py
      merge.py
      validator.py
    core/
      types.py
      protocols.py
      registry.py
      exceptions.py
      context.py
    sources/
      image_folder.py
    generators/
      degradation.py
      teacher_sr.py
      restoration.py
    runners/
      degradation/
        classical.py
      teacher/
        pytorch_runner.py
      restoration/
        pytorch_runner.py
    adapters/
      degradation/
      teacher/
      restoration/
    dataset/
      writer.py
      manifest.py
      naming.py
    orchestration/
      executor.py
      orchestrator.py
      state_store.py
      retry.py
    validation/
      config_validator.py
      output_validator.py
      manifest_validator.py
    cli/
      main.py
      commands/
        run.py
        validate.py
        inspect.py
```

核心职责：

- `SourceReader` 读取源图片记录，并基于 `paths.input_root` 计算 `rel_path`。
- `Generator` 定义样本生成语义。
- `Runner` 执行一个具体算法或模型。
- `Adapter` 把外部 repo 或非标准模型 API 封装成 runner。
- `RestorationGenerator` 预留给降噪、去紫边等非超分 image-to-image 修复任务。
- `DatasetWriter` 按镜像相对路径布局写入生成图片和 manifest。
- `PipelineExecutor` 驱动整条流水线。
- `RunStateStore` 记录 completed、skipped、failed 状态，用于 resume。

## 接口设计

MVP 阶段接口保持小而稳定。

```python
class SourceRecord:
    source_id: str
    path: str
    rel_path: str
    meta: dict
```

```python
class SourceReader:
    def iter_sources(self):
        ...
```

```python
class RunnerOutput:
    outputs: dict
    meta: dict
```

```python
class Runner:
    name: str

    def run(self, inputs: dict, context):
        ...
```

```python
class Generator:
    name: str

    def generate(self, source: SourceRecord, context):
        ...
```

`Generator` 负责 manifest 语义。
`Runner` 负责实际执行。
`DatasetWriter` 负责输出路径生成、文件写入和 manifest 持久化。

## 执行流程

运行流程如下：

1. 读取并合并 YAML 配置。
2. 校验配置、路径、registry、CUDA 状态和输出语义。
3. 构建 source reader、generator、runner、dataset writer 和 run state store。
4. 遍历 source records，并计算稳定的 `rel_path`。
5. 对每个 source 和 task 检查 resume 状态。
6. 执行 generator 和 runner。
7. 根据输出分支、任务或模型文件夹、`source.rel_path` 拼出输出路径。
8. 校验生成文件和 manifest record。
9. 写入生成图片，并追加 `samples.jsonl`。
10. 失败样本写入 `failures.jsonl`。
11. 写入 `run_summary.json`。

除配置错误外，其他失败应尽量隔离在单个 sample 或 source-task 粒度。

MVP 支持的运行模式：

- `resume`：跳过已完成的 source-task。
- `force_rerun`：忽略历史状态，重新生成。
- `retry_failed`：只重跑失败记录。

第一版可以使用 JSONL 作为状态存储。
设计上保留未来替换为 SQLite 的空间。

## MVP 范围

第一版包含：

- `ImageFolderSourceReader`
- `DegradationGenerator`
- `TeacherSRGenerator`
- 为未来 image-to-image 修复任务预留 `RestorationGenerator` 接口
- `ClassicalDegradationRunner`
- Real-ESRGAN、SwinIR、HAT 这类 `PyTorchTeacherRunner` 风格的 teacher adapter
- `DatasetWriter`
- `PipelineExecutor`
- 支持简单 `base` 继承的 YAML 配置加载器
- 基于 registry 的组件构建
- CLI 命令：`run`、`validate`、`inspect`、`setup realesrgan`
- `samples.jsonl`、`failures.jsonl`、`run_summary.json`
- 退化图和教师图的镜像输出目录
- 为未来修复模型预留 `restored/<model_name>/<source_rel_path>` 输出布局
- resume 支持
- Windows 和 Linux 路径兼容

第一版不包含：

- 把源图片复制到输出数据集
- 之前的 `artifacts` 和 `views` 目录层
- 分布式执行
- 多节点调度
- 复杂多 GPU 调度
- Web UI
- 大量外部模型 adapter
- 学习型退化网络
- 除预留接口和配置形态以外的完整图像修复模型实现
- 生成过程中隐式下载模型。Real-ESRGAN 本地仓库和权重通过 `setup realesrgan` 显式准备。
- 复杂报表看板
- 完整 SQLite 状态存储

## 校验策略

启动前校验应该发现：

- YAML 结构非法
- 路径缺失
- 模型任务缺少权重
- 未知 `type`
- 未知 runner
- 非法 `target_type`
- task 类型和 output 语义不匹配
- 配置要求 CUDA 但 CUDA 不可用
- 输出目录无写入权限
- 输出文件夹名非法

运行时 output 校验应该发现：

- 生成文件缺失
- 生成图片不可读
- 图片尺寸非法
- manifest 字段缺失
- 输出路径非法
- sample ID 重复
- 输出路径没有保持源图片相对路径

`inspect` 命令应该输出：

- source 图片数量
- 生成样本数量
- generation mode 分布
- target type 分布
- degradation task 分布
- teacher model 分布
- 如果存在图像修复任务，输出 restoration model 分布
- 失败数量
- source 和 generated 图片尺寸分布
- 缺失输出数量

## 测试策略

第一版测试先验证框架本身，不依赖真实大模型：

- 配置加载和 base merge 单元测试。
- registry 构建单元测试。
- manifest record 校验单元测试。
- 镜像输出路径生成单元测试。
- 使用嵌套图片目录、mock degradation runner、mock teacher runner 的集成测试。
- 实现 `RestorationGenerator` 后，可以增加 mock restoration runner 的接口级测试。
- resume 测试，确认同一配置重跑会跳过已完成任务。
- 基于 `pathlib` 的跨平台路径测试。

真实开源模型 adapter 应该单独做 smoke test，因为它们依赖权重和 GPU 环境。

## 实现顺序

推荐实现顺序：

1. 项目骨架和 packaging。
2. 配置加载器、merge 逻辑和 validator。
3. core types、protocols、registry 和 context。
4. 支持嵌套目录遍历和 `rel_path` 生成的图片文件夹 source reader。
5. dataset writer、naming、manifest writer 和镜像输出路径 writer。
6. mock 或最小 classical degradation runner。
7. degradation generator。
8. 最小 PyTorch teacher runner 和 teacher generator。
9. pipeline executor 和 run state。
10. CLI 命令。
11. inspect 命令和基础 summary 报告。
12. 测试和示例配置。
13. 可选后续项：`RestorationGenerator` 和 mock restoration runner。

## MVP 默认决策

- 源图片默认不复制到 `output_root`。
- 退化图输出到 `degraded/<task_name>/<source_rel_path>`。
- 教师图输出到 `teacher/<model_name>/<source_rel_path>`。
- 未来图像修复输出到 `restored/<model_name>/<source_rel_path>`。
- 教师文件夹名默认使用 `model.name`，也允许 task 配置用 `output.folder_name` 覆盖。
- 图像修复文件夹名默认使用 `model.name`，也允许 task 配置用 `output.folder_name` 覆盖。
- 退化文件夹名默认使用 `task.name`，也允许 task 配置用 `output.folder_name` 覆盖。
- 第一批真实超分 teacher adapter 覆盖 Real-ESRGAN 2x、SwinIR x2 和 HAT x2。它们统一使用 `model.name`、`weights`、`scale`、`tile`、`tile_pad`、`half` 等 YAML 字段。
- MVP 退化使用常规算子：resize、blur、noise 和 JPEG compression。
- 退化任务和 teacher 任务都可以在 YAML 中独立开启或关闭。
- 初始状态存储使用 JSONL。等数据规模带来查询瓶颈后，再替换为 SQLite。
