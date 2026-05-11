# Diffusion SR Models Design

## Purpose

This design extends SR Data Maker with a first batch of real-world diffusion-based super-resolution teacher models while keeping the existing YAML-driven teacher pipeline intact.

The first batch includes:

- `StableSRRunner`
- `ResShiftRunner`
- `SUPIRRunner`

These models are integrated as `teacher_superres` tasks for the current phase. Their outputs continue to be written under:

```text
teacher/<model_name>/<source_rel_path>
```

This keeps them compatible with the current dataset layout, manifests, resume behavior, and downstream comparison workflow.

## Scope Decision

The user explicitly wants real-world diffusion super-resolution models that are academically or industrially well known and reliable enough for actual local testing on a `5070 Ti`, with room to scale up to multiple `V100` GPUs later.

To avoid unnecessary architectural churn, this phase does **not** introduce a new generator type or a parallel diffusion-only pipeline. Instead, it reuses `TeacherSRGenerator` and the existing generic teacher structure, while extending the runner/config/setup layers so diffusion-specific parameters can be expressed cleanly.

## Goals

- Reuse the current `TeacherSRGenerator` execution path.
- Add local setup commands for `StableSR`, `ResShift`, and `SUPIR`.
- Add runnable teacher runners for the three models.
- Keep configuration YAML-driven and aligned with existing Real-ESRGAN, SwinIR, HAT, and face-teacher support.
- Preserve Windows and Linux compatibility.
- Record diffusion-model-specific provenance and inference parameters in the manifest.
- Provide example YAMLs that are conservative enough for first-pass local testing.

## Non-Goals

- No new `DiffusionTeacherGenerator` in this phase.
- No distributed multi-GPU orchestration in this phase.
- No benchmark harness or metric dashboard in this phase.
- No prompt search or automatic hyperparameter tuning.
- No attempt to normalize every third-party repository into one fully generic diffusion inference backend.

## Semantic Decision

The output semantics for these models in phase 1 are:

- `task_type`: `superres`
- `generation_mode`: `teacher_superres`
- `target.target_type`: `pseudo_gt`
- output root: `teacher/<model_name>/...`

The manifest should additionally record that the producing model is a diffusion-based real-world SR teacher, along with the actual inference controls used for the sample.

## Integration Strategy

### Recommended approach

Use the existing `TeacherSRGenerator` and add three new runner classes:

- `StableSRRunner`
- `ResShiftRunner`
- `SUPIRRunner`

Each runner should:

1. Accept one Pillow image from the standard runner input contract.
2. Load repository code lazily when executed.
3. Load local weights and model assets from YAML.
4. Produce one Pillow output image.
5. Return extra metadata for manifest provenance.

### Why this approach

- It stays aligned with the current plugin, executor, and dataset architecture.
- It minimizes invasive changes to the stable pipeline.
- It lets later diffusion models such as `SeeSR` or `InvSR` reuse the same extension pattern.
- It keeps this phase focused on reliable first-batch integration instead of broad framework invention.

## Configuration Shape

Each diffusion teacher task should keep the same top-level task shape already used by current teacher tasks:

```yaml
- name: teacher_resshift_x4
  enabled: true
  type: TeacherSRGenerator
  runner:
    type: ResShiftRunner
  model:
    name: ResShift_x4
    weights: ./weights/resshift/resshift_x4.safetensors
    repo_root: ./third_party/ResShift
    scale: 4
    steps: 15
    tile_size: 256
    tile_stride: 128
    precision: fp16
    download_url: <model-url>
  output:
    folder_name: ResShift_x4
    target_type: pseudo_gt
```

Shared required fields:

- `model.name`
- `model.weights`
- `model.repo_root`
- `model.scale`

Shared optional fields:

- `model.download_url`
- `model.device`
- `model.precision`
- `model.steps`
- `model.tile_size`
- `model.tile_stride`
- `model.max_input_size`
- `model.repo_url`
- `model.extra_repo_roots`

Model-specific fields:

- StableSR:
  - `vae_weights`
  - `tile_vae`
  - `guidance_scale`
  - `latent_tiled`
- ResShift:
  - `chop_size`
  - `chop_stride`
  - `sampler`
- SUPIR:
  - `prompt`
  - `negative_prompt`
  - `edm_steps`
  - `s_stage1`
  - `s_stage2`
  - `color_fix_type`
  - `use_llava`

The configuration should remain explicit. Diffusion-specific behavior must not be silently enabled.

## Manifest Provenance

The existing teacher manifest stays structurally the same, but runner metadata should include diffusion-model-specific provenance. At minimum, the runner meta should include:

```json
{
  "diffusion_model": true,
  "diffusion_model_family": "ResShift",
  "real_world_sr": true,
  "scale": 4,
  "steps": 15,
  "precision": "fp16"
}
```

Required diffusion metadata:

- `diffusion_model: true`
- `diffusion_model_family`
- `real_world_sr: true`
- `scale`

Optional metadata depending on runner:

- `steps`
- `precision`
- `tile_size`
- `tile_stride`
- `prompt`
- `negative_prompt`
- `guidance_scale`
- `color_fix_type`
- `sampler`

This metadata should be emitted by the runner via `RunnerOutput.meta`, not hard-coded into the generator.

## Setup Commands

Add three explicit setup commands:

- `setup stablesr`
- `setup resshift`
- `setup supir`

Each setup command should:

- read enabled matching tasks from the YAML config
- clone required repositories into `third_party/`
- download configured weights into `weights/`
- skip existing resources
- print an idempotent summary

Expected repository dependencies:

- StableSR:
  - `third_party/StableSR`
- ResShift:
  - `third_party/ResShift`
- SUPIR:
  - `third_party/SUPIR`

Some repositories may require additional helper repos or model assets. The setup path should keep those dependencies explicit rather than hidden inside the runner.

## Runner Behavior

### StableSRRunner

Responsibilities:

- import the local StableSR repo lazily
- validate required weights and repository paths
- support conservative tiled inference controls for limited VRAM
- return the final whole-image SR output

Expected metadata:

- `diffusion_model: true`
- `diffusion_model_family: StableSR`
- `scale`
- `steps`
- `tile_size`
- `precision`

### ResShiftRunner

Responsibilities:

- import the local ResShift repo lazily
- load configured weights
- support reduced-step inference suitable for real-world SR
- expose chopping/tile controls for smaller local GPUs
- return the final whole-image SR output

Expected metadata:

- `diffusion_model: true`
- `diffusion_model_family: ResShift`
- `scale`
- `steps`
- `precision`
- `chop_size`
- `chop_stride`

### SUPIRRunner

Responsibilities:

- import the local SUPIR repo lazily
- load configured weights and optional prompt controls
- support conservative memory-aware inference defaults
- return the final restored/SR image

Expected metadata:

- `diffusion_model: true`
- `diffusion_model_family: SUPIR`
- `scale`
- `steps`
- `precision`
- `prompt`
- `negative_prompt`
- `color_fix_type`

## Hardware Strategy

This phase should explicitly support two testing modes:

### Local-first mode

Target: single `5070 Ti`

Expectations:

- `ResShift` should be the easiest first-pass diffusion model to run locally.
- `StableSR` should prefer tiled or chopped inference when configured.
- `SUPIR` should ship with a conservative example config rather than an aggressive quality-first config.

### Scale-up mode

Target: one or more `V100` GPUs

Expectations:

- the same runners and YAMLs should remain valid
- users may raise `steps`, input size, or disable conservative tiling later
- no multi-GPU orchestration is required in this phase, but config semantics should not block it

## Integration Plan

### Files to add or extend

- `src/sr_data_maker/runners/teacher/diffusion_base.py`
- `src/sr_data_maker/runners/teacher/stablesr.py`
- `src/sr_data_maker/runners/teacher/resshift.py`
- `src/sr_data_maker/runners/teacher/supir.py`
- `src/sr_data_maker/setup/diffusion_teacher.py`
- `src/sr_data_maker/plugins.py`
- `src/sr_data_maker/cli/main.py`
- `README.md`
- `README.zh-CN.md`
- `configs/examples/local_stablesr_x4.yaml`
- `configs/examples/local_resshift_x4.yaml`
- `configs/examples/local_supir_x4.yaml`
- tests for setup, registry, runner contract, and CLI dispatch

### Files that should stay stable

- `TeacherSRGenerator`
- `PipelineExecutor`
- dataset layout
- state store format

If the implementation pressures these stable files too much, that is a sign the phase is trying to do too much and should be reduced.

## Testing Strategy

The test plan should match the rest of the project: verify the framework contract first, keep fast tests independent from third-party repositories, and defer real third-party inference to smoke-level validation.

Required tests:

- registry tests confirm the three runners are registered
- setup discovery tests confirm enabled YAML tasks are found correctly
- setup tests confirm repo clone and weight download behavior
- runner tests confirm missing weights fail clearly
- runner tests confirm diffusion parameters are propagated into model config and provenance
- CLI tests confirm `setup stablesr`, `setup resshift`, and `setup supir` dispatch correctly

Optional smoke tests:

- local one-image ResShift inference
- local one-image StableSR inference
- local one-image SUPIR inference

These smoke tests should remain separate from the fast unit suite because they depend on weights, third-party repos, and GPU state.

## Risks

### Repository dependency mismatch

Diffusion repos often have different assumptions about Torch versions, auxiliary assets, and environment configuration. Setup must keep repo roots explicit, and the first implementation should prefer clear failure over hidden environment guessing.

### Semantic mismatch between SR and restoration

`SUPIR` in particular is closer to real-world restoration than strict deterministic SR. Treating it as `teacher_superres` is acceptable only because this project is using it as a pseudo-target teacher and recording provenance clearly.

### VRAM pressure

These models are materially heavier than current CNN/GAN teachers. The first implementation must ship conservative example YAMLs and expose tiling/chopping controls in config.

### Over-abstraction too early

The three first-batch models should share only the setup and runner responsibilities that are genuinely common. Do not force a fully generic diffusion backend in phase 1.

## Phase Exit Criteria

This phase is complete when:

- all three diffusion teacher runners are registered
- all three setup commands work from YAML
- example configs exist for all three models
- the main test suite passes
- manifest provenance distinguishes these models as diffusion real-world SR teachers
- the implementation lives entirely on the isolated feature branch/worktree

## Recommendation

Proceed with a conservative first implementation:

1. Keep `TeacherSRGenerator` unchanged.
2. Add `StableSRRunner`, `ResShiftRunner`, and `SUPIRRunner`.
3. Add explicit setup commands for each model family.
4. Extend manifest provenance through runner metadata only.
5. Defer real multi-GPU orchestration and benchmark tooling to a later phase.
