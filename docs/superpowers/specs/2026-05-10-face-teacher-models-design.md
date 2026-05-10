# Face Teacher Models Design

## Purpose

This design extends SR Data Maker with a first batch of face-focused open-source models while keeping the existing teacher pipeline intact.

The first batch includes:

- `GFPGANRunner`
- `CodeFormerRunner`
- `VQFRRunner`

These models are integrated as `teacher_superres` tasks for the current phase. Their outputs are written under:

```text
teacher/<model_name>/<source_rel_path>
```

This keeps them compatible with the existing dataset layout, resume behavior, manifests, and comparison workflow.

## Scope Decision

Although GFPGAN, CodeFormer, and VQFR are closer to face restoration than pure super-resolution, the user explicitly wants the first batch to behave as teacher models.

To avoid unnecessary architectural churn, this phase does **not** introduce a new generator type. Instead, it reuses `TeacherSRGenerator` and adds face-model-specific provenance metadata so later phases can distinguish these outputs from general super-resolution teachers.

## Goals

- Reuse the current `TeacherSRGenerator` execution path.
- Add local setup commands for GFPGAN, CodeFormer, and VQFR.
- Add runnable teacher adapters for the three models.
- Keep configuration YAML-driven and consistent with existing Real-ESRGAN, SwinIR, and HAT support.
- Preserve Windows and Linux compatibility.
- Record model-specific restoration parameters in the manifest.

## Non-Goals

- No new `FaceTeacherGenerator` in this phase.
- No new `restoration` output branch for these models in this phase.
- No automatic model download during `run`.
- No face-specific evaluation metrics or quality scoring.
- No attempt to normalize model behavior across repositories beyond a shared runner contract.

## Semantic Decision

The output semantics for these models in phase 1 are:

- `task_type`: `superres`
- `generation_mode`: `teacher_superres`
- `target.target_type`: `pseudo_gt`
- output root: `teacher/<model_name>/...`

The manifest should additionally record that the producing model is a face-focused teacher. This keeps downstream training compatible with the current teacher-based workflow while preserving enough provenance to separate face restoration teachers from generic SR teachers later.

## Adapter Strategy

### Recommended approach

Use the existing `TeacherSRGenerator` and add three new runner classes:

- `GFPGANRunner`
- `CodeFormerRunner`
- `VQFRRunner`

Each runner should:

1. Accept one Pillow image from the standard runner input contract.
2. Load repository code only when executed.
3. Load local weights from YAML.
4. Produce one Pillow output image.
5. Return extra metadata for manifest provenance.

### Why this approach

- It keeps the implementation aligned with the current plugin and executor architecture.
- It minimizes invasive changes to pipeline semantics.
- It allows us to ship working support faster.
- It leaves room for a future `RestorationGenerator` without forcing that migration now.

## Configuration Shape

Each face teacher task should keep the same top-level task structure used by current teacher tasks:

```yaml
- name: teacher_face_codeformer
  enabled: true
  type: TeacherSRGenerator
  runner:
    type: CodeFormerRunner
  model:
    name: CodeFormer_face_upsample_x2
    weights: ./weights/codeformer.pth
    repo_root: ./third_party/CodeFormer
    facelib_root: ./third_party/facexlib
    basicsr_root: ./third_party/BasicSR
    fidelity_weight: 0.7
    face_upsample: true
    background_upsampler: realesrgan
    bg_tile: 512
    scale: 2
    half: false
    download_url: <model-url>
  output:
    folder_name: CodeFormer_face_upsample_x2
    target_type: pseudo_gt
```

Shared required fields:

- `model.name`
- `model.weights`
- `model.repo_root`
- `model.scale`

Shared optional fields:

- `model.download_url`
- `model.half`
- `model.device`
- `model.extra_repo_roots`

Model-specific fields:

- GFPGAN:
  - `channel_multiplier`
  - `arch`
  - `bg_upsampler`
  - `bg_tile`
  - `only_center_face`
- CodeFormer:
  - `fidelity_weight`
  - `face_upsample`
  - `background_upsampler`
  - `bg_tile`
  - `has_aligned`
  - `only_center_face`
- VQFR:
  - `fidelity_ratio`
  - `bg_upsampler`
  - `bg_tile`
  - `arch`

## Manifest Provenance

The existing teacher manifest stays structurally the same, but runner metadata should include face-model-specific provenance. At minimum, the runner meta should include:

```json
{
  "face_model": true,
  "face_model_family": "CodeFormer",
  "scale": 2,
  "fidelity_weight": 0.7,
  "background_upsampler": "realesrgan"
}
```

Required face metadata:

- `face_model: true`
- `face_model_family`
- `scale`

Optional metadata depending on runner:

- `fidelity_weight`
- `fidelity_ratio`
- `bg_upsampler`
- `face_upsample`
- `only_center_face`
- `arch`

This metadata should be emitted by the runner via `RunnerOutput.meta`, not hard-coded into the generator.

## Setup Commands

Add three explicit setup commands:

- `setup gfpgan`
- `setup codeformer`
- `setup vqfr`

Each setup command should:

- read enabled matching tasks from the YAML config
- clone required repositories into `third_party/`
- download configured weights into `weights/`
- skip existing resources
- print an idempotent summary

Expected repository dependencies:

- GFPGAN:
  - `third_party/GFPGAN`
  - `third_party/BasicSR`
  - `third_party/facexlib`
- CodeFormer:
  - `third_party/CodeFormer`
  - `third_party/BasicSR`
  - `third_party/facexlib`
- VQFR:
  - `third_party/VQFR`
  - `third_party/BasicSR`
  - `third_party/facexlib`

The exact dependency set may vary by repository layout, but setup should be explicit and local rather than hidden inside the runner.

## Runner Behavior

### GFPGANRunner

Responsibilities:

- import the local GFPGAN repo lazily
- build the restorer with configured architecture parameters
- optionally use a background upsampler when configured
- run whole-image face restoration and return the merged final image

Expected metadata:

- `face_model: true`
- `face_model_family: GFPGAN`
- `scale`
- `arch`
- `channel_multiplier`
- `bg_upsampler`

### CodeFormerRunner

Responsibilities:

- import the local CodeFormer repo lazily
- build the restorer/inference helper
- support `fidelity_weight`
- optionally enable face upsample or background upsample
- return the merged final image

Expected metadata:

- `face_model: true`
- `face_model_family: CodeFormer`
- `scale`
- `fidelity_weight`
- `face_upsample`
- `background_upsampler`

### VQFRRunner

Responsibilities:

- import the local VQFR repo lazily
- load the configured restoration model
- optionally use configured background upsampling
- return the merged final image

Expected metadata:

- `face_model: true`
- `face_model_family: VQFR`
- `scale`
- `fidelity_ratio`
- `bg_upsampler`

## Integration Plan

### Files to add or extend

- `src/sr_data_maker/runners/teacher/gfpgan.py`
- `src/sr_data_maker/runners/teacher/codeformer.py`
- `src/sr_data_maker/runners/teacher/vqfr.py`
- `src/sr_data_maker/setup/face_teacher.py`
- `src/sr_data_maker/plugins.py`
- `src/sr_data_maker/cli/main.py`
- `README.md`
- `configs/examples/local_gfpgan_x2.yaml`
- `configs/examples/local_codeformer_x2.yaml`
- `configs/examples/local_vqfr_x2.yaml`
- tests for setup, registry, and runner contract

### Files that should stay stable

- `TeacherSRGenerator`
- `PipelineExecutor`
- dataset layout
- state store format

If any implementation change pressures these stable files too much, that is a signal that the phase is trying to do too much and should be reduced.

## Testing Strategy

The test plan should match the rest of the project: verify the framework contract first, and keep full third-party inference as smoke-level validation.

Required tests:

- registry tests confirm the three runners are registered
- setup discovery tests confirm enabled YAML tasks are found correctly
- setup tests confirm repo clone and weight download behavior
- runner tests confirm missing weights fail clearly
- runner tests confirm YAML parameters are propagated into model config
- CLI tests confirm `setup gfpgan`, `setup codeformer`, and `setup vqfr` dispatch correctly

Optional smoke tests:

- local one-image GFPGAN inference
- local one-image CodeFormer inference
- local one-image VQFR inference

These smoke tests should be separate from the fast unit suite because they depend on weights and repo state.

## Risks

### Repository dependency mismatch

Face restoration repos often pin different `BasicSR` or `facexlib` revisions. Setup must make repo roots explicit, and the first implementation should prefer clear failure over hidden version guessing.

### Semantic mismatch

These are not pure SR models. Treating them as `teacher_superres` is acceptable only because we are explicitly recording them as face-focused pseudo-target teachers in the manifest.

### Background upsampler coupling

GFPGAN and CodeFormer often rely on Real-ESRGAN as a background upsampler. In this phase, that dependency should be optional and explicit in YAML, not silently enabled.

### Whole-image vs aligned-face assumptions

Some repos support both aligned face input and whole-image restoration. The first implementation should target whole-image processing because it fits the project's folder-driven pipeline better.

## Phase Exit Criteria

This phase is complete when:

- all three face teacher runners are registered
- all three setup commands work from YAML
- example configs exist for all three models
- the main test suite passes
- manifest provenance distinguishes these models as face-focused teachers
- the implementation lives entirely on the isolated feature branch/worktree

## Recommendation

Proceed with a conservative first implementation:

1. Keep `TeacherSRGenerator` unchanged.
2. Add `GFPGANRunner`, `CodeFormerRunner`, and `VQFRRunner`.
3. Add explicit setup commands for each model family.
4. Extend manifest provenance through runner metadata only.
5. Defer any new generator type or `restored/` output semantics to a later phase.
