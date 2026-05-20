# Diffusion SR Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add StableSR, ResShift, and SUPIR as first-batch diffusion-based real-world super-resolution teachers in the existing `teacher_superres` pipeline.

**Architecture:** Keep `TeacherSRGenerator`, `PipelineExecutor`, and the dataset layout unchanged. Extend the project by adding three new teacher runners, a shared diffusion-teacher setup module, CLI setup entrypoints, example YAML configs, and tests that verify registry/setup/runner contract behavior before any real third-party inference.

**Tech Stack:** Python 3.10+, PyTorch, Pillow, pytest, argparse, pathlib, existing registry/setup helpers, local third-party repos under `third_party/`.

---

## File Structure

- Create `src/sr_data_maker/runners/teacher/diffusion_base.py`: shared helper for diffusion-teacher runners to validate weights, add repo roots to `sys.path`, and return standardized provenance metadata.
- Create `src/sr_data_maker/runners/teacher/stablesr.py`: StableSR runner implementation.
- Create `src/sr_data_maker/runners/teacher/resshift.py`: ResShift runner implementation.
- Create `src/sr_data_maker/runners/teacher/supir.py`: SUPIR runner implementation.
- Create `src/sr_data_maker/setup/diffusion_teacher.py`: generic setup support for StableSR, ResShift, and SUPIR tasks.
- Modify `src/sr_data_maker/plugins.py`: register the three new runners.
- Modify `src/sr_data_maker/cli/main.py`: add `setup stablesr`, `setup resshift`, and `setup supir`.
- Modify `README.md`: document setup and run flow for the diffusion teacher models.
- Modify `README.zh-CN.md`: add the corresponding Chinese documentation.
- Create `configs/examples/local_stablesr_x4.yaml`: example config for StableSR teacher mode.
- Create `configs/examples/local_resshift_x4.yaml`: example config for ResShift teacher mode.
- Create `configs/examples/local_supir_x4.yaml`: example config for SUPIR teacher mode.
- Create `tests/test_diffusion_teacher_setup.py`: setup discovery and repo/weight preparation tests.
- Create `tests/test_diffusion_teacher_runners.py`: missing-weight and parameter-propagation tests.
- Modify `tests/test_executor_cli.py`: CLI dispatch coverage for new setup commands.
- Modify `tests/test_pytorch_teacher_adapters.py`: registry coverage for new diffusion runners.
- Modify `tests/test_generators.py`: teacher manifest provenance coverage for diffusion-model metadata.

---

## Task 1: Add diffusion teacher setup tests

**Files:**
- Create: `tests/test_diffusion_teacher_setup.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_diffusion_teacher_setup.py -q`
- [ ] **Step 3: Write minimal implementation in `src/sr_data_maker/setup/diffusion_teacher.py`**
- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_diffusion_teacher_setup.py -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "feat: add diffusion teacher setup support"`

## Task 2: Add diffusion runner contract tests

**Files:**
- Create: `tests/test_diffusion_teacher_runners.py`
- Create: `src/sr_data_maker/runners/teacher/diffusion_base.py`
- Create: `src/sr_data_maker/runners/teacher/stablesr.py`
- Create: `src/sr_data_maker/runners/teacher/resshift.py`
- Create: `src/sr_data_maker/runners/teacher/supir.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_diffusion_teacher_runners.py -q`
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_diffusion_teacher_runners.py -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "feat: add diffusion teacher runner contracts"`

## Task 3: Register new runners and cover registry behavior

**Files:**
- Modify: `src/sr_data_maker/plugins.py`
- Modify: `tests/test_pytorch_teacher_adapters.py`

- [ ] **Step 1: Write the failing registry test**
- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_pytorch_teacher_adapters.py::test_builtin_registry_includes_diffusion_teacher_runners -q`
- [ ] **Step 3: Register `StableSRRunner`, `ResShiftRunner`, and `SUPIRRunner`**
- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_pytorch_teacher_adapters.py::test_builtin_registry_includes_diffusion_teacher_runners -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "feat: register diffusion teacher runners"`

## Task 4: Add CLI setup command tests

**Files:**
- Modify: `tests/test_executor_cli.py`
- Modify: `src/sr_data_maker/cli/main.py`

- [ ] **Step 1: Write failing tests for `setup stablesr`, `setup resshift`, and `setup supir`**
- [ ] **Step 2: Run tests to verify they fail**
  Run: `python -m pytest tests/test_executor_cli.py -q`
- [ ] **Step 3: Add the setup subcommands and config dispatch**
- [ ] **Step 4: Run tests to verify they pass**
  Run: `python -m pytest tests/test_executor_cli.py -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "feat: add diffusion teacher setup commands"`

## Task 5: Extend teacher manifest coverage for diffusion provenance

**Files:**
- Modify: `tests/test_generators.py`
- Modify production code only if required by the test result

- [ ] **Step 1: Write the failing manifest provenance test**
- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_generators.py::test_teacher_generator_preserves_diffusion_runner_provenance -q`
- [ ] **Step 3: Fix only the minimal manifest merge path if needed**
- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_generators.py::test_teacher_generator_preserves_diffusion_runner_provenance -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "test: cover diffusion teacher provenance in manifest"`

## Task 6: Add example configs and README coverage

**Files:**
- Create: `configs/examples/local_stablesr_x4.yaml`
- Create: `configs/examples/local_resshift_x4.yaml`
- Create: `configs/examples/local_supir_x4.yaml`
- Modify: `README.md`
- Modify: `README.zh-CN.md`

- [ ] **Step 1: Write a failing existence test for the new example configs**
- [ ] **Step 2: Run test to verify it fails**
  Run: `python -m pytest tests/test_cli_import.py -q`
- [ ] **Step 3: Add the three YAML examples and README entries**
- [ ] **Step 4: Run test to verify it passes**
  Run: `python -m pytest tests/test_cli_import.py -q`
- [ ] **Step 5: Commit**
  Run: `git commit -m "docs: add diffusion teacher configs and usage"`

## Task 7: Add real third-party smoke-test helper coverage

**Files:**
- Modify or create focused smoke-test documentation and helper config files as needed

- [ ] **Step 1: Add a conservative real-run note for 5070 Ti usage**
- [ ] **Step 2: Add one smoke-test config per model if it differs from the main local config**
- [ ] **Step 3: Verify docs and config paths are consistent**
- [ ] **Step 4: Commit**
  Run: `git commit -m "docs: add diffusion teacher smoke test guidance"`

## Task 8: Full verification

**Files:**
- Modify as needed only if verification reveals issues

- [ ] **Step 1: Run focused diffusion-teacher tests**
  Run: `python -m pytest tests/test_diffusion_teacher_setup.py tests/test_diffusion_teacher_runners.py tests/test_executor_cli.py tests/test_generators.py tests/test_pytorch_teacher_adapters.py -q`
- [ ] **Step 2: Run the full test suite**
  Run: `python -m pytest -q`
- [ ] **Step 3: Check branch status**
  Run: `git status --short --branch`
- [ ] **Step 4: Review recent commits**
  Run: `git log --oneline -5`
- [ ] **Step 5: Push the feature branch**
  Run: `git push -u origin codex/diffusion-sr-models`
