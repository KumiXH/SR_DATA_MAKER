# SR Data Maker MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working SR Data Maker MVP with YAML-driven degradation and Real-ESRGAN 2x teacher task support.

**Architecture:** The project uses a `src/sr_data_maker` package with small modules for config loading, registries, source scanning, output naming, manifest writing, runners, generators, orchestration, and CLI commands. The MVP keeps real model execution optional through a Real-ESRGAN runner that imports third-party packages only when the task is executed.

**Tech Stack:** Python 3.10+, PyYAML, Pillow, pytest, argparse, pathlib, dataclasses.

---

## File Structure

- Create `pyproject.toml`: package metadata, runtime dependencies, pytest config, console script.
- Create `configs/examples/sr_mixed_v1.yaml`: example config with conventional degradation enabled and Real-ESRGAN 2x teacher configurable.
- Create `src/sr_data_maker/config/loader.py`: YAML loading, simple base merge, path resolution.
- Create `src/sr_data_maker/config/validator.py`: startup config validation.
- Create `src/sr_data_maker/core/types.py`: dataclasses for source records, context, runner outputs, task results.
- Create `src/sr_data_maker/core/registry.py`: component registries and registration helpers.
- Create `src/sr_data_maker/sources/image_folder.py`: nested image scanning with stable relative paths.
- Create `src/sr_data_maker/dataset/naming.py`: mirrored output path generation.
- Create `src/sr_data_maker/dataset/manifest.py`: JSONL append/read helpers and summary writing.
- Create `src/sr_data_maker/dataset/writer.py`: image writing, manifest writing, failure writing, summary writing.
- Create `src/sr_data_maker/runners/degradation/classical.py`: conventional blur, resize, noise, JPEG degradation.
- Create `src/sr_data_maker/runners/teacher/realesrgan.py`: Real-ESRGAN 2x runner with optional dependency import.
- Create `src/sr_data_maker/generators/degradation.py`: degradation sample semantics and provenance.
- Create `src/sr_data_maker/generators/teacher_sr.py`: teacher sample semantics and provenance.
- Create `src/sr_data_maker/orchestration/state_store.py`: JSONL state tracking for resume.
- Create `src/sr_data_maker/orchestration/executor.py`: end-to-end pipeline execution.
- Create `src/sr_data_maker/cli/main.py`: `run`, `validate`, and `inspect` CLI commands.
- Create `src/sr_data_maker/plugins.py`: built-in component registration.
- Create tests under `tests/` for each behavior before implementation.

---

### Task 1: Packaging And Imports

**Files:**
- Create: `pyproject.toml`
- Create: `src/sr_data_maker/__init__.py`
- Test: `tests/test_package_import.py`

- [ ] **Step 1: Write the failing test**

```python
def test_package_imports_version():
    import sr_data_maker

    assert isinstance(sr_data_maker.__version__, str)
    assert sr_data_maker.__version__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_package_import.py -q`
Expected: FAIL with `ModuleNotFoundError` or missing `__version__`.

- [ ] **Step 3: Write minimal implementation**

Create `pyproject.toml` with package metadata and dependencies. Create `src/sr_data_maker/__init__.py` with `__version__ = "0.1.0"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_package_import.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add pyproject.toml src/sr_data_maker/__init__.py tests/test_package_import.py && git commit -m "chore: add package skeleton"`

### Task 2: Config Loading And Validation

**Files:**
- Create: `src/sr_data_maker/config/loader.py`
- Create: `src/sr_data_maker/config/validator.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest

from sr_data_maker.config.loader import load_config
from sr_data_maker.config.validator import ConfigError, validate_config


def test_load_config_merges_base_and_child(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text("runtime:\n  device: cpu\n  num_workers: 1\n", encoding="utf-8")
    child.write_text("base:\n  - base.yaml\nruntime:\n  num_workers: 3\nname: demo\n", encoding="utf-8")

    config = load_config(child)

    assert config["name"] == "demo"
    assert config["runtime"]["device"] == "cpu"
    assert config["runtime"]["num_workers"] == 3


def test_validate_config_rejects_missing_input_root(tmp_path):
    config = {
        "name": "demo",
        "runtime": {"device": "cpu"},
        "paths": {"input_root": str(tmp_path / "missing"), "output_root": str(tmp_path / "out")},
        "source": {"type": "ImageFolderSourceReader"},
        "tasks": [],
    }

    with pytest.raises(ConfigError, match="input_root"):
        validate_config(config)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_config.py -q`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement config loading and validation**

Implement `load_config(path)` with YAML parsing and simple recursive dict merge. Implement `validate_config(config)` checking required keys, existing input root, output root writability, and enabled task basics.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/config tests/test_config.py && git commit -m "feat: add yaml config loading and validation"`

### Task 3: Registry, Types, And Source Reader

**Files:**
- Create: `src/sr_data_maker/core/types.py`
- Create: `src/sr_data_maker/core/registry.py`
- Create: `src/sr_data_maker/sources/image_folder.py`
- Test: `tests/test_source_reader.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from PIL import Image

from sr_data_maker.core.registry import Registry
from sr_data_maker.sources.image_folder import ImageFolderSourceReader


def test_registry_builds_registered_class():
    registry = Registry("demo")

    @registry.register("Thing")
    class Thing:
        def __init__(self, value):
            self.value = value

    instance = registry.build({"type": "Thing", "value": 7})

    assert instance.value == 7


def test_image_folder_reader_preserves_nested_relative_paths(tmp_path):
    root = tmp_path / "raw"
    nested = root / "city" / "day"
    nested.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(nested / "img001.png")
    (nested / "ignore.txt").write_text("skip", encoding="utf-8")

    reader = ImageFolderSourceReader(root=root, recursive=True, exts=["png"])
    records = list(reader.iter_sources())

    assert len(records) == 1
    assert records[0].rel_path == "city/day/img001.png"
    assert records[0].path == nested / "img001.png"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_source_reader.py -q`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement minimal types, registry, and source reader**

Use dataclasses for `SourceRecord`, `RunnerOutput`, and `GeneratedSample`. `ImageFolderSourceReader` should use `Path.rglob` when recursive and normalize relative paths with `/`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_source_reader.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/core src/sr_data_maker/sources tests/test_source_reader.py && git commit -m "feat: add registry and image folder source reader"`

### Task 4: Dataset Writer And Manifest

**Files:**
- Create: `src/sr_data_maker/dataset/naming.py`
- Create: `src/sr_data_maker/dataset/manifest.py`
- Create: `src/sr_data_maker/dataset/writer.py`
- Test: `tests/test_dataset_writer.py`

- [ ] **Step 1: Write failing tests**

```python
import json

from PIL import Image

from sr_data_maker.dataset.naming import output_path_for
from sr_data_maker.dataset.writer import DatasetWriter


def test_output_path_mirrors_source_relative_path(tmp_path):
    path = output_path_for(tmp_path, "degraded", "degradation_x2", "city/day/img001.png")

    assert path == tmp_path / "degraded" / "degradation_x2" / "city" / "day" / "img001.png"


def test_dataset_writer_writes_image_and_manifest(tmp_path):
    writer = DatasetWriter(output_root=tmp_path)
    image = Image.new("RGB", (3, 3), "blue")
    record = {"sample_id": "sample-1", "outputs": [{"path": "degraded/degradation_x2/a.png"}]}

    writer.write_image("degraded/degradation_x2/a.png", image)
    writer.append_sample(record)

    assert (tmp_path / "degraded" / "degradation_x2" / "a.png").exists()
    lines = (tmp_path / "manifests" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["sample_id"] == "sample-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset_writer.py -q`
Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement naming, manifest append, and writer**

`DatasetWriter` should create parent directories, save Pillow images, append JSON lines, append failures, and write summary JSON.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_dataset_writer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/dataset tests/test_dataset_writer.py && git commit -m "feat: add mirrored dataset writer"`

### Task 5: Classical Degradation Runner

**Files:**
- Create: `src/sr_data_maker/runners/degradation/classical.py`
- Test: `tests/test_classical_degradation.py`

- [ ] **Step 1: Write failing tests**

```python
from PIL import Image

from sr_data_maker.runners.degradation.classical import ClassicalDegradationRunner


def test_classical_degradation_resizes_by_scale_and_records_params():
    runner = ClassicalDegradationRunner(
        scale=2,
        blur={"enabled": False},
        resize={"enabled": True, "mode": "bicubic"},
        noise={"enabled": False},
        jpeg={"enabled": False},
        seed=123,
    )
    image = Image.new("RGB", (8, 6), "white")

    output = runner.run({"image": image}, context=None)

    assert output.outputs["image"].size == (4, 3)
    assert output.meta["params"]["scale"] == 2
    assert output.meta["params"]["resize"]["enabled"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classical_degradation.py -q`
Expected: FAIL because runner does not exist.

- [ ] **Step 3: Implement minimal conventional degradation**

Implement resize, optional Gaussian blur, optional Gaussian noise, and optional JPEG roundtrip with deterministic RNG from seed.

- [ ] **Step 4: Run test**

Run: `python -m pytest tests/test_classical_degradation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/runners tests/test_classical_degradation.py && git commit -m "feat: add classical degradation runner"`

### Task 6: Generators

**Files:**
- Create: `src/sr_data_maker/generators/degradation.py`
- Create: `src/sr_data_maker/generators/teacher_sr.py`
- Test: `tests/test_generators.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from PIL import Image

from sr_data_maker.core.types import RunnerOutput, SourceRecord
from sr_data_maker.generators.degradation import DegradationGenerator
from sr_data_maker.generators.teacher_sr import TeacherSRGenerator


class FixedRunner:
    name = "FixedRunner"

    def run(self, inputs, context):
        return RunnerOutput(outputs={"image": Image.new("RGB", (2, 2), "black")}, meta={"params": {"scale": 2}})


def test_degradation_generator_uses_mirrored_degraded_path(tmp_path):
    source = SourceRecord(source_id="a/b.png", path=tmp_path / "a" / "b.png", rel_path="a/b.png", meta={})
    generator = DegradationGenerator(name="degradation_x2", runner=FixedRunner(), output={"folder_name": "degradation_x2"})

    sample = generator.generate(source, context=None)[0]

    assert sample.image.size == (2, 2)
    assert sample.output_path == "degraded/degradation_x2/a/b.png"
    assert sample.manifest["target"]["target_type"] == "real_gt"


def test_teacher_generator_uses_model_folder_name(tmp_path):
    source = SourceRecord(source_id="a/b.png", path=tmp_path / "a" / "b.png", rel_path="a/b.png", meta={})
    generator = TeacherSRGenerator(
        name="teacher_sr_realesrgan",
        runner=FixedRunner(),
        model={"name": "RealESRGAN_x2plus"},
        output={"folder_name": "RealESRGAN_x2plus"},
    )

    sample = generator.generate(source, context=None)[0]

    assert sample.output_path == "teacher/RealESRGAN_x2plus/a/b.png"
    assert sample.manifest["target"]["target_type"] == "pseudo_gt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generators.py -q`
Expected: FAIL because generators do not exist.

- [ ] **Step 3: Implement generators**

Generators should call the runner, build mirrored output paths, and return `GeneratedSample` objects with image, output path, and manifest.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_generators.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/generators tests/test_generators.py && git commit -m "feat: add degradation and teacher generators"`

### Task 7: Executor, State, CLI, And Example Config

**Files:**
- Create: `src/sr_data_maker/orchestration/state_store.py`
- Create: `src/sr_data_maker/orchestration/executor.py`
- Create: `src/sr_data_maker/plugins.py`
- Create: `src/sr_data_maker/cli/main.py`
- Create: `configs/examples/sr_mixed_v1.yaml`
- Test: `tests/test_executor_cli.py`

- [ ] **Step 1: Write failing integration test**

```python
import json

from PIL import Image

from sr_data_maker.orchestration.executor import PipelineExecutor


def test_executor_runs_degradation_and_skips_disabled_teacher(tmp_path):
    source_root = tmp_path / "raw"
    nested = source_root / "city" / "day"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(nested / "img001.png")
    output_root = tmp_path / "out"
    config = {
        "name": "demo",
        "runtime": {"device": "cpu", "resume": True, "seed": 1},
        "paths": {"input_root": str(source_root), "output_root": str(output_root)},
        "source": {"type": "ImageFolderSourceReader", "recursive": True, "exts": ["png"]},
        "tasks": [
            {
                "name": "degradation_x2",
                "enabled": True,
                "type": "DegradationGenerator",
                "runner": {"type": "ClassicalDegradationRunner"},
                "degradation": {
                    "scale": 2,
                    "blur": {"enabled": False},
                    "resize": {"enabled": True, "mode": "bicubic"},
                    "noise": {"enabled": False},
                    "jpeg": {"enabled": False},
                },
                "output": {"folder_name": "degradation_x2"},
            },
            {
                "name": "teacher_sr_realesrgan",
                "enabled": False,
                "type": "TeacherSRGenerator",
                "runner": {"type": "RealESRGANRunner"},
                "model": {"name": "RealESRGAN_x2plus", "scale": 2},
                "output": {"folder_name": "RealESRGAN_x2plus"},
            },
        ],
    }

    summary = PipelineExecutor().run(config)

    assert summary["succeeded"] == 1
    assert summary["skipped_tasks"] == 1
    assert (output_root / "degraded" / "degradation_x2" / "city" / "day" / "img001.png").exists()
    samples = (output_root / "manifests" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(samples[0])["source"]["rel_path"] == "city/day/img001.png"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_executor_cli.py -q`
Expected: FAIL because executor does not exist.

- [ ] **Step 3: Implement state, built-in registration, executor, CLI, and example config**

Executor should register built-ins, skip disabled tasks, run enabled tasks, write outputs, append state, handle failures, and write summary.

- [ ] **Step 4: Run integration test**

Run: `python -m pytest tests/test_executor_cli.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/orchestration src/sr_data_maker/plugins.py src/sr_data_maker/cli configs tests/test_executor_cli.py && git commit -m "feat: add pipeline executor and cli"`

### Task 8: Full Verification And Push

**Files:**
- Modify as needed only when verification reveals an issue.

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Run CLI validation against example config**

Run: `python -m sr_data_maker.cli.main validate --config configs/examples/sr_mixed_v1.yaml`
Expected: exits 0 when example input path exists, or exits with a clear input path error when it does not.

- [ ] **Step 3: Review git diff**

Run: `git status --short --branch` and `git log --oneline -5`
Expected: feature branch contains committed MVP work.

- [ ] **Step 4: Push branch**

Run: `git push -u origin feature/mvp-implementation`
Expected: branch pushed to GitHub.

