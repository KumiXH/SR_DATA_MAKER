# Face Teacher Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add GFPGAN, CodeFormer, and VQFR as first-batch face-focused teacher models in the existing `teacher_superres` pipeline.

**Architecture:** Keep `TeacherSRGenerator`, `PipelineExecutor`, and the dataset layout unchanged. Extend the project by adding three new teacher runners, a shared face-teacher setup module, CLI setup entrypoints, example YAML configs, and tests that verify registry/setup/runner contract behavior before any real third-party inference.

**Tech Stack:** Python 3.10+, PyTorch, Pillow, pytest, argparse, pathlib, existing registry/setup helpers, local third-party repos under `third_party/`.

---

## File Structure

- Create `src/sr_data_maker/runners/teacher/face_base.py`: shared helper for face-teacher runners to validate weights, add repo roots to `sys.path`, and return standardized provenance metadata.
- Create `src/sr_data_maker/runners/teacher/gfpgan.py`: GFPGAN runner implementation.
- Create `src/sr_data_maker/runners/teacher/codeformer.py`: CodeFormer runner implementation.
- Create `src/sr_data_maker/runners/teacher/vqfr.py`: VQFR runner implementation.
- Create `src/sr_data_maker/setup/face_teacher.py`: generic setup support for GFPGAN, CodeFormer, and VQFR tasks.
- Modify `src/sr_data_maker/plugins.py`: register the three new runners.
- Modify `src/sr_data_maker/cli/main.py`: add `setup gfpgan`, `setup codeformer`, and `setup vqfr`.
- Modify `README.md`: document setup and run flow for the new face teacher models.
- Create `configs/examples/local_gfpgan_x2.yaml`: example config for GFPGAN teacher mode.
- Create `configs/examples/local_codeformer_x2.yaml`: example config for CodeFormer teacher mode.
- Create `configs/examples/local_vqfr_x2.yaml`: example config for VQFR teacher mode.
- Create `tests/test_face_teacher_setup.py`: setup discovery and repo/weight preparation tests.
- Create `tests/test_face_teacher_runners.py`: missing-weight and parameter-propagation tests.
- Modify `tests/test_executor_cli.py`: CLI dispatch coverage for new setup commands.
- Modify `tests/test_generators.py`: teacher manifest provenance coverage for face-model metadata.

---

### Task 1: Add face teacher setup tests

**Files:**
- Create: `tests/test_face_teacher_setup.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from sr_data_maker.setup.face_teacher import (
    CODEFORMER_REPO_URL,
    FACEXLIB_REPO_URL,
    GFPGAN_REPO_URL,
    VQFR_REPO_URL,
    FaceTeacherSetup,
    find_face_teacher_models,
)


def test_find_face_teacher_models_reads_enabled_matching_tasks():
    config = {
        "tasks": [
            {
                "enabled": True,
                "runner": {"type": "GFPGANRunner"},
                "model": {"name": "GFPGAN_x2", "weights": "./weights/gfpgan.pth"},
            },
            {
                "enabled": True,
                "runner": {"type": "CodeFormerRunner"},
                "model": {"name": "CodeFormer_x2", "weights": "./weights/codeformer.pth"},
            },
            {
                "enabled": False,
                "runner": {"type": "VQFRRunner"},
                "model": {"name": "disabled"},
            },
        ]
    }

    models = find_face_teacher_models(config, {"GFPGANRunner", "CodeFormerRunner", "VQFRRunner"})

    assert [item["runner_type"] for item in models] == ["GFPGANRunner", "CodeFormerRunner"]


def test_gfpgan_setup_clones_expected_repos_and_downloads_weights(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="GFPGANRunner",
        default_repo_url=GFPGAN_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "GFPGAN_x2",
            "weights": "./weights/GFPGAN_x2.pth",
            "repo_root": "./third_party/GFPGAN",
            "facelib_root": "./third_party/facexlib",
            "basicsr_root": "./third_party/BasicSR",
            "download_url": "https://example.test/gfpgan.pth",
        }
    )

    assert result["repos"]["main"]["status"] == "cloned"
    assert result["repos"]["facexlib"]["status"] == "cloned"
    assert result["repos"]["basicsr"]["status"] == "cloned"
    assert result["weights"]["status"] == "downloaded"
    assert calls == [
        ("clone", GFPGAN_REPO_URL, tmp_path / "third_party" / "GFPGAN"),
        ("clone", FACEXLIB_REPO_URL, tmp_path / "third_party" / "facexlib"),
        ("clone", "https://github.com/XPixelGroup/BasicSR.git", tmp_path / "third_party" / "BasicSR"),
        ("download", "https://example.test/gfpgan.pth", tmp_path / "weights" / "GFPGAN_x2.pth"),
    ]


def test_codeformer_setup_requires_download_url_when_weight_missing(tmp_path):
    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="CodeFormerRunner",
        default_repo_url=CODEFORMER_REPO_URL,
        clone=lambda _repo_url, destination: destination.mkdir(parents=True),
        download=lambda _url, _destination: None,
    )

    with pytest.raises(ValueError, match="download_url"):
        setup.prepare_model(
            {
                "name": "CodeFormer_x2",
                "weights": "./weights/codeformer.pth",
                "repo_root": "./third_party/CodeFormer",
                "facelib_root": "./third_party/facexlib",
                "basicsr_root": "./third_party/BasicSR",
            }
        )


def test_vqfr_setup_uses_default_repo_url(tmp_path):
    calls: list[tuple[str, str, Path]] = []

    def clone(repo_url: str, destination: Path) -> None:
        calls.append(("clone", repo_url, destination))
        destination.mkdir(parents=True)

    def download(url: str, destination: Path) -> None:
        calls.append(("download", url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"weights")

    setup = FaceTeacherSetup(
        project_root=tmp_path,
        runner_type="VQFRRunner",
        default_repo_url=VQFR_REPO_URL,
        clone=clone,
        download=download,
    )

    result = setup.prepare_model(
        {
            "name": "VQFR_x2",
            "weights": "./weights/VQFR_x2.pth",
            "repo_root": "./third_party/VQFR",
            "facelib_root": "./third_party/facexlib",
            "basicsr_root": "./third_party/BasicSR",
            "download_url": "https://example.test/vqfr.pth",
        }
    )

    assert result["repos"]["main"]["path"].endswith("third_party\\VQFR")
    assert result["weights"]["path"].endswith("weights\\VQFR_x2.pth")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_face_teacher_setup.py -q`
Expected: FAIL with `ModuleNotFoundError` for `sr_data_maker.setup.face_teacher`.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from sr_data_maker.setup.realesrgan import BASICSR_REPO_URL, clone_git_repo, download_file

GFPGAN_REPO_URL = "https://github.com/TencentARC/GFPGAN.git"
CODEFORMER_REPO_URL = "https://github.com/sczhou/CodeFormer.git"
VQFR_REPO_URL = "https://github.com/TencentARC/VQFR.git"
FACEXLIB_REPO_URL = "https://github.com/xinntao/facexlib.git"

CloneFunc = Callable[[str, Path], None]
DownloadFunc = Callable[[str, Path], None]


def find_face_teacher_models(config: dict[str, Any], runner_types: set[str]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for task in config.get("tasks", []):
        if task.get("enabled", True) is False:
            continue
        runner = task.get("runner") or {}
        runner_type = runner.get("type")
        if runner_type not in runner_types:
            continue
        model = dict(task.get("model") or {})
        if not model:
            continue
        model["runner_type"] = runner_type
        models.append(model)
    return models


class FaceTeacherSetup:
    def __init__(
        self,
        project_root: str | Path,
        runner_type: str,
        default_repo_url: str,
        clone: CloneFunc | None = None,
        download: DownloadFunc | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runner_type = runner_type
        self.default_repo_url = default_repo_url
        self.clone = clone or clone_git_repo
        self.download = download or download_file

    def prepare_model(self, model: dict[str, Any]) -> dict[str, Any]:
        ...
```

Implement `prepare_model()` by reusing the existing `PyTorchTeacherSetup` style: clone the main repo, clone `facexlib` when configured, clone `BasicSR` when configured, and download weights from `model.download_url`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_face_teacher_setup.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add tests/test_face_teacher_setup.py src/sr_data_maker/setup/face_teacher.py`
Expected: files staged for face-teacher setup support

Run: `git commit -m "feat: add face teacher setup support"`
Expected: commit created

### Task 2: Add face runner contract tests

**Files:**
- Create: `tests/test_face_teacher_runners.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from PIL import Image
import pytest

from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.vqfr import VQFRRunner


def test_gfpgan_runner_rejects_missing_weights(tmp_path):
    runner = GFPGANRunner(name="GFPGAN_x2", weights=str(tmp_path / "missing.pth"), scale=2)

    with pytest.raises(FileNotFoundError, match="GFPGAN weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_codeformer_runner_rejects_missing_weights(tmp_path):
    runner = CodeFormerRunner(name="CodeFormer_x2", weights=str(tmp_path / "missing.pth"), scale=2)

    with pytest.raises(FileNotFoundError, match="CodeFormer weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_vqfr_runner_rejects_missing_weights(tmp_path):
    runner = VQFRRunner(name="VQFR_x2", weights=str(tmp_path / "missing.pth"), scale=2)

    with pytest.raises(FileNotFoundError, match="VQFR weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_codeformer_runner_exposes_face_provenance(tmp_path):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")
    captured = {}

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _run_inference(self, image, torch):
            captured["model"] = dict(self.model)
            return image

        def _import_torch(self):
            return object()

    runner = TestableCodeFormerRunner(
        name="CodeFormer_x2",
        weights=str(weights),
        scale=2,
        fidelity_weight=0.7,
        face_upsample=True,
        background_upsampler="realesrgan",
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert output.meta["face_model"] is True
    assert output.meta["face_model_family"] == "CodeFormer"
    assert output.meta["fidelity_weight"] == 0.7
    assert captured["model"]["background_upsampler"] == "realesrgan"


def test_vqfr_runner_exposes_fidelity_ratio(tmp_path):
    weights = tmp_path / "vqfr.pth"
    weights.write_bytes(b"weights")

    class TestableVQFRRunner(VQFRRunner):
        def _run_inference(self, image, torch):
            return image

        def _import_torch(self):
            return object()

    runner = TestableVQFRRunner(
        name="VQFR_x2",
        weights=str(weights),
        scale=2,
        fidelity_ratio=0.85,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.meta["face_model_family"] == "VQFR"
    assert output.meta["fidelity_ratio"] == 0.85
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_face_teacher_runners.py -q`
Expected: FAIL with `ModuleNotFoundError` for the new runner modules.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from sr_data_maker.core.types import RunnerOutput


class FaceTeacherRunnerBase:
    family = "FaceTeacher"
    display_name = "Face Teacher"

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        weights = Path(str(self.model.get("weights", "")))
        if not weights.exists():
            raise FileNotFoundError(f"{self.display_name} weights not found: {weights}")

        torch = self._import_torch()
        output = self._run_inference(image, torch)
        return RunnerOutput(outputs={"image": output}, meta=self._provenance())

    def _provenance(self) -> dict[str, Any]:
        return {
            "face_model": True,
            "face_model_family": self.family,
            "scale": int(self.model.get("scale", 1)),
        }
```

Add one subclass per model and extend `_provenance()` with the model-specific fields from the design. Keep `_run_inference()` as a separate method so tests can override it without requiring real third-party repos yet.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_face_teacher_runners.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add tests/test_face_teacher_runners.py src/sr_data_maker/runners/teacher/face_base.py src/sr_data_maker/runners/teacher/gfpgan.py src/sr_data_maker/runners/teacher/codeformer.py src/sr_data_maker/runners/teacher/vqfr.py`
Expected: new runner files staged

Run: `git commit -m "feat: add face teacher runner contracts"`
Expected: commit created

### Task 3: Register new runners and cover registry behavior

**Files:**
- Modify: `src/sr_data_maker/plugins.py`
- Modify: `tests/test_pytorch_teacher_adapters.py`

- [ ] **Step 1: Write the failing test**

```python
from sr_data_maker.plugins import RUNNERS, register_builtins
from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.vqfr import VQFRRunner


def test_builtin_registry_includes_face_teacher_runners():
    register_builtins()

    gfpgan = RUNNERS.build({"type": "GFPGANRunner", "name": "GFPGAN_x2", "weights": "missing.pth", "scale": 2})
    codeformer = RUNNERS.build({"type": "CodeFormerRunner", "name": "CodeFormer_x2", "weights": "missing.pth", "scale": 2})
    vqfr = RUNNERS.build({"type": "VQFRRunner", "name": "VQFR_x2", "weights": "missing.pth", "scale": 2})

    assert isinstance(gfpgan, GFPGANRunner)
    assert isinstance(codeformer, CodeFormerRunner)
    assert isinstance(vqfr, VQFRRunner)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_pytorch_teacher_adapters.py::test_builtin_registry_includes_face_teacher_runners -q`
Expected: FAIL because the runners are not registered.

- [ ] **Step 3: Write minimal implementation**

```python
from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.vqfr import VQFRRunner

...
        RUNNERS.register("GFPGANRunner")(GFPGANRunner)
        RUNNERS.register("CodeFormerRunner")(CodeFormerRunner)
        RUNNERS.register("VQFRRunner")(VQFRRunner)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_pytorch_teacher_adapters.py::test_builtin_registry_includes_face_teacher_runners -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add src/sr_data_maker/plugins.py tests/test_pytorch_teacher_adapters.py`
Expected: registry update staged

Run: `git commit -m "feat: register face teacher runners"`
Expected: commit created

### Task 4: Add CLI setup command tests

**Files:**
- Modify: `tests/test_executor_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from sr_data_maker.cli.main import main


def test_cli_setup_gfpgan_uses_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
tasks:
  - enabled: true
    runner:
      type: GFPGANRunner
    model:
      name: GFPGAN_x2
      weights: ./weights/GFPGAN_x2.pth
      repo_root: ./third_party/GFPGAN
      download_url: https://example.test/gfpgan.pth
""",
        encoding="utf-8",
    )
    calls: list[tuple[dict, Path]] = []

    def fake_setup(config, project_root):
        calls.append((config, Path(project_root)))
        return [{"model_name": "GFPGAN_x2"}]

    monkeypatch.setattr("sr_data_maker.cli.main.setup_gfpgan_from_config", fake_setup)
    monkeypatch.setattr(
        "sys.argv",
        ["sr-data-maker", "setup", "gfpgan", "--config", str(config_path), "--project-root", str(tmp_path)],
    )

    assert main() == 0
    assert calls[0][1] == tmp_path


def test_cli_setup_codeformer_uses_config(monkeypatch, tmp_path):
    ...


def test_cli_setup_vqfr_uses_config(monkeypatch, tmp_path):
    ...
```

For the `CodeFormer` and `VQFR` tests, follow the same structure with the corresponding runner type, setup function name, and expected `model_name`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_executor_cli.py -q`
Expected: FAIL because the CLI does not yet expose those setup subcommands.

- [ ] **Step 3: Write minimal implementation**

```python
from sr_data_maker.setup.face_teacher import (
    print_face_teacher_setup_summary,
    setup_codeformer_from_config,
    setup_gfpgan_from_config,
    setup_vqfr_from_config,
)

...
    gfpgan = setup_sub.add_parser("gfpgan")
    gfpgan.add_argument("--config", required=True)
    gfpgan.add_argument("--project-root", default=".")
    codeformer = setup_sub.add_parser("codeformer")
    codeformer.add_argument("--config", required=True)
    codeformer.add_argument("--project-root", default=".")
    vqfr = setup_sub.add_parser("vqfr")
    vqfr.add_argument("--config", required=True)
    vqfr.add_argument("--project-root", default=".")
```

Add `_setup_gfpgan()`, `_setup_codeformer()`, and `_setup_vqfr()` mirroring the existing `setup_swinir` and `setup_hat` flow.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_executor_cli.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add tests/test_executor_cli.py src/sr_data_maker/cli/main.py`
Expected: CLI changes staged

Run: `git commit -m "feat: add face teacher setup commands"`
Expected: commit created

### Task 5: Extend teacher manifest coverage for face provenance

**Files:**
- Modify: `tests/test_generators.py`

- [ ] **Step 1: Write the failing test**

```python
from PIL import Image

from sr_data_maker.core.types import RunnerOutput, SourceRecord
from sr_data_maker.generators.teacher_sr import TeacherSRGenerator


class FaceRunner:
    name = "CodeFormerRunner"

    def run(self, inputs, context):
        return RunnerOutput(
            outputs={"image": Image.new("RGB", (2, 2), "black")},
            meta={
                "face_model": True,
                "face_model_family": "CodeFormer",
                "fidelity_weight": 0.7,
            },
        )


def test_teacher_generator_preserves_face_runner_provenance(tmp_path):
    source = SourceRecord(source_id="a/b.png", path=tmp_path / "a" / "b.png", rel_path="a/b.png", meta={})
    generator = TeacherSRGenerator(
        name="teacher_face_codeformer",
        runner=FaceRunner(),
        model={"name": "CodeFormer_x2"},
        output={"folder_name": "CodeFormer_x2"},
    )

    sample = generator.generate(source, context=None)[0]

    assert sample.manifest["provenance"]["face_model"] is True
    assert sample.manifest["provenance"]["face_model_family"] == "CodeFormer"
    assert sample.manifest["provenance"]["fidelity_weight"] == 0.7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_generators.py::test_teacher_generator_preserves_face_runner_provenance -q`
Expected: FAIL if provenance is not preserved as expected.

- [ ] **Step 3: Write minimal implementation**

No production code should be needed if `TeacherSRGenerator` continues to merge `result.meta` into `provenance`. If the test fails, fix only the minimal manifest merge path inside `src/sr_data_maker/generators/teacher_sr.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_generators.py::test_teacher_generator_preserves_face_runner_provenance -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add tests/test_generators.py src/sr_data_maker/generators/teacher_sr.py`
Expected: manifest coverage update staged

Run: `git commit -m "test: cover face teacher provenance in manifest"`
Expected: commit created

### Task 6: Add example configs and README coverage

**Files:**
- Create: `configs/examples/local_gfpgan_x2.yaml`
- Create: `configs/examples/local_codeformer_x2.yaml`
- Create: `configs/examples/local_vqfr_x2.yaml`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_face_teacher_example_configs_exist():
    assert Path("configs/examples/local_gfpgan_x2.yaml").exists()
    assert Path("configs/examples/local_codeformer_x2.yaml").exists()
    assert Path("configs/examples/local_vqfr_x2.yaml").exists()
```

Add this test to `tests/test_cli_import.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_import.py -q`
Expected: FAIL because the new example configs do not exist.

- [ ] **Step 3: Write minimal implementation**

Create the three configs following the current style used by `configs/examples/local_realesrgan_x2.yaml`, but with the new runner names and required repo/weights fields. Update `README.md` with:

- setup commands for `gfpgan`, `codeformer`, and `vqfr`
- one short YAML example per model family
- one short note that these models are integrated as `teacher` outputs in this phase

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

Run: `git add configs/examples/local_gfpgan_x2.yaml configs/examples/local_codeformer_x2.yaml configs/examples/local_vqfr_x2.yaml README.md tests/test_cli_import.py`
Expected: docs and config changes staged

Run: `git commit -m "docs: add face teacher configs and usage"`
Expected: commit created

### Task 7: Full verification

**Files:**
- Modify as needed only if verification reveals issues

- [ ] **Step 1: Run focused face-teacher tests**

Run: `python -m pytest tests/test_face_teacher_setup.py tests/test_face_teacher_runners.py tests/test_executor_cli.py tests/test_generators.py -q`
Expected: PASS

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Check branch status**

Run: `git status --short --branch`
Expected: clean working tree on `feature/face-models`

Run: `git log --oneline -5`
Expected: recent commits for setup support, runner contracts, registry, CLI, and docs

- [ ] **Step 4: Push feature branch**

Run: `git push -u origin feature/face-models`
Expected: branch pushed successfully
