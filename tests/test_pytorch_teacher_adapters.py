from __future__ import annotations

import sys

from PIL import Image
import pytest

from sr_data_maker.plugins import RUNNERS, register_builtins
from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.hat import HATAdapter
from sr_data_maker.runners.teacher.swinir import SwinIRAdapter
from sr_data_maker.runners.teacher.vqfr import VQFRRunner


def test_swinir_adapter_rejects_missing_weights(tmp_path):
    runner = SwinIRAdapter(
        name="SwinIR_x2_classical",
        weights=str(tmp_path / "missing.pth"),
        scale=2,
    )

    with pytest.raises(FileNotFoundError, match="SwinIR weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_hat_adapter_rejects_missing_weights(tmp_path):
    runner = HATAdapter(
        name="HAT_SRx2",
        weights=str(tmp_path / "missing.pth"),
        scale=2,
    )

    with pytest.raises(FileNotFoundError, match="HAT weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_swinir_adapter_builds_model_from_yaml_params(tmp_path):
    weights = tmp_path / "swinir.pth"
    weights.write_bytes(b"weights")
    captured = {}

    class DummyModel:
        pass

    class TestableSwinIRAdapter(SwinIRAdapter):
        def _load_weights(self, model, weights_path, torch):
            captured["weights_path"] = weights_path

        def _run_model(self, image, model, torch):
            return image

        def _import_torch(self):
            return object()

        def _build_model(self):
            captured["model_config"] = dict(self.model)
            return DummyModel()

    runner = TestableSwinIRAdapter(
        name="SwinIR_x2_classical",
        weights=str(weights),
        scale=2,
        tile=128,
        tile_pad=8,
        half=True,
        depths=[6, 6, 6, 6, 6, 6],
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert output.meta["model"]["name"] == "SwinIR_x2_classical"
    assert captured["weights_path"] == weights
    assert captured["model_config"]["tile"] == 128
    assert captured["model_config"]["half"] is True
    assert captured["model_config"]["depths"] == [6, 6, 6, 6, 6, 6]


def test_hat_adapter_builds_model_from_yaml_params(tmp_path):
    weights = tmp_path / "hat.pth"
    weights.write_bytes(b"weights")
    captured = {}

    class DummyModel:
        pass

    class TestableHATAdapter(HATAdapter):
        def _load_weights(self, model, weights_path, torch):
            captured["weights_path"] = weights_path

        def _run_model(self, image, model, torch):
            return image

        def _import_torch(self):
            return object()

        def _build_model(self):
            captured["model_config"] = dict(self.model)
            return DummyModel()

    runner = TestableHATAdapter(
        name="HAT_SRx2",
        weights=str(weights),
        scale=2,
        tile=128,
        tile_pad=8,
        half=False,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert output.meta["model"]["name"] == "HAT_SRx2"
    assert captured["weights_path"] == weights
    assert captured["model_config"]["tile"] == 128
    assert captured["model_config"]["embed_dim"] == 180


def test_pytorch_adapter_caches_loaded_model_across_multiple_runs(tmp_path):
    weights = tmp_path / "cached.pth"
    weights.write_bytes(b"weights")
    captured = {"builds": 0, "loads": 0}

    class DummyModel:
        pass

    class TestableSwinIRAdapter(SwinIRAdapter):
        def _import_torch(self):
            return object()

        def _build_model(self):
            captured["builds"] += 1
            return DummyModel()

        def _load_weights(self, model, weights_path, torch):
            captured["loads"] += 1

        def _run_model(self, image, model, torch):
            return image

    runner = TestableSwinIRAdapter(
        name="SwinIR_x2_classical",
        weights=str(weights),
        scale=2,
    )

    image = Image.new("RGB", (8, 8), "white")
    runner.run({"image": image}, context=None)
    runner.run({"image": image}, context=None)

    assert captured["builds"] == 1
    assert captured["loads"] == 1


def test_builtin_registry_includes_swinir_and_hat_adapters():
    register_builtins()

    swinir = RUNNERS.build({"type": "SwinIRAdapter", "name": "SwinIR_x2_classical", "weights": "missing.pth", "scale": 2})
    hat = RUNNERS.build({"type": "HATAdapter", "name": "HAT_SRx2", "weights": "missing.pth", "scale": 2})

    assert isinstance(swinir, SwinIRAdapter)
    assert isinstance(hat, HATAdapter)


def test_builtin_registry_includes_face_teacher_runners():
    register_builtins()

    gfpgan = RUNNERS.build({"type": "GFPGANRunner", "name": "GFPGAN_x2", "weights": "missing.pth", "scale": 2})
    codeformer = RUNNERS.build({"type": "CodeFormerRunner", "name": "CodeFormer_x2", "weights": "missing.pth", "scale": 2})
    vqfr = RUNNERS.build({"type": "VQFRRunner", "name": "VQFR_x2", "weights": "missing.pth", "scale": 2})

    assert isinstance(gfpgan, GFPGANRunner)
    assert isinstance(codeformer, CodeFormerRunner)
    assert isinstance(vqfr, VQFRRunner)


def test_adapter_adds_repo_and_dependency_roots_to_python_path(monkeypatch, tmp_path):
    repo_root = tmp_path / "HAT"
    basicsr_root = tmp_path / "BasicSR"
    extra_root = tmp_path / "ExtraDependency"
    repo_root.mkdir()
    basicsr_root.mkdir()
    extra_root.mkdir()

    runner = HATAdapter(
        name="HAT_SRx2",
        weights=str(tmp_path / "missing.pth"),
        repo_root=str(repo_root),
        basicsr_root=str(basicsr_root),
        extra_repo_roots=[str(extra_root)],
        scale=2,
    )
    monkeypatch.setattr(sys, "path", [])

    runner._add_repo_to_path()

    assert sys.path == [str(extra_root), str(basicsr_root), str(repo_root)]


def test_tile_buffer_allocation_uses_model_output_channels():
    class FakeTensor:
        def __init__(self):
            self.shape = None

        def new_zeros(self, shape):
            result = FakeTensor()
            result.shape = shape
            return result

    class FakeOutput:
        def size(self, dim=None):
            shape = (1, 1, 8, 8)
            if dim is None:
                return shape
            return shape[dim]

    adapter = SwinIRAdapter(name="SwinIR_x2_classical", weights="unused.pth", scale=2, tile=4, tile_pad=0)
    output, weight = adapter._allocate_tile_buffers(FakeTensor(), FakeOutput(), scale=2, height=4, width=4)

    assert output.shape == (1, 1, 8, 8)
    assert weight.shape == (1, 1, 8, 8)
