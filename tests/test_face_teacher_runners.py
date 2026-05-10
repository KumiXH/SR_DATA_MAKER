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


def test_gfpgan_runner_uses_restorer_and_returns_restored_image(tmp_path):
    weights = tmp_path / "gfpgan.pth"
    weights.write_bytes(b"weights")

    class FakeRestorer:
        def __init__(self):
            self.calls = []

        def enhance(self, img, has_aligned, only_center_face, paste_back, weight):
            self.calls.append((img.shape, has_aligned, only_center_face, paste_back, weight))
            return [], [], img

    class TestableGFPGANRunner(GFPGANRunner):
        def _import_torch(self):
            return object()

        def _build_restorer(self, torch):
            self.restorer = FakeRestorer()
            return self.restorer

    runner = TestableGFPGANRunner(
        name="GFPGAN_x2",
        weights=str(weights),
        scale=2,
        only_center_face=True,
        weight=0.6,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert runner.restorer.calls[0][1:] == (False, True, True, 0.6)


def test_codeformer_runner_uses_restoration_pipeline_and_returns_image(tmp_path):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _import_torch(self):
            return object()

        def _restore_image(self, image, torch):
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    runner = TestableCodeFormerRunner(
        name="CodeFormer_x2",
        weights=str(weights),
        scale=2,
        fidelity_weight=0.7,
    )

    source = Image.new("RGB", (8, 8), "white")
    output = runner.run({"image": source}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert output.outputs["image"] is not source


def test_vqfr_runner_uses_demo_restorer_and_returns_restored_image(tmp_path):
    weights = tmp_path / "vqfr.pth"
    weights.write_bytes(b"weights")

    class FakeRestorer:
        def __init__(self):
            self.calls = []

        def enhance(self, img, fidelity_ratio, has_aligned, only_center_face, paste_back):
            self.calls.append((img.shape, fidelity_ratio, has_aligned, only_center_face, paste_back))
            return [], [], img

    class TestableVQFRRunner(VQFRRunner):
        def _import_torch(self):
            return object()

        def _build_restorer(self, torch):
            self.restorer = FakeRestorer()
            return self.restorer

    runner = TestableVQFRRunner(
        name="VQFR_x2",
        weights=str(weights),
        scale=2,
        fidelity_ratio=0.85,
        only_center_face=False,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert runner.restorer.calls[0][1:] == (0.85, False, False, True)


def test_codeformer_runner_restore_image_uses_run_inference_hook(tmp_path):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")
    called = {}

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _import_torch(self):
            return object()

        def _run_inference(self, image, torch):
            called["used"] = True
            return image.copy()

    runner = TestableCodeFormerRunner(name="CodeFormer_x2", weights=str(weights), scale=2)
    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert called["used"] is True
