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
