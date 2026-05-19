from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from sr_data_maker.runners.teacher.codeformer import CodeFormerRunner
from sr_data_maker.runners.teacher.gfpgan import GFPGANRunner
from sr_data_maker.runners.teacher.vqfr import VQFRRunner
from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


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


def test_codeformer_runner_can_return_upscaled_restored_image(tmp_path):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _import_torch(self):
            return object()

        def _run_inference(self, image, torch):
            return image.resize((16, 16))

    runner = TestableCodeFormerRunner(
        name="CodeFormer_x2",
        weights=str(weights),
        scale=2,
        face_upsample=True,
        paste_back=True,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (16, 16)


def test_codeformer_runner_raises_when_single_face_restore_fails(tmp_path, monkeypatch):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")

    class FakeTorch:
        class cuda:
            @staticmethod
            def empty_cache():
                return None

        @staticmethod
        def no_grad():
            class _Ctx:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Ctx()

    class FakeTensor:
        def unsqueeze(self, dim):
            return self

        def to(self, device):
            return self

    def fake_img2tensor(*args, **kwargs):
        return FakeTensor()

    def fake_tensor2img(*args, **kwargs):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setitem(__import__("sys").modules, "basicsr.utils", type("M", (), {"img2tensor": fake_img2tensor, "tensor2img": fake_tensor2img})())
    monkeypatch.setitem(
        __import__("sys").modules,
        "torchvision.transforms.functional",
        type("M", (), {"normalize": lambda *args, **kwargs: None})(),
    )

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _build_network(self, torch):
            class FailingNet:
                def __call__(self, *args, **kwargs):
                    raise ValueError("boom")

            return FailingNet()

    runner = TestableCodeFormerRunner(name="CodeFormer_x2", weights=str(weights), scale=2, device="cpu")

    with pytest.raises(RuntimeError, match="CodeFormer failed to restore face 1/1: boom"):
        runner._restore_faces([np.zeros((8, 8, 3), dtype=np.uint8)], torch=FakeTorch())


def test_codeformer_runner_provenance_exposes_paste_back_and_alignment(tmp_path):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _import_torch(self):
            return object()

        def _run_inference(self, image, torch):
            return image

    runner = TestableCodeFormerRunner(
        name="CodeFormer_x2",
        weights=str(weights),
        scale=2,
        has_aligned=False,
        paste_back=True,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.meta["has_aligned"] is False
    assert output.meta["paste_back"] is True


def test_face_teacher_runner_add_repo_to_path_preserves_declared_priority(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    facelib_root = tmp_path / "facelib"
    basicsr_root = tmp_path / "basicsr"
    for path in (repo_root, facelib_root, basicsr_root):
        path.mkdir()

    class DummyRunner(FaceTeacherRunnerBase):
        display_name = "Dummy"

        def _run_inference(self, image, torch):
            return image

    runner = DummyRunner(
        weights=str(tmp_path / "dummy.pth"),
        repo_root=str(repo_root),
        facelib_root=str(facelib_root),
        basicsr_root=str(basicsr_root),
    )

    import sys

    original = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original))
    runner._add_repo_to_path()

    assert sys.path[0:3] == [str(repo_root), str(facelib_root), str(basicsr_root)]


def test_codeformer_runner_passes_device_to_face_helper_detection_model(tmp_path, monkeypatch):
    weights = tmp_path / "codeformer.pth"
    weights.write_bytes(b"weights")

    captured = {}

    class FakeFaceRestoreHelper:
        def __init__(self, *args, **kwargs):
            captured["device"] = kwargs.get("device")
            self.cropped_faces = []
            self.det_faces = []

        def clean_all(self):
            return None

        def read_image(self, image):
            return None

        def get_face_landmarks_5(self, **kwargs):
            return 0

        def align_warp_face(self):
            return None

    import types
    import sys

    module = types.ModuleType("facelib.utils.face_restoration_helper")
    module.FaceRestoreHelper = FakeFaceRestoreHelper
    monkeypatch.setitem(sys.modules, "facelib.utils.face_restoration_helper", module)

    class TestableCodeFormerRunner(CodeFormerRunner):
        def _import_torch(self):
            return object()

        def _restore_faces(self, cropped_faces, torch):
            return []

    runner = TestableCodeFormerRunner(
        name="CodeFormer_x2",
        weights=str(weights),
        repo_root=str(tmp_path),
        scale=2,
        device="cpu",
        has_aligned=False,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert captured["device"] == "cpu"
