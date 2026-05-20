from PIL import Image

from sr_data_maker.core.types import RunnerOutput, SourceRecord
from sr_data_maker.generators.degradation import DegradationGenerator
from sr_data_maker.generators.teacher_sr import TeacherSRGenerator


class FixedRunner:
    name = "FixedRunner"

    def run(self, inputs, context):
        return RunnerOutput(outputs={"image": Image.new("RGB", (2, 2), "black")}, meta={"params": {"scale": 2}})


def test_degradation_generator_uses_mirrored_degraded_path(tmp_path):
    source = SourceRecord(
        source_id="a/b.png",
        path=tmp_path / "a" / "b.png",
        rel_path="a/b.png",
        meta={"image": Image.new("RGB", (4, 4), "white")},
    )
    generator = DegradationGenerator(name="degradation_x2", runner=FixedRunner(), output={"folder_name": "degradation_x2"})

    sample = generator.generate(source, context=None)[0]

    assert sample.image.size == (2, 2)
    assert sample.output_path == "degraded/degradation_x2/a/b.png"
    assert sample.manifest["target"]["target_type"] == "real_gt"


def test_teacher_generator_uses_model_folder_name(tmp_path):
    source = SourceRecord(
        source_id="a/b.png",
        path=tmp_path / "a" / "b.png",
        rel_path="a/b.png",
        meta={"image": Image.new("RGB", (4, 4), "white")},
    )
    generator = TeacherSRGenerator(
        name="teacher_sr_realesrgan",
        runner=FixedRunner(),
        model={"name": "RealESRGAN_x2plus"},
        output={"folder_name": "RealESRGAN_x2plus"},
    )

    sample = generator.generate(source, context=None)[0]

    assert sample.output_path == "teacher/RealESRGAN_x2plus/a/b.png"
    assert sample.manifest["target"]["target_type"] == "pseudo_gt"


def test_teacher_generator_preserves_face_runner_provenance(tmp_path):
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

    source_path = tmp_path / "a" / "b.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "white").save(source_path)
    source = SourceRecord(source_id="a/b.png", path=source_path, rel_path="a/b.png", meta={})
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


def test_teacher_generator_preserves_diffusion_runner_provenance(tmp_path):
    class DiffusionRunner:
        name = "ResShiftRunner"

        def run(self, inputs, context):
            return RunnerOutput(
                outputs={"image": Image.new("RGB", (2, 2), "black")},
                meta={
                    "diffusion_model": True,
                    "diffusion_model_family": "ResShift",
                    "steps": 15,
                    "precision": "fp16",
                },
            )

    source_path = tmp_path / "a" / "b.png"
    source_path.parent.mkdir(parents=True)
    Image.new("RGB", (2, 2), "white").save(source_path)
    source = SourceRecord(source_id="a/b.png", path=source_path, rel_path="a/b.png", meta={})
    generator = TeacherSRGenerator(
        name="teacher_sr_resshift",
        runner=DiffusionRunner(),
        model={"name": "ResShift_x4"},
        output={"folder_name": "ResShift_x4"},
    )

    sample = generator.generate(source, context=None)[0]

    assert sample.manifest["provenance"]["diffusion_model"] is True
    assert sample.manifest["provenance"]["diffusion_model_family"] == "ResShift"
    assert sample.manifest["provenance"]["steps"] == 15
    assert sample.manifest["provenance"]["precision"] == "fp16"
