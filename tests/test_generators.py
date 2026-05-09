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
