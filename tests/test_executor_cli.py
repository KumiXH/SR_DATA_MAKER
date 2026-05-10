import json

from PIL import Image

from sr_data_maker.core.types import RunnerOutput
from sr_data_maker.orchestration.executor import PipelineExecutor
from sr_data_maker.plugins import RUNNERS, register_builtins


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


def test_executor_reuses_runner_for_same_task_across_sources(tmp_path):
    class CountingRunner:
        init_count = 0

        def __init__(self, **_: object) -> None:
            type(self).init_count += 1

        def run(self, inputs, context):
            return RunnerOutput(outputs={"image": inputs["image"]}, meta={})

    register_builtins()
    RUNNERS.register("CountingRunner")(CountingRunner)

    source_root = tmp_path / "raw"
    nested = source_root / "faces"
    nested.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(nested / "img001.png")
    Image.new("RGB", (8, 8), "white").save(nested / "img002.png")
    output_root = tmp_path / "out"
    config = {
        "name": "runner_reuse",
        "runtime": {"device": "cpu", "resume": False, "seed": 1},
        "paths": {"input_root": str(source_root), "output_root": str(output_root)},
        "source": {"type": "ImageFolderSourceReader", "recursive": True, "exts": ["png"]},
        "tasks": [
            {
                "name": "teacher_sr_counting",
                "enabled": True,
                "type": "TeacherSRGenerator",
                "runner": {"type": "CountingRunner"},
                "model": {"name": "Counting_x1", "scale": 1},
                "output": {"folder_name": "Counting_x1"},
            }
        ],
    }

    summary = PipelineExecutor().run(config)

    assert summary["succeeded"] == 2
    assert CountingRunner.init_count == 1
