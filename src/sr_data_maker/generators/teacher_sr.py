from __future__ import annotations

from typing import Any

from sr_data_maker.core.types import GeneratedSample, SourceRecord
from sr_data_maker.generators.degradation import load_source_image


class TeacherSRGenerator:
    def __init__(self, name: str, runner: Any, model: dict[str, Any] | None = None, output: dict[str, Any] | None = None, **_: Any) -> None:
        self.name = name
        self.runner = runner
        self.model = model or {}
        self.output = output or {}

    def generate(self, source: SourceRecord, context: Any) -> list[GeneratedSample]:
        result = self.runner.run({"image": load_source_image(source)}, context)
        folder = self.output.get("folder_name", self.model.get("name", self.name))
        output_path = f"teacher/{folder}/{source.rel_path}"
        manifest = {
            "sample_id": f"{folder}::{source.rel_path}",
            "task_type": "superres",
            "generation_mode": "teacher_superres",
            "source": {"root": "", "rel_path": source.rel_path, "role": "lq"},
            "input": {"source_ref": source.rel_path, "role": "lq", "storage": "source_reference"},
            "target": {"path": output_path, "role": "target", "target_type": "pseudo_gt"},
            "outputs": [{"name": "teacher", "path": output_path, "type": "image", "model_name": folder}],
            "provenance": {
                "generator": "TeacherSRGenerator",
                "runner": getattr(self.runner, "name", type(self.runner).__name__),
                "task_name": self.name,
                "model_name": folder,
                **result.meta,
            },
        }
        return [GeneratedSample(image=result.outputs["image"], output_path=output_path, manifest=manifest)]
