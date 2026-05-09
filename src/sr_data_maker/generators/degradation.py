from __future__ import annotations

from typing import Any

from sr_data_maker.core.types import GeneratedSample, SourceRecord


class DegradationGenerator:
    def __init__(self, name: str, runner: Any, output: dict[str, Any] | None = None, **_: Any) -> None:
        self.name = name
        self.runner = runner
        self.output = output or {}

    def generate(self, source: SourceRecord, context: Any) -> list[GeneratedSample]:
        result = self.runner.run({"image": load_source_image(source)}, context)
        folder = self.output.get("folder_name", self.name)
        output_path = f"degraded/{folder}/{source.rel_path}"
        manifest = {
            "sample_id": f"{self.name}::{source.rel_path}",
            "task_type": "superres",
            "generation_mode": "degradation",
            "source": {"root": "", "rel_path": source.rel_path, "role": "hr"},
            "input": {"path": output_path, "role": "lq"},
            "target": {
                "source_ref": source.rel_path,
                "role": "target",
                "target_type": "real_gt",
                "storage": "source_reference",
            },
            "outputs": [{"name": "degraded", "path": output_path, "type": "image"}],
            "provenance": {"generator": "DegradationGenerator", "runner": getattr(self.runner, "name", type(self.runner).__name__), **result.meta},
        }
        return [GeneratedSample(image=result.outputs["image"], output_path=output_path, manifest=manifest)]


def load_source_image(source: SourceRecord):
    from PIL import Image

    if "image" in source.meta:
        return source.meta["image"].copy()
    return Image.open(source.path)
