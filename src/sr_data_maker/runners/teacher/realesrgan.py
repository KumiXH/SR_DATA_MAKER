from __future__ import annotations

from typing import Any

from sr_data_maker.core.types import RunnerOutput


class RealESRGANRunner:
    name = "RealESRGANRunner"

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        return RunnerOutput(outputs={"image": image.copy()}, meta={"model": dict(self.model)})
