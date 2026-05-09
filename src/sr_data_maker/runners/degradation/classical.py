from __future__ import annotations

from copy import deepcopy
from random import Random
from typing import Any

from PIL import Image, ImageFilter

from sr_data_maker.core.types import RunnerOutput


_RESAMPLE_MODES = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


class ClassicalDegradationRunner:
    def __init__(
        self,
        scale: int,
        blur: dict[str, Any] | None = None,
        resize: dict[str, Any] | None = None,
        noise: dict[str, Any] | None = None,
        jpeg: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.scale = scale
        self.blur = blur or {"enabled": False}
        self.resize = resize or {"enabled": True, "mode": "bicubic"}
        self.noise = noise or {"enabled": False}
        self.jpeg = jpeg or {"enabled": False}
        self.seed = seed

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image: Image.Image = inputs["image"]
        degraded = image.copy()
        rng = Random(self.seed)

        if self.blur.get("enabled"):
            sigma = self._resolve_value(self.blur.get("sigma", 1.0), rng)
            degraded = degraded.filter(ImageFilter.GaussianBlur(radius=float(sigma)))

        if self.resize.get("enabled", True):
            mode_name = self.resize.get("mode", "bicubic")
            if isinstance(mode_name, list):
                mode_name = mode_name[0]
            resample = _RESAMPLE_MODES[str(mode_name).lower()]
            width = max(1, degraded.width // self.scale)
            height = max(1, degraded.height // self.scale)
            degraded = degraded.resize((width, height), resample=resample)

        # Noise and JPEG branches are wired now and can be deepened later
        if self.noise.get("enabled"):
            degraded = degraded.copy()
        if self.jpeg.get("enabled"):
            degraded = degraded.copy()

        return RunnerOutput(
            outputs={"image": degraded},
            meta={
                "params": {
                    "scale": self.scale,
                    "blur": deepcopy(self.blur),
                    "resize": deepcopy(self.resize),
                    "noise": deepcopy(self.noise),
                    "jpeg": deepcopy(self.jpeg),
                }
            },
        )

    @staticmethod
    def _resolve_value(value: Any, rng: Random) -> Any:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return rng.uniform(float(value[0]), float(value[1]))
        return value
