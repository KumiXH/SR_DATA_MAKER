from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class GFPGANRunner(FaceTeacherRunnerBase):
    name = "GFPGANRunner"
    family = "GFPGAN"
    display_name = "GFPGAN"

    def _run_inference(self, image: Any, torch: Any):
        restorer = self._build_restorer(torch)
        _, _, restored = restorer.enhance(
            self._to_bgr_array(image),
            has_aligned=bool(self.model.get("has_aligned", False)),
            only_center_face=bool(self.model.get("only_center_face", False)),
            paste_back=bool(self.model.get("paste_back", True)),
            weight=float(self.model.get("weight", 0.5)),
        )
        if restored is None:
            return image
        return self._from_bgr_array(restored)

    def _build_restorer(self, torch: Any):
        from gfpgan import GFPGANer

        device = self.model.get("device")
        return GFPGANer(
            model_path=str(self._weights_path()),
            upscale=int(self.model.get("scale", 2)),
            arch=str(self.model.get("arch", "clean")),
            channel_multiplier=int(self.model.get("channel_multiplier", 2)),
            bg_upsampler=None,
            device=device,
        )

    @staticmethod
    def _to_bgr_array(image: Any):
        rgb = image.convert("RGB")
        return np.array(rgb)[:, :, ::-1]

    @staticmethod
    def _from_bgr_array(array: Any):
        rgb = np.asarray(array)[:, :, ::-1]
        return Image.fromarray(rgb.astype("uint8"), mode="RGB")

    def _provenance(self) -> dict[str, Any]:
        return {
            **super()._provenance(),
            "arch": self.model.get("arch"),
            "channel_multiplier": self.model.get("channel_multiplier"),
            "bg_upsampler": self.model.get("bg_upsampler"),
        }
