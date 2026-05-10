from __future__ import annotations

from typing import Any

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class GFPGANRunner(FaceTeacherRunnerBase):
    name = "GFPGANRunner"
    family = "GFPGAN"
    display_name = "GFPGAN"

    def _run_inference(self, image: Any, torch: Any):
        return image

    def _provenance(self) -> dict[str, Any]:
        return {
            **super()._provenance(),
            "arch": self.model.get("arch"),
            "channel_multiplier": self.model.get("channel_multiplier"),
            "bg_upsampler": self.model.get("bg_upsampler"),
        }
