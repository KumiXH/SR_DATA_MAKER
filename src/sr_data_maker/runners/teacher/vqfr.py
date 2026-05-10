from __future__ import annotations

from typing import Any

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class VQFRRunner(FaceTeacherRunnerBase):
    name = "VQFRRunner"
    family = "VQFR"
    display_name = "VQFR"

    def _run_inference(self, image: Any, torch: Any):
        return image

    def _provenance(self) -> dict[str, Any]:
        return {
            **super()._provenance(),
            "fidelity_ratio": self.model.get("fidelity_ratio"),
            "bg_upsampler": self.model.get("bg_upsampler"),
            "arch": self.model.get("arch"),
        }
