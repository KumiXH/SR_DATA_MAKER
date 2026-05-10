from __future__ import annotations

from typing import Any

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class CodeFormerRunner(FaceTeacherRunnerBase):
    name = "CodeFormerRunner"
    family = "CodeFormer"
    display_name = "CodeFormer"

    def _run_inference(self, image: Any, torch: Any):
        return image

    def _provenance(self) -> dict[str, Any]:
        return {
            **super()._provenance(),
            "fidelity_weight": self.model.get("fidelity_weight"),
            "face_upsample": self.model.get("face_upsample"),
            "background_upsampler": self.model.get("background_upsampler"),
        }
