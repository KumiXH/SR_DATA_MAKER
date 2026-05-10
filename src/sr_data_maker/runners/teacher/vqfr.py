from __future__ import annotations

import importlib
import sys
from typing import Any

import numpy as np
from PIL import Image

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class VQFRRunner(FaceTeacherRunnerBase):
    name = "VQFRRunner"
    family = "VQFR"
    display_name = "VQFR"

    def _run_inference(self, image: Any, torch: Any):
        restorer = self._build_restorer(torch)
        _, _, restored = restorer.enhance(
            self._to_bgr_array(image),
            fidelity_ratio=self.model.get("fidelity_ratio"),
            has_aligned=bool(self.model.get("has_aligned", False)),
            only_center_face=bool(self.model.get("only_center_face", False)),
            paste_back=bool(self.model.get("paste_back", True)),
        )
        if restored is None:
            return image
        return self._from_bgr_array(restored)

    def _build_restorer(self, torch: Any):
        self._ensure_torchvision_compat()
        from vqfr.demo_util import VQFR_Demo

        device = self.model.get("device")
        return VQFR_Demo(
            model_path=str(self._weights_path()),
            upscale=int(self.model.get("scale", 2)),
            arch=str(self.model.get("arch", "v1")).lower().replace("vqfr", ""),
            bg_upsampler=None,
            device=device,
        )

    @staticmethod
    def _ensure_torchvision_compat() -> None:
        if "torchvision.transforms.functional_tensor" in sys.modules:
            return
        functional = importlib.import_module("torchvision.transforms.functional")
        sys.modules["torchvision.transforms.functional_tensor"] = functional

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
            "fidelity_ratio": self.model.get("fidelity_ratio"),
            "bg_upsampler": self.model.get("bg_upsampler"),
            "arch": self.model.get("arch"),
        }
