from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sr_data_maker.core.types import RunnerOutput


class RealESRGANRunner:
    name = "RealESRGANRunner"

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        weights = Path(self.model.get("weights", ""))
        if not weights.exists():
            raise FileNotFoundError(f"RealESRGAN weights not found: {weights}")

        torch = self._import_torch()
        RealESRGANer, rrdbnet_cls = self._import_realesrgan()
        model = self._build_model(rrdbnet_cls)

        device = self.model.get("device")
        half = bool(self.model.get("half", False))
        upsampler = RealESRGANer(
            scale=int(self.model.get("scale", 2)),
            model_path=str(weights),
            model=model,
            tile=int(self.model.get("tile", 0) or 0),
            tile_pad=int(self.model.get("tile_pad", 10) or 10),
            pre_pad=int(self.model.get("pre_pad", 0) or 0),
            half=half and bool(torch.cuda.is_available()),
            gpu_id=self._gpu_id(device),
        )

        output, _ = upsampler.enhance(self._to_bgr_array(image), outscale=float(self.model.get("scale", 2)))
        return RunnerOutput(outputs={"image": self._from_bgr_array(output)}, meta={"model": dict(self.model)})

    @staticmethod
    def _import_torch():
        import torch

        return torch

    def _import_realesrgan(self):
        real_esrgan_root, basicsr_root = self._repo_roots()
        for root in (real_esrgan_root, basicsr_root):
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        return RealESRGANer, RRDBNet

    def _repo_roots(self) -> tuple[Path, Path]:
        repo_root = self.model.get("repo_root")
        basicsr_root = self.model.get("basicsr_root")
        if repo_root and basicsr_root:
            return Path(repo_root), Path(basicsr_root)

        current = Path(__file__).resolve()
        project_root = current.parents[4]
        return project_root / "third_party" / "Real-ESRGAN", project_root / "third_party" / "BasicSR"

    def _build_model(self, rrdbnet_cls):
        name = str(self.model.get("name", "RealESRGAN_x2plus"))
        scale = int(self.model.get("scale", 2))
        if name == "RealESRGAN_x2plus":
            return rrdbnet_cls(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        if name == "RealESRGAN_x4plus":
            return rrdbnet_cls(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
        raise ValueError(f"Unsupported RealESRGAN model name: {name}")

    @staticmethod
    def _to_bgr_array(image):
        import numpy as np

        rgb = image.convert("RGB")
        return np.array(rgb)[:, :, ::-1]

    @staticmethod
    def _from_bgr_array(array):
        import numpy as np
        from PIL import Image

        rgb = np.asarray(array)[:, :, ::-1]
        return Image.fromarray(rgb.astype("uint8"), mode="RGB")

    @staticmethod
    def _gpu_id(device: Any) -> int | None:
        if device is None:
            return None
        if isinstance(device, str) and device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        if isinstance(device, int):
            return device
        return None
