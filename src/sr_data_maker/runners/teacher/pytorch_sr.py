from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sr_data_maker.core.types import RunnerOutput


class PyTorchSRAdapter:
    name = "PyTorchSRAdapter"
    display_name = "PyTorch SR"
    package_dirs: tuple[str, ...] = ()

    def __init__(self, **model: Any) -> None:
        self.model = model

    def run(self, inputs: dict[str, Any], context: Any) -> RunnerOutput:
        image = inputs["image"]
        weights = self._weights_path()
        if not weights.exists():
            raise FileNotFoundError(f"{self.display_name} weights not found: {weights}")

        self._add_repo_to_path()
        torch = self._import_torch()
        sr_model = self._build_model()
        self._load_weights(sr_model, weights, torch)
        output = self._run_model(image, sr_model, torch)
        return RunnerOutput(outputs={"image": output}, meta={"model": dict(self.model)})

    def _weights_path(self) -> Path:
        weights = self.model.get("weights")
        if not weights:
            raise FileNotFoundError(f"{self.display_name} weights not configured")
        return Path(str(weights))

    def _add_repo_to_path(self) -> None:
        for repo_path in self._repo_roots():
            if not repo_path.exists():
                raise FileNotFoundError(f"{self.display_name} repo root not found: {repo_path}")
            repo_str = str(repo_path)
            if repo_str not in sys.path:
                sys.path.insert(0, repo_str)

    def _repo_roots(self) -> list[Path]:
        roots: list[Path] = []
        repo_root = self.model.get("repo_root")
        if repo_root:
            roots.append(Path(str(repo_root)))
        basicsr_root = self.model.get("basicsr_root")
        if basicsr_root:
            roots.append(Path(str(basicsr_root)))
        extra_roots = self.model.get("extra_repo_roots") or []
        if isinstance(extra_roots, (str, Path)):
            extra_roots = [extra_roots]
        roots.extend(Path(str(root)) for root in extra_roots)
        return roots

    @staticmethod
    def _import_torch():
        import torch

        return torch

    def _build_model(self):
        raise NotImplementedError

    def _load_weights(self, model: Any, weights_path: Path, torch: Any) -> None:
        state = torch.load(str(weights_path), map_location=self._torch_device(torch))
        if isinstance(state, dict):
            state = state.get("params_ema") or state.get("params") or state.get("state_dict") or state
        model.load_state_dict(state, strict=bool(self.model.get("strict_load", True)))

    def _run_model(self, image: Any, model: Any, torch: Any):
        device = self._torch_device(torch)
        model = model.to(device)
        model.eval()
        if bool(self.model.get("half", False)) and str(device).startswith("cuda"):
            model = model.half()

        tensor = self._image_to_tensor(image, torch).to(device)
        if bool(self.model.get("half", False)) and str(device).startswith("cuda"):
            tensor = tensor.half()

        with torch.no_grad():
            output = self._forward_tensor(model, tensor, torch)
        return self._tensor_to_image(output, torch)

    def _forward_tensor(self, model: Any, tensor: Any, torch: Any):
        tile = int(self.model.get("tile", 0) or 0)
        if tile <= 0:
            return model(tensor)
        return self._tile_forward(model, tensor, torch, tile=tile, tile_pad=int(self.model.get("tile_pad", 10) or 10))

    def _tile_forward(self, model: Any, tensor: Any, torch: Any, tile: int, tile_pad: int):
        _, _, height, width = tensor.size()
        scale = int(self.model.get("scale", 2))
        output = None
        weight = None

        for top in range(0, height, tile):
            for left in range(0, width, tile):
                bottom = min(top + tile, height)
                right = min(left + tile, width)
                top_pad = max(top - tile_pad, 0)
                left_pad = max(left - tile_pad, 0)
                bottom_pad = min(bottom + tile_pad, height)
                right_pad = min(right + tile_pad, width)

                input_tile = tensor[:, :, top_pad:bottom_pad, left_pad:right_pad]
                output_tile = model(input_tile)
                if output is None or weight is None:
                    output, weight = self._allocate_tile_buffers(tensor, output_tile, scale, height, width)

                inner_top = (top - top_pad) * scale
                inner_left = (left - left_pad) * scale
                inner_bottom = inner_top + (bottom - top) * scale
                inner_right = inner_left + (right - left) * scale
                out_top = top * scale
                out_left = left * scale
                out_bottom = bottom * scale
                out_right = right * scale
                output[:, :, out_top:out_bottom, out_left:out_right] += output_tile[
                    :, :, inner_top:inner_bottom, inner_left:inner_right
                ]
                weight[:, :, out_top:out_bottom, out_left:out_right] += 1

        return output / weight.clamp_min(1)

    @staticmethod
    def _allocate_tile_buffers(tensor: Any, output_tile: Any, scale: int, height: int, width: int):
        channels = int(output_tile.size(1))
        output = tensor.new_zeros((1, channels, height * scale, width * scale))
        weight = tensor.new_zeros((1, 1, height * scale, width * scale))
        return output, weight

    def _torch_device(self, torch: Any):
        device = self.model.get("device")
        if device:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _image_to_tensor(image: Any, torch: Any):
        import numpy as np

        array = np.array(image.convert("RGB")).astype("float32") / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor

    @staticmethod
    def _tensor_to_image(tensor: Any, torch: Any):
        import numpy as np
        from PIL import Image

        tensor = tensor.detach().float().clamp_(0, 1).squeeze(0).permute(1, 2, 0).cpu()
        array = (tensor.numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array, mode="RGB")


def as_tuple(value: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value,)
    return tuple(int(item) for item in value)
