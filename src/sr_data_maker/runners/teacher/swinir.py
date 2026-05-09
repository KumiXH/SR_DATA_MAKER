from __future__ import annotations

from typing import Any

from sr_data_maker.runners.teacher.pytorch_sr import PyTorchSRAdapter, as_tuple


class SwinIRAdapter(PyTorchSRAdapter):
    name = "SwinIRAdapter"
    display_name = "SwinIR"

    def _build_model(self):
        try:
            from models.network_swinir import SwinIR
        except ImportError as exc:
            raise ImportError(
                "SwinIRAdapter requires the official SwinIR repo. Configure model.repo_root or clone it under third_party/SwinIR."
            ) from exc

        scale = int(self.model.get("scale", 2))
        return SwinIR(
            upscale=scale,
            patch_size=int(self.model.get("patch_size", 1)),
            in_chans=int(self.model.get("in_chans", 3)),
            img_size=int(self.model.get("img_size", 64)),
            window_size=int(self.model.get("window_size", 8)),
            img_range=float(self.model.get("img_range", 1.0)),
            depths=as_tuple(self.model.get("depths"), (6, 6, 6, 6, 6, 6)),
            embed_dim=int(self.model.get("embed_dim", 180)),
            num_heads=as_tuple(self.model.get("num_heads"), (6, 6, 6, 6, 6, 6)),
            mlp_ratio=float(self.model.get("mlp_ratio", 2.0)),
            qkv_bias=bool(self.model.get("qkv_bias", True)),
            qk_scale=self.model.get("qk_scale"),
            drop_rate=float(self.model.get("drop_rate", 0.0)),
            attn_drop_rate=float(self.model.get("attn_drop_rate", 0.0)),
            drop_path_rate=float(self.model.get("drop_path_rate", 0.1)),
            ape=bool(self.model.get("ape", False)),
            patch_norm=bool(self.model.get("patch_norm", True)),
            use_checkpoint=bool(self.model.get("use_checkpoint", False)),
            upsampler=str(self.model.get("upsampler", "pixelshuffle")),
            resi_connection=str(self.model.get("resi_connection", "1conv")),
        )
