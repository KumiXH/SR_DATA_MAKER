from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

from sr_data_maker.runners.teacher.pytorch_sr import PyTorchSRAdapter, as_tuple

_HAT_ARCH_MODULE: ModuleType | None = None


class HATAdapter(PyTorchSRAdapter):
    name = "HATAdapter"
    display_name = "HAT"

    def _build_model(self):
        repo_root = self.model.get("repo_root")
        if repo_root:
            HAT = self._load_hat_arch_from_file(
                ImportError("Loading HAT directly from repo_root to avoid package side effects.")
            )
        else:
            try:
                from hat.archs.hat_arch import HAT
            except ImportError as exc:
                HAT = self._load_hat_arch_from_file(exc)

        scale = int(self.model.get("scale", 2))
        return HAT(
            upscale=scale,
            patch_size=int(self.model.get("patch_size", 1)),
            in_chans=int(self.model.get("in_chans", 3)),
            img_size=int(self.model.get("img_size", 64)),
            window_size=int(self.model.get("window_size", 16)),
            img_range=float(self.model.get("img_range", 1.0)),
            depths=as_tuple(self.model.get("depths"), (6, 6, 6, 6, 6, 6)),
            embed_dim=int(self.model.get("embed_dim", 180)),
            num_heads=as_tuple(self.model.get("num_heads"), (6, 6, 6, 6, 6, 6)),
            compress_ratio=int(self.model.get("compress_ratio", 3)),
            squeeze_factor=int(self.model.get("squeeze_factor", 30)),
            conv_scale=float(self.model.get("conv_scale", 0.01)),
            overlap_ratio=float(self.model.get("overlap_ratio", 0.5)),
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

    def _load_hat_arch_from_file(self, original_error: Exception):
        global _HAT_ARCH_MODULE
        repo_root = self.model.get("repo_root")
        if not repo_root:
            raise ImportError(
                "HATAdapter requires the official HAT repo. Configure model.repo_root or clone it under third_party/HAT."
            ) from original_error

        module_path = Path(str(repo_root)) / "hat" / "archs" / "hat_arch.py"
        if not module_path.exists():
            raise ImportError(
                "HATAdapter requires the official HAT repo. Configure model.repo_root or clone it under third_party/HAT."
            ) from original_error

        spec = importlib.util.spec_from_file_location("sr_data_maker_hat_arch", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load HAT architecture module from: {module_path}") from original_error

        if _HAT_ARCH_MODULE is None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _HAT_ARCH_MODULE = module
        return _HAT_ARCH_MODULE.HAT
