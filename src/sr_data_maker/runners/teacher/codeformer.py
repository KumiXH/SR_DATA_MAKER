from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from sr_data_maker.runners.teacher.face_base import FaceTeacherRunnerBase


class CodeFormerRunner(FaceTeacherRunnerBase):
    name = "CodeFormerRunner"
    family = "CodeFormer"
    display_name = "CodeFormer"

    def _run_inference(self, image: Any, torch: Any):
        if bool(self.model.get("has_aligned", False)):
            return self._run_aligned_inference(image, torch)
        return self._run_whole_image_inference(image, torch)

    def _run_aligned_inference(self, image: Any, torch: Any):
        restored_faces = self._restore_faces(
            [self._to_bgr_array(image.resize((512, 512)))],
            torch=torch,
        )
        if not restored_faces:
            return image
        return self._from_bgr_array(restored_faces[0])

    def _run_whole_image_inference(self, image: Any, torch: Any):
        import cv2
        from facelib.utils.face_restoration_helper import FaceRestoreHelper

        face_helper = FaceRestoreHelper(
            int(self.model.get("scale", 2)),
            face_size=512,
            crop_ratio=(1, 1),
            det_model=str(self.model.get("detection_model", "retinaface_resnet50")),
            save_ext="png",
            use_parse=True,
            device=self.model.get("device"),
        )
        face_helper.clean_all()
        bgr_image = self._to_bgr_array(image)
        face_helper.read_image(bgr_image)
        face_helper.get_face_landmarks_5(
            only_center_face=bool(self.model.get("only_center_face", False)),
            resize=640,
            eye_dist_threshold=5,
        )
        face_helper.align_warp_face()
        restored_faces = self._restore_faces(face_helper.cropped_faces, torch=torch)
        for cropped_face, restored_face in zip(face_helper.cropped_faces, restored_faces):
            face_helper.add_restored_face(restored_face, cropped_face)

        if not restored_faces:
            return image

        bg_upsampler = self._build_background_upsampler(torch)
        bg_image = None
        if bg_upsampler is not None:
            bg_image = bg_upsampler.enhance(
                bgr_image,
                outscale=int(self.model.get("scale", 2)),
            )[0]

        face_helper.get_inverse_affine(None)
        restored = face_helper.paste_faces_to_input_image(
            upsample_img=bg_image,
            face_upsampler=bg_upsampler if bool(self.model.get("face_upsample", False)) else None,
            draw_box=bool(self.model.get("draw_box", False)),
        )
        if restored is None:
            restored = restored_faces[0]
        return self._from_bgr_array(restored)

    def _restore_faces(self, cropped_faces: list[Any], torch: Any) -> list[Any]:
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        net = self._build_network(torch)
        restored_faces: list[Any] = []
        total_faces = len(cropped_faces)
        for index, cropped_face in enumerate(cropped_faces, start=1):
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.model.get("device"))

            try:
                with torch.no_grad():
                    output = net(
                        cropped_face_t,
                        w=float(self.model.get("fidelity_weight", 0.7)),
                        adain=True,
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as exc:
                raise RuntimeError(f"CodeFormer failed to restore face {index}/{total_faces}: {exc}") from exc
            restored_faces.append(restored_face.astype("uint8"))
        return restored_faces

    def _build_network(self, torch: Any):
        from basicsr.utils.registry import ARCH_REGISTRY

        device = self.model.get("device")
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        checkpoint = torch.load(str(self._weights_path()), map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["params_ema"])
        net.eval()
        return net

    def _build_background_upsampler(self, torch: Any):
        upsampler_name = self.model.get("background_upsampler")
        if upsampler_name not in {None, "realesrgan"} and not bool(self.model.get("face_upsample", False)):
            return None
        if upsampler_name != "realesrgan" and not bool(self.model.get("face_upsample", False)):
            return None

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from basicsr.utils.realesrgan_utils import RealESRGANer

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        return RealESRGANer(
            scale=2,
            model_path=str(
                self.model.get(
                    "background_weights",
                    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
                )
            ),
            model=model,
            tile=int(self.model.get("bg_tile", 400)),
            tile_pad=40,
            pre_pad=0,
            half=bool(self.model.get("half", False)),
            device=self.model.get("device"),
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
            "fidelity_weight": self.model.get("fidelity_weight"),
            "face_upsample": self.model.get("face_upsample"),
            "background_upsampler": self.model.get("background_upsampler"),
            "has_aligned": self.model.get("has_aligned"),
            "paste_back": self.model.get("paste_back"),
            "detection_model": self.model.get("detection_model"),
        }
