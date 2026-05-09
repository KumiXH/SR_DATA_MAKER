from PIL import Image
import pytest

from sr_data_maker.runners.teacher.realesrgan import RealESRGANRunner


def test_realesrgan_runner_rejects_missing_weights(tmp_path):
    runner = RealESRGANRunner(
        name="RealESRGAN_x2plus",
        weights=str(tmp_path / "missing.pth"),
        scale=2,
    )

    with pytest.raises(FileNotFoundError, match="weights"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)
