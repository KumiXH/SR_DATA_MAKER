from pathlib import Path

from sr_data_maker.runners.teacher.realesrgan import RealESRGANRunner


def test_realesrgan_runner_resolves_repo_paths(tmp_path):
    real_esrgan = tmp_path / "Real-ESRGAN"
    basicsr = tmp_path / "BasicSR"
    real_esrgan.mkdir()
    basicsr.mkdir()

    runner = RealESRGANRunner(
        name="RealESRGAN_x2plus",
        weights=str(tmp_path / "missing.pth"),
        repo_root=str(real_esrgan),
        basicsr_root=str(basicsr),
        scale=2,
    )

    real_path, basic_path = runner._repo_roots()

    assert real_path == real_esrgan
    assert basic_path == basicsr
