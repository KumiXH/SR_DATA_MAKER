from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from sr_data_maker.runners.teacher.resshift import ResShiftRunner
from sr_data_maker.runners.teacher.stablesr import StableSRRunner
from sr_data_maker.runners.teacher.supir import SUPIRRunner


def test_stablesr_runner_rejects_missing_weights(tmp_path):
    runner = StableSRRunner(name="StableSR_x4", weights=str(tmp_path / "missing.safetensors"), scale=4)

    with pytest.raises(FileNotFoundError, match="StableSR weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_resshift_runner_rejects_missing_weights(tmp_path):
    runner = ResShiftRunner(name="ResShift_x4", weights=str(tmp_path / "missing.safetensors"), scale=4)

    with pytest.raises(FileNotFoundError, match="ResShift weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_supir_runner_rejects_missing_weights(tmp_path):
    runner = SUPIRRunner(name="SUPIR_x4", weights=str(tmp_path / "missing.safetensors"), scale=4)

    with pytest.raises(FileNotFoundError, match="SUPIR weights not found"):
        runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)


def test_stablesr_runner_exposes_diffusion_provenance(tmp_path):
    weights = tmp_path / "stablesr.safetensors"
    weights.write_bytes(b"weights")
    captured = {}

    class TestableStableSRRunner(StableSRRunner):
        def _run_inference(self, image, torch):
            captured["model"] = dict(self.model)
            return image

        def _import_torch(self):
            return object()

    runner = TestableStableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        scale=4,
        steps=50,
        tile_size=256,
        precision="fp16",
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.outputs["image"].size == (8, 8)
    assert output.meta["diffusion_model"] is True
    assert output.meta["diffusion_model_family"] == "StableSR"
    assert output.meta["steps"] == 50
    assert captured["model"]["tile_size"] == 256


def test_stablesr_runner_builds_official_script_command(tmp_path):
    weights = tmp_path / "stablesr.safetensors"
    vqgan = tmp_path / "vqgan.ckpt"
    weights.write_bytes(b"weights")
    vqgan.write_bytes(b"weights")
    repo_root = tmp_path / "StableSR"
    repo_root.mkdir()

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        vqgan_weights=str(vqgan),
        scale=4,
        steps=20,
        input_size=512,
        tile_overlap=48,
        vqgan_tile_size=1024,
        vqgan_tile_stride=768,
        color_fix_type="wavelet",
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    assert command[0].endswith("python.exe") or command[0].endswith("python")
    assert command[1].replace("\\", "/").endswith("scripts/sr_val_ddpm_text_T_vqganfin_oldcanvas_tile.py")
    assert "--ddpm_steps" in command
    assert "20" in command
    assert "--colorfix_type" in command
    assert "wavelet" in command
    assert "--input_size" in command
    assert "512" in command
    assert "--vqgantile_size" in command
    assert "1024" in command
    assert "--vqgantile_stride" in command
    assert "768" in command


def test_stablesr_runner_resolves_relative_config_path_to_absolute(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "stablesr.ckpt"
    vqgan = tmp_path / "weights" / "vqgan.ckpt"
    repo_root = tmp_path / "third_party" / "StableSR"
    config_path = tmp_path / "third_party" / "StableSR" / "configs" / "stableSRNew" / "v2-finetune_text_T_512.yaml"
    weights.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    vqgan.write_bytes(b"weights")
    config_path.write_text("model: {}", encoding="utf-8")
    repo_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root="./third_party/StableSR",
        vqgan_weights=str(vqgan),
        config_path="./third_party/StableSR/configs/stableSRNew/v2-finetune_text_T_512.yaml",
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    config_index = command.index("--config") + 1
    assert Path(command[config_index]).is_absolute()
    assert Path(command[config_index]) == config_path.resolve()


def test_stablesr_runner_resolves_relative_vqgan_path_to_absolute(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "stablesr.ckpt"
    vqgan = tmp_path / "weights" / "vqgan.ckpt"
    repo_root = tmp_path / "third_party" / "StableSR"
    weights.parent.mkdir(parents=True, exist_ok=True)
    repo_root.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"weights")
    vqgan.write_bytes(b"weights")
    monkeypatch.chdir(tmp_path)

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root="./third_party/StableSR",
        vqgan_weights="./weights/vqgan.ckpt",
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    vqgan_index = command.index("--vqgan_ckpt") + 1
    assert Path(command[vqgan_index]).is_absolute()
    assert Path(command[vqgan_index]) == vqgan.resolve()


def test_stablesr_runner_sets_repo_pythonpath_for_subprocess(tmp_path, monkeypatch):
    weights = tmp_path / "stablesr.safetensors"
    vqgan = tmp_path / "vqgan.ckpt"
    repo_root = tmp_path / "StableSR"
    weights.write_bytes(b"weights")
    vqgan.write_bytes(b"weights")
    repo_root.mkdir()
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        vqgan_weights=str(vqgan),
    )

    env = runner._subprocess_env()

    assert env["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    assert str(repo_root) in env["PYTHONPATH"]
    assert "existing-path" in env["PYTHONPATH"]


def test_stablesr_runner_adds_local_dependency_sources_to_pythonpath(tmp_path, monkeypatch):
    weights = tmp_path / "stablesr.safetensors"
    vqgan = tmp_path / "vqgan.ckpt"
    repo_root = tmp_path / "StableSR"
    taming_root = tmp_path / "src" / "taming-transformers"
    clip_root = tmp_path / "src" / "clip"
    weights.write_bytes(b"weights")
    vqgan.write_bytes(b"weights")
    repo_root.mkdir()
    taming_root.mkdir(parents=True)
    clip_root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        vqgan_weights=str(vqgan),
    )

    env = runner._subprocess_env()

    assert str(taming_root) in env["PYTHONPATH"]
    assert str(clip_root) in env["PYTHONPATH"]


def test_diffusion_runner_uses_model_python_executable(tmp_path):
    weights = tmp_path / "model.ckpt"
    repo_root = tmp_path / "StableSR"
    python_executable = tmp_path / "env" / "python.exe"
    weights.write_bytes(b"weights")
    repo_root.mkdir()
    python_executable.parent.mkdir(parents=True)
    python_executable.write_bytes(b"")

    runner = StableSRRunner(
        name="StableSR_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        vqgan_weights=str(weights),
        python_executable=str(python_executable),
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    assert Path(command[0]) == python_executable


def test_resshift_runner_exposes_chop_controls(tmp_path):
    weights = tmp_path / "resshift.safetensors"
    weights.write_bytes(b"weights")

    class TestableResShiftRunner(ResShiftRunner):
        def _run_inference(self, image, torch):
            return image

        def _import_torch(self):
            return object()

    runner = TestableResShiftRunner(
        name="ResShift_x4",
        weights=str(weights),
        scale=4,
        steps=15,
        chop_size=256,
        chop_stride=128,
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.meta["diffusion_model_family"] == "ResShift"
    assert output.meta["chop_size"] == 256
    assert output.meta["chop_stride"] == 128


def test_resshift_runner_builds_official_script_command(tmp_path):
    weights = tmp_path / "resshift.safetensors"
    weights.write_bytes(b"weights")
    repo_root = tmp_path / "ResShift"
    repo_root.mkdir()

    runner = ResShiftRunner(
        name="ResShift_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        scale=4,
        steps=15,
        version="v3",
        chop_size=256,
        chop_stride=128,
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    assert command[0].endswith("python.exe") or command[0].endswith("python")
    assert command[1].replace("\\", "/").endswith("inference_resshift.py")
    assert "--task" in command
    assert "realsr" in command
    assert "--version" in command
    assert "v3" in command


def test_resshift_runner_stages_weights_into_official_repo_layout(tmp_path):
    weights = tmp_path / "weights" / "resshift_realsrx4_s4_v3.pth"
    autoencoder = tmp_path / "weights" / "autoencoder_vq_f4.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"main-weights")
    autoencoder.write_bytes(b"autoencoder-weights")
    repo_root = tmp_path / "ResShift"
    (repo_root / "weights").mkdir(parents=True)

    runner = ResShiftRunner(
        name="ResShift_x4",
        weights=str(weights),
        autoencoder_weights=str(autoencoder),
        repo_root=str(repo_root),
        scale=4,
        version="v3",
        task="realsr",
    )

    runner._prepare_runtime_artifacts()

    staged_main = repo_root / "weights" / "resshift_realsrx4_s4_v3.pth"
    staged_autoencoder = repo_root / "weights" / "autoencoder_vq_f4.pth"
    assert staged_main.exists()
    assert staged_autoencoder.exists()
    assert staged_main.read_bytes() == b"main-weights"
    assert staged_autoencoder.read_bytes() == b"autoencoder-weights"


def test_resshift_runner_resolves_relative_repo_root_to_absolute(tmp_path, monkeypatch):
    weights = tmp_path / "weights" / "resshift_realsrx4_s4_v3.pth"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"main-weights")
    repo_root = tmp_path / "third_party" / "ResShift"
    repo_root.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    runner = ResShiftRunner(
        name="ResShift_x4",
        weights=str(weights),
        repo_root="./third_party/ResShift",
        scale=4,
        version="v3",
        task="realsr",
    )

    command = runner._build_command(tmp_path / "in.png", tmp_path / "out")

    assert Path(command[1]).is_absolute()
    assert Path(command[1]) == repo_root / "inference_resshift.py"
    assert runner._repo_root() == repo_root.resolve()


def test_supir_runner_exposes_prompt_controls(tmp_path):
    weights = tmp_path / "supir.safetensors"
    weights.write_bytes(b"weights")

    class TestableSUPIRRunner(SUPIRRunner):
        def _run_inference(self, image, torch):
            return image

        def _import_torch(self):
            return object()

    runner = TestableSUPIRRunner(
        name="SUPIR_x4",
        weights=str(weights),
        scale=4,
        steps=30,
        prompt="clean realistic photo",
        negative_prompt="oversmoothed, blurry",
        color_fix_type="wavelet",
    )

    output = runner.run({"image": Image.new("RGB", (8, 8), "white")}, context=None)

    assert output.meta["diffusion_model_family"] == "SUPIR"
    assert output.meta["prompt"] == "clean realistic photo"
    assert output.meta["color_fix_type"] == "wavelet"


def test_supir_runner_builds_official_script_command(tmp_path):
    weights = tmp_path / "supir.safetensors"
    weights.write_bytes(b"weights")
    repo_root = tmp_path / "SUPIR"
    repo_root.mkdir()
    opt_path = repo_root / "options" / "SUPIR_v0_tiled.yaml"
    opt_path.parent.mkdir()
    opt_path.write_text("model: {}\n", encoding="utf-8")

    runner = SUPIRRunner(
        name="SUPIR_x4",
        weights=str(weights),
        repo_root=str(repo_root),
        opt_path=str(opt_path),
        scale=1,
        steps=2,
        prompt="clean realistic photo",
        negative_prompt="oversmoothed, blurry",
        color_fix_type="Wavelet",
        no_llava=True,
        loading_half_params=True,
        use_tile_vae=True,
        encoder_tile_size=192,
        decoder_tile_size=32,
        min_size=256,
        ae_dtype="bf16",
        diff_dtype="fp16",
    )

    command = runner._build_command(tmp_path / "inputs", tmp_path / "outputs")

    assert command[0].endswith("python.exe") or command[0].endswith("python")
    assert command[1].replace("\\", "/").endswith("test.py")
    assert command[2] == "--opt"
    assert command[3].replace("\\", "/").endswith("SUPIR_v0_tiled.yaml")
    assert "--img_dir" in command
    assert command[command.index("--img_dir") + 1] == str(tmp_path / "inputs")
    assert "--save_dir" in command
    assert "--upscale" in command
    assert "--loading_half_params" in command
    assert "--use_tile_vae" in command
    assert "--encoder_tile_size" in command
    assert "--decoder_tile_size" in command
    assert "--min_size" in command
