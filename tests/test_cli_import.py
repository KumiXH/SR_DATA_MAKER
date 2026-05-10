from pathlib import Path

from sr_data_maker.cli.main import main


def test_cli_main_is_importable():
    assert callable(main)


def test_face_teacher_example_configs_exist():
    assert Path("configs/examples/local_gfpgan_x2.yaml").exists()
    assert Path("configs/examples/local_codeformer_x2.yaml").exists()
    assert Path("configs/examples/local_vqfr_x2.yaml").exists()
