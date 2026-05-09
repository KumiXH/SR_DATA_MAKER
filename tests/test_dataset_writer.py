import json

from PIL import Image

from sr_data_maker.dataset.naming import output_path_for
from sr_data_maker.dataset.writer import DatasetWriter


def test_output_path_mirrors_source_relative_path(tmp_path):
    path = output_path_for(tmp_path, "degraded", "degradation_x2", "city/day/img001.png")

    assert path == tmp_path / "degraded" / "degradation_x2" / "city" / "day" / "img001.png"


def test_dataset_writer_writes_image_and_manifest(tmp_path):
    writer = DatasetWriter(output_root=tmp_path)
    image = Image.new("RGB", (3, 3), "blue")
    record = {"sample_id": "sample-1", "outputs": [{"path": "degraded/degradation_x2/a.png"}]}

    writer.write_image("degraded/degradation_x2/a.png", image)
    writer.append_sample(record)

    assert (tmp_path / "degraded" / "degradation_x2" / "a.png").exists()
    lines = (tmp_path / "manifests" / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["sample_id"] == "sample-1"
