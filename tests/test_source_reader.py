from PIL import Image

from sr_data_maker.core.registry import Registry
from sr_data_maker.sources.image_folder import ImageFolderSourceReader


def test_registry_builds_registered_class():
    registry = Registry("demo")

    @registry.register("Thing")
    class Thing:
        def __init__(self, value):
            self.value = value

    instance = registry.build({"type": "Thing", "value": 7})

    assert instance.value == 7


def test_image_folder_reader_preserves_nested_relative_paths(tmp_path):
    root = tmp_path / "raw"
    nested = root / "city" / "day"
    nested.mkdir(parents=True)
    Image.new("RGB", (4, 4), "red").save(nested / "img001.png")
    (nested / "ignore.txt").write_text("skip", encoding="utf-8")

    reader = ImageFolderSourceReader(root=root, recursive=True, exts=["png"])
    records = list(reader.iter_sources())

    assert len(records) == 1
    assert records[0].rel_path == "city/day/img001.png"
    assert records[0].path == nested / "img001.png"
