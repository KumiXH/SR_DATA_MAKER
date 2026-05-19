from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from sr_data_maker.dataset.manifest import append_jsonl, write_json


class DatasetWriter:
    def __init__(self, output_root: str | Path, manifest_namespace: str | None = None) -> None:
        self.output_root = Path(output_root)
        self.manifest_root = self.output_root / "manifests"
        if manifest_namespace:
            self.manifest_root = self.manifest_root / manifest_namespace

    def write_image(self, rel_output_path: str, image: Image.Image) -> Path:
        target = self.output_root / Path(rel_output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
        return target

    def append_sample(self, record: dict[str, Any]) -> None:
        append_jsonl(self.manifest_root / "samples.jsonl", record)

    def append_failure(self, record: dict[str, Any]) -> None:
        append_jsonl(self.manifest_root / "failures.jsonl", record)

    def write_summary(self, payload: dict[str, Any]) -> None:
        write_json(self.manifest_root / "run_summary.json", payload)
