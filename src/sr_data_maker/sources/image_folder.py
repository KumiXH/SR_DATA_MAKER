from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sr_data_maker.core.types import SourceRecord


class ImageFolderSourceReader:
    def __init__(self, root: str | Path, recursive: bool = True, exts: list[str] | None = None) -> None:
        self.root = Path(root)
        self.recursive = recursive
        self.exts = {ext.lower().lstrip(".") for ext in (exts or ["png", "jpg", "jpeg", "webp"])}

    def iter_sources(self) -> Iterator[SourceRecord]:
        pattern = "**/*" if self.recursive else "*"
        for path in sorted(self.root.glob(pattern)):
            if not path.is_file():
                continue
            if path.suffix.lower().lstrip(".") not in self.exts:
                continue
            rel_path = path.relative_to(self.root).as_posix()
            yield SourceRecord(source_id=rel_path, path=path, rel_path=rel_path, meta={})
