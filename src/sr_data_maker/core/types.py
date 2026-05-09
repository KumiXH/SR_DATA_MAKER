from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    path: Path
    rel_path: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunnerOutput:
    outputs: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedSample:
    image: Any
    output_path: str
    manifest: dict[str, Any]
