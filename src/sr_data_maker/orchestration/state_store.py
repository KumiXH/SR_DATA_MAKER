from __future__ import annotations

import json
from pathlib import Path


class RunStateStore:
    def __init__(self, output_root: str | Path) -> None:
        self.path = Path(output_root) / "manifests" / "state.jsonl"
        self._completed: set[str] = set()
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                self._completed.add(json.loads(line)["key"])

    def has(self, key: str) -> bool:
        return key in self._completed

    def add(self, key: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key}, ensure_ascii=False) + "\n")
        self._completed.add(key)
