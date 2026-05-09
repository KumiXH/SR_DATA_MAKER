from __future__ import annotations

from pathlib import Path


def output_path_for(output_root: str | Path, branch: str, folder_name: str, rel_path: str) -> Path:
    return Path(output_root) / branch / folder_name / Path(rel_path)
