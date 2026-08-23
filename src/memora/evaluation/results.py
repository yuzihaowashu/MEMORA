"""Atomic result writing for MEMORA-Bench evaluations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_json_atomic(path: str | Path, data: Any) -> None:
    """Write JSON atomically so interrupted runs do not leave partial results."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def is_complete_eam_qa_result(path: str | Path) -> bool:
    """Return whether an existing file is a complete EAM-QA result."""
    result_path = Path(path)
    try:
        with result_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return False
    total = data.get("total_questions")
    results = data["results"]
    if not isinstance(total, int) or total <= 0 or len(results) != total:
        return False
    return all(
        isinstance(item, dict)
        and bool(str(item.get("question_id") or "").strip())
        and not item.get("error")
        and isinstance(item.get("is_correct"), bool)
        for item in results
    )
