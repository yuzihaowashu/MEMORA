"""Persist Segment Encoder records and resume interrupted runs safely."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set


def load_completed_video_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read completion manifest {path}: {exc}") from exc
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"Completion manifest must be a JSON list of video ids: {path}")
    return set(payload)


def write_completed_video_ids(path: Path, video_ids: Set[str]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(sorted(video_ids), indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def prune_uncommitted_records(path: Path, completed_video_ids: Set[str]) -> None:
    """Discard records from videos interrupted before their completion marker."""
    if not path.exists():
        return
    retained = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            if record.get("video_id") in completed_video_ids:
                retained.append(json.dumps(record, ensure_ascii=False))
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(("\n".join(retained) + "\n") if retained else "", encoding="utf-8")
    temporary_path.replace(path)


def summarize_records(path: Path, observation_format: str) -> Dict[str, Any]:
    """Count records and typed observations in a persisted JSONL file."""
    summary: Dict[str, Any] = {"records": 0}
    if observation_format == "memora":
        summary["objects"] = 0
    else:
        summary["by_type"] = {"state": 0, "activity": 0, "environment": 0}
    if not path.exists():
        return summary

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            summary["records"] += 1
            if observation_format == "memora":
                registry = record.get("object_registry") or {}
                if not isinstance(registry, dict):
                    raise ValueError(f"object_registry must be an object in {path} at line {line_number}")
                summary["objects"] += len(registry)
            else:
                fact_type = record.get("fact_type")
                if fact_type not in summary["by_type"]:
                    raise ValueError(f"Unknown fact_type {fact_type!r} in {path} at line {line_number}")
                summary["by_type"][fact_type] += 1
    return summary


def append_records(path: Path, records: List[Dict[str, Any]]) -> None:
    """Append one completed video's records and flush them to disk."""
    with path.open("a", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
