"""Locate input videos and divide them into fixed-duration segments."""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def get_video_duration(video_path: str) -> float:
    """Read a video's duration in seconds with ffprobe."""
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "csv=p=0", video_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return float(result.stdout.strip())
    raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")


def generate_segments_from_duration(
    video_duration: float,
    segment_length: float = 10.0,
) -> List[Dict]:
    """Generate non-overlapping segment boundaries for a video."""
    segments = []
    start = 0.0
    while start < video_duration:
        end = min(start + segment_length, video_duration)
        if end - start < 1.0:
            break
        segments.append({"segment_start": round(start, 2), "segment_end": round(end, 2)})
        start = end
    return segments


def load_video_ids(filepath: str) -> List[str]:
    """Read unique video IDs from a text file."""
    video_ids = []
    seen = set()
    with open(filepath, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            video_id = line.strip()
            if not video_id or video_id.startswith("#"):
                continue
            if video_id in seen:
                raise ValueError(
                    f"Duplicate video id {video_id!r} in {filepath} at line {line_number}"
                )
            seen.add(video_id)
            video_ids.append(video_id)
    if not video_ids:
        raise ValueError(f"No video ids found in {filepath}")
    return video_ids


def find_video_path(video_id: str, video_dir: Path) -> Optional[Path]:
    """Locate an EPIC-KITCHENS video, with direct-path fallbacks."""
    extensions = [".MP4", ".mp4", ".MOV", ".mov", ".avi", ".mkv"]
    participant = video_id.split("_")[0] if "_" in video_id else None
    candidate_dirs = []
    if participant:
        candidate_dirs.extend([
            video_dir / participant / "videos",
            video_dir / participant,
        ])
    candidate_dirs.append(video_dir)
    for directory in candidate_dirs:
        for extension in extensions:
            path = directory / f"{video_id}{extension}"
            if path.exists():
                return path
    return None
