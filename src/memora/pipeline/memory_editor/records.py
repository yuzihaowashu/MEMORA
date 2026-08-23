"""Load and group Segment Encoder records for memory formation."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """One entry in the Flat-1D memory baseline."""

    id: str
    text: str
    episode_id: Optional[str] = None
    time_window: Optional[Dict[str, int]] = None
    fact_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "text": self.text}
        if self.episode_id:
            result["episode_id"] = self.episode_id
        if self.time_window:
            result["time_window"] = self.time_window
        if self.fact_type:
            result["fact_type"] = self.fact_type
        return result


def load_segment_observations(input_file: Path) -> List[Dict[str, Any]]:
    """Load Segment Encoder observations from JSONL."""
    observations = []
    with input_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                observation = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_file} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(observation, dict):
                raise ValueError(
                    f"Expected a JSON object in {input_file} at line {line_number}"
                )
            observations.append(observation)
    logger.info("Loaded %d segment observations", len(observations))
    return observations


def load_flat_1d_memory(memory_file: Path) -> List[MemoryEntry]:
    """Load an initial Flat-1D memory state."""
    with memory_file.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    memory = [
        MemoryEntry(
            id=entry["id"],
            text=entry["text"],
            episode_id=entry.get("episode_id"),
            time_window=entry.get("time_window"),
            fact_type=entry.get("fact_type"),
        )
        for entry in data.get("memory", [])
    ]
    logger.info("Loaded initial Flat-1D memory with %d entries", len(memory))
    return memory


def group_observations_by_scope(
    observations: List[Dict[str, Any]],
    scope: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """Group observations by video, participant, or the complete collection."""
    valid_scopes = {"per_video", "per_participant", "global"}
    if scope not in valid_scopes:
        raise ValueError(f"Unknown memory scope {scope!r}; expected one of {sorted(valid_scopes)}")

    grouped = defaultdict(list)
    for observation in observations:
        episode_id = observation.get("episode_id", "unknown")
        if scope == "per_video":
            key = episode_id
        elif scope == "per_participant":
            key = episode_id.split("_")[0] if "_" in episode_id else episode_id
        else:
            key = "global"
        grouped[key].append(observation)
    return grouped
