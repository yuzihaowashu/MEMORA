"""Configuration for the paper's EPIC-KITCHENS memory-formation pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FormationConfig:
    """Settings shared across the memory-formation lifecycle."""

    experience_description: str
    segment_length: float = 10.0
    video_fps: Optional[float] = None
    location_keywords: Dict[str, str] = field(default_factory=dict)
    default_location: str = "kitchen_general"

    def copy(self) -> "FormationConfig":
        """Return an independent configuration instance."""
        return copy.deepcopy(self)

    @property
    def segment_length_int(self) -> int:
        """Whole-second segment duration used by formation prompts."""
        return int(self.segment_length)

    def get_location_id(self, description: str) -> str:
        """Map a place description to an Environment Memory identifier."""
        description_lower = description.lower()
        for keyword, location_id in self.location_keywords.items():
            if keyword in description_lower:
                return location_id
        return self.default_location

    def __repr__(self) -> str:
        return (
            "FormationConfig("
            f"experience={self.experience_description!r}, "
            f"segment_length={self.segment_length}s)"
        )


EPIC_KITCHENS_CONFIG = FormationConfig(
    experience_description="egocentric kitchen activity",
    segment_length=10.0,
    video_fps=None,
    location_keywords={
        "sink": "sink_area",
        "faucet": "sink_area",
        "dishwasher": "dishwasher_area",
        "stove": "stove_area",
        "oven": "stove_area",
        "hob": "stove_area",
        "refrigerator": "fridge_area",
        "fridge": "fridge_area",
        "counter": "counter_area",
        "countertop": "counter_area",
        "table": "table_area",
        "dining": "table_area",
        "cabinet": "cabinet_area",
        "drawer": "cabinet_area",
        "cupboard": "cabinet_area",
        "microwave": "counter_area",
    },
)
