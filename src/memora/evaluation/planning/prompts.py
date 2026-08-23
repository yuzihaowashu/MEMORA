"""Public prompt registry for MEMORA-Planning evaluation settings."""

from memora.evaluation.planning.prompt_profiles import (
    GRAPH_2D_PROMPT,
    MEMORA_EPISODIC_PROMPT,
    MEMORA_FULL_PROMPT,
    PLANNING_FORCED_ANSWER_PROMPT,
    PLANNING_SYSTEM_PROMPT_FLAT_1D,
    PLANNING_SYSTEM_PROMPT_NO_MEMORY,
)


PLANNER_PROFILES = {
    "graph_2d": GRAPH_2D_PROMPT,
    "memora_full": MEMORA_FULL_PROMPT,
    "memora_episodic": MEMORA_EPISODIC_PROMPT,
}

DEFAULT_PLANNER_PROFILE = "memora_full"

__all__ = [
    "DEFAULT_PLANNER_PROFILE",
    "PLANNER_PROFILES",
    "PLANNING_FORCED_ANSWER_PROMPT",
    "PLANNING_SYSTEM_PROMPT_FLAT_1D",
    "PLANNING_SYSTEM_PROMPT_NO_MEMORY",
]
