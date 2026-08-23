"""Tools and prompts for the paper's controlled memory conditions."""

from typing import Any

from .flat import FLAT_1D_SYSTEM_PROMPT, FlatMemoryTools
from .graph import GRAPH_2D_SYSTEM_PROMPT, GraphMemoryTools
from .memora import (
    MEMORATools,
    TYPED_MEMORY_SYSTEM_PROMPT,
    TYPED_MEMORY_SYSTEM_PROMPT_WITH_CATEGORY,
)
from .shared import (
    FORCED_ANSWER_PROMPT,
    FORCED_ANSWER_PROMPT_SHORT_ANSWER,
    SHORT_ANSWER_GUIDANCE,
)


def create_memory_tools(
    memory_type: str,
    memory_file: str,
    **kwargs: Any,
) -> MEMORATools | FlatMemoryTools | GraphMemoryTools:
    """Create the tools exposed by a MEMORA-Bench memory condition."""
    tool_classes = {
        "memora": MEMORATools,
        "flat_1d": FlatMemoryTools,
        "graph_2d": GraphMemoryTools,
    }
    try:
        tool_class = tool_classes[memory_type]
    except KeyError as exc:
        supported = ", ".join(tool_classes)
        raise ValueError(
            f"Unknown memory_type {memory_type!r}; expected one of: {supported}"
        ) from exc
    return tool_class(memory_file, **kwargs)


__all__ = [
    "FLAT_1D_SYSTEM_PROMPT",
    "FORCED_ANSWER_PROMPT",
    "FORCED_ANSWER_PROMPT_SHORT_ANSWER",
    "GRAPH_2D_SYSTEM_PROMPT",
    "MEMORATools",
    "SHORT_ANSWER_GUIDANCE",
    "TYPED_MEMORY_SYSTEM_PROMPT",
    "TYPED_MEMORY_SYSTEM_PROMPT_WITH_CATEGORY",
    "FlatMemoryTools",
    "GraphMemoryTools",
    "create_memory_tools",
]
