"""Read-time tools for MEMORA's four-store memory condition."""

from memora.memory_agent.tools.tool_interface import TypedMemoryTools


class MEMORATools(TypedMemoryTools):
    """Expose MEMORA's type-aware retrieval and planning tools."""
