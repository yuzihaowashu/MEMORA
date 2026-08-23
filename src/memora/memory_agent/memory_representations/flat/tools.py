"""Read-time tools for the Flat-1D chronological-text baseline."""

from typing import Any, Dict, List

from memora.memory_agent.tools.tool_interface import TypedMemoryTools


class FlatMemoryTools(TypedMemoryTools):
    """Expose a single search interface over chronological memory records.

    Released Flat-1D artifacts use the common participant-memory envelope so
    that context selection and temporal filtering remain identical across
    controlled conditions. Their records occupy one chronological activity
    stream; the agent receives only the shared ``search`` tool.
    """

    def get_tools_definition(
        self,
        allow_category: bool = False,
    ) -> List[Dict[str, Any]]:
        del allow_category
        tools = super().get_tools_definition(allow_category=False)
        return [
            tool
            for tool in tools
            if tool.get("function", {}).get("name") == "search"
        ]
