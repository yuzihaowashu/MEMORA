"""Shared configuration for OpenAI-compatible pipeline APIs."""

import os
from typing import Optional, Tuple


DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def resolve_api_credentials(
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    default_base: Optional[str] = DASHSCOPE_API_BASE,
) -> Tuple[Optional[str], str]:
    """Resolve an OpenAI-compatible endpoint and its matching credentials.

    ``default_base=None`` preserves the standard OpenAI endpoint unless the
    environment contains only DashScope credentials. Formation callers retain
    the DashScope-compatible default used by the released API workflow.
    """
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get(
        "DASHSCOPE_API"
    )
    openai_key = os.environ.get("OPENAI_API_KEY")
    resolved_base = (
        api_base
        or os.environ.get("DASHSCOPE_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or (
            DASHSCOPE_API_BASE
            if default_base is None and dashscope_key and not openai_key
            else None
        )
        or default_base
    )
    if api_key:
        return resolved_base, api_key

    if resolved_base and "dashscope.aliyuncs.com" in resolved_base.lower():
        resolved_key = dashscope_key or "EMPTY"
    else:
        resolved_key = openai_key or "EMPTY"
    return resolved_base, resolved_key
