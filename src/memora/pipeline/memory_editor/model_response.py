"""Parse structured edit operations from language-model responses."""

import json
import re
from typing import Any, Dict, Optional


def parse_json_response(content: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from a direct or Markdown-fenced response."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    patterns = [
        r"```json\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
        r'\{[\s\S]*"object_operations"[\s\S]*\}',
        r'\{[\s\S]*"memory"[\s\S]*\}',
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        try:
            payload = match.group(1) if "```" in pattern else match.group(0)
            return json.loads(payload)
        except (json.JSONDecodeError, IndexError):
            continue
    return None
