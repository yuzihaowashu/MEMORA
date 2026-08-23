"""Answer parsing for EAM-QA."""

import re


def extract_multiple_choice_answer(response: str) -> str:
    """Extract an EAM-QA answer letter from a model response."""
    match = re.search(
        r"\*?\*?Answer:\s*\*?\*?\s*([A-E])[\)\s\*]",
        response,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\)", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-E])\b", response)
    if match:
        return match.group(1).upper()
    return ""
