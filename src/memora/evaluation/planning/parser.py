"""Parse a planner's final text response into an ordered action list."""

import re
from typing import List


MAX_PLAN_STEPS = 15


def _deduplicate_consecutive_steps(steps: List[str]) -> List[str]:
    """Remove consecutive exact repetitions emitted by a planner."""
    if not steps:
        return steps
    output = [steps[0]]
    for step in steps[1:]:
        if step.strip().lower() != output[-1].strip().lower():
            output.append(step)
    return output


def _strip_tool_markup(text: str) -> str:
    """Remove tool-call markup accidentally repeated in the final response."""
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
    text = re.sub(r"<tool_response>[\s\S]*?</tool_response>", "", text)
    text = re.sub(r'\{"query"\s*:[\s\S]*?\}\s*\}(?:\s*\})?', "", text)
    return text.strip()


def extract_plan_from_response(response_text: str) -> List[str]:
    """Extract numbered or bulleted plan steps from a planner response.

    The parser accepts an explicit ``Plan:`` block, a standalone numbered list,
    or a bullet list. It removes consecutive exact duplicates and limits the
    result to ``MAX_PLAN_STEPS`` so malformed generations cannot grow without
    bound. It does not score or judge plan quality.
    """
    if not response_text:
        return []

    cleaned = _strip_tool_markup(response_text)
    if not cleaned:
        return []

    raw_steps: List[str] = []
    plan_match = re.search(
        r"Plan\s*:\s*\n((?:\s*\d+[\.\)]\s*.+\n?)+)",
        cleaned,
        re.IGNORECASE,
    )
    if plan_match:
        raw_steps = re.findall(r"\d+[\.\)]\s*(.+)", plan_match.group(1))
        raw_steps = [step.strip() for step in raw_steps if step.strip()]

    if not raw_steps:
        numbered = re.findall(r"^\s*\d+[\.\)]\s+(.+)$", cleaned, re.MULTILINE)
        if len(numbered) >= 2:
            raw_steps = [step.strip() for step in numbered if step.strip()]

    if not raw_steps:
        bullets = re.findall(r"^\s*[-*]\s+(.+)$", cleaned, re.MULTILINE)
        if len(bullets) >= 2:
            raw_steps = [step.strip() for step in bullets if step.strip()]

    if not raw_steps:
        return []

    return _deduplicate_consecutive_steps(raw_steps)[:MAX_PLAN_STEPS]
