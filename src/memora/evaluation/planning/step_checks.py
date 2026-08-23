#!/usr/bin/env python3
"""Deterministic action, object, and location checks for generated plan steps."""

import re
from typing import List


ACTION_VERBS = {
    "pick up", "put down", "open", "close", "wash", "pour", "cut", "turn on",
    "turn off", "place", "move", "take", "grab", "rinse", "dry", "wipe",
    "slice", "chop", "peel", "stir", "mix", "squeeze", "fill", "empty",
    "flip", "scoop", "spread", "throw", "shake", "scrub", "adjust",
    "pull", "push", "set", "remove", "add", "press", "lift", "lower",
    "fold", "unfold", "heat", "boil", "fry", "bake", "toast", "grate",
    "crack", "whisk", "measure", "transfer", "serve", "bring", "carry",
    "reach", "reach for", "grasp", "hold", "grip", "release", "position",
    "continue", "approach", "locate", "identify", "prepare", "ensure",
    "gather", "collect", "arrange", "toss", "dispose", "discard",
    "rotate", "check", "inspect", "retrieve", "return", "stack",
    "pick_up", "reach_for", "picks up", "places", "holding",
    "scrubbing", "rinsing", "wiping", "peeling", "stirring", "cutting",
}

LOCATION_PREPOSITIONS = {
    "on", "in", "from", "near", "next to", "under", "above", "inside",
    "at", "behind", "beside", "between", "onto", "into", "out of",
    "toward", "towards", "across",
}

LOCATION_NOUNS = {
    "counter", "countertop", "cabinet", "sink", "shelf", "drawer",
    "fridge", "refrigerator", "stove", "oven", "table", "rack",
    "drying rack", "dish rack", "worktop", "pan", "pot", "board",
    "cutting board", "chopping board", "microwave", "dishwasher",
    "bin", "trash", "garbage", "hob", "burner", "plate rack",
    "sink area", "sink_area", "counter area", "countertop_area",
    "left zone", "right zone", "center zone", "table_area",
    "drying_rack_area", "left_zone", "right_zone", "center_zone",
    "colander", "bowl", "plate", "tap", "faucet",
}

_ACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(v) for v in sorted(ACTION_VERBS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_LOCATION_PREPOSITION_PATTERN = re.compile(
    r"\b("
    + "|".join(re.escape(p) for p in sorted(LOCATION_PREPOSITIONS, key=len, reverse=True))
    + r")\b",
    re.IGNORECASE,
)
_LOCATION_NOUN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in sorted(LOCATION_NOUNS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_PRONOUN_ONLY_PATTERN = re.compile(
    r"^(it|the thing|that|this|them|those|these|something|stuff)$",
    re.IGNORECASE,
)
_SPECIFIC_OBJECT_PATTERN = re.compile(
    r"\b(the\s+)?(red|blue|green|white|black|large|small|big|little|metal|wooden|glass|plastic|ceramic|left|right|top|bottom|first|second)\s+\w+",
    re.IGNORECASE,
)


def _step_has_action(step: str) -> bool:
    return bool(_ACTION_PATTERN.search(step))


def _step_has_object(step: str) -> bool:
    tokens = re.findall(r"\b\w+(?:\s+\w+)?\b", step.lower())
    noun_candidates = [
        token
        for token in tokens
        if not _ACTION_PATTERN.match(token)
        and not _LOCATION_PREPOSITION_PATTERN.match(token)
        and token not in {
            "the", "a", "an", "and", "or", "then", "to", "with", "for", "is", "are"
        }
    ]
    return any(len(candidate) > 1 and not _PRONOUN_ONLY_PATTERN.match(candidate)
               for candidate in noun_candidates)


def _step_has_location(step: str) -> bool:
    return bool(
        _LOCATION_NOUN_PATTERN.search(step)
        or _LOCATION_PREPOSITION_PATTERN.search(step)
    )


def _step_is_specific(step: str) -> bool:
    return bool(_SPECIFIC_OBJECT_PATTERN.search(step))


class PlanStepChecks:
    """Predicates used by the paper's rule-based OrderExec calculation."""

    @staticmethod
    def compute_executability(plan_steps: List[str]) -> dict:
        """Return action/object/location checks for each generated plan step."""
        if not plan_steps:
            return {
                "executability_rate": 0.0,
                "avg_specificity": 0.0,
                "step_scores": [],
                "num_steps": 0,
            }

        step_scores = []
        fully_executable = 0
        for step in plan_steps:
            has_action = _step_has_action(step)
            has_object = _step_has_object(step)
            has_location = _step_has_location(step)
            score = int(has_action) + int(has_object) + int(has_location)
            step_scores.append({
                "step": step,
                "has_action": has_action,
                "has_object": has_object,
                "has_location": has_location,
                "is_specific": _step_is_specific(step),
                "score": score,
            })
            fully_executable += int(score == 3)

        count = len(plan_steps)
        return {
            "executability_rate": round(fully_executable / count, 4),
            "avg_specificity": round(
                sum(item["score"] for item in step_scores) / (3 * count), 4
            ),
            "specificity_score_mean": round(
                sum(item["score"] for item in step_scores) / count, 4
            ),
            "step_scores": step_scores,
            "num_steps": count,
            "num_fully_executable": fully_executable,
            "num_specific": sum(1 for item in step_scores if item["is_specific"]),
        }
