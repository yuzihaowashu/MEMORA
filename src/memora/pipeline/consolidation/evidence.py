"""Compile compact cross-episode evidence from Activity and Entity Memory."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def _short(value: Any, words: int = 24) -> str:
    tokens = str(value or "").replace("\n", " ").split()
    return " ".join(tokens[:words])


def _activity_text(activity: Dict[str, Any]) -> str:
    return _short(
        activity.get("summary")
        or activity.get("detailed_narrative")
        or activity.get("action")
        or "",
        30,
    )


def _activity_objects(activity: Dict[str, Any]) -> List[str]:
    objects = []
    seen = set()
    for step in activity.get("action_breakdown", []) or []:
        if not isinstance(step, dict):
            continue
        value = str(step.get("object") or "").strip()
        key = value.lower()
        if value and key not in {"none", "unknown", "n/a"} and key not in seen:
            seen.add(key)
            objects.append(value)
    return objects


def select_balanced_activities(
    activities: Iterable[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Round-robin episodes so a bounded prompt represents the full history."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        groups[str(activity.get("source_video") or "unknown_episode")].append(activity)
    for values in groups.values():
        values.sort(key=lambda item: float((item.get("time_window") or {}).get("start", 0) or 0))

    selected: List[Dict[str, Any]] = []
    episode_ids = sorted(groups)
    offset = 0
    while len(selected) < limit:
        added = False
        for episode_id in episode_ids:
            values = groups[episode_id]
            if offset < len(values):
                selected.append(values[offset])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def compile_cross_episode_evidence(
    activities: Iterable[Dict[str, Any]],
    max_records: int = 120,
) -> Dict[str, List[Dict[str, Any]]]:
    """Summarize repeated objects, contexts, and transitions across episodes."""
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for activity in activities:
        if isinstance(activity, dict):
            by_episode[str(activity.get("source_video") or "unknown_episode")].append(activity)
    for values in by_episode.values():
        values.sort(key=lambda item: float((item.get("time_window") or {}).get("start", 0) or 0))

    object_counts: Counter[str] = Counter()
    object_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    context_counts: Counter[str] = Counter()
    transition_counts: Counter[tuple[str, str]] = Counter()
    transition_examples: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for episode_id, episode in by_episode.items():
        descriptions = [_activity_text(activity) for activity in episode]
        for index, activity in enumerate(episode):
            description = descriptions[index]
            turn_id = activity.get("turn_id")
            example = {
                "source_video": episode_id,
                "turn_id": turn_id,
                "time_window": activity.get("time_window", {}),
                "summary": description,
            }
            for obj in _activity_objects(activity):
                key = obj.lower()
                object_counts[key] += 1
                if len(object_examples[key]) < 4:
                    object_examples[key].append(example)
                if description:
                    context_counts[f"{obj}: {description}"] += 1
            if index + 1 < len(episode) and description and descriptions[index + 1]:
                pair = (description, descriptions[index + 1])
                transition_counts[pair] += 1
                if len(transition_examples[pair]) < 4:
                    transition_examples[pair].append(example)

    frequent_objects = [
        {"object": name, "count": count, "examples": object_examples[name]}
        for name, count in object_counts.most_common(max_records)
    ]
    common_transitions = [
        {
            "trigger": trigger,
            "next_action": next_action,
            "count": count,
            "supporting_episodes": transition_examples[(trigger, next_action)],
        }
        for (trigger, next_action), count in transition_counts.most_common(max_records)
    ]
    object_contexts = [
        {"object_context": context, "count": count}
        for context, count in context_counts.most_common(max_records)
    ]
    return {
        "frequent_objects": frequent_objects,
        "common_transitions": common_transitions,
        "object_contexts": object_contexts,
    }


def compile_reusable_procedure_evidence(
    activities: Iterable[Dict[str, Any]],
    cross_episode_evidence: Dict[str, List[Dict[str, Any]]],
    max_records: int = 200,
) -> Dict[str, List[Dict[str, Any]]]:
    """Expose transitions and within-activity object handling as procedure evidence."""
    object_events: Counter[tuple[str, str, str, str]] = Counter()
    examples: Dict[tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        steps = [step for step in (activity.get("action_breakdown") or []) if isinstance(step, dict)]
        for first, second in zip(steps, steps[1:]):
            key = (
                _short(first.get("action"), 8),
                _short(first.get("object"), 8),
                _short(second.get("action"), 8),
                _short(second.get("object"), 8),
            )
            if not key[0] or not key[2]:
                continue
            object_events[key] += 1
            if len(examples[key]) < 4:
                examples[key].append({
                    "source_video": activity.get("source_video"),
                    "turn_id": activity.get("turn_id"),
                    "time_window": activity.get("time_window", {}),
                    "summary": _activity_text(activity),
                })

    atomic_transitions = [
        {
            "trigger": item["trigger"],
            "outcome": item["next_action"],
            "count": item["count"],
            "supporting_episodes": item.get("supporting_episodes", []),
        }
        for item in cross_episode_evidence.get("common_transitions", [])
    ]
    object_handling_events = [
        {
            "trigger_action": key[0],
            "trigger_object": key[1],
            "outcome_action": key[2],
            "outcome_object": key[3],
            "count": count,
            "supporting_episodes": examples[key],
        }
        for key, count in object_events.most_common(max_records)
    ]
    return {
        "atomic_transitions": atomic_transitions,
        "object_handling_events": object_handling_events,
        "procedure_templates": [],
    }
