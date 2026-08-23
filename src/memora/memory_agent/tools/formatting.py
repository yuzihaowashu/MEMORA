"""Formatting helpers for compact MEMORA retrieval outputs."""

import re
from typing import Any, Dict, List, Optional, Tuple

QUERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been", "being",
    "what", "when", "where", "how", "does", "do", "did", "they", "them",
    "it", "this", "that", "their", "person", "typically", "usually",
    "after", "before", "next", "immediately", "then",
}


def shorten_text(text: Any, limit: int = 220) -> str:
    """Collapse whitespace and truncate long text for compact tool output."""
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def activity_time_bounds(activity: Dict[str, Any]) -> Tuple[float, float]:
    """Extract numeric start/end seconds from an activity."""
    tw = activity.get("time_window", activity.get("time", {}))
    if isinstance(tw, dict):
        return float(tw.get("start", 0) or 0), float(tw.get("end", 0) or 0)
    if isinstance(tw, str):
        nums = re.findall(r"\d+(?:\.\d+)?", tw)
        if nums:
            start = float(nums[0])
            end = float(nums[1]) if len(nums) > 1 else start
            return start, end
    return 0.0, 0.0


def activity_text(activity: Dict[str, Any], current_video_id: str = "") -> str:
    """Build a searchable narrative string from one activity."""
    parts = [
        activity.get("source_video", current_video_id or ""),
        activity.get("summary", ""),
        activity.get("detailed_narrative", ""),
        activity.get("high_level_goal", ""),
        activity.get("video_summary", ""),
        activity.get("local_sequence", ""),
        activity.get("preceding_action", ""),
        activity.get("following_action", ""),
        activity.get("retrieval_text", ""),
    ]
    local_event = activity.get("local_event", {})
    if isinstance(local_event, dict):
        parts.extend([
            local_event.get("previous", ""),
            local_event.get("next", ""),
            local_event.get("sequence", ""),
            " ".join(str(o) for o in local_event.get("action_objects", [])),
            " ".join(
                str(o.get("name", ""))
                for o in local_event.get("nearby_objects", [])
                if isinstance(o, dict)
            ),
        ])
    for breakdown in activity.get("action_breakdown", []):
        if isinstance(breakdown, dict):
            parts.extend([
                breakdown.get("timestamp", ""),
                breakdown.get("action", ""),
                breakdown.get("object", ""),
                breakdown.get("hand", ""),
                breakdown.get("manner", ""),
                breakdown.get("direction", ""),
            ])
    return " ".join(str(p) for p in parts if p).lower()


def semantic_activity_text(activity: Dict[str, Any], current_video_id: str = "") -> str:
    """Build text for semantic aggregation without expanded retrieval duplication."""
    parts = [
        activity.get("source_video", current_video_id or ""),
        activity.get("summary", ""),
        activity.get("detailed_narrative", ""),
        activity.get("preceding_action", ""),
        activity.get("following_action", ""),
    ]
    for breakdown in activity.get("action_breakdown", []):
        if isinstance(breakdown, dict):
            parts.extend([
                breakdown.get("timestamp", ""),
                breakdown.get("action", ""),
                breakdown.get("object", ""),
                breakdown.get("hand", ""),
                breakdown.get("manner", ""),
                breakdown.get("direction", ""),
            ])
    return " ".join(str(p) for p in parts if p).lower()


def query_terms(query: str) -> List[str]:
    """Tokenize a query and keep terms useful for narrative matching."""
    terms = []
    for term in re.findall(r"[a-z0-9]+", (query or "").lower()):
        if len(term) <= 1 or term in QUERY_STOPWORDS:
            continue
        terms.append(term)
        if term.endswith("ing") and len(term) > 5:
            stem = term[:-3]
            terms.append(stem)
            if len(stem) > 2 and stem[-1] == stem[-2]:
                terms.append(stem[:-1])
        elif term.endswith("ed") and len(term) > 4:
            terms.append(term[:-2])
        elif term.endswith("s") and len(term) > 4:
            terms.append(term[:-1])
    seen = set()
    return [t for t in terms if not (t in seen or seen.add(t))]


def query_video_ids(query: str) -> List[str]:
    """Extract EPIC-style video IDs from a query, e.g. P01_104."""
    return [m.upper() for m in re.findall(r"\bP\d{2}_\d{3}\b", query or "", flags=re.IGNORECASE)]


def select_bounded_ordered_items(
    ordered: Any,
    max_activities: int,
    focus_turn_id: Any = None,
    query: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Select a small, chronological set of activities from a goal segment."""
    if not isinstance(ordered, list):
        return []
    items = [item for item in ordered if isinstance(item, dict)]
    if len(items) <= max_activities:
        return items

    selected: set[int] = set()

    if focus_turn_id is not None:
        for idx, item in enumerate(items):
            if item.get("turn_id") == focus_turn_id:
                selected.update(i for i in (idx - 1, idx, idx + 1) if 0 <= i < len(items))
                break

    terms = query_terms(query or "")
    if terms:
        scored = []
        for idx, item in enumerate(items):
            text = " ".join([
                str(item.get("summary", "")),
                str(item.get("detailed_narrative", "")),
                " ".join(str(o) for o in item.get("action_objects", [])),
            ]).lower()
            hits = sum(1 for term in terms if term in text)
            if hits:
                scored.append((hits, idx))
        for _, idx in sorted(scored, reverse=True):
            selected.add(idx)
            if len(selected) >= max_activities:
                break

    if len(selected) < max_activities:
        if max_activities <= 1:
            representative = [len(items) // 2]
        else:
            representative = [
                round(i * (len(items) - 1) / (max_activities - 1))
                for i in range(max_activities)
            ]
        for idx in representative:
            selected.add(int(idx))
            if len(selected) >= max_activities:
                break

    return [items[idx] for idx in sorted(selected)[:max_activities]]


def compact_action_breakdown(breakdown: Any, max_steps: int = 2) -> List[Dict[str, Any]]:
    """Keep only the fields needed for reasoning about action sequences."""
    if not isinstance(breakdown, list):
        return []
    compact_steps = []
    for step in breakdown[:max_steps]:
        if not isinstance(step, dict):
            continue
        compact_steps.append({
            "timestamp": shorten_text(step.get("timestamp", ""), 24),
            "action": shorten_text(step.get("action", ""), 40),
            "object": shorten_text(step.get("object", ""), 48),
            "hand": shorten_text(step.get("hand", ""), 24),
        })
    return compact_steps


def compact_goal_activity_view(
    view: Dict[str, Any],
    max_activities: int = 5,
    focus_turn_id: Any = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the goal/sub-goal/activity layer used by expanded memory outputs."""
    ordered = view.get("ordered_activity_log", [])
    if not isinstance(ordered, list):
        ordered = []
    selected = select_bounded_ordered_items(
        ordered,
        max_activities=max_activities,
        focus_turn_id=focus_turn_id,
        query=query,
    )
    return {
        "goal": shorten_text(view.get("goal", ""), 180),
        "sub_goal": shorten_text(view.get("sub_goal", ""), 140),
        "goal_time_range": view.get("goal_time_range", {}),
        "abstract_steps": view.get("abstract_steps", [])[:5],
        "matching_goal_steps": view.get("matching_goal_steps", [])[:3],
        "ordered_activity_log": [
            {
                "turn_id": item.get("turn_id"),
                "time_window": item.get("time_window", {}),
                "summary": shorten_text(item.get("summary", ""), 120),
                "detailed_narrative": shorten_text(item.get("detailed_narrative", ""), 180),
                "action_objects": item.get("action_objects", [])[:6],
            }
            for item in selected
        ],
        "_selection": "bounded_relevant_or_representative",
        "_truncated": len(ordered) > len(selected),
    }


def compact_activity(activity: Dict[str, Any], current_video_id: str = "") -> Dict[str, Any]:
    """Return a compact activity snippet suitable for tool output."""
    start, end = activity_time_bounds(activity)
    breakdown = activity.get("action_breakdown", [])
    expanded_steps = 4 if isinstance(activity.get("goal_activity_view"), dict) else 2
    result = {
        "source_video": activity.get("source_video", current_video_id),
        "turn_id": activity.get("turn_id"),
        "time": activity.get("time_window", activity.get("time", {})),
        "time_seconds": {"start": start, "end": end},
        "summary": shorten_text(activity.get("summary", ""), 140),
        "detailed_narrative": shorten_text(activity.get("detailed_narrative", ""), 160),
        "action_breakdown": compact_action_breakdown(breakdown, max_steps=expanded_steps),
        "local_event": {
            "previous": shorten_text(activity.get("local_event", {}).get("previous", ""), 80)
            if isinstance(activity.get("local_event"), dict) else "",
            "next": shorten_text(activity.get("local_event", {}).get("next", ""), 80)
            if isinstance(activity.get("local_event"), dict) else "",
            "action_objects": activity.get("local_event", {}).get("action_objects", [])[:6]
            if isinstance(activity.get("local_event"), dict) else [],
        },
    }
    if isinstance(activity.get("goal_activity_view"), dict):
        result["goal_activity_view"] = compact_goal_activity_view(
            activity["goal_activity_view"],
            max_activities=5,
            focus_turn_id=activity.get("turn_id"),
        )
        result["high_level_goal"] = shorten_text(activity.get("high_level_goal", ""), 180)
    return result


def compact_semantic_activity(activity: Dict[str, Any], current_video_id: str = "") -> Dict[str, Any]:
    """Compact support example for semantic aggregation without episodic goal views."""
    start, end = activity_time_bounds(activity)
    return {
        "source_video": activity.get("source_video", current_video_id),
        "turn_id": activity.get("turn_id"),
        "time": activity.get("time_window", activity.get("time", {})),
        "time_seconds": {"start": start, "end": end},
        "summary": shorten_text(activity.get("summary", ""), 130),
        "detailed_narrative": shorten_text(activity.get("detailed_narrative", ""), 160),
        "action_breakdown": compact_action_breakdown(
            activity.get("action_breakdown", []),
            max_steps=2,
        ),
        "high_level_goal": shorten_text(activity.get("high_level_goal", ""), 120),
    }


def episode_to_context_format(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an episode record into compact activity context format."""
    result = {
        "time": episode.get("time", ""),
        "summary": episode.get("summary", ""),
    }

    context = {}
    if episode.get("scene_objects"):
        context["objects_involved"] = episode["scene_objects"]
    if episode.get("environment"):
        context["environment"] = {"description": episode["environment"]}
    if episode.get("previous_action"):
        context["previous_action"] = episode["previous_action"]
    if episode.get("next_action"):
        context["next_action"] = episode["next_action"]

    if context:
        result["_context"] = context

    if episode.get("action_breakdown"):
        result["action_breakdown"] = compact_action_breakdown(
            episode["action_breakdown"],
            max_steps=4 if isinstance(episode.get("goal_activity_view"), dict) else 5,
        )
    if episode.get("detailed_narrative"):
        result["detailed_narrative"] = shorten_text(
            episode["detailed_narrative"],
            240 if isinstance(episode.get("goal_activity_view"), dict) else 360,
        )
    if episode.get("local_sequence"):
        result["local_sequence"] = shorten_text(episode["local_sequence"], 180)
    if isinstance(episode.get("local_event"), dict):
        local_event = episode["local_event"]
        result["local_event"] = {
            "previous": shorten_text(local_event.get("previous", ""), 90),
            "next": shorten_text(local_event.get("next", ""), 90),
            "sequence": shorten_text(local_event.get("sequence", ""), 180),
            "action_objects": local_event.get("action_objects", [])[:8],
            "nearby_objects": local_event.get("nearby_objects", [])[:6],
        }

    if isinstance(episode.get("goal_activity_view"), dict):
        result["goal_activity_view"] = compact_goal_activity_view(
            episode["goal_activity_view"],
            max_activities=5,
            focus_turn_id=episode.get("turn_id"),
        )

    for field in ("high_level_goal", "video_summary", "goal_turn_range", "similarity"):
        if field in episode:
            result[field] = episode[field]

    return result
