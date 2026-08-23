"""Activity-memory retrieval helpers used by ``TypedMemoryTools``."""

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional

from memora.memory_agent.tools.embedding import get_for

def search_activities(
    memory_tools: Any,
    query: str,
    top_k: int = None,
    _skip_synonym_expansion: bool = False,
) -> List[Dict[str, Any]]:
    """Search Activity Memory for actions, events, and temporal procedure evidence."""
    if not _skip_synonym_expansion:
        return memory_tools._search_with_synonym_expansion(memory_tools.search_activities, query, top_k)

    top_k = min(top_k or memory_tools.DEFAULT_TOP_K, memory_tools.MAX_TOP_K)

    if not memory_tools.current_memory:
        return [{"error": "No memory context set."}]

    activity_log = memory_tools.current_memory.get("activity_log", [])
    if not activity_log:
        return []

    docs = []
    for act in activity_log:
        text_parts = [
            act.get("summary", ""),
            act.get("detailed_narrative", ""),
            act.get("high_level_goal", ""),
            act.get("local_sequence", ""),
            act.get("retrieval_text", ""),
            act.get("video_summary", ""),
        ]
        local_event = act.get("local_event", {})
        if isinstance(local_event, dict):
            text_parts.extend([
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

        for breakdown in act.get("action_breakdown", []):
            text_parts.append(breakdown.get("action", ""))
            text_parts.append(breakdown.get("object", ""))

        docs.append({
            "text": " ".join(str(p) for p in text_parts if p),
            "data": act,
        })

    model = get_for(memory_tools)
    if model:
        import numpy as np

        texts = [f"passage: {doc['text']}" for doc in docs]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_emb = model.encode(f"query: {query}", normalize_embeddings=True)

        similarities = embeddings @ query_emb
        indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in indices:
            sim_score = float(similarities[idx])
            if memory_tools.always_return_top_k or sim_score > memory_tools.similarity_threshold:
                act = docs[idx]["data"]
                episode = memory_tools._build_episode(act)
                episode["similarity"] = sim_score
                if memory_tools.include_tips and sim_score < memory_tools.similarity_threshold:
                    episode["_confidence"] = "low"
                results.append(episode)

        return results

    query_lower = query.lower()
    results = []
    for doc in docs:
        if query_lower in doc["text"].lower():
            results.append(memory_tools._build_episode(doc["data"]))
    return results[:top_k]


def activity_overlaps_window(
    memory_tools: Any,
    activity: Dict[str, Any],
    window_start: float,
    window_end: float,
) -> bool:
    """Return whether an activity overlaps a time window."""
    start, end = memory_tools._activity_time_bounds(activity)
    if start == 0.0 and end == 0.0:
        return False
    return start <= window_end and end >= window_start


def objects_in_time_window(
    memory_tools: Any,
    center_time: Optional[float] = None,
    window: float = 30.0,
    max_objects: int = 12,
) -> List[Dict[str, Any]]:
    """Return objects mentioned or observed near a time window."""
    if not memory_tools.current_memory:
        return []

    if center_time is None:
        center_time = memory_tools.time_threshold
    if center_time is None:
        return []

    window = max(float(window or 30.0), 1.0)
    window_start = max(0.0, float(center_time) - window)
    window_end = float(center_time) + window
    candidates: Dict[str, Dict[str, Any]] = {}

    for activity in memory_tools.current_memory.get("activity_log", []):
        if not memory_tools._activity_overlaps_window(activity, window_start, window_end):
            continue
        act_summary = memory_tools._shorten_text(activity.get("summary", ""), 100)
        for step in activity.get("action_breakdown", []):
            if not isinstance(step, dict):
                continue
            obj = memory_tools._shorten_text(step.get("object", ""), 80)
            if not obj or obj.lower() in {"none", "null", "n/a"}:
                continue
            key = obj.lower()
            entry = candidates.setdefault(key, {
                "name": obj,
                "source": "activity_window",
                "mentions": 0,
                "example_actions": [],
            })
            entry["mentions"] += 1
            if len(entry["example_actions"]) < 2:
                entry["example_actions"].append({
                    "action": memory_tools._shorten_text(step.get("action", ""), 48),
                    "summary": act_summary,
                })

    for obj_id, obj_data in memory_tools.current_memory.get("object_registry", {}).items():
        times = []
        for field in ("last_seen_time", "last_seen", "first_seen_time", "first_seen"):
            value = obj_data.get(field)
            if isinstance(value, (int, float)):
                times.append(float(value))
        for hist in obj_data.get("state_history", []) or []:
            value = hist.get("time_seconds", hist.get("time"))
            if isinstance(value, (int, float)):
                times.append(float(value))
        if not times or not any(window_start <= t <= window_end for t in times):
            continue

        name = memory_tools._shorten_text(obj_data.get("name", obj_id), 80)
        key = name.lower() or str(obj_id).lower()
        state_val = obj_data.get("state", {})
        state = state_val.get("current_state", "") if isinstance(state_val, dict) else state_val
        location = obj_data.get("spatial_info", {}).get("location", "")
        entry = candidates.setdefault(key, {
            "name": name,
            "source": "object_registry_window",
            "mentions": 0,
            "example_actions": [],
        })
        entry.update({
            "object_id": obj_id,
            "state": memory_tools._shorten_text(state, 60),
            "location": memory_tools._shorten_text(location, 80),
        })

    ranked = sorted(
        candidates.values(),
        key=lambda x: (x.get("mentions", 0), bool(x.get("object_id"))),
        reverse=True,
    )
    return ranked[:max_objects]


def get_video_summary(
    memory_tools: Any,
    video_id: Optional[str] = None,
    max_activities: Optional[int] = None,
) -> Dict[str, Any]:
    """Return a video-level narrative assembled from activity summaries."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    activities = memory_tools.current_memory.get("activity_log", [])
    if not activities:
        return {"error": "No activities in memory"}

    if video_id is None and memory_tools.current_video_id:
        video_id = memory_tools.current_video_id
    expanded = memory_tools._has_expanded_activity_view()
    default_max = 6 if expanded else 3
    cap = 6 if expanded else 3
    max_activities = min(max(max_activities or default_max, 1), cap)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for activity in activities:
        src = activity.get("source_video", memory_tools.current_video_id or "current_video")
        if video_id and src != video_id:
            continue
        grouped[src].append(activity)

    if video_id and not grouped:
        available = sorted({
            a.get("source_video", memory_tools.current_video_id or "current_video")
            for a in activities
        })
        return {"error": f"No activities found for video_id '{video_id}'", "available_videos": available[:20]}

    result = {"video_id": video_id, "videos": {}}
    for src, acts in grouped.items():
        acts = sorted(acts, key=lambda a: memory_tools._activity_time_bounds(a)[0])
        if expanded and len(acts) > max_activities:
            if max_activities <= 1:
                selected = [acts[len(acts) // 2]]
            else:
                selected = [
                    acts[round(i * (len(acts) - 1) / (max_activities - 1))]
                    for i in range(max_activities)
                ]
        elif len(acts) <= max_activities:
            selected = acts
        else:
            mid_idx = len(acts) // 2
            selected = [acts[0], acts[mid_idx], acts[-1]]
        snippets = [memory_tools._compact_activity(a) for a in selected[:max_activities]]
        summaries = [s["summary"] for s in snippets if s.get("summary")]
        result["videos"][src] = {
            "num_activities": len(acts),
            "shown_activities": len(snippets),
            "overview": memory_tools._shorten_text(" ".join(summaries[:3]), 300),
            "chronological_activities": snippets,
            "_truncated": len(acts) > len(snippets),
        }
        if expanded:
            goal_views = []
            seen_goals = set()
            for act in acts:
                view = act.get("goal_activity_view")
                if not isinstance(view, dict):
                    continue
                goal = view.get("goal", "")
                goal_range = view.get("goal_time_range", {})
                key = (goal, str(goal_range))
                if key in seen_goals:
                    continue
                seen_goals.add(key)
                goal_views.append(memory_tools._compact_goal_activity_view(view, max_activities=4))
                if len(goal_views) >= 3:
                    break
            if goal_views:
                result["videos"][src]["goal_activity_views"] = goal_views
    return result


def get_video_activities(
    memory_tools: Any,
    video_id: Optional[str] = None,
    compact: bool = True,
    max_activities: Optional[int] = None,
) -> Dict[str, Any]:
    """Return the chronological activity stream for a single video."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    activities = memory_tools.current_memory.get("activity_log", [])
    if not activities:
        return {"error": "No activities in memory"}

    if video_id is None and memory_tools.current_video_id:
        video_id = memory_tools.current_video_id

    same_video: List[Dict[str, Any]] = []
    for activity in activities:
        src = str(activity.get("source_video", memory_tools.current_video_id or "")).upper()
        if video_id and src != str(video_id).upper():
            continue
        same_video.append(activity)

    if video_id and not same_video:
        available = sorted({
            str(a.get("source_video", memory_tools.current_video_id or "current_video"))
            for a in activities
        })
        return {
            "error": f"No activities found for video_id '{video_id}'",
            "available_videos": available[:20],
        }

    same_video = sorted(same_video, key=lambda a: memory_tools._activity_time_bounds(a)[0])
    total = len(same_video)
    cap = min(max(max_activities or 80, 1), 120)
    used = same_video[:cap]

    if compact:
        stream = []
        for a in used:
            start, end = memory_tools._activity_time_bounds(a)
            entry = {
                "turn_id": a.get("turn_id"),
                "t": [round(start, 1), round(end, 1)],
                "summary": memory_tools._shorten_text(a.get("summary", ""), 200),
            }
            narrative = a.get("detailed_narrative", "") or ""
            if narrative:
                entry["narrative"] = memory_tools._shorten_text(narrative, 350)
            stream.append(entry)
    else:
        stream = [memory_tools._compact_activity(a) for a in used]

    return {
        "video_id": video_id,
        "num_activities": total,
        "shown_activities": len(stream),
        "compact": compact,
        "activities": stream,
        "_truncated": total > len(stream),
    }


def get_local_narrative(
    memory_tools: Any,
    time_seconds: Optional[float] = None,
    window: float = 30.0,
    video_id: Optional[str] = None,
    max_activities: int = 5,
) -> Dict[str, Any]:
    """Return local activity narrative and nearby objects around a timestamp."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}
    if time_seconds is None:
        time_seconds = memory_tools.time_threshold
    if time_seconds is None:
        return {
            "error": "No time_seconds provided and no temporal context is set.",
            "_tip": "Use get_video_summary for broad video recall, or provide time_seconds for local narrative.",
        }

    if video_id is None and memory_tools.current_video_id:
        video_id = memory_tools.current_video_id
    window = max(float(window or 30.0), 1.0)
    max_activities = min(max(max_activities or 5, 1), 8)
    window_start = max(0.0, float(time_seconds) - window)
    window_end = float(time_seconds) + window

    activities = []
    for activity in memory_tools.current_memory.get("activity_log", []):
        src = str(activity.get("source_video", memory_tools.current_video_id or ""))
        if video_id and src and src.upper() != str(video_id).upper():
            continue
        if memory_tools._activity_overlaps_window(activity, window_start, window_end):
            activities.append(activity)

    activities = sorted(activities, key=lambda a: memory_tools._activity_time_bounds(a)[0])
    compact_activities = [memory_tools._compact_activity(a) for a in activities[:max_activities]]
    return {
        "video_id": video_id,
        "time_seconds": time_seconds,
        "window": {"start": window_start, "end": window_end},
        "activities": compact_activities,
        "candidates_in_window": memory_tools._objects_in_time_window(time_seconds, window=window, max_objects=12),
        "_tip": "Use candidates_in_window as faithful context when a queried object is not tracked by name.",
    }


def get_narrative_evidence(
    memory_tools: Any,
    query: str,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Search activity narratives directly for open-vocabulary objects/events."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}
    expanded = memory_tools._has_expanded_activity_view()
    default_top_k = 4 if expanded else 3
    cap = 4 if expanded else 3
    top_k = min(max(top_k or default_top_k, 1), cap)

    activities = memory_tools.current_memory.get("activity_log", [])
    if not activities:
        return {"query": query, "evidence": []}

    query_lower = query.lower().strip()
    query_terms = memory_tools._query_terms(query_lower)
    query_video_ids = set(memory_tools._query_video_ids(query))
    scored = []
    for activity in activities:
        src = str(activity.get("source_video", memory_tools.current_video_id or "")).upper()
        if query_video_ids and src not in query_video_ids:
            continue
        text = memory_tools._activity_text(activity)
        phrase_hit = bool(query_lower and query_lower in text)
        term_hits = sum(1 for t in query_terms if t in text)
        if phrase_hit or term_hits:
            video_bonus = 8 if query_video_ids and src in query_video_ids else 0
            score = video_bonus + (10 if phrase_hit else 0) + term_hits
            scored.append((score, activity, "exact_or_term"))

    if len(scored) < top_k:
        semantic_hits = memory_tools.search_activities(
            query,
            top_k=max(top_k * 2, 8),
            _skip_synonym_expansion=True,
        )
        for episode in semantic_hits:
            if not isinstance(episode, dict) or "error" in episode:
                continue
            src = str(episode.get("source_video", memory_tools.current_video_id or "")).upper()
            if query_video_ids and src not in query_video_ids:
                continue
            compact = {
                "source_video": episode.get("source_video", memory_tools.current_video_id),
                "turn_id": episode.get("turn_id"),
                "time": episode.get("time", {}),
                "summary": memory_tools._shorten_text(episode.get("summary", ""), 140),
                "detailed_narrative": memory_tools._shorten_text(episode.get("detailed_narrative", ""), 160),
                "action_breakdown": memory_tools._compact_action_breakdown(
                    episode.get("action_breakdown", []),
                    max_steps=2,
                ),
                "similarity": episode.get("similarity"),
            }
            if isinstance(episode.get("goal_activity_view"), dict):
                compact["goal_activity_view"] = memory_tools._compact_goal_activity_view(
                    episode["goal_activity_view"],
                    max_activities=4,
                    focus_turn_id=episode.get("turn_id"),
                    query=query,
                )
            scored.append((float(episode.get("similarity", 0)), compact, "semantic_activity"))

    evidence = []
    seen = set()
    for score, activity, match_type in sorted(scored, key=lambda x: x[0], reverse=True):
        compact = activity if "time_seconds" in activity else memory_tools._compact_activity(activity)
        key = (compact.get("source_video"), compact.get("turn_id"), compact.get("summary"))
        if key in seen:
            continue
        seen.add(key)
        compact = copy.deepcopy(compact)
        compact["match_type"] = match_type
        compact["score"] = score
        evidence.append(compact)
        if len(evidence) >= top_k:
            break

    return {
        "query": query,
        "evidence": evidence,
        "_tip": "Use these narrative snippets as fallback evidence when structured object lookup misses.",
    }
