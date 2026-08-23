"""Semantic-evidence aggregation helpers used by ``TypedMemoryTools``."""

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

def count_object_uses(
    memory_tools,
    object_query: str,
    context_query: str = "",
    top_k_examples: int = 4,
) -> Dict[str, Any]:
    """
    Count activity mentions of an object, optionally conditioned on context.

    This supports habit and preference questions. A context query such as
    "preparing food" or "cleaning" prevents irrelevant object mentions from
    dominating the count.
    """
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}
    top_k_examples = min(max(top_k_examples or 4, 1), 4)

    obj_terms = memory_tools._query_terms(object_query)
    ctx_terms = memory_tools._query_terms(context_query)
    matched = []
    action_counter: Counter = Counter()
    video_counter: Counter = Counter()

    for activity in memory_tools.current_memory.get("activity_log", []):
        text = memory_tools._semantic_activity_text(activity)
        obj_hit = object_query.lower() in text or all(t in text for t in obj_terms)
        if not obj_hit:
            continue
        ctx_hit = True
        if ctx_terms:
            phrase_hit = bool(context_query.lower() and context_query.lower() in text)
            term_hit, _ = memory_tools._matches_terms(text, ctx_terms)
            ctx_hit = phrase_hit or term_hit
        if not ctx_hit:
            continue

        matched.append(activity)
        src = activity.get("source_video", memory_tools.current_video_id or "current_video")
        video_counter[src] += 1
        for step in activity.get("action_breakdown", []):
            if not isinstance(step, dict):
                continue
            step_text = " ".join(str(step.get(k, "")) for k in ("action", "object")).lower()
            if object_query.lower() in step_text or any(t in step_text for t in obj_terms):
                action = step.get("action")
                if action:
                    action_counter[action] += 1

    examples = [memory_tools._compact_semantic_activity(a) for a in matched[:top_k_examples]]
    return {
        "object_query": object_query,
        "context_query": context_query,
        "count": len(matched),
        "videos": dict(video_counter.most_common()),
        "common_actions": dict(action_counter.most_common(10)),
        "examples": examples,
    }

def matches_terms(memory_tools, text: str, terms: List[str]) -> Tuple[bool, int]:
    """Return whether enough query terms match a text, plus the hit count."""
    if not terms:
        return False, 0
    hits = sum(1 for term in terms if term in text)
    if len(terms) == 1:
        required = 1
    elif len(terms) <= 3:
        required = 2
    else:
        required = min(4, max(3, len(terms) // 2))
    return hits >= required, hits

def transition_next_text(
    memory_tools,
    activities_by_video: Dict[str, List[Dict[str, Any]]],
    video_id: str,
    activity_index: int,
    activity: Dict[str, Any],
) -> str:
    """Get the best available text describing what happens after an activity."""
    following = activity.get("following_action", "")
    if following:
        return str(following)
    video_activities = activities_by_video.get(video_id, [])
    if activity_index + 1 < len(video_activities):
        return str(video_activities[activity_index + 1].get("summary", ""))
    return ""

def candidate_support(memory_tools, text: str, candidate: str) -> int:
    terms = memory_tools._query_terms(candidate)
    if not terms:
        return 0
    return sum(1 for term in terms if term in text.lower())

def normalize_semantic_candidates(
    memory_tools,
    candidates: Any = None,
    candidate_a: str = "",
    candidate_b: str = "",
    choice_a: str = "",
    choice_b: str = "",
    choice_c: str = "",
    choice_d: str = "",
) -> List[Dict[str, str]]:
    """Normalize semantic answer choices while preserving labels."""
    normalized: List[Dict[str, str]] = []

    def add(label: str, text: Any) -> None:
        value = str(text or "").strip()
        if not value:
            return
        if value.lower() in {"e", "option e"}:
            return
        if "information is not available" in value.lower():
            return
        if any(item["text"].lower() == value.lower() for item in normalized):
            return
        normalized.append({"label": label, "text": value})

    if isinstance(candidates, dict):
        for label, text in candidates.items():
            add(str(label).upper()[:1] or f"C{len(normalized) + 1}", text)
    elif isinstance(candidates, list):
        for idx, item in enumerate(candidates):
            label = chr(ord("A") + idx)
            text = item
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("choice") or label).upper()[:1]
                text = item.get("text") or item.get("answer") or item.get("content") or ""
            add(label, text)
    elif isinstance(candidates, str):
        for idx, part in enumerate(re.split(r"\s*[|;]\s*", candidates)):
            add(chr(ord("A") + idx), part)

    for label, text in (
        ("A", choice_a),
        ("B", choice_b),
        ("C", choice_c),
        ("D", choice_d),
    ):
        add(label, text)

    if not normalized:
        add("A", candidate_a)
        add("B", candidate_b)

    return normalized

def pick_candidate_winner(memory_tools, candidate_counts: Counter) -> str:
    if not candidate_counts:
        return "insufficient_evidence"
    ranked = candidate_counts.most_common()
    top_label, top_count = ranked[0]
    if top_count <= 0:
        return "insufficient_evidence"
    if len(ranked) > 1 and ranked[1][1] == top_count:
        return "tie"
    return top_label

def score_all_candidates(
    memory_tools,
    candidates: List[Dict[str, str]],
    context_query: str,
) -> Dict[str, Any]:
    """Count support for each semantic candidate under the same context."""
    counts: Counter = Counter()
    activity_counts: Counter = Counter()
    pattern_counts: Counter = Counter()
    details: Dict[str, Any] = {}
    pattern_docs = memory_tools.search_patterns(context_query, top_k=10, _skip_synonym_expansion=True) if context_query else []
    if pattern_docs and len(pattern_docs) == 1 and "error" in pattern_docs[0]:
        pattern_docs = []
    for item in candidates:
        label = item["label"]
        text = item["text"]
        counted = memory_tools.count_object_uses(text, context_query=context_query, top_k_examples=2)
        count = counted.get("count", 0)
        pattern_support = 0
        pattern_examples = []
        for doc in pattern_docs:
            doc_text = " ".join(str(doc.get(k, "")) for k in ("description", "text")).lower()
            support = memory_tools._candidate_support(doc_text, text)
            if support <= 0:
                continue
            pattern_support += support
            if len(pattern_examples) < 2:
                pattern_examples.append({
                    "category": doc.get("category"),
                    "description": doc.get("description"),
                    "source": doc.get("source"),
                })
        activity_counts[label] = count
        pattern_counts[label] = pattern_support
        counts[label] = count + pattern_support
        details[label] = {
            "candidate": text,
            "count": count + pattern_support,
            "activity_count": count,
            "pattern_support": pattern_support,
            "common_actions": counted.get("common_actions", {}),
            "examples": counted.get("examples", [])[:2],
            "pattern_examples": pattern_examples,
        }

    winner = memory_tools._pick_candidate_winner(counts)
    return {
        "winner": winner,
        "counts": dict(counts),
        "activity_counts": dict(activity_counts),
        "pattern_support": dict(pattern_counts),
        "ranked_candidates": [
            {
                "label": label,
                "candidate": next((c["text"] for c in candidates if c["label"] == label), ""),
                "count": count,
            }
            for label, count in counts.most_common()
        ],
        "details": details,
        "_tip": "Prefer the highest-count candidate only when examples match the question context; use E/no-pattern if all counts are zero or tied without supporting patterns.",
    }

def find_action_transitions(
    memory_tools,
    trigger_query: str,
    candidate_a: str = "",
    candidate_b: str = "",
    top_k_examples: int = 4,
    candidates: Any = None,
) -> Dict[str, Any]:
    """
    Aggregate what usually happens after a trigger action.

    This is designed for questions like "after picking up X, what does
    the user do next?" where single semantic search snippets are noisy.
    """
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    activities = memory_tools.current_memory.get("activity_log", [])
    if not activities:
        return {"trigger_query": trigger_query, "matches": 0, "transitions": []}

    top_k_examples = min(max(top_k_examples or 4, 1), 4)
    trigger_lower = (trigger_query or "").lower().strip()
    trigger_terms = memory_tools._query_terms(trigger_query)

    activities_by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for activity in activities:
        src = str(activity.get("source_video", memory_tools.current_video_id or "current_video"))
        activities_by_video[src].append(activity)
    for video_activities in activities_by_video.values():
        video_activities.sort(key=lambda a: memory_tools._activity_time_bounds(a)[0])

    matches = []
    transition_counter: Counter = Counter()
    normalized_candidates = memory_tools._normalize_semantic_candidates(
        candidates=candidates,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )
    candidate_counts: Counter = Counter()

    for video_id, video_activities in activities_by_video.items():
        for idx, activity in enumerate(video_activities):
            text = memory_tools._semantic_activity_text(activity)
            phrase_hit = bool(trigger_lower and trigger_lower in text)
            term_hit, hit_count = memory_tools._matches_terms(text, trigger_terms)
            if not (phrase_hit or term_hit):
                continue

            next_text = memory_tools._transition_next_text(activities_by_video, video_id, idx, activity)
            if not next_text:
                continue

            compact_next = memory_tools._shorten_text(next_text, 120)
            transition_counter[compact_next] += 1

            next_lower = next_text.lower()
            support_by_choice = {}
            for item in normalized_candidates:
                support = memory_tools._candidate_support(next_lower, item["text"])
                support_by_choice[item["label"]] = support
                if support:
                    candidate_counts[item["label"]] += support

            matches.append({
                "source_video": video_id,
                "turn_id": activity.get("turn_id"),
                "time": activity.get("time_window", activity.get("time", {})),
                "trigger_activity": memory_tools._shorten_text(activity.get("summary", ""), 120),
                "next_action": compact_next,
                "match_score": (10 if phrase_hit else 0) + hit_count,
                "candidate_support": support_by_choice,
            })

    matches.sort(key=lambda item: item["match_score"], reverse=True)
    winner = None
    if normalized_candidates:
        winner = memory_tools._pick_candidate_winner(candidate_counts)

    return {
        "trigger_query": trigger_query,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "candidates": normalized_candidates,
        "matches": len(matches),
        "winner": winner,
        "candidate_counts": dict(candidate_counts),
        "common_next_actions": dict(transition_counter.most_common(5)),
        "transitions": matches[:top_k_examples],
        "_tip": "Use this for after/next/typically-do questions. If matches are sparse or winner is insufficient_evidence, prefer option E/no-pattern when available.",
    }

def compare_objects(
    memory_tools,
    query_a: str,
    query_b: str,
    context_query: str = "",
) -> Dict[str, Any]:
    """Compare two object candidates under the same context."""
    a = memory_tools.count_object_uses(query_a, context_query=context_query)
    b = memory_tools.count_object_uses(query_b, context_query=context_query)
    a_count = a.get("count", 0)
    b_count = b.get("count", 0)
    if a_count > b_count:
        winner = query_a
    elif b_count > a_count:
        winner = query_b
    else:
        winner = "tie"
    return {
        "query_a": query_a,
        "query_b": query_b,
        "context_query": context_query,
        "winner": winner,
        "counts": {query_a: a_count, query_b: b_count},
        "details": {
            query_a: {
                "count": a_count,
                "common_actions": a.get("common_actions", {}),
                "examples": a.get("examples", []),
            },
            query_b: {
                "count": b_count,
                "common_actions": b.get("common_actions", {}),
                "examples": b.get("examples", []),
            },
        },
        "_tip": "For preference/habit questions, prefer the higher count only when examples match the question context.",
    }

def get_semantic_evidence(
    memory_tools,
    query: str,
    candidate_a: str = "",
    candidate_b: str = "",
    context_query: str = "",
    trigger_query: str = "",
    candidates: Any = None,
    choice_a: str = "",
    choice_b: str = "",
    choice_c: str = "",
    choice_d: str = "",
) -> Dict[str, Any]:
    """
    Return typed evidence for semantic memory questions.

    This deliberately avoids broad episodic goal/activity dumps. It combines
    pattern search, transition aggregation, and candidate comparison into a
    compact semantic evidence packet.
    """
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    query = query or ""
    context = context_query or query
    normalized_candidates = memory_tools._normalize_semantic_candidates(
        candidates=candidates,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        choice_a=choice_a,
        choice_b=choice_b,
        choice_c=choice_c,
        choice_d=choice_d,
    )
    result: Dict[str, Any] = {
        "query": query,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "candidates": normalized_candidates,
        "context_query": context_query,
        "trigger_query": trigger_query,
        "patterns": [],
        "transition_evidence": None,
        "candidate_comparison": None,
        "all_candidate_comparison": None,
        "candidate_counts": {},
        "best_supported_choice": None,
        "_tip": (
            "Use this typed semantic evidence for habit/preference/routine questions. "
            "Prefer best_supported_choice, all_candidate_comparison, or transition_evidence when they match the answer choices; "
            "use patterns for strategy/routine questions."
        ),
    }

    patterns = memory_tools.search_patterns(query, top_k=5, _skip_synonym_expansion=True)
    if patterns and not (len(patterns) == 1 and "error" in patterns[0]):
        result["patterns"] = patterns[:5]

    transition_markers = ("after", "immediately", "next", "following")
    should_transition = bool(trigger_query) or any(marker in query.lower() for marker in transition_markers)
    if should_transition:
        result["transition_evidence"] = memory_tools.find_action_transitions(
            trigger_query or query,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            top_k_examples=4,
            candidates=normalized_candidates,
        )

    if len(normalized_candidates) > 2:
        result["all_candidate_comparison"] = memory_tools._score_all_candidates(
            normalized_candidates,
            context_query=context,
        )
    elif candidate_a and candidate_b:
        result["candidate_comparison"] = memory_tools.compare_objects(
            candidate_a,
            candidate_b,
            context_query=context,
        )
    elif candidate_a:
        result["candidate_counts"][candidate_a] = memory_tools.count_object_uses(
            candidate_a,
            context_query=context,
        )
    elif candidate_b:
        result["candidate_counts"][candidate_b] = memory_tools.count_object_uses(
            candidate_b,
            context_query=context,
        )

    transition_winner = None
    if result["transition_evidence"]:
        transition_winner = result["transition_evidence"].get("winner")
    all_winner = None
    if result["all_candidate_comparison"]:
        all_winner = result["all_candidate_comparison"].get("winner")
    if transition_winner and transition_winner not in {"tie", "insufficient_evidence"}:
        result["best_supported_choice"] = transition_winner
    elif all_winner and all_winner not in {"tie", "insufficient_evidence"}:
        result["best_supported_choice"] = all_winner

    result["evidence_summary"] = {
        "patterns_found": len(result["patterns"]),
        "has_transition_evidence": bool(result["transition_evidence"]),
        "has_candidate_comparison": bool(result["candidate_comparison"]),
        "has_all_candidate_comparison": bool(result["all_candidate_comparison"]),
        "candidate_counts_found": list(result["candidate_counts"].keys()),
        "best_supported_choice": result["best_supported_choice"],
    }
    return result
