"""Entity-memory retrieval helpers used by ``TypedMemoryTools``."""

import logging
from typing import Any, Dict, List

from memora.memory_agent.tools.embedding import get_for
from memora.memory_agent.tools.stores import environment

logger = logging.getLogger(__name__)


def search_objects(
    memory_tools: Any,
    query: str,
    top_k: int = None,
    _skip_synonym_expansion: bool = False,
) -> List[Dict[str, Any]]:
    """Search objects in the entity memory."""
    if not _skip_synonym_expansion:
        return memory_tools._search_with_synonym_expansion(memory_tools.search_objects, query, top_k)

    top_k = min(top_k or memory_tools.DEFAULT_TOP_K, memory_tools.MAX_TOP_K)

    if not memory_tools.current_memory:
        return [{"error": "No memory context set. Call set_temporal_context first."}]

    object_registry = memory_tools.current_memory.get("object_registry", {})
    if not object_registry:
        return []

    docs = []
    for obj_id, obj_data in object_registry.items():
        if not isinstance(obj_data, dict):
            logger.warning("Skipping non-dict object data for %s: %s", obj_id, type(obj_data))
            continue

        text_parts = [
            obj_data.get("name", ""),
            obj_id,
        ]

        vp = obj_data.get("visual_properties", {})
        if not isinstance(vp, dict):
            vp = {}
        text_parts.extend([
            vp.get("color", ""),
            vp.get("material", ""),
            vp.get("size", ""),
            vp.get("condition", ""),
        ])

        sp = obj_data.get("spatial_info", {})
        if not isinstance(sp, dict):
            sp = {}
        text_parts.extend([
            sp.get("location", ""),
            sp.get("zone", ""),
            sp.get("relative_to", ""),
        ])

        state = obj_data.get("state", {})
        if isinstance(state, dict):
            text_parts.append(state.get("current_state", ""))
        elif isinstance(state, str):
            text_parts.append(state)

        docs.append({
            "object_id": obj_id,
            "text": " ".join(str(part) for part in text_parts if part),
            "data": obj_data,
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
                doc = docs[idx]
                data = doc["data"]
                result = {
                    "object_id": doc["object_id"],
                    "name": data.get("name", ""),
                    "visual_properties": data.get("visual_properties", {}),
                    "spatial_info": data.get("spatial_info", {}),
                    "state": data.get("state", {}),
                    "similarity": sim_score,
                }
                first = data.get("first_seen_time") or data.get("first_seen")
                last = data.get("last_seen_time") or data.get("last_seen")
                if first is not None or last is not None:
                    result["time_seen_seconds"] = {"first_seen": first, "last_seen": last}
                if memory_tools.include_tips and sim_score < memory_tools.similarity_threshold:
                    result["_confidence"] = "low"
                results.append(result)
        return results

    query_lower = query.lower()
    results = []
    for doc in docs:
        if query_lower in doc["text"].lower():
            data = doc["data"]
            result = {
                "object_id": doc["object_id"],
                "name": data.get("name", ""),
                "visual_properties": data.get("visual_properties", {}),
                "spatial_info": data.get("spatial_info", {}),
                "state": data.get("state", {}),
            }
            first = data.get("first_seen_time") or data.get("first_seen")
            last = data.get("last_seen_time") or data.get("last_seen")
            if first is not None or last is not None:
                result["time_seen_seconds"] = {"first_seen": first, "last_seen": last}
            results.append(result)
    return results[:top_k]


def search_objects_by_name_only(
    memory_tools: Any,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Search objects by name and object id only."""
    if not memory_tools.current_memory:
        return []

    object_registry = memory_tools.current_memory.get("object_registry", {})
    if not object_registry:
        return []

    docs = []
    for obj_id, obj_data in object_registry.items():
        if not isinstance(obj_data, dict):
            continue
        name = obj_data.get("name", "")
        searchable_text = f"{name} {obj_id}".strip()
        if searchable_text:
            docs.append({"object_id": obj_id, "text": searchable_text, "data": obj_data})

    if not docs:
        return []

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
            doc = docs[idx]
            results.append({
                "object_id": doc["object_id"],
                "name": doc["data"].get("name", ""),
                "similarity": sim_score,
                "data": doc["data"],
            })
        return results

    query_lower = query.lower()
    results = []
    for doc in docs:
        if query_lower in doc["text"].lower():
            results.append({
                "object_id": doc["object_id"],
                "name": doc["data"].get("name", ""),
                "similarity": 1.0,
                "data": doc["data"],
            })
    return results[:top_k]


def get_state_at_time(memory_tools: Any, time_seconds: float) -> Dict[str, Any]:
    """Get a complete state snapshot at a specific time."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    result = {
        "time": time_seconds,
        "visible_objects": [],
        "environment": None,
        "current_activity": None,
    }

    for obj_id, obj_data in memory_tools.current_memory.get("object_registry", {}).items():
        last_seen = obj_data.get("last_seen_time", obj_data.get("last_seen", float("inf")))
        first_seen = obj_data.get("first_seen_time", obj_data.get("first_seen", 0))
        if first_seen <= time_seconds <= last_seen + 60:
            state_at_time = memory_tools._find_state_at_time(obj_data, time_seconds)
            state_val = obj_data.get("state", {})
            fallback_state = (
                state_val.get("current_state", "")
                if isinstance(state_val, dict)
                else state_val
            )
            result["visible_objects"].append({
                "object_id": obj_id,
                "name": obj_data.get("name", ""),
                "location": state_at_time.get(
                    "location",
                    obj_data.get("spatial_info", {}).get("location", ""),
                ),
                "state": state_at_time.get("state", fallback_state),
                "_from_history": state_at_time.get("_from_history", False),
            })

    for env in memory_tools.current_memory.get("environment_log", []):
        first_seen = env.get("first_seen", 0)
        last_seen = env.get("last_seen", float("inf"))
        if first_seen <= time_seconds <= last_seen:
            state = env.get("current_state", env)
            result["environment"] = {
                "location_id": env.get("location_id", ""),
                "layout_description": state.get("layout_description", ""),
                "features": state.get("features", []),
            }
            break

    for activity in memory_tools.current_memory.get("activity_log", []):
        time_window = activity.get("time_window", activity.get("time", {}))
        if isinstance(time_window, dict):
            start = time_window.get("start", 0)
            end = time_window.get("end", 0)
        else:
            start = 0
            end = 0

        if start <= time_seconds <= end:
            result["current_activity"] = {
                "time": f"{start}-{end}s",
                "summary": activity.get("summary", ""),
                "action_breakdown": activity.get("action_breakdown", []),
            }
            break

    return result

def find_state_at_time(obj_data: Dict[str, Any], time_seconds: float) -> Dict[str, Any]:
    """Find an object's state at a specific time using state_history."""
    state_history = obj_data.get("state_history", [])

    if not state_history:
        state_val = obj_data.get("state", {})
        return {
            "state": state_val.get("current_state", "") if isinstance(state_val, dict) else state_val,
            "location": obj_data.get("spatial_info", {}).get("location", ""),
            "_from_history": False,
        }

    best_match = None
    for entry in state_history:
        entry_time = entry.get("time_seconds", entry.get("time", 0))
        if entry_time <= time_seconds:
            best_match = entry
        else:
            break

    if best_match:
        return {
            "state": best_match.get("state", ""),
            "location": best_match.get("location", ""),
            "_from_history": True,
            "_history_time": best_match.get("time_seconds", best_match.get("time")),
        }

    first_entry = state_history[0]
    return {
        "state": first_entry.get("state", ""),
        "location": first_entry.get("location", ""),
        "_from_history": True,
        "_note": "Using earliest known state",
    }


def get_object_history(memory_tools: Any, object_query: str) -> Dict[str, Any]:
    """Get the history of an object's states and locations."""
    if not memory_tools.current_memory:
        return {"error": "No memory context set."}

    object_registry = memory_tools.current_memory.get("object_registry", {})
    if not object_registry:
        local_narrative = memory_tools.get_local_narrative() if memory_tools.time_threshold is not None else {}
        return {
            "error": "No objects in memory",
            "match_type": "miss",
            "local_narrative": local_narrative.get("activities", []),
            "candidates_in_window": local_narrative.get("candidates_in_window", []),
            "_tip": "Structured objects are unavailable; use local_narrative and candidates_in_window as faithful episode context.",
        }

    query_lower = object_query.lower()
    matched_obj = None
    matched_id = None
    match_type = None

    for obj_id, obj_data in object_registry.items():
        name = obj_data.get("name", "").lower()
        if query_lower == obj_id.lower() or query_lower == name:
            matched_obj = obj_data
            matched_id = obj_id
            match_type = "exact"
            break
        if query_lower in obj_id.lower() or query_lower in name:
            matched_obj = obj_data
            matched_id = obj_id
            match_type = "substring"
            break

    if not matched_obj:
        expanded_queries = memory_tools._expand_query_with_synonyms(object_query)
        for expanded in expanded_queries[1:]:
            expanded_lower = expanded.lower()
            for obj_id, obj_data in object_registry.items():
                name = obj_data.get("name", "").lower()
                if expanded_lower in obj_id.lower() or expanded_lower in name:
                    matched_obj = obj_data
                    matched_id = obj_id
                    match_type = f"synonym ({expanded})"
                    break
            if matched_obj:
                break

    if not matched_obj:
        search_results = memory_tools._search_objects_by_name_only(object_query, top_k=3)
        if search_results:
            best_match = search_results[0]
            matched_id = best_match.get("object_id")
            matched_obj = best_match.get("data") or object_registry.get(matched_id)
            similarity = best_match.get("similarity", 0)
            if matched_obj and similarity > 0.85:
                match_type = f"semantic_name_only (similarity={similarity:.2f})"
                logger.info(
                    "Name-only semantic match: %r -> %r (sim=%.2f)",
                    object_query,
                    best_match.get("name"),
                    similarity,
                )
            else:
                logger.info(
                    "Rejected semantic match: %r -> %r (sim=%.2f < 0.85)",
                    object_query,
                    best_match.get("name"),
                    similarity,
                )
                matched_obj = None
                matched_id = None

    if not matched_obj:
        available_objects = [
            f"{oid}: {data.get('name', '')}"
            for oid, data in list(object_registry.items())[:10]
        ]
        narrative_evidence = memory_tools.get_narrative_evidence(object_query, top_k=3)
        local_narrative = memory_tools.get_local_narrative() if memory_tools.time_threshold is not None else {}
        return {
            "error": f"Object '{object_query}' not found in memory",
            "match_type": "miss",
            "suggestion": "Try a different object name or check spelling",
            "available_objects_sample": available_objects,
            "narrative_evidence": narrative_evidence.get("evidence", []),
            "local_narrative": local_narrative.get("activities", []),
            "candidates_in_window": local_narrative.get("candidates_in_window", []),
            "_tip": (
                "The object may not be a tracked entity. Use narrative_evidence, "
                "local_narrative, and candidates_in_window as faithful context before giving up."
            ),
        }

    state_history = matched_obj.get("state_history", [])
    all_states = set()
    all_locations = set()
    location_transitions = []
    prev_location = None

    for entry in state_history:
        if entry.get("state"):
            all_states.add(entry["state"])
        current_loc = entry.get("location")
        if current_loc:
            all_locations.add(current_loc)
            if prev_location and prev_location != current_loc:
                location_transitions.append({
                    "from_location": prev_location,
                    "to_location": current_loc,
                    "timestamp_seconds": entry.get("time_seconds", entry.get("turn_id", 0) * 10),
                    "turn_id": entry.get("turn_id"),
                    "action": entry.get("action", "moved"),
                })
            prev_location = current_loc

    state_val = matched_obj.get("state", {})
    current_state = state_val.get("current_state", "") if isinstance(state_val, dict) else state_val
    current_location = matched_obj.get("spatial_info", {}).get("location", "")

    if current_state:
        all_states.add(current_state)
    if current_location:
        all_locations.add(current_location)

    location_synonyms_map = {}
    for loc in all_locations:
        synonyms = environment.get_location_synonyms(memory_tools, loc)
        if len(synonyms) > 1:
            location_synonyms_map[loc] = synonyms

    enhanced_transitions = []
    for trans in location_transitions:
        enhanced = trans.copy()
        to_loc = trans.get("to_location", "")
        if to_loc:
            to_synonyms = environment.get_location_synonyms(memory_tools, to_loc)
            if len(to_synonyms) > 1 and memory_tools.include_tips:
                enhanced["to_location_synonyms"] = to_synonyms
                enhanced["_note"] = f"'{to_loc}' is equivalent to: {', '.join(to_synonyms[:3])}"
        enhanced_transitions.append(enhanced)

    result = {
        "object_id": matched_id,
        "name": matched_obj.get("name", ""),
        "match_type": match_type,
        "current_state": current_state,
        "current_location": current_location,
        "all_states_observed": list(all_states),
        "all_locations_observed": list(all_locations),
        "location_transitions": enhanced_transitions,
        "state_history": state_history,
        "first_seen": matched_obj.get("first_seen_time", matched_obj.get("first_seen")),
        "last_seen": matched_obj.get("last_seen_time", matched_obj.get("last_seen")),
    }

    if memory_tools.include_tips:
        tips = [
            "Check 'all_states_observed' for 'Was X ever in state Y?' questions",
            "Check 'location_transitions' for 'When did X move to Y?' questions",
            " IMPORTANT: 'on counter' = 'on table' = 'on countertop' (same surface)",
            " IMPORTANT: 'on dish rack' = 'on drying rack' (same location)",
            f"Object was matched via: {match_type}",
        ]
        if not enhanced_transitions:
            tips.insert(0, " WARNING: No location transitions recorded! The object was only seen in ONE location.")
            tips.insert(1, " 'last_seen' is the last observation time, not the time when the object moved.")
            tips.insert(2, " The movement time is unknown because it was not observed by the Segment Encoder.")
        result["_tips"] = tips

        if not enhanced_transitions:
            result["_data_limitation"] = (
                "No movement evidence is available; the Segment Encoder observed "
                "this object at only one location."
            )
        if location_synonyms_map:
            result["_location_equivalents"] = location_synonyms_map
        if match_type and "semantic" in match_type:
            result["_note"] = f" Object '{object_query}' not found exactly. Best match: '{matched_obj.get('name', matched_id)}'"

    return result
