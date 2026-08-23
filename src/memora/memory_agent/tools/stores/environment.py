"""Environment-memory retrieval helpers used by ``TypedMemoryTools``."""

from typing import Any, Dict, List

from memora.memory_agent.tools.embedding import get_for


def get_location_synonyms(memory_tools: Any, location: str) -> List[str]:
    """Return equivalent location names, including the original location."""
    if not location:
        return []

    loc_lower = location.lower().strip()
    result = [location]

    for key, synonyms in memory_tools.LOCATION_SYNONYMS.items():
        key_lower = key.lower()
        all_variants = [key] + synonyms

        if loc_lower in key_lower or key_lower in loc_lower:
            result.extend([s for s in all_variants if s.lower() != loc_lower])
        else:
            for syn in synonyms:
                if loc_lower in syn.lower() or syn.lower() in loc_lower:
                    result.extend([s for s in all_variants if s.lower() != loc_lower])
                    break

    return list(set(result))


def search_environment(
    memory_tools: Any,
    query: str,
    top_k: int = None,
    _skip_synonym_expansion: bool = False,
) -> List[Dict[str, Any]]:
    """Search Environment Memory for places, zones, features, and relations."""
    if not _skip_synonym_expansion:
        return memory_tools._search_with_synonym_expansion(memory_tools.search_environment, query, top_k)

    top_k = min(top_k or memory_tools.DEFAULT_TOP_K, memory_tools.MAX_TOP_K)

    if not memory_tools.current_memory:
        return [{"error": "No memory context set."}]

    environment_log = memory_tools.current_memory.get("environment_log", [])
    if not environment_log:
        return []

    docs = []
    for env in environment_log:
        state = env["current_state"] if "current_state" in env else env

        text_parts = [
            env.get("location_id", ""),
            state.get("layout_description", ""),
            " ".join(state.get("features") or []),
            " ".join(state.get("spatial_relations") or []),
        ]

        zones = state.get("zones", {})
        if isinstance(zones, dict):
            for zone_name, zone_data in zones.items():
                text_parts.append(zone_name)
                if isinstance(zone_data, dict):
                    text_parts.append(zone_data.get("anchor", ""))
                    contents = zone_data.get("contents", [])
                    if contents:
                        text_parts.extend(contents)
                elif isinstance(zone_data, list):
                    text_parts.extend(zone_data)
                elif isinstance(zone_data, str):
                    text_parts.append(zone_data)
        elif isinstance(zones, list):
            text_parts.extend(str(z) for z in zones)

        docs.append({
            "location_id": env.get("location_id", "unknown"),
            "text": " ".join(str(p) for p in text_parts if p),
            "state": state,
            "first_seen": env.get("first_seen"),
            "last_seen": env.get("last_seen"),
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
                result = {
                    "location_id": doc["location_id"],
                    "layout_description": doc["state"].get("layout_description", ""),
                    "zones": doc["state"].get("zones", {}),
                    "spatial_relations": doc["state"].get("spatial_relations", []),
                    "features": doc["state"].get("features", []),
                    "first_seen": doc["first_seen"],
                    "last_seen": doc["last_seen"],
                    "similarity": sim_score,
                }
                if memory_tools.include_tips and sim_score < memory_tools.similarity_threshold:
                    result["_confidence"] = "low"
                results.append(result)

        return results

    query_lower = query.lower()
    results = []
    for doc in docs:
        if query_lower in doc["text"].lower():
            results.append({
                "location_id": doc["location_id"],
                "layout_description": doc["state"].get("layout_description", ""),
                "zones": doc["state"].get("zones", {}),
                "spatial_relations": doc["state"].get("spatial_relations", []),
                "features": doc["state"].get("features", []),
            })
    return results[:top_k]
