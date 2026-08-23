"""Planning-oriented retrieval helpers for consolidated MEMORA memory."""

from typing import Any, Dict, List, Tuple

from memora.memory_agent.tools.embedding import get_for

def get_routine_skill(memory_tools: Any, goal_query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieve consolidated routine templates by semantic match on a goal."""
    knowledge = memory_tools.current_memory.get("inferred_knowledge", {})
    procedures = knowledge.get("reusable_procedures", {})
    routines = procedures.get("procedure_templates", [])
    if not routines:
        return {
            "error": "No consolidated procedure templates are available. Fall back "
                     "to search_activities(query) for episode evidence.",
            "num_matched": 0,
            "routines": [],
        }

    docs: List[Dict[str, Any]] = []
    for item in routines:
        if not isinstance(item, dict):
            continue
        goal = item.get("goal", "")
        steps = item.get("canonical_steps", []) or []
        step_actions = []
        for step in steps:
            if isinstance(step, dict):
                action = step.get("action", "")
                obj = step.get("object", "")
                step_actions.append(f"{action} {obj}".strip())
            elif isinstance(step, str):
                step_actions.append(step)
        steps_text = " ".join(part for part in step_actions if part)
        objects_text = " ".join(
            obj.get("object", "") if isinstance(obj, dict) else str(obj)
            for obj in (item.get("key_objects", []) or [])
        )
        searchable = f"{goal} {steps_text} {objects_text}".strip()
        docs.append({"item": item, "text": searchable})

    model = get_for(memory_tools)
    ranked: List[Dict[str, Any]] = []
    if model and goal_query.strip():
        import numpy as np

        texts = [f"passage: {doc['text']}" for doc in docs]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_emb = model.encode(f"query: {goal_query}", normalize_embeddings=True)
        sims = embeddings @ query_emb
        order = np.argsort(sims)[::-1]
        for idx in order[: max(top_k, 1)]:
            ranked.append({"item": docs[idx]["item"], "similarity": float(sims[idx])})
    else:
        query_words = goal_query.lower().split()
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for doc in docs:
            text = doc["text"].lower()
            score = sum(1 for word in query_words if word and word in text)
            scored.append((score, doc["item"]))
        scored.sort(key=lambda item: item[0], reverse=True)
        for score, item in scored[: max(top_k, 1)]:
            ranked.append({"item": item, "similarity": float(score)})

    results = []
    for ranked_item in ranked:
        item = ranked_item["item"]
        results.append({
            "goal": item.get("goal", ""),
            "canonical_steps": item.get("canonical_steps", []),
            "key_objects": item.get("key_objects", []),
            "supporting_episodes": item.get("supporting_episodes", []),
            "count": item.get("count", 0),
            "similarity": round(ranked_item["similarity"], 4),
            "source": "procedure_template",
        })
    return {
        "query": goal_query,
        "num_matched": len(results),
        "routines": results,
    }


def get_preferences(memory_tools: Any, query: str = "", top_k: int = 5) -> Dict[str, Any]:
    """Retrieve this participant's observed preferences and habits."""
    knowledge = memory_tools.current_memory.get("inferred_knowledge", {})
    prefs = knowledge.get("preferences", {}).get("statements", [])
    if not prefs:
        return {
            "error": "No consolidated preference statements are available. "
                     "Try search_patterns(query) for consolidated pattern evidence.",
            "num_matched": 0,
            "preferences": [],
        }

    docs: List[Dict[str, Any]] = []
    for item in prefs:
        if not isinstance(item, dict):
            continue
        preference = item.get("preference", "")
        evidence = item.get("evidence_summary", "")
        subtype = item.get("subtype", "")
        text = item.get("text") or f"{preference} {evidence} {subtype}"
        docs.append({"item": item, "text": text})

    if query.strip():
        model = get_for(memory_tools)
        ranked: List[Dict[str, Any]] = []
        if model:
            import numpy as np

            texts = [f"passage: {doc['text']}" for doc in docs]
            embeddings = model.encode(texts, normalize_embeddings=True)
            query_emb = model.encode(f"query: {query}", normalize_embeddings=True)
            sims = embeddings @ query_emb
            order = np.argsort(sims)[::-1]
            for idx in order[: max(top_k, 1)]:
                ranked.append({"item": docs[idx]["item"], "similarity": float(sims[idx])})
        else:
            query_words = query.lower().split()
            scored = []
            for doc in docs:
                text = doc["text"].lower()
                score = sum(1 for word in query_words if word and word in text)
                scored.append((score, doc["item"]))
            scored.sort(key=lambda item: item[0], reverse=True)
            for score, item in scored[: max(top_k, 1)]:
                ranked.append({"item": item, "similarity": float(score)})
    else:
        sorted_items = sorted(
            docs,
            key=lambda doc: float(doc["item"].get("confidence", 0.5) or 0.5),
            reverse=True,
        )[: max(top_k, 1)]
        ranked = [{"item": doc["item"], "similarity": -1.0} for doc in sorted_items]

    results = []
    for ranked_item in ranked:
        item = ranked_item["item"]
        results.append({
            "preference": item.get("preference", ""),
            "evidence_summary": item.get("evidence_summary", ""),
            "subtype": item.get("subtype", ""),
            "confidence": item.get("confidence", 0.5),
            "supporting_episodes": item.get("supporting_episodes", []),
            "similarity": round(ranked_item["similarity"], 4),
            "source": "consolidated_preference",
        })
    return {
        "query": query,
        "num_matched": len(results),
        "preferences": results,
    }
