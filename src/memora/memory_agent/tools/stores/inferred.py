"""Inferred-knowledge retrieval helpers used by ``TypedMemoryTools``."""

import logging
import math
import re
from typing import Any, Dict, List

from memora.memory_agent.tools.embedding import get_for
from memora.memory_agent.tools.ranking import balanced_pattern_rank

logger = logging.getLogger(__name__)


def pattern_lexical_term_bonus(memory_tools, query: str, doc: Dict[str, Any]) -> float:
    """Reward concrete token overlap between query and doc (dims generic embedding hits)."""
    q = (query or "").lower()
    blob = f"{doc.get('text', '')} {doc.get('description', '')}".lower()
    terms = [t for t in re.findall(r"[a-z][a-z0-9]{2,}", q) if len(t) >= 3]
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in blob)
    return min(0.07, 0.014 * hits)

def pattern_choice_overlap_bonus(memory_tools, query: str, doc: Dict[str, Any]) -> float:
    """Extra lift when query aligns with compiled choice phrases on profile cards."""
    ql = (query or "").lower().strip()
    if not ql:
        return 0.0
    qtoks = set(re.findall(r"[a-z][a-z0-9]{2,}", ql))
    best = 0.0
    for p in doc.get("choice_phrases") or []:
        pl = str(p).lower().strip()
        if len(pl) < 5:
            continue
        if pl in ql or ql in pl:
            return 0.055
        ptoks = set(re.findall(r"[a-z][a-z0-9]{2,}", pl))
        if ptoks and qtoks:
            inter = ptoks & qtoks
            if not inter:
                continue
            overlap = len(inter) / max(1, min(len(ptoks), len(qtoks)))
            if overlap >= 0.34:
                best = max(best, 0.038)
    return best

def pattern_evidence_strength_bonus(memory_tools, doc: Dict[str, Any]) -> float:
    bonus = 0.0
    c = doc.get("confidence")
    if isinstance(c, (int, float)):
        cf = float(c)
        if cf >= 0.82:
            bonus += 0.022
        elif cf >= 0.65:
            bonus += 0.013
    sc = doc.get("support_count")
    if isinstance(sc, (int, float)) and float(sc) > 0:
        bonus += min(0.03, 0.007 * math.log1p(float(sc)))
    return bonus

def pattern_source_priority_bonus(memory_tools, doc: Dict[str, Any], strategy_name: str = "") -> float:
    """Prefer consolidated records over coarse cross-episode counts."""
    src = doc.get("source") or ""
    cat = doc.get("category") or ""
    if src == "consolidated_preference":
        return 0.078
    if src == "reusable_procedures":
        return 0.052
    if src == "offline_consolidation":
        if cat == "action_sequence":
            return 0.022
        return 0.032
    if src == "cross_episode_evidence":
        if cat == "participant_object_context":
            return -0.052
        if cat == "participant_object_frequency":
            return -0.015
        if cat == "participant_transition":
            return 0.026
        return 0.0
    if src == "online":
        return 0.017
    return 0.0

def pattern_rank_spec_for_query(memory_tools, query: str) -> Dict[str, Any]:
    return balanced_pattern_rank()

def scale_signed_source_bonus(memory_tools, bonus: float, spec: Dict[str, Any]) -> float:
    """Apply strategy weights; negative bonuses scale with neg_source_scale (stronger penalty)."""
    ws = float(spec["w_source"])
    if bonus >= 0:
        return bonus * ws
    return bonus * ws * float(spec["neg_source_scale"])

def pattern_search_rank_score(memory_tools, query: str, doc: Dict[str, Any], raw_similarity: float) -> float:
    spec = memory_tools._pattern_rank_spec_for_query(query)
    if spec["embedding_only"]:
        return float(raw_similarity)
    src = memory_tools._scale_signed_source_bonus(
        memory_tools._pattern_source_priority_bonus(doc, strategy_name=spec.get("name", "")),
        spec,
    )
    return (
        float(raw_similarity)
        + src
        + memory_tools._pattern_lexical_term_bonus(query, doc) * float(spec["w_lexical"])
        + memory_tools._pattern_choice_overlap_bonus(query, doc) * float(spec["w_choice"])
        + memory_tools._pattern_evidence_strength_bonus(doc) * float(spec["w_evidence"])
    )

def pattern_doc_allowed_for_question_type(memory_tools, doc: Dict[str, Any]) -> bool:
    """Hard-filter pattern documents by benchmark question type.

    These rules align retrieval pools with the per-question prompts:
    preference questions see preference memory; routine questions see
    routine memory.
    """
    qtype = (memory_tools.current_question_type or "").strip()
    if not qtype:
        return True

    src = doc.get("source") or ""
    cat = doc.get("category") or ""
    if qtype == "SPref":
        if src == "reusable_procedures" and cat in {"procedure_template", "procedure_object_handling"}:
            return False
        if src == "cross_episode_evidence" and cat == "participant_object_frequency":
            return False
        return True

    if qtype == "SHabit":
        if src == "consolidated_preference":
            return False
        if src == "cross_episode_evidence" and cat == "participant_object_frequency":
            return False
        return True

    if qtype == "SRoutine":
        if src == "consolidated_preference":
            return False
        return True

    if qtype == "ERecall":
        return src != "consolidated_preference"

    return True

# ==================== Tool 4: search_patterns ====================
def search_patterns(memory_tools, query: str, top_k: int = None, _skip_synonym_expansion: bool = False) -> List[Dict[str, Any]]:
    """
    Search inferred knowledge for patterns, preferences, and habits.

    Args:
        query: Search by pattern type or keyword
               Examples: "hand preference", "workflow", "habit", "routine"

    Returns:
        List of matching patterns:
        - category: Pattern type (behavior_patterns, preferences, spatial_habits, efficiency_notes)
        - pattern_id: Unique identifier
        - description: Pattern description
    """
    # Try synonym expansion if no results
    if not _skip_synonym_expansion:
        return memory_tools._search_with_synonym_expansion(memory_tools.search_patterns, query, top_k)

    # Event-recall and temporal questions often need a wider candidate pool
    # because the relevant evidence may be a specific episode rather than a
    # consolidated routine.
    episodic_qtype = (memory_tools.current_question_type or "").strip() == "ERecall"
    effective_top_k = top_k or memory_tools.DEFAULT_TOP_K
    if episodic_qtype:
        effective_top_k = max(int(effective_top_k), 20)
    top_k = min(effective_top_k, memory_tools.MAX_TOP_K)

    if not memory_tools.current_memory:
        return [{"error": "No memory context set."}]

    docs = []
    inferred_knowledge = memory_tools.current_memory.get("inferred_knowledge", {})
    if inferred_knowledge:
        # Handle preferences sub-structure
        preferences = inferred_knowledge.get("preferences", {})
        if preferences:
            # storage_preferences
            for pref in preferences.get("storage_preferences", []):
                obj = pref.get("object", "")
                loc = pref.get("preferred_location", "")
                ctx = pref.get("context", "")
                conf = pref.get("confidence", 0)
                evidence = pref.get("evidence", "")
                desc = f"{obj} is typically kept {loc} ({ctx}). {evidence}"
                docs.append({
                    "category": "storage_preference",
                    "pattern_id": f"storage_{obj}",
                    "description": desc,
                    "text": f"storage preference {obj} {loc} {ctx} {evidence}",
                    "source": "offline_consolidation",
                    "confidence": conf
                })

            # organizational_habits
            for habit in preferences.get("organizational_habits", []):
                habit_desc = habit.get("habit", "")
                objects = ", ".join(habit.get("objects_involved", []))
                conf = habit.get("confidence", 0)
                docs.append({
                    "category": "organizational_habit",
                    "pattern_id": f"habit_{len(docs)}",
                    "description": f"{habit_desc} (involves: {objects})",
                    "text": f"organizational habit {habit_desc} {objects}",
                    "source": "offline_consolidation",
                    "confidence": conf
                })

            # workflow_patterns
            for wf in preferences.get("workflow_patterns", []):
                pattern = wf.get("pattern", "")
                objects = ", ".join(wf.get("objects", []))
                conf = wf.get("confidence", 0)
                docs.append({
                    "category": "workflow_pattern",
                    "pattern_id": f"workflow_{len(docs)}",
                    "description": f"{pattern} (objects: {objects})",
                    "text": f"workflow pattern {pattern} {objects}",
                    "source": "offline_consolidation",
                    "confidence": conf
                })

        # Handle consolidated action-sequence evidence.
        for seq in inferred_knowledge.get("action_sequences", []):
            content = seq.get("content", seq)
            title = content.get("title", seq.get("pattern_name", ""))
            goal = content.get("goal", "")
            activity_type = content.get("activity_type", "")
            key_objects = ", ".join(content.get("key_objects", []))
            steps = content.get("abstract_steps", content.get("key_steps", []))
            steps_text = " -> ".join([s.get("action", s) if isinstance(s, dict) else str(s) for s in steps[:5]])

            docs.append({
                "category": "action_sequence",
                "pattern_id": seq.get("knowledge_id", f"action_{len(docs)}"),
                "description": f"{title}: {goal}. Steps: {steps_text}",
                "text": f"action sequence {title} {goal} {activity_type} {key_objects} {steps_text}",
                "source": "offline_consolidation",
                "activity_type": activity_type,
                "key_objects": content.get("key_objects", [])
            })

    # Search participant-level semantic evidence derived during consolidation.
    semantic_evidence = inferred_knowledge.get("cross_episode_evidence", {})
    if semantic_evidence:
        for item in semantic_evidence.get("common_transitions", []):
            trigger = item.get("trigger", "")
            next_action = item.get("next_action", "")
            count = item.get("count", 0)
            docs.append({
                "category": "participant_transition",
                "pattern_id": f"transition_{len(docs)}",
                "description": f"After {trigger}, the next action is often {next_action} (count={count}).",
                "text": f"participant transition after {trigger} next {next_action} count {count}",
                "source": "cross_episode_evidence",
                "confidence": min(1.0, 0.4 + 0.1 * float(count or 0)),
            })
        for item in semantic_evidence.get("frequent_objects", []):
            obj = item.get("object", "")
            count = item.get("count", 0)
            examples = item.get("examples", [])
            example_text = " ".join(e.get("summary", "") for e in examples if isinstance(e, dict))
            docs.append({
                "category": "participant_object_frequency",
                "pattern_id": f"object_freq_{obj}",
                "description": f"{obj} appears frequently in participant activities (count={count}). Examples: {memory_tools._shorten_text(example_text, 180)}",
                "text": f"participant object frequency {obj} count {count} examples {example_text}",
                "source": "cross_episode_evidence",
                "confidence": min(1.0, 0.3 + 0.05 * float(count or 0)),
            })
        for item in semantic_evidence.get("object_contexts", []):
            ctx = item.get("object_context", "")
            count = item.get("count", 0)
            docs.append({
                "category": "participant_object_context",
                "pattern_id": f"object_context_{len(docs)}",
                "description": f"Object-context evidence: {ctx} (count={count}).",
                "text": f"participant object context {ctx} count {count}",
                "source": "cross_episode_evidence",
                "confidence": min(1.0, 0.3 + 0.05 * float(count or 0)),
            })

    # Search reusable procedures derived from activity traces.
    reusable_procedures = inferred_knowledge.get("reusable_procedures", {})
    if reusable_procedures:
        for item in reusable_procedures.get("atomic_transitions", []):
            trigger = item.get("trigger", "")
            outcome = item.get("outcome", "")
            count = item.get("count", 0)
            examples = item.get("supporting_episodes", [])
            example_text = " ".join(e.get("summary", "") for e in examples if isinstance(e, dict))
            docs.append({
                "category": "procedure_transition",
                "pattern_id": f"procedure_transition_{len(docs)}",
                "description": f"Reusable transition: after {trigger}, the next action is {outcome} (count={count}).",
                "text": f"reusable procedure transition after {trigger} next outcome {outcome} count {count} examples {example_text}",
                "source": "reusable_procedures",
                "confidence": min(1.0, 0.45 + 0.08 * float(count or 0)),
                "supporting_episodes": examples,
            })
        for item in reusable_procedures.get("object_handling_events", []):
            trigger_action = item.get("trigger_action", "")
            trigger_object = item.get("trigger_object", "")
            outcome_action = item.get("outcome_action", "")
            outcome_object = item.get("outcome_object", "")
            context = item.get("context", "")
            count = item.get("count", 0)
            examples = item.get("supporting_episodes", [])
            docs.append({
                "category": "procedure_object_handling",
                "pattern_id": f"procedure_object_{len(docs)}",
                "description": (
                    f"Object-handling skill in context '{context}': "
                    f"{trigger_action} {trigger_object} -> {outcome_action} {outcome_object} "
                    f"(count={count})."
                ),
                "text": (
                    f"reusable procedure object handling context {context} trigger {trigger_action} "
                    f"{trigger_object} outcome {outcome_action} {outcome_object} count {count}"
                ),
                "source": "reusable_procedures",
                "confidence": min(1.0, 0.45 + 0.08 * float(count or 0)),
                "supporting_episodes": examples,
            })
        for item in reusable_procedures.get("procedure_templates", []):
            goal = item.get("goal", "")
            steps = item.get("canonical_steps", [])
            objects = item.get("key_objects", [])
            steps_text = " -> ".join(
                step.get("action", "") for step in steps if isinstance(step, dict)
            )
            objects_text = " ".join(
                obj.get("object", "") for obj in objects if isinstance(obj, dict)
            )
            docs.append({
                "category": "procedure_template",
                "pattern_id": f"procedure_template_{len(docs)}",
                "description": f"Reusable procedure: {goal}. Canonical steps: {steps_text}. Key objects: {objects_text}.",
                "text": f"reusable procedure goal {goal} canonical steps {steps_text} key objects {objects_text}",
                "source": "reusable_procedures",
                "confidence": min(1.0, 0.45 + 0.03 * float(item.get("count", 0) or 0)),
                "supporting_episodes": item.get("supporting_episodes", []),
            })
    for item in inferred_knowledge.get("preferences", {}).get("statements", []):
        preference = item.get("preference", "")
        evidence = item.get("evidence_summary", "")
        pref_text = item.get("text") or f"{preference} {evidence}"
        docs.append({
            "category": "consolidated_preference",
            "pattern_id": f"consolidated_preference_{len(docs)}",
            "description": f"{preference} {evidence}".strip(),
            "text": f"consolidated preference {item.get('subtype', '')} {pref_text}",
            "source": "consolidated_preference",
            "confidence": item.get("confidence", 0.5),
            "supporting_episodes": item.get("supporting_episodes", []),
            "generator": item.get("generator", ""),
        })

    total_docs_built = len(docs)
    docs = [doc for doc in docs if memory_tools._pattern_doc_allowed_for_question_type(doc)]
    if len(docs) != total_docs_built:
        logger.debug(
            "Pattern type filter (%s): %d -> %d docs",
            memory_tools.current_question_type or "unknown",
            total_docs_built,
            len(docs),
        )
    if not docs:
        return []

    # Semantic search
    model = get_for(memory_tools)
    if model:
        import numpy as np

        texts = [f"passage: {doc['text']}" for doc in docs]
        embeddings = model.encode(texts, normalize_embeddings=True)
        query_emb = model.encode(f"query: {query}", normalize_embeddings=True)

        similarities = embeddings @ query_emb
        pattern_threshold = memory_tools.similarity_threshold * 0.75
        spec = memory_tools._pattern_rank_spec_for_query(query)
        pool_mult = int(spec["pool_mult"])
        pool_min = int(spec["pool_min"])
        pool_by_sim = min(len(docs), max(top_k * pool_mult, pool_min))
        top_sim_idx = np.argsort(similarities)[::-1][:pool_by_sim]
        candidate_set = {int(i) for i in top_sim_idx.tolist()}
        always_include = spec.get("always_include_sources") or ()
        for i, d in enumerate(docs):
            if d.get("source") in always_include:
                candidate_set.add(i)

        if spec["embedding_only"]:
            ranked_indices = sorted(
                candidate_set,
                key=lambda idx: float(similarities[idx]),
                reverse=True,
            )
        else:
            ranked_indices = sorted(
                candidate_set,
                key=lambda idx: memory_tools._pattern_search_rank_score(query, docs[idx], float(similarities[idx])),
                reverse=True,
            )

        rank_delta = float(spec["rank_pass_delta"])

        results = []
        for idx in ranked_indices:
            if len(results) >= top_k:
                break
            sim_score = float(similarities[idx])
            if spec["embedding_only"]:
                rank_score = sim_score
                passes = memory_tools.always_return_top_k or sim_score > pattern_threshold
            else:
                rank_score = memory_tools._pattern_search_rank_score(query, docs[idx], sim_score)
                passes = (
                    memory_tools.always_return_top_k
                    or sim_score > pattern_threshold
                    or rank_score > pattern_threshold + rank_delta
                )
            if not passes:
                continue
            doc = docs[idx]
            result = {
                "category": doc["category"],
                "pattern_id": doc["pattern_id"],
                "description": doc["description"],
                "similarity": sim_score,
                "source": doc.get("source"),
            }
            if doc.get("supporting_episodes"):
                result["supporting_episodes"] = doc["supporting_episodes"][:3]
            if doc.get("confidence") is not None:
                result["confidence"] = doc.get("confidence")
            low_tip = sim_score < pattern_threshold
            if not spec["embedding_only"]:
                low_tip = low_tip and rank_score <= pattern_threshold + rank_delta
            if memory_tools.include_tips and low_tip:
                result["_confidence"] = "low"
            results.append(result)

        logger.debug("Returning %d pattern results", len(results))
        return results
    else:
        # Fallback
        query_lower = query.lower()
        results = []
        for doc in docs:
            if query_lower in doc["text"].lower():
                results.append({
                    "category": doc["category"],
                    "pattern_id": doc["pattern_id"],
                    "description": doc["description"],
                    "source": doc.get("source"),
                })
        return results[:top_k]
