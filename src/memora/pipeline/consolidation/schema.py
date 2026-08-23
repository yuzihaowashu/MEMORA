"""Build the public Inferred Knowledge schema from consolidation outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List


def _preference_statements(preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
    statements: List[Dict[str, Any]] = []
    for item in preferences.get("storage_preferences", []) or []:
        obj = item.get("object", "object")
        location = item.get("preferred_location", "")
        statements.append({
            "preference": f"Typically stores {obj} {location}".strip(),
            "evidence_summary": item.get("evidence", ""),
            "subtype": "storage",
            "confidence": item.get("confidence", 0.5),
            "supporting_episodes": item.get("supporting_episodes", []),
        })
    for item in preferences.get("organizational_habits", []) or []:
        statements.append({
            "preference": item.get("habit", ""),
            "evidence_summary": item.get("evidence", ""),
            "subtype": "habit",
            "confidence": item.get("confidence", 0.5),
            "supporting_episodes": item.get("supporting_episodes", []),
        })
    for item in preferences.get("workflow_patterns", []) or []:
        statements.append({
            "preference": item.get("pattern", ""),
            "evidence_summary": item.get("evidence", ""),
            "subtype": "workflow",
            "confidence": item.get("confidence", 0.5),
            "supporting_episodes": item.get("supporting_episodes", []),
        })
    return [item for item in statements if item["preference"]]


def _procedure_templates(action_sequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates = []
    for sequence in action_sequences:
        content = sequence.get("content", sequence)
        steps = []
        for step in content.get("abstract_steps", []) or []:
            action = step.get("action", "") if isinstance(step, dict) else str(step)
            if action:
                steps.append({"action": action})
        key_objects = []
        for obj in content.get("key_objects", []) or []:
            key_objects.append(obj if isinstance(obj, dict) else {"object": str(obj)})
        provenance = sequence.get("provenance", {})
        templates.append({
            "goal": content.get("goal") or content.get("title", ""),
            "canonical_steps": steps,
            "key_objects": key_objects,
            "supporting_episodes": provenance.get("source_videos", []),
            "count": provenance.get("supporting_activity_count", 0),
            "confidence": sequence.get("confidence", 0.5),
        })
    return [item for item in templates if item["goal"] and item["canonical_steps"]]


def build_inferred_knowledge(
    participant_id: str,
    source_videos: List[str],
    preferences: Dict[str, Any],
    action_sequences: List[Dict[str, Any]],
    cross_episode_evidence: Dict[str, Any],
    reusable_procedures: Dict[str, Any],
) -> Dict[str, Any]:
    preferences = dict(preferences)
    preferences["statements"] = _preference_statements(preferences)
    reusable_procedures = dict(reusable_procedures)
    reusable_procedures["procedure_templates"] = _procedure_templates(action_sequences)
    return {
        "participant_id": participant_id,
        "source_videos": source_videos,
        "generated_at": datetime.now().isoformat(),
        "source": "offline_consolidation",
        "preferences": preferences,
        "action_sequences": action_sequences,
        "cross_episode_evidence": cross_episode_evidence,
        "reusable_procedures": reusable_procedures,
    }
