#!/usr/bin/env python3
"""
Offline consolidation for the Inferred Knowledge store.

This module is called by ``pipeline.memory_editor.cli`` and reuses the loaded Memory
Editor model.

Two LLM calls (separate for quality and context management):
  Call 1: Preferences - LLM analyzes object_registry + activity_log
  Call 2: Action sequences - LLM identifies procedural patterns from activity_log
"""

import json
import logging
from typing import Dict, List, Any

from memora.pipeline.consolidation.evidence import (
    compile_cross_episode_evidence,
    compile_reusable_procedure_evidence,
    select_balanced_activities,
)
from memora.pipeline.consolidation.schema import build_inferred_knowledge
from memora.pipeline.consolidation.prompts import (
    build_preference_prompt,
    build_procedure_prompt,
)
from memora.pipeline.formation_config import EPIC_KITCHENS_CONFIG

logger = logging.getLogger(__name__)


# ============================================================================
# LLM-Enhanced Preference Extractor (Call 1)
# ============================================================================

class LLMPreferenceExtractor:
    """Extract user preferences using LLM for semantic understanding."""

    def __init__(self, llm, sampling_params, tokenizer, config=None):
        self.llm = llm
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.config = config  # Optional formation prompt configuration.

    def _format_object_registry(self, object_registry: Dict) -> str:
        """Format object registry for prompt (condensed)."""
        lines = []
        for obj_id, obj_data in list(object_registry.items())[:30]:  # Limit
            name = obj_data.get('name', obj_id)
            spatial = obj_data.get('spatial_info', {})
            location = spatial.get('location', 'unknown')
            if isinstance(location, list):
                location = ', '.join(str(l) for l in location)

            # Include history if available
            history_locs = []
            for h in obj_data.get('state_history', [])[:5]:
                loc = h.get('location', '')
                if isinstance(loc, list):
                    loc = ', '.join(str(l) for l in loc)
                else:
                    loc = str(loc)
                if loc and loc not in history_locs:
                    history_locs.append(loc)

            if history_locs:
                lines.append(f"- {name}: current={location}, history=[{', '.join(history_locs)}]")
            else:
                lines.append(f"- {name}: {location}")

        return "\n".join(lines) if lines else "No objects tracked"

    def _format_activity_log(self, activity_log: List[Dict]) -> str:
        """Format activity log for prompt (condensed for preferences)."""
        lines = []
        for activity in select_balanced_activities(activity_log, limit=120):
            tw = activity.get('time_window', {})
            summary = (activity.get('summary', '') or
                      activity.get('action', '') or
                      activity.get('detailed_narrative', ''))
            if summary:
                lines.append(f"- [{tw.get('start', 0):.0f}s] {summary}")

        return "\n".join(lines) if lines else "No activities recorded"

    def extract(self, object_registry: Dict, activity_log: List[Dict],
                participant_id: str, evidence_scope: str) -> Dict:
        """Extract preferences using LLM."""

        prompt_template = build_preference_prompt(
            self.config or EPIC_KITCHENS_CONFIG
        )
        prompt = prompt_template.format(
            participant_id=participant_id,
            evidence_scope=evidence_scope,
            object_registry=self._format_object_registry(object_registry),
            activity_log=self._format_activity_log(activity_log),
        )

        # Use chat template for correct special tokens (system + user roles)
        system_message = "You are an expert at understanding human behavior patterns."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            outputs = self.llm.generate([full_prompt], self.sampling_params)
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty output for preference extraction")
            result_text = outputs[0].outputs[0].text.strip()

            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            llm_result = json.loads(result_text)
            if not isinstance(llm_result, dict):
                raise ValueError("Preference response is not a JSON object")
            for field in ("storage_preferences", "organizational_habits", "workflow_patterns"):
                if not isinstance(llm_result.get(field, []), list):
                    raise ValueError(f"Preference response field {field!r} is not a list")

            # Convert to the Inferred Knowledge schema.
            preferences = {
                "storage_preferences": llm_result.get("storage_preferences", []),
                "organizational_habits": llm_result.get("organizational_habits", []),
                "workflow_patterns": llm_result.get("workflow_patterns", []),
                # Mirror storage preferences into an object-location view for
                # downstream retrieval.
                "object_locations": self._convert_to_default_format(
                    llm_result.get("storage_preferences", [])
                ),
                "summary": self._generate_summary(llm_result),
                "extraction_method": "llm"
            }

            return preferences

        except Exception as e:
            raise RuntimeError(
                f"Preference consolidation failed: {type(e).__name__}: {e}"
            ) from e

    def _convert_to_default_format(self, storage_prefs: List[Dict]) -> Dict:
        """Convert LLM format to default object_locations format."""
        result = {}
        for pref in storage_prefs:
            obj = pref.get("object", "").lower()
            if obj:
                result[obj] = {
                    "preferred_location": pref.get("preferred_location", "unknown"),
                    "confidence": pref.get("confidence", 0.8),
                    "context": pref.get("context", "default")
                }
        return result

    def _generate_summary(self, llm_result: Dict) -> List[str]:
        """Generate human-readable summary."""
        summary = []

        for pref in llm_result.get("storage_preferences", [])[:5]:
            summary.append(
                f"User typically keeps {pref.get('object')} {pref.get('preferred_location')} "
                f"({pref.get('context', 'default')} context)"
            )

        for habit in llm_result.get("organizational_habits", [])[:3]:
            summary.append(f"Habit: {habit.get('habit')}")

        return summary

# ============================================================================
# Reusable procedure extractor (Call 2)
# ============================================================================

class LLMActionSequenceExtractor:
    """Extract action sequences using LLM."""

    def __init__(self, llm, sampling_params, tokenizer, config=None):
        self.llm = llm
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        self.config = config  # Optional formation prompt configuration.

    def _format_activity_list(self, activity_log: List[Dict]) -> str:
        """Format activity log for prompt."""
        lines = []
        for activity in select_balanced_activities(activity_log, limit=160):
            tw = activity.get('time_window', {})
            start = tw.get('start', 0)
            end = tw.get('end', 0)
            source_video = activity.get("source_video", "unknown_episode")
            summary = (activity.get('summary', '') or
                      activity.get('action', '') or
                      activity.get('detailed_narrative', ''))
            if summary:
                lines.append(f"[{source_video} {start:.0f}-{end:.0f}s] {summary}")
        return "\n".join(lines)

    def extract(
        self,
        activity_log: List[Dict],
        evidence_scope: str,
        participant_id: str,
        source_videos: List[str],
    ) -> List[Dict]:
        """Extract action sequences using LLM."""

        if not activity_log:
            return []

        episode_bounds: Dict[str, List[float]] = {}
        for activity in activity_log:
            source_video = str(activity.get("source_video", "unknown_episode"))
            time_window = activity.get("time_window", {}) or {}
            start = float(time_window.get("start", 0) or 0)
            end = float(time_window.get("end", start) or start)
            bounds = episode_bounds.setdefault(source_video, [start, end])
            bounds[0] = min(bounds[0], start)
            bounds[1] = max(bounds[1], end)
        duration = sum(max(end - start, 0) for start, end in episode_bounds.values())
        formatted = self._format_activity_list(activity_log)

        config = self.config or EPIC_KITCHENS_CONFIG
        prompt_template = build_procedure_prompt(config)
        prompt = prompt_template.format(
            evidence_scope=evidence_scope,
            duration=int(duration),
            num_activities=len(activity_log),
            activities=formatted,
        )
        experience_desc = config.experience_description

        # Use chat template for correct special tokens (system + user roles)
        system_message = f"You are an expert at analyzing human activities in {experience_desc} videos."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        full_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            outputs = self.llm.generate([full_prompt], self.sampling_params)
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty output for action pattern extraction")
            result_text = outputs[0].outputs[0].text.strip()

            # Parse JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            llm_result = json.loads(result_text)
            if not isinstance(llm_result, dict):
                raise ValueError("Procedure response is not a JSON object")
            patterns = llm_result.get("patterns")
            if not isinstance(patterns, list):
                raise ValueError("Procedure response has no patterns list")

            # Convert to the Inferred Knowledge schema.
            action_sequences = []

            source_video_set = set(source_videos)
            for index, pattern in enumerate(patterns):
                if not isinstance(pattern, dict):
                    raise ValueError(f"Procedure pattern {index} is not a JSON object")
                if not isinstance(pattern.get("pattern_name"), str) or not pattern["pattern_name"].strip():
                    raise ValueError(f"Procedure pattern {index} has no pattern_name")
                if not isinstance(pattern.get("key_steps"), list) or not pattern["key_steps"]:
                    raise ValueError(f"Procedure pattern {index} has no key_steps")
                supporting_episodes = pattern.get("supporting_episodes")
                if not isinstance(supporting_episodes, list) or not supporting_episodes:
                    raise ValueError(f"Procedure pattern {index} has no supporting_episodes")
                unknown_episodes = sorted(set(supporting_episodes) - source_video_set)
                if unknown_episodes:
                    raise ValueError(
                        f"Procedure pattern {index} cites unknown episodes: "
                        + ", ".join(unknown_episodes)
                    )
                supporting_activity_count = pattern.get("supporting_activity_count")
                if not isinstance(supporting_activity_count, int) or supporting_activity_count < 1:
                    raise ValueError(
                        f"Procedure pattern {index} has invalid supporting_activity_count"
                    )
                # Build abstract_steps from key_steps
                abstract_steps = [
                    {"order": j + 1, "type": "mandatory", "action": step}
                    for j, step in enumerate(pattern.get("key_steps", []))
                ]

                detailed_steps = [
                    {"order": step["order"], "action": step["action"]}
                    for step in abstract_steps
                ]

                sequence = {
                    "knowledge_id": f"act_seq_{evidence_scope}_p{pattern.get('pattern_id', len(action_sequences)+1)}",
                    "type": "action_sequence",
                    "confidence": pattern.get("confidence", 0.85),
                    "content": {
                        "title": pattern["pattern_name"],
                        "goal": pattern.get("goal", ""),
                        "activity_type": pattern.get("activity_type", "other"),
                        "key_objects": pattern.get("key_objects", []),
                        "abstract_steps": abstract_steps,
                        "detailed_steps": detailed_steps
                    },
                    "provenance": {
                        "source_videos": supporting_episodes,
                        "total_activities": len(activity_log),
                        "supporting_activity_count": supporting_activity_count,
                        "observed_duration": duration,
                        "extraction_method": "memory_editor",
                        "participant_id": participant_id
                    }
                }
                action_sequences.append(sequence)

            logger.info(f"     LLM found {len(action_sequences)} action sequences")
            return action_sequences

        except Exception as e:
            raise RuntimeError(
                f"Procedure consolidation failed: {type(e).__name__}: {e}"
            ) from e


# ============================================================================
# Main Entry Point
# ============================================================================

def run_offline_consolidation(
    participant_memory: Dict[str, Dict],
    llm,
    sampling_params,
    tokenizer,
    config=None,
) -> Dict[str, Any]:
    """Consolidate all episodes for one participant into Inferred Knowledge."""
    source_videos = sorted(participant_memory)
    participant_ids = {video_id.split("_")[0] for video_id in source_videos}
    if len(participant_ids) > 1:
        raise ValueError(
            "Offline consolidation expects one participant at a time; found "
            + ", ".join(sorted(participant_ids))
        )
    participant_id = next(iter(participant_ids), "unknown")

    object_registry: Dict[str, Any] = {}
    activity_log: List[Dict[str, Any]] = []
    for video_id in source_videos:
        video_data = participant_memory[video_id]
        for object_id, object_data in video_data.get("object_registry", {}).items():
            object_registry[f"{video_id}:{object_id}"] = object_data
        for activity in video_data.get("activity_log", []):
            record = dict(activity)
            record["source_video"] = video_id
            activity_log.append(record)

    print(f"\n{'═' * 60}", flush=True)
    print("Offline consolidation", flush=True)
    print(f"   Participant: {participant_id}", flush=True)
    print(f"   Source videos: {len(source_videos)}", flush=True)
    print("   Model calls: 2 (preferences + reusable procedures)", flush=True)
    print(f"{'═' * 60}", flush=True)

    preference_extractor = LLMPreferenceExtractor(
        llm, sampling_params, tokenizer, config=config
    )
    sequence_extractor = LLMActionSequenceExtractor(
        llm, sampling_params, tokenizer, config=config
    )
    evidence_scope = f"{participant_id} across {len(source_videos)} episodes"
    preferences = preference_extractor.extract(
        object_registry=object_registry,
        activity_log=activity_log,
        participant_id=participant_id,
        evidence_scope=evidence_scope,
    )
    action_sequences = sequence_extractor.extract(
        activity_log=activity_log,
        evidence_scope=participant_id,
        participant_id=participant_id,
        source_videos=source_videos,
    )

    cross_episode_evidence = compile_cross_episode_evidence(activity_log)
    reusable_procedures = compile_reusable_procedure_evidence(
        activity_log,
        cross_episode_evidence,
    )
    inferred_knowledge = build_inferred_knowledge(
        participant_id=participant_id,
        source_videos=source_videos,
        preferences=preferences,
        action_sequences=action_sequences,
        cross_episode_evidence=cross_episode_evidence,
        reusable_procedures=reusable_procedures,
    )
    print(f"   Preferences: {len(preferences.get('storage_preferences', []))}", flush=True)
    print(f"   Reusable procedures: {len(action_sequences)}", flush=True)
    print(f"{'═' * 60}\n", flush=True)
    return inferred_knowledge
