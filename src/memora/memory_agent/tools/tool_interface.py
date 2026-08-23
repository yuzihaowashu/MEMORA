#!/usr/bin/env python3
"""
Read-time facade for MEMORA's typed embodied memory.

``TypedMemoryTools`` is the public tool surface used by the LLM agent. It keeps
the current temporal or participant context, exposes store-specific retrieval
tools, composes cross-store evidence views, and dispatches function calls.

The implementation follows the paper's memory architecture:
- ``tools/stores/environment.py``: Environment Memory, for places and spatial context.
- ``tools/stores/entity.py``: Entity Memory, for object identity and state history.
- ``tools/stores/activity.py``: Activity Memory, for timestamped action evidence.
- ``tools/stores/inferred.py``: Inferred Knowledge, for routines and regularities.
- ``tools/evidence/``: cross-store evidence for episodes, semantics, and planning.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from memora.memory_agent.tools.evidence import episode, planning
from memora.memory_agent.tools.stores import activity, entity, environment, inferred
from memora.memory_agent.tools.context import MemoryContextMixin
from memora.memory_agent.tools.embedding import resolve_device
from memora.memory_agent.tools.evidence.cross_store import CrossStoreEvidenceMixin
from memora.memory_agent.tools.stores.entity_normalization import (
    normalize_object_name,
    strip_attributes_from_name,
)
from memora.memory_agent.tools.lexicon import (
    ACTION_VERBS as LEXICON_ACTION_VERBS,
    LOCATION_SYNONYMS,
    OBJECT_SYNONYMS,
    STATE_ADJECTIVES as LEXICON_STATE_ADJECTIVES,
)
from memora.memory_agent.tools.schemas import build_tools_definition

logger = logging.getLogger(__name__)

class TypedMemoryTools(MemoryContextMixin, CrossStoreEvidenceMixin):
    """
    Read-time facade for MEMORA's four stores.

    It provides unified and store-aware search plus temporal, semantic, and
    planning helpers over Environment Memory, Entity Memory, Activity Memory,
    and Inferred Knowledge.
    """

    DEFAULT_TOP_K = 10
    MAX_TOP_K = 30

    SYNONYMS = OBJECT_SYNONYMS
    LOCATION_SYNONYMS = LOCATION_SYNONYMS
    ACTION_VERBS = LEXICON_ACTION_VERBS
    STATE_ADJECTIVES = LEXICON_STATE_ADJECTIVES

    def __init__(
        self,
        memory_file: str,
        e5_model_name: str = "intfloat/e5-base-v2",
        device: Optional[str] = None,
        include_tips: bool = False,
    ):
        """
        Initialize the MEMORA typed-memory tool interface.

        Args:
            memory_file: Path to a participant memory JSON/JSONL file.
            e5_model_name: E5 model for semantic search
            device: Device for E5 model
        """
        resolved_device = resolve_device(device)

        self.memory_file = memory_file
        self.e5_model_name = e5_model_name
        self.device = resolved_device
        self.include_tips = include_tips
        self.inferred_knowledge = {}

        self.participant_memory = self._load_participant_memory(memory_file)

        # Context for filtering
        self.context_type = None  # "episodic" or "semantic"
        self.current_video_id = None
        self.time_threshold = None
        self.participant_videos = None
        self.current_question_type = ""

        # Current memory state (rebuilt based on context)
        self.current_memory = None

        # Query cache to prevent duplicate searches (cleared on context change)
        self.query_cache = {}

        # Search configuration
        # Return broad candidates and let the planner judge semantic relevance.
        self.similarity_threshold = 0.2
        self.always_return_top_k = True  # Always return top-K even if below threshold
        logger.info("TypedMemoryTools initialized with %d videos", len(self.participant_memory))

    def set_question_context(self, question_type: str = "") -> None:
        """Set the current benchmark question type for type-aware retrieval.

        The model prompt already differs by question type; this makes tool-side
        retrieval follow the same contract instead of relying only on query words.
        """
        self.current_question_type = str(question_type or "").strip()

    def _expand_query_with_synonyms(self, query: str) -> List[str]:
        """
        Expand a query with synonyms.
        Returns list of queries to try (original + synonyms).
        """
        queries = [query]
        query_lower = query.lower()

        for key, synonyms in self.SYNONYMS.items():
            pattern = re.compile(rf"\b{re.escape(key)}\b")
            if pattern.search(query_lower):
                for syn in synonyms:
                    expanded = pattern.sub(syn, query_lower)
                    if expanded not in queries:
                        queries.append(expanded)

        return queries

    def _is_preference_query(self, query: str) -> bool:
        """
        Detect if a query asks about preferences or habits.

        Examples of preference queries:
        - "P01 prefer clean up" → True
        - "typically use which tool" → True
        - "habit after cooking" → True
        - "storage preference" → True
        """
        query_lower = query.lower()
        preference_keywords = {
            "prefer", "preference", "typically", "usually", "habit", "always",
            "tends to", "likes to", "pattern", "routine", "often", "frequently",
            "storage", "organize", "clean up", "after meal", "after cooking",
            "before cooking", "workflow"
        }

        for keyword in preference_keywords:
            if keyword in query_lower:
                return True

        return False

    def _is_action_query(self, query: str) -> bool:
        """
        Detect if query is asking about ACTIONS (should prioritize activities).

        Examples of action queries:
        - "How does P03 open jar?" → True
        - "When did person wash dishes?" → True
        - "Where is the plate?" → False

        NOTE: Preference queries are NOT action queries!
        """
        # First check if it's a preference query - those should prioritize patterns
        if self._is_preference_query(query):
            return False

        query_words = set(query.lower().split())

        # Check for action verbs
        for verb in self.ACTION_VERBS:
            if verb in query_words:
                return True

        # Check for "how" or "when" questions (often action-related)
        if query_words & {"how", "when", "what"}:
            return True

        return False

    def _filter_object_noise(self, objects: List[Dict], query: str) -> List[Dict]:
        """
        Filter out objects that match query only on STATE/ADJECTIVE fields.

        Problem: Query "open jar" matches "cabinet" because cabinet has state="open".
        This is noise - the model thinks "can_opener" is relevant to "open jar".

        Solution: If the match is only on visual_properties.condition or state fields,
        and those words are in STATE_ADJECTIVES, demote the object.
        """
        query_words = set(query.lower().split())
        state_words = query_words & self.STATE_ADJECTIVES

        if not state_words:
            return objects  # No state words in query, no filtering needed

        filtered = []
        for obj in objects:
            # Get the object name
            obj_name = obj.get("name", "").lower()

            # Check if name directly matches any query word (excluding state words)
            name_words = set(obj_name.split())
            non_state_query_words = query_words - state_words

            # If object name matches non-state query words, keep it
            if name_words & non_state_query_words:
                filtered.append(obj)
                continue

            # Check if the only match is on state/condition
            visual_props = obj.get("visual_properties", {})
            condition = visual_props.get("condition", "").lower()
            state_info = obj.get("state", {})
            current_state = state_info.get("current_state", "").lower() if isinstance(state_info, dict) else ""

            # If condition/state matches a state word but name doesn't match query
            # This is likely noise (e.g., "cabinet (open)" for query "open jar")
            is_state_only_match = False
            for sw in state_words:
                if sw in condition or sw in current_state:
                    # Check if name has any relevance to query
                    if not (name_words & non_state_query_words):
                        is_state_only_match = True
                        break

            if not is_state_only_match:
                filtered.append(obj)
            else:
                logger.debug("Filtered noise: %r (matched on state only)", obj_name)

        return filtered

    def _search_with_synonym_expansion(
        self,
        search_func,
        query: str,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Perform search with automatic synonym expansion.
        If initial query returns no results, try synonyms.
        """
        # Try original query first
        results = search_func(query, top_k, _skip_synonym_expansion=True)

        if results:
            return results

        # No results - try synonyms
        expanded_queries = self._expand_query_with_synonyms(query)

        for expanded_query in expanded_queries[1:]:  # Skip original (index 0)
            logger.info(f"    Trying synonym: '{query}' → '{expanded_query}'")
            results = search_func(expanded_query, top_k, _skip_synonym_expansion=True)
            if results:
                # Add note about synonym match
                if self.include_tips:
                    for r in results:
                        r["_synonym_match"] = f"Found via synonym: '{expanded_query}' (original: '{query}')"
                return results

        return []


    # ==================== Tool 1: search_objects ====================
    def search_objects(self, query: str, top_k: int = None, _skip_synonym_expansion: bool = False) -> List[Dict[str, Any]]:
        """
        Search objects in the object registry.

        Args:
            query: Search by name, color, material, state, or location
                   Examples: "plate", "blue cup", "dirty", "on counter"

        Returns:
            List of matching objects with their properties:
            - object_id: Canonical object identifier (e.g., "plate", "plate_white")
            - name: Object name
            - visual_properties: {color, material, size, condition}
            - spatial_info: {location, zone, relative_to}
            - state: {current_state, held_by}
        """
        return entity.search_objects(
            self,
            query=query,
            top_k=top_k,
            _skip_synonym_expansion=_skip_synonym_expansion,
        )

    def _search_objects_by_name_only(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search objects by name only for entity-centric recall.

        Unlike search_objects which embeds all fields (name, color, location, state),
        this method ONLY embeds name and object_id for more accurate object matching.

        Args:
            query: Object name to search (e.g., "cup", "spatula", "plate")
            top_k: Number of results to return

        Returns:
            List of matching objects with similarity scores
        """
        return entity.search_objects_by_name_only(self, query=query, top_k=top_k)

    # ==================== Tool 2: search_environment ====================
    def search_environment(self, query: str, top_k: int = None, _skip_synonym_expansion: bool = False) -> List[Dict[str, Any]]:
        """
        Search environment locations and spatial information.

        Args:
            query: Search by location name, zone, feature, or spatial relation
                   Examples: "sink area", "drying rack", "left of sink"

        Returns:
            List of matching environments:
            - location_id: Unique identifier (e.g., "sink_area")
            - layout_description: Natural language description
            - zones: {zone_name: {anchor, position, contents}}
            - spatial_relations: ["faucet ABOVE sink", ...]
            - features: ["stainless steel sink", "blue tiles", ...]
        """
        return environment.search_environment(
            self,
            query=query,
            top_k=top_k,
            _skip_synonym_expansion=_skip_synonym_expansion,
        )

    # ==================== Tool 3: search_activities ====================
    def search_activities(self, query: str, top_k: int = None, _skip_synonym_expansion: bool = False) -> List[Dict[str, Any]]:
        """
        Search activity log for actions and events.

        Args:
            query: Search by action, object, or time
                   Examples: "washing", "plate", "picking up", "1:30"

        Returns:
            List of matching activities:
            - time: Time range (e.g., "1:30-1:40")
            - summary: Brief description
            - detailed_narrative: Full description
            - action_breakdown: [{timestamp, action, object, hand}]
        """
        return activity.search_activities(
            self,
            query=query,
            top_k=top_k,
            _skip_synonym_expansion=_skip_synonym_expansion,
        )

    # ==================== Episode Reconstruction ====================
    def _build_episode(self, activity_record: Dict) -> Dict[str, Any]:
        """
        Build a complete episode from an activity hit by reconstructing
        cross-layer context (objects, environment, temporal neighbors).

        Returns a flat dict ready to be returned to the LLM.
        """
        return episode.build_episode(self, activity_record)

    def _pattern_lexical_term_bonus(self, query: str, doc: Dict[str, Any]) -> float:
        return inferred.pattern_lexical_term_bonus(self, query, doc)

    def _pattern_choice_overlap_bonus(self, query: str, doc: Dict[str, Any]) -> float:
        return inferred.pattern_choice_overlap_bonus(self, query, doc)

    def _pattern_evidence_strength_bonus(self, doc: Dict[str, Any]) -> float:
        return inferred.pattern_evidence_strength_bonus(self, doc)

    def _pattern_source_priority_bonus(self, doc: Dict[str, Any], strategy_name: str = "") -> float:
        return inferred.pattern_source_priority_bonus(self, doc, strategy_name=strategy_name)

    def _pattern_rank_spec_for_query(self, query: str) -> Dict[str, Any]:
        return inferred.pattern_rank_spec_for_query(self, query)

    def _scale_signed_source_bonus(self, bonus: float, spec: Dict[str, Any]) -> float:
        return inferred.scale_signed_source_bonus(self, bonus, spec)

    def _pattern_search_rank_score(self, query: str, doc: Dict[str, Any], raw_similarity: float) -> float:
        return inferred.pattern_search_rank_score(self, query, doc, raw_similarity)

    def _pattern_doc_allowed_for_question_type(self, doc: Dict[str, Any]) -> bool:
        return inferred.pattern_doc_allowed_for_question_type(self, doc)

    # ==================== Tool 4: search_patterns ====================
    def search_patterns(self, query: str, top_k: int = None, _skip_synonym_expansion: bool = False) -> List[Dict[str, Any]]:
        """
        Search inferred knowledge for patterns, preferences, and habits.
        """
        return inferred.search_patterns(
            self,
            query=query,
            top_k=top_k,
            _skip_synonym_expansion=_skip_synonym_expansion,
        )

    # ==================== Tool 5: get_state_at_time ====================
    def get_state_at_time(self, time_seconds: float) -> Dict[str, Any]:
        """
        Get complete state snapshot at a specific time.

        Uses state_history to reconstruct historical object states.

        Args:
            time_seconds: The time point in seconds (e.g., 0.0, 30.5, 120.0)

        Returns:
            Dict with:
            - time: The requested time
            - visible_objects: List of objects visible at this time (with HISTORICAL state/location)
            - environment: Environment state at this time
            - current_activity: What was happening at this time
        """
        return entity.get_state_at_time(self, time_seconds=time_seconds)

    def _find_state_at_time(self, obj_data: Dict, time_seconds: float) -> Dict[str, Any]:
        """
        Find the object's state at a specific time using state_history.

        Logic: Find the LATEST history entry with time <= requested time.
        If no state_history, fall back to current state.

        Args:
            obj_data: Object data from object_registry
            time_seconds: Time point to query

        Returns:
            Dict with 'state', 'location', and '_from_history' flag
        """
        return entity.find_state_at_time(obj_data, time_seconds=time_seconds)

    # ==================== Tool 5b: get_object_history ====================
    def get_object_history(self, object_query: str) -> Dict[str, Any]:
        """
        Get COMPLETE history of an object's states and locations.

        Designed for object-state and object-location recall questions like:
        - "When did the spatula move to the dish rack?"
        - "Was the plate ever dirty?"
        - "Did the cup move during the video?"
        - "What states has the fork been in?"

        Args:
            object_query: Object name or ID to track (e.g., "plate", "cup", "fork", "spatula")

        Returns:
            Dict with:
            - object_id: Matched object ID
            - name: Object name
            - current_state: Current state
            - current_location: Current location
            - all_states_observed: List of ALL states seen (for "Was X ever Y?" questions)
            - all_locations_observed: List of ALL locations seen
            - state_history: Full timeline of state/location changes with timestamps
        """
        return entity.get_object_history(self, object_query=object_query)

    # ==================== Planning Tool: get_routine_skill ====================
    def get_routine_skill(self, goal_query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve consolidated procedure templates by semantic match on `goal`.

        Planning-specific entry point into
        ``inferred_knowledge.reusable_procedures.procedure_templates``. Unlike
        ``search_patterns`` (which dumps every pattern source into a flat list and
        truncates ``canonical_steps`` to a 5-step ``" -> ".join`` preview), this
        tool returns the **full structured routine** so a planning agent can
        consume ``canonical_steps`` directly as a plan skeleton.

        Args:
            goal_query: natural-language goal (e.g. "wash dishes", "make tea",
                        "peel and chop vegetables").
            top_k:     how many top-matching routines to return (default 3).

        Returns:
            {
              "query": goal_query,
              "num_matched": N,
              "routines": [
                {
                  "goal": "...",
                  "canonical_steps": [
                      {"action": "...", "object": "...", "hand": "...",
                       "direction": "..."},
                      ...
                  ],
                  "key_objects": [{"object": "...", ...}, ...],
                  "supporting_episodes": [...],
                  "count": int,
                  "similarity": float,
                  "source": "procedure_template"
                },
                ...
              ]
            }

        Returns ``{"error": ...}`` if no procedure templates are available
        (e.g. memory without offline consolidation). The agent should fall back to
        ``search_activities`` for raw episode evidence in that case.
        """
        return planning.get_routine_skill(self, goal_query=goal_query, top_k=top_k)

    # ==================== Planning Tool: get_preferences ====================
    def get_preferences(self, query: str = "", top_k: int = 5) -> Dict[str, Any]:
        """
        Retrieve this participant's observed preferences and habits.

        Reads ``inferred_knowledge.preferences.statements``: concise statements
        abstracted during offline consolidation, with linked supporting episodes.
        Planning-specific: returns at most ``top_k`` entries ranked
        by relevance to ``query`` (or by confidence when ``query`` is empty).

        Args:
            query:  optional focus (e.g. "where knives go", "post-meal
                    cleanup"). Empty -> return the most confident preferences.
            top_k:  max number of preferences to return.

        Returns:
            {
              "query": query,
              "num_matched": N,
              "preferences": [
                {"preference": "...", "evidence_summary": "...",
                 "subtype": "...", "confidence": float,
                 "supporting_episodes": [...], "similarity": float,
                 "source": "consolidated_preference"},
                ...
              ]
            }
        """
        return planning.get_preferences(self, query=query, top_k=top_k)

    # ==================== Activity Context Enrichment ====================

    # ==================== UNIFIED SEARCH (4-Category + Reconstruction) ====================
    _SEARCH_CATEGORIES = ("objects", "activities", "environment", "patterns")

    def search(self, query: str, top_k_per_category: int = 3, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Unified search across ALL memory categories, or a single category.

        Activities now include scene reconstruction (_context has objects_involved
        from object_registry + environment from environment_log).
        """
        top_k_per_category = min(max(int(top_k_per_category or 3), 1), 3)
        if category is not None:
            category = "objects" if category == "object_registry" else category
            category = "activities" if category == "activity_log" else category
            if category not in self._SEARCH_CATEGORIES:
                logger.warning(f"    Unknown search category '{category}', ignoring (search all)")
                category = None

        cache_key = (
            query,
            self.current_video_id,
            self.context_type,
            self.current_question_type,
            category or "all",
            top_k_per_category,
        )
        if cache_key in self.query_cache:
            cached = self.query_cache[cache_key].copy()
            if self.include_tips:
                cached["_cached"] = True
                cached["_note"] = "This query was already executed. Try a different search term or approach."
            logger.info(f"   Returning cached results for: '{query}'")
            return cached

        results = {"query": query, "objects": [], "activities": [], "environment": [], "patterns": []}

        if category is not None:
            k = top_k_per_category
            if category == "objects":
                obj_results = self.search_objects(query, top_k=k, _skip_synonym_expansion=True)
                if obj_results and not (len(obj_results) == 1 and "error" in obj_results[0]):
                    results["objects"] = obj_results[:k]
            elif category == "activities":
                act_results = self.search_activities(query, top_k=k, _skip_synonym_expansion=True)
                if act_results and not (len(act_results) == 1 and "error" in act_results[0]):
                    results["activities"] = [self._episode_to_context_format(a) for a in act_results[:k]]
            elif category == "environment":
                env_results = self.search_environment(query, top_k=k, _skip_synonym_expansion=True)
                if env_results and not (len(env_results) == 1 and "error" in env_results[0]):
                    results["environment"] = env_results[:k]
            elif category == "patterns":
                pat_results = self.search_patterns(query, top_k=k, _skip_synonym_expansion=True)
                if pat_results and not (len(pat_results) == 1 and "error" in pat_results[0]):
                    results["patterns"] = pat_results[:k]
            self.query_cache[cache_key] = results.copy()
            return results

        is_preference_query = self._is_preference_query(query)
        is_action_query = self._is_action_query(query)

        if is_preference_query:
            logger.info(f"    PREFERENCE query detected: '{query}' - prioritizing PATTERNS")
        elif is_action_query:
            logger.info(f"    Action query detected: '{query}' - prioritizing activities")

        act_top_k = top_k_per_category * 2 if is_action_query else top_k_per_category
        obj_top_k = top_k_per_category
        pat_boost = 3 if is_preference_query else 1

        act_results = self.search_activities(query, top_k=act_top_k, _skip_synonym_expansion=True)
        if act_results and not (len(act_results) == 1 and "error" in act_results[0]):
            results["activities"] = [self._episode_to_context_format(a) for a in act_results[:act_top_k]]

        obj_results = self.search_objects(query, top_k=obj_top_k, _skip_synonym_expansion=True)
        if obj_results and not (len(obj_results) == 1 and "error" in obj_results[0]):
            if is_action_query:
                obj_results = self._filter_object_noise(obj_results, query)
            results["objects"] = obj_results[:obj_top_k]

        env_results = self.search_environment(query, top_k=top_k_per_category, _skip_synonym_expansion=True)
        if env_results and not (len(env_results) == 1 and "error" in env_results[0]):
            results["environment"] = env_results[:top_k_per_category]

        pat_top_k = top_k_per_category * pat_boost
        pat_results = self.search_patterns(query, top_k=pat_top_k, _skip_synonym_expansion=True)
        if pat_results and not (len(pat_results) == 1 and "error" in pat_results[0]):
            results["patterns"] = pat_results[:pat_top_k]

        if self.include_tips:
            if is_action_query and not results["activities"]:
                results["_note"] = "No matching activities found."
        if is_preference_query and self.include_tips:
            results["_preference_guidance"] = {
                "message": "This is a PREFERENCE/HABIT question. Focus on 'patterns' first.",
                "priority": "patterns > activities > objects > environment"
            }

        total_results = sum(len(results[k]) for k in ["objects", "activities", "environment", "patterns"])
        if total_results == 0:
            logger.info(f"    No results for '{query}', trying synonym expansion...")
            for expanded_query in self._expand_query_with_synonyms(query)[1:]:
                logger.info(f"    Trying synonym: '{query}' → '{expanded_query}'")
                act_results = self.search_activities(expanded_query, top_k=act_top_k, _skip_synonym_expansion=True)
                if act_results and not (len(act_results) == 1 and "error" in act_results[0]):
                    results["activities"] = [self._episode_to_context_format(a) for a in act_results[:act_top_k]]
                obj_results = self.search_objects(expanded_query, top_k=obj_top_k, _skip_synonym_expansion=True)
                if obj_results and not (len(obj_results) == 1 and "error" in obj_results[0]):
                    if is_action_query:
                        obj_results = self._filter_object_noise(obj_results, expanded_query)
                    results["objects"] = obj_results[:obj_top_k]
                env_results = self.search_environment(expanded_query, top_k=top_k_per_category, _skip_synonym_expansion=True)
                if env_results and not (len(env_results) == 1 and "error" in env_results[0]):
                    results["environment"] = env_results[:top_k_per_category]
                pat_results = self.search_patterns(expanded_query, top_k=top_k_per_category, _skip_synonym_expansion=True)
                if pat_results and not (len(pat_results) == 1 and "error" in pat_results[0]):
                    results["patterns"] = pat_results[:top_k_per_category]
                total_results = sum(len(results[k]) for k in ["objects", "activities", "environment", "patterns"])
                if total_results > 0:
                    if self.include_tips:
                        results["_synonym_used"] = expanded_query
                    break

        total_found = sum(len(results[k]) for k in ["objects", "activities", "environment", "patterns"])

        if results["objects"] and len(results["objects"]) > 1:
            base_names: dict = {}
            for obj in results["objects"]:
                bn = normalize_object_name(obj.get("name", ""))
                base_names.setdefault(bn, []).append(obj)
            for bn, objs in base_names.items():
                if len(objs) > 1:
                    candidates = []
                    for o in objs:
                        vp = o.get("visual_properties", {})
                        sp = o.get("spatial_info", {})
                        c = str(vp.get("color", "")).strip() if isinstance(vp, dict) else ""
                        m = str(vp.get("material", "")).strip() if isinstance(vp, dict) else ""
                        loc = str(sp.get("location", "")).strip() if isinstance(sp, dict) else ""
                        base = strip_attributes_from_name(o.get("name", ""), c, m)
                        desc = " ".join(p for p in (c, m, base) if p)
                        if loc:
                            desc += f" (at {loc})"
                        candidates.append(desc)
                    results.setdefault("_disambiguation", {})[bn] = {
                        "message": f"Multiple {bn}s found. Pick the one matching your task.",
                        "candidates": candidates
                    }

        if self.include_tips:
            results["_summary"] = {
                "objects_found": len(results["objects"]),
                "activities_found": len(results["activities"]),
                "environment_found": len(results["environment"]),
                "patterns_found": len(results["patterns"]),
                "total": total_found
            }
            if total_found == 0:
                suggestions = []
                query_words = query.lower().split()
                if len(query_words) > 1:
                    suggestions.append(f"Try searching for individual words: {', '.join(query_words)}")
                suggestions.append("Try synonyms (e.g., 'tray' for 'baking tray', 'pan' for 'frying pan')")
                results["_no_results_guidance"] = {"message": f"No results found for '{query}'.", "suggestions": suggestions}

        if is_preference_query and results.get("patterns"):
            reordered = {"query": results["query"]}
            if "_preference_guidance" in results:
                reordered["_preference_guidance"] = results["_preference_guidance"]
            reordered["patterns"] = results["patterns"]
            for key in ["activities", "objects", "environment"]:
                if results.get(key):
                    reordered[key] = results[key]
            for key in results:
                if key not in reordered:
                    reordered[key] = results[key]
            results = reordered

        self.query_cache[cache_key] = results.copy()
        return results

    # ==================== Tool Definitions ====================
    def get_tools_definition(self, allow_category: bool = False) -> List[Dict[str, Any]]:
        """Get OpenAI-compatible tool definitions."""
        return build_tools_definition(allow_category=allow_category)

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool by name"""
        # === PRIMARY TOOLS ===
        if tool_name == "search":
            return self.search(
                arguments.get("query", ""),
                top_k_per_category=arguments.get("top_k_per_category", 3),
                category=arguments.get("category"),
            )
        elif tool_name == "get_state_at_time":
            return self.get_state_at_time(arguments.get("time_seconds", 0.0))
        elif tool_name == "get_object_history":
            return self.get_object_history(arguments.get("object_query", ""))
        elif tool_name == "get_routine_skill":
            return self.get_routine_skill(
                arguments.get("goal_query", arguments.get("query", "")),
                top_k=int(arguments.get("top_k", 3)),
            )
        elif tool_name == "get_preferences":
            return self.get_preferences(
                arguments.get("query", ""),
                top_k=int(arguments.get("top_k", 5)),
            )
        elif tool_name == "get_video_summary":
            return self.get_video_summary(arguments.get("video_id"))
        elif tool_name == "get_video_activities":
            return self.get_video_activities(
                arguments.get("video_id"),
                compact=arguments.get("compact", True),
                max_activities=arguments.get("max_activities"),
            )
        elif tool_name == "get_narrative_evidence":
            return self.get_narrative_evidence(arguments.get("query", ""))
        elif tool_name == "get_local_narrative":
            return self.get_local_narrative(
                arguments.get("time_seconds"),
                arguments.get("window", 30.0),
                arguments.get("video_id"),
            )
        elif tool_name == "get_semantic_evidence":
            return self.get_semantic_evidence(
                arguments.get("query", ""),
                arguments.get("candidate_a", ""),
                arguments.get("candidate_b", ""),
                arguments.get("context_query", ""),
                arguments.get("trigger_query", ""),
                arguments.get("candidates"),
                arguments.get("choice_a", ""),
                arguments.get("choice_b", ""),
                arguments.get("choice_c", ""),
                arguments.get("choice_d", ""),
            )
        elif tool_name == "count_object_uses":
            return self.count_object_uses(
                arguments.get("object_query", ""),
                arguments.get("context_query", ""),
            )
        elif tool_name == "find_action_transitions":
            return self.find_action_transitions(
                arguments.get("trigger_query", ""),
                arguments.get("candidate_a", ""),
                arguments.get("candidate_b", ""),
                candidates=arguments.get("candidates"),
            )
        elif tool_name == "compare_objects":
            return self.compare_objects(
                arguments.get("query_a", ""),
                arguments.get("query_b", ""),
                arguments.get("context_query", ""),
            )

        # === ADDITIONAL TYPED SEARCH ENTRY POINTS ===
        # These are available for direct tool calls but are not advertised in
        # the compact default prompt.
        elif tool_name == "search_objects":
            return self.search_objects(arguments.get("query", ""))
        elif tool_name == "search_environment":
            return self.search_environment(arguments.get("query", ""))
        elif tool_name == "search_activities":
            return self.search_activities(arguments.get("query", ""))
        elif tool_name == "search_patterns":
            return self.search_patterns(arguments.get("query", ""))
        else:
            return {"error": f"Unknown tool: {tool_name}"}
