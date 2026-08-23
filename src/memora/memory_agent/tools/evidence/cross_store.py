"""Cross-store episodic and semantic evidence views."""

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from memora.memory_agent.tools.evidence import semantic
from memora.memory_agent.tools.stores import activity
from memora.memory_agent.tools.formatting import (
    QUERY_STOPWORDS,
    activity_text,
    activity_time_bounds,
    compact_action_breakdown,
    compact_activity,
    compact_goal_activity_view,
    compact_semantic_activity,
    episode_to_context_format,
    query_terms,
    query_video_ids,
    semantic_activity_text,
    shorten_text,
)


class CrossStoreEvidenceMixin:
    """Derived evidence views composed from the four memory stores."""

    def _episode_to_context_format(self, episode_record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a retrieved episode into compact activity-context format."""
        return episode_to_context_format(episode_record)

    # ==================== Narrative and Aggregation Tools ====================
    def _activity_time_bounds(self, activity_record: Dict[str, Any]) -> Tuple[float, float]:
        """Extract numeric start/end seconds from an activity."""
        return activity_time_bounds(activity_record)

    def _activity_text(self, activity_record: Dict[str, Any]) -> str:
        """Build a searchable narrative string from one activity."""
        return activity_text(activity_record, current_video_id=self.current_video_id)

    def _semantic_activity_text(self, activity_record: Dict[str, Any]) -> str:
        """Build text for semantic aggregation without expanded retrieval duplication."""
        return semantic_activity_text(activity_record, current_video_id=self.current_video_id)

    _QUERY_STOPWORDS = QUERY_STOPWORDS

    def _query_terms(self, query: str) -> List[str]:
        """Tokenize a query and keep terms useful for narrative matching."""
        return query_terms(query)

    def _query_video_ids(self, query: str) -> List[str]:
        """Extract EPIC-style video IDs from a query, e.g. P01_104."""
        return query_video_ids(query)

    def _shorten_text(self, text: Any, limit: int = 220) -> str:
        """Collapse whitespace and truncate long text for compact tool output."""
        return shorten_text(text, limit)

    def _has_expanded_activity_view(self) -> bool:
        if not self.current_memory:
            return False
        return any(
            isinstance(activity.get("goal_activity_view"), dict)
            for activity in self.current_memory.get("activity_log", [])
            if isinstance(activity, dict)
        )

    def _compact_action_breakdown(
        self,
        breakdown: Any,
        max_steps: int = 2,
    ) -> List[Dict[str, Any]]:
        """Keep only the fields needed for reasoning about action sequences."""
        return compact_action_breakdown(breakdown, max_steps=max_steps)

    def _compact_goal_activity_view(
        self,
        view: Dict[str, Any],
        max_activities: int = 5,
        focus_turn_id: Any = None,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the goal/sub-goal/activity layer used by expanded memory outputs."""
        return compact_goal_activity_view(
            view,
            max_activities=max_activities,
            focus_turn_id=focus_turn_id,
            query=query,
        )

    def _compact_activity(self, activity_record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact activity snippet suitable for tool output."""
        return compact_activity(activity_record, current_video_id=self.current_video_id)

    def _compact_semantic_activity(self, activity_record: Dict[str, Any]) -> Dict[str, Any]:
        """Compact support example for semantic aggregation without episodic goal views."""
        return compact_semantic_activity(activity_record, current_video_id=self.current_video_id)

    def _activity_overlaps_window(
        self,
        activity_record: Dict[str, Any],
        window_start: float,
        window_end: float,
    ) -> bool:
        """Return whether an activity overlaps a time window."""
        return activity.activity_overlaps_window(
            self,
            activity_record,
            window_start,
            window_end,
        )

    def _objects_in_time_window(
        self,
        center_time: Optional[float] = None,
        window: float = 30.0,
        max_objects: int = 12,
    ) -> List[Dict[str, Any]]:
        """Return objects mentioned or observed near a time window."""
        return activity.objects_in_time_window(
            self,
            center_time=center_time,
            window=window,
            max_objects=max_objects,
        )

    def get_video_summary(self, video_id: Optional[str] = None, max_activities: Optional[int] = None) -> Dict[str, Any]:
        """
        Return a video-level narrative assembled from activity summaries.

        Current participant-memory files do not reliably store a pre-written video_summary field, so
        this tool builds one from activity_log entries already in memory.
        """
        return activity.get_video_summary(
            self,
            video_id=video_id,
            max_activities=max_activities,
        )

    def get_video_activities(
        self,
        video_id: Optional[str] = None,
        compact: bool = True,
        max_activities: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return the chronological activity stream for a single video.

        Unlike `get_video_summary` (which keeps only ~6 evenly-spaced snippets so
        it fits in a short tool reply), this tool returns every activity for the
        named video so event-recall questions can be answered without
        relying on what the truncation happens to retain.

        - `compact=True` (default): one summary line per activity.
        - `compact=False`: full compact-activity payload (summary, detailed
          narrative, action_breakdown, local_event, goal_activity_view).
        """
        return activity.get_video_activities(
            self,
            video_id=video_id,
            compact=compact,
            max_activities=max_activities,
        )

    def get_local_narrative(
        self,
        time_seconds: Optional[float] = None,
        window: float = 30.0,
        video_id: Optional[str] = None,
        max_activities: int = 5,
    ) -> Dict[str, Any]:
        """
        Return local activity narrative and nearby objects around a timestamp.

        This exposes the episode neighborhood already present in memory, without
        pretending an untracked object was found in the entity registry.
        """
        return activity.get_local_narrative(
            self,
            time_seconds=time_seconds,
            window=window,
            video_id=video_id,
            max_activities=max_activities,
        )

    def get_narrative_evidence(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Search activity narratives directly for open-vocabulary objects/events.

        Use when object_registry misses an entity (trash bag, rolling pin, cling
        film) or for event-recall questions where text evidence is more useful
        than object trajectories.
        """
        return activity.get_narrative_evidence(self, query=query, top_k=top_k)

    def count_object_uses(
        self,
        object_query: str,
        context_query: str = "",
        top_k_examples: int = 4,
    ) -> Dict[str, Any]:
        """Count activity mentions of an object, optionally conditioned on context."""
        return semantic.count_object_uses(
            self,
            object_query=object_query,
            context_query=context_query,
            top_k_examples=top_k_examples,
        )

    def _matches_terms(self, text: str, terms: List[str]) -> Tuple[bool, int]:
        return semantic.matches_terms(self, text, terms)

    def _transition_next_text(
        self,
        activities_by_video: Dict[str, List[Dict[str, Any]]],
        video_id: str,
        activity_index: int,
        activity_record: Dict[str, Any],
    ) -> str:
        return semantic.transition_next_text(
            self,
            activities_by_video=activities_by_video,
            video_id=video_id,
            activity_index=activity_index,
            activity=activity_record,
        )

    def _candidate_support(self, text: str, candidate: str) -> int:
        return semantic.candidate_support(self, text, candidate)

    def _normalize_semantic_candidates(
        self,
        candidates: Any = None,
        candidate_a: str = "",
        candidate_b: str = "",
        choice_a: str = "",
        choice_b: str = "",
        choice_c: str = "",
        choice_d: str = "",
    ) -> List[Dict[str, str]]:
        return semantic.normalize_semantic_candidates(
            self,
            candidates=candidates,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
        )

    def _pick_candidate_winner(self, candidate_counts: Counter) -> str:
        return semantic.pick_candidate_winner(self, candidate_counts)

    def _score_all_candidates(
        self,
        candidates: List[Dict[str, str]],
        context_query: str,
    ) -> Dict[str, Any]:
        return semantic.score_all_candidates(
            self,
            candidates=candidates,
            context_query=context_query,
        )

    def find_action_transitions(
        self,
        trigger_query: str,
        candidate_a: str = "",
        candidate_b: str = "",
        top_k_examples: int = 4,
        candidates: Any = None,
    ) -> Dict[str, Any]:
        """Aggregate what usually happens after a trigger action."""
        return semantic.find_action_transitions(
            self,
            trigger_query=trigger_query,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            top_k_examples=top_k_examples,
            candidates=candidates,
        )

    def compare_objects(
        self,
        query_a: str,
        query_b: str,
        context_query: str = "",
    ) -> Dict[str, Any]:
        """Compare two object candidates under the same context."""
        return semantic.compare_objects(
            self,
            query_a=query_a,
            query_b=query_b,
            context_query=context_query,
        )

    def get_semantic_evidence(
        self,
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
        """Return typed evidence for semantic memory questions."""
        return semantic.get_semantic_evidence(
            self,
            query=query,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            context_query=context_query,
            trigger_query=trigger_query,
            candidates=candidates,
            choice_a=choice_a,
            choice_b=choice_b,
            choice_c=choice_c,
            choice_d=choice_d,
        )
