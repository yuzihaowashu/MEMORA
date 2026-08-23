"""Memory loading, temporal reconstruction, and participant scoping."""

import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional

from memora.memory_agent.tools.formatting import activity_time_bounds
from memora.memory_agent.tools.stores.entity_normalization import deduplicate_objects

logger = logging.getLogger(__name__)


class MemoryContextMixin:
    """Context lifecycle shared by the typed-memory retrieval facade."""

    def _load_participant_memory(self, file_path: str) -> Dict[str, Any]:
        """
        Load memory data from either:
        - participant_memory.json (WITH memory editing - aggregated)
        - segment_observations.jsonl (WITHOUT memory editing - per-segment)
        """
        # The paper's no-memory EAM-QA condition uses the same agent and tool
        # interface over an empty memory state.
        if not file_path:
            return {}
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Participant memory file not found: {file_path}")

        logger.debug("Loading participant memory file: %s", file_path)

        # Check file extension to determine format
        if file_path.endswith('.jsonl'):
            # Per-segment format (WITHOUT memory editing)
            # Aggregate segments into the runtime memory structure
            return self._load_from_segments(file_path)
        else:
            # Aggregated format (WITH memory editing)
            with open(file_path, 'r') as f:
                data = json.load(f)

            if "memories_by_video" in data:
                result = data["memories_by_video"]
                self.inferred_knowledge = data.get("inferred_knowledge", {})
                return result
            return data

    def _load_from_segments(self, file_path: str) -> Dict[str, Any]:
        """
        Load and aggregate segment_observations.jsonl into runtime memory format.

        This simulates "WITHOUT memory editing" - just accumulate all data
        without intelligent ADD/UPDATE/DELETE processing.
        """
        logger.info("Loading raw per-segment memory file: %s", file_path)

        # Count total lines first for progress
        with open(file_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        participant_memory = {}

        with open(file_path, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                if line_num % 100 == 0:
                    logger.debug("Parsed %d/%d raw memory lines", line_num, total_lines)

                if not line.strip():
                    continue
                try:
                    segment = json.loads(line)
                    if not isinstance(segment, dict):
                        raise ValueError("segment is not a JSON object")
                    video_id = segment.get('video_id', 'unknown')
                    turn_id = segment.get('turn_id', 0)
                    time_window = segment.get('time_window', {})

                    # Initialize video entry if needed
                    if video_id not in participant_memory:
                        participant_memory[video_id] = {
                            "environment_log": [],
                            "object_registry": {},
                            "activity_log": [],
                            "inferred_knowledge": {},
                        }

                    memory = participant_memory[video_id]

                    # 1. Accumulate environment records without merging.
                    env = segment.get('environment', {})
                    if env:
                        env_entry = {
                            "location_id": f"segment_{turn_id}",
                            "turn_id": turn_id,
                            "first_seen": time_window.get('start', 0),
                            "last_seen": time_window.get('end', 0),
                            "current_state": env
                        }
                        memory["environment_log"].append(env_entry)

                    # 2. Accumulate exact-ID observations without semantic editing.
                    # State history remains available for time-restricted retrieval.
                    obj_reg = segment.get('object_registry', {})
                    if isinstance(obj_reg, dict):
                        for obj_id, obj_data in obj_reg.items():
                            if not isinstance(obj_data, dict):
                                raise ValueError(
                                    f"object {obj_id!r} is {type(obj_data).__name__}, not a JSON object"
                                )
                            obj_data_copy = copy.deepcopy(obj_data)
                            observed_at = time_window.get('start', turn_id * 10)
                            state_data = obj_data_copy.get('state', {})
                            spatial_data = obj_data_copy.get('spatial_info', {})
                            history_entry = {
                                'turn_id': turn_id,
                                'time_seconds': observed_at,
                                'time': observed_at,
                                'state': state_data.get('current_state', '') if isinstance(state_data, dict) else state_data,
                                'location': spatial_data.get('location', '') if isinstance(spatial_data, dict) else '',
                            }
                            previous = memory["object_registry"].get(obj_id)
                            if isinstance(previous, dict):
                                history = copy.deepcopy(previous.get('state_history', []))
                                first_seen_time = previous.get('first_seen_time', observed_at)
                                first_seen_turn = previous.get('first_seen_turn', turn_id)
                                merged = copy.deepcopy(previous)
                                merged.update(obj_data_copy)
                                obj_data_copy = merged
                            else:
                                history = []
                                first_seen_time = observed_at
                                first_seen_turn = turn_id
                            history.append(history_entry)
                            history.sort(key=lambda item: float(item.get('time_seconds', item.get('time', 0)) or 0))
                            obj_data_copy['state_history'] = history
                            obj_data_copy['first_seen_time'] = first_seen_time
                            obj_data_copy['first_seen_turn'] = first_seen_turn
                            obj_data_copy['last_seen_turn'] = turn_id
                            obj_data_copy['last_seen_time'] = time_window.get('end', 0)
                            memory["object_registry"][obj_id] = obj_data_copy

                    # 3. Accumulate activities
                    activity = segment.get('activity_narrative', {})
                    if activity:
                        activity_entry = {
                            "turn_id": turn_id,
                            "time_window": time_window,
                            "summary": activity.get('summary', ''),
                            "detailed_narrative": activity.get('detailed_narrative', ''),
                            "action_breakdown": activity.get('action_breakdown', [])
                        }
                        memory["activity_log"].append(activity_entry)

                except Exception as e:
                    raise ValueError(
                        f"Invalid raw segment at {file_path}:{line_num}: "
                        f"{type(e).__name__}: {e}"
                    ) from e

        logger.info(f"   Loaded {len(participant_memory)} videos from raw segments")

        for vid, data in list(participant_memory.items())[:3]:
            logger.info(f"   - {vid}: {len(data['object_registry'])} objects, {len(data['activity_log'])} activities")

        return participant_memory

    def _rebuild_memory_at_time(
        self,
        video_id: str,
        time_threshold_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Rebuild memory state at a specific time using history.

        For environment_log and object_registry, we use the history to
        reconstruct what the memory looked like at that time.
        """
        if video_id not in self.participant_memory:
            return {
                "environment_log": [],
                "object_registry": {},
                "activity_log": [],
                "inferred_knowledge": {},
            }

        full_memory = self.participant_memory[video_id]

        if time_threshold_seconds is None or time_threshold_seconds == float('inf'):
            # Return current state but WITHOUT patterns for episodic context
            # Event-recall questions should focus on activities, not generalized patterns.
            result = copy.deepcopy(full_memory)
            result["inferred_knowledge"] = {}  # Clear inferred knowledge for episodic
            return result

        # Rebuild state at specific time
        # Temporal and event-recall contexts exclude consolidated patterns so
        # answers stay grounded in episode-specific evidence.
        rebuilt = {
            "environment_log": [],
            "object_registry": {},
            "activity_log": [],
            "inferred_knowledge": {},
        }

        # 1. Rebuild environment_log from history
        for env in full_memory.get("environment_log", []):
            first_seen = env.get("first_seen", 0)
            if first_seen <= time_threshold_seconds:
                # Find the state at this time from history
                history = env.get("history", [])
                reconstructed_state = None

                for entry in history:
                    entry_time = entry.get("time", 0)
                    if entry_time > time_threshold_seconds:
                        break

                    if entry.get("event") == "ADD":
                        reconstructed_state = copy.deepcopy(entry.get("snapshot", {}))
                    elif entry.get("event") == "UPDATE" and reconstructed_state:
                        # Apply changes
                        changes = entry.get("changes", {})
                        if "layout_description" in changes:
                            reconstructed_state["layout_description"] = changes["layout_description"]
                        if "features_added" in changes:
                            existing = set(reconstructed_state.get("features", []))
                            reconstructed_state["features"] = list(existing | set(changes["features_added"]))

                if reconstructed_state:
                    rebuilt_env = {
                        "location_id": env.get("location_id"),
                        "first_seen": first_seen,
                        "last_seen": min(env.get("last_seen", time_threshold_seconds), time_threshold_seconds),
                        "current_state": reconstructed_state
                    }
                    rebuilt["environment_log"].append(rebuilt_env)

        # 2. Filter activity_log by time
        for activity_record in full_memory.get("activity_log", []):
            activity_start, _ = activity_time_bounds(activity_record)
            if activity_start <= time_threshold_seconds:
                rebuilt["activity_log"].append(copy.deepcopy(activity_record))

        # 3. Object registry - filter by time (only include objects seen before threshold)
        # Objects may have first_seen_time, last_seen_time, first_seen, last_seen fields
        for obj_id, obj_data in full_memory.get("object_registry", {}).items():
            # Get first seen time (try multiple field names)
            first_seen = (
                obj_data.get("first_seen_time") or
                obj_data.get("first_seen") or
                0
            )

            # Include object if it was first seen before the threshold
            if first_seen <= time_threshold_seconds:
                obj_copy = copy.deepcopy(obj_data)

                if "history" in obj_copy:
                    obj_copy["history"] = [
                        copy.deepcopy(entry)
                        for entry in obj_copy["history"]
                        if float(entry.get("time", entry.get("time_seconds", 0)) or 0)
                        <= time_threshold_seconds
                    ]

                # Update last_seen to be at most the threshold
                if "last_seen_time" in obj_copy:
                    obj_copy["last_seen_time"] = min(obj_copy["last_seen_time"], time_threshold_seconds)
                if "last_seen" in obj_copy:
                    obj_copy["last_seen"] = min(obj_copy["last_seen"], time_threshold_seconds)

                rebuilt["object_registry"][obj_id] = obj_copy

        return rebuilt

    def set_temporal_context(
        self,
        video_id: str,
        ask_turn_id: Optional[int] = None,
        time_threshold_seconds: Optional[float] = None
    ):
        """Set context for episodic questions (single video, specific time)

        ``turn_id`` is a zero-based 10-second segment index. When an explicit
        timestamp is unavailable, retrieval includes evidence through the end
        of that segment: ``(turn_id + 1) * 10`` seconds.
        """
        # Clear query cache on context change
        self.query_cache = {}

        if ask_turn_id is not None and time_threshold_seconds is None:
            time_threshold_seconds = (ask_turn_id + 1) * 10
            logger.info(
                "turn_id %d → time threshold %.1fs (end of segment)",
                ask_turn_id,
                time_threshold_seconds,
            )

        self.context_type = "episodic"
        self.current_video_id = video_id
        self.time_threshold = time_threshold_seconds
        self.participant_videos = None

        # Rebuild memory at this time
        self.current_memory = self._rebuild_memory_at_time(video_id, time_threshold_seconds)

        time_label = "full video" if time_threshold_seconds is None else f"time<={time_threshold_seconds}s"
        logger.info("Temporal context: video=%s, %s", video_id, time_label)
        logger.info(f"  Objects: {len(self.current_memory.get('object_registry', {}))}")
        logger.info(f"  Environments: {len(self.current_memory.get('environment_log', []))}")
        logger.info(f"  Activities: {len(self.current_memory.get('activity_log', []))}")

    def set_participant_context(
        self,
        participant_id: str,
        video_ids: Optional[List[str]] = None
    ):
        """Set context for semantic questions (cross-video aggregation)"""
        # Clear query cache on context change
        self.query_cache = {}

        if video_ids:
            self.participant_videos = set(video_ids)
        else:
            self.participant_videos = set(
                vid for vid in self.participant_memory.keys()
                if vid.startswith(participant_id)
            )

        self.context_type = "semantic"
        self.current_video_id = None
        self.time_threshold = None

        # Aggregate memory across all participant videos
        self.current_memory = {
            "environment_log": [],
            "object_registry": {},
            "activity_log": [],
            "inferred_knowledge": copy.deepcopy(self.inferred_knowledge),
        }

        for video_id in sorted(self.participant_videos):
            video_memory = self.participant_memory.get(video_id, {})

            # Merge environments (with video_id prefix)
            for env in video_memory.get("environment_log", []):
                env_copy = copy.deepcopy(env)
                env_copy["source_video"] = video_id
                self.current_memory["environment_log"].append(env_copy)

            # Merge objects (with video_id prefix)
            for obj_id, obj_data in video_memory.get("object_registry", {}).items():
                prefixed_id = f"{video_id}_{obj_id}"
                # Handle both dict and list formats for obj_data
                if isinstance(obj_data, dict):
                    obj_copy = copy.deepcopy(obj_data)
                    obj_copy["source_video"] = video_id
                    self.current_memory["object_registry"][prefixed_id] = obj_copy
                elif isinstance(obj_data, list):
                    # If obj_data is a list, wrap it with source_video
                    self.current_memory["object_registry"][prefixed_id] = {
                        "instances": obj_data,
                        "source_video": video_id
                    }
                else:
                    # Fallback for other types
                    self.current_memory["object_registry"][prefixed_id] = {
                        "data": obj_data,
                        "source_video": video_id
                    }

            # Merge activities
            for act in video_memory.get("activity_log", []):
                act_copy = copy.deepcopy(act)
                act_copy["source_video"] = video_id
                self.current_memory["activity_log"].append(act_copy)


        knowledge = self.current_memory.get("inferred_knowledge", {})
        prefs = knowledge.get("preferences", {})
        storage_prefs = len(prefs.get("storage_preferences", []))
        org_habits = len(prefs.get("organizational_habits", []))
        workflow_patterns = len(prefs.get("workflow_patterns", []))
        action_seqs = len(knowledge.get("action_sequences", []))
        logger.debug(
            "Participant context %s: videos=%s objects=%d environments=%d patterns=(storage=%d habits=%d workflow=%d actions=%d)",
            participant_id,
            self.participant_videos,
            len(self.current_memory.get("object_registry", {})),
            len(self.current_memory.get("environment_log", [])),
            storage_prefs,
            org_habits,
            workflow_patterns,
            action_seqs,
        )

        raw_obj_count = len(self.current_memory.get("object_registry", {}))
        deduplicate_objects(self.current_memory)
        dedup_obj_count = len(self.current_memory.get("object_registry", {}))
        logger.debug("Object deduplication: %d -> %d", raw_obj_count, dedup_obj_count)

        logger.info(f"Participant context: {participant_id}, videos={self.participant_videos}")
        logger.info(f"  Objects: {dedup_obj_count} (raw: {raw_obj_count})")
        logger.info(f"  Environments: {len(self.current_memory.get('environment_log', []))}")
        logger.info(
            "  Inferred Knowledge: storage=%d, habits=%d, workflow=%d, actions=%d",
            storage_prefs,
            org_habits,
            workflow_patterns,
            action_seqs,
        )
