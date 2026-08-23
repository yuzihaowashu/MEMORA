"""
MEMORA typed memory processing.

Contains:
- TYPED_MEMORY_EDITOR_PROMPT
- EmbodiedMemoryState dataclass
- Rule-based operations (activity_log, environment)
- LLM-based operations (object_registry only)
- Helper functions: extract_location_id, merge_environment_features, etc.

Design:
  - Rule-based: activity_log (always ADD), environment (ADD/UPDATE)
  - LLM-based: object_registry (ADD/UPDATE/DELETE/NOOP) — single focused prompt
  - Offline consolidation: inferred_knowledge (via pipeline.consolidation)
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from memora.pipeline.formation_config import EPIC_KITCHENS_CONFIG
from memora.pipeline.memory_editor.prompts import build_memory_editor_prompt

logger = logging.getLogger(__name__)

# ============================================================================
# MEMORA Typed Memory Processing
# ============================================================================
# For MEMORA typed-memory format, we split operations:
#   - Rule-based: activity_log (always ADD), environment (ADD/UPDATE)
#   - LLM-based: object_registry (ADD/UPDATE/DELETE/NOOP) — single focused prompt
#   - Offline consolidation: inferred_knowledge (participant-level, not per-segment)
# ============================================================================

TYPED_MEMORY_EDITOR_PROMPT = build_memory_editor_prompt(EPIC_KITCHENS_CONFIG)


@dataclass
class EmbodiedMemoryState:
    """Online memory state for one video; Inferred Knowledge is consolidated later."""
    # ============================================================
    # Environment Log: List of environment entries by location
    # Each entry accumulates features seen at that location
    # ============================================================
    environment_log: List[Dict[str, Any]]  # List of {location_id, first_seen, last_seen, description, features}
    object_registry: Dict[str, Any]  # LLM manages (object_id -> object_data)
    activity_log: List[Dict[str, Any]]  # Append-only (rule-based)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment_log": self.environment_log,
            "object_registry": self.object_registry,
            "activity_log": self.activity_log,
        }

def extract_location_id(description: str, config=None) -> str:
    """Map a place description to its configured Environment Memory ID."""
    location_config = config or EPIC_KITCHENS_CONFIG
    return location_config.get_location_id(description)


def _normalize_zone(zone) -> Dict[str, Any]:
    """VLM outputs zones in inconsistent formats. Normalize to dict."""
    if isinstance(zone, dict):
        return zone
    if isinstance(zone, list):
        return {"contents": zone}
    if isinstance(zone, str):
        return {"contents": [zone]}
    return {}


def merge_environment_features(old_env: Dict[str, Any], new_env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two environment descriptions, accumulating features, zones, and spatial relations.
    """
    merged = old_env.copy()

    # 1. Merge features lists
    old_features = set(old_env.get("features", []))
    new_features = set(new_env.get("features", []))
    merged["features"] = list(old_features | new_features)

    # 2. Use newer/longer description
    if new_env.get("layout_description"):
        old_desc = old_env.get("layout_description", "")
        new_desc = new_env.get("layout_description", "")
        if len(new_desc) > len(old_desc):
            merged["layout_description"] = new_desc

    # 3. Merge zones (accumulate contents, keep best description)
    old_zones = old_env.get("zones", {})
    new_zones = new_env.get("zones", {})
    merged_zones = {}

    all_zone_names = set(old_zones.keys()) | set(new_zones.keys())
    for zone_name in all_zone_names:
        old_zone = _normalize_zone(old_zones.get(zone_name, {}))
        new_zone = _normalize_zone(new_zones.get(zone_name, {}))

        if old_zone and new_zone:
            old_contents = set(old_zone.get("contents", []))
            new_contents = set(new_zone.get("contents", []))
            merged_zones[zone_name] = {
                "anchor": new_zone.get("anchor") or old_zone.get("anchor"),
                "position": new_zone.get("position") or old_zone.get("position"),
                "contents": list(old_contents | new_contents),
                "description": new_zone.get("description") or old_zone.get("description")
            }
        else:
            merged_zones[zone_name] = new_zone or old_zone

    if merged_zones:
        merged["zones"] = merged_zones

    # 4. Merge spatial relations (accumulate unique relations)
    old_relations = set(old_env.get("spatial_relations", []))
    new_relations = set(new_env.get("spatial_relations", []))
    merged_relations = list(old_relations | new_relations)
    if merged_relations:
        merged["spatial_relations"] = merged_relations

    # 5. Update other fields with new values
    for key in ["lighting", "ambient", "key_locations"]:
        if new_env.get(key):
            merged[key] = new_env[key]

    return merged


def apply_rule_based_operations(
    current_memory: EmbodiedMemoryState,
    new_segment: Dict[str, Any],
    config=None,
) -> List[Dict[str, Any]]:
    """
    Apply rule-based operations for activity_log and environment_log.
    Returns a list of operation records.
    """
    operations = []
    turn_id = new_segment.get("turn_id", 0)
    time_window = new_segment.get("time_window", {})

    # 1. Activity Narrative -> ALWAYS ADD to activity_log
    activity_narrative = new_segment.get("activity_narrative", {})
    if activity_narrative:
        activity_entry = {
            "turn_id": turn_id,
            "time_window": time_window,
            "summary": activity_narrative.get("summary", ""),
            "detailed_narrative": activity_narrative.get("detailed_narrative", ""),
            "action_breakdown": activity_narrative.get("action_breakdown", []),
            "concurrent_actions": activity_narrative.get("concurrent_actions", [])
        }

        current_memory.activity_log.append(activity_entry)
        operations.append({
            "layer": "activity_log",
            "event": "ADD",
            "turn_id": turn_id,
            "summary": activity_narrative.get("summary", "")
        })

    # ============================================================
    # 2. Environment Log -> Rule-based ADD/UPDATE (never DELETE)
    # ============================================================
    new_environment = new_segment.get("environment", {})
    if new_environment:
        current_time = time_window.get("end", turn_id * 10)
        location_id = extract_location_id(
            new_environment.get("layout_description", ""),
            config=config,
        )

        existing_entry_idx = None
        for idx, entry in enumerate(current_memory.environment_log):
            if entry.get("location_id") == location_id:
                existing_entry_idx = idx
                break

        if existing_entry_idx is not None:
            old_entry = current_memory.environment_log[existing_entry_idx]
            old_state = old_entry.get("current_state", old_entry)
            try:
                merged_state = merge_environment_features(old_state, new_environment)
                changes = compute_environment_changes(old_state, merged_state)
            except Exception as e:
                logger.warning(f"    Environment merge failed ({type(e).__name__}: {e}), falling back to last_seen update only")
                current_memory.environment_log[existing_entry_idx]["last_seen"] = current_time
                current_memory.environment_log[existing_entry_idx]["turn_count"] = old_entry.get("turn_count", 1) + 1
                operations.append({
                    "layer": "environment_log",
                    "event": "UPDATE",
                    "location_id": location_id,
                    "status": "last_seen_only",
                    "error": f"{type(e).__name__}: {e}",
                })
                return operations

            history = old_entry.get("history", [])
            if changes:
                history.append({
                    "turn_id": turn_id,
                    "time": current_time,
                    "event": "UPDATE",
                    "changes": changes,
                })

            current_memory.environment_log[existing_entry_idx] = {
                "location_id": location_id,
                "first_seen": old_entry.get("first_seen", current_time),
                "last_seen": current_time,
                "turn_count": old_entry.get("turn_count", 1) + 1,
                "current_state": merged_state,
                "history": history,
            }
            operations.append({
                "layer": "environment_log",
                "event": "UPDATE",
                "location_id": location_id,
                "turn_count": old_entry.get("turn_count", 1) + 1,
                "changes_recorded": bool(changes),
            })
        else:
            current_state = {
                "layout_description": new_environment.get("layout_description", ""),
                "zones": new_environment.get("zones", {}),
                "spatial_relations": new_environment.get("spatial_relations", []),
                "features": new_environment.get("features", []),
                "lighting": new_environment.get("lighting", ""),
                "ambient": new_environment.get("ambient", ""),
            }
            current_memory.environment_log.append({
                "location_id": location_id,
                "first_seen": current_time,
                "last_seen": current_time,
                "turn_count": 1,
                "current_state": current_state,
                "history": [{
                    "turn_id": turn_id,
                    "time": current_time,
                    "event": "ADD",
                    "snapshot": copy.deepcopy(current_state),
                }],
            })
            operations.append({
                "layer": "environment_log",
                "event": "ADD",
                "location_id": location_id,
            })

    return operations


def compute_environment_changes(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute the differences between old and new environment states.

    Returns a dict describing what changed (for history tracking).
    """
    changes = {}

    # Layout description change
    old_layout = old_state.get("layout_description", "")
    new_layout = new_state.get("layout_description", "")
    if old_layout != new_layout:
        changes["layout_description"] = new_layout

    # Features changes
    old_features = set(old_state.get("features", []))
    new_features = set(new_state.get("features", []))
    features_added = new_features - old_features
    features_removed = old_features - new_features
    if features_added:
        changes["features_added"] = list(features_added)
    if features_removed:
        changes["features_removed"] = list(features_removed)

    # Spatial relations changes
    old_relations = set(old_state.get("spatial_relations", []))
    new_relations = set(new_state.get("spatial_relations", []))
    relations_added = new_relations - old_relations
    if relations_added:
        changes["spatial_relations_added"] = list(relations_added)

    # Zones changes
    old_zones = set(old_state.get("zones", {}).keys())
    new_zones = set(new_state.get("zones", {}).keys())
    zones_added = new_zones - old_zones
    zones_removed = old_zones - new_zones
    if zones_added:
        changes["zones_added"] = {k: new_state["zones"][k] for k in zones_added}
    if zones_removed:
        changes["zones_removed"] = list(zones_removed)

    # Check for modified zones (existing zones with changed contents)
    common_zones = old_zones & new_zones
    zones_modified = {}
    for zone_name in common_zones:
        old_zone = _normalize_zone(old_state.get("zones", {}).get(zone_name, {}))
        new_zone = _normalize_zone(new_state.get("zones", {}).get(zone_name, {}))
        old_contents = set(old_zone.get("contents", []))
        new_contents = set(new_zone.get("contents", []))
        contents_added = new_contents - old_contents
        if contents_added:
            zones_modified[zone_name] = {"contents_added": list(contents_added)}
    if zones_modified:
        changes["zones_modified"] = zones_modified

    # Lighting/ambient changes
    if old_state.get("lighting") != new_state.get("lighting") and new_state.get("lighting"):
        changes["lighting"] = new_state["lighting"]
    if old_state.get("ambient") != new_state.get("ambient") and new_state.get("ambient"):
        changes["ambient"] = new_state["ambient"]

    return changes




def _compact_registry_for_prompt(registry: dict) -> str:
    """One-line-per-object summary for the LLM prompt."""
    if not registry:
        return "(empty)"
    lines = []
    for obj_id, obj in registry.items():
        loc = "?"
        state_str = "?"
        held = None
        spatial = obj.get("spatial_info") or {}
        if isinstance(spatial, dict):
            loc = spatial.get("location", "?")
        st = obj.get("state") or {}
        if isinstance(st, dict):
            state_str = st.get("current_state", "?")
            held = st.get("held_by")
        first = obj.get("first_seen_turn", "?")
        last = obj.get("last_seen_turn", "?")
        parts = [f"loc={loc}", f"state={state_str}"]
        if held:
            parts.append(f"held={held}")
        parts.append(f"seen=t{first}-t{last}")
        lines.append(f"  {obj_id}: {', '.join(parts)}")
    return "\n".join(lines)


def _find_matching_existing_object(new_id: str, registry: dict) -> Optional[str]:
    """Find the best surface-form match used by the paper Memory Editor."""
    if not new_id or not registry:
        return None

    new_lower = new_id.lower().replace("_", " ").replace("-", " ")
    new_words = new_lower.split()
    new_noun = new_words[-1] if new_words else ""

    best_match = None
    best_score = 0

    for existing_id in registry.keys():
        ex_lower = existing_id.lower().replace("_", " ").replace("-", " ")
        if new_lower == ex_lower:
            continue

        ex_words = ex_lower.split()
        ex_noun = ex_words[-1] if ex_words else ""
        score = 0

        if new_lower in ex_lower or ex_lower in new_lower:
            score = 3
        elif new_noun and new_noun == ex_noun:
            overlap = len(set(new_words) & set(ex_words))
            min_len = min(len(new_words), len(ex_words))
            if min_len > 0 and overlap / min_len >= 0.5:
                score = 2 if overlap >= 2 else 1

        if score > best_score:
            best_score = score
            best_match = existing_id

    if best_match:
        logger.info(
            f"      🔗 Rename match: '{new_id}' → '{best_match}' "
            f"(score={best_score})"
        )
    return best_match


def _create_state_history_entry(
    obj_data: dict,
    turn_id: int,
    time_window: Dict[str, Any],
    location_override: str = None,
    time_offset: float = 0
) -> dict:
    """Create a state_history entry with rule-based timestamps.
    Every entry has time_seconds and time so consumers can use either.
    """
    base_time = time_window.get("start", turn_id * 10)
    t = base_time + time_offset
    return {
        "turn_id": turn_id,
        "time_seconds": t,
        "time": t,  # compact timestamp mirror used by retrieval tools
        "state": obj_data.get("state", {}).get("current_state", "") if isinstance(obj_data.get("state"), dict) else str(obj_data.get("state", "")),
        "location": location_override if location_override else (obj_data.get("spatial_info", {}).get("location", "") if isinstance(obj_data.get("spatial_info"), dict) else ""),
    }


def _sort_state_history(state_history: list, set_order_index: bool = True) -> None:
    """Sort state_history in place by time (ascending). Optionally set order_index on each entry.
    Key: time_seconds or time, then turn_id. Ensures consumers can assume chronological order.
    """
    if not state_history:
        return

    def _entry_time(entry: dict) -> float:
        t = entry.get("time_seconds", entry.get("time", 0))
        try:
            return float(t)
        except (TypeError, ValueError):
            return 0.0

    state_history.sort(key=lambda e: (_entry_time(e), e.get("turn_id", 0)))
    if set_order_index:
        for i, entry in enumerate(state_history):
            entry["order_index"] = i


def _ensure_state_history_sorted(obj_data: dict) -> None:
    """If obj_data has state_history list, sort it by time and set order_index. In-place."""
    hist = obj_data.get("state_history")
    if isinstance(hist, list) and len(hist) > 0:
        _sort_state_history(hist, set_order_index=True)


def _expand_movement_trajectory(
    obj_data: dict,
    turn_id: int,
    time_window: Dict[str, Any],
    action: str = "movement"
) -> list:
    """
    If object has movement_trajectory, create state_history entries for each location.
    This captures intermediate movements within a single segment.

    Example: movement_trajectory = ["on counter", "in hand", "in sink"]
    Creates 3 entries with estimated timestamps spread across the segment.
    """
    spatial_info = obj_data.get("spatial_info", {})
    if not isinstance(spatial_info, dict):
        return []

    trajectory = spatial_info.get("movement_trajectory")
    if not trajectory or not isinstance(trajectory, list) or len(trajectory) <= 1:
        return []  # No trajectory or only one location - no intermediate movements

    # Calculate time interval for each movement
    segment_duration = time_window.get("end", (turn_id + 1) * 10) - time_window.get("start", turn_id * 10)
    num_locations = len(trajectory)
    time_per_location = segment_duration / num_locations

    entries = []
    for i, loc in enumerate(trajectory):
        time_offset = i * time_per_location
        entry = _create_state_history_entry(obj_data, turn_id, time_window, location_override=loc, time_offset=time_offset)
        entry["action"] = f"{action}_step_{i+1}_of_{num_locations}"
        entry["from_trajectory"] = True  # Mark as expanded from trajectory
        if i > 0:
            entry["previous_location"] = trajectory[i-1]
        entries.append(entry)

    return entries


def _recover_movement_trajectory(
    data: dict,
    object_id: str,
    new_object_registry: Dict[str, Any]
) -> None:
    """
    Recover movement_trajectory from VLM output if LLM didn't include it.
    Modifies `data` in-place.
    """
    data_spatial = data.get("spatial_info", {})
    if not data_spatial.get("movement_trajectory") and object_id in new_object_registry:
        vlm_spatial = new_object_registry[object_id].get("spatial_info", {})
        if isinstance(vlm_spatial, dict) and vlm_spatial.get("movement_trajectory"):
            if "spatial_info" not in data:
                data["spatial_info"] = {}
            data["spatial_info"]["movement_trajectory"] = vlm_spatial["movement_trajectory"]
            logger.info(f"    Recovered movement_trajectory from VLM for '{object_id}'")
