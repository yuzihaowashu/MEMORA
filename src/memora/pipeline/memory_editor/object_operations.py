"""Apply validated Memory Editor operations to Entity Memory."""

import copy
import logging
from typing import Any, Dict, List

from memora.pipeline.memory_editor.typed_memory import (
    EmbodiedMemoryState,
    _create_state_history_entry,
    _ensure_state_history_sorted,
    _expand_movement_trajectory,
    _find_matching_existing_object,
    _recover_movement_trajectory,
)
logger = logging.getLogger(__name__)


def apply_object_operations(
    current_memory: EmbodiedMemoryState,
    object_operations: List[Dict[str, Any]],
    new_object_registry: Dict[str, Any],
    turn_id: int,
    time_window: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply Add, Update, Delete, or Noop while preserving state history."""
    llm_ops: List[Dict[str, Any]] = []
    # Track which objects LLM processed (for default NOOP fallback)
    llm_processed_object_ids = set()

    for op in object_operations:
        event = op.get("event", "NOOP")
        object_id = op.get("object_id", "")

        # Track this object as processed by LLM
        if object_id:
            llm_processed_object_ids.add(object_id)

        if event == "ADD":
            data = op.get("data", {})

            # Initialize state_history for new object
            if "state_history" not in data:
                data["state_history"] = []

            # ROBUST: Recover movement_trajectory from VLM if LLM didn't include it
            _recover_movement_trajectory(data, object_id, new_object_registry)

            # Check for movement_trajectory and expand it
            trajectory_entries = _expand_movement_trajectory(data, turn_id, time_window, action="initial_movement")
            if trajectory_entries:
                # Use expanded trajectory entries instead of single entry
                data["state_history"].extend(trajectory_entries)
                logger.info(f"    Object '{object_id}' trajectory expanded: {len(trajectory_entries)} locations")
            else:
                # No trajectory - create single entry for final location
                initial_history = _create_state_history_entry(data, turn_id, time_window)
                initial_history["action"] = "initial_observation"
                data["state_history"].append(initial_history)
            data["first_seen_time"] = time_window.get("start", turn_id * 10)
            data["first_seen_turn"] = turn_id
            data["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
            data["last_seen_turn"] = turn_id
            _ensure_state_history_sorted(data)

            # Check if object_id already exists (exact match)
            if object_id in current_memory.object_registry:
                # Error handling: LLM tried to ADD existing object
                # Convert to UPDATE instead
                logger.warning(f"    ADD on existing object_id '{object_id}' - converting to UPDATE")
                old_data = copy.deepcopy(current_memory.object_registry[object_id])

                # Preserve existing state_history and append new entries
                existing_history = old_data.get("state_history", [])

                _recover_movement_trajectory(data, object_id, new_object_registry)

                # Check for movement_trajectory and expand it
                trajectory_entries = _expand_movement_trajectory(data, turn_id, time_window, action="update_movement")
                if trajectory_entries:
                    existing_history.extend(trajectory_entries)
                    logger.info(f"    Object '{object_id}' trajectory expanded: {len(trajectory_entries)} locations")
                else:
                    new_history_entry = _create_state_history_entry(data, turn_id, time_window)
                    new_history_entry["action"] = "state_update"
                    existing_history.append(new_history_entry)
                data["state_history"] = existing_history
                data["first_seen_time"] = old_data.get("first_seen_time", time_window.get("start", 0))
                data["first_seen_turn"] = old_data.get("first_seen_turn", turn_id)
                _ensure_state_history_sorted(data)

                current_memory.object_registry[object_id] = data
                llm_ops.append({
                    "layer": "object_registry",
                    "event": "UPDATE",
                    "object_id": object_id,
                    "old": old_data,
                    "new": data,
                    "note": "converted_from_add_exact_match"
                })
            else:
                # NEW: Check for similar existing objects (fuzzy match)
                # Handles VLM ID inconsistency: "cutting_board" vs "cutting_board_wooden"
                matching_existing_id = _find_matching_existing_object(object_id, current_memory.object_registry)

                if matching_existing_id:
                    # Found a similar object - convert to UPDATE on that object
                    logger.info(f"   🔗 ADD on similar object '{object_id}' → UPDATE '{matching_existing_id}'")
                    old_data = copy.deepcopy(current_memory.object_registry[matching_existing_id])

                    # Preserve existing state_history and append new entries
                    existing_history = old_data.get("state_history", [])

                    _recover_movement_trajectory(data, object_id, new_object_registry)

                    # Check for movement_trajectory and expand it
                    trajectory_entries = _expand_movement_trajectory(data, turn_id, time_window, action="update_movement")
                    if trajectory_entries:
                        for entry in trajectory_entries:
                            entry["original_object_id"] = object_id  # Track original ID
                        existing_history.extend(trajectory_entries)
                        logger.info(f"    Object '{matching_existing_id}' trajectory expanded: {len(trajectory_entries)} locations")
                    else:
                        new_history_entry = _create_state_history_entry(data, turn_id, time_window)
                        new_history_entry["action"] = "state_update"
                        new_history_entry["original_object_id"] = object_id  # Track original ID
                        existing_history.append(new_history_entry)

                    # Merge data into existing object
                    merged_data = copy.deepcopy(old_data)
                    # Update state and spatial_info with new data
                    if "state" in data:
                        merged_data["state"] = data["state"]
                    if "spatial_info" in data:
                        merged_data["spatial_info"] = data["spatial_info"]
                    if "visual_properties" in data:
                        # Merge visual properties (prefer new if present)
                        merged_data.setdefault("visual_properties", {}).update(data.get("visual_properties", {}))
                    merged_data["state_history"] = existing_history
                    merged_data["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
                    merged_data["last_seen_turn"] = turn_id
                    _ensure_state_history_sorted(merged_data)

                    current_memory.object_registry[matching_existing_id] = merged_data
                    llm_ops.append({
                        "layer": "object_registry",
                        "event": "UPDATE",
                        "object_id": matching_existing_id,
                        "original_object_id": object_id,
                        "old": old_data,
                        "new": merged_data,
                        "note": "converted_from_add_fuzzy_match"
                    })
                else:
                    # Truly new object
                    current_memory.object_registry[object_id] = data
                    llm_ops.append({
                        "layer": "object_registry",
                        "event": "ADD",
                        "object_id": object_id,
                        "data": data
                    })

        elif event == "UPDATE":
            changes = op.get("changes", {})

            # First check for exact match, then fuzzy match
            target_object_id = object_id
            if object_id not in current_memory.object_registry:
                # Try fuzzy match
                matching_existing_id = _find_matching_existing_object(object_id, current_memory.object_registry)
                if matching_existing_id:
                    logger.info(f"   🔗 UPDATE on '{object_id}' → found match '{matching_existing_id}'")
                    target_object_id = matching_existing_id

            if target_object_id in current_memory.object_registry:
                old_data = copy.deepcopy(current_memory.object_registry[target_object_id])

                # Check if state or location changed (for state_history)
                # Get entire state dict for comprehensive comparison
                old_state_dict = old_data.get("state", {}) if isinstance(old_data.get("state"), dict) else {}
                old_spatial_dict = old_data.get("spatial_info", {}) if isinstance(old_data.get("spatial_info"), dict) else {}

                # Extract specific fields for comparison
                old_current_state = old_state_dict.get("current_state", "")
                old_held_by = old_state_dict.get("held_by", None)
                old_location = old_spatial_dict.get("location", "")

                # Apply changes
                for key, value in changes.items():
                    if isinstance(value, dict) and isinstance(current_memory.object_registry[target_object_id].get(key), dict):
                        current_memory.object_registry[target_object_id][key].update(value)
                    else:
                        current_memory.object_registry[target_object_id][key] = value

                # Get new state/location after applying changes
                new_state_dict = current_memory.object_registry[target_object_id].get("state", {}) if isinstance(current_memory.object_registry[target_object_id].get("state"), dict) else {}
                new_spatial_dict = current_memory.object_registry[target_object_id].get("spatial_info", {}) if isinstance(current_memory.object_registry[target_object_id].get("spatial_info"), dict) else {}

                new_current_state = new_state_dict.get("current_state", "")
                new_held_by = new_state_dict.get("held_by", None)
                new_location = new_spatial_dict.get("location", "")

                # Check for ANY meaningful state change (not just current_state)
                current_state_changed = old_current_state != new_current_state
                held_by_changed = old_held_by != new_held_by
                location_changed = old_location != new_location

                # Any of these changes should trigger state_history update
                any_significant_change = current_state_changed or held_by_changed or location_changed

                if any_significant_change:
                    if "state_history" not in current_memory.object_registry[target_object_id]:
                        current_memory.object_registry[target_object_id]["state_history"] = []

                    _recover_movement_trajectory(changes, object_id, new_object_registry)

                    # Check for movement_trajectory in changes and expand it
                    trajectory_entries = _expand_movement_trajectory(changes, turn_id, time_window, action="update_movement")
                    if trajectory_entries:
                        for entry in trajectory_entries:
                            if target_object_id != object_id:
                                entry["original_object_id"] = object_id
                            if location_changed:
                                entry["from_location"] = old_location
                        current_memory.object_registry[target_object_id]["state_history"].extend(trajectory_entries)
                        logger.info(f"    Object '{target_object_id}' trajectory expanded: {len(trajectory_entries)} locations")
                    else:
                        history_entry = _create_state_history_entry(current_memory.object_registry[target_object_id], turn_id, time_window)
                        history_entry["action"] = op.get("reason", "state_update")

                        # Record what changed
                        if current_state_changed:
                            history_entry["old_state"] = old_current_state
                        if held_by_changed:
                            history_entry["old_held_by"] = old_held_by
                            history_entry["new_held_by"] = new_held_by
                        if location_changed:
                            history_entry["old_location"] = old_location

                        # Track if this was a fuzzy match
                        if target_object_id != object_id:
                            history_entry["original_object_id"] = object_id

                        current_memory.object_registry[target_object_id]["state_history"].append(history_entry)

                    _ensure_state_history_sorted(current_memory.object_registry[target_object_id])

                # Update last_seen timestamps (Rule-based)
                current_memory.object_registry[target_object_id]["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
                current_memory.object_registry[target_object_id]["last_seen_turn"] = turn_id

                llm_ops.append({
                    "layer": "object_registry",
                    "event": "UPDATE",
                    "object_id": target_object_id,
                    "original_object_id": object_id if target_object_id != object_id else None,
                    "old": old_data,
                    "changes": changes,
                    "state_history_appended": any_significant_change
                })
            else:
                # Error handling: UPDATE on non-existent object_id
                # Convert to ADD instead (the object may be new)
                logger.warning(f"    UPDATE on non-existent object_id '{object_id}' - converting to ADD")
                data = changes  # Use changes as the new data

                # Initialize state_history for converted ADD
                if "state_history" not in data:
                    data["state_history"] = []

                _recover_movement_trajectory(data, object_id, new_object_registry)

                # Check for movement_trajectory and expand it
                trajectory_entries = _expand_movement_trajectory(data, turn_id, time_window, action="initial_movement_from_update")
                if trajectory_entries:
                    data["state_history"].extend(trajectory_entries)
                    logger.info(f"    Object '{object_id}' trajectory expanded: {len(trajectory_entries)} locations")
                else:
                    initial_history = _create_state_history_entry(data, turn_id, time_window)
                    initial_history["action"] = "initial_observation_from_update"
                    data["state_history"].append(initial_history)
                data["first_seen_time"] = time_window.get("start", turn_id * 10)
                data["first_seen_turn"] = turn_id
                data["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
                data["last_seen_turn"] = turn_id
                _ensure_state_history_sorted(data)

                current_memory.object_registry[object_id] = data
                llm_ops.append({
                    "layer": "object_registry",
                    "event": "ADD",
                    "object_id": object_id,
                    "data": data,
                    "note": "converted_from_update_missing_id"
                })

        elif event == "DELETE":
            reason = op.get("reason", "unknown")
            if object_id in current_memory.object_registry:
                deleted_data = current_memory.object_registry.pop(object_id)
                llm_ops.append({
                    "layer": "object_registry",
                    "event": "DELETE",
                    "object_id": object_id,
                    "data": deleted_data,
                    "reason": reason
                })
            else:
                # Error handling: DELETE on non-existent object_id
                # Record as INVALID operation for diagnostics
                logger.warning(f"    DELETE on non-existent object_id '{object_id}' - skipping")
                llm_ops.append({
                    "layer": "object_registry",
                    "event": "DELETE_INVALID",
                    "object_id": object_id,
                    "reason": f"object_id not found: {reason}",
                    "note": "skipped_missing_id"
                })

        else:  # NOOP
            reason = op.get("reason", "unchanged")

            # Still update last_seen timestamps for NOOP (object was observed)
            if object_id in current_memory.object_registry:
                current_memory.object_registry[object_id]["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
                current_memory.object_registry[object_id]["last_seen_turn"] = turn_id

            llm_ops.append({
                "layer": "object_registry",
                "event": "NOOP",
                "object_id": object_id,
                "reason": reason
            })

    # Inferred Knowledge is handled by offline consolidation.

    return llm_ops


def maintain_unmentioned_observations(
    current_memory: EmbodiedMemoryState,
    new_object_registry: Dict[str, Any],
    llm_ops: List[Dict[str, Any]],
    turn_id: int,
    time_window: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Maintain observed existing objects omitted by the Memory Editor.

    The paper pipeline treated an omitted observation as a deterministic Noop,
    while still refreshing recency and retaining an observed location change.
    """
    for observed_id, observed_data in new_object_registry.items():
        already_processed = any(
            op.get("object_id") == observed_id
            or op.get("original_object_id") == observed_id
            for op in llm_ops
            if op.get("layer") == "object_registry"
        )

        if not already_processed:
            observed_name = observed_id.lower().replace("_", " ")
            for op in llm_ops:
                if op.get("layer") != "object_registry":
                    continue
                operation_name = str(op.get("object_id", "")).lower().replace("_", " ")
                if (
                    observed_name == operation_name
                    or observed_name in operation_name
                    or operation_name in observed_name
                ):
                    already_processed = True
                    break

        if already_processed:
            continue

        matching_memory_id = observed_id if observed_id in current_memory.object_registry else None
        if not matching_memory_id:
            matching_memory_id = _find_matching_existing_object(
                observed_id, current_memory.object_registry
            )
        if not matching_memory_id:
            continue

        memory_object = current_memory.object_registry[matching_memory_id]
        memory_object["last_seen_time"] = time_window.get("end", (turn_id + 1) * 10)
        memory_object["last_seen_turn"] = turn_id

        spatial_info = observed_data.get("spatial_info") or {}
        observed_location = (
            spatial_info.get("location", "") if isinstance(spatial_info, dict) else ""
        )
        state_history = memory_object.get("state_history", [])
        previous_location = state_history[-1].get("location", "") if state_history else ""

        if observed_location and observed_location != previous_location:
            state = observed_data.get("state") or {}
            current_state = (
                state.get("current_state", "") if isinstance(state, dict) else str(state)
            )
            memory_object.setdefault("state_history", []).append({
                "turn_id": turn_id,
                "time_seconds": time_window.get("start", turn_id * 10),
                "state": current_state,
                "location": observed_location,
                "action": "fallback_noop_location_update",
            })
            operation = {
                "layer": "object_registry",
                "event": "FALLBACK_NOOP_WITH_UPDATE",
                "object_id": matching_memory_id,
                "observed_object_id": observed_id,
                "reason": (
                    "Memory Editor omitted the object; location changed: "
                    f"{previous_location} -> {observed_location}"
                ),
            }
        else:
            operation = {
                "layer": "object_registry",
                "event": "FALLBACK_NOOP",
                "object_id": matching_memory_id,
                "observed_object_id": observed_id,
                "reason": "Memory Editor omitted the object; recency updated",
            }

        llm_ops.append(operation)

    return llm_ops
