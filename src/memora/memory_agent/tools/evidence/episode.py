"""Episode reconstruction helpers used by ``TypedMemoryTools``."""

import re
from typing import Any, Dict


def build_episode(memory_tools: Any, activity: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an episode from an activity hit by adding temporal, object, and scene context.
    """
    tw = activity.get("time_window", activity.get("time", {}))
    if isinstance(tw, dict):
        t_start = tw.get("start", 0)
        t_end = tw.get("end", 0)
    elif isinstance(tw, str):
        match = re.search(r"(\d+(?:\.\d+)?)", tw)
        t_start = float(match.group(1)) if match else 0
        t_end = t_start
    else:
        t_start, t_end = 0, 0

    episode: Dict[str, Any] = {
        "source_video": activity.get("source_video", memory_tools.current_video_id),
        "turn_id": activity.get("turn_id"),
        "time": tw,
        "summary": activity.get("summary", ""),
    }

    for field in ("high_level_goal", "video_summary", "goal_turn_range"):
        val = activity.get(field)
        if val:
            episode[field] = val

    action_breakdown = activity.get("action_breakdown", [])
    if action_breakdown and isinstance(action_breakdown, list):
        episode["action_breakdown"] = action_breakdown

    detailed_narrative = activity.get("detailed_narrative", "")
    if detailed_narrative:
        episode["detailed_narrative"] = detailed_narrative

    local_sequence = activity.get("local_sequence", "")
    if local_sequence:
        episode["local_sequence"] = local_sequence

    local_event = activity.get("local_event")
    if isinstance(local_event, dict):
        episode["local_event"] = local_event

    goal_activity_view = activity.get("goal_activity_view")
    if isinstance(goal_activity_view, dict):
        episode["goal_activity_view"] = goal_activity_view

    preceding_action = activity.get("preceding_action")
    following_action = activity.get("following_action")
    if preceding_action:
        episode["previous_action"] = preceding_action
    if following_action:
        episode["next_action"] = following_action

    if not preceding_action and not following_action and memory_tools.current_memory:
        activity_log = memory_tools.current_memory.get("activity_log", [])
        current_idx = None
        for i, act in enumerate(activity_log):
            act_tw = act.get("time_window", {})
            if isinstance(act_tw, dict) and abs(act_tw.get("start", -999) - t_start) < 5:
                current_idx = i
                break
        if current_idx is not None:
            if current_idx > 0:
                episode["previous_action"] = activity_log[current_idx - 1].get("summary", "")[:100]
            if current_idx < len(activity_log) - 1:
                episode["next_action"] = activity_log[current_idx + 1].get("summary", "")[:100]

    if memory_tools.current_memory:
        obj_summaries = []
        for obj_id, obj_data in memory_tools.current_memory.get("object_registry", {}).items():
            if not isinstance(obj_data, dict):
                continue
            first_seen = obj_data.get(
                "first_seen_time",
                obj_data.get("first_seen_turn", obj_data.get("first_seen", 0)),
            )
            last_seen = obj_data.get(
                "last_seen_time",
                obj_data.get("last_seen_turn", obj_data.get("last_seen", float("inf"))),
            )
            if not isinstance(first_seen, (int, float)):
                first_seen = 0
            if not isinstance(last_seen, (int, float)):
                last_seen = float("inf")

            if first_seen <= t_end and last_seen >= t_start:
                state_at_time = memory_tools._find_state_at_time(obj_data, (t_start + t_end) / 2)
                name = obj_data.get("name", obj_id)
                loc = state_at_time.get("location", "")
                if not loc:
                    spatial_info = obj_data.get("spatial_info", {})
                    loc = spatial_info.get("location", "") if isinstance(spatial_info, dict) else ""
                state = state_at_time.get("state", "")
                desc = name
                if isinstance(loc, str) and loc:
                    desc += f" @ {loc}"
                elif isinstance(loc, dict):
                    zone = loc.get("zone", loc.get("description", ""))
                    if zone:
                        desc += f" @ {zone}"
                if isinstance(state, str) and state:
                    desc += f" ({state})"
                obj_summaries.append(desc)

        if obj_summaries:
            episode["scene_objects"] = obj_summaries[:10]

    if memory_tools.current_memory:
        for env in memory_tools.current_memory.get("environment_log", []):
            first_seen = env.get("first_seen", 0)
            last_seen = env.get("last_seen", float("inf"))
            if first_seen <= t_end and last_seen >= t_start:
                current_state = env.get("current_state", env)
                if isinstance(current_state, dict):
                    layout = current_state.get("layout_description", "")
                    if layout:
                        episode["environment"] = layout[:150]
                        break

    return episode
