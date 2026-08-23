"""Build the controlled Graph-2D representation from MEMORA observations.

The builder creates activity, object, environment, and inferred-knowledge
nodes together with the deterministic edges used by the paper baseline.
It accepts either participant memory JSON or raw Segment Encoder JSONL.
"""

import argparse
import json
import logging
import os
import re
import time
from collections import defaultdict
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Object matching

_STRIP_ARTICLES = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
_NON_ALPHA = re.compile(r"[^a-z0-9 ]")


def _normalize(text) -> str:
    """Lowercase, remove articles/underscores/punctuation, collapse whitespace.

    Tolerant of non-string values from Segment Encoder observations."""
    if text is None:
        return ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t)
    elif not isinstance(text, str):
        text = str(text)
    text = text.lower().replace("_", " ")
    text = _STRIP_ARTICLES.sub("", text)
    text = _NON_ALPHA.sub(" ", text)
    return " ".join(text.split())


def _fuzzy_match_object(mention: str, registry_keys: List[str],
                        norm_cache: Dict[str, str]) -> Optional[str]:
    """Return the best-matching object_id from registry, or None.

    Strategy: exact normalised match > substring containment (prefer longest key).
    """
    norm_mention = _normalize(mention)
    if not norm_mention:
        return None

    best_key: Optional[str] = None
    best_len = 0

    for key in registry_keys:
        norm_key = norm_cache[key]
        if norm_key == norm_mention:
            return key
        if norm_mention in norm_key or norm_key in norm_mention:
            if len(norm_key) > best_len:
                best_key = key
                best_len = len(norm_key)

    return best_key


# Node builders

def _build_activity_nodes(activity_log: List[dict]) -> Dict[str, dict]:
    nodes: Dict[str, dict] = {}
    for act in activity_log:
        turn_id = act.get("turn_id")
        if turn_id is None:
            continue
        node_id = f"act_{turn_id}"
        tw = act.get("time_window", {})
        start = tw.get("start", 0.0)
        end = tw.get("end", 0.0)
        timestamp = (start + end) / 2.0

        summary = act.get("summary", "")
        narrative = act.get("detailed_narrative", "")
        if isinstance(narrative, list):
            narrative = " ".join(str(s) for s in narrative if s)
        if not isinstance(narrative, str):
            narrative = ""
        content = summary
        if narrative:
            content += " | " + narrative[:200]

        nodes[node_id] = {
            "node_id": node_id,
            "node_type": "activity",
            "timestamp": timestamp,
            "content": content,
            "metadata": {
                "turn_id": turn_id,
                "time_window": tw,
                "summary": summary,
                "detailed_narrative": narrative,
                # The controlled Graph-2D artifact excludes additional goal labels.
                "high_level_goal": "",
            },
        }
    return nodes


def _build_object_nodes(object_registry: Dict[str, dict]) -> Dict[str, dict]:
    nodes: Dict[str, dict] = {}
    for obj_id, obj_data in object_registry.items():
        if not isinstance(obj_data, dict):
            continue
        node_id = f"obj_{obj_id}"
        name = obj_data.get("name", obj_id)
        if isinstance(name, list):
            name = " ".join(str(x) for x in name if x)
        elif not isinstance(name, str):
            name = str(name) if name else str(obj_id)

        spatial = obj_data.get("spatial_info", {}) if isinstance(obj_data.get("spatial_info"), dict) else {}
        location = spatial.get("location", "")
        if not location:
            state_info = obj_data.get("state", {})
            if isinstance(state_info, dict):
                location = state_info.get("current_state", "")
        if isinstance(location, list):
            location = " ".join(str(s) for s in location if s)
        if not isinstance(location, str):
            location = str(location) if location else ""

        vp = obj_data.get("visual_properties", {}) if isinstance(obj_data.get("visual_properties"), dict) else {}
        vis_parts = []
        for attr in ("color", "material", "size"):
            val = vp.get(attr, "")
            if isinstance(val, list):
                val = val[0] if val else ""
            if not isinstance(val, str):
                val = str(val) if val else ""
            if val and val.lower() not in ("unknown", "null", "none", "n/a"):
                vis_parts.append(val)
        vis_desc = " ".join(vis_parts)

        content_parts = [name]
        if location:
            content_parts.append(location)
        if vis_desc:
            content_parts.append(f"{vis_desc} {name.split()[-1] if name else ''}")
        content = ", ".join(p for p in content_parts if p)

        first_seen = obj_data.get("first_seen_time", 0.0)
        if not isinstance(first_seen, (int, float)):
            first_seen = 0.0

        nodes[node_id] = {
            "node_id": node_id,
            "node_type": "object",
            "timestamp": float(first_seen),
            "content": content,
            "metadata": {
                "name": name,
                "location": location,
                "state": obj_data.get("state", {}).get("current_state", "") if isinstance(obj_data.get("state"), dict) else "",
                "visual_properties": vp,
            },
        }
    return nodes


def _build_environment_nodes(environment_log: List[dict]) -> Dict[str, dict]:
    nodes: Dict[str, dict] = {}
    for env in environment_log:
        if not isinstance(env, dict):
            continue
        loc_id = env.get("location_id", "")
        if not loc_id:
            continue
        node_id = f"env_{loc_id}"

        cs = env.get("current_state", {})
        layout = cs.get("layout_description", "") if isinstance(cs, dict) else ""
        if isinstance(layout, list):
            layout = " ".join(str(s) for s in layout if s)
        elif isinstance(layout, dict):
            layout = "; ".join(f"{k}={v}" for k, v in layout.items())
        elif not isinstance(layout, str):
            layout = str(layout) if layout else ""

        content = f"{loc_id}: {layout[:300]}" if layout else loc_id

        first_seen = env.get("first_seen", 0.0)
        if not isinstance(first_seen, (int, float)):
            first_seen = 0.0

        nodes[node_id] = {
            "node_id": node_id,
            "node_type": "environment",
            "timestamp": float(first_seen),
            "content": content,
            "metadata": {
                "location_id": loc_id,
                "first_seen": env.get("first_seen", 0.0),
                "last_seen": env.get("last_seen", 0.0),
            },
        }
    return nodes


def _build_inferred_nodes(inferred_knowledge: dict) -> Dict[str, dict]:
    nodes: Dict[str, dict] = {}
    if not isinstance(inferred_knowledge, dict):
        return nodes

    action_sequences = inferred_knowledge.get("action_sequences", [])
    if not isinstance(action_sequences, list):
        return nodes

    for seq in action_sequences:
        if not isinstance(seq, dict):
            continue
        kid = seq.get("knowledge_id", "")
        if not kid:
            continue
        node_id = f"pat_{kid}"
        content_data = seq.get("content", {}) if isinstance(seq.get("content"), dict) else {}
        title = content_data.get("title", "")
        goal = content_data.get("goal", "")
        key_objects = content_data.get("key_objects", [])
        if isinstance(key_objects, list):
            ko_str = ", ".join(str(o) for o in key_objects)
        else:
            ko_str = str(key_objects)

        content_parts = []
        if title:
            content_parts.append(title)
        if goal:
            content_parts.append(goal)
        if ko_str:
            content_parts.append(f"Key objects: {ko_str}")
        content = ": ".join(content_parts[:2])
        if ko_str:
            content += f". {content_parts[-1]}" if len(content_parts) > 2 else ""

        nodes[node_id] = {
            "node_id": node_id,
            "node_type": "pattern",
            "timestamp": -1,
            "content": content,
            "metadata": {
                "knowledge_id": kid,
                "title": title,
                "goal": goal,
                "key_objects": key_objects,
                "time_range": content_data.get("time_range", {}),
            },
        }
    return nodes


# Edge builders

def _build_next_edges(activity_nodes: Dict[str, dict]) -> List[dict]:
    """NEXT: consecutive activities sorted by time_window.start."""
    sorted_ids = sorted(
        activity_nodes.keys(),
        key=lambda nid: activity_nodes[nid]["metadata"]["time_window"].get("start", 0.0),
    )
    edges = []
    for i in range(len(sorted_ids) - 1):
        edges.append({
            "source": sorted_ids[i],
            "target": sorted_ids[i + 1],
            "edge_type": "NEXT",
            "weight": 1.0,
        })
    return edges


def _build_used_in_edges(activity_log: List[dict],
                         object_registry: Dict[str, dict]) -> List[dict]:
    """USED_IN: object -> activity.

    Two sources (union):
      1. action_breakdown object mentions (fuzzy text match)
      2. state_history turn co-occurrence (object observed at same turn as activity)
    """
    if not object_registry:
        return []

    reg_keys = list(object_registry.keys())
    norm_cache = {k: _normalize(k) for k in reg_keys}
    for k, v in object_registry.items():
        if isinstance(v, dict) and v.get("name"):
            norm_cache.setdefault(k, _normalize(v["name"]))

    edges = []
    seen: Set[Tuple[str, str]] = set()

    # Source 1: action_breakdown text matching
    for act in activity_log:
        turn_id = act.get("turn_id")
        if turn_id is None:
            continue
        act_node_id = f"act_{turn_id}"
        breakdown = act.get("action_breakdown", [])
        if not isinstance(breakdown, list):
            continue

        for entry in breakdown:
            if not isinstance(entry, dict):
                continue
            obj_mention = entry.get("object", "")
            if not obj_mention or not isinstance(obj_mention, str):
                continue

            matched_key = _fuzzy_match_object(obj_mention, reg_keys, norm_cache)
            if matched_key is None:
                continue

            obj_node_id = f"obj_{matched_key}"
            pair = (obj_node_id, act_node_id)
            if pair in seen:
                continue
            seen.add(pair)

            edges.append({
                "source": obj_node_id,
                "target": act_node_id,
                "edge_type": "USED_IN",
                "weight": 1.0,
                "metadata": {"action": entry.get("action", "")},
            })

    # Source 2: state_history turn co-occurrence
    turn_to_activity = {}
    for act in activity_log:
        tid = act.get("turn_id")
        if tid is not None:
            turn_to_activity[tid] = f"act_{tid}"

    for obj_id, obj_data in object_registry.items():
        if not isinstance(obj_data, dict):
            continue
        state_history = obj_data.get("state_history", [])
        if not isinstance(state_history, list):
            continue
        for sh_entry in state_history:
            if not isinstance(sh_entry, dict):
                continue
            tid = sh_entry.get("turn_id")
            if tid is None:
                continue
            act_nid = turn_to_activity.get(tid)
            if act_nid is None:
                continue
            obj_node_id = f"obj_{obj_id}"
            pair = (obj_node_id, act_nid)
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": obj_node_id,
                "target": act_nid,
                "edge_type": "USED_IN",
                "weight": 0.8,
                "metadata": {"source": "state_history_co_occurrence"},
            })

    return edges


def _build_located_at_edges(object_nodes: Dict[str, dict],
                            environment_nodes: Dict[str, dict]) -> List[dict]:
    """LOCATED_AT: object -> environment, by substring matching location text."""
    if not environment_nodes:
        return []

    env_entries = []
    for env_nid, env_node in environment_nodes.items():
        loc_id = env_node["metadata"]["location_id"]
        norm_loc = _normalize(loc_id)
        env_entries.append((env_nid, loc_id, norm_loc))

    edges = []
    for obj_nid, obj_node in object_nodes.items():
        location_text = obj_node["metadata"].get("location", "")
        if not location_text:
            continue
        norm_loc_text = _normalize(location_text)
        if not norm_loc_text:
            continue

        best_env: Optional[str] = None
        best_score = 0
        for env_nid, loc_id, norm_env in env_entries:
            if norm_env in norm_loc_text or norm_loc_text in norm_env:
                score = len(norm_env)
                if score > best_score:
                    best_env = env_nid
                    best_score = score
            for env_word in norm_env.split():
                if len(env_word) >= 4 and env_word in norm_loc_text:
                    if len(env_word) > best_score:
                        best_env = env_nid
                        best_score = len(env_word)

        if best_env:
            edges.append({
                "source": obj_nid,
                "target": best_env,
                "edge_type": "LOCATED_AT",
                "weight": 1.0,
            })

    return edges


def _build_co_occurs_edges(activity_log: List[dict],
                           object_registry: Dict[str, dict]) -> List[dict]:
    """CO_OCCURS: objects that appear together in the same activity segment."""
    if not object_registry:
        return []

    reg_keys = list(object_registry.keys())
    norm_cache = {k: _normalize(k) for k in reg_keys}

    turn_to_objects: Dict[int, Set[str]] = defaultdict(set)

    for act in activity_log:
        turn_id = act.get("turn_id")
        if turn_id is None:
            continue
        breakdown = act.get("action_breakdown", [])
        if isinstance(breakdown, list):
            for entry in breakdown:
                if not isinstance(entry, dict):
                    continue
                obj_mention = entry.get("object", "")
                if obj_mention and isinstance(obj_mention, str):
                    matched = _fuzzy_match_object(obj_mention, reg_keys, norm_cache)
                    if matched:
                        turn_to_objects[turn_id].add(matched)

    for obj_id, obj_data in object_registry.items():
        if not isinstance(obj_data, dict):
            continue
        state_history = obj_data.get("state_history", [])
        if isinstance(state_history, list):
            for entry in state_history:
                if isinstance(entry, dict):
                    tid = entry.get("turn_id")
                    if tid is not None:
                        turn_to_objects[tid].add(obj_id)

    seen_pairs: Dict[Tuple[str, str], int] = {}
    edges = []

    for turn_id, obj_set in turn_to_objects.items():
        if len(obj_set) < 2:
            continue
        for a, b in combinations(sorted(obj_set), 2):
            pair = (a, b)
            if pair not in seen_pairs:
                seen_pairs[pair] = turn_id
                edges.append({
                    "source": f"obj_{a}",
                    "target": f"obj_{b}",
                    "edge_type": "CO_OCCURS",
                    "weight": 1.0,
                    "metadata": {"segment_turn": turn_id},
                })

    return edges


def _build_happened_at_edges(activity_nodes: Dict[str, dict],
                             environment_nodes: Dict[str, dict]) -> List[dict]:
    """HAPPENED_AT: activity -> environment, by time overlap."""
    if not environment_nodes:
        return []

    edges = []
    for act_nid, act_node in activity_nodes.items():
        tw = act_node["metadata"]["time_window"]
        act_start = tw.get("start", 0.0)
        act_end = tw.get("end", 0.0)

        for env_nid, env_node in environment_nodes.items():
            env_first = env_node["metadata"].get("first_seen", 0.0)
            env_last = env_node["metadata"].get("last_seen", 0.0)
            if not isinstance(env_first, (int, float)) or not isinstance(env_last, (int, float)):
                continue
            if env_first <= act_end and env_last >= act_start:
                edges.append({
                    "source": act_nid,
                    "target": env_nid,
                    "edge_type": "HAPPENED_AT",
                    "weight": 1.0,
                })

    return edges


def _build_implies_edges(inferred_nodes: Dict[str, dict],
                         activity_nodes: Dict[str, dict]) -> List[dict]:
    """IMPLIES: pattern -> activities that fall within the pattern's time_range."""
    edges = []
    for pat_nid, pat_node in inferred_nodes.items():
        time_range = pat_node["metadata"].get("time_range", {})
        if not isinstance(time_range, dict):
            continue
        pat_start = time_range.get("start")
        pat_end = time_range.get("end")
        if pat_start is None or pat_end is None:
            continue

        for act_nid, act_node in activity_nodes.items():
            tw = act_node["metadata"]["time_window"]
            act_start = tw.get("start", 0.0)
            act_end = tw.get("end", 0.0)
            if act_start >= pat_start and act_end <= pat_end:
                edges.append({
                    "source": pat_nid,
                    "target": act_nid,
                    "edge_type": "IMPLIES",
                    "weight": 1.0,
                })

    return edges


# Adjacency list

def _build_adjacency(edges: List[dict], all_node_ids: Set[str]) -> Dict[str, list]:
    adj: Dict[str, list] = {nid: [] for nid in all_node_ids}
    for e in edges:
        src = e["source"]
        tgt = e["target"]
        entry = {"target": tgt, "edge_type": e["edge_type"]}
        if src in adj:
            adj[src].append(entry)
        reverse_entry = {"target": src, "edge_type": e["edge_type"]}
        if tgt in adj:
            adj[tgt].append(reverse_entry)
    return adj


# Per-video graph builder

def build_video_graph(
    video_id: str,
    video_data: dict,
    inferred_knowledge: Optional[dict] = None,
) -> dict:
    """Build the complete graph for a single video."""
    activity_log = video_data.get("activity_log", [])
    if not isinstance(activity_log, list):
        activity_log = []
    object_registry = video_data.get("object_registry", {})
    if not isinstance(object_registry, dict):
        object_registry = {}
    environment_log = video_data.get("environment_log", [])
    if not isinstance(environment_log, list):
        environment_log = []
    if inferred_knowledge is None:
        inferred_knowledge = video_data.get("inferred_knowledge", {})
    if not isinstance(inferred_knowledge, dict):
        inferred_knowledge = {}

    activity_nodes = _build_activity_nodes(activity_log)
    object_nodes = _build_object_nodes(object_registry)
    environment_nodes = _build_environment_nodes(environment_log)
    inferred_nodes = _build_inferred_nodes(inferred_knowledge)

    all_nodes: Dict[str, dict] = {}
    all_nodes.update(activity_nodes)
    all_nodes.update(object_nodes)
    all_nodes.update(environment_nodes)
    all_nodes.update(inferred_nodes)

    edges: List[dict] = []
    edges.extend(_build_next_edges(activity_nodes))
    edges.extend(_build_used_in_edges(activity_log, object_registry))
    edges.extend(_build_located_at_edges(object_nodes, environment_nodes))
    edges.extend(_build_co_occurs_edges(activity_log, object_registry))
    edges.extend(_build_happened_at_edges(activity_nodes, environment_nodes))
    edges.extend(_build_implies_edges(inferred_nodes, activity_nodes))

    valid_ids = set(all_nodes.keys())
    edges = [e for e in edges if e["source"] in valid_ids and e["target"] in valid_ids]

    adjacency = _build_adjacency(edges, valid_ids)

    return {
        "nodes": all_nodes,
        "edges": edges,
        "adjacency": adjacency,
    }


# Raw observation loader

def _load_jsonl_as_aggregated(path: str) -> Dict[str, dict]:
    """Aggregate Segment Encoder JSONL records by video."""
    import copy
    videos: Dict[str, dict] = {}

    with open(path) as f:
        for line in f:
            seg = json.loads(line.strip())
            vid = seg.get("video_id", "unknown")
            turn_id = seg.get("turn_id", 0)
            tw = seg.get("time_window", {})

            if vid not in videos:
                videos[vid] = {
                    "environment_log": [],
                    "object_registry": {},
                    "activity_log": [],
                    "inferred_knowledge": {},
                }

            vd = videos[vid]

            env = seg.get("environment", {})
            if env:
                vd["environment_log"].append({
                    "location_id": f"segment_{turn_id}",
                    "turn_id": turn_id,
                    "first_seen": tw.get("start", 0),
                    "last_seen": tw.get("end", 0),
                    "current_state": env,
                })

            obj_reg = seg.get("object_registry", {})
            if isinstance(obj_reg, dict):
                for oid, odata in obj_reg.items():
                    if not isinstance(odata, dict):
                        continue
                    odata_copy = copy.deepcopy(odata)
                    odata_copy["last_seen_turn"] = turn_id
                    odata_copy["last_seen_time"] = tw.get("end", 0)
                    odata_copy.setdefault("first_seen_time", tw.get("start", 0))
                    vd["object_registry"][oid] = odata_copy

            activity = seg.get("activity_narrative", {})
            if activity:
                vd["activity_log"].append({
                    "turn_id": turn_id,
                    "time_window": tw,
                    "summary": activity.get("summary", ""),
                    "detailed_narrative": activity.get("detailed_narrative", ""),
                    "action_breakdown": activity.get("action_breakdown", []),
                })

            knowledge = seg.get("inferred_knowledge", {})
            if knowledge:
                for cat, patterns in knowledge.items():
                    if cat not in vd["inferred_knowledge"]:
                        vd["inferred_knowledge"][cat] = {}
                    if isinstance(patterns, dict):
                        for k, v in patterns.items():
                            vd["inferred_knowledge"][cat][f"t{turn_id}_{k}"] = v

    logger.info("Loaded JSONL: %d videos", len(videos))
    return videos


# Participant graph builder


def _group_action_sequences_by_video(
    inferred_knowledge: dict,
    video_ids: Set[str],
) -> Dict[str, dict]:
    """Assign consolidated action sequences to their supporting videos."""
    grouped = {video_id: {"action_sequences": []} for video_id in video_ids}
    if not isinstance(inferred_knowledge, dict):
        return grouped

    sequences = inferred_knowledge.get("action_sequences", [])
    if not isinstance(sequences, list):
        return grouped

    for sequence in sequences:
        if not isinstance(sequence, dict):
            continue
        provenance = sequence.get("provenance", {})
        source_videos = provenance.get("source_videos", []) if isinstance(provenance, dict) else []
        if not isinstance(source_videos, list):
            continue
        for video_id in source_videos:
            if video_id in grouped:
                grouped[video_id]["action_sequences"].append(sequence)
    return grouped


def build_graph_memory(
    memory_data: dict,
    video_ids: Optional[List[str]] = None,
    evaluation_setting: Optional[str] = None,
    participant_id: Optional[str] = None,
) -> dict:
    """Build one participant-level Graph-2D artifact."""
    all_videos = memory_data.get("memories_by_video", {})
    if not isinstance(all_videos, dict):
        raise ValueError("memories_by_video must be a JSON object")

    selected_video_ids = sorted(all_videos) if video_ids is None else [
        video_id for video_id in video_ids if video_id in all_videos
    ]
    inferred_by_video = _group_action_sequences_by_video(
        memory_data.get("inferred_knowledge", {}),
        set(selected_video_ids),
    )

    graphs_by_video: Dict[str, dict] = {}
    nodes_by_type: Dict[str, int] = defaultdict(int)
    edges_by_type: Dict[str, int] = defaultdict(int)

    for index, video_id in enumerate(selected_video_ids, 1):
        graph = build_video_graph(
            video_id,
            all_videos[video_id],
            inferred_knowledge=inferred_by_video[video_id],
        )
        graphs_by_video[video_id] = graph

        video_nodes_by_type: Dict[str, int] = defaultdict(int)
        for node in graph["nodes"].values():
            node_type = node["node_type"]
            nodes_by_type[node_type] += 1
            video_nodes_by_type[node_type] += 1

        video_edges_by_type: Dict[str, int] = defaultdict(int)
        for edge in graph["edges"]:
            edge_type = edge["edge_type"]
            edges_by_type[edge_type] += 1
            video_edges_by_type[edge_type] += 1

        logger.info(
            "[%3d/%d] %-12s nodes=%4d (%s) edges=%4d (%s)",
            index,
            len(selected_video_ids),
            video_id,
            len(graph["nodes"]),
            ", ".join(f"{key}={value}" for key, value in sorted(video_nodes_by_type.items())),
            len(graph["edges"]),
            ", ".join(f"{key}={value}" for key, value in sorted(video_edges_by_type.items())),
        )

    output = {
        "graph_format": "graph_2d",
        "videos_processed": selected_video_ids,
        "stats": {
            "num_nodes": sum(nodes_by_type.values()),
            "num_edges": sum(edges_by_type.values()),
            "nodes_by_type": dict(nodes_by_type),
            "edges_by_type": dict(edges_by_type),
        },
        "graphs_by_video": graphs_by_video,
    }
    if evaluation_setting:
        output["evaluation_setting"] = evaluation_setting
    resolved_participant = participant_id or memory_data.get("participant_id")
    if not resolved_participant and selected_video_ids:
        participant_prefixes = {
            video_id.split("_", 1)[0]
            for video_id in selected_video_ids
            if "_" in video_id
        }
        if len(participant_prefixes) == 1:
            resolved_participant = participant_prefixes.pop()
    if resolved_participant:
        output["participant_id"] = resolved_participant
    return output


# Command-line interface

def main():
    parser = argparse.ArgumentParser(
        description="Build the controlled Graph-2D memory representation",
    )
    parser.add_argument("--input", required=True,
                        help="Participant memory JSON or Segment Encoder JSONL")
    parser.add_argument("--output", required=True,
                        help="Output Graph-2D JSON")
    parser.add_argument("--video-ids", default=None,
                        help="Comma-separated list of video IDs to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only print statistics without writing output")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument(
        "--evaluation-setting",
        choices=["graph_2d_raw", "graph_2d_edited"],
        default=None,
        help="Optional paper condition recorded in the output",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Loading memory source from %s", args.input)
    t0 = time.time()

    if args.input.endswith(".jsonl"):
        memory_data = {
            "memories_by_video": _load_jsonl_as_aggregated(args.input),
            "inferred_knowledge": {},
        }
    else:
        with open(args.input) as file:
            memory_data = json.load(file)
    all_videos = memory_data.get("memories_by_video", {})

    logger.info("Loaded in %.1fs (%d videos)", time.time() - t0, len(all_videos))
    if args.video_ids:
        requested = [v.strip() for v in args.video_ids.split(",")]
        missing = [v for v in requested if v not in all_videos]
        if missing:
            logger.warning("Video IDs not found in participant memory: %s", missing)
        video_ids = [v for v in requested if v in all_videos]
    else:
        video_ids = sorted(all_videos.keys())

    logger.info("Processing %d / %d videos", len(video_ids), len(all_videos))

    output_data = build_graph_memory(
        memory_data,
        video_ids=video_ids,
        evaluation_setting=args.evaluation_setting,
    )
    logger.info("Total nodes: %d", output_data["stats"]["num_nodes"])
    logger.info("Total edges: %d", output_data["stats"]["num_edges"])

    if args.dry_run:
        logger.info("Dry run; skipping output write")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        t0 = time.time()
        with open(args.output, "w") as f:
            json.dump(output_data, f, ensure_ascii=False)
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        logger.info("Wrote %s (%.1f MB) in %.1fs", args.output, size_mb, time.time() - t0)

    logger.info("Done.")


if __name__ == "__main__":
    main()
