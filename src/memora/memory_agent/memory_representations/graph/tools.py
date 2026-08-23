#!/usr/bin/env python3
"""
Read-time retrieval for the Graph-2D baseline.

Operates on the released participant-specific graphs whose nodes are typed
(activity, object, environment, pattern) and connected via edges
(NEXT, USED_IN, LOCATED_AT, CO_OCCURS, HAPPENED_AT, IMPLIES).

Provides the same external interface as ``TypedMemoryTools`` so it can be
used as a drop-in replacement in ``agent_environment.py``.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from memora.memory_agent.tools.embedding import get_model, resolve_device

logger = logging.getLogger(__name__)

# Node-type → search-result category mapping
_NODE_TYPE_TO_CATEGORY = {
    "object": "objects",
    "activity": "activities",
    "environment": "environment",
    "pattern": "patterns",
}

_SEARCH_CATEGORIES = ("objects", "activities", "environment", "patterns")


class GraphMemoryTools:
    """Memory tools for the Graph-2D entity-relation baseline.

    Drop-in replacement for ``TypedMemoryTools`` – exposes identical
    ``set_temporal_context``, ``set_participant_context``, ``search``,
    ``get_state_at_time``, ``get_object_history``, ``get_tools_definition``
    and ``execute_tool`` methods.
    """

    DEFAULT_TOP_K = 5

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        graph_file: str,
        e5_model_name: str = "intfloat/e5-base-v2",
        device: Optional[str] = None,
        include_tips: bool = False,
    ):
        self.graph_file = graph_file
        self.e5_model_name = e5_model_name
        self.device = resolve_device(device)
        self.include_tips = include_tips

        self.full_graph = self._load_graph(graph_file)

        self.context_type: Optional[str] = None
        self.current_video_id: Optional[str] = None
        self.time_threshold: Optional[float] = None
        self.participant_videos: Optional[Set[str]] = None

        # Active (possibly filtered) graph used by search methods
        self.current_graph: Optional[Dict[str, Any]] = None

        self.query_cache: Dict[str, Any] = {}

        # Pre-computed embeddings keyed by ``(video_id, node_id)``
        self._node_embeddings: Optional[np.ndarray] = None
        self._node_ids: Optional[List[str]] = None

        self._precompute_embeddings()

        logger.info(
            "GraphMemoryTools initialised – %d video(s), embeddings shape %s",
            len(self.full_graph.get("graphs_by_video", {})),
            self._node_embeddings.shape if self._node_embeddings is not None else "N/A",
        )

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_graph(self, graph_file: str) -> Dict[str, Any]:
        path = Path(graph_file)
        if not path.exists():
            raise FileNotFoundError(f"Graph memory file not found: {graph_file}")
        logger.info("Loading Graph-2D memory from %s", graph_file)
        with open(path) as f:
            data = json.load(f)
        fmt = data.get("graph_format", "")
        if fmt != "graph_2d":
            raise ValueError(
                f"Unexpected graph_format {fmt!r} in {graph_file}; expected 'graph_2d'"
            )
        return data

    # ------------------------------------------------------------------
    # Embedding pre-computation
    # ------------------------------------------------------------------

    def _precompute_embeddings(self) -> None:
        """Compute E5 embeddings for every node's *content* field across all videos."""
        all_ids: List[str] = []
        all_texts: List[str] = []

        for _vid, vgraph in self.full_graph.get("graphs_by_video", {}).items():
            for node_id, node in vgraph.get("nodes", {}).items():
                all_ids.append(node_id)
                all_texts.append(f"passage: {node.get('content', '')}")

        if not all_ids:
            logger.warning("No nodes found in graph – nothing to embed")
            return

        model = get_model(self.e5_model_name, self.device)
        if model is None:
            logger.warning("E5 model unavailable; semantic search will use keyword matching")
            return

        logger.info("Computing E5 embeddings for %d nodes …", len(all_ids))
        embeddings = model.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)
        self._node_embeddings = np.asarray(embeddings, dtype=np.float32)
        self._node_ids = all_ids
        logger.info("Embeddings computed: shape %s", self._node_embeddings.shape)

    # ------------------------------------------------------------------
    # Context setters (API-compatible with TypedMemoryTools)
    # ------------------------------------------------------------------

    def set_temporal_context(
        self,
        video_id: str,
        ask_turn_id: Optional[int] = None,
        time_threshold_seconds: Optional[float] = None,
    ) -> None:
        """Filter the graph to a single video up to *time_threshold_seconds*.

        turn_id semantics follow the same convention as ``TypedMemoryTools``:
        ``time_threshold = (turn_id + 1) * 10`` seconds.
        """
        self.query_cache = {}

        if ask_turn_id is not None and time_threshold_seconds is None:
            time_threshold_seconds = (ask_turn_id + 1) * 10
            logger.info(
                "turn_id %d → time threshold %.1fs", ask_turn_id, time_threshold_seconds
            )

        self.context_type = "episodic"
        self.current_video_id = video_id
        self.time_threshold = time_threshold_seconds
        self.participant_videos = None

        video_graph = self.full_graph.get("graphs_by_video", {}).get(video_id)
        if video_graph is None:
            logger.warning("Video '%s' not found in graph", video_id)
            self.current_graph = {"nodes": {}, "edges": [], "adjacency": {}}
            return

        self.current_graph = self._filter_graph_by_time(video_graph, time_threshold_seconds)

        logger.info(
            "Temporal context: video=%s, time<=%.1fs – %d nodes, %d edges",
            video_id,
            time_threshold_seconds or 0,
            len(self.current_graph["nodes"]),
            len(self.current_graph["edges"]),
        )

    def set_participant_context(
        self,
        participant_id: str,
        video_ids: Optional[List[str]] = None,
    ) -> None:
        """Merge graphs from all videos belonging to *participant_id*."""
        self.query_cache = {}

        all_vids = set(self.full_graph.get("graphs_by_video", {}).keys())
        if video_ids:
            self.participant_videos = set(video_ids)
        else:
            self.participant_videos = {v for v in all_vids if v.startswith(participant_id)}

        self.context_type = "semantic"
        self.current_video_id = None
        self.time_threshold = None

        self.current_graph = self._merge_graphs(self.participant_videos)

        logger.info(
            "Participant context: %s – %d video(s), %d nodes, %d edges",
            participant_id,
            len(self.participant_videos),
            len(self.current_graph["nodes"]),
            len(self.current_graph["edges"]),
        )

    # ------------------------------------------------------------------
    # Graph filtering / merging
    # ------------------------------------------------------------------

    def _filter_graph_by_time(
        self, video_graph: Dict[str, Any], threshold: Optional[float]
    ) -> Dict[str, Any]:
        """Return a copy of *video_graph* retaining only nodes whose timestamp
        is ``<= threshold`` or ``< 0`` (patterns are timeless)."""
        if threshold is None:
            return copy.deepcopy(video_graph)

        kept_nodes: Dict[str, dict] = {}
        for nid, node in video_graph.get("nodes", {}).items():
            ts = node.get("timestamp", 0.0)
            if ts < 0 or ts <= threshold:
                kept_nodes[nid] = node

        kept_ids = set(kept_nodes.keys())

        kept_edges = [
            e
            for e in video_graph.get("edges", [])
            if e["source"] in kept_ids and e["target"] in kept_ids
        ]

        kept_adj: Dict[str, list] = {}
        for nid in kept_ids:
            orig = video_graph.get("adjacency", {}).get(nid, [])
            kept_adj[nid] = [
                entry for entry in orig if entry["target"] in kept_ids
            ]

        return {"nodes": kept_nodes, "edges": kept_edges, "adjacency": kept_adj}

    def _merge_graphs(self, video_ids: Set[str]) -> Dict[str, Any]:
        """Merge multiple per-video graphs into one, prefixing node IDs with
        the video ID to avoid collisions."""
        merged_nodes: Dict[str, dict] = {}
        merged_edges: List[dict] = []
        merged_adj: Dict[str, list] = {}

        for vid in sorted(video_ids):
            vgraph = self.full_graph.get("graphs_by_video", {}).get(vid)
            if vgraph is None:
                continue

            for nid, node in vgraph.get("nodes", {}).items():
                prefixed = f"{vid}/{nid}"
                merged_node = copy.deepcopy(node)
                merged_node["node_id"] = prefixed
                merged_node["_source_video"] = vid
                merged_nodes[prefixed] = merged_node

            for edge in vgraph.get("edges", []):
                merged_edges.append({
                    **edge,
                    "source": f"{vid}/{edge['source']}",
                    "target": f"{vid}/{edge['target']}",
                })

            for nid, neighbors in vgraph.get("adjacency", {}).items():
                prefixed = f"{vid}/{nid}"
                merged_adj[prefixed] = [
                    {**entry, "target": f"{vid}/{entry['target']}"}
                    for entry in neighbors
                ]

        return {"nodes": merged_nodes, "edges": merged_edges, "adjacency": merged_adj}

    # ------------------------------------------------------------------
    # Core graph search
    # ------------------------------------------------------------------

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        hops: int = 1,
    ) -> Dict[str, Any]:
        """Semantic + structural graph search.

        1. Embed *query* with E5.
        2. Find top-K seed nodes by cosine similarity within ``current_graph``.
        3. Expand via *hops*-hop BFS over the adjacency list.
        4. Return seed nodes, expanded neighbours, and a formatted text summary.
        """
        if self.current_graph is None:
            return {"error": "No context set. Call set_temporal_context or set_participant_context first."}

        current_nodes = self.current_graph["nodes"]
        adjacency = self.current_graph.get("adjacency", {})

        if not current_nodes:
            return {"seed_nodes": [], "expanded_nodes": [], "subgraph_text": "No nodes available."}

        seed_nodes = self._find_seed_nodes(query, current_nodes, top_k)

        seed_ids = {n["node_id"] for n in seed_nodes}
        expanded_ids = self._expand_hops(seed_ids, adjacency, current_nodes, hops)

        expanded_nodes = [current_nodes[nid] for nid in sorted(expanded_ids) if nid in current_nodes]

        subgraph_text = self._format_subgraph(seed_nodes, expanded_nodes, adjacency)

        return {
            "seed_nodes": seed_nodes,
            "expanded_nodes": expanded_nodes,
            "subgraph_text": subgraph_text,
        }

    # ------------------------------------------------------------------
    # Seed-node retrieval (semantic or keyword fallback)
    # ------------------------------------------------------------------

    def _find_seed_nodes(
        self,
        query: str,
        current_nodes: Dict[str, dict],
        top_k: int,
    ) -> List[dict]:
        model = get_model(self.e5_model_name, self.device)

        if model is not None and self._node_embeddings is not None and self._node_ids is not None:
            return self._semantic_seed_search(query, model, current_nodes, top_k)

        return self._keyword_seed_search(query, current_nodes, top_k)

    def _semantic_seed_search(
        self,
        query: str,
        model: Any,
        current_nodes: Dict[str, dict],
        top_k: int,
    ) -> List[dict]:
        """Use pre-computed embeddings + cosine similarity."""
        current_ids_set = set(current_nodes.keys())

        # Participant context may merge with prefixed IDs, so
        # strip the prefix to match against the global embedding list and
        # also handle the raw (unprefixed) IDs for episodic context.
        # Build a mapping from global embedding index → current_graph node ID.
        idx_to_current: List[Tuple[int, str]] = []
        for global_idx, global_nid in enumerate(self._node_ids):
            if global_nid in current_ids_set:
                idx_to_current.append((global_idx, global_nid))
            else:
                # For merged graphs the current_graph keys are "vid/nid";
                # the global embedding list uses raw "nid".
                for cur_nid in current_ids_set:
                    if cur_nid.endswith(f"/{global_nid}"):
                        idx_to_current.append((global_idx, cur_nid))
                        break

        if not idx_to_current:
            return self._keyword_seed_search(query, current_nodes, top_k)

        indices = np.array([i for i, _ in idx_to_current], dtype=np.int64)
        cur_nids = [c for _, c in idx_to_current]
        sub_embeddings = self._node_embeddings[indices]  # (N, D)

        query_emb = model.encode(
            [f"query: {query}"], normalize_embeddings=True, show_progress_bar=False
        )
        query_emb = np.asarray(query_emb, dtype=np.float32)  # (1, D)

        sims = (sub_embeddings @ query_emb.T).squeeze(-1)  # (N,)

        k = min(top_k, len(sims))
        top_indices = np.argsort(sims)[::-1][:k]

        results: List[dict] = []
        for ti in top_indices:
            nid = cur_nids[int(ti)]
            node = current_nodes[nid]
            results.append({
                **node,
                "similarity": float(sims[int(ti)]),
            })
        return results

    def _keyword_seed_search(
        self,
        query: str,
        current_nodes: Dict[str, dict],
        top_k: int,
    ) -> List[dict]:
        """Simple keyword overlap fallback when E5 is unavailable."""
        query_lower = query.lower()
        query_tokens = set(query_lower.split())

        scored: List[Tuple[float, str]] = []
        for nid, node in current_nodes.items():
            content = node.get("content", "").lower()
            if not content:
                continue
            content_tokens = set(content.split())
            overlap = len(query_tokens & content_tokens)
            if overlap > 0 or query_lower in content:
                score = overlap + (1.0 if query_lower in content else 0.0)
                scored.append((score, nid))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [current_nodes[nid] for _, nid in scored[:top_k]]

    # ------------------------------------------------------------------
    # N-hop expansion
    # ------------------------------------------------------------------

    _SKIP_EDGE_TYPES_FOR_HOP = frozenset({"CO_OCCURS"})

    @staticmethod
    def _expand_hops(
        seed_ids: Set[str],
        adjacency: Dict[str, list],
        current_nodes: Dict[str, dict],
        hops: int,
    ) -> Set[str]:
        """BFS expansion from *seed_ids* for *hops* hops.

        Skips noisy edge types (CO_OCCURS) to keep expansions relevant.
        """
        visited = set(seed_ids)
        frontier = set(seed_ids)

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for nid in frontier:
                for neighbor in adjacency.get(nid, []):
                    if neighbor.get("edge_type") in GraphMemoryTools._SKIP_EDGE_TYPES_FOR_HOP:
                        continue
                    tgt = neighbor["target"]
                    if tgt not in visited and tgt in current_nodes:
                        next_frontier.add(tgt)
                        visited.add(tgt)
            frontier = next_frontier
            if not frontier:
                break

        return visited - seed_ids

    # ------------------------------------------------------------------
    # Subgraph formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_subgraph(
        seed_nodes: List[dict],
        expanded_nodes: List[dict],
        adjacency: Dict[str, list],
    ) -> str:
        lines: List[str] = []

        lines.append("=== Seed Nodes ===")
        for node in seed_nodes:
            sim = node.get("similarity")
            sim_str = f" (sim={sim:.3f})" if sim is not None else ""
            lines.append(
                f"[{node.get('node_type', '?')}] {node['node_id']}{sim_str}: "
                f"{node.get('content', '')[:200]}"
            )
            for nb in adjacency.get(node["node_id"], []):
                lines.append(f"  --{nb['edge_type']}--> {nb['target']}")

        if expanded_nodes:
            lines.append("")
            lines.append("=== Expanded Neighbours ===")
            for node in expanded_nodes:
                lines.append(
                    f"[{node.get('node_type', '?')}] {node['node_id']}: "
                    f"{node.get('content', '')[:200]}"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Canonical search() interface – 4-category result format
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k_per_category: int = 3,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return graph results in the canonical MEMORA search format.

        Runs ``graph_search`` then maps results into the canonical
        ``{query, objects, activities, environment, patterns}`` dict.
        """
        if category is not None:
            category = {"object_registry": "objects", "activity_log": "activities"}.get(
                category, category
            )
            if category not in _SEARCH_CATEGORIES:
                logger.warning("Unknown category '%s', searching all", category)
                category = None

        cache_key = (
            query,
            self.current_video_id,
            self.context_type,
            category or "all",
            top_k_per_category,
        )
        if cache_key in self.query_cache:
            logger.info("Returning cached results for '%s'", query)
            return self.query_cache[cache_key]

        total_k = top_k_per_category * 4
        gs = self.graph_search(query, top_k=total_k, hops=1)

        if "error" in gs:
            return {"query": query, "objects": [], "activities": [], "environment": [], "patterns": [], "_error": gs["error"]}

        results: Dict[str, Any] = {
            "query": query,
            "objects": [],
            "activities": [],
            "environment": [],
            "patterns": [],
        }

        all_nodes = gs.get("seed_nodes", []) + gs.get("expanded_nodes", [])
        adjacency = self.current_graph.get("adjacency", {}) if self.current_graph else {}

        seen_ids: Set[str] = set()
        for node in all_nodes:
            nid = node.get("node_id", "")
            if nid in seen_ids:
                continue
            seen_ids.add(nid)

            cat = _NODE_TYPE_TO_CATEGORY.get(node.get("node_type"), None)
            if cat is None:
                continue
            if category is not None and cat != category:
                continue
            if len(results[cat]) >= top_k_per_category:
                continue

            entry = self._node_to_result_entry(node, cat, adjacency)
            results[cat].append(entry)

        self.query_cache[cache_key] = results
        return results

    def _node_to_result_entry(
        self,
        node: dict,
        category: str,
        adjacency: Dict[str, list],
    ) -> Dict[str, Any]:
        """Convert a graph node into the result dict expected by the agent."""
        nid = node["node_id"]
        meta = node.get("metadata", {})
        sim = node.get("similarity")

        if category == "activities":
            tw = meta.get("time_window", {})
            if isinstance(tw, dict):
                s, e = tw.get("start", 0), tw.get("end", 0)
                time_str = f"{int(s)}-{int(e)}s"
            else:
                time_str = str(tw)
            entry: Dict[str, Any] = {
                "time": time_str,
                "summary": meta.get("summary", node.get("content", "")),
            }

            context: Dict[str, Any] = {}
            objects_involved: List[str] = []
            env_desc: Optional[str] = None
            prev_action: Optional[str] = None
            next_action: Optional[str] = None

            for nb in adjacency.get(nid, []):
                tgt_id = nb["target"]
                tgt_node = self.current_graph["nodes"].get(tgt_id) if self.current_graph else None
                if tgt_node is None:
                    continue
                etype = nb["edge_type"]

                if etype == "USED_IN" and tgt_node.get("node_type") == "object":
                    objects_involved.append(tgt_node.get("metadata", {}).get("name", tgt_id))
                elif etype == "USED_IN" and tgt_node.get("node_type") == "activity":
                    objects_involved.append(tgt_node.get("metadata", {}).get("summary", tgt_id))
                elif etype == "HAPPENED_AT":
                    env_desc = tgt_node.get("content", "")
                elif etype == "NEXT":
                    ts_self = node.get("timestamp", 0)
                    ts_other = tgt_node.get("timestamp", 0)
                    summary = tgt_node.get("metadata", {}).get("summary", tgt_node.get("content", ""))
                    if ts_other > ts_self:
                        next_action = summary
                    else:
                        prev_action = summary

            # Also collect objects from USED_IN edges pointing TO this activity
            for nb in adjacency.get(nid, []):
                tgt_id = nb["target"]
                tgt_node = self.current_graph["nodes"].get(tgt_id) if self.current_graph else None
                if tgt_node is None:
                    continue
                if nb["edge_type"] == "USED_IN" and tgt_node.get("node_type") == "object":
                    name = tgt_node.get("metadata", {}).get("name", tgt_id)
                    if name not in objects_involved:
                        objects_involved.append(name)

            if objects_involved:
                context["objects_involved"] = objects_involved
            if env_desc:
                context["environment"] = {"description": env_desc}
            if prev_action:
                context["previous_action"] = prev_action
            if next_action:
                context["next_action"] = next_action

            if context:
                entry["_context"] = context
            if sim is not None:
                entry["similarity"] = sim
            return entry

        if category == "objects":
            entry = {
                "object_id": nid,
                "name": meta.get("name", ""),
                "spatial_info": {"location": meta.get("location", "")},
                "state": {"current_state": meta.get("state", "")},
            }
            if sim is not None:
                entry["similarity"] = sim
            return entry

        if category == "environment":
            entry = {
                "location_id": meta.get("location_id", nid),
                "layout_description": node.get("content", ""),
                "first_seen": meta.get("first_seen", 0.0),
                "last_seen": meta.get("last_seen", 0.0),
            }
            if sim is not None:
                entry["similarity"] = sim
            return entry

        # patterns
        entry = {
            "knowledge_id": meta.get("knowledge_id", nid),
            "title": meta.get("title", ""),
            "goal": meta.get("goal", ""),
            "key_objects": meta.get("key_objects", []),
            "content": node.get("content", ""),
        }
        if sim is not None:
            entry["similarity"] = sim
        return entry

    # ------------------------------------------------------------------
    # get_state_at_time
    # ------------------------------------------------------------------

    def get_state_at_time(self, time_seconds: float) -> Dict[str, Any]:
        """Point-in-time snapshot using graph edges for cross-layer context."""
        if self.current_graph is None:
            return {"error": "No context set."}

        nodes = self.current_graph["nodes"]
        adjacency = self.current_graph.get("adjacency", {})

        result: Dict[str, Any] = {
            "time": time_seconds,
            "visible_objects": [],
            "environment": None,
            "current_activity": None,
        }

        # Objects visible at *time_seconds*: first_seen <= t (use node timestamp
        # as a proxy for first_seen; objects with timestamp=0 are always visible).
        for nid, node in nodes.items():
            if node.get("node_type") != "object":
                continue
            ts = node.get("timestamp", 0.0)
            if ts < 0:
                continue
            if ts <= time_seconds:
                meta = node.get("metadata", {})
                result["visible_objects"].append({
                    "object_id": nid,
                    "name": meta.get("name", ""),
                    "location": meta.get("location", ""),
                    "state": meta.get("state", ""),
                })

        # Current activity: the activity node whose time window spans *time_seconds*
        best_act = None
        best_dist = float("inf")
        for nid, node in nodes.items():
            if node.get("node_type") != "activity":
                continue
            tw = node.get("metadata", {}).get("time_window", {})
            start = tw.get("start", 0.0)
            end = tw.get("end", 0.0)
            if start <= time_seconds <= end:
                best_act = node
                best_dist = 0
                break
            dist = min(abs(time_seconds - start), abs(time_seconds - end))
            if dist < best_dist:
                best_dist = dist
                best_act = node

        if best_act is not None:
            meta = best_act.get("metadata", {})
            tw = meta.get("time_window", {})
            if isinstance(tw, dict):
                s, e = tw.get("start", 0), tw.get("end", 0)
                time_str = f"{int(s)}-{int(e)}s"
            else:
                time_str = str(tw)
            result["current_activity"] = {
                "time": time_str,
                "summary": meta.get("summary", best_act.get("content", "")),
            }
            # Attach environment via HAPPENED_AT edge
            for nb in adjacency.get(best_act["node_id"], []):
                if nb["edge_type"] == "HAPPENED_AT":
                    env_node = nodes.get(nb["target"])
                    if env_node:
                        result["environment"] = {
                            "location_id": env_node.get("metadata", {}).get("location_id", ""),
                            "layout_description": env_node.get("content", ""),
                        }
                        break

        # If environment still None, find by time overlap
        if result["environment"] is None:
            for nid, node in nodes.items():
                if node.get("node_type") != "environment":
                    continue
                meta = node.get("metadata", {})
                first = meta.get("first_seen", 0.0)
                last = meta.get("last_seen", float("inf"))
                if isinstance(first, (int, float)) and isinstance(last, (int, float)):
                    if first <= time_seconds <= last:
                        result["environment"] = {
                            "location_id": meta.get("location_id", ""),
                            "layout_description": node.get("content", ""),
                        }
                        break

        return result

    # ------------------------------------------------------------------
    # get_object_history
    # ------------------------------------------------------------------

    def get_object_history(self, object_query: str) -> Dict[str, Any]:
        """Track an object across activities via USED_IN edges.

        Returns a chronologically-ordered list of activities involving the object.
        """
        if self.current_graph is None:
            return {"error": "No context set."}

        nodes = self.current_graph["nodes"]
        adjacency = self.current_graph.get("adjacency", {})

        # --- locate the object node ---
        obj_node, obj_nid = self._find_object_node(object_query, nodes)
        if obj_node is None:
            return {"error": f"Object '{object_query}' not found in graph."}

        meta = obj_node.get("metadata", {})

        # Gather all activities connected via USED_IN (the adjacency is
        # bidirectional, so activity neighbours are inspected directly.)
        activity_entries: List[Dict[str, Any]] = []
        for nb in adjacency.get(obj_nid, []):
            tgt_node = nodes.get(nb["target"])
            if tgt_node is None:
                continue
            if tgt_node.get("node_type") == "activity":
                tw = tgt_node.get("metadata", {}).get("time_window", {})
                start = tw.get("start", 0.0) if isinstance(tw, dict) else 0.0
                end = tw.get("end", 0.0) if isinstance(tw, dict) else 0.0
                activity_entries.append({
                    "activity_id": tgt_node["node_id"],
                    "time": f"{int(start)}-{int(end)}s",
                    "summary": tgt_node.get("metadata", {}).get("summary", tgt_node.get("content", "")),
                    "_sort_key": start,
                })

        activity_entries.sort(key=lambda x: x.pop("_sort_key", 0.0))

        # Collect locations and environments from edges
        locations: List[str] = []
        loc = meta.get("location", "")
        if loc:
            locations.append(loc)
        for nb in adjacency.get(obj_nid, []):
            if nb["edge_type"] == "LOCATED_AT":
                env_node = nodes.get(nb["target"])
                if env_node:
                    env_loc = env_node.get("metadata", {}).get("location_id", "")
                    if env_loc and env_loc not in locations:
                        locations.append(env_loc)

        return {
            "object_id": obj_nid,
            "name": meta.get("name", object_query),
            "current_state": meta.get("state", ""),
            "current_location": meta.get("location", ""),
            "all_locations_observed": locations,
            "activity_history": activity_entries,
            "num_activities": len(activity_entries),
        }

    def _find_object_node(
        self, query: str, nodes: Dict[str, dict]
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Find the best-matching object node for *query*."""
        query_lower = query.lower()

        # Pass 1: exact or substring on name / node_id
        for nid, node in nodes.items():
            if node.get("node_type") != "object":
                continue
            name = node.get("metadata", {}).get("name", "").lower()
            if query_lower == name or query_lower == nid.lower():
                return node, nid
            if query_lower in name or query_lower in nid.lower():
                return node, nid

        # Pass 2: semantic similarity among object nodes
        model = get_model(self.e5_model_name, self.device)
        if model is None:
            return None, None

        obj_nodes = [
            (nid, node) for nid, node in nodes.items() if node.get("node_type") == "object"
        ]
        if not obj_nodes:
            return None, None

        texts = [f"passage: {n.get('metadata', {}).get('name', '')}" for _, n in obj_nodes]
        embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        q_emb = model.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False)

        sims = (np.asarray(embs) @ np.asarray(q_emb).T).squeeze(-1)
        best_idx = int(np.argmax(sims))
        if sims[best_idx] < 0.3:
            return None, None
        return obj_nodes[best_idx][1], obj_nodes[best_idx][0]

    # ------------------------------------------------------------------
    # Tool definitions (OpenAI-format, compatible with TypedMemoryTools)
    # ------------------------------------------------------------------

    def get_tools_definition(
        self,
        allow_category: bool = False,
        allow_graph_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        search_properties: Dict[str, Any] = {
            "query": {
                "type": "string",
                "description": (
                    "Free-form search query (e.g., 'cloth', 'washing plate', "
                    "'turn off tap', 'coffee routine')"
                ),
            },
        }
        if allow_category:
            search_properties["category"] = {
                "type": "string",
                "description": (
                    "Optional. Search only this category: "
                    "'objects', 'activities', 'environment', 'patterns'."
                ),
            }

        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": (
                        "Unified search across ALL memory categories. Returns results from:\n"
                        "- objects: Items and their locations/states\n"
                        "- activities: Timestamped actions\n"
                        "- environment: Locations and spatial info\n"
                        "- patterns: Behavioral habits\n\n"
                        "Use this for ANY question about what/where/when/how."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": search_properties,
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_state_at_time",
                    "description": (
                        "Get complete state snapshot at a SPECIFIC time point.\n"
                        "Returns: visible objects, environment, and current activity.\n\n"
                        "Use for questions like:\n"
                        '- "At 0.0s, was X visible?"\n'
                        '- "What was happening at time X?"'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_seconds": {
                                "type": "number",
                                "description": "Time point in seconds (e.g., 0.0, 30.5, 120.0)",
                            }
                        },
                        "required": ["time_seconds"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_object_history",
                    "description": (
                        "Get COMPLETE history of an object across activities.\n\n"
                        "Returns:\n"
                        "- all_locations_observed: ALL locations the object has been in\n"
                        "- activity_history: Chronological list of activities involving the object\n\n"
                        "Use for:\n"
                        '- "Was the plate ever dirty?"\n'
                        '- "Did the cup move during the video?"\n'
                        '- "What was the fork used for?"'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "object_query": {
                                "type": "string",
                                "description": "Object name to track (e.g., 'plate', 'cup', 'fork')",
                            }
                        },
                        "required": ["object_query"],
                    },
                },
            },
        ]

        if allow_graph_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": "graph_search",
                    "description": (
                        "Graph-aware search: finds semantically similar nodes "
                        "then expands via N-hop graph traversal to discover "
                        "connected context (objects, environments, activities)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "hops": {
                                "type": "integer",
                                "description": "Number of graph hops to expand (default 1)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            })

        return tools

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        if tool_name == "search":
            return self.search(
                arguments.get("query", ""),
                top_k_per_category=arguments.get("top_k_per_category", 3),
                category=arguments.get("category"),
            )
        if tool_name == "get_state_at_time":
            return self.get_state_at_time(arguments.get("time_seconds", 0.0))
        if tool_name == "get_object_history":
            return self.get_object_history(arguments.get("object_query", ""))
        if tool_name == "graph_search":
            return self.graph_search(
                arguments.get("query", ""),
                top_k=arguments.get("top_k", self.DEFAULT_TOP_K),
                hops=arguments.get("hops", 1),
            )

        logger.warning("Unknown tool '%s'", tool_name)
        return {"error": f"Unknown tool: {tool_name}"}
