"""Memory-context construction for MEMORA-Planning."""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class PlanningContextMixin:
    """Compose procedure, activity, entity, and place evidence for planning."""

    @staticmethod
    def _format_action_breakdown(ab_list: list) -> str:
        """Format action_breakdown into a concise step sequence."""
        steps = []
        for ab in ab_list:
            if not isinstance(ab, dict):
                continue
            action = ab.get("action", "")
            obj = ab.get("object", "")
            hand = ab.get("hand", "")
            direction = ab.get("direction", "")
            parts = [action]
            if obj:
                parts.append(obj)
            detail = []
            if hand:
                detail.append(f"{hand} hand")
            if direction:
                detail.append(direction)
            if detail:
                parts.append(f"({', '.join(detail)})")
            steps.append(" ".join(parts))
        return " → ".join(steps)

    _STOP_WORDS = frozenset({
        "help", "the", "a", "an", "and", "to", "in", "on", "at", "of",
        "for", "with", "this", "that", "from", "person", "kitchen",
        "please", "p01", "p02", "p03", "entire", "all",
    })

    _COMPOUND_EXPANSIONS = {
        "countertop": "counter", "worktop": "counter",
        "faucet": "tap", "cooktop": "stove",
        "refrigerator": "fridge", "rubbish": "trash",
        "dishcloth": "cloth", "sponge": "cloth",
    }

    _GENERIC_VERBS = frozenset({
        "clean", "wash", "prepare", "cook", "make", "get", "put",
        "take", "pick", "move", "use", "set", "clear", "handle",
    })

    @staticmethod
    def _stem(word: str) -> str:
        """Minimal suffix stripping for keyword matching."""
        w = word.lower()
        for suffix in ("ing", "tion", "ed", "es", "er", "ly"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[: -len(suffix)]
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            return w[:-1]
        return w

    @classmethod
    def _normalize_terms(cls, words: set) -> set:
        """Stem words and expand compounds for matching."""
        stems = set()
        for w in words:
            stems.add(cls._stem(w))
            stems.add(w)
            if w in cls._COMPOUND_EXPANSIONS:
                stems.add(cls._COMPOUND_EXPANSIONS[w])
                stems.add(cls._stem(cls._COMPOUND_EXPANSIONS[w]))
        return stems

    @classmethod
    def _extract_nouns_verbs(cls, text: str) -> tuple:
        """Split text terms into (noun-like, verb-like) sets after stemming."""
        words = set(text.lower().split())
        raw = {w for w in words if len(w) > 2 and w not in cls._STOP_WORDS}
        normed = cls._normalize_terms(raw)
        nouns = {t for t in normed if cls._stem(t) not in cls._GENERIC_VERBS}
        verbs = normed - nouns
        return nouns, verbs

    @classmethod
    def _goal_is_relevant(cls, task_query: str, goal_name: str,
                          activity_summaries: list = None) -> bool:
        """Check whether a subgoal is relevant to the task using keyword overlap.

        Requires at least one NOUN (non-generic-verb) overlap between the task
        and the goal (or its activity summaries).  Pure verb overlap (e.g.,
        both mention "clean") is not sufficient.
        """
        if not goal_name:
            return True

        task_nouns, task_verbs = cls._extract_nouns_verbs(task_query)
        if not task_nouns and not task_verbs:
            return True

        # Check goal name
        goal_nouns, _ = cls._extract_nouns_verbs(goal_name)
        if task_nouns & goal_nouns:
            return True

        # Check activity summaries
        if activity_summaries:
            all_text = " ".join(activity_summaries)
            sum_nouns, _ = cls._extract_nouns_verbs(all_text)
            if task_nouns & sum_nouns:
                return True

        return False

    @staticmethod
    def _decompose_task_query(task_query: str) -> List[str]:
        """Split a task query into atomic sub-goal search queries.

        Uses rule-based parsing:
        1. Split on conjunctions / commas / "and" / "then"
        2. Extract verb+object pairs from each fragment
        3. Also extract bare object nouns for object-level searches
        """
        import re as _re

        q = task_query.lower()
        # Remove "help Pxx" prefix
        q = _re.sub(r"^help\s+p\d+\s*", "", q).strip().rstrip(".")

        # Split on commas, "and", "then", semicolons
        fragments = _re.split(r"\s*(?:,\s*and\s+|,\s+and\s+|,\s+|\band\b|\bthen\b|;)\s*", q)
        fragments = [f.strip() for f in fragments if f.strip() and len(f.strip()) > 2]

        sub_queries = []
        seen = set()

        # Extract listed objects: "with plate, bowl, and cutlery" → plate, bowl, cutlery
        obj_list_match = _re.search(
            r"(?:with|using|including)\s+(.+)", q
        )
        if obj_list_match:
            obj_text = obj_list_match.group(1)
            objs = _re.split(r"\s*(?:,\s*and\s+|,\s+and\s+|,\s+|\band\b)\s*", obj_text)
            for obj in objs:
                obj = obj.strip().rstrip(".")
                if obj and len(obj) > 2 and obj not in seen:
                    sub_queries.append(obj)
                    seen.add(obj)

        for frag in fragments:
            if frag not in seen:
                sub_queries.append(frag)
                seen.add(frag)

        # If only one fragment and it contains "using/with", also add the
        # phrase without the "using ..." clause as a procedure search
        if len(fragments) <= 1:
            core = _re.sub(r"\s+(?:using|with|including)\s+.*$", "", q).strip()
            if core and core not in seen:
                sub_queries.insert(0, core)
                seen.add(core)

        return sub_queries if sub_queries else [q]

    def _build_memory_context(self, task_query: str) -> Tuple[str, str]:
        """Build activity-centric memory context for the planning agent.

        Returns (context_string, confidence_level) where confidence_level is
        one of "high", "medium", or "low":
          high   – matching procedure sequences found
          medium – relevant activities/objects found but no procedure match
          low    – almost nothing relevant found

        Context includes activity breakdowns, temporal neighbors, entity
        records, and inferred knowledge when those stores are present.
        """
        memory_tools = self._env.memory_tools
        sections = []
        seen_objects = set()
        cm = getattr(memory_tools, "current_memory", {})
        obj_reg = cm.get("object_registry", {})

        include_breakdown = True
        include_temporal = True
        include_objects = True
        include_patterns = True

        # --- 1. Activity-centric retrieval with continuous sequence expansion ---
        # Use compositional decomposition for richer retrieval
        sub_goals = self._decompose_task_query(task_query)
        task_words = task_query.lower().split()
        key_nouns = [w for w in task_words if len(w) > 3 and w not in
                     {"help", "with", "this", "that", "person", "kitchen", "please"}]
        search_queries = [task_query] + sub_goals + key_nouns[:3]
        # Deduplicate while preserving order
        _seen_q = set()
        unique_queries = []
        for sq in search_queries:
            sq_lower = sq.lower().strip()
            if sq_lower not in _seen_q:
                _seen_q.add(sq_lower)
                unique_queries.append(sq)
        search_queries = unique_queries[:8]

        activity_results = []
        seen_summaries = set()
        for query in search_queries:
            try:
                results = memory_tools.search(query, top_k_per_category=5, category="activities")
                acts = results.get("activities", [])
                for act in acts:
                    if not isinstance(act, dict):
                        continue
                    summary = act.get("summary", "")
                    if summary in seen_summaries:
                        continue
                    seen_summaries.add(summary)
                    activity_results.append(act)
            except Exception as exc:
                logger.warning("Planning context retrieval failed for %r: %s", query, exc)

        activity_log = cm.get("activity_log", [])

        # Build per-video index: (source_video, turn_id) -> index in merged log.
        # Coerce turn_id to int so later lookups with int(t) also match memory
        # files that store identifiers as strings.
        vid_turn_index: dict = {}
        for i, a in enumerate(activity_log):
            vid = a.get("source_video", "")
            tid = a.get("turn_id")
            if vid and tid is not None:
                try:
                    tid = int(tid)
                except (TypeError, ValueError):
                    continue
                vid_turn_index[(vid, tid)] = i

        # Build continuous procedure sequence using goal_turn_range
        MAX_GOAL_TURNS = 15
        FALLBACK_WINDOW = 5
        sequence_sections = []
        seen_goal_ranges = set()
        if activity_results and activity_log and include_temporal:
            for hit in activity_results[:3]:
                hit_time = hit.get("time", {})
                if isinstance(hit_time, dict):
                    anchor_start = hit_time.get("start", -1)
                else:
                    anchor_start = -1

                anchor_idx = None
                anchor_video = ""
                if anchor_start >= 0:
                    for idx, a in enumerate(activity_log):
                        tw = a.get("time_window", {})
                        if abs(tw.get("start", -999) - anchor_start) < 1.0:
                            anchor_idx = idx
                            anchor_video = a.get("source_video", "")
                            break

                if anchor_idx is None:
                    continue

                anchor_act = activity_log[anchor_idx]
                goal_range = anchor_act.get("goal_turn_range", [])
                goal_name = anchor_act.get("high_level_goal", "")

                # Dedup by (video, goal_start, goal_end)
                if goal_range and len(goal_range) == 2:
                    goal_start, goal_end = goal_range
                    # Some memory variants store turn ids as strings ("12") instead
                    # of ints. Coerce here so the arithmetic below cannot crash.
                    try:
                        goal_start = int(goal_start)
                        goal_end = int(goal_end)
                    except (TypeError, ValueError):
                        # Malformed range — skip this anchor entirely.
                        continue
                    range_key = (anchor_video, goal_start, goal_end)
                    if range_key in seen_goal_ranges:
                        continue
                    seen_goal_ranges.add(range_key)

                    lo_turn, hi_turn = goal_start, goal_end
                    if hi_turn - lo_turn + 1 > MAX_GOAL_TURNS:
                        half = MAX_GOAL_TURNS // 2
                        try:
                            anchor_turn = int(anchor_act.get("turn_id", lo_turn))
                        except (TypeError, ValueError):
                            anchor_turn = lo_turn
                        lo_turn = max(goal_start, anchor_turn - half)
                        hi_turn = min(goal_end, lo_turn + MAX_GOAL_TURNS - 1)

                    # Collect indices for this video's goal range
                    goal_indices = []
                    for t in range(lo_turn, hi_turn + 1):
                        idx = vid_turn_index.get((anchor_video, t))
                        if idx is not None:
                            goal_indices.append(idx)
                    if not goal_indices:
                        lo = max(0, anchor_idx - FALLBACK_WINDOW)
                        hi = min(len(activity_log) - 1, anchor_idx + FALLBACK_WINDOW)
                        goal_indices = list(range(lo, hi + 1))
                else:
                    # No goal range — use fallback window within same video
                    goal_indices = []
                    for delta in range(-FALLBACK_WINDOW, FALLBACK_WINDOW + 1):
                        idx = anchor_idx + delta
                        if 0 <= idx < len(activity_log):
                            if activity_log[idx].get("source_video", "") == anchor_video or not anchor_video:
                                goal_indices.append(idx)
                    goal_name = goal_name or "Unknown"

                if not goal_indices:
                    continue

                # --- Relevance gate: skip goals unrelated to the task ---
                range_summaries = [
                    activity_log[idx].get("summary", "")
                    for idx in goal_indices
                    if idx < len(activity_log)
                ]
                if not self._goal_is_relevant(task_query, goal_name, range_summaries):
                    logger.info(
                        "  Skipping irrelevant goal '%s' (no keyword overlap with task)",
                        goal_name,
                    )
                    continue

                src_label = f" (from {anchor_video})" if anchor_video else ""
                goal_label = f'"{goal_name}"' if goal_name else "related activities"
                seq_lines = [
                    f"Subgoal: {goal_label}{src_label} "
                    f"(turns {activity_log[goal_indices[0]].get('turn_id','?')}"
                    f"-{activity_log[goal_indices[-1]].get('turn_id','?')}):"
                ]
                for step_num, idx in enumerate(goal_indices, 1):
                    a = activity_log[idx]
                    tw = a.get("time_window", {})
                    s = a.get("summary", "")
                    ab = a.get("action_breakdown", [])
                    marker = " ◀ best match" if idx == anchor_idx else ""
                    step_line = (
                        f"  {step_num}. [{tw.get('start',0):.0f}-{tw.get('end',0):.0f}s] "
                        f"{s[:120]}{marker}"
                    )
                    seq_lines.append(step_line)
                    if include_breakdown and ab:
                        ab_str = self._format_action_breakdown(ab)
                        seq_lines.append(f"     Actions: {ab_str}")
                    if include_objects:
                        for abd in ab:
                            obj_name = abd.get("object", "")
                            if obj_name and len(obj_name) > 2:
                                seen_objects.add(obj_name)
                sequence_sections.append("\n".join(seq_lines))

        # Assemble procedure sequences (goal-aligned) + individual hits
        if sequence_sections or activity_results:
            full_section = ""
            if sequence_sections:
                full_section += "### Observed Procedure Sequences (from memory)\n"
                full_section += "Each sequence shows the COMPLETE subgoal as observed. "
                full_section += "Use the action steps as a template for your plan.\n\n"
                full_section += "\n\n".join(sequence_sections)

            if activity_results:
                extra_lines = []
                for i, act in enumerate(activity_results[:5], 1):
                    summary = act.get("summary", "")
                    time_info = act.get("time", "")
                    ctx = act.get("_context", {})
                    ab = act.get("action_breakdown", [])

                    header = f"{i}. \"{summary}\""
                    if time_info and isinstance(time_info, dict):
                        header += f" ({time_info.get('start',0):.0f}-{time_info.get('end',0):.0f}s)"
                    extra_lines.append(header)

                    if include_breakdown and ab:
                        extra_lines.append(f"   Steps: {self._format_action_breakdown(ab)}")

                    if include_objects:
                        scene_objs = ctx.get("objects_involved", []) if isinstance(ctx, dict) else []
                        if isinstance(scene_objs, list):
                            for so in scene_objs:
                                if isinstance(so, dict):
                                    seen_objects.add(so.get("object_id", so.get("name", "")))

                if full_section:
                    full_section += "\n\n"
                full_section += "### Additional Relevant Activities\n" + "\n".join(extra_lines)

            sections.append(full_section)

        # --- 2. Objects from matched activities + memory attributes ---
        if include_objects:
            obj_lines = []
            if seen_objects and obj_reg:
                for oid in seen_objects:
                    odata = obj_reg.get(oid, {})
                    if not isinstance(odata, dict):
                        for k, v in obj_reg.items():
                            if isinstance(v, dict) and v.get("name", "").lower() == oid.lower():
                                odata = v
                                break
                    if not odata:
                        continue
                    name = odata.get("name", oid)
                    vp = odata.get("visual_properties", {})
                    sp = odata.get("spatial_info", {})
                    color = vp.get("color", "")
                    material = vp.get("material", "")
                    loc = sp.get("location", odata.get("last_location", ""))
                    desc = f"{color} {material} {name}".strip()
                    if loc:
                        desc += f" (at {loc})"
                    obj_lines.append(f"  - {desc}")

            if not obj_lines and obj_reg:
                for oid, odata in list(obj_reg.items())[:10]:
                    if isinstance(odata, dict):
                        name = odata.get("name", oid)
                        vp = odata.get("visual_properties", {})
                        sp = odata.get("spatial_info", {})
                        color = vp.get("color", "")
                        material = vp.get("material", "")
                        loc = sp.get("location", odata.get("last_location", ""))
                        desc = f"{color} {material} {name}".strip()
                        if loc:
                            desc += f" (at {loc})"
                        obj_lines.append(f"  - {desc}")

            if obj_lines:
                sections.append("### Objects in This Kitchen\n" + "\n".join(obj_lines[:15]))

        # --- 3. Habits & patterns ---
        if include_patterns:
            knowledge = cm.get("inferred_knowledge", {})
            prefs = knowledge.get("preferences", {})
            pref_lines = []
            for cat_key in ["storage_preferences", "organizational_habits", "workflow_patterns",
                            "activity_patterns", "temporal_patterns"]:
                items = prefs.get(cat_key, [])
                if isinstance(items, list):
                    for item in items[:3]:
                        if isinstance(item, dict):
                            if "object" in item and "preferred_location" in item:
                                pref_lines.append(f"  - {item['object']} → usually at {item['preferred_location']}")
                            elif "habit" in item:
                                pref_lines.append(f"  - Habit: {item['habit'][:120]}")
                            elif "pattern" in item:
                                pref_lines.append(f"  - Pattern: {item['pattern'][:120]}")
                            elif "description" in item:
                                pref_lines.append(f"  - {item['description'][:120]}")
            if pref_lines:
                sections.append("### Person's Habits & Preferences\n" + "\n".join(pref_lines))

        ctx = "\n\n".join(sections) if sections else ""
        if len(ctx) > 6000:
            ctx = ctx[:6000] + "\n... [truncated]"

        if sequence_sections:
            task_nouns, task_verbs = self._extract_nouns_verbs(task_query)
            any_verb_match = False
            for sec in sequence_sections:
                sec_nouns, sec_verbs = self._extract_nouns_verbs(sec)
                if task_verbs & sec_verbs:
                    any_verb_match = True
                    break
            confidence = "high" if any_verb_match else "medium"
        elif activity_results:
            confidence = "medium"
        else:
            confidence = "low"

        return ctx, confidence
