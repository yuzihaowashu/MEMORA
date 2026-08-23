#!/usr/bin/env python3
"""Extract Replay task candidates from EPIC-KITCHENS-100 annotations.

Segments action annotations into coherent candidate tasks for the released
MEMORA-Planning Replay protocol.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from memora_bench.paths import memora_planning_root

logger = logging.getLogger(__name__)

CLEANUP_VERBS = frozenset({
    "wash", "rinse", "clean", "wipe", "dry", "sponge", "scrub", "squeeze",
})
COOKING_VERBS = frozenset({
    "fry", "boil", "cook", "bake", "roast", "simmer", "heat",
    "stir", "flip", "turn-over", "pour-into", "pour-from", "pour",
    "mix", "season", "add", "fold", "stretch", "roll", "roll-out",
    "shape", "knead", "spread",
})
PREP_VERBS = frozenset({
    "cut", "chop", "peel", "slice", "grate", "crush", "mash", "break",
})
FILLER_VERBS = frozenset({
    "pick-up", "put-down", "open", "close", "take", "put", "get", "hold",
    "put-in", "put-into", "put-on", "put-onto", "move", "move-into",
})
SETUP_VERBS = frozenset({
    "turn-on", "turn-off", "switch-on", "switch-off", "adjust",
    "check", "shake", "scoop", "sprinkle", "fill-with", "empty",
})
UTENSIL_NOUNS = frozenset({
    "spoon", "knife", "fork", "spatula", "tongs", "ladle", "wooden spoon",
    "rolling pin", "whisk", "grater", "peeler", "scissors", "chopstick",
    "cloth", "towel", "sponge", "brush", "tap", "hand", "finger", "lid",
    "bottle", "bag", "bin", "container", "box", "tray", "timer", "button",
    "hob", "sink", "surface", "counter",
})
TASK_TYPES = [
    "meal_preparation",
    "cleanup_organization",
    "object_retrieval_setup",
    "multi_step_cooking",
    "routine_reproduction",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_target_video_ids(path: str) -> set[str]:
    """Load the released participant-video selection from JSONL."""
    video_ids = set()
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            video_id = record.get("video_id")
            if not video_id:
                raise ValueError(
                    f"Missing video_id in {path} at line {line_number}"
                )
            video_ids.add(video_id)
    if not video_ids:
        raise ValueError(f"No video IDs found in {path}")
    return video_ids


def timestamp_to_seconds(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def clean_noun(noun: str) -> str:
    """EPIC-Kitchens 'base:modifier' → natural language form."""
    if ":" not in noun:
        return noun
    base, modifier = noun.split(":", 1)
    if " " in modifier:
        return f"{base} {modifier}"
    return f"{modifier} {base}"


def get_verb_category(verb: str) -> str:
    if verb in CLEANUP_VERBS:
        return "cleanup"
    if verb in COOKING_VERBS:
        return "cooking"
    if verb in PREP_VERBS:
        return "prep"
    return "general"


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_video(df, max_gap: float,
                  min_steps: int, max_steps: int) -> list:
    if len(df) < min_steps:
        return []

    df = df.copy()
    df["start_sec"] = df["start_timestamp"].apply(timestamp_to_seconds)
    df["stop_sec"] = df["stop_timestamp"].apply(timestamp_to_seconds)
    df["verb_cat"] = df["verb"].apply(get_verb_category)

    splits = _find_split_points(df, max_gap)
    raw = [df.iloc[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
    merged = _merge_short(raw, min_steps)
    final = _split_long(merged, min_steps, max_steps)
    return [s for s in final if min_steps <= len(s) <= max_steps]


def _find_split_points(df: pd.DataFrame, max_gap: float) -> list:
    points = [0]
    for i in range(1, len(df)):
        gap = max(0, df.iloc[i]["start_sec"] - df.iloc[i - 1]["stop_sec"])

        if gap > max_gap:
            points.append(i)
            continue

        # Medium gap + sustained verb-category transition
        prev_cat = df.iloc[i - 1]["verb_cat"]
        curr_cat = df.iloc[i]["verb_cat"]
        if (gap > max_gap / 3
                and prev_cat != curr_cat
                and prev_cat != "general"
                and curr_cat != "general"
                and i + 1 < len(df)
                and df.iloc[i + 1]["verb_cat"] == curr_cat):
            points.append(i)

    points.append(len(df))
    return points


def _merge_short(segments: list, min_steps: int) -> list:
    if not segments:
        return []
    merged = []
    buf = segments[0]
    for seg in segments[1:]:
        if len(buf) < min_steps:
            buf = pd.concat([buf, seg])
        else:
            merged.append(buf)
            buf = seg
    if len(buf) >= min_steps:
        merged.append(buf)
    elif merged:
        merged[-1] = pd.concat([merged[-1], buf])
    return merged


def _split_long(segments: list, min_steps: int, max_steps: int) -> list:
    result = []
    target_size = (min_steps + max_steps) // 2

    for seg in segments:
        n = len(seg)
        if n <= max_steps:
            result.append(seg)
            continue

        num_chunks = max(
            -(-n // max_steps),                        # ceil(n / max_steps)
            min(n // min_steps, -(-n // target_size)),  # prefer target_size
        )
        chunk_base = n // num_chunks
        remainder = n % num_chunks

        pos = 0
        for i in range(num_chunks):
            size = chunk_base + (1 if i < remainder else 0)
            result.append(seg.iloc[pos:pos + size])
            pos += size

    return result


# ---------------------------------------------------------------------------
# Classification & query generation
# ---------------------------------------------------------------------------

def classify_task_type(verbs: list, nouns: list) -> str:
    meaningful = [v for v in verbs if v not in FILLER_VERBS]
    if not meaningful:
        storage_nouns = {"cupboard", "drawer", "fridge", "shelf", "cabinet", "bin", "bag"}
        if set(nouns) & storage_nouns:
            return "object_retrieval_setup"
        return "routine_reproduction"

    total = len(meaningful)
    counts = Counter(meaningful)
    cleanup = sum(counts.get(v, 0) for v in CLEANUP_VERBS)
    cooking = sum(counts.get(v, 0) for v in COOKING_VERBS)
    prep = sum(counts.get(v, 0) for v in PREP_VERBS)
    setup = sum(counts.get(v, 0) for v in SETUP_VERBS)

    if cleanup / total > 0.35:
        return "cleanup_organization"
    if cooking / total > 0.3:
        return "multi_step_cooking"
    if prep / total > 0.3:
        return "meal_preparation"
    if setup / total > 0.3:
        return "object_retrieval_setup"

    max_ratio = max(cleanup / total, cooking / total, prep / total, setup / total)
    if max_ratio < 0.15:
        return "routine_reproduction"

    if prep + cooking > cleanup and prep > 0 and cooking > 0:
        return "multi_step_cooking"
    if cooking > cleanup:
        return "multi_step_cooking"
    if setup >= cleanup and setup >= cooking:
        return "object_retrieval_setup"
    if prep > cooking:
        return "meal_preparation"
    if cleanup > 0:
        return "cleanup_organization"
    return "object_retrieval_setup"


def _best_noun(noun_counts: Counter) -> str:
    """Pick the most frequent non-utensil noun; fall back to most frequent."""
    for noun, _ in noun_counts.most_common():
        cleaned = clean_noun(noun)
        if noun not in UTENSIL_NOUNS and cleaned not in UTENSIL_NOUNS:
            return cleaned
    return clean_noun(noun_counts.most_common(1)[0][0]) if noun_counts else "items"


def generate_task_query(pid: str, verbs: list, nouns: list,
                        task_type: str) -> str:
    verb_set = set(verbs)
    noun_counts = Counter(nouns)
    top_noun = _best_noun(noun_counts)

    if task_type == "cleanup_organization":
        if verb_set & {"wash", "rinse"}:
            dishware = {"plate", "bowl", "cup", "glass", "mug", "pot", "pan", "dish"}
            if set(nouns) & dishware:
                return f"Help {pid} wash and dry the dishes"
            return f"Help {pid} wash the {top_noun}"
        if verb_set & {"wipe", "clean"}:
            return f"Help {pid} wipe down the kitchen surfaces"
        return f"Help {pid} clean up the kitchen after cooking"

    if task_type == "multi_step_cooking":
        if verb_set & {"fry", "stir"}:
            return f"Help {pid} stir-fry the {top_noun}"
        if verb_set & {"boil", "simmer"}:
            return f"Help {pid} boil and prepare the {top_noun}"
        if verb_set & {"pour-into", "pour", "mix"}:
            return f"Help {pid} mix and cook the {top_noun}"
        if verb_set & {"fold", "stretch", "roll", "roll-out", "knead", "shape"}:
            return f"Help {pid} prepare and shape the {top_noun}"
        return f"Help {pid} cook a dish with {top_noun}"

    if task_type == "meal_preparation":
        if verb_set & {"cut", "chop", "slice"}:
            return f"Help {pid} cut and prepare the {top_noun}"
        if verb_set & {"peel"}:
            return f"Help {pid} peel and prepare the {top_noun}"
        return f"Help {pid} prepare the {top_noun} for cooking"

    if task_type == "object_retrieval_setup":
        storage = {"cupboard", "drawer", "fridge", "shelf", "cabinet"}
        if set(nouns) & storage:
            return f"Help {pid} get items from storage and set up"
        return f"Help {pid} organize the kitchen workspace"

    return f"Help {pid} complete the kitchen routine"


# ---------------------------------------------------------------------------
# Task building & scoring
# ---------------------------------------------------------------------------

def build_task(seg: pd.DataFrame, video_id: str, pid: str,
               seg_idx: int) -> dict:
    verbs = seg["verb"].tolist()
    nouns = seg["noun"].tolist()
    task_type = classify_task_type(verbs, nouns)
    query = generate_task_query(pid, verbs, nouns, task_type)

    steps = []
    for _, row in seg.iterrows():
        steps.append({
            "narration_id": row["narration_id"],
            "narration": row["narration"],
            "verb": row["verb"],
            "noun": row["noun"],
            "start_time": row["start_timestamp"],
            "end_time": row["stop_timestamp"],
        })

    time_span = seg["stop_sec"].iloc[-1] - seg["start_sec"].iloc[0]

    return {
        "task_id": f"plan_{video_id}_{seg_idx:03d}",
        "participant_id": pid,
        "video_id": video_id,
        "task_type": task_type,
        "task_query": query,
        "ground_truth_steps": steps,
        "num_steps": len(steps),
        "time_span_seconds": round(time_span, 1),
        "primary_objects": [clean_noun(n) for n, _ in Counter(nouns).most_common(5)],
        "primary_verbs": [v for v, _ in Counter(verbs).most_common(5)],
    }


def score_task(task: dict, min_steps: int, max_steps: int) -> float:
    n = task["num_steps"]
    ideal = (min_steps + max_steps) / 2
    length_score = 1.0 - abs(n - ideal) / max_steps

    unique_verbs = len(set(task["primary_verbs"]))
    diversity_score = unique_verbs / max(len(task["primary_verbs"]), 1)

    meaningful_ratio = sum(
        1 for s in task["ground_truth_steps"] if s["verb"] not in FILLER_VERBS
    ) / max(n, 1)

    return length_score + diversity_score + meaningful_ratio


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def select_diverse_tasks(candidates: dict,
                         target_per_participant: int) -> list:
    selected = []
    max_per_type = (target_per_participant // len(TASK_TYPES)) + 2

    for pid in sorted(candidates):
        pool = candidates[pid]
        if not pool:
            continue

        pool.sort(key=lambda x: x[0], reverse=True)
        type_counts: Counter = Counter()
        chosen, overflow = [], []

        for score, task in pool:
            tt = task["task_type"]
            if type_counts[tt] < max_per_type and len(chosen) < target_per_participant:
                chosen.append(task)
                type_counts[tt] += 1
            else:
                overflow.append((score, task))

        for _, task in overflow:
            if len(chosen) >= target_per_participant:
                break
            chosen.append(task)

        logger.info(f"{pid}: {len(chosen)} tasks selected from {len(pool)} candidates")
        type_dist = Counter(t["task_type"] for t in chosen)
        for tt in TASK_TYPES:
            if type_dist[tt]:
                logger.info(f"  {tt}: {type_dist[tt]}")

        selected.extend(chosen)

    return selected


def deduplicate_queries(tasks: list) -> list:
    used: set = set()
    for task in tasks:
        pid = task["participant_id"]
        base_query = task["task_query"]
        candidate = base_query
        key = (pid, candidate)

        if key not in used:
            used.add(key)
            continue

        # Try appending different primary objects to disambiguate
        resolved = False
        for obj in task["primary_objects"]:
            candidate = f"{base_query} ({obj})"
            key = (pid, candidate)
            if key not in used:
                task["task_query"] = candidate
                used.add(key)
                resolved = True
                break

        if not resolved:
            candidate = f"{base_query} [{task['video_id']}]"
            task["task_query"] = candidate
            used.add((pid, candidate))

    return tasks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_tasks(csv_path: str, memory_jsonl: str, min_steps: int,
                  max_steps: int, max_gap: float,
                  video_ids_file: str,
                  target_per_participant: int = 12) -> list:
    df = pd.read_csv(csv_path)
    target_vids = load_target_video_ids(video_ids_file)

    memory_vids: set = set()
    with open(memory_jsonl) as f:
        for line in f:
            memory_vids.add(json.loads(line)["video_id"])

    valid_vids = target_vids & memory_vids & set(df["video_id"].unique())
    logger.info(f"Valid video IDs: {len(valid_vids)} (target ∩ participant memory ∩ CSV)")

    filtered = df[df["video_id"].isin(valid_vids)]
    logger.info(f"Filtered annotations: {len(filtered)}")

    candidates: dict = {}

    for vid in sorted(filtered["video_id"].unique()):
        vid_df = (filtered[filtered["video_id"] == vid]
                  .sort_values("start_timestamp")
                  .reset_index(drop=True))
        pid = vid_df.iloc[0]["participant_id"]
        segments = segment_video(vid_df, max_gap, min_steps, max_steps)

        if pid not in candidates:
            candidates[pid] = []

        for idx, seg in enumerate(segments):
            task = build_task(seg, vid, pid, idx)
            score = score_task(task, min_steps, max_steps)
            candidates[pid].append((score, task))

    tasks = select_diverse_tasks(candidates, target_per_participant)
    tasks = deduplicate_queries(tasks)
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Extract planning tasks from EPIC-Kitchens annotations",
    )
    benchmark_root = memora_planning_root()
    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the EPIC-KITCHENS-100 training annotation CSV",
    )
    parser.add_argument(
        "--memory-jsonl",
        required=True,
        help="Segment Encoder observation JSONL (segment_observations.jsonl)",
    )
    parser.add_argument(
        "--video-ids-file",
        default=str(benchmark_root / "data" / "participant_video_ids.jsonl"),
        help="Released participant-video selection in JSONL format",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for extracted task candidates",
    )
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--max-gap-seconds", type=float, default=30.0)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print task summaries without writing the file",
    )
    args = parser.parse_args()

    global pd
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "Planning-task extraction requires the analysis dependencies: "
            "pip install -e '.[analysis]'"
        ) from exc

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tasks = extract_tasks(
        args.csv_path, args.memory_jsonl,
        args.min_steps, args.max_steps, args.max_gap_seconds,
        args.video_ids_file,
    )

    type_counts = Counter(t["task_type"] for t in tasks)
    logger.info(f"Total tasks: {len(tasks)}")
    for tt, c in sorted(type_counts.items()):
        logger.info(f"  {tt}: {c}")

    if args.dry_run:
        for t in tasks:
            print(f"\n{t['task_id']} [{t['task_type']}]")
            print(f"  Query: {t['task_query']}")
            print(f"  Steps: {t['num_steps']}, Span: {t['time_span_seconds']}s")
            print(f"  Verbs: {t['primary_verbs'][:3]}")
            print(f"  Objects: {t['primary_objects'][:3]}")
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(tasks, indent=2) + "\n")
        logger.info(f"Written to {args.output}")


if __name__ == "__main__":
    main()
