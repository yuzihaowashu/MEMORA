#!/usr/bin/env python3
"""Build Replay tasks from candidate EPIC-KITCHENS action sequences.

Pipeline:
  1. Filter highly repetitive tasks (>70% repeated narrations)
  2. Measure repetition and filter trivially repetitive tasks
  3. Match EPIC nouns to participant memory object_registry (color, material, location)
  4. Generate enriched narrations grounded in participant memory data
  5. (Optional) Use LLM to generate better task queries from GT narrations
  6. Write replay planning tasks

Usage:
    # participant memory grounding only (no LLM needed)
    python3 scripts/benchmark_construction/build_replay_suite.py \
        --input /path/to/benchmark_build/planning_candidates_p01.json \
        --memory-file /path/to/participant_memory.json \
        --output planning/suites/replay/p01.json

    # With LLM query improvement (needs vLLM)
    python3 scripts/benchmark_construction/build_replay_suite.py \
        --input /path/to/benchmark_build/planning_candidates_p01.json \
        --memory-file /path/to/participant_memory.json \
        --output planning/suites/replay/p01.json \
        --use-llm --model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SYNONYMS = {
    "tap": ["faucet", "water_tap"],
    "faucet": ["tap", "water_tap"],
    "cloth": ["towel", "dishcloth", "rag"],
    "towel": ["cloth", "dishcloth"],
    "hob": ["stove", "burner", "stovetop", "cooktop"],
    "stove": ["hob", "burner", "stovetop"],
    "pan": ["frying_pan", "saucepan", "skillet"],
    "pot": ["saucepan", "cooking_pot"],
    "saucepan": ["pot", "pan"],
    "bin": ["trash", "garbage", "waste"],
    "chopping board": ["cutting_board", "board"],
    "cutting board": ["chopping_board", "board"],
    "cup": ["mug", "glass"],
    "mug": ["cup"],
    "glass": ["cup", "tumbler"],
    "spoon": ["ladle", "spatula"],
    "knife": ["blade", "cutter"],
    "plate": ["dish"],
    "dish": ["plate"],
    "bowl": ["mixing_bowl", "container"],
    "bottle": ["flask", "water_bottle", "milk_bottle"],
    "bag": ["sack", "packet"],
    "lid": ["cover", "cap"],
    "container": ["box", "tub", "tupperware", "jar"],
    "tupperware": ["container", "tub"],
    "brush": ["scrubber", "washing_up_brush"],
    "sponge": ["scrubber"],
}


# ---------------------------------------------------------------------------
# Step 1 & 2: Filter and metadata
# ---------------------------------------------------------------------------

def compute_repeat_rate(task: dict) -> float:
    """Fraction of steps that are duplicates."""
    steps = task.get("ground_truth_steps", [])
    narrations = []
    for s in steps:
        n = s.get("narration", str(s)) if isinstance(s, dict) else str(s)
        narrations.append(n.lower().strip())
    if not narrations:
        return 0.0
    return 1.0 - len(set(narrations)) / len(narrations)


def filter_and_annotate(tasks: List[dict],
                        max_repeat_rate: float = 0.70) -> Tuple[List[dict], List[dict]]:
    """Add repeat_rate and filter out trivially repetitive tasks."""
    kept, removed = [], []
    for t in tasks:
        rr = compute_repeat_rate(t)
        t["repeat_rate"] = round(rr, 3)
        if rr > max_repeat_rate:
            t["filtered_reason"] = f"repeat_rate={rr:.2f}"
            removed.append(t)
        else:
            kept.append(t)
    return kept, removed


# ---------------------------------------------------------------------------
# Step 3: match EPIC nouns to participant memory
# ---------------------------------------------------------------------------

def load_memory_objects(memory_file: str) -> Dict[str, Dict[str, dict]]:
    """Load the object registry per video from enriched participant memory.

    Returns: {video_id: {object_id: object_data}}
    """
    with open(memory_file) as f:
        memory = json.load(f)

    result = {}
    memories = memory.get("memories_by_video", {})
    for vid, mem in memories.items():
        obj_reg = mem.get("object_registry", {})
        if obj_reg:
            result[vid] = obj_reg
    return result


def match_noun_to_memory(noun: str, object_registry: dict) -> Optional[dict]:
    """Match an EPIC noun to a participant memory object using multi-strategy matching.

    Strategies (in order):
    1. Exact match on object_id or name
    2. Substring match
    3. Synonym expansion + rematch
    """
    noun_lower = noun.lower().strip().replace(":", " ").replace("_", " ")
    noun_parts = set(noun_lower.split())

    best_match = None
    best_score = 0

    for obj_id, odata in object_registry.items():
        if not isinstance(odata, dict):
            continue
        obj_name = odata.get("name", obj_id).lower().strip()
        obj_id_lower = obj_id.lower().replace("_", " ")

        score = 0

        if noun_lower == obj_id_lower or noun_lower == obj_name:
            score = 10
        elif noun_lower in obj_id_lower:
            score = 8
        elif noun_lower in obj_name:
            score = 7
        elif obj_id_lower in noun_lower or obj_name in noun_lower:
            score = 6
        else:
            obj_parts = set(obj_name.split()) | set(obj_id_lower.split())
            overlap = noun_parts & obj_parts
            if overlap and len(overlap) >= len(noun_parts) * 0.5:
                score = 3 + len(overlap)

        if score == 0:
            syns = SYNONYMS.get(noun_lower, [])
            for syn in syns:
                syn_lower = syn.lower().replace("_", " ")
                if syn_lower == obj_id_lower or syn_lower == obj_name:
                    score = 6
                    break
                if syn_lower in obj_name or syn_lower in obj_id_lower:
                    score = 4
                    break

        if score > best_score:
            best_score = score
            best_match = {"object_id": obj_id, **odata, "_match_score": score}

    if best_score >= 4:
        return best_match
    return None


def enrich_step(step: dict, object_registry: dict) -> dict:
    """Enrich a single GT step with participant memory object data."""
    if not isinstance(step, dict):
        return step

    noun = step.get("noun", "")
    if not noun:
        return step

    matched = match_noun_to_memory(noun, object_registry)
    if not matched:
        return step

    vp = matched.get("visual_properties", {})
    si = matched.get("spatial_info", {})

    color = vp.get("color", "")
    material = vp.get("material", "")
    location = si.get("location", "")
    memory_name = matched.get("name", "")

    parts = []
    if color and color.lower() not in ("unknown", "n/a", "null"):
        parts.append(color)
    if material and material.lower() not in ("unknown", "n/a", "null"):
        parts.append(material)

    narration = step.get("narration", "")
    verb = step.get("verb", "").replace("-", " ")
    clean_noun = noun.replace(":", " ")

    if parts:
        descriptor = " ".join(parts)
        enriched_obj = f"the {descriptor} {clean_noun}"
    elif memory_name and memory_name.lower() != clean_noun.lower():
        enriched_obj = f"the {memory_name}"
    else:
        enriched_obj = f"the {clean_noun}"

    enriched_narr = narration
    if verb and enriched_obj:
        enriched_narr = narration.replace(clean_noun, enriched_obj.replace("the ", "", 1), 1)
        if location and location.lower() not in ("unknown",):
            loc = location.strip()
            if loc.startswith(("on ", "in ", "near ", "under ", "above ", "beside ")):
                enriched_narr += f" {loc}"
            else:
                enriched_narr += f" at {loc}"

    step["enriched_narration"] = enriched_narr
    step["matched_object"] = {
        "memory_id": matched.get("object_id", ""),
        "memory_name": memory_name,
        "color": color,
        "material": material,
        "location": location,
    }
    return step


# ---------------------------------------------------------------------------
# Step 4: LLM query improvement
# ---------------------------------------------------------------------------

QUERY_IMPROVEMENT_PROMPT = """Given action annotations from a kitchen video, generate a natural task request that someone would say to a robot assistant.

Participant: {participant_id}
Action narrations (what the person actually did):
{narrations_text}

Current task query (too vague): "{current_query}"

Requirements:
- Describe WHAT to do, not the exact step sequence
- Mention the 1-2 key objects involved
- Be natural and concise (8-15 words)
- Start with "Help {participant_id} ..."
- Do NOT list the steps

Output ONLY the rewritten query, nothing else."""


def build_query_prompts(tasks: List[dict]) -> List[Tuple[int, str]]:
    """Build LLM prompts for task query improvement."""
    prompts = []
    for i, t in enumerate(tasks):
        gt = t.get("ground_truth_steps", [])
        narrations = []
        for s in gt:
            n = s.get("narration", str(s)) if isinstance(s, dict) else str(s)
            narrations.append(n)
        unique_narrs = list(dict.fromkeys(narrations))[:8]
        narr_text = "\n".join(f"- {n}" for n in unique_narrs)

        prompt = QUERY_IMPROVEMENT_PROMPT.format(
            participant_id=t.get("participant_id", ""),
            narrations_text=narr_text,
            current_query=t.get("task_query", ""),
        )
        prompts.append((i, prompt))
    return prompts


def improve_queries_with_llm(tasks: List[dict], model_name: str,
                             gpu_memory: float = 0.85) -> List[dict]:
    """Use vLLM to generate better task queries."""
    from memora.memory_agent.agent import VLLMInference

    logger.info("Initializing vLLM for query improvement...")
    model = VLLMInference(
        model_name=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory,
        max_model_len=4096,
    )
    model.initialize()

    prompts = build_query_prompts(tasks)

    for idx, prompt in prompts:
        try:
            messages = [{"role": "user", "content": prompt}]
            result = model.chat_completion(messages)
            new_query = result.get("content", "").strip()
            new_query = new_query.strip('"').strip("'").strip()

            if new_query and new_query.lower().startswith("help"):
                tasks[idx]["task_query_original"] = tasks[idx]["task_query"]
                tasks[idx]["task_query"] = new_query
                logger.info(f"  [{idx}] {tasks[idx]['task_query_original']}")
                logger.info(f"    -> {new_query}")
            else:
                logger.warning(f"  [{idx}] LLM returned invalid query: {new_query[:80]}")
        except Exception as e:
            logger.warning(f"  [{idx}] LLM query improvement failed: {e}")

    return tasks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def improve_tasks(input_path: str, memory_file: str, output_path: str,
                  use_llm: bool = False, model_name: str = "",
                  max_repeat_rate: float = 0.70,
                  gpu_memory: float = 0.85) -> List[dict]:
    """Run the full improvement pipeline."""

    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON task array")
    tasks = data

    logger.info(f"Loaded {len(tasks)} tasks from {input_path}")

    # Step 1-2: Filter and annotate
    kept, removed = filter_and_annotate(tasks, max_repeat_rate)
    logger.info(f"Filtered: {len(removed)} tasks removed (repeat_rate > {max_repeat_rate})")
    for r in removed:
        logger.info(f"  Removed: {r['task_id']} — {r.get('task_query','')} "
                     f"(repeat={r['repeat_rate']:.2f})")
    logger.info(f"Remaining: {len(kept)} tasks")

    # Step 3: participant memory grounding
    if memory_file:
        memory_objects = load_memory_objects(memory_file)
        logger.info(f"Loaded participant memory with {len(memory_objects)} videos")

        enriched_count = 0
        total_steps = 0
        for t in kept:
            vid = t.get("video_id", "")
            obj_reg = memory_objects.get(vid, {})
            if not obj_reg:
                logger.warning(f"  No participant memory objects for {vid}")
                continue
            for step in t.get("ground_truth_steps", []):
                total_steps += 1
                enrich_step(step, obj_reg)
                if "enriched_narration" in step:
                    enriched_count += 1
        logger.info(f"participant memory grounding: {enriched_count}/{total_steps} steps enriched "
                     f"({enriched_count/max(total_steps,1)*100:.1f}%)")

    # Step 4: LLM query improvement
    if use_llm and model_name:
        kept = improve_queries_with_llm(kept, model_name, gpu_memory)
    else:
        logger.info("Skipping LLM query improvement (--use-llm not set)")

    # Step 5: Output
    # Summary stats
    enriched_steps = sum(
        1 for t in kept for s in t.get("ground_truth_steps", [])
        if isinstance(s, dict) and "enriched_narration" in s
    )
    total_steps = sum(len(t.get("ground_truth_steps", [])) for t in kept)
    query_rewritten = sum(1 for t in kept if "task_query_original" in t)

    logger.info("\n=== Summary ===")
    logger.info(f"  Tasks: {len(kept)} kept, {len(removed)} filtered")
    logger.info(f"  GT enrichment: {enriched_steps}/{total_steps} steps")
    logger.info(f"  Query rewrite: {query_rewritten}/{len(kept)} tasks")

    released_tasks = []
    for task in kept:
        released = dict(task)
        released.pop("repeat_rate", None)
        released.pop("task_query_original", None)
        released_tasks.append(released)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(released_tasks, indent=2, ensure_ascii=False) + "\n"
    )
    logger.info(f"Written to {output_path}")
    return released_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Build replay planning tasks from observed routines",
    )
    parser.add_argument(
        "--input", required=True,
        help="Input planning_tasks.json",
    )
    parser.add_argument(
        "--memory-file",
        required=True,
        help="Path to one participant-memory JSON file",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output replay-task JSON",
    )
    parser.add_argument(
        "--max-repeat-rate", type=float, default=0.70,
        help="Filter tasks above this repeat rate (default: 0.70)",
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use an LLM to rewrite task queries",
    )
    parser.add_argument(
        "--model",
        help="Model name for optional LLM query rewriting (required with --use-llm)",
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=0.85,
        help="GPU memory utilization for vLLM",
    )
    args = parser.parse_args()

    if args.use_llm and not args.model:
        parser.error("--model is required with --use-llm")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    improve_tasks(
        input_path=args.input,
        memory_file=args.memory_file,
        output_path=args.output,
        use_llm=args.use_llm,
        model_name=args.model,
        max_repeat_rate=args.max_repeat_rate,
        gpu_memory=args.gpu_memory,
    )


if __name__ == "__main__":
    main()
