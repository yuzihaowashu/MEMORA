#!/usr/bin/env python3
"""Generate Transfer, Composition, and Fully Novel MEMORA-Planning tasks.

The goal is to test whether a memory-grounded agent can plan beyond exact
replay by recombining and adapting evidence from prior embodied experience.

Three task types:
  1. Transfer – Known action + known object, but in an unseen combination
  2. Composition – Combine 2-3 observed sub-procedures into one new multi-step task
  3. Fully Novel – Form a new task from participant-specific experience

Usage:
  python3 scripts/benchmark_construction/generate_generalize_candidates.py \
      --memory-file /path/to/participant_memory.json \
      --participant P01 \
      --output /path/to/benchmark_build/generalize_candidates_p01.json
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

random.seed(42)


def mine_participant_knowledge(memory: dict, participant: str) -> dict:
    """Extract a knowledge profile from all videos of a participant."""
    mbv = memory.get("memories_by_video", memory)

    vids = sorted(v for v in mbv if v.startswith(participant))
    if not vids:
        raise ValueError(f"No videos found for participant {participant}")

    action_obj_pairs: Counter = Counter()
    action_counter: Counter = Counter()
    object_counter: Counter = Counter()
    obj_locations: Dict[str, Set[str]] = defaultdict(set)
    obj_attributes: Dict[str, Dict] = {}
    activity_summaries: List[Tuple[str, str, dict]] = []
    high_level_goals: Set[str] = set()

    for vid in vids:
        data = mbv[vid]
        for act in data.get("activity_log", []):
            summary = act.get("summary", "")
            hlg = act.get("high_level_goal", "")
            if hlg:
                high_level_goals.add(hlg)
            activity_summaries.append((vid, summary, act))

            for step in act.get("action_breakdown", []):
                if not isinstance(step, dict):
                    continue
                action = step.get("action") or ""
                action = action if isinstance(action, str) else str(action)
                action = action.strip().lower()
                obj = step.get("object") or ""
                obj = obj if isinstance(obj, str) else " ".join(obj) if isinstance(obj, list) else str(obj)
                obj = obj.strip().lower()
                direction = step.get("direction") or ""
                direction = direction if isinstance(direction, str) else str(direction)
                direction = direction.strip().lower()
                if not action or action == "none":
                    continue
                action_counter[action] += 1
                if obj and obj != "none":
                    action_obj_pairs[(action, obj)] += 1
                    object_counter[obj] += 1
                    if "to" in direction:
                        loc_part = direction.split("to")[-1].strip()
                        if loc_part and loc_part != "none":
                            obj_locations[obj].add(loc_part)

        for oid, odata in data.get("object_registry", {}).items():
            if not isinstance(odata, dict):
                continue
            name = (odata.get("name") or oid).lower()
            vp = odata.get("visual_properties") or {}
            si = odata.get("spatial_info") or {}
            obj_attributes[name] = {
                "name": odata.get("name", oid),
                "color": (vp.get("color") or ""),
                "material": (vp.get("material") or ""),
                "location": (si.get("location") or ""),
            }
            loc = obj_attributes[name]["location"]
            if loc:
                obj_locations[name].add(loc)

    return {
        "participant": participant,
        "videos": vids,
        "action_obj_pairs": action_obj_pairs,
        "action_counter": action_counter,
        "object_counter": object_counter,
        "obj_locations": obj_locations,
        "obj_attributes": obj_attributes,
        "activity_summaries": activity_summaries,
        "high_level_goals": high_level_goals,
    }


def _describe_obj(name: str, attrs: dict) -> str:
    """Format an object with its participant memory attributes."""
    color = attrs.get("color", "")
    material = attrs.get("material", "")
    loc = attrs.get("location", "")
    desc = f"{color} {material} {name}".strip()
    if loc:
        desc += f" (at {loc})"
    return desc


PLAUSIBLE_ACTION_OBJECTS = {
    "washes":     {"plate", "bowl", "pot", "pan", "frying pan", "cup", "mug", "sponge",
                   "knife", "spatula", "cutting board", "vegetables", "bok choy", "onion",
                   "garlic", "cucumber", "potato"},
    "scrubbing":  {"plate", "bowl", "pot", "pan", "frying pan", "cup", "mug", "sink",
                   "cutting board", "stove", "stovetop"},
    "scrubs":     {"plate", "bowl", "pot", "pan", "frying pan", "cup", "mug", "sink",
                   "cutting board"},
    "rinsing":    {"plate", "bowl", "pot", "pan", "frying pan", "cup", "mug", "sponge",
                   "knife", "spatula", "cloth", "vegetables", "bok choy", "onion",
                   "garlic", "cucumber", "potato"},
    "wiping":     {"stove", "stovetop", "counter", "cutting board", "plate", "bowl",
                   "pot", "pan", "frying pan", "table"},
    "peeling":    {"potato", "onion", "garlic", "cucumber", "carrot"},
    "stirring":   {"potatoes", "vegetables", "onions", "bok choy", "garlic", "sauce",
                   "soup", "pasta"},
    "cutting":    {"potato", "onion", "garlic", "cucumber", "bok choy", "vegetables",
                   "bread", "carrot"},
}


def generate_transfer_tasks(knowledge: dict) -> List[dict]:
    """Transfer: known action applied to a different known object (plausible combos only)."""
    aop = knowledge["action_obj_pairs"]
    obj_count = knowledge["object_counter"]
    act_count = knowledge["action_counter"]
    pid = knowledge["participant"]

    available_actions = {a for a in act_count
                        if a in PLAUSIBLE_ACTION_OBJECTS and act_count[a] >= 5}
    frequent_objects = {o for o, c in obj_count.most_common(40)}

    known_pairs = set(aop.keys())

    tasks = []
    for action in sorted(available_actions):
        known_objects = {o for (a, o) in known_pairs if a == action}
        plausible_targets = PLAUSIBLE_ACTION_OBJECTS.get(action, set())
        novel_objects = (frequent_objects & plausible_targets) - known_objects
        if not novel_objects or not known_objects:
            continue
        for novel_obj in sorted(novel_objects)[:2]:
            template_obj = sorted(known_objects, key=lambda o: aop[(action, o)], reverse=True)[0]

            gt_steps = _build_transfer_gt(action, novel_obj, template_obj, knowledge)
            if not gt_steps:
                continue

            task_query = _goal_oriented_query(pid, action, novel_obj)
            task_id = f"generalize_transfer_{pid}_{action.replace(' ', '_')}_{novel_obj.replace(' ', '_')}"

            tasks.append({
                "task_id": task_id,
                "task_type": "transfer",
                "task_query": task_query,
                "video_id": knowledge["videos"][0],
                "participant_id": pid,
                "source_action": action,
                "source_object": template_obj,
                "target_object": novel_obj,
                "ground_truth_steps": gt_steps,
                "rationale": (
                    f"Person knows how to '{action}' the '{template_obj}' "
                    f"({aop[(action, template_obj)]}x observed). "
                    f"Transfer to '{novel_obj}' (never observed with '{action}')."
                ),
            })

    return tasks[:6]


FOOD_OBJECTS = {"cucumber", "onion", "carrot", "bok choy", "lettuce",
                "potato", "garlic", "vegetables"}
FIXED_SURFACES = {"sink", "counter", "countertop", "stove", "stovetop", "table"}


def _action_to_verb(action: str) -> str:
    mapping = {
        "picks up": "pick up",
        "places": "place",
        "washes": "wash",
        "scrubbing": "scrub",
        "rinsing": "rinse",
        "wiping": "wipe",
        "peeling": "peel",
        "stirring": "stir",
        "cutting": "cut",
        "scrubs": "scrub",
    }
    return mapping.get(action, action)


def _goal_oriented_query(pid: str, action: str, obj: str) -> str:
    """Generate goal-oriented task query for rinse/scrub actions.

    For food items:  "Help Pxx wash the X before preparing it."
    For dishware:    "Help Pxx clean the X."
    Other actions fall back to the standard verb-based phrasing.
    """
    a = action.lower()
    if a in ("rinsing", "scrubbing", "scrubs"):
        if obj.lower() in FOOD_OBJECTS:
            return f"Help {pid} wash the {obj} before preparing it."
        return f"Help {pid} clean the {obj}."
    return f"Help {pid} {_action_to_verb(action)} the {obj}."


def _goal_oriented_gt_verb(action: str, obj: str) -> str:
    """Return the GT verb aligned with the goal-oriented query."""
    a = action.lower()
    if a in ("rinsing", "scrubbing", "scrubs"):
        if obj.lower() in FOOD_OBJECTS:
            return "Wash"
        return "Clean"
    return _action_to_verb(action).capitalize()


def _build_transfer_gt(action: str, target_obj: str, template_obj: str,
                       knowledge: dict) -> List[str]:
    """Build ground truth steps by adapting a known procedure to a new object."""
    attrs = knowledge["obj_attributes"]
    target_desc = _describe_obj(target_obj, attrs.get(target_obj, {}))
    locs = list(knowledge["obj_locations"].get(target_obj, set()))
    target_loc = locs[0] if locs else "the counter"

    verb = _action_to_verb(action)
    gt_verb = _goal_oriented_gt_verb(action, target_obj)

    if verb in ("wash", "scrub", "rinse"):
        if target_obj.lower() in FIXED_SURFACES:
            return [
                "Pick up the sponge",
                "Apply dish soap to the sponge",
                f"{gt_verb} the {target_obj} in place",
                "Rinse the sponge",
                f"Wipe residual water from the {target_obj}",
                "Return the sponge to its place",
            ]
        return [
            f"Locate the {target_desc}",
            f"Pick up the {target_obj} from {target_loc}",
            f"Carry the {target_obj} to the sink",
            "Turn on the tap",
            f"{gt_verb} the {target_obj} under running water",
            "Turn off the tap",
            f"Place the {target_obj} on the drying rack",
        ]
    elif verb == "wipe":
        cloth_desc = _describe_obj("cloth", attrs.get("cloth", {}))
        return [
            f"Pick up the {cloth_desc}",
            "Wet the cloth under the tap",
            f"Wipe the {target_obj} surface thoroughly",
            "Rinse the cloth",
            f"Wipe the {target_obj} again to remove residue",
            "Return the cloth to its place",
        ]
    elif verb == "peel":
        peeler_desc = _describe_obj("peeler", attrs.get("peeler", {}))
        return [
            f"Pick up the {target_obj} from {target_loc}",
            f"Pick up the {peeler_desc}",
            f"Hold the {target_obj} steady with one hand",
            f"Peel the {target_obj} skin with the peeler",
            "Rotate and continue peeling until fully peeled",
            f"Place peeled {target_obj} on the cutting board",
            "Discard peelings in the bin",
        ]
    elif verb == "stir":
        spatula_desc = _describe_obj("spatula", attrs.get("spatula", {}))
        return [
            f"Pick up the {spatula_desc}",
            f"Position the spatula over the pan with {target_obj}",
            f"Stir the {target_obj} gently in the pan",
            "Continue stirring to ensure even cooking",
            f"Check the {target_obj} for doneness",
            "Rest the spatula on the pan edge",
        ]
    elif verb == "cut":
        knife_desc = _describe_obj("knife", attrs.get("knife", {}))
        return [
            f"Place the {target_obj} on the cutting board",
            f"Pick up the {knife_desc}",
            f"Hold the {target_obj} steady with one hand",
            f"Cut the {target_obj} into pieces",
            "Gather the cut pieces",
            f"Transfer cut {target_obj} to a bowl",
        ]
    return []


def generate_composition_tasks(knowledge: dict) -> List[dict]:
    """Composition: combine 2-3 observed sub-procedures into a new task."""
    pid = knowledge["participant"]
    aop = knowledge["action_obj_pairs"]
    attrs = knowledge["obj_attributes"]

    compositions = []

    # 1) Peel + Cut + Stir (potato workflow variant)
    if (aop.get(("peeling", "potato"), 0) > 0 and
        aop.get(("stirring", "vegetables"), 0) > 0):
        compositions.append({
            "name": "prepare_and_cook_onion",
            "query": f"Help {pid} peel, chop, and stir-fry the onion.",
            "components": ["peeling onion", "cutting onion", "stirring onion in pan"],
            "gt_steps": [
                f"Pick up the onion from {_get_loc('onion', knowledge)}",
                f"Pick up the {_describe_obj('peeler', attrs.get('peeler', {}))}",
                "Peel the onion skin with the peeler",
                "Place peeled onion on the cutting board",
                f"Pick up the {_describe_obj('knife', attrs.get('knife', {}))}",
                "Cut the onion into small pieces on the cutting board",
                f"Pick up the {_describe_obj('spatula', attrs.get('spatula', {}))}",
                "Transfer cut onion pieces into the heated pan",
                "Stir-fry the onion pieces in the pan with the spatula",
                "Continue stirring until onion is translucent",
            ],
            "rationale": (
                f"Person knows: peeling potato ({aop.get(('peeling potato', 'potato'), aop.get(('peeling', 'potato'), 0))}x), "
                f"stirring vegetables ({aop.get(('stirring', 'vegetables'), 0)}x). "
                f"Composition: apply peel+cut+stir to onion."
            ),
        })

    # 2) Wash + Dry + Store (extend dish-washing to a full cycle)
    if (aop.get(("washes", "plate"), 0) > 0 or aop.get(("scrubbing", "pot"), 0) > 0):
        compositions.append({
            "name": "full_dishwashing_cycle",
            "query": f"Help {pid} wash all the dirty dishes, dry them, and put them away.",
            "components": ["scrubbing dishes", "rinsing dishes", "drying", "storing"],
            "gt_steps": [
                "Collect dirty dishes from the counter to the sink",
                f"Pick up the {_describe_obj('sponge', attrs.get('sponge', {}))}",
                "Apply soap to the sponge",
                "Scrub each dish thoroughly with the sponge",
                "Rinse each dish under running water",
                "Turn off the tap",
                f"Pick up the {_describe_obj('cloth', attrs.get('cloth', {}))}",
                "Dry each dish with the cloth",
                "Stack dried plates in the plate rack",
                "Place dried pots on the shelf",
            ],
            "rationale": (
                f"Person knows: scrubbing pot ({aop.get(('scrubbing', 'pot'), 0)}x), "
                f"rinsing pot ({aop.get(('rinsing', 'pot'), 0)}x). "
                "Composition: extend to full wash-dry-store cycle for all dishes."
            ),
        })

    # 3) Cut + Season + Cook (vegetable prep)
    if aop.get(("stirring", "vegetables"), 0) > 0:
        compositions.append({
            "name": "prep_cook_bok_choy",
            "query": f"Help {pid} wash, cut, and stir-fry the bok choy.",
            "components": ["washing bok choy", "cutting bok choy", "stir-frying"],
            "gt_steps": [
                f"Pick up the bok choy from {_get_loc('bok choy', knowledge)}",
                "Bring the bok choy to the sink",
                "Turn on the tap and rinse the bok choy under running water",
                "Turn off the tap",
                "Place the bok choy on the cutting board",
                f"Pick up the {_describe_obj('knife', attrs.get('knife', {}))}",
                "Cut the bok choy into bite-sized pieces",
                f"Pick up the {_describe_obj('spatula', attrs.get('spatula', {}))}",
                "Add the cut bok choy to the heated pan",
                "Stir-fry the bok choy with the spatula until wilted",
            ],
            "rationale": (
                f"Person knows: stirring vegetables ({aop.get(('stirring', 'vegetables'), 0)}x), "
                f"rinsing various items, cutting on board. "
                "Composition: wash+cut+cook applied to bok choy specifically."
            ),
        })

    tasks = []
    for i, comp in enumerate(compositions):
        tasks.append({
            "task_id": f"generalize_composition_{pid}_{comp['name']}",
            "task_type": "composition",
            "task_query": comp["query"],
            "video_id": knowledge["videos"][0],
            "participant_id": pid,
            "components": comp["components"],
            "ground_truth_steps": comp["gt_steps"],
            "rationale": comp["rationale"],
        })
    return tasks


def generate_fully_novel_tasks(knowledge: dict) -> List[dict]:
    """Generate fully novel tasks grounded in participant experience."""
    pid = knowledge["participant"]
    attrs = knowledge["obj_attributes"]
    tasks_def = [
        {
            "name": "set_table_for_meal",
            "query": f"Help {pid} set the table for a meal with plate, bowl, and cutlery.",
            "gt_steps": [
                f"Pick up a clean {_describe_obj('plate', attrs.get('plate', {}))}",
                "Place the plate on the table",
                f"Pick up a clean {_describe_obj('bowl', attrs.get('bowl', {}))}",
                "Place the bowl above the plate on the table",
                f"Pick up the {_describe_obj('knife', attrs.get('knife', {}))}",
                "Place the knife to the right of the plate",
                "Pick up a fork",
                "Place the fork to the left of the plate",
                "Pick up a glass from the shelf",
                "Place the glass above the knife on the table",
            ],
            "rationale": (
                "Person never observed setting a table, but knows locations of "
                "plates, bowls, knives from extensive kitchen activity."
            ),
        },
        {
            "name": "clean_kitchen_countertop",
            "query": f"Help {pid} clear and clean the entire kitchen countertop.",
            "gt_steps": [
                "Survey the countertop for items to clear",
                f"Pick up the {_describe_obj('cutting board', attrs.get('cutting board', {}))} and store it",
                "Collect any food scraps and dispose in the bin",
                f"Pick up the {_describe_obj('cloth', attrs.get('cloth', {}))}",
                "Wet the cloth under the tap",
                "Wipe down the entire countertop surface from left to right",
                "Rinse the cloth under the tap",
                "Wipe the countertop again to remove soap residue",
                "Dry the countertop with a dry cloth",
                "Return cleaning supplies to their storage locations",
            ],
            "rationale": (
                "Person frequently wipes surfaces (stove, hob) but never observed "
                "doing a full countertop clean. Requires generalizing wiping patterns."
            ),
        },
        {
            "name": "prepare_simple_salad",
            "query": f"Help {pid} prepare a simple salad using available vegetables.",
            "gt_steps": [
                "Gather available vegetables from the counter/fridge",
                "Bring vegetables to the sink",
                "Rinse all vegetables under running water",
                "Place vegetables on the cutting board",
                f"Pick up the {_describe_obj('knife', attrs.get('knife', {}))}",
                "Chop the vegetables into bite-sized pieces",
                f"Pick up a clean {_describe_obj('bowl', attrs.get('bowl', {}))}",
                "Transfer chopped vegetables into the bowl",
                "Toss the vegetables in the bowl",
                "Place the salad bowl on the table",
            ],
            "rationale": (
                "Person has extensive cutting and vegetable handling experience, "
                "but salad preparation (raw assembly) is a fully novel task category."
            ),
        },
    ]

    tasks = []
    for td in tasks_def:
        tasks.append({
            # Preserve the released task-ID namespace for result joins.
            "task_id": f"generalize_novel_{pid}_{td['name']}",
            "task_type": "fully_novel",
            "task_query": td["query"],
            "video_id": knowledge["videos"][0],
            "participant_id": pid,
            "ground_truth_steps": td["gt_steps"],
            "rationale": td["rationale"],
        })
    return tasks


def _get_loc(obj: str, knowledge: dict) -> str:
    locs = list(knowledge["obj_locations"].get(obj, set()))
    return locs[0] if locs else "the counter"


def main():
    parser = argparse.ArgumentParser(description="Generate generalize planning tasks")
    parser.add_argument(
        "--memory-file",
        required=True,
        help="Path to participant memory JSON",
    )
    parser.add_argument("--participant", required=True, help="Participant ID")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    memory = json.loads(Path(args.memory_file).read_text(encoding="utf-8"))
    knowledge = mine_participant_knowledge(memory, args.participant)

    print(f"Participant evidence summary for {args.participant}:")
    print(f"  Videos: {len(knowledge['videos'])}")
    print(f"  Activities: {len(knowledge['activity_summaries'])}")
    print(f"  Unique action-object pairs: {len(knowledge['action_obj_pairs'])}")
    print(f"  Unique objects: {len(knowledge['object_counter'])}")
    print(f"  High-level goals: {len(knowledge['high_level_goals'])}")
    print()

    transfer = generate_transfer_tasks(knowledge)
    composition = generate_composition_tasks(knowledge)
    generalize = generate_fully_novel_tasks(knowledge)

    all_tasks = transfer + composition + generalize

    print("Generated tasks:")
    print(f"  Transfer:    {len(transfer)}")
    print(f"  Composition: {len(composition)}")
    print(f"  Fully Novel: {len(generalize)}")
    print(f"  Total:       {len(all_tasks)}")
    print()

    for t in all_tasks:
        print(f"  [{t['task_type']:11s}] {t['task_id']}")
        print(f"    Query: {t['task_query']}")
        print(f"    GT steps: {len(t['ground_truth_steps'])}")
        print(f"    Rationale: {t['rationale'][:100]}...")
        print()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(all_tasks, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
