#!/usr/bin/env python3
"""Build participant-grounded reference plans for Generalize tasks.

Principles:
  1. Every object mention includes color + material + location from participant memory
  2. Physically necessary state changes are included (turn on stove, turn on tap)
  3. Granularity = goal-level (no excessive carry steps), but each step is atomic
  4. GT is participant-specific (different kitchens have different objects)

Usage:
  python3 scripts/benchmark_construction/build_generalize_suite.py \
      --memory-file /path/to/participant_memory.json \
      --participant P01 \
      --input  /path/to/benchmark_build/generalize_candidates_p01.json \
      --output planning/suites/generalize/p01.json
"""

import argparse
import json
import copy
from pathlib import Path
from typing import Dict, List


INVALID_LOCATION_VALUES = {"in sink", "on counter", "in hand", "on stove", "in pan",
                  "on countertop", "on table", "in drawer", "in dishwasher",
                  "in cabinet", "on shelf", "on rack", "in colander"}
STATE_WORDS = {"dirty", "clean", "used", "empty", "full", "old", "new"}


def _is_valid_attr(val: str) -> bool:
    """Check if a participant-memory attribute is meaningful, not a location or state label."""
    if not val or not val.strip():
        return False
    v = val.strip().lower()
    if v in INVALID_LOCATION_VALUES or v in STATE_WORDS or v in INVALID_LOCATIONS:
        return False
    if v.startswith("in ") or v.startswith("on ") or v.startswith("at "):
        return False
    return True


def load_memory_objects(memory_file: str, participant: str) -> Dict[str, dict]:
    """Load and merge all object registries for a participant, preferring high-quality data."""
    with open(memory_file) as f:
        memory = json.load(f)
    mbv = memory.get("memories_by_video", memory)
    vids = sorted(v for v in mbv if v.startswith(participant))

    objects = {}
    for vid in vids:
        for oid, odata in mbv[vid].get("object_registry", {}).items():
            if not isinstance(odata, dict):
                continue
            if oid not in objects:
                objects[oid] = copy.deepcopy(odata)
            else:
                cur = objects[oid]
                cur_vp = cur.get("visual_properties", {})
                new_vp = odata.get("visual_properties", {})
                for attr in ("color", "material"):
                    cur_val = cur_vp.get(attr, "")
                    new_val = new_vp.get(attr, "")
                    if not _is_valid_attr(cur_val) and _is_valid_attr(new_val):
                        if "visual_properties" not in cur:
                            cur["visual_properties"] = {}
                        cur["visual_properties"][attr] = new_val

                cur_name = cur.get("name", "")
                new_name = odata.get("name", "")
                if new_name and (not cur_name or any(w in cur_name.lower() for w in STATE_WORDS)):
                    if not any(w in new_name.lower() for w in STATE_WORDS):
                        cur["name"] = new_name

                cur_si = cur.get("spatial_info", {})
                new_si = odata.get("spatial_info", {})
                new_loc = new_si.get("location", "")
                cur_loc = cur_si.get("location", "")
                if isinstance(new_loc, str) and new_loc.strip():
                    new_loc_clean = new_loc.strip().lower()
                    cur_loc_clean = (cur_loc or "").strip().lower()
                    BAD_LOCS = {"in hand", "in hands", "", "in washing machine",
                                "not specified", "unknown"}
                    GOOD_LOCS = {"counter", "countertop", "on counter", "on countertop",
                                 "on table", "on stove", "in sink", "on shelf",
                                 "on drying rack", "on drying_rack", "in drawer"}
                    if new_loc_clean not in BAD_LOCS:
                        should_update = (
                            not cur_loc or cur_loc_clean in BAD_LOCS
                            or (cur_loc_clean not in GOOD_LOCS
                                and any(g in new_loc_clean for g in GOOD_LOCS))
                        )
                        if should_update:
                            if "spatial_info" not in cur:
                                cur["spatial_info"] = {}
                            cur["spatial_info"]["location"] = new_loc

    return objects


def _entry_quality(entry: dict) -> int:
    """Score how good an entry's attributes are (higher = better)."""
    score = 0
    if _is_valid_attr(entry.get("color", "")):
        score += 2
    if _is_valid_attr(entry.get("material", "")):
        score += 2
    if entry.get("location", ""):
        score += 1
    name = entry.get("name", "")
    if name and not any(w in name.lower() for w in STATE_WORDS):
        score += 1
    return score


def build_object_lookup(objects: Dict[str, dict]) -> Dict[str, dict]:
    """Build a lookup from common names to participant memory attributes.

    Priority: exact oid match > exact name match > partial token match.
    Within same priority, higher quality wins.
    """
    all_oids = {oid.lower() for oid in objects}
    lookup = {}
    priority = {}

    for oid, odata in objects.items():
        name = (odata.get("name") or oid).lower().strip()
        name = name.replace("_", " ")
        vp = odata.get("visual_properties", {})
        si = odata.get("spatial_info", {})
        entry = {
            "id": oid,
            "name": name,
            "color": (vp.get("color") or "").strip(),
            "material": (vp.get("material") or "").strip(),
            "location": "",
        }
        loc = si.get("location") or odata.get("location", "")
        if isinstance(loc, dict):
            loc = loc.get("location", "")
        entry["location"] = str(loc).strip() if loc else ""

        quality = _entry_quality(entry)
        oid_clean = oid.lower().replace("_", " ")

        def try_register(token: str, prio: int):
            token = token.strip()
            if len(token) <= 2:
                return
            cur_prio = priority.get(token, -1)
            if prio > cur_prio:
                lookup[token] = entry
                priority[token] = prio
            elif prio == cur_prio and quality > _entry_quality(lookup.get(token, {})):
                lookup[token] = entry

        try_register(oid_clean, 30)
        try_register(name, 20)

        for token in oid_clean.split() + name.split():
            if token in all_oids:
                continue
            try_register(token, 10)

    return lookup


INVALID_LOCATIONS = {"hand", "hands", "air", "midair", "unknown", ""}


def _clean_loc(loc: str) -> str:
    """Normalize participant memory location: 'on cutting board' → 'the cutting board'."""
    import re
    loc = loc.strip()
    loc = re.sub(r"^(on|in|at|near|beside|next to)\s+", "", loc, flags=re.IGNORECASE)
    loc = re.sub(r"^(the|a|an)\s+", "", loc, flags=re.IGNORECASE)
    loc = loc.replace("_", " ")
    if loc.lower() in INVALID_LOCATIONS:
        return "the counter"
    return f"the {loc}"


def _clean_desc(raw: str) -> str:
    """Remove duplicate/adjacent words and fix preposition issues."""
    import re
    words = raw.split()
    cleaned = []
    for i, w in enumerate(words):
        if i > 0 and w.lower() == words[i - 1].lower():
            continue
        cleaned.append(w)
    result = " ".join(cleaned)
    result = re.sub(r"\bthe the\b", "the", result)
    return result


MATERIAL_ADJECTIVE_MAP = {"wood": "wooden", "metal": "metal", "plastic": "plastic",
                           "ceramic": "ceramic", "foam": "foam", "cotton": "cotton",
                           "glass": "glass", "rubber": "rubber", "silicone": "silicone"}


def _simplify_name(name: str) -> str:
    """Extract a clean short name from participant memory name, removing noise."""
    name = name.split("(")[0].strip()
    name = name.replace("_", " ")
    noise = {"stainless", "steel", "gas", "kitchen",
             "clean", "dirty", "used", "empty", "full",
             "in", "on", "at", "near", "the", "a",
             "batter", "slices", "pieces", "chopped"}
    loc_words = {"sink", "counter", "stove", "table", "shelf", "rack", "floor"}
    tokens = name.split()
    result = []
    for i, t in enumerate(tokens):
        tl = t.lower()
        if tl in noise:
            continue
        if tl in loc_words and i > 0 and tokens[i-1].lower() in ("in", "on", "at", "near", "from"):
            continue
        adj = MATERIAL_ADJECTIVE_MAP.get(tl)
        if adj and adj != tl and adj in [x.lower() for x in tokens]:
            continue
        result.append(t)
    return " ".join(result)


def describe(obj_name: str, lookup: Dict[str, dict], with_location: bool = True) -> str:
    """Describe an object using participant memory attributes: 'color material name at location'."""
    obj_name_lower = obj_name.lower().strip()
    entry = lookup.get(obj_name_lower)

    if not entry:
        for key, val in lookup.items():
            if obj_name_lower in key or key in obj_name_lower:
                entry = val
                break

    if not entry:
        return obj_name

    color = entry["color"].lower().strip()
    material = entry["material"].lower().strip()
    base = _simplify_name(entry["name"]) if entry["name"] else obj_name

    skip_material = material in ("vegetable", "food", "produce", "organic",
                                  "skin", "flesh", "liquid", "powder")

    COLORS = {"red","orange","yellow","green","blue","purple","pink","brown",
              "black","white","silver","grey","gray","gold","beige","cream"}

    base_lower = base.lower()
    base_tokens = base_lower.split()
    base_has_color_word = any(t in COLORS for t in base_tokens)
    if base_has_color_word:
        base_tokens = [t for t in base_tokens if t not in COLORS]

    mat_adj = MATERIAL_ADJECTIVE_MAP.get(material, material)
    mat_forms = {material, mat_adj}

    parts = []
    if color:
        parts.append(color)
    has_material_in_base = bool(mat_forms & set(base_tokens))
    if material and not skip_material and material not in parts and not has_material_in_base:
        parts.append(material)

    exclude = set(parts) | mat_forms
    clean_base_tokens = [t for t in base_tokens if t not in exclude]
    if has_material_in_base:
        clean_base_tokens = [mat_adj if t in mat_forms else t for t in base_tokens if t not in set(parts)]
    clean_base = " ".join(clean_base_tokens) if clean_base_tokens else obj_name
    parts.append(clean_base)

    desc = " ".join(parts)

    if with_location and entry["location"]:
        desc += f" from {_clean_loc(entry['location'])}"

    return _clean_desc(desc)


def get_location(obj_name: str, lookup: Dict[str, dict]) -> str:
    """Get location of an object from participant memory."""
    obj_name_lower = obj_name.lower().strip()
    entry = lookup.get(obj_name_lower)
    if not entry:
        for key, val in lookup.items():
            if obj_name_lower in key or key in obj_name_lower:
                entry = val
                break
    if entry and entry["location"]:
        return _clean_loc(entry["location"])
    return "the counter"


def build_grounded_gt(task: dict, lookup: Dict[str, dict], pid: str) -> List[str]:
    """Build memory-grounded GT steps for a Generalize task."""
    task_type = task["task_type"]
    if task_type == "transfer":
        return build_transfer_gt(task, lookup, pid)
    elif task_type == "composition":
        return build_composition_gt(task, lookup, pid)
    elif task_type == "fully_novel":
        return build_fully_novel_gt(task, lookup, pid)
    return task["ground_truth_steps"]


def build_transfer_gt(task: dict, lookup: Dict[str, dict], pid: str) -> List[str]:
    """Build grounded GT for transfer tasks (known action → new object)."""
    query = task.get("task_query", "").lower()
    target = task.get("target_object", "")

    target_desc = describe(target, lookup, with_location=False)
    target_loc = get_location(target, lookup)

    import re
    is_cut = bool(re.search(r"\bcut\b", query)) and "cutting board" not in query
    if is_cut:
        knife_desc = describe("knife", lookup, with_location=False)
        knife_loc = get_location("knife", lookup)
        board_desc = describe("cutting board", lookup, with_location=False)
        board_loc = get_location("cutting board", lookup)
        bowl_desc = describe("bowl", lookup, with_location=False)
        return [
            f"Pick up the {target_desc} from {target_loc}.",
            f"Place the {target_desc} on the {board_desc} on {board_loc}.",
            f"Pick up the {knife_desc} from {knife_loc}.",
            f"Hold the {target_desc} steady with one hand on the {board_desc}.",
            f"Cut the {target_desc} into pieces with the {knife_desc}.",
            f"Place the {knife_desc} back on {knife_loc}.",
            f"Transfer the cut {target} pieces to the {bowl_desc}.",
        ]

    elif "peel" in query:
        peeler_desc = describe("peeler", lookup, with_location=False)
        peeler_loc = get_location("peeler", lookup)
        board_desc = describe("cutting board", lookup, with_location=False)
        return [
            f"Pick up the {target_desc} from {target_loc}.",
            f"Pick up the {peeler_desc} from {peeler_loc}.",
            f"Hold the {target_desc} steady with one hand.",
            f"Peel the {target_desc} skin with the {peeler_desc}.",
            f"Rotate the {target} and continue peeling until fully peeled.",
            f"Place the peeled {target} on the {board_desc}.",
            "Discard the peelings in the bin.",
        ]

    elif "wash" in query or "clean" in query:
        sponge_desc = describe("sponge", lookup, with_location=False)
        sink_desc = describe("sink", lookup, with_location=False)
        faucet_desc = describe("faucet", lookup, with_location=False)
        is_food = target.lower() in {
            "cucumber", "carrot", "onion", "potato", "garlic",
            "bok choy", "lettuce", "vegetables",
        }
        steps = [
            f"Pick up the {target_desc} from {target_loc}.",
            f"Carry the {target} to the {sink_desc}.",
            f"Turn on the {faucet_desc}.",
        ]
        if not is_food:
            steps.append(f"Apply dish soap to the {sponge_desc}.")
            steps.append(f"Scrub the {target} with the {sponge_desc}.")
        steps.append(f"Rinse the {target} under running water.")
        steps.append(f"Turn off the {faucet_desc}.")
        steps.append(f"Place the clean {target} on the drying rack.")
        return steps

    elif "stir" in query:
        spatula_desc = describe("spatula", lookup, with_location=False)
        spatula_loc = get_location("spatula", lookup)
        pan_desc = describe("frying pan", lookup, with_location=False)
        return [
            f"Pick up the {spatula_desc} from {spatula_loc}.",
            f"Position the {spatula_desc} in the {pan_desc} with the {target}.",
            f"Stir the {target} gently in the {pan_desc}.",
            "Continue stirring until evenly cooked.",
            f"Place the {spatula_desc} on the counter.",
        ]

    return task["ground_truth_steps"]


def build_composition_gt(task: dict, lookup: Dict[str, dict], pid: str) -> List[str]:
    """Build grounded GT for composition tasks (combine sub-procedures)."""
    query = task.get("task_query", "").lower()

    if "peel" in query and "chop" in query and "stir-fry" in query:
        obj = "onion"
        for word in ["onion", "potato", "carrot", "garlic"]:
            if word in query:
                obj = word
                break
        obj_desc = describe(obj, lookup, with_location=False)
        obj_loc = get_location(obj, lookup)
        knife_desc = describe("knife", lookup, with_location=False)
        knife_loc = get_location("knife", lookup)
        board_desc = describe("cutting board", lookup, with_location=False)
        pan_desc = describe("frying pan", lookup, with_location=False)
        spatula_desc = describe("spatula", lookup, with_location=False)
        spatula_loc = get_location("spatula", lookup)
        stove_desc = describe("stove", lookup, with_location=False)
        return [
            f"Pick up the {obj_desc} from {obj_loc}.",
            f"Place the {obj_desc} on the {board_desc}.",
            f"Pick up the {knife_desc} from {knife_loc}.",
            f"Peel the {obj} skin with the {knife_desc}.",
            f"Cut the peeled {obj} into small pieces on the {board_desc}.",
            f"Place the {knife_desc} back on {knife_loc}.",
            f"Turn on the {stove_desc} to medium heat.",
            f"Place the {pan_desc} on the {stove_desc}.",
            f"Transfer the cut {obj} pieces into the {pan_desc}.",
            f"Pick up the {spatula_desc} from {spatula_loc}.",
            f"Stir-fry the {obj} pieces with the {spatula_desc} until translucent.",
            f"Turn off the {stove_desc}.",
        ]

    elif "wash" in query and "cut" in query and "stir-fry" in query:
        obj = "bok choy"
        for word in ["bok choy", "spinach", "vegetables"]:
            if word in query:
                obj = word
                break
        obj_desc = describe(obj, lookup, with_location=False)
        obj_loc = get_location(obj, lookup)
        knife_desc = describe("knife", lookup, with_location=False)
        knife_loc = get_location("knife", lookup)
        board_desc = describe("cutting board", lookup, with_location=False)
        pan_desc = describe("frying pan", lookup, with_location=False)
        spatula_desc = describe("spatula", lookup, with_location=False)
        spatula_loc = get_location("spatula", lookup)
        stove_desc = describe("stove", lookup, with_location=False)
        sink_desc = describe("sink", lookup, with_location=False)
        faucet_desc = describe("faucet", lookup, with_location=False)
        return [
            f"Pick up the {obj_desc} from {obj_loc}.",
            f"Carry the {obj} to the {sink_desc}.",
            f"Turn on the {faucet_desc}.",
            f"Rinse the {obj} under running water.",
            f"Turn off the {faucet_desc}.",
            f"Place the rinsed {obj} on the {board_desc}.",
            f"Pick up the {knife_desc} from {knife_loc}.",
            f"Cut the {obj} into bite-sized pieces on the {board_desc}.",
            f"Place the {knife_desc} back on {knife_loc}.",
            f"Turn on the {stove_desc} to medium heat.",
            f"Place the {pan_desc} on the {stove_desc}.",
            f"Transfer the cut {obj} into the {pan_desc}.",
            f"Pick up the {spatula_desc} from {spatula_loc}.",
            f"Stir-fry the {obj} with the {spatula_desc} until wilted.",
            f"Turn off the {stove_desc}.",
        ]

    elif "wash" in query and "dishes" in query:
        sponge_desc = describe("sponge", lookup, with_location=False)
        sink_desc = describe("sink", lookup, with_location=False)
        faucet_desc = describe("faucet", lookup, with_location=False)
        plate_desc = describe("plate", lookup, with_location=False)
        bowl_desc = describe("bowl", lookup, with_location=False)
        return [
            f"Collect dirty dishes from the counter to the {sink_desc}.",
            f"Turn on the {faucet_desc}.",
            f"Apply dish soap to the {sponge_desc}.",
            f"Scrub the {plate_desc} with the {sponge_desc}.",
            f"Rinse the {plate_desc} under running water.",
            f"Place the clean {plate_desc.split()[0]} plate on the drying rack.",
            f"Scrub the {bowl_desc} with the {sponge_desc}.",
            f"Rinse the {bowl_desc} under running water.",
            "Place the clean bowl on the drying rack.",
            f"Turn off the {faucet_desc}.",
        ]

    return task["ground_truth_steps"]


def build_fully_novel_gt(task: dict, lookup: Dict[str, dict], pid: str) -> List[str]:
    """Build grounded GT for fully novel tasks."""
    query = task.get("task_query", "").lower()

    if "set the table" in query:
        plate_desc = describe("plate", lookup, with_location=False)
        plate_loc = get_location("plate", lookup)
        bowl_desc = describe("bowl", lookup, with_location=False)
        bowl_loc = get_location("bowl", lookup)
        knife_desc = describe("knife", lookup, with_location=False)
        knife_loc = get_location("knife", lookup)
        fork_desc = describe("fork", lookup, with_location=False)
        fork_loc = get_location("fork", lookup)
        cup_desc = describe("cup", lookup, with_location=False)
        cup_loc = get_location("cup", lookup)
        return [
            f"Pick up the {plate_desc} from {plate_loc}.",
            f"Place the {plate_desc.split('from')[0].strip()} on the table.",
            f"Pick up the {bowl_desc} from {bowl_loc}.",
            f"Place the {bowl_desc.split('from')[0].strip()} above the plate on the table.",
            f"Pick up the {knife_desc} from {knife_loc}.",
            "Place the knife to the right of the plate on the table.",
            f"Pick up the {fork_desc} from {fork_loc}.",
            "Place the fork to the left of the plate on the table.",
            f"Pick up the {cup_desc} from {cup_loc}.",
            "Place the cup above the knife on the table.",
        ]

    elif "clean" in query and "countertop" in query:
        cloth_desc = describe("cloth", lookup, with_location=False)
        cloth_loc = get_location("cloth", lookup)
        sink_desc = describe("sink", lookup, with_location=False)
        faucet_desc = describe("faucet", lookup, with_location=False)
        return [
            f"Clear loose items from the countertop to the {sink_desc}.",
            f"Pick up the {cloth_desc} from {cloth_loc}.",
            f"Turn on the {faucet_desc}.",
            f"Wet the {cloth_desc} under the {faucet_desc}.",
            f"Turn off the {faucet_desc}.",
            f"Wipe the countertop surface from one end to the other with the {cloth_desc}.",
            f"Rinse the {cloth_desc} under the {faucet_desc}.",
            "Wipe the countertop again to remove residue.",
            f"Rinse and wring out the {cloth_desc}.",
            f"Place the {cloth_desc} back on {cloth_loc}.",
        ]

    elif "salad" in query:
        cucumber_desc = describe("cucumber", lookup, with_location=False)
        cucumber_loc = get_location("cucumber", lookup)
        knife_desc = describe("knife", lookup, with_location=False)
        knife_loc = get_location("knife", lookup)
        board_desc = describe("cutting board", lookup, with_location=False)
        bowl_desc = describe("bowl", lookup, with_location=False)
        bowl_loc = get_location("bowl", lookup)
        sink_desc = describe("sink", lookup, with_location=False)
        faucet_desc = describe("faucet", lookup, with_location=False)
        return [
            f"Pick up the {cucumber_desc} from {cucumber_loc}.",
            f"Turn on the {faucet_desc}.",
            f"Rinse the cucumber under running water in the {sink_desc}.",
            f"Turn off the {faucet_desc}.",
            f"Place the cucumber on the {board_desc}.",
            f"Pick up the {knife_desc} from {knife_loc}.",
            f"Cut the cucumber into slices on the {board_desc}.",
            f"Transfer the cucumber slices to the {bowl_desc}.",
            f"Place the {knife_desc} back on {knife_loc}.",
            f"Mix the salad ingredients in the {bowl_desc}.",
        ]

    return task["ground_truth_steps"]


def main():
    parser = argparse.ArgumentParser(
        description="Build Generalize references from participant memory"
    )
    parser.add_argument("--memory-file", required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--input", required=True, help="Input benchmark JSON")
    parser.add_argument("--output", required=True, help="Output benchmark JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without saving")
    args = parser.parse_args()

    objects = load_memory_objects(args.memory_file, args.participant)
    lookup = build_object_lookup(objects)

    with open(args.input) as f:
        tasks = json.load(f)

    new_tasks = []
    for task in tasks:
        new_task = copy.deepcopy(task)
        original_gt = task["ground_truth_steps"]
        new_gt = build_grounded_gt(task, lookup, args.participant)
        new_task["ground_truth_steps"] = new_gt
        new_tasks.append(new_task)

        if args.dry_run:
            print(f"\n{'='*80}")
            print(f"Task: {task['task_query']}")
            print(f"Type: {task['task_type']}")
            print(f"\nOriginal GT ({len(original_gt)} steps):")
            for i, s in enumerate(original_gt):
                txt = s if isinstance(s, str) else s.get("narration", str(s))
                print(f"  {i+1}. {txt}")
            print(f"\nGrounded GT ({len(new_gt)} steps):")
            for i, s in enumerate(new_gt):
                print(f"  {i+1}. {s}")

    if not args.dry_run:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(new_tasks, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(new_tasks)} tasks to {args.output}")
    else:
        print(f"\n\n{'='*80}")
        print(f"DRY RUN: {len(new_tasks)} tasks would be written to {args.output}")


if __name__ == "__main__":
    main()
