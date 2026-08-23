#!/usr/bin/env python3
"""Compute the paper's Robot-Grounded Plan (RGP) metric.

RGP is the unweighted mean of three condition-level scores:

* OrderExec: ordered coverage of reference actions by executable plan steps.
* KeyObj: coverage of reference objects in the generated plan.
* PrefAdh: adherence to relevant participant preferences.

All three scores are deterministic and rule-based. This module reads saved
planning outputs, the released MEMORA-Planning suites, and the participant
memory files; it does not call an LLM.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Optional

from memora.evaluation.settings import PUBLIC_EVALUATION_CONDITIONS

CONDITION_ORDER = list(PUBLIC_EVALUATION_CONDITIONS)
MEMORA_CONDITIONS = {"memora_episodic", "memora_full"}
ROUTINE_MATCH_THRESHOLD = 0.10
PREFERENCE_RELEVANCE_THRESHOLD = 0.05

KITCHEN_NOUNS = {
    "pot", "pan", "wok", "saucepan", "kettle", "bowl", "plate", "dish",
    "cup", "mug", "glass", "bottle", "jar", "can", "tin", "container",
    "tray", "board", "tupperware", "box", "bag", "basket", "carton",
    "knife", "spoon", "fork", "spatula", "peeler", "grater", "tongs",
    "whisk", "ladle", "strainer", "colander", "opener", "scoop",
    "microwave", "toaster", "blender", "stove", "oven", "hob", "fridge",
    "refrigerator", "dishwasher", "sink", "freezer", "cooker",
    "counter", "countertop", "drawer", "cabinet", "cupboard", "shelf",
    "rack", "table", "tap", "faucet", "burner", "grill", "worktop",
    "bread", "toast", "potato", "onion", "garlic", "carrot", "tomato",
    "pepper", "vegetable", "veg", "meat", "chicken", "rice", "pasta",
    "egg", "milk", "cheese", "butter", "oil", "salt", "sugar", "flour",
    "sauce", "spice", "cereal", "fruit", "apple", "banana", "lemon",
    "herb", "seasoning", "bok", "choy", "noodle", "broth",
    "cloth", "sponge", "towel", "soap", "detergent", "spray", "bin",
    "trash", "rubbish", "tissue", "paper", "wipe",
    "lid", "cap", "handle", "skin", "core", "wrapper", "peel", "rind",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "in", "on", "at", "and", "or", "for", "with", "by", "from", "this",
    "that", "it", "its", "person", "participant", "p01", "p02", "p03",
    "p04", "p05", "p06", "p07", "p08", "p09", "tends", "typically",
    "usually", "while", "after", "before", "when", "their", "they", "user",
    "help",
}

VERB_SYNONYMS = {
    "grab": "grasp", "hold": "grasp", "grip": "grasp",
    "pick up": "grasp", "take": "grasp", "get": "grasp",
    "put": "place", "set": "place", "lay": "place", "rest": "place",
    "drop": "place", "position": "place",
    "carry": "move", "bring": "move", "transfer": "move", "transport": "move",
    "twist": "turn", "rotate": "turn", "switch": "turn",
    "wash": "wash", "rinse": "rinse", "clean": "wash", "scrub": "wash",
    "cut": "cut", "slice": "cut", "chop": "cut", "dice": "cut",
    "pour": "pour", "drizzle": "pour", "add": "pour",
    "open": "open", "close": "close", "shut": "close",
    "stir": "stir", "mix": "stir", "whisk": "stir",
    "peel": "peel", "reach": "reach", "extend": "reach", "wipe": "wipe",
    "fill": "fill", "refill": "fill", "the person is": "",
}

_VERB_INFLECTIONS = {
    "grasp": ["grasping", "grasps", "grasped", "picking", "picks", "picked",
              "grabbing", "grabs", "grabbed", "taking", "takes", "took",
              "taken", "holding", "holds", "held", "getting", "gets", "got"],
    "place": ["placing", "places", "placed", "putting", "puts", "set",
              "setting", "sets", "laying", "lays", "laid", "dropping",
              "drops", "dropped", "positioning", "positions", "positioned"],
    "move": ["moving", "moves", "moved", "carrying", "carries", "carried",
             "bringing", "brings", "brought", "transferring", "transfers",
             "transferred"],
    "turn": ["turning", "turns", "turned", "twisting", "twists", "twisted",
             "rotating", "rotates", "rotated", "switching", "switches",
             "switched"],
    "wash": ["washing", "washes", "washed", "cleaning", "cleans", "cleaned",
             "scrubbing", "scrubs", "scrubbed", "rinsing", "rinses", "rinsed"],
    "cut": ["cutting", "cuts", "slicing", "slices", "sliced", "chopping",
            "chops", "chopped", "dicing", "dices", "diced"],
    "pour": ["pouring", "pours", "poured", "drizzling", "drizzles",
             "drizzled", "adding", "adds", "added"],
    "open": ["opening", "opens", "opened"],
    "close": ["closing", "closes", "closed", "shutting", "shuts"],
    "stir": ["stirring", "stirs", "stirred", "mixing", "mixes", "mixed",
             "whisking", "whisks", "whisked"],
    "peel": ["peeling", "peels", "peeled"],
    "reach": ["reaching", "reaches", "reached", "extending", "extends",
              "extended"],
    "wipe": ["wiping", "wipes", "wiped"],
    "fill": ["filling", "fills", "filled", "refilling", "refills", "refilled"],
    "prepare": ["preparing", "prepares", "prepared"],
    "cook": ["cooking", "cooks", "cooked"],
    "use": ["using", "uses", "used"],
    "begin": ["beginning", "begins", "began", "starting", "starts", "started"],
    "continue": ["continuing", "continues", "continued"],
    "knead": ["kneading", "kneads", "kneaded"],
    "fold": ["folding", "folds", "folded"],
    "press": ["pressing", "presses", "pressed"],
    "shake": ["shaking", "shakes", "shook", "shaken"],
    "squeeze": ["squeezing", "squeezes", "squeezed"],
    "drain": ["draining", "drains", "drained"],
    "boil": ["boiling", "boils", "boiled"],
    "fry": ["frying", "fries", "fried"],
}
for _canonical_verb, _forms in _VERB_INFLECTIONS.items():
    VERB_SYNONYMS.setdefault(_canonical_verb, _canonical_verb)
    for _form in _forms:
        VERB_SYNONYMS.setdefault(_form, _canonical_verb)

_NON_VERB_TOKENS = {
    "person", "a", "the", "he", "she", "they", "it",
    "user", "participant", "subject",
}
_STEM_SUFFIXES = ("ing", "ied", "ies", "ed", "es", "s")
_COLOR_TOKENS = {
    "black", "white", "red", "green", "blue", "yellow", "silver", "grey",
    "gray", "brown", "pink", "orange", "purple", "gold", "transparent",
    "clear", "metal", "metallic",
}


def _memora_source_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "memora"


def _memory_root() -> Path:
    configured = os.environ.get("PLANNING_MEMORY_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    data_root = Path(
        os.environ.get("MEMORA_DATA_ROOT")
        or Path(__file__).resolve().parents[2] / "data"
    ).expanduser().resolve()
    return data_root / "participant_memory" / "memora_paper"


def _stem(token: str) -> str:
    for suffix in _STEM_SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            base = token[:-len(suffix)]
            if len(base) >= 3 and base[-1] == base[-2]:
                base = base[:-1]
            return base
    return token


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"\b[a-z][a-z\-]{2,}\b", text.lower())
    return {_stem(token) for token in tokens if token not in STOPWORDS}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _extract_action_verb(step: str) -> str:
    text = re.sub(r"^\d+[\.\)]\s*", "", step.lower().strip())
    text = re.sub(
        r"^(the\s+|a\s+)?person\s+"
        r"(is\s+|are\s+|was\s+|were\s+|will\s+|"
        r"begins?\s+(to\s+)?|continues?\s+(to\s+)?)?",
        "",
        text,
    )
    text = re.sub(
        r"^(he|she|they|it)\s+(is\s+|are\s+|was\s+|were\s+|will\s+)?",
        "",
        text,
    ).strip()
    for phrase, canonical in sorted(
        VERB_SYNONYMS.items(), key=lambda item: -len(item[0])
    ):
        if text == phrase or text.startswith(phrase + " ") or (
            text.startswith(phrase + "s ") and phrase.endswith(("s", "h"))
        ):
            return canonical or "_filler_"
    tokens = text.split()
    if not tokens:
        return ""
    head = tokens[0]
    if head in _NON_VERB_TOKENS and len(tokens) > 1:
        head = tokens[1]
    if head in _NON_VERB_TOKENS:
        return "_filler_"
    stemmed = _stem(head)
    return VERB_SYNONYMS.get(stemmed, stemmed)


def _planning_suite_patterns() -> list[str]:
    source_root = _memora_source_root().parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from memora_bench.paths import glob_planning_suite_patterns

    return glob_planning_suite_patterns(source_root)


def _load_benchmark_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for pattern in _planning_suite_patterns():
        for filename in sorted(glob.glob(pattern)):
            try:
                payload = json.loads(Path(filename).read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot load benchmark file {filename}: {exc}") from exc
            tasks = payload
            if not isinstance(tasks, list):
                raise ValueError(f"Benchmark file must contain a task list: {filename}")
            for position, task in enumerate(tasks):
                if not isinstance(task, dict):
                    raise ValueError(f"Task {position} is not an object: {filename}")
                task_id = str(task.get("task_id") or "").strip()
                if not task_id:
                    raise ValueError(f"Task {position} has no task_id: {filename}")
                if task_id in index:
                    raise ValueError(f"Duplicate benchmark task_id {task_id!r}")
                index[task_id] = task
    if not index:
        raise FileNotFoundError("No released MEMORA-Planning tasks found")
    return index


def _load_memory(participant_id: str) -> dict[str, Any]:
    path = (
        _memory_root()
        / "memora_full"
        / f"participant_memory_{participant_id.lower()}.json"
    )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load participant memory {path}: {exc}") from exc


def _planning_result_files(output_root: Path) -> list[Path]:
    files: list[Path] = []
    for participant_dir in sorted(output_root.glob("P*")):
        if not participant_dir.is_dir():
            continue
        released = sorted(participant_dir.glob("results_*.json"))
        files.extend(released or sorted(participant_dir.glob("planning_results_*.json")))
    return files


def _condition_from_result(path: Path, payload: dict[str, Any]) -> str:
    released = set(CONDITION_ORDER)
    filename_condition = (
        path.name.removeprefix("results_").removesuffix(".json")
        if path.name.startswith("results_")
        else ""
    )
    metadata_condition = str(payload.get("condition") or "").strip()
    if filename_condition in released:
        if metadata_condition and metadata_condition != filename_condition:
            raise ValueError(
                f"Result condition {metadata_condition!r} disagrees with {path.name}"
            )
        return filename_condition
    if metadata_condition in released:
        return metadata_condition
    raise ValueError(f"Result does not identify a released memory setting: {path}")


def _memory_nouns(memory: dict[str, Any]) -> set[str]:
    nouns: set[str] = set()
    for video_memory in (memory.get("memories_by_video") or {}).values():
        registry = video_memory.get("object_registry") or {}
        if isinstance(registry, dict):
            entries = registry.values()
        elif isinstance(registry, list):
            entries = registry
        else:
            continue
        for entry in entries:
            name = entry.get("name", "") if isinstance(entry, dict) else ""
            tokens = re.findall(r"[a-z]+", name.lower())
            if not tokens:
                continue
            nouns.add(_stem(tokens[-1]))
            nouns.update(
                _stem(token)
                for token in tokens
                if token not in _COLOR_TOKENS and len(token) >= 4
            )
    procedures = (
        (memory.get("inferred_knowledge") or {})
        .get("reusable_procedures", {})
        .get("procedure_templates", [])
    )
    for procedure in procedures:
        for key_object in procedure.get("key_objects", []) or []:
            value = (
                key_object.get("object", "")
                if isinstance(key_object, dict)
                else str(key_object)
            )
            tokens = re.findall(r"[a-z]+", value.lower().replace("_", " "))
            if not tokens:
                continue
            nouns.add(_stem(tokens[-1]))
            nouns.update(
                _stem(token)
                for token in tokens
                if token not in _COLOR_TOKENS and len(token) >= 4
            )
    return nouns

def _reference_object_coverage(
    plan: list[str], task: dict[str, Any], memory: dict[str, Any]
) -> Optional[float]:
    if not plan:
        return None
    task_id = str(task.get("task_id") or "")
    if task_id.startswith("generalize_"):
        ground_truth = _ground_truth_steps(task.get("ground_truth_steps") or [])
        vocabulary = _memory_nouns(memory) or KITCHEN_NOUNS
        reference_objects = {
            _stem(token)
            for token in re.findall(r"[a-z]+", " ".join(ground_truth).lower())
            if _stem(token) in vocabulary
        }
    else:
        procedures = (
            (memory.get("inferred_knowledge") or {})
            .get("reusable_procedures", {})
            .get("procedure_templates", [])
        )
        query_tokens = _tokenize(str(task.get("task_query") or ""))
        best: Optional[dict[str, Any]] = None
        best_score = 0.0
        for procedure in procedures:
            candidate_tokens = _tokenize(str(procedure.get("goal") or ""))
            for key_object in procedure.get("key_objects", []) or []:
                value = (
                    key_object.get("object", "")
                    if isinstance(key_object, dict)
                    else str(key_object)
                )
                candidate_tokens |= _tokenize(value.replace("_", " "))
            for step in procedure.get("canonical_steps", []) or []:
                value = step.get("action", "") if isinstance(step, dict) else str(step)
                candidate_tokens |= _tokenize(value)
            score = _jaccard(query_tokens, candidate_tokens)
            if score > best_score:
                best_score = score
                best = procedure
        if best is None or best_score < ROUTINE_MATCH_THRESHOLD:
            return None
        # Preserve the procedure record exactly. Repeated key-object entries
        # remain repeated in the denominator, matching the reported metric.
        reference_objects: list[str] = []
        for key_object in best.get("key_objects", []) or []:
            value = (
                key_object.get("object", "")
                if isinstance(key_object, dict)
                else str(key_object)
            )
            value = value.lower().strip().replace("_", " ")
            if value:
                reference_objects.append(value)

    if not reference_objects:
        return None
    plan_text = " ".join(plan).lower()
    if task_id.startswith("generalize_"):
        plan_stems = {_stem(token) for token in re.findall(r"[a-z]+", plan_text)}
        hits = sum(reference in plan_stems for reference in reference_objects)
    else:
        hits = sum(
            reference in plan_text
            or any(
                token in plan_text
                for token in reference.split()
                if len(token) > 2
            )
            for reference in reference_objects
        )
    return hits / len(reference_objects)


def _preference_adherence(
    plan: list[str], task_query: str, preferences: list[dict[str, Any]]
) -> Optional[float]:
    if not plan or not preferences:
        return None
    query_tokens = _tokenize(task_query)
    plan_text = " ".join(plan).lower()
    plan_tokens = _tokenize(plan_text)
    scores: list[float] = []
    for preference in preferences:
        if not isinstance(preference, dict):
            continue
        text = str(preference.get("preference") or preference.get("text") or "")
        if not text:
            continue
        keywords = preference.get("keywords") or []
        if keywords:
            content: set[str] = set()
            for keyword in keywords:
                content |= _tokenize(str(keyword))
        else:
            content = _tokenize(text) - {"used", "typically", "use", "user"}
        if not content or _jaccard(query_tokens, content) < PREFERENCE_RELEVANCE_THRESHOLD:
            continue
        hits = sum(token in plan_tokens or token in plan_text for token in content)
        scores.append(hits / len(content))
    return mean(scores) if scores else None


def _ground_truth_steps(raw_steps: list[Any]) -> list[str]:
    steps: list[str] = []
    for step in raw_steps:
        if isinstance(step, dict):
            value = step.get("narration") or step.get("action") or ""
        else:
            value = step if isinstance(step, str) else ""
        if value.strip():
            steps.append(value.strip())
    return steps


def _ordered_executability(plan: list[str], reference: list[str]) -> Optional[float]:
    if not plan or not reference:
        return None
    reference_verbs = [
        verb for verb in map(_extract_action_verb, reference)
        if verb and verb != "_filler_"
    ]
    if not reference_verbs:
        return None
    plan_verbs = [
        "" if verb == "_filler_" else verb
        for verb in map(_extract_action_verb, plan)
    ]
    source_root = _memora_source_root().parent
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from memora.evaluation.planning.step_checks import PlanStepChecks

    step_scores = PlanStepChecks.compute_executability(plan).get("step_scores") or []
    plan_position = 0
    matched = 0
    for reference_verb in reference_verbs:
        found = False
        while plan_position < len(plan):
            score = step_scores[plan_position]
            valid = bool(
                score.get("has_action")
                and score.get("has_object")
                and score.get("has_location")
            )
            if valid and plan_verbs[plan_position] == reference_verb:
                matched += 1
                plan_position += 1
                found = True
                break
            plan_position += 1
        if not found:
            break
    # The released metric rounds each task before condition-level averaging.
    return round(matched / len(reference_verbs), 4)


def _mean_or_none(values: list[float]) -> Optional[float]:
    return mean(values) if values else None


def compute_planning_metrics(output_root: Path) -> dict[str, Any]:
    """Compute all RGP axes from one model/split output directory."""
    output_root = output_root.expanduser().resolve()
    benchmark = _load_benchmark_index()
    result_files = _planning_result_files(output_root)
    if not result_files:
        raise FileNotFoundError(f"No planning result files found under {output_root}")

    memories: dict[str, dict[str, Any]] = {}
    per_condition: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    scored_plan_counts: dict[str, int] = defaultdict(int)
    empty_plan_counts: dict[str, int] = defaultdict(int)

    for path in result_files:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load planning result {path}: {exc}") from exc
        participant_id = path.parent.name
        condition = _condition_from_result(path, payload)
        memory = memories.setdefault(participant_id, _load_memory(participant_id))
        knowledge = memory.get("inferred_knowledge") or {}
        preferences = (knowledge.get("preferences") or {}).get("statements") or []

        for result in payload.get("results", []):
            plan = result.get("generated_plan") or result.get("plan_steps") or []
            if isinstance(plan, str):
                plan = [line.strip() for line in plan.splitlines() if line.strip()]
            if not plan:
                empty_plan_counts[condition] += 1
                continue
            task_id = str(result.get("task_id") or "")
            task = benchmark.get(task_id)
            if task is None:
                raise ValueError(f"Unknown benchmark task_id {task_id!r} in {path}")
            task_query = str(
                result.get("task_query")
                or task.get("task_query")
                or ""
            )
            reference_steps = _ground_truth_steps(task.get("ground_truth_steps") or [])
            order_exec = _ordered_executability(plan, reference_steps)
            key_object = _reference_object_coverage(plan, task, memory)
            preference = _preference_adherence(plan, task_query, preferences)
            scored_plan_counts[condition] += 1
            for name, value in (
                ("order_exec", order_exec),
                ("key_object", key_object),
                ("preference", preference),
            ):
                if value is not None:
                    per_condition[condition][name].append(float(value))

    by_condition: dict[str, dict[str, Any]] = {}
    ordered = [
        name
        for name in CONDITION_ORDER
        if name in scored_plan_counts or name in empty_plan_counts
    ]
    for condition in ordered:
        order_exec = _mean_or_none(per_condition[condition]["order_exec"])
        key_object = _mean_or_none(per_condition[condition]["key_object"])
        preference = _mean_or_none(per_condition[condition]["preference"])
        if None in (order_exec, key_object, preference):
            raise ValueError(f"RGP axis is undefined for condition {condition}")
        rgp = mean([order_exec, key_object, preference])
        by_condition[condition] = {
            "n_plans_scored": scored_plan_counts[condition],
            "n_empty_plans": empty_plan_counts[condition],
            "order_exec": order_exec,
            "key_object": key_object,
            "preference_adherence": preference,
            "rgp": rgp,
        }
    return {"output_root": str(output_root), "by_condition": by_condition}


def write_planning_metrics(output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    output = output_root / "planning_metrics.json"
    payload = compute_planning_metrics(output_root)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return output


def aggregate_planning_rgp(
    output_root: Path,
    *,
    condition: Optional[str] = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """Load or compute RGP values and summarize MEMORA vs. baselines."""
    output_root = output_root.expanduser().resolve()
    metrics_path = output_root / "planning_metrics.json"
    if refresh or not metrics_path.exists():
        write_planning_metrics(output_root)
    payload = json.loads(metrics_path.read_text())
    panel = {
        name: float(row["rgp"])
        for name, row in payload["by_condition"].items()
    }
    result: dict[str, Any] = {
        "output_root": str(output_root),
        "panel_rgp": panel,
    }
    if condition:
        result["condition"] = condition
        result["rgp"] = panel.get(condition)

    memora = [(name, value) for name, value in panel.items() if name in MEMORA_CONDITIONS]
    baselines = [(name, value) for name, value in panel.items() if name not in MEMORA_CONDITIONS]
    if memora:
        name, value = max(memora, key=lambda item: item[1])
        result["best_memora"] = {"condition": name, "rgp": value}
    if baselines:
        name, value = max(baselines, key=lambda item: item[1])
        result["best_baseline"] = {"condition": name, "rgp": value}
        if memora:
            result["delta_vs_best_baseline"] = result["best_memora"]["rgp"] - value
    return result


def _print_metrics(payload: dict[str, Any]) -> None:
    print("condition              OrderExec   KeyObj   PrefAdh      RGP")
    print("-" * 64)
    for condition in CONDITION_ORDER:
        row = payload["by_condition"].get(condition)
        if row is None:
            continue
        print(
            f"{condition:<22}"
            f"{row['order_exec']:>10.3f}"
            f"{row['key_object']:>9.3f}"
            f"{row['preference_adherence']:>10.3f}"
            f"{row['rgp']:>9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = compute_planning_metrics(args.output_root)
    output = args.out or args.output_root / "planning_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    _print_metrics(payload)
    print(f"\nSaved {output}")


if __name__ == "__main__":
    main()
