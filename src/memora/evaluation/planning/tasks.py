"""Load and validate MEMORA-Planning tasks and multi-run specifications."""

import json
from pathlib import Path
from typing import Any, Dict, List

from memora.evaluation.settings import PUBLIC_EVALUATION_CONDITIONS


def resolve_task_instruction(task: Dict[str, Any]) -> str:
    """Return the benchmark's natural-language planning instruction."""
    return str(task.get("task_query") or "").strip()


def load_planning_suite(path: Path) -> List[Dict[str, Any]]:
    """Load and validate one Replay or Generalize planning suite."""
    try:
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load MEMORA-Planning benchmark {path}: {exc}") from exc

    tasks = data if isinstance(data, list) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(
            f"MEMORA-Planning benchmark {path} must contain a non-empty task list"
        )

    seen_ids = set()
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"{path}: task {index} is not a JSON object")
        if not resolve_task_instruction(task):
            raise ValueError(f"{path}: task {index} has no task query")
        steps = task.get("ground_truth_steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{path}: task {index} has no ground-truth steps")
        task_id = str(task.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"{path}: task {index} has no task ID")
        if task_id in seen_ids:
            raise ValueError(f"{path}: duplicate task ID {task_id!r}")
        seen_ids.add(task_id)
    return tasks


def load_planning_run_list(path: Path) -> List[Dict[str, Any]]:
    """Load and validate multi-run specifications before model initialization."""
    try:
        with path.open(encoding="utf-8") as stream:
            specs = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load planning task list {path}: {exc}") from exc
    if not isinstance(specs, list) or not specs:
        raise ValueError(f"Planning task list {path} must be a non-empty JSON array")

    for index, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: spec {index} is not a JSON object")
        for field in ("benchmark_file", "condition"):
            if not str(spec.get(field) or "").strip():
                raise ValueError(f"{path}: spec {index} is missing {field!r}")
        condition = spec["condition"]
        if condition not in PUBLIC_EVALUATION_CONDITIONS:
            raise ValueError(f"{path}: spec {index} has unknown condition {condition!r}")
        if condition != "no_memory" and not str(spec.get("memory_file") or "").strip():
            raise ValueError(f"{path}: spec {index} requires 'memory_file'")
    return specs
