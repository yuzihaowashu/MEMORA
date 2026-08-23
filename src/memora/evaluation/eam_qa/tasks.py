"""Load and validate EAM-QA questions and evaluation task lists."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from memora.evaluation.settings import PUBLIC_EVALUATION_CONDITIONS


QUESTION_TYPES = {"SPref", "SHabit", "SRoutine", "ERecall"}


def expand_path(path_value):
    """Expand environment variables and ``~`` in a configured path."""
    if path_value is None:
        return None
    return os.path.expanduser(os.path.expandvars(str(path_value)))


def question_id(task: Dict[str, Any]) -> str:
    """Return the immutable ID stored with a released EAM-QA item."""
    value = str(task.get("question_id") or "").strip()
    if not value:
        raise ValueError("EAM-QA item has no question_id")
    return value


def load_benchmark(path: str) -> List[Dict[str, Any]]:
    """Load and validate one released EAM-QA question file."""
    expanded = Path(expand_path(path))
    try:
        with expanded.open() as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load EAM-QA benchmark {expanded}: {exc}") from exc

    if isinstance(data, dict):
        questions = data.get("questions")
    elif isinstance(data, list):
        questions = data
    else:
        questions = None
    if not isinstance(questions, list) or not questions:
        raise ValueError(
            f"EAM-QA benchmark {expanded} must contain a non-empty question list"
        )

    seen_ids = set()
    for index, task in enumerate(questions):
        if not isinstance(task, dict):
            raise ValueError(f"{expanded}: question {index} is not a JSON object")
        if not str(task.get("question") or "").strip():
            raise ValueError(f"{expanded}: question {index} has no question text")
        choices = task.get("choices")
        if not isinstance(choices, list) or len(choices) != 5:
            raise ValueError(f"{expanded}: question {index} must have five choices")
        answer = str(task.get("correct_answer") or "").strip().upper()
        if answer not in {"A", "B", "C", "D", "E"}:
            raise ValueError(
                f"{expanded}: question {index} has invalid answer {answer!r}"
            )
        question_type = str(task.get("qa_type") or "").strip()
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"{expanded}: question {index} has invalid EAM-QA type "
                f"{question_type!r}"
            )
        identifier = question_id(task)
        if identifier in seen_ids:
            raise ValueError(f"{expanded}: duplicate question ID {identifier!r}")
        seen_ids.add(identifier)
    return questions


def load_task_list(path: str) -> List[Dict[str, Any]]:
    """Load and validate a multi-run task list before model initialization."""
    expanded = Path(expand_path(path))
    try:
        with expanded.open(encoding="utf-8") as stream:
            tasks = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load EAM-QA task list {expanded}: {exc}") from exc
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"EAM-QA task list {expanded} must be a non-empty JSON array")

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"{expanded}: task {index} is not a JSON object")
        for field in ("benchmark_file", "condition", "output"):
            if not str(task.get(field) or "").strip():
                raise ValueError(f"{expanded}: task {index} is missing {field!r}")
        condition = task["condition"]
        if condition not in PUBLIC_EVALUATION_CONDITIONS:
            raise ValueError(
                f"{expanded}: task {index} has unknown condition {condition!r}"
            )
        if condition != "no_memory" and not str(task.get("memory_file") or "").strip():
            raise ValueError(f"{expanded}: task {index} requires 'memory_file'")
    return tasks
