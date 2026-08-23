#!/usr/bin/env python3
"""Build the static data bundle used by the MEMORA-Bench web explorer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "src" / "memora_bench"
DEFAULT_OUTPUT = REPO_ROOT / "website" / "assets" / "benchmark-explorer-data.js"


def _clean_question(text: str) -> str:
    return text.split("\n\nIMPORTANT:", 1)[0].strip()


def _valid_video_ids(values: list[Any] | None) -> list[str]:
    return [str(value) for value in (values or []) if value and "???" not in str(value)]


def _load_eam_qa() -> tuple[list[dict[str, Any]], dict[str, int]]:
    items: list[dict[str, Any]] = []
    core_count = 0
    control_count = 0

    for path in sorted((BENCH_ROOT / "eam_qa" / "questions").glob("p*.json")):
        payload = json.loads(path.read_text())
        participant = path.stem.upper()
        for question in payload["questions"]:
            is_control = bool(question.get("is_unanswerable"))
            core_count += int(not is_control)
            control_count += int(is_control)
            items.append(
                {
                    "id": question["question_id"],
                    "participant": question.get("participant_id", participant),
                    "video": question.get("video_id"),
                    "videoIds": _valid_video_ids(question.get("video_ids")),
                    "type": question["qa_type"],
                    "question": _clean_question(question["question"]),
                    "choices": question["choices"],
                    "correctAnswer": question["correct_answer"],
                    "groundTruth": question["ground_truth"],
                    "isUnanswerableControl": is_control,
                }
            )

    return items, {"core": core_count, "controls": control_count}


def _normalize_steps(steps: list[Any]) -> list[str]:
    normalized = []
    for step in steps:
        if isinstance(step, str):
            normalized.append(step.strip())
        elif isinstance(step, dict):
            normalized.append(str(step.get("narration") or step.get("step") or step).strip())
        else:
            normalized.append(str(step).strip())
    return normalized


def _load_participant_video_ids() -> dict[str, list[str]]:
    path = BENCH_ROOT / "planning" / "data" / "participant_video_ids.jsonl"
    videos: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        video_id = json.loads(line)["video_id"]
        participant = video_id.split("_", 1)[0]
        videos.setdefault(participant, []).append(video_id)
    return videos


def _load_planning(split: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    suite_dir = BENCH_ROOT / "planning" / "suites" / split
    participant_videos = _load_participant_video_ids()

    for path in sorted(suite_dir.glob("p*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a top-level task array")
        for task in payload:
            participant = task.get("participant_id", path.stem.upper())
            video_ids = _valid_video_ids(task.get("video_ids")) or participant_videos.get(participant, [])
            items.append(
                {
                    "id": task["task_id"],
                    "participant": participant,
                    "video": task.get("video_id"),
                    "videoIds": video_ids,
                    "type": task.get("task_type", "planning"),
                    "query": task.get("task_query"),
                    "groundTruthSteps": _normalize_steps(task.get("ground_truth_steps", [])),
                    "primaryObjects": task.get("primary_objects", []),
                    "sourceAction": task.get("source_action"),
                    "sourceObject": task.get("source_object"),
                    "targetObject": task.get("target_object"),
                    "rationale": task.get("rationale"),
                }
            )
    return items


def build_bundle() -> dict[str, Any]:
    eam_qa, qa_counts = _load_eam_qa()
    replay = _load_planning("replay")
    generalize = _load_planning("generalize")
    participants = sorted({item["participant"] for item in eam_qa})

    return {
        "version": 1,
        "stats": {
            "participants": len(participants),
            "videoHours": 45,
            "eamQaCore": qa_counts["core"],
            "eamQaControls": qa_counts["controls"],
            "eamQaStored": len(eam_qa),
            "qaTypes": dict(sorted(Counter(item["type"] for item in eam_qa).items())),
            "planningReplay": len(replay),
            "planningGeneralize": len(generalize),
            "replayTypes": dict(sorted(Counter(item["type"] for item in replay).items())),
            "generalizeTypes": dict(sorted(Counter(item["type"] for item in generalize).items())),
        },
        "participants": participants,
        "eamQa": eam_qa,
        "planningReplay": replay,
        "planningGeneralize": generalize,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    bundle = build_bundle()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    args.output.write_text(f"window.MEMORA_BENCHMARK = {serialized};\n")
    print(
        f"Wrote {args.output} with {len(bundle['eamQa'])} EAM-QA items, "
        f"{len(bundle['planningReplay'])} Replay tasks, and "
        f"{len(bundle['planningGeneralize'])} Generalize tasks."
    )


if __name__ == "__main__":
    main()
