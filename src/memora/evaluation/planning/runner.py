#!/usr/bin/env python3
# MEMORA-Planning benchmark evaluation.
"""
planning/runner.py – Planning benchmark evaluation for MEMORA.

Runs a ReAct-style agent on MEMORA-Planning tasks under the seven evaluation
settings reported in the paper. Each setting determines both the participant
memory representation and the planner interface used to query it.

Usage:
    memora-plan \
        --benchmark-file src/memora_bench/planning/suites/replay/p01.json \
        --memory-file /path/to/memora_full/participant_memory_p01.json \
        --condition memora_full \
        --output-dir results/planning
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memora.memory_agent.agent import VLLMInference, OpenAIInference, run_agent_loop
from memora.evaluation.planning.prompts import (
    DEFAULT_PLANNER_PROFILE,
    PLANNING_SYSTEM_PROMPT_NO_MEMORY,
)
from memora.evaluation.results import write_json_atomic
from memora.evaluation.settings import (
    CONDITION_MEMORY_TYPE,
    PUBLIC_EVALUATION_CONDITIONS,
)


_CONDITION_PLANNER_PROFILE = {
    "no_memory": None,
    "flat_1d_raw": None,
    "flat_1d_edited": None,
    "graph_2d_raw": "graph_2d",
    "graph_2d_edited": "graph_2d",
    "memora_episodic": "memora_episodic",
    "memora_full": "memora_full",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


from memora.evaluation.planning.planning_environment import PlanningEnvironment
from memora.evaluation.planning.parser import extract_plan_from_response
from memora.evaluation.planning.tasks import (
    load_planning_run_list,
    load_planning_suite,
    resolve_task_instruction,
)

# ---------------------------------------------------------------------------
# No-memory single-shot evaluation
# ---------------------------------------------------------------------------

def run_no_memory(
    model: VLLMInference,
    task: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Single-shot plan generation without any memory or tools."""

    task_query = resolve_task_instruction(task)
    participant_id = task.get("participant_id", "")

    user_content = (
        f"The person is {participant_id}.\n\n"
        f"Task: {task_query}\n\n"
        f"Generate a detailed step-by-step plan."
    )

    messages = [
        {"role": "system", "content": PLANNING_SYSTEM_PROMPT_NO_MEMORY},
        {"role": "user", "content": user_content},
    ]

    if verbose:
        logger.info("[no_memory] Generating plan for: %s", task_query)

    response = model.chat_completion(messages, tools=None)
    raw_text = response.get("content", "")
    plan = extract_plan_from_response(raw_text)

    if verbose:
        logger.info("[no_memory] Extracted %d steps", len(plan))

    return {
        "task_id": task.get("task_id", ""),
        "video_id": task.get("video_id", ""),
        "participant_id": task.get("participant_id", ""),
        "task_query": task_query,
        "task_type": task.get("task_type", ""),
        "generated_plan": plan,
        "ground_truth_steps": task.get("ground_truth_steps", []),
        "num_generated_steps": len(plan),
        "num_gt_steps": len(task.get("ground_truth_steps", [])),
        "tool_calls": [],
        "iterations": 1,
        "raw_response": raw_text,
    }


# ---------------------------------------------------------------------------
# Memory-backed evaluation (memora / flat_1d)
# ---------------------------------------------------------------------------

def run_with_memory(
    model: VLLMInference,
    env: PlanningEnvironment,
    task: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the ReAct agent loop and collect the generated plan."""

    result = run_agent_loop(model, env, task, verbose=verbose)

    final_info = result.get("final_info", {})
    generated_plan = final_info.get("generated_plan", [])
    raw_response = final_info.get("raw_response", "")
    tool_calls_log = final_info.get("tool_calls", [])

    if not generated_plan and raw_response:
        generated_plan = extract_plan_from_response(raw_response)

    if not raw_response:
        for resp in reversed(env._env._responses_log):
            if resp:
                raw_response = resp
                break
        if raw_response and not generated_plan:
            generated_plan = extract_plan_from_response(raw_response)

    serialisable_tools = []
    for tc in tool_calls_log:
        entry = {
            "tool": tc.get("tool", ""),
            "arguments": tc.get("arguments", {}),
        }
        serialisable_tools.append(entry)

    return {
        "task_id": task.get("task_id", ""),
        "video_id": task.get("video_id", ""),
        "participant_id": task.get("participant_id", ""),
        "task_query": resolve_task_instruction(task),
        "task_type": task.get("task_type", ""),
        "generated_plan": generated_plan,
        "ground_truth_steps": task.get("ground_truth_steps", []),
        "num_generated_steps": len(generated_plan),
        "num_gt_steps": len(task.get("ground_truth_steps", [])),
        "tool_calls": serialisable_tools,
        "iterations": result.get("iterations", 0),
        "raw_response": raw_response,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    model: VLLMInference,
    benchmark_data: List[Dict[str, Any]],
    condition: str,
    memory_file: Optional[str] = None,
    memory_type: str = "memora",
    max_iterations: int = 8,
    verbose: bool = False,
    planner_profile: Optional[str] = DEFAULT_PLANNER_PROFILE,
) -> List[Dict[str, Any]]:
    """Iterate over planning tasks and collect results."""

    results: List[Dict[str, Any]] = []
    total = len(benchmark_data)

    env: Optional[PlanningEnvironment] = None
    if condition != "no_memory":
        if not memory_file:
            raise ValueError(f"Condition '{condition}' requires --memory-file")
        env = PlanningEnvironment(
            memory_file=memory_file,
            max_iterations=max_iterations,
            memory_type=memory_type,
            planner_profile=planner_profile,
        )

    for idx, task in enumerate(benchmark_data, 1):
        task_id = task.get("task_id", f"task_{idx}")
        task_query = resolve_task_instruction(task)

        logger.info(
            "=== Task %d/%d [%s]: %s ===",
            idx, total, task_id, task_query[:80],
        )

        t0 = time.time()

        try:
            if condition == "no_memory":
                result = run_no_memory(model, task, verbose=verbose)
            else:
                result = run_with_memory(model, env, task, verbose=verbose)
        except Exception:
            logger.exception("Error on task %s", task_id)
            result = {
                "task_id": task_id,
                "video_id": task.get("video_id", ""),
                "participant_id": task.get("participant_id", ""),
                "task_query": task_query,
                "task_type": task.get("task_type", ""),
                "generated_plan": [],
                "ground_truth_steps": task.get("ground_truth_steps", []),
                "num_generated_steps": 0,
                "num_gt_steps": len(task.get("ground_truth_steps", [])),
                "tool_calls": [],
                "iterations": 0,
                "raw_response": "",
                "error": True,
            }

        elapsed = time.time() - t0
        result["elapsed_seconds"] = round(elapsed, 2)
        results.append(result)

        logger.info(
            "  -> %d steps generated (gt=%d) in %.1fs",
            result["num_generated_steps"],
            result["num_gt_steps"],
            elapsed,
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MEMORA-Planning evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        required=True,
        help="Hugging Face model name or API model identifier",
    )
    # NOTE: `--benchmark-file` / `--condition` are required ONLY in
    # single-task mode. In multi-task mode (`--task-list`) every spec in the
    # JSON list carries its own benchmark/condition/etc. — see the
    # multi-task path in main().
    p.add_argument(
        "--benchmark-file",
        default=None,
        help="Path to a MEMORA-Planning benchmark JSON file (required unless --task-list is set)",
    )
    p.add_argument(
        "--memory-file",
        dest="memory_file",
        default=None,
        help="Path to participant memory JSON file (not needed for no_memory)",
    )
    p.add_argument(
        "--condition",
        default=None,
        choices=PUBLIC_EVALUATION_CONDITIONS,
        help="Paper evaluation setting (required unless --task-list is set)",
    )
    p.add_argument(
        "--task-list",
        default=None,
        help="Path to a JSON task list evaluated with one loaded model",
    )
    p.add_argument(
        "--output-dir",
        default="results/planning",
        help="Output directory for results JSON",
    )
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Generation-token budget per model call",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=8,
        help="Maximum model iterations per task, including the forced final answer",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit the number of tasks to evaluate",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--require-e5",
        action="store_true",
        help="Fail if E5 cannot be loaded instead of using keyword retrieval",
    )

    # --- OpenAI API mode (alternative to vLLM) ---
    api = p.add_argument_group("API mode (use instead of local vLLM)")
    api.add_argument(
        "--use-api",
        action="store_true",
        help="Use OpenAI-compatible API instead of local vLLM",
    )
    api.add_argument(
        "--api-key",
        default=None,
        help="API key (or set DASHSCOPE_API_KEY / OPENAI_API_KEY)",
    )
    api.add_argument(
        "--api-base",
        default=None,
        help="Custom API base URL (for Azure, local proxies, etc.)",
    )
    return p.parse_args()


def _resolve_evaluation_setting(setting: str) -> Tuple[str, Optional[str]]:
    """Map a paper evaluation setting to its runtime memory and prompt types."""
    try:
        return CONDITION_MEMORY_TYPE[setting], _CONDITION_PLANNER_PROFILE[setting]
    except KeyError as exc:
        raise ValueError(
            f"Unknown evaluation setting {setting!r}; choose from "
            f"{', '.join(PUBLIC_EVALUATION_CONDITIONS)}"
        ) from exc


def _execute_planning_task(
    model,
    *,
    benchmark_file: str,
    evaluation_setting: str,
    memory_file: Optional[str],
    output_dir: str,
    max_iterations: int,
    num_samples: Optional[int],
    verbose: bool,
    model_name: str,
    pid_label: Optional[str] = None,
) -> Optional[Path]:
    """Run ONE (condition, memory, benchmark) tuple end-to-end and write a
    ``planning_results_<cond>_<ts>.json`` file. Returns the output file path
    on success, or None on a fatal error (with the error already logged).

    Designed to be safe to call repeatedly on the same in-process
    ``VLLMInference`` model — env / memory file / E5 cache are recreated per call but
    the LLM weights stay loaded (this is exactly the property EAM-QA's
    ``run-multi`` relies on).
    """
    memory_type, planner_profile = _resolve_evaluation_setting(evaluation_setting)
    runtime_condition = memory_type

    benchmark_path = Path(benchmark_file).expanduser().resolve()
    if not benchmark_path.exists():
        logger.error("Benchmark file not found: %s", benchmark_path)
        return None

    benchmark_data = load_planning_suite(benchmark_path)

    if num_samples and num_samples < len(benchmark_data):
        benchmark_data = benchmark_data[:num_samples]

    label = f"[{pid_label}] " if pid_label else ""
    logger.info("%sLoaded %d planning tasks from %s",
                label, len(benchmark_data), benchmark_path)
    logger.info("%sEvaluation setting: %s", label, evaluation_setting)

    results = run_evaluation(
        model=model,
        benchmark_data=benchmark_data,
        condition=runtime_condition,
        memory_file=memory_file,
        memory_type=memory_type,
        max_iterations=max_iterations,
        verbose=verbose,
        planner_profile=planner_profile,
    )

    output = {
        "condition": evaluation_setting,
        "model": model_name,
        "participant_memory_file": Path(memory_file).name if memory_file else None,
        "memory_type": memory_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "num_tasks": len(results),
        "max_iterations": max_iterations,
        "results": results,
    }

    plans_with_steps = [r for r in results if r["num_generated_steps"] > 0]
    avg_steps = (
        sum(r["num_generated_steps"] for r in plans_with_steps) / len(plans_with_steps)
        if plans_with_steps
        else 0
    )
    output["summary"] = {
        "tasks_with_plans": len(plans_with_steps),
        "tasks_failed": len(results) - len(plans_with_steps),
        "avg_generated_steps": round(avg_steps, 2),
        "avg_gt_steps": round(
            sum(r["num_gt_steps"] for r in results) / max(len(results), 1), 2
        ),
    }

    output_dir_path = Path(output_dir).expanduser().resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir_path / f"planning_results_{evaluation_setting}_{ts}.json"

    write_json_atomic(output_file, output)

    logger.info("%sResults written to %s", label, output_file)
    logger.info(
        "%sSummary: %d/%d tasks produced plans, avg %.1f steps (gt avg %.1f)",
        label,
        output["summary"]["tasks_with_plans"],
        len(results),
        output["summary"]["avg_generated_steps"],
        output["summary"]["avg_gt_steps"],
    )

    if output["summary"]["tasks_failed"]:
        logger.error(
            "%sRun completed with %d task(s) that produced no plan; "
            "the diagnostic output was retained at %s",
            label,
            output["summary"]["tasks_failed"],
            output_file,
        )
        return None

    return output_file


def _init_model_once(args: argparse.Namespace):
    """Create either ``OpenAIInference`` or ``VLLMInference`` once and force
    eager initialization. Used by both single-task and multi-task paths so
    the LLM weights stay in memory across many evaluations.
    """
    if args.use_api:
        model = OpenAIInference(
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        model.initialize()
    else:
        # IMPORTANT: vLLM must init BEFORE sentence_transformers (E5) to
        # avoid CUDA fork deadlock. Force eager initialization here.
        model = VLLMInference(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        model.initialize()
    return model


def main() -> None:
    args = parse_args()
    if args.require_e5:
        os.environ["MEMORA_REQUIRE_E5"] = "1"

    if args.task_list:
        # ------------------------------------------------------------------
        # Multi-task mode (mirrors multi-task runner.job): load the model
        # ONCE, loop through the task-list, write one results file per spec.
        # ------------------------------------------------------------------
        tasklist_path = Path(args.task_list).expanduser().resolve()
        if not tasklist_path.exists():
            logger.error("Task list not found: %s", tasklist_path)
            sys.exit(1)

        task_specs = load_planning_run_list(tasklist_path)

        logger.info("Loaded %d planning specs from %s",
                    len(task_specs), tasklist_path)
        for i, spec in enumerate(task_specs, 1):
            tag = spec.get("tag") or (
                f"{spec.get('pid', '?')}/{spec.get('condition', '?')}"
            )
            logger.info("  %2d. %s", i, tag)

        model = _init_model_once(args)

        n_ok = 0
        n_fail = 0
        for idx, spec in enumerate(task_specs, 1):
            bench = spec["benchmark_file"]
            condition = spec["condition"]
            pid_label = spec.get("pid")
            logger.info(
                "\n%s\n[%d/%d] %s\n%s",
                "=" * 72,
                idx, len(task_specs),
                spec.get("tag")
                or f"{pid_label or '?'} / {condition}",
                "=" * 72,
            )
            try:
                out = _execute_planning_task(
                    model,
                    benchmark_file=bench,
                    evaluation_setting=condition,
                    memory_file=spec.get("memory_file"),
                    output_dir=spec.get("output_dir", args.output_dir),
                    max_iterations=int(
                        spec.get("max_iterations", args.max_iterations)
                    ),
                    num_samples=spec.get("num_samples", args.num_samples),
                    verbose=args.verbose,
                    model_name=args.model,
                    pid_label=pid_label,
                )
                if out is None:
                    n_fail += 1
                else:
                    n_ok += 1
            except Exception:
                logger.exception("Spec %d failed (%s)", idx, spec)
                n_fail += 1

        logger.info(
            "\n%s\nMulti-task done: %d ok, %d failed (of %d)\n%s",
            "=" * 72, n_ok, n_fail, len(task_specs), "=" * 72,
        )
        sys.exit(0 if n_fail == 0 else 2)

    # ------------------------------------------------------------------
    # Single-task mode
    # ------------------------------------------------------------------
    if not args.benchmark_file or not args.condition:
        logger.error(
            "Single-task mode requires --benchmark-file and --condition "
            "(or pass --task-list for multi-task mode)."
        )
        sys.exit(1)

    model = _init_model_once(args)
    output_file = _execute_planning_task(
        model,
        benchmark_file=args.benchmark_file,
        evaluation_setting=args.condition,
        memory_file=args.memory_file,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        num_samples=args.num_samples,
        verbose=args.verbose,
        model_name=args.model,
    )
    if output_file is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
