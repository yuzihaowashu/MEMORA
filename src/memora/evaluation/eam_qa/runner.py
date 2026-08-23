#!/usr/bin/env python3
# EAM-QA: MEMORA Embodied Memory Assessment.
"""EAM-QA evaluation: multiple-choice with 5 options (A–E; E = unanswerable).

Uses the shared AgentEnvironment and ReAct agent loop. Supports single-run and
run-multi (model loaded once, multiple tasks).

Usage:
    # Single run
    memora-eam-qa run \
        --model google/gemma-4-26B-A4B-it \
        --benchmark-file src/memora_bench/eam_qa/questions/p01.json \
        --condition memora_full \
        --memory-file participant_memory/memora_paper/memora_full/participant_memory_p01.json

    # Multi-task (model loaded once)
    memora-eam-qa run-multi \
        --model google/gemma-4-26B-A4B-it \
        --task-list /path/to/eam_qa_tasks.json
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

from memora.memory_agent.agent_environment import AgentEnvironment
from memora.evaluation.eam_qa.answers import extract_multiple_choice_answer
from memora.evaluation.eam_qa.prompts import (
    NO_MEMORY_SYSTEM_PROMPT,
    format_multiple_choice_question,
)
from memora.evaluation.eam_qa.tasks import (
    expand_path,
    load_benchmark,
    load_task_list,
    question_id,
)
from memora.evaluation.results import is_complete_eam_qa_result, write_json_atomic
from memora.evaluation.settings import (
    CONDITION_MEMORY_TYPE,
    PUBLIC_EVALUATION_CONDITIONS,
)


def _run_one_eval(model, benchmark, condition, memory_file, output_path, verbose=True):
    """Run EAM-QA for one paper condition and participant-memory file."""
    from memora.memory_agent.agent import run_agent_loop

    if condition == "no_memory":
        env = None
    else:
        if not str(memory_file or "").strip():
            raise ValueError(f"--memory-file is required for {condition}")
        runtime_memory_type = CONDITION_MEMORY_TYPE[condition]
        env = AgentEnvironment(
            memory_file=memory_file,
            memory_type=runtime_memory_type,
        )

    start = time.time()
    results = []
    total = len(benchmark)
    type_scores = defaultdict(list)
    e_selections = defaultdict(int)
    type_counts = defaultdict(int)

    for i, task in enumerate(benchmark):
        print(f"\n{'='*70}")
        print(f"  [{i+1}/{total}] EAM-QA Q")
        print(f"{'='*70}")
        print(f"Q: {task.get('question', '')}")
        print(f"Choices: {len(task.get('choices', []))}")
        print(f"GT: {task.get('correct_answer', '')}")
        print(f"Type: {task.get('qa_type', '')}")

        try:
            if env is None:
                response = model.chat_completion(
                    [
                        {"role": "system", "content": NO_MEMORY_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": format_multiple_choice_question(task),
                        },
                    ],
                    tools=None,
                )
                response_text = response.get("content", "")
                if verbose:
                    print(f"Response: {response_text}")
            else:
                agent_result = run_agent_loop(
                    model=model, env=env, task=task, verbose=verbose
                )
                info = agent_result["final_info"]
                response_text = info.get("final_answer", "")

            predicted = extract_multiple_choice_answer(response_text)

            question_type = str(task["qa_type"])
            correct_answer = task.get("correct_answer", "").strip().upper()
            is_correct = predicted == correct_answer
            selected_e = predicted == "E"

            result = {
                "question_id": question_id(task),
                "question": task.get("question", ""),
                "correct_answer": correct_answer,
                "predicted": predicted,
                "is_correct": is_correct,
                "selected_e": selected_e,
                "qa_type": question_type,
            }
            results.append(result)
            type_scores[question_type].append(1.0 if is_correct else 0.0)
            type_counts[question_type] += 1
            if selected_e:
                e_selections[question_type] += 1

            print(f"Predicted: {predicted}  Correct: {correct_answer}  "
                  f"{'✓' if is_correct else '✗'}"
                  f"{'  [SELECTED E]' if selected_e else ''}")

        except Exception as e:
            import traceback
            print(f"ERROR: {e}\n{traceback.format_exc()}")
            question_type = str(task["qa_type"])
            results.append({
                "question_id": question_id(task),
                "question": task.get("question", ""),
                "correct_answer": task.get("correct_answer", "").strip().upper(),
                "error": str(e),
                "is_correct": False,
                "selected_e": False,
                "predicted": "",
                "qa_type": question_type,
            })
            type_scores[question_type].append(0.0)
            type_counts[question_type] += 1

    elapsed = time.time() - start

    # Aggregate
    n = len(results) or 1
    overall_acc = sum(r["is_correct"] for r in results) / n
    overall_e_rate = sum(r["selected_e"] for r in results) / n

    output_data = {
        "overall_accuracy": overall_acc,
        "overall_e_rate": overall_e_rate,
        "total_questions": total,
        "correct_answers": sum(r["is_correct"] for r in results),
        "e_selections": sum(r["selected_e"] for r in results),
        "failed_questions": sum(bool(r.get("error")) for r in results),
        "per_type_accuracy": {t: sum(v)/len(v) for t, v in type_scores.items()},
        "per_type_e_rate": {t: e_selections[t]/type_counts[t] for t in type_counts},
        "results": results,
        "config": {
            "condition": condition,
            "memory_type": CONDITION_MEMORY_TYPE[condition],
            "participant_memory_file": memory_file or None,
            "format": "mc5",
            "elapsed_time": elapsed,
        },
    }

    write_json_atomic(output_path, output_data)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  EAM-QA Evaluation Results ({condition})")
    print(f"{'='*60}")
    print(f"Accuracy: {overall_acc*100:.1f}%")
    print(f"E-selection rate: {overall_e_rate*100:.1f}%")
    print(f"N={total}, Time={elapsed:.1f}s")
    if output_data["failed_questions"]:
        print(f"Failed questions: {output_data['failed_questions']}")
    print("\nBy Type:")
    for t in sorted(type_scores.keys()):
        acc = sum(type_scores[t]) / len(type_scores[t]) * 100
        e_rate = e_selections[t] / type_counts[t] * 100
        print(f"  {t}: Acc={acc:.1f}%  E-rate={e_rate:.1f}%")
    print(f"\nResults saved to {output_path}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description="EAM-QA Evaluation (5-choice with option E)")
    sub = parser.add_subparsers(dest="command")

    # Single run
    run = sub.add_parser("run", help="Run single EAM-QA evaluation")
    run.add_argument("--model", type=str, required=True)
    run.add_argument("--benchmark-file", type=str, required=True)
    run.add_argument(
        "--memory-file",
        dest="memory_file",
        type=str,
        default=None,
        help="Path to participant memory JSON file",
    )
    run.add_argument("--output", type=str, default="results_eam_qa.json")
    run.add_argument(
        "--condition",
        type=str,
        default="memora_full",
        choices=PUBLIC_EVALUATION_CONDITIONS,
        help="Paper evaluation setting represented by --memory-file",
    )
    run.add_argument("--tensor-parallel-size", type=int, default=1)
    run.add_argument("--max-model-len", type=int, default=32768)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--temperature", type=float, default=0.6)
    run.add_argument(
        "--require-e5",
        action="store_true",
        help="Fail if E5 cannot be loaded instead of using keyword retrieval",
    )
    run.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the full agent trace (default: disabled)",
    )
    # API mode (drop-in replacement for vLLM)
    run.add_argument("--use-api", action="store_true",
                     help="Use an OpenAI-compatible API instead of local vLLM")
    run.add_argument("--api-key", type=str, default=None,
                     help="API key (or set DASHSCOPE_API_KEY / OPENAI_API_KEY)")
    run.add_argument("--api-base", type=str, default=None,
                     help="Custom API base URL (e.g. DashScope endpoint)")

    # Multi-task
    multi = sub.add_parser("run-multi",
        help="Run multiple EAM-QA eval tasks sharing one model")
    multi.add_argument("--model", type=str, required=True)
    multi.add_argument("--task-list", type=str, required=True)
    multi.add_argument("--tensor-parallel-size", type=int, default=1)
    multi.add_argument("--max-model-len", type=int, default=32768)
    multi.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    multi.add_argument("--temperature", type=float, default=0.6)
    multi.add_argument(
        "--require-e5",
        action="store_true",
        help="Fail if E5 cannot be loaded instead of using keyword retrieval",
    )
    multi.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print the full agent trace (default: disabled)",
    )
    multi.add_argument("--skip-existing", action="store_true",
                       help="Skip tasks whose output file already exists (resume-safe)")
    # API mode
    multi.add_argument("--use-api", action="store_true",
                       help="Use OpenAI-compatible API instead of local vLLM")
    multi.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set DASHSCOPE_API_KEY / OPENAI_API_KEY)",
    )
    multi.add_argument("--api-base", type=str, default=None)

    args = parser.parse_args()
    if getattr(args, "require_e5", False):
        os.environ["MEMORA_REQUIRE_E5"] = "1"

    if args.command == "run-multi":
        tasks = load_task_list(args.task_list)

        if args.use_api:
            from memora.memory_agent.agent import OpenAIInference
            print(f"Initialising API client once: {args.model}")
            model = OpenAIInference(
                model_name=args.model,
                api_key=args.api_key,
                api_base=args.api_base,
                temperature=args.temperature,
            )
            model.initialize()
        else:
            from memora.memory_agent.agent import VLLMInference
            print(f"Loading model once: {args.model}")
            model = VLLMInference(
                model_name=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                temperature=args.temperature,
            )

        total_start = time.time()
        ran = 0
        skipped = 0
        failed = 0
        for ti, task_cfg in enumerate(tasks):
            out_path = Path(expand_path(task_cfg["output"]))
            if args.skip_existing and is_complete_eam_qa_result(out_path):
                skipped += 1
                print(f"\n{'#'*70}")
                print(f"  TASK {ti+1}/{len(tasks)}: SKIP (output exists)")
                print(f"  {out_path}")
                print(f"{'#'*70}")
                continue
            if args.skip_existing and out_path.exists():
                print(f"Incomplete or invalid output will be rerun: {out_path}")

            print(f"\n{'#'*70}")
            print(f"  TASK {ti+1}/{len(tasks)}: {task_cfg.get('condition','?')} "
                  f"| {Path(task_cfg['benchmark_file']).stem}")
            print(f"{'#'*70}")

            benchmark = load_benchmark(task_cfg["benchmark_file"])

            output_data = _run_one_eval(
                model=model,
                benchmark=benchmark,
                condition=task_cfg.get("condition", "memora_full"),
                memory_file=expand_path(
                    task_cfg.get("memory_file", "")
                ) or "",
                output_path=str(out_path),
                verbose=args.verbose,
            )
            ran += 1
            if output_data["failed_questions"]:
                failed += 1

        total_elapsed = time.time() - total_start
        print(f"\n{'='*70}")
        print(
            f"  DONE: {ran} ran, {skipped} skipped, {failed} failed, "
            f"{len(tasks)} in list  ({total_elapsed:.0f}s total)"
        )
        print(f"{'='*70}")
        if failed:
            raise SystemExit(2)
        return

    if args.command != "run":
        parser.print_help()
        return

    benchmark = load_benchmark(args.benchmark_file)

    print(f"  Loaded {args.benchmark_file}: {len(benchmark)} questions")
    if args.use_api:
        from memora.memory_agent.agent import OpenAIInference
        print(f"\nInitialising API client: {args.model}")
        model = OpenAIInference(
            model_name=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
            temperature=args.temperature,
        )
        model.initialize()
    else:
        from memora.memory_agent.agent import VLLMInference
        print(f"\nLoading model: {args.model}")
        model = VLLMInference(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            temperature=args.temperature,
        )

    output_data = _run_one_eval(
        model=model,
        benchmark=benchmark,
        condition=args.condition,
        memory_file=args.memory_file or "",
        output_path=args.output,
        verbose=args.verbose,
    )
    if output_data["failed_questions"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
