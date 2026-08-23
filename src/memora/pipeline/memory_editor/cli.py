"""
Memory Editor CLI entry-point.

Contains:
- run_typed_memory_editor  – orchestrates MEMORA typed memory across videos
- main()  – argparse CLI
- _print_final_summary
"""

import argparse
import copy
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from memora.pipeline.formation_config import EPIC_KITCHENS_CONFIG
from memora.pipeline.memory_editor.records import (
    load_flat_1d_memory,
    load_segment_observations,
)
from memora.pipeline.memory_editor.flat_1d import run_memory_editor
from memora.pipeline.memory_editor.segment_processing import process_typed_memory_segment
from memora.pipeline.memory_editor.typed_memory import EmbodiedMemoryState
from memora.pipeline.api_client import resolve_api_credentials

logger = logging.getLogger(__name__)
PAPER_MEMORY_EDITOR_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


class APIChatTokenizer:
    """Minimal tokenizer shim exposing apply_chat_template for the memory editor."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}:\n{content}")
        if add_generation_prompt:
            parts.append("ASSISTANT:\n")
        return "\n\n".join(parts)


class APIChatLLM:
    """Small vLLM-compatible wrapper around an OpenAI-compatible chat API."""

    def __init__(self, model_name: str, api_base: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_base, self.api_key = resolve_api_credentials(api_base, api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)

    def generate(self, prompts, sampling_params=None):
        outputs = []
        for prompt in prompts:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=getattr(sampling_params, "max_tokens", 4096) if sampling_params else 4096,
                temperature=getattr(sampling_params, "temperature", 0.2) if sampling_params else 0.2,
            )
            text = resp.choices[0].message.content or ""
            outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=text)]))
        return outputs


# ============================================================================
# MEMORA typed-memory orchestrator
# ============================================================================

def run_typed_memory_editor(
    segment_observations: List[Dict[str, Any]],
    llm,
    sampling_params,
    tokenizer,
    output_dir: Optional[Path] = None,
    max_objects: int = 50,  # Max objects in prompt
    use_e5_retrieval: bool = False,  # Use E5 for large registries
    video_ids_filter: Optional[List[str]] = None,  # Filter to specific videos
    config=None,  # Optional formation prompt configuration.
) -> tuple:
    """
    Run MEMORA typed-memory editor on segment observations.

    Args:
        segment_observations: Segment observations from the Segment Encoder
        llm: vLLM model
        sampling_params: Sampling parameters
        tokenizer: Tokenizer
        output_dir: Output directory for saving results
        max_objects: Maximum objects to include in prompt (default: 50)
        use_e5_retrieval: Use E5 semantic retrieval when object registry is large
        video_ids_filter: If provided, only process these video IDs (for testing)
        config: Optional formation configuration for prompt wording and place IDs.

    Returns:
        (all_memories, all_history, total_ops)
    """
    # Group segments by video_id
    grouped_segments = defaultdict(list)
    for seg in segment_observations:
        video_id = seg.get("video_id", "unknown")
        grouped_segments[video_id].append(seg)

    logger.info("MEMORA online memory editing")
    logger.info(f"   Segments: {len(segment_observations)}")
    logger.info(f"   Videos: {len(grouped_segments)}")
    logger.info("   Episodic organization: one edited state per video")
    logger.info(f"   Max objects per prompt: {max_objects}")
    logger.info(f"   E5 retrieval: {'enabled' if use_e5_retrieval else 'disabled'}")

    all_memories = {}
    all_history = []
    total_ops = {
        "activity_log_ADD": 0,
        "environment_ADD": 0,
        "environment_UPDATE": 0,
        "object_registry_ADD": 0,
        "object_registry_UPDATE": 0,
        "object_registry_DELETE": 0,
        "object_registry_NOOP": 0,
    }

    # Status file for batch-job monitoring
    status_file = output_dir / ".status" if output_dir else None
    def write_status(msg):
        if status_file:
            try:
                status_file.write_text(msg + "\n")
            except OSError:
                pass

    # Open files for incremental writing
    history_file_handle = None
    if output_dir:
        history_file = output_dir / "memory_edit_history.jsonl"
        history_file_handle = open(history_file, 'w', encoding='utf-8')

    try:
        # Get video IDs (filtered if specified)
        all_video_ids = sorted(grouped_segments.keys())

        if video_ids_filter:
            # Filter to only process specified videos
            video_ids_set = set(video_ids_filter)
            video_ids = [vid for vid in all_video_ids if vid in video_ids_set]
            skipped = len(all_video_ids) - len(video_ids)
            logger.info(f"Video filter applied: {len(video_ids)}/{len(all_video_ids)} videos selected, {skipped} skipped")
        else:
            video_ids = all_video_ids

        participant_ids = {
            video_id.split("_", 1)[0]
            for video_id in video_ids
            if video_id and video_id != "unknown"
        }
        if len(participant_ids) != 1:
            raise ValueError(
                "One Memory Editor run must contain videos from exactly one "
                f"participant; found {sorted(participant_ids) or ['unknown']}"
            )
        participant_id = next(iter(participant_ids))
        logger.info(f"   Participant: {participant_id}")

        total_videos = len(video_ids)
        total_segments = sum(len(grouped_segments[vid]) for vid in video_ids)
        global_seg_done = 0

        # Print header for batch-job logs.
        print(f"\n{'═'*60}", flush=True)
        print("MEMORA online memory editing", flush=True)
        print(f"   Total videos: {total_videos}" + (f" (filtered from {len(all_video_ids)})" if video_ids_filter else ""), flush=True)
        print(f"   Total segments: {total_segments}", flush=True)
        print(f"{'═'*60}", flush=True)
        write_status("loading model...")

        for vid_idx, video_id in enumerate(video_ids):
            segments = sorted(grouped_segments[video_id], key=lambda x: x.get("turn_id", 0))

            # Progress line for batch-job logs.
            progress_pct = int((vid_idx + 1) / total_videos * 100)
            print(f"\n[Online editing] Video {vid_idx+1}/{total_videos} ({progress_pct}%): {video_id} ({len(segments)} segments)", flush=True)
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({video_id}) preparing...")

            # Initialize empty MEMORA memory for this video
            memory = EmbodiedMemoryState(
                environment_log=[],
                object_registry={},
                activity_log=[],
            )

            logger.info(f"\nProcessing {video_id}: {len(segments)} segments")

            for seg in segments:
                turn_id = seg.get("turn_id", 0)
                time_window = seg.get("time_window", {})

                # Capture memory before
                memory_before = copy.deepcopy(memory.to_dict())

                # Process segment
                memory, rule_ops, llm_ops = process_typed_memory_segment(
                    memory, seg, llm, sampling_params, tokenizer,
                    max_objects=max_objects,
                    use_e5_retrieval=use_e5_retrieval,
                    config=config
                )

                # Count explicit Memory Editor operations.
                for op in rule_ops + llm_ops:
                    layer = op.get("layer", "unknown")
                    event = op.get("event", "NOOP")
                    key = f"{layer}_{event}"
                    if key in total_ops:
                        total_ops[key] += 1

                global_seg_done += 1
                seg_num = segments.index(seg) + 1
                status_msg = (
                    f"[{global_seg_done}/{total_segments}] "
                    f"seg {seg_num}/{len(segments)} | "
                    f"vid {vid_idx+1}/{total_videos} ({video_id})"
                )
                write_status(status_msg)

                # Build history entry
                history_entry = {
                    "video_id": video_id,
                    "turn_id": turn_id,
                    "time_window": time_window,
                    "memory_before": memory_before,
                    "new_segment_summary": seg.get("activity_narrative", {}).get("summary", ""),
                    "rule_based_operations": rule_ops,
                    "llm_operations": llm_ops,
                    "memory_after": memory.to_dict()
                }
                all_history.append(history_entry)

                # Incremental write
                if history_file_handle:
                    history_file_handle.write(json.dumps(history_entry, ensure_ascii=False) + '\n')
                    history_file_handle.flush()

            # Store final memory for this video
            all_memories[video_id] = memory.to_dict()

            # Progress output for batch-job logs.
            print(f"    Completed: {len(memory.object_registry)} objects, {len(memory.activity_log)} activities", flush=True)
            logger.info(f"    {video_id}: {len(memory.object_registry)} objects, "
                       f"{len(memory.activity_log)} activities")
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({video_id}) DONE")

        # Final summary for batch-job logs.
        total_objects = sum(len(m.get("object_registry", {})) for m in all_memories.values())
        total_activities = sum(len(m.get("activity_log", [])) for m in all_memories.values())
        print(f"\n{'═'*60}", flush=True)
        print("Online memory editing complete.", flush=True)
        print(f"    Videos processed: {total_videos}", flush=True)
        print(f"    Total objects tracked: {total_objects}", flush=True)
        print(f"    Total activities logged: {total_activities}", flush=True)
        print(f"{'═'*60}\n", flush=True)
        write_status(
            f"FINISHED {total_videos} videos {total_objects} objects "
            f"{total_activities} activities"
        )

    finally:
        if history_file_handle:
            history_file_handle.close()

    return all_memories, all_history, total_ops


# ============================================================================
# CLI entry-point
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="MEMORA Memory Editor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  memora-memory-edit \
    --input segment_observations.jsonl \
    --output-dir participant_memory
"""
    )

    # Input/Output
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Path to Segment Encoder observations (JSONL)")
    parser.add_argument("--output-dir", "-o", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--initial-flat-1d-memory", type=str, default=None,
                       help="Initial memory used only by the Flat-1D baseline")
    parser.add_argument("--video-ids-file", type=str, default=None,
                       help="Path to file with video IDs to process (one per line). "
                            "If not provided, all videos in input file will be processed.")

    # Memory scope settings
    parser.add_argument("--flat-1d-scope", type=str,
                       choices=["per_video", "per_participant", "global"],
                       default="per_participant",
                       help="Grouping used only by the Flat-1D baseline")
    parser.add_argument("--max-records-in-prompt", type=int, default=60,
                       help="Maximum memory records shown to the model per edit")
    parser.add_argument("--use-e5-retrieval", action="store_true",
                       help="Use E5 semantic retrieval to select records shown to the Memory Editor")
    parser.add_argument(
        "--require-e5",
        action="store_true",
        help="Fail if E5 cannot be loaded instead of using keyword retrieval",
    )

    # Memory representation
    parser.add_argument("--memory-format", type=str,
                       choices=["memora", "flat_1d"],
                       default="memora",
                       help="Memory representation (default: memora)")

    # Model settings
    parser.add_argument(
        "--model-name",
        type=str,
        help=(
            "Model name. Defaults to the paper Memory Editor for local vLLM; "
            "required for API mode because endpoint aliases differ."
        ),
    )
    parser.add_argument("--backend", type=str, default="vllm",
                       choices=["vllm", "api"],
                       help="Text LLM backend for memory editing")
    parser.add_argument("--api-base", type=str, default=None,
                       help="(api) OpenAI-compatible API base URL. Defaults to DashScope compatible endpoint.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="(api) API key. Otherwise uses the environment key matching --api-base.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=16384,
                       help="Maximum model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")

    # Visualization options
    parser.add_argument("--enable-visualization", action="store_true",
                       help="Enable Flat-1D operation visualization")
    parser.add_argument("--print-summary", action="store_true",
                       help="Print operation statistics summary at the end")

    # Offline consolidation
    parser.add_argument("--run-offline-consolidation", action="store_true",
                       help="Run offline consolidation after online memory editing. "
                            "Extracts recurring procedures and participant-specific regularities "
                            "with the same Memory Editor model.")

    args = parser.parse_args()

    if args.require_e5 and not args.use_e5_retrieval:
        parser.error("--require-e5 requires --use-e5-retrieval")
    if args.require_e5:
        os.environ["MEMORA_REQUIRE_E5"] = "1"

    if args.model_name is None:
        if args.backend == "api":
            parser.error("--model-name is required with --backend api")
        args.model_name = PAPER_MEMORY_EDITOR_MODEL

    formation_config = EPIC_KITCHENS_CONFIG.copy()
    logger.info("Formation config: %s", formation_config)

    # Validate paths
    input_file = Path(args.input)
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load video IDs filter if provided
    video_ids_filter = None
    if args.video_ids_file:
        video_ids_file = Path(args.video_ids_file)
        if video_ids_file.exists():
            with open(video_ids_file, 'r') as f:
                video_ids_filter = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            logger.info(f" Video IDs filter loaded: {len(video_ids_filter)} videos from {video_ids_file}")
        else:
            logger.warning(f" Video IDs file not found: {video_ids_file}")

    # Load extracted facts
    extracted_facts = load_segment_observations(input_file)

    if not extracted_facts:
        logger.error("No extracted facts found")
        sys.exit(1)

    is_typed_memory = args.memory_format == "memora"

    # Load initial memory if provided (only for flat-memory format)
    initial_memory = []
    if args.initial_flat_1d_memory and not is_typed_memory:
        initial_memory_file = Path(args.initial_flat_1d_memory)
        if initial_memory_file.exists():
            initial_memory = load_flat_1d_memory(initial_memory_file)
        else:
            logger.warning(f"Initial memory file not found: {initial_memory_file}")

    # Initialize model backend
    logger.info(f" Loading memory editor backend: {args.backend} ({args.model_name})")
    logger.info(f"   Memory format: {'MEMORA typed memory' if is_typed_memory else 'flat facts'}")
    if args.memory_format == "flat_1d":
        logger.info(f"   Flat-1D scope: {args.flat_1d_scope}")
    if not is_typed_memory:
        logger.info(f"   Max records per prompt: {args.max_records_in_prompt}")
        logger.info(
            "   E5 retrieval: %s",
            "enabled" if args.use_e5_retrieval else "disabled (using recency)",
        )

    if args.backend == "api":
        llm = APIChatLLM(args.model_name, api_base=args.api_base, api_key=args.api_key)
        tokenizer = APIChatTokenizer()
        sampling_params = SimpleNamespace(temperature=0.2, max_tokens=4096)
        logger.info(f" API memory editor ready: {llm.api_base}")
    else:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        llm = LLM(
            model=args.model_name,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True
        )

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=8192,
            stop=["<|im_end|>"],
        )

        logger.info(" vLLM model loaded")

    # Run memory editor based on format
    if is_typed_memory:
        # MEMORA typed-memory processing
        logger.info(f" Processing {len(extracted_facts)} typed-memory segments...")
        logger.info(f"   Max Entity records per prompt: {args.max_records_in_prompt}")
        logger.info(f"   E5 Retrieval: {' Enabled (for large registries)' if args.use_e5_retrieval else ' Disabled'}")
        logger.info(f"Incremental saving enabled - results saved immediately to: {output_dir}")

        all_memories, all_history, total_ops = run_typed_memory_editor(
            extracted_facts,
            llm,
            sampling_params,
            tokenizer,
            output_dir=output_dir,
            max_objects=args.max_records_in_prompt,
            use_e5_retrieval=args.use_e5_retrieval,
            video_ids_filter=video_ids_filter,  # Filter to specific videos
            config=formation_config,
        )
        all_group_memories = all_memories  # For typed memory, all_memories is dict by video_id
    else:
        # Flat-memory format processing
        logger.info(f" Processing {len(extracted_facts)} fact entries...")
        logger.info(f"Incremental saving enabled - results saved immediately to: {output_dir}")

        all_memories, all_history, total_ops, all_group_memories = run_memory_editor(
            extracted_facts,
            initial_memory,
            llm,
            sampling_params,
            tokenizer,
            memory_scope=args.flat_1d_scope,
            max_memories=args.max_records_in_prompt,
            enable_visualization=args.enable_visualization,
            output_dir=output_dir,
            use_e5_retrieval=args.use_e5_retrieval
        )

    # operation_history.jsonl and participant_memory_per_group.jsonl
    # are already saved incrementally during processing!

    # Save combined memory file (final summary)
    if is_typed_memory:
        # MEMORA typed-memory format: all_memories is dict by video_id
        memory_data = {
            "memory_format": "memora",
            "participant_id": next(iter(all_memories)).split("_", 1)[0],
            "memories_by_video": all_memories,
            "inferred_knowledge": {},
            "total_operations": total_ops
        }
        history_file = output_dir / "memory_edit_history.jsonl"
    else:
        memory_data = {
            "memory_format": "flat_1d",
            "memory": [m.to_dict() for m in all_memories] if isinstance(all_memories, list) else all_memories,
            "memory_scope": args.flat_1d_scope,
            "total_operations": total_ops
        }
        history_file = output_dir / "operation_history.jsonl"

    memory_file = output_dir / "participant_memory.json"
    with open(memory_file, 'w', encoding='utf-8') as f:
        json.dump(memory_data, f, ensure_ascii=False, indent=2)

    # Note: per-group memory files are saved incrementally as JSONL; the
    # aggregate JSON is convenient for inspection and downstream tooling.
    if not is_typed_memory and args.flat_1d_scope in ["per_video", "per_participant"]:
        memory_per_group_json_file = output_dir / "participant_memory_per_group.json"
        with open(memory_per_group_json_file, 'w', encoding='utf-8') as f:
            json.dump(all_group_memories, f, ensure_ascii=False, indent=2)
        logger.info(f"   Saved {len(all_group_memories)} group memory files (JSON format)")

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": timestamp,
        "model": args.model_name,
        "input_file": str(input_file),
        "memory_format": "memora" if is_typed_memory else "flat_1d",
        "episodic_organization": "per_video" if is_typed_memory else args.flat_1d_scope,
        "max_records_in_prompt": args.max_records_in_prompt,
        "use_e5_retrieval": args.use_e5_retrieval,
        "initial_memory_size": len(initial_memory) if not is_typed_memory else 0,
        "total_entries": len(extracted_facts),
        "final_memory_size": len(all_memories) if isinstance(all_memories, list) else len(all_memories),
        "num_groups": len(all_group_memories) if isinstance(all_group_memories, dict) else "N/A",
        "operations": total_ops,
        "output_files": {
            "participant_memory": str(memory_file),
            "operation_history": str(history_file)
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save this execution's runtime settings.
    run_config = {
        "input_file": str(input_file),
        "initial_flat_1d_memory": args.initial_flat_1d_memory,
        "memory_format": "memora" if is_typed_memory else "flat_1d",
        "episodic_organization": "per_video" if is_typed_memory else args.flat_1d_scope,
        "max_records_in_prompt": args.max_records_in_prompt,
        "use_e5_retrieval": args.use_e5_retrieval,
        "model_name": args.model_name,
        "backend": args.backend,
        "api_base": args.api_base if args.backend == "api" else None,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization
    }

    config_file = output_dir / "config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    # ============================================================================
    # Offline consolidation
    # ============================================================================
    # Consolidate Inferred Knowledge with the loaded Memory Editor model.
    # ============================================================================

    if is_typed_memory and args.run_offline_consolidation:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Offline consolidation")
        logger.info("=" * 60)
        logger.info("   Reusing the Memory Editor model")

        from memora.pipeline.consolidation import run_offline_consolidation

        inferred_knowledge = run_offline_consolidation(
            participant_memory=all_memories,
            llm=llm,
            sampling_params=sampling_params,
            tokenizer=tokenizer,
            config=formation_config,
        )
        memory_data["inferred_knowledge"] = inferred_knowledge

        memory_file_tmp = memory_file.with_suffix('.json.tmp')
        with open(memory_file_tmp, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        memory_file_tmp.replace(memory_file)

        preferences = inferred_knowledge.get('preferences', {})
        logger.info("   Offline consolidation complete.")
        logger.info(
            "   Preferences: %d",
            len(preferences.get('storage_preferences', [])),
        )
        logger.info(
            "   Reusable procedures: %d",
            len(inferred_knowledge.get('action_sequences', [])),
        )

    logger.info("=" * 60)
    logger.info("Memory editing complete.")
    logger.info(f"   Memory format: {'MEMORA typed memory' if is_typed_memory else 'flat facts'}")
    if is_typed_memory:
        logger.info("   Episodic organization: one edited state per video")
    else:
        logger.info(f"   Flat-1D scope: {args.flat_1d_scope}")

    if is_typed_memory:
        logger.info(f"   Processed: {len(extracted_facts)} segment observations")
        logger.info(f"   Videos processed: {len(all_memories)}")
        # Print typed-memory specific stats
        total_objects = sum(len(m.get("object_registry", {})) for m in all_memories.values())
        total_activities = sum(len(m.get("activity_log", [])) for m in all_memories.values())
        logger.info(f"   Total objects tracked: {total_objects}")
        logger.info(f"   Total activities logged: {total_activities}")
        logger.info("   Operations summary:")
        for key, val in total_ops.items():
            if val > 0:
                logger.info(f"     - {key}: {val}")
    else:
        logger.info(f"   Max records per prompt: {args.max_records_in_prompt}")
        logger.info(f"   E5 Retrieval: {' Enabled' if args.use_e5_retrieval else ' Disabled'}")
        logger.info(f"   Initial memory: {len(initial_memory)} entries")
        logger.info(f"   Processed: {len(extracted_facts)} fact entries")
        logger.info(f"   Groups processed: {len(all_group_memories)}")
        logger.info(f"   Final memory: {len(all_memories) if isinstance(all_memories, list) else 'per-group'} entries")
        logger.info("   Operations:")
        logger.info(f"     - ADD: {total_ops['ADD']}")
        logger.info(f"     - UPDATE: {total_ops['UPDATE']}")
        logger.info(f"     - DELETE: {total_ops['DELETE']}")

    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 60)

    # Print detailed summary if requested (flat-memory format only)
    if args.print_summary and not is_typed_memory:
        _print_final_summary(
            total_ops=total_ops,
            all_group_memories=all_group_memories,
            all_history=all_history,
            memory_scope="per_video" if is_typed_memory else args.flat_1d_scope,
            initial_memory_size=len(initial_memory),
            final_memory_size=len(all_memories) if isinstance(all_memories, list) else sum(len(group_memory['memory']) for group_memory in all_group_memories.values())
        )


def _print_final_summary(
    total_ops: Dict[str, int],
    all_group_memories: Dict[str, Dict],
    all_history: List[Dict],
    memory_scope: str,
    initial_memory_size: int,
    final_memory_size: int
):
    """
     Print comprehensive final summary of all memory operations.
    """
    print(f"\n{'='*70}")
    print("MEMORA MEMORY EDITING - FINAL OPERATION SUMMARY")
    print(f"{'='*70}")

    print("\n OVERALL STATISTICS:")
    print(f"   Total operations: {sum(total_ops.values())}")
    print(f"    ADD:    {total_ops['ADD']:5d}")
    print(f"    UPDATE: {total_ops['UPDATE']:5d}")
    print(f"    DELETE: {total_ops['DELETE']:5d}")
    print(f"    NOOP:   {total_ops['NOOP']:5d}")

    print("\n MEMORY EVOLUTION:")
    print(f"   Initial: {initial_memory_size} entries")
    print(f"   Final:   {final_memory_size} entries")
    print(f"   Change:  {final_memory_size - initial_memory_size:+d}")

    print(f"\n BY GROUP ({memory_scope}):")
    # Sort by total operations
    sorted_groups = sorted(
        all_group_memories.items(),
        key=lambda x: sum(x[1].get('operations', {}).values()),
        reverse=True
    )

    for group_id, data in sorted_groups[:10]:
        ops = data.get('operations', {})
        mem_size = len(data.get('memory', []))
        total = sum(ops.values())
        print(f"   {group_id}: {mem_size} mem, {total} ops "
              f"(ADD:{ops.get('ADD', 0)}, UPD:{ops.get('UPDATE', 0)}, DEL:{ops.get('DELETE', 0)})")

    if len(sorted_groups) > 10:
        print(f"   ... and {len(sorted_groups) - 10} more groups")

    # Time window analysis
    if all_history:
        print("\nTIME WINDOW ANALYSIS:")
        print(f"   Total time windows: {len(all_history)}")

        # Count operations per time window
        ops_per_window = []
        for h in all_history:
            ops_count = len(h.get('operations', []))
            ops_per_window.append(ops_count)

        if ops_per_window:
            import statistics
            print(f"   Ops per window: min={min(ops_per_window)}, max={max(ops_per_window)}, "
                  f"avg={statistics.mean(ops_per_window):.1f}")

    print(f"\n{'='*70}")
    print(" Summary complete.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
