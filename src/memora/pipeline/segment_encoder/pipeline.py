"""Command-line orchestration for the MEMORA Segment Encoder."""

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from memora.pipeline.formation_config import EPIC_KITCHENS_CONFIG
from memora.pipeline.segment_encoder import (
    APISegmentEncoder,
    HuggingFaceSegmentEncoder,
    TYPED_MEMORY_VLM_PROMPT,
    VLLMSegmentEncoder,
)
from memora.pipeline.segment_encoder.core import DIRECT_VIDEO_INSTRUCTION
from memora.pipeline.segment_encoder.records import (
    append_records,
    load_completed_video_ids,
    prune_uncommitted_records,
    summarize_records,
    write_completed_video_ids,
)
from memora.pipeline.segment_encoder.video import (
    find_video_path,
    generate_segments_from_duration,
    get_video_duration,
    load_video_ids,
)

logger = logging.getLogger(__name__)
PAPER_SEGMENT_ENCODER_MODEL = "Qwen/Qwen2.5-Omni-7B"

def main():
    parser = argparse.ArgumentParser(description="Encode video segments into MEMORA observations")

    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--video-ids-file", type=str, required=True)
    parser.add_argument(
        "--model-name",
        type=str,
        help=(
            "Model name. Defaults to the paper Segment Encoder for local backends; "
            "required for API mode because endpoint aliases differ."
        ),
    )
    parser.add_argument(
        "--segment-length",
        type=float,
        default=None,
        help="Seconds per segment. Defaults to the paper formation configuration.",
    )
    parser.add_argument(
        "--observation-format",
        default="memora",
        choices=["memora", "flat_1d"],
        help="Segment Encoder output: MEMORA typed observations or Flat-1D observations",
    )
    parser.add_argument("--segment-cache-dir", type=str, default=None,
                        help="Directory to cache cut video segments. If set, segments are reused across runs.")
    parser.add_argument("--video-fps", type=float, default=None,
                        help="Output FPS for video segments. Defaults to the paper formation configuration.")

    parser.add_argument("--backend", type=str, default="huggingface",
                        choices=["huggingface", "vllm-omni", "api"],
                        help="Inference backend: huggingface (Transformers) or "
                             "vllm-omni (vLLM offline inference for Omni video models) "
                             "or api (OpenAI-compatible vision API)")

    # vLLM-Omni specific options (only used when --backend=vllm-omni)
    parser.add_argument("--max-model-len", type=int, default=12800,
                        help="(vllm-omni) Maximum model context length")
    parser.add_argument("--num-frames", type=int, default=-1,
                        help="(vllm-omni) Frames to sample per segment. "
                             "-1 (default) = all frames (best when --video-fps is set). "
                             "Positive int = cap at N frames (saves memory for long segments)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="(vllm-omni) GPU memory utilization (0.0-1.0)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="(vllm-omni) Split model across N GPUs. Use >1 for large models "
                             "(e.g. 30B needs TP=2+ on H100 80GB)")
    parser.add_argument("--api-base", type=str, default=None,
                        help="(api) OpenAI-compatible API base URL. Defaults to DashScope compatible endpoint.")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="(api) API key. Otherwise uses the environment key matching --api-base.",
    )
    parser.add_argument("--api-max-frames", type=int, default=4,
                        help="(api) Maximum sampled frames per segment.")

    args = parser.parse_args()

    if args.model_name is None:
        if args.backend == "api":
            parser.error("--model-name is required with --backend api")
        args.model_name = PAPER_SEGMENT_ENCODER_MODEL

    formation_config = EPIC_KITCHENS_CONFIG.copy()
    # Explicit CLI values take precedence over the paper configuration.
    if args.segment_length is None:
        args.segment_length = formation_config.segment_length
    if args.video_fps is None:
        args.video_fps = formation_config.video_fps
    logger.info("Formation config: %s", formation_config)

    if args.segment_length is None or args.segment_length <= 0:
        parser.error("--segment-length must be greater than zero")
    if args.video_fps is not None and args.video_fps <= 0:
        parser.error("--video-fps must be greater than zero")
    if args.api_max_frames <= 0:
        parser.error("--api-max-frames must be greater than zero")

    # Select the Segment Encoder output representation.
    if args.observation_format == "memora":
        if TYPED_MEMORY_VLM_PROMPT is None:
            logger.error("Could not build the MEMORA Segment Encoder prompt.")
            sys.exit(1)
        logger.info("Using MEMORA Segment Encoder prompt (Environment, Entity, Activity)")
    else:
        logger.info("Using flat-fact prompt")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(args.video_dir)
    video_ids = load_video_ids(args.video_ids_file)

    total_videos = len(video_ids)
    logger.info(f"Processing {total_videos} videos")

    # Initialize processor based on backend
    if args.backend == "vllm-omni":
        logger.info(f"Using vLLM-Omni backend (TP={args.tensor_parallel_size})")
        processor = VLLMSegmentEncoder(
            model_name=args.model_name,
            observation_format=args.observation_format,
            video_fps=args.video_fps,
            config=formation_config,
            max_model_len=args.max_model_len,
            num_frames=args.num_frames,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    elif args.backend == "api":
        logger.info(" Using OpenAI-compatible vision API backend")
        processor = APISegmentEncoder(
            model_name=args.model_name,
            observation_format=args.observation_format,
            video_fps=args.video_fps,
            config=formation_config,
            api_base=args.api_base,
            api_key=args.api_key,
            api_max_frames=args.api_max_frames,
        )
    else:
        logger.info("Using HuggingFace Transformers backend")
        processor = HuggingFaceSegmentEncoder(
            model_name=args.model_name,
            observation_format=args.observation_format,
            video_fps=args.video_fps,
            config=formation_config,
        )
    # Pre-scan to compute total segment count for progress tracking
    total_segments = 0
    _vid_seg_counts = {}
    for _vid in video_ids:
        vp = find_video_path(_vid, video_dir)
        if vp is None:
            _vid_seg_counts[_vid] = 0
            continue
        try:
            dur = get_video_duration(str(vp))
            _n = len(generate_segments_from_duration(dur, args.segment_length))
        except Exception as e:
            logger.warning(f"Could not probe {_vid}: {e}")
            _n = 0
        _vid_seg_counts[_vid] = _n
        total_segments += _n
    logger.info(f"Total segments across {total_videos} videos: {total_segments}")

    # Status file for batch-job monitoring
    status_file = output_dir / ".status"
    global_seg_done = 0
    def write_status(msg):
        try:
            status_file.write_text(msg + "\n")
        except OSError:
            pass

    write_status("loading model...")
    processor.load_model()
    write_status("model loaded, starting...")

    # Log FPS setting
    if args.video_fps:
        logger.info(f"Video FPS: {args.video_fps}")
    else:
        logger.info(" Video FPS: original (no resampling)")

    # The two representations have different output structures.
    is_typed_memory = args.observation_format == "memora"

    # Resume support. A video is complete only after its output has been
    # durably appended and its id committed to the atomic manifest.
    segments_file = output_dir / "segment_observations.jsonl"
    facts_file = output_dir / "flat_observations.jsonl"
    completion_file = output_dir / f"completed_{args.observation_format}_videos.json"
    output_file = segments_file if is_typed_memory else facts_file
    completed_video_ids = load_completed_video_ids(completion_file)
    prune_uncommitted_records(output_file, completed_video_ids)
    write_completed_video_ids(completion_file, completed_video_ids)
    requested_completed_ids = completed_video_ids.intersection(video_ids)
    if requested_completed_ids:
        logger.info(
            "Resume: found %d fully committed videos, skipping them",
            len(requested_completed_ids),
        )
        for skip_vid in sorted(requested_completed_ids):
            global_seg_done += _vid_seg_counts.get(skip_vid, 0)

    try:
        processed_count = 0
        skipped_count = 0

        for vid_idx, vid_id in enumerate(video_ids):
          if vid_id in completed_video_ids:
            skipped_count += 1
            logger.info(f"Skipping {vid_id} (already processed in previous run)")
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({vid_id}) SKIPPED (resume)")
            continue

          try:
            video_path = find_video_path(vid_id, video_dir)
            if video_path is None:
                logger.warning(f"Video not found: {vid_id}")
                continue

            try:
                vid_dur = get_video_duration(str(video_path))
                segments = generate_segments_from_duration(vid_dur, args.segment_length)
            except Exception as e:
                logger.warning(f"Cannot generate segments for {vid_id}: {e}")
                continue
            if not segments:
                logger.warning(f"No segments generated for {vid_id} (too short?)")
                continue

            processed_count += 1
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({vid_id}) preparing...")
            logger.info(f"Processing: {vid_id} ({_vid_seg_counts.get(vid_id, '?')} segs)")

            video_facts = []
            video_segment_observations = []

            # ==============================================================
            # Flat-1D segments are independent and can use vLLM batch inference.
            # ==============================================================
            use_batch = (
                args.backend == "vllm-omni"
                and not is_typed_memory
                and hasattr(processor, 'process_video_segments_batch')
            )
            if use_batch:
                logger.info(f"Batch mode: {len(segments)} segments -> single vLLM call")
                batch_results = processor.process_video_segments_batch(
                    video_path=str(video_path),
                    segments=segments,
                    video_id=vid_id,
                    segment_cache_dir=args.segment_cache_dir,
                )
                for seg_idx, result in enumerate(batch_results):
                    if isinstance(result, list):
                        video_facts.extend(result)
                    global_seg_done += 1
                    write_status(
                        f"[{global_seg_done}/{total_segments}] "
                        f"seg {seg_idx+1}/{len(segments)} | "
                        f"vid {vid_idx+1}/{total_videos} ({vid_id})"
                    )

                append_records(facts_file, [asdict(fact) for fact in video_facts])
                completed_video_ids.add(vid_id)
                write_completed_video_ids(completion_file, completed_video_ids)
                logger.info(f"  Video total: {len(video_facts)} facts (batch)")
                write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({vid_id}) DONE batch")
                continue  # skip sequential loop

            # ==============================================================
            # SEQUENTIAL PATH: HuggingFace or MEMORA typed memory (context chain)
            # ==============================================================
            # Cumulative summary: high-level description of what happened so far
            # Previous segment: detailed info from immediately previous segment
            cumulative_summaries = []  # List of summaries from all segments
            all_object_ids = set()  # Track all objects seen so far
            previous_context = ""  # For MEMORA typed memory context continuity

            MAX_SEGMENT_RETRIES = 3
            for seg_idx, segment in enumerate(segments):
              seg_start = segment["segment_start"]
              seg_end = segment["segment_end"]
              for _attempt in range(MAX_SEGMENT_RETRIES):
                # Snapshot mutable state for rollback on retry
                _snapshot_observations = len(video_segment_observations)
                _snap_summaries = len(cumulative_summaries)
                _snap_facts = len(video_facts)
                _snap_context = previous_context
                try:
                    # Process segment
                    result = processor.process_video_segment(
                        video_path=str(video_path),
                        start_time=seg_start,
                        end_time=seg_end,
                        segment_instruction=DIRECT_VIDEO_INSTRUCTION,
                        video_id=vid_id,
                        turn_id=seg_idx,
                        previous_context=previous_context,
                        segment_cache_dir=args.segment_cache_dir
                    )

                    if is_typed_memory:
                        # MEMORA returns one structured segment observation.
                        if result is not None and not isinstance(result, dict):
                            logger.warning(f"    Segment {seg_idx} returned {type(result).__name__} instead of dict. Skipping.")
                            result = None
                        if result is not None:
                            # Post-process: Convert relative timestamps to absolute
                            try:
                                activity_narrative = result.get("activity_narrative", {})
                                if isinstance(activity_narrative, dict):
                                    action_breakdown = activity_narrative.get("action_breakdown", [])
                                else:
                                    logger.warning(f"    activity_narrative is {type(activity_narrative).__name__}, expected dict. Skipping timestamp conversion.")
                                    action_breakdown = []
                                for action in action_breakdown:
                                    rel_ts = action.get("timestamp", "")
                                    if not isinstance(rel_ts, str):
                                        rel_ts = str(rel_ts) if rel_ts is not None else ""
                                    match = re.match(r'(\d+)(?:-(\d+))?s?', rel_ts)
                                    if match:
                                        rel_start = int(match.group(1))
                                        rel_end = int(match.group(2)) if match.group(2) else rel_start + 2
                                        if rel_end <= 10:
                                            abs_start = int(seg_start) + rel_start
                                            abs_end = int(seg_start) + rel_end
                                            action["timestamp"] = f"{abs_start}-{abs_end}s"
                            except Exception as e:
                                logger.warning(f"    Segment {seg_idx} timestamp conversion failed: {e}. Keeping original timestamps.")

                            video_segment_observations.append(result)

                            # Build Enhanced Context for Next Segment
                            activity_nar = result.get('activity_narrative', {})
                            if not isinstance(activity_nar, dict):
                                activity_nar = {}
                            summary = activity_nar.get('summary', '')
                            if summary:
                                cumulative_summaries.append(f"[{int(seg_start)}-{int(seg_end)}s] {summary}")

                            obj_registry = result.get('object_registry', {})
                            if isinstance(obj_registry, dict):
                                all_object_ids.update(obj_registry.keys())

                            prev_summary = summary or 'No summary'
                            prev_objects = list(obj_registry.keys()) if isinstance(obj_registry, dict) else []
                            prev_obj_count = len(prev_objects)
                            env_data = result.get('environment', {})
                            if isinstance(env_data, dict):
                                layout_desc = env_data.get('layout_description', '')
                                if not isinstance(layout_desc, str):
                                    layout_desc = str(layout_desc) if layout_desc else ''
                            else:
                                layout_desc = ''
                            prev_env = layout_desc[:100] if layout_desc else ''

                            # Extract last action for continuity
                            action_breakdown = activity_nar.get('action_breakdown', [])
                            if action_breakdown and isinstance(action_breakdown, list):
                                last_act = action_breakdown[-1]
                                if isinstance(last_act, dict):
                                    last_action_str = f"{last_act.get('action', '?')} {last_act.get('object', '?')} ({last_act.get('hand', '?')} hand)"
                                else:
                                    last_action_str = str(last_act)[:80]
                            else:
                                last_action_str = 'None'

                            previous_context = f"""=== VIDEO CONTEXT ===
[Cumulative Summary - Segments 0~{seg_idx}]:
{chr(10).join(cumulative_summaries) if cumulative_summaries else 'This is the first segment.'}

[Known Objects So Far ({len(all_object_ids)})]: {', '.join(sorted(all_object_ids)) if all_object_ids else 'None yet'}

[Previous Segment {seg_idx} ({int(seg_start)}-{int(seg_end)}s)]:
- Summary: {prev_summary}
- Last action: {last_action_str}
- Objects ({prev_obj_count}): {', '.join(prev_objects) if prev_objects else 'None'}
- Environment: {prev_env if prev_env else 'Not described'}

RULES:
- Reuse EXACT object_ids from [Known Objects] when the same object appears (e.g., if "baking_tray" exists, do NOT create "baking_sheet" for the same tray)
- Your summary MUST differ from previous segment's summary — describe what CHANGED or PROGRESSED
- Include at least 3 objects in object_registry (kitchen scenes always have multiple visible objects)
"""

                            obj_count = len(obj_registry) if isinstance(obj_registry, dict) else 0
                            logger.info(f"  Segment {seg_start:.0f}-{seg_end:.0f}s: {obj_count} objects tracked")
                        else:
                            if _attempt < MAX_SEGMENT_RETRIES - 1:
                                logger.warning(f"  Segment {seg_start:.0f}-{seg_end:.0f}s: Failed to parse (attempt {_attempt+1}/{MAX_SEGMENT_RETRIES}, retrying...)")
                                continue
                            logger.warning(f"  Segment {seg_start:.0f}-{seg_end:.0f}s: Failed to parse after {MAX_SEGMENT_RETRIES} attempts, skipping")
                    else:
                        # Flat-1D output is a list of ExtractedFact records.
                        video_facts.extend(result)

                        state_count = len([f for f in result if f.fact_type == "state"])
                        activity_count = len([f for f in result if f.fact_type == "activity"])
                        env_count = len([f for f in result if f.fact_type == "environment"])
                        logger.info(f"  Segment {seg_start:.0f}-{seg_end:.0f}s: {len(result)} facts "
                               f"({state_count}S, {activity_count}A, {env_count}E)")

                    global_seg_done += 1
                    write_status(
                        f"[{global_seg_done}/{total_segments}] "
                        f"seg {seg_idx+1}/{len(segments)} | "
                        f"vid {vid_idx+1}/{total_videos} ({vid_id})"
                    )
                    break  # success — exit retry loop
                except Exception as e:
                    # Rollback: discard any partial data added during failed attempt
                    del video_segment_observations[_snapshot_observations:]
                    del cumulative_summaries[_snap_summaries:]
                    del video_facts[_snap_facts:]
                    previous_context = _snap_context
                    if _attempt < MAX_SEGMENT_RETRIES - 1:
                        logger.warning(f"   Segment {seg_idx} ({seg_start}-{seg_end}s) attempt {_attempt+1}/{MAX_SEGMENT_RETRIES} failed: {e}. Retrying...")
                    else:
                        logger.error(f"   Segment {seg_idx} ({seg_start}-{seg_end}s) failed after {MAX_SEGMENT_RETRIES} attempts: {e}. Skipping.")

            if is_typed_memory:
                append_records(segments_file, video_segment_observations)
                logger.info(f"  Video total: {len(video_segment_observations)} segments (flushed to disk)")
            else:
                append_records(facts_file, [asdict(fact) for fact in video_facts])
                logger.info(f"  Video total: {len(video_facts)} facts (flushed to disk)")
            completed_video_ids.add(vid_id)
            write_completed_video_ids(completion_file, completed_video_ids)
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({vid_id}) DONE")
          except Exception as e:
            logger.error(f"   Video {vid_id} failed unexpectedly: {e}. Skipping to next video.")
            global_seg_done += _vid_seg_counts.get(vid_id, 0)
            write_status(f"[{global_seg_done}/{total_segments}] vid {vid_idx+1}/{total_videos} ({vid_id}) FAILED")
            continue

        if is_typed_memory:
            persisted = summarize_records(segments_file, "memora")
            total_segs_on_disk = persisted["records"]

            summary = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model_name,
                "backend": args.backend,
                "api_base": args.api_base if args.backend == "api" else None,
                "api_max_frames": args.api_max_frames if args.backend == "api" else None,
                "observation_format": "memora",
                "videos_processed": processed_count,
                "videos_resumed": skipped_count,
                "total_segments": total_segs_on_disk,
                "total_objects_tracked": persisted["objects"],
            }

            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            write_status(f"FINISHED {processed_count} videos (+{skipped_count} resumed) {total_segs_on_disk} segments")
            print(f"\n{'═'*60}", flush=True)
            print("Segment encoding complete. (MEMORA observations)", flush=True)
            print(f"    Videos processed: {processed_count} (+{skipped_count} resumed)", flush=True)
            print(f"    Total segments: {total_segs_on_disk}", flush=True)
            print(f"    Objects tracked: {summary['total_objects_tracked']}", flush=True)
            print(f"{'═'*60}\n", flush=True)

            logger.info("=" * 60)
            logger.info("Segment encoding complete. (MEMORA observations)")
            logger.info(f"   Segments: {total_segs_on_disk}")
            logger.info(f"   Objects tracked: {summary['total_objects_tracked']}")
            logger.info(f"   Output: {output_dir}")
            logger.info("=" * 60)
        else:
            persisted = summarize_records(facts_file, "flat_1d")
            total_facts_on_disk = persisted["records"]

            summary = {
                "timestamp": datetime.now().isoformat(),
                "model": args.model_name,
                "backend": args.backend,
                "api_base": args.api_base if args.backend == "api" else None,
                "api_max_frames": args.api_max_frames if args.backend == "api" else None,
                "observation_format": args.observation_format,
                "videos_processed": processed_count,
                "videos_resumed": skipped_count,
                "total_facts": total_facts_on_disk,
                "by_type": persisted["by_type"],
            }

            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)

            write_status(f"FINISHED {processed_count} videos {summary['total_facts']} facts")
            # Final summary for batch log
            print(f"\n{'═'*60}", flush=True)
            print("Observation structuring complete. (flat facts)", flush=True)
            print(f"    Videos processed: {processed_count}", flush=True)
            print(f"    Total facts: {summary['total_facts']}", flush=True)
            print(f"    By type: {summary['by_type']}", flush=True)
            print(f"{'═'*60}\n", flush=True)

            logger.info("=" * 60)
            logger.info("Observation structuring complete.")
            logger.info(f"   Facts: {summary['total_facts']}")
            logger.info(f"   By type: {summary['by_type']}")
            logger.info(f"   Output: {output_dir}")
            logger.info("=" * 60)

    finally:
        processor.unload_model()
