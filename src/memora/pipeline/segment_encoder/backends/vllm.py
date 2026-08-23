"""vLLM-Omni backend for the MEMORA Segment Encoder."""

import gc
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from memora.pipeline.segment_encoder.core import (
    DIRECT_VIDEO_INSTRUCTION,
    OMNI_SYSTEM_MESSAGE,
    SegmentEncoderBackend,
    _build_segment_question,
    _new_temporary_video_path,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Segment Encoder (vLLM-Omni)
# ============================================================================

class VLLMSegmentEncoder(SegmentEncoderBackend):
    """Encode video segments with an Omni model through vLLM-Omni.

    This implementation also supports batched segment inference.

    Key advantages over HuggingFace backend:
      - Continuous batching: process_video_segments_batch() sends all
        segments to vLLM in one call; the engine schedules them across
        KV-cache pages automatically.
      - Tensor parallelism: set tensor_parallel_size>1 for models too
        large for a single GPU (e.g. Qwen3-Omni-30B).
      - Accepts both numpy arrays and torch.Tensor for video frames.
    """

    SYSTEM_MESSAGE = OMNI_SYSTEM_MESSAGE

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        observation_format: str = "memora",
        video_fps: float = None,
        config=None,
        max_model_len: int = 12800,
        num_frames: int = -1,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
    ):
        """
        Args:
            num_frames: Frames to sample per segment.
                -1 (default) = use ALL frames in the cut segment. This is
                the correct choice when video_fps is set, because ffmpeg
                already controls temporal density (e.g. 5fps × 10s = 50
                frames). Passing all of them preserves the same information
                the HF backend's processor would see.
                Positive int = uniformly sample exactly N frames (useful for
                capping memory on very long / high-fps segments).
        """
        super().__init__(
            model_name=model_name,
            observation_format=observation_format,
            video_fps=video_fps,
            config=config,
        )
        self.max_model_len = max_model_len
        self.num_frames = num_frames
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.sampling_params = None

    def load_model(self):
        logger.info(f" Loading Omni: {self.model_name}")
        logger.info("   Backend: vLLM thinker-only (text output)")
        logger.info(f"   TP: {self.tensor_parallel_size} GPU(s)")
        if self.video_fps is not None:
            logger.info(f"    Video FPS: {self.video_fps} (segments will be re-encoded)")
        else:
            logger.info("    Video FPS: Original (no re-encoding)")

        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=5,
            limit_mm_per_prompt={"video": 1},
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
        )
        self.sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=8192,
            stop=["<|im_end|>"],
        )

        logger.info(f" vLLM model loaded: {self.model_name}")

    def unload_model(self):
        logger.info("Unloading vLLM model...")
        if self.llm is not None:
            del self.llm
            self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(" vLLM model unloaded")

    # ------------------------------------------------------------------
    # Video frame loading
    # ------------------------------------------------------------------

    def _load_video_frames(self, video_path: str) -> np.ndarray:
        """Load video frames as (N, H, W, 3) uint8 array.

        Resolution is NOT scaled here — the model's vision processor
        handles resizing internally.  Frame count is controlled by
        self.num_frames via uniform temporal sampling.

        Prefers vLLM's built-in video_to_ndarrays (uses OpenCV, only
        decompresses sampled frames). Falls back to decord, then pyav.
        """
        try:
            from vllm.assets.video import video_to_ndarrays
            frames = video_to_ndarrays(video_path, num_frames=self.num_frames)
            logger.debug(f"    [vllm/cv2] {len(frames)} frames, shape={frames.shape}")
            return frames
        except Exception as e:
            logger.debug(f"   video_to_ndarrays unavailable ({e}), trying fallbacks")

        try:
            from decord import VideoReader, cpu as decord_cpu
            vr = VideoReader(video_path, ctx=decord_cpu(0))
            total_frames = len(vr)
            num = total_frames if self.num_frames <= 0 else min(self.num_frames, total_frames)
            indices = np.linspace(0, total_frames - 1, num, dtype=int)
            frames = vr.get_batch(indices).asnumpy()
            logger.debug(f"    [decord] {num}/{total_frames} frames, shape={frames.shape}")
            return frames
        except ImportError:
            pass

        import av
        container = av.open(video_path)
        all_frames = [
            frame.to_ndarray(format='rgb24')
            for frame in container.decode(video=0)
        ]
        container.close()
        if not all_frames:
            return np.array([])
        total_frames = len(all_frames)
        num = total_frames if self.num_frames <= 0 else min(self.num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num, dtype=int)
        frames = np.stack([all_frames[i] for i in indices])
        logger.debug(f"    [pyav] {num}/{total_frames} frames, shape={frames.shape}")
        return frames

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_vllm_prompt(self, question: str) -> str:
        """Build prompt with correct chat format and video placeholder tokens.

        Token formats differ between model families:
          Qwen2.5-Omni: <|vision_bos|><|VIDEO|><|vision_eos|>
          Qwen3-Omni:   <|vision_start|><|video_pad|><|vision_end|>
        """
        if "qwen2" in self.model_name.lower():
            video_placeholder = "<|vision_bos|><|VIDEO|><|vision_eos|>"
        else:
            video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"
        return (
            f"<|im_start|>system\n{self.SYSTEM_MESSAGE}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{video_placeholder}"
            f"{question}<|im_end|>\n"
            "<|im_start|>assistant\n{{"
        )

    def _build_question(
        self,
        segment_instruction: str,
        turn_id: int = 0,
        start_time: float = 0,
        end_time: float = 0,
        previous_context: str = "",
    ) -> str:
        """Build the Segment Encoder question from context and task guidance."""
        return _build_segment_question(
            observation_format=self.observation_format,
            segment_instruction=segment_instruction,
            turn_id=turn_id,
            start_time=start_time,
            end_time=end_time,
            previous_context=previous_context,
            configured_typed_prompt=self._config_prompt_typed,
            configured_flat_prompt=self._config_prompt_flat,
        )

    # ------------------------------------------------------------------
    # Segment preparation (ffmpeg cut + frame load)
    # ------------------------------------------------------------------

    def _prepare_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        video_id: str,
        segment_cache_dir: str = None,
    ) -> Optional[tuple]:
        """Cut a segment with ffmpeg and load frames.

        Returns (segment_path, frames, is_temp) or None on failure.
        is_temp=True means the file should be cleaned up after use.
        """
        duration = end_time - start_time
        use_cache = segment_cache_dir is not None

        if use_cache:
            cache_dir = Path(segment_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            fps_suffix = f"_fps{int(self.video_fps)}" if self.video_fps else ""
            segment_path = str(
                cache_dir / f"{video_id}_{int(start_time)}_{int(end_time)}{fps_suffix}.mp4"
            )
            is_temp = False
            segment_exists = os.path.exists(segment_path)
        else:
            segment_path = _new_temporary_video_path()
            is_temp = True
            segment_exists = False

        if not segment_exists:
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
                "-t", str(duration),
            ]
            if self.video_fps is not None:
                cmd.extend([
                    "-r", str(self.video_fps),
                    "-c:v", "libx264", "-c:a", "aac", "-preset", "ultrafast",
                    segment_path,
                ])
            else:
                cmd.extend(["-c", "copy", segment_path])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"ffmpeg failed for {video_id} [{start_time}-{end_time}s]: {result.stderr[:200]}")
                    if is_temp:
                        Path(segment_path).unlink(missing_ok=True)
                    return None
            except Exception as e:
                logger.error(f"ffmpeg error: {e}")
                if is_temp:
                    Path(segment_path).unlink(missing_ok=True)
                return None

        if self.video_fps is not None:
            min_duration = 2.0 / self.video_fps
            if duration < min_duration:
                logger.warning(f"    Segment too short ({duration:.1f}s < {min_duration:.1f}s)")
                if is_temp and os.path.exists(segment_path):
                    os.remove(segment_path)
                return None

        try:
            frames = self._load_video_frames(segment_path)
            if frames.size == 0:
                logger.warning(f"    No frames from {video_id} [{start_time}-{end_time}s]")
                if is_temp and os.path.exists(segment_path):
                    os.remove(segment_path)
                return None
        except Exception as e:
            logger.error(f"   Frame load error: {e}")
            if is_temp and os.path.exists(segment_path):
                os.remove(segment_path)
            return None

        return (segment_path, frames, is_temp)

    # ------------------------------------------------------------------
    # Single-segment inference used by MEMORA typed memory
    # ------------------------------------------------------------------

    def process_video_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        segment_instruction: str,
        video_id: str,
        turn_id: int = 0,
        previous_context: str = "",
        segment_cache_dir: str = None,
    ) -> Any:
        """Process one segment. Same interface as parent class."""
        prep = self._prepare_segment(video_path, start_time, end_time, video_id, segment_cache_dir)
        if prep is None:
            return [] if self.observation_format != "memora" else None
        segment_path, frames, is_temp = prep

        try:
            question = self._build_question(
                segment_instruction, turn_id, start_time, end_time, previous_context
            )
            prompt = self._build_vllm_prompt(question)

            outputs = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"video": frames}},
                sampling_params=self.sampling_params,
            )
            if not outputs or not outputs[0].outputs:
                raise RuntimeError("vLLM returned empty output for video segment")
            completion = outputs[0].outputs[0]
            response_text = completion.text
            gen_len = len(completion.token_ids) if hasattr(completion, 'token_ids') else len(response_text) // 4
            finish = getattr(completion, 'finish_reason', 'unknown')
            logger.info(
                f"   [vLLM] {video_id} {start_time:.0f}-{end_time:.0f}s: "
                f"gen≈{gen_len}tok finish={finish} "
                f"{' HIT MAX_TOKENS' if finish == 'length' else ' stopped naturally'}"
            )
            # Prompt ends with "{" to guide JSON generation; prepend it back
            response_text = "{" + response_text

            if self.observation_format == "memora":
                return self._parse_typed_memory_response(
                    response_text, video_id, turn_id, start_time, end_time
                )
            return self._parse_facts_response(response_text, video_id, start_time, end_time)

        except Exception as e:
            logger.error(f"vLLM inference error [{video_id} {start_time}-{end_time}s]: {e}")
            import traceback
            traceback.print_exc()
            return [] if self.observation_format != "memora" else None
        finally:
            if is_temp and os.path.exists(segment_path):
                os.remove(segment_path)

    # ------------------------------------------------------------------
    # Batch inference for Flat-1D observations, whose segments are independent.
    # ------------------------------------------------------------------

    def process_video_segments_batch(
        self,
        video_path: str,
        segments: List[Dict],
        video_id: str,
        segment_cache_dir: str = None,
    ) -> List[Any]:
        """Process ALL segments of a video in one batched vLLM call.

        Only for Flat-1D observations, where segments are independent. MEMORA
        observations use the sequential ``process_video_segment`` path.

        Args:
            video_path: Full path to the source video file.
            segments: Segment boundaries with ``segment_start`` and ``segment_end``.
            video_id: Video identifier string.
            segment_cache_dir: Optional segment cache directory.

        Returns:
            List of results (one per segment). Failed segments return [].
        """
        if self.observation_format == "memora":
            raise ValueError(
                "Batch mode is incompatible with typed memory (context chain). "
                "Use process_video_segment() sequentially instead."
            )

        # Phase 1: prepare all segments (ffmpeg cut + frame load)
        prepared = []  # list of (idx, seg, segment_path, frames, is_temp)
        for idx, seg in enumerate(segments):
            start = seg["segment_start"]
            end = seg["segment_end"]
            prep = self._prepare_segment(video_path, start, end, video_id, segment_cache_dir)
            if prep is None:
                prepared.append((idx, seg, None, None, False))
            else:
                prepared.append((idx, seg, prep[0], prep[1], prep[2]))

        # Phase 2: build vLLM inputs for all valid segments
        batch_inputs = []
        valid_indices = []
        for idx, seg, seg_path, frames, is_temp in prepared:
            if frames is None:
                continue
            question = self._build_question(DIRECT_VIDEO_INSTRUCTION)
            prompt = self._build_vllm_prompt(question)
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": {"video": frames},
            })
            valid_indices.append(idx)

        # Phase 3: single batched vLLM call
        results = [[] for _ in segments]
        if batch_inputs:
            logger.info(f"Batch inference: {len(batch_inputs)} segments in one vLLM call")
            try:
                outputs = self.llm.generate(batch_inputs, sampling_params=self.sampling_params)
                if len(outputs) != len(batch_inputs):
                    logger.error(f"   vLLM batch mismatch: sent {len(batch_inputs)}, got {len(outputs)}; processing available outputs only")
                for out_idx, vllm_out in enumerate(outputs):
                    if out_idx >= len(valid_indices):
                        break
                    seg_idx = valid_indices[out_idx]
                    seg = segments[seg_idx]
                    if not vllm_out.outputs:
                        logger.warning(f"    vLLM returned no completions for segment {seg_idx}, skipping")
                        continue
                    response_text = vllm_out.outputs[0].text
                    try:
                        parsed = self._parse_facts_response(
                            response_text, video_id,
                            seg["segment_start"], seg["segment_end"],
                        )
                        results[seg_idx] = parsed
                    except Exception as e:
                        logger.warning(f"   Parse failed for segment {seg_idx}: {e}")
            except Exception as e:
                logger.error(f"    Batch inference failed: {e}")
                import traceback
                traceback.print_exc()

        # Cleanup temp files
        for idx, seg, seg_path, frames, is_temp in prepared:
            if is_temp and seg_path and os.path.exists(seg_path):
                os.remove(seg_path)

        return results
