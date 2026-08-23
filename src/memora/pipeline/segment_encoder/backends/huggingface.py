"""Hugging Face implementation of the MEMORA Segment Encoder."""

import gc
import logging
import os
import subprocess
from pathlib import Path
from typing import List

try:
    import torch
except ImportError:
    torch = None

from memora.pipeline.segment_encoder.core import (
    ExtractedFact,
    OMNI_SYSTEM_MESSAGE,
    SegmentEncoderBackend,
    _build_segment_question,
    _new_temporary_video_path,
)

logger = logging.getLogger(__name__)


class HuggingFaceSegmentEncoder(SegmentEncoderBackend):
    """Encode video segments with Qwen2.5-Omni through Hugging Face."""

    def load_model(self):
        logger.info(f" Loading Omni: {self.model_name}")
        logger.info("   Backend: HuggingFace Transformers (single GPU)")
        if torch is None:
            raise RuntimeError(
                "The Hugging Face backend requires PyTorch. Install the GPU extras "
                "with `pip install -e '.[gpu]'`, or use `--backend api`."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The Hugging Face backend is configured for a CUDA GPU, but CUDA is "
                "not available. Use a GPU environment or select `--backend api`."
            )
        if self.video_fps is not None:
            logger.info(
                f"    Video FPS: {self.video_fps} (segments will be re-encoded)"
            )
        else:
            logger.info("    Video FPS: Original (no re-encoding)")

        from transformers import (
            Qwen2_5OmniProcessor,
            Qwen2_5OmniThinkerForConditionalGeneration,
            GenerationConfig,
        )

        model_kwargs = {
            "device_map": "cuda",  # Single GPU for data parallel
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }

        try:
            __import__("flash_attn")
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model.eval()

        self.generation_config = GenerationConfig(
            pad_token_id=151643,
            bos_token_id=151644,
            eos_token_id=151645,
            max_new_tokens=8192,
            do_sample=True,
            temperature=0.3,
        )

        logger.info(f" Omni model loaded (device: {self.model.device})")

    def unload_model(self):
        logger.info("Unloading Omni model...")

        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info(" Omni model unloaded")

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
    ) -> List[ExtractedFact]:
        """Process a video segment and extract facts.

        Args:
            segment_cache_dir: If provided, cache cut segments to this directory.
                              Segments are reused on subsequent runs, saving ffmpeg time.
        """

        # Segment caching: reuse pre-cut segments if available
        duration = end_time - start_time
        use_cache = segment_cache_dir is not None

        if use_cache:
            cache_dir = Path(segment_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Deterministic path: video_id_start_end_fps.mp4 (include FPS to avoid cache conflicts)
            fps_suffix = f"_fps{int(self.video_fps)}" if self.video_fps else ""
            segment_path = str(
                cache_dir
                / f"{video_id}_{int(start_time)}_{int(end_time)}{fps_suffix}.mp4"
            )
            segment_exists = os.path.exists(segment_path)
        else:
            segment_path = _new_temporary_video_path()
            segment_exists = False

        # Cut video segment only if not cached
        if not segment_exists:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_time),
                "-i",
                video_path,
                "-t",
                str(duration),
            ]
            if self.video_fps is not None:
                # Re-encode with target FPS (e.g., 5 FPS)
                cmd.extend(["-r", str(self.video_fps)])
                cmd.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-c:a",
                        "aac",
                        "-preset",
                        "ultrafast",
                        segment_path,
                    ]
                )
                logger.info(
                    f"    [FPS] Encoding segment at {self.video_fps} FPS (re-encode)"
                )
            else:
                # No FPS change: stream copy.
                cmd.extend(["-c", "copy", segment_path])
                logger.debug("    [COPY] Stream copy segment (no re-encode)")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Failed to cut segment: {result.stderr}")
                    if not use_cache:
                        Path(segment_path).unlink(missing_ok=True)
                    return []

                # Verify actual FPS with ffprobe
                if self.video_fps is not None:
                    try:
                        probe_cmd = [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=r_frame_rate,nb_frames,duration",
                            "-of",
                            "csv=p=0",
                            segment_path,
                        ]
                        probe_result = subprocess.run(
                            probe_cmd, capture_output=True, text=True
                        )
                        if probe_result.returncode == 0 and probe_result.stdout.strip():
                            parts = probe_result.stdout.strip().split(",")
                            if len(parts) >= 1:
                                fps_str = parts[0]  # e.g., "5/1" or "30000/1001"
                                if "/" in fps_str:
                                    num, den = map(int, fps_str.split("/"))
                                    actual_fps = num / den if den else 0
                                else:
                                    actual_fps = float(fps_str)
                                logger.info(
                                    f"    [FPS VERIFIED] Segment FPS: {actual_fps:.2f} (target: {self.video_fps})"
                                )
                                if len(parts) >= 3:
                                    nb_frames = parts[1] if parts[1] != "N/A" else "?"
                                    seg_duration = (
                                        parts[2] if parts[2] != "N/A" else "?"
                                    )
                                    logger.info(
                                        f"       Frames: {nb_frames}, Duration: {seg_duration}s"
                                    )
                    except Exception as probe_err:
                        logger.warning(f"    Could not verify FPS: {probe_err}")

            except Exception as e:
                logger.error(f"Error cutting video: {e}")
                if not use_cache:
                    Path(segment_path).unlink(missing_ok=True)
                return []
        else:
            logger.debug(f"   Using cached segment: {segment_path}")
            # Also verify FPS for cached segments (first time only per run)
            if self.video_fps is not None and not getattr(
                self, "_fps_cache_verified", False
            ):
                try:
                    probe_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        segment_path,
                    ]
                    probe_result = subprocess.run(
                        probe_cmd, capture_output=True, text=True
                    )
                    if probe_result.returncode == 0 and probe_result.stdout.strip():
                        fps_str = probe_result.stdout.strip()
                        if "/" in fps_str:
                            num, den = map(int, fps_str.split("/"))
                            actual_fps = num / den if den else 0
                        else:
                            actual_fps = float(fps_str)
                        logger.info(
                            f"   [CACHED FPS] {actual_fps:.2f} FPS (target: {self.video_fps})"
                        )
                        self._fps_cache_verified = True
                except Exception as exc:
                    logger.debug("Could not verify cached segment FPS: %s", exc)

        prompt = _build_segment_question(
            observation_format=self.observation_format,
            segment_instruction=segment_instruction,
            turn_id=turn_id,
            start_time=start_time,
            end_time=end_time,
            previous_context=previous_context,
            configured_typed_prompt=self._config_prompt_typed,
            configured_flat_prompt=self._config_prompt_flat,
        )

        # Build messages for Omni (same system message as vLLM backend)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": OMNI_SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": segment_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        try:
            # Pre-check: skip segments too short for the target FPS
            # At FPS=5, a segment needs at least 0.4s (2 frames minimum)
            if self.video_fps is not None:
                min_duration = 2.0 / self.video_fps  # Need at least 2 frames
                if duration < min_duration:
                    logger.warning(
                        f"    Segment too short for {self.video_fps} FPS: {duration:.1f}s < {min_duration:.1f}s (need ≥2 frames). Skipping."
                    )
                    return [] if self.observation_format != "memora" else None

            # Process with Omni
            try:
                from qwen_omni_utils import process_mm_info
            except ImportError:

                def process_mm_info(messages, use_audio_in_video=False):
                    audios, images, videos = [], [], []
                    for msg in messages:
                        for content in msg.get("content", []):
                            if content.get("type") == "video":
                                videos.append(content["video"])
                    return audios, images, videos

            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            # JSON guidance: append "{" to prompt (same as vLLM backend)
            text += "{"

            USE_AUDIO_IN_VIDEO = False
            audios, images, videos = process_mm_info(
                messages, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            inputs = self.processor(
                text=text,
                audios=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)

            input_len = inputs.input_ids.size(1)
            with torch.no_grad():
                generation = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                generate_ids = generation[:, input_len:]
                gen_len = generate_ids.size(1)
                response_text = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            hit_max = gen_len >= self.generation_config.max_new_tokens
            logger.info(
                f"   [HF] {video_id} {start_time:.0f}-{end_time:.0f}s: "
                f"input={input_len}tok gen={gen_len}tok "
                f"{' HIT MAX_TOKENS' if hit_max else ' stopped naturally'}"
            )
            # Prepend "{" back (prompt ended with "{", model continues from there)
            response_text = "{" + response_text

            # Parse according to the selected observation representation.
            if self.observation_format == "memora":
                # Return one structured observation for this segment.
                return self._parse_typed_memory_response(
                    response_text, video_id, turn_id, start_time, end_time
                )
            else:
                facts = self._parse_facts_response(
                    response_text, video_id, start_time, end_time
                )
                return facts

        except ValueError as e:
            if "nframes" in str(e):
                logger.warning(
                    f"    Segment too short for model (nframes error): {e}. Skipping."
                )
                return [] if self.observation_format != "memora" else None
            else:
                logger.error(f"ValueError processing segment: {e}")
                import traceback

                traceback.print_exc()
                return [] if self.observation_format != "memora" else None
        except Exception as e:
            logger.error(f"Error processing segment: {e}")
            import traceback

            traceback.print_exc()
            return [] if self.observation_format != "memora" else None
        finally:
            # Only delete non-cached (temp) segments
            if not use_cache and os.path.exists(segment_path):
                os.remove(segment_path)
