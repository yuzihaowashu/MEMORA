"""OpenAI-compatible API backend for the MEMORA Segment Encoder."""

import base64
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from memora.pipeline.api_client import DASHSCOPE_API_BASE, resolve_api_credentials
from memora.pipeline.segment_encoder.core import (
    OMNI_SYSTEM_MESSAGE,
    SegmentEncoderBackend,
    _build_segment_question,
    _new_temporary_video_path,
)

logger = logging.getLogger(__name__)

class APISegmentEncoder(SegmentEncoderBackend):
    """Video segment processing through an OpenAI-compatible vision API.

    DashScope's compatible endpoint accepts image inputs for Qwen-VL models.
    We approximate the Omni segment interface by sampling a small set of frames
    from each segment and sending them together with the segment instruction.
    """

    DEFAULT_API_BASE = DASHSCOPE_API_BASE

    def __init__(
        self,
        model_name: str,
        observation_format: str = "memora",
        video_fps: float = None,
        config=None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        api_max_frames: int = 4,
    ):
        super().__init__(
            model_name=model_name,
            observation_format=observation_format,
            video_fps=video_fps,
            config=config,
        )
        self.api_base, self.api_key = resolve_api_credentials(api_base, api_key)
        self.api_max_frames = max(1, api_max_frames)
        self.client = None

    def load_model(self):
        logger.info(f" Loading video API client: {self.model_name}")
        logger.info("   Backend: OpenAI-compatible vision API")
        logger.info(f"   API Base: {self.api_base}")
        logger.info(f"   API Key: {'set' if self.api_key and self.api_key != 'EMPTY' else 'EMPTY'}")
        logger.info(f"   Frames per segment: {self.api_max_frames}")
        from openai import OpenAI

        self.client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        logger.info(" Video API client ready")

    def unload_model(self):
        logger.info("Unloading video API client...")
        self.client = None
        logger.info(" Video API client disconnected")

    def _build_question(
        self,
        segment_instruction: str,
        turn_id: int = 0,
        start_time: float = 0,
        end_time: float = 0,
        previous_context: str = "",
    ) -> str:
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

    def _cut_segment(self, video_path: str, start_time: float, end_time: float, video_id: str, segment_cache_dir: str = None):
        duration = end_time - start_time
        use_cache = segment_cache_dir is not None
        if use_cache:
            cache_dir = Path(segment_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            segment_path = str(cache_dir / f"{video_id}_{int(start_time)}_{int(end_time)}_api.mp4")
            is_temp = False
            segment_exists = os.path.exists(segment_path)
        else:
            segment_path = _new_temporary_video_path()
            is_temp = True
            segment_exists = False

        if not segment_exists:
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-i", video_path,
                "-t", str(duration), "-an", "-c:v", "libx264", "-preset", "ultrafast",
                segment_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg failed for {video_id} [{start_time}-{end_time}s]: {result.stderr[:300]}")
                if is_temp and os.path.exists(segment_path):
                    os.remove(segment_path)
                return None
        return segment_path, is_temp

    def _extract_frame_data_urls(self, segment_path: str, duration: float) -> List[str]:
        frame_dir = Path(tempfile.mkdtemp(prefix="memora_api_frames_"))
        fps = min(1.0, max(0.1, self.api_max_frames / max(duration, 1.0)))
        pattern = str(frame_dir / "frame_%03d.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", segment_path,
            "-vf", f"fps={fps},scale='min(960,iw)':-2",
            "-q:v", "3", pattern,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"ffmpeg frame extraction failed: {result.stderr[:300]}")
                return []
            urls = []
            for frame_path in sorted(frame_dir.glob("frame_*.jpg"))[: self.api_max_frames]:
                b64 = base64.b64encode(frame_path.read_bytes()).decode("utf-8")
                urls.append(f"data:image/jpeg;base64,{b64}")
            return urls
        finally:
            for frame_path in frame_dir.glob("*"):
                try:
                    frame_path.unlink()
                except OSError:
                    pass
            try:
                frame_dir.rmdir()
            except OSError:
                pass

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
        duration = end_time - start_time
        cut = self._cut_segment(video_path, start_time, end_time, video_id, segment_cache_dir)
        if cut is None:
            return [] if self.observation_format != "memora" else None
        segment_path, is_temp = cut
        try:
            image_urls = self._extract_frame_data_urls(segment_path, duration)
            if not image_urls:
                logger.warning(f"    No API frames from {video_id} [{start_time}-{end_time}s]")
                return [] if self.observation_format != "memora" else None

            question = self._build_question(
                segment_instruction, turn_id, start_time, end_time, previous_context
            )
            content = [{"type": "text", "text": question}]
            content.extend(
                {"type": "image_url", "image_url": {"url": url}}
                for url in image_urls
            )

            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": OMNI_SYSTEM_MESSAGE},
                    {"role": "user", "content": content},
                ],
                max_tokens=4096,
                temperature=0.2,
            )
            response_text = resp.choices[0].message.content or ""
            logger.info(
                f"   [API] {video_id} {start_time:.0f}-{end_time:.0f}s: "
                f"{len(image_urls)} frames, {len(response_text)} chars"
            )

            if self.observation_format == "memora":
                return self._parse_typed_memory_response(
                    response_text, video_id, turn_id, start_time, end_time
                )
            return self._parse_facts_response(response_text, video_id, start_time, end_time)
        except Exception as e:
            logger.error(f"API video inference error [{video_id} {start_time}-{end_time}s]: {e}")
            import traceback
            traceback.print_exc()
            return [] if self.observation_format != "memora" else None
        finally:
            if is_temp and os.path.exists(segment_path):
                os.remove(segment_path)
