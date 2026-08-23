"""Shared contracts, prompts, and response parsing for Segment Encoder backends."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from memora.pipeline.formation_config import EPIC_KITCHENS_CONFIG
from memora.pipeline.segment_encoder.observations import (
    fix_environment_format,
    fix_object_registry_format,
    sanitize_object_registry,
)
from memora.pipeline.segment_encoder.prompts import SegmentEncoderPrompts

logger = logging.getLogger(__name__)

_DEFAULT_PROMPTS = SegmentEncoderPrompts(EPIC_KITCHENS_CONFIG)
TYPED_MEMORY_VLM_PROMPT = _DEFAULT_PROMPTS.build_typed_observation_prompt()
FLAT_FACT_PROMPT = _DEFAULT_PROMPTS.build_flat_observation_prompt()
OMNI_SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)
DIRECT_VIDEO_INSTRUCTION = (
    "Inspect the current video segment directly. Record the activity, the entities "
    "involved in it, their visible states and locations, and the environmental "
    "context needed to ground the activity. Reuse identifiers from the previous "
    "context when the same entity persists, and do not infer unsupported entities."
)


def _new_temporary_video_path() -> str:
    """Create a private temporary path that ffmpeg may safely overwrite."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        return tmp.name


def _build_segment_question(
    observation_format: str,
    segment_instruction: str,
    turn_id: int,
    start_time: float,
    end_time: float,
    previous_context: str,
    configured_typed_prompt: Optional[str] = None,
    configured_flat_prompt: Optional[str] = None,
) -> str:
    """Build one backend-independent Segment Encoder question."""
    if observation_format == "memora":
        template = configured_typed_prompt or TYPED_MEMORY_VLM_PROMPT
        if template is None:
            raise RuntimeError("The MEMORA Segment Encoder prompt is unavailable")
        return template.format(
            turn_id=turn_id,
            start_time=int(start_time),
            end_time=int(end_time),
            previous_context=previous_context
            or "This is the first segment of the video.",
            segment_instruction=segment_instruction,
        )

    template = configured_flat_prompt or FLAT_FACT_PROMPT
    return template.format(segment_instruction=segment_instruction)


@dataclass
class ExtractedFact:
    fact_type: str  # "state", "activity", "environment"
    text: str
    video_id: str
    start_time: float
    end_time: float


class SegmentEncoderBackend:
    """Common configuration and response parsing for Segment Encoder backends."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        observation_format: str = "memora",
        video_fps: float = None,
        config=None,
    ):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.generation_config = None
        self.observation_format = observation_format
        self.video_fps = video_fps
        self.config = config

        self._config_prompt_typed = None
        self._config_prompt_flat = None
        if config is not None:
            prompts = SegmentEncoderPrompts(config)
            self._config_prompt_typed = prompts.build_typed_observation_prompt()
            self._config_prompt_flat = prompts.build_flat_observation_prompt()

    def _parse_facts_response(
        self, raw_text: str, video_id: str, start_time: float, end_time: float
    ) -> List[ExtractedFact]:
        try:
            if "```json" in raw_text:
                json_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                json_text = raw_text.split("```")[1].split("```")[0]
            else:
                json_text = raw_text

            data = json.loads(json_text.strip())

            facts = []

            if "activities" in data or "states" in data or "environment" in data:
                # Flat-1D format: separate arrays for each observation type.
                # CRITICAL: Add time prefix to activity text for temporal QA support
                # This ensures "Person washes plate [0-10s]" and "Person washes plate [60-70s]"
                # are treated as different events, enabling temporal recall questions
                for text in data.get("activities", []):
                    if isinstance(text, str) and text.strip():
                        # Add time prefix to activity: "[0-10s] Person washes plate"
                        time_prefix = f"[{int(start_time)}-{int(end_time)}s]"
                        activity_text = f"{time_prefix} {text.strip()}"
                        facts.append(
                            ExtractedFact(
                                fact_type="activity",
                                text=activity_text,
                                video_id=video_id,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        )
                # States: No time prefix needed (only care about latest state)
                for text in data.get("states", []):
                    if isinstance(text, str) and text.strip():
                        facts.append(
                            ExtractedFact(
                                fact_type="state",
                                text=text.strip(),
                                video_id=video_id,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        )
                # Environment: No time prefix needed (stable features)
                for text in data.get("environment", []):
                    if isinstance(text, str) and text.strip():
                        facts.append(
                            ExtractedFact(
                                fact_type="environment",
                                text=text.strip(),
                                video_id=video_id,
                                start_time=start_time,
                                end_time=end_time,
                            )
                        )
            return facts
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse facts: {e}")
            return []

    def _parse_typed_memory_response(
        self,
        raw_text: str,
        video_id: str,
        turn_id: int,
        start_time: float,
        end_time: float,
    ) -> Optional[Dict[str, Any]]:
        """Parse MEMORA typed memory segment-observation response into structured data.

        The VLM sometimes closes the root JSON object after object_registry,
        then generates activity_narrative as a separate JSON block.  We merge
        all top-level JSON objects found in the raw text so nothing is lost.
        """
        try:
            if "```json" in raw_text:
                json_text = raw_text.split("```json")[1].split("```")[0]
            elif "```" in raw_text:
                json_text = raw_text.split("```")[1].split("```")[0]
            else:
                json_text = raw_text

            json_text = json_text.strip()
            data = self._parse_merged_json(json_text)

            nested_activity = None
            nested_registry = None
            environment_raw = data.get("environment", {})
            if isinstance(environment_raw, dict):
                nested_activity = environment_raw.pop("activity_narrative", None)
                nested_registry = environment_raw.pop("object_registry", None)
                if nested_activity is not None or nested_registry is not None:
                    logger.info(
                        "    Extracted activity_narrative/object_registry from nested environment"
                    )

            object_registry = data.get("object_registry") or nested_registry or {}
            object_registry = fix_object_registry_format(object_registry)
            object_registry = sanitize_object_registry(object_registry)

            environment = fix_environment_format(environment_raw)
            if environment is None:
                logger.warning("    Environment data unusable — will trigger retry")
                return None

            activity_narrative = data.get("activity_narrative") or nested_activity or {}
            if not activity_narrative:
                logger.debug("    activity_narrative empty after parse")

            segment_observation = {
                "video_id": video_id,
                "turn_id": turn_id,
                "time_window": {"start": start_time, "end": end_time},
                "environment": environment,
                "object_registry": object_registry,
                "activity_narrative": activity_narrative,
            }

            return segment_observation

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse MEMORA typed memory response: {e}")
            logger.debug(f"Raw text: {raw_text[:500]}...")
            return None
        except Exception as e:
            logger.warning(f"Error processing MEMORA typed memory response: {e}")
            return None

    def _parse_merged_json(self, text: str) -> dict:
        """Parse JSON text, merging multiple consecutive top-level objects.

        The VLM may output: {"environment":..., "object_registry":...}{"activity_narrative":...}
        This method detects that pattern and merges all objects into one dict.
        """
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        merged = {}
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            # Skip whitespace
            while pos < len(text) and text[pos] in " \t\n\r":
                pos += 1
            if pos >= len(text):
                break
            if text[pos] != "{":
                break
            try:
                obj, end = decoder.raw_decode(text, pos)
                if isinstance(obj, dict):
                    merged.update(obj)
                pos = end
            except json.JSONDecodeError:
                break

        if merged:
            logger.info(f"   Merged {len(merged)} keys from multi-object response")
            return merged

        # Fall back to repair
        repaired = self._try_repair_json(text)
        if repaired:
            logger.info("   JSON repaired successfully")
            return json.loads(repaired)

        raise json.JSONDecodeError("Could not parse or repair JSON", text, 0)

    def _try_repair_json(self, json_text: str) -> Optional[str]:
        """
        Attempt to repair common JSON errors from VLM output.

        Common issues:
        1. Trailing commas before } or ]
        2. Unclosed strings (missing closing quote)
        3. Unbalanced braces/brackets
        4. Single quotes instead of double quotes
        5. Unescaped special characters in strings
        """
        import re

        try:
            fixed = json_text

            # 1. Remove trailing commas before } or ]
            fixed = re.sub(r",\s*}", "}", fixed)
            fixed = re.sub(r",\s*]", "]", fixed)

            # 2. Replace single quotes with double quotes (careful with apostrophes)
            # Only replace if it looks like a JSON key/value pattern
            fixed = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', fixed)  # 'key': -> "key":

            # 3. Try to find and extract valid JSON object
            # Find first { and last }
            start_idx = fixed.find("{")
            end_idx = fixed.rfind("}")

            if start_idx >= 0 and end_idx > start_idx:
                fixed = fixed[start_idx : end_idx + 1]

            # 4. Try to balance braces/brackets by counting
            open_braces = fixed.count("{")
            close_braces = fixed.count("}")
            open_brackets = fixed.count("[")
            close_brackets = fixed.count("]")

            # Add missing closing braces/brackets at the end
            if open_braces > close_braces:
                fixed += "}" * (open_braces - close_braces)
            if open_brackets > close_brackets:
                fixed += "]" * (open_brackets - close_brackets)

            # 5. Close a likely unterminated string on each malformed line.
            lines = fixed.split("\n")
            fixed_lines = []
            for line in lines:
                # Count quotes in line
                quote_count = line.count('"') - line.count('\\"')
                if quote_count % 2 == 1:
                    # An odd quote count indicates a likely missing terminator.
                    if line.rstrip().endswith(",") or line.rstrip().endswith(":"):
                        line = line.rstrip()[:-1] + '"' + line.rstrip()[-1]
                    else:
                        line = line.rstrip() + '"'
                fixed_lines.append(line)
            fixed = "\n".join(fixed_lines)

            # 6. Final cleanup - remove any trailing content after last }
            last_brace = fixed.rfind("}")
            if last_brace >= 0:
                fixed = fixed[: last_brace + 1]

            # Verify the repair worked
            json.loads(fixed)
            return fixed

        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None
