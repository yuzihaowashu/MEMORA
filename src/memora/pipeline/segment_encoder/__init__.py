"""Public interface for encoding video segments into typed observations."""

from memora.pipeline.segment_encoder.backends import (
    APISegmentEncoder,
    HuggingFaceSegmentEncoder,
    VLLMSegmentEncoder,
)
from memora.pipeline.segment_encoder.core import (
    ExtractedFact,
    FLAT_FACT_PROMPT,
    OMNI_SYSTEM_MESSAGE,
    TYPED_MEMORY_VLM_PROMPT,
)

__all__ = [
    "APISegmentEncoder",
    "ExtractedFact",
    "FLAT_FACT_PROMPT",
    "HuggingFaceSegmentEncoder",
    "OMNI_SYSTEM_MESSAGE",
    "TYPED_MEMORY_VLM_PROMPT",
    "VLLMSegmentEncoder",
    "main",
]


def main():
    from memora.pipeline.segment_encoder.pipeline import main as run_cli

    return run_cli()
