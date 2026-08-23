"""Shared optional embedding model for semantic memory retrieval."""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MODEL: Any = None
_LOAD_ATTEMPTED = False
_MODEL_LOCK = threading.Lock()


def embeddings_required() -> bool:
    """Return whether semantic retrieval must fail instead of degrading."""
    return os.environ.get("MEMORA_REQUIRE_E5", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def resolve_device(device: Optional[str] = None) -> str:
    """Resolve the configured embedding device, preferring a visible GPU."""
    requested = device or os.environ.get("MEMORA_E5_DEVICE", "auto")
    if requested != "auto":
        return requested

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:
        logger.debug("Could not inspect CUDA availability; using CPU: %s", exc)
    return "cpu"


def get_model(model_name: str, device: Optional[str] = None):
    """Load the shared E5 model once, or return ``None`` for keyword retrieval."""
    global _MODEL, _LOAD_ATTEMPTED
    resolved_device = resolve_device(device)

    if _MODEL is None and not _LOAD_ATTEMPTED:
        with _MODEL_LOCK:
            if _MODEL is None and not _LOAD_ATTEMPTED:
                _LOAD_ATTEMPTED = True
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading E5 model: %s", model_name)
                    _MODEL = SentenceTransformer(model_name, device=resolved_device)
                    logger.info("E5 model loaded on %s", resolved_device)
                except Exception as exc:
                    if embeddings_required():
                        raise RuntimeError(
                            f"Required E5 model {model_name!r} could not be loaded"
                        ) from exc
                    logger.warning(
                        "E5 model unavailable (%s); using keyword retrieval",
                        exc,
                    )
    if _MODEL is None and embeddings_required():
        raise RuntimeError(f"Required E5 model {model_name!r} is unavailable")
    return _MODEL


def get_for(memory_tools: Any):
    """Return the shared model configured by a ``TypedMemoryTools`` instance."""
    return get_model(memory_tools.e5_model_name, memory_tools.device)
