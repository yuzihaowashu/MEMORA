"""Semantic retrieval for selecting memory records shown to the editor."""

import logging
import os
from typing import Any, List

logger = logging.getLogger(__name__)


def _embeddings_required() -> bool:
    return os.environ.get("MEMORA_REQUIRE_E5", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# ============================================================================
# E5 Memory Retriever
# ============================================================================

class E5MemoryRetriever:
    """Select relevant memory records with E5 semantic similarity."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._e5_model = None
        self._initialized = True

    def _ensure_loaded(self):
        """Lazy load E5 model on CPU."""
        if self._e5_model is not None:
            return True

        try:
            logger.info(" Loading E5 embedding model for memory retrieval (CPU only)...")
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid GPU conflict with vLLM
            self._e5_model = SentenceTransformer('intfloat/e5-base-v2', device='cpu')
            logger.info(" E5 model loaded on CPU")
            return True
        except ImportError as exc:
            if _embeddings_required():
                raise RuntimeError(
                    "E5 retrieval was required, but sentence-transformers is unavailable"
                ) from exc
            logger.warning(" sentence_transformers not installed, using keyword fallback")
            return False
        except Exception as e:
            if _embeddings_required():
                raise RuntimeError(
                    "E5 retrieval was required, but intfloat/e5-base-v2 could not be loaded"
                ) from e
            logger.warning(f" Failed to load E5 model: {e}, using keyword fallback")
            return False

    def retrieve(self, query: str, memories: List[Any], top_k: int = 10) -> List[Any]:
        """
        Retrieve top-k most relevant memories for the query.

        Uses E5 embeddings with simple numpy matrix multiplication (no FAISS).
        This avoids GPU memory conflicts and is fast enough for small memory banks.

        Args:
            query: Search query (e.g., new facts combined)
            memories: List of MemoryEntry objects or dicts
            top_k: Maximum memories to return

        Returns:
            List of most relevant memories (same type as input)
        """
        if len(memories) <= top_k:
            return memories

        if not self._ensure_loaded():
            return self._keyword_retrieve(query, memories, top_k)

        try:
            import numpy as np

            # Extract text from memories
            texts = []
            for m in memories:
                if hasattr(m, 'text'):
                    texts.append(m.text)
                elif isinstance(m, dict):
                    texts.append(m.get('text', str(m)))
                else:
                    texts.append(str(m))

            # Encode query and memories (CPU only, normalized)
            query_embedding = self._e5_model.encode(
                [f"query: {query}"],
                normalize_embeddings=True
            )
            memory_embeddings = self._e5_model.encode(
                [f"passage: {t}" for t in texts],
                normalize_embeddings=True
            )

            # Simple numpy matrix multiplication (no FAISS needed)
            # Cosine similarity = dot product of normalized vectors
            similarities = memory_embeddings @ query_embedding.T  # (N, 1)
            similarities = similarities.flatten()  # (N,)

            # Get top-k indices
            k = min(top_k, len(texts))
            top_indices = np.argsort(similarities)[-k:][::-1]  # Descending order

            # Get results (preserve original order by sorting indices)
            sorted_indices = sorted(top_indices)
            results = [memories[i] for i in sorted_indices]

            return results

        except Exception as e:
            logger.warning(f" E5 retrieval error: {e}, using keyword fallback")
            return self._keyword_retrieve(query, memories, top_k)

    def _keyword_retrieve(self, query: str, memories: List[Any], top_k: int) -> List[Any]:
        """Fallback keyword-based retrieval."""
        query_words = set(query.lower().split())

        scored = []
        for m in memories:
            if hasattr(m, 'text'):
                text = m.text
            elif isinstance(m, dict):
                text = m.get('text', str(m))
            else:
                text = str(m)

            text_words = set(text.lower().split())
            score = len(query_words & text_words)
            scored.append((score, m))

        # Sort by score descending, then take top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]


def get_e5_retriever() -> E5MemoryRetriever:
    """Get E5 retriever singleton (class-level __new__ ensures single instance)."""
    return E5MemoryRetriever()
