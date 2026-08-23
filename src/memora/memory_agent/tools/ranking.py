"""Ranking configuration used by the paper's Inferred Knowledge retrieval."""

from typing import Any, Dict

_BALANCED_PATTERN_RANK: Dict[str, Any] = {
    "name": "balanced",
    "embedding_only": False,
    "pool_mult": 8,
    "pool_min": 56,
    "always_include_sources": (
        "consolidated_preference",
        "reusable_procedures",
    ),
    "w_source": 1.0,
    "w_lexical": 1.0,
    "w_choice": 1.0,
    "w_evidence": 1.0,
    "neg_source_scale": 1.0,
    "rank_pass_delta": 0.055,
}

def balanced_pattern_rank() -> Dict[str, Any]:
    """Return the fixed ranking profile used in the reported experiments."""
    return dict(_BALANCED_PATTERN_RANK)
