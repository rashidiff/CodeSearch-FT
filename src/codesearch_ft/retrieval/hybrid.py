"""Rank-based deterministic lexical/semantic fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from codesearch_ft.domain.models import RetrievalMode, SearchHit


def reciprocal_rank_fusion(
    lexical_results: Sequence[SearchHit],
    semantic_results: Sequence[SearchHit],
    *,
    top_k: int,
    constant: int = 60,
) -> tuple[SearchHit, ...]:
    """Fuse one-based ranks with ``sum(1 / (constant + rank))``."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if constant < 1:
        raise ValueError("RRF constant must be positive.")
    scores: dict[str, float] = {}
    candidates: dict[str, SearchHit] = {}
    for results in (lexical_results, semantic_results):
        for hit in results:
            scores[hit.function_id] = scores.get(hit.function_id, 0.0) + 1.0 / (constant + hit.rank)
            candidates.setdefault(hit.function_id, hit)
    ordered = sorted(
        scores,
        key=lambda function_id: (-scores[function_id], _stable_id(function_id)),
    )[:top_k]
    return tuple(
        replace(
            candidates[function_id],
            retrieval_mode=RetrievalMode.HYBRID,
            rank=rank,
            score=scores[function_id],
            score_type=f"reciprocal_rank_fusion(k={constant})",
        )
        for rank, function_id in enumerate(ordered, start=1)
    )


def _stable_id(function_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(function_id))
    except ValueError:
        return (1, function_id)
