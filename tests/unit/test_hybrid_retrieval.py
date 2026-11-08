"""Reciprocal-rank fusion formula, labels, ties, and validation."""

import pytest

from codesearch_ft.domain.models import RetrievalMode, SearchHit
from codesearch_ft.retrieval.hybrid import reciprocal_rank_fusion


def _hit(function_id: str, rank: int) -> SearchHit:
    return SearchHit(
        function_id=function_id,
        name=f"function_{function_id}",
        qualified_name=f"module.function_{function_id}",
        file_path="module.py",
        line_start=1,
        line_end=2,
        code_snippet="def f(): pass",
        retrieval_mode=RetrievalMode.LEXICAL,
        rank=rank,
        score=-0.1,
        score_type="test score",
    )


def test_rrf_uses_one_based_ranks_and_sums_duplicate_candidates() -> None:
    fused = reciprocal_rank_fusion(
        (_hit("1", 1), _hit("2", 2)),
        (_hit("2", 1), _hit("3", 2)),
        top_k=3,
        constant=2,
    )

    assert [hit.function_id for hit in fused] == ["2", "1", "3"]
    assert abs(fused[0].score - (1 / 4 + 1 / 3)) < 1e-12
    assert abs(fused[1].score - 1 / 3) < 1e-12
    assert abs(fused[2].score - 1 / 4) < 1e-12
    assert all(hit.retrieval_mode is RetrievalMode.HYBRID for hit in fused)
    assert [hit.rank for hit in fused] == [1, 2, 3]
    assert all(hit.score_type == "reciprocal_rank_fusion(k=2)" for hit in fused)


def test_rrf_ties_use_stable_function_id_and_validate_parameters() -> None:
    tied = reciprocal_rank_fusion((_hit("10", 1), _hit("2", 1)), (), top_k=3)

    assert [hit.function_id for hit in tied] == ["2", "10"]
    with pytest.raises(ValueError, match="top_k"):
        reciprocal_rank_fusion((), (), top_k=0)
    with pytest.raises(ValueError, match="constant"):
        reciprocal_rank_fusion((), (), top_k=1, constant=0)
