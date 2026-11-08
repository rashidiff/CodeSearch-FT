"""FAISS CPU exact-search and persistence tests on small synthetic vectors."""

from pathlib import Path

import pytest

from codesearch_ft.domain.errors import IndexCompatibilityError
from codesearch_ft.retrieval.faiss_index import FaissFlatIPIndex


def test_index_flat_ip_normalizes_vectors_searches_by_cosine_and_reloads(
    tmp_path: Path,
) -> None:
    index = FaissFlatIPIndex(2)
    index.add((10, 20, 30), ((3.0, 4.0), (0.0, 5.0), (-1.0, 0.0)))

    results = index.search((6.0, 8.0), 10)

    assert [result.vector_id for result in results] == [10, 20, 30]
    assert all(
        abs(actual - expected) < 1e-6
        for actual, expected in zip(
            [result.score for result in results], [1.0, 0.8, -0.6], strict=True
        )
    )
    assert index.vector_ids == (10, 20, 30)

    artifact = tmp_path / "nested" / "index.faiss"
    index.save(artifact)
    restored = FaissFlatIPIndex(2)
    restored.load(artifact)
    assert restored.count == index.count
    assert restored.vector_ids == index.vector_ids
    assert restored.search((1.0, 0.0), 3) == index.search((1.0, 0.0), 3)


def test_faiss_validates_ids_dimensions_finiteness_and_nonzero_norms() -> None:
    with pytest.raises(ValueError, match="positive"):
        FaissFlatIPIndex(0)
    index = FaissFlatIPIndex(2)
    with pytest.raises(ValueError, match="equal lengths"):
        index.add((1,), ())
    with pytest.raises(ValueError, match="unique non-negative"):
        index.add((1, 1), ((1.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="unique non-negative"):
        index.add((-1,), ((1.0, 0.0),))
    with pytest.raises(ValueError, match="shape"):
        index.add((1,), ((1.0,),))
    with pytest.raises(ValueError, match="finite"):
        index.add((1,), ((float("nan"), 0.0),))
    with pytest.raises(ValueError, match="zero-norm"):
        index.add((1,), ((0.0, 0.0),))
    index.add((1,), ((1.0, 0.0),))
    with pytest.raises(ValueError, match="more than once"):
        index.add((1,), ((0.0, 1.0),))
    with pytest.raises(ValueError, match="top_k"):
        index.search((1.0, 0.0), 0)
    with pytest.raises(ValueError, match="non-zero norm"):
        index.search((0.0, 0.0), 1)


def test_corrupted_or_dimensionally_incompatible_artifact_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "index.faiss"
    source.write_bytes(b"not a FAISS index")
    with pytest.raises(IndexCompatibilityError, match="Unable to load"):
        FaissFlatIPIndex(2).load(source)

    index = FaissFlatIPIndex(2)
    index.add((1,), ((1.0, 0.0),))
    index.save(source)
    with pytest.raises(IndexCompatibilityError, match="dimension=2"):
        FaissFlatIPIndex(3).load(source)
