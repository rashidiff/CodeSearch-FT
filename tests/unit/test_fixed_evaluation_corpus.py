"""Fixed held-out set preparation is stable and labels remain source-traceable."""

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest

from codesearch_ft.config.models import EvaluationConfig, TrainingConfig
from codesearch_ft.datasets.codesearchnet import TrainingPair
from codesearch_ft.evaluation.fixed_corpus import prepare_fixed_evaluation_set


def _pair(query: str, code: str, identity: str, split: str) -> TrainingPair:
    return TrainingPair(
        query=query,
        code=code,
        query_hash=hashlib.sha256(query.encode()).hexdigest(),
        code_hash=hashlib.sha256(code.encode()).hexdigest(),
        repository=f"repo-{split}-{identity}",
        identity=identity,
        split=split,
    )


def test_preparation_keeps_test_associations_and_reproduces_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codesearch_ft.evaluation.fixed_corpus as fixed

    raw = {
        "train": [
            _pair(
                "train docs explain sorting",
                "def train_sort(x):\n    return sorted(x)",
                "t1",
                "train",
            ),
            _pair(
                "train docs explain reversing",
                "def train_reverse(x):\n    return x[::-1]",
                "t2",
                "train",
            ),
        ],
        "validation": [
            _pair(
                "validation docs explain joining",
                "def validate_join(x):\n    return ''.join(x)",
                "v1",
                "validation",
            ),
            _pair(
                "validation docs explain parsing",
                "def validate_parse(x):\n    return int(x)",
                "v2",
                "validation",
            ),
        ],
        "test": [
            _pair(
                "sort values in ascending order",
                "def sort_values(x):\n    return sorted(x)",
                "e1",
                "test",
            ),
            _pair(
                "reverse values in a sequence",
                "def reverse_values(x):\n    return x[::-1]",
                "e2",
                "test",
            ),
        ],
    }

    def fake_iter_source_splits(
        _config: TrainingConfig, _limits: Mapping[str, int | None]
    ) -> dict[str, list[TrainingPair]]:
        return raw

    monkeypatch.setattr(fixed, "iter_source_splits", fake_iter_source_splits)
    fixed_dir = tmp_path / "fixed"
    config = EvaluationConfig(
        profile="evaluation",
        candidate_corpus=fixed_dir / "candidates.jsonl",
        queries=fixed_dir / "queries.jsonl",
        output_dir=tmp_path / "run",
        candidate_limit=2,
        query_limit=2,
    )

    first = prepare_fixed_evaluation_set(config)
    first_candidates = config.candidate_corpus.read_bytes()
    first_queries = config.queries.read_bytes()
    second = prepare_fixed_evaluation_set(config)

    assert first["fixed_set_sha256"] == second["fixed_set_sha256"]
    assert first_candidates == config.candidate_corpus.read_bytes()
    assert first_queries == config.queries.read_bytes()
    assert first["counts"]["unique_candidates"] == 2
    assert first["counts"]["unique_queries"] == 2
    assert first["label_type"] == "automatic association; not human-verified relevance"
    assert first["partition_manifest"]["row_counts"] == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert first["partition_manifest"]["cross_split_duplicate_groups"] == 0
