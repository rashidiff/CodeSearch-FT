"""Offline fixed-pool runner tests use deterministic local encoder fixtures."""

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import pytest

import codesearch_ft.evaluation.benchmark as benchmark_module
from codesearch_ft.config.models import EvaluationConfig
from codesearch_ft.domain.models import VectorMatch
from codesearch_ft.evaluation.benchmark import run_benchmark
from codesearch_ft.evaluation.fixed_corpus import EvaluationCandidate, EvaluationQuery


class FixtureEncoder:
    dimension = 2
    model_revision = "3b0952feddeffad0063f274080e3c23d75e7eb39"

    def __init__(self, *, correct: bool) -> None:
        self.correct = correct
        self.model_id = "microsoft/codebert-base"

    def encode_code(self, snippets: Sequence[str]) -> tuple[tuple[float, float], ...]:
        return tuple((1.0, 0.0) if "sort_values" in text else (0.0, 1.0) for text in snippets)

    def encode_queries(self, texts: Sequence[str]) -> tuple[tuple[float, float], ...]:
        vectors: list[tuple[float, float]] = []
        for text in texts:
            vector = (1.0, 0.0) if "sort" in text else (0.0, 1.0)
            if not self.correct:
                vector = (vector[1], vector[0])
            vectors.append(vector)
        return tuple(vectors)


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def test_runner_compares_every_mode_on_one_fixed_candidate_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixtureIndex:
        def __init__(self, dimension: int) -> None:
            self.dimension = dimension
            self.rows: list[tuple[int, tuple[float, ...]]] = []

        def add(self, vector_ids: Sequence[int], vectors: Sequence[tuple[float, ...]]) -> None:
            self.rows.extend(zip(vector_ids, vectors, strict=True))

        def save(self, destination: Path) -> None:
            destination.write_bytes(repr(self.rows).encode())

        def search(self, query: tuple[float, ...], top_k: int) -> tuple[VectorMatch, ...]:
            matches = sorted(
                (
                    VectorMatch(
                        vector_id,
                        sum(left * right for left, right in zip(query, vector, strict=True)),
                    )
                    for vector_id, vector in self.rows
                ),
                key=lambda match: (-match.score, match.vector_id),
            )
            return tuple(matches[:top_k])

    monkeypatch.setattr(benchmark_module, "FaissFlatIPIndex", FixtureIndex)
    fixed_dir = tmp_path / "fixed"
    fixed_dir.mkdir()
    output_dir = tmp_path / "run"
    candidate_path = fixed_dir / "candidates.jsonl"
    query_path = fixed_dir / "queries.jsonl"
    candidates = (
        EvaluationCandidate(
            "cand-sort",
            "def sort_values(values):\n    return sorted(values)\n",
            "a" * 64,
            "repo/a.py::sort_values",
        ),
        EvaluationCandidate(
            "cand-reverse",
            "def reverse_values(values):\n    return list(reversed(values))\n",
            "b" * 64,
            "repo/b.py::reverse_values",
        ),
    )
    queries = (
        EvaluationQuery("q-sort", "sort values", {"cand-sort": 1}, "automatic"),
        EvaluationQuery("q-reverse", "reverse values", {"cand-reverse": 1}, "automatic"),
    )
    _jsonl(candidate_path, [asdict(candidate) for candidate in candidates])
    _jsonl(query_path, [asdict(query) for query in queries])
    (fixed_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "dataset_revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
                "limitations": ["Fixture labels are automatic."],
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "fine-tuned-codebert"
    model_dir = checkpoint / "model"
    model_dir.mkdir(parents=True)
    (checkpoint / "metadata.json").write_text(
        json.dumps(
            {
                "model_id": "microsoft/codebert-base",
                "model_revision": "3b0952feddeffad0063f274080e3c23d75e7eb39",
                "dataset_revision": "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555",
                "embedding_dimension": 768,
                "embedding_pooling": "attention-mask-mean-l2-normalized",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"fixture weights")
    config = EvaluationConfig(
        profile="evaluation",
        candidate_corpus=candidate_path,
        queries=query_path,
        output_dir=output_dir,
        fine_tuned_checkpoint=checkpoint,
        candidate_limit=2,
        query_limit=2,
    )

    result = run_benchmark(
        config,
        pretrained_encoder=FixtureEncoder(correct=False),
        fine_tuned_encoder=FixtureEncoder(correct=True),
    )

    assert set(result["results"]) == {
        "bm25",
        "pretrained_codebert",
        "fine_tuned_codebert",
        "hybrid",
    }
    pool_hashes = {item["candidate_pool_sha256"] for item in result["results"].values()}
    assert len(pool_hashes) == 1
    assert result["results"]["fine_tuned_codebert"]["mrr_at_10"] == 1
    assert result["results"]["fine_tuned_codebert"]["ndcg_at_10"] is None
    assert result["models"]["pretrained_codebert"]["model_revision"] == (
        "3b0952feddeffad0063f274080e3c23d75e7eb39"
    )
    assert result["dataset"]["dataset_revision"] == ("bd0cf261e357a3eb5c8fba490d23ec1a1cd59555")
    assert result["environment"]["encoder_device"] == "cpu"
    for mode in result["results"].values():
        assert mode["latency_unit"] == "seconds"
        assert mode["latency_sample_count"] == 2
        assert mode["indexing_duration_seconds"] >= 0
        assert mode["indexing_duration_sample_count"] == 1
        assert mode["indexing_duration_unit"] == "seconds"
        assert mode["index_size_bytes"] > 0
        assert mode["index_size_unit"] == "bytes"
        assert mode["index_sha256"]
        if mode["memory_rss_before_bytes"] is None:
            assert mode["memory_unavailable_reason"]
        else:
            assert mode["indexing_memory_sample_count"] == 2
    for name in ("benchmark.json", "benchmark.csv", "benchmark.md", "metrics.svg"):
        assert (output_dir / name).is_file()
    report_json = json.loads((output_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert report_json["results"] == result["results"]
    assert len((output_dir / "benchmark.csv").read_text(encoding="utf-8").splitlines()) == 9
    repeated = run_benchmark(
        config,
        pretrained_encoder=FixtureEncoder(correct=False),
        fine_tuned_encoder=FixtureEncoder(correct=True),
    )
    assert {
        query["query_id"]: {
            mode: values["ranking_top_k"] for mode, values in query["modes"].items()
        }
        for query in repeated["per_query"]
    } == {
        query["query_id"]: {
            mode: values["ranking_top_k"] for mode, values in query["modes"].items()
        }
        for query in result["per_query"]
    }
    assert {mode: values["mrr_at_10"] for mode, values in repeated["results"].items()} == {
        mode: values["mrr_at_10"] for mode, values in result["results"].items()
    }
