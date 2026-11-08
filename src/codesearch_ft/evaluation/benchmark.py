"""Run fixed-pool CodeSearchNet retrieval comparisons with auditable outputs."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import platform
import shutil
import tempfile
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

from codesearch_ft.application.indexing import RepositoryIndexingService
from codesearch_ft.config.models import EvaluationConfig
from codesearch_ft.domain.models import RetrievalMode, SearchHit
from codesearch_ft.evaluation.fixed_corpus import (
    EvaluationCandidate,
    EvaluationQuery,
    load_jsonl,
    prepare_fixed_evaluation_set,
)
from codesearch_ft.evaluation.metrics import (
    QueryMetric,
    aggregate_metrics,
    calculate_query_metrics,
    query_metric_dict,
)
from codesearch_ft.ingestion.ast_parser import AstCodeParser
from codesearch_ft.ingestion.repository_scanner import RepositoryScanner
from codesearch_ft.ml.codebert_encoder import MODEL_ID, CodeBERTEncoder
from codesearch_ft.retrieval.faiss_index import FaissFlatIPIndex
from codesearch_ft.retrieval.hybrid import reciprocal_rank_fusion
from codesearch_ft.storage.sqlite_repository import SQLiteRepository


def run_benchmark(
    config: EvaluationConfig,
    *,
    pretrained_encoder: Any | None = None,
    fine_tuned_encoder: Any | None = None,
) -> dict[str, Any]:
    """Compare BM25, base CodeBERT, fine-tuned CodeBERT, and BM25+fine-tuned RRF."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if not config.candidate_corpus.exists() and not config.queries.exists():
        prepare_fixed_evaluation_set(config)
    elif not config.candidate_corpus.is_file() or not config.queries.is_file():
        raise ValueError("Both fixed candidate corpus and query JSONL files must exist together.")
    candidates = load_jsonl(config.candidate_corpus, EvaluationCandidate)
    queries = load_jsonl(config.queries, EvaluationQuery)
    if len(candidates) < 2 or not queries:
        raise ValueError("The fixed benchmark needs at least two candidates and one query.")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_by_id) != len(candidates):
        raise ValueError("Candidate IDs in the fixed corpus are not unique.")
    for query in queries:
        unknown = set(query.relevance) - candidate_by_id.keys()
        if unknown:
            raise ValueError(f"Query {query.query_id} refers to absent candidate IDs.")

    source_manifest_path = config.candidate_corpus.parent / "dataset-manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("Fixed corpus is missing its dataset-manifest.json provenance file.")
    candidate_file_sha256 = _sha256(config.candidate_corpus)
    query_file_sha256 = _sha256(config.queries)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("dataset_revision") != config.dataset_revision:
        raise ValueError("Fixed corpus dataset revision differs from evaluation configuration.")
    if source_manifest.get("candidates_sha256") not in (None, candidate_file_sha256):
        raise ValueError("Fixed candidate corpus hash differs from its manifest.")
    if source_manifest.get("queries_sha256") not in (None, query_file_sha256):
        raise ValueError("Fixed query set hash differs from its manifest.")
    fixed_set_sha256 = source_manifest.get("fixed_set_sha256")
    if fixed_set_sha256 is not None:
        unsigned_manifest = dict(source_manifest)
        unsigned_manifest.pop("fixed_set_sha256", None)
        if _canonical_hash(unsigned_manifest) != fixed_set_sha256:
            raise ValueError("Fixed dataset manifest failed its integrity hash.")
    manifest_sha256 = _sha256(source_manifest_path)
    config_sha256 = _canonical_hash(config.model_dump(mode="json"))
    candidate_pool_sha256 = hashlib.sha256("\n".join(sorted(candidate_by_id)).encode()).hexdigest()
    checkpoint_metadata = _checkpoint_metadata(
        config.fine_tuned_checkpoint, expected_revision=config.model_revision
    )
    checkpoint_weights_sha256 = _checkpoint_weights_sha256(config.fine_tuned_checkpoint)
    index_dir = config.output_dir / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    bm25_hits, bm25_timings, bm25_index = _bm25_search(config, candidates, queries, index_dir)
    rankings: dict[str, dict[str, tuple[str, ...]]] = {"bm25": _candidate_ids_by_query(bm25_hits)}
    timings: dict[str, dict[str, float]] = {"bm25": bm25_timings}
    index_records: dict[str, dict[str, Any]] = {"bm25": bm25_index}

    if pretrained_encoder is None:
        pretrained_encoder = CodeBERTEncoder.from_pretrained(
            MODEL_ID,
            revision=config.model_revision,
            local_files_only=True,
        )
    base_rank, base_time, base_index = _semantic_search(
        config,
        "pretrained_codebert",
        pretrained_encoder,
        candidates,
        queries,
        index_dir / "pretrained-codebert.faiss",
    )
    rankings["pretrained_codebert"] = base_rank
    timings["pretrained_codebert"] = base_time
    index_records["pretrained_codebert"] = base_index
    del pretrained_encoder
    gc.collect()

    if fine_tuned_encoder is None:
        fine_tuned_encoder = CodeBERTEncoder.from_pretrained(
            str(config.fine_tuned_checkpoint), local_files_only=True
        )
    if fine_tuned_encoder.model_revision != config.model_revision:
        raise ValueError("Fine-tuned checkpoint base revision differs from evaluation config.")
    fine_rank, fine_time, fine_index = _semantic_search(
        config,
        "fine_tuned_codebert",
        fine_tuned_encoder,
        candidates,
        queries,
        index_dir / "fine-tuned-codebert.faiss",
    )
    rankings["fine_tuned_codebert"] = fine_rank
    timings["fine_tuned_codebert"] = fine_time
    index_records["fine_tuned_codebert"] = fine_index

    if "hybrid" in config.modes:
        hybrid_rank, hybrid_time = _hybrid_search(
            queries,
            bm25_hits,
            bm25_timings,
            fine_rank,
            fine_time,
            candidate_by_id,
            config,
        )
        rankings["hybrid"] = hybrid_rank
        timings["hybrid"] = hybrid_time
        index_records["hybrid"] = {
            "indexing_duration_seconds": bm25_index["indexing_duration_seconds"]
            + fine_index["indexing_duration_seconds"],
            "indexing_duration_sample_count": 1,
            "indexing_duration_unit": "seconds",
            "index_size_bytes": bm25_index["index_size_bytes"] + fine_index["index_size_bytes"],
            "index_size_unit": "bytes",
            "index_sha256": hashlib.sha256(
                (bm25_index["index_sha256"] + fine_index["index_sha256"]).encode()
            ).hexdigest(),
            "memory_rss_before_bytes": None,
            "memory_rss_after_bytes": None,
            "memory_rss_delta_bytes": None,
            "memory_unavailable_reason": (
                "Hybrid reuses its component indexes; no separate index is materialized."
            ),
        }

    per_query: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    per_mode_metrics: dict[str, list[QueryMetric]] = {}
    for mode in config.modes:
        mode_metrics = [
            calculate_query_metrics(
                query,
                rankings[mode].get(query.query_id, ()),
                candidate_count=len(candidates),
                top_k=config.top_k,
            )
            for query in queries
        ]
        per_mode_metrics[mode] = mode_metrics
        summaries[mode] = {
            **aggregate_metrics(mode_metrics),
            **_latency_summary(timings[mode]),
            **index_records[mode],
            "candidate_count": len(candidates),
            "candidate_pool_sha256": candidate_pool_sha256,
        }
    for query in queries:
        per_query.append(
            {
                "query_id": query.query_id,
                "label_type": query.label_type,
                "known_relevant_candidate_ids": sorted(query.relevance),
                "modes": {
                    mode: {
                        "ranking_top_k": list(rankings[mode].get(query.query_id, ()))[
                            : config.top_k
                        ],
                        **query_metric_dict(metric),
                        "latency_seconds": timings[mode][query.query_id],
                    }
                    for mode in config.modes
                    for metric in per_mode_metrics[mode]
                    if metric.query_id == query.query_id
                },
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "config_sha256": config_sha256,
        "dataset": {
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "dataset_config": config.dataset_config,
            "official_split": "test",
            "candidate_count": len(candidates),
            "candidate_id_set_sha256": candidate_pool_sha256,
            "query_count": len(queries),
            "source_manifest_sha256": manifest_sha256,
            "candidate_file_sha256": candidate_file_sha256,
            "query_file_sha256": query_file_sha256,
            "fixed_set_sha256": source_manifest.get("fixed_set_sha256"),
            "label_source": source_manifest.get("label_source"),
            "label_type": "automatic associations; not human-verified judgments",
            "unjudged_policy": (
                "Candidates without a recorded automatic association remain unknown, "
                "not true negatives."
            ),
        },
        "models": {
            "pretrained_codebert": {
                "model_id": MODEL_ID,
                "model_revision": config.model_revision,
            },
            "fine_tuned_codebert": {
                "model_id": checkpoint_metadata.get("model_id"),
                "model_revision": checkpoint_metadata.get("model_revision"),
                "dataset_revision": checkpoint_metadata.get("dataset_revision"),
                "dataset_manifest_sha256": checkpoint_metadata.get("dataset_manifest_sha256"),
                "metadata_sha256": checkpoint_metadata.get("metadata_sha256"),
                "weights_sha256": checkpoint_weights_sha256,
            },
        },
        "results": summaries,
        "per_query": per_query,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "torch_version": distribution_version("torch"),
            "encoder_device": str(getattr(fine_tuned_encoder, "device", "cpu")),
            "vector_index": "FAISS IndexFlatIP CPU",
        },
        "limitations": [
            *source_manifest.get("limitations", []),
            "No optional hard-negative-trained checkpoint was supplied for this comparison.",
        ],
        "measurement_notes": {
            "latency": (
                "Per-query wall clock seconds; semantic timing includes query encoding "
                "and exact FAISS search."
            ),
            "indexing_duration": (
                "Includes source scanning/FTS indexing or code encoding/FAISS "
                "construction and save."
            ),
            "index_size": (
                "Bytes for persisted SQLite FTS or exact FAISS vector index; excludes "
                "model weights and source JSONL."
            ),
            "memory": (
                "Process RSS snapshots in bytes before and after index build; unavailable "
                "is explicit when psutil is absent."
            ),
            "ndcg": (
                "Reported only for queries with explicit graded candidate judgments; "
                "otherwise null with a reason."
            ),
        },
    }
    _write_outputs(config, report)
    return report


def _bm25_search(
    config: EvaluationConfig,
    candidates: Sequence[EvaluationCandidate],
    queries: Sequence[EvaluationQuery],
    index_dir: Path,
) -> tuple[dict[str, tuple[SearchHit, ...]], dict[str, float], dict[str, Any]]:
    memory_before = _rss_bytes()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="codesearch-ft-fts-", dir=config.output_dir
    ) as temporary:
        temporary_root = Path(temporary)
        corpus_root = temporary_root / "corpus"
        corpus_root.mkdir()
        for candidate in candidates:
            (corpus_root / f"{candidate.candidate_id}.py").write_text(
                candidate.code, encoding="utf-8"
            )
        database_path = temporary_root / "metadata.sqlite3"
        repository = SQLiteRepository(database_path)
        service = RepositoryIndexingService(RepositoryScanner(), AstCodeParser(), repository)
        summary = service.index(corpus_root)
        if summary.function_count < 1:
            raise ValueError("No Python functions were available for BM25 indexing.")
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        hits: dict[str, tuple[SearchHit, ...]] = {}
        query_timings: dict[str, float] = {}
        for query in queries:
            query_started = time.perf_counter()
            raw_hits = repository.search_lexical(
                query.query, summary.function_count, summary.repository_id
            )
            unique: list[SearchHit] = []
            seen: set[str] = set()
            for hit in raw_hits:
                candidate_id = Path(hit.file_path).stem
                if candidate_id in seen or candidate_id not in candidate_by_id:
                    continue
                seen.add(candidate_id)
                candidate = candidate_by_id[candidate_id]
                unique.append(
                    SearchHit(
                        function_id=candidate_id,
                        name=candidate_id,
                        qualified_name=candidate.source_identity,
                        file_path=candidate_id,
                        line_start=hit.line_start,
                        line_end=hit.line_end,
                        code_snippet=candidate.code,
                        retrieval_mode=RetrievalMode.LEXICAL,
                        rank=len(unique) + 1,
                        score=hit.score,
                        score_type=hit.score_type,
                    )
                )
            hits[query.query_id] = tuple(unique)
            query_timings[query.query_id] = time.perf_counter() - query_started
        repository.close()
        index_bytes = database_path.stat().st_size
        target = index_dir / "bm25.sqlite3"
        shutil.copy2(database_path, target)
    elapsed = time.perf_counter() - started
    memory_after = _rss_bytes()
    return (
        hits,
        query_timings,
        {
            "indexing_duration_seconds": elapsed - sum(query_timings.values()),
            "indexing_duration_sample_count": 1,
            "indexing_duration_unit": "seconds",
            "index_size_bytes": index_bytes,
            "index_size_unit": "bytes",
            "index_sha256": _sha256(target),
            "indexed_function_count": summary.function_count,
            "indexing_memory_sample_count": 2
            if memory_before is not None and memory_after is not None
            else 0,
            "memory_rss_before_bytes": memory_before,
            "memory_rss_after_bytes": memory_after,
            "memory_rss_delta_bytes": memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None,
            "memory_unavailable_reason": None
            if memory_before is not None and memory_after is not None
            else "psutil is unavailable in this runtime.",
        },
    )


def _semantic_search(
    config: EvaluationConfig,
    mode: str,
    encoder: Any,
    candidates: Sequence[EvaluationCandidate],
    queries: Sequence[EvaluationQuery],
    index_path: Path,
) -> tuple[dict[str, tuple[str, ...]], dict[str, float], dict[str, Any]]:
    memory_before = _rss_bytes()
    index_started = time.perf_counter()
    ordered_candidates = tuple(sorted(candidates, key=lambda row: row.candidate_id))
    candidate_vectors = encoder.encode_code(tuple(row.code for row in ordered_candidates))
    if len(candidate_vectors) != len(ordered_candidates):
        raise ValueError(f"{mode} encoder returned a different number of candidate vectors.")
    vector_index = FaissFlatIPIndex(encoder.dimension)
    vector_index.add(tuple(range(len(ordered_candidates))), candidate_vectors)
    vector_index.save(index_path)
    index_duration = time.perf_counter() - index_started
    candidate_ids = {index: row.candidate_id for index, row in enumerate(ordered_candidates)}
    rankings: dict[str, tuple[str, ...]] = {}
    query_timings: dict[str, float] = {}
    for query in queries:
        started = time.perf_counter()
        vectors = encoder.encode_queries((query.query,))
        if len(vectors) != 1:
            raise ValueError(f"{mode} encoder did not return one query vector.")
        matches = vector_index.search(vectors[0], len(ordered_candidates))
        ordered_matches = sorted(
            matches,
            key=lambda match: (-match.score, candidate_ids[match.vector_id]),
        )
        rankings[query.query_id] = tuple(candidate_ids[item.vector_id] for item in ordered_matches)
        query_timings[query.query_id] = time.perf_counter() - started
    memory_after = _rss_bytes()
    return (
        rankings,
        query_timings,
        {
            "indexing_duration_seconds": index_duration,
            "indexing_duration_sample_count": 1,
            "indexing_duration_unit": "seconds",
            "index_size_bytes": index_path.stat().st_size,
            "index_size_unit": "bytes",
            "index_sha256": _sha256(index_path),
            "indexed_candidate_count": len(ordered_candidates),
            "embedding_dimension": encoder.dimension,
            "indexing_memory_sample_count": 2
            if memory_before is not None and memory_after is not None
            else 0,
            "memory_rss_before_bytes": memory_before,
            "memory_rss_after_bytes": memory_after,
            "memory_rss_delta_bytes": memory_after - memory_before
            if memory_before is not None and memory_after is not None
            else None,
            "memory_unavailable_reason": None
            if memory_before is not None and memory_after is not None
            else "psutil is unavailable in this runtime.",
        },
    )


def _candidate_ids_by_query(
    hits: dict[str, tuple[SearchHit, ...]],
) -> dict[str, tuple[str, ...]]:
    return {query_id: tuple(hit.function_id for hit in values) for query_id, values in hits.items()}


def _hybrid_search(
    queries: Sequence[EvaluationQuery],
    lexical_hits: dict[str, tuple[SearchHit, ...]],
    lexical_timings: dict[str, float],
    fine_rankings: dict[str, tuple[str, ...]],
    fine_timings: dict[str, float],
    candidate_by_id: dict[str, EvaluationCandidate],
    config: EvaluationConfig,
) -> tuple[dict[str, tuple[str, ...]], dict[str, float]]:
    hybrid: dict[str, tuple[str, ...]] = {}
    timings: dict[str, float] = {}
    for query in queries:
        semantic_hits = tuple(
            SearchHit(
                function_id=candidate_id,
                name=candidate_id,
                qualified_name=candidate_by_id[candidate_id].source_identity,
                file_path=candidate_id,
                line_start=1,
                line_end=max(len(candidate_by_id[candidate_id].code.splitlines()), 1),
                code_snippet=candidate_by_id[candidate_id].code,
                retrieval_mode=RetrievalMode.SEMANTIC,
                rank=rank,
                score=0.0,
                score_type="rank-only input to RRF",
            )
            for rank, candidate_id in enumerate(fine_rankings[query.query_id], start=1)
        )
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            lexical_hits[query.query_id],
            semantic_hits,
            top_k=len(candidate_by_id),
            constant=config.rrf_constant,
        )
        fusion_duration = time.perf_counter() - started
        hybrid[query.query_id] = tuple(hit.function_id for hit in fused)
        timings[query.query_id] = (
            fine_timings[query.query_id] + lexical_timings[query.query_id] + fusion_duration
        )
    return hybrid, timings


def _latency_summary(values: dict[str, float]) -> dict[str, Any]:
    samples = sorted(values.values())
    if not samples:
        return {
            "latency_unit": "seconds",
            "latency_sample_count": 0,
            "latency_mean_seconds": None,
            "latency_p50_seconds": None,
            "latency_p95_seconds": None,
        }
    return {
        "latency_unit": "seconds",
        "latency_sample_count": len(samples),
        "latency_mean_seconds": sum(samples) / len(samples),
        "latency_p50_seconds": _percentile(samples, 0.5),
        "latency_p95_seconds": _percentile(samples, 0.95),
    }


def _percentile(sorted_values: Sequence[float], quantile: float) -> float:
    index = max(math.ceil(quantile * len(sorted_values)) - 1, 0)
    return sorted_values[index]


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        return None


def _checkpoint_metadata(checkpoint: Path, *, expected_revision: str) -> dict[str, Any]:
    metadata_path = checkpoint / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Fine-tuned checkpoint metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("model_id") != MODEL_ID:
        raise ValueError("Fine-tuned checkpoint base model does not match CodeBERT.")
    if metadata.get("model_revision") != expected_revision:
        raise ValueError("Fine-tuned checkpoint base revision differs from evaluation config.")
    if metadata.get("dataset_revision") != "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555":
        raise ValueError("Fine-tuned checkpoint dataset revision does not match the benchmark.")
    if metadata.get("embedding_dimension") != 768:
        raise ValueError("Fine-tuned checkpoint must retain the CodeBERT 768-dimension contract.")
    if metadata.get("embedding_pooling") != "attention-mask-mean-l2-normalized":
        raise ValueError("Fine-tuned checkpoint pooling is incompatible with retrieval.")
    expected_hash = metadata.get("metadata_sha256")
    if expected_hash is not None:
        unsigned_metadata = dict(metadata)
        unsigned_metadata.pop("metadata_sha256", None)
        payload = json.dumps(unsigned_metadata, sort_keys=True, indent=2).encode()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError("Fine-tuned checkpoint metadata failed SHA-256 validation.")
    return metadata


def _checkpoint_weights_sha256(checkpoint: Path) -> str:
    weight_path = checkpoint / "model" / "model.safetensors"
    if not weight_path.is_file():
        raise ValueError(f"Fine-tuned checkpoint weights are missing: {weight_path}")
    return _sha256(weight_path)


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_outputs(config: EvaluationConfig, report: dict[str, Any]) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = (
            "query_id",
            "mode",
            "label_type",
            "mrr_at_10",
            "recall_at_1",
            "recall_at_10",
            "first_known_relevant_rank",
            "known_relevant_count",
            "unjudged_candidate_count",
            "latency_seconds",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for query in report["per_query"]:
            for mode, values in query["modes"].items():
                writer.writerow(
                    {
                        "query_id": query["query_id"],
                        "mode": mode,
                        "label_type": query["label_type"],
                        "mrr_at_10": values["reciprocal_rank_at_10"],
                        "recall_at_1": values["recall_at_1"],
                        "recall_at_10": values["recall_at_10"],
                        "first_known_relevant_rank": values["first_known_relevant_rank"],
                        "known_relevant_count": values["known_relevant_count"],
                        "unjudged_candidate_count": values["unjudged_candidate_count"],
                        "latency_seconds": values["latency_seconds"],
                    }
                )
    _write_markdown(output_dir / "benchmark.md", report)
    _write_svg(output_dir / "metrics.svg", report["results"])


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = [
        "# CodeSearch-FT fixed-pool retrieval benchmark",
        "",
        f"Generated: {report['created_at_utc']}",
        "",
        f"- Dataset: `{report['dataset']['dataset_id']}` @ "
        f"`{report['dataset']['dataset_revision']}` (Python, official test split)",
        f"- Pool: {report['dataset']['candidate_count']} candidates; "
        f"{report['dataset']['query_count']} queries",
        f"- Labels: {report['dataset']['label_type']}",
        f"- Dataset manifest SHA-256: `{report['dataset']['source_manifest_sha256']}`",
        f"- Fine-tuned weights SHA-256: "
        f"`{report['models']['fine_tuned_codebert']['weights_sha256']}`",
        "",
        "| Mode | MRR@10 | Recall@1 | Recall@10 | nDCG@10 | Mean query ms | "
        "p95 query ms | Index seconds | Index bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in report["results"].items():
        rows.append(
            f"| {mode} | {metrics['mrr_at_10']:.4f} | {metrics['recall_at_1']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {_format_optional(metrics['ndcg_at_10'])} | "
            f"{metrics['latency_mean_seconds'] * 1000:.2f} | "
            f"{metrics['latency_p95_seconds'] * 1000:.2f} | "
            f"{metrics['indexing_duration_seconds']:.3f} | {metrics['index_size_bytes']} |"
        )
    rows.extend(["", "## Interpretation and limits", ""])
    rows.extend(f"- {item}" for item in report["limitations"])
    rows.extend(
        [
            "- These automatic documentation-to-function associations are positive "
            "labels, not human-verified judgments.",
            "- Unjudged candidates are unknown. MRR/recall assess discovery of recorded "
            "positives only and do not establish that other candidates are irrelevant.",
            "- No nDCG is calculated because this fixed set contains no explicit graded judgments.",
            "- BM25 indexes the docstring-free source metadata produced by the same AST "
            "pipeline; target documentation text is not inserted as searchable content.",
            "- Pretrained CodeBERT has CodeSearchNet-related pretraining overlap; this "
            "benchmark is not an unseen-corpus generalization claim.",
            "- Measurements are local to the recorded hardware/software runtime and are "
            "not GPU pilot throughput measurements.",
            "",
            "Artifacts: `benchmark.json` (full per-query rankings and measurements), "
            "`benchmark.csv`, and `metrics.svg`.",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_svg(path: Path, results: dict[str, Any]) -> None:
    modes = tuple(results)
    width, height = 900, 90 + 62 * len(modes)
    bars: list[str] = []
    for index, mode in enumerate(modes):
        y = 48 + index * 62
        mrr = max(0.0, min(float(results[mode]["mrr_at_10"]), 1.0))
        recall = max(0.0, min(float(results[mode]["recall_at_10"]), 1.0))
        bars.extend(
            [
                f'<text x="12" y="{y + 18}" font-size="14">{mode}</text>',
                f'<rect x="210" y="{y}" width="{mrr * 500:.2f}" height="20" fill="#31688e"/>',
                f'<text x="720" y="{y + 16}" font-size="12">MRR@10 {mrr:.3f}</text>',
                f'<rect x="210" y="{y + 25}" width="{recall * 500:.2f}" '
                'height="20" fill="#35b779"/>',
                f'<text x="720" y="{y + 41}" font-size="12">Recall@10 {recall:.3f}</text>',
            ]
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="12" y="24" font-size="18" font-family="sans-serif">'
        "Held-out retrieval metrics</text>"
        f'<g font-family="sans-serif">{"".join(bars)}</g></svg>'
    )
    path.write_text(svg + "\n", encoding="utf-8")
