"""Create a deterministic, leakage-aware CodeSearchNet held-out retrieval pool."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codesearch_ft.config.models import EvaluationConfig, TrainingConfig
from codesearch_ft.datasets.codesearchnet import (
    DATASET_ID,
    DATASET_REVISION,
    TrainingPair,
    iter_source_splits,
    prepare_partitions,
)
from codesearch_ft.ingestion.ast_parser import AstCodeParser


@dataclass(frozen=True, slots=True)
class EvaluationCandidate:
    candidate_id: str
    code: str
    code_sha256: str
    source_identity: str


@dataclass(frozen=True, slots=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevance: dict[str, int]
    label_type: str
    judged_candidate_ids: tuple[str, ...] | None = None


def prepare_fixed_evaluation_set(config: EvaluationConfig) -> dict[str, Any]:
    """Write the pinned test-split candidate/query set and its provenance manifest."""
    if config.dataset_id != DATASET_ID or config.dataset_revision != DATASET_REVISION:
        raise ValueError("Evaluation supports only the pinned CodeSearchNet Python source.")
    if config.candidate_corpus.parent != config.queries.parent:
        raise ValueError("Candidate and query files must share one fixed-set directory.")

    source_config = TrainingConfig(
        profile="cpu-smoke",
        dataset_id=config.dataset_id,
        dataset_revision=config.dataset_revision,
        dataset_config=config.dataset_config,
        max_train_samples=max(config.candidate_limit, 128),
        validation_max_samples=max(config.candidate_limit, 128),
        test_max_samples=config.candidate_limit,
        random_seed=config.random_seed,
        output_dir=config.output_dir / "preparation",
        tracking_uri=f"sqlite:///{(config.output_dir / 'preparation' / 'mlflow.db').as_posix()}",
    )
    try:
        raw = iter_source_splits(
            source_config,
            {
                "train": source_config.max_train_samples,
                "validation": source_config.validation_max_samples,
                "test": config.candidate_limit,
            },
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to read the pinned CodeSearchNet Python Parquet splits. "
            "Check network access and Hugging Face availability, then rerun the benchmark."
        ) from exc
    partitions, source_manifest = prepare_partitions(raw, source_config)
    selected_rows = partitions["test"][: config.candidate_limit]
    parser = AstCodeParser()
    candidates_by_code: dict[str, EvaluationCandidate] = {}
    valid_pairs: list[tuple[TrainingPair, str]] = []
    syntax_rejected = 0
    for pair in selected_rows:
        try:
            functions = parser.extract(pair.code, f"{pair.code_hash[:20]}.py")
        except SyntaxError:
            syntax_rejected += 1
            continue
        if not functions:
            syntax_rejected += 1
            continue
        candidate_id = f"csn-{pair.code_hash}"
        candidates_by_code.setdefault(
            pair.code_hash,
            EvaluationCandidate(
                candidate_id=candidate_id,
                code=pair.code,
                code_sha256=pair.code_hash,
                source_identity=pair.identity,
            ),
        )
        valid_pairs.append((pair, candidate_id))
    if len(candidates_by_code) < 2:
        raise ValueError("Fewer than two valid unique CodeSearchNet test candidates remain.")

    query_groups: dict[str, EvaluationQuery] = {}
    for pair, candidate_id in valid_pairs[: config.query_limit]:
        previous = query_groups.get(pair.query_hash)
        if previous is None:
            query_groups[pair.query_hash] = EvaluationQuery(
                query_id=f"q-{pair.query_hash}",
                query=pair.query,
                relevance={candidate_id: 1},
                label_type="automatic_codesearchnet_association",
            )
        else:
            query_groups[pair.query_hash] = EvaluationQuery(
                query_id=previous.query_id,
                query=previous.query,
                relevance={**previous.relevance, candidate_id: 1},
                label_type=previous.label_type,
            )
    candidates = tuple(candidates_by_code.values())
    queries = tuple(query_groups.values())
    if not queries:
        raise ValueError("No traceable automatic query-code associations remain in test split.")

    candidate_payload = b"".join(
        (json.dumps(asdict(candidate), sort_keys=True) + "\n").encode("utf-8")
        for candidate in candidates
    )
    query_payload = b"".join(
        (json.dumps(asdict(query), sort_keys=True) + "\n").encode("utf-8") for query in queries
    )
    fixed_identity = {
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "dataset_config": config.dataset_config,
        "official_split": "test",
        "selection": (
            "first accepted rows in immutable Parquet source order after cross-split quarantine"
        ),
        "random_seed": config.random_seed,
        "candidate_limit": config.candidate_limit,
        "query_limit": config.query_limit,
        "label_source": ("CodeSearchNet test-row documentation-to-function association"),
        "label_type": "automatic association; not human-verified relevance",
        "counts": {
            "source_train_rows": len(raw["train"]),
            "source_validation_rows": len(raw["validation"]),
            "source_test_rows": len(raw["test"]),
            "test_rows_after_cross_split_quarantine": len(partitions["test"]),
            "syntax_rejected": syntax_rejected,
            "unique_candidates": len(candidates),
            "unique_queries": len(queries),
            "queries_with_traceable_positive": len(queries),
            "unjudged_candidates_per_query": (
                "all candidates without a recorded automatic association"
            ),
        },
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "partition_manifest": asdict(source_manifest),
        "candidates_sha256": hashlib.sha256(candidate_payload).hexdigest(),
        "queries_sha256": hashlib.sha256(query_payload).hexdigest(),
        "candidates": [candidate.candidate_id for candidate in candidates],
        "queries": [query.query_id for query in queries],
        "preprocessing": (
            "CodeSearchNet adapter docstring removal; AST syntax/function validation; "
            "exact cross-split quarantine"
        ),
        "limitations": [
            "Automatic paired positives are not human judgments.",
            "Unjudged candidates are unknown; the report must not call them true negatives.",
            "CodeBERT pretraining overlaps CodeSearchNet, including potential test material.",
            "CodeSearchNet reupload has license=other and no per-example license metadata.",
        ],
    }
    fixed_identity["fixed_set_sha256"] = hashlib.sha256(
        json.dumps(fixed_identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory = config.candidate_corpus.parent
    directory.mkdir(parents=True, exist_ok=True)
    config.candidate_corpus.write_bytes(candidate_payload)
    config.queries.write_bytes(query_payload)
    (directory / "dataset-manifest.json").write_text(
        json.dumps(fixed_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fixed_identity


def load_jsonl(path: Path, record_type: type[Any]) -> tuple[Any, ...]:
    records: list[Any] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            payload = json.loads(line)
            records.append(record_type(**payload))
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Invalid JSONL record in {path} at line {line_number}.") from exc
    return tuple(records)
