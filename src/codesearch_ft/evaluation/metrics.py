"""Ranking metrics that keep automatic positives separate from graded judgments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from codesearch_ft.evaluation.fixed_corpus import EvaluationQuery


@dataclass(frozen=True, slots=True)
class QueryMetric:
    query_id: str
    label_type: str
    reciprocal_rank_at_10: float
    recall_at_1: float
    recall_at_10: float
    first_known_relevant_rank: int | None
    known_relevant_count: int
    retrieved_known_relevant_at_10: int
    unjudged_candidate_count: int
    ndcg_at_10: float | None


def calculate_query_metrics(
    query: EvaluationQuery,
    ranking: Sequence[str],
    *,
    candidate_count: int,
    top_k: int = 10,
) -> QueryMetric:
    """Score only traceable positives; report unknown labels without calling them negatives."""
    if top_k < 1:
        raise ValueError("top_k must be positive.")
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    if not query.relevance:
        raise ValueError(f"Query {query.query_id} has no traceable positive labels.")
    relevant = {candidate_id for candidate_id, grade in query.relevance.items() if grade > 0}
    if not relevant:
        raise ValueError(f"Query {query.query_id} has no positive relevance labels.")
    visible = tuple(ranking[:top_k])
    first_rank = next(
        (rank for rank, candidate_id in enumerate(visible, start=1) if candidate_id in relevant),
        None,
    )
    retrieved = sum(candidate_id in relevant for candidate_id in visible)
    denominator = len(relevant)
    recall_at_1 = float(bool(visible) and visible[0] in relevant) / denominator
    recall_at_10 = retrieved / denominator
    ndcg = _ndcg_at_k(query, visible, top_k)
    judged = set(query.judged_candidate_ids or query.relevance)
    return QueryMetric(
        query_id=query.query_id,
        label_type=query.label_type,
        reciprocal_rank_at_10=1.0 / first_rank if first_rank is not None else 0.0,
        recall_at_1=recall_at_1,
        recall_at_10=recall_at_10,
        first_known_relevant_rank=first_rank,
        known_relevant_count=denominator,
        retrieved_known_relevant_at_10=retrieved,
        unjudged_candidate_count=max(candidate_count - len(judged), 0),
        ndcg_at_10=ndcg,
    )


def aggregate_metrics(rows: Sequence[QueryMetric]) -> dict[str, Any]:
    """Macro-average query metrics and include nDCG only for explicit graded pools."""
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation query set.")
    summary: dict[str, Any] = {
        "query_count": len(rows),
        "mrr_at_10": sum(row.reciprocal_rank_at_10 for row in rows) / len(rows),
        "recall_at_1": sum(row.recall_at_1 for row in rows) / len(rows),
        "recall_at_10": sum(row.recall_at_10 for row in rows) / len(rows),
        "known_relevant_labels": sum(row.known_relevant_count for row in rows),
        "unjudged_candidates_per_query_mean": sum(row.unjudged_candidate_count for row in rows)
        / len(rows),
    }
    graded = [row.ndcg_at_10 for row in rows if row.ndcg_at_10 is not None]
    summary["ndcg_at_10"] = sum(graded) / len(graded) if graded else None
    summary["ndcg_at_10_unavailable_reason"] = (
        None if graded else "No explicit graded relevance judgments were supplied."
    )
    return summary


def _ndcg_at_k(query: EvaluationQuery, ranking: Sequence[str], top_k: int) -> float | None:
    judged_ids = query.judged_candidate_ids
    if judged_ids is None:
        return None
    judged = set(judged_ids)
    grades = {candidate_id: grade for candidate_id, grade in query.relevance.items()}
    known_grades = [grade for candidate_id, grade in grades.items() if candidate_id in judged]
    if not known_grades or max(known_grades) <= 1:
        return None
    dcg = 0.0
    judged_rank = 0
    for candidate_id in ranking:
        if candidate_id not in judged:
            continue
        judged_rank += 1
        if judged_rank > top_k:
            break
        relevance = grades.get(candidate_id, 0)
        dcg += (2**relevance - 1) / math.log2(judged_rank + 1)
    ideal = sorted(known_grades, reverse=True)[:top_k]
    idcg = sum((2**grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def query_metric_dict(metric: QueryMetric) -> Mapping[str, Any]:
    """Serialize a per-query metric row without dataclass-specific consumers."""
    return {
        "query_id": metric.query_id,
        "label_type": metric.label_type,
        "reciprocal_rank_at_10": metric.reciprocal_rank_at_10,
        "recall_at_1": metric.recall_at_1,
        "recall_at_10": metric.recall_at_10,
        "first_known_relevant_rank": metric.first_known_relevant_rank,
        "known_relevant_count": metric.known_relevant_count,
        "retrieved_known_relevant_at_10": metric.retrieved_known_relevant_at_10,
        "unjudged_candidate_count": metric.unjudged_candidate_count,
        "ndcg_at_10": metric.ndcg_at_10,
    }
