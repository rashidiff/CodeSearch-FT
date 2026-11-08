"""Hand-checked ranking metrics and unjudged-label behavior."""

from codesearch_ft.evaluation.fixed_corpus import EvaluationQuery
from codesearch_ft.evaluation.metrics import aggregate_metrics, calculate_query_metrics


def test_automatic_positive_metrics_report_unknown_candidates_without_ndcg() -> None:
    query = EvaluationQuery(
        query_id="q-1",
        query="sort values",
        relevance={"candidate-a": 1},
        label_type="automatic_codesearchnet_association",
    )

    metric = calculate_query_metrics(
        query,
        ("candidate-b", "candidate-c", "candidate-a"),
        candidate_count=3,
    )

    assert metric.reciprocal_rank_at_10 == 1 / 3
    assert metric.recall_at_1 == 0
    assert metric.recall_at_10 == 1
    assert metric.first_known_relevant_rank == 3
    assert metric.unjudged_candidate_count == 2
    assert metric.ndcg_at_10 is None
    summary = aggregate_metrics((metric,))
    assert summary["ndcg_at_10"] is None
    assert summary["ndcg_at_10_unavailable_reason"]


def test_graded_ndcg_ignores_unjudged_candidates_in_rank_positions() -> None:
    query = EvaluationQuery(
        query_id="q-graded",
        query="find important items",
        relevance={"high": 2, "low": 1, "judged-none": 0},
        label_type="human_graded",
        judged_candidate_ids=("high", "low", "judged-none"),
    )
    metric = calculate_query_metrics(
        query,
        ("unjudged", "low", "high", "judged-none"),
        candidate_count=4,
    )

    assert metric.reciprocal_rank_at_10 == 0.5
    assert metric.recall_at_1 == 0
    assert metric.recall_at_10 == 1
    assert metric.unjudged_candidate_count == 1
    assert metric.ndcg_at_10 is not None
    assert 0 < metric.ndcg_at_10 < 1
