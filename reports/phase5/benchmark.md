# CodeSearch-FT fixed-pool retrieval benchmark

Generated: 2026-11-08T12:05:32.130023+00:00

- Dataset: `code-search-net/code_search_net` @ `bd0cf261e357a3eb5c8fba490d23ec1a1cd59555` (Python, official test split)
- Pool: 128 candidates; 124 queries
- Labels: automatic associations; not human-verified judgments
- Dataset manifest SHA-256: `bc1f08e4a87fdf6e6053f02e4ef8c3dc6b3812b141bf47cb6c447b61845643ee`
- Fine-tuned weights SHA-256: `811ab9b99ddb7cf2ebc779c22ab55dde3ba4c1decc8477050a59e4038fc9c9d7`

| Mode | MRR@10 | Recall@1 | Recall@10 | nDCG@10 | Mean query ms | p95 query ms | Index seconds | Index bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.5975 | 0.4422 | 0.8763 | n/a | 5.03 | 10.70 | 2.310 | 442368 |
| pretrained_codebert | 0.0340 | 0.0081 | 0.1089 | n/a | 13.87 | 20.45 | 1.823 | 394330 |
| fine_tuned_codebert | 0.6724 | 0.5565 | 0.9355 | n/a | 11.44 | 16.27 | 1.797 | 394330 |
| hybrid | 0.7547 | 0.6290 | 0.9409 | n/a | 17.43 | 26.06 | 4.107 | 836698 |

## Interpretation and limits

- Automatic paired positives are not human judgments.
- Unjudged candidates are unknown; the report must not call them true negatives.
- CodeBERT pretraining overlaps CodeSearchNet, including potential test material.
- CodeSearchNet reupload has license=other and no per-example license metadata.
- No optional hard-negative-trained checkpoint was supplied for this comparison.
- These automatic documentation-to-function associations are positive labels, not human-verified judgments.
- Unjudged candidates are unknown. MRR/recall assess discovery of recorded positives only and do not establish that other candidates are irrelevant.
- No nDCG is calculated because this fixed set contains no explicit graded judgments.
- BM25 indexes the docstring-free source metadata produced by the same AST pipeline; target documentation text is not inserted as searchable content.
- Pretrained CodeBERT has CodeSearchNet-related pretraining overlap; this benchmark is not an unseen-corpus generalization claim.
- Measurements are local to the recorded hardware/software runtime and are not GPU pilot throughput measurements.

Artifacts: `benchmark.json` (full per-query rankings and measurements), `benchmark.csv`, and `metrics.svg`.
