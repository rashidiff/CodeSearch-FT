# Phase 5 retrieval evaluation

## Fixed initial benchmark set

The initial benchmark is built from the pinned `code-search-net/code_search_net`
Python configuration at revision
`bd0cf261e357a3eb5c8fba490d23ec1a1cd59555`. The candidate corpus and queries
come from the official `test` partition only. Selection is deterministic: use
the first accepted rows in immutable Parquet source order, up to 128 rows by
default. The existing dataset adapter removes function docstrings from code,
preserves official split assignments, and quarantines detected exact
query/code duplicates crossing its sampled train, validation, and test rows.
AST parsing then rejects invalid candidate code. Candidates with identical
normalized code hashes share one candidate ID. Identical queries are grouped
and retain all associated candidate IDs as positives.

Each positive label is the source row's automatic documentation-to-function
association. It is not human-verified relevance. Candidate code is indexed
without the target function docstring, so BM25 searches names, signatures and
docstring-free source through the same AST/SQLite FTS5 path used by the product.
Every mode receives identical candidate and query IDs. Hybrid retrieval is
one-based reciprocal-rank fusion of BM25 and fine-tuned CodeBERT with `k=60`.

The generated fixed inputs and their `dataset-manifest.json` are written under
`artifacts/evaluation/codesearchnet-test-v1/`, an ignored local artifact path.
The manifest records source revisions, limits, filters, counts, IDs and file
hashes. The default config points at the extracted checkpoint in
`artifacts/evaluation/fine-tuned-codebert/`.

## Verified Colab run (2026-11-08)

The submitted Colab run evaluated 124 unique queries against 128 candidates
from the pinned official Python test partition. All 128 positive labels are
traceable automatic documentation-to-function associations. The run metadata
records a Tesla T4 and `encoder_device: cuda:0`; the Colab notebook loads both
CodeBERT encoders through that same device setting. The lexical and exact FAISS
indexes ran on CPU. Both CodeBERT modes report revision
`3b0952feddeffad0063f274080e3c23d75e7eb39`.

| Mode | MRR@10 | Recall@10 | Mean query latency | p95 query latency |
|---|---:|---:|---:|---:|
| BM25 | 0.5975 | 0.8763 | 5.03 ms | 10.70 ms |
| Pretrained CodeBERT | 0.0340 | 0.1089 | 13.87 ms | 20.45 ms |
| Fine-tuned CodeBERT | 0.6724 | 0.9355 | 11.44 ms | 16.27 ms |
| Hybrid | 0.7547 | 0.9409 | 17.43 ms | 26.06 ms |

The report's per-query MRR@10 and Recall@10 were independently recomputed from
the raw rankings, and the JSON, CSV and bootstrap intervals agree. The
submitted percentile bootstrap intervals (5,000 resamples, seed 4205) are:

| Mode | MRR@10 95% interval | Recall@10 95% interval |
|---|---:|---:|
| BM25 | [0.5299, 0.6641] | [0.8145, 0.9274] |
| Pretrained CodeBERT | [0.0150, 0.0582] | [0.0565, 0.1653] |
| Fine-tuned CodeBERT | [0.6026, 0.7397] | [0.8871, 0.9758] |
| Hybrid | [0.6947, 0.8121] | [0.8952, 0.9758] |

The pretrained-to-fine-tuned difference is large. The artifacts show the same
pinned model revision, evaluation query IDs, candidate pool hash, encoder
device and embedding dimension; the checkpoint weight hash matches the
separately verified local checkpoint. A paired bootstrap over the 124 raw
query rows gives a derived MRR@10 difference of +0.6383 (95% interval
[0.5677, 0.7067]) and Recall@10 difference of +0.8266 ([0.7581, 0.8911]); this
is a diagnostic derived from the raw results, not a replacement measurement.
These checks found no candidate-pool or model-revision mismatch. They cannot
establish broad generalization or exclude all data overlap.

The fixed-set manifest records zero exact cross-split duplicate groups among
its bounded 128-row train/validation/test prefixes. The checkpoint metadata
records a hash for the training manifest created for its 2,000-pair pilot,
but the submitted results archive does not include that full training
manifest or training rows. An independent local attempt to reconstruct that
manifest from the pinned Parquet source failed because this Windows
environment could not resolve the Hugging Face host. Therefore the archived
manifest does not let us independently intersect all 2,000 fine-tuning pairs
with the held-out candidates. Treat possible training-set overlap as an
unverified limitation. The report also notes CodeBERT pretraining overlap
with CodeSearchNet.

The untouched submitted archive is
`artifacts/evaluation/codesearch_ft_phase5_results.zip`; it is also extracted
to `artifacts/evaluation/phase5-colab-results/`. The companion
`verification.json` records member hashes and verification checks. The run's
checkpoint weight SHA-256 is
`811ab9b99ddb7cf2ebc779c22ab55dde3ba4c1decc8477050a59e4038fc9c9d7`.

These results use automatic positives, not human judgments. Unjudged candidates
remain unknown, and nDCG@10 is unavailable because there are no graded labels.
The fixed source-order sample contains 124 queries, so its bootstrap intervals
are descriptive and do not correct for source-order selection or query
dependence. No improvement claim should be generalized beyond this benchmark.

## Labels and interpretation

Only recorded automatic positives are known. A candidate without an associated
label remains unjudged; it is not asserted to be irrelevant. MRR@10 and
Recall@1/@10 measure discovery of the known positives. nDCG@10 is emitted only
when a query supplies explicit graded labels and a judged-candidate pool;
otherwise the result is `null` with an explanation. The initial CodeSearchNet
set has binary automatic associations, so it does not produce nDCG.

This is a reproducible benchmark of automatic paired examples, not a human
quality study. CodeBERT's pretraining overlaps CodeSearchNet, including
potential evaluation material. The dataset reupload declares `license: other`
and has no per-example license metadata. These results do not establish
generalization to unseen repositories or exhaustive relevance.

## Run the benchmark

First extract `artifacts/fine-tuned-codebert.zip` so that
`artifacts/evaluation/fine-tuned-codebert/metadata.json` and its `model/`
directory exist. Then run:

```powershell
uv sync --extra inference --group training
uv run codesearch config validate configs/evaluation.yaml --kind evaluation
uv run codesearch benchmark --config configs/evaluation.yaml
```

The first run streams bounded source rows from the pinned dataset; it does not
train or download the full corpus. Pretrained model weights must already be in
the local Hugging Face cache or be downloaded explicitly with
`codesearch models download`. The fine-tuned model is loaded from its local
checkpoint directory. No benchmark mode receives a different candidate pool.

Outputs under `artifacts/evaluation/codesearchnet-test-v1/run/` include
`benchmark.json`, per-query `benchmark.csv`, `benchmark.md`, `metrics.svg`, and
exact SQLite/FAISS search indexes. JSON records hashes for configuration,
dataset inputs, models, and indexes; measured query latency, indexing duration,
index bytes, and process RSS snapshots include units/sample counts or explicit
unavailable values. Runtime timings depend on host hardware and software.

Do not tune implementation choices against the test labels. Use a separate
validation protocol for development and preserve the fixed test set for
reported comparisons.
