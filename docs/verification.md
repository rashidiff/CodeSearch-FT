# Local release verification

Checks recorded on 2026-11-08 for the development project and its standalone public copy.

## Passed

## Commands used

From each project root, Ruff, formatting, Pyright and lock checks were run with the locked project tool environment:

```text
ruff check .
ruff format --check .
pyright
uv lock --check --offline
```

The development suite ran as `uv run --locked pytest -q --basetemp .phase6-tmp`; the public suite ran as `pytest -q --basetemp .test-tmp` from the public root. Both used `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. Isolated builds ran as `uv build --out-dir .phase6-dist` in development and `uv build --out-dir .release-build` in the public root.

- Ruff lint and formatting checks pass in both directories.
- Pyright reports 0 errors in both directories.
- `uv lock --check --offline` passes in both directories.
- Full offline-flagged pytest suites pass: 94 tests in the development project and 80 tests in the public copy. Coverage is 82.55%. Hugging Face Hub, Transformers, and Datasets were set to offline mode. The trainer checkpoint/resume test injects a deterministic tiny model/tokenizer and synthetic pairs; it does not download data or model weights.
- API route, validation, error handling, localhost default, and path/training restrictions are exercised by the passing API tests.
- Isolated `uv build` succeeded in both directories, producing source archives and wheels through the declared setuptools backend. The public wheel contains evaluation and training modules and imports with functional CLI help and API schema generation.
- Both Colab notebooks validate as nbformat 4, contain executable code cells, and have no saved outputs.
- Six benchmark report and manifest hashes match the recorded evaluation verification file.
- The recursive public-tree audit found no disallowed paths or text, host-specific Windows paths, credential patterns, databases, indexes, checkpoints, or raw dataset rows. Colab notebooks and the benchmark configuration use only the standard `/content/drive/MyDrive/codesearch_ft_colab_pilot` runtime path.
- Docker image `codesearch-ft:phase6-smoke` built successfully from the public release with Docker Desktop Engine 29.1.3 (`linux/amd64`). Container health returned `status=ok` and `search_ready=true`; the injected `tests/fixtures/docker_sample.py` indexed as one file/one function; lexical search returned `normalize_values_preserving_order` through both the CLI and HTTP `POST /v1/search` (MRR/search latency are not benchmark claims).

## Remaining environment checks

- The pinned CodeBERT weights are not cached locally. The explicit model smoke returned the expected local-cache-missing error under offline mode; no model or dataset download was attempted. This optional smoke was unavailable.

The earlier Windows `TestClient` socket stall and FAISS native failure did not reproduce outside the restricted sandbox: full suites, including API and FAISS tests, passed in the approved execution context. No code skip or test exclusion was added for those tests.

One non-failing warning remains from MLflow's SQLAlchemy dependency using the deprecated `noload` loader strategy.
